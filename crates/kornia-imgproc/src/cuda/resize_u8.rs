//! Native CUDA u8 resize kernels, byte-exact with the CPU u8 fast paths.
//!
//! # Integer-LUT strategy (why this is NOT the f32 `PixelMapping` design)
//!
//! The f32 resize kernels achieve CPU/GPU bit-parity by mirroring float
//! expression trees under `--fmad=false`. The u8 CPU paths are different
//! animals: they precompute coordinates in **f64** on the host and run pure
//! integer fixed-point pixel loops (Q14 weights, round-half-up `+half >> Q`).
//! No device float expression can reproduce host f64 rounding — so instead we
//! upload the host-built integer tables and keep every kernel integer-only.
//! Byte-exactness then holds by construction, and the tests `assert_eq!`.
//!
//! The tables MUST come from the same builders the CPU uses
//! (`resize::bilinear::bilinear_axis_lut`, `resize::nearest::nearest_axis_lut`,
//! `resize::common::precompute_contribs`); the `resize::cuda` adapter is the
//! one caller and enforces this. The launchers here only check shapes.
//!
//! # Kernel ↔ CPU reference map (each pinned by a parity test)
//!
//! | Kernel                      | CPU reference                             | Rounding |
//! |-----------------------------|-------------------------------------------|----------|
//! | `resize_u8_pyrdown2x_rgb`   | `resize::kernels::pyrdown_row_rgb_u8`     | `(a+b+c+d+2)>>2` |
//! | `resize_u8_pyrup2x_rgb`     | `resize::kernels::{hinterp_row_rgb_u8, blend_75_25_row}` | nested `(a+b+1)>>1` twice |
//! | `resize_u8_nearest`         | `resize::nearest::resize_nearest_u8`      | pure gather |
//! | `resize_u8_bilinear`        | `resize::kernels::bilinear_row_u8`        | u64 `(...+ 1<<27) >> 28` |
//! | `resize_u8_sep_h` / `_v`    | `resize::kernels::{horizontal_row_scalar, vertical_row_scalar}` | i32 `(acc + 1<<13) >> 14`, i16 intermediate |
//!
//! The pyrup kernel reproduces the CPU's **nested rounding averages**
//! (`avg=(a+b+1)>>1; out=(x+avg+1)>>1`), NOT the algebraic `(3a+b+2)>>2` —
//! the two round differently (e.g. a=1, b=2).
//!
//! Signed `>>` is an arithmetic shift in both CUDA and Rust, which the
//! separable kernels rely on for negative bicubic/lanczos accumulators —
//! same guarantee `cuda/color` documents.

use std::sync::{Arc, OnceLock};

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use kornia_tensor::CudaKernel;

use super::resize::CudaResizeError;
use super::{make_config, try_compile_with_l1};

// ── CUDA C sources ────────────────────────────────────────────────────────────

static PYRDOWN2X_SRC: &str = r#"
extern "C" __global__ void resize_u8_pyrdown2x_rgb(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__       dst,
    unsigned int dst_w,
    unsigned int dst_h
) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;

    unsigned int src_w = dst_w * 2u;
    size_t r0 = (size_t)(2u * y) * src_w * 3u;
    size_t r1 = r0 + (size_t)src_w * 3u;
    size_t c0 = (size_t)(2u * x) * 3u;
    size_t d  = ((size_t)y * dst_w + x) * 3u;

    #pragma unroll
    for (int ch = 0; ch < 3; ++ch) {
        unsigned int sum = (unsigned int)__ldg(&src[r0 + c0 + ch])
                         + (unsigned int)__ldg(&src[r0 + c0 + 3u + ch])
                         + (unsigned int)__ldg(&src[r1 + c0 + ch])
                         + (unsigned int)__ldg(&src[r1 + c0 + 3u + ch]);
        dst[d + ch] = (unsigned char)((sum + 2u) >> 2);
    }
}
"#;

static PYRUP2X_SRC: &str = r#"
// Horizontal 2x sample of one source row at dst column x (see
// hinterp_row_rgb_u8): edges are clamped copies, interior pixels are the
// nested {0.75, 0.25} rounding blend of neighbours j and j+1.
__device__ __forceinline__ unsigned int pyrup_hval(
    const unsigned char* __restrict__ row,
    unsigned int x, unsigned int src_w, unsigned int ch
) {
    if (x == 0u) return (unsigned int)__ldg(&row[ch]);
    if (x == 2u * src_w - 1u) return (unsigned int)__ldg(&row[(size_t)(src_w - 1u) * 3u + ch]);
    unsigned int j = (x - 1u) >> 1;
    unsigned int a = (unsigned int)__ldg(&row[(size_t)j * 3u + ch]);
    unsigned int b = (unsigned int)__ldg(&row[(size_t)(j + 1u) * 3u + ch]);
    unsigned int avg = (a + b + 1u) >> 1;
    // dst[2j+1] leans on a (0.75a + 0.25b), dst[2j+2] leans on b.
    return (x & 1u) ? ((a + avg + 1u) >> 1) : ((b + avg + 1u) >> 1);
}

extern "C" __global__ void resize_u8_pyrup2x_rgb(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__       dst,
    unsigned int src_w,
    unsigned int src_h
) {
    unsigned int dst_w = src_w * 2u;
    unsigned int dst_h = src_h * 2u;
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;

    size_t src_stride = (size_t)src_w * 3u;
    size_t d = ((size_t)y * dst_w + x) * 3u;

    if (y == 0u || y == dst_h - 1u) {
        // Edge rows: horizontal pass of the clamped source row only.
        const unsigned char* row = src + (y == 0u ? 0 : (size_t)(src_h - 1u) * src_stride);
        #pragma unroll
        for (int ch = 0; ch < 3; ++ch) {
            dst[d + ch] = (unsigned char)pyrup_hval(row, x, src_w, ch);
        }
        return;
    }

    // Inner rows: block I consumes source rows (I, I+1); dst row 2I+1 leans
    // on row I, dst row 2I+2 leans on row I+1 (same nested blend as the
    // horizontal direction).
    unsigned int i = (y - 1u) >> 1;
    const unsigned char* ra = src + (size_t)i * src_stride;
    const unsigned char* rb = src + (size_t)(i + 1u) * src_stride;
    #pragma unroll
    for (int ch = 0; ch < 3; ++ch) {
        unsigned int a = pyrup_hval(ra, x, src_w, ch);
        unsigned int b = pyrup_hval(rb, x, src_w, ch);
        unsigned int avg = (a + b + 1u) >> 1;
        dst[d + ch] = (unsigned char)((y & 1u) ? ((a + avg + 1u) >> 1)
                                               : ((b + avg + 1u) >> 1));
    }
}
"#;

static NEAREST_SRC: &str = r#"
extern "C" __global__ void resize_u8_nearest(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__       dst,
    const int* __restrict__           xmap,
    const int* __restrict__           ymap,
    unsigned int src_w,
    unsigned int dst_w,
    unsigned int dst_h,
    unsigned int channels
) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;

    size_t so = ((size_t)__ldg(&ymap[y]) * src_w + (size_t)__ldg(&xmap[x])) * channels;
    size_t d  = ((size_t)y * dst_w + x) * channels;
    for (unsigned int ch = 0; ch < channels; ++ch) {
        dst[d + ch] = __ldg(&src[so + ch]);
    }
}
"#;

static BILINEAR_SRC: &str = r#"
// Q14 bilinear, byte-exact with bilinear_row_u8: the blend runs in 64-bit,
// with ONE round-half-up at the final Q28 shift. fx/fy come from the host
// f64 LUTs; fx1 = 16384 - fx is exact in integer arithmetic.
extern "C" __global__ void resize_u8_bilinear(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__       dst,
    const unsigned int* __restrict__  xofs,
    const unsigned int* __restrict__  xfx,
    const unsigned int* __restrict__  yofs,
    const unsigned int* __restrict__  yfy,
    unsigned int src_w,
    unsigned int dst_w,
    unsigned int dst_h,
    unsigned int channels
) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;

    const unsigned long long SCALE = 16384ull;
    unsigned long long fx  = (unsigned long long)__ldg(&xfx[x]);
    unsigned long long fy  = (unsigned long long)__ldg(&yfy[y]);
    unsigned long long fx1 = SCALE - fx;
    unsigned long long fy1 = SCALE - fy;

    size_t r0 = ((size_t)__ldg(&yofs[y]) * src_w + (size_t)__ldg(&xofs[x])) * channels;
    size_t r1 = r0 + (size_t)src_w * channels;
    size_t d  = ((size_t)y * dst_w + x) * channels;

    for (unsigned int ch = 0; ch < channels; ++ch) {
        unsigned long long p00 = (unsigned long long)__ldg(&src[r0 + ch]);
        unsigned long long p01 = (unsigned long long)__ldg(&src[r0 + channels + ch]);
        unsigned long long p10 = (unsigned long long)__ldg(&src[r1 + ch]);
        unsigned long long p11 = (unsigned long long)__ldg(&src[r1 + channels + ch]);
        unsigned long long top = p00 * fx1 + p01 * fx;
        unsigned long long bot = p10 * fx1 + p11 * fx;
        dst[d + ch] = (unsigned char)((top * fy1 + bot * fy + (1ull << 27)) >> 28);
    }
}
"#;

static SEP_H_SRC: &str = r#"
// Q14 separable horizontal pass, byte-exact with horizontal_row_scalar:
// i32 accumulate over kx taps, round-half-up at Q14, CLAMP TO i16 — the
// signed 16-bit intermediate (overshooting lobes included) is part of the
// CPU contract, not an optimization.
extern "C" __global__ void resize_u8_sep_h(
    const unsigned char* __restrict__  src,
    short* __restrict__                hbuf,
    const unsigned short* __restrict__ xsrc,
    const short* __restrict__          xw,
    unsigned int kx,
    unsigned int src_w,
    unsigned int dst_w,
    unsigned int src_h,
    unsigned int channels
) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= src_h) return;

    size_t row = (size_t)y * src_w * channels;
    size_t ibase = (size_t)x * kx;
    size_t d = ((size_t)y * dst_w + x) * channels;

    for (unsigned int ch = 0; ch < channels; ++ch) {
        int acc = 0;
        for (unsigned int t = 0; t < kx; ++t) {
            unsigned int sx = (unsigned int)__ldg(&xsrc[ibase + t]);
            acc += (int)__ldg(&src[row + (size_t)sx * channels + ch])
                 * (int)__ldg(&xw[ibase + t]);
        }
        int v = (acc + (1 << 13)) >> 14;
        v = min(max(v, -32768), 32767);
        hbuf[d + ch] = (short)v;
    }
}
"#;

static SEP_V_SRC: &str = r#"
// Q14 separable vertical pass, byte-exact with vertical_row_scalar: i16
// intermediate in, i32 accumulate, round-half-up at Q14, clamp to u8.
// Row indices clamp to [0, src_h-1] per tap (border replicate), exactly as
// the CPU's per-tap clamp.
extern "C" __global__ void resize_u8_sep_v(
    const short* __restrict__ hbuf,
    unsigned char* __restrict__ dst,
    const int* __restrict__     yofs,
    const short* __restrict__   yw,
    unsigned int ky,
    unsigned int src_h,
    unsigned int dst_w,
    unsigned int dst_h,
    unsigned int channels
) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;

    int y0 = __ldg(&yofs[y]);
    size_t wbase = (size_t)y * ky;
    size_t d = ((size_t)y * dst_w + x) * channels;

    for (unsigned int ch = 0; ch < channels; ++ch) {
        int acc = 0;
        for (unsigned int k = 0; k < ky; ++k) {
            int sy = min(max(y0 + (int)k, 0), (int)src_h - 1);
            acc += (int)__ldg(&hbuf[((size_t)sy * dst_w + x) * channels + ch])
                 * (int)__ldg(&yw[wbase + k]);
        }
        int v = (acc + (1 << 13)) >> 14;
        dst[d + ch] = (unsigned char)min(max(v, 0), 255);
    }
}
"#;

static PYRDOWN2X_KERNEL: OnceLock<Result<CudaKernel, String>> = OnceLock::new();
static PYRUP2X_KERNEL: OnceLock<Result<CudaKernel, String>> = OnceLock::new();
static NEAREST_KERNEL: OnceLock<Result<CudaKernel, String>> = OnceLock::new();
static BILINEAR_KERNEL: OnceLock<Result<CudaKernel, String>> = OnceLock::new();
static SEP_H_KERNEL: OnceLock<Result<CudaKernel, String>> = OnceLock::new();
static SEP_V_KERNEL: OnceLock<Result<CudaKernel, String>> = OnceLock::new();

fn get_kernel<'a>(
    cell: &'a OnceLock<Result<CudaKernel, String>>,
    ctx: &Arc<CudaContext>,
    src: &str,
    name: &str,
) -> Result<&'a CudaKernel, CudaResizeError> {
    cell.get_or_init(|| try_compile_with_l1(ctx, src, name))
        .as_ref()
        .map_err(|e| CudaResizeError::Cuda(e.clone()))
}

fn check_dst_len(dst_len: usize, w: u32, h: u32, channels: u32) -> Result<(), CudaResizeError> {
    let need = (w as usize) * (h as usize) * (channels as usize);
    if dst_len < need {
        return Err(CudaResizeError::SliceTooSmall { got: dst_len, need });
    }
    Ok(())
}

fn to_cuda_err(e: cudarc::driver::result::DriverError) -> CudaResizeError {
    CudaResizeError::Cuda(e.to_string())
}

// ── Public launchers ──────────────────────────────────────────────────────────

/// Exact-2× box-average downscale for RGB u8 (`dst = src/2` on both axes).
/// Byte-exact with `pyrdown_2x_rgb_u8`.
pub fn launch_resize_u8_pyrdown2x_rgb_cuda(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<u8>,
    dst: &mut CudaSlice<u8>,
    dst_width: u32,
    dst_height: u32,
    block_dim: Option<(u32, u32)>,
) -> Result<(), CudaResizeError> {
    super::check_geometry(
        dst_width * 2,
        dst_height * 2,
        dst_width,
        dst_height,
        block_dim,
    )
    .map_err(CudaResizeError::Cuda)?;
    check_dst_len(dst.len(), dst_width, dst_height, 3)?;

    let kernel = get_kernel(
        &PYRDOWN2X_KERNEL,
        ctx,
        PYRDOWN2X_SRC,
        "resize_u8_pyrdown2x_rgb",
    )?;
    kernel
        .launch_builder(stream)
        .arg(src)
        .arg(dst)
        .arg(&dst_width)
        .arg(&dst_height)
        .launch_2d(
            dst_width,
            dst_height,
            make_config(dst_width, dst_height, block_dim),
        )
        .map_err(|e| CudaResizeError::Cuda(e.to_string()))
}

/// Exact-2× bilinear upscale for RGB u8 (`dst = src*2` on both axes).
/// Byte-exact with `pyrup_2x_rgb_u8` (nested rounding-average blend).
pub fn launch_resize_u8_pyrup2x_rgb_cuda(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<u8>,
    dst: &mut CudaSlice<u8>,
    src_width: u32,
    src_height: u32,
    block_dim: Option<(u32, u32)>,
) -> Result<(), CudaResizeError> {
    let (dst_w, dst_h) = (src_width * 2, src_height * 2);
    super::check_geometry(src_width, src_height, dst_w, dst_h, block_dim)
        .map_err(CudaResizeError::Cuda)?;
    if src_width < 2 || src_height < 2 {
        return Err(CudaResizeError::Cuda(
            "pyrup2x requires source at least 2x2".into(),
        ));
    }
    check_dst_len(dst.len(), dst_w, dst_h, 3)?;

    let kernel = get_kernel(&PYRUP2X_KERNEL, ctx, PYRUP2X_SRC, "resize_u8_pyrup2x_rgb")?;
    kernel
        .launch_builder(stream)
        .arg(src)
        .arg(dst)
        .arg(&src_width)
        .arg(&src_height)
        .launch_2d(dst_w, dst_h, make_config(dst_w, dst_h, block_dim))
        .map_err(|e| CudaResizeError::Cuda(e.to_string()))
}

/// Nearest-neighbor u8 resize via host-built index LUTs (pure gather).
///
/// `xmap`/`ymap` MUST come from `nearest_axis_lut` for CPU parity; entries
/// must lie in `[0, src_len-1]` (the builders clamp).
#[allow(clippy::too_many_arguments)]
pub fn launch_resize_u8_nearest_cuda(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<u8>,
    dst: &mut CudaSlice<u8>,
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    channels: u32,
    xmap: &[i32],
    ymap: &[i32],
    block_dim: Option<(u32, u32)>,
) -> Result<(), CudaResizeError> {
    super::check_geometry(src_width, src_height, dst_width, dst_height, block_dim)
        .map_err(CudaResizeError::Cuda)?;
    check_dst_len(dst.len(), dst_width, dst_height, channels)?;
    if xmap.len() != dst_width as usize || ymap.len() != dst_height as usize {
        return Err(CudaResizeError::Cuda(
            "xmap/ymap length must equal dst width/height".into(),
        ));
    }

    let kernel = get_kernel(&NEAREST_KERNEL, ctx, NEAREST_SRC, "resize_u8_nearest")?;
    let d_xmap = stream.clone_htod(xmap).map_err(to_cuda_err)?;
    let d_ymap = stream.clone_htod(ymap).map_err(to_cuda_err)?;

    kernel
        .launch_builder(stream)
        .arg(src)
        .arg(dst)
        .arg(&d_xmap)
        .arg(&d_ymap)
        .arg(&src_width)
        .arg(&dst_width)
        .arg(&dst_height)
        .arg(&channels)
        .launch_2d(
            dst_width,
            dst_height,
            make_config(dst_width, dst_height, block_dim),
        )
        .map_err(|e| CudaResizeError::Cuda(e.to_string()))
}

/// Q14 bilinear u8 resize via host-built tap LUTs.
///
/// `xofs`/`xfx` and `yofs`/`yfy` MUST come from `bilinear_axis_lut` for CPU
/// parity: offsets clamped to `[0, src_len-2]`, weights in `[0, 16384]`.
/// Requires a source of at least 2×2 (the `ofs+1` neighbor read).
#[allow(clippy::too_many_arguments)]
pub fn launch_resize_u8_bilinear_cuda(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<u8>,
    dst: &mut CudaSlice<u8>,
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    channels: u32,
    xofs: &[u32],
    xfx: &[u32],
    yofs: &[u32],
    yfy: &[u32],
    block_dim: Option<(u32, u32)>,
) -> Result<(), CudaResizeError> {
    super::check_geometry(src_width, src_height, dst_width, dst_height, block_dim)
        .map_err(CudaResizeError::Cuda)?;
    if src_width < 2 || src_height < 2 {
        return Err(CudaResizeError::Cuda(
            "bilinear requires source at least 2x2".into(),
        ));
    }
    check_dst_len(dst.len(), dst_width, dst_height, channels)?;
    if xofs.len() != dst_width as usize
        || xfx.len() != dst_width as usize
        || yofs.len() != dst_height as usize
        || yfy.len() != dst_height as usize
    {
        return Err(CudaResizeError::Cuda(
            "bilinear LUT lengths must equal dst width/height".into(),
        ));
    }

    let kernel = get_kernel(&BILINEAR_KERNEL, ctx, BILINEAR_SRC, "resize_u8_bilinear")?;
    let d_xofs = stream.clone_htod(xofs).map_err(to_cuda_err)?;
    let d_xfx = stream.clone_htod(xfx).map_err(to_cuda_err)?;
    let d_yofs = stream.clone_htod(yofs).map_err(to_cuda_err)?;
    let d_yfy = stream.clone_htod(yfy).map_err(to_cuda_err)?;

    kernel
        .launch_builder(stream)
        .arg(src)
        .arg(dst)
        .arg(&d_xofs)
        .arg(&d_xfx)
        .arg(&d_yofs)
        .arg(&d_yfy)
        .arg(&src_width)
        .arg(&dst_width)
        .arg(&dst_height)
        .arg(&channels)
        .launch_2d(
            dst_width,
            dst_height,
            make_config(dst_width, dst_height, block_dim),
        )
        .map_err(|e| CudaResizeError::Cuda(e.to_string()))
}

/// Q14 separable bicubic / lanczos u8 resize (two passes, i16 intermediate).
///
/// Tables MUST come from `precompute_contribs` + `build_xsrc_lut` +
/// `pack_xw_i16` — including the antialias kernel widening — for CPU parity.
/// The i16 scratch (`src_h × dst_w × channels`) is stream-ordered allocated
/// per call; with the mempool release threshold raised, steady-state cost
/// is a pool lookup.
#[allow(clippy::too_many_arguments)]
pub fn launch_resize_u8_separable_cuda(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<u8>,
    dst: &mut CudaSlice<u8>,
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    channels: u32,
    xsrc: &[u16],
    xw: &[i16],
    kx: u32,
    yofs: &[i32],
    yw: &[i16],
    ky: u32,
    block_dim: Option<(u32, u32)>,
) -> Result<(), CudaResizeError> {
    super::check_geometry(src_width, src_height, dst_width, dst_height, block_dim)
        .map_err(CudaResizeError::Cuda)?;
    check_dst_len(dst.len(), dst_width, dst_height, channels)?;
    if kx == 0
        || ky == 0
        || xsrc.len() != (dst_width * kx) as usize
        || xw.len() != (dst_width * kx) as usize
        || yofs.len() != dst_height as usize
        || yw.len() != (dst_height * ky) as usize
    {
        return Err(CudaResizeError::Cuda(
            "separable LUT lengths must match dst dims x tap counts".into(),
        ));
    }

    let kernel_h = get_kernel(&SEP_H_KERNEL, ctx, SEP_H_SRC, "resize_u8_sep_h")?;
    let kernel_v = get_kernel(&SEP_V_KERNEL, ctx, SEP_V_SRC, "resize_u8_sep_v")?;

    let d_xsrc = stream.clone_htod(xsrc).map_err(to_cuda_err)?;
    let d_xw = stream.clone_htod(xw).map_err(to_cuda_err)?;
    let d_yofs = stream.clone_htod(yofs).map_err(to_cuda_err)?;
    let d_yw = stream.clone_htod(yw).map_err(to_cuda_err)?;

    // Intermediate: src_h rows of dst_w pixels, i16. Pass 1 writes every
    // element before pass 2 reads; no zero-fill needed.
    let inter_len = (src_height as usize) * (dst_width as usize) * (channels as usize);
    let mut hbuf = unsafe { stream.alloc::<i16>(inter_len) }
        .map_err(|e| CudaResizeError::Cuda(e.to_string()))?;

    kernel_h
        .launch_builder(stream)
        .arg(src)
        .arg(&mut hbuf)
        .arg(&d_xsrc)
        .arg(&d_xw)
        .arg(&kx)
        .arg(&src_width)
        .arg(&dst_width)
        .arg(&src_height)
        .arg(&channels)
        .launch_2d(
            dst_width,
            src_height,
            make_config(dst_width, src_height, block_dim),
        )
        .map_err(|e| CudaResizeError::Cuda(e.to_string()))?;

    kernel_v
        .launch_builder(stream)
        .arg(&hbuf)
        .arg(dst)
        .arg(&d_yofs)
        .arg(&d_yw)
        .arg(&ky)
        .arg(&src_height)
        .arg(&dst_width)
        .arg(&dst_height)
        .arg(&channels)
        .launch_2d(
            dst_width,
            dst_height,
            make_config(dst_width, dst_height, block_dim),
        )
        .map_err(|e| CudaResizeError::Cuda(e.to_string()))
}
