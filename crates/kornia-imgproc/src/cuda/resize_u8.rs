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

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

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

/// Nearest kernel source, specialized per channel count: pure gather, four
/// output pixels per thread with the channel loop fully unrolled so every
/// load issues independently.
fn nearest_src(channels: usize) -> String {
    // Per-C gather bodies: 4 output pixels per thread, output assembled in
    // registers and written as whole u32 words when the destination offset
    // is 4-byte aligned (checked at runtime; byte-store fallback otherwise).
    // C1: 4 px = one word. C3: 4 px = 12 B = three words packed from the
    // gathered bytes. C4: each px is naturally one word.
    let body = match channels {
        1 => r#"
        unsigned int b0 = (unsigned int)__ldg(&src[row + (size_t)__ldg(&xmap[x0 + 0u])]);
        unsigned int b1 = (unsigned int)__ldg(&src[row + (size_t)__ldg(&xmap[x0 + 1u])]);
        unsigned int b2 = (unsigned int)__ldg(&src[row + (size_t)__ldg(&xmap[x0 + 2u])]);
        unsigned int b3 = (unsigned int)__ldg(&src[row + (size_t)__ldg(&xmap[x0 + 3u])]);
        unsigned int w0 = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
        if (((size_t)(dst + d) & 3u) == 0u) {
            *(unsigned int*)(dst + d) = w0;
        } else {
            dst[d + 0] = (unsigned char)b0;
            dst[d + 1] = (unsigned char)b1;
            dst[d + 2] = (unsigned char)b2;
            dst[d + 3] = (unsigned char)b3;
        }
"#
        .to_string(),
        3 => r#"
        size_t s0 = row3 + (size_t)__ldg(&xmap[x0 + 0u]) * 3u;
        size_t s1 = row3 + (size_t)__ldg(&xmap[x0 + 1u]) * 3u;
        size_t s2 = row3 + (size_t)__ldg(&xmap[x0 + 2u]) * 3u;
        size_t s3 = row3 + (size_t)__ldg(&xmap[x0 + 3u]) * 3u;
        unsigned int a0 = (unsigned int)__ldg(&src[s0 + 0u]);
        unsigned int a1 = (unsigned int)__ldg(&src[s0 + 1u]);
        unsigned int a2 = (unsigned int)__ldg(&src[s0 + 2u]);
        unsigned int b0 = (unsigned int)__ldg(&src[s1 + 0u]);
        unsigned int b1 = (unsigned int)__ldg(&src[s1 + 1u]);
        unsigned int b2 = (unsigned int)__ldg(&src[s1 + 2u]);
        unsigned int c0 = (unsigned int)__ldg(&src[s2 + 0u]);
        unsigned int c1 = (unsigned int)__ldg(&src[s2 + 1u]);
        unsigned int c2 = (unsigned int)__ldg(&src[s2 + 2u]);
        unsigned int e0 = (unsigned int)__ldg(&src[s3 + 0u]);
        unsigned int e1 = (unsigned int)__ldg(&src[s3 + 1u]);
        unsigned int e2 = (unsigned int)__ldg(&src[s3 + 2u]);
        if (((size_t)(dst + d) & 3u) == 0u) {
            *(unsigned int*)(dst + d)      = a0 | (a1 << 8) | (a2 << 16) | (b0 << 24);
            *(unsigned int*)(dst + d + 4)  = b1 | (b2 << 8) | (c0 << 16) | (c1 << 24);
            *(unsigned int*)(dst + d + 8)  = c2 | (e0 << 8) | (e1 << 16) | (e2 << 24);
        } else {
            dst[d + 0]  = (unsigned char)a0;  dst[d + 1]  = (unsigned char)a1;
            dst[d + 2]  = (unsigned char)a2;  dst[d + 3]  = (unsigned char)b0;
            dst[d + 4]  = (unsigned char)b1;  dst[d + 5]  = (unsigned char)b2;
            dst[d + 6]  = (unsigned char)c0;  dst[d + 7]  = (unsigned char)c1;
            dst[d + 8]  = (unsigned char)c2;  dst[d + 9]  = (unsigned char)e0;
            dst[d + 10] = (unsigned char)e1;  dst[d + 11] = (unsigned char)e2;
        }
"#
        .to_string(),
        4 => r#"
        #pragma unroll
        for (unsigned int px = 0; px < 4u; ++px) {
            size_t so = rowC + (size_t)__ldg(&xmap[x0 + px]) * C;
            unsigned int w = (unsigned int)__ldg(&src[so + 0u])
                           | ((unsigned int)__ldg(&src[so + 1u]) << 8)
                           | ((unsigned int)__ldg(&src[so + 2u]) << 16)
                           | ((unsigned int)__ldg(&src[so + 3u]) << 24);
            if (((size_t)(dst + d + (size_t)px * 4u) & 3u) == 0u) {
                *(unsigned int*)(dst + d + (size_t)px * 4u) = w;
            } else {
                dst[d + px * 4u + 0u] = (unsigned char)(w & 0xffu);
                dst[d + px * 4u + 1u] = (unsigned char)((w >> 8) & 0xffu);
                dst[d + px * 4u + 2u] = (unsigned char)((w >> 16) & 0xffu);
                dst[d + px * 4u + 3u] = (unsigned char)((w >> 24) & 0xffu);
            }
        }
"#
        .to_string(),
        // Any other channel count: plain per-byte gather. The 4-byte packed
        // form above is only correct when a pixel is exactly one word — the
        // old catch-all applied it to every C, which over-read the source
        // and stored at the wrong stride for C != 4.
        _ => r#"
        #pragma unroll
        for (unsigned int px = 0; px < 4u; ++px) {
            size_t so = rowC + (size_t)__ldg(&xmap[x0 + px]) * C;
            size_t dd = d + (size_t)px * C;
            #pragma unroll
            for (unsigned int ch = 0; ch < C; ++ch) {
                dst[dd + ch] = __ldg(&src[so + ch]);
            }
        }
"#
        .to_string(),
    };
    format!(
        r#"
#define C {channels}
extern "C" __global__ void resize_u8_nearest_c{channels}(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__       dst,
    const int* __restrict__           xmap,
    const int* __restrict__           ymap,
    unsigned int src_w,
    unsigned int dst_w,
    unsigned int dst_h
) {{
    unsigned int x0 = 4u * (blockIdx.x * blockDim.x + threadIdx.x);
    unsigned int y  = blockIdx.y * blockDim.y + threadIdx.y;
    if (y >= dst_h) return;
    size_t row  = (size_t)__ldg(&ymap[y]) * src_w;
    size_t row3 = row * 3u;
    size_t rowC = row * C;
    (void)row3; (void)rowC;
    size_t d = ((size_t)y * dst_w + x0) * C;

    if (x0 + 3u < dst_w) {{
{body}        return;
    }}
    if (x0 >= dst_w) return;
    for (unsigned int x = x0; x < dst_w; ++x) {{
        size_t so = (row + (size_t)__ldg(&xmap[x])) * C;
        #pragma unroll
        for (unsigned int ch = 0; ch < C; ++ch) {{
            dst[((size_t)y * dst_w + x) * C + ch] = __ldg(&src[so + ch]);
        }}
    }}
}}
"#
    )
}

/// Q14 bilinear kernel source, specialized per channel count.
///
/// Byte-exact with `bilinear_row_u8`: the blend runs in 64-bit with ONE
/// round-half-up at the final Q28 shift; fx/fy come from the host f64 LUTs
/// and `fx1 = 16384 - fx` is exact integer arithmetic.
///
/// The channel count is baked into the source (`#pragma unroll` on a
/// compile-time bound), and each thread produces TWO output pixels, so all
/// corner loads are named registers issuing independently — the runtime
/// channel loop of the first version serialized them (the same
/// latency-bound profile the morphology kernels had before specialization).
fn bilinear_src(channels: usize) -> String {
    format!(
        r#"
#define C {channels}
extern "C" __global__ void resize_u8_bilinear_c{channels}(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__       dst,
    const unsigned int* __restrict__  xofs,
    const unsigned int* __restrict__  xfx,
    const unsigned int* __restrict__  yofs,
    const unsigned int* __restrict__  yfy,
    unsigned int src_w,
    unsigned int dst_w,
    unsigned int dst_h
) {{
    // One pixel per thread (occupancy beats ILP for gather kernels — the
    // 2-px pair variant measured no better than 1-px, same as nearest's
    // 4-px quad); the baked channel count unrolls the corner loads into
    // independent issues.
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;

    const unsigned long long SCALE = 16384ull;
    unsigned long long fy  = (unsigned long long)__ldg(&yfy[y]);
    unsigned long long fy1 = SCALE - fy;
    unsigned long long fx  = (unsigned long long)__ldg(&xfx[x]);
    unsigned long long fx1 = SCALE - fx;
    size_t row0 = (size_t)__ldg(&yofs[y]) * src_w * C;
    size_t row1 = row0 + (size_t)src_w * C;
    size_t c0   = (size_t)__ldg(&xofs[x]) * C;
    size_t d    = ((size_t)y * dst_w + x) * C;

    #pragma unroll
    for (unsigned int ch = 0; ch < C; ++ch) {{
        unsigned long long p00 = (unsigned long long)__ldg(&src[row0 + c0 + ch]);
        unsigned long long p01 = (unsigned long long)__ldg(&src[row0 + c0 + C + ch]);
        unsigned long long p10 = (unsigned long long)__ldg(&src[row1 + c0 + ch]);
        unsigned long long p11 = (unsigned long long)__ldg(&src[row1 + c0 + C + ch]);
        unsigned long long top = p00 * fx1 + p01 * fx;
        unsigned long long bot = p10 * fx1 + p11 * fx;
        dst[d + ch] = (unsigned char)((top * fy1 + bot * fy + (1ull << 27)) >> 28);
    }}
}}
"#
    )
}

/// Separable-pass kernel sources, specialized per channel count and —
/// when the element is narrow enough (`k <= 32`) — per tap count, so both
/// loops fully unroll into independent load issues. Wide antialiased
/// kernels (k > 32) keep a runtime tap loop (KTAPS == 0 variant).
/// Byte-exact either way: same i32 accumulation order, same
/// round-half-up, same i16 clamp.
fn sep_h_src(channels: usize, ktaps: usize) -> String {
    let (loop_head, bound) = if ktaps > 0 {
        ("#pragma unroll".to_string(), format!("{ktaps}u"))
    } else {
        (String::new(), "kx".to_string())
    };
    format!(
        r#"
#define C {channels}
extern "C" __global__ void resize_u8_sep_h_c{channels}_k{ktaps}(
    const unsigned char* __restrict__  src,
    short* __restrict__                hbuf,
    const unsigned short* __restrict__ xsrc,
    const short* __restrict__          xw,
    unsigned int kx,
    unsigned int src_w,
    unsigned int dst_w,
    unsigned int src_h
) {{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= src_h) return;

    size_t row = (size_t)y * src_w * C;
    size_t ibase = (size_t)x * kx;
    size_t d = ((size_t)y * dst_w + x) * C;

    int acc[C] = {{0}};
    {loop_head}
    for (unsigned int t = 0; t < {bound}; ++t) {{
        int w = (int)__ldg(&xw[ibase + t]);
        const unsigned char* p =
            src + row + (size_t)__ldg(&xsrc[ibase + t]) * C;
        #pragma unroll
        for (unsigned int ch = 0; ch < C; ++ch) {{
            acc[ch] += (int)__ldg(&p[ch]) * w;
        }}
    }}
    #pragma unroll
    for (unsigned int ch = 0; ch < C; ++ch) {{
        int v = (acc[ch] + (1 << 13)) >> 14;
        v = min(max(v, -32768), 32767);
        hbuf[d + ch] = (short)v;
    }}
}}
"#
    )
}

fn sep_v_src(channels: usize, ktaps: usize) -> String {
    let (loop_head, bound) = if ktaps > 0 {
        ("#pragma unroll".to_string(), format!("{ktaps}u"))
    } else {
        (String::new(), "ky".to_string())
    };
    format!(
        r#"
#define C {channels}
extern "C" __global__ void resize_u8_sep_v_c{channels}_k{ktaps}(
    const short* __restrict__ hbuf,
    unsigned char* __restrict__ dst,
    const int* __restrict__     yofs,
    const short* __restrict__   yw,
    unsigned int ky,
    unsigned int src_h,
    unsigned int dst_w,
    unsigned int dst_h
) {{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;

    int y0 = __ldg(&yofs[y]);
    size_t wbase = (size_t)y * ky;
    size_t d = ((size_t)y * dst_w + x) * C;

    int acc[C] = {{0}};
    {loop_head}
    for (unsigned int k = 0; k < {bound}; ++k) {{
        int w = (int)__ldg(&yw[wbase + k]);
        int sy = min(max(y0 + (int)k, 0), (int)src_h - 1);
        const short* p = hbuf + ((size_t)sy * dst_w + x) * C;
        #pragma unroll
        for (unsigned int ch = 0; ch < C; ++ch) {{
            acc[ch] += (int)__ldg(&p[ch]) * w;
        }}
    }}
    #pragma unroll
    for (unsigned int ch = 0; ch < C; ++ch) {{
        int v = (acc[ch] + (1 << 13)) >> 14;
        dst[d + ch] = (unsigned char)min(max(v, 0), 255);
    }}
}}
"#
    )
}

static PYRDOWN2X_KERNEL: OnceLock<Result<CudaKernel, String>> = OnceLock::new();
static PYRUP2X_KERNEL: OnceLock<Result<CudaKernel, String>> = OnceLock::new();
type PerCKernelCache = Mutex<HashMap<u32, Arc<CudaKernel>>>;
static NEAREST_KERNELS: OnceLock<PerCKernelCache> = OnceLock::new();
/// Per-channel-count specialized bilinear kernels (C is baked into the
/// source so the pixel loop fully unrolls).
static BILINEAR_KERNELS: OnceLock<PerCKernelCache> = OnceLock::new();
type SepKernelCache = Mutex<HashMap<(u32, u32, bool), Arc<CudaKernel>>>;
static SEP_KERNELS: OnceLock<SepKernelCache> = OnceLock::new();
/// Tap counts up to this are baked into specialized kernels; wider
/// (strongly antialiased) elements use the runtime-loop variant.
const SEP_MAX_BAKED_TAPS: u32 = 32;

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

/// Nearest-neighbor u8 resize via device index LUTs (pure gather).
///
/// `xmap`/`ymap` are device uploads of `nearest_axis_lut` output (the
/// `resize::cuda` adapter caches them per geometry — per-call pageable H2D
/// uploads on Jetson have a catastrophic latency tail, ~250 µs average).
/// Entries must lie in `[0, src_len-1]` (the builders clamp).
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
    xmap: &CudaSlice<i32>,
    ymap: &CudaSlice<i32>,
    block_dim: Option<(u32, u32)>,
) -> Result<(), CudaResizeError> {
    super::check_geometry(src_width, src_height, dst_width, dst_height, block_dim)
        .map_err(CudaResizeError::Cuda)?;
    if channels == 0 {
        return Err(CudaResizeError::Cuda("channels must be at least 1".into()));
    }
    check_dst_len(dst.len(), dst_width, dst_height, channels)?;
    if xmap.len() != dst_width as usize || ymap.len() != dst_height as usize {
        return Err(CudaResizeError::Cuda(
            "xmap/ymap length must equal dst width/height".into(),
        ));
    }

    // Per-C specialized kernel (see bilinear for the cache pattern).
    let cache = NEAREST_KERNELS.get_or_init(Default::default);
    let cached = cache
        .lock()
        .expect("nearest kernel cache poisoned")
        .get(&channels)
        .cloned();
    let kernel = if let Some(hit) = cached {
        hit
    } else {
        let src_code = nearest_src(channels as usize);
        let name = format!("resize_u8_nearest_c{channels}");
        let built =
            Arc::new(try_compile_with_l1(ctx, &src_code, &name).map_err(CudaResizeError::Cuda)?);
        cache
            .lock()
            .expect("nearest kernel cache poisoned")
            .entry(channels)
            .or_insert(built)
            .clone()
    };

    // Four output pixels per thread in x.
    let grid_w = dst_width.div_ceil(4);
    kernel
        .launch_builder(stream)
        .arg(src)
        .arg(dst)
        .arg(xmap)
        .arg(ymap)
        .arg(&src_width)
        .arg(&dst_width)
        .arg(&dst_height)
        .launch_2d(
            grid_w,
            dst_height,
            make_config(grid_w, dst_height, block_dim),
        )
        .map_err(|e| CudaResizeError::Cuda(e.to_string()))
}

/// Q14 bilinear u8 resize via device tap LUTs.
///
/// `xofs`/`xfx` and `yofs`/`yfy` are device uploads of `bilinear_axis_lut`
/// output (cached per geometry by the `resize::cuda` adapter): offsets
/// clamped to `[0, src_len-2]`, weights in `[0, 16384]`. Requires a source
/// of at least 2×2 (the `ofs+1` neighbor read).
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
    xofs: &CudaSlice<u32>,
    xfx: &CudaSlice<u32>,
    yofs: &CudaSlice<u32>,
    yfy: &CudaSlice<u32>,
    block_dim: Option<(u32, u32)>,
) -> Result<(), CudaResizeError> {
    super::check_geometry(src_width, src_height, dst_width, dst_height, block_dim)
        .map_err(CudaResizeError::Cuda)?;
    if !(channels == 1 || channels == 3 || channels == 4) {
        return Err(CudaResizeError::Cuda(
            "bilinear supports 1/3/4-channel u8 images".into(),
        ));
    }
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

    // Per-C specialized kernel (scoped-guard lookup: binding `.get()` in an
    // `if let` scrutinee would hold the lock through the else branch).
    let cache = BILINEAR_KERNELS.get_or_init(Default::default);
    let cached = cache
        .lock()
        .expect("bilinear kernel cache poisoned")
        .get(&channels)
        .cloned();
    let kernel = if let Some(hit) = cached {
        hit
    } else {
        let src_code = bilinear_src(channels as usize);
        let name = format!("resize_u8_bilinear_c{channels}");
        let built =
            Arc::new(try_compile_with_l1(ctx, &src_code, &name).map_err(CudaResizeError::Cuda)?);
        cache
            .lock()
            .expect("bilinear kernel cache poisoned")
            .entry(channels)
            .or_insert(built)
            .clone()
    };

    let grid_w = dst_width;
    kernel
        .launch_builder(stream)
        .arg(src)
        .arg(dst)
        .arg(xofs)
        .arg(xfx)
        .arg(yofs)
        .arg(yfy)
        .arg(&src_width)
        .arg(&dst_width)
        .arg(&dst_height)
        .launch_2d(
            grid_w,
            dst_height,
            make_config(grid_w, dst_height, block_dim),
        )
        .map_err(|e| CudaResizeError::Cuda(e.to_string()))
}

/// Q14 separable bicubic / lanczos u8 resize (two passes, i16 intermediate).
///
/// Device tables are uploads of `precompute_contribs` + `build_xsrc_lut` +
/// `pack_xw_i16` output — including the antialias kernel widening — cached
/// per geometry by the `resize::cuda` adapter. The i16 scratch
/// (`src_h × dst_w × channels`) is stream-ordered allocated per call; with
/// the mempool release threshold raised, steady-state cost is a pool lookup.
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
    xsrc: &CudaSlice<u16>,
    xw: &CudaSlice<i16>,
    kx: u32,
    yofs: &CudaSlice<i32>,
    yw: &CudaSlice<i16>,
    ky: u32,
    block_dim: Option<(u32, u32)>,
) -> Result<(), CudaResizeError> {
    super::check_geometry(src_width, src_height, dst_width, dst_height, block_dim)
        .map_err(CudaResizeError::Cuda)?;
    if !(channels == 1 || channels == 3 || channels == 4) {
        return Err(CudaResizeError::Cuda(
            "separable resize supports 1/3/4-channel u8 images".into(),
        ));
    }
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

    // Per-(C, taps) specialized kernels; wide antialiased elements fall back
    // to the runtime-loop variant (ktaps = 0). Scoped-guard cache lookups.
    let sep_kernel = |horizontal: bool, k: u32| -> Result<Arc<CudaKernel>, CudaResizeError> {
        let baked = if k <= SEP_MAX_BAKED_TAPS { k } else { 0 };
        let cache = SEP_KERNELS.get_or_init(Default::default);
        let key = (channels, baked, horizontal);
        let cached = cache
            .lock()
            .expect("separable kernel cache poisoned")
            .get(&key)
            .cloned();
        if let Some(hit) = cached {
            return Ok(hit);
        }
        let (src_code, name) = if horizontal {
            (
                sep_h_src(channels as usize, baked as usize),
                format!("resize_u8_sep_h_c{channels}_k{baked}"),
            )
        } else {
            (
                sep_v_src(channels as usize, baked as usize),
                format!("resize_u8_sep_v_c{channels}_k{baked}"),
            )
        };
        let built =
            Arc::new(try_compile_with_l1(ctx, &src_code, &name).map_err(CudaResizeError::Cuda)?);
        Ok(cache
            .lock()
            .expect("separable kernel cache poisoned")
            .entry(key)
            .or_insert(built)
            .clone())
    };
    let kernel_h = sep_kernel(true, kx)?;
    let kernel_v = sep_kernel(false, ky)?;

    // Intermediate: src_h rows of dst_w pixels, i16. Pass 1 writes every
    // element before pass 2 reads; no zero-fill needed.
    let inter_len = (src_height as usize) * (dst_width as usize) * (channels as usize);
    let mut hbuf = unsafe { stream.alloc::<i16>(inter_len) }
        .map_err(|e| CudaResizeError::Cuda(e.to_string()))?;

    kernel_h
        .launch_builder(stream)
        .arg(src)
        .arg(&mut hbuf)
        .arg(xsrc)
        .arg(xw)
        .arg(&kx)
        .arg(&src_width)
        .arg(&dst_width)
        .arg(&src_height)
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
        .arg(yofs)
        .arg(yw)
        .arg(&ky)
        .arg(&src_height)
        .arg(&dst_width)
        .arg(&dst_height)
        .launch_2d(
            dst_width,
            dst_height,
            make_config(dst_width, dst_height, block_dim),
        )
        .map_err(|e| CudaResizeError::Cuda(e.to_string()))
}
