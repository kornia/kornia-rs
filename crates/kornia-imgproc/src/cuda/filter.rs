//! CUDA separable-filter engine (f32 skip-zero + u8 Q8 + 3×3 binomial u8).
//!
//! Three kernel families, each the textual mirror of a CPU path:
//!
//! * **f32 H/V pair** — mirrors `filter/separable_filter.rs`: taps applied in
//!   sequential order with plain `acc += v * k` (compiled `--fmad=false`), and
//!   out-of-bounds taps SKIPPED (constant-zero border, no renormalize). The
//!   intermediate is a full f32 image, matching the CPU `temp` buffer.
//!   Bit-exact with the CPU engine by construction.
//! * **u8 Q8 H/V pair** — mirrors `filter/ops.rs::separable_blur_u8_striped`:
//!   host-quantized `quantize_kernel_256` weights, replicate-clamped borders,
//!   `(acc + 128) >> 8` rounding per pass with a u8 intermediate.
//! * **u8 3×3 binomial** — mirrors the `[1,2,1]/4` fast path's halving-add
//!   nesting `rhadd(rhadd(a,b), rhadd(b,d))` with replicate borders.
//!
//! House rules per the codebase: per-C NVRTC codegen with baked tap counts
//! (runtime-loop fallback above [`SEP_FILTER_MAX_BAKED_TAPS`]), scoped-guard
//! kernel caches with single-entry eviction, no shared-memory tiling
//! (occupancy beats staging on Orin — measured repeatedly), and pub launchers
//! that validate everything the kernels assume.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use kornia_tensor::CudaKernel;

use super::{make_config, try_compile_with_l1};

/// Above this tap count the runtime-loop kernel variant is used.
pub const SEP_FILTER_MAX_BAKED_TAPS: u32 = 32;

/// Error type for the CUDA filter launchers.
#[derive(Debug, thiserror::Error)]
pub enum CudaFilterError {
    /// CUDA driver / compile / launch error.
    #[error("CUDA filter error: {0}")]
    Cuda(String),
    /// A slice is smaller than required.
    #[error("device slice '{what}' length {got} < required {need}")]
    SliceTooSmall {
        /// Which operand was too small.
        what: &'static str,
        /// Actual length (elements).
        got: usize,
        /// Required length (elements).
        need: usize,
    },
}

fn check_slice(what: &'static str, got: usize, need: usize) -> Result<(), CudaFilterError> {
    if got < need {
        return Err(CudaFilterError::SliceTooSmall { what, got, need });
    }
    Ok(())
}

// ── f32 kernels ──────────────────────────────────────────────────────────────

/// f32 horizontal/vertical pass source. `horizontal` picks the axis; taps are
/// applied in ascending order with plain multiply-add (`--fmad=false` at
/// compile keeps the expression tree identical to the CPU's `acc += v * k`).
/// Out-of-bounds taps are skipped — the CPU engine's constant-zero border.
fn sep_f32_src(channels: usize, ktaps: usize, horizontal: bool) -> String {
    let (loop_head, bound) = if ktaps > 0 {
        ("#pragma unroll".to_string(), format!("{ktaps}u"))
    } else {
        (String::new(), "ktaps".to_string())
    };
    let axis = if horizontal { "h" } else { "v" };
    let coord = if horizontal {
        r#"
        int t = (int)x + (int)ti - half;
        bool inb = (t >= 0 && t < (int)cols);
        size_t si = inb ? (((size_t)y * cols + (size_t)t) * C) : 0;
"#
    } else {
        r#"
        int t = (int)y + (int)ti - half;
        bool inb = (t >= 0 && t < (int)rows);
        size_t si = inb ? (((size_t)t * cols + (size_t)x) * C) : 0;
"#
    };
    format!(
        r#"
#define C {channels}
extern "C" __global__ void sep_filter_f32_{axis}_c{channels}_k{ktaps}(
    const float* __restrict__ src,
    float* __restrict__       dst,
    const float* __restrict__ taps,
    unsigned int ktaps,
    unsigned int cols,
    unsigned int rows
) {{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cols || y >= rows) return;
    int half = (int)(ktaps / 2u);

    float acc[C];
    #pragma unroll
    for (unsigned int ch = 0; ch < C; ++ch) acc[ch] = 0.0f;

    {loop_head}
    for (unsigned int ti = 0; ti < {bound}; ++ti) {{
        float k = __ldg(&taps[ti]);
{coord}
        if (inb) {{
            #pragma unroll
            for (unsigned int ch = 0; ch < C; ++ch) {{
                acc[ch] += __ldg(&src[si + ch]) * k;
            }}
        }}
    }}
    size_t d = ((size_t)y * cols + (size_t)x) * C;
    #pragma unroll
    for (unsigned int ch = 0; ch < C; ++ch) {{
        dst[d + ch] = acc[ch];
    }}
}}
"#
    )
}

// ── u8 Q8 kernels ────────────────────────────────────────────────────────────

/// u8 Q8 pass source. Mirrors `hpass_u8_row`/the striped V-pass: replicate-
/// clamped taps, `(acc + 128) >> 8` rounding, u8 output per pass.
fn sep_u8q8_src(channels: usize, ktaps: usize, horizontal: bool) -> String {
    let (loop_head, bound) = if ktaps > 0 {
        ("#pragma unroll".to_string(), format!("{ktaps}u"))
    } else {
        (String::new(), "ktaps".to_string())
    };
    let axis = if horizontal { "h" } else { "v" };
    let coord = if horizontal {
        r#"
        int t = min(max((int)x + (int)ti - half, 0), (int)cols - 1);
        size_t si = ((size_t)y * cols + (size_t)t) * C;
"#
    } else {
        r#"
        int t = min(max((int)y + (int)ti - half, 0), (int)rows - 1);
        size_t si = ((size_t)t * cols + (size_t)x) * C;
"#
    };
    format!(
        r#"
#define C {channels}
extern "C" __global__ void sep_filter_u8_{axis}_c{channels}_k{ktaps}(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__       dst,
    const unsigned char* __restrict__ taps,
    unsigned int ktaps,
    unsigned int cols,
    unsigned int rows
) {{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cols || y >= rows) return;
    int half = (int)(ktaps / 2u);

    unsigned int acc[C];
    #pragma unroll
    for (unsigned int ch = 0; ch < C; ++ch) acc[ch] = 0u;

    {loop_head}
    for (unsigned int ti = 0; ti < {bound}; ++ti) {{
        unsigned int k = (unsigned int)__ldg(&taps[ti]);
{coord}
        #pragma unroll
        for (unsigned int ch = 0; ch < C; ++ch) {{
            acc[ch] += (unsigned int)__ldg(&src[si + ch]) * k;
        }}
    }}
    size_t d = ((size_t)y * cols + (size_t)x) * C;
    #pragma unroll
    for (unsigned int ch = 0; ch < C; ++ch) {{
        dst[d + ch] = (unsigned char)((acc[ch] + 128u) >> 8);
    }}
}}
"#
    )
}

// ── u8 3×3 binomial kernels ──────────────────────────────────────────────────

/// `[1,2,1]/4` pass via the same halving-add nesting the CPU NEON bulk uses:
/// `rhadd(rhadd(a, b), rhadd(b, d))` with `rhadd(x, y) = (x + y + 1) >> 1`,
/// replicate borders. One source per axis.
fn binomial3_src(channels: usize, horizontal: bool) -> String {
    let axis = if horizontal { "h" } else { "v" };
    let coord = if horizontal {
        r#"
    int xm = max((int)x - 1, 0);
    int xp = min((int)x + 1, (int)cols - 1);
    size_t ia = ((size_t)y * cols + (size_t)xm) * C;
    size_t ib = ((size_t)y * cols + (size_t)x) * C;
    size_t id = ((size_t)y * cols + (size_t)xp) * C;
"#
    } else {
        r#"
    int ym = max((int)y - 1, 0);
    int yp = min((int)y + 1, (int)rows - 1);
    size_t ia = ((size_t)ym * cols + (size_t)x) * C;
    size_t ib = ((size_t)y * cols + (size_t)x) * C;
    size_t id = ((size_t)yp * cols + (size_t)x) * C;
"#
    };
    format!(
        r#"
#define C {channels}
__device__ __forceinline__ unsigned int rhadd_u8(unsigned int a, unsigned int b)
{{
    return (a + b + 1u) >> 1;
}}

extern "C" __global__ void binomial3_u8_{axis}_c{channels}(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__       dst,
    unsigned int cols,
    unsigned int rows
) {{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cols || y >= rows) return;
{coord}
    #pragma unroll
    for (unsigned int ch = 0; ch < C; ++ch) {{
        unsigned int a = (unsigned int)__ldg(&src[ia + ch]);
        unsigned int b = (unsigned int)__ldg(&src[ib + ch]);
        unsigned int d = (unsigned int)__ldg(&src[id + ch]);
        dst[((size_t)y * cols + (size_t)x) * C + ch] =
            (unsigned char)rhadd_u8(rhadd_u8(a, b), rhadd_u8(b, d));
    }}
}}
"#
    )
}

// ── Kernel cache ─────────────────────────────────────────────────────────────

type SepKernelCache = Mutex<HashMap<(u32, u32, bool, u8), Arc<CudaKernel>>>;
static SEP_FILTER_KERNELS: OnceLock<SepKernelCache> = OnceLock::new();
const SEP_FILTER_CACHE_CAP: usize = 64;

const VAR_F32: u8 = 0;
const VAR_U8Q8: u8 = 1;
const VAR_BINOMIAL: u8 = 2;

fn get_or_compile(
    ctx: &Arc<CudaContext>,
    key: (u32, u32, bool, u8),
    name: &str,
    src_code: &str,
) -> Result<Arc<CudaKernel>, CudaFilterError> {
    let cache = SEP_FILTER_KERNELS.get_or_init(Default::default);
    let cached = cache
        .lock()
        .expect("separable filter kernel cache poisoned")
        .get(&key)
        .cloned();
    if let Some(hit) = cached {
        return Ok(hit);
    }
    let built = Arc::new(try_compile_with_l1(ctx, src_code, name).map_err(CudaFilterError::Cuda)?);
    let mut map = cache
        .lock()
        .expect("separable filter kernel cache poisoned");
    if map.len() >= SEP_FILTER_CACHE_CAP {
        // Evict one entry, not the whole map (stampede guard).
        if let Some(k) = map.keys().next().copied() {
            map.remove(&k);
        }
    }
    Ok(map.entry(key).or_insert(built).clone())
}

fn baked(k: u32) -> u32 {
    if k <= SEP_FILTER_MAX_BAKED_TAPS {
        k
    } else {
        0
    }
}

// ── Launchers ────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)] // mirrors the kernel's parameter surface
fn launch_sep_f32_pass(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    taps: &CudaSlice<f32>,
    ktaps: u32,
    cols: u32,
    rows: u32,
    channels: u32,
    horizontal: bool,
) -> Result<(), CudaFilterError> {
    let kb = baked(ktaps);
    let axis = if horizontal { "h" } else { "v" };
    let name = format!("sep_filter_f32_{axis}_c{channels}_k{kb}");
    let src_code = sep_f32_src(channels as usize, kb as usize, horizontal);
    let kernel = get_or_compile(ctx, (channels, kb, horizontal, VAR_F32), &name, &src_code)?;
    kernel
        .launch_builder(stream)
        .arg(src)
        .arg(dst)
        .arg(taps)
        .arg(&ktaps)
        .arg(&cols)
        .arg(&rows)
        .launch_2d(cols, rows, make_config(cols, rows, None))
        .map_err(|e| CudaFilterError::Cuda(e.to_string()))
}

/// f32 separable filter: H pass into a caller-provided f32 scratch, V pass
/// into `dst`. Bit-exact with `filter/separable_filter.rs::separable_filter`
/// (skip-zero border, sequential taps, `--fmad=false`).
///
/// `scratch` must hold `cols * rows * channels` f32 (same as src/dst).
#[allow(clippy::too_many_arguments)] // mirrors the kernel's parameter surface
pub fn launch_separable_filter_f32(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    scratch: &mut CudaSlice<f32>,
    kx: &CudaSlice<f32>,
    kx_len: u32,
    ky: &CudaSlice<f32>,
    ky_len: u32,
    cols: u32,
    rows: u32,
    channels: u32,
) -> Result<(), CudaFilterError> {
    super::check_geometry(cols, rows, cols, rows, None).map_err(CudaFilterError::Cuda)?;
    if channels == 0 || kx_len == 0 || ky_len == 0 {
        return Err(CudaFilterError::Cuda(
            "channels and tap counts must be at least 1".into(),
        ));
    }
    let n = cols as usize * rows as usize * channels as usize;
    check_slice("src", src.len(), n)?;
    check_slice("dst", dst.len(), n)?;
    check_slice("scratch", scratch.len(), n)?;
    check_slice("kx", kx.len(), kx_len as usize)?;
    check_slice("ky", ky.len(), ky_len as usize)?;

    launch_sep_f32_pass(
        ctx, stream, src, scratch, kx, kx_len, cols, rows, channels, true,
    )?;
    launch_sep_f32_pass(
        ctx, stream, scratch, dst, ky, ky_len, cols, rows, channels, false,
    )
}

#[allow(clippy::too_many_arguments)] // mirrors the kernel's parameter surface
fn launch_sep_u8q8_pass(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<u8>,
    dst: &mut CudaSlice<u8>,
    taps: &CudaSlice<u8>,
    ktaps: u32,
    cols: u32,
    rows: u32,
    channels: u32,
    horizontal: bool,
) -> Result<(), CudaFilterError> {
    let kb = baked(ktaps);
    let axis = if horizontal { "h" } else { "v" };
    let name = format!("sep_filter_u8_{axis}_c{channels}_k{kb}");
    let src_code = sep_u8q8_src(channels as usize, kb as usize, horizontal);
    let kernel = get_or_compile(ctx, (channels, kb, horizontal, VAR_U8Q8), &name, &src_code)?;
    kernel
        .launch_builder(stream)
        .arg(src)
        .arg(dst)
        .arg(taps)
        .arg(&ktaps)
        .arg(&cols)
        .arg(&rows)
        .launch_2d(cols, rows, make_config(cols, rows, None))
        .map_err(|e| CudaFilterError::Cuda(e.to_string()))
}

/// u8 Q8 separable blur: H pass into a u8 scratch, V pass into `dst`.
/// Byte-exact with `separable_blur_u8_striped` (replicate borders,
/// `(acc + 128) >> 8` per pass, u8 intermediate). Taps are the
/// `quantize_kernel_256` weights, uploaded by the adapter.
#[allow(clippy::too_many_arguments)] // mirrors the kernel's parameter surface
pub fn launch_separable_blur_u8q8(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<u8>,
    dst: &mut CudaSlice<u8>,
    scratch: &mut CudaSlice<u8>,
    kx: &CudaSlice<u8>,
    kx_len: u32,
    ky: &CudaSlice<u8>,
    ky_len: u32,
    cols: u32,
    rows: u32,
    channels: u32,
) -> Result<(), CudaFilterError> {
    super::check_geometry(cols, rows, cols, rows, None).map_err(CudaFilterError::Cuda)?;
    if channels == 0 || kx_len == 0 || ky_len == 0 {
        return Err(CudaFilterError::Cuda(
            "channels and tap counts must be at least 1".into(),
        ));
    }
    let n = cols as usize * rows as usize * channels as usize;
    check_slice("src", src.len(), n)?;
    check_slice("dst", dst.len(), n)?;
    check_slice("scratch", scratch.len(), n)?;
    check_slice("kx", kx.len(), kx_len as usize)?;
    check_slice("ky", ky.len(), ky_len as usize)?;

    launch_sep_u8q8_pass(
        ctx, stream, src, scratch, kx, kx_len, cols, rows, channels, true,
    )?;
    launch_sep_u8q8_pass(
        ctx, stream, scratch, dst, ky, ky_len, cols, rows, channels, false,
    )
}

/// u8 3×3 binomial blur (`[1,2,1]/4` per axis via nested halving-adds,
/// replicate borders) — the GPU twin of the CPU binomial fast path.
#[allow(clippy::too_many_arguments)] // mirrors the kernel's parameter surface
pub fn launch_binomial3_u8(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<u8>,
    dst: &mut CudaSlice<u8>,
    scratch: &mut CudaSlice<u8>,
    cols: u32,
    rows: u32,
    channels: u32,
) -> Result<(), CudaFilterError> {
    super::check_geometry(cols, rows, cols, rows, None).map_err(CudaFilterError::Cuda)?;
    if channels == 0 {
        return Err(CudaFilterError::Cuda("channels must be at least 1".into()));
    }
    let n = cols as usize * rows as usize * channels as usize;
    check_slice("src", src.len(), n)?;
    check_slice("dst", dst.len(), n)?;
    check_slice("scratch", scratch.len(), n)?;

    let h_kernel = get_or_compile(
        ctx,
        (channels, 3, true, VAR_BINOMIAL),
        &format!("binomial3_u8_h_c{channels}"),
        &binomial3_src(channels as usize, true),
    )?;
    h_kernel
        .launch_builder(stream)
        .arg(src)
        .arg(&mut *scratch)
        .arg(&cols)
        .arg(&rows)
        .launch_2d(cols, rows, make_config(cols, rows, None))
        .map_err(|e| CudaFilterError::Cuda(e.to_string()))?;

    let v_kernel = get_or_compile(
        ctx,
        (channels, 3, false, VAR_BINOMIAL),
        &format!("binomial3_u8_v_c{channels}"),
        &binomial3_src(channels as usize, false),
    )?;
    v_kernel
        .launch_builder(stream)
        .arg(&*scratch)
        .arg(dst)
        .arg(&cols)
        .arg(&rows)
        .launch_2d(cols, rows, make_config(cols, rows, None))
        .map_err(|e| CudaFilterError::Cuda(e.to_string()))
}

// ── sobel/scharr magnitude ───────────────────────────────────────────────────

static MAGNITUDE_SRC: &str = r#"
extern "C" __global__ void gradient_magnitude_f32(
    const float* __restrict__ gx,
    const float* __restrict__ gy,
    float* __restrict__       dst,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float a = __ldg(&gx[i]);
    float b = __ldg(&gy[i]);
    // Same expression tree as the CPU's (gx*gx + gy*gy).sqrt() under
    // --fmad=false.
    dst[i] = sqrtf(a * a + b * b);
}
"#;

/// `dst[i] = sqrt(gx[i]^2 + gy[i]^2)` — the CPU magnitude fold of
/// `sobel`/`scharr`, mirrored under `--fmad=false`.
pub fn launch_gradient_magnitude_f32(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    gx: &CudaSlice<f32>,
    gy: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    n: usize,
) -> Result<(), CudaFilterError> {
    check_slice("gx", gx.len(), n)?;
    check_slice("gy", gy.len(), n)?;
    check_slice("dst", dst.len(), n)?;
    let kernel = get_or_compile(
        ctx,
        (0, 0, false, 3),
        "gradient_magnitude_f32",
        MAGNITUDE_SRC,
    )?;
    let n_u32 = u32::try_from(n).map_err(|_| CudaFilterError::Cuda("n exceeds u32".into()))?;
    let cfg = cudarc::driver::LaunchConfig {
        block_dim: (256, 1, 1),
        grid_dim: (n_u32.div_ceil(256), 1, 1),
        shared_mem_bytes: 0,
    };
    kernel
        .launch_builder(stream)
        .arg(gx)
        .arg(gy)
        .arg(dst)
        .arg(&n_u32)
        .launch_cfg(cfg)
        .map_err(|e| CudaFilterError::Cuda(e.to_string()))
}
