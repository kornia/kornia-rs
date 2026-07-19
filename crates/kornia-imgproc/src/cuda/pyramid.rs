//! CUDA Gaussian-pyramid kernels — textual mirrors of `pyramid.rs`.
//!
//! * `pyrdown_f32`: ONE fused 5×5-Gaussian + subsample kernel. The 25
//!   weights are the same f32 products the CPU precomputes, baked as exact
//!   bit patterns; interior pixels use direct indexing and the 1-px border
//!   ring uses `reflect_101` on both axes — branch shapes identical to the
//!   CPU's split loops, so output is bit-exact under `--fmad=false`.
//! * `pyrup_f32`: the CPU's polyphase H/V pair (even `(1·p + 6·c + 1·n)/8`,
//!   odd `(c + n)/2`, border phases special-cased) with an f32 intermediate
//!   of `dst_w × src_h` — bit-exact.
//! * `pyrdown_u8`: separable `[1,4,6,4,1]` with a u16 H-intermediate and the
//!   V-pass `(sum + 128) >> 8` (clamped) — byte-exact.
//! * `pyrup_u8`: integer polyphase pair (`(p + 6c + n + 4) >> 3`,
//!   `(c + n + 1) >> 1`) with a u8 intermediate — byte-exact.
//!
//! Launchers take caller-provided scratch so `PyramidPlan` can own every
//! buffer (zero-alloc steady state, graph-capturable); they validate all
//! slice lengths and channel counts (pub-launcher hygiene).

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use kornia_tensor::CudaKernel;

use super::{make_config, try_compile_with_l1};

/// Error type for the CUDA pyramid launchers.
#[derive(Debug, thiserror::Error)]
pub enum CudaPyramidError {
    /// CUDA driver / compile / launch error.
    #[error("CUDA pyramid error: {0}")]
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

fn check_slice(what: &'static str, got: usize, need: usize) -> Result<(), CudaPyramidError> {
    if got < need {
        return Err(CudaPyramidError::SliceTooSmall { what, got, need });
    }
    Ok(())
}

/// `reflect_101` device preamble, shared by every kernel family that
/// samples reflected borders (pyramids, bilateral).
pub(crate) const REFLECT_101: &str = r#"
__device__ __forceinline__ int reflect_101(int p, int len) {
    if (len == 1) return 0;
    if (p < 0) p = -p;
    int period = 2 * (len - 1);
    p %= period;
    if (p >= len) p = period - p;
    return p;
}
"#;

// ── f32 pyrdown (fused 5x5 + subsample) ──────────────────────────────────────

/// The CPU's 25 precomputed `ky_w * kx_w` f32 products as exact bit patterns.
const PYRDOWN_W_BITS: [u32; 25] = [
    0x3b800000, 0x3c800000, 0x3cc00000, 0x3c800000, 0x3b800000, //
    0x3c800000, 0x3d800000, 0x3dc00000, 0x3d800000, 0x3c800000, //
    0x3cc00000, 0x3dc00000, 0x3e100000, 0x3dc00000, 0x3cc00000, //
    0x3c800000, 0x3d800000, 0x3dc00000, 0x3d800000, 0x3c800000, //
    0x3b800000, 0x3c800000, 0x3cc00000, 0x3c800000, 0x3b800000,
];

fn pyrdown_f32_src(channels: usize) -> String {
    let weights: Vec<String> = PYRDOWN_W_BITS
        .iter()
        .map(|b| format!("__uint_as_float(0x{b:08x}u)"))
        .collect();
    let weights = weights.join(", ");
    format!(
        r#"
#define C {channels}
{REFLECT_101}
extern "C" __global__ void pyrdown_f32_c{channels}(
    const float* __restrict__ src,
    float* __restrict__       dst,
    unsigned int src_w,
    unsigned int src_h,
    unsigned int dst_w,
    unsigned int dst_h
) {{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;
    const float kw[25] = {{ {weights} }};

    // Same interior test as the CPU's split loops: the 1-px dst border ring
    // (and everything when dst is tiny) goes through reflect_101.
    bool interior = (x >= 1u && x + 1u < dst_w && y >= 1u && y + 1u < dst_h);
    int scx = (int)(x * 2u);
    int scy = (int)(y * 2u);

    float sum[C];
    #pragma unroll
    for (unsigned int ch = 0; ch < C; ++ch) sum[ch] = 0.0f;

    int k_idx = 0;
    #pragma unroll
    for (int ky = 0; ky < 5; ++ky) {{
        int sy = interior ? (scy + ky - 2) : reflect_101(scy + ky - 2, (int)src_h);
        #pragma unroll
        for (int kx = 0; kx < 5; ++kx) {{
            int sx = interior ? (scx + kx - 2) : reflect_101(scx + kx - 2, (int)src_w);
            size_t si = ((size_t)sy * src_w + (size_t)sx) * C;
            float w = kw[k_idx];
            #pragma unroll
            for (unsigned int ch = 0; ch < C; ++ch) {{
                sum[ch] += __ldg(&src[si + ch]) * w;
            }}
            ++k_idx;
        }}
    }}
    size_t d = ((size_t)y * dst_w + (size_t)x) * C;
    #pragma unroll
    for (unsigned int ch = 0; ch < C; ++ch) {{
        dst[d + ch] = sum[ch];
    }}
}}
"#
    )
}

// ── f32 pyrup (polyphase H/V pair) ───────────────────────────────────────────

fn pyrup_h_f32_src(channels: usize) -> String {
    format!(
        r#"
#define C {channels}
{REFLECT_101}
// One thread per SOURCE pixel; writes the even and odd output columns with
// the exact expression groupings of pyrup_horizontal_pass_f32 (border
// phases special-cased identically).
extern "C" __global__ void pyrup_h_f32_c{channels}(
    const float* __restrict__ src,
    float* __restrict__       dst,
    unsigned int src_w,
    unsigned int src_h,
    unsigned int dst_w
) {{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= src_w || y >= src_h) return;
    size_t row = (size_t)y * src_w * C;
    size_t drow = (size_t)y * dst_w * C;

    #pragma unroll
    for (unsigned int ch = 0; ch < C; ++ch) {{
        if (src_w == 1u) {{
            float v = __ldg(&src[row + ch]);
            dst[drow + ch] = v;
            if (C + ch < (size_t)dst_w * C) dst[drow + C + ch] = v;
            continue;
        }}
        float curr = __ldg(&src[row + (size_t)x * C + ch]);
        if (x == 0u) {{
            float next = __ldg(&src[row + C + ch]);
            dst[drow + ch] = (6.0f * curr + 2.0f * next) * 0.125f;
            dst[drow + C + ch] = (curr + next) * 0.5f;
        }} else if (x == src_w - 1u) {{
            float prev = __ldg(&src[row + (size_t)(x - 1u) * C + ch]);
            dst[drow + (size_t)(2u * x) * C + ch] = (1.0f * prev + 7.0f * curr) * 0.125f;
            dst[drow + (size_t)(2u * x + 1u) * C + ch] = curr;
        }} else {{
            float prev = __ldg(&src[row + (size_t)(x - 1u) * C + ch]);
            float next = __ldg(&src[row + (size_t)(x + 1u) * C + ch]);
            dst[drow + (size_t)(2u * x) * C + ch] =
                (1.0f * prev + 6.0f * curr + 1.0f * next) * 0.125f;
            dst[drow + (size_t)(2u * x + 1u) * C + ch] = (curr + next) * 0.5f;
        }}
    }}
}}
"#
    )
}

fn pyrup_v_f32_src(channels: usize) -> String {
    format!(
        r#"
#define C {channels}
{REFLECT_101}
// One thread per intermediate element (i in [0, dst_w*C), y in [0, src_h));
// writes the even and odd output rows — pyrup_vertical_pass_f32 mirrored.
extern "C" __global__ void pyrup_v_f32_c{channels}(
    const float* __restrict__ buf,
    float* __restrict__       dst,
    unsigned int stride_elems,
    unsigned int src_h
) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= stride_elems || y >= src_h) return;

    unsigned int top, center, bottom;
    if (src_h == 1u)      {{ top = 0u; center = 0u; bottom = 0u; }}
    else if (y == 0u)     {{ top = 0u; center = 0u; bottom = 1u; }}
    else if (y == src_h - 1u) {{ top = src_h - 2u; center = src_h - 1u; bottom = src_h - 1u; }}
    else                  {{ top = y - 1u; center = y; bottom = y + 1u; }}

    float t = __ldg(&buf[(size_t)top * stride_elems + i]);
    float c = __ldg(&buf[(size_t)center * stride_elems + i]);
    float b = __ldg(&buf[(size_t)bottom * stride_elems + i]);

    float even, odd;
    if (y == 0u) {{
        even = (6.0f * c + 2.0f * b) * 0.125f;
        odd = (c + b) * 0.5f;
    }} else if (y == src_h - 1u) {{
        even = (1.0f * t + 7.0f * c) * 0.125f;
        odd = c;
    }} else {{
        even = (1.0f * t + 6.0f * c + 1.0f * b) * 0.125f;
        odd = (c + b) * 0.5f;
    }}
    dst[(size_t)(2u * y) * stride_elems + i] = even;
    dst[(size_t)(2u * y + 1u) * stride_elems + i] = odd;
}}
"#
    )
}

// ── u8 pyrdown (separable 1,4,6,4,1 with u16 intermediate) ───────────────────

fn pyrdown_h_u8_src(channels: usize) -> String {
    format!(
        r#"
#define C {channels}
{REFLECT_101}
extern "C" __global__ void pyrdown_h_u8_c{channels}(
    const unsigned char* __restrict__ src,
    unsigned short* __restrict__      buf,
    unsigned int src_w,
    unsigned int src_h,
    unsigned int dst_w
) {{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= src_h) return;
    size_t row = (size_t)y * src_w * C;
    int scx = (int)(x * 2u);

    // Same split as the CPU: the safe interior range [1, (src_w-1)/2) uses
    // direct indexing; everything else reflects.
    unsigned int safe_end = (src_w > 2u) ? ((src_w - 1u) / 2u) : 1u;
    bool interior = (x >= 1u && x < safe_end);

    int xm2, xm1, x0, xp1, xp2;
    if (interior) {{
        xm2 = scx - 2; xm1 = scx - 1; x0 = scx; xp1 = scx + 1; xp2 = scx + 2;
    }} else {{
        xm2 = reflect_101(scx - 2, (int)src_w);
        xm1 = reflect_101(scx - 1, (int)src_w);
        x0  = reflect_101(scx,     (int)src_w);
        xp1 = reflect_101(scx + 1, (int)src_w);
        xp2 = reflect_101(scx + 2, (int)src_w);
    }}
    size_t d = ((size_t)y * dst_w + (size_t)x) * C;
    #pragma unroll
    for (unsigned int ch = 0; ch < C; ++ch) {{
        unsigned int vm2 = __ldg(&src[row + (size_t)xm2 * C + ch]);
        unsigned int vm1 = __ldg(&src[row + (size_t)xm1 * C + ch]);
        unsigned int v0  = __ldg(&src[row + (size_t)x0  * C + ch]);
        unsigned int vp1 = __ldg(&src[row + (size_t)xp1 * C + ch]);
        unsigned int vp2 = __ldg(&src[row + (size_t)xp2 * C + ch]);
        buf[d + ch] = (unsigned short)(vm2 + 4u * vm1 + 6u * v0 + 4u * vp1 + vp2);
    }}
}}
"#
    )
}

fn pyrdown_v_u8_src(channels: usize) -> String {
    format!(
        r#"
#define C {channels}
{REFLECT_101}
extern "C" __global__ void pyrdown_v_u8_c{channels}(
    const unsigned short* __restrict__ buf,
    unsigned char* __restrict__        dst,
    unsigned int stride_elems,
    unsigned int src_h,
    unsigned int dst_h
) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= stride_elems || y >= dst_h) return;
    int scy = (int)(y * 2u);
    int ym2 = reflect_101(scy - 2, (int)src_h);
    int ym1 = reflect_101(scy - 1, (int)src_h);
    int y0  = reflect_101(scy,     (int)src_h);
    int yp1 = reflect_101(scy + 1, (int)src_h);
    int yp2 = reflect_101(scy + 2, (int)src_h);

    unsigned int vm2 = __ldg(&buf[(size_t)ym2 * stride_elems + i]);
    unsigned int vm1 = __ldg(&buf[(size_t)ym1 * stride_elems + i]);
    unsigned int v0  = __ldg(&buf[(size_t)y0  * stride_elems + i]);
    unsigned int vp1 = __ldg(&buf[(size_t)yp1 * stride_elems + i]);
    unsigned int vp2 = __ldg(&buf[(size_t)yp2 * stride_elems + i]);

    unsigned int sum = vm2 + 4u * vm1 + 6u * v0 + 4u * vp1 + vp2;
    unsigned int val = (sum + 128u) >> 8;
    dst[(size_t)y * stride_elems + i] = (unsigned char)min(val, 255u);
}}
"#
    )
}

// ── u8 pyrup (integer polyphase pair) ────────────────────────────────────────

fn pyrup_h_u8_src(channels: usize) -> String {
    format!(
        r#"
#define C {channels}
{REFLECT_101}
extern "C" __global__ void pyrup_h_u8_c{channels}(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__       buf,
    unsigned int src_w,
    unsigned int src_h,
    unsigned int dst_w
) {{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= src_w || y >= src_h) return;
    size_t row = (size_t)y * src_w * C;
    size_t drow = (size_t)y * dst_w * C;
    size_t dst_stride = (size_t)dst_w * C;

    int prev_i = reflect_101((int)x - 1, (int)src_w);
    int next_i = reflect_101((int)x + 1, (int)src_w);
    #pragma unroll
    for (unsigned int ch = 0; ch < C; ++ch) {{
        unsigned int curr = __ldg(&src[row + (size_t)x * C + ch]);
        unsigned int prev = __ldg(&src[row + (size_t)prev_i * C + ch]);
        unsigned int next = __ldg(&src[row + (size_t)next_i * C + ch]);
        buf[drow + (size_t)(2u * x) * C + ch] =
            (unsigned char)((prev + 6u * curr + next + 4u) >> 3);
        if ((size_t)(2u * x + 1u) * C + ch < dst_stride) {{
            buf[drow + (size_t)(2u * x + 1u) * C + ch] =
                (unsigned char)((curr + next + 1u) >> 1);
        }}
    }}
}}
"#
    )
}

fn pyrup_v_u8_src(channels: usize) -> String {
    format!(
        r#"
#define C {channels}
{REFLECT_101}
extern "C" __global__ void pyrup_v_u8_c{channels}(
    const unsigned char* __restrict__ buf,
    unsigned char* __restrict__       dst,
    unsigned int stride_elems,
    unsigned int src_h
) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= stride_elems || y >= src_h) return;
    int yp = reflect_101((int)y - 1, (int)src_h);
    int yn = reflect_101((int)y + 1, (int)src_h);

    unsigned int c = __ldg(&buf[(size_t)y  * stride_elems + i]);
    unsigned int p = __ldg(&buf[(size_t)yp * stride_elems + i]);
    unsigned int n = __ldg(&buf[(size_t)yn * stride_elems + i]);

    dst[(size_t)(2u * y) * stride_elems + i] =
        (unsigned char)((p + 6u * c + n + 4u) >> 3);
    dst[(size_t)(2u * y + 1u) * stride_elems + i] =
        (unsigned char)((c + n + 1u) >> 1);
}}
"#
    )
}

// ── Kernel cache ─────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum PyrKernelKey {
    DownF32 { channels: u32 },
    UpHF32 { channels: u32 },
    UpVF32 { channels: u32 },
    DownHU8 { channels: u32 },
    DownVU8 { channels: u32 },
    UpHU8 { channels: u32 },
    UpVU8 { channels: u32 },
}

type PyrKernelCache = Mutex<HashMap<PyrKernelKey, Arc<CudaKernel>>>;
static PYR_KERNELS: OnceLock<PyrKernelCache> = OnceLock::new();
const PYR_CACHE_CAP: usize = 32;

/// `make` renders `(kernel_name, source)` and runs ONLY on a cache miss —
/// steady-state launches allocate nothing host-side.
fn get_or_compile(
    ctx: &Arc<CudaContext>,
    key: PyrKernelKey,
    make: impl FnOnce() -> (String, String),
) -> Result<Arc<CudaKernel>, CudaPyramidError> {
    let cache = PYR_KERNELS.get_or_init(Default::default);
    let cached = cache
        .lock()
        .expect("pyramid kernel cache poisoned")
        .get(&key)
        .cloned();
    if let Some(hit) = cached {
        return Ok(hit);
    }
    let (name, src_code) = make();
    let built =
        Arc::new(try_compile_with_l1(ctx, &src_code, &name).map_err(CudaPyramidError::Cuda)?);
    let mut map = cache.lock().expect("pyramid kernel cache poisoned");
    if map.len() >= PYR_CACHE_CAP {
        // Evict one entry, not the whole map (stampede guard).
        if let Some(k) = map.keys().next().copied() {
            map.remove(&k);
        }
    }
    Ok(map.entry(key).or_insert(built).clone())
}

// ── Launchers ────────────────────────────────────────────────────────────────

/// Fused f32 pyrdown. `dst` is `ceil(src/2)` in each dimension.
#[allow(clippy::too_many_arguments)] // mirrors the kernel's parameter surface
pub fn launch_pyrdown_f32(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    channels: u32,
) -> Result<(), CudaPyramidError> {
    super::check_geometry(src_w, src_h, dst_w, dst_h, None).map_err(CudaPyramidError::Cuda)?;
    if channels == 0 {
        return Err(CudaPyramidError::Cuda("channels must be at least 1".into()));
    }
    check_slice(
        "src",
        src.len(),
        src_w as usize * src_h as usize * channels as usize,
    )?;
    check_slice(
        "dst",
        dst.len(),
        dst_w as usize * dst_h as usize * channels as usize,
    )?;
    let kernel = get_or_compile(ctx, PyrKernelKey::DownF32 { channels }, || {
        (
            format!("pyrdown_f32_c{channels}"),
            pyrdown_f32_src(channels as usize),
        )
    })?;
    kernel
        .launch_builder(stream)
        .arg(src)
        .arg(dst)
        .arg(&src_w)
        .arg(&src_h)
        .arg(&dst_w)
        .arg(&dst_h)
        .launch_2d(dst_w, dst_h, make_config(dst_w, dst_h, None))
        .map_err(|e| CudaPyramidError::Cuda(e.to_string()))
}

/// Polyphase f32 pyrup: H pass into `scratch` (`2*src_w × src_h`), V pass
/// into `dst` (`2*src_w × 2*src_h`).
#[allow(clippy::too_many_arguments)] // mirrors the kernel's parameter surface
pub fn launch_pyrup_f32(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    scratch: &mut CudaSlice<f32>,
    src_w: u32,
    src_h: u32,
    channels: u32,
) -> Result<(), CudaPyramidError> {
    super::check_geometry(src_w, src_h, src_w, src_h, None).map_err(CudaPyramidError::Cuda)?;
    if channels == 0 {
        return Err(CudaPyramidError::Cuda("channels must be at least 1".into()));
    }
    let dst_w = src_w * 2;
    let stride = dst_w as usize * channels as usize;
    check_slice(
        "src",
        src.len(),
        src_w as usize * src_h as usize * channels as usize,
    )?;
    check_slice("scratch", scratch.len(), stride * src_h as usize)?;
    check_slice("dst", dst.len(), stride * 2 * src_h as usize)?;

    let h = get_or_compile(ctx, PyrKernelKey::UpHF32 { channels }, || {
        (
            format!("pyrup_h_f32_c{channels}"),
            pyrup_h_f32_src(channels as usize),
        )
    })?;
    h.launch_builder(stream)
        .arg(src)
        .arg(&mut *scratch)
        .arg(&src_w)
        .arg(&src_h)
        .arg(&dst_w)
        .launch_2d(src_w, src_h, make_config(src_w, src_h, None))
        .map_err(|e| CudaPyramidError::Cuda(e.to_string()))?;

    let stride_elems = stride as u32;
    let v = get_or_compile(ctx, PyrKernelKey::UpVF32 { channels }, || {
        (
            format!("pyrup_v_f32_c{channels}"),
            pyrup_v_f32_src(channels as usize),
        )
    })?;
    v.launch_builder(stream)
        .arg(&*scratch)
        .arg(dst)
        .arg(&stride_elems)
        .arg(&src_h)
        .launch_2d(stride_elems, src_h, make_config(stride_elems, src_h, None))
        .map_err(|e| CudaPyramidError::Cuda(e.to_string()))
}

/// Separable u8 pyrdown: H pass into a u16 `scratch` (`dst_w × src_h`), V
/// pass into `dst` (`dst_w × dst_h`).
#[allow(clippy::too_many_arguments)] // mirrors the kernel's parameter surface
pub fn launch_pyrdown_u8(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<u8>,
    dst: &mut CudaSlice<u8>,
    scratch: &mut CudaSlice<u16>,
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    channels: u32,
) -> Result<(), CudaPyramidError> {
    super::check_geometry(src_w, src_h, dst_w, dst_h, None).map_err(CudaPyramidError::Cuda)?;
    if channels == 0 {
        return Err(CudaPyramidError::Cuda("channels must be at least 1".into()));
    }
    let stride = dst_w as usize * channels as usize;
    check_slice(
        "src",
        src.len(),
        src_w as usize * src_h as usize * channels as usize,
    )?;
    check_slice("scratch", scratch.len(), stride * src_h as usize)?;
    check_slice("dst", dst.len(), stride * dst_h as usize)?;

    let h = get_or_compile(ctx, PyrKernelKey::DownHU8 { channels }, || {
        (
            format!("pyrdown_h_u8_c{channels}"),
            pyrdown_h_u8_src(channels as usize),
        )
    })?;
    h.launch_builder(stream)
        .arg(src)
        .arg(&mut *scratch)
        .arg(&src_w)
        .arg(&src_h)
        .arg(&dst_w)
        .launch_2d(dst_w, src_h, make_config(dst_w, src_h, None))
        .map_err(|e| CudaPyramidError::Cuda(e.to_string()))?;

    let stride_elems = stride as u32;
    let v = get_or_compile(ctx, PyrKernelKey::DownVU8 { channels }, || {
        (
            format!("pyrdown_v_u8_c{channels}"),
            pyrdown_v_u8_src(channels as usize),
        )
    })?;
    v.launch_builder(stream)
        .arg(&*scratch)
        .arg(dst)
        .arg(&stride_elems)
        .arg(&src_h)
        .arg(&dst_h)
        .launch_2d(stride_elems, dst_h, make_config(stride_elems, dst_h, None))
        .map_err(|e| CudaPyramidError::Cuda(e.to_string()))
}

/// Integer polyphase u8 pyrup: H pass into a u8 `scratch` (`2*src_w ×
/// src_h`), V pass into `dst` (`2*src_w × 2*src_h`).
#[allow(clippy::too_many_arguments)] // mirrors the kernel's parameter surface
pub fn launch_pyrup_u8(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<u8>,
    dst: &mut CudaSlice<u8>,
    scratch: &mut CudaSlice<u8>,
    src_w: u32,
    src_h: u32,
    channels: u32,
) -> Result<(), CudaPyramidError> {
    super::check_geometry(src_w, src_h, src_w, src_h, None).map_err(CudaPyramidError::Cuda)?;
    if channels == 0 {
        return Err(CudaPyramidError::Cuda("channels must be at least 1".into()));
    }
    let dst_w = src_w * 2;
    let stride = dst_w as usize * channels as usize;
    check_slice(
        "src",
        src.len(),
        src_w as usize * src_h as usize * channels as usize,
    )?;
    check_slice("scratch", scratch.len(), stride * src_h as usize)?;
    check_slice("dst", dst.len(), stride * 2 * src_h as usize)?;

    let h = get_or_compile(ctx, PyrKernelKey::UpHU8 { channels }, || {
        (
            format!("pyrup_h_u8_c{channels}"),
            pyrup_h_u8_src(channels as usize),
        )
    })?;
    h.launch_builder(stream)
        .arg(src)
        .arg(&mut *scratch)
        .arg(&src_w)
        .arg(&src_h)
        .arg(&dst_w)
        .launch_2d(src_w, src_h, make_config(src_w, src_h, None))
        .map_err(|e| CudaPyramidError::Cuda(e.to_string()))?;

    let stride_elems = stride as u32;
    let v = get_or_compile(ctx, PyrKernelKey::UpVU8 { channels }, || {
        (
            format!("pyrup_v_u8_c{channels}"),
            pyrup_v_u8_src(channels as usize),
        )
    })?;
    v.launch_builder(stream)
        .arg(&*scratch)
        .arg(dst)
        .arg(&stride_elems)
        .arg(&src_h)
        .launch_2d(stride_elems, src_h, make_config(stride_elems, src_h, None))
        .map_err(|e| CudaPyramidError::Cuda(e.to_string()))
}
