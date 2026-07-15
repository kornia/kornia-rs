//! Native CUDA warp-affine kernels for `kornia-imgproc`.
//!
//! # Algorithm
//!
//! All kernels use **inverse mapping**: each output thread computes the
//! floating-point source coordinate by applying the inverted 2×3 affine matrix
//! to its `(dst_x, dst_y)` position.
//!
//! # Border handling
//!
//! The kernels reproduce the CPU [`warp_affine`](crate::warp::warp_affine)
//! contract:
//!
//! * A destination pixel whose inverse-mapped source coordinate falls outside
//!   `[0, src_w) × [0, src_h)` is written as 0 (`BORDER_CONSTANT`), matching the
//!   CPU valid-range guard.
//! * Inside that range, bilinear clamps the coordinate to `[0, src-1]` and takes
//!   the `+1` tap as `min(x0 + 1, src_w - 1)` — a per-axis edge clamp. Note this
//!   is *not* the rule in `interpolation::bilinear::bilinear_interpolation`
//!   (which replicates `val00`); the f32 `warp_affine` inner loop does not use
//!   that helper. `warp_perspective` does, and its kernels match it instead.
//! * Nearest rounds half-away-from-zero then clamps.
//!
//! **Exact-tie caveat.** The CPU walks a row by accumulating `sx += dsx`, while
//! each GPU thread evaluates `m0*x + m1*y + m2` independently. The two agree to
//! within a float ulp, so a source coordinate landing exactly on a `.5` boundary
//! can round to different pixels on the two paths. This affects only nearest,
//! only at exact ties, and is inherent to any parallel evaluation of the map.
//!
//! # Optimisations applied
//!
//! * **32×8 thread block (default)** — full warp per output row, same
//!   reasoning as `resize`: better write coalescing and source reads
//!   confined to nearby rows per warp.
//! * **`__ldg` source reads** — all four kernels read the source through the
//!   read-only data cache from a raw pointer. An earlier revision routed
//!   bilinear/nearest through a 1-channel pitch-2D texture object; that made
//!   `pitchInBytes = src_w * 3 * 4`, which CUDA requires to be a multiple of
//!   `CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT` (32 B), so the kernels
//!   failed outright for every source width not a multiple of 8 (issue #1000).
//!   The source rows are densely packed, so an aligned pitch would have needed
//!   a padded device-to-device copy per call. `__ldg` avoids the constraint,
//!   lets the kernels implement the CPU border rules above, and matches
//!   what `cuda::resize` already found faster for interleaved RGB.
//! * **`CU_FUNC_CACHE_PREFER_L1`** — enlarges L1 to 64 KB since no kernel here
//!   uses shared memory.
//! * **Dynamic block size** — every launcher accepts an optional `block_dim`;
//!   `None` selects 32×8 (optimal for most sizes) while `Some` lets callers
//!   override for small images.
//!
//! # Public API
//!
//! * [`launch_warp_affine_bilinear_cuda`] — bilinear warp affine, 3-ch f32.
//! * [`launch_warp_affine_nearest_cuda`]  — nearest-neighbor warp affine, 3-ch f32.
//! * [`launch_warp_affine_bicubic_cuda`]  — bicubic warp affine, 3-ch f32.
//! * [`launch_warp_affine_lanczos_cuda`]  — Lanczos-3 warp affine, 3-ch f32.

use std::sync::{Arc, OnceLock};

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use kornia_tensor::CudaKernel;

use super::{check_geometry, make_config, try_compile_with_l1};
use crate::warp::invert_affine_transform;

// ── CUDA C source: bilinear warp affine ──────────────────────────────────────
//
// Reads the source through __ldg on a raw pointer. Edge rule: the coordinate is
// clamped into [0, src-1] and the +1 tap per axis is `min(x0+1, src_w-1)` — the
// per-axis edge clamp of the CPU f32 `warp_affine` inner loop. (This is NOT the
// val00-replicate rule of `bilinear_interpolation`; that one belongs to
// `warp_perspective`, whose kernels implement it instead.)

static BILINEAR_SRC: &str = r#"
extern "C" __global__ void warp_affine_bilinear_3c(
    const float* __restrict__ src,
    float* __restrict__       dst,
    unsigned int src_w,
    unsigned int src_h,
    unsigned int dst_w,
    unsigned int dst_h,
    float m0, float m1, float m2,
    float m3, float m4, float m5
) {
    unsigned int gx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int gy = blockIdx.y * blockDim.y + threadIdx.y;
    if (gx >= dst_w || gy >= dst_h) return;

    // 64-bit index math: (y*w + x)*3 wraps 32 bits at ~1.43 gigapixels.
    unsigned long long out = ((unsigned long long)gy * dst_w + gx) * 3ull;

    // Inverse affine: map output pixel → floating-point source coordinate.
    float sx = m0 * (float)gx + m1 * (float)gy + m2;
    float sy = m3 * (float)gx + m4 * (float)gy + m5;

    // BORDER_CONSTANT: outside the source, emit 0 (CPU valid-range guard).
    if (sx < 0.0f || sx >= (float)src_w || sy < 0.0f || sy >= (float)src_h) {
        dst[out] = 0.0f; dst[out+1] = 0.0f; dst[out+2] = 0.0f;
        return;
    }

    // Clamp into [0, src-1] and take the +1 tap with a per-axis edge clamp —
    // the rule the CPU f32 `warp_affine` inner loop uses.
    float sxc = fminf(sx, (float)(src_w - 1u));
    float syc = fminf(sy, (float)(src_h - 1u));

    unsigned int x0 = (unsigned int)sxc;
    unsigned int y0 = (unsigned int)syc;
    unsigned int x1 = min(x0 + 1u, src_w - 1u);
    unsigned int y1 = min(y0 + 1u, src_h - 1u);
    float fx = sxc - (float)x0;
    float fy = syc - (float)y0;

    unsigned long long r0 = (unsigned long long)y0 * src_w;
    unsigned long long r1 = (unsigned long long)y1 * src_w;
    unsigned long long b00 = (r0 + x0) * 3ull;
    unsigned long long b10 = (r0 + x1) * 3ull;
    unsigned long long b01 = (r1 + x0) * 3ull;
    unsigned long long b11 = (r1 + x1) * 3ull;

    float fxx = 1.0f - fx;
    float fyy = 1.0f - fy;
    float w00 = fyy * fxx;
    float w10 = fyy * fx;
    float w01 = fy  * fxx;
    float w11 = fy  * fx;

    #pragma unroll
    for (unsigned int c = 0u; c < 3u; ++c) {
        float v00 = __ldg(&src[b00 + c]);
        float v10 = __ldg(&src[b10 + c]);
        float v01 = __ldg(&src[b01 + c]);
        float v11 = __ldg(&src[b11 + c]);
        dst[out + c] = fmaf(w00, v00, fmaf(w10, v10, fmaf(w01, v01, w11 * v11)));
    }
}
"#;

// ── CUDA C source: nearest-neighbor warp affine ──────────────────────────────
//
// roundf is half-away-from-zero, matching Rust's f32::round; the clamp then
// mirrors `nearest_neighbor_interpolation`, where a coordinate like src_w-0.2
// rounds up to src_w and is clamped back to the last column.

static NEAREST_SRC: &str = r#"
extern "C" __global__ void warp_affine_nearest_3c(
    const float* __restrict__ src,
    float* __restrict__       dst,
    unsigned int src_w,
    unsigned int src_h,
    unsigned int dst_w,
    unsigned int dst_h,
    float m0, float m1, float m2,
    float m3, float m4, float m5
) {
    unsigned int gx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int gy = blockIdx.y * blockDim.y + threadIdx.y;
    if (gx >= dst_w || gy >= dst_h) return;

    unsigned long long out = ((unsigned long long)gy * dst_w + gx) * 3ull;

    float sx = m0 * (float)gx + m1 * (float)gy + m2;
    float sy = m3 * (float)gx + m4 * (float)gy + m5;

    if (sx < 0.0f || sx >= (float)src_w || sy < 0.0f || sy >= (float)src_h) {
        dst[out] = 0.0f; dst[out+1] = 0.0f; dst[out+2] = 0.0f;
        return;
    }

    unsigned int xi = min((unsigned int)roundf(sx), src_w - 1u);
    unsigned int yi = min((unsigned int)roundf(sy), src_h - 1u);

    unsigned long long b = ((unsigned long long)yi * src_w + xi) * 3ull;
    dst[out]   = __ldg(&src[b]);
    dst[out+1] = __ldg(&src[b+1]);
    dst[out+2] = __ldg(&src[b+2]);
}
"#;

// ── CUDA C source: bicubic warp affine ───────────────────────────────────────
//
// Keys cubic (a = -0.5, matches OpenCV INTER_CUBIC). Weights are computed via
// Horner form without branches: frac ∈ [0,1) places each tap in a known
// polynomial region, eliminating fabsf and the two conditionals of a generic
// cubic_w helper. Precomputing wx[4]/wy[4] before the loop removes the 12
// redundant x-weight evaluations the naive loop would incur. Row base addresses
// are precomputed outside the inner loop to move the row-multiply out of the
// critical path. #pragma unroll lets ptxas allocate wx/wy in registers and
// issue all 16 __ldg loads without loop overhead.

static BICUBIC_SRC: &str = r#"
extern "C" __global__ void warp_affine_bicubic_3c(
    const float* __restrict__ src,
    float* __restrict__       dst,
    unsigned int src_w,
    unsigned int src_h,
    unsigned int dst_w,
    unsigned int dst_h,
    float m0, float m1, float m2,
    float m3, float m4, float m5
) {
    unsigned int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (dst_x >= dst_w || dst_y >= dst_h) return;

    float sx = m0 * (float)dst_x + m1 * (float)dst_y + m2;
    float sy = m3 * (float)dst_x + m4 * (float)dst_y + m5;

    unsigned long long out = ((unsigned long long)dst_y * dst_w + dst_x) * 3ull;

    if (sx < 0.0f || sx >= (float)src_w || sy < 0.0f || sy >= (float)src_h) {
        dst[out] = 0.0f; dst[out+1] = 0.0f; dst[out+2] = 0.0f;
        return;
    }

    int x0 = (int)floorf(sx);
    int y0 = (int)floorf(sy);
    float frac_x = sx - (float)x0;
    float frac_y = sy - (float)y0;

    // Horner-form cubic weights, branch-free. frac in [0,1) gives:
    //   i=0 (tap -1): |t|=1+frac → second poly  -0.5t³+2.5t²-4t+2
    //   i=1 (tap  0): |t|=frac   → first poly    1.5t³-2.5t²+1
    //   i=2 (tap +1): |t|=1-frac → first poly
    //   i=3 (tap +2): |t|=2-frac → second poly
    float wx[4], wy[4];
    {
        float t;
        t = 1.0f + frac_x; wx[0] = ((-0.5f*t + 2.5f)*t - 4.0f)*t + 2.0f;
        t =         frac_x; wx[1] = (( 1.5f*t - 2.5f)*t       )*t + 1.0f;
        t = 1.0f - frac_x; wx[2] = (( 1.5f*t - 2.5f)*t       )*t + 1.0f;
        t = 2.0f - frac_x; wx[3] = ((-0.5f*t + 2.5f)*t - 4.0f)*t + 2.0f;
        t = 1.0f + frac_y; wy[0] = ((-0.5f*t + 2.5f)*t - 4.0f)*t + 2.0f;
        t =         frac_y; wy[1] = (( 1.5f*t - 2.5f)*t       )*t + 1.0f;
        t = 1.0f - frac_y; wy[2] = (( 1.5f*t - 2.5f)*t       )*t + 1.0f;
        t = 2.0f - frac_y; wy[3] = ((-0.5f*t + 2.5f)*t - 4.0f)*t + 2.0f;
    }

    // Row base addresses precomputed: moves the row multiply outside the inner loop.
    unsigned long long row[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int yi = max(0, min(y0 + i - 1, (int)src_h - 1));
        row[i] = (unsigned long long)yi * src_w * 3ull;
    }

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f;
    #pragma unroll
    for (int dy = 0; dy < 4; ++dy) {
        #pragma unroll
        for (int dx = 0; dx < 4; ++dx) {
            int xi = max(0, min(x0 + dx - 1, (int)src_w - 1));
            unsigned long long b = row[dy] + (unsigned long long)xi * 3ull;
            float w = wx[dx] * wy[dy];
            acc0 = fmaf(w, __ldg(&src[b]),   acc0);
            acc1 = fmaf(w, __ldg(&src[b+1]), acc1);
            acc2 = fmaf(w, __ldg(&src[b+2]), acc2);
        }
    }
    dst[out]     = acc0;
    dst[out + 1] = acc1;
    dst[out + 2] = acc2;
}
"#;

// ── CUDA C source: Lanczos-3 warp affine ─────────────────────────────────────
//
// 3-lobe Lanczos filter using a 6×6 tap grid (36 source reads per output pixel).
// Weights are normalised after computation to prevent brightness drift at clamped
// boundaries. OOB pixels (centre outside source) are zero-filled (BORDER_CONSTANT).
// Taps within the 6×6 neighbourhood that fall outside are clamped (BORDER_REPLICATE).

static LANCZOS_SRC: &str = r#"
__device__ inline float lanczos3(float x) {
    const float PI = 3.14159265358979f;
    if (fabsf(x) < 1e-5f) return 1.0f;
    if (fabsf(x) >= 3.0f) return 0.0f;
    float pix  = PI * x;
    float pix3 = pix * 0.33333333f;
    return __sinf(pix) * __sinf(pix3) * __fdividef(1.0f, pix * pix3);
}

extern "C" __global__ void warp_affine_lanczos_3c(
    const float* __restrict__ src,
    float* __restrict__       dst,
    unsigned int src_w,
    unsigned int src_h,
    unsigned int dst_w,
    unsigned int dst_h,
    float m0, float m1, float m2,
    float m3, float m4, float m5
) {
    unsigned int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (dst_x >= dst_w || dst_y >= dst_h) return;

    float sx = m0 * (float)dst_x + m1 * (float)dst_y + m2;
    float sy = m3 * (float)dst_x + m4 * (float)dst_y + m5;

    unsigned long long out = ((unsigned long long)dst_y * dst_w + dst_x) * 3ull;

    if (sx < 0.0f || sx >= (float)src_w || sy < 0.0f || sy >= (float)src_h) {
        dst[out] = 0.0f; dst[out+1] = 0.0f; dst[out+2] = 0.0f;
        return;
    }

    int x0 = (int)floorf(sx);
    int y0 = (int)floorf(sy);
    float frac_x = sx - (float)x0;
    float frac_y = sy - (float)y0;

    // 6 weights per axis; tap i is at offset (i-2) from x0/y0.
    float wx[6], wy[6];
    wx[0] = lanczos3(frac_x + 2.0f); wx[1] = lanczos3(frac_x + 1.0f);
    wx[2] = lanczos3(frac_x);        wx[3] = lanczos3(frac_x - 1.0f);
    wx[4] = lanczos3(frac_x - 2.0f); wx[5] = lanczos3(frac_x - 3.0f);
    wy[0] = lanczos3(frac_y + 2.0f); wy[1] = lanczos3(frac_y + 1.0f);
    wy[2] = lanczos3(frac_y);        wy[3] = lanczos3(frac_y - 1.0f);
    wy[4] = lanczos3(frac_y - 2.0f); wy[5] = lanczos3(frac_y - 3.0f);

    // Normalise to guard against brightness drift at clamped boundaries.
    float sum_wx = wx[0]+wx[1]+wx[2]+wx[3]+wx[4]+wx[5];
    float sum_wy = wy[0]+wy[1]+wy[2]+wy[3]+wy[4]+wy[5];
    float inv_x = __fdividef(1.0f, sum_wx);
    float inv_y = __fdividef(1.0f, sum_wy);
    #pragma unroll
    for (int i = 0; i < 6; i++) { wx[i] *= inv_x; wy[i] *= inv_y; }

    // Row base addresses precomputed to move the multiply outside the inner loop.
    unsigned long long row[6];
    #pragma unroll
    for (int i = 0; i < 6; i++) {
        int yi = max(0, min(y0 + i - 2, (int)src_h - 1));
        row[i] = (unsigned long long)yi * src_w * 3ull;
    }

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f;
    #pragma unroll
    for (int dy = 0; dy < 6; dy++) {
        float rx = 0.0f, gx = 0.0f, bx = 0.0f;
        #pragma unroll
        for (int dx = 0; dx < 6; dx++) {
            int xi = max(0, min(x0 + dx - 2, (int)src_w - 1));
            unsigned long long b = row[dy] + (unsigned long long)xi * 3ull;
            rx = fmaf(wx[dx], __ldg(&src[b]),   rx);
            gx = fmaf(wx[dx], __ldg(&src[b+1]), gx);
            bx = fmaf(wx[dx], __ldg(&src[b+2]), bx);
        }
        acc0 = fmaf(wy[dy], rx, acc0);
        acc1 = fmaf(wy[dy], gx, acc1);
        acc2 = fmaf(wy[dy], bx, acc2);
    }

    dst[out]     = acc0;
    dst[out + 1] = acc1;
    dst[out + 2] = acc2;
}
"#;

// ── Kernel caches ─────────────────────────────────────────────────────────────

static BILINEAR_KERNEL: OnceLock<Result<CudaKernel, String>> = OnceLock::new();
static NEAREST_KERNEL: OnceLock<Result<CudaKernel, String>> = OnceLock::new();
static BICUBIC_KERNEL: OnceLock<Result<CudaKernel, String>> = OnceLock::new();
static LANCZOS_KERNEL: OnceLock<Result<CudaKernel, String>> = OnceLock::new();

// ── Error type ────────────────────────────────────────────────────────────────

/// Error type returned by the CUDA warp-affine launchers.
#[derive(Debug, thiserror::Error)]
pub enum CudaWarpAffineError {
    /// CUDA kernel compilation or launch failure.
    #[error("CUDA kernel error: {0}")]
    Cuda(String),
    /// Output device slice is smaller than the required pixel count.
    #[error("output slice length {got} < required {need}")]
    SliceTooSmall {
        /// Actual slice length (in elements).
        got: usize,
        /// Minimum required length (dst_w × dst_h × 3).
        need: usize,
    },
}

// ── Internal helpers ──────────────────────────────────────────────────────────

fn check_slice(slice: &CudaSlice<f32>, width: u32, height: u32) -> Result<(), CudaWarpAffineError> {
    let need = (width as usize) * (height as usize) * 3;
    if slice.len() < need {
        return Err(CudaWarpAffineError::SliceTooSmall {
            got: slice.len(),
            need,
        });
    }
    Ok(())
}

/// Shared pre-launch validation for all four launchers: non-zero dims and
/// block components (a zero would underflow the kernels' `-1u` clamps or
/// divide-by-zero in `make_config`), and both slice lengths — every kernel
/// here indexes `src` directly via `__ldg`, so a short source slice is an
/// out-of-bounds device read rather than a recoverable error.
#[allow(clippy::too_many_arguments)]
fn validate(
    src: &CudaSlice<f32>,
    dst: &CudaSlice<f32>,
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    block_dim: Option<(u32, u32)>,
) -> Result<(), CudaWarpAffineError> {
    check_geometry(src_width, src_height, dst_width, dst_height, block_dim)
        .map_err(CudaWarpAffineError::Cuda)?;
    check_slice(src, src_width, src_height)?;
    check_slice(dst, dst_width, dst_height)
}

// ── Public launchers ──────────────────────────────────────────────────────────

/// Launch the bilinear warp-affine kernel for a 3-channel f32 image.
///
/// Applies the 2×3 forward affine matrix `m` to warp `src` into `dst`.
/// The matrix is inverted internally so the API matches the CPU
/// [`warp_affine`](crate::warp::warp_affine).
///
/// Source pixels are read via `__ldg`. Out-of-bounds source coordinates yield 0
/// (`BORDER_CONSTANT`); inside the source, the coordinate is clamped and the
/// +1 tap is edge-clamped per axis, matching the CPU `warp_affine` inner loop.
/// Any source width is supported.
///
/// # Arguments
///
/// * `ctx`        – CUDA context for one-time kernel compilation.
/// * `stream`     – Stream for kernel execution.
/// * `src`        – Device slice: `src_h × src_w × 3` f32 values (HWC).
/// * `dst`        – Device slice: `dst_h × dst_w × 3` f32 values (written).
/// * `src_width`, `src_height` – Source image dimensions.
/// * `dst_width`, `dst_height` – Output canvas dimensions (may differ from source).
/// * `m`          – Forward 2×3 affine matrix `[a, b, tx, d, e, ty]`.
/// * `block_dim`  – Optional thread-block override `(width, height)`.
///   Pass `None` to use the default 32×8 block (optimal for large images).
///   For small images (≤ 128×128), passing a smaller block avoids wasted
///   threads: e.g. `Some((16, 4))` for a 64×64 output.
///
/// # Errors
///
/// Returns [`CudaWarpAffineError`] on compile failure, launch error, or if
/// either slice is too small for its stated dimensions.
#[allow(clippy::too_many_arguments)]
pub fn launch_warp_affine_bilinear_cuda(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    m: &[f32; 6],
    block_dim: Option<(u32, u32)>,
) -> Result<(), CudaWarpAffineError> {
    validate(
        src, dst, src_width, src_height, dst_width, dst_height, block_dim,
    )?;

    let kernel = BILINEAR_KERNEL
        .get_or_init(|| try_compile_with_l1(ctx, BILINEAR_SRC, "warp_affine_bilinear_3c"))
        .as_ref()
        .map_err(|e| CudaWarpAffineError::Cuda(e.clone()))?;

    let mi = invert_affine_transform(m);

    kernel
        .launch_builder(stream)
        .arg(src)
        .arg(dst)
        .arg(&src_width)
        .arg(&src_height)
        .arg(&dst_width)
        .arg(&dst_height)
        .arg(&mi[0])
        .arg(&mi[1])
        .arg(&mi[2])
        .arg(&mi[3])
        .arg(&mi[4])
        .arg(&mi[5])
        .launch_2d(
            dst_width,
            dst_height,
            make_config(dst_width, dst_height, block_dim),
        )
        .map_err(|e| CudaWarpAffineError::Cuda(e.to_string()))
}

/// Launch the nearest-neighbor warp-affine kernel for a 3-channel f32 image.
///
/// Same as [`launch_warp_affine_bilinear_cuda`] but uses round-to-nearest
/// source sampling.  Faster; suitable for integer-aligned transforms (pure
/// translation, 90°/180° rotation, flip).
///
/// Source reads go through `__ldg`; out-of-bounds coords yield 0. Rounding is
/// half-away-from-zero then clamped, matching the CPU nearest interpolator.
///
/// # Arguments
///
/// See [`launch_warp_affine_bilinear_cuda`] — arguments are identical.
///
/// # Errors
///
/// Returns [`CudaWarpAffineError`] on compile failure, launch error, or if
/// either slice is too small for its stated dimensions.
#[allow(clippy::too_many_arguments)]
pub fn launch_warp_affine_nearest_cuda(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    m: &[f32; 6],
    block_dim: Option<(u32, u32)>,
) -> Result<(), CudaWarpAffineError> {
    validate(
        src, dst, src_width, src_height, dst_width, dst_height, block_dim,
    )?;

    let kernel = NEAREST_KERNEL
        .get_or_init(|| try_compile_with_l1(ctx, NEAREST_SRC, "warp_affine_nearest_3c"))
        .as_ref()
        .map_err(|e| CudaWarpAffineError::Cuda(e.clone()))?;

    let mi = invert_affine_transform(m);

    kernel
        .launch_builder(stream)
        .arg(src)
        .arg(dst)
        .arg(&src_width)
        .arg(&src_height)
        .arg(&dst_width)
        .arg(&dst_height)
        .arg(&mi[0])
        .arg(&mi[1])
        .arg(&mi[2])
        .arg(&mi[3])
        .arg(&mi[4])
        .arg(&mi[5])
        .launch_2d(
            dst_width,
            dst_height,
            make_config(dst_width, dst_height, block_dim),
        )
        .map_err(|e| CudaWarpAffineError::Cuda(e.to_string()))
}

/// Launch the bicubic warp-affine kernel for a 3-channel f32 image.
///
/// Uses Keys cubic interpolation (`a = -0.5`, matches OpenCV `INTER_CUBIC`)
/// with a 4×4 tap neighborhood.  Out-of-range taps are clamped to the image
/// border; pixels whose centre maps outside the source are zero-filled.
///
/// Bicubic reads 16 source values per output pixel via `__ldg`, making it
/// more compute- and bandwidth-intensive than bilinear (4 reads).  Best used
/// when visual quality matters more than raw throughput.
///
/// Hardware texture bilinear cannot accelerate a 4×4 bicubic stencil, so this
/// kernel continues to use `__ldg` with explicit clamping.
///
/// # Arguments
///
/// See [`launch_warp_affine_bilinear_cuda`] — arguments are identical including
/// the optional `block_dim` override.
///
/// # Errors
///
/// Returns [`CudaWarpAffineError`] on compile failure, launch error, zero
/// dimensions, or if either slice is too small for its stated dimensions.
#[allow(clippy::too_many_arguments)]
pub fn launch_warp_affine_bicubic_cuda(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    m: &[f32; 6],
    block_dim: Option<(u32, u32)>,
) -> Result<(), CudaWarpAffineError> {
    validate(
        src, dst, src_width, src_height, dst_width, dst_height, block_dim,
    )?;

    let kernel = BICUBIC_KERNEL
        .get_or_init(|| try_compile_with_l1(ctx, BICUBIC_SRC, "warp_affine_bicubic_3c"))
        .as_ref()
        .map_err(|e| CudaWarpAffineError::Cuda(e.clone()))?;

    let mi = invert_affine_transform(m);

    kernel
        .launch_builder(stream)
        .arg(src)
        .arg(dst)
        .arg(&src_width)
        .arg(&src_height)
        .arg(&dst_width)
        .arg(&dst_height)
        .arg(&mi[0])
        .arg(&mi[1])
        .arg(&mi[2])
        .arg(&mi[3])
        .arg(&mi[4])
        .arg(&mi[5])
        .launch_2d(
            dst_width,
            dst_height,
            make_config(dst_width, dst_height, block_dim),
        )
        .map_err(|e| CudaWarpAffineError::Cuda(e.to_string()))
}

/// Launch the Lanczos-3 warp-affine kernel for a 3-channel f32 image.
///
/// Uses a 6×6 tap Lanczos-3 filter (36 source reads per output pixel).
/// Weights are normalised after computation to prevent brightness drift at
/// clamped image borders.  Pixels whose centre maps outside the source are
/// zero-filled (`BORDER_CONSTANT`); taps within the neighbourhood that fall
/// outside are clamped (`BORDER_REPLICATE`).
///
/// Lanczos reads 36 source values per output pixel via `__ldg`.  Best used
/// when highest visual quality is required regardless of throughput cost.
///
/// # Arguments
///
/// See [`launch_warp_affine_bilinear_cuda`] — arguments are identical except
/// no `block_dim` parameter (always uses 32×8).
///
/// # Errors
///
/// Returns [`CudaWarpAffineError`] on compile failure, launch error, zero
/// dimensions, or if either slice is too small for its stated dimensions.
#[allow(clippy::too_many_arguments)]
pub fn launch_warp_affine_lanczos_cuda(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    m: &[f32; 6],
    block_dim: Option<(u32, u32)>,
) -> Result<(), CudaWarpAffineError> {
    validate(
        src, dst, src_width, src_height, dst_width, dst_height, block_dim,
    )?;

    let kernel = LANCZOS_KERNEL
        .get_or_init(|| try_compile_with_l1(ctx, LANCZOS_SRC, "warp_affine_lanczos_3c"))
        .as_ref()
        .map_err(|e| CudaWarpAffineError::Cuda(e.clone()))?;

    let mi = invert_affine_transform(m);

    kernel
        .launch_builder(stream)
        .arg(src)
        .arg(dst)
        .arg(&src_width)
        .arg(&src_height)
        .arg(&dst_width)
        .arg(&dst_height)
        .arg(&mi[0])
        .arg(&mi[1])
        .arg(&mi[2])
        .arg(&mi[3])
        .arg(&mi[4])
        .arg(&mi[5])
        .launch_2d(
            dst_width,
            dst_height,
            make_config(dst_width, dst_height, block_dim),
        )
        .map_err(|e| CudaWarpAffineError::Cuda(e.to_string()))
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::cuda::color::test_utils::{default_stream, pattern_f32};
    use crate::interpolation::InterpolationMode;
    use crate::warp::warp_affine;
    use kornia_image::{Image, ImageSize};

    /// A transform whose **inverse** has dyadic coefficients (exact in binary).
    /// The CPU walks a row by accumulating `sx += dsx` where `dsx` comes from the
    /// *inverted* matrix; when that is dyadic the accumulation is exact, so CPU
    /// and GPU derive bit-identical source coordinates and the comparison
    /// isolates what we actually want to test — the interpolation and edge rules
    /// — instead of measuring the CPU's coordinate drift.
    ///
    /// Integer entries with a power-of-two determinant guarantee it: here
    /// `det = 2*4 - 2*2 = 4`, so every inverse entry is an integer over 4.
    /// `dyadic_m_inverse_is_exact` pins that property. Both axes shear (`dsx` and
    /// `dsy` are both non-zero) and the translation pushes part of the output off
    /// the source, so the border path is exercised too.
    const DYADIC_M: [f32; 6] = [2.0, 2.0, -8.0, 2.0, 4.0, 6.0];

    /// Guards the premise of every strict comparison below: if this ever stops
    /// holding, the tight tolerances are measuring float drift, not the kernel.
    #[test]
    fn dyadic_m_inverse_is_exact() {
        let mi = crate::warp::invert_affine_transform(&DYADIC_M);
        for (i, v) in mi.iter().enumerate() {
            let scaled = v * 4.0;
            assert_eq!(
                scaled,
                scaled.round(),
                "m_inv[{i}] = {v} is not an exact multiple of 1/4; the CPU's \
                 `sx += dsx` accumulation will drift and the strict tolerances \
                 in this module are no longer meaningful"
            );
        }
    }

    /// With DYADIC_M the only remaining divergence is `fmaf` fusing the weighted
    /// accumulate where the CPU rounds each product — a few ulps on values in
    /// [0, 1]. An edge-rule mismatch is ~0.5 (an earlier revision of this kernel
    /// used the wrong rule and the test caught it at 0.7), so this is strict.
    const BILINEAR_TOL: f32 = 1e-5;

    /// A realistic non-dyadic transform. Here the CPU's accumulated `sx` drifts a
    /// few ulps from the GPU's direct evaluation, and on the deliberately random
    /// test pattern (adjacent pixels differ by up to 1.0) that surfaces as a
    /// ~1e-3 value difference. Loose on purpose: this case guards against gross
    /// errors under a rotation, while DYADIC_M does the strict rule-checking.
    const ROTATION_TOL: f32 = 5e-3;

    /// Run the CPU reference and the CUDA kernel on the same input and return
    /// both buffers. `dst` dims equal `src` dims.
    fn cpu_and_gpu(
        w: usize,
        h: usize,
        m: &[f32; 6],
        interpolation: InterpolationMode,
    ) -> (Vec<f32>, Vec<f32>) {
        let data = pattern_f32(w * h * 3);
        let size = ImageSize {
            width: w,
            height: h,
        };

        let src = Image::<f32, 3>::new(size, data.clone()).unwrap();
        let mut cpu = Image::<f32, 3>::from_size_val(size, 0.0).unwrap();
        warp_affine(&src, &mut cpu, m, interpolation).unwrap();

        let stream = default_stream();
        let ctx = &stream.context();
        let d_src = stream.clone_htod(&data).unwrap();
        let mut d_dst = stream.alloc_zeros::<f32>(w * h * 3).unwrap();

        let (wu, hu) = (w as u32, h as u32);
        match interpolation {
            InterpolationMode::Bilinear => launch_warp_affine_bilinear_cuda(
                ctx, &stream, &d_src, &mut d_dst, wu, hu, wu, hu, m, None,
            ),
            InterpolationMode::Nearest => launch_warp_affine_nearest_cuda(
                ctx, &stream, &d_src, &mut d_dst, wu, hu, wu, hu, m, None,
            ),
            other => panic!("unsupported mode in test: {other:?}"),
        }
        .unwrap_or_else(|e| panic!("launch failed at {w}x{h} ({interpolation:?}): {e}"));

        let gpu: Vec<f32> = stream.clone_dtoh(&d_dst).unwrap();
        stream.synchronize().unwrap();

        (cpu.as_slice().to_vec(), gpu)
    }

    /// Pixels whose CPU/GPU results may legitimately differ, for either of two
    /// reasons — both rooted in the same ulp-level coordinate drift:
    ///
    /// * **Validity boundary.** The CPU decides in/out-of-source from `sx`
    ///   accumulated along the row (`compute_range`); the GPU evaluates the map
    ///   directly. A coordinate sitting within an ulp of 0 or `src_w` can be
    ///   judged inside by one and outside by the other — one writes an
    ///   interpolated value, the other writes the `BORDER_CONSTANT` 0.
    /// * **Rounding tie (nearest only).** A coordinate landing on an exact `.5`
    ///   can round to either neighbour.
    ///
    /// Excluding only these keeps the assertion strict everywhere else — nearest
    /// stays bit-exact — instead of hiding real disagreement behind a tolerance.
    ///
    /// Pass `eps = 0.0` to disable exclusion entirely. That is correct whenever
    /// both sides compute the coordinate *exactly* (see [`DYADIC_M`]): with no
    /// drift there is nothing to be ambiguous about — a `.5` tie resolves the
    /// same way on both paths, since `roundf` and Rust's `f32::round` are both
    /// half-away-from-zero — so those runs demand agreement even on ties.
    fn ambiguous_pixels(
        w: usize,
        h: usize,
        m: &[f32; 6],
        mode: InterpolationMode,
        eps: f32,
    ) -> Vec<bool> {
        let mi = crate::warp::invert_affine_transform(m);
        let (fw, fh) = (w as f32, h as f32);
        let mut out = vec![false; w * h];
        for y in 0..h {
            for x in 0..w {
                let sx = mi[0] * x as f32 + mi[1] * y as f32 + mi[2];
                let sy = mi[3] * x as f32 + mi[4] * y as f32 + mi[5];

                let near = |v: f32, edge: f32| (v - edge).abs() < eps;
                let mut amb = near(sx, 0.0) || near(sx, fw) || near(sy, 0.0) || near(sy, fh);

                if matches!(mode, InterpolationMode::Nearest) {
                    let near_half = |v: f32| (v - v.floor() - 0.5).abs() < eps;
                    amb = amb || near_half(sx) || near_half(sy);
                }
                out[y * w + x] = amb;
            }
        }
        out
    }

    /// Compare CPU vs GPU, skipping the ambiguous pixels above. Returns how many
    /// were skipped so callers can assert they stay a rounding curiosity.
    fn assert_matches_cpu(
        w: usize,
        h: usize,
        m: &[f32; 6],
        mode: InterpolationMode,
        tol: f32,
        eps: f32,
    ) -> usize {
        let (cpu, gpu) = cpu_and_gpu(w, h, m, mode);
        let amb = ambiguous_pixels(w, h, m, mode, eps);
        let mut skipped = 0;
        for (p, &is_amb) in amb.iter().enumerate() {
            if is_amb {
                skipped += 1;
                continue;
            }
            for c in 0..3 {
                let i = p * 3 + c;
                let d = (gpu[i] - cpu[i]).abs();
                assert!(
                    d <= tol,
                    "{w}x{h} {mode:?}: pixel {p} ch {c} diff {d} > {tol} (cpu {} gpu {})",
                    cpu[i],
                    gpu[i]
                );
            }
        }
        assert!(
            skipped * 20 < w * h,
            "{skipped}/{} pixels excluded as ambiguous — too many to still be a \
             boundary/tie effect; the coordinate mapping itself is probably wrong",
            w * h
        );
        skipped
    }

    /// Regression for issue #1000: the pitch-2D texture path required
    /// `src_w * 3 * 4` bytes to be a multiple of the 32-byte texture pitch
    /// alignment, so every width not a multiple of 8 failed the launch outright
    /// with `CUDA_ERROR_INVALID_VALUE`. The `__ldg` kernels have no such
    /// constraint — every width below must both launch and match the CPU.
    #[test]
    fn warp_affine_unaligned_widths_match_cpu() {
        // 128/136 are 8-aligned (worked before); the rest are the regression.
        const WIDTHS: &[usize] = &[128, 129, 130, 132, 133, 135, 136];

        for &w in WIDTHS {
            assert_matches_cpu(
                w,
                49,
                &DYADIC_M,
                InterpolationMode::Bilinear,
                BILINEAR_TOL,
                0.0,
            );
            assert_matches_cpu(w, 49, &DYADIC_M, InterpolationMode::Nearest, 0.0, 0.0);
        }
    }

    #[test]
    fn warp_affine_bilinear_matches_cpu() {
        assert_matches_cpu(
            320,
            240,
            &DYADIC_M,
            InterpolationMode::Bilinear,
            BILINEAR_TOL,
            0.0,
        );
    }

    /// Non-dyadic rotation: guards against gross errors on a realistic transform.
    /// See ROTATION_TOL for why this cannot be strict.
    #[test]
    fn warp_affine_rotation_matches_cpu() {
        let m = crate::warp::get_rotation_matrix2d((160.0, 120.0), 37.0, 1.3);
        assert_matches_cpu(
            320,
            240,
            &m,
            InterpolationMode::Bilinear,
            ROTATION_TOL,
            1e-3,
        );
    }

    /// Nearest selects a single source pixel on both paths, so away from exact
    /// rounding ties any difference is a coordinate-rule bug, not float error.
    #[test]
    fn warp_affine_nearest_is_bit_exact_vs_cpu() {
        assert_matches_cpu(320, 240, &DYADIC_M, InterpolationMode::Nearest, 0.0, 0.0);
    }

    #[test]
    fn warp_affine_identity_is_exact_copy() {
        let m = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        for mode in [InterpolationMode::Bilinear, InterpolationMode::Nearest] {
            let (cpu, gpu) = cpu_and_gpu(65, 33, &m, mode);
            assert_eq!(
                gpu, cpu,
                "identity warp ({mode:?}) must reproduce the source"
            );
        }
    }

    /// Every launcher must reject a short src slice — the kernels read src via
    /// raw `__ldg`, so an unvalidated short slice is an OOB device read.
    #[test]
    fn all_launchers_reject_short_src_slices() {
        let stream = default_stream();
        let ctx = &stream.context();
        let m = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];

        // src holds only half the pixels it claims to.
        let d_src = stream.clone_htod(&vec![0.0f32; 32 * 32 * 3 / 2]).unwrap();
        let mut d_dst = stream.alloc_zeros::<f32>(32 * 32 * 3).unwrap();

        type Launcher = fn(
            &std::sync::Arc<cudarc::driver::CudaContext>,
            &std::sync::Arc<cudarc::driver::CudaStream>,
            &cudarc::driver::CudaSlice<f32>,
            &mut cudarc::driver::CudaSlice<f32>,
            u32,
            u32,
            u32,
            u32,
            &[f32; 6],
            Option<(u32, u32)>,
        ) -> Result<(), CudaWarpAffineError>;
        const LAUNCHERS: [(&str, Launcher); 4] = [
            ("bilinear", launch_warp_affine_bilinear_cuda),
            ("nearest", launch_warp_affine_nearest_cuda),
            ("bicubic", launch_warp_affine_bicubic_cuda),
            ("lanczos", launch_warp_affine_lanczos_cuda),
        ];

        for (name, launch) in LAUNCHERS {
            let err =
                launch(ctx, &stream, &d_src, &mut d_dst, 32, 32, 32, 32, &m, None).expect_err(name);
            assert!(
                matches!(err, CudaWarpAffineError::SliceTooSmall { .. }),
                "{name}: expected SliceTooSmall, got {err:?}"
            );

            // Zero dst dims and a zero block component must be clean errors,
            // not the div_ceil(0) panic an earlier revision hit in make_config.
            let ok_src = stream.clone_htod(&vec![0.0f32; 8 * 8 * 3]).unwrap();
            let mut ok_dst = stream.alloc_zeros::<f32>(8 * 8 * 3).unwrap();
            assert!(
                launch(ctx, &stream, &ok_src, &mut ok_dst, 8, 8, 0, 8, &m, None).is_err(),
                "{name}: zero dst width must error"
            );
            assert!(
                launch(
                    ctx,
                    &stream,
                    &ok_src,
                    &mut ok_dst,
                    8,
                    8,
                    8,
                    8,
                    &m,
                    Some((0, 8))
                )
                .is_err(),
                "{name}: zero block component must error"
            );
        }
    }

    /// Bicubic identity: Keys weights at frac 0 are exactly [0, 1, 0, 0], so an
    /// identity warp must be a bit-exact copy. Guards the (otherwise untested)
    /// bicubic kernel, including its 64-bit index math.
    #[test]
    fn warp_affine_bicubic_identity_is_exact_copy() {
        let (w, h) = (65usize, 33usize);
        let data = pattern_f32(w * h * 3);
        let stream = default_stream();
        let ctx = &stream.context();
        let d_src = stream.clone_htod(&data).unwrap();
        let mut d_dst = stream.alloc_zeros::<f32>(w * h * 3).unwrap();
        launch_warp_affine_bicubic_cuda(
            ctx,
            &stream,
            &d_src,
            &mut d_dst,
            w as u32,
            h as u32,
            w as u32,
            h as u32,
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            None,
        )
        .unwrap();
        let gpu: Vec<f32> = stream.clone_dtoh(&d_dst).unwrap();
        stream.synchronize().unwrap();
        assert_eq!(gpu, data, "bicubic identity warp must reproduce the source");
    }

    /// Lanczos identity: lanczos3(0) = 1 exactly, but the ±1/±2 weights are
    /// only ~1e-7 (sin(πk) under a float π), so identity is near-copy within
    /// normalization error, not bit-exact.
    #[test]
    fn warp_affine_lanczos_identity_is_near_copy() {
        let (w, h) = (65usize, 33usize);
        let data = pattern_f32(w * h * 3);
        let stream = default_stream();
        let ctx = &stream.context();
        let d_src = stream.clone_htod(&data).unwrap();
        let mut d_dst = stream.alloc_zeros::<f32>(w * h * 3).unwrap();
        launch_warp_affine_lanczos_cuda(
            ctx,
            &stream,
            &d_src,
            &mut d_dst,
            w as u32,
            h as u32,
            w as u32,
            h as u32,
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            None,
        )
        .unwrap();
        let gpu: Vec<f32> = stream.clone_dtoh(&d_dst).unwrap();
        stream.synchronize().unwrap();
        let diff = crate::cuda::color::test_utils::max_abs_diff_f32(&gpu, &data);
        assert!(diff <= 1e-4, "lanczos identity max abs diff {diff} > 1e-4");
    }
}
