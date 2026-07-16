//! Native CUDA warp-perspective kernels for `kornia-imgproc`.
//!
//! # Algorithm
//!
//! All kernels use **inverse mapping**: each output thread computes the
//! floating-point source coordinate by applying the inverted 3×3 homography
//! to its `(dst_x, dst_y)` position and dividing by the homogeneous weight `w`.
//! Pixels whose `|w| < 1e-10` (degenerate projection) are written as 0.
//!
//! **Border handling.** A destination pixel whose inverse-mapped source
//! coordinate falls outside `[0, src_w) × [0, src_h)` is written as 0
//! (`BORDER_CONSTANT`). Inside that range:
//! * Bilinear replicates `val00` for taps that leave the right/bottom edge, and
//!   nearest rounds-then-clamps — both matching the CPU interpolators exactly.
//! * Bicubic / Lanczos-3 clamp OOB taps in the 4×4 / 6×6 support window to the
//!   nearest source border pixel (`BORDER_REPLICATE`).
//!
//! # Optimisations
//!
//! * **`__ldg` source reads** — all kernels read the source through the read-only
//!   data cache. Bilinear/nearest previously used a pitch-2D texture object,
//!   which constrained `src_w` to multiples of 8 (issue #1000); see
//!   [`super::warp_affine`] for the full rationale.
//! * **`CU_FUNC_CACHE_PREFER_L1`** — enlarges L1 to 64 KB; no shared memory used.
//! * **32×8 thread block (default)** — full warp per output row for coalesced writes.
//! * **`1/w` reciprocal** (bicubic/Lanczos only) — one FP division replaces two
//!   per output pixel. Bilinear/nearest divide directly to stay byte-exact with
//!   the CPU `transform_point`, which divides.
//!
//! # Public API
//!
//! * [`launch_warp_perspective_bilinear_cuda`] — bilinear, 3-ch f32.
//! * [`launch_warp_perspective_nearest_cuda`]  — nearest-neighbor, 3-ch f32.
//! * [`launch_warp_perspective_bicubic_cuda`]  — bicubic (Keys a=-0.5), 3-ch f32.
//! * [`launch_warp_perspective_lanczos_cuda`]  — Lanczos-3 (6×6 taps), 3-ch f32.

use std::sync::{Arc, OnceLock};

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use kornia_tensor::CudaKernel;

use super::{check_geometry, make_config, try_compile_with_l1};
use crate::warp::invert_homography;

// ── CUDA C source: bilinear warp perspective ─────────────────────────────────
//
// The only difference from warp_affine_bilinear_3c is the coordinate transform:
// affine uses a 2×3 linear map; perspective adds a w-divide from the third row
// of the 3×3 homography.

static BILINEAR_SRC: &str = r#"
extern "C" __global__ void warp_perspective_bilinear_3c(
    const float* __restrict__ src,
    float* __restrict__       dst,
    unsigned int src_w,
    unsigned int src_h,
    unsigned int dst_w,
    unsigned int dst_h,
    float h0, float h1, float h2,
    float h3, float h4, float h5,
    float h6, float h7, float h8
) {
    unsigned int gx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int gy = blockIdx.y * blockDim.y + threadIdx.y;
    if (gx >= dst_w || gy >= dst_h) return;

    unsigned long long out = ((unsigned long long)gy * dst_w + gx) * 3ull;
    float w = h6 * (float)gx + h7 * (float)gy + h8;
    if (fabsf(w) < 1e-10f) {
        dst[out] = 0.0f; dst[out+1] = 0.0f; dst[out+2] = 0.0f;
        return;
    }
    // Direct division, NOT reciprocal-multiply: the CPU `transform_point`
    // divides, and x/w rounds differently from x*(1/w). Numerator grouping is
    // the CPU's left-to-right `m0*x + m1*y + m2`. Byte-exact contract with the
    // parity tests; do not "optimize" back to a reciprocal.
    float sx = (h0 * (float)gx + h1 * (float)gy + h2) / w;
    float sy = (h3 * (float)gx + h4 * (float)gy + h5) / w;

    // BORDER_CONSTANT: outside the source, emit 0 (CPU valid-range guard).
    if (sx < 0.0f || sx >= (float)src_w || sy < 0.0f || sy >= (float)src_h) {
        dst[out] = 0.0f; dst[out+1] = 0.0f; dst[out+2] = 0.0f;
        return;
    }

    // sx, sy >= 0 here, so truncation == floor (matches CPU `u.trunc()`).
    unsigned int x0 = (unsigned int)sx;
    unsigned int y0 = (unsigned int)sy;
    float fx = sx - (float)x0;
    float fy = sy - (float)y0;

    bool has_x1 = (x0 + 1u) < src_w;
    bool has_y1 = (y0 + 1u) < src_h;

    // 64-bit index math: (y*w + x)*3 wraps 32 bits at ~1.43 gigapixels.
    unsigned long long b00 = ((unsigned long long)y0 * src_w + x0) * 3ull;
    unsigned long long b01 = has_x1 ? b00 + 3ull                : b00;
    unsigned long long b10 = has_y1 ? b00 + (unsigned long long)src_w * 3ull : b00;
    // val11 replicates val00 when EITHER axis overflows — CPU rule.
    unsigned long long b11 =
        (has_x1 && has_y1) ? b00 + (unsigned long long)src_w * 3ull + 3ull : b00;

    float fxx = 1.0f - fx;
    float fyy = 1.0f - fy;
    float w00 = fxx * fyy;
    float w01 = fx  * fyy;
    float w10 = fxx * fy;
    float w11 = fx  * fy;

    #pragma unroll
    for (unsigned int c = 0u; c < 3u; ++c) {
        float v00 = __ldg(&src[b00 + c]);
        float v01 = __ldg(&src[b01 + c]);
        float v10 = __ldg(&src[b10 + c]);
        float v11 = __ldg(&src[b11 + c]);
        // Plain sum in the CPU tap order (val00, x+1, y+1, both): matches
        // `bilinear_interpolation` under --fmad=false. Same ops, same roundings.
        dst[out + c] = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11;
    }
}
"#;

// ── CUDA C source: nearest-neighbor warp perspective ─────────────────────────

static NEAREST_SRC: &str = r#"
extern "C" __global__ void warp_perspective_nearest_3c(
    const float* __restrict__ src,
    float* __restrict__       dst,
    unsigned int src_w,
    unsigned int src_h,
    unsigned int dst_w,
    unsigned int dst_h,
    float h0, float h1, float h2,
    float h3, float h4, float h5,
    float h6, float h7, float h8
) {
    unsigned int gx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int gy = blockIdx.y * blockDim.y + threadIdx.y;
    if (gx >= dst_w || gy >= dst_h) return;

    unsigned long long out = ((unsigned long long)gy * dst_w + gx) * 3ull;
    float w = h6 * (float)gx + h7 * (float)gy + h8;
    if (fabsf(w) < 1e-10f) {
        dst[out] = 0.0f; dst[out+1] = 0.0f; dst[out+2] = 0.0f;
        return;
    }
    // Direct division, NOT reciprocal-multiply: the CPU `transform_point`
    // divides, and x/w rounds differently from x*(1/w). Numerator grouping is
    // the CPU's left-to-right `m0*x + m1*y + m2`. Byte-exact contract with the
    // parity tests; do not "optimize" back to a reciprocal.
    float sx = (h0 * (float)gx + h1 * (float)gy + h2) / w;
    float sy = (h3 * (float)gx + h4 * (float)gy + h5) / w;

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

// ── CUDA C source: bicubic warp perspective ───────────────────────────────────
//
// Keys cubic (a = -0.5, matches OpenCV INTER_CUBIC).  Same weight computation
// as warp_affine_bicubic_3c; only the coordinate transform differs (w-divide).

static BICUBIC_SRC: &str = r#"
extern "C" __global__ void warp_perspective_bicubic_3c(
    const float* __restrict__ src,
    float* __restrict__       dst,
    unsigned int src_w,
    unsigned int src_h,
    unsigned int dst_w,
    unsigned int dst_h,
    float h0, float h1, float h2,
    float h3, float h4, float h5,
    float h6, float h7, float h8
) {
    unsigned int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (dst_x >= dst_w || dst_y >= dst_h) return;

    unsigned long long out = ((unsigned long long)dst_y * dst_w + dst_x) * 3ull;
    float w = h6 * (float)dst_x + h7 * (float)dst_y + h8;
    if (fabsf(w) < 1e-10f) {
        dst[out] = 0.0f; dst[out+1] = 0.0f; dst[out+2] = 0.0f;
        return;
    }
    // Direct division like the CPU transform_point — x/w rounds differently
    // from x*(1/w). Same contract as the bilinear/nearest kernels above.
    float sx = (h0 * (float)dst_x + h1 * (float)dst_y + h2) / w;
    float sy = (h3 * (float)dst_x + h4 * (float)dst_y + h5) / w;

    if (sx < 0.0f || sx >= (float)src_w || sy < 0.0f || sy >= (float)src_h) {
        dst[out] = 0.0f; dst[out+1] = 0.0f; dst[out+2] = 0.0f;
        return;
    }

    int x0 = (int)floorf(sx);
    int y0 = (int)floorf(sy);
    float frac_x = sx - (float)x0;
    float frac_y = sy - (float)y0;

    // Horner-form Keys cubic weights (a = -0.5), branch-free.
    float wx[4], wy[4];
    {
        float t;
        // Explicit fmaf: --fmad=false no longer contracts plain expressions,
        // and the CPU twin (keys_weights) uses mul_add — same fused chain.
        t = 1.0f + frac_x; wx[0] = fmaf(fmaf(fmaf(-0.5f, t, 2.5f), t, -4.0f), t, 2.0f);
        t =         frac_x; wx[1] = fmaf(fmaf( 1.5f, t, -2.5f) * t,       t, 1.0f);
        t = 1.0f - frac_x; wx[2] = fmaf(fmaf( 1.5f, t, -2.5f) * t,       t, 1.0f);
        t = 2.0f - frac_x; wx[3] = fmaf(fmaf(fmaf(-0.5f, t, 2.5f), t, -4.0f), t, 2.0f);
        t = 1.0f + frac_y; wy[0] = fmaf(fmaf(fmaf(-0.5f, t, 2.5f), t, -4.0f), t, 2.0f);
        t =         frac_y; wy[1] = fmaf(fmaf( 1.5f, t, -2.5f) * t,       t, 1.0f);
        t = 1.0f - frac_y; wy[2] = fmaf(fmaf( 1.5f, t, -2.5f) * t,       t, 1.0f);
        t = 2.0f - frac_y; wy[3] = fmaf(fmaf(fmaf(-0.5f, t, 2.5f), t, -4.0f), t, 2.0f);
    }

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
            float wt = wx[dx] * wy[dy];
            acc0 = fmaf(wt, __ldg(&src[b]),   acc0);
            acc1 = fmaf(wt, __ldg(&src[b+1]), acc1);
            acc2 = fmaf(wt, __ldg(&src[b+2]), acc2);
        }
    }
    dst[out]   = acc0;
    dst[out+1] = acc1;
    dst[out+2] = acc2;
}
"#;

// ── CUDA C source: Lanczos-3 warp perspective ────────────────────────────────
//
// 3-lobe Lanczos filter (6×6 taps, 36 source reads per output pixel).
// Same weight computation as warp_affine_lanczos_3c; only the coordinate
// transform differs (w-divide from the homography third row).

static LANCZOS_SRC: &str = r#"
__device__ inline float sin_pi(float x) {
    // sin(pi*x) via exact integer reduction + odd Taylor polynomial in plain
    // mul/add. Deterministic and bit-identical to the Rust twin under
    // --fmad=false — unlike sinf, whose rounding differs between the CUDA
    // math library and host libm. |x| <= 3 here, so k is exact.
    float k = roundf(x);
    float r = x - k;
    float z = 3.14159265358979f * r;
    float z2 = z * z;
    float p = -2.5052108385e-08f;
    p = p * z2 + 2.7557319224e-06f;
    p = p * z2 + -1.9841269841e-04f;
    p = p * z2 + 8.3333333333e-03f;
    p = p * z2 + -1.6666666667e-01f;
    float s = z + z * z2 * p;
    return (((int)k) & 1) ? -s : s;
}

__device__ inline float lanczos3(float x) {
    const float PI = 3.14159265358979f;
    if (fabsf(x) < 1e-5f) return 1.0f;
    if (fabsf(x) >= 3.0f) return 0.0f;
    float pix  = PI * x;
    float pix3 = pix * 0.33333333f;
    return sin_pi(x) * sin_pi(x * (1.0f / 3.0f)) / (pix * pix3);
}

extern "C" __global__ void warp_perspective_lanczos_3c(
    const float* __restrict__ src,
    float* __restrict__       dst,
    unsigned int src_w,
    unsigned int src_h,
    unsigned int dst_w,
    unsigned int dst_h,
    float h0, float h1, float h2,
    float h3, float h4, float h5,
    float h6, float h7, float h8
) {
    unsigned int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (dst_x >= dst_w || dst_y >= dst_h) return;

    unsigned long long out = ((unsigned long long)dst_y * dst_w + dst_x) * 3ull;
    float w = h6 * (float)dst_x + h7 * (float)dst_y + h8;
    if (fabsf(w) < 1e-10f) {
        dst[out] = 0.0f; dst[out+1] = 0.0f; dst[out+2] = 0.0f;
        return;
    }
    // Direct division like the CPU transform_point — x/w rounds differently
    // from x*(1/w). Same contract as the bilinear/nearest kernels above.
    float sx = (h0 * (float)dst_x + h1 * (float)dst_y + h2) / w;
    float sy = (h3 * (float)dst_x + h4 * (float)dst_y + h5) / w;

    if (sx < 0.0f || sx >= (float)src_w || sy < 0.0f || sy >= (float)src_h) {
        dst[out] = 0.0f; dst[out+1] = 0.0f; dst[out+2] = 0.0f;
        return;
    }

    int x0 = (int)floorf(sx);
    int y0 = (int)floorf(sy);
    float frac_x = sx - (float)x0;
    float frac_y = sy - (float)y0;

    float wx[6], wy[6];
    wx[0] = lanczos3(frac_x + 2.0f); wx[1] = lanczos3(frac_x + 1.0f);
    wx[2] = lanczos3(frac_x);        wx[3] = lanczos3(frac_x - 1.0f);
    wx[4] = lanczos3(frac_x - 2.0f); wx[5] = lanczos3(frac_x - 3.0f);
    wy[0] = lanczos3(frac_y + 2.0f); wy[1] = lanczos3(frac_y + 1.0f);
    wy[2] = lanczos3(frac_y);        wy[3] = lanczos3(frac_y - 1.0f);
    wy[4] = lanczos3(frac_y - 2.0f); wy[5] = lanczos3(frac_y - 3.0f);

    float sum_wx = wx[0]+wx[1]+wx[2]+wx[3]+wx[4]+wx[5];
    float sum_wy = wy[0]+wy[1]+wy[2]+wy[3]+wy[4]+wy[5];
    float inv_x = 1.0f / sum_wx;
    float inv_y = 1.0f / sum_wy;
    #pragma unroll
    for (int i = 0; i < 6; i++) { wx[i] *= inv_x; wy[i] *= inv_y; }

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
    dst[out]   = acc0;
    dst[out+1] = acc1;
    dst[out+2] = acc2;
}
"#;

// ── Kernel caches ─────────────────────────────────────────────────────────────

static BILINEAR_KERNEL: OnceLock<Result<CudaKernel, String>> = OnceLock::new();
static NEAREST_KERNEL: OnceLock<Result<CudaKernel, String>> = OnceLock::new();
static BICUBIC_KERNEL: OnceLock<Result<CudaKernel, String>> = OnceLock::new();
static LANCZOS_KERNEL: OnceLock<Result<CudaKernel, String>> = OnceLock::new();

// ── Error type ────────────────────────────────────────────────────────────────

/// Error returned by the CUDA warp-perspective launchers.
#[derive(Debug, thiserror::Error)]
pub enum CudaWarpPerspectiveError {
    /// CUDA driver / compile / launch error.
    #[error("CUDA warp-perspective error: {0}")]
    Cuda(String),
    /// A device slice is smaller than required.
    #[error("device slice '{what}' length {got} < required {need}")]
    SliceTooSmall {
        /// Which operand was too small (`"src"` or `"dst"`).
        what: &'static str,
        /// Actual slice length (in elements).
        got: usize,
        /// Minimum required length.
        need: usize,
    },
    /// The homography matrix is singular and cannot be inverted.
    #[error("homography matrix is singular (|det| < 1e-10)")]
    SingularHomography,
}

// ── Internal helpers ──────────────────────────────────────────────────────────

fn check_image_slice(
    slice: &CudaSlice<f32>,
    what: &'static str,
    width: u32,
    height: u32,
) -> Result<(), CudaWarpPerspectiveError> {
    let need = (width as usize) * (height as usize) * 3;
    if slice.len() < need {
        return Err(CudaWarpPerspectiveError::SliceTooSmall {
            what,
            got: slice.len(),
            need,
        });
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_and_invert(
    src: &CudaSlice<f32>,
    dst: &CudaSlice<f32>,
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    h: &[f32; 9],
    block_dim: Option<(u32, u32)>,
) -> Result<[f32; 9], CudaWarpPerspectiveError> {
    check_geometry(src_width, src_height, dst_width, dst_height, block_dim)
        .map_err(CudaWarpPerspectiveError::Cuda)?;
    check_image_slice(src, "src", src_width, src_height)?;
    check_image_slice(dst, "dst", dst_width, dst_height)?;
    invert_homography(h).ok_or(CudaWarpPerspectiveError::SingularHomography)
}

// ── Public launchers ──────────────────────────────────────────────────────────

/// Launch the bilinear warp-perspective kernel for a 3-channel f32 image.
///
/// Applies the 3×3 forward homography `h` to warp `src` into `dst`.
/// The matrix is inverted internally so the API matches the CPU
/// [`warp_perspective`](crate::warp::warp_perspective).
///
/// Source pixels are read via `__ldg`; any source width is supported.
/// Out-of-bounds coordinates return 0 (`BORDER_CONSTANT`), and taps that leave
/// the right/bottom edge replicate `val00`, matching the CPU interpolator.
/// Pixels whose homogeneous weight `|w| < 1e-10` after inversion are
/// written as 0 (degenerate projection).
///
/// # Arguments
///
/// * `ctx`        — CUDA context for one-time kernel compilation.
/// * `stream`     — Stream for kernel execution.
/// * `src`        — Source image, `src_h × src_w × 3` f32 (HWC).
/// * `dst`        — Destination buffer, at least `dst_h × dst_w × 3` f32.
/// * `src_width`  / `src_height` — Source image dimensions.
/// * `dst_width`  / `dst_height` — Output canvas dimensions.
/// * `h`          — Forward 3×3 homography in row-major order.
/// * `block_dim`  — Optional thread-block override; `None` → 32×8.
///
/// # Errors
///
/// Returns [`CudaWarpPerspectiveError`] on compile failure, singular homography,
/// launch error, or if any slice is too small.
#[allow(clippy::too_many_arguments)]
pub fn launch_warp_perspective_bilinear_cuda(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    h: &[f32; 9],
    block_dim: Option<(u32, u32)>,
) -> Result<(), CudaWarpPerspectiveError> {
    let hi = validate_and_invert(
        src, dst, src_width, src_height, dst_width, dst_height, h, block_dim,
    )?;
    let kernel = BILINEAR_KERNEL
        .get_or_init(|| try_compile_with_l1(ctx, BILINEAR_SRC, "warp_perspective_bilinear_3c"))
        .as_ref()
        .map_err(|e| CudaWarpPerspectiveError::Cuda(e.clone()))?;

    kernel
        .launch_builder(stream)
        .arg(src)
        .arg(dst)
        .arg(&src_width)
        .arg(&src_height)
        .arg(&dst_width)
        .arg(&dst_height)
        .arg(&hi[0])
        .arg(&hi[1])
        .arg(&hi[2])
        .arg(&hi[3])
        .arg(&hi[4])
        .arg(&hi[5])
        .arg(&hi[6])
        .arg(&hi[7])
        .arg(&hi[8])
        .launch_2d(
            dst_width,
            dst_height,
            make_config(dst_width, dst_height, block_dim),
        )
        .map_err(|e| CudaWarpPerspectiveError::Cuda(e.to_string()))
}

/// Launch the nearest-neighbor warp-perspective kernel for a 3-channel f32 image.
///
/// Same as [`launch_warp_perspective_bilinear_cuda`] but uses round-to-nearest
/// source sampling (half-away-from-zero, then clamped — matching the CPU
/// nearest interpolator).  Faster; suitable for integer-aligned transforms.
///
/// # Arguments
///
/// See [`launch_warp_perspective_bilinear_cuda`] — arguments are identical.
///
/// # Errors
///
/// Returns [`CudaWarpPerspectiveError`] on compile failure, singular homography,
/// launch error, or if any slice is too small.
#[allow(clippy::too_many_arguments)]
pub fn launch_warp_perspective_nearest_cuda(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    h: &[f32; 9],
    block_dim: Option<(u32, u32)>,
) -> Result<(), CudaWarpPerspectiveError> {
    let hi = validate_and_invert(
        src, dst, src_width, src_height, dst_width, dst_height, h, block_dim,
    )?;
    let kernel = NEAREST_KERNEL
        .get_or_init(|| try_compile_with_l1(ctx, NEAREST_SRC, "warp_perspective_nearest_3c"))
        .as_ref()
        .map_err(|e| CudaWarpPerspectiveError::Cuda(e.clone()))?;

    kernel
        .launch_builder(stream)
        .arg(src)
        .arg(dst)
        .arg(&src_width)
        .arg(&src_height)
        .arg(&dst_width)
        .arg(&dst_height)
        .arg(&hi[0])
        .arg(&hi[1])
        .arg(&hi[2])
        .arg(&hi[3])
        .arg(&hi[4])
        .arg(&hi[5])
        .arg(&hi[6])
        .arg(&hi[7])
        .arg(&hi[8])
        .launch_2d(
            dst_width,
            dst_height,
            make_config(dst_width, dst_height, block_dim),
        )
        .map_err(|e| CudaWarpPerspectiveError::Cuda(e.to_string()))
}

/// Launch the bicubic warp-perspective kernel for a 3-channel f32 image.
///
/// Uses Keys cubic (a = −0.5), matching OpenCV `INTER_CUBIC`.
/// Reads the source via `__ldg` with a 4×4 tap window, like all kernels in
/// this module.
/// Out-of-bounds taps are clamped to the source border (BORDER_REPLICATE);
/// pixels whose inverse-mapped centre falls outside the source are written as 0.
///
/// # Arguments
///
/// See [`launch_warp_perspective_bilinear_cuda`] — arguments are identical.
///
/// # Errors
///
/// Returns [`CudaWarpPerspectiveError::Cuda`] if the kernel fails to compile
/// (NVRTC; happens at most once per process) or fails to launch.
/// Returns [`CudaWarpPerspectiveError::SingularHomography`] for degenerate `h`.
#[allow(clippy::too_many_arguments)]
pub fn launch_warp_perspective_bicubic_cuda(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    h: &[f32; 9],
    block_dim: Option<(u32, u32)>,
) -> Result<(), CudaWarpPerspectiveError> {
    let hi = validate_and_invert(
        src, dst, src_width, src_height, dst_width, dst_height, h, block_dim,
    )?;
    let kernel = BICUBIC_KERNEL
        .get_or_init(|| try_compile_with_l1(ctx, BICUBIC_SRC, "warp_perspective_bicubic_3c"))
        .as_ref()
        .map_err(|e| CudaWarpPerspectiveError::Cuda(e.clone()))?;

    kernel
        .launch_builder(stream)
        .arg(src)
        .arg(dst)
        .arg(&src_width)
        .arg(&src_height)
        .arg(&dst_width)
        .arg(&dst_height)
        .arg(&hi[0])
        .arg(&hi[1])
        .arg(&hi[2])
        .arg(&hi[3])
        .arg(&hi[4])
        .arg(&hi[5])
        .arg(&hi[6])
        .arg(&hi[7])
        .arg(&hi[8])
        .launch_2d(
            dst_width,
            dst_height,
            make_config(dst_width, dst_height, block_dim),
        )
        .map_err(|e| CudaWarpPerspectiveError::Cuda(e.to_string()))
}

/// Launch the Lanczos-3 warp-perspective kernel for a 3-channel f32 image.
///
/// Uses a 6×6 tap (36 source reads per output pixel) with sinc-based Lanczos
/// weights normalised per output pixel to prevent brightness drift at clamped
/// boundaries.  Highest quality; slower than bicubic.
///
/// Like bicubic, reads from raw device memory via `__ldg`.  OOB taps within the
/// 6×6 neighbourhood are clamped (BORDER_REPLICATE); pixels whose inverse-mapped
/// centre falls outside the source are written as 0 (BORDER_CONSTANT).
///
/// # Arguments
///
/// See [`launch_warp_perspective_bilinear_cuda`] — arguments are identical.
///
/// # Errors
///
/// Returns [`CudaWarpPerspectiveError::Cuda`] on compile or launch failure.
/// Returns [`CudaWarpPerspectiveError::SingularHomography`] for degenerate `h`.
#[allow(clippy::too_many_arguments)]
pub fn launch_warp_perspective_lanczos_cuda(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    h: &[f32; 9],
    block_dim: Option<(u32, u32)>,
) -> Result<(), CudaWarpPerspectiveError> {
    let hi = validate_and_invert(
        src, dst, src_width, src_height, dst_width, dst_height, h, block_dim,
    )?;
    let kernel = LANCZOS_KERNEL
        .get_or_init(|| try_compile_with_l1(ctx, LANCZOS_SRC, "warp_perspective_lanczos_3c"))
        .as_ref()
        .map_err(|e| CudaWarpPerspectiveError::Cuda(e.clone()))?;

    kernel
        .launch_builder(stream)
        .arg(src)
        .arg(dst)
        .arg(&src_width)
        .arg(&src_height)
        .arg(&dst_width)
        .arg(&dst_height)
        .arg(&hi[0])
        .arg(&hi[1])
        .arg(&hi[2])
        .arg(&hi[3])
        .arg(&hi[4])
        .arg(&hi[5])
        .arg(&hi[6])
        .arg(&hi[7])
        .arg(&hi[8])
        .launch_2d(
            dst_width,
            dst_height,
            make_config(dst_width, dst_height, block_dim),
        )
        .map_err(|e| CudaWarpPerspectiveError::Cuda(e.to_string()))
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::cuda::color::test_utils::{default_stream, pattern_f32};
    use crate::interpolation::InterpolationMode;
    use crate::warp::warp_perspective;
    use kornia_image::{Image, ImageSize};

    /// Run the CPU reference and the CUDA kernel on the same input and return
    /// both buffers. Both destinations start zeroed — the CPU leaves
    /// out-of-bounds pixels untouched while the kernel writes 0, so a zeroed
    /// destination is part of the comparison contract.
    fn cpu_and_gpu(
        w: usize,
        h: usize,
        hm: &[f32; 9],
        interpolation: InterpolationMode,
    ) -> (Vec<f32>, Vec<f32>) {
        let data = pattern_f32(w * h * 3);
        let size = ImageSize {
            width: w,
            height: h,
        };

        let src = Image::<f32, 3>::new(size, data.clone()).unwrap();
        let mut cpu = Image::<f32, 3>::from_size_val(size, 0.0).unwrap();
        warp_perspective(&src, &mut cpu, hm, interpolation).unwrap();

        let stream = default_stream();
        let ctx = &stream.context();
        let d_src = stream.clone_htod(&data).unwrap();
        let mut d_dst = stream.alloc_zeros::<f32>(w * h * 3).unwrap();

        let (wu, hu) = (w as u32, h as u32);
        match interpolation {
            InterpolationMode::Bilinear => launch_warp_perspective_bilinear_cuda(
                ctx, &stream, &d_src, &mut d_dst, wu, hu, wu, hu, hm, None,
            ),
            InterpolationMode::Nearest => launch_warp_perspective_nearest_cuda(
                ctx, &stream, &d_src, &mut d_dst, wu, hu, wu, hu, hm, None,
            ),
            other => panic!("unsupported mode in test: {other:?}"),
        }
        .unwrap_or_else(|e| panic!("launch failed at {w}x{h} ({interpolation:?}): {e}"));

        let gpu: Vec<f32> = stream.clone_dtoh(&d_dst).unwrap();
        stream.synchronize().unwrap();

        (cpu.as_slice().to_vec(), gpu)
    }

    /// Byte-exact: the CPU `transform_point` divides the identically grouped
    /// numerator by `w`, and the kernels do the same (direct division, not
    /// reciprocal-multiply) under `--fmad=false`; `bilinear_interpolation`'s
    /// expression shape matches the kernel's tap sum. Both paths evaluate the
    /// coordinate per pixel, so there is no span/accumulation caveat at all.
    fn assert_bit_exact(w: usize, h: usize, hm: &[f32; 9], interpolation: InterpolationMode) {
        let (cpu, gpu) = cpu_and_gpu(w, h, hm, interpolation);
        for (i, (c, g)) in cpu.iter().zip(&gpu).enumerate() {
            assert!(
                c.to_bits() == g.to_bits(),
                "{w}x{h} {interpolation:?}: element {i}: cpu {c} ({:#010x}) gpu {g} ({:#010x})",
                c.to_bits(),
                g.to_bits()
            );
        }
    }

    /// Genuinely projective homography (non-zero third row) at a non-dyadic,
    /// unaligned size: the w-divide is exercised on every pixel.
    #[test]
    fn warp_perspective_projective_matches_cpu() {
        let hm = [
            1.03,
            0.05,
            -3.0,
            -0.02,
            0.97,
            4.0,
            2.0 / (97.0 * 129.0),
            1.5 / (129.0 * 97.0),
            1.0,
        ];
        assert_bit_exact(129, 97, &hm, InterpolationMode::Bilinear);
        assert_bit_exact(129, 97, &hm, InterpolationMode::Nearest);
    }

    /// An affine-equivalent homography (third row [0, 0, 1]) must also agree —
    /// and cross-checks the internal inversion against the CPU's.
    #[test]
    fn warp_perspective_affine_equivalent_matches_cpu() {
        let hm = [0.9, 0.15, 10.0, -0.1, 1.1, -6.0, 0.0, 0.0, 1.0];
        assert_bit_exact(320, 240, &hm, InterpolationMode::Bilinear);
        assert_bit_exact(320, 240, &hm, InterpolationMode::Nearest);
    }

    /// Identity homography reproduces the source exactly.
    #[test]
    fn warp_perspective_identity_is_exact_copy() {
        let hm = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        for mode in [InterpolationMode::Bilinear, InterpolationMode::Nearest] {
            let (cpu, gpu) = cpu_and_gpu(65, 33, &hm, mode);
            assert_eq!(
                gpu, cpu,
                "identity warp ({mode:?}) must reproduce the source"
            );
        }
    }

    /// A singular homography must be rejected before any launch.
    #[test]
    fn warp_perspective_rejects_singular_homography() {
        let stream = default_stream();
        let ctx = &stream.context();
        let d_src = stream.clone_htod(&vec![0.0f32; 8 * 8 * 3]).unwrap();
        let mut d_dst = stream.alloc_zeros::<f32>(8 * 8 * 3).unwrap();
        // Rank-1 matrix.
        let hm = [1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0];
        let err = launch_warp_perspective_bilinear_cuda(
            ctx, &stream, &d_src, &mut d_dst, 8, 8, 8, 8, &hm, None,
        )
        .unwrap_err();
        assert!(matches!(err, CudaWarpPerspectiveError::SingularHomography));
    }
}
