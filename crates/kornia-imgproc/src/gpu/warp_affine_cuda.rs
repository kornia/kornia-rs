//! Native CUDA warp-affine kernels for `kornia-imgproc`.
//!
//! # Algorithm
//!
//! Both kernels use **inverse mapping**: each output thread computes the
//! floating-point source coordinate by applying the inverted 2×3 affine matrix
//! to its `(dst_x, dst_y)` position.  Source coordinates that fall outside the
//! image are filled with zero (`BORDER_CONSTANT`), matching the CPU
//! [`warp_affine`](crate::warp::warp_affine) default.
//!
//! # Optimisations applied
//!
//! * **32×8 thread block** — full warp per output row, same reasoning as
//!   `resize_cuda`: better write coalescing and source reads confined to nearby
//!   rows per warp.
//! * **`__ldg` source reads** — routes through the read-only L1 cache.
//!   Warp-affine source accesses are scattered (angle determines stride between
//!   consecutive threads), so the 2D spatial locality of `__ldg` is critical.
//! * **`CU_FUNC_CACHE_PREFER_L1`** — enlarges L1 to 64 KB on Turing since
//!   neither kernel uses shared memory.
//!
//! # Public API
//!
//! * [`launch_warp_affine_bilinear_cuda`] — bilinear warp affine, 3-ch f32.
//! * [`launch_warp_affine_nearest_cuda`]  — nearest-neighbor warp affine, 3-ch f32.
//! * [`launch_warp_affine_bicubic_cuda`]  — bicubic warp affine, 3-ch f32.

use std::sync::{Arc, OnceLock};

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, LaunchConfig};
use kornia_tensor::CudaKernel;

use crate::warp::invert_affine_transform;

// ── CUDA C source: bilinear warp affine ──────────────────────────────────────

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
    unsigned int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (dst_x >= dst_w || dst_y >= dst_h) return;

    // Inverse affine: map output pixel → floating-point source coordinate.
    float sx = m0 * (float)dst_x + m1 * (float)dst_y + m2;
    float sy = m3 * (float)dst_x + m4 * (float)dst_y + m5;

    unsigned int out = (dst_y * dst_w + dst_x) * 3u;

    // Out-of-bounds: BORDER_CONSTANT = 0.
    if (sx < 0.0f || sx >= (float)src_w || sy < 0.0f || sy >= (float)src_h) {
        dst[out] = 0.0f; dst[out + 1] = 0.0f; dst[out + 2] = 0.0f;
        return;
    }

    // Clamp so that x0+1 / y0+1 stay within the last valid index.
    sx = fminf(sx, (float)(src_w - 1u));
    sy = fminf(sy, (float)(src_h - 1u));

    unsigned int x0 = (unsigned int)sx;
    unsigned int y0 = (unsigned int)sy;
    unsigned int x1 = min(x0 + 1u, src_w - 1u);
    unsigned int y1 = min(y0 + 1u, src_h - 1u);

    float fx  = sx - (float)x0;
    float fy  = sy - (float)y0;
    float w00 = (1.0f - fy) * (1.0f - fx);
    float w10 = (1.0f - fy) * fx;
    float w01 = fy * (1.0f - fx);
    float w11 = fy * fx;

    unsigned int b00 = (y0 * src_w + x0) * 3u;
    unsigned int b10 = (y0 * src_w + x1) * 3u;
    unsigned int b01 = (y1 * src_w + x0) * 3u;
    unsigned int b11 = (y1 * src_w + x1) * 3u;

    dst[out]     = w00*__ldg(&src[b00])   + w10*__ldg(&src[b10])   + w01*__ldg(&src[b01])   + w11*__ldg(&src[b11]);
    dst[out + 1] = w00*__ldg(&src[b00+1]) + w10*__ldg(&src[b10+1]) + w01*__ldg(&src[b01+1]) + w11*__ldg(&src[b11+1]);
    dst[out + 2] = w00*__ldg(&src[b00+2]) + w10*__ldg(&src[b10+2]) + w01*__ldg(&src[b01+2]) + w11*__ldg(&src[b11+2]);
}
"#;

// ── CUDA C source: nearest-neighbor warp affine ───────────────────────────────

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
    unsigned int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (dst_x >= dst_w || dst_y >= dst_h) return;

    float sx = m0 * (float)dst_x + m1 * (float)dst_y + m2;
    float sy = m3 * (float)dst_x + m4 * (float)dst_y + m5;

    unsigned int out = (dst_y * dst_w + dst_x) * 3u;

    if (sx < 0.0f || sx >= (float)src_w || sy < 0.0f || sy >= (float)src_h) {
        dst[out] = 0.0f; dst[out + 1] = 0.0f; dst[out + 2] = 0.0f;
        return;
    }

    unsigned int xi = min((unsigned int)roundf(sx), src_w - 1u);
    unsigned int yi = min((unsigned int)roundf(sy), src_h - 1u);

    unsigned int base = (yi * src_w + xi) * 3u;
    dst[out]     = __ldg(&src[base]);
    dst[out + 1] = __ldg(&src[base + 1]);
    dst[out + 2] = __ldg(&src[base + 2]);
}
"#;

// ── CUDA C source: bicubic warp affine ───────────────────────────────────────
//
// Keys cubic weight (a = -0.5, matches OpenCV INTER_CUBIC):
//   w(t) = 1.5t³ - 2.5t² + 1       for |t| ≤ 1
//   w(t) = -0.5t³ + 2.5t² - 4t + 2 for 1 < |t| ≤ 2
//
// The 4×4 tap neighborhood spans dx, dy ∈ {-1, 0, 1, 2} relative to
// floor(sx)/floor(sy). Out-of-range taps are clamped to the image border
// (BORDER_REPLICATE). Pixels whose centre falls outside the source image
// are filled with zero (BORDER_CONSTANT), matching the bilinear kernel.

static BICUBIC_SRC: &str = r#"
__device__ __forceinline__ float cubic_w(float t) {
    t = fabsf(t);
    if (t < 1.0f) return 1.5f*t*t*t - 2.5f*t*t + 1.0f;
    if (t < 2.0f) return -0.5f*t*t*t + 2.5f*t*t - 4.0f*t + 2.0f;
    return 0.0f;
}

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

    unsigned int out = (dst_y * dst_w + dst_x) * 3u;

    if (sx < 0.0f || sx >= (float)src_w || sy < 0.0f || sy >= (float)src_h) {
        dst[out] = 0.0f; dst[out+1] = 0.0f; dst[out+2] = 0.0f;
        return;
    }

    int x0 = (int)floorf(sx);
    int y0 = (int)floorf(sy);
    float frac_x = sx - (float)x0;
    float frac_y = sy - (float)y0;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f;
    for (int dy = -1; dy <= 2; ++dy) {
        float wy = cubic_w((float)dy - frac_y);
        for (int dx = -1; dx <= 2; ++dx) {
            float wx = cubic_w((float)dx - frac_x);
            int xi = max(0, min(x0 + dx, (int)src_w - 1));
            int yi = max(0, min(y0 + dy, (int)src_h - 1));
            unsigned int b = ((unsigned int)yi * src_w + (unsigned int)xi) * 3u;
            float w = wx * wy;
            acc0 += w * __ldg(&src[b]);
            acc1 += w * __ldg(&src[b+1]);
            acc2 += w * __ldg(&src[b+2]);
        }
    }
    dst[out]     = acc0;
    dst[out + 1] = acc1;
    dst[out + 2] = acc2;
}
"#;

// ── Kernel caches ─────────────────────────────────────────────────────────────

static BILINEAR_KERNEL: OnceLock<CudaKernel> = OnceLock::new();
static NEAREST_KERNEL: OnceLock<CudaKernel> = OnceLock::new();
static BICUBIC_KERNEL: OnceLock<CudaKernel> = OnceLock::new();

const BLOCK_W: u32 = 32;
const BLOCK_H: u32 = 8;

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

fn make_config(dst_width: u32, dst_height: u32) -> LaunchConfig {
    LaunchConfig {
        block_dim: (BLOCK_W, BLOCK_H, 1),
        grid_dim: (dst_width.div_ceil(BLOCK_W), dst_height.div_ceil(BLOCK_H), 1),
        shared_mem_bytes: 0,
    }
}

fn compile_with_l1(ctx: &Arc<CudaContext>, src: &str, fn_name: &str) -> CudaKernel {
    let k = CudaKernel::compile(ctx, src, fn_name)
        .unwrap_or_else(|e| panic!("failed to compile {fn_name}: {e}"));
    let _ = k.prefer_l1_cache();
    k
}

fn check_dst(
    dst: &CudaSlice<f32>,
    dst_width: u32,
    dst_height: u32,
) -> Result<(), CudaWarpAffineError> {
    let need = (dst_width as usize) * (dst_height as usize) * 3;
    if dst.len() < need {
        return Err(CudaWarpAffineError::SliceTooSmall {
            got: dst.len(),
            need,
        });
    }
    Ok(())
}

// ── Public launchers ──────────────────────────────────────────────────────────

/// Launch the bilinear warp-affine kernel for a 3-channel f32 image.
///
/// Applies the 2×3 forward affine matrix `m` to warp `src` into `dst`.
/// The matrix is inverted internally so the API matches the CPU
/// [`warp_affine`](crate::warp::warp_affine).  Source coordinates that fall
/// outside the image boundary are filled with zero (`BORDER_CONSTANT`).
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
///
/// # Errors
///
/// Returns [`CudaWarpAffineError`] on compile failure, launch error, or if
/// `dst` is too small.
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
) -> Result<(), CudaWarpAffineError> {
    check_dst(dst, dst_width, dst_height)?;

    let kernel = BILINEAR_KERNEL
        .get_or_init(|| compile_with_l1(ctx, BILINEAR_SRC, "warp_affine_bilinear_3c"));

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
        .launch_2d(dst_width, dst_height, make_config(dst_width, dst_height))
        .map_err(|e| CudaWarpAffineError::Cuda(e.to_string()))
}

/// Launch the nearest-neighbor warp-affine kernel for a 3-channel f32 image.
///
/// Same as [`launch_warp_affine_bilinear_cuda`] but uses round-to-nearest
/// source sampling.  Faster; suitable for integer-aligned transforms (pure
/// translation, 90°/180° rotation, flip).
///
/// # Arguments
///
/// See [`launch_warp_affine_bilinear_cuda`] — arguments are identical.
///
/// # Errors
///
/// Returns [`CudaWarpAffineError`] on compile failure, launch error, or if
/// `dst` is too small.
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
) -> Result<(), CudaWarpAffineError> {
    check_dst(dst, dst_width, dst_height)?;

    let kernel =
        NEAREST_KERNEL.get_or_init(|| compile_with_l1(ctx, NEAREST_SRC, "warp_affine_nearest_3c"));

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
        .launch_2d(dst_width, dst_height, make_config(dst_width, dst_height))
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
/// # Arguments
///
/// See [`launch_warp_affine_bilinear_cuda`] — arguments are identical.
///
/// # Errors
///
/// Returns [`CudaWarpAffineError`] on compile failure, launch error, or if
/// `dst` is too small.
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
) -> Result<(), CudaWarpAffineError> {
    check_dst(dst, dst_width, dst_height)?;

    let kernel =
        BICUBIC_KERNEL.get_or_init(|| compile_with_l1(ctx, BICUBIC_SRC, "warp_affine_bicubic_3c"));

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
        .launch_2d(dst_width, dst_height, make_config(dst_width, dst_height))
        .map_err(|e| CudaWarpAffineError::Cuda(e.to_string()))
}
