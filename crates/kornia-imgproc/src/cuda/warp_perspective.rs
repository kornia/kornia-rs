//! Native CUDA warp-perspective kernels for `kornia-imgproc`.
//!
//! # Algorithm
//!
//! Both kernels use **inverse mapping**: each output thread computes the
//! floating-point source coordinate by applying the inverted 3×3 homography
//! to its `(dst_x, dst_y)` position and dividing by the homogeneous weight `w`.
//! Pixels whose `|w| < 1e-10` (degenerate projection) are written as 0.
//! Source coordinates outside the image also return 0 (`BORDER_CONSTANT`),
//! matching [`warp_perspective`](crate::warp::warp_perspective).
//!
//! # Optimisations
//!
//! * **CUDA texture objects** — source reads via the 2D-spatial texture cache
//!   with `CU_TR_ADDRESS_MODE_BORDER`, eliminating divergent OOB branches.
//! * **`CU_FUNC_CACHE_PREFER_L1`** — enlarges L1 to 64 KB; no shared memory used.
//! * **32×8 thread block (default)** — full warp per output row for coalesced writes.
//! * **`1/w` reciprocal** — one FP division replaces two per output pixel.
//!
//! # Public API
//!
//! * [`launch_warp_perspective_bilinear_cuda`] — bilinear, 3-ch f32.
//! * [`launch_warp_perspective_nearest_cuda`]  — nearest-neighbor, 3-ch f32.
//! * [`launch_warp_perspective_bicubic_cuda`]  — bicubic (Keys a=-0.5), 3-ch f32.

use std::sync::{Arc, OnceLock};

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, LaunchConfig};
use kornia_tensor::CudaKernel;

use crate::warp::invert_homography;

use super::texture::CudaTexObject;

// ── CUDA C source: bilinear warp perspective via texture object ───────────────
//
// The only difference from warp_affine_bilinear_tex_3c is the coordinate
// transform: affine uses a 2×3 linear map; perspective adds a w-divide from
// the third row of the 3×3 homography.

static BILINEAR_SRC: &str = r#"
extern "C" __global__ void warp_perspective_bilinear_tex_3c(
    unsigned long long tex,
    float* __restrict__  dst,
    unsigned int dst_w,
    unsigned int dst_h,
    float h0, float h1, float h2,
    float h3, float h4, float h5,
    float h6, float h7, float h8
) {
    unsigned int gx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int gy = blockIdx.y * blockDim.y + threadIdx.y;
    if (gx >= dst_w || gy >= dst_h) return;

    unsigned int out = (gy * dst_w + gx) * 3u;
    float w = h6 * (float)gx + h7 * (float)gy + h8;
    if (fabsf(w) < 1e-10f) {
        dst[out] = 0.0f; dst[out+1] = 0.0f; dst[out+2] = 0.0f;
        return;
    }
    float rw = 1.0f / w;
    float sx = (h0 * (float)gx + h1 * (float)gy + h2) * rw;
    float sy = (h3 * (float)gx + h4 * (float)gy + h5) * rw;

    // Bilinear weights.
    float x0f = floorf(sx);
    float y0f = floorf(sy);
    float fx   = sx - x0f;
    float fy   = sy - y0f;
    float w00 = (1.0f - fy) * (1.0f - fx);
    float w10 = (1.0f - fy) * fx;
    float w01 = fy * (1.0f - fx);
    float w11 = fy * fx;

    float tx0 = x0f * 3.0f;
    float ty1 = y0f + 1.0f;

    // Border mode: OOB fetches return 0.0.
    cudaTextureObject_t t = (cudaTextureObject_t)tex;

    float r00 = tex2D<float>(t, tx0,       y0f);
    float r10 = tex2D<float>(t, tx0+3.0f,  y0f);
    float r01 = tex2D<float>(t, tx0,       ty1);
    float r11 = tex2D<float>(t, tx0+3.0f,  ty1);
    dst[out]   = fmaf(w00, r00, fmaf(w10, r10, fmaf(w01, r01, w11 * r11)));

    float g00 = tex2D<float>(t, tx0+1.0f,  y0f);
    float g10 = tex2D<float>(t, tx0+4.0f,  y0f);
    float g01 = tex2D<float>(t, tx0+1.0f,  ty1);
    float g11 = tex2D<float>(t, tx0+4.0f,  ty1);
    dst[out+1] = fmaf(w00, g00, fmaf(w10, g10, fmaf(w01, g01, w11 * g11)));

    float b00 = tex2D<float>(t, tx0+2.0f,  y0f);
    float b10 = tex2D<float>(t, tx0+5.0f,  y0f);
    float b01 = tex2D<float>(t, tx0+2.0f,  ty1);
    float b11 = tex2D<float>(t, tx0+5.0f,  ty1);
    dst[out+2] = fmaf(w00, b00, fmaf(w10, b10, fmaf(w01, b01, w11 * b11)));
}
"#;

// ── CUDA C source: nearest-neighbor warp perspective via texture object ───────

static NEAREST_SRC: &str = r#"
extern "C" __global__ void warp_perspective_nearest_tex_3c(
    unsigned long long tex,
    float* __restrict__  dst,
    unsigned int dst_w,
    unsigned int dst_h,
    float h0, float h1, float h2,
    float h3, float h4, float h5,
    float h6, float h7, float h8
) {
    unsigned int gx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int gy = blockIdx.y * blockDim.y + threadIdx.y;
    if (gx >= dst_w || gy >= dst_h) return;

    unsigned int out = (gy * dst_w + gx) * 3u;
    float w = h6 * (float)gx + h7 * (float)gy + h8;
    if (fabsf(w) < 1e-10f) {
        dst[out] = 0.0f; dst[out+1] = 0.0f; dst[out+2] = 0.0f;
        return;
    }
    float rw = 1.0f / w;
    float sx = (h0 * (float)gx + h1 * (float)gy + h2) * rw;
    float sy = (h3 * (float)gx + h4 * (float)gy + h5) * rw;

    // Round to nearest; border mode handles OOB → 0.
    float tx = roundf(sx) * 3.0f;
    float ty = roundf(sy);

    cudaTextureObject_t t = (cudaTextureObject_t)tex;
    dst[out]   = tex2D<float>(t, tx,       ty);
    dst[out+1] = tex2D<float>(t, tx+1.0f,  ty);
    dst[out+2] = tex2D<float>(t, tx+2.0f,  ty);
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

    unsigned int out = (dst_y * dst_w + dst_x) * 3u;
    float w = h6 * (float)dst_x + h7 * (float)dst_y + h8;
    if (fabsf(w) < 1e-10f) {
        dst[out] = 0.0f; dst[out+1] = 0.0f; dst[out+2] = 0.0f;
        return;
    }
    float rw = 1.0f / w;
    float sx = (h0 * (float)dst_x + h1 * (float)dst_y + h2) * rw;
    float sy = (h3 * (float)dst_x + h4 * (float)dst_y + h5) * rw;

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
        t = 1.0f + frac_x; wx[0] = ((-0.5f*t + 2.5f)*t - 4.0f)*t + 2.0f;
        t =         frac_x; wx[1] = (( 1.5f*t - 2.5f)*t       )*t + 1.0f;
        t = 1.0f - frac_x; wx[2] = (( 1.5f*t - 2.5f)*t       )*t + 1.0f;
        t = 2.0f - frac_x; wx[3] = ((-0.5f*t + 2.5f)*t - 4.0f)*t + 2.0f;
        t = 1.0f + frac_y; wy[0] = ((-0.5f*t + 2.5f)*t - 4.0f)*t + 2.0f;
        t =         frac_y; wy[1] = (( 1.5f*t - 2.5f)*t       )*t + 1.0f;
        t = 1.0f - frac_y; wy[2] = (( 1.5f*t - 2.5f)*t       )*t + 1.0f;
        t = 2.0f - frac_y; wy[3] = ((-0.5f*t + 2.5f)*t - 4.0f)*t + 2.0f;
    }

    unsigned int row[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int yi = max(0, min(y0 + i - 1, (int)src_h - 1));
        row[i] = (unsigned int)yi * src_w * 3u;
    }

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f;
    #pragma unroll
    for (int dy = 0; dy < 4; ++dy) {
        #pragma unroll
        for (int dx = 0; dx < 4; ++dx) {
            int xi = max(0, min(x0 + dx - 1, (int)src_w - 1));
            unsigned int b = row[dy] + (unsigned int)xi * 3u;
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

// ── Kernel caches ─────────────────────────────────────────────────────────────

static BILINEAR_KERNEL: OnceLock<CudaKernel> = OnceLock::new();
static NEAREST_KERNEL: OnceLock<CudaKernel> = OnceLock::new();
static BICUBIC_KERNEL: OnceLock<Result<CudaKernel, String>> = OnceLock::new();

const BLOCK_W: u32 = 32;
const BLOCK_H: u32 = 8;

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

fn make_config(dst_width: u32, dst_height: u32, block_dim: Option<(u32, u32)>) -> LaunchConfig {
    let (bw, bh) = block_dim.unwrap_or_else(|| (BLOCK_W.min(dst_width), BLOCK_H.min(dst_height)));
    LaunchConfig {
        block_dim: (bw, bh, 1),
        grid_dim: (dst_width.div_ceil(bw), dst_height.div_ceil(bh), 1),
        shared_mem_bytes: 0,
    }
}

fn compile_with_l1(ctx: &Arc<CudaContext>, src: &str, fn_name: &str) -> CudaKernel {
    let k = CudaKernel::compile(ctx, src, fn_name)
        .unwrap_or_else(|e| panic!("failed to compile {fn_name}: {e}"));
    let _ = k.prefer_l1_cache();
    k
}

fn try_compile_with_l1(
    ctx: &Arc<CudaContext>,
    src: &str,
    fn_name: &str,
) -> Result<CudaKernel, String> {
    let k = CudaKernel::compile(ctx, src, fn_name)
        .map_err(|e| format!("failed to compile {fn_name}: {e}"))?;
    let _ = k.prefer_l1_cache();
    Ok(k)
}

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

fn make_tex(
    src: &CudaSlice<f32>,
    stream: &Arc<CudaStream>,
    src_width: u32,
    src_height: u32,
) -> Result<(CudaTexObject, u64), CudaWarpPerspectiveError> {
    let (dev_ptr, _guard) = src.device_ptr(stream);
    let tex =
        CudaTexObject::new_pitch2d_border(dev_ptr, src_width as usize * 3, src_height as usize)
            .map_err(CudaWarpPerspectiveError::Cuda)?;
    let handle = tex.handle();
    Ok((tex, handle))
}

// ── Public launchers ──────────────────────────────────────────────────────────

/// Launch the bilinear warp-perspective kernel for a 3-channel f32 image.
///
/// Applies the 3×3 forward homography `h` to warp `src` into `dst`.
/// The matrix is inverted internally so the API matches the CPU
/// [`warp_perspective`](crate::warp::warp_perspective).
///
/// Source pixels are fetched via a CUDA texture object (border-mode).
/// Out-of-bounds coordinates return 0 (`BORDER_CONSTANT`).
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
/// texture-creation error, launch error, or if any slice is too small.
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
    if src_width == 0 || src_height == 0 || dst_width == 0 || dst_height == 0 {
        return Err(CudaWarpPerspectiveError::Cuda(
            "image dimensions must be non-zero".into(),
        ));
    }
    check_image_slice(src, "src", src_width, src_height)?;
    check_image_slice(dst, "dst", dst_width, dst_height)?;

    let hi = invert_homography(h).ok_or(CudaWarpPerspectiveError::SingularHomography)?;
    let kernel = BILINEAR_KERNEL
        .get_or_init(|| compile_with_l1(ctx, BILINEAR_SRC, "warp_perspective_bilinear_tex_3c"));
    let (_tex, tex_handle) = make_tex(src, stream, src_width, src_height)?;

    kernel
        .launch_builder(stream)
        .arg(&tex_handle)
        .arg(dst)
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
/// source sampling.  Faster; suitable for integer-aligned transforms.
///
/// # Arguments
///
/// See [`launch_warp_perspective_bilinear_cuda`] — arguments are identical.
///
/// # Errors
///
/// Returns [`CudaWarpPerspectiveError`] on compile failure, singular homography,
/// texture-creation error, launch error, or if any slice is too small.
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
    if src_width == 0 || src_height == 0 || dst_width == 0 || dst_height == 0 {
        return Err(CudaWarpPerspectiveError::Cuda(
            "image dimensions must be non-zero".into(),
        ));
    }
    check_image_slice(src, "src", src_width, src_height)?;
    check_image_slice(dst, "dst", dst_width, dst_height)?;

    let hi = invert_homography(h).ok_or(CudaWarpPerspectiveError::SingularHomography)?;
    let kernel = NEAREST_KERNEL
        .get_or_init(|| compile_with_l1(ctx, NEAREST_SRC, "warp_perspective_nearest_tex_3c"));
    let (_tex, tex_handle) = make_tex(src, stream, src_width, src_height)?;

    kernel
        .launch_builder(stream)
        .arg(&tex_handle)
        .arg(dst)
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
/// Unlike bilinear/nearest, this kernel reads from a raw device pointer via
/// `__ldg` rather than a texture object (required for the 4×4 tap window).
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
    if src_width == 0 || src_height == 0 || dst_width == 0 || dst_height == 0 {
        return Err(CudaWarpPerspectiveError::Cuda(
            "image dimensions must be non-zero".into(),
        ));
    }
    check_image_slice(src, "src", src_width, src_height)?;
    check_image_slice(dst, "dst", dst_width, dst_height)?;

    let hi = invert_homography(h).ok_or(CudaWarpPerspectiveError::SingularHomography)?;
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
