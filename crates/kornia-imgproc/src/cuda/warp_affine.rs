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
//! * **32×8 thread block (default)** — full warp per output row, same
//!   reasoning as `resize`: better write coalescing and source reads
//!   confined to nearby rows per warp.
//! * **CUDA texture objects** — source reads route through the dedicated
//!   2D-spatial texture cache.  `CU_TR_ADDRESS_MODE_BORDER` returns 0 for
//!   any out-of-bounds fetch, eliminating the divergent `if OOB return 0`
//!   branch that affects every corner pixel under rotation.
//! * **`CU_FUNC_CACHE_PREFER_L1`** — enlarges L1 to 64 KB on Turing since
//!   neither kernel uses shared memory.
//! * **Dynamic block size** — [`launch_warp_affine_bilinear_cuda`] and
//!   [`launch_warp_affine_nearest_cuda`] accept an optional `block_dim`
//!   parameter; `None` selects 32×8 (optimal for most sizes) while `Some`
//!   lets callers override for small images.
//!
//! # Public API
//!
//! * [`launch_warp_affine_bilinear_cuda`] — bilinear warp affine, 3-ch f32.
//! * [`launch_warp_affine_nearest_cuda`]  — nearest-neighbor warp affine, 3-ch f32.
//! * [`launch_warp_affine_bicubic_cuda`]  — bicubic warp affine, 3-ch f32.

use std::sync::{Arc, OnceLock};

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, LaunchConfig};
use kornia_tensor::CudaKernel;

use crate::warp::invert_affine_transform;

use super::texture::CudaTexObject;

// ── CUDA C source: bilinear warp affine via texture object ───────────────────
//
// Passes the source as a 1-channel pitch-2D texture (width = src_w * 3) so
// that interleaved RGB channels are accessible as individual texels.  Border
// mode (CU_TR_ADDRESS_MODE_BORDER) returns 0 for any OOB fetch, implementing
// BORDER_CONSTANT=0 without an explicit bounds-check branch.

static BILINEAR_SRC: &str = r#"
extern "C" __global__ void warp_affine_bilinear_tex_3c(
    unsigned long long tex,
    float* __restrict__  dst,
    unsigned int dst_w,
    unsigned int dst_h,
    float m0, float m1, float m2,
    float m3, float m4, float m5
) {
    unsigned int gx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int gy = blockIdx.y * blockDim.y + threadIdx.y;
    if (gx >= dst_w || gy >= dst_h) return;

    // Inverse affine: map output pixel → floating-point source coordinate.
    float sx = m0 * (float)gx + m1 * (float)gy + m2;
    float sy = m3 * (float)gx + m4 * (float)gy + m5;

    // Bilinear weights.
    float x0f = floorf(sx);
    float y0f = floorf(sy);
    float fx   = sx - x0f;
    float fy   = sy - y0f;
    float w00 = (1.0f - fy) * (1.0f - fx);
    float w10 = (1.0f - fy) * fx;
    float w01 = fy * (1.0f - fx);
    float w11 = fy * fx;

    // Texture x-base for the left column of the 2×2 tap (pixel x0, RGB offset 0).
    // Right column (pixel x0+1) is at tx0 + 3.0f.
    float tx0 = x0f * 3.0f;
    float ty0 = y0f;
    float ty1 = y0f + 1.0f;

    // Border mode: OOB fetches return 0.0 — no explicit bounds check needed.
    cudaTextureObject_t t = (cudaTextureObject_t)tex;
    unsigned int out = (gy * dst_w + gx) * 3u;

    float r00 = tex2D<float>(t, tx0,       ty0);
    float r10 = tex2D<float>(t, tx0+3.0f,  ty0);
    float r01 = tex2D<float>(t, tx0,       ty1);
    float r11 = tex2D<float>(t, tx0+3.0f,  ty1);
    dst[out]   = fmaf(w00, r00, fmaf(w10, r10, fmaf(w01, r01, w11 * r11)));

    float g00 = tex2D<float>(t, tx0+1.0f,  ty0);
    float g10 = tex2D<float>(t, tx0+4.0f,  ty0);
    float g01 = tex2D<float>(t, tx0+1.0f,  ty1);
    float g11 = tex2D<float>(t, tx0+4.0f,  ty1);
    dst[out+1] = fmaf(w00, g00, fmaf(w10, g10, fmaf(w01, g01, w11 * g11)));

    float b00 = tex2D<float>(t, tx0+2.0f,  ty0);
    float b10 = tex2D<float>(t, tx0+5.0f,  ty0);
    float b01 = tex2D<float>(t, tx0+2.0f,  ty1);
    float b11 = tex2D<float>(t, tx0+5.0f,  ty1);
    dst[out+2] = fmaf(w00, b00, fmaf(w10, b10, fmaf(w01, b01, w11 * b11)));
}
"#;

// ── CUDA C source: nearest-neighbor warp affine via texture object ────────────

static NEAREST_SRC: &str = r#"
extern "C" __global__ void warp_affine_nearest_tex_3c(
    unsigned long long tex,
    float* __restrict__  dst,
    unsigned int dst_w,
    unsigned int dst_h,
    float m0, float m1, float m2,
    float m3, float m4, float m5
) {
    unsigned int gx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int gy = blockIdx.y * blockDim.y + threadIdx.y;
    if (gx >= dst_w || gy >= dst_h) return;

    float sx = m0 * (float)gx + m1 * (float)gy + m2;
    float sy = m3 * (float)gx + m4 * (float)gy + m5;

    // Round to nearest; border mode handles OOB → 0.
    float xi = roundf(sx);
    float yi  = roundf(sy);
    float tx  = xi * 3.0f;

    cudaTextureObject_t t = (cudaTextureObject_t)tex;
    unsigned int out = (gy * dst_w + gx) * 3u;
    dst[out]   = tex2D<float>(t, tx,       yi);
    dst[out+1] = tex2D<float>(t, tx+1.0f,  yi);
    dst[out+2] = tex2D<float>(t, tx+2.0f,  yi);
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

    unsigned int out = (dst_y * dst_w + dst_x) * 3u;

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

// ── Kernel caches ─────────────────────────────────────────────────────────────

static BILINEAR_KERNEL: OnceLock<CudaKernel> = OnceLock::new();
static NEAREST_KERNEL: OnceLock<CudaKernel> = OnceLock::new();
static BICUBIC_KERNEL: OnceLock<Result<CudaKernel, String>> = OnceLock::new();

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

fn make_tex(
    src: &CudaSlice<f32>,
    stream: &Arc<CudaStream>,
    src_width: u32,
    src_height: u32,
) -> Result<(CudaTexObject, u64), CudaWarpAffineError> {
    // Safety: DevicePtr::device_ptr returns the raw CUdeviceptr; the _guard
    // records a stream event on drop (after the texture is destroyed) which is
    // the correct ordering.
    let (dev_ptr, _guard) = src.device_ptr(stream);
    let tex =
        CudaTexObject::new_pitch2d_border(dev_ptr, src_width as usize * 3, src_height as usize)
            .map_err(CudaWarpAffineError::Cuda)?;
    let handle = tex.handle();
    Ok((tex, handle))
}

// ── Public launchers ──────────────────────────────────────────────────────────

/// Launch the bilinear warp-affine kernel for a 3-channel f32 image.
///
/// Applies the 2×3 forward affine matrix `m` to warp `src` into `dst`.
/// The matrix is inverted internally so the API matches the CPU
/// [`warp_affine`](crate::warp::warp_affine).
///
/// Source pixels are fetched via a CUDA texture object (1-channel pitch-2D,
/// border-mode).  Out-of-bounds source coordinates return 0 (`BORDER_CONSTANT`)
/// without a divergent branch in the inner loop.
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
/// Returns [`CudaWarpAffineError`] on compile failure, texture-creation error,
/// launch error, or if `dst` is too small.
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
    check_dst(dst, dst_width, dst_height)?;

    let kernel = BILINEAR_KERNEL
        .get_or_init(|| compile_with_l1(ctx, BILINEAR_SRC, "warp_affine_bilinear_tex_3c"));

    let (_tex, tex_handle) = make_tex(src, stream, src_width, src_height)?;
    let mi = invert_affine_transform(m);

    kernel
        .launch_builder(stream)
        .arg(&tex_handle)
        .arg(dst)
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
/// Source reads are texture-cached; out-of-bounds coords return 0 without
/// a divergent branch.
///
/// # Arguments
///
/// See [`launch_warp_affine_bilinear_cuda`] — arguments are identical.
///
/// # Errors
///
/// Returns [`CudaWarpAffineError`] on compile failure, texture-creation error,
/// launch error, or if `dst` is too small.
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
    check_dst(dst, dst_width, dst_height)?;

    let kernel = NEAREST_KERNEL
        .get_or_init(|| compile_with_l1(ctx, NEAREST_SRC, "warp_affine_nearest_tex_3c"));

    let (_tex, tex_handle) = make_tex(src, stream, src_width, src_height)?;
    let mi = invert_affine_transform(m);

    kernel
        .launch_builder(stream)
        .arg(&tex_handle)
        .arg(dst)
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
    block_dim: Option<(u32, u32)>,
) -> Result<(), CudaWarpAffineError> {
    if src_width == 0 || src_height == 0 || dst_width == 0 || dst_height == 0 {
        return Err(CudaWarpAffineError::Cuda(
            "src and dst dimensions must be non-zero".into(),
        ));
    }
    check_dst(dst, dst_width, dst_height)?;

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
