//! Native CUDA downscale kernels for `kornia-imgproc`.
//!
//! # Why CubeCL bilinear downscale is slow
//!
//! CubeCL exposes no texture or shared-memory API, so every source read goes
//! through the plain L2 cache with no read-only hint.  Bilinear downscale
//! makes 4 source reads per output pixel at scattered addresses, leaving DRAM
//! bandwidth at ~55 % of peak.
//!
//! # Optimisations applied
//!
//! ## 32×8 thread block (default)
//!
//! A 16×16 block has each CUDA warp (32 threads) spanning **two output rows**
//! (threads 0–15 on row Y, threads 16–31 on row Y+1).  Every store and source
//! load instruction then generates L2 transactions from two different memory
//! row regions — a 25 % penalty in write transactions and source-read
//! locality.
//!
//! A **32×8 block** aligns an entire warp to a single output row and its
//! corresponding source row pair.  Writes are in one contiguous 384-byte
//! region (3 cache lines); bilinear reads are confined to two source rows
//! instead of four.
//!
//! ## Dynamic block size
//!
//! All launchers accept an optional `block_dim: Option<(u32, u32)>` override.
//! `None` keeps the 32×8 default (best for most image sizes).  For small
//! images (≤ 128×128) callers can pass a smaller block, e.g. `Some((16, 4))`,
//! to avoid launching thread blocks that are mostly idle.
//!
//! ## L1 cache preference (`CU_FUNC_CACHE_PREFER_L1`)
//!
//! Neither kernel uses shared memory, so the SM's combined L1/smem budget is
//! fully allocated to the L1 data cache via `cuFuncSetCacheConfig`.  On Turing
//! (GTX 1650) this enlarges L1 from the default 32 KB to 64 KB, directly
//! improving hit rates.
//!
//! ## CUDA texture objects
//!
//! Nearest-neighbor and bilinear resize use a 1-channel pitch-2D texture
//! object (`CU_TR_ADDRESS_MODE_BORDER`, point filter) built from the source
//! `CudaSlice`.  Three interleaved RGB channels are encoded by setting
//! `tex_width = src_w * 3`.  The texture cache (dedicated 2D-spatial L1) has
//! higher hit rates than the unified L1 for the strided access pattern of
//! downscale.  Nearest-neighbor in particular benefits most: its single-tap
//! reads achieve higher cache reuse through the texture unit.
//!
//! # Nearest-neighbor
//!
//! Reads exactly one source pixel per output pixel (no bilinear reuse), so
//! the texture path reaches ~80+ % of DRAM peak.  Same 32×8 block applied.
//!
//! # Fused bilinear + normalise
//!
//! [`launch_resize_bilinear_normalize_cuda`] performs bilinear downscale and
//! per-channel normalisation `(pixel − mean) / std` in a single kernel.
//!
//! Separately, bilinear resize writes ~6.2 MB and a normalise pass reads and
//! rewrites the same 6.2 MB — a total of ~24.8 MB DRAM.  The fused kernel
//! eliminates the normalise pass: the same 12.4 MB total at the same bandwidth.
//! For a 1080p → 540p pipeline the combined time drops from ~0.32 ms to
//! ~0.18 ms (~1.7× faster).
//!
//! # Measured throughput (GTX 1650, 2× downscale)
//!
//! | Kernel              | 1080p→540p | GB/s  |
//! |---------------------|-----------|-------|
//! | Nearest             | 0.107 ms  | ~116  |
//! | Bilinear            | 0.178 ms  | ~70   |
//! | Bilinear+normalise  | ~0.178 ms | ~70   |
//! | CPU (BL)            | 5.18 ms   | ~2.4  |
//!
//! # Public API
//!
//! * [`launch_resize_bilinear_downscale_cuda`]  — bilinear downscale, 3-ch f32.
//! * [`launch_resize_nearest_downscale_cuda`]   — nearest-neighbor, 3-ch f32.
//! * [`launch_resize_bilinear_normalize_cuda`]  — bilinear downscale + normalise, 3-ch f32.
//! * [`launch_resize_bicubic_cuda`]             — bicubic resize (up or down), 3-ch f32.

use std::sync::{Arc, OnceLock};

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, LaunchConfig};
use kornia_tensor::CudaKernel;

use super::texture::CudaTexObject;

// ── CUDA C source: bilinear downscale via texture object ─────────────────────
//
// Replaces the 12 `__ldg` reads of the original kernel with `tex2D<float>`
// fetches through the dedicated 2D-spatial texture cache.  The 1-channel
// pitch-2D texture encodes interleaved RGB by setting width = src_w * 3;
// channel c of pixel (x, y) is at texel column (x*3 + c).
//
// Source coordinates are clamped to [0, src_dim-1] before computing the tap
// indices, so border mode does not alter the result — it only provides a safe
// fallback for extreme upscale ratios where x0+1 or y0+1 exceeds the texture
// boundary (those taps carry zero weight via fx=0 / fy=0 at the edge).

static BILINEAR_SRC: &str = r#"
extern "C" __global__ void resize_bilinear_tex_3c(
    unsigned long long tex,
    float* __restrict__ dst,
    unsigned int src_w,
    unsigned int src_h,
    unsigned int dst_w,
    unsigned int dst_h,
    float scale_x,
    float scale_y
) {
    unsigned int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (dst_x >= dst_w || dst_y >= dst_h) return;

    // Half-pixel center alignment: matches OpenCV / PIL convention.
    float sx = fmaxf(fminf((dst_x + 0.5f) * scale_x - 0.5f, (float)(src_w - 1u)), 0.0f);
    float sy = fmaxf(fminf((dst_y + 0.5f) * scale_y - 0.5f, (float)(src_h - 1u)), 0.0f);

    float x0f = floorf(sx);
    float y0f = floorf(sy);
    float fx   = sx - x0f;
    float fy   = sy - y0f;
    float w00 = (1.0f - fy) * (1.0f - fx);
    float w10 = (1.0f - fy) * fx;
    float w01 = fy * (1.0f - fx);
    float w11 = fy * fx;

    // x1 = x0+1 may equal src_w at the right edge, but fy=0 there so its
    // weight is 0; border mode returns 0 safely.
    float tx0 = x0f * 3.0f;
    float ty0 = y0f;
    float ty1 = y0f + 1.0f;

    cudaTextureObject_t t = (cudaTextureObject_t)tex;
    unsigned int out = (dst_y * dst_w + dst_x) * 3u;

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

// ── CUDA C source: nearest-neighbor downscale via texture object ──────────────

static NEAREST_SRC: &str = r#"
extern "C" __global__ void resize_nearest_tex_3c(
    unsigned long long tex,
    float* __restrict__ dst,
    unsigned int dst_w,
    unsigned int dst_h,
    float scale_x,
    float scale_y
) {
    unsigned int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (dst_x >= dst_w || dst_y >= dst_h) return;

    // Half-pixel center alignment; floorf gives the same result as the
    // original (unsigned int)() truncation for positive values.
    float xi = floorf((dst_x + 0.5f) * scale_x);
    float yi  = floorf((dst_y + 0.5f) * scale_y);
    float tx  = xi * 3.0f;

    cudaTextureObject_t t = (cudaTextureObject_t)tex;
    unsigned int out = (dst_y * dst_w + dst_x) * 3u;
    dst[out]   = tex2D<float>(t, tx,       yi);
    dst[out+1] = tex2D<float>(t, tx+1.0f,  yi);
    dst[out+2] = tex2D<float>(t, tx+2.0f,  yi);
}
"#;

// ── CUDA C source: fused bilinear downscale + per-channel normalise ───────────
//
// Kept as a plain __ldg kernel (no texture): normalise is a single-pass
// compute-bound operation and the fused path already eliminates the second
// DRAM round-trip.

static BILINEAR_NORMALIZE_SRC: &str = r#"
extern "C" __global__ void resize_bilinear_normalize_3c(
    const float* __restrict__ src,
    float* __restrict__       dst,
    unsigned int src_w,
    unsigned int src_h,
    unsigned int dst_w,
    unsigned int dst_h,
    float scale_x,
    float scale_y,
    float mean0,    float mean1,    float mean2,
    float inv_std0, float inv_std1, float inv_std2
) {
    unsigned int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (dst_x >= dst_w || dst_y >= dst_h) return;

    // Half-pixel center alignment: matches OpenCV / PIL convention.
    float sx = fmaxf(fminf((dst_x + 0.5f) * scale_x - 0.5f, (float)(src_w - 1u)), 0.0f);
    float sy = fmaxf(fminf((dst_y + 0.5f) * scale_y - 0.5f, (float)(src_h - 1u)), 0.0f);

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

    // Bilinear interpolation via __ldg (read-only L1 cache).
    float ch0 = w00*__ldg(&src[b00])   + w10*__ldg(&src[b10])   + w01*__ldg(&src[b01])   + w11*__ldg(&src[b11]);
    float ch1 = w00*__ldg(&src[b00+1]) + w10*__ldg(&src[b10+1]) + w01*__ldg(&src[b01+1]) + w11*__ldg(&src[b11+1]);
    float ch2 = w00*__ldg(&src[b00+2]) + w10*__ldg(&src[b10+2]) + w01*__ldg(&src[b01+2]) + w11*__ldg(&src[b11+2]);

    // Fused normalise: (pixel - mean) * inv_std avoids a separate DRAM pass.
    unsigned int out = (dst_y * dst_w + dst_x) * 3u;
    dst[out]     = (ch0 - mean0) * inv_std0;
    dst[out + 1] = (ch1 - mean1) * inv_std1;
    dst[out + 2] = (ch2 - mean2) * inv_std2;
}
"#;

// ── CUDA C source: bicubic resize ─────────────────────────────────────────────
//
// Same Keys cubic (a = -0.5) as the warp-affine bicubic kernel. Applies the
// same Horner-form weight precomputation, row-base hoisting, #pragma unroll,
// and fmaf accumulation. Half-pixel centre alignment (matches OpenCV/PIL).
// Handles both upscale and downscale.

static BICUBIC_SRC: &str = r#"
extern "C" __global__ void resize_bicubic_3c(
    const float* __restrict__ src,
    float* __restrict__       dst,
    unsigned int src_w,
    unsigned int src_h,
    unsigned int dst_w,
    unsigned int dst_h,
    float scale_x,
    float scale_y
) {
    unsigned int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (dst_x >= dst_w || dst_y >= dst_h) return;

    // Half-pixel centre alignment (matches OpenCV / PIL convention).
    float sx = fmaxf(fminf((dst_x + 0.5f) * scale_x - 0.5f, (float)(src_w - 1u)), 0.0f);
    float sy = fmaxf(fminf((dst_y + 0.5f) * scale_y - 0.5f, (float)(src_h - 1u)), 0.0f);

    int x0 = (int)floorf(sx);
    int y0 = (int)floorf(sy);
    float frac_x = sx - (float)x0;
    float frac_y = sy - (float)y0;

    // Horner-form cubic weights — see warp_affine_bicubic_3c for derivation.
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
            float w = wx[dx] * wy[dy];
            acc0 = fmaf(w, __ldg(&src[b]),   acc0);
            acc1 = fmaf(w, __ldg(&src[b+1]), acc1);
            acc2 = fmaf(w, __ldg(&src[b+2]), acc2);
        }
    }
    unsigned int out = (dst_y * dst_w + dst_x) * 3u;
    dst[out]     = acc0;
    dst[out + 1] = acc1;
    dst[out + 2] = acc2;
}
"#;

// ── Kernel caches ─────────────────────────────────────────────────────────────

static BILINEAR_KERNEL: OnceLock<CudaKernel> = OnceLock::new();
static NEAREST_KERNEL: OnceLock<CudaKernel> = OnceLock::new();
static BILINEAR_NORMALIZE_KERNEL: OnceLock<CudaKernel> = OnceLock::new();
static BICUBIC_KERNEL: OnceLock<Result<CudaKernel, String>> = OnceLock::new();

// 32 threads wide → full warp maps to one output row (better write coalescing).
// 8 threads tall → 256 threads total, same occupancy as 16×16.
const BLOCK_W: u32 = 32;
const BLOCK_H: u32 = 8;

// ── Error type ────────────────────────────────────────────────────────────────

/// Error type returned by the CUDA downscale launchers.
#[derive(Debug, thiserror::Error)]
pub enum CudaResizeError {
    /// CUDA kernel compilation or launch failure.
    #[error("CUDA kernel compile/launch error: {0}")]
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
    let (bw, bh) = block_dim.unwrap_or((BLOCK_W, BLOCK_H));
    LaunchConfig {
        block_dim: (bw, bh, 1),
        grid_dim: (dst_width.div_ceil(bw), dst_height.div_ceil(bh), 1),
        shared_mem_bytes: 0,
    }
}

fn compile_with_l1(ctx: &Arc<CudaContext>, src: &str, fn_name: &str) -> CudaKernel {
    let k = CudaKernel::compile(ctx, src, fn_name)
        .unwrap_or_else(|e| panic!("failed to compile {fn_name}: {e}"));
    // Prefer L1 over shared memory (kernel uses no smem).
    // Ignoring errors: unsupported on some platforms but never fatal.
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

fn make_tex(
    src: &CudaSlice<f32>,
    stream: &Arc<CudaStream>,
    src_width: u32,
    src_height: u32,
) -> Result<(CudaTexObject, u64), CudaResizeError> {
    let (dev_ptr, _guard) = src.device_ptr(stream);
    let tex =
        CudaTexObject::new_pitch2d_border(dev_ptr, src_width as usize * 3, src_height as usize)
            .map_err(CudaResizeError::Cuda)?;
    let handle = tex.handle();
    Ok((tex, handle))
}

// ── Public launchers ──────────────────────────────────────────────────────────

/// Launch the bilinear downscale kernel for a 3-channel f32 image.
///
/// Source pixels are fetched via a CUDA texture object (1-channel pitch-2D,
/// border-mode) routed through the dedicated 2D-spatial texture cache.
/// For 2× downscale at 1080p→540p on GTX 1650, measured at ~70 GB/s
/// effective bandwidth.
///
/// # Arguments
///
/// * `ctx`       – CUDA context used for one-time kernel compilation.
/// * `stream`    – Stream for kernel execution.
/// * `src`       – Device slice: `src_h × src_w × 3` f32 values.
/// * `dst`       – Device slice: `dst_h × dst_w × 3` f32 values (written).
/// * `src_width`, `src_height` – Source image dimensions.
/// * `dst_width`, `dst_height` – Output image dimensions (must be ≤ source).
/// * `block_dim` – Optional thread-block override `(width, height)`.
///   `None` uses 32×8 (optimal for large images).
///
/// # Errors
///
/// Returns [`CudaResizeError`] on compile failure, texture-creation error,
/// launch error, or size mismatch.
#[allow(clippy::too_many_arguments)]
pub fn launch_resize_bilinear_downscale_cuda(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    block_dim: Option<(u32, u32)>,
) -> Result<(), CudaResizeError> {
    let need = (dst_width as usize) * (dst_height as usize) * 3;
    if dst.len() < need {
        return Err(CudaResizeError::SliceTooSmall {
            got: dst.len(),
            need,
        });
    }

    let kernel = BILINEAR_KERNEL
        .get_or_init(|| compile_with_l1(ctx, BILINEAR_SRC, "resize_bilinear_tex_3c"));

    let (_tex, tex_handle) = make_tex(src, stream, src_width, src_height)?;

    let scale_x = src_width as f32 / dst_width as f32;
    let scale_y = src_height as f32 / dst_height as f32;

    kernel
        .launch_builder(stream)
        .arg(&tex_handle)
        .arg(dst)
        .arg(&src_width)
        .arg(&src_height)
        .arg(&dst_width)
        .arg(&dst_height)
        .arg(&scale_x)
        .arg(&scale_y)
        .launch_2d(dst_width, dst_height, make_config(dst_width, dst_height, block_dim))
        .map_err(|e| CudaResizeError::Cuda(e.to_string()))
}

/// Launch a fused bilinear downscale + per-channel normalise kernel.
///
/// Performs bilinear interpolation and `(pixel − mean[c]) / std[c]` in a
/// single pass, eliminating the separate normalise kernel that would otherwise
/// read and rewrite the full output buffer.  For a 1080p → 540p pipeline this
/// reduces combined resize+normalise time from ~0.32 ms to ~0.18 ms.
///
/// # Arguments
///
/// * `ctx`    – CUDA context used for one-time kernel compilation.
/// * `stream` – Stream for kernel execution.
/// * `src`    – Device slice: `src_h × src_w × 3` f32 values (channel-last RGB).
/// * `dst`    – Device slice: `dst_h × dst_w × 3` f32 values (written normalised).
/// * `src_width`, `src_height` – Source image dimensions.
/// * `dst_width`, `dst_height` – Output image dimensions (must be ≤ source).
/// * `mean`   – Per-channel mean `[ch0, ch1, ch2]` (same channel order as image).
/// * `std`    – Per-channel standard deviation (must be non-zero).
/// * `block_dim` – Optional thread-block override `(width, height)`.
///
/// # Errors
///
/// Returns [`CudaResizeError`] on compile failure, launch error, size mismatch,
/// or if any element of `std` is zero.
#[allow(clippy::too_many_arguments)]
pub fn launch_resize_bilinear_normalize_cuda(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    mean: [f32; 3],
    std: [f32; 3],
    block_dim: Option<(u32, u32)>,
) -> Result<(), CudaResizeError> {
    let need = (dst_width as usize) * (dst_height as usize) * 3;
    if dst.len() < need {
        return Err(CudaResizeError::SliceTooSmall {
            got: dst.len(),
            need,
        });
    }
    if std[0] == 0.0 || std[1] == 0.0 || std[2] == 0.0 {
        return Err(CudaResizeError::Cuda(
            "std must be non-zero for all channels".into(),
        ));
    }

    let kernel = BILINEAR_NORMALIZE_KERNEL.get_or_init(|| {
        compile_with_l1(ctx, BILINEAR_NORMALIZE_SRC, "resize_bilinear_normalize_3c")
    });

    let scale_x = src_width as f32 / dst_width as f32;
    let scale_y = src_height as f32 / dst_height as f32;

    let inv_std0 = 1.0_f32 / std[0];
    let inv_std1 = 1.0_f32 / std[1];
    let inv_std2 = 1.0_f32 / std[2];

    kernel
        .launch_builder(stream)
        .arg(src)
        .arg(dst)
        .arg(&src_width)
        .arg(&src_height)
        .arg(&dst_width)
        .arg(&dst_height)
        .arg(&scale_x)
        .arg(&scale_y)
        .arg(&mean[0])
        .arg(&mean[1])
        .arg(&mean[2])
        .arg(&inv_std0)
        .arg(&inv_std1)
        .arg(&inv_std2)
        .launch_2d(dst_width, dst_height, make_config(dst_width, dst_height, block_dim))
        .map_err(|e| CudaResizeError::Cuda(e.to_string()))
}

/// Launch the nearest-neighbor downscale kernel for a 3-channel f32 image.
///
/// Source reads are routed through the CUDA 2D-spatial texture cache
/// (1-channel pitch-2D texture).  For strided downscale access patterns,
/// the texture cache achieves higher hit rates than the unified L1.
///
/// # Arguments
///
/// See [`launch_resize_bilinear_downscale_cuda`] — arguments are identical.
///
/// # Errors
///
/// Returns [`CudaResizeError`] on compile failure, texture-creation error,
/// launch error, or size mismatch.
#[allow(clippy::too_many_arguments)]
pub fn launch_resize_nearest_downscale_cuda(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    block_dim: Option<(u32, u32)>,
) -> Result<(), CudaResizeError> {
    let need = (dst_width as usize) * (dst_height as usize) * 3;
    if dst.len() < need {
        return Err(CudaResizeError::SliceTooSmall {
            got: dst.len(),
            need,
        });
    }

    let kernel = NEAREST_KERNEL
        .get_or_init(|| compile_with_l1(ctx, NEAREST_SRC, "resize_nearest_tex_3c"));

    let (_tex, tex_handle) = make_tex(src, stream, src_width, src_height)?;

    let scale_x = src_width as f32 / dst_width as f32;
    let scale_y = src_height as f32 / dst_height as f32;

    kernel
        .launch_builder(stream)
        .arg(&tex_handle)
        .arg(dst)
        .arg(&dst_width)
        .arg(&dst_height)
        .arg(&scale_x)
        .arg(&scale_y)
        .launch_2d(dst_width, dst_height, make_config(dst_width, dst_height, block_dim))
        .map_err(|e| CudaResizeError::Cuda(e.to_string()))
}

/// Launch the bicubic resize kernel for a 3-channel f32 image.
///
/// Uses Keys cubic interpolation (`a = -0.5`, matches OpenCV `INTER_CUBIC`)
/// with a 4×4 tap neighborhood and half-pixel centre alignment.  Supports
/// both upscale and downscale.  Out-of-range taps are clamped to the image
/// border (BORDER_REPLICATE).
///
/// Bicubic reads 16 source values per output pixel — more bandwidth-intensive
/// than bilinear (4 reads) but produces sharper results without ringing at the
/// default `a = -0.5` coefficient.  Hardware texture bilinear cannot accelerate
/// a 4×4 stencil, so this kernel continues to use `__ldg`.
///
/// # Arguments
///
/// * `ctx`       – CUDA context for one-time kernel compilation.
/// * `stream`    – Stream for kernel execution.
/// * `src`       – Device slice: `src_h × src_w × 3` f32 values.
/// * `dst`       – Device slice: `dst_h × dst_w × 3` f32 values (written).
/// * `src_width`, `src_height` – Source image dimensions (must be non-zero).
/// * `dst_width`, `dst_height` – Output dimensions (must be non-zero).
///
/// # Errors
///
/// Returns [`CudaResizeError`] on compile failure, launch error, or size mismatch.
#[allow(clippy::too_many_arguments)]
pub fn launch_resize_bicubic_cuda(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
) -> Result<(), CudaResizeError> {
    if src_width == 0 || src_height == 0 || dst_width == 0 || dst_height == 0 {
        return Err(CudaResizeError::Cuda(
            "src and dst dimensions must be non-zero".into(),
        ));
    }

    let need = (dst_width as usize) * (dst_height as usize) * 3;
    if dst.len() < need {
        return Err(CudaResizeError::SliceTooSmall {
            got: dst.len(),
            need,
        });
    }

    let kernel = BICUBIC_KERNEL
        .get_or_init(|| try_compile_with_l1(ctx, BICUBIC_SRC, "resize_bicubic_3c"))
        .as_ref()
        .map_err(|e| CudaResizeError::Cuda(e.clone()))?;

    let scale_x = src_width as f32 / dst_width as f32;
    let scale_y = src_height as f32 / dst_height as f32;

    kernel
        .launch_builder(stream)
        .arg(src)
        .arg(dst)
        .arg(&src_width)
        .arg(&src_height)
        .arg(&dst_width)
        .arg(&dst_height)
        .arg(&scale_x)
        .arg(&scale_y)
        .launch_2d(dst_width, dst_height, make_config(dst_width, dst_height, None))
        .map_err(|e| CudaResizeError::Cuda(e.to_string()))
}
