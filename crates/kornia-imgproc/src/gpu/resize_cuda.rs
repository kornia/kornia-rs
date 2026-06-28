//! Native CUDA downscale kernels for `kornia-imgproc`.
//!
//! # Why CubeCL bilinear downscale is slow
//!
//! CubeCL exposes no texture or shared-memory API, so every source read goes
//! through the plain L2 cache with no read-only hint.  Bilinear downscale
//! makes 4 source reads per output pixel at scattered addresses (stride ≈
//! scale_factor × 12 bytes between adjacent warp threads), leaving DRAM
//! bandwidth at ~55 % of peak.
//!
//! # This implementation
//!
//! Both kernels use a 2D 16×16 thread grid (adjacent threads produce adjacent
//! output pixels — coalesced output writes) and route all source reads through
//! `__ldg()`, which uses the 48 KB read-only L1 (same hardware as the texture
//! cache) without requiring a texture object setup.
//!
//! **Bilinear**: `__ldg` for all 4 bilinear corner reads.  For ≤ 2× downscale
//! all 4 corners span ≤ 2 cache lines, so the read-only L1 serves them with
//! near-zero extra DRAM traffic.  Shared-memory tiling was benchmarked and
//! found to be slower at 2× downscale because the L2 already handles the
//! bilinear source reuse efficiently; the smem overhead (tile load,
//! `__syncthreads`, bank conflicts) outweighed the benefit.
//!
//! **Nearest-neighbor**: one `__ldg` read per output pixel; reaches ~91 % of
//! theoretical DRAM bandwidth on GTX 1650.
//!
//! # Measured throughput (GTX 1650, 2× downscale)
//!
//! | Kernel    | 1080p→540p | GB/s formula   |
//! |-----------|-----------|----------------|
//! | Nearest   | 0.107 ms  | ~116 GB/s      |
//! | Bilinear  | 0.178 ms  | ~70 GB/s       |
//! | CPU (BL)  | 5.18 ms   | ~2.4 GB/s      |
//!
//! The formula counts 1 src read + 1 dst write per output pixel; bilinear's
//! true effective bandwidth is ~3× higher due to L2 cache hits on the 4
//! bilinear source reads.
//!
//! # Public API
//!
//! * [`launch_resize_bilinear_downscale_cuda`] — bilinear downscale, 3-ch f32.
//! * [`launch_resize_nearest_downscale_cuda`]  — nearest-neighbor, 3-ch f32.

use std::sync::{Arc, OnceLock};

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, LaunchConfig};
use kornia_tensor::CudaKernel;

// ── CUDA C source: bilinear, __ldg ───────────────────────────────────────────

static BILINEAR_SRC: &str = r#"
extern "C" __global__ void resize_bilinear_downscale_3c(
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

    unsigned int out = (dst_y * dst_w + dst_x) * 3u;
    dst[out]     = w00*__ldg(&src[b00])   + w10*__ldg(&src[b10])   + w01*__ldg(&src[b01])   + w11*__ldg(&src[b11]);
    dst[out + 1] = w00*__ldg(&src[b00+1]) + w10*__ldg(&src[b10+1]) + w01*__ldg(&src[b01+1]) + w11*__ldg(&src[b11+1]);
    dst[out + 2] = w00*__ldg(&src[b00+2]) + w10*__ldg(&src[b10+2]) + w01*__ldg(&src[b01+2]) + w11*__ldg(&src[b11+2]);
}
"#;

// ── CUDA C source: nearest-neighbor, __ldg ───────────────────────────────────

// Nearest downscale reads exactly one source pixel per output pixel — no data
// is shared between threads, so __ldg alone reaches ~91 % of DRAM peak.

static NEAREST_SRC: &str = r#"
extern "C" __global__ void resize_nearest_downscale_3c(
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

    // Half-pixel center alignment.
    unsigned int src_xi = min((unsigned int)((dst_x + 0.5f) * scale_x), src_w - 1u);
    unsigned int src_yi = min((unsigned int)((dst_y + 0.5f) * scale_y), src_h - 1u);

    unsigned int src_base = (src_yi * src_w + src_xi) * 3u;
    unsigned int dst_base = (dst_y  * dst_w + dst_x)  * 3u;

    dst[dst_base]     = __ldg(&src[src_base]);
    dst[dst_base + 1] = __ldg(&src[src_base + 1]);
    dst[dst_base + 2] = __ldg(&src[src_base + 2]);
}
"#;

// ── Kernel caches ─────────────────────────────────────────────────────────────

static BILINEAR_KERNEL: OnceLock<CudaKernel> = OnceLock::new();
static NEAREST_KERNEL: OnceLock<CudaKernel> = OnceLock::new();

// ── Constants ─────────────────────────────────────────────────────────────────

const BLOCK_W: u32 = 16;
const BLOCK_H: u32 = 16;

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

fn make_config(dst_width: u32, dst_height: u32) -> LaunchConfig {
    LaunchConfig {
        block_dim: (BLOCK_W, BLOCK_H, 1),
        grid_dim: (dst_width.div_ceil(BLOCK_W), dst_height.div_ceil(BLOCK_H), 1),
        shared_mem_bytes: 0,
    }
}

// ── Public launchers ──────────────────────────────────────────────────────────

/// Launch the bilinear downscale kernel for a 3-channel f32 image.
///
/// Uses `__ldg()` for all source reads (read-only L1 / texture cache hardware).
/// For ≤ 2× downscale all 4 bilinear corners land within 1–2 cache lines, so
/// the L2 already provides near-perfect source data reuse.  Measured throughput:
/// ~70 GB/s formula-bandwidth on GTX 1650 for 1080p → 540p.
///
/// # Arguments
///
/// * `ctx`    – CUDA context used for one-time kernel compilation.
/// * `stream` – Stream for kernel execution.
/// * `src`    – Device slice: `src_h × src_w × 3` f32 values.
/// * `dst`    – Device slice: `dst_h × dst_w × 3` f32 values (written).
/// * `src_width`, `src_height` – Source image dimensions.
/// * `dst_width`, `dst_height` – Output image dimensions (must be ≤ source).
///
/// # Errors
///
/// Returns [`CudaResizeError`] on compile failure, launch error, or size mismatch.
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
) -> Result<(), CudaResizeError> {
    let need = (dst_width as usize) * (dst_height as usize) * 3;
    if dst.len() < need {
        return Err(CudaResizeError::SliceTooSmall { got: dst.len(), need });
    }

    let kernel = BILINEAR_KERNEL.get_or_init(|| {
        CudaKernel::compile(ctx, BILINEAR_SRC, "resize_bilinear_downscale_3c")
            .expect("failed to compile resize_bilinear_downscale_3c")
    });

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
        .launch_2d(dst_width, dst_height, make_config(dst_width, dst_height))
        .map_err(|e| CudaResizeError::Cuda(e.to_string()))
}

/// Launch the nearest-neighbor downscale kernel for a 3-channel f32 image.
///
/// Uses `__ldg()` for all source reads (read-only L1 cache), reaching ~91 %
/// of theoretical DRAM bandwidth on GTX 1650.
///
/// # Arguments
///
/// See [`launch_resize_bilinear_downscale_cuda`] — arguments are identical.
///
/// # Errors
///
/// Returns [`CudaResizeError`] on compile failure, launch error, or size mismatch.
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
) -> Result<(), CudaResizeError> {
    let need = (dst_width as usize) * (dst_height as usize) * 3;
    if dst.len() < need {
        return Err(CudaResizeError::SliceTooSmall { got: dst.len(), need });
    }

    let kernel = NEAREST_KERNEL.get_or_init(|| {
        CudaKernel::compile(ctx, NEAREST_SRC, "resize_nearest_downscale_3c")
            .expect("failed to compile resize_nearest_downscale_3c")
    });

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
        .launch_2d(dst_width, dst_height, make_config(dst_width, dst_height))

        .map_err(|e| CudaResizeError::Cuda(e.to_string()))
}
