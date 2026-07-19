//! CUDA histogram + histogram-equalization kernels.
//!
//! * `histogram_u8`: grid-stride over the pixels, per-block shared `u32`
//!   bins accumulated with `atomicAdd`, flushed to the global histogram.
//!   Binning mirrors the CPU expression (`floorf(px / scale)` with the same
//!   f32 `scale = 256 / num_bins`), and integer atomic addition is
//!   commutative — counts are EXACTLY equal to the CPU's.
//! * `equalize_lut_u8`: one block builds the 256-entry LUT. Thread 0 runs
//!   the sequential cdf scan (256 adds — deterministic, sync-free), then
//!   all threads evaluate OpenCV's exact formula:
//!   `lut[i] = rint((cdf[i] - cdf_min) * (255f / (N - cdf_min)))` — f32
//!   scale, round-half-to-even, identity LUT for a constant image — the
//!   same expression the CPU twin computes, so CPU/GPU/cv2 agree
//!   byte-for-byte.
//! * `apply_lut_u8`: `dst[i] = lut[src[i]]`.

use std::sync::{Arc, OnceLock};

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use kornia_tensor::CudaKernel;

use super::try_compile_with_l1;

super::define_cuda_error!(
    /// Error type for the CUDA histogram launchers.
    CudaHistogramError,
    "CUDA histogram error: {0}"
);

static HISTOGRAM_SRC: &str = r#"
extern "C" __global__ void histogram_u8(
    const unsigned char* __restrict__ src,
    unsigned int* __restrict__        hist,
    unsigned int n,
    unsigned int num_bins,
    float scale
) {
    extern __shared__ unsigned int bins[];
    for (unsigned int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        bins[i] = 0u;
    }
    __syncthreads();

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = gridDim.x * blockDim.x;
    for (unsigned int i = idx; i < n; i += stride) {
        // Same binning expression as the CPU: (px as f32 / scale).floor().
        unsigned int bin = (unsigned int)floorf((float)__ldg(&src[i]) / scale);
        atomicAdd(&bins[bin], 1u);
    }
    __syncthreads();

    for (unsigned int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        if (bins[i] > 0u) {
            atomicAdd(&hist[i], bins[i]);
        }
    }
}
"#;

static EQUALIZE_LUT_SRC: &str = r#"
extern "C" __global__ void equalize_lut_u8(
    const unsigned int* __restrict__ hist,
    unsigned char* __restrict__      lut,
    unsigned int total
) {
    __shared__ unsigned int cdf[256];
    __shared__ unsigned int cdf_min_s;

    if (threadIdx.x == 0) {
        unsigned int acc = 0u;
        unsigned int cdf_min = 0u;
        bool found = false;
        for (int i = 0; i < 256; ++i) {
            acc += __ldg(&hist[i]);
            cdf[i] = acc;
            if (!found && acc > 0u) {
                cdf_min = acc;
                found = true;
            }
        }
        cdf_min_s = cdf_min;
    }
    __syncthreads();

    unsigned int i = threadIdx.x;
    if (i >= 256u) return;
    unsigned int cdf_min = cdf_min_s;
    if (total == cdf_min) {
        // Constant image: OpenCV leaves it unchanged (identity LUT).
        lut[i] = (unsigned char)i;
        return;
    }
    // OpenCV's exact formula: f32 scale, round-half-to-even (rint). Bins
    // below the first occupied one have cdf < cdf_min — signed difference,
    // negative product, saturates to 0 exactly like cv2's saturate_cast.
    float scale = 255.0f / (float)(total - cdf_min);
    float v = (float)((int)cdf[i] - (int)cdf_min) * scale;
    lut[i] = (unsigned char)min(max((int)rintf(v), 0), 255);
}
"#;

static APPLY_LUT_SRC: &str = r#"
extern "C" __global__ void apply_lut_u8(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__       dst,
    const unsigned char* __restrict__ lut,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __ldg(&lut[__ldg(&src[i])]);
}
"#;

static HISTOGRAM_KERNEL: OnceLock<Result<CudaKernel, String>> = OnceLock::new();
static EQUALIZE_LUT_KERNEL: OnceLock<Result<CudaKernel, String>> = OnceLock::new();
static APPLY_LUT_KERNEL: OnceLock<Result<CudaKernel, String>> = OnceLock::new();

fn get_kernel(
    cell: &'static OnceLock<Result<CudaKernel, String>>,
    ctx: &Arc<CudaContext>,
    src: &str,
    name: &str,
) -> Result<&'static CudaKernel, CudaHistogramError> {
    super::get_kernel_cached(cell, ctx, src, name).map_err(CudaHistogramError::Cuda)
}

/// Accumulate the u8 histogram of `src` into `hist` (`num_bins` u32 bins,
/// zeroed by the caller). Counts are exactly equal to the CPU's
/// `compute_histogram` (same binning expression, commutative adds).
pub fn launch_histogram_u8(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<u8>,
    hist: &mut CudaSlice<u32>,
    n: usize,
    num_bins: u32,
    scale: f32,
) -> Result<(), CudaHistogramError> {
    if num_bins == 0 || num_bins > 256 {
        return Err(CudaHistogramError::Cuda(
            "num_bins must be in [1, 256]".into(),
        ));
    }
    CudaHistogramError::check_slice("src", src.len(), n)?;
    CudaHistogramError::check_slice("hist", hist.len(), num_bins as usize)?;
    let n_u32 = u32::try_from(n).map_err(|_| CudaHistogramError::Cuda("n exceeds u32".into()))?;
    let kernel = get_kernel(&HISTOGRAM_KERNEL, ctx, HISTOGRAM_SRC, "histogram_u8")?;
    // Grid-stride: enough blocks to fill the device, capped for tiny inputs.
    let blocks = n_u32.div_ceil(256).clamp(1, 1024);
    let cfg = cudarc::driver::LaunchConfig {
        block_dim: (256, 1, 1),
        grid_dim: (blocks, 1, 1),
        shared_mem_bytes: num_bins * 4,
    };
    kernel
        .launch_builder(stream)
        .arg(src)
        .arg(hist)
        .arg(&n_u32)
        .arg(&num_bins)
        .arg(&scale)
        .launch_cfg(cfg)
        .map_err(|e| CudaHistogramError::Cuda(e.to_string()))
}

/// Build the OpenCV-exact equalization LUT from a 256-bin u32 histogram.
pub fn launch_equalize_lut_u8(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    hist: &CudaSlice<u32>,
    lut: &mut CudaSlice<u8>,
    total: usize,
) -> Result<(), CudaHistogramError> {
    CudaHistogramError::check_slice("hist", hist.len(), 256)?;
    CudaHistogramError::check_slice("lut", lut.len(), 256)?;
    let total_u32 =
        u32::try_from(total).map_err(|_| CudaHistogramError::Cuda("total exceeds u32".into()))?;
    let kernel = get_kernel(
        &EQUALIZE_LUT_KERNEL,
        ctx,
        EQUALIZE_LUT_SRC,
        "equalize_lut_u8",
    )?;
    let cfg = cudarc::driver::LaunchConfig {
        block_dim: (256, 1, 1),
        grid_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    kernel
        .launch_builder(stream)
        .arg(hist)
        .arg(lut)
        .arg(&total_u32)
        .launch_cfg(cfg)
        .map_err(|e| CudaHistogramError::Cuda(e.to_string()))
}

/// `dst[i] = lut[src[i]]`.
pub fn launch_apply_lut_u8(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<u8>,
    dst: &mut CudaSlice<u8>,
    lut: &CudaSlice<u8>,
    n: usize,
) -> Result<(), CudaHistogramError> {
    CudaHistogramError::check_slice("src", src.len(), n)?;
    CudaHistogramError::check_slice("dst", dst.len(), n)?;
    CudaHistogramError::check_slice("lut", lut.len(), 256)?;
    let n_u32 = u32::try_from(n).map_err(|_| CudaHistogramError::Cuda("n exceeds u32".into()))?;
    let kernel = get_kernel(&APPLY_LUT_KERNEL, ctx, APPLY_LUT_SRC, "apply_lut_u8")?;
    let cfg = cudarc::driver::LaunchConfig {
        block_dim: (256, 1, 1),
        grid_dim: (n_u32.div_ceil(256), 1, 1),
        shared_mem_bytes: 0,
    };
    kernel
        .launch_builder(stream)
        .arg(src)
        .arg(dst)
        .arg(lut)
        .arg(&n_u32)
        .launch_cfg(cfg)
        .map_err(|e| CudaHistogramError::Cuda(e.to_string()))
}
