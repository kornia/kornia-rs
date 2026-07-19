//! CUDA bilateral-filter kernel — textual twin of `filter/bilateral.rs`.
//!
//! The kernel receives the SAME host-built tables the CPU path uses
//! (`BilateralTables`: cv2's v_exp color table, f64-exp space weights,
//! row-major circular taps) and mirrors the accumulation loop textually:
//! `w = space_w[k] · color_w[|val − val0|]`, `wsum += w`,
//! `sum = fmaf(val, w, sum)`, output `rintf(sum / wsum)` (IEEE division —
//! NVRTC's default `prec-div=true`). Byte parity with the CPU (and hence
//! with `cv2.bilateralFilter`) by construction.

use std::sync::{Arc, OnceLock};

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use kornia_tensor::CudaKernel;

use super::try_compile_with_l1;
use crate::filter::bilateral::BilateralTables;

/// Error type for the CUDA bilateral launcher.
#[derive(Debug, thiserror::Error)]
pub enum CudaBilateralError {
    /// CUDA driver / compile / launch error.
    #[error("CUDA bilateral error: {0}")]
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

fn check_slice(what: &'static str, got: usize, need: usize) -> Result<(), CudaBilateralError> {
    if got < need {
        return Err(CudaBilateralError::SliceTooSmall { what, got, need });
    }
    Ok(())
}

static BILATERAL_SRC: &str = r#"
// Textual twin of bilateral.rs::reflect_101.
__device__ __forceinline__ int reflect_101(int p, int len) {
    if (len == 1) return 0;
    while (p < 0 || p >= len) {
        if (p < 0) p = -p; else p = 2 * (len - 1) - p;
    }
    return p;
}

extern "C" __global__ void bilateral_u8(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__       dst,
    const int* __restrict__           tap_dy,
    const int* __restrict__           tap_dx,
    const float* __restrict__         space_weight,
    const float* __restrict__         color_weight,
    const int* __restrict__           simd_order,
    int w, int h, int ntaps, int simd_end
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int val0 = __ldg(&src[y * w + x]);
    float wsum = 0.0f;
    float sum = 0.0f;
    bool in_simd = x < simd_end;
    for (int kk = 0; kk < ntaps; ++kk) {
        // Twin of the CPU loop: cv2's SIMD-region tap order for pixels
        // below simd_end, sequential order for the scalar-tail pixels.
        int k = in_simd ? __ldg(&simd_order[kk]) : kk;
        int sy = reflect_101(y + __ldg(&tap_dy[k]), h);
        int sx = reflect_101(x + __ldg(&tap_dx[k]), w);
        int val = __ldg(&src[sy * w + sx]);
        float wgt = __ldg(&space_weight[k]) * __ldg(&color_weight[abs(val - val0)]);
        wsum += wgt;
        sum = fmaf((float)val, wgt, sum);
    }
    dst[y * w + x] = (unsigned char)(int)rintf(sum / wsum);
}
"#;

static KERNEL: OnceLock<Result<CudaKernel, String>> = OnceLock::new();

/// Bilateral filter on device with the host-built `BilateralTables` —
/// byte-identical to the CPU `bilateral_filter`.
pub fn launch_bilateral_u8(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<u8>,
    dst: &mut CudaSlice<u8>,
    width: usize,
    height: usize,
    t: &BilateralTables,
) -> Result<(), CudaBilateralError> {
    if width == 0 || height == 0 {
        return Err(CudaBilateralError::Cuda(
            "image dimensions must be non-zero".into(),
        ));
    }
    let ntaps = t.taps.len();
    if ntaps == 0 || t.space_weight.len() != ntaps || t.color_weight.len() != 256 {
        return Err(CudaBilateralError::Cuda(
            "inconsistent bilateral tables".into(),
        ));
    }
    check_slice("src", src.len(), width * height)?;
    check_slice("dst", dst.len(), width * height)?;
    let w =
        i32::try_from(width).map_err(|_| CudaBilateralError::Cuda("width exceeds i32".into()))?;
    let h =
        i32::try_from(height).map_err(|_| CudaBilateralError::Cuda("height exceeds i32".into()))?;
    let n =
        i32::try_from(ntaps).map_err(|_| CudaBilateralError::Cuda("ntaps exceeds i32".into()))?;

    if t.simd_order.len() != ntaps {
        return Err(CudaBilateralError::Cuda(
            "inconsistent bilateral tables".into(),
        ));
    }
    let simd_end = i32::try_from(crate::filter::bilateral::simd_region_end(width))
        .map_err(|_| CudaBilateralError::Cuda("width exceeds i32".into()))?;

    let err = |e: cudarc::driver::DriverError| CudaBilateralError::Cuda(e.to_string());
    let dys: Vec<i32> = t.taps.iter().map(|&(dy, _)| dy).collect();
    let dxs: Vec<i32> = t.taps.iter().map(|&(_, dx)| dx).collect();
    let ord: Vec<i32> = t.simd_order.iter().map(|&k| k as i32).collect();
    let d_dy = stream.clone_htod(&dys).map_err(err)?;
    let d_dx = stream.clone_htod(&dxs).map_err(err)?;
    let d_sw = stream.clone_htod(&t.space_weight).map_err(err)?;
    let d_cw = stream.clone_htod(&t.color_weight).map_err(err)?;
    let d_or = stream.clone_htod(&ord).map_err(err)?;

    let kernel = KERNEL
        .get_or_init(|| try_compile_with_l1(ctx, BILATERAL_SRC, "bilateral_u8"))
        .as_ref()
        .map_err(|e| CudaBilateralError::Cuda(e.clone()))?;
    let cfg = super::make_config(w as u32, h as u32, None);
    kernel
        .launch_builder(stream)
        .arg(src)
        .arg(dst)
        .arg(&d_dy)
        .arg(&d_dx)
        .arg(&d_sw)
        .arg(&d_cw)
        .arg(&d_or)
        .arg(&w)
        .arg(&h)
        .arg(&n)
        .arg(&simd_end)
        .launch_cfg(cfg)
        .map_err(|e| CudaBilateralError::Cuda(e.to_string()))
}
