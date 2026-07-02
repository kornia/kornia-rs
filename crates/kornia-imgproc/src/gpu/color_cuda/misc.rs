//! CUDA kernels for miscellaneous color ops: sepia (Phase 2); colormap joins
//! in a later phase.
//!
//! Sepia u8 mirrors the CPU Q8 fixed-point path bit-for-bit
//! (`color/sepia.rs::sepia_u8_scalar`); sepia f32 mirrors the shared
//! `matrix3_affine_f32` MAC (`b + m0·c0 + m1·c1 + m2·c2` evaluation order).

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};

use super::{check_len, get_kernel, CudaColorError, KernelCell};

static SEPIA_U8_SRC: &str = r#"
extern "C" __global__ void sepia_from_rgb_u8(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    unsigned int npixels)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npixels) return;
    unsigned int b = i * 3u;
    unsigned int r  = __ldg(&src[b]);
    unsigned int g  = __ldg(&src[b + 1u]);
    unsigned int bl = __ldg(&src[b + 2u]);
    // Q8 coefficients round(coeff*256) — matches color/sepia.rs::Q.
    unsigned int rr = (101u * r + 197u * g + 48u * bl + 128u) >> 8;
    unsigned int gg = ( 89u * r + 176u * g + 43u * bl + 128u) >> 8;
    unsigned int bb = ( 70u * r + 137u * g + 34u * bl + 128u) >> 8;
    dst[b]      = (unsigned char)min(rr, 255u);
    dst[b + 1u] = (unsigned char)min(gg, 255u);
    dst[b + 2u] = (unsigned char)min(bb, 255u);
}
"#;

static SEPIA_F32_SRC: &str = r#"
extern "C" __global__ void sepia_from_rgb_f32(
    const float* __restrict__ src,
    float* __restrict__ dst,
    unsigned int npixels)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npixels) return;
    unsigned int b = i * 3u;
    float r  = __ldg(&src[b]);
    float g  = __ldg(&src[b + 1u]);
    float bl = __ldg(&src[b + 2u]);
    // Same MAC order as matrix3_affine_f32 with SEPIA_M and zero offset.
    dst[b]      = 0.0f + 0.393f * r + 0.769f * g + 0.189f * bl;
    dst[b + 1u] = 0.0f + 0.349f * r + 0.686f * g + 0.168f * bl;
    dst[b + 2u] = 0.0f + 0.272f * r + 0.534f * g + 0.131f * bl;
}
"#;

static SEPIA_U8: KernelCell = KernelCell::new();
static SEPIA_F32: KernelCell = KernelCell::new();

/// Launch sepia tone on RGB8 (Q8 fixed point, bit-exact vs CPU).
pub fn launch_sepia_from_rgb_u8(
    stream: &Arc<CudaStream>,
    src: &CudaSlice<u8>,
    dst: &mut CudaSlice<u8>,
    npixels: usize,
) -> Result<(), CudaColorError> {
    check_len("src", src.len(), npixels * 3)?;
    check_len("dst", dst.len(), npixels * 3)?;
    let kernel = get_kernel(&SEPIA_U8, stream, SEPIA_U8_SRC, "sepia_from_rgb_u8")?;
    let n = npixels as u32;
    kernel
        .launch_builder(stream)
        .arg(src)
        .arg(dst)
        .arg(&n)
        .launch_1d(n)?;
    Ok(())
}

/// Launch sepia tone on RGB f32 (linear matrix, no clamp — like the CPU path).
pub fn launch_sepia_from_rgb_f32(
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    npixels: usize,
) -> Result<(), CudaColorError> {
    check_len("src", src.len(), npixels * 3)?;
    check_len("dst", dst.len(), npixels * 3)?;
    let kernel = get_kernel(&SEPIA_F32, stream, SEPIA_F32_SRC, "sepia_from_rgb_f32")?;
    let n = npixels as u32;
    kernel
        .launch_builder(stream)
        .arg(src)
        .arg(dst)
        .arg(&n)
        .launch_1d(n)?;
    Ok(())
}

#[cfg(all(test, feature = "gpu-cuda"))]
mod tests {
    use super::*;
    use crate::gpu::color_cuda::test_utils::{default_stream, pattern_u8};

    #[test]
    fn sepia_u8_bit_exact_vs_cpu() {
        let stream = default_stream();
        let n = 640 * 480;
        let rgb = pattern_u8(n * 3);
        let mut cpu = vec![0u8; n * 3];
        crate::color::sepia::sepia_u8_scalar_oracle(&rgb, &mut cpu, n);

        let d_src = stream.clone_htod(&rgb).unwrap();
        let mut d_dst = stream.alloc_zeros::<u8>(n * 3).unwrap();
        launch_sepia_from_rgb_u8(&stream, &d_src, &mut d_dst, n).unwrap();
        let gpu: Vec<u8> = stream.clone_dtoh(&d_dst).unwrap();
        stream.synchronize().unwrap();
        assert_eq!(gpu, cpu);
    }
}
