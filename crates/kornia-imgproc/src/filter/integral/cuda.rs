use cudarc::driver::CudaStream;
use kornia_image::{Image, ImageError};
use kornia_tensor::CudaKernel;
use std::sync::{Arc, OnceLock};

use crate::cuda::try_compile_with_l1;

static INTEGRAL_H_SRC: &str = r#"
extern "C" __global__ void integral_h_u8(
    const unsigned char* __restrict__ src,
    float* __restrict__ dst,
    unsigned int w,
    unsigned int h,
    unsigned int c
) {
    unsigned int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= h) return;

    if (c == 1) {
        float sum = 0.0f;
        for (unsigned int x = 0; x < w; ++x) {
            sum += (float)src[y * w + x];
            dst[y * w + x] = sum;
        }
    } else if (c == 3) {
        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f;
        for (unsigned int x = 0; x < w; ++x) {
            unsigned int idx = (y * w + x) * 3;
            sum0 += (float)src[idx];
            sum1 += (float)src[idx + 1];
            sum2 += (float)src[idx + 2];
            dst[idx] = sum0;
            dst[idx + 1] = sum1;
            dst[idx + 2] = sum2;
        }
    }
}

extern "C" __global__ void integral_h_f32(
    const float* __restrict__ src,
    float* __restrict__ dst,
    unsigned int w,
    unsigned int h,
    unsigned int c
) {
    unsigned int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= h) return;

    if (c == 1) {
        float sum = 0.0f;
        for (unsigned int x = 0; x < w; ++x) {
            sum += src[y * w + x];
            dst[y * w + x] = sum;
        }
    } else if (c == 3) {
        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f;
        for (unsigned int x = 0; x < w; ++x) {
            unsigned int idx = (y * w + x) * 3;
            sum0 += src[idx];
            sum1 += src[idx + 1];
            sum2 += src[idx + 2];
            dst[idx] = sum0;
            dst[idx + 1] = sum1;
            dst[idx + 2] = sum2;
        }
    }
}
"#;

static INTEGRAL_V_SRC: &str = r#"
extern "C" __global__ void integral_v(
    float* __restrict__ dst,
    unsigned int w,
    unsigned int h,
    unsigned int c
) {
    // 1 thread per element (x, c)
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= w * c) return;

    float sum = dst[x];
    for (unsigned int y = 1; y < h; ++y) {
        unsigned int idx = y * w * c + x;
        sum += dst[idx];
        dst[idx] = sum;
    }
}
"#;

static INTEGRAL_H_U8_KERNEL: OnceLock<Result<CudaKernel, String>> = OnceLock::new();
static INTEGRAL_H_F32_KERNEL: OnceLock<Result<CudaKernel, String>> = OnceLock::new();
static INTEGRAL_V_KERNEL: OnceLock<Result<CudaKernel, String>> = OnceLock::new();

/// Native CUDA implementation for integral image computation.
///
/// Dispatches internally based on the element type.
pub fn integral_image_cuda<T, const C: usize>(
    src: &Image<T, C>,
    dst: &mut Image<f32, C>,
    stream: &Arc<CudaStream>,
) -> Result<(), ImageError>
where
    T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits + 'static,
{
    let ctx = stream.context();
    let src_slice = src
        .0
        .as_cudaslice()
        .ok_or(ImageError::Cuda("not cuda".into()))?;
    let dst_slice = dst
        .0
        .as_cudaslice_mut()
        .ok_or(ImageError::Cuda("not cuda".into()))?;
    let w = src.cols() as u32;
    let h = src.rows() as u32;
    let c = C as u32;

    let is_u8 = std::any::type_name::<T>() == "u8";

    let kernel_h = if is_u8 {
        INTEGRAL_H_U8_KERNEL
            .get_or_init(|| try_compile_with_l1(ctx, INTEGRAL_H_SRC, "integral_h_u8"))
            .as_ref()
            .map_err(|e| ImageError::Cuda(e.clone()))?
    } else {
        INTEGRAL_H_F32_KERNEL
            .get_or_init(|| try_compile_with_l1(ctx, INTEGRAL_H_SRC, "integral_h_f32"))
            .as_ref()
            .map_err(|e| ImageError::Cuda(e.clone()))?
    };

    let kernel_v = INTEGRAL_V_KERNEL
        .get_or_init(|| try_compile_with_l1(ctx, INTEGRAL_V_SRC, "integral_v"))
        .as_ref()
        .map_err(|e| ImageError::Cuda(e.clone()))?;

    // Pass 1: Horizontal sum (1 thread per row)
    kernel_h
        .launch_builder(stream)
        .arg(src_slice)
        .arg(&mut *dst_slice)
        .arg(&w)
        .arg(&h)
        .arg(&c)
        .launch_1d(h)
        .map_err(|e| ImageError::Cuda(e.to_string()))?;

    // Pass 2: Vertical sum (1 thread per column * channel)
    kernel_v
        .launch_builder(stream)
        .arg(dst_slice)
        .arg(&w)
        .arg(&h)
        .arg(&c)
        .launch_1d(w * c)
        .map_err(|e| ImageError::Cuda(e.to_string()))?;

    Ok(())
}
