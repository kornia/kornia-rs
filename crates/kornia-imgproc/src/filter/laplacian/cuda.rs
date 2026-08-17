use cudarc::driver::CudaStream;
use kornia_image::{Image, ImageError};
use kornia_tensor::CudaKernel;
use std::sync::{Arc, OnceLock};

use crate::cuda::try_compile_with_l1;

static LAPLACIAN_SRC: &str = r#"
extern "C" __global__ void laplacian_u8_to_i16(
    const unsigned char* __restrict__ src,
    short* __restrict__ dst,
    unsigned int w,
    unsigned int h,
    unsigned int c
) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return;

    unsigned int x_left = max((int)x - 1, 0);
    unsigned int x_right = min(x + 1, w - 1);
    unsigned int y_up = max((int)y - 1, 0);
    unsigned int y_down = min(y + 1, h - 1);

    for (unsigned int ch = 0; ch < c; ++ch) {
        short v_center = (short)src[(y * w + x) * c + ch];
        short v_up = (short)src[(y_up * w + x) * c + ch];
        short v_down = (short)src[(y_down * w + x) * c + ch];
        short v_left = (short)src[(y * w + x_left) * c + ch];
        short v_right = (short)src[(y * w + x_right) * c + ch];

        short val = v_up + v_down + v_left + v_right - 4 * v_center;
        dst[(y * w + x) * c + ch] = val;
    }
}
"#;

static LAPLACIAN_U8_KERNEL: OnceLock<Result<CudaKernel, String>> = OnceLock::new();

/// Native CUDA implementation for laplacian filter on u8 images.
pub fn laplacian_u8_cuda<const C: usize>(
    src: &Image<u8, C>,
    dst: &mut Image<i16, C>,
    stream: &Arc<CudaStream>,
) -> Result<(), ImageError> {
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

    let kernel = LAPLACIAN_U8_KERNEL
        .get_or_init(|| try_compile_with_l1(ctx, LAPLACIAN_SRC, "laplacian_u8_to_i16"))
        .as_ref()
        .map_err(|e| ImageError::Cuda(e.clone()))?;

    // We use a 2D grid block. 32x8 is standard for 2D image kernels in kornia.
    kernel
        .launch_builder(stream)
        .arg(src_slice)
        .arg(dst_slice)
        .arg(&w)
        .arg(&h)
        .arg(&c)
        .launch_2d(w, h, crate::cuda::make_config(w, h, None))
        .map_err(|e| ImageError::Cuda(e.to_string()))?;

    Ok(())
}
