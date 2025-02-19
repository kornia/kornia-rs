use cubecl::calculate_cube_count_elemwise;
use cubecl::prelude::*;
use cubecl::Runtime;
use kornia_image::Image;
use serde::{Deserialize, Serialize};

#[derive(CubeLaunch, Deserialize, Serialize)]
struct GrayFromRgbArgs {
    cols: u32,
    rows: u32,
}

#[cube(launch_unchecked)]
fn gray_from_rgb_float_cl_kernel<F: Float>(
    src: &Array<Line<F>>,
    dst: &mut Array<Line<F>>,
    args: GrayFromRgbArgs,
) {
    let x = CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X;
    let y = CUBE_POS_Y * CUBE_DIM_Y + UNIT_POS_Y;

    let cols = args.cols;
    let rows = args.rows;

    if x < cols && y < rows {
        let idx = y * cols + x;
        let r = src[3 * idx] / 255.0;
        let g = src[3 * idx + 1] / 255.0;
        let b = src[3 * idx + 2] / 255.0;
        let gray = r * 0.299 + g * 0.587 + b * 0.114;
        dst[idx] = gray * 255.0;
    }
}

/// Convert a RGB image to a grayscale image on the GPU.
fn gray_from_rgb_float_cl_impl<R: Runtime>(
    src: &[f32],
    dst: &mut [f32],
    cols: u32,
    rows: u32,
    device: &R::Device,
) {
    let client = R::client(&device);

    let input_handle = client.create(f32::as_bytes(&src));
    let output_handle = client.create(f32::as_bytes(&dst));

    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(dst.len(), cube_dim);

    unsafe {
        gray_from_rgb_float_cl_kernel::launch_unchecked::<f32, R>(
            &client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts::<f32>(&input_handle, src.len(), 1),
            ArrayArg::from_raw_parts::<f32>(&output_handle, dst.len(), 1),
            GrayFromRgbArgsLaunch::new(ScalarArg::new(cols), ScalarArg::new(rows)),
        );
    }

    // put the result back to the dst
    let out_bytes = client.read_one(output_handle.binding());
    let out_dst = f32::from_bytes(&out_bytes);
    dst.copy_from_slice(&out_dst);
}

/// Convert a RGB image to a grayscale image on the GPU using cubecl with wgpu.
pub fn gray_from_rgb_float_cl_wgpu(src: &Image<f32, 3>, dst: &mut Image<f32, 1>) {
    let (cols, rows) = (dst.cols() as u32, dst.rows() as u32);
    let device = cubecl::wgpu::WgpuDevice::DefaultDevice;
    gray_from_rgb_float_cl_impl::<cubecl::wgpu::WgpuRuntime>(
        src.as_slice(),
        dst.as_slice_mut(),
        cols,
        rows,
        &device,
    );
}

/// Convert a RGB image to a grayscale image on the GPU using cubecl with cuda.
pub fn gray_from_rgb_float_cl_cuda(src: &Image<f32, 3>, dst: &mut Image<f32, 1>) {
    let (cols, rows) = (dst.cols() as u32, dst.rows() as u32);
    let device = cubecl::cuda::CudaDevice::new(0);
    gray_from_rgb_float_cl_impl::<cubecl::cuda::CudaRuntime>(
        src.as_slice(),
        dst.as_slice_mut(),
        cols,
        rows,
        &device,
    );
}

/// Convert a RGB8 image to a grayscale image on the GPU using cubecl with cuda.
pub fn gray_from_rgb_u8_cl_cuda(src: &Image<u8, 3>, dst: &mut Image<u8, 1>) {
    let (cols, rows) = (dst.cols() as u32, dst.rows() as u32);
    let device = cubecl::cuda::CudaDevice::new(0);
    let src_f32 = src.cast::<f32>().unwrap();
    let mut dst_f32 = dst.cast::<f32>().unwrap();
    gray_from_rgb_float_cl_impl::<cubecl::cuda::CudaRuntime>(
        src_f32.as_slice(),
        dst_f32.as_slice_mut(),
        cols,
        rows,
        &device,
    );
    for (dst, out) in dst_f32.as_slice().into_iter().zip(dst.as_slice_mut()) {
        *out = dst.round() as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::Image;

    #[test]
    fn test_cubecl_float_small() -> Result<(), Box<dyn std::error::Error>> {
        let src = Image::new([2, 1].into(), vec![0.0, 128.0, 255.0, 128.0, 0.0, 128.0])?;
        let mut dst = Image::from_size_val([2, 1].into(), 1.0)?;

        gray_from_rgb_float_cl_wgpu(&src, &mut dst);

        assert_eq!(dst.as_slice(), &[104.206, 52.864002]);

        Ok(())
    }

    #[test]
    fn test_cubecl_numeric_small() -> Result<(), Box<dyn std::error::Error>> {
        let src = Image::new([2, 1].into(), vec![0u8, 128, 255, 128, 0, 128])?;
        let mut dst = Image::from_size_val([2, 1].into(), 0)?;

        gray_from_rgb_u8_cl_cuda(&src, &mut dst);

        assert_eq!(dst.as_slice(), &[104, 53]);

        Ok(())
    }

    #[test]
    fn test_cubecl_large() -> Result<(), Box<dyn std::error::Error>> {
        let src = Image::new([1024, 1024].into(), vec![0.0; 1024 * 1024 * 3])?;
        let mut dst = Image::from_size_val([1024, 1024].into(), 0.0)?;

        gray_from_rgb_float_cl_cuda(&src, &mut dst);

        Ok(())
    }
}
