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
fn gray_from_rgb_kernel<F: Float>(src: &Array<F>, dst: &mut Array<F>, args: GrayFromRgbArgs) {
    let x = CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X;
    let y = CUBE_POS_Y * CUBE_DIM_Y + UNIT_POS_Y;

    let cols = args.cols;
    let rows = args.rows;

    if x < cols && y < rows {
        let idx = y * cols + x;
        let r = src[3 * idx];
        let g = src[3 * idx + 1];
        let b = src[3 * idx + 2];

        let gray = r * comptime! {F::new(0.299)}
            + g * comptime! {F::new(0.587)}
            + b * comptime! {F::new(0.114)};

        dst[idx] = gray;
    }
}

/// Convert a RGB image to a grayscale image on the GPU.
pub fn gray_from_rgb_cubecl(src: &Image<u8, 3>, dst: &mut Image<u8, 1>) {
    let cols = src.cols() as u32;
    let rows = src.rows() as u32;

    let src_vec = src
        .as_slice()
        .iter()
        .map(|x| (*x as f32) / 255.0)
        .collect::<Vec<_>>();

    let dst_vec = dst
        .as_slice()
        .iter()
        .map(|x| (*x as f32) / 255.0)
        .collect::<Vec<_>>();

    type R = cubecl::wgpu::WgpuRuntime;
    let device = cubecl::wgpu::WgpuDevice::DefaultDevice;
    //let device = cubecl::wgpu::WgpuDevice::Cpu;

    //type R = cubecl::cuda::CudaRuntime;
    //let device = cubecl::cuda::CudaDevice::new(0);

    let client = R::client(&device);

    let input_handle = client.create(f32::as_bytes(&src_vec));
    let output_handle = client.create(f32::as_bytes(&dst_vec));

    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(dst.numel(), cube_dim);

    unsafe {
        gray_from_rgb_kernel::launch_unchecked::<f32, R>(
            &client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts::<f32>(&input_handle, src.numel(), 1),
            ArrayArg::from_raw_parts::<f32>(&output_handle, dst.numel(), 1),
            GrayFromRgbArgsLaunch::new(ScalarArg::new(cols), ScalarArg::new(rows)),
        );
    }

    // put the result back to the dst
    let out_bytes = client.read_one(output_handle.binding());
    let out_dst = f32::from_bytes(&out_bytes);

    for (out_i, out_dst_i) in dst.as_slice_mut().iter_mut().zip(out_dst.iter()) {
        *out_i = (*out_dst_i * 255.0) as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::Image;

    #[test]
    fn test_cubecl_small() -> Result<(), Box<dyn std::error::Error>> {
        let src = Image::new([2, 1].into(), vec![0, 128, 255, 128, 0, 128])?;
        let mut dst = Image::from_size_val([2, 1].into(), 0)?;

        gray_from_rgb_cubecl(&src, &mut dst);

        assert_eq!(dst.as_slice(), &[104, 52]);

        Ok(())
    }

    #[test]
    fn test_cubecl_large() -> Result<(), Box<dyn std::error::Error>> {
        let src = Image::new([1024, 1024].into(), vec![0; 1024 * 1024 * 3])?;
        let mut dst = Image::from_size_val([1024, 1024].into(), 0)?;

        gray_from_rgb_cubecl(&src, &mut dst);

        Ok(())
    }
}
