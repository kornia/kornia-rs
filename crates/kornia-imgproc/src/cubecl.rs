use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

use cubecl::calculate_cube_count_elemwise;
use cubecl::prelude::*;
use cubecl::{client::ComputeClient, server::Handle, Runtime};

/// 3D shape of the image.
pub struct Shape2 {
    /// Number of columns in the image, or width of the image.
    pub cols: usize,
    /// Number of rows in the image, or height of the image.
    pub rows: usize,
}

/// Cubecl image
pub struct ImageCl<T, const C: usize, R: Runtime> {
    client: ComputeClient<R::Server, R::Channel>,
    device: R::Device,
    handle: Handle,
    shape: Shape2,
    _marker: PhantomData<T>,
}

impl<T, const C: usize, R: Runtime> ImageCl<T, C, R> {
    /// Get the shape of the image.
    #[inline]
    pub fn shape(&self) -> &Shape2 {
        &self.shape
    }

    /// Create an empty image.
    pub fn empty(shape: Shape2, device: R::Device) -> Self {
        let client = R::client(&device);
        let handle = client.empty(shape.cols * shape.rows);

        Self {
            client,
            device,
            handle,
            shape,
            _marker: PhantomData,
        }
    }

    /// Create an image from a slice.
    pub fn from_slice(shape: Shape2, slice: &[u8], device: R::Device) -> Self {
        let client = R::client(&device);
        let handle = client.create(slice);

        Self {
            client,
            device,
            handle,
            shape,
            _marker: PhantomData,
        }
    }

    /// Convert the image to a vector of u8.
    pub fn to_vec(&self) -> Vec<u8> {
        let client = R::client(&self.device);
        let binding = self.handle.clone().binding();
        client.read_one(binding)
    }
}

#[derive(CubeLaunch, Deserialize, Serialize)]
struct GrayFromRgbArgs {
    cols: u32,
    rows: u32,
}

#[cube(launch_unchecked)]
fn gray_from_rgb_u8_cl_kernel(
    src: &Array<Line<u8>>,
    dst: &mut Array<Line<u8>>,
    args: GrayFromRgbArgs,
) {
    let x = CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X;
    let y = CUBE_POS_Y * CUBE_DIM_Y + UNIT_POS_Y;

    let cols = args.cols;
    let rows = args.rows;

    if x < cols && y < rows {
        let idx = y * cols + x;
        let r = u16::cast_from(src[3 * idx]);
        let g = u16::cast_from(src[3 * idx + 1]);
        let b = u16::cast_from(src[3 * idx + 2]);
        let gray = u8::cast_from(((r * 77 + g * 150 + b * 29) + 128) >> 8);
        dst[idx] = Line::new(gray);
    }
}

/// Convert a RGB8 image to a grayscale image on the GPU using cubecl with cuda.
pub fn gray_from_rgb_u8_cl<R: Runtime>(src: &ImageCl<u8, 3, R>, dst: &mut ImageCl<u8, 1, R>) {
    let cols = dst.shape.cols;
    let rows = dst.shape.rows;
    let num_elems = cols * rows;

    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_elems, cube_dim);

    unsafe {
        gray_from_rgb_u8_cl_kernel::launch_unchecked::<R>(
            &src.client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts::<u8>(&src.handle, num_elems, 1),
            ArrayArg::from_raw_parts::<u8>(&dst.handle, num_elems, 1),
            GrayFromRgbArgsLaunch::new(ScalarArg::new(cols as u32), ScalarArg::new(rows as u32)),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gray_from_rgb_u8_cl_cuda() -> Result<(), Box<dyn std::error::Error>> {
        let device = cubecl::cuda::CudaDevice::new(0);
        let src = ImageCl::from_slice(
            Shape2 { cols: 2, rows: 1 },
            &[0u8, 128, 255, 128, 0, 128],
            device.clone(),
        );

        let mut dst = ImageCl::empty(Shape2 { cols: 2, rows: 1 }, device.clone());

        gray_from_rgb_u8_cl::<cubecl::cuda::CudaRuntime>(&src, &mut dst);

        let dst_data = dst.to_vec();
        assert_eq!(dst_data, vec![104, 53]);

        Ok(())
    }
}
