use pyo3::prelude::*;

use crate::image::{
    alloc_output_pyarray, numpy_to_f32_image, parse_interpolation, to_pyerr, PyImage,
};
use kornia_image::{allocator::CpuAllocator, Image, ImageError, ImageSize};
use kornia_imgproc::warp;

#[pyfunction]
pub fn warp_affine(
    py: Python<'_>,
    image: PyImage,
    m: [f32; 6],
    new_size: (usize, usize),
    interpolation: &str,
) -> PyResult<PyImage> {
    let new_size = ImageSize {
        height: new_size.0,
        width: new_size.1,
    };
    let interpolation = parse_interpolation(interpolation)?;
    let src_f32 = numpy_to_f32_image::<3>(py, &image)?;
    let (mut dst_u8, out) = unsafe { alloc_output_pyarray::<3>(py, new_size)? };

    py.detach(|| -> Result<(), ImageError> {
        let mut dst_f32 = Image::from_size_val(new_size, 0f32, CpuAllocator)?;
        warp::warp_affine(&src_f32, &mut dst_f32, &m, interpolation)?;
        dst_u8
            .as_slice_mut()
            .iter_mut()
            .zip(dst_f32.as_slice().iter())
            .for_each(|(d, &s)| *d = s as u8);
        Ok(())
    })
    .map_err(to_pyerr)?;

    Ok(out)
}

#[pyfunction]
pub fn warp_perspective(
    py: Python<'_>,
    image: PyImage,
    m: [f32; 9],
    new_size: (usize, usize),
    interpolation: &str,
) -> PyResult<PyImage> {
    let new_size = ImageSize {
        height: new_size.0,
        width: new_size.1,
    };
    let interpolation = parse_interpolation(interpolation)?;
    let src_f32 = numpy_to_f32_image::<3>(py, &image)?;
    let (mut dst_u8, out) = unsafe { alloc_output_pyarray::<3>(py, new_size)? };

    py.detach(|| -> Result<(), ImageError> {
        let mut dst_f32 = Image::from_size_val(new_size, 0f32, CpuAllocator)?;
        warp::warp_perspective(&src_f32, &mut dst_f32, &m, interpolation)?;
        dst_u8
            .as_slice_mut()
            .iter_mut()
            .zip(dst_f32.as_slice().iter())
            .for_each(|(d, &s)| *d = s as u8);
        Ok(())
    })
    .map_err(to_pyerr)?;

    Ok(out)
}
