use pyo3::prelude::*;

use crate::image::{alloc_output_pyarray, numpy_to_f32_image, to_pyerr, PyImage};
use kornia_image::{allocator::CpuAllocator, Image, ImageError};
use kornia_imgproc::filter;

#[pyfunction]
pub fn gaussian_blur(
    py: Python<'_>,
    image: PyImage,
    kernel_size: (usize, usize),
    sigma: (f32, f32),
) -> PyResult<PyImage> {
    let src_f32 = numpy_to_f32_image::<3>(py, &image)?;
    let size = src_f32.size();
    let (mut dst_u8, out) = unsafe { alloc_output_pyarray::<3>(py, size)? };

    py.detach(|| -> Result<(), ImageError> {
        let mut dst_f32 = Image::from_size_val(size, 0.0f32, CpuAllocator)?;
        filter::gaussian_blur(&src_f32, &mut dst_f32, kernel_size, sigma)?;
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
pub fn box_blur(py: Python<'_>, image: PyImage, kernel_size: (usize, usize)) -> PyResult<PyImage> {
    let src_f32 = numpy_to_f32_image::<3>(py, &image)?;
    let size = src_f32.size();
    let (mut dst_u8, out) = unsafe { alloc_output_pyarray::<3>(py, size)? };

    py.detach(|| -> Result<(), ImageError> {
        let mut dst_f32 = Image::from_size_val(size, 0.0f32, CpuAllocator)?;
        filter::box_blur(&src_f32, &mut dst_f32, kernel_size)?;
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
