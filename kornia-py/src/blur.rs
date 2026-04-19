use pyo3::prelude::*;

use crate::image::{alloc_output_pyarray, numpy_as_image, to_pyerr, PyImage};
use kornia_imgproc::filter;

#[pyfunction]
pub fn gaussian_blur(
    py: Python<'_>,
    image: PyImage,
    kernel_size: (usize, usize),
    sigma: (f32, f32),
) -> PyResult<PyImage> {
    let src = unsafe { numpy_as_image::<3>(py, &image)? };
    let (mut dst, out) = unsafe { alloc_output_pyarray::<3>(py, src.size())? };

    py.detach(|| filter::gaussian_blur_u8(&src, &mut dst, kernel_size, sigma))
        .map_err(to_pyerr)?;

    Ok(out)
}

#[pyfunction]
pub fn box_blur(py: Python<'_>, image: PyImage, kernel_size: (usize, usize)) -> PyResult<PyImage> {
    let src = unsafe { numpy_as_image::<3>(py, &image)? };
    let size = src.size();
    let (mut dst, out) = unsafe { alloc_output_pyarray::<3>(py, size)? };

    py.detach(|| -> Result<(), kornia_image::ImageError> {
        let mut src_f32 = kornia_image::Image::from_size_val(
            size,
            0.0f32,
            kornia_image::allocator::CpuAllocator,
        )?;
        kornia_image::ops::cast_and_scale(&src, &mut src_f32, 1.0)?;
        let mut dst_f32 = kornia_image::Image::from_size_val(
            size,
            0.0f32,
            kornia_image::allocator::CpuAllocator,
        )?;
        filter::box_blur(&src_f32, &mut dst_f32, kernel_size)?;
        dst.as_slice_mut()
            .iter_mut()
            .zip(dst_f32.as_slice().iter())
            .for_each(|(d, &s)| *d = s as u8);
        Ok(())
    })
    .map_err(to_pyerr)?;

    Ok(out)
}
