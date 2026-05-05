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
    let (mut dst, out) = unsafe { alloc_output_pyarray::<3>(py, src.size())? };

    py.detach(|| filter::box_blur_u8(&src, &mut dst, kernel_size))
        .map_err(to_pyerr)?;

    Ok(out)
}
