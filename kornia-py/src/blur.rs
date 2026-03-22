use pyo3::prelude::*;

use crate::image::{FromPyImage, PyImage, ToPyImage};
use kornia_image::{allocator::CpuAllocator, Image};
use kornia_imgproc::filter;

#[pyfunction]
pub fn gaussian_blur(
    image: PyImage,
    kernel_size: (usize, usize),
    sigma: (f32, f32),
) -> PyResult<PyImage> {
    let image: Image<u8, 3, _> = Image::from_pyimage(image)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;

    let image = image
        .cast::<f32>()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;

    let mut dst: Image<f32, 3, _> = Image::from_size_val(image.size(), 0.0f32, CpuAllocator)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;

    filter::gaussian_blur(&image, &mut dst, kernel_size, sigma)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;

    let dst = dst
        .cast::<u8>()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;

    let pyimage = dst.to_pyimage().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyException, _>(format!("failed to convert image: {}", e))
    })?;

    Ok(pyimage)
}

#[pyfunction]
pub fn box_blur(image: PyImage, kernel_size: (usize, usize)) -> PyResult<PyImage> {
    let image: Image<u8, 3, _> = Image::from_pyimage(image)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;

    let image = image
        .cast::<f32>()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;

    let mut dst: Image<f32, 3, _> = Image::from_size_val(image.size(), 0.0f32, CpuAllocator)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;

    filter::box_blur(&image, &mut dst, kernel_size)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;

    let dst = dst
        .cast::<u8>()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;

    let pyimage = dst.to_pyimage().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyException, _>(format!("failed to convert image: {}", e))
    })?;

    Ok(pyimage)
}
