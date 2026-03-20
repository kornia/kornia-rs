use pyo3::prelude::*;

use crate::image::{FromPyImage, PyImage, ToPyImage};
use kornia_image::{allocator::CpuAllocator, Image};
use kornia_imgproc::flip;

#[pyfunction]
pub fn horizontal_flip(image: PyImage) -> PyResult<PyImage> {
    let image: Image<u8, 3, _> = Image::from_pyimage(image)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;

    let mut dst = Image::from_size_val(image.size(), 0u8, CpuAllocator)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;

    flip::horizontal_flip(&image, &mut dst)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;

    let pyimage = dst.to_pyimage().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyException, _>(format!("failed to convert image: {}", e))
    })?;

    Ok(pyimage)
}

#[pyfunction]
pub fn vertical_flip(image: PyImage) -> PyResult<PyImage> {
    let image: Image<u8, 3, _> = Image::from_pyimage(image)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;

    let mut dst = Image::from_size_val(image.size(), 0u8, CpuAllocator)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;

    flip::vertical_flip(&image, &mut dst)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;

    let pyimage = dst.to_pyimage().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyException, _>(format!("failed to convert image: {}", e))
    })?;

    Ok(pyimage)
}
