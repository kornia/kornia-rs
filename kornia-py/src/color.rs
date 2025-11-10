use pyo3::prelude::*;

use crate::image::{FromPyImage, PyImage, ToPyImage};
use kornia_image::{allocator::CpuAllocator, Image};
use kornia_imgproc::color;

#[pyfunction]
pub fn rgb_from_gray(image: PyImage) -> PyResult<PyImage> {
    let image_gray = Image::from_pyimage(image)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("src image: {}", e)))?;

    let mut image_rgb = Image::from_size_val(image_gray.size(), 0u8, CpuAllocator)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("dst image: {}", e)))?;

    color::rgb_from_gray(&image_gray, &mut image_rgb).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyException, _>(format!("failed to convert image: {}", e))
    })?;

    let pyimage_rgb = image_rgb.to_pyimage().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyException, _>(format!("failed to convert image: {}", e))
    })?;

    Ok(pyimage_rgb)
}

#[pyfunction]
pub fn bgr_from_rgb(image: PyImage) -> PyResult<PyImage> {
    let image_rgb = Image::from_pyimage(image)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("src image: {}", e)))?;

    let mut image_bgr = Image::from_size_val(image_rgb.size(), 0u8, CpuAllocator)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("dst image: {}", e)))?;

    color::bgr_from_rgb(&image_rgb, &mut image_bgr).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyException, _>(format!("failed to convert image: {}", e))
    })?;

    let pyimage_bgr = image_bgr.to_pyimage().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyException, _>(format!("failed to convert image: {}", e))
    })?;

    Ok(pyimage_bgr)
}

#[pyfunction]
pub fn gray_from_rgb(image: PyImage) -> PyResult<PyImage> {
    let image_rgb = Image::from_pyimage(image)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("src image: {}", e)))?;

    let image_rgb = image_rgb.cast::<f32>().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyException, _>(format!("failed to convert image: {}", e))
    })?;

    let mut image_gray = Image::from_size_val(image_rgb.size(), 0f32, CpuAllocator)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("dst image: {}", e)))?;

    color::gray_from_rgb(&image_rgb, &mut image_gray).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyException, _>(format!("failed to convert image: {}", e))
    })?;

    let image_gray = image_gray.cast::<u8>().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyException, _>(format!("failed to convert image: {}", e))
    })?;

    let pyimage_gray = image_gray.to_pyimage().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyException, _>(format!("failed to convert image: {}", e))
    })?;

    Ok(pyimage_gray)
}

#[pyfunction]
#[pyo3(signature = (image, background=None))]
pub fn rgb_from_rgba(image: PyImage, background: Option<[u8; 3]>) -> PyResult<PyImage> {
    let image_rgba = Image::from_pyimage(image)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("src image: {}", e)))?;

    let mut image_rgb = Image::from_size_val(image_rgba.size(), 0u8, CpuAllocator)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("dst image: {}", e)))?;

    color::rgb_from_rgba(&image_rgba, &mut image_rgb, background).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyException, _>(format!("failed to convert image: {}", e))
    })?;

    let pyimage_rgb = image_rgb.to_pyimage().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyException, _>(format!("failed to convert image: {}", e))
    })?;

    Ok(pyimage_rgb)
}

#[pyfunction]
#[pyo3(signature = (image, background=None))]
pub fn rgb_from_bgra(image: PyImage, background: Option<[u8; 3]>) -> PyResult<PyImage> {
    let image_bgra = Image::from_pyimage(image)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("src image: {}", e)))?;

    let mut image_rgb = Image::from_size_val(image_bgra.size(), 0u8, CpuAllocator)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("dst image: {}", e)))?;

    color::rgb_from_bgra(&image_bgra, &mut image_rgb, background).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyException, _>(format!("failed to convert image: {}", e))
    })?;

    let pyimage_rgb = image_rgb.to_pyimage().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyException, _>(format!("failed to convert image: {}", e))
    })?;

    Ok(pyimage_rgb)
}
