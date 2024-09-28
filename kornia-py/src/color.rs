use pyo3::prelude::*;

use crate::image::{FromPyImage, PyImage, ToPyImage};
use kornia_image::Image;
use kornia_imgproc::color;

#[pyfunction]
pub fn rgb_from_gray(image: PyImage) -> PyResult<PyImage> {
    let image_gray = Image::from_pyimage(image)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("src image: {}", e)))?;

    let mut image_rgb = Image::from_size_val(image_gray.size(), 0u8)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("dst image: {}", e)))?;

    color::rgb_from_gray(&image_gray, &mut image_rgb).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyException, _>(format!("failed to convert image: {}", e))
    })?;

    Ok(image_rgb.to_pyimage())
}

#[pyfunction]
pub fn bgr_from_rgb(image: PyImage) -> PyResult<PyImage> {
    let image_rgb = Image::from_pyimage(image)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("src image: {}", e)))?;

    let mut image_bgr = Image::from_size_val(image_rgb.size(), 0u8)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("dst image: {}", e)))?;

    color::bgr_from_rgb(&image_rgb, &mut image_bgr).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyException, _>(format!("failed to convert image: {}", e))
    })?;

    Ok(image_bgr.to_pyimage())
}

#[pyfunction]
pub fn gray_from_rgb(image: PyImage) -> PyResult<PyImage> {
    let image_rgb = Image::from_pyimage(image)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("src image: {}", e)))?;

    let image_rgb = image_rgb.cast::<f32>().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyException, _>(format!("failed to convert image: {}", e))
    })?;

    let mut image_gray = Image::from_size_val(image_rgb.size(), 0f32)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("dst image: {}", e)))?;

    color::gray_from_rgb(&image_rgb, &mut image_gray).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyException, _>(format!("failed to convert image: {}", e))
    })?;

    let image_gray = image_gray.cast::<u8>().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyException, _>(format!("failed to convert image: {}", e))
    })?;

    Ok(image_gray.to_pyimage())
}
