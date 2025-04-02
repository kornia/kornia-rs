use pyo3::prelude::*;

use crate::image::{PyImage, ToPyImage};
use kornia_io::png as P;

#[pyfunction]
pub fn decode_image_png_mono8(png_data: &[u8]) -> PyResult<PyImage> {
    let image = P::decode_image_png_mono8(png_data)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(e.to_string()))?;
    Ok(image.to_pyimage())
}

#[pyfunction]
pub fn decode_image_png_rgb8(png_data: &[u8]) -> PyResult<PyImage> {
    let image = P::decode_image_png_rgb8(png_data)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(e.to_string()))?;
    Ok(image.to_pyimage())
}

#[pyfunction]
pub fn decode_image_png_rgba8(png_data: &[u8]) -> PyResult<PyImage> {
    let image = P::decode_image_png_rgba8(png_data)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(e.to_string()))?;
    Ok(image.to_pyimage())
}

#[pyfunction]
pub fn decode_image_png_mono16(png_data: &[u8]) -> PyResult<PyImage> {
    let image = P::decode_image_png_mono16(png_data)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(e.to_string()))?;
    Ok(image.to_pyimage())
}

#[pyfunction]
pub fn decode_image_png_rgb16(png_data: &[u8]) -> PyResult<PyImage> {
    let image = P::decode_image_png_rgb16(png_data)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(e.to_string()))?;
    Ok(image.to_pyimage())
}

#[pyfunction]
pub fn decode_image_png_rgba16(png_data: &[u8]) -> PyResult<PyImage> {
    let image = P::decode_image_png_rgba16(png_data)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(e.to_string()))?;
    Ok(image.to_pyimage())
}
