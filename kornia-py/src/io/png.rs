use pyo3::prelude::*;

use crate::image::{PyImage, ToPyImage};
use kornia_io::png as P;

pub fn decode_image_png_mono8(png_data: &[u8]) -> PyResult<PyImage> {
    let image = P::decode_image_png_mono8(png_data)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;
    Ok(image.to_pyimage())
}

pub fn decode_image_png_rgb8(png_data: &[u8]) -> PyResult<PyImage> {
    let image = P::decode_image_png_rgb8(png_data)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;
    Ok(image.to_pyimage())
}

pub fn decode_image_png_rgba8(png_data: &[u8]) -> PyResult<PyImage> {
    let image = P::decode_image_png_rgba8(png_data)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;
    Ok(image.to_pyimage())
}

pub fn decode_image_png_mono16(png_data: &[u8]) -> PyResult<PyImage> {
    let image = P::decode_image_png_mono16(png_data)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;
    Ok(image.to_pyimage())
}

pub fn decode_image_png_rgb16(png_data: &[u8]) -> PyResult<PyImage> {
    let image = P::decode_image_png_rgb16(png_data)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;
    Ok(image.to_pyimage())
}

pub fn decode_image_png_rgba16(png_data: &[u8]) -> PyResult<PyImage> {
    let image = P::decode_image_png_rgba16(png_data)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;
    Ok(image.to_pyimage())
}

#[pyfunction]
pub fn decode_image_png(png_data: &[u8], mode: &str) -> PyResult<PyImage> {
    match mode {
        "rgb8" => decode_image_png_rgb8(png_data),
        "rgba8" => decode_image_png_rgba8(png_data),
        "mono8" => decode_image_png_mono8(png_data),
        "rgb16" => decode_image_png_rgb16(png_data),
        "rgba16" => decode_image_png_rgba16(png_data),
        "mono16" => decode_image_png_mono16(png_data),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            String::from(
                r#"\
        The following are the supported values of mode:
        1) "rgb8" -> 8-bit RGB
        2) "rgba8" -> 8-bit RGBA
        3) "mono8" -> 8-bit Monochrome
        4) "rgb16" -> 16-bit RGB
        5) "rgba16" -> 16-bit RGBA
        6) "mono16" -> 16-bit Monochrome
        "#,
            ),
        )),
    }
}
