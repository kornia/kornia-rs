use pyo3::prelude::*;

use crate::image::{FromPyImage, PyImage, ToPyImage};
use kornia_image::Image;
use kornia_imgproc::enhance;

#[pyfunction]
pub fn add_weighted(
    src1: PyImage,
    alpha: f32,
    src2: PyImage,
    beta: f32,
    gamma: f32,
) -> PyResult<PyImage> {
    let image1: Image<u8, 3> = Image::from_pyimage(src1).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyException, _>(format!("src1 image: {}", e))
    })?;

    let image2: Image<u8, 3> = Image::from_pyimage(src2).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyException, _>(format!("src2 image: {}", e))
    })?;

    // cast input images to f32
    let image1 = image1.cast::<f32>().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyException, _>(format!("src1 image: {}", e))
    })?;

    let image2 = image2.cast::<f32>().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyException, _>(format!("src2 image: {}", e))
    })?;

    let mut dst: Image<f32, 3> = Image::from_size_val(image1.size(), 0.0f32)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("dst image: {}", e)))?;

    enhance::add_weighted(&image1, alpha, &image2, beta, gamma, &mut dst)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("dst image: {}", e)))?;

    // cast dst image to u8
    let dst = dst
        .cast::<u8>()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("dst image: {}", e)))?;

    Ok(dst.to_pyimage())
}
