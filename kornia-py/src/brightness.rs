use pyo3::prelude::*;

use crate::image::{pyarray_data, vec_to_pyarray, PyImage};

#[pyfunction]
#[pyo3(name = "adjust_brightness")]
pub fn adjust_brightness_py(py: Python<'_>, image: PyImage, factor: f32) -> PyResult<PyImage> {
    let bound = image.bind(py);
    let (src, h, w, c) = pyarray_data(bound);
    let offset = factor * 255.0;
    let out: Vec<u8> = src
        .iter()
        .map(|&v| (v as f32 + offset).clamp(0.0, 255.0) as u8)
        .collect();
    Ok(vec_to_pyarray(py, out, h, w, c))
}
