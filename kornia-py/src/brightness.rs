use pyo3::prelude::*;

use crate::image::{adjust_brightness_into_pyarray, pyarray_data, PyImage};

#[pyfunction]
#[pyo3(name = "adjust_brightness")]
pub fn adjust_brightness_py(py: Python<'_>, image: PyImage, factor: f32) -> PyResult<PyImage> {
    let bound = image.bind(py);
    let (src, h, w, c) = pyarray_data(bound);
    Ok(adjust_brightness_into_pyarray(
        py,
        src,
        factor * 255.0,
        h,
        w,
        c,
    ))
}
