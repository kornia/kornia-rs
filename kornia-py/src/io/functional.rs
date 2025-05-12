use crate::image::{PyImage, ToPyImage};
use kornia_io::functional as F;
use pyo3::prelude::*;

#[pyfunction]
pub fn read_image_any(file_path: &str) -> PyResult<PyImage> {
    let image = F::read_image_any_rgb8(file_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyFileExistsError, _>(format!("{}", e)))?;
    Ok(image.to_pyimage())
}
