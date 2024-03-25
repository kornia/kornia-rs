use pyo3::prelude::*;
use std::path::Path;

use crate::image::{FromPyImage, PyImage, ToPyImage};
use kornia_rs::{image::Image, io::functional as F};

#[pyfunction]
pub fn read_image_jpeg(file_path: &str) -> PyResult<PyImage> {
    let image = F::read_image_jpeg(Path::new(file_path))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyFileExistsError, _>(format!("{}", e)))?;
    Ok(image.to_pyimage())
}

#[pyfunction]
pub fn write_image_jpeg(file_path: &str, image: PyImage) -> PyResult<()> {
    let image = Image::from_pyimage(image)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;
    F::write_image_jpeg(Path::new(file_path), &image)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;
    Ok(())
}

#[pyfunction]
pub fn read_image_any(file_path: &str) -> PyResult<PyImage> {
    let image = F::read_image_any(Path::new(&file_path))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyFileExistsError, _>(format!("{}", e)))?;
    Ok(image.to_pyimage())
}
