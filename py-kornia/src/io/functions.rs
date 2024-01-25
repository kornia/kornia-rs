use pyo3::prelude::*;
use std::path::Path;

use crate::tensor::PyTensor;
use kornia_rs::io::functions as F;

#[pyfunction]
pub fn read_image_jpeg(file_path: String) -> PyResult<PyTensor> {
    let tensor = F::read_image_jpeg(Path::new(&file_path));
    Ok(tensor.into())
}

#[pyfunction]
pub fn write_image_jpeg(file_path: String, image: PyTensor) -> PyResult<()> {
    F::write_image_jpeg(Path::new(&file_path), image.into());
    Ok(())
}

#[pyfunction]
pub fn read_image_any(file_path: String) -> PyResult<PyTensor> {
    let tensor = F::read_image_any(Path::new(&file_path));
    Ok(tensor.into())
}
