use pyo3::prelude::*;
use std::path::Path;

use crate::image::PyImage;
use kornia_rs::io::functional as F;

#[pyfunction]
pub fn read_image_jpeg(file_path: String) -> PyResult<PyImage> {
    let image = F::read_image_jpeg(Path::new(&file_path)).unwrap();
    Ok(image.into())
}

#[pyfunction]
pub fn write_image_jpeg(file_path: String, image: PyImage) -> PyResult<()> {
    F::write_image_jpeg(Path::new(&file_path), image.into()).unwrap();
    Ok(())
}

#[pyfunction]
pub fn read_image_any(file_path: String) -> PyResult<PyImage> {
    let image = F::read_image_any(Path::new(&file_path)).unwrap();
    Ok(image.into())
}
