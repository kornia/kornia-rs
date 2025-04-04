use crate::image::{FromPyImage, PyImage, ToPyImage};
use kornia_image::Image;
use kornia_io::functional as F;
use pyo3::prelude::*;

#[pyfunction]
pub fn read_image_jpeg(file_path: &str) -> PyResult<PyImage> {
    let image = F::read_image_jpegturbo_rgb8(file_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyFileExistsError, _>(format!("{}", e)))?;
    Ok(image.to_pyimage())
}

#[pyfunction]
pub fn write_image_jpeg(file_path: &str, image: PyImage) -> PyResult<()> {
    let image = Image::from_pyimage(image)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;
    F::write_image_jpegturbo_rgb8(file_path, &image)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;
    Ok(())
}

#[pyfunction]
pub fn read_image_any(file_path: &str) -> PyResult<PyImage> {
    let image = F::read_image_any_rgb8(file_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyFileExistsError, _>(format!("{}", e)))?;
    Ok(image.to_pyimage())
}
