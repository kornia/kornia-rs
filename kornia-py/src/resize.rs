use pyo3::prelude::*;

use crate::image::{FromPyImage, PyImage, ToPyImage};
use kornia_image::{Image, ImageSize};
use kornia_imgproc::{interpolation::InterpolationMode, resize::resize_fast};

#[pyfunction]
pub fn resize(image: PyImage, new_size: (usize, usize), interpolation: &str) -> PyResult<PyImage> {
    let image = Image::from_pyimage(image)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;

    let new_size = ImageSize {
        height: new_size.0,
        width: new_size.1,
    };

    let interpolation = match interpolation.to_lowercase().as_str() {
        "nearest" => InterpolationMode::Nearest,
        "bilinear" => InterpolationMode::Bilinear,
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Invalid interpolation mode",
            ))
        }
    };

    let mut image_resized = Image::from_size_val(new_size, 0u8)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;

    resize_fast(&image, &mut image_resized, interpolation)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;

    Ok(image_resized.to_pyimage())
}
