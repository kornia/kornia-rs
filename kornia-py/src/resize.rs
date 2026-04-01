use pyo3::prelude::*;

use crate::image::{alloc_output_pyarray, numpy_as_image, to_pyerr, PyImage};
use kornia_image::ImageSize;
use kornia_imgproc::{interpolation::InterpolationMode, resize::resize_fast_rgb};

#[pyfunction]
pub fn resize(
    py: Python<'_>,
    image: PyImage,
    new_size: (usize, usize),
    interpolation: &str,
) -> PyResult<PyImage> {
    let interpolation = match interpolation.to_lowercase().as_str() {
        "nearest" => InterpolationMode::Nearest,
        "bilinear" => InterpolationMode::Bilinear,
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Invalid interpolation mode",
            ))
        }
    };
    let new_size = ImageSize {
        height: new_size.0,
        width: new_size.1,
    };
    let src = unsafe { numpy_as_image::<3>(py, &image)? };
    let (mut dst, out) = unsafe { alloc_output_pyarray::<3>(py, new_size)? };
    py.detach(|| resize_fast_rgb(&src, &mut dst, interpolation))
        .map_err(to_pyerr)?;
    Ok(out)
}
