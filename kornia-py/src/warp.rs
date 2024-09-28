use pyo3::prelude::*;

use crate::image::{FromPyImage, PyImage, ToPyImage};
use kornia_image::{Image, ImageSize};
use kornia_imgproc::interpolation::InterpolationMode;
use kornia_imgproc::warp;

#[pyfunction]
pub fn warp_affine(
    image: PyImage,
    m: [f32; 6],
    new_size: (usize, usize),
    interpolation: &str,
) -> PyResult<PyImage> {
    // have to add annotation Image<u8, 3>, otherwise the compiler will complain
    // NOTE: do we support images with channels != 3?
    let image: Image<u8, 3> = Image::from_pyimage(image)
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

    // we need to cast to f32 for now since kornia-rs interpolation function only works with f32
    let image = image
        .cast::<f32>()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;

    let mut image_warped = Image::from_size_val(new_size, 0f32)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;

    warp::warp_affine(&image, &mut image_warped, &m, interpolation)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;

    // NOTE: for bicubic interpolation (not implemented yet), f32 may overshoot 255
    let image_warped = image_warped
        .cast::<u8>()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;

    Ok(image_warped.to_pyimage())
}

#[pyfunction]
pub fn warp_perspective(
    image: PyImage,
    m: [f32; 9],
    new_size: (usize, usize),
    interpolation: &str,
) -> PyResult<PyImage> {
    let image: Image<u8, 3> = Image::from_pyimage(image)
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

    let image = image
        .cast::<f32>()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;

    let mut image_warped = Image::from_size_val(new_size, 0f32)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;

    warp::warp_perspective(&image, &mut image_warped, &m, interpolation)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;

    let image_warped = image_warped
        .cast::<u8>()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;

    Ok(image_warped.to_pyimage())
}
