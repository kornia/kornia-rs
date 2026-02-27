use pyo3::prelude::*;

use crate::image::{FromPyImage, PyImage, ToPyImage};
use kornia_image::{allocator::CpuAllocator, Image, ImageSize};
use kornia_imgproc::{interpolation::InterpolationMode, resize::resize_fast_rgb};

/// Resize an image to a new size using the specified interpolation mode.
///
/// This function resizes an RGB image to the specified dimensions using
/// one of the supported interpolation methods.
///
/// Parameters
/// ----------
/// image : PyImage
///     Input RGB image as a PyImage object (HxWx3 numpy array).
/// new_size : Tuple[int, int]
///     Target size as (height, width).
/// interpolation : str
///     Interpolation method. Supported values:
///     - "nearest": Nearest neighbor interpolation (fastest, lowest quality)
///     - "bilinear": Bilinear interpolation (default, good balance)
///
/// Returns
/// -------
/// PyImage
///     Resized RGB image as a PyImage object.
///
/// Raises
/// ------
/// PyValueError
///     If interpolation mode is not supported.
///
/// Examples
/// --------
/// >>> import kornia_rs
/// >>> import numpy as np
/// >>>
/// >>> # Create a small test image
/// >>> img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
///
/// >>> # Resize to 50x50 using bilinear interpolation
/// >>> resized = kornia_rs.resize(img, (50, 50), interpolation="bilinear")
/// >>> print(resized.shape)
/// (50, 50, 3)
///
/// >>> # Resize using nearest neighbor (faster)
/// >>> resized = kornia_rs.resize(img, (200, 200), interpolation="nearest")
/// >>> print(resized.shape)
/// (200, 200, 3)
///
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

    let mut image_resized = Image::from_size_val(new_size, 0u8, CpuAllocator)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;

    resize_fast_rgb(&image, &mut image_resized, interpolation)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;

    let pyimage_resized = image_resized.to_pyimage().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyException, _>(format!("failed to convert image: {}", e))
    })?;

    Ok(pyimage_resized)
}
