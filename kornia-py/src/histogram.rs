use pyo3::prelude::*;

use crate::image::{FromPyImage, PyImage};
use kornia_rs::image::Image;

/// Compute the pixel-wise histogram of an image.
/// --
///
/// This function computes the histogram of an image with a given number of bins.
///
/// # Arguments
///
/// * `image` - The input image to compute the histogram with shape (H, W, 1) and dtype uint8.
/// * `num_bins` - The number of bins to use for the histogram.
///
/// # Returns
///
/// A vector of size `num_bins` containing the histogram.
#[pyfunction]
pub fn compute_histogram(image: PyImage, num_bins: usize) -> PyResult<Vec<usize>> {
    let image = Image::from_pyimage(image)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;

    let histogram = kornia_rs::histogram::compute_histogram(&image, num_bins)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;

    Ok(histogram)
}
