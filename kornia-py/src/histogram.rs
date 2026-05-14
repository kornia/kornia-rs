use pyo3::prelude::*;

use crate::image::{numpy_as_image, to_pyerr, PyImage};
use kornia_imgproc as imgproc;

#[pyfunction]
pub fn compute_histogram(py: Python<'_>, image: PyImage, num_bins: usize) -> PyResult<Vec<usize>> {
    let image = unsafe { numpy_as_image::<1>(py, &image)? };
    let mut histogram = vec![0; num_bins];
    py.detach(|| imgproc::histogram::compute_histogram(&image, &mut histogram, num_bins))
        .map_err(to_pyerr)?;
    Ok(histogram)
}
