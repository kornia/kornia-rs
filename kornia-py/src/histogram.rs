use pyo3::prelude::*;

use crate::image::{alloc_output_pyarray, numpy_as_image, to_pyerr, PyImage};
use kornia_imgproc as imgproc;

#[pyfunction]
pub fn compute_histogram(
    py: Python<'_>,
    image: &Bound<'_, PyAny>,
    num_bins: usize,
) -> PyResult<Vec<usize>> {
    let mut histogram = vec![0; num_bins];

    #[cfg(feature = "cuda")]
    if let Ok(api) = image.cast::<crate::image::PyImageApi>() {
        let img = api.borrow();
        if img.is_device() {
            crate::cuda_ext::histogram::compute_histogram(py, &img, &mut histogram, num_bins)?;
            return Ok(histogram);
        }
    }

    let image: PyImage = image.extract()?;
    let image = unsafe { numpy_as_image::<1>(py, &image)? };
    py.detach(|| imgproc::histogram::compute_histogram(&image, &mut histogram, num_bins))
        .map_err(to_pyerr)?;
    Ok(histogram)
}

/// Histogram-equalize an 8-bit single-channel image — byte-for-byte with
/// `cv2.equalizeHist`.
///
/// Residency-dispatched: a u8 device `Image` runs the CUDA
/// histogram → LUT → apply chain (byte-identical to the CPU path); a numpy
/// u8 array of shape (H, W, 1) runs the CPU path.
#[pyfunction]
pub fn equalize_hist(py: Python<'_>, image: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    #[cfg(feature = "cuda")]
    if let Ok(api) = image.cast::<crate::image::PyImageApi>() {
        let img = api.borrow();
        if img.is_device() {
            return crate::cuda_ext::histogram::equalize_hist(py, &img)?.into_py(py);
        }
    }

    crate::dispatch::cpu_op(py, image, move |py, arr: Py<numpy::PyArray3<u8>>| {
        let src = unsafe { numpy_as_image::<1>(py, &arr)? };
        let (mut dst, out) = unsafe { alloc_output_pyarray::<1>(py, src.size())? };
        py.detach(|| imgproc::histogram::equalize_hist(&src, &mut dst))
            .map_err(to_pyerr)?;
        Ok(out)
    })
}
