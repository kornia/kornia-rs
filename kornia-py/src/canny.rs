use pyo3::prelude::*;

use crate::dispatch::cpu_op;
use crate::image::{alloc_output_pyarray, numpy_as_image, to_pyerr};
use kornia_imgproc as imgproc;

/// Canny edge detection — byte-for-byte with
/// `cv2.Canny(src, low_threshold, high_threshold, L2gradient=l2_gradient)`
/// (aperture size 3). Output is 255 on edges, 0 elsewhere.
///
/// Residency-dispatched: a u8 single-channel device `Image` runs the CUDA
/// pipeline (byte-identical to the CPU path); a numpy u8 array of shape
/// (H, W, 1) runs the CPU path.
#[pyfunction]
#[pyo3(signature = (image, low_threshold=50.0, high_threshold=150.0, l2_gradient=false))]
pub fn canny(
    py: Python<'_>,
    image: &Bound<'_, PyAny>,
    low_threshold: f64,
    high_threshold: f64,
    l2_gradient: bool,
) -> PyResult<Py<PyAny>> {
    #[cfg(feature = "cuda")]
    if let Ok(api) = image.cast::<crate::image::PyImageApi>() {
        let img = api.borrow();
        if img.is_device() {
            return crate::cuda_ext::canny_dev::canny(
                py,
                &img,
                low_threshold,
                high_threshold,
                l2_gradient,
            )?
            .into_py(py);
        }
    }
    cpu_op(py, image, move |py, arr: Py<numpy::PyArray3<u8>>| {
        let src = unsafe { numpy_as_image::<1>(py, &arr)? };
        let (mut dst, out) = unsafe { alloc_output_pyarray::<1>(py, src.size())? };
        py.detach(|| {
            imgproc::canny::canny(&src, &mut dst, low_threshold, high_threshold, l2_gradient)
        })
        .map_err(to_pyerr)?;
        Ok(out)
    })
}
