use pyo3::prelude::*;

use crate::image::{alloc_output_pyarray_i32, numpy_as_image, to_pyerr};
use kornia_imgproc::connected_components::{connected_components, Connectivity};

fn parse_conn(connectivity: u8) -> PyResult<Connectivity> {
    match connectivity {
        4 => Ok(Connectivity::Four),
        8 => Ok(Connectivity::Eight),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "connectivity must be 4 or 8; got {other}"
        ))),
    }
}

/// Connected-component labeling — label-exact with OpenCV's SAUF
/// algorithm (`cv2.connectedComponentsWithAlgorithm(img, connectivity,
/// cv2.CV_32S, cv2.CCL_WU)`): background 0, components numbered 1..N in
/// raster order of first appearance. Returns `(n, labels)` with `n`
/// including the background label, like cv2.
///
/// Residency-dispatched: a u8 single-channel device `Image` runs the CUDA
/// label-equivalence kernels and returns an int32 device `Image`
/// (label-identical to the CPU path); a numpy u8 array of shape (H, W, 1)
/// runs the CPU path and returns an int32 ndarray.
#[pyfunction(name = "connected_components")]
#[pyo3(signature = (image, connectivity=8))]
pub fn connected_components_op(
    py: Python<'_>,
    image: &Bound<'_, PyAny>,
    connectivity: u8,
) -> PyResult<(i32, Py<PyAny>)> {
    let conn = parse_conn(connectivity)?;

    #[cfg(feature = "cuda")]
    if let Ok(api) = image.cast::<crate::image::PyImageApi>() {
        let img = api.borrow();
        if img.is_device() {
            let (n, out) = crate::cuda_ext::ccl_dev::connected_components(py, &img, conn)?;
            return Ok((n, out));
        }
    }

    let arr: Py<numpy::PyArray3<u8>> = if let Ok(api) = image.cast::<crate::image::PyImageApi>() {
        // Host Image: operate on its zero-copy numpy view.
        api.call_method0("numpy")?.extract()?
    } else {
        image.extract()?
    };
    let src = unsafe { numpy_as_image::<1>(py, &arr)? };
    let (mut dst, out) = unsafe { alloc_output_pyarray_i32::<1>(py, src.size())? };
    let n = py
        .detach(|| connected_components(&src, &mut dst, conn))
        .map_err(to_pyerr)?;
    Ok((n, out.into_any()))
}
