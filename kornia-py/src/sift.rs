//! CUDA SIFT detector/descriptor (`kornia_rs.imgproc.sift_cuda`).

use pyo3::prelude::*;

#[cfg(feature = "cuda")]
use crate::image::PyImageApi;
#[cfg(feature = "cuda")]
use numpy::PyArray2;

/// Detect SIFT keypoints and compute 128-D descriptors on the GPU.
///
/// `image` must be an f32 single-channel device `Image` with values in 0..255 —
/// the reference's own internal representation. Normalising to 0..1 changes the
/// contrast threshold's meaning and will silently return far fewer keypoints.
///
/// `upsample=True` reproduces `first_octave = -1`, the OpenCV / COLMAP / VLFeat
/// default: it doubles the base image, roughly 2.4x'ing the correct-match count
/// at about 3.6x the cost. `max_octaves=0` means unlimited.
///
/// Returns `(keypoints, descriptors)`: `(N, 6)` of
/// `x, y, size, angle, response, octave`, and `(N, 128)`.
#[cfg(feature = "cuda")]
#[pyfunction]
#[pyo3(signature = (image, n_octave_layers=3, contrast_threshold=0.04, edge_threshold=10.0,
                    sigma=1.6, max_keypoints=8192, upsample=true, max_octaves=0))]
#[allow(clippy::too_many_arguments)]
pub fn sift_cuda<'py>(
    py: Python<'py>,
    image: PyRef<'py, PyImageApi>,
    n_octave_layers: usize,
    contrast_threshold: f64,
    edge_threshold: f64,
    sigma: f64,
    max_keypoints: usize,
    upsample: bool,
    max_octaves: usize,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>)> {
    crate::cuda_ext::cuda_sift::sift_cuda(
        py,
        &image,
        n_octave_layers,
        contrast_threshold,
        edge_threshold,
        sigma,
        max_keypoints,
        upsample,
        max_octaves,
    )
}

/// Stub for builds without the `cuda` feature, so the symbol always exists.
#[cfg(not(feature = "cuda"))]
#[pyfunction]
#[pyo3(signature = (image, n_octave_layers=3, contrast_threshold=0.04, edge_threshold=10.0,
                    sigma=1.6, max_keypoints=8192, upsample=true, max_octaves=0))]
#[allow(clippy::too_many_arguments)]
pub fn sift_cuda(
    image: Py<PyAny>,
    n_octave_layers: usize,
    contrast_threshold: f64,
    edge_threshold: f64,
    sigma: f64,
    max_keypoints: usize,
    upsample: bool,
    max_octaves: usize,
) -> PyResult<()> {
    let _ = (
        image,
        n_octave_layers,
        contrast_threshold,
        edge_threshold,
        sigma,
        max_keypoints,
        upsample,
        max_octaves,
    );
    Err(pyo3::exceptions::PyRuntimeError::new_err(
        "sift_cuda: CUDA support is not compiled in",
    ))
}
