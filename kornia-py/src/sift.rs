//! SIFT detector/descriptor (`kornia_rs.imgproc.sift`).
//!
//! # Residency
//!
//! [`sift`] dispatches on where the image lives, like the `color`, `resize` and
//! `warp` ops. There is one asymmetry: SIFT has no CPU implementation in this
//! workspace, so a host `Image` or a numpy array is a typed error rather than a
//! fall-through. Uploading on the caller's behalf would hide a full-frame H2D
//! copy inside what looks like a detector call, which is exactly the kind of
//! implicit transfer [`crate::dispatch::no_gpu_kernel_if_device`] refuses in the
//! other direction.

use pyo3::prelude::*;

#[cfg(feature = "cuda")]
use crate::image::PyImageApi;
#[cfg(feature = "cuda")]
use numpy::PyArray2;

/// Detect SIFT keypoints and compute 128-D descriptors.
///
/// `image` must be an f32 single-channel **device** `Image` with values in
/// 0..255 — the reference's own internal representation. Normalising to 0..1
/// changes the contrast threshold's meaning and will silently return far fewer
/// keypoints. Move a host image across with `.to_cuda(stream)` first.
///
/// `upsample=True` reproduces `first_octave = -1`, the OpenCV / COLMAP / VLFeat
/// default: it doubles the base image, roughly 3.5x'ing the correct-match count
/// at about 2.5x the cost. `max_octaves=0` means unlimited.
///
/// Returns `(keypoints, descriptors)`: `(N, 6)` of
/// `x, y, size, angle, response, octave`, and `(N, 128)`.
#[cfg(feature = "cuda")]
#[pyfunction]
#[pyo3(signature = (image, n_octave_layers=3, contrast_threshold=0.04, edge_threshold=10.0,
                    sigma=1.6, max_keypoints=8192, upsample=true, max_octaves=0))]
#[allow(clippy::too_many_arguments)]
pub fn sift<'py>(
    py: Python<'py>,
    image: &Bound<'py, PyAny>,
    n_octave_layers: usize,
    contrast_threshold: f64,
    edge_threshold: f64,
    sigma: f64,
    max_keypoints: usize,
    upsample: bool,
    max_octaves: usize,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>)> {
    let api = image.cast::<PyImageApi>().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err(
            "sift: expected a device Image; SIFT has no CPU path, so convert with \
             Image.from_numpy(a).to_cuda(stream) first",
        )
    })?;
    let img = api.borrow();
    if !img.is_device() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "sift: this Image lives on the host and SIFT has no CPU kernel; move it \
             with .to_cuda(stream) first",
        ));
    }
    crate::cuda_ext::cuda_sift::sift_cuda(
        py,
        &img,
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
pub fn sift(
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
        "sift: CUDA support is not compiled in",
    ))
}

/// A reusable SIFT detector, shaped like `cv2.SIFT`.
///
/// ```python
/// sift = kornia_rs.imgproc.Sift(contrast_threshold=0.04)
/// kp, desc = sift.detect_and_compute(device_image)
/// ```
///
/// Prefer this over the free [`sift`] function when processing more than one
/// frame. The instance owns the pipeline's scratch — roughly a dozen
/// full-resolution planes — and rebuilds it only when the image size changes, so
/// a streaming caller allocates once instead of per frame.
///
/// Parameters match `cv2.SIFT_create`, minus `nfeatures` (there is no
/// `retainBest` on device yet) and plus two knobs OpenCV hardcodes: `upsample`
/// selects `first_octave`, and `max_octaves` caps the pyramid.
///
/// `unsendable`: the plan holds CUDA stream and buffer handles that are not
/// `Sync`, so an instance stays on the thread that built it.
#[cfg(feature = "cuda")]
#[pyclass(unsendable)]
pub struct Sift {
    n_octave_layers: usize,
    contrast_threshold: f64,
    edge_threshold: f64,
    sigma: f64,
    max_keypoints: usize,
    upsample: bool,
    max_octaves: usize,
    plan: crate::cuda_ext::cuda_sift::PlanSlot,
}

#[cfg(feature = "cuda")]
#[pymethods]
impl Sift {
    #[new]
    #[pyo3(signature = (n_octave_layers=3, contrast_threshold=0.04, edge_threshold=10.0,
                        sigma=1.6, max_keypoints=8192, upsample=true, max_octaves=0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_octave_layers: usize,
        contrast_threshold: f64,
        edge_threshold: f64,
        sigma: f64,
        max_keypoints: usize,
        upsample: bool,
        max_octaves: usize,
    ) -> PyResult<Self> {
        if n_octave_layers == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Sift: n_octave_layers must be non-zero",
            ));
        }
        if max_keypoints == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Sift: max_keypoints must be non-zero",
            ));
        }
        Ok(Self {
            n_octave_layers,
            contrast_threshold,
            edge_threshold,
            sigma,
            max_keypoints,
            upsample,
            max_octaves,
            plan: None,
        })
    }

    /// Detect keypoints and compute their descriptors.
    ///
    /// `image` must be an f32 single-channel **device** `Image` in 0..255.
    /// Returns `(keypoints, descriptors)`: `(N, 6)` of
    /// `x, y, size, angle, response, octave`, and `(N, 128)`.
    fn detect_and_compute<'py>(
        &mut self,
        py: Python<'py>,
        image: &Bound<'py, PyAny>,
    ) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>)> {
        let api = image.cast::<PyImageApi>().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err(
                "Sift.detect_and_compute: expected a device Image; SIFT has no CPU \
                 path, so convert with Image.from_numpy(a).to_cuda(stream) first",
            )
        })?;
        let img = api.borrow();
        if !img.is_device() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Sift.detect_and_compute: this Image lives on the host and SIFT has \
                 no CPU kernel; move it with .to_cuda(stream) first",
            ));
        }
        crate::cuda_ext::cuda_sift::sift_cuda_with_plan(
            py,
            &img,
            &mut self.plan,
            self.n_octave_layers,
            self.contrast_threshold,
            self.edge_threshold,
            self.sigma,
            self.max_keypoints,
            self.upsample,
            self.max_octaves,
        )
    }

    /// Alias of [`Sift::detect_and_compute`] spelled the way `cv2` spells it, so
    /// existing OpenCV code ports with only the constructor changed.
    #[pyo3(name = "detectAndCompute")]
    #[pyo3(signature = (image, mask=None))]
    fn detect_and_compute_cv2<'py>(
        &mut self,
        py: Python<'py>,
        image: &Bound<'py, PyAny>,
        mask: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>)> {
        if mask.is_some_and(|m| !m.is_none()) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Sift.detectAndCompute: a mask is not supported; the argument exists \
                 only so cv2 call sites port unchanged",
            ));
        }
        self.detect_and_compute(py, image)
    }

    fn __repr__(&self) -> String {
        format!(
            "Sift(n_octave_layers={}, contrast_threshold={}, edge_threshold={}, \
             sigma={}, max_keypoints={}, upsample={}, max_octaves={})",
            self.n_octave_layers,
            self.contrast_threshold,
            self.edge_threshold,
            self.sigma,
            self.max_keypoints,
            // Python spelling: a bare Rust `true` in a repr is a NameError if
            // anyone pastes it back.
            if self.upsample { "True" } else { "False" },
            self.max_octaves,
        )
    }
}

/// Stub for builds without the `cuda` feature, so the symbol always exists.
#[cfg(not(feature = "cuda"))]
#[pyclass]
pub struct Sift;

#[cfg(not(feature = "cuda"))]
#[pymethods]
impl Sift {
    #[new]
    #[pyo3(signature = (n_octave_layers=3, contrast_threshold=0.04, edge_threshold=10.0,
                        sigma=1.6, max_keypoints=8192, upsample=true, max_octaves=0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_octave_layers: usize,
        contrast_threshold: f64,
        edge_threshold: f64,
        sigma: f64,
        max_keypoints: usize,
        upsample: bool,
        max_octaves: usize,
    ) -> PyResult<Self> {
        let _ = (
            n_octave_layers,
            contrast_threshold,
            edge_threshold,
            sigma,
            max_keypoints,
            upsample,
            max_octaves,
        );
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Sift: CUDA support is not compiled in",
        ))
    }
}

/// Device-only entry point, kept as an explicit alias of [`sift`].
///
/// Prefer `sift`; this name stays so callers that want the residency
/// requirement spelled out at the call site keep working.
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
