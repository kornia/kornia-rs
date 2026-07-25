//! SIFT detector/descriptor (`kornia_rs.imgproc.Sift`).
//!
//! # Residency
//!
//! SIFT has no CPU implementation in this workspace, so unlike the `color`,
//! `resize` and `warp` ops there is nothing to fall through to: a host `Image`
//! or a numpy array is a typed error naming the fix. Uploading on the caller's
//! behalf would hide a full-frame H2D copy inside what looks like a detector
//! call, which is the mirror of the implicit download
//! [`crate::dispatch::no_gpu_kernel_if_device`] refuses in the other direction.

use pyo3::prelude::*;

#[cfg(feature = "cuda")]
use crate::image::PyImageApi;
#[cfg(feature = "cuda")]
use numpy::PyArray2;

/// A reusable SIFT detector, shaped like `cv2.SIFT`.
///
/// ```python
/// sift = kornia_rs.imgproc.Sift(contrast_threshold=0.04)
/// kp, desc = sift.detect_and_compute(device_image)
/// ```
///
/// The instance owns the pipeline's scratch — roughly a dozen full-resolution
/// planes — and rebuilds it only when the image size or a parameter changes, so
/// a streaming caller allocates once rather than per frame. Keep the instance
/// alive across frames; constructing one per call gives up that reuse.
///
/// Parameters match `cv2.SIFT_create`, plus two knobs OpenCV hardcodes:
/// `upsample` selects `first_octave`, and `max_octaves` caps the pyramid.
///
/// `unsendable`: the plan holds CUDA stream and buffer handles that are not
/// `Sync`, so an instance stays on the thread that built it.
#[cfg(feature = "cuda")]
#[pyclass(unsendable)]
pub struct Sift {
    n_features: usize,
    n_octave_layers: usize,
    contrast_threshold: f64,
    edge_threshold: f64,
    sigma: f64,
    max_keypoints: usize,
    upsample: bool,
    max_octaves: usize,
    fast_descriptor: bool,
    plan: crate::cuda_ext::cuda_sift::PlanSlot,
    store: crate::cuda_ext::cuda_sift::MatchStore,
}

/// Borrow a device `Image` from a Python object, or explain what is wrong.
#[cfg(feature = "cuda")]
fn device_image<'py>(obj: &Bound<'py, PyAny>, who: &str) -> PyResult<Bound<'py, PyImageApi>> {
    let api = obj.cast::<PyImageApi>().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "{who}: expected a device Image; SIFT has no CPU path, so convert with \
             Image.from_numpy(a).to_cuda(stream) first"
        ))
    })?;
    if !api.borrow().is_device() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{who}: this Image lives on the host and SIFT has no CPU kernel; move \
             it with .to_cuda(stream) first"
        )));
    }
    Ok(api.clone())
}

#[cfg(feature = "cuda")]
#[pymethods]
impl Sift {
    #[new]
    #[pyo3(signature = (n_features=0, n_octave_layers=3, contrast_threshold=0.04,
                        edge_threshold=10.0, sigma=1.6, max_keypoints=8192,
                        upsample=true, max_octaves=0, fast_descriptor=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_features: usize,
        n_octave_layers: usize,
        contrast_threshold: f64,
        edge_threshold: f64,
        sigma: f64,
        max_keypoints: usize,
        upsample: bool,
        max_octaves: usize,
        fast_descriptor: bool,
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
        // A non-positive sigma reaches an `assert!` deep in the Gaussian kernel
        // builder, which surfaces as a PanicException rather than a ValueError.
        if !(sigma.is_finite() && sigma > 0.0) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Sift: sigma must be finite and positive, got {sigma}"
            )));
        }
        Ok(Self {
            n_features,
            n_octave_layers,
            contrast_threshold,
            edge_threshold,
            sigma,
            max_keypoints,
            upsample,
            max_octaves,
            fast_descriptor,
            plan: None,
            store: Default::default(),
        })
    }

    /// Detect keypoints and compute their descriptors.
    ///
    /// `image` must be an f32 single-channel **device** `Image` with values in
    /// 0..255 — the reference's own internal representation. Normalising to
    /// 0..1 changes the contrast threshold's meaning and will silently return
    /// far fewer keypoints.
    ///
    /// Returns `(keypoints, descriptors)`: `(N, 6)` of
    /// `x, y, size, angle, response, octave`, and `(N, 128)`.
    fn detect_and_compute<'py>(
        &mut self,
        py: Python<'py>,
        image: &Bound<'py, PyAny>,
    ) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>)> {
        let api = device_image(image, "Sift.detect_and_compute")?;
        crate::cuda_ext::cuda_sift::sift_cuda_with_plan(
            py,
            &api.borrow(),
            &mut self.plan,
            self.n_features,
            self.n_octave_layers,
            self.fast_descriptor,
            self.contrast_threshold,
            self.edge_threshold,
            self.sigma,
            self.max_keypoints,
            self.upsample,
            self.max_octaves,
        )
    }

    /// Detect in both images and match, without the descriptors ever leaving
    /// the device.
    ///
    /// This is the reason to prefer it over calling `detect_and_compute` twice
    /// and matching on the host: the descriptors are the bulk of the data (128
    /// floats per keypoint, megabytes a frame) and here they are produced,
    /// stored and compared entirely in device memory. Only the keypoints and the
    /// surviving pairs cross the bus.
    ///
    /// `ratio` is Lowe's ratio; `>= 1.0` disables it. `cross_check` requires
    /// each pair to be a mutual nearest neighbour.
    ///
    /// Returns `(keypoints_a, keypoints_b, matches)`, where `matches` is
    /// `(M, 2)` of indices into the two keypoint arrays.
    #[pyo3(signature = (image_a, image_b, ratio=0.8, cross_check=true))]
    fn r#match<'py>(
        &mut self,
        py: Python<'py>,
        image_a: &Bound<'py, PyAny>,
        image_b: &Bound<'py, PyAny>,
        ratio: f32,
        cross_check: bool,
    ) -> PyResult<(
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray2<i32>>,
    )> {
        let a = device_image(image_a, "Sift.match")?;
        let b = device_image(image_b, "Sift.match")?;
        crate::cuda_ext::cuda_sift::sift_match(
            py,
            &a.borrow(),
            &b.borrow(),
            &mut self.plan,
            &mut self.store,
            ratio,
            cross_check,
            self.n_features,
            self.n_octave_layers,
            self.fast_descriptor,
            self.contrast_threshold,
            self.edge_threshold,
            self.sigma,
            self.max_keypoints,
            self.upsample,
            self.max_octaves,
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "Sift(n_features={}, n_octave_layers={}, contrast_threshold={}, \
             edge_threshold={}, sigma={}, max_keypoints={}, upsample={}, \
             max_octaves={}, fast_descriptor={})",
            self.n_features,
            self.n_octave_layers,
            self.contrast_threshold,
            self.edge_threshold,
            self.sigma,
            self.max_keypoints,
            // Python spelling: a bare Rust `true` in a repr is a NameError if
            // anyone pastes it back.
            if self.upsample { "True" } else { "False" },
            self.max_octaves,
            if self.fast_descriptor {
                "True"
            } else {
                "False"
            },
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
    #[pyo3(signature = (n_features=0, n_octave_layers=3, contrast_threshold=0.04,
                        edge_threshold=10.0, sigma=1.6, max_keypoints=8192,
                        upsample=true, max_octaves=0, fast_descriptor=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_features: usize,
        n_octave_layers: usize,
        contrast_threshold: f64,
        edge_threshold: f64,
        sigma: f64,
        max_keypoints: usize,
        upsample: bool,
        max_octaves: usize,
        fast_descriptor: bool,
    ) -> PyResult<Self> {
        let _ = (
            n_features,
            n_octave_layers,
            contrast_threshold,
            edge_threshold,
            sigma,
            max_keypoints,
            upsample,
            max_octaves,
            fast_descriptor,
        );
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Sift: CUDA support is not compiled in",
        ))
    }
}
