//! SIFT detector/descriptor (`kornia_rs.imgproc.Sift`).
//!
//! # Residency
//!
//! Dispatches like the other unified-`Image` ops: a **device** `Image` runs the
//! CUDA pipeline, a host `Image` or a numpy array runs the NEON one. Both
//! backends share their host-side numerics and their input validation, so the
//! result and the set of rejected inputs do not depend on where the image lives
//! — only the speed does.
//!
//! Residency does decide the *container* the descriptors come back in: a device
//! image yields a device `Tensor`, a host image a numpy array. That is the point
//! — a 2515x128 block costs more to move across the bus than the detection that
//! produced it, so the device path never moves it. `Sift.match` takes those
//! containers back and dispatches on them, so a device-to-device match never
//! touches host memory.
//!
//! Keypoints always come back on the host, as a list of `SiftKeypoint` — the
//! shape `cv2.SIFT.detectAndCompute` returns. They are two orders of magnitude
//! smaller than the descriptors and every consumer of them (drawing, pose,
//! bookkeeping) is host code.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use kornia_imgproc::features::{
    sift_detect_and_compute, sift_match_descriptors, FirstOctave as CpuFirstOctave, SiftConfig,
    SiftKeypoint as CoreKeypoint, SiftWorkspace, DESCR_LEN,
};
use numpy::{PyArray2, PyArray3, PyArrayMethods, PyUntypedArrayMethods};

#[cfg(feature = "cuda")]
use crate::image::PyImageApi;

/// `(keypoints, descriptors)`.
///
/// `keypoints` is a Python list of `SiftKeypoint`, shaped like what
/// `cv2.SIFT.detectAndCompute` returns. `descriptors` is a numpy `(N, 128)` on
/// the host path and a device `Tensor` on the CUDA one, so it is typed as
/// `Py<PyAny>`; the residency of the image decides which. Both backends and both
/// the public method and its two private halves return this shape, so it is
/// named once rather than spelled out at each of them.
pub(crate) type DetectOut = PyResult<(Vec<PyKeypoint>, Py<PyAny>)>;

// ── Keypoints ────────────────────────────────────────────────────────────────

/// A single detected keypoint. `detect_and_compute` returns a list of these,
/// matching the shape of `cv2.SIFT.detectAndCompute`.
///
/// `octave`, `layer` and `xi` are decoded from OpenCV's packed `KeyPoint.octave`
/// field; `packed_octave` is that field verbatim, for comparing against `cv2`.
///
/// Every field is a plain attribute read, but the list itself is one Python
/// object per keypoint — a few thousand of them on a 752x480 frame. Bulk numeric
/// work wants a column, so build it once
/// (`np.fromiter((k.x for k in kp), np.float32, len(kp))`) rather than
/// re-walking the list per operation.
#[pyclass(
    name = "SiftKeypoint",
    module = "kornia_rs.imgproc",
    frozen,
    get_all,
    skip_from_py_object
)]
#[derive(Clone, Copy)]
pub struct PyKeypoint {
    /// Column coordinate, in pixels of the **input** image.
    pub x: f32,
    /// Row coordinate, in pixels of the input image.
    pub y: f32,
    /// Diameter of the meaningful neighbourhood.
    pub size: f32,
    /// Dominant gradient orientation, in degrees clockwise from +x.
    pub angle: f32,
    /// Contrast at the interpolated extremum; the `n_features` ranking key.
    pub response: f32,
    /// Signed octave index. `-1` when `upsample=True`, which is OpenCV's own
    /// `firstOctave = -1`.
    pub octave: i32,
    /// Layer within the octave.
    pub layer: i32,
    /// Sub-layer offset in `[-0.5, 0.5)`, quantised to 1/255 by the packing.
    pub xi: f32,
    /// The raw packed field, as `cv2.KeyPoint.octave` reports it.
    pub packed_octave: i32,
}

#[pymethods]
impl PyKeypoint {
    fn __repr__(&self) -> String {
        format!(
            "SiftKeypoint(x={:.3}, y={:.3}, size={:.3}, angle={:.2}, \
             response={:.6}, octave={}, layer={}, xi={:.4})",
            self.x, self.y, self.size, self.angle, self.response, self.octave, self.layer, self.xi
        )
    }
}

/// Decode OpenCV's packed octave field into `(octave, layer, xi)`.
///
/// Mirrors `unpackOctave` in `sift.dispatch.cpp`: the low byte is a *signed*
/// octave, so `firstOctave = -1` arrives as `255` and must be sign-extended, not
/// read as 255.
fn unpack_octave(packed: i32) -> (i32, i32, f32) {
    let lo = packed & 255;
    let octave = if lo < 128 { lo } else { lo | -128 };
    let layer = (packed >> 8) & 255;
    let xi = ((packed >> 16) & 255) as f32 / 255.0 - 0.5;
    (octave, layer, xi)
}

/// Transpose a core keypoint list into Python objects.
///
/// The packed octave is decoded here rather than exposed raw: OpenCV stores
/// three values in `KeyPoint.octave` — the signed octave index in the low byte,
/// the layer in the next, and a quantised sub-layer offset above that — and
/// making every caller remember that is how the field ends up misread.
/// `packed_octave` keeps the original for anyone comparing against `cv2`.
pub(crate) fn keypoints_to_list(kps: &[CoreKeypoint]) -> Vec<PyKeypoint> {
    kps.iter()
        .map(|k| {
            let (octave, layer, xi) = unpack_octave(k.octave);
            PyKeypoint {
                x: k.x,
                y: k.y,
                size: k.size,
                angle: k.angle,
                response: k.response,
                octave,
                layer,
                xi,
                packed_octave: k.octave,
            }
        })
        .collect()
}

// ── Detector ─────────────────────────────────────────────────────────────────

/// A reusable SIFT detector, shaped like `cv2.SIFT`.
///
/// ```python
/// sift = kornia_rs.imgproc.Sift(contrast_threshold=0.04)
/// kp, desc = sift.detect_and_compute(device_image)   # CUDA, desc is a Tensor
/// kp, desc = sift.detect_and_compute(numpy_array)    # NEON, desc is ndarray
/// kp[0].x, kp[0].octave, kp[0].response
/// pairs = sift.match(desc_a, desc_b)
/// ```
///
/// The instance owns both backends' scratch — the CUDA plan and the CPU
/// workspace are each around twenty full-resolution planes — and rebuilds them
/// only when the image size or a parameter changes. Keep the instance alive
/// across frames; constructing one per call gives up that reuse.
///
/// Parameters match `cv2.SIFT_create`, plus three knobs OpenCV hardcodes:
/// `upsample` selects `first_octave`, `max_octaves` caps the pyramid, and
/// `fast_descriptor` trades bit-exactness for speed on the GPU.
///
/// `max_keypoints` is **not** one of those: it sizes the CUDA plan's device
/// buffers, and the device path silently truncates to it. The host path has no
/// such ceiling, so it is the one parameter that makes the two backends return
/// different results — leave it at the default unless a frame genuinely exceeds
/// it, and raise it rather than relying on the truncation. Use `n_features` for
/// an actual keypoint budget: that one is the reference's `retainBest` and both
/// backends apply it identically.
///
/// `unsendable`: the CUDA plan holds stream and buffer handles that are not
/// `Sync`, so an instance stays on the thread that built it.
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
    ws: SiftWorkspace,
    #[cfg(feature = "cuda")]
    plan: crate::cuda_ext::cuda_sift::PlanSlot,
    #[cfg(feature = "cuda")]
    store: crate::cuda_ext::cuda_sift::MatchStore,
}

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
        if max_keypoints == 0 {
            return Err(PyValueError::new_err(
                "Sift: max_keypoints must be non-zero",
            ));
        }
        // The same validator both backends use, so a bad configuration is
        // rejected at construction rather than at the first frame.
        let cfg = SiftConfig {
            n_features,
            n_octave_layers,
            contrast_threshold,
            edge_threshold,
            sigma,
        };
        let cap = if max_octaves == 0 {
            usize::MAX
        } else {
            max_octaves
        };
        cfg.validate(cap)
            .map_err(|e| PyValueError::new_err(format!("Sift: {e}")))?;

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
            ws: SiftWorkspace::new(),
            #[cfg(feature = "cuda")]
            plan: None,
            #[cfg(feature = "cuda")]
            store: Default::default(),
        })
    }

    /// Detect keypoints and compute their descriptors.
    ///
    /// `image` may be a device `Image` (CUDA), or a host `Image` or numpy array
    /// (NEON). It must be single-channel `f32` with values in **0..255** — the
    /// reference's own internal representation. Normalising to 0..1 changes what
    /// the contrast threshold means and will silently return far fewer
    /// keypoints.
    ///
    /// Returns `(keypoints, descriptors)`. `keypoints` is a list of
    /// `SiftKeypoint`, always on the host, shaped like what
    /// `cv2.SIFT.detectAndCompute` returns. `descriptors` follows the input's
    /// residency: a device `Tensor` of shape `(N, 128)` for a device
    /// image, a numpy `(N, 128)` for a host one. Feed either straight back to
    /// `match`.
    ///
    /// The device descriptors are a fresh allocation, not a view into the plan:
    /// the plan has one output buffer and the next call overwrites it, so a view
    /// would silently change under a caller holding two frames.
    fn detect_and_compute(&mut self, py: Python<'_>, image: &Bound<'_, PyAny>) -> DetectOut {
        #[cfg(feature = "cuda")]
        if let Ok(api) = image.cast::<PyImageApi>() {
            let img = api.borrow();
            if img.is_device() {
                // Read the parameters before borrowing the plan mutably.
                let params = self.detect_params();
                return crate::cuda_ext::cuda_sift::sift_cuda_with_plan(
                    py,
                    &img,
                    &mut self.plan,
                    params,
                );
            }
        }
        self.detect_host(py, image)
    }

    /// Match two descriptor blocks.
    ///
    /// Takes what `detect_and_compute` returned, so detection and matching are
    /// separable: detect once, match against several frames, or match
    /// descriptors that came from somewhere else entirely.
    ///
    /// Dispatches on the descriptors, not on an image. Two device `Tensor`s
    /// match on device and never cross the bus; two numpy arrays run the NEON
    /// matcher. Mixing the two is refused rather than silently transferred —
    /// the transfer is the expensive part and hiding it is how a frame budget
    /// disappears.
    ///
    /// `ratio` is Lowe's ratio; `>= 1.0` disables it. `cross_check` requires
    /// each pair to be a mutual nearest neighbour. Returns an `(M, 2)` int32
    /// array of indices into the two keypoint lists.
    #[pyo3(signature = (descriptors_a, descriptors_b, ratio=0.8, cross_check=true))]
    fn r#match<'py>(
        &mut self,
        py: Python<'py>,
        descriptors_a: &Bound<'py, PyAny>,
        descriptors_b: &Bound<'py, PyAny>,
        ratio: f32,
        cross_check: bool,
    ) -> PyResult<Bound<'py, PyArray2<i32>>> {
        #[cfg(feature = "cuda")]
        {
            use crate::cuda_ext::PyTensor;
            let (a, b) = (
                descriptors_a.cast::<PyTensor>(),
                descriptors_b.cast::<PyTensor>(),
            );
            match (a, b) {
                (Ok(a), Ok(b)) => {
                    return crate::cuda_ext::cuda_sift::sift_match_device(
                        py,
                        &a.borrow(),
                        &b.borrow(),
                        &mut self.store,
                        self.max_keypoints,
                        ratio,
                        cross_check,
                    );
                }
                (Ok(_), Err(_)) | (Err(_), Ok(_)) => {
                    return Err(PyValueError::new_err(
                        "Sift.match: got one device Tensor and one host array; both \
                         descriptor blocks must have the same residency. Detect both \
                         frames from device images, or both from host ones.",
                    ));
                }
                _ => {}
            }
        }
        self.match_host(py, descriptors_a, descriptors_b, ratio, cross_check)
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

impl Sift {
    /// Everything the CUDA plan's cache key and constructor need, in one value
    /// so the eleven-argument call site cannot silently reorder.
    #[cfg(feature = "cuda")]
    fn detect_params(&self) -> crate::cuda_ext::cuda_sift::DetectParams {
        crate::cuda_ext::cuda_sift::DetectParams {
            n_features: self.n_features,
            n_octave_layers: self.n_octave_layers,
            fast_descriptor: self.fast_descriptor,
            contrast_threshold: self.contrast_threshold,
            edge_threshold: self.edge_threshold,
            sigma: self.sigma,
            max_keypoints: self.max_keypoints,
            upsample: self.upsample,
            max_octaves: self.max_octaves,
        }
    }

    /// Match two host `(N, 128)` descriptor blocks with the NEON matcher.
    fn match_host<'py>(
        &mut self,
        py: Python<'py>,
        a: &Bound<'py, PyAny>,
        b: &Bound<'py, PyAny>,
        ratio: f32,
        cross_check: bool,
    ) -> PyResult<Bound<'py, PyArray2<i32>>> {
        let (da, db) = (
            host_descriptors(a, "descriptors_a")?,
            host_descriptors(b, "descriptors_b")?,
        );
        let (na, nb) = (da.shape()[0], db.shape()[0]);
        let (ra, rb) = (da.readonly(), db.readonly());
        let (sa, sb) = (
            ra.as_slice().map_err(|_| contiguity_err())?,
            rb.as_slice().map_err(|_| contiguity_err())?,
        );
        let pairs = py.detach(|| sift_match_descriptors(sa, na, sb, nb, ratio, cross_check));
        let flat: Vec<i32> = pairs.iter().flat_map(|p| [p[0], p[1]]).collect();
        crate::pyutils::rows_to_numpy(py, flat, 2)
    }

    /// The NEON path, for a numpy array or a host `Image`.
    fn detect_host(&mut self, py: Python<'_>, image: &Bound<'_, PyAny>) -> DetectOut {
        // A host `Image` exposes its buffer through `numpy()`; a numpy array is
        // already one. Either way the CPU path wants a contiguous `(H, W, 1)`
        // f32 view, and neither is copied.
        let arr: Bound<'_, PyArray3<f32>> = match image.call_method0("numpy") {
            Ok(v) => v.extract()?,
            Err(_) => image.extract().map_err(|_| {
                PyValueError::new_err(
                    "Sift.detect_and_compute: expected a device Image, a host Image, \
                     or a single-channel float32 array of shape (H, W, 1)",
                )
            })?,
        };
        let shape = arr.shape().to_vec();
        if shape.len() != 3 || shape[2] != 1 {
            return Err(PyValueError::new_err(format!(
                "Sift.detect_and_compute: expected a single-channel image of shape \
                 (H, W, 1), got {shape:?}"
            )));
        }
        let (h, w) = (shape[0], shape[1]);
        let ro = arr.readonly();
        let src = ro.as_slice().map_err(|_| {
            PyValueError::new_err(
                "Sift.detect_and_compute: expected a C-contiguous array; call \
                 np.ascontiguousarray(a) first",
            )
        })?;

        let cfg = SiftConfig {
            n_features: self.n_features,
            n_octave_layers: self.n_octave_layers,
            contrast_threshold: self.contrast_threshold,
            edge_threshold: self.edge_threshold,
            sigma: self.sigma,
        };
        let first_octave = if self.upsample {
            CpuFirstOctave::Double
        } else {
            CpuFirstOctave::Native
        };
        let max_octaves = if self.max_octaves == 0 {
            usize::MAX
        } else {
            self.max_octaves
        };
        // Detection is pure compute over a borrowed buffer, so the interpreter
        // lock is not needed for the duration.
        let ws = &mut self.ws;
        let fast = self.fast_descriptor;
        let feats = py
            .detach(|| {
                sift_detect_and_compute(ws, src, w, h, &cfg, first_octave, max_octaves, fast)
            })
            .map_err(|e| PyValueError::new_err(format!("Sift: {e}")))?;

        let desc = crate::pyutils::rows_to_numpy(py, feats.descriptors, DESCR_LEN)?;
        Ok((
            keypoints_to_list(&feats.keypoints),
            desc.into_any().unbind(),
        ))
    }
}

/// Validate a host descriptor block: `(N, 128)` float32.
///
/// The column count is checked here rather than left to the matcher, which
/// derives `N` from the buffer length and would happily read a `(N, 64)` block
/// as half as many 128-D rows.
fn host_descriptors<'py>(
    obj: &Bound<'py, PyAny>,
    who: &str,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let arr: Bound<'py, PyArray2<f32>> = obj.extract().map_err(|_| {
        PyValueError::new_err(format!(
            "Sift.match: {who} must be a float32 array of shape (N, {DESCR_LEN}), \
             or a device Tensor from a device detect"
        ))
    })?;
    let shape = arr.shape();
    if shape[1] != DESCR_LEN {
        return Err(PyValueError::new_err(format!(
            "Sift.match: {who} has {} columns, expected {DESCR_LEN}",
            shape[1]
        )));
    }
    Ok(arr)
}

fn contiguity_err() -> PyErr {
    PyValueError::new_err(
        "Sift.match: descriptors must be C-contiguous; call \
         np.ascontiguousarray(d) first",
    )
}
