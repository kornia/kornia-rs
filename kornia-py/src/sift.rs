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
//! `Sift.match` is the exception, and says so: the brute-force matcher is a CUDA
//! kernel, so it requires device images rather than silently uploading.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use kornia_imgproc::features::{
    sift_detect_with, FirstOctave as CpuFirstOctave, SiftConfig, SiftKeypoint, SiftWorkspace,
    DESCR_LEN,
};
use numpy::{PyArray2, PyArray3, PyArrayMethods, PyUntypedArrayMethods};

#[cfg(feature = "cuda")]
use crate::image::PyImageApi;

/// A reusable SIFT detector, shaped like `cv2.SIFT`.
///
/// ```python
/// sift = kornia_rs.imgproc.Sift(contrast_threshold=0.04)
/// kp, desc = sift.detect_and_compute(device_image)   # CUDA
/// kp, desc = sift.detect_and_compute(numpy_array)    # NEON
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

/// `(N, 6)` of `x, y, size, angle, response, octave`.
fn keypoints_to_numpy<'py>(
    py: Python<'py>,
    kps: &[SiftKeypoint],
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let rows: Vec<Vec<f32>> = kps
        .iter()
        .map(|k| vec![k.x, k.y, k.size, k.angle, k.response, k.octave as f32])
        .collect();
    PyArray2::from_vec2(py, &rows).map_err(|e| PyValueError::new_err(e.to_string()))
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
    /// Returns `(keypoints, descriptors)`: `(N, 6)` of
    /// `x, y, size, angle, response, octave`, and `(N, 128)`.
    fn detect_and_compute<'py>(
        &mut self,
        py: Python<'py>,
        image: &Bound<'py, PyAny>,
    ) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>)> {
        #[cfg(feature = "cuda")]
        if let Ok(api) = image.cast::<PyImageApi>() {
            let img = api.borrow();
            if img.is_device() {
                return crate::cuda_ext::cuda_sift::sift_cuda_with_plan(
                    py,
                    &img,
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
                );
            }
        }
        self.detect_host(py, image)
    }

    /// Detect in both images and match, without the descriptors leaving the
    /// device.
    ///
    /// Device images only: the matcher is a CUDA kernel, and uploading on the
    /// caller's behalf would hide a full-frame transfer inside what looks like a
    /// matching call. Detection alone runs on either backend — see
    /// [`Sift::detect_and_compute`].
    ///
    /// `ratio` is Lowe's ratio; `>= 1.0` disables it. `cross_check` requires
    /// each pair to be a mutual nearest neighbour.
    ///
    /// Returns `(keypoints_a, keypoints_b, matches)`, where `matches` is
    /// `(M, 2)` of indices into the two keypoint arrays.
    #[pyo3(signature = (image_a, image_b, ratio=0.8, cross_check=true))]
    #[allow(unused_variables)]
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
        #[cfg(feature = "cuda")]
        {
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
        #[cfg(not(feature = "cuda"))]
        Err(PyValueError::new_err(
            "Sift.match: the matcher is a CUDA kernel and this build has no CUDA \
             support; detect on the CPU and match with your own routine",
        ))
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
    /// The NEON path, for a numpy array or a host `Image`.
    fn detect_host<'py>(
        &mut self,
        py: Python<'py>,
        image: &Bound<'py, PyAny>,
    ) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>)> {
        // A host `Image` exposes its buffer through `numpy()`; a numpy array is
        // already one. Either way the CPU path wants a contiguous `(H, W, 1)`
        // f32 view, and neither is copied.
        let arr: Bound<'py, PyArray3<f32>> = match image.call_method0("numpy") {
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
            .detach(|| sift_detect_with(ws, src, w, h, &cfg, first_octave, max_octaves, fast))
            .map_err(|e| PyValueError::new_err(format!("Sift: {e}")))?;

        let kp = keypoints_to_numpy(py, &feats.keypoints)?;
        let desc = PyArray2::from_vec2(
            py,
            &feats
                .descriptors
                .chunks(DESCR_LEN)
                .map(|c| c.to_vec())
                .collect::<Vec<_>>(),
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((kp, desc))
    }
}

/// Borrow a device `Image`, or explain what is wrong.
#[cfg(feature = "cuda")]
fn device_image<'py>(obj: &Bound<'py, PyAny>, who: &str) -> PyResult<Bound<'py, PyImageApi>> {
    let api = obj.cast::<PyImageApi>().map_err(|_| {
        PyValueError::new_err(format!(
            "{who}: expected a device Image; convert with \
             Image.from_numpy(a).to_cuda(stream)"
        ))
    })?;
    if !api.borrow().is_device() {
        return Err(PyValueError::new_err(format!(
            "{who}: this Image lives on the host; move it with .to_cuda(stream)"
        )));
    }
    Ok(api.clone())
}
