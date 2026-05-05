//! Python bindings for two-view relative pose estimation.
//!
//! Two surfaces are exposed on `kornia_rs.k3d`:
//!
//! 1. The [`PyTwoViewEstimator`] pyclass + plug-in solver/refiner pyclasses
//!    ([`PyFundamental8ptSolver`], [`PyEssentialNister5ptSolver`],
//!    [`PyLmRefiner`], [`PyNoopRefiner`]) that mirror the Rust trait/builder
//!    API — for callers that want to swap epipolar solvers or refinement
//!    strategies without restringifying.
//! 2. [`two_view_estimate_py`] — a one-shot convenience function that picks
//!    solver/refiner from string kwargs. Same defaults as
//!    `TwoViewEstimator()`. Easier from a notebook; equivalent under the hood.
//!
//! Both run the ORB-SLAM3-style monocular bootstrap: F+H RANSAC race → model
//! selection → essential decomposition → cheirality vote → optional LM polish.
//!
//! All matrices are row-major float64. Returns a [`PyTwoViewPose`]
//! `#[pyclass(frozen)]` — named attributes beat a positional tuple because
//! downstream SLAM code cares about *which* field it's reading and future
//! additions (e.g. pose covariance) stay non-breaking.

use numpy::{PyArray, PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;

use kornia_3d::pose::{
    EssentialNister5ptSolver, Fundamental8ptSolver, LmPoseConfig, LmRefiner, NoopRefiner,
    RansacParams, TriangulationConfig, TwoViewEstimator, TwoViewModel, TwoViewResult,
};
use kornia_algebra::{Mat3F64, Vec2F64};

use crate::pyutils::{mask_to_py, mat3_to_py, unpack_mat3, unpack_pts};

/// Result of a two-view pose estimate. Field access mirrors the Rust
/// `TwoViewResult`; all matrices are row-major float64.
#[pyclass(name = "TwoViewPose", module = "kornia_rs.k3d", frozen)]
pub struct PyTwoViewPose {
    /// `(3, 3)` float64 relative rotation from view 1 to view 2.
    #[pyo3(get)]
    pub rotation: Py<PyArray2<f64>>,
    /// `(3,)` float64 unit-length translation direction from view 1 to view 2.
    /// Scale is unobservable from monocular two-view geometry — magnitude is
    /// not metric.
    #[pyo3(get)]
    pub translation: Py<PyArray1<f64>>,
    /// Which model won inlier selection: `"fundamental"`, `"essential"`, or
    /// `"homography"`.
    #[pyo3(get)]
    pub model_type: String,
    /// `(3, 3)` float64 winning model matrix (F, E, or H).
    #[pyo3(get)]
    pub model: Py<PyArray2<f64>>,
    /// `(N,)` uint8 inlier mask over the input correspondences (1 = inlier).
    #[pyo3(get)]
    pub inliers: Py<PyArray1<u8>>,
    /// Total inlier count used by the winning model.
    #[pyo3(get)]
    pub inlier_count: usize,
    /// `(M,)` int64 indices into the input `pts1` / `pts2` arrays for each
    /// triangulated 3D point in `points3d`. `M ≤ inlier_count` — cheirality
    /// drops points behind a camera or outside the parallax/reprojection gate.
    #[pyo3(get)]
    pub inlier_indices: Py<PyArray1<i64>>,
    /// `(M, 3)` float64 triangulated 3D points in the view-1 camera frame.
    #[pyo3(get)]
    pub points3d: Py<PyArray2<f64>>,
}

#[pymethods]
impl PyTwoViewPose {
    fn __repr__(&self, py: Python<'_>) -> String {
        let n_pts = self.points3d.bind(py).shape()[0];
        format!(
            "TwoViewPose(model_type='{}', inlier_count={}, points3d={})",
            self.model_type, self.inlier_count, n_pts
        )
    }
}

// --- shared helpers -------------------------------------------------------

/// Validate shapes/dtypes/contiguity of the three (or four) input arrays and
/// copy them into kornia-algebra types. Centralized so both
/// [`two_view_estimate_py`] and [`PyTwoViewEstimator::estimate`] reject bad
/// inputs identically.
fn unpack_two_view_inputs(
    pts1: &Bound<'_, PyArray2<f64>>,
    pts2: &Bound<'_, PyArray2<f64>>,
    k1: &Bound<'_, PyArray2<f64>>,
    k2: Option<&Bound<'_, PyArray2<f64>>>,
) -> PyResult<(Vec<Vec2F64>, Vec<Vec2F64>, Mat3F64, Mat3F64)> {
    if !pts1.is_c_contiguous() || !pts2.is_c_contiguous() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "point arrays must be C-contiguous",
        ));
    }
    let s1 = pts1.shape();
    let s2 = pts2.shape();
    if s1[1] != 2 || s2[1] != 2 || s1[0] != s2[0] {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "expected (N, 2) float64 arrays with matching length, got ({}, {}) and ({}, {})",
            s1[0], s1[1], s2[0], s2[1]
        )));
    }
    let x1 = unpack_pts(pts1);
    let x2 = unpack_pts(pts2);
    let k1_mat = unpack_mat3(k1)?;
    let k2_mat = match k2 {
        Some(k) => unpack_mat3(k)?,
        None => k1_mat,
    };
    Ok((x1, x2, k1_mat, k2_mat))
}

/// Pack a [`TwoViewResult`] into a [`PyTwoViewPose`] (numpy arrays, owned
/// `Py<...>` handles). Single-pass over inliers/indices/points to avoid
/// extra walks. Unsafe is contained to the numpy buffer fills, all of which
/// stay within the just-allocated arrays.
fn result_to_py(py: Python<'_>, result: TwoViewResult) -> PyTwoViewPose {
    let rotation = mat3_to_py(py, &result.rotation).unbind();
    let t_unit = result.translation.normalize();
    let translation = unsafe {
        let arr = PyArray::<f64, _>::new(py, [3], false);
        let slice = std::slice::from_raw_parts_mut(arr.data(), 3);
        slice[0] = t_unit.x;
        slice[1] = t_unit.y;
        slice[2] = t_unit.z;
        arr.unbind()
    };

    let (model_type, model_mat) = match result.model {
        TwoViewModel::Fundamental(f) => ("fundamental".to_string(), f),
        TwoViewModel::Essential(e) => ("essential".to_string(), e),
        TwoViewModel::Homography(h) => ("homography".to_string(), h),
    };
    let model = mat3_to_py(py, &model_mat).unbind();

    let inlier_count = result.inliers.iter().filter(|b| **b).count();
    let inliers = mask_to_py(py, &result.inliers).unbind();

    // Triangulated inlier indices → int64, 3D points → (M, 3) float64.
    let m = result.inlier_indices.len();
    let inlier_indices = unsafe {
        let arr = PyArray::<i64, _>::new(py, [m], false);
        let slice = std::slice::from_raw_parts_mut(arr.data(), m);
        for (dst, src) in slice.iter_mut().zip(result.inlier_indices.iter()) {
            *dst = *src as i64;
        }
        arr.unbind()
    };

    let points3d = unsafe {
        let arr = PyArray::<f64, _>::new(py, [m, 3], false);
        let slice = std::slice::from_raw_parts_mut(arr.data(), m * 3);
        for (i, p) in result.points3d.iter().enumerate() {
            slice[i * 3] = p.x;
            slice[i * 3 + 1] = p.y;
            slice[i * 3 + 2] = p.z;
        }
        arr.unbind()
    };

    PyTwoViewPose {
        rotation,
        translation,
        model_type,
        model,
        inliers,
        inlier_count,
        inlier_indices,
        points3d,
    }
}

// --- solver pyclasses -----------------------------------------------------

/// Eight-point fundamental solver — F via DLT, σ-equalized lift to E. Faster
/// than the 5-point path; cleaner rotation. Default in
/// [`PyTwoViewEstimator`].
///
/// **Accuracy tradeoff vs [`EssentialNister5ptSolver`]** (EuRoC MH_01,
/// median over 20 RANSAC seeds):
/// - 8pt: rot ≈ 0.04°, t ≈ 3.7° — *best rotation*
/// - 5pt: rot ≈ 0.15°, t ≈ 3.1° — *best translation direction*
///
/// 8pt+lift solves F in pixel space then projects onto the E-manifold via
/// σ-equalization. The projection noise lands in `t` (the right null space
/// of the SVD clip) while leaving `R` clean. Pick 8pt when downstream
/// cares about rotation accuracy first (visual odometry rotation drift,
/// IMU-camera alignment); pick 5pt when translation direction matters
/// most (sparse SLAM bootstrap, structure-from-motion baselines).
///
/// Args:
///     max_iterations: RANSAC iteration cap. Default 2000.
///     threshold: inlier threshold in pixels (Sampson). Default 1.0.
///     min_inliers: required inliers to accept the model. Default 15.
///     seed: optional RNG seed for deterministic runs.
///     refit: re-fit across all inliers after the main loop (LO-RANSAC).
///         Default True. Only the homography arm honors this today.
#[pyclass(
    name = "Fundamental8ptSolver",
    module = "kornia_rs.k3d",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyFundamental8ptSolver {
    inner: Fundamental8ptSolver,
}

#[pymethods]
impl PyFundamental8ptSolver {
    #[new]
    #[pyo3(signature = (
        max_iterations = 2000,
        threshold = 1.0,
        min_inliers = 15,
        seed = None,
        refit = true,
    ))]
    fn new(
        max_iterations: usize,
        threshold: f64,
        min_inliers: usize,
        seed: Option<u64>,
        refit: bool,
    ) -> Self {
        Self {
            inner: Fundamental8ptSolver {
                ransac: RansacParams {
                    max_iterations,
                    threshold,
                    min_inliers,
                    random_seed: seed,
                    refit,
                },
            },
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Fundamental8ptSolver(max_iterations={}, threshold={}, min_inliers={}, seed={:?}, refit={})",
            self.inner.ransac.max_iterations,
            self.inner.ransac.threshold,
            self.inner.ransac.min_inliers,
            self.inner.ransac.random_seed,
            self.inner.ransac.refit,
        )
    }
}

/// Nistér five-point essential solver — calibrated rays, on-manifold by
/// construction. Preserves translation-direction accuracy at the cost of a
/// 10-degree polynomial solve per RANSAC sample.
///
/// **Accuracy tradeoff vs [`PyFundamental8ptSolver`]** (EuRoC MH_01,
/// median over 20 RANSAC seeds):
/// - 5pt: rot ≈ 0.15°, t ≈ 3.1° — *best translation direction*
/// - 8pt: rot ≈ 0.04°, t ≈ 3.7° — *best rotation*
///
/// 5pt stays on the E-manifold without a σ-clipping projection round-trip,
/// so `t`-direction stays clean; the polynomial-root branch landing has
/// slightly more rotation jitter than 8pt+lift. Pick 5pt for sparse-SLAM
/// bootstrap and SfM baselines where translation direction is the
/// downstream signal; pick 8pt when rotation dominates downstream cost
/// (visual-odometry drift, IMU-camera calibration).
///
/// Pose-stage cost: ~3.2 ms vs 8pt's ~1.4 ms on this image pair (Jetson
/// Orin) — the 10-degree polynomial solve is the dominant factor.
///
/// Args mirror [`PyFundamental8ptSolver`].
#[pyclass(
    name = "EssentialNister5ptSolver",
    module = "kornia_rs.k3d",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyEssentialNister5ptSolver {
    inner: EssentialNister5ptSolver,
}

#[pymethods]
impl PyEssentialNister5ptSolver {
    #[new]
    #[pyo3(signature = (
        max_iterations = 2000,
        threshold = 1.0,
        min_inliers = 15,
        seed = None,
        refit = true,
    ))]
    fn new(
        max_iterations: usize,
        threshold: f64,
        min_inliers: usize,
        seed: Option<u64>,
        refit: bool,
    ) -> Self {
        Self {
            inner: EssentialNister5ptSolver {
                ransac: RansacParams {
                    max_iterations,
                    threshold,
                    min_inliers,
                    random_seed: seed,
                    refit,
                },
            },
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "EssentialNister5ptSolver(max_iterations={}, threshold={}, min_inliers={}, seed={:?}, refit={})",
            self.inner.ransac.max_iterations,
            self.inner.ransac.threshold,
            self.inner.ransac.min_inliers,
            self.inner.ransac.random_seed,
            self.inner.ransac.refit,
        )
    }
}

// --- refiner pyclasses ----------------------------------------------------

/// Levenberg-Marquardt pose refiner with optional threshold-annealed polish
/// (OpenCV USAC LO+ pattern). Default refiner in [`PyTwoViewEstimator`].
///
/// Args:
///     max_iters: hard cap on LM iterations. Default 10.
///     initial_lambda: initial Levenberg damping. Default 1e-3.
///     lambda_up: λ multiplier on rejected step. Default 10.0.
///     lambda_down: λ multiplier on accepted step. Default 0.5.
///     gradient_tol: convergence on `‖J^T r‖_∞`. Default 1e-9.
///     step_tol: convergence on `‖δ‖`. Default 1e-9.
///     cost_tol: convergence on `|Δcost|/|cost|`. Default 1e-12.
///     anneal_thresholds: multipliers on the RANSAC threshold for each polish
///         pass. Empty list disables annealing. Default `[0.5, 0.25]`.
///     anneal_min_inliers: minimum inliers required to admit an annealed
///         pass. Default 30.
#[pyclass(
    name = "LmRefiner",
    module = "kornia_rs.k3d",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyLmRefiner {
    inner: LmRefiner,
}

#[pymethods]
impl PyLmRefiner {
    #[new]
    #[pyo3(signature = (
        max_iters = 10,
        initial_lambda = 1e-3,
        lambda_up = 10.0,
        lambda_down = 0.5,
        gradient_tol = 1e-9,
        step_tol = 1e-9,
        cost_tol = 1e-12,
        anneal_thresholds = vec![0.5, 0.25],
        anneal_min_inliers = 30,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        max_iters: usize,
        initial_lambda: f64,
        lambda_up: f64,
        lambda_down: f64,
        gradient_tol: f64,
        step_tol: f64,
        cost_tol: f64,
        anneal_thresholds: Vec<f64>,
        anneal_min_inliers: usize,
    ) -> Self {
        Self {
            inner: LmRefiner {
                config: LmPoseConfig {
                    max_iters,
                    initial_lambda,
                    lambda_up,
                    lambda_down,
                    gradient_tol,
                    step_tol,
                    cost_tol,
                    // Kernel fields default to Identity / ∞ scale — no
                    // behaviour change vs. the pre-kernel API. Pythonside
                    // wiring of robust loss is a follow-up.
                    ..Default::default()
                },
                anneal_thresholds,
                anneal_min_inliers,
            },
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "LmRefiner(max_iters={}, anneal_thresholds={:?}, anneal_min_inliers={})",
            self.inner.config.max_iters,
            self.inner.anneal_thresholds,
            self.inner.anneal_min_inliers,
        )
    }
}

/// Pass-through refiner — returns the cheirality-vote pose unchanged.
/// Use for measuring raw F/E pipeline accuracy or when downstream code owns
/// its own refinement.
#[pyclass(
    name = "NoopRefiner",
    module = "kornia_rs.k3d",
    frozen,
    skip_from_py_object
)]
#[derive(Clone, Copy, Default)]
pub struct PyNoopRefiner;

#[pymethods]
impl PyNoopRefiner {
    #[new]
    fn new() -> Self {
        Self
    }

    fn __repr__(&self) -> String {
        "NoopRefiner()".to_string()
    }
}

// --- estimator pyclass ----------------------------------------------------

/// Configured two-view pose estimator — Python mirror of
/// [`kornia_3d::pose::TwoViewEstimator`]. Build once with the desired solver
/// + refiner, then reuse `estimate(...)` across many image pairs.
///
/// Args:
///     solver: epipolar solver instance — [`PyFundamental8ptSolver`] (default)
///         or [`PyEssentialNister5ptSolver`].
///     refiner: post-cheirality refiner — [`PyLmRefiner`] (default) or
///         [`PyNoopRefiner`].
///     homography_max_iterations: H-arm RANSAC iteration cap. Default 2000.
///     homography_threshold: H-arm reprojection threshold (px). Default 1.0.
///     homography_min_inliers: required inliers to accept H. Default 8.
///     homography_inlier_ratio: prefer H when
///         `H_inliers > ratio * epipolar_inliers`. Default 0.8.
///     min_parallax_deg: triangulation-time parallax floor. Default 1.0.
///     seed: RNG seed for the **homography RANSAC arm only**. The epipolar
///         RANSAC seed lives on the solver itself
///         (`Fundamental8ptSolver(seed=...)` or
///         `EssentialNister5ptSolver(seed=...)`); pass both to make the whole
///         pipeline deterministic.
#[pyclass(name = "TwoViewEstimator", module = "kornia_rs.k3d", frozen)]
pub struct PyTwoViewEstimator {
    inner: TwoViewEstimator,
}

#[pymethods]
impl PyTwoViewEstimator {
    #[new]
    #[pyo3(signature = (
        solver = None,
        refiner = None,
        homography_max_iterations = 2000,
        homography_threshold = 1.0,
        homography_min_inliers = 8,
        homography_inlier_ratio = 0.8,
        min_parallax_deg = 1.0,
        seed = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        solver: Option<&Bound<'_, PyAny>>,
        refiner: Option<&Bound<'_, PyAny>>,
        homography_max_iterations: usize,
        homography_threshold: f64,
        homography_min_inliers: usize,
        homography_inlier_ratio: f64,
        min_parallax_deg: f64,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let homography_ransac = RansacParams {
            max_iterations: homography_max_iterations,
            threshold: homography_threshold,
            min_inliers: homography_min_inliers,
            random_seed: seed,
            refit: true,
        };
        let triangulation = TriangulationConfig {
            min_parallax_deg,
            ..TriangulationConfig::default()
        };

        let mut builder = TwoViewEstimator::builder()
            .homography_ransac(homography_ransac)
            .homography_inlier_ratio(homography_inlier_ratio)
            .triangulation(triangulation);

        // Try-extract the solver pyclass; default Fundamental8pt is already on
        // the builder so `None` is a no-op.
        if let Some(s) = solver {
            if let Ok(f) = s.extract::<PyRef<PyFundamental8ptSolver>>() {
                builder = builder.epipolar_solver(f.inner.clone());
            } else if let Ok(e) = s.extract::<PyRef<PyEssentialNister5ptSolver>>() {
                builder = builder.epipolar_solver(e.inner.clone());
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "solver must be a Fundamental8ptSolver or EssentialNister5ptSolver instance",
                ));
            }
        }

        if let Some(r) = refiner {
            if let Ok(lm) = r.extract::<PyRef<PyLmRefiner>>() {
                builder = builder.refiner(lm.inner.clone());
            } else if r.extract::<PyRef<PyNoopRefiner>>().is_ok() {
                builder = builder.refiner(NoopRefiner);
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "refiner must be an LmRefiner or NoopRefiner instance",
                ));
            }
        }

        Ok(Self {
            inner: builder.build(),
        })
    }

    /// Estimate the relative pose between two views.
    ///
    /// Args:
    ///     pts1, pts2: `(N, 2)` float64 matched pixel coordinates (N ≥ 8).
    ///     k1: `(3, 3)` float64 intrinsics for view 1.
    ///     k2: `(3, 3)` float64 intrinsics for view 2. If `None`, reuses `k1`.
    ///
    /// Returns:
    ///     [`PyTwoViewPose`] with rotation, unit translation direction, the
    ///     winning model matrix, inlier mask, and triangulated inlier 3D
    ///     points.
    ///
    /// Raises:
    ///     ValueError: bad shapes, non-contiguous arrays, or RANSAC failure.
    #[pyo3(signature = (pts1, pts2, k1, k2 = None))]
    fn estimate(
        &self,
        py: Python<'_>,
        pts1: Bound<'_, PyArray2<f64>>,
        pts2: Bound<'_, PyArray2<f64>>,
        k1: Bound<'_, PyArray2<f64>>,
        k2: Option<Bound<'_, PyArray2<f64>>>,
    ) -> PyResult<PyTwoViewPose> {
        let (x1, x2, k1_mat, k2_mat) = unpack_two_view_inputs(&pts1, &pts2, &k1, k2.as_ref())?;
        let result = py
            .detach(|| self.inner.estimate(&x1, &x2, &k1_mat, &k2_mat))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        Ok(result_to_py(py, result))
    }
}

// --- one-shot convenience function ---------------------------------------

/// Estimate the relative camera pose between two views — convenience wrapper
/// around [`PyTwoViewEstimator`] that picks solver/refiner from string kwargs.
///
/// Args:
///     pts1, pts2: `(N, 2)` float64 matched pixel coordinates (N ≥ 8).
///     k1: `(3, 3)` float64 intrinsics for view 1.
///     k2: `(3, 3)` float64 intrinsics for view 2. If `None`, reuses `k1`.
///     ransac_threshold: inlier threshold in pixels (Sampson for F,
///                       reprojection for H). Default 1.0.
///     max_iterations: RANSAC iteration cap. Default 2000.
///     min_inliers_f / min_inliers_h: required inliers to accept each model.
///     homography_inlier_ratio: prefer H when
///         `H_inliers > homography_inlier_ratio * F_inliers`. Default 0.8.
///     min_parallax_deg: minimum parallax for triangulation. Default 1.0.
///     seed: RNG seed.
///     solver: epipolar solver — `"fundamental_8pt"` (default; faster pose,
///         cleaner rotation) or `"essential_5pt"` (on-manifold;
///         translation-direction priority).
///     refine: post-cheirality refinement — `"lm"` (default; LM on Σ Sampson²
///         with `[0.5, 0.25]` annealing) or `"none"` (skip refinement).
///
/// Returns:
///     [`TwoViewPose`] with rotation, unit translation direction, inlier mask,
///     triangulated inlier 3D points, and the selected model matrix.
///
/// Raises ValueError on bad shapes, unknown `solver`/`refine` value, or RANSAC failure.
#[pyfunction(name = "two_view_estimate")]
#[pyo3(signature = (
    pts1,
    pts2,
    k1,
    k2=None,
    ransac_threshold=1.0,
    max_iterations=2000,
    min_inliers_f=15,
    min_inliers_h=8,
    homography_inlier_ratio=0.8,
    min_parallax_deg=1.0,
    seed=None,
    solver="fundamental_8pt",
    refine="lm",
))]
#[allow(clippy::too_many_arguments)]
pub fn two_view_estimate_py(
    py: Python<'_>,
    pts1: Bound<'_, PyArray2<f64>>,
    pts2: Bound<'_, PyArray2<f64>>,
    k1: Bound<'_, PyArray2<f64>>,
    k2: Option<Bound<'_, PyArray2<f64>>>,
    ransac_threshold: f64,
    max_iterations: usize,
    min_inliers_f: usize,
    min_inliers_h: usize,
    homography_inlier_ratio: f64,
    min_parallax_deg: f64,
    seed: Option<u64>,
    solver: &str,
    refine: &str,
) -> PyResult<PyTwoViewPose> {
    let (x1, x2, k1_mat, k2_mat) = unpack_two_view_inputs(&pts1, &pts2, &k1, k2.as_ref())?;

    let ransac_f = RansacParams {
        max_iterations,
        threshold: ransac_threshold,
        min_inliers: min_inliers_f,
        random_seed: seed,
        refit: true,
    };
    let ransac_h = RansacParams {
        max_iterations,
        threshold: ransac_threshold,
        min_inliers: min_inliers_h,
        random_seed: seed,
        refit: true,
    };
    let triangulation = TriangulationConfig {
        min_parallax_deg,
        ..TriangulationConfig::default()
    };

    let mut builder = TwoViewEstimator::builder()
        .homography_ransac(ransac_h)
        .homography_inlier_ratio(homography_inlier_ratio)
        .triangulation(triangulation);

    builder = match solver {
        "fundamental_8pt" => builder.epipolar_solver(Fundamental8ptSolver { ransac: ransac_f }),
        "essential_5pt" => builder.epipolar_solver(EssentialNister5ptSolver { ransac: ransac_f }),
        other => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "unknown solver '{other}': expected 'fundamental_8pt' or 'essential_5pt'"
            )))
        }
    };

    builder = match refine {
        "lm" => builder, // LmRefiner is the builder default.
        "none" => builder.refiner(NoopRefiner),
        other => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "unknown refine '{other}': expected 'lm' or 'none'"
            )))
        }
    };

    let estimator = builder.build();
    let result = py
        .detach(|| estimator.estimate(&x1, &x2, &k1_mat, &k2_mat))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;

    Ok(result_to_py(py, result))
}
