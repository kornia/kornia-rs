//! Python bindings for two-view relative pose estimation.
//!
//! Exposes `kornia_rs.features.two_view_estimate`, the monocular-SLAM bootstrap
//! pipeline used by ORB-SLAM3 et al.:
//!
//! 1. RANSAC a fundamental matrix and a homography in parallel.
//! 2. Pick whichever model has more inliers (H preferred on near-planar scenes).
//! 3. Decompose the winner into 4 candidate (R, t) poses.
//! 4. Triangulate inlier correspondences for each candidate; pick the pose that
//!    produces the most points in front of both cameras (cheirality).
//!
//! Returns a [`PyTwoViewPose`] `#[pyclass(frozen)]` — named attributes beat a
//! positional tuple because downstream SLAM code cares about *which* field it's
//! reading and future additions (e.g. pose covariance) stay non-breaking.

use numpy::{PyArray, PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;

use kornia_3d::pose::{
    two_view_estimate as two_view_estimate_fn, RansacParams, TriangulationConfig, TwoViewConfig,
    TwoViewModel,
};
use kornia_algebra::{Mat3F64, Vec2F64};

/// Result of [`two_view_estimate_py`]. Field access mirrors the Rust
/// `TwoViewResult`; all matrices are row-major float64.
#[pyclass(name = "TwoViewPose", module = "kornia_rs.features", frozen)]
pub struct PyTwoViewPose {
    /// `(3, 3)` float64 relative rotation from view 1 to view 2.
    #[pyo3(get)]
    pub rotation: Py<PyArray2<f64>>,
    /// `(3,)` float64 unit-length translation direction from view 1 to view 2.
    /// Scale is unobservable from monocular two-view geometry — magnitude is
    /// not metric.
    #[pyo3(get)]
    pub translation: Py<PyArray1<f64>>,
    /// Which model won inlier selection: `"fundamental"` or `"homography"`.
    #[pyo3(get)]
    pub model_type: String,
    /// `(3, 3)` float64 winning model matrix (F or H).
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

fn unpack_pts(arr: &Bound<'_, PyArray2<f64>>) -> Vec<Vec2F64> {
    let n = arr.shape()[0];
    unsafe {
        let raw = std::slice::from_raw_parts(arr.data(), n * 2);
        (0..n)
            .map(|i| Vec2F64::new(raw[i * 2], raw[i * 2 + 1]))
            .collect()
    }
}

fn unpack_mat3(arr: &Bound<'_, PyArray2<f64>>) -> PyResult<Mat3F64> {
    if !arr.is_c_contiguous() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "intrinsics matrix must be C-contiguous",
        ));
    }
    let s = arr.shape();
    if s[0] != 3 || s[1] != 3 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "expected (3, 3) float64 intrinsics, got ({}, {})",
            s[0], s[1]
        )));
    }
    // Numpy is row-major, Mat3F64 is column-major.
    let raw = unsafe { std::slice::from_raw_parts(arr.data(), 9) };
    Ok(Mat3F64::from_cols_array(&[
        raw[0], raw[3], raw[6], raw[1], raw[4], raw[7], raw[2], raw[5], raw[8],
    ]))
}

fn mat3_to_py<'py>(py: Python<'py>, m: &Mat3F64) -> Bound<'py, PyArray2<f64>> {
    let cols = m.to_cols_array();
    unsafe {
        let arr = PyArray::<f64, _>::new(py, [3, 3], false);
        let slice = std::slice::from_raw_parts_mut(arr.data(), 9);
        for r in 0..3 {
            for c in 0..3 {
                slice[r * 3 + c] = cols[c * 3 + r];
            }
        }
        arr
    }
}

/// Estimate the relative camera pose between two views — the ORB-SLAM-style
/// bootstrap pipeline: F + H RANSAC → model selection → essential decomposition
/// → cheirality.
///
/// Args:
///     pts1, pts2: `(N, 2)` float64 matched pixel coordinates (N ≥ 8).
///     k1: `(3, 3)` float64 intrinsics for view 1.
///     k2: `(3, 3)` float64 intrinsics for view 2. If `None`, reuses `k1`.
///     ransac_threshold: inlier threshold in pixels (Sampson for F,
///                        reprojection for H). Default 1.0.
///     max_iterations: RANSAC iteration cap. Default 2000.
///     min_inliers_f / min_inliers_h: required inliers to accept each model.
///     homography_inlier_ratio: prefer H when
///         `H_inliers > homography_inlier_ratio * F_inliers`. Default 0.8.
///     min_parallax_deg: minimum parallax for triangulation. Default 1.0.
///     seed: RNG seed.
///
/// Returns:
///     [`TwoViewPose`] with rotation, unit translation direction, inlier mask,
///     triangulated inlier 3D points, and the selected model matrix.
///
/// Raises ValueError on bad shapes or RANSAC failure.
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
) -> PyResult<PyTwoViewPose> {
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

    let x1 = unpack_pts(&pts1);
    let x2 = unpack_pts(&pts2);
    let k1_mat = unpack_mat3(&k1)?;
    let k2_mat = match k2 {
        Some(k) => unpack_mat3(&k)?,
        None => k1_mat,
    };

    let config = TwoViewConfig {
        ransac_f: RansacParams {
            max_iterations,
            threshold: ransac_threshold,
            min_inliers: min_inliers_f,
            random_seed: seed,
            refit: false,
        },
        ransac_h: RansacParams {
            max_iterations,
            threshold: ransac_threshold,
            min_inliers: min_inliers_h,
            random_seed: seed,
            refit: false,
        },
        homography_inlier_ratio,
        triangulation: TriangulationConfig {
            min_parallax_deg,
            ..TriangulationConfig::default()
        },
    };

    let result = py
        .detach(|| two_view_estimate_fn(&x1, &x2, &k1_mat, &k2_mat, &config))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;

    // Pack rotation, translation, and winning model matrix.
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
        TwoViewModel::Homography(h) => ("homography".to_string(), h),
    };
    let model = mat3_to_py(py, &model_mat).unbind();

    // Inlier mask: Vec<bool> → uint8 numpy. Count while copying instead of a
    // second pass.
    let n = result.inliers.len();
    let mut inlier_count = 0usize;
    let inliers = unsafe {
        let arr = PyArray::<u8, _>::new(py, [n], false);
        let slice = std::slice::from_raw_parts_mut(arr.data(), n);
        for (dst, src) in slice.iter_mut().zip(result.inliers.iter()) {
            *dst = *src as u8;
            inlier_count += *src as usize;
        }
        arr.unbind()
    };

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

    Ok(PyTwoViewPose {
        rotation,
        translation,
        model_type,
        model,
        inliers,
        inlier_count,
        inlier_indices,
        points3d,
    })
}
