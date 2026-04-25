use numpy::{PyArray1, PyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

use kornia_3d::pose::{
    fundamental_8point as fundamental_8point_fn, homography_dlt as homography_dlt_fn,
    ransac_fundamental as ransac_fundamental_fn, ransac_homography as ransac_homography_fn,
    RansacParams,
};

use crate::pyutils::{mask_to_py, mat3_to_py, unpack_pts};

const METHOD_RANSAC: i32 = 8;

/// Estimate a homography with RANSAC using the normalized 4-point solver.
///
/// Args:
///     pts1: `(N, 2)` float64 source points.
///     pts2: `(N, 2)` float64 destination points.
///     threshold: inlier reprojection-error threshold in pixels (default 3.0).
///     max_iterations: RANSAC iteration cap (default 2000).
///     min_inliers: minimum inliers required for a valid fit (default 15).
///     seed: optional RNG seed for deterministic runs.
///
/// Returns:
///     `(H, inlier_mask, inlier_count)` where:
///     * `H` — `(3, 3)` float64 row-major homography mapping `pts1 → pts2`.
///     * `inlier_mask` — `(N,)` uint8 per-correspondence mask (1 = inlier).
///     * `inlier_count` — total inlier count (int).
///
/// Raises ValueError if RANSAC fails to find a model meeting `min_inliers`.
#[pyfunction(name = "ransac_homography")]
#[pyo3(signature = (pts1, pts2, threshold=3.0, max_iterations=2000, min_inliers=15, seed=None))]
pub fn ransac_homography_py(
    py: Python<'_>,
    pts1: Bound<'_, PyArray2<f64>>,
    pts2: Bound<'_, PyArray2<f64>>,
    threshold: f64,
    max_iterations: usize,
    min_inliers: usize,
    seed: Option<u64>,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray1<u8>>, usize)> {
    if !pts1.is_c_contiguous() || !pts2.is_c_contiguous() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "point arrays must be C-contiguous",
        ));
    }
    let s1 = pts1.shape();
    let s2 = pts2.shape();
    if s1[1] != 2 || s2[1] != 2 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "expected (N, 2) float64 arrays, got ({}, {}) and ({}, {})",
            s1[0], s1[1], s2[0], s2[1]
        )));
    }
    if s1[0] != s2[0] {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "pts1 and pts2 must have the same length: {} vs {}",
            s1[0], s2[0]
        )));
    }

    let x1 = unpack_pts(&pts1);
    let x2 = unpack_pts(&pts2);

    let params = RansacParams {
        max_iterations,
        threshold,
        min_inliers,
        random_seed: seed,
        ..Default::default()
    };

    let result = py
        .detach(|| ransac_homography_fn(&x1, &x2, &params))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;

    let h_arr = mat3_to_py(py, &result.model);
    let mask_arr = mask_to_py(py, &result.inliers);
    Ok((h_arr.unbind(), mask_arr.unbind(), result.inlier_count))
}

/// Estimate a homography between two point sets — cv2.findHomography-style.
///
/// Args:
///     pts1: `(N, 2)` float64 source points (N ≥ 4).
///     pts2: `(N, 2)` float64 destination points (same length as pts1).
///     method: `0` for direct DLT (least-squares over all points, no outlier rejection),
///             `8` (alias for `cv2.RANSAC`) for RANSAC + inlier refit (LO-RANSAC).
///     ransac_threshold: inlier reprojection-error threshold in pixels (RANSAC only).
///     max_iterations: RANSAC iteration cap.
///     min_inliers: minimum inliers required for a valid RANSAC fit.
///     seed: optional RNG seed for deterministic RANSAC runs.
///
/// Returns:
///     `(H, inlier_mask)` where:
///     * `H` — `(3, 3)` float64 row-major homography mapping `pts1 → pts2`.
///     * `inlier_mask` — `(N,)` uint8; for method=0 it's all-ones; for RANSAC
///       it's 1 on inliers, 0 on outliers.
///
/// Raises ValueError on singular/insufficient input or RANSAC failure.
#[pyfunction(name = "find_homography")]
#[pyo3(signature = (
    pts1,
    pts2,
    method=0,
    ransac_threshold=3.0,
    max_iterations=2000,
    min_inliers=4,
    seed=None,
))]
pub fn find_homography_py(
    py: Python<'_>,
    pts1: Bound<'_, PyArray2<f64>>,
    pts2: Bound<'_, PyArray2<f64>>,
    method: i32,
    ransac_threshold: f64,
    max_iterations: usize,
    min_inliers: usize,
    seed: Option<u64>,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray1<u8>>)> {
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
    let n = s1[0];
    let x1 = unpack_pts(&pts1);
    let x2 = unpack_pts(&pts2);

    let (h, inliers) = match method {
        0 => {
            let h = py
                .detach(|| homography_dlt_fn(&x1, &x2))
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
            (h, vec![true; n])
        }
        METHOD_RANSAC => {
            let params = RansacParams {
                max_iterations,
                threshold: ransac_threshold,
                min_inliers,
                random_seed: seed,
                refit: true,
            };
            let res = py
                .detach(|| ransac_homography_fn(&x1, &x2, &params))
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
            (res.model, res.inliers)
        }
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "unsupported method {method} — use 0 (DLT) or 8 (RANSAC)"
            )));
        }
    };

    let h_arr = mat3_to_py(py, &h);
    let mask_arr = mask_to_py(py, &inliers);
    Ok((h_arr.unbind(), mask_arr.unbind()))
}

/// Estimate a fundamental matrix between two point sets — cv2.findFundamentalMat-style.
///
/// Args:
///     pts1: `(N, 2)` float64 source points (N ≥ 8).
///     pts2: `(N, 2)` float64 destination points (same length as pts1).
///     method: `0` for direct 8-point DLT (least-squares over all points, no
///             outlier rejection), `8` (alias for `cv2.FM_RANSAC`) for RANSAC
///             on the 8-point solver with Sampson-distance inlier scoring.
///     ransac_threshold: inlier Sampson-distance threshold in pixels (RANSAC only).
///     max_iterations: RANSAC iteration cap.
///     min_inliers: minimum inliers required for a valid RANSAC fit.
///     seed: optional RNG seed for deterministic RANSAC runs.
///
/// Returns:
///     `(F, inlier_mask)` where:
///     * `F` — `(3, 3)` float64 row-major fundamental matrix satisfying
///       `x2ᵀ · F · x1 = 0` for corresponding points.
///     * `inlier_mask` — `(N,)` uint8; for method=0 it's all-ones; for RANSAC
///       it's 1 on inliers, 0 on outliers.
///
/// Raises ValueError on singular/insufficient input or RANSAC failure.
#[pyfunction(name = "find_fundamental")]
#[pyo3(signature = (
    pts1,
    pts2,
    method=0,
    ransac_threshold=3.0,
    max_iterations=2000,
    min_inliers=8,
    seed=None,
))]
pub fn find_fundamental_py(
    py: Python<'_>,
    pts1: Bound<'_, PyArray2<f64>>,
    pts2: Bound<'_, PyArray2<f64>>,
    method: i32,
    ransac_threshold: f64,
    max_iterations: usize,
    min_inliers: usize,
    seed: Option<u64>,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray1<u8>>)> {
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
    let n = s1[0];
    let x1 = unpack_pts(&pts1);
    let x2 = unpack_pts(&pts2);

    let (f, inliers) = match method {
        0 => {
            let f = py
                .detach(|| fundamental_8point_fn(&x1, &x2))
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
            (f, vec![true; n])
        }
        METHOD_RANSAC => {
            let params = RansacParams {
                max_iterations,
                threshold: ransac_threshold,
                min_inliers,
                random_seed: seed,
                refit: false,
            };
            let res = py
                .detach(|| ransac_fundamental_fn(&x1, &x2, &params))
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
            (res.model, res.inliers)
        }
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "unsupported method {method} — use 0 (8-point DLT) or 8 (RANSAC)"
            )));
        }
    };

    let f_arr = mat3_to_py(py, &f);
    let mask_arr = mask_to_py(py, &inliers);
    Ok((f_arr.unbind(), mask_arr.unbind()))
}
