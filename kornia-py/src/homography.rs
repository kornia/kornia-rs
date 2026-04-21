use numpy::{PyArray, PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;

use kornia_3d::pose::{
    homography_dlt as homography_dlt_fn, ransac_homography as ransac_homography_fn, RansacParams,
};
use kornia_algebra::{Mat3F64, Vec2F64};

/// Matches cv2.RANSAC numeric value for the `method` argument of find_homography.
const METHOD_RANSAC: i32 = 8;

/// Copy (N, 2) f64 numpy array into a Vec<Vec2F64>. Caller guarantees shape.
fn unpack_pts(arr: &Bound<'_, PyArray2<f64>>) -> Vec<Vec2F64> {
    let n = arr.shape()[0];
    unsafe {
        let raw = std::slice::from_raw_parts(arr.data(), n * 2);
        (0..n)
            .map(|i| Vec2F64::new(raw[i * 2], raw[i * 2 + 1]))
            .collect()
    }
}

/// Pack a Mat3F64 into a row-major (3, 3) numpy array.
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

    let n = s1[0];
    let x1 = unpack_pts(&pts1);
    let x2 = unpack_pts(&pts2);

    let params = RansacParams {
        max_iterations,
        threshold,
        min_inliers,
        random_seed: seed,
    };

    let result = py
        .detach(|| ransac_homography_fn(&x1, &x2, &params))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;

    let h_arr = mat3_to_py(py, &result.model);
    let mask_arr = unsafe {
        let arr = PyArray::<u8, _>::new(py, [n], false);
        let slice = std::slice::from_raw_parts_mut(arr.data(), n);
        for (i, b) in result.inliers.iter().enumerate() {
            slice[i] = if *b { 1 } else { 0 };
        }
        arr
    };

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

    // Method 0 = direct DLT over all points (cv2.findHomography(method=0)).
    // Method 8 = RANSAC (cv2.RANSAC == 8) with a DLT refit on the inlier set.
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
            };
            let (h, inl) = py
                .detach(|| -> Result<(Mat3F64, Vec<bool>), String> {
                    let res = ransac_homography_fn(&x1, &x2, &params)
                        .map_err(|e| format!("{}", e))?;
                    // LO-RANSAC: refit H via DLT across ALL inliers, then keep
                    // whichever model (RANSAC-minimal or refit) has lower inlier
                    // reprojection error. Plain RANSAC picks H from 4 points that
                    // happen to bracket many inliers — those 4 may not be a
                    // well-conditioned basis for H. Refitting on the full inlier
                    // set closes most of the gap to cv2.findHomography.
                    let inl_x1: Vec<Vec2F64> = x1
                        .iter()
                        .zip(res.inliers.iter())
                        .filter_map(|(p, k)| if *k { Some(*p) } else { None })
                        .collect();
                    let inl_x2: Vec<Vec2F64> = x2
                        .iter()
                        .zip(res.inliers.iter())
                        .filter_map(|(p, k)| if *k { Some(*p) } else { None })
                        .collect();
                    if inl_x1.len() >= 4 {
                        if let Ok(h_refit) = homography_dlt_fn(&inl_x1, &inl_x2) {
                            let score_refit = score_model(&h_refit, &x1, &x2, &res.inliers);
                            if score_refit < res.score {
                                return Ok((h_refit, res.inliers));
                            }
                        }
                    }
                    Ok((res.model, res.inliers))
                })
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
            (h, inl)
        }
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "unsupported method {method} — use 0 (DLT) or 8 (RANSAC)"
            )));
        }
    };

    let h_arr = mat3_to_py(py, &h);
    let mask_arr = unsafe {
        let arr = PyArray::<u8, _>::new(py, [n], false);
        let slice = std::slice::from_raw_parts_mut(arr.data(), n);
        for (i, b) in inliers.iter().enumerate() {
            slice[i] = if *b { 1 } else { 0 };
        }
        arr
    };
    Ok((h_arr.unbind(), mask_arr.unbind()))
}

/// Sum of squared reprojection errors for an H applied to a set of correspondences,
/// restricted to indices where `mask[i] == true`. Mirrors `homography_reproj_error`
/// from kornia-3d but inlined here because that helper is private to twoview.rs.
fn score_model(h: &Mat3F64, x1: &[Vec2F64], x2: &[Vec2F64], mask: &[bool]) -> f64 {
    let cols = h.to_cols_array();
    // column-major: H[r][c] = cols[c * 3 + r].
    let mut score = 0.0;
    for i in 0..x1.len() {
        if !mask[i] {
            continue;
        }
        let (u, v) = (x1[i].x, x1[i].y);
        let w = cols[2] * u + cols[5] * v + cols[8];
        let px = (cols[0] * u + cols[3] * v + cols[6]) / w;
        let py = (cols[1] * u + cols[4] * v + cols[7]) / w;
        let dx = px - x2[i].x;
        let dy = py - x2[i].y;
        score += dx * dx + dy * dy;
    }
    score
}
