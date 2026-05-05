//! Python bindings for `kornia_3d::ransac`.
//!
//! Exposes the generic RANSAC machinery + the four built-in estimators
//! (Fundamental, Essential, Homography, EPnP) as plain functions returning
//! frozen result classes. The intent is `pytest-benchmark` parity with
//! `cv2.findFundamentalMat` / `findEssentialMat` / `findHomography` /
//! `solvePnPRansac`, so the Python signatures follow the OpenCV calling
//! convention as closely as makes sense.
//!
//! NumPy is the IO format: `(N, 4) float64` for two-view matches
//! (`x1.x, x1.y, x2.x, x2.y`), `(N, 5) float64` for PnP
//! (`world.x, world.y, world.z, image.x, image.y`).
//!
//! The result types are `#[pyclass(frozen)]` and own all their data — no
//! lifetimes leak across the FFI boundary.

use kornia_3d::ransac::{
    estimators::{EPnPEstimator, EssentialEstimator, FundamentalEstimator, HomographyEstimator},
    run, Match2d2d, Match2d3d, RansacConfig, ThresholdConsensus, UniformSampler,
};
use kornia_algebra::{Mat3AF32, Vec2F64, Vec3AF32, Vec3F64};
use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use rand::{rngs::StdRng, SeedableRng};

/// A 3×3 model (F / E / H) returned from a RANSAC run, plus its inlier mask.
#[pyclass(frozen, name = "RansacTwoViewResult", module = "kornia_rs.ransac")]
pub struct PyRansacTwoViewResult {
    /// Best-scoring 3×3 model (row-major, 9 floats).
    #[pyo3(get)]
    pub model: Option<Vec<f64>>,
    /// Inlier mask aligned to the input rows.
    #[pyo3(get)]
    pub inliers: Vec<bool>,
    /// Number of hypotheses evaluated.
    #[pyo3(get)]
    pub num_iters: u32,
    /// Best consensus score (higher is better).
    #[pyo3(get)]
    pub score: f64,
}

/// EPnP RANSAC result. Rotation is a row-major 3×3 in `model[..9]` and
/// translation is in `translation`.
#[pyclass(frozen, name = "RansacPnPResult", module = "kornia_rs.ransac")]
pub struct PyRansacPnPResult {
    /// Best-scoring rotation (row-major 3×3, 9 floats), or `None`.
    #[pyo3(get)]
    pub rotation: Option<Vec<f64>>,
    /// Translation vector (3 floats), or `None`.
    #[pyo3(get)]
    pub translation: Option<Vec<f64>>,
    /// Inlier mask.
    #[pyo3(get)]
    pub inliers: Vec<bool>,
    /// Number of hypotheses evaluated.
    #[pyo3(get)]
    pub num_iters: u32,
    /// Best consensus score.
    #[pyo3(get)]
    pub score: f64,
}

fn parse_two_view_matches(arr: PyReadonlyArray2<'_, f64>) -> PyResult<Vec<Match2d2d>> {
    let shape = arr.shape();
    if shape.len() != 2 || shape[1] != 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "expected (N, 4) float64 array of [x1.x, x1.y, x2.x, x2.y] rows",
        ));
    }
    let view = arr.as_array();
    Ok((0..shape[0])
        .map(|i| {
            Match2d2d::new(
                Vec2F64::new(view[[i, 0]], view[[i, 1]]),
                Vec2F64::new(view[[i, 2]], view[[i, 3]]),
            )
        })
        .collect())
}

fn make_cfg(threshold: f64, max_iters: u32, confidence: f64) -> RansacConfig {
    RansacConfig {
        max_iters,
        confidence,
        inlier_threshold: threshold,
        ..Default::default()
    }
}

fn mat3_to_row_major_vec(m: &kornia_algebra::Mat3F64) -> Vec<f64> {
    let cols: [f64; 9] = (*m).into(); // column-major
    vec![
        cols[0], cols[3], cols[6], cols[1], cols[4], cols[7], cols[2], cols[5], cols[8],
    ]
}

/// `cv2.findFundamentalMat` analog. Returns a `RansacTwoViewResult`.
#[pyfunction]
#[pyo3(signature = (matches, threshold=1.0, max_iters=1000, confidence=0.999, seed=None))]
pub fn fundamental(
    matches: PyReadonlyArray2<'_, f64>,
    threshold: f64,
    max_iters: u32,
    confidence: f64,
    seed: Option<u64>,
) -> PyResult<PyRansacTwoViewResult> {
    let samples = parse_two_view_matches(matches)?;
    let mut sampler = UniformSampler::new(StdRng::seed_from_u64(seed.unwrap_or(0)));
    let consensus = ThresholdConsensus { threshold };
    let cfg = make_cfg(threshold, max_iters, confidence);
    let result = run(
        &FundamentalEstimator,
        &consensus,
        &mut sampler,
        &samples,
        &cfg,
    );
    Ok(PyRansacTwoViewResult {
        model: result.model.as_ref().map(mat3_to_row_major_vec),
        inliers: result.inliers,
        num_iters: result.num_iters,
        score: result.score,
    })
}

/// `cv2.findEssentialMat` analog. Inputs are **normalised** correspondences
/// (apply `K⁻¹` before calling). Threshold is in normalised units.
#[pyfunction]
#[pyo3(signature = (matches, threshold=1e-4, max_iters=1000, confidence=0.999, seed=None))]
pub fn essential(
    matches: PyReadonlyArray2<'_, f64>,
    threshold: f64,
    max_iters: u32,
    confidence: f64,
    seed: Option<u64>,
) -> PyResult<PyRansacTwoViewResult> {
    let samples = parse_two_view_matches(matches)?;
    let mut sampler = UniformSampler::new(StdRng::seed_from_u64(seed.unwrap_or(0)));
    let consensus = ThresholdConsensus { threshold };
    let cfg = make_cfg(threshold, max_iters, confidence);
    let result = run(&EssentialEstimator, &consensus, &mut sampler, &samples, &cfg);
    Ok(PyRansacTwoViewResult {
        model: result.model.as_ref().map(mat3_to_row_major_vec),
        inliers: result.inliers,
        num_iters: result.num_iters,
        score: result.score,
    })
}

/// `cv2.findHomography` analog.
#[pyfunction]
#[pyo3(signature = (matches, threshold=4.0, max_iters=1000, confidence=0.999, seed=None))]
pub fn homography(
    matches: PyReadonlyArray2<'_, f64>,
    threshold: f64,
    max_iters: u32,
    confidence: f64,
    seed: Option<u64>,
) -> PyResult<PyRansacTwoViewResult> {
    let samples = parse_two_view_matches(matches)?;
    let mut sampler = UniformSampler::new(StdRng::seed_from_u64(seed.unwrap_or(0)));
    let consensus = ThresholdConsensus { threshold };
    let cfg = make_cfg(threshold, max_iters, confidence);
    let result = run(
        &HomographyEstimator,
        &consensus,
        &mut sampler,
        &samples,
        &cfg,
    );
    Ok(PyRansacTwoViewResult {
        model: result.model.as_ref().map(mat3_to_row_major_vec),
        inliers: result.inliers,
        num_iters: result.num_iters,
        score: result.score,
    })
}

/// `cv2.solvePnPRansac` analog (EPnP kernel).
///
/// `matches` is `(N, 5)` with rows `[Xw, Yw, Zw, u, v]`. `k` is `(3, 3)` row-major
/// pinhole intrinsics (no distortion supported in v1).
#[pyfunction]
#[pyo3(signature = (matches, k, threshold=4.0, max_iters=1000, confidence=0.999, seed=None))]
pub fn solve_pnp(
    py: Python<'_>,
    matches: PyReadonlyArray2<'_, f64>,
    k: PyReadonlyArray2<'_, f64>,
    threshold: f64,
    max_iters: u32,
    confidence: f64,
    seed: Option<u64>,
) -> PyResult<PyRansacPnPResult> {
    let _ = py;
    let shape = matches.shape();
    if shape.len() != 2 || shape[1] != 5 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "expected (N, 5) float64 array [Xw, Yw, Zw, u, v]",
        ));
    }
    let m = matches.as_array();
    let samples: Vec<Match2d3d> = (0..shape[0])
        .map(|i| {
            Match2d3d::new(
                Vec3F64::new(m[[i, 0]], m[[i, 1]], m[[i, 2]]),
                Vec2F64::new(m[[i, 3]], m[[i, 4]]),
            )
        })
        .collect();

    let kshape = k.shape();
    if kshape.len() != 2 || kshape[0] != 3 || kshape[1] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "expected (3, 3) float64 K matrix",
        ));
    }
    let kv = k.as_array();
    // Row-major numpy → column-major Mat3AF32: col 0 is the first column of K.
    let k_mat = Mat3AF32::from_cols(
        Vec3AF32::new(kv[[0, 0]] as f32, kv[[1, 0]] as f32, kv[[2, 0]] as f32),
        Vec3AF32::new(kv[[0, 1]] as f32, kv[[1, 1]] as f32, kv[[2, 1]] as f32),
        Vec3AF32::new(kv[[0, 2]] as f32, kv[[1, 2]] as f32, kv[[2, 2]] as f32),
    );
    let est = EPnPEstimator::new(k_mat);
    let mut sampler = UniformSampler::new(StdRng::seed_from_u64(seed.unwrap_or(0)));
    let consensus = ThresholdConsensus { threshold };
    let cfg = make_cfg(threshold, max_iters, confidence);
    let result = run(&est, &consensus, &mut sampler, &samples, &cfg);
    let (rot, tr) = match &result.model {
        Some(m) => {
            let r_arr = m.rotation.to_cols_array(); // column-major [c0.x..c2.z]
            // Row-major repack: row i = (col0[i], col1[i], col2[i])
            let rmaj = vec![
                r_arr[0] as f64, r_arr[3] as f64, r_arr[6] as f64,
                r_arr[1] as f64, r_arr[4] as f64, r_arr[7] as f64,
                r_arr[2] as f64, r_arr[5] as f64, r_arr[8] as f64,
            ];
            (
                Some(rmaj),
                Some(vec![
                    m.translation.x as f64,
                    m.translation.y as f64,
                    m.translation.z as f64,
                ]),
            )
        }
        None => (None, None),
    };
    Ok(PyRansacPnPResult {
        rotation: rot,
        translation: tr,
        inliers: result.inliers,
        num_iters: result.num_iters,
        score: result.score,
    })
}

/// Build the `kornia_rs.ransac` submodule.
pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "ransac")?;
    m.add_class::<PyRansacTwoViewResult>()?;
    m.add_class::<PyRansacPnPResult>()?;
    m.add_function(wrap_pyfunction!(fundamental, &m)?)?;
    m.add_function(wrap_pyfunction!(essential, &m)?)?;
    m.add_function(wrap_pyfunction!(homography, &m)?)?;
    m.add_function(wrap_pyfunction!(solve_pnp, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}
