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
    estimators::{EssentialEstimator, FundamentalEstimator, HomographyEstimator},
    run, Match2d2d, RansacConfig, ThresholdConsensus, UniformSampler,
};
use kornia_algebra::Vec2F64;
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
    let result = run(
        &EssentialEstimator,
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

// PnP RANSAC moved to `kornia_rs.k3d.solve_pnp_ransac` (see `crate::pnp`)
// to align with the OpenCV-shaped `find_fundamental` / `find_homography`
// neighbours. The lower-level generic-trait path stays accessible via
// the Rust API (`kornia_3d::ransac::run` + `EPnPEstimator`).

/// Build the `kornia_rs.ransac` submodule.
///
/// Two-view-only here; PnP/EPnP RANSAC has moved to
/// [`kornia_rs.k3d.solve_pnp_ransac`] to align with the OpenCV-shape
/// namespace convention used by `find_fundamental` / `find_homography`.
pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "ransac")?;
    m.add_class::<PyRansacTwoViewResult>()?;
    m.add_function(wrap_pyfunction!(fundamental, &m)?)?;
    m.add_function(wrap_pyfunction!(essential, &m)?)?;
    m.add_function(wrap_pyfunction!(homography, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}
