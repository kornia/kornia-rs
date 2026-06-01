//! `kornia_rs.k3d.solve_pnp_ransac` — `cv2.solvePnPRansac`-shaped binding.
//!
//! Wraps `kornia_3d::ransac::run` over `EPnPEstimator` so the Python surface
//! matches OpenCV's calling convention while the underlying solver is the
//! generic kornia RANSAC driver (NEON/AVX2 scoring, adaptive iter cap).
//!
//! P3P kernel is queued as a follow-up; today's RANSAC kernel is EPnP.

use kornia_3d::ransac::{
    estimators::EPnPEstimator, run, Match2d3d, RansacConfig, ThresholdConsensus, UniformSampler,
};
use kornia_algebra::{Mat3AF32, Vec2F64, Vec3AF32, Vec3F64};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods, ToPyArray};
use pyo3::prelude::*;
use rand::{rngs::StdRng, SeedableRng};

/// `cv2.solvePnPRansac` analog. Recovers `(R, t)` from 3D-2D correspondences.
///
/// Args:
///     world: `(N, 3)` float64 world points.
///     image: `(N, 2)` float64 pixel observations (same length as world).
///     k: `(3, 3)` float64 row-major pinhole intrinsics.
///     threshold: reprojection-error threshold in pixels (default 4.0).
///     max_iterations: RANSAC iteration cap (default 1000).
///     confidence: target probability of an all-inlier sample (default 0.999).
///     seed: optional RNG seed for deterministic runs.
///
/// Returns:
///     `(R, t, inlier_mask, inlier_count)` where:
///     * `R` — `(3, 3)` float64 row-major rotation (world → camera).
///     * `t` — `(3,)` float64 translation in the camera frame.
///     * `inlier_mask` — `(N,)` uint8 per-correspondence mask (1 = inlier).
///     * `inlier_count` — total inliers (int).
///
/// Raises ValueError on shape mismatches or when RANSAC produces no model.
#[pyfunction(name = "solve_pnp_ransac")]
#[pyo3(signature = (world, image, k, threshold=4.0, max_iterations=1000, confidence=0.999, lo_every = 2, seed=None))]
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn solve_pnp_ransac_py<'py>(
    py: Python<'py>,
    world: Bound<'py, PyArray2<f64>>,
    image: Bound<'py, PyArray2<f64>>,
    k: Bound<'py, PyArray2<f64>>,
    threshold: f64,
    max_iterations: u32,
    confidence: f64,
    lo_every: u32,
    seed: Option<u64>,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<u8>>,
    usize,
)> {
    let world_shape = world.shape();
    let image_shape = image.shape();
    let k_shape = k.shape();
    if world_shape.len() != 2 || world_shape[1] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "world must be (N, 3) float64",
        ));
    }
    if image_shape.len() != 2 || image_shape[1] != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "image must be (N, 2) float64",
        ));
    }
    if world_shape[0] != image_shape[0] {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "world and image must have matching N",
        ));
    }
    if k_shape.len() != 2 || k_shape[0] != 3 || k_shape[1] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "k must be (3, 3) float64",
        ));
    }
    let n = world_shape[0];
    if n < 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "solve_pnp_ransac requires N >= 4 correspondences",
        ));
    }

    // SAFETY: only used to read; no aliasing concerns since we copy into samples below.
    let world_ro = unsafe { world.as_array() };
    let image_ro = unsafe { image.as_array() };
    let k_ro = unsafe { k.as_array() };

    let samples: Vec<Match2d3d> = (0..n)
        .map(|i| {
            Match2d3d::new(
                Vec3F64::new(world_ro[[i, 0]], world_ro[[i, 1]], world_ro[[i, 2]]),
                Vec2F64::new(image_ro[[i, 0]], image_ro[[i, 1]]),
            )
        })
        .collect();

    // Row-major numpy → column-major Mat3AF32: col j is K's column j.
    let k_mat = Mat3AF32::from_cols(
        Vec3AF32::new(
            k_ro[[0, 0]] as f32,
            k_ro[[1, 0]] as f32,
            k_ro[[2, 0]] as f32,
        ),
        Vec3AF32::new(
            k_ro[[0, 1]] as f32,
            k_ro[[1, 1]] as f32,
            k_ro[[2, 1]] as f32,
        ),
        Vec3AF32::new(
            k_ro[[0, 2]] as f32,
            k_ro[[1, 2]] as f32,
            k_ro[[2, 2]] as f32,
        ),
    );

    let est = EPnPEstimator::new(k_mat);
    let mut sampler = UniformSampler::new(StdRng::seed_from_u64(seed.unwrap_or(0)));
    let consensus = ThresholdConsensus { threshold };
    let cfg = RansacConfig {
        max_iters: max_iterations,
        confidence,
        inlier_threshold: threshold,
        lo_every,
        ..Default::default()
    };

    let result = run(&est, &consensus, &mut sampler, &samples, &cfg);
    let model = result.model.ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(
            "solve_pnp_ransac: no model recovered (check threshold + iter budget)",
        )
    })?;

    // Row-major repack of column-major Mat3AF32: row i = (col0[i], col1[i], col2[i]).
    let r_arr = model.rotation.to_cols_array();
    let r_rmaj: Vec<f64> = vec![
        r_arr[0] as f64,
        r_arr[3] as f64,
        r_arr[6] as f64,
        r_arr[1] as f64,
        r_arr[4] as f64,
        r_arr[7] as f64,
        r_arr[2] as f64,
        r_arr[5] as f64,
        r_arr[8] as f64,
    ];
    let r_py = numpy::PyArray::from_vec(py, r_rmaj).reshape([3usize, 3])?;
    let t_py = [
        model.translation.x as f64,
        model.translation.y as f64,
        model.translation.z as f64,
    ]
    .to_pyarray(py);
    let mask: Vec<u8> = result.inliers.iter().map(|&b| b as u8).collect();
    let mask_py = mask.to_pyarray(py);
    let inlier_count = result.inliers.iter().filter(|&&b| b).count();
    Ok((r_py, t_py, mask_py, inlier_count))
}
