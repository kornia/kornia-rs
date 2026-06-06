//! `kornia_rs.k3d.solve_pnp_ransac` — `cv2.solvePnPRansac`-shaped binding.
//!
//! Wraps `kornia_3d::ransac::run` over `EPnPEstimator` or `AP3PEstimator` so the Python
//! surface matches OpenCV's calling convention while the underlying solver is the
//! generic kornia RANSAC driver (NEON/AVX2 scoring, adaptive iter cap).

use kornia_3d::pnp::refine::{refine_pose_lm, LMRefineParams};
use kornia_3d::ransac::{
    estimators::{AP3PEstimator, EPnPEstimator},
    run, Match2d3d, RansacConfig, ThresholdConsensus, UniformSampler,
};
use kornia_algebra::{Mat3AF32, Vec2F32, Vec2F64, Vec3AF32, Vec3F64};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods, ToPyArray};
use pyo3::prelude::*;
use rand::{rngs::StdRng, SeedableRng};

/// Defines the underlying algebraic kernel used by the PnP RANSAC estimator.
#[pyclass(eq, eq_int, from_py_object)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PnPSolverMethod {
    /// Efficient Perspective-n-Point (requires N >= 4)
    EPnP = 0,
    /// Algebraic Perspective-3-Point (requires N >= 3)
    AP3P = 1,
}

/// `cv2.solvePnPRansac` analog. Recovers `(R, t)` from 3D-2D correspondences.
///
/// Args:
///     world: `(N, 3)` float64 world points.
///     image: `(N, 2)` float64 pixel observations (same length as world).
///     k: `(3, 3)` float64 row-major pinhole intrinsics.
///     method: `PnPSolverMethod` to use (EPnP or AP3P). Default is EPnP.
///     threshold: reprojection-error threshold in pixels (default 4.0).
///     max_iterations: RANSAC iteration cap (default 1000).
///     confidence: target probability of an all-inlier sample (default 0.999).
///     lo_every: frequency of local optimization passes. 0 means disabled (default 0).
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
#[pyo3(signature = (world, image, k, method=PnPSolverMethod::EPnP, threshold=4.0, max_iterations=1000, confidence=0.999, lo_every=0, seed=None))]
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn solve_pnp_ransac_py<'py>(
    py: Python<'py>,
    world: Bound<'py, PyArray2<f64>>,
    image: Bound<'py, PyArray2<f64>>,
    k: Bound<'py, PyArray2<f64>>,
    method: PnPSolverMethod,
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

    // Dynamically check the minimal sample constraints based on the chosen algorithm
    match method {
        PnPSolverMethod::AP3P if n < 3 => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "AP3P requires N >= 3 correspondences",
            ));
        }
        PnPSolverMethod::EPnP if n < 4 => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "EPnP requires N >= 4 correspondences",
            ));
        }
        _ => {}
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

    let threshold_sq = threshold * threshold;

    let mut sampler = UniformSampler::new(StdRng::seed_from_u64(seed.unwrap_or(0)));

    let consensus = ThresholdConsensus {
        threshold: threshold_sq,
    };
    let cfg = RansacConfig {
        max_iters: max_iterations,
        confidence,
        inlier_threshold: threshold_sq,
        lo_every,
        ..Default::default()
    };

    // Helper closure to process output and prevent code duplication between match arms
    let process_model =
        |rotation: Mat3AF32, translation: Vec3AF32, inliers: &[bool]| -> PyResult<_> {
            // Row-major repack of column-major Mat3AF32: row i = (col0[i], col1[i], col2[i]).
            let r_arr = rotation.to_cols_array();
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
                translation.x as f64,
                translation.y as f64,
                translation.z as f64,
            ]
            .to_pyarray(py);

            let mask: Vec<u8> = inliers.iter().map(|&b| b as u8).collect();
            let mask_py = mask.to_pyarray(py);
            let inlier_count = inliers.iter().filter(|&&b| b).count();

            Ok((r_py, t_py, mask_py, inlier_count))
        };

    // 1. Execute RANSAC and extract the base model + inlier mask
    let (mut final_rot, mut final_trans, inliers) = match method {
        PnPSolverMethod::EPnP => {
            let est = EPnPEstimator::new(k_mat);
            let result = run(&est, &consensus, &mut sampler, &samples, &cfg);
            let model = result.model.ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "solve_pnp_ransac (EPnP): no model recovered (check threshold + iter budget)",
                )
            })?;
            (model.rotation, model.translation, result.inliers)
        }
        PnPSolverMethod::AP3P => {
            let est = AP3PEstimator::new(k_mat);
            let result = run(&est, &consensus, &mut sampler, &samples, &cfg);
            let model = result.model.ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "solve_pnp_ransac (AP3P): no model recovered (check threshold + iter budget)",
                )
            })?;
            (model.rotation, model.translation, result.inliers)
        }
    };

    // 2. --- THE TERMINAL POLISH ---
    // Extract the 3D and 2D points directly from our Match2d3d `samples` array
    let mut inlier_world_pts = Vec::with_capacity(samples.len());
    let mut inlier_image_pts = Vec::with_capacity(samples.len());

    for (i, &is_inlier) in inliers.iter().enumerate() {
        if is_inlier {
            inlier_world_pts.push(Vec3AF32::new(
                samples[i].object.x as f32,
                samples[i].object.y as f32,
                samples[i].object.z as f32,
            ));
            inlier_image_pts.push(Vec2F32::new(
                samples[i].image.x as f32,
                samples[i].image.y as f32,
            ));
        }
    }

    // Only run LM if we have an overdetermined system (N >= 4)
    if inlier_world_pts.len() >= 4 {
        let refine_params = LMRefineParams::default()
            .with_max_iterations(20)
            .with_cost_tolerance(1e-6);

        // Run the Image-Space 2D reprojection minimizer
        if let Ok(refined) = refine_pose_lm(
            &inlier_world_pts,
            &inlier_image_pts,
            &k_mat,
            &final_rot,
            &final_trans,
            None, // Distortion is safely bypassed here
            &refine_params,
        ) {
            // Overwrite the algebraic pose with the polished geometric pose
            final_rot = refined.rotation;
            final_trans = refined.translation;
        }
    }

    // 3. Return the polished data to Python
    process_model(final_rot, final_trans, &inliers)
}
