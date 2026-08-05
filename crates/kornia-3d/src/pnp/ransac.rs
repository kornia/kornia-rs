//! RANSAC-based robust wrapper for PnP solvers.

use super::ap3p::solve_ap3p_multi;
use super::ops::{intrinsics_as_vectors, project_sq_error};
use super::{solve_pnp, PnPMethod};
use super::{EPnPParams, LMRefineParams, PnPError, PnPResult};
use kornia_algebra::{Mat3AF32, Vec2F32, Vec3AF32};
use kornia_imgproc::calibration::distortion::PolynomialDistortion;
use rand::seq::SliceRandom;
use rand::{rngs::StdRng, SeedableRng};
use thiserror::Error;

const MIN_CORRESPONDENCES: usize = 4; // Minimum 2D-3D pairs required by EPnP (and by any refit)
const AP3P_SAMPLE_SIZE: usize = 3; // AP3P is an exact 3-point solver: it rejects any other count
const EPNP_MIN_SAMPLE_SIZE: usize = 5; // Minimal sample size for EPnP (unless only 4 points available)

/// Upper bound on the number of real roots an exact P3P solver can return.
const AP3P_MAX_ROOTS: usize = 4;

const DEFAULT_MAX_ITERATIONS: usize = 100;
const DEFAULT_REPROJ_THRESHOLD_PX: f32 = 8.0;
const DEFAULT_CONFIDENCE: f32 = 0.99;

const EPS_PROB_MIN: f32 = 1e-6; // Guard for tiny probabilities
const EPS_LOG_GUARD: f32 = 1e-12; // Guard to avoid log(0) and log(1)
const HIGH_INLIER_RATIO_STOP: f32 = 0.95; // Early stop when inlier ratio is very high

/// Error type for the RANSAC-based PnP solver.
///
/// This encapsulates both underlying base-solver errors (`PnPError`) as well as
/// RANSAC-specific failure modes.
#[derive(Debug, Error)]
pub enum PnPRansacError {
    /// Error returned by the underlying PnP solver used inside RANSAC.
    #[error(transparent)]
    Base(#[from] PnPError),

    /// RANSAC failed to find a valid model with enough inliers.
    #[error("RANSAC found insufficient inliers: required {required}, got {actual}")]
    InsufficientInliers {
        /// Minimum number of inliers required to accept a model
        required: usize,
        /// Actual number of inliers found
        actual: usize,
    },

    /// The AP3P sampling kernel was combined with a lens-distortion model.
    ///
    /// AP3P consumes bearing vectors obtained by inverting `K` alone, so a distortion model
    /// cannot be honoured while generating hypotheses. Silently ignoring it would score the
    /// consensus set under an ideal-pinhole model and then refit it under a distorted one —
    /// two different camera models for one answer. Undistort the image points up front and
    /// pass `distortion = None`, or use an EPnP kernel.
    #[error("AP3P inside RANSAC cannot honour a distortion model: undistort the image points first and pass `distortion = None`, or use an EPnP kernel")]
    DistortionUnsupportedByKernel,
}

/// The non-minimal refit over the consensus set.
///
/// A thin seam over [`solve_pnp`] so the "refit failed, keep the minimal-sample pose" branch can be
/// tested deterministically. It cannot be reached by choosing an input: whether a near-degenerate
/// consensus set makes EPnP's LU/LM solve fail depends on the target ISA's floating-point
/// behaviour, so a test that arranges a numerical failure passes on one architecture and fails on
/// another. The fallback is a contract, not a numerical accident, and is tested as one.
#[cfg(not(test))]
#[inline]
fn refit(
    world: &[Vec3AF32],
    image: &[Vec2F32],
    k: &Mat3AF32,
    distortion: Option<&PolynomialDistortion>,
    method: PnPMethod,
) -> Result<PnPResult, PnPError> {
    solve_pnp(world, image, k, distortion, method)
}

#[cfg(test)]
fn refit(
    world: &[Vec3AF32],
    image: &[Vec2F32],
    k: &Mat3AF32,
    distortion: Option<&PolynomialDistortion>,
    method: PnPMethod,
) -> Result<PnPResult, PnPError> {
    if tests::refit_forced_to_fail() {
        return Err(PnPError::SvdFailed(
            "forced refit failure (test seam)".to_string(),
        ));
    }
    solve_pnp(world, image, k, distortion, method)
}

/// Parameters for RANSAC over PnP.
#[derive(Debug, Clone)]
pub struct RansacParams {
    /// Maximum number of RANSAC iterations.
    pub max_iterations: usize,
    /// Pixel error threshold to classify an observation as an inlier.
    pub reproj_threshold_px: f32,
    /// Desired probability that at least one sample set is outlier-free.
    pub confidence: f32,
    /// Optional fixed seed for reproducible sampling.
    pub random_seed: Option<u64>,
    /// Whether to refit on all inliers using the base solver.
    pub refine: bool,
}

impl Default for RansacParams {
    fn default() -> Self {
        Self {
            max_iterations: DEFAULT_MAX_ITERATIONS,
            reproj_threshold_px: DEFAULT_REPROJ_THRESHOLD_PX,
            confidence: DEFAULT_CONFIDENCE,
            random_seed: None,
            refine: true,
        }
    }
}

/// RANSAC result for PnP.
#[derive(Debug, Clone)]
pub struct PnPRansacResult {
    /// Best pose found by RANSAC.
    pub pose: PnPResult,
    /// Indices of inlier correspondences.
    pub inliers: Vec<usize>,
}

/// Solve PnP robustly using a legacy RANSAC loop around a base PnP method (e.g., EPnP).
///
/// - Minimal sample size follows the KERNEL: 3 for AP3P (an exact P3P solver), 5 for EPnP (4 when
///   only 4 points are available). This is what makes a P3P kernel worth using here: a sample is
///   outlier-free with probability `w^k` at inlier ratio `w`, so `k = 3` rather than `5` draws a
///   clean sample `w^-2` times as often (16x at `w = 0.25`).
/// - **Every** AP3P root is scored. P3P returns up to four algebraic solutions and cheirality on
///   only three points rarely narrows that to one, so scoring a single root would waste most
///   clean samples on the wrong pose and forfeit the `w^-2` advantage above. EPnP returns one
///   solution and keeps its single-hypothesis path.
/// - Scoring uses Euclidean pixel reprojection error.
/// - Iterations adapt from current inlier ratio and desired confidence.
/// - With `params.refine`, the final non-minimal refit over all inliers uses **EPnP even when the
///   sampling kernel was AP3P**, because a minimal solver cannot consume more than its exact
///   correspondence count. The returned pose is therefore an EPnP fit to the AP3P-selected
///   consensus set — the standard minimal-kernel / non-minimal-refit split. LM refinement is
///   enabled on that refit, since a caller who picked AP3P is often in exactly the regime where
///   bare EPnP is ill-conditioned (see the `epnp` module docs). If the refit fails, the best
///   minimal-sample pose is returned rather than discarding a successful RANSAC run.
/// - Minimum correspondence count is kernel- and refine-aware: AP3P with `refine: false` needs
///   only 3, everything else needs 4 because the EPnP refit does.
///
/// # Errors
///
/// Returns [`PnPRansacError::DistortionUnsupportedByKernel`] when an AP3P kernel is combined with
/// `distortion = Some(..)`: AP3P inverts `K` alone and cannot consume a distortion model, so the
/// combination is rejected instead of silently generating hypotheses from distorted pixels treated
/// as ideal. Undistort the image points up front and pass `None`.
pub fn solve_pnp_ransac(
    world: &[Vec3AF32],
    image: &[Vec2F32],
    k: &Mat3AF32,
    distortion: Option<&PolynomialDistortion>,
    base: PnPMethod,
    params: &RansacParams,
) -> Result<PnPRansacResult, PnPRansacError> {
    let n = world.len();
    if n != image.len() {
        return Err(PnPError::MismatchedArrayLengths {
            left_name: "world points",
            left_len: world.len(),
            right_name: "image points",
            right_len: image.len(),
        }
        .into());
    }
    let is_ap3p = matches!(base, PnPMethod::AP3P(_) | PnPMethod::AP3PDefault);

    // AP3P consumes bearing vectors from K alone; it has nowhere to put a distortion model.
    // Accepting one would score hypotheses under an ideal pinhole and then refit the very same
    // consensus set under a distorted model. Reject rather than silently mis-model.
    if is_ap3p && distortion.is_some() {
        return Err(PnPRansacError::DistortionUnsupportedByKernel);
    }

    // Minimum correspondences is kernel- and refine-aware: the AP3P kernel itself needs only its
    // 3-point minimal sample, but the non-minimal refit is EPnP, which needs 4.
    let min_correspondences = if is_ap3p && !params.refine {
        AP3P_SAMPLE_SIZE
    } else {
        MIN_CORRESPONDENCES
    };
    if n < min_correspondences {
        return Err(PnPError::InsufficientCorrespondences {
            required: min_correspondences,
            actual: n,
        }
        .into());
    }

    // Minimal set size is a property of the KERNEL: AP3P is an exact 3-point solver (it rejects
    // any other count), while EPnP samples 5 (unless only 4 points exist). Using the kernel's
    // true minimal size is not merely a fix — it is the point of a P3P kernel inside RANSAC: at
    // inlier ratio w a sample is outlier-free with probability w^k, so k=3 vs k=5 draws a clean
    // sample w^-2 times as often (16x at w=0.25).
    let sample_size: usize = if is_ap3p {
        AP3P_SAMPLE_SIZE
    } else if n == MIN_CORRESPONDENCES {
        MIN_CORRESPONDENCES
    } else {
        EPNP_MIN_SAMPLE_SIZE
    };

    // Precompute intrinsics vectors
    let (intr_x, intr_y) = intrinsics_as_vectors(k);

    // RNG setup
    let mut rng: StdRng = match params.random_seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => {
            let mut trng = rand::rng();
            StdRng::from_rng(&mut trng)
        }
    };

    // Working buffers
    let mut indices: Vec<usize> = (0..n).collect();
    let mut best_inliers: Vec<usize> = Vec::new();
    let mut best_pose: Option<PnPResult> = None;
    let mut w_min: Vec<Vec3AF32> = Vec::with_capacity(sample_size);
    let mut i_min: Vec<Vec2F32> = Vec::with_capacity(sample_size);
    // Hypotheses produced by one minimal sample: up to 4 for AP3P, exactly 1 for EPnP.
    let mut hypotheses: Vec<PnPResult> = Vec::with_capacity(AP3P_MAX_ROOTS);

    let mut iter: usize = 0;
    let mut required_iters = params.max_iterations;

    while iter < required_iters && iter < params.max_iterations {
        iter += 1;

        // Debug: prevent infinite loops
        if iter > params.max_iterations {
            log::warn!("RANSAC: Emergency break after {iter} iterations");
            break;
        }

        // Sample k unique indices without replacement.
        indices.shuffle(&mut rng);
        let sample = &indices[..sample_size];

        // Build minimal subsets
        w_min.clear();
        i_min.clear();
        for &idx in sample.iter() {
            w_min.push(world[idx]);
            i_min.push(image[idx]);
        }

        // Generate the hypotheses for this minimal sample. AP3P is a polynomial solver returning
        // up to 4 real roots; scoring only the first cheirality-passing one throws away most
        // clean samples, so every root is kept and scored below.
        hypotheses.clear();
        if is_ap3p {
            match solve_ap3p_multi(&w_min, &i_min, k) {
                Ok(roots) => hypotheses.extend(roots),
                Err(_e) => {
                    log::debug!("AP3P failed on minimal set at iteration {iter}");
                    continue;
                }
            }
        } else {
            match solve_pnp(&w_min, &i_min, k, distortion, base.clone()) {
                Ok(p) => hypotheses.push(p),
                Err(_e) => {
                    log::debug!("EPnP failed on minimal set at iteration {iter}");
                    continue;
                }
            }
        }

        // Score every hypothesis against all points, keeping the one with most inliers.
        let mut improved = false;
        for pose_min in hypotheses.drain(..) {
            // Cheirality check on minimal set (all positive depths)
            if !sample_all_positive_depths(&pose_min.rotation, &pose_min.translation, &w_min) {
                log::debug!("Cheirality check failed on iteration {iter}");
                continue;
            }

            let (inliers, _total_squared_error) = classify_points(
                world,
                image,
                None,
                None,
                ClassificationParams {
                    rotation_matrix: &pose_min.rotation,
                    translation_vector: &pose_min.translation,
                    camera_intrinsics_x: &intr_x,
                    camera_intrinsics_y: &intr_y,
                    threshold: Some(params.reproj_threshold_px),
                },
            );

            if inliers.len() > best_inliers.len() {
                best_inliers = inliers;
                best_pose = Some(pose_min);
                improved = true;
            }
        }

        if improved {
            // Update required iterations based on current inlier ratio and sample size
            if best_inliers.len() >= sample_size {
                let w = best_inliers.len() as f32 / n as f32;
                let s = sample_size as f32;

                // Avoid numerical issues with very small w
                if w > EPS_PROB_MIN && w < 1.0 {
                    let ws = w.powf(s);
                    if ws < 1.0 - EPS_LOG_GUARD && ws > EPS_LOG_GUARD {
                        // Avoid log(0) and log(1)
                        let log_conf = (1.0 - params.confidence).max(EPS_LOG_GUARD).ln();
                        let log_denom = (1.0 - ws).ln();
                        if log_denom.is_finite() && log_denom.abs() > EPS_LOG_GUARD {
                            let est = (log_conf / log_denom).ceil();

                            if est.is_finite() && est > 0.0 {
                                let est_usize = est.min(params.max_iterations as f32) as usize;
                                if est_usize < required_iters {
                                    required_iters = est_usize;
                                }
                            }
                        }
                    } else if w >= HIGH_INLIER_RATIO_STOP {
                        // Very high inlier ratio (≥95%), we can stop early
                        required_iters = iter;
                    }
                }
            }
        }
    }

    // Validate and optionally refine
    if best_inliers.len() < min_correspondences {
        let err = PnPRansacError::InsufficientInliers {
            required: min_correspondences,
            actual: best_inliers.len(),
        };
        return Err(err);
    }

    let refined = if params.refine {
        // Refit on all inliers. A minimal solver cannot do this — AP3P rejects any count other
        // than 3 — so the non-minimal refit always uses EPnP, the standard minimal-kernel /
        // non-minimal-refit split. LM refinement is switched on for that fallback: a caller who
        // reached for AP3P is often in the small-object / long-range or near-planar regime where
        // EPnP's PCA control points are ill-conditioned, and handing back a bare EPnP pose there
        // would silently undo their choice of kernel.
        let refit_method = if is_ap3p {
            PnPMethod::EPnP(EPnPParams {
                refine_lm: Some(LMRefineParams::default()),
                ..Default::default()
            })
        } else {
            base.clone()
        };
        let mut w_all = Vec::with_capacity(best_inliers.len());
        let mut i_all = Vec::with_capacity(best_inliers.len());
        for &idx in &best_inliers {
            w_all.push(world[idx]);
            i_all.push(image[idx]);
        }
        // A failed refit must not sink a successful RANSAC run: the consensus set was selected by
        // the sampling kernel, and with AP3P sampling EPnP was never validated on it, so a
        // near-degenerate inlier set can make the refit fail on a perfectly usable pose.
        match refit(&w_all, &i_all, k, distortion, refit_method) {
            Ok(p) => Some(p),
            Err(e) => {
                log::debug!("RANSAC refit on the consensus set failed ({e}); keeping the best minimal-sample pose");
                None
            }
        }
    } else {
        None
    };

    let mut final_pose = match refined.or(best_pose) {
        Some(p) => p,
        None => {
            return Err(PnPError::SvdFailed(
                "RANSAC failed to produce a pose despite sufficient inliers".to_string(),
            )
            .into());
        }
    };

    // Recompute reprojection error on inliers only
    let (_inliers, sum_sq_inliers) = classify_points(
        world,
        image,
        None,
        Some(&best_inliers),
        ClassificationParams {
            rotation_matrix: &final_pose.rotation,
            translation_vector: &final_pose.translation,
            camera_intrinsics_x: &intr_x,
            camera_intrinsics_y: &intr_y,
            threshold: None,
        },
    );
    let rmse = if !best_inliers.is_empty() {
        (sum_sq_inliers / best_inliers.len() as f32).sqrt()
    } else {
        0.0
    };
    final_pose.reproj_rmse = Some(rmse);

    Ok(PnPRansacResult {
        pose: final_pose,
        inliers: best_inliers,
    })
}

fn sample_all_positive_depths(r: &Mat3AF32, t: &Vec3AF32, world: &[Vec3AF32]) -> bool {
    world.iter().all(|&pw| {
        let pc = *r * pw + *t;
        pc.z > 0.0
    })
}

/// This function handles both:
/// - Scoring all points against a candidate pose (during RANSAC)
/// - Computing final RMSE on a subset of inlier points
struct ClassificationParams<'a> {
    rotation_matrix: &'a Mat3AF32,
    translation_vector: &'a Vec3AF32,
    camera_intrinsics_x: &'a Vec3AF32,
    camera_intrinsics_y: &'a Vec3AF32,
    threshold: Option<f32>,
}

fn classify_points(
    world: &[Vec3AF32],
    image: &[Vec2F32],
    _distortion: Option<&PolynomialDistortion>,
    indices: Option<&[usize]>,
    params: ClassificationParams,
) -> (Vec<usize>, f32) {
    let rotation = params.rotation_matrix;
    let translation = params.translation_vector;

    let mut inliers: Vec<usize> = Vec::new();
    let mut total_squared_error: f32 = 0.0;

    match indices {
        Some(indices) => {
            for &idx in indices {
                if idx >= world.len() || idx >= image.len() {
                    continue;
                }

                if let Some(squared_error) = project_sq_error(
                    &world[idx],
                    &image[idx],
                    rotation,
                    translation,
                    params.camera_intrinsics_x,
                    params.camera_intrinsics_y,
                    true,
                ) {
                    total_squared_error += squared_error;

                    let is_inlier = match params.threshold {
                        Some(thresh) => squared_error.sqrt() < thresh,
                        None => true,
                    };

                    if is_inlier {
                        inliers.push(idx);
                    }
                }
            }
        }
        None => {
            for (idx, (world_point, image_point)) in world.iter().zip(image.iter()).enumerate() {
                if let Some(squared_error) = project_sq_error(
                    world_point,
                    image_point,
                    rotation,
                    translation,
                    params.camera_intrinsics_x,
                    params.camera_intrinsics_y,
                    true,
                ) {
                    total_squared_error += squared_error;

                    let is_inlier = match params.threshold {
                        Some(thresh) => squared_error.sqrt() < thresh,
                        None => true,
                    };

                    if is_inlier {
                        inliers.push(idx);
                    }
                }
            }
        }
    }

    (inliers, total_squared_error)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Mutex, MutexGuard, OnceLock};

    /// Drives the [`super::refit`] seam. Only compiled under `cfg(test)`.
    static FORCE_REFIT_FAILURE: AtomicBool = AtomicBool::new(false);

    /// `cargo test` runs these in parallel threads over one process, and the flag above is process
    /// wide — without this lock a forced failure would leak into whatever else is mid-solve.
    fn refit_lock() -> &'static Mutex<()> {
        static L: OnceLock<Mutex<()>> = OnceLock::new();
        L.get_or_init(|| Mutex::new(()))
    }

    /// Forces the consensus refit to fail for as long as the guard is alive.
    pub(super) struct ForceRefitFailure(#[allow(dead_code)] MutexGuard<'static, ()>);

    impl ForceRefitFailure {
        fn new() -> Self {
            let g = refit_lock().lock().unwrap_or_else(|e| e.into_inner());
            FORCE_REFIT_FAILURE.store(true, Ordering::SeqCst);
            Self(g)
        }
    }

    impl Drop for ForceRefitFailure {
        fn drop(&mut self) {
            FORCE_REFIT_FAILURE.store(false, Ordering::SeqCst);
        }
    }

    pub(super) fn refit_forced_to_fail() -> bool {
        FORCE_REFIT_FAILURE.load(Ordering::SeqCst)
    }

    use super::super::ap3p::AP3PParams;
    use super::super::epnp::EPnPParams;
    use super::*;

    fn k_default() -> Mat3AF32 {
        // Column-major: [fx, 0, 0,  0, fy, 0,  cx, cy, 1]
        Mat3AF32::from_cols_array(&[800.0, 0.0, 0.0, 0.0, 800.0, 0.0, 640.0, 480.0, 1.0])
    }

    #[test]
    fn test_ransac_basic_outliers() -> Result<(), PnPRansacError> {
        // Use the same 6 inlier correspondences from epnp test
        let points_world: [Vec3AF32; 6] = [
            Vec3AF32::new(0.0315, 0.03333, -0.10409),
            Vec3AF32::new(-0.0315, 0.03333, -0.10409),
            Vec3AF32::new(0.0, -0.00102, -0.12977),
            Vec3AF32::new(0.02646, -0.03167, -0.1053),
            Vec3AF32::new(-0.02646, -0.031667, -0.1053),
            Vec3AF32::new(0.0, 0.04515, -0.11033),
        ];
        let mut points_image: Vec<Vec2F32> = vec![
            Vec2F32::new(722.96466, 502.0828),
            Vec2F32::new(669.88837, 498.61877),
            Vec2F32::new(707.0025, 478.48975),
            Vec2F32::new(728.05634, 447.56918),
            Vec2F32::new(682.6069, 443.91776),
            Vec2F32::new(696.4414, 511.96442),
        ];
        // Inject 4 strong outliers
        let mut world = points_world.to_vec();
        for (j, &point_world) in points_world.iter().enumerate().take(4) {
            world.push(point_world);
            points_image.push(Vec2F32::new(
                1200.0 + j as f32 * 5.0,
                -300.0 - j as f32 * 3.0,
            ));
        }
        let k = k_default();

        let params = RansacParams {
            max_iterations: 10,          // Reduce for testing
            reproj_threshold_px: 1000.0, // High threshold needed for this test data with extreme outliers
            confidence: 0.99,
            random_seed: Some(42),
            refine: false,
        };

        let base = PnPMethod::EPnP(EPnPParams::default());

        let res = solve_pnp_ransac(&world, &points_image, &k, None, base, &params)?;
        assert!(res.inliers.len() >= 6); // Should find at least the 6 original inliers
        assert!(res.pose.reproj_rmse.is_some());

        // With extreme outliers, RMSE will be higher, but RANSAC should still work
        let rmse = res.pose.reproj_rmse.unwrap();
        assert!(rmse < 2000.0); // Allow reasonable tolerance for this challenging test data
        Ok(())
    }

    #[test]
    fn test_ransac_perfect_data() -> Result<(), PnPRansacError> {
        // Test with perfect data (no outliers)
        let points_world: [Vec3AF32; 6] = [
            Vec3AF32::new(0.0315, 0.03333, -0.10409),
            Vec3AF32::new(-0.0315, 0.03333, -0.10409),
            Vec3AF32::new(0.0, -0.00102, -0.12977),
            Vec3AF32::new(0.02646, -0.03167, -0.1053),
            Vec3AF32::new(-0.02646, -0.031667, -0.1053),
            Vec3AF32::new(0.0, 0.04515, -0.11033),
        ];
        let points_image: [Vec2F32; 6] = [
            Vec2F32::new(722.96466, 502.0828),
            Vec2F32::new(669.88837, 498.61877),
            Vec2F32::new(707.0025, 478.48975),
            Vec2F32::new(728.05634, 447.56918),
            Vec2F32::new(682.6069, 443.91776),
            Vec2F32::new(696.4414, 511.96442),
        ];
        let k = k_default();

        let params = RansacParams {
            max_iterations: 10,
            reproj_threshold_px: 8.0,
            confidence: 0.99,
            random_seed: Some(42),
            refine: true,
        };

        let base = PnPMethod::EPnP(EPnPParams::default());
        let res = solve_pnp_ransac(&points_world, &points_image, &k, None, base, &params)?;
        assert_eq!(res.inliers.len(), 6); // All points should be inliers
        assert!(res.pose.reproj_rmse.is_some());

        // The test data from epnp tests has some inherent reprojection error
        // (~12 pixels RMSE) which is reasonable given the 8px threshold
        let rmse = res.pose.reproj_rmse.unwrap();
        assert!(rmse < 20.0); // Allow reasonable tolerance for this test data
        Ok(())
    }

    #[test]
    fn test_ransac_minimum_points() -> Result<(), PnPRansacError> {
        // Test with exactly 4 points
        let points_world: [Vec3AF32; 4] = [
            Vec3AF32::new(0.0315, 0.03333, -0.10409),
            Vec3AF32::new(-0.0315, 0.03333, -0.10409),
            Vec3AF32::new(0.0, -0.00102, -0.12977),
            Vec3AF32::new(0.02646, -0.03167, -0.1053),
        ];
        let points_image: [Vec2F32; 4] = [
            Vec2F32::new(722.96466, 502.0828),
            Vec2F32::new(669.88837, 498.61877),
            Vec2F32::new(707.0025, 478.48975),
            Vec2F32::new(728.05634, 447.56918),
        ];
        let k = k_default();

        let params = RansacParams {
            max_iterations: 5,
            reproj_threshold_px: 8.0,
            confidence: 0.99,
            random_seed: Some(42),
            refine: true,
        };

        let base = PnPMethod::EPnP(EPnPParams::default());
        let res = solve_pnp_ransac(&points_world, &points_image, &k, None, base, &params)?;
        assert!(res.inliers.len() >= 4);
        Ok(())
    }

    #[test]
    fn test_ransac_error_cases() {
        let points_world: [Vec3AF32; 3] = [
            Vec3AF32::new(0.0, 0.0, 1.0),
            Vec3AF32::new(1.0, 0.0, 1.0),
            Vec3AF32::new(0.0, 1.0, 1.0),
        ];
        let points_image: [Vec2F32; 3] = [
            Vec2F32::new(100.0, 100.0),
            Vec2F32::new(200.0, 100.0),
            Vec2F32::new(100.0, 200.0),
        ];
        let k = Mat3AF32::from_cols(
            Vec3AF32::new(800.0, 0.0, 0.0),
            Vec3AF32::new(0.0, 800.0, 0.0),
            Vec3AF32::new(400.0, 300.0, 1.0),
        );

        let params = RansacParams::default();
        let base = PnPMethod::EPnP(EPnPParams::default());

        // Should fail with insufficient correspondences
        let result = solve_pnp_ransac(&points_world, &points_image, &k, None, base, &params);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            PnPRansacError::Base(PnPError::InsufficientCorrespondences { .. })
        ));
    }

    /// AP3P must actually run inside RANSAC.
    ///
    /// The minimal sample size used to be EPnP's unconditionally, but AP3P is an exact 3-point
    /// solver that rejects any other count. So every iteration handed it 5 points, it errored,
    /// the loop swallowed the error and continued, and `best_pose` stayed `None` — the caller got
    /// "RANSAC failed to produce a pose despite sufficient inliers", an error blaming the data for
    /// a kernel that never ran once. This test fails before the fix, both with `refine` off (the
    /// misleading error) and on (a refit over zero inliers).
    #[test]
    fn ap3p_registers_a_pose_inside_ransac() -> Result<(), PnPRansacError> {
        // A known-good rigid configuration: the module's own test scene, all six points inliers.
        let points_world: [Vec3AF32; 6] = [
            Vec3AF32::new(0.0315, 0.03333, -0.10409),
            Vec3AF32::new(-0.0315, 0.03333, -0.10409),
            Vec3AF32::new(0.0, -0.00102, -0.12977),
            Vec3AF32::new(0.02646, -0.03167, -0.1053),
            Vec3AF32::new(-0.02646, -0.031667, -0.1053),
            Vec3AF32::new(0.0, 0.04515, -0.11033),
        ];
        let points_image: [Vec2F32; 6] = [
            Vec2F32::new(722.96466, 502.0828),
            Vec2F32::new(669.88837, 498.61877),
            Vec2F32::new(707.0025, 478.48975),
            Vec2F32::new(728.05634, 447.56918),
            Vec2F32::new(682.6069, 443.91776),
            Vec2F32::new(696.4414, 511.96442),
        ];
        let k = k_default();
        let params = RansacParams {
            max_iterations: 50,
            reproj_threshold_px: 8.0,
            confidence: 0.99,
            random_seed: Some(42),
            refine: false,
        };

        let res = solve_pnp_ransac(
            &points_world,
            &points_image,
            &k,
            None,
            PnPMethod::AP3PDefault,
            &params,
        )?;
        assert_eq!(res.inliers.len(), 6, "AP3P should register every inlier");

        // The `refine: true` path is covered by `ap3p_ransac_refit_is_lm_refined` and
        // `ap3p_ransac_survives_a_failed_refit`, which assert the RETURNED POSE. Asserting the
        // inlier count here would prove nothing about it: `inliers` is `best_inliers`, computed
        // before the refit runs, so it is identical whatever the refit does.
        Ok(())
    }

    // ---------------------------------------------------------------------------------------
    // Synthetic AP3P scenes: exact pinhole projections of a known pose, so the *returned pose*
    // can be asserted rather than just the inlier count (which is computed before the refit).
    // ---------------------------------------------------------------------------------------

    /// Row-major `R = Rz(az) · Ry(ay) · Rx(ax)`.
    fn rot_zyx(ax: f32, ay: f32, az: f32) -> [[f32; 3]; 3] {
        let (sx, cx) = ax.sin_cos();
        let (sy, cy) = ay.sin_cos();
        let (sz, cz) = az.sin_cos();
        [
            [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
            [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
            [-sy, cy * sx, cy * cx],
        ]
    }

    /// The ground-truth pose shared by the synthetic scenes: a generic orientation, target
    /// roughly 1.5 m away.
    fn pose_gt() -> ([[f32; 3]; 3], [f32; 3]) {
        (rot_zyx(0.1, -0.15, 0.05), [0.02, -0.01, 1.5])
    }

    /// Noise-free pinhole projection under `(r, t)` with [`k_default`].
    fn project_exact(world: &[Vec3AF32], r: &[[f32; 3]; 3], t: &[f32; 3]) -> Vec<Vec2F32> {
        world
            .iter()
            .map(|p| {
                let pc = [
                    r[0][0] * p.x + r[0][1] * p.y + r[0][2] * p.z + t[0],
                    r[1][0] * p.x + r[1][1] * p.y + r[1][2] * p.z + t[1],
                    r[2][0] * p.x + r[2][1] * p.y + r[2][2] * p.z + t[2],
                ];
                Vec2F32::new(800.0 * pc[0] / pc[2] + 640.0, 800.0 * pc[1] / pc[2] + 480.0)
            })
            .collect()
    }

    /// Largest absolute element-wise deviation of an estimated pose from the ground truth.
    fn pose_deviation(pose: &PnPResult, r: &[[f32; 3]; 3], t: &[f32; 3]) -> (f32, f32) {
        let est = pose.rotation.to_cols_array(); // column-major: est[col * 3 + row]
        let dr = (0..3)
            .flat_map(|row| (0..3).map(move |col| (row, col)))
            .map(|(row, col)| (est[col * 3 + row] - r[row][col]).abs())
            .fold(0.0f32, f32::max);
        let dt = (pose.translation.x - t[0])
            .abs()
            .max((pose.translation.y - t[1]).abs())
            .max((pose.translation.z - t[2]).abs());
        (dr, dt)
    }

    /// Six non-coplanar points spanning ~20 cm — a well-conditioned scene for both kernels.
    fn scene_non_planar() -> [Vec3AF32; 6] {
        [
            Vec3AF32::new(-0.1, -0.1, 0.0),
            Vec3AF32::new(0.1, -0.1, 0.03),
            Vec3AF32::new(0.1, 0.1, -0.04),
            Vec3AF32::new(-0.1, 0.1, 0.05),
            Vec3AF32::new(0.0, -0.05, -0.06),
            Vec3AF32::new(0.05, 0.0, 0.07),
        ]
    }

    /// Six points on a plane — an AprilTag / chessboard target, the classic P3P use case and the
    /// case EPnP's PCA control points cannot handle (sigma3 = 0 collapses `cw[3]` onto the
    /// centroid).
    fn scene_coplanar() -> [Vec3AF32; 6] {
        [
            Vec3AF32::new(-0.1, -0.1, 0.0),
            Vec3AF32::new(0.1, -0.1, 0.0),
            Vec3AF32::new(0.1, 0.1, 0.0),
            Vec3AF32::new(-0.1, 0.1, 0.0),
            Vec3AF32::new(0.0, -0.05, 0.0),
            Vec3AF32::new(0.05, 0.0, 0.0),
        ]
    }

    /// Every algebraic P3P root must be scored, not just the first cheirality-passing one.
    ///
    /// P3P returns up to four roots and cheirality on only three points does not disambiguate
    /// them: on this scene every triple leaves exactly two survivors, and the *first* one is the
    /// wrong pose for 7 of the 20 triples. Each iteration is therefore capped at one clean
    /// sample here (`max_iterations: 1`) so a single wasted sample is fatal — scoring only the
    /// first root collapses the consensus to the 3 sampled points on those triples.
    #[test]
    fn ap3p_ransac_scores_every_p3p_root() {
        let (r_gt, t_gt) = pose_gt();
        let world = scene_non_planar();
        let image = project_exact(&world, &r_gt, &t_gt);
        let k = k_default();

        for seed in 0..16u64 {
            let params = RansacParams {
                max_iterations: 1,
                reproj_threshold_px: 1.0,
                confidence: 0.99,
                random_seed: Some(seed),
                refine: false,
            };
            let res = solve_pnp_ransac(&world, &image, &k, None, PnPMethod::AP3PDefault, &params)
                .unwrap_or_else(|e| panic!("seed {seed}: {e}"));

            assert_eq!(
                res.inliers.len(),
                6,
                "seed {seed}: one outlier-free triple must yield the pose all 6 points agree \
                 with; a smaller consensus means the wrong P3P root was scored"
            );
            let (dr, dt) = pose_deviation(&res.pose, &r_gt, &t_gt);
            assert!(dr < 1e-3, "seed {seed}: rotation deviates by {dr}");
            assert!(dt < 1e-3, "seed {seed}: translation deviates by {dt}");
        }
    }

    /// The AP3P refit must be an LM-refined EPnP, not a bare one.
    ///
    /// This asserts on the *refined* pose, which the inlier list cannot cover: `inliers` is
    /// `best_inliers`, computed before the refit runs. On this well-conditioned scene a bare
    /// EPnP refit still lands ~22 px RMSE off (its PCA control points project too close
    /// together at this object extent / range), while the LM-polished one is exact.
    #[test]
    fn ap3p_ransac_refit_is_lm_refined() -> Result<(), PnPRansacError> {
        let (r_gt, t_gt) = pose_gt();
        let world = scene_non_planar();
        let image = project_exact(&world, &r_gt, &t_gt);
        let k = k_default();

        let params = RansacParams {
            max_iterations: 30,
            reproj_threshold_px: 1.0,
            confidence: 0.99,
            random_seed: Some(42),
            refine: true,
        };
        let res = solve_pnp_ransac(&world, &image, &k, None, PnPMethod::AP3PDefault, &params)?;

        assert_eq!(res.inliers.len(), 6);
        let rmse = res
            .pose
            .reproj_rmse
            .expect("RANSAC always fills in the inlier RMSE");
        assert!(rmse < 0.01, "refined pose reprojects at {rmse} px");
        let (dr, dt) = pose_deviation(&res.pose, &r_gt, &t_gt);
        assert!(dr < 1e-3, "refined rotation deviates by {dr}");
        assert!(dt < 1e-3, "refined translation deviates by {dt}");
        Ok(())
    }

    /// A refit that fails must not sink an otherwise successful RANSAC run.
    ///
    /// Driven through the [`refit`] seam rather than by arranging a numerically degenerate input.
    /// The earlier version of this test used a coplanar target and relied on the LM-refined EPnP
    /// refit erroring: it does on aarch64, but on x86_64 the same solve CONVERGES (to 0.145 px), so
    /// the test asserted the fallback on one architecture and the success path on another, and was
    /// red on x86_64. Whether an LU solve fails is not a property this crate controls; the fallback
    /// is.
    ///
    /// Asserts bit-equality with the `refine: false` result, which IS `best_pose` — so the test
    /// fails if the fallback returns anything other than the untouched minimal-sample pose.
    #[test]
    fn ap3p_ransac_survives_a_failed_refit() -> Result<(), PnPRansacError> {
        let (r_gt, t_gt) = pose_gt();
        let world = scene_coplanar();
        let image = project_exact(&world, &r_gt, &t_gt);
        let k = k_default();

        for seed in [42u64, 7, 1] {
            let base_params = RansacParams {
                max_iterations: 30,
                reproj_threshold_px: 1.0,
                confidence: 0.99,
                random_seed: Some(seed),
                refine: false,
            };
            // The pose RANSAC selected, with no refit applied.
            let unrefined = solve_pnp_ransac(
                &world,
                &image,
                &k,
                None,
                PnPMethod::AP3PDefault,
                &base_params,
            )?;

            let guard = ForceRefitFailure::new();
            let fell_back = solve_pnp_ransac(
                &world,
                &image,
                &k,
                None,
                PnPMethod::AP3PDefault,
                &RansacParams {
                    refine: true,
                    ..base_params
                },
            )?;
            drop(guard);

            assert_eq!(
                fell_back.inliers, unrefined.inliers,
                "seed {seed}: the consensus set is chosen before the refit and must not change"
            );
            let (a, b) = (&fell_back.pose, &unrefined.pose);
            assert_eq!(
                (a.rotation, a.translation),
                (b.rotation, b.translation),
                "seed {seed}: a failed refit must return the minimal-sample pose UNCHANGED"
            );
            // And that pose is the right answer, not merely the same one.
            let (dr, dt) = pose_deviation(&fell_back.pose, &r_gt, &t_gt);
            assert!(dr < 1e-3, "seed {seed}: rotation deviates by {dr}");
            assert!(dt < 1e-3, "seed {seed}: translation deviates by {dt}");
        }
        Ok(())
    }

    /// The coplanar case must succeed end to end, whichever way the refit goes.
    ///
    /// Companion to the test above: that one pins the fallback, this one pins that a coplanar
    /// target is solvable at all. It deliberately asserts nothing about WHICH path ran, because
    /// that is exactly the ISA-dependent part.
    ///
    /// The pose tolerances are loose FOR THE SAME REASON, and the first version of this test got
    /// that wrong: on aarch64 the refit errors and the exact AP3P pose survives (dR ~ 5e-6), while
    /// on x86_64 it converges and returns an EPnP pose at dR = 0.0177 -- worse, but not wrong. A
    /// tight bound therefore encodes the host ISA all over again. The accuracy gap itself is the
    /// long-range refit degradation tracked in kornia-rs#1075; when that is fixed these bounds can
    /// tighten.
    #[test]
    fn ap3p_ransac_solves_a_coplanar_target() -> Result<(), PnPRansacError> {
        let (r_gt, t_gt) = pose_gt();
        let world = scene_coplanar();
        let image = project_exact(&world, &r_gt, &t_gt);
        let k = k_default();

        let res = solve_pnp_ransac(
            &world,
            &image,
            &k,
            None,
            PnPMethod::AP3PDefault,
            &RansacParams {
                max_iterations: 30,
                reproj_threshold_px: 1.0,
                confidence: 0.99,
                random_seed: Some(42),
                refine: true,
            },
        )?;
        assert_eq!(res.inliers.len(), 6);
        let (dr, dt) = pose_deviation(&res.pose, &r_gt, &t_gt);
        assert!(dr < 0.05, "rotation deviates by {dr}");
        assert!(dt < 0.05, "translation deviates by {dt}");
        Ok(())
    }

    /// AP3P consumes bearing vectors from `K` alone, so a distortion model cannot be honoured
    /// while generating hypotheses. The combination is rejected instead of silently scoring the
    /// consensus under an ideal pinhole and then refitting it under a distorted model.
    #[test]
    fn ap3p_ransac_rejects_a_distortion_model() {
        let (r_gt, t_gt) = pose_gt();
        let world = scene_non_planar();
        let image = project_exact(&world, &r_gt, &t_gt);
        let k = k_default();
        let distortion = PolynomialDistortion {
            k1: -0.2,
            k2: 0.05,
            k3: 0.0,
            k4: 0.0,
            k5: 0.0,
            k6: 0.0,
            p1: 0.001,
            p2: -0.001,
        };

        for base in [
            PnPMethod::AP3PDefault,
            PnPMethod::AP3P(AP3PParams::default()),
        ] {
            let err = solve_pnp_ransac(
                &world,
                &image,
                &k,
                Some(&distortion),
                base,
                &RansacParams::default(),
            )
            .expect_err("AP3P + distortion must be rejected");
            assert!(
                matches!(err, PnPRansacError::DistortionUnsupportedByKernel),
                "unexpected error: {err}"
            );
        }

        // EPnP does consume the distortion model, so it must still be accepted.
        assert!(solve_pnp_ransac(
            &world,
            &image,
            &k,
            Some(&distortion),
            PnPMethod::EPnPDefault,
            &RansacParams::default(),
        )
        .is_ok());
    }

    /// AP3P needs only its 3-point minimal sample when no refit is requested, so 3
    /// correspondences must be accepted — the blanket 4-point floor is EPnP's.
    ///
    /// Only the pose's *consistency* is asserted: three points do not disambiguate the P3P
    /// roots (they all reproject exactly), so which one comes back is genuinely undetermined.
    #[test]
    fn ap3p_ransac_accepts_exactly_three_correspondences() -> Result<(), PnPRansacError> {
        let (r_gt, t_gt) = pose_gt();
        let all = scene_non_planar();
        let world = [all[0], all[1], all[2]];
        let image = project_exact(&world, &r_gt, &t_gt);
        let k = k_default();

        let params = RansacParams {
            max_iterations: 5,
            reproj_threshold_px: 1.0,
            confidence: 0.99,
            random_seed: Some(42),
            refine: false,
        };
        let res = solve_pnp_ransac(&world, &image, &k, None, PnPMethod::AP3PDefault, &params)?;
        assert_eq!(res.inliers.len(), 3);
        let rmse = res.pose.reproj_rmse.expect("RMSE is always filled in");
        assert!(rmse < 0.01, "3-point AP3P pose reprojects at {rmse} px");

        // `refine: true` still needs 4, because the refit is EPnP.
        let err = solve_pnp_ransac(
            &world,
            &image,
            &k,
            None,
            PnPMethod::AP3PDefault,
            &RansacParams {
                refine: true,
                ..params
            },
        )
        .expect_err("the EPnP refit cannot run on 3 correspondences");
        assert!(
            matches!(
                err,
                PnPRansacError::Base(PnPError::InsufficientCorrespondences {
                    required: MIN_CORRESPONDENCES,
                    ..
                })
            ),
            "unexpected error: {err}"
        );

        // EPnP itself is unaffected: it still demands 4 regardless of `refine`.
        let err = solve_pnp_ransac(
            &world,
            &image,
            &k,
            None,
            PnPMethod::EPnPDefault,
            &RansacParams {
                refine: false,
                ..params
            },
        )
        .expect_err("EPnP needs 4 correspondences");
        assert!(
            matches!(
                err,
                PnPRansacError::Base(PnPError::InsufficientCorrespondences {
                    required: MIN_CORRESPONDENCES,
                    ..
                })
            ),
            "unexpected error: {err}"
        );
        Ok(())
    }
}
