//! AP3P (Algebraic Perspective-3-Point) estimator.
//!
//! Wraps the AP3P minimal solver behind the generic [`Estimator`] trait for use
//! in RANSAC pipelines.
//!
//! **Architecture:** This acts as an *Asymmetric Estimator*. It uses the lightning-fast
//! AP3P algorithm to generate hypotheses from exactly 3 points during the minimal
//! sampling phase. However, it delegates overdetermined Local Optimization (LO)
//! refit passes (where $N > 3$) to EPnP. This prevents the solver from truncating
//! valuable inliers, unlocking RANSAC's early-termination math while maintaining
//! strict $O(1)$ allocations in the inner loop.

use kornia_algebra::{Mat3AF32, Vec2F32, Vec3AF32};
use kornia_imgproc::calibration::distortion::PolynomialDistortion;

use crate::pnp::ap3p::{solve_ap3p_multi, AP3PParams};
use crate::pnp::epnp::{solve_epnp, EPnPParams};
use crate::pnp::LMRefineParams;
use crate::ransac::{Estimator, Match2d3d};

/// One absolute-pose hypothesis produced by the AP3P solver.
///
/// AP3P natively computes up to 4 polynomial roots. During the `fit` phase,
/// all cheirality-passing roots are emitted as distinct `AP3PModel` candidates.
/// The RANSAC driver evaluates all candidates against the consensus set to find
/// the true camera pose.
#[derive(Debug, Clone, Copy)]
pub struct AP3PModel {
    /// Rotation matrix mapping coordinates from the **world** frame to the **camera** frame.
    pub rotation: Mat3AF32,
    /// Translation vector in the **camera** frame.
    pub translation: Vec3AF32,
}

/// Estimator for absolute camera pose from 3D-2D correspondences via AP3P.
///
/// Carries the camera intrinsics, optional lens distortion, and solver hyperparameters
/// as state. Intrinsics are pre-extracted into scalar fields upon initialization to
/// completely eliminate matrix destructuring overhead during the millions of
/// `residual()` calls made by the RANSAC inner loop.
#[derive(Debug, Clone)]
pub struct AP3PEstimator {
    /// Full intrinsics matrix (f32, SIMD-aligned).
    pub k: Mat3AF32,
    /// Pre-extracted focal length X to bypass inner-loop matrix copies.
    pub fx: f32,
    /// Pre-extracted focal length Y to bypass inner-loop matrix copies.
    pub fy: f32,
    /// Pre-extracted principal point X to bypass inner-loop matrix copies.
    pub cx: f32,
    /// Pre-extracted principal point Y to bypass inner-loop matrix copies.
    pub cy: f32,
    /// Optional lens distortion model. Currently utilized only during the LO refit step.
    pub distortion: Option<PolynomialDistortion>,
    /// Tuning parameters for the AP3P algebraic solver.
    pub params: AP3PParams,
}

impl AP3PEstimator {
    /// Builds an AP3P estimator with default parameters and no distortion.
    ///
    /// The intrinsics matrix is immediately destructured to cache `fx`, `fy`, `cx`, and `cy`
    /// for high-performance residual scoring.
    pub fn new(k: Mat3AF32) -> Self {
        let arr = k.to_cols_array();
        Self {
            k,
            fx: arr[0],
            fy: arr[4],
            cx: arr[6],
            cy: arr[7],
            distortion: None,
            params: AP3PParams::default(),
        }
    }

    /// Attaches a polynomial distortion model to the estimator.
    ///
    /// *Note:* This distortion model is applied during the overdetermined EPnP LO refit phase.
    pub fn with_distortion(mut self, distortion: PolynomialDistortion) -> Self {
        self.distortion = Some(distortion);
        self
    }

    /// Overrides the underlying [`AP3PParams`] used during hypothesis generation.
    pub fn with_params(mut self, params: AP3PParams) -> Self {
        self.params = params;
        self
    }
}

impl Estimator for AP3PEstimator {
    type Model = AP3PModel;
    type Sample = Match2d3d;

    /// AP3P is a minimal solver requiring exactly 3 geometric points.
    const SAMPLE_SIZE: usize = 3;

    /// Generates camera pose hypotheses from a minimal 3-point sample.
    fn fit(&self, samples: &[Self::Sample], out: &mut Vec<Self::Model>) {
        if samples.len() < Self::SAMPLE_SIZE {
            return;
        }

        let mut world = [Vec3AF32::default(); 3];
        let mut image = [Vec2F32::default(); 3];

        for i in 0..3 {
            world[i] = Vec3AF32::new(
                samples[i].object.x as f32,
                samples[i].object.y as f32,
                samples[i].object.z as f32,
            );
            image[i] = Vec2F32::new(samples[i].image.x as f32, samples[i].image.y as f32);
        }

        if let Ok(results) = solve_ap3p_multi(&world, &image, &self.k) {
            for res in results {
                out.push(AP3PModel {
                    rotation: res.rotation,
                    translation: res.translation,
                });
            }
        }
    }

    /// Computes an overdetermined least-squares camera pose from all current inliers.
    ///
    /// **Asymmetric Routing:** Because AP3P strictly requires 3 points, passing $N > 3$
    /// inliers to it would require truncating valuable data. Instead, this explicitly
    /// delegates the RANSAC Local Optimization (LO) phase to the EPnP solver, allowing
    /// it to consume all inliers for a highly accurate, noise-averaged model.
    fn refit(&self, samples: &[Self::Sample], out: &mut Vec<Self::Model>) {
        let n = samples.len();
        if n < 4 {
            return;
        }

        let mut world = Vec::with_capacity(n);
        let mut image = Vec::with_capacity(n);

        for s in samples {
            world.push(Vec3AF32::new(
                s.object.x as f32,
                s.object.y as f32,
                s.object.z as f32,
            ));
            image.push(Vec2F32::new(s.image.x as f32, s.image.y as f32));
        }

        let epnp_params = EPnPParams {
            refine_lm: Some(LMRefineParams::default()),
            ..Default::default()
        };

        if let Ok(result) = solve_epnp(
            &world,
            &image,
            &self.k,
            self.distortion.as_ref(),
            &epnp_params,
        ) {
            out.push(AP3PModel {
                rotation: result.rotation,
                translation: result.translation,
            });
        }
    }

    /// Scores a sample against a hypothesis model.
    ///
    /// Returns the squared pixel reprojection error of the 3D point under the given
    /// `(R, t)` camera pose. Calculates using an ideal pinhole projection (distortion
    /// is not applied during scoring). Uses pre-extracted intrinsics to maximize throughput.
    fn residual(&self, model: &Self::Model, sample: &Self::Sample) -> f64 {
        let p = Vec3AF32::new(
            sample.object.x as f32,
            sample.object.y as f32,
            sample.object.z as f32,
        );
        let pc = model.rotation * p + model.translation;

        if pc.z.abs() < 1e-6 {
            return f64::INFINITY;
        }

        let xn = pc.x / pc.z;
        let yn = pc.y / pc.z;

        let u = self.fx * xn + self.cx;
        let v = self.fy * yn + self.cy;

        let du = (u as f64) - sample.image.x;
        let dv = (v as f64) - sample.image.y;

        du * du + dv * dv
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_algebra::{Vec2F64, Vec3F64};

    /// Generates 3 perfectly-projected correspondences, fits the minimal solver,
    /// and ensures at least one returned algebraic root perfectly matches the ground truth.
    #[test]
    fn fits_and_scores_clean_correspondences() {
        let fx = 600.0_f32;
        let fy = 600.0_f32;
        let cx = 320.0_f32;
        let cy = 240.0_f32;
        let k = Mat3AF32::from_cols(
            Vec3AF32::new(fx, 0.0, 0.0),
            Vec3AF32::new(0.0, fy, 0.0),
            Vec3AF32::new(cx, cy, 1.0),
        );

        let world_pts = [
            Vec3F64::new(-0.4, -0.3, 5.0),
            Vec3F64::new(0.3, -0.2, 4.5),
            Vec3F64::new(-0.2, 0.4, 6.0),
        ];

        let angle = 0.05_f64;
        let r = [
            [angle.cos(), 0.0, -angle.sin()],
            [0.0, 1.0, 0.0],
            [angle.sin(), 0.0, angle.cos()],
        ];
        let t = [0.1_f64, -0.05, 0.2];

        let matches: Vec<Match2d3d> = world_pts
            .iter()
            .map(|p| {
                let pc = [
                    r[0][0] * p.x + r[0][1] * p.y + r[0][2] * p.z + t[0],
                    r[1][0] * p.x + r[1][1] * p.y + r[1][2] * p.z + t[1],
                    r[2][0] * p.x + r[2][1] * p.y + r[2][2] * p.z + t[2],
                ];
                let u = fx as f64 * pc[0] / pc[2] + cx as f64;
                let v = fy as f64 * pc[1] / pc[2] + cy as f64;
                Match2d3d::new(*p, Vec2F64::new(u, v))
            })
            .collect();

        let est = AP3PEstimator::new(k);
        let mut models = Vec::new();
        est.fit(&matches, &mut models);

        assert!(!models.is_empty(), "expected at least one AP3P solution");

        // Verify that at least one returned root represents the true pose
        let mut best_rmse = f64::INFINITY;
        for model in &models {
            let mut sum = 0.0;
            for m in &matches {
                sum += est.residual(model, m);
            }
            if sum < best_rmse {
                best_rmse = sum;
            }
        }
        assert!(
            best_rmse < 1e-4,
            "No returned model matched the ground truth"
        );
    }

    /// Verifies that providing an overdetermined sample (e.g. via LO) successfully
    /// routes the data to the EPnP refit delegate without triggering SVD length panics.
    #[test]
    fn safely_routes_lo_refit() {
        let k = Mat3AF32::from_cols(
            Vec3AF32::new(600.0, 0.0, 0.0),
            Vec3AF32::new(0.0, 600.0, 0.0),
            Vec3AF32::new(320.0, 240.0, 1.0),
        );
        let est = AP3PEstimator::new(k);
        let mut models = Vec::new();

        let dummy_match = Match2d3d::new(Vec3F64::new(0.0, 0.0, 5.0), Vec2F64::new(320.0, 240.0));
        let oversampled = vec![dummy_match; 6];

        // Should successfully delegate to EPnP rather than panicking on length checks
        est.refit(&oversampled, &mut models);
    }
}
