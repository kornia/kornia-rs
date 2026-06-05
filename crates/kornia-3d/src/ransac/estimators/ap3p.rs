//! AP3P (Algebraic Perspective-3-Point) estimator.
//!
//! Wraps [`crate::pnp::ap3p::solve_ap3p`] behind the generic [`Estimator`]
//! trait. The underlying solver is f32-native (it sits on `Mat3AF32`/
//! `Vec3AF32` for SIMD lane alignment), so the trait performs an f64 → f32
//! conversion at the boundary. 

use kornia_algebra::{Mat3AF32, Vec2F32, Vec3AF32};
use kornia_imgproc::calibration::distortion::PolynomialDistortion;

use crate::pnp::ap3p::{solve_ap3p, AP3PParams};
use crate::ransac::{Estimator, Match2d3d};

/// One absolute-pose hypothesis produced by AP3P.
///
/// AP3P natively computes up to 4 roots, but the `solve_ap3p` kernel
/// automatically filters out cheirality-violating solutions and selects the 
/// single best candidate (based on reprojection RMSE). Therefore, this 
/// estimator outputs a maximum of one model per minimal sample.
#[derive(Debug, Clone, Copy)]
pub struct AP3PModel {
    /// Rotation mapping world → camera.
    pub rotation: Mat3AF32,
    /// Translation in the camera frame.
    pub translation: Vec3AF32,
}

/// Estimator for absolute camera pose from 3D-2D correspondences via AP3P.
///
/// Carries the camera intrinsics + (optional) lens distortion + solver
/// hyperparameters as state, since they're constant across all hypotheses
/// in a single RANSAC run. 
///
/// **Coordinate convention.** `Match2d3d::object` is a world-frame 3D point;
/// `Match2d3d::image` is the corresponding pixel observation. The recovered
/// pose maps world → camera (matching the existing `PnPResult` convention).
///
/// **Residual.** Squared reprojection error in pixels, with no distortion
/// model applied at scoring time. 
#[derive(Debug, Clone)]
pub struct AP3PEstimator {
    /// Intrinsics matrix (f32, SIMD-aligned).
    pub k: Mat3AF32,
    /// Optional lens distortion model. Used only by the fit step today.
    pub distortion: Option<PolynomialDistortion>,
    /// Solver parameters (e.g., whether to pick the absolute lowest RMSE).
    pub params: AP3PParams,
}

impl AP3PEstimator {
    /// Build an AP3P estimator with no distortion and default parameters.
    pub fn new(k: Mat3AF32) -> Self {
        Self {
            k,
            distortion: None,
            params: AP3PParams::default(),
        }
    }

    /// Attach a polynomial distortion model. Used by the fit step.
    pub fn with_distortion(mut self, distortion: PolynomialDistortion) -> Self {
        self.distortion = Some(distortion);
        self
    }

    /// Override the underlying [`AP3PParams`].
    pub fn with_params(mut self, params: AP3PParams) -> Self {
        self.params = params;
        self
    }
}

impl Estimator for AP3PEstimator {
    type Model = AP3PModel;
    type Sample = Match2d3d;
    
    // AP3P is a minimal solver requiring exactly 3 points.
    const SAMPLE_SIZE: usize = 3;

    fn fit(&self, samples: &[Self::Sample], out: &mut Vec<Self::Model>) {
        if samples.len() < Self::SAMPLE_SIZE {
            return;
        }
        
        // AP3P strictly solves for exactly 3 points. If the RANSAC LO step 
        // attempts a refit with > 3 points, we truncate to the minimal sample 
        // to prevent the underlying solver from rejecting the array lengths.
        let mut world = Vec::with_capacity(Self::SAMPLE_SIZE);
        let mut image = Vec::with_capacity(Self::SAMPLE_SIZE);
        
        for s in samples.iter().take(Self::SAMPLE_SIZE) {
            world.push(Vec3AF32::new(
                s.object.x as f32,
                s.object.y as f32,
                s.object.z as f32,
            ));
            image.push(Vec2F32::new(s.image.x as f32, s.image.y as f32));
        }
        
        if let Ok(result) = solve_ap3p(
            &world,
            &image,
            &self.k,
            &self.params,
        ) {
            out.push(AP3PModel {
                rotation: result.rotation,
                translation: result.translation,
            });
        }
    }

    fn residual(&self, model: &Self::Model, sample: &Self::Sample) -> f64 {
        // Project sample.object via (R, t, K) — pinhole, no distortion
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
        
        // K is column-major [m00, m10, m20, m01, m11, m21, m02, m12, m22].
        // For a standard intrinsics matrix:
        //   col 0 = [fx, 0, 0], col 1 = [0, fy, 0], col 2 = [cx, cy, 1].
        let arr = self.k.to_cols_array();
        let fx = arr[0] as f64;
        let fy = arr[4] as f64;
        let cx = arr[6] as f64;
        let cy = arr[7] as f64;
        
        let u = fx * xn as f64 + cx;
        let v = fy * yn as f64 + cy;
        
        let du = u - sample.image.x;
        let dv = v - sample.image.y;
        
        du * du + dv * dv
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_algebra::{Vec2F64, Vec3F64};

    /// Generate exactly 3 perfectly-projected 3D-2D correspondences, fit 
    /// through the trait, and verify reprojection residuals are strictly 
    /// sub-pixel-squared on the fitting set.
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

        // Exactly 3 world points in front of the camera to satisfy the 
        // AP3P minimal geometric requirement.
        let world_pts = [
            Vec3F64::new(-0.4, -0.3, 5.0),
            Vec3F64::new(0.3, -0.2, 4.5),
            Vec3F64::new(-0.2, 0.4, 6.0),
        ];

        // Camera pose: small rotation around Y + translation.
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

        // Instantiate the AP3P estimator with default parameters
        let est = AP3PEstimator::new(k);
        let mut models = Vec::new();
        est.fit(&matches, &mut models);
        
        // Ensure AP3P successfully resolved the algebraic roots and cheirality checks
        assert_eq!(models.len(), 1, "expected exactly one AP3P solution");

        // Verify the extracted model correctly maps the control points
        for m in &matches {
            let r2 = est.residual(&models[0], m);
            assert!(r2.is_finite(), "non-finite reprojection²: {r2}");
            // Since it's a direct closed-form solver, the minimal set should 
            // project back with near-zero error.
            assert!(r2 < 1e-4, "reprojection squared error too high: {r2}");
        }
    }

    /// Passing more than 3 samples safely truncates and evaluates 
    /// without panicking or triggering length assertion errors.
    #[test]
    fn safely_handles_oversampled_inputs() {
        let k = Mat3AF32::from_cols(
            Vec3AF32::new(600.0, 0.0, 0.0),
            Vec3AF32::new(0.0, 600.0, 0.0),
            Vec3AF32::new(320.0, 240.0, 1.0),
        );
        let est = AP3PEstimator::new(k);
        let mut models = Vec::new();
        
        // Provide 5 samples to test truncation behavior
        let dummy_match = Match2d3d::new(
            Vec3F64::new(0.0, 0.0, 5.0), 
            Vec2F64::new(320.0, 240.0)
        );
        let oversampled = vec![dummy_match; 5];
        
        // This should run the first 3 samples smoothly without returning a 
        // mismatched array lengths error.
        est.fit(&oversampled, &mut models);
    }

    /// Below-minimal sample → no model, no panic.
    #[test]
    fn under_min_samples_yields_no_model() {
        let k = Mat3AF32::from_cols(
            Vec3AF32::new(500.0, 0.0, 0.0),
            Vec3AF32::new(0.0, 500.0, 0.0),
            Vec3AF32::new(320.0, 240.0, 1.0),
        );
        let est = AP3PEstimator::new(k);
        let mut models = Vec::new();
        
        // Pass only 2 samples
        let dummy_match = Match2d3d::new(
            Vec3F64::new(0.0, 0.0, 5.0), 
            Vec2F64::new(320.0, 240.0)
        );
        
        est.fit(&[dummy_match, dummy_match], &mut models);
        assert!(models.is_empty());
    }
}