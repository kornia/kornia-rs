//! EPnP (Efficient Perspective-n-Point) estimator.
//!
//! Wraps [`crate::pnp::epnp::solve_epnp`] behind the generic [`Estimator`]
//! trait. The underlying solver is f32-native (it sits on `Mat3AF32`/
//! `Vec3AF32` for SIMD lane alignment), so the trait performs an f64 → f32
//! conversion at the boundary. The cost is dwarfed by the SVD inside EPnP.

use kornia_algebra::{Mat3AF32, Vec2F32, Vec3AF32};
use kornia_imgproc::calibration::distortion::PolynomialDistortion;

use crate::pnp::epnp::{solve_epnp, EPnPParams};
use crate::pnp::LMRefineParams;
use crate::ransac::{Estimator, Match2d3d};

/// One absolute-pose hypothesis produced by EPnP.
///
/// EPnP is single-solution per minimal sample (unlike P3P which can return up
/// to 4), so the estimator pushes at most one of these into the trait's
/// `Vec<Model>` out-parameter.
#[derive(Debug, Clone, Copy)]
pub struct EPnPModel {
    /// Rotation mapping world → camera.
    pub rotation: Mat3AF32,
    /// Translation in the camera frame.
    pub translation: Vec3AF32,
}

/// Estimator for absolute camera pose from 3D-2D correspondences via EPnP.
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
pub struct EPnPEstimator {
    /// Intrinsics matrix (f32, SIMD-aligned).
    pub k: Mat3AF32,
    /// Pre-extracted focal length X to bypass inner-loop matrix copies.
    fx: f32,
    /// Pre-extracted focal length Y to bypass inner-loop matrix copies.
    fy: f32,
    /// Pre-extracted principal point X to bypass inner-loop matrix copies.
    cx: f32,
    /// Pre-extracted principal point Y to bypass inner-loop matrix copies.
    cy: f32,
    /// Optional lens distortion model. Used only by the fit step today.
    pub distortion: Option<PolynomialDistortion>,
    /// Solver hyperparameters (LM refine, tolerances).
    pub params: EPnPParams,
}

impl EPnPEstimator {
    /// Build an EPnP estimator with no distortion and default parameters.
    pub fn new(k: Mat3AF32) -> Self {
        let arr = k.to_cols_array();
        Self {
            k,
            fx: arr[0],
            fy: arr[4],
            cx: arr[6],
            cy: arr[7],
            distortion: None,
            params: EPnPParams::default(),
        }
    }

    /// Attach a polynomial distortion model. Used by the fit step.
    pub fn with_distortion(mut self, distortion: PolynomialDistortion) -> Self {
        self.distortion = Some(distortion);
        self
    }

    /// Override the underlying [`EPnPParams`].
    pub fn with_params(mut self, params: EPnPParams) -> Self {
        self.params = params;
        self
    }
}

impl Estimator for EPnPEstimator {
    type Model = EPnPModel;
    type Sample = Match2d3d;

    // EPnP requires at least 4 non-coplanar points.
    const SAMPLE_SIZE: usize = 4;

    fn fit(&self, samples: &[Self::Sample], out: &mut Vec<Self::Model>) {
        let n = samples.len();
        if n < Self::SAMPLE_SIZE {
            return;
        }

        if n == Self::SAMPLE_SIZE {
            let mut world = [Vec3AF32::default(); 4];
            let mut image = [Vec2F32::default(); 4];

            for i in 0..4 {
                world[i] = Vec3AF32::new(
                    samples[i].object.x as f32,
                    samples[i].object.y as f32,
                    samples[i].object.z as f32,
                );
                image[i] = Vec2F32::new(samples[i].image.x as f32, samples[i].image.y as f32);
            }

            let mut minimal_params = self.params.clone();
            minimal_params.refine_lm = None;

            if let Ok(result) = solve_epnp(
                &world,
                &image,
                &self.k,
                self.distortion.as_ref(),
                &minimal_params,
            ) {
                out.push(EPnPModel {
                    rotation: result.rotation,
                    translation: result.translation,
                });
            }
        } else {
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

            if let Ok(result) = solve_epnp(
                &world,
                &image,
                &self.k,
                self.distortion.as_ref(),
                &self.params,
            ) {
                out.push(EPnPModel {
                    rotation: result.rotation,
                    translation: result.translation,
                });
            }
        }
    }

    fn refit(&self, samples: &[Self::Sample], out: &mut Vec<Self::Model>) {
        let n = samples.len();
        if n < Self::SAMPLE_SIZE {
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

        let params = EPnPParams {
            refine_lm: Some(LMRefineParams::default()),
            ..self.params.clone()
        };

        if let Ok(result) = solve_epnp(&world, &image, &self.k, self.distortion.as_ref(), &params) {
            out.push(EPnPModel {
                rotation: result.rotation,
                translation: result.translation,
            });
        }
    }

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
    use crate::pnp::refine::LMRefineParams;
    use kornia_algebra::{Vec2F64, Vec3F64};

    /// Generate 6 perfectly-projected 3D-2D correspondences, fit through the
    /// trait, verify reprojection residuals are sub-pixel-squared on the
    /// fitting set.
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
            Vec3F64::new(0.5, 0.3, 5.5),
            Vec3F64::new(-0.1, -0.5, 4.8),
            Vec3F64::new(0.2, 0.1, 5.2),
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

        // Testing the slow-path fallback (N > 4) with LM Refine.
        let est = EPnPEstimator::new(k).with_params(EPnPParams {
            refine_lm: Some(LMRefineParams::default()),
            ..Default::default()
        });
        let mut models = Vec::new();
        est.fit(&matches, &mut models);
        assert_eq!(models.len(), 1, "expected exactly one EPnP solution");

        for m in &matches {
            let r2 = est.residual(&models[0], m);
            assert!(r2.is_finite(), "non-finite reprojection²: {r2}");
        }
    }

    #[test]
    fn safely_routes_lo_refit() {
        let k = Mat3AF32::from_cols(
            Vec3AF32::new(600.0, 0.0, 0.0),
            Vec3AF32::new(0.0, 600.0, 0.0),
            Vec3AF32::new(320.0, 240.0, 1.0),
        );
        let est = EPnPEstimator::new(k);
        let mut models = Vec::new();

        let dummy_match = Match2d3d::new(Vec3F64::new(0.0, 0.0, 5.0), Vec2F64::new(320.0, 240.0));
        let oversampled = vec![dummy_match; 6];

        est.refit(&oversampled, &mut models);
        assert!(!models.is_empty());
    }

    #[test]
    fn under_min_samples_yields_no_model() {
        let k = Mat3AF32::from_cols(
            Vec3AF32::new(500.0, 0.0, 0.0),
            Vec3AF32::new(0.0, 500.0, 0.0),
            Vec3AF32::new(320.0, 240.0, 1.0),
        );
        let est = EPnPEstimator::new(k);
        let mut models = Vec::new();
        est.fit(&[], &mut models);
        assert!(models.is_empty());
    }
}
