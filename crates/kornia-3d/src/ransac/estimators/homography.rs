//! Homography estimator (4-point minimal solver).
//!
//! Wraps [`crate::pose::homography::homography_4pt2d`], the 8×8 LU-based
//! minimal solver kept hot for inside-RANSAC use. Non-minimal refits (LO
//! step, final inlier-set polish) should call
//! `crate::pose::homography::homography_dlt` directly.

use kornia_algebra::{Mat3F64, Vec3F64};

use crate::pose::homography_4pt2d;
use crate::ransac::{Estimator, Match2d2d};

/// Estimator for a planar homography from 2D-2D pixel correspondences.
///
/// **Coordinate convention.** Pixel coordinates. The minimal solver does
/// not Hartley-normalize internally — the 4-point system is small enough
/// that LU on the raw 8×8 stays well-conditioned for typical image scales.
///
/// **Residual.** One-sided forward transfer error squared, i.e.
/// `‖x2 - π(H · [x1, 1])‖²` in pixel units. Symmetric transfer error is
/// available as a future variant; the asymmetric form is the OpenCV
/// `findHomography` default and matches what most downstream code expects.
#[derive(Debug, Clone, Copy, Default)]
pub struct HomographyEstimator;

impl Estimator for HomographyEstimator {
    type Model = Mat3F64;
    type Sample = Match2d2d;
    const SAMPLE_SIZE: usize = 4;

    fn fit(&self, samples: &[Self::Sample], out: &mut Vec<Self::Model>) {
        if samples.len() < Self::SAMPLE_SIZE {
            return;
        }
        let x1 = [
            [samples[0].x1.x, samples[0].x1.y],
            [samples[1].x1.x, samples[1].x1.y],
            [samples[2].x1.x, samples[2].x1.y],
            [samples[3].x1.x, samples[3].x1.y],
        ];
        let x2 = [
            [samples[0].x2.x, samples[0].x2.y],
            [samples[1].x2.x, samples[1].x2.y],
            [samples[2].x2.x, samples[2].x2.y],
            [samples[3].x2.x, samples[3].x2.y],
        ];
        let mut h = [[0.0f64; 3]; 3];
        if homography_4pt2d(&x1, &x2, &mut h).is_ok() {
            out.push(Mat3F64::from_cols(
                Vec3F64::new(h[0][0], h[1][0], h[2][0]),
                Vec3F64::new(h[0][1], h[1][1], h[2][1]),
                Vec3F64::new(h[0][2], h[1][2], h[2][2]),
            ));
        }
    }

    #[inline]
    fn residual(&self, model: &Self::Model, sample: &Self::Sample) -> f64 {
        let x1h = Vec3F64::new(sample.x1.x, sample.x1.y, 1.0);
        let mapped = *model * x1h;
        // Behind-the-plane / numerically degenerate H · x → ∞ residual so
        // the consensus step rejects it without poisoning the score.
        if mapped.z.abs() < 1e-12 {
            return f64::INFINITY;
        }
        let dx = mapped.x / mapped.z - sample.x2.x;
        let dy = mapped.y / mapped.z - sample.x2.y;
        dx * dx + dy * dy
    }

    /// Override to keep the H matrix in registers across the inner loop.
    /// Smaller win than F (no transpose to hoist) but still removes the
    /// per-call dispatch through [`Self::residual`].
    fn residual_batch(&self, model: &Self::Model, samples: &[Self::Sample], out: &mut [f64]) {
        debug_assert_eq!(out.len(), samples.len());
        let h = *model;
        for (i, s) in samples.iter().enumerate() {
            let x1h = Vec3F64::new(s.x1.x, s.x1.y, 1.0);
            let mapped = h * x1h;
            if mapped.z.abs() < 1e-12 {
                out[i] = f64::INFINITY;
                continue;
            }
            let inv_z = 1.0 / mapped.z;
            let dx = mapped.x * inv_z - s.x2.x;
            let dy = mapped.y * inv_z - s.x2.y;
            out[i] = dx * dx + dy * dy;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_algebra::Vec2F64;

    /// Build 4 correspondences from a known H, fit through the trait, verify
    /// transfer error is ~0 on the fitting set.
    #[test]
    fn fits_and_scores_clean_correspondences() {
        let h_true = Mat3F64::from_cols(
            Vec3F64::new(1.2, 0.05, 0.0),
            Vec3F64::new(0.03, 0.95, 0.0),
            Vec3F64::new(7.0, -3.0, 1.0),
        );
        let pts = [
            Vec2F64::new(10.0, 20.0),
            Vec2F64::new(100.0, 30.0),
            Vec2F64::new(80.0, 200.0),
            Vec2F64::new(20.0, 180.0),
        ];
        let matches: Vec<Match2d2d> = pts
            .iter()
            .map(|p| {
                let mapped = h_true * Vec3F64::new(p.x, p.y, 1.0);
                let mp = Vec2F64::new(mapped.x / mapped.z, mapped.y / mapped.z);
                Match2d2d::new(*p, mp)
            })
            .collect();

        let est = HomographyEstimator;
        let mut models = Vec::new();
        est.fit(&matches, &mut models);
        assert_eq!(models.len(), 1);

        let h = models[0];
        for m in &matches {
            let r = est.residual(&h, m);
            assert!(r < 1e-10, "transfer error too large: {r}");
        }
    }

    /// Below-minimal sample → no model.
    #[test]
    fn under_min_samples_yields_no_model() {
        let est = HomographyEstimator;
        let mut models = Vec::new();
        est.fit(&[], &mut models);
        assert!(models.is_empty());
    }
}
