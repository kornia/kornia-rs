//! Essential-matrix estimator (Nistér 5-point).
//!
//! Wraps [`crate::pose::essential_5pt::essential_5pt`] which is itself a
//! multi-solution kernel: each minimal sample yields up to 10 candidate Es
//! that satisfy the on-manifold polynomial system. The trait's `Vec<Model>`
//! out-parameter exists precisely for this case — we push every candidate
//! and let the driver score them all.

use kornia_algebra::{Mat3F64, Vec2F64};

use crate::pose::{essential_5pt, sampson_distance};
use crate::ransac::{Estimator, Match2d2d};

/// Estimator for the essential matrix from 2D-2D **normalized** correspondences.
///
/// **Coordinate convention.** Samples must be in calibrated coordinates,
/// i.e. `K⁻¹ · [u, v, 1]ᵀ`. Pre-normalize once before constructing the
/// `Match2d2d` slice you hand to RANSAC; the kernel does no further
/// normalization (this matches the existing `essential_5pt` API).
///
/// **Residual.** Sampson distance evaluated on the calibrated
/// correspondences. Algebraically identical to the F-matrix Sampson form;
/// numerically the residual is in normalized image units (≈ pixel / focal),
/// so RANSAC thresholds need to be scaled accordingly (e.g. `(1.0 / fx)²`
/// for a 1-pixel target).
#[derive(Debug, Clone, Copy, Default)]
pub struct EssentialEstimator;

impl Estimator for EssentialEstimator {
    type Model = Mat3F64;
    type Sample = Match2d2d;
    const SAMPLE_SIZE: usize = 5;

    fn fit(&self, samples: &[Self::Sample], out: &mut Vec<Self::Model>) {
        if samples.len() < Self::SAMPLE_SIZE {
            return;
        }
        // The 5-point solver takes fixed-size arrays — copy the first 5
        // samples into stack-resident arrays.
        let mut x1 = [Vec2F64::ZERO; 5];
        let mut x2 = [Vec2F64::ZERO; 5];
        for i in 0..5 {
            x1[i] = samples[i].x1;
            x2[i] = samples[i].x2;
        }
        let candidates = essential_5pt(&x1, &x2);
        out.extend(candidates);
    }

    #[inline]
    fn residual(&self, model: &Self::Model, sample: &Self::Sample) -> f64 {
        sampson_distance(model, &sample.x1, &sample.x2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_algebra::Vec3F64;

    /// Generates 5 calibrated correspondences from a known (R, t) and feeds
    /// them through the estimator. At least one of the up-to-10 candidate
    /// Es must satisfy the epipolar constraint to within roundoff on the
    /// fitting set — this is the canonical 5-point sanity check.
    #[test]
    fn fits_and_scores_clean_normalized_correspondences() {
        let pair = synthetic_normalized_pair();
        let est = EssentialEstimator;
        let mut models = Vec::new();
        est.fit(&pair.matches, &mut models);
        assert!(!models.is_empty(), "5-point should produce ≥1 candidate");

        // Pick the candidate with the smallest aggregate residual on the
        // fitting set — that's the "true" E up to 4-fold cheirality
        // ambiguity (which RANSAC resolves later via cheirality vote, not
        // here).
        let mut best = f64::INFINITY;
        for e in &models {
            let total: f64 = pair.matches.iter().map(|m| est.residual(e, m)).sum();
            if total < best {
                best = total;
            }
        }
        assert!(best < 1e-10, "best 5-pt E residual too large: {best}");
    }

    struct Pair {
        matches: Vec<Match2d2d>,
    }

    fn synthetic_normalized_pair() -> Pair {
        // Pure rotation + translation; no intrinsics — samples are already
        // in calibrated coordinates.
        let angle = 0.15_f64;
        let r = [
            [angle.cos(), 0.0, -angle.sin()],
            [0.0, 1.0, 0.0],
            [angle.sin(), 0.0, angle.cos()],
        ];
        let t = [0.8_f64, 0.05, 0.1];
        let pts = [
            Vec3F64::new(-0.4, -0.2, 3.0),
            Vec3F64::new(0.3, -0.3, 2.5),
            Vec3F64::new(-0.2, 0.4, 4.0),
            Vec3F64::new(0.5, 0.2, 3.5),
            Vec3F64::new(-0.1, -0.5, 2.8),
        ];
        let mut matches = Vec::with_capacity(pts.len());
        for p in &pts {
            let m1 = Vec2F64::new(p.x / p.z, p.y / p.z);
            let pc2 = [
                r[0][0] * p.x + r[0][1] * p.y + r[0][2] * p.z + t[0],
                r[1][0] * p.x + r[1][1] * p.y + r[1][2] * p.z + t[1],
                r[2][0] * p.x + r[2][1] * p.y + r[2][2] * p.z + t[2],
            ];
            let m2 = Vec2F64::new(pc2[0] / pc2[2], pc2[1] / pc2[2]);
            matches.push(Match2d2d::new(m1, m2));
        }
        Pair { matches }
    }
}
