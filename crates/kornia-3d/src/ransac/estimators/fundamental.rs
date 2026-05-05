//! Fundamental-matrix estimator.
//!
//! Wraps [`crate::pose::fundamental::fundamental_8point`] — the normalized
//! 8-point algorithm — behind the generic [`Estimator`] trait so the RANSAC
//! driver can drive it the same way it drives any other model.

use kornia_algebra::{Mat3F64, Vec2F64};

use crate::pose::{fundamental_8point, sampson_distance};
use crate::ransac::{Estimator, Match2d2d};

/// Estimator for the fundamental matrix from 2D-2D pixel correspondences.
///
/// **Coordinate convention.** Samples are in raw pixel coordinates; Hartley
/// normalization is applied internally, mirroring the existing solver.
///
/// **Residual.** Sampson distance — the standard first-order epipolar
/// approximation. Squared pixel units, so the matching RANSAC threshold
/// is also squared (e.g. `1.0` ≈ 1 px).
#[derive(Debug, Clone, Copy, Default)]
pub struct FundamentalEstimator;

impl Estimator for FundamentalEstimator {
    type Model = Mat3F64;
    type Sample = Match2d2d;
    const SAMPLE_SIZE: usize = 8;

    fn fit(&self, samples: &[Self::Sample], out: &mut Vec<Self::Model>) {
        // Translate AoS samples into the parallel-slice form fundamental_8point
        // expects. SAMPLE_SIZE=8 means we never spill past the stack array;
        // larger inputs (LO-RANSAC refits) fall back to a Vec but only when
        // the driver explicitly passes >8 — they're rare and amortized.
        const STACK_N: usize = 32;
        let n = samples.len();
        if n < Self::SAMPLE_SIZE {
            return;
        }
        if n <= STACK_N {
            let mut x1 = [Vec2F64::ZERO; STACK_N];
            let mut x2 = [Vec2F64::ZERO; STACK_N];
            for (i, s) in samples.iter().enumerate() {
                x1[i] = s.x1;
                x2[i] = s.x2;
            }
            if let Ok(f) = fundamental_8point(&x1[..n], &x2[..n]) {
                out.push(f);
            }
        } else {
            let x1: Vec<Vec2F64> = samples.iter().map(|s| s.x1).collect();
            let x2: Vec<Vec2F64> = samples.iter().map(|s| s.x2).collect();
            if let Ok(f) = fundamental_8point(&x1, &x2) {
                out.push(f);
            }
        }
    }

    #[inline]
    fn residual(&self, model: &Self::Model, sample: &Self::Sample) -> f64 {
        sampson_distance(model, &sample.x1, &sample.x2)
    }

    /// 2-lane f64 NEON Sampson scorer on aarch64; scalar fallback elsewhere.
    ///
    /// `Match2d2d` is `{x1: Vec2F64, x2: Vec2F64}` = 4 contiguous f64s, so
    /// two consecutive matches are exactly the 8 f64s `vld4q_f64` reads
    /// and deinterleaves into 4 lane-vectors — `(x1_x, x1_y, x2_x, x2_y)`
    /// across two correspondences in one instruction. No SoA scratch
    /// needed.
    fn residual_batch(
        &self,
        model: &Self::Model,
        samples: &[Self::Sample],
        out: &mut [f64],
    ) {
        debug_assert_eq!(out.len(), samples.len());
        // Pull 9 F entries into scalars once per hypothesis. Naming follows
        // the row-major mathematical convention (F · x = ... f00*x + f01*y + f02).
        let f00 = model.x_axis.x;
        let f01 = model.y_axis.x;
        let f02 = model.z_axis.x;
        let f10 = model.x_axis.y;
        let f11 = model.y_axis.y;
        let f12 = model.z_axis.y;
        let f20 = model.x_axis.z;
        let f21 = model.y_axis.z;
        let f22 = model.z_axis.z;

        #[cfg(target_arch = "aarch64")]
        let mut idx = unsafe {
            sampson_residual_neon(
                (f00, f01, f02, f10, f11, f12, f20, f21, f22),
                samples,
                out,
            )
        };
        #[cfg(not(target_arch = "aarch64"))]
        let mut idx = 0usize;

        // Scalar tail (and full fallback on non-aarch64).
        while idx < samples.len() {
            let s = &samples[idx];
            let (x1, y1) = (s.x1.x, s.x1.y);
            let (x2, y2) = (s.x2.x, s.x2.y);
            let fx1x = f00 * x1 + f01 * y1 + f02;
            let fx1y = f10 * x1 + f11 * y1 + f12;
            let fx1z = f20 * x1 + f21 * y1 + f22;
            let ftx2x = f00 * x2 + f10 * y2 + f20;
            let ftx2y = f01 * x2 + f11 * y2 + f21;
            let err = fx1x * x2 + fx1y * y2 + fx1z;
            let denom = fx1x * fx1x + fx1y * fx1y + ftx2x * ftx2x + ftx2y * ftx2y;
            out[idx] = if denom <= 1e-12 {
                err * err
            } else {
                err * err / denom
            };
            idx += 1;
        }
    }
}

/// 2-lane f64 NEON kernel for `FundamentalEstimator::residual_batch`.
///
/// Returns the index up to which it processed (caller does scalar tail for
/// the remainder).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn sampson_residual_neon(
    f: (f64, f64, f64, f64, f64, f64, f64, f64, f64),
    samples: &[Match2d2d],
    out: &mut [f64],
) -> usize {
    use std::arch::aarch64::*;
    let (f00, f01, f02, f10, f11, f12, f20, f21, f22) = f;
    let f00v = vdupq_n_f64(f00);
    let f01v = vdupq_n_f64(f01);
    let f02v = vdupq_n_f64(f02);
    let f10v = vdupq_n_f64(f10);
    let f11v = vdupq_n_f64(f11);
    let f12v = vdupq_n_f64(f12);
    let f20v = vdupq_n_f64(f20);
    let f21v = vdupq_n_f64(f21);
    let f22v = vdupq_n_f64(f22);
    let eps = vdupq_n_f64(1e-12);
    let one = vdupq_n_f64(1.0);

    let n = samples.len();
    let mut idx = 0usize;
    // Process two matches per iteration. samples[i..i+2] is exactly the
    // 8 f64s vld4q_f64 deinterleaves into (x1_x, x1_y, x2_x, x2_y) lane
    // vectors — perfect AoS-to-SoA load in one instruction.
    while idx + 2 <= n {
        let base = samples.as_ptr().add(idx) as *const f64;
        let lanes = vld4q_f64(base);
        let x1 = lanes.0; // (m[i].x1.x, m[i+1].x1.x)
        let y1 = lanes.1;
        let x2 = lanes.2;
        let y2 = lanes.3;

        // fx1 = F · [x1, y1, 1]
        let fx1x = vfmaq_f64(vfmaq_f64(f02v, x1, f00v), y1, f01v);
        let fx1y = vfmaq_f64(vfmaq_f64(f12v, x1, f10v), y1, f11v);
        let fx1z = vfmaq_f64(vfmaq_f64(f22v, x1, f20v), y1, f21v);
        // ftx2 = Fᵀ · [x2, y2, 1] (only x,y needed for denom)
        let ftx2x = vfmaq_f64(vfmaq_f64(f20v, x2, f00v), y2, f10v);
        let ftx2y = vfmaq_f64(vfmaq_f64(f21v, x2, f01v), y2, f11v);

        let err = vfmaq_f64(vfmaq_f64(fx1z, x2, fx1x), y2, fx1y);
        let denom = vfmaq_f64(
            vfmaq_f64(vfmaq_f64(vmulq_f64(fx1x, fx1x), fx1y, fx1y), ftx2x, ftx2x),
            ftx2y,
            ftx2y,
        );
        let err_sq = vmulq_f64(err, err);
        // Branchless guard: if denom > 1e-12 use err²/denom, else err²
        // (matches scalar `sampson_distance`).
        let denom_ok = vcgtq_f64(denom, eps);
        let safe_denom = vbslq_f64(denom_ok, denom, one);
        let div_val = vdivq_f64(err_sq, safe_denom);
        let dd = vbslq_f64(denom_ok, div_val, err_sq);

        vst1q_f64(out.as_mut_ptr().add(idx), dd);
        idx += 2;
    }
    idx
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_algebra::Vec3F64;

    /// Smoke test: feed 8 noise-free correspondences from a synthetic geometry,
    /// verify the trait produces an F whose Sampson residuals are ~0 on the
    /// fitting set. Mirrors the existing `fundamental_8point` test but routes
    /// through the trait surface so a future signature change breaks loudly.
    #[test]
    fn fits_and_scores_clean_correspondences() {
        let k_inv_t_e_k_inv = synthetic_pair();
        let mut models = Vec::new();
        let est = FundamentalEstimator;
        est.fit(&k_inv_t_e_k_inv.matches, &mut models);
        assert_eq!(models.len(), 1, "expected exactly one F from 8-pt");
        let f = models[0];
        for m in &k_inv_t_e_k_inv.matches {
            let d = est.residual(&f, m);
            assert!(d < 1e-8, "Sampson residual too large: {d}");
        }
    }

    /// NEON `residual_batch` must match the scalar `residual` path on the
    /// same model + samples. Catches lane-ordering bugs in the vld4q load
    /// or bad branchless masking on the denom guard.
    #[test]
    fn neon_batch_matches_scalar_residual() {
        let pair = synthetic_pair();
        let est = FundamentalEstimator;
        let mut models = Vec::new();
        est.fit(&pair.matches, &mut models);
        let f = models[0];

        // Build a longer sample slice (odd N → exercises the scalar tail).
        let mut samples = pair.matches.clone();
        samples.extend(pair.matches.iter().take(3).copied());
        assert_eq!(samples.len() % 2, 1, "odd N to hit the scalar tail");

        let mut batched = vec![0.0f64; samples.len()];
        est.residual_batch(&f, &samples, &mut batched);

        for (i, s) in samples.iter().enumerate() {
            let scalar = est.residual(&f, s);
            // 1e-12 absolute is the NEON↔scalar floor for FMA reordering.
            assert!(
                (batched[i] - scalar).abs() < 1e-12 * scalar.max(1.0).abs(),
                "lane {i}: batched={} scalar={} (Δ={})",
                batched[i],
                scalar,
                batched[i] - scalar
            );
        }
    }

    /// Below-minimal input must yield zero candidate models without panicking.
    #[test]
    fn under_min_samples_yields_no_model() {
        let est = FundamentalEstimator;
        let mut models = Vec::new();
        est.fit(&[], &mut models);
        assert!(models.is_empty());
        let one = Match2d2d::new(Vec2F64::new(0.0, 0.0), Vec2F64::new(0.0, 0.0));
        est.fit(&[one; 7], &mut models);
        assert!(models.is_empty());
    }

    struct Pair {
        matches: Vec<Match2d2d>,
    }

    fn synthetic_pair() -> Pair {
        let k_fx = 500.0;
        let k_fy = 500.0;
        let k_cx = 320.0;
        let k_cy = 240.0;
        let angle = 0.1_f64;
        let r = [
            [angle.cos(), 0.0, -angle.sin()],
            [0.0, 1.0, 0.0],
            [angle.sin(), 0.0, angle.cos()],
        ];
        let t = [1.0_f64, 0.0, 0.2];

        let pts = [
            Vec3F64::new(-0.5, -0.3, 4.0),
            Vec3F64::new(0.4, -0.2, 3.5),
            Vec3F64::new(-0.3, 0.5, 5.0),
            Vec3F64::new(0.6, 0.4, 4.5),
            Vec3F64::new(-0.1, -0.6, 3.0),
            Vec3F64::new(0.2, 0.3, 6.0),
            Vec3F64::new(-0.4, 0.1, 3.8),
            Vec3F64::new(0.5, -0.5, 4.2),
        ];

        let mut matches = Vec::with_capacity(pts.len());
        for p in &pts {
            let u1 = k_fx * p.x / p.z + k_cx;
            let v1 = k_fy * p.y / p.z + k_cy;
            let pc2 = [
                r[0][0] * p.x + r[0][1] * p.y + r[0][2] * p.z + t[0],
                r[1][0] * p.x + r[1][1] * p.y + r[1][2] * p.z + t[1],
                r[2][0] * p.x + r[2][1] * p.y + r[2][2] * p.z + t[2],
            ];
            let u2 = k_fx * pc2[0] / pc2[2] + k_cx;
            let v2 = k_fy * pc2[1] / pc2[2] + k_cy;
            matches.push(Match2d2d::new(Vec2F64::new(u1, v1), Vec2F64::new(u2, v2)));
        }
        Pair { matches }
    }
}
