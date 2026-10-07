//! Fundamental-matrix estimator.
//!
//! Seven-point minimal hypotheses and eight-point inlier refinement share
//! the same SIMD Sampson scoring. The eight-point estimator is also available
//! for callers that need the previous sampling behavior.

mod kernels;

use kernels::{pack_f, sampson_residual_batch, sampson_residual_batch_threshold};
use kornia_algebra::{Mat3F64, Vec2F64};

use crate::pose::fundamental_7point_into;
use crate::pose::{fundamental_8point, sampson_distance};
use crate::ransac::{clamp_pair, Estimator, Match2d2d, ThresholdInlierResult};

/// Seven-point fundamental matrix estimator from 2D-2D pixel correspondences.
///
/// Minimal fitting returns up to three real solutions; inlier sets of eight
/// or more matches are refined with the normalized eight-point solver.
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
    const SAMPLE_SIZE: usize = 7;

    fn fit(&self, samples: &[Self::Sample], out: &mut Vec<Self::Model>) {
        if samples.len() == Self::SAMPLE_SIZE {
            let x1 = std::array::from_fn::<_, 7, _>(|i| samples[i].x1);
            let x2 = std::array::from_fn::<_, 7, _>(|i| samples[i].x2);
            let mut models = [Mat3F64::ZERO; 3];
            if let Ok(count) = fundamental_7point_into(&x1, &x2, &mut models) {
                out.extend_from_slice(&models[..count]);
            }
        } else {
            Fundamental8PointEstimator.fit(samples, out);
        }
    }

    fn refit(&self, inliers: &[Self::Sample], out: &mut Vec<Self::Model>) {
        self.fit(inliers, out);
    }

    #[inline]
    fn residual(&self, model: &Self::Model, sample: &Self::Sample) -> f64 {
        sampson_distance(model, &sample.x1, &sample.x2)
    }

    fn residual_batch(&self, model: &Self::Model, samples: &[Self::Sample], out: &mut [f64]) {
        Fundamental8PointEstimator.residual_batch(model, samples, out);
    }

    /// Scores in fixed SIMD-friendly chunks. A losing seven-point root often
    /// needs only the first chunk: after it cannot exceed the incumbent,
    /// RANSAC can skip both the remaining residuals and mask writes.
    fn threshold_inliers(
        &self,
        model: &Self::Model,
        samples: &[Self::Sample],
        threshold: f64,
        best_inlier_count: usize,
        residuals: &mut [f64],
        inliers_out: &mut Vec<bool>,
    ) -> Option<ThresholdInlierResult> {
        let n = samples.len().min(residuals.len());
        if n != samples.len() {
            return None;
        }

        // Pack once per candidate rather than once per scoring chunk.
        let f = pack_f(model);
        // The selected backend keeps its vector constants live across the
        // fixed-size chunks and performs the same safe pruning check.
        let Some(count) =
            sampson_residual_batch_threshold(f, samples, residuals, threshold, best_inlier_count)
        else {
            return Some(ThresholdInlierResult::Pruned);
        };

        // Only a candidate that can become the winner needs its full mask.
        inliers_out.resize(n, false);
        for (mask, &residual) in inliers_out.iter_mut().zip(residuals.iter()) {
            *mask = residual < threshold;
        }
        Some(ThresholdInlierResult::Complete(count))
    }
}

/// Eight-point fundamental matrix estimator retaining the original RANSAC path.
///
/// Uses normalized eight-point fitting for both minimal samples and inlier
/// refinement, with the same squared-pixel Sampson residuals as
/// [`FundamentalEstimator`].
#[derive(Debug, Clone, Copy, Default)]
pub struct Fundamental8PointEstimator;

impl Estimator for Fundamental8PointEstimator {
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

    /// LO-friendly refit. `fundamental_8point` is a least-squares solver
    /// that takes any N ≥ 8, so we forward straight to `fit` — the existing
    /// implementation already routes through Householder for n ≤ 64 and
    /// MtM/LDLᵀ otherwise.
    fn refit(&self, inliers: &[Self::Sample], out: &mut Vec<Self::Model>) {
        self.fit(inliers, out);
    }

    /// Dispatcher matching the kornia-imgproc `warp/kernels.rs` convention:
    /// aarch64 → NEON unconditionally (NEON is baseline for the supported
    /// linux-aarch64 target), x86_64 → AVX2+FMA when probed at runtime,
    /// otherwise the portable scalar reference.
    ///
    /// All three paths produce identical results to within FMA reordering
    /// noise (≤ 1e-12 relative) — a unit test pins the equivalence.
    fn residual_batch(&self, model: &Self::Model, samples: &[Self::Sample], out: &mut [f64]) {
        // The SIMD kernels below write `out` through raw pointers for every sample;
        // only the first `min(out.len(), samples.len())` entries are computed.
        let (samples, out) = clamp_pair(samples, out);
        sampson_residual_batch(pack_f(model), samples, out);
    }
}

#[cfg(test)]
mod tests;
