//! Generic RANSAC infrastructure shared by all robust geometry estimators.
//!
//! The module is split into three orthogonal traits so that each axis can vary
//! independently:
//!
//! - [`Estimator`] knows how to fit a model from a minimal sample and how to
//!   compute the per-sample residual under that model. One impl per geometric
//!   problem (fundamental, essential, homography, PnP, triangulation, ...).
//! - [`Consensus`] turns a vector of residuals into a scalar score plus an
//!   inlier mask. Swap [`ThresholdConsensus`] for vanilla RANSAC or a future
//!   MAGSAC++ scorer for threshold-free σ-consensus, without touching the
//!   estimator.
//! - [`Sampler`] draws minimal subsets. [`UniformSampler`] is the default;
//!   PROSAC-style guided sampling can plug in later.
//!
//! The driver loop (`core::run`, landing in a follow-up) consumes one of each
//! and emits a [`RansacResult`].
//!
//! # Why the split?
//! Existing estimators in [`crate::pose`] each carry their own copy of the
//! RANSAC loop. Centralising the loop here lets us add LO-RANSAC, adaptive
//! iteration caps, and σ-consensus once instead of per-estimator, and keeps
//! the public surface small enough to bind cleanly from `kornia-py`.

mod config;
mod driver;
pub mod estimators;
pub mod kernels;
pub mod magsac;
mod result;
pub mod samples;
pub mod sprt;

pub use config::{ConsensusKind, RansacConfig};
pub(crate) use driver::adaptive_max_iters;
pub use driver::{run, run_parallel, run_with_rng};
pub use kernels::{
    CauchyKernel, HuberKernel, IdentityKernel, RobustKernel, RobustKernelKind, TukeyKernel,
};
pub use magsac::MagsacConsensus;
pub use result::RansacResult;
pub use samples::{Match2d2d, Match2d3d};
pub use sprt::{SPRTConfig, SPRTState};

use rand::Rng;

/// A minimal-sample model fitter and residual evaluator.
///
/// Implementations are typically zero-sized config carriers (e.g. a
/// `FundamentalEstimator` with a normalisation flag) — the `&self` receiver
/// keeps the door open for tunable solver parameters without leaking them
/// through a thread-local.
pub trait Estimator {
    /// The fitted geometric model (e.g. `Mat3F64` for F/E/H).
    type Model;

    /// One observation consumed by the estimator (e.g. a 2D-2D match for
    /// two-view geometry, a 2D-3D pair for PnP).
    type Sample;

    /// Number of samples a minimal solver needs (7 for F7pt, 8 for F8pt, 5 for E5pt,
    /// 4 for H, 3 for P3P).
    const SAMPLE_SIZE: usize;

    /// Fit candidate models from exactly `SAMPLE_SIZE` samples.
    ///
    /// Pushes 0 or more candidate models into `out`. Most solvers produce
    /// at most one (F-8pt, H-4pt, EPnP); F-7pt returns up to three, while
    /// Nistér's 5-point essential or P3P may produce up to ~10. The driver
    /// clears `out` before each call and scores every candidate it returns.
    ///
    /// A degenerate sample (collinear points, numerical collapse) should
    /// leave `out` empty so the driver skips the hypothesis without
    /// polluting the score distribution.
    fn fit(&self, samples: &[Self::Sample], out: &mut Vec<Self::Model>);

    /// Per-sample residual under `model`. Lower is better. The numeric scale
    /// is estimator-defined (Sampson distance squared, reprojection error
    /// squared, ...) — see each impl for details.
    fn residual(&self, model: &Self::Model, sample: &Self::Sample) -> f64;

    /// Non-minimal refit on a (typically larger) inlier subset.
    ///
    /// Used by the LO-RANSAC step: after a hypothesis lands, the driver
    /// collects its inlier set and calls `refit` to produce a polished
    /// model from the full inlier population. Estimators backed by a
    /// least-squares solver that scales with N (`fundamental_8point`,
    /// `homography_dlt`) override this to call the over-determined path;
    /// the default implementation calls [`Self::fit`] on the first
    /// `SAMPLE_SIZE` inliers, which is correct but loses the LO benefit.
    fn refit(&self, inliers: &[Self::Sample], out: &mut Vec<Self::Model>) {
        if inliers.len() < Self::SAMPLE_SIZE {
            return;
        }
        self.fit(&inliers[..Self::SAMPLE_SIZE], out);
    }

    /// Compute residuals for an entire sample slice in one call.
    ///
    /// Default impl loops [`Self::residual`]. Estimators with non-trivial
    /// per-hypothesis precomputation (transpose, intrinsics caching, etc.)
    /// should override to hoist that work out of the inner loop — the RANSAC
    /// driver calls this once per scored model, so any setup amortises across
    /// all `samples.len()` evaluations.
    ///
    /// `out.len() == samples.len()` is expected (the driver pre-sizes the
    /// scratch buffer). If the lengths differ, only the first
    /// `min(out.len(), samples.len())` residuals are written; implementations
    /// must never write past either slice.
    fn residual_batch(&self, model: &Self::Model, samples: &[Self::Sample], out: &mut [f64]) {
        for (o, s) in out.iter_mut().zip(samples) {
            *o = self.residual(model, s);
        }
    }

    /// Evaluate a hard inlier threshold while producing its mask.
    ///
    /// Estimators with a vectorized residual kernel can override this hook to
    /// fuse residual evaluation and thresholding, and may stop once the
    /// remaining observations cannot beat `best_inlier_count`. Returning
    /// `None` requests the portable `residual_batch` plus [`Consensus`]
    /// path. This hook is only used with [`ThresholdConsensus`] and without
    /// SPRT, so custom consensus strategies retain their exact behavior.
    ///
    /// `residuals` is reusable driver scratch and is provided so an override
    /// can batch residual evaluation without allocating. On
    /// [`ThresholdInlierResult::Complete`], `inliers_out` must contain one
    /// entry per sample. Its contents are ignored after `Pruned`.
    ///
    /// # Arguments
    ///
    /// * `model` - Hypothesis being evaluated.
    /// * `samples` - All observations, in their original scoring order.
    /// * `threshold` - Strict residual cutoff, in the estimator's units.
    /// * `best_inlier_count` - Support of the incumbent hypothesis.
    /// * `residuals` - Reusable scratch with one entry per observation.
    /// * `inliers_out` - Reusable mask, complete only on `Complete`.
    ///
    /// # Returns
    ///
    /// `None` to use the generic path, or a completed count or exact rejection.
    ///
    /// # Errors
    ///
    /// This method is infallible; unsupported inputs should return `None`.
    fn threshold_inliers(
        &self,
        _model: &Self::Model,
        _samples: &[Self::Sample],
        _threshold: f64,
        _best_inlier_count: usize,
        _residuals: &mut [f64],
        _inliers_out: &mut Vec<bool>,
    ) -> Option<ThresholdInlierResult> {
        None
    }
}

/// Result of an estimator-specific hard-threshold scoring pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThresholdInlierResult {
    /// All samples were evaluated and this many satisfy the strict threshold.
    Complete(usize),
    /// The unevaluated suffix cannot beat the current best inlier count.
    Pruned,
}

/// Reduces a vector of residuals to a scalar score plus an inlier mask.
///
/// Higher `score` = better hypothesis. The driver picks the maximum.
pub trait Consensus {
    /// Compute consensus for a single hypothesis.
    ///
    /// `inliers_out` is a caller-owned scratch buffer the driver re-uses
    /// across hypotheses; impls must clear and refill it without keeping
    /// references after the call.
    fn consensus(&self, residuals: &[f64], inliers_out: &mut Vec<bool>) -> ConsensusOutcome;

    /// Return the strict hard threshold when this consensus is plain RANSAC.
    ///
    /// The default keeps arbitrary consensus implementations on the generic
    /// residual-vector path. [`ThresholdConsensus`] exposes its threshold so
    /// estimators with an opt-in fused kernel can avoid a second pass.
    ///
    /// # Arguments
    ///
    /// This method takes no arguments beyond the consensus strategy.
    ///
    /// # Returns
    ///
    /// A strict residual cutoff, or `None` for any other scoring strategy.
    /// Returning `Some` promises a score equal to the strict inlier count.
    ///
    /// # Errors
    ///
    /// This method is infallible.
    ///
    /// # Example
    ///
    /// ```
    /// use kornia_3d::ransac::{Consensus, ThresholdConsensus};
    /// assert_eq!(ThresholdConsensus { threshold: 1.0 }.threshold(), Some(1.0));
    /// ```
    fn threshold(&self) -> Option<f64> {
        None
    }
}

/// Outcome of one consensus evaluation.
#[derive(Debug, Clone, Copy)]
pub struct ConsensusOutcome {
    /// Hypothesis quality. Higher is better.
    pub score: f64,
    /// Number of samples flagged as inliers (informational; redundant with
    /// the mask but cached to avoid a second scan in the driver).
    pub inlier_count: usize,
}

/// Strategy for drawing minimal samples from `[0, n)`.
pub trait Sampler {
    /// Fill `out` with `out.len()` distinct indices in `[0, n)`.
    ///
    /// The driver allocates `out` once and reuses it every iteration, so
    /// impls should write in place without growing the slice.
    fn sample(&mut self, n: usize, out: &mut [usize]);
}

/// Hard-threshold consensus — classic RANSAC.
///
/// Inlier iff `residual < threshold`; score = inlier count. The threshold is
/// in the same units the [`Estimator::residual`] returns (commonly squared
/// pixels for Sampson / reprojection).
#[derive(Debug, Clone, Copy)]
pub struct ThresholdConsensus {
    /// Inlier acceptance threshold (residual units defined by the estimator).
    pub threshold: f64,
}

impl Consensus for ThresholdConsensus {
    fn consensus(&self, residuals: &[f64], inliers_out: &mut Vec<bool>) -> ConsensusOutcome {
        inliers_out.clear();
        inliers_out.reserve(residuals.len());
        let mut count = 0usize;
        for &r in residuals {
            let is_in = r < self.threshold;
            inliers_out.push(is_in);
            count += is_in as usize;
        }
        ConsensusOutcome {
            score: count as f64,
            inlier_count: count,
        }
    }

    #[inline]
    fn threshold(&self) -> Option<f64> {
        Some(self.threshold)
    }
}

/// Uniform-without-replacement sampler.
///
/// Wraps `rand::seq::index::sample`, matching the pattern used elsewhere in
/// this crate (see `pose::twoview`). Carries its own RNG so the driver stays
/// deterministic when seeded.
pub struct UniformSampler<R: Rng> {
    rng: R,
}

impl<R: Rng> UniformSampler<R> {
    /// Wrap an RNG.
    pub fn new(rng: R) -> Self {
        Self { rng }
    }
}

impl<R: Rng> Sampler for UniformSampler<R> {
    fn sample(&mut self, n: usize, out: &mut [usize]) {
        let k = out.len();
        debug_assert!(k <= n, "sample size {k} exceeds population {n}");
        let drawn = rand::seq::index::sample(&mut self.rng, n, k);
        for (slot, idx) in out.iter_mut().zip(drawn.iter()) {
            *slot = idx;
        }
    }
}

/// Clamps `samples` and `out` to their common length, so a batch residual kernel
/// that walks both slices in lockstep (e.g. through raw SIMD pointers) can never
/// write past either of them.
pub(crate) fn clamp_pair<'s, 'o, S>(
    samples: &'s [S],
    out: &'o mut [f64],
) -> (&'s [S], &'o mut [f64]) {
    let n = samples.len().min(out.len());
    (&samples[..n], &mut out[..n])
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_algebra::{Mat3F64, Vec2F64};

    /// Regression: `out.len() == samples.len()` was only a debug_assert, so in
    /// release builds the SIMD kernels wrote past a short `out` slice.
    fn check_residual_batch_stays_in_bounds<E>(est: &E)
    where
        E: Estimator<Sample = Match2d2d, Model = Mat3F64>,
    {
        let samples = vec![Match2d2d::new(Vec2F64::new(1.0, 2.0), Vec2F64::new(3.0, 4.0)); 64];
        let sentinel = -12345.0;
        let mut buf = vec![sentinel; 68];
        est.residual_batch(&Mat3F64::IDENTITY, &samples, &mut buf[..4]);
        assert!(buf[..4].iter().all(|&r| r != sentinel));
        assert!(buf[4..].iter().all(|&r| r == sentinel), "wrote past `out`");

        // Longer `out` than `samples`: only the prefix is written.
        let mut buf = vec![sentinel; 8];
        est.residual_batch(&Mat3F64::IDENTITY, &samples[..3], &mut buf);
        assert!(buf[3..].iter().all(|&r| r == sentinel));
    }

    #[test]
    fn residual_batch_short_out_stays_in_bounds() {
        check_residual_batch_stays_in_bounds(&estimators::FundamentalEstimator);
        check_residual_batch_stays_in_bounds(&estimators::HomographyEstimator);
    }
}
