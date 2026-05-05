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

pub use config::{ConsensusKind, RansacConfig};
pub use driver::run;
pub use kernels::{
    CauchyKernel, HuberKernel, IdentityKernel, RobustKernel, RobustKernelKind, TukeyKernel,
};
pub use magsac::MagsacConsensus;
pub use result::RansacResult;
pub use samples::{Match2d2d, Match2d3d};

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

    /// Number of samples a minimal solver needs (8 for F8pt, 5 for E5pt,
    /// 4 for H, 3 for P3P).
    const SAMPLE_SIZE: usize;

    /// Fit candidate models from exactly `SAMPLE_SIZE` samples.
    ///
    /// Pushes 0 or more candidate models into `out`. Most solvers produce
    /// at most one (F-8pt, H-4pt, EPnP); multi-solution kernels like
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

    /// Compute residuals for an entire sample slice in one call.
    ///
    /// Default impl loops [`Self::residual`]. Estimators with non-trivial
    /// per-hypothesis precomputation (transpose, intrinsics caching, etc.)
    /// should override to hoist that work out of the inner loop — the RANSAC
    /// driver calls this once per scored model, so any setup amortises across
    /// all `samples.len()` evaluations.
    ///
    /// `out.len() == samples.len()` must hold; the driver pre-sizes the
    /// scratch buffer.
    fn residual_batch(
        &self,
        model: &Self::Model,
        samples: &[Self::Sample],
        out: &mut [f64],
    ) {
        debug_assert_eq!(out.len(), samples.len());
        for (i, s) in samples.iter().enumerate() {
            out[i] = self.residual(model, s);
        }
    }
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
    fn consensus(
        &self,
        residuals: &[f64],
        inliers_out: &mut Vec<bool>,
    ) -> ConsensusOutcome;
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
    fn consensus(
        &self,
        residuals: &[f64],
        inliers_out: &mut Vec<bool>,
    ) -> ConsensusOutcome {
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
