//! Generic RANSAC driver loop.
//!
//! Drives any [`Estimator`] + [`Consensus`] + [`Sampler`] triple to
//! produce a [`RansacResult`]. Sequential v1; LO step and parallel
//! hypothesis evaluation are deferred to follow-up modules to keep the
//! generic signature clean.
//!
//! Pre-allocates every per-iteration scratch buffer (residuals, inlier
//! masks, sample-index slots, model out-vec) so the hot loop performs
//! zero allocations. The adaptive iteration cap implements the classic
//! Fischler-Bolles stopping rule:
//!
//! ```text
//! m ≥ log(1 - p) / log(1 - w^k)
//! ```
//!
//! where `w` is the observed inlier ratio of the current best model,
//! `k` = `SAMPLE_SIZE`, and `p` = `RansacConfig::confidence`. Each time a
//! better hypothesis lands we tighten `max_iters` downward, never upward.

use crate::ransac::{Consensus, Estimator, RansacConfig, RansacResult, Sampler};

/// Run RANSAC.
///
/// `samples` is the full input correspondence set. The driver draws
/// `E::SAMPLE_SIZE` indices each iteration via `sampler`, calls
/// `estimator.fit` to obtain candidate models, scores each candidate's
/// per-sample residuals through `consensus`, and tracks the best.
///
/// Returns a [`RansacResult`] with the best model, its inlier mask,
/// the score, and the number of iterations actually consumed (which can
/// be well below `cfg.max_iters` once the adaptive cap kicks in).
///
/// # Bounds
/// `E::Sample: Copy` lets the driver index-copy minimal samples into a
/// contiguous scratch buffer without needing `Clone` calls or
/// per-iteration allocations. Both existing sample types
/// ([`super::Match2d2d`], [`super::Match2d3d`]) satisfy this for free.
pub fn run<E, C, S>(
    estimator: &E,
    consensus: &C,
    sampler: &mut S,
    samples: &[E::Sample],
    cfg: &RansacConfig,
) -> RansacResult<E::Model>
where
    E: Estimator,
    E::Sample: Copy,
    E::Model: Clone,
    C: Consensus,
    S: Sampler,
{
    let n = samples.len();
    if n < E::SAMPLE_SIZE || cfg.max_iters == 0 {
        return RansacResult {
            model: None,
            inliers: Vec::new(),
            num_iters: 0,
            score: f64::NEG_INFINITY,
        };
    }

    // Per-iteration scratch buffers — allocated once, reused every loop.
    let mut residuals = vec![0.0f64; n];
    let mut current_inliers: Vec<bool> = Vec::with_capacity(n);
    let mut best_inliers: Vec<bool> = Vec::with_capacity(n);
    let mut sample_idx = vec![0usize; E::SAMPLE_SIZE];
    let mut sample_buf: Vec<E::Sample> = Vec::with_capacity(E::SAMPLE_SIZE);
    // Models out-vec sized for the worst-case multi-solution kernel
    // (Nistér 5pt → up to 10 candidates, P3P → up to 4).
    let mut models: Vec<E::Model> = Vec::with_capacity(10);

    let mut best_score = f64::NEG_INFINITY;
    let mut best_model: Option<E::Model> = None;

    let mut max_iters = cfg.max_iters;
    let mut i: u32 = 0;
    while i < max_iters {
        sampler.sample(n, &mut sample_idx);
        sample_buf.clear();
        for &idx in sample_idx.iter() {
            sample_buf.push(samples[idx]);
        }

        models.clear();
        estimator.fit(&sample_buf, &mut models);

        // Multi-solution kernels: score every candidate; the best across
        // all candidates from this minimal sample feeds the adaptive cap.
        for model in models.iter() {
            // Single batch call — estimators may hoist per-hypothesis work
            // (e.g. F.transpose()) out of the per-sample loop here.
            estimator.residual_batch(model, samples, &mut residuals);
            let outcome = consensus.consensus(&residuals, &mut current_inliers);
            if outcome.score > best_score {
                best_score = outcome.score;
                best_model = Some(model.clone());
                std::mem::swap(&mut best_inliers, &mut current_inliers);

                // Adaptive cap update — only ever tightens.
                if outcome.inlier_count > 0 {
                    let w = outcome.inlier_count as f64 / n as f64;
                    let new_max = adaptive_max_iters(w, E::SAMPLE_SIZE, cfg.confidence, max_iters);
                    if new_max < max_iters {
                        max_iters = new_max;
                    }
                }
            }
        }

        i += 1;
    }

    RansacResult {
        model: best_model,
        inliers: best_inliers,
        num_iters: i,
        score: best_score,
    }
}

/// Recompute the adaptive iteration cap.
///
/// Returns the minimum of `current` and the analytic estimate. Never
/// increases the cap and never returns 0.
#[inline]
fn adaptive_max_iters(inlier_ratio: f64, sample_size: usize, confidence: f64, current: u32) -> u32 {
    if inlier_ratio <= 0.0 {
        return current;
    }
    let p_all_inlier = inlier_ratio.powi(sample_size as i32);
    if p_all_inlier >= 1.0 {
        return 1;
    }
    let conf = confidence.clamp(0.0, 1.0 - 1e-12);
    let denom = (1.0 - p_all_inlier).ln();
    if denom >= 0.0 || !denom.is_finite() {
        return current;
    }
    let raw = (1.0 - conf).ln() / denom;
    if !raw.is_finite() || raw <= 0.0 {
        return current;
    }
    let ceiled = raw.ceil() as u32;
    ceiled.min(current).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ransac::{
        estimators::FundamentalEstimator, Match2d2d, ThresholdConsensus, UniformSampler,
    };
    use kornia_algebra::{Vec2F64, Vec3F64};
    use rand::{rngs::StdRng, SeedableRng};

    /// Below-minimal input → result with no model, zero iters, doesn't panic.
    #[test]
    fn under_min_samples_returns_empty() {
        let est = FundamentalEstimator;
        let consensus = ThresholdConsensus { threshold: 1.0 };
        let mut sampler = UniformSampler::new(StdRng::seed_from_u64(0));
        let result: RansacResult<_> = run(
            &est,
            &consensus,
            &mut sampler,
            &[Match2d2d::new(Vec2F64::new(0.0, 0.0), Vec2F64::new(0.0, 0.0)); 5],
            &RansacConfig::default(),
        );
        assert!(result.model.is_none());
        assert_eq!(result.num_iters, 0);
        assert!(result.inliers.is_empty());
    }

    /// 60 inliers + 40 random outliers → driver should land a model whose
    /// inlier count is close to the ground-truth count (60). Adaptive
    /// stopping should kick in well below `max_iters = 1000`.
    #[test]
    fn recovers_inliers_under_outliers() {
        let pair = synthetic_with_outliers(60, 40, 12345);
        let est = FundamentalEstimator;
        let consensus = ThresholdConsensus { threshold: 4.0 }; // 2px Sampson²
        let mut sampler = UniformSampler::new(StdRng::seed_from_u64(0xC0FFEE));
        let cfg = RansacConfig {
            max_iters: 1000,
            confidence: 0.999,
            inlier_threshold: 4.0,
            ..Default::default()
        };
        let result = run(&est, &consensus, &mut sampler, &pair.matches, &cfg);

        assert!(result.model.is_some(), "driver returned no model");
        let recovered = result.inlier_count();
        // Allow some misclassification under noise — recover at least 80% of true inliers.
        assert!(
            recovered >= 48,
            "recovered only {recovered}/60 true inliers (score = {})",
            result.score
        );
        // The adaptive cap must engage at all — i.e. the loop terminates
        // before the configured `max_iters` because a good hypothesis
        // tightened the bound. (At 60% inliers + 8-pt samples + 0.999
        // confidence, the analytic minimum is ~408 iters, so we don't
        // pin a tighter magic number than `cfg.max_iters` here.)
        assert!(
            result.num_iters < cfg.max_iters,
            "adaptive cap never engaged: ran {} of {} iters",
            result.num_iters,
            cfg.max_iters,
        );
    }

    struct Pair {
        matches: Vec<Match2d2d>,
    }

    /// Generate `n_inliers` epipolar-consistent correspondences plus
    /// `n_outliers` uniformly-random pixel pairs in the same image bounds.
    /// Inliers come first; outliers are appended (the driver doesn't see
    /// the partition — it has to discover it).
    fn synthetic_with_outliers(n_inliers: usize, n_outliers: usize, seed: u64) -> Pair {
        // Same camera + relative pose as fundamental.rs unit tests.
        let fx = 500.0_f64;
        let fy = 500.0_f64;
        let cx = 320.0_f64;
        let cy = 240.0_f64;
        let angle = 0.1_f64;
        let r = [
            [angle.cos(), 0.0, -angle.sin()],
            [0.0, 1.0, 0.0],
            [angle.sin(), 0.0, angle.cos()],
        ];
        let t = [1.0_f64, 0.0, 0.2];

        // Deterministic LCG for both inlier 3D points and outlier pixels.
        let mut state = seed;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f64) / (1u64 << 31) as f64 - 1.0
        };

        let mut matches = Vec::with_capacity(n_inliers + n_outliers);
        for _ in 0..n_inliers {
            let p = Vec3F64::new(next() * 0.6, next() * 0.6, 3.0 + next().abs() * 3.0);
            let u1 = fx * p.x / p.z + cx;
            let v1 = fy * p.y / p.z + cy;
            let pc2 = [
                r[0][0] * p.x + r[0][1] * p.y + r[0][2] * p.z + t[0],
                r[1][0] * p.x + r[1][1] * p.y + r[1][2] * p.z + t[1],
                r[2][0] * p.x + r[2][1] * p.y + r[2][2] * p.z + t[2],
            ];
            let u2 = fx * pc2[0] / pc2[2] + cx;
            let v2 = fy * pc2[1] / pc2[2] + cy;
            matches.push(Match2d2d::new(Vec2F64::new(u1, v1), Vec2F64::new(u2, v2)));
        }
        for _ in 0..n_outliers {
            let u1 = (next() * 0.5 + 0.5) * 640.0;
            let v1 = (next() * 0.5 + 0.5) * 480.0;
            let u2 = (next() * 0.5 + 0.5) * 640.0;
            let v2 = (next() * 0.5 + 0.5) * 480.0;
            matches.push(Match2d2d::new(Vec2F64::new(u1, v1), Vec2F64::new(u2, v2)));
        }
        Pair { matches }
    }
}
