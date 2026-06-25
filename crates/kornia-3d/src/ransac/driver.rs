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
use rayon::prelude::*;

/// Per-hypothesis chunk result inside `run_parallel`: best `(model,
/// score, inlier_count, inlier_mask)` produced from one minimal sample,
/// or `None` if the kernel returned no candidates. Aliased to keep the
/// inner `par_iter` closure's return type readable.
type ChunkBest<M> = Option<(M, f64, usize, Vec<bool>)>;

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
    // LO-RANSAC scratch: when the configured `lo_every` is non-zero we
    // periodically refit on the *current best* inlier set and try the
    // polished model. Allocated once, reused across all LO triggers.
    let mut lo_inlier_buf: Vec<E::Sample> = Vec::new();
    let mut lo_models: Vec<E::Model> = Vec::new();
    let mut accepted_since_lo: u32 = 0;
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
                accepted_since_lo += 1;

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

        // LO-RANSAC step: every `lo_every` accepted hypotheses, refit on
        // the current best inlier set and try the polished model. Skipped
        // when `lo_every == 0` (default) — keeps vanilla RANSAC behaviour
        // identical to the pre-LO driver.
        if cfg.lo_every > 0 && accepted_since_lo >= cfg.lo_every && best_inliers.iter().any(|&b| b)
        {
            accepted_since_lo = 0;
            lo_inlier_buf.clear();
            for (idx, &is_in) in best_inliers.iter().enumerate() {
                if is_in {
                    lo_inlier_buf.push(samples[idx]);
                }
            }
            // Need at least the minimal-sample size to refit at all, and
            // strictly more than that to actually benefit from LO.
            if lo_inlier_buf.len() > E::SAMPLE_SIZE {
                lo_models.clear();
                estimator.refit(&lo_inlier_buf, &mut lo_models);
                for lo_model in lo_models.iter() {
                    estimator.residual_batch(lo_model, samples, &mut residuals);
                    let lo_outcome = consensus.consensus(&residuals, &mut current_inliers);
                    if lo_outcome.score > best_score {
                        best_score = lo_outcome.score;
                        best_model = Some(lo_model.clone());
                        std::mem::swap(&mut best_inliers, &mut current_inliers);
                        // LO acceptance also tightens the adaptive cap.
                        if lo_outcome.inlier_count > 0 {
                            let w = lo_outcome.inlier_count as f64 / n as f64;
                            let new_max =
                                adaptive_max_iters(w, E::SAMPLE_SIZE, cfg.confidence, max_iters);
                            if new_max < max_iters {
                                max_iters = new_max;
                            }
                        }
                    }
                }
            }
        }

        i += 1;
    }

    let mut final_model = best_model.clone();

    if final_model.is_some() {
        let mut final_inliers_buf = Vec::with_capacity(n);
        for (idx, &is_in) in best_inliers.iter().enumerate() {
            if is_in {
                final_inliers_buf.push(samples[idx]);
            }
        }

        if final_inliers_buf.len() > E::SAMPLE_SIZE {
            let mut final_models = Vec::new();
            estimator.refit(&final_inliers_buf, &mut final_models);

            for fm in final_models.iter() {
                estimator.residual_batch(fm, samples, &mut residuals);
                let final_outcome = consensus.consensus(&residuals, &mut current_inliers);
                if final_outcome.score >= best_score {
                    best_score = final_outcome.score;
                    final_model = Some(fm.clone());
                    best_inliers = current_inliers.clone();
                }
            }
        }
    }

    RansacResult {
        model: final_model,
        inliers: best_inliers,
        num_iters: i,
        score: best_score,
    }
}

/// Run RANSAC with parallel hypothesis evaluation via `rayon`.
///
/// Same semantics as [`run`] but evaluates batches of minimal-sample
/// hypotheses in parallel. Sampling stays serial (the [`Sampler`] is
/// `&mut`, so single-threaded by construction); only the per-hypothesis
/// `fit` + `residual_batch` + `consensus` work distributes across the
/// rayon thread pool.
///
/// **When to pick this over [`run`]:** the rayon overhead (~1 µs per
/// chunk dispatch) is worth it once `samples.len() ≳ 200` *and* the
/// minimal solver is non-trivial (5-point E, EPnP). For F-8pt with
/// N ≤ 100 the serial driver is faster.
///
/// **Bounds note:** the extra `Sync` requirements on `E` and `C` are
/// trivially met by every estimator in this crate (they're all unit
/// structs or POD-state holders). Provided as a separate function so
/// callers with non-Sync estimators can keep using [`run`].
pub fn run_parallel<E, C, S>(
    estimator: &E,
    consensus: &C,
    sampler: &mut S,
    samples: &[E::Sample],
    cfg: &RansacConfig,
) -> RansacResult<E::Model>
where
    E: Estimator + Sync,
    E::Sample: Copy + Send + Sync,
    E::Model: Clone + Send + Sync,
    C: Consensus + Sync,
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

    // Chunk size: enough work per rayon dispatch to amortise the ~1 µs
    // task-spawn overhead, but small enough that adaptive cap shrinks
    // can fire between chunks (fewer wasted hypotheses on easy data).
    let n_threads = rayon::current_num_threads().max(1);
    let chunk_size = (n_threads * 4).max(32);

    // Per-chunk minimal samples are pre-generated serially (Sampler is
    // single-threaded by trait bound).
    let mut chunk_samples: Vec<Vec<E::Sample>> = Vec::with_capacity(chunk_size);
    let mut sample_idx = vec![0usize; E::SAMPLE_SIZE];

    let mut best_score = f64::NEG_INFINITY;
    let mut best_model: Option<E::Model> = None;
    let mut best_inliers: Vec<bool> = Vec::with_capacity(n);

    let mut max_iters = cfg.max_iters;
    let mut i: u32 = 0;

    while i < max_iters {
        // Generate up to `chunk_size` minimal samples without exceeding
        // the remaining iteration budget.
        let remaining = max_iters - i;
        let this_chunk = chunk_size.min(remaining as usize);
        chunk_samples.clear();
        for _ in 0..this_chunk {
            sampler.sample(n, &mut sample_idx);
            let mut buf: Vec<E::Sample> = Vec::with_capacity(E::SAMPLE_SIZE);
            for &idx in sample_idx.iter() {
                buf.push(samples[idx]);
            }
            chunk_samples.push(buf);
        }

        // Per-chunk parallel evaluation. Each thread allocates its own
        // `models`, `residuals`, and `inliers` buffers — small (10 ×
        // model + N × f64 + N × bool ≈ a few KB) and short-lived.
        let chunk_results: Vec<ChunkBest<E::Model>> = chunk_samples
            .par_iter()
            .map(|sample_buf| {
                let mut models: Vec<E::Model> = Vec::with_capacity(10);
                let mut residuals = vec![0.0f64; n];
                let mut inliers: Vec<bool> = Vec::with_capacity(n);
                estimator.fit(sample_buf, &mut models);

                let mut local_best: Option<(E::Model, f64, usize, Vec<bool>)> = None;
                for model in models.iter() {
                    estimator.residual_batch(model, samples, &mut residuals);
                    let outcome = consensus.consensus(&residuals, &mut inliers);
                    let take = match &local_best {
                        None => true,
                        Some((_, s, _, _)) => outcome.score > *s,
                    };
                    if take {
                        local_best = Some((
                            model.clone(),
                            outcome.score,
                            outcome.inlier_count,
                            inliers.clone(),
                        ));
                    }
                }
                local_best
            })
            .collect();

        // Reduce: pick best across the chunk, update global best.
        for entry in chunk_results.into_iter().flatten() {
            let (model, score, inlier_count, inliers) = entry;
            if score > best_score {
                best_score = score;
                best_model = Some(model);
                best_inliers = inliers;
                if inlier_count > 0 {
                    let w = inlier_count as f64 / n as f64;
                    let new_max = adaptive_max_iters(w, E::SAMPLE_SIZE, cfg.confidence, max_iters);
                    if new_max < max_iters {
                        max_iters = new_max;
                    }
                }
            }
        }

        i += this_chunk as u32;
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

    /// `run_parallel` must produce a result at least as good as `run` on
    /// identical input. (Strict equality is not guaranteed because the
    /// parallel path evaluates hypotheses in chunks and may visit a
    /// slightly different prefix before the adaptive cap fires; both
    /// must still recover a model with ≥ 80% true-inlier recall.)
    #[test]
    fn run_parallel_matches_serial_quality() {
        let pair = synthetic_with_outliers(80, 50, 0xCAFE);
        let est = FundamentalEstimator;
        let consensus = ThresholdConsensus { threshold: 4.0 };
        let cfg = RansacConfig {
            max_iters: 600,
            confidence: 0.999,
            inlier_threshold: 4.0,
            ..Default::default()
        };

        let mut sampler_serial = UniformSampler::new(StdRng::seed_from_u64(11));
        let serial = run(&est, &consensus, &mut sampler_serial, &pair.matches, &cfg);

        let mut sampler_par = UniformSampler::new(StdRng::seed_from_u64(11));
        let par = run_parallel(&est, &consensus, &mut sampler_par, &pair.matches, &cfg);

        assert!(serial.model.is_some() && par.model.is_some());
        // Both must recover ≥ 64/80 true inliers (80%).
        let serial_inliers = serial.inliers[..80].iter().filter(|&&b| b).count();
        let par_inliers = par.inliers[..80].iter().filter(|&&b| b).count();
        assert!(serial_inliers >= 64, "serial: {serial_inliers}");
        assert!(par_inliers >= 64, "parallel: {par_inliers}");
        // Scores in the same ballpark — parallel can be slightly off
        // due to chunked scheduling, but not by much.
        let ratio = par.score / serial.score;
        assert!(
            ratio > 0.85,
            "parallel score {} regressed vs serial {}",
            par.score,
            serial.score
        );
    }

    /// LO-RANSAC with `lo_every=5` should produce a strictly better
    /// (or at minimum equal) score than plain RANSAC on the same noisy
    /// outlier-laden data, because the LO refit polishes the F matrix
    /// over the full inlier set instead of the 8-point minimal sample.
    #[test]
    fn lo_ransac_does_not_regress_plain_ransac() {
        let pair = synthetic_with_outliers(60, 40, 0xBEEF);
        let est = FundamentalEstimator;
        let consensus = ThresholdConsensus { threshold: 4.0 };
        let cfg_plain = RansacConfig {
            max_iters: 500,
            confidence: 0.999,
            inlier_threshold: 4.0,
            lo_every: 0,
            ..Default::default()
        };
        let cfg_lo = RansacConfig {
            lo_every: 5,
            ..cfg_plain.clone()
        };

        let mut sampler_plain = UniformSampler::new(StdRng::seed_from_u64(7));
        let plain = run(
            &est,
            &consensus,
            &mut sampler_plain,
            &pair.matches,
            &cfg_plain,
        );

        let mut sampler_lo = UniformSampler::new(StdRng::seed_from_u64(7));
        let lo = run(&est, &consensus, &mut sampler_lo, &pair.matches, &cfg_lo);

        // LO must not regress on plain RANSAC — it has strictly more chances
        // to find a better hypothesis (it considers every plain candidate
        // *plus* periodic refits on the inlier set).
        assert!(
            lo.score >= plain.score,
            "LO regressed: plain.score={} lo.score={}",
            plain.score,
            lo.score
        );
        assert!(lo.model.is_some());
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
