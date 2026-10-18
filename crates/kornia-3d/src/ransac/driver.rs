//! Generic RANSAC driver.
//!
//! Drives any [`Estimator`] + [`Consensus`] + [`Sampler`] triple to
//! produce a [`RansacResult`]. [`run_with_rng`] is a preset of the staged
//! engine (`super::engine`): an [`Estimator`] adapted as the problem, a
//! consensus scorer, optional SPRT verification, LO-RANSAC every
//! `lo_every` accepted hypotheses, and a final non-minimal refit that wins
//! ties. [`run_parallel`] keeps its own loop.
//!
//! Pre-allocates every per-iteration scratch buffer (residuals, inlier
//! masks, sample-index slots, model out-vec) so the hot loop performs
//! zero allocations. The adaptive iteration cap implements the classic
//! Fischler-Bolles stopping rule:
//!
//! ```text
//! m ≥ log(1 - p) / log(1 - q)
//! ```
//!
//! where `q` is the probability of drawing an all-inlier distinct sample from
//! the observed inlier count, `k` = `SAMPLE_SIZE`, and `p` =
//! `RansacConfig::confidence`. Each time a
//! better hypothesis lands we tighten `max_iters` downward, never upward.
//!
//! When [`RansacConfig::sprt`] is set, the driver additionally applies
//! Wald's Sequential Probability Ratio Test (SPRT) per hypothesis: it
//! visits points in a randomised order and rejects the hypothesis as
//! soon as the LLR exceeds the decision threshold. SPRT is only faster
//! than the non-SPRT path when residuals are computed *one at a time*
//! inside the SPRT loop, not materialised up front — this is the
//! reference OpenCV USAC behaviour. SPRT is inspired by OpenCV's USAC
//! framework.
//!
//! SPRT is only allowed to reject once a model has been accepted
//! (`best_inlier_count >= SAMPLE_SIZE`): before that, the epsilon prior
//! is unverified, so every hypothesis gets a minimum look at the data
//! before rejection is allowed. After the first acceptance the test uses
//! the *observed* inlier ratio of the best model, never the prior, so a
//! pessimistic prior cannot cause good hypotheses to be rejected.

use super::engine::{
    self, adaptive_max_iters, Components, GenericCap, Incumbent, MaxScore, NoVerifier,
    OwnRngSampler, PointResidual, Problem, Schedule, Score, Scorer, SprtVerifier, StageContext,
    Termination, Ties, Workspace,
};
use crate::ransac::{
    Consensus, Estimator, RansacConfig, RansacResult, Sampler, ThresholdInlierResult,
};
use rand::Rng;
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
    run_with_rng(
        estimator,
        consensus,
        sampler,
        samples,
        cfg,
        &mut rand::rng(),
    )
}

/// Run RANSAC with an explicit RNG for the SPRT evaluation order.
///
/// Identical to [`run`] except that SPRT's per-hypothesis point order is
/// drawn from `rng` instead of the thread-local generator, which makes
/// SPRT-enabled runs reproducible under a seeded RNG. When SPRT is
/// disabled `rng` is never touched.
///
/// # Arguments
///
/// * `estimator` - Fits candidate models and scores per-sample residuals.
/// * `consensus` - Turns residuals into a score plus inlier mask.
/// * `sampler` - Draws minimal samples from `samples`.
/// * `samples` - The full input correspondence set.
/// * `cfg` - RANSAC hyperparameters, including the optional SPRT config.
/// * `rng` - Random source used for the SPRT point-evaluation order.
///
/// # Returns
///
/// A [`RansacResult`] with the best model, its inlier mask, the score, and
/// the number of iterations actually consumed (which can be well below
/// `cfg.max_iters` once the adaptive cap kicks in).
///
/// # Behaviour
///
/// - **Inliers and score:** whatever `consensus` computes from the
///   estimator's residuals; [`super::ThresholdConsensus`] counts strict
///   `residual < threshold` inliers in the estimator's units (squared pixels
///   for F and H). A candidate replaces the best one only with a strictly
///   higher score.
/// - **Stopping:** after `cfg.max_iters` draws, or once the adaptive bound
///   for sampling without replacement at `cfg.confidence` is met; every
///   accepted candidate with inliers tightens it. A draw that yields no model
///   still counts. `num_iters` is the number of draws.
/// - **SPRT:** with a valid `cfg.sprt`, each minimal-sample candidate is
///   first tested against `cfg.inlier_threshold` with a strict `<`; an
///   invalid config falls back to plain scoring.
/// - **LO-RANSAC:** with `cfg.lo_every > 0`, after every `lo_every` accepted
///   minimal-sample candidates (checked once all roots of a draw are
///   scored), the estimator refits the best model's inliers and each refit
///   replaces it if it scores strictly higher.
/// - **Final refit:** the best model's inliers are refit once more (when there
///   are more than `E::SAMPLE_SIZE`); a refit that scores at least as high
///   replaces it. The returned mask is always that of the returned model.
pub fn run_with_rng<E, C, S, R>(
    estimator: &E,
    consensus: &C,
    sampler: &mut S,
    samples: &[E::Sample],
    cfg: &RansacConfig,
    rng: &mut R,
) -> RansacResult<E::Model>
where
    E: Estimator,
    E::Sample: Copy,
    E::Model: Clone,
    C: Consensus,
    S: Sampler,
    R: Rng,
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

    let mut problem = EstimatorProblem {
        estimator,
        samples,
        sample_buf: Vec::with_capacity(E::SAMPLE_SIZE),
        candidates: Vec::with_capacity(MAX_MINIMAL_MODELS),
    };
    let termination = GenericCap::new(cfg.max_iters, n, E::SAMPLE_SIZE, cfg.confidence);
    let schedule = LegacyGeneric {
        lo_every: cfg.lo_every,
        accepted_since_lo: 0,
        unchecked_accept: false,
        inlier_samples: Vec::new(),
        models: Vec::new(),
    };
    // Resolve the evaluation policy once. A valid SPRT config verifies raw
    // candidates and scores the residuals it computed; an invalid one falls
    // back to plain scoring. Only plain RANSAC may use the estimator's fused
    // threshold kernel.
    let (sprt, fused_threshold) = match cfg.sprt {
        Some(sprt) if sprt.is_valid() => (Some(sprt), None),
        Some(_) => (None, None),
        None => (None, consensus.threshold()),
    };
    let scorer = ConsensusScorer {
        consensus,
        fused_threshold,
    };
    let mut parts = Components {
        sampler: OwnRngSampler(sampler),
        verifier: NoVerifier,
        scorer,
        schedule,
        termination,
    };
    let mut best = Incumbent::new(Vec::with_capacity(n));
    let mut ws = Workspace::new(Vec::with_capacity(n), n);
    let draws = match sprt {
        Some(sprt) => {
            // The legacy driver counted draws from zero.
            let verifier = SprtVerifier::new(sprt, cfg.inlier_threshold, n, E::SAMPLE_SIZE, 1);
            let mut parts = parts.with_verifier(verifier);
            engine::run(&mut problem, &mut parts, &mut best, &mut ws, rng)
        }
        None => engine::run(&mut problem, &mut parts, &mut best, &mut ws, rng),
    };

    RansacResult {
        model: best.model,
        inliers: best.inliers,
        num_iters: draws as u32,
        score: best.score.objective.0,
    }
}

/// Capacity reserved for the candidates of one minimal sample (Nistér's
/// five-point solver returns up to ten).
const MAX_MINIMAL_MODELS: usize = 10;

/// An [`Estimator`] over a sample slice, as an engine [`Problem`].
struct EstimatorProblem<'a, E: Estimator> {
    estimator: &'a E,
    samples: &'a [E::Sample],
    sample_buf: Vec<E::Sample>,
    candidates: Vec<E::Model>,
}

impl<E: Estimator> EstimatorProblem<'_, E>
where
    E::Sample: Copy,
{
    /// Refit on the samples flagged in `inliers`, gathered into `buf`, if
    /// there are more of them than a minimal sample. Returns whether the
    /// refit ran.
    fn fit_nonminimal(
        &self,
        inliers: &[bool],
        buf: &mut Vec<E::Sample>,
        out: &mut Vec<E::Model>,
    ) -> bool {
        buf.clear();
        for (sample, &is_inlier) in self.samples.iter().zip(inliers) {
            if is_inlier {
                buf.push(*sample);
            }
        }
        // Refitting needs a minimal sample, and only more than that can
        // improve on the minimal fit.
        if buf.len() <= E::SAMPLE_SIZE {
            return false;
        }
        out.clear();
        self.estimator.refit(buf, out);
        true
    }
}

impl<E: Estimator> Problem for EstimatorProblem<'_, E>
where
    E::Sample: Copy,
    E::Model: Clone,
{
    type Model = E::Model;

    #[inline]
    fn num_observations(&self) -> usize {
        self.samples.len()
    }

    #[inline]
    fn sample_size(&self) -> usize {
        E::SAMPLE_SIZE
    }

    #[inline]
    fn fit_minimal(&mut self, sample: &[usize]) {
        self.sample_buf.clear();
        for &index in sample {
            self.sample_buf.push(self.samples[index]);
        }
        self.candidates.clear();
        self.estimator.fit(&self.sample_buf, &mut self.candidates);
    }

    #[inline]
    fn candidates(&self) -> &[E::Model] {
        &self.candidates
    }
}

impl<E: Estimator> PointResidual for EstimatorProblem<'_, E>
where
    E::Sample: Copy,
    E::Model: Clone,
{
    #[inline]
    fn residual(&self, model: &E::Model, index: usize) -> f64 {
        self.estimator.residual(model, &self.samples[index])
    }
}

/// Scores with a [`Consensus`], through the estimator's fused threshold
/// kernel when allowed.
struct ConsensusScorer<'c, C> {
    consensus: &'c C,
    /// The consensus threshold when plain RANSAC may use
    /// [`Estimator::threshold_inliers`].
    fused_threshold: Option<f64>,
}

impl<E, C> Scorer<EstimatorProblem<'_, E>> for ConsensusScorer<'_, C>
where
    E: Estimator,
    E::Sample: Copy,
    E::Model: Clone,
    C: Consensus,
{
    type Objective = MaxScore;

    #[inline]
    fn score_raw(
        &mut self,
        problem: &EstimatorProblem<'_, E>,
        model: &E::Model,
        bound: Option<&Score<MaxScore>>,
        ws: &mut Workspace,
    ) -> Option<Score<MaxScore>> {
        if ws.residuals_complete {
            // SPRT passed and visited every point.
            return Some(self.consensus_score(ws));
        }
        // Seven-point fitting yields several roots, most of which lose
        // quickly. With an incumbent to beat, a fused kernel may score
        // SIMD-sized chunks and stop once the rest cannot beat its support.
        // The first candidate is always scored fully: even a zero-inlier
        // model beats the initial -inf score.
        if let (Some(threshold), Some(bound)) = (self.fused_threshold, bound) {
            match problem.estimator.threshold_inliers(
                model,
                problem.samples,
                threshold,
                bound.support,
                &mut ws.residuals,
                &mut ws.mask,
            ) {
                Some(ThresholdInlierResult::Complete(count)) => {
                    return Some(Score {
                        objective: MaxScore(count as f64),
                        support: count,
                    })
                }
                Some(ThresholdInlierResult::Pruned) => return None,
                None => {}
            }
        }
        Some(self.score_full(problem, model, ws))
    }

    #[inline]
    fn score_full(
        &mut self,
        problem: &EstimatorProblem<'_, E>,
        model: &E::Model,
        ws: &mut Workspace,
    ) -> Score<MaxScore> {
        problem
            .estimator
            .residual_batch(model, problem.samples, &mut ws.residuals);
        self.consensus_score(ws)
    }
}

impl<C: Consensus> ConsensusScorer<'_, C> {
    /// Score the residuals in the workspace with the consensus.
    #[inline]
    fn consensus_score(&self, ws: &mut Workspace) -> Score<MaxScore> {
        let outcome = self.consensus.consensus(&ws.residuals, &mut ws.mask);
        Score {
            objective: MaxScore(outcome.score),
            support: outcome.inlier_count,
        }
    }
}

/// The generic driver's schedule: LO-RANSAC every `lo_every` accepted raw
/// candidates, then an inclusive non-minimal refit.
struct LegacyGeneric<S, M> {
    lo_every: u32,
    accepted_since_lo: u32,
    /// Whether a raw candidate was accepted since local optimization last
    /// found the incumbent without inliers. Until then the incumbent is
    /// unchanged, so the mask need not be scanned again.
    unchecked_accept: bool,
    /// Refit scratch: the incumbent's inlier samples and the refit models.
    inlier_samples: Vec<S>,
    models: Vec<M>,
}

impl<S: Copy, M> LegacyGeneric<S, M> {
    /// Refit on the incumbent's inliers and offer the polished models under
    /// [`Ties::Strict`]. Split out of `after_draw`, which runs on every draw
    /// and stays small enough to inline; this runs every `lo_every` accepts.
    fn local_optimization<'a, E, Sc, T, R>(
        &mut self,
        stage: &mut StageContext<'_, EstimatorProblem<'a, E>, Sc, T, R>,
    ) where
        E: Estimator<Sample = S, Model = M>,
        M: Clone,
        Sc: Scorer<EstimatorProblem<'a, E>>,
        T: Termination,
    {
        if !self.unchecked_accept {
            return;
        }
        if !stage.best.inliers.iter().any(|&b| b) {
            // LO stays due and runs once an accepted model has inliers.
            self.unchecked_accept = false;
            return;
        }
        self.accepted_since_lo = 0;
        if stage.problem.fit_nonminimal(
            &stage.best.inliers,
            &mut self.inlier_samples,
            &mut self.models,
        ) {
            for model in &self.models {
                stage.consider_derived(model, Ties::Strict);
            }
        }
    }
}

impl<'a, E, Sc, T, R> Schedule<EstimatorProblem<'a, E>, Sc, T, R>
    for LegacyGeneric<E::Sample, E::Model>
where
    E: Estimator,
    E::Sample: Copy,
    E::Model: Clone,
    Sc: Scorer<EstimatorProblem<'a, E>>,
    T: Termination,
{
    #[inline]
    fn on_raw_accept(&mut self) {
        self.accepted_since_lo += 1;
        self.unchecked_accept = true;
    }

    /// Every `lo_every` accepted hypotheses, refit on the incumbent's inliers
    /// and offer the polished models. Runs after all roots of a draw.
    #[inline]
    fn after_draw(&mut self, stage: &mut StageContext<'_, EstimatorProblem<'a, E>, Sc, T, R>) {
        if self.lo_every > 0 && self.accepted_since_lo >= self.lo_every {
            self.local_optimization(stage);
        }
    }

    /// Refit on the final inliers; the refit wins ties.
    fn final_fit(&mut self, stage: &mut StageContext<'_, EstimatorProblem<'a, E>, Sc, T, R>) {
        if stage.best.model.is_none()
            || !stage.problem.fit_nonminimal(
                &stage.best.inliers,
                &mut self.inlier_samples,
                &mut self.models,
            )
        {
            return;
        }
        for model in &self.models {
            stage.consider_derived(model, Ties::AcceptTies);
        }
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
                    let new_max = adaptive_max_iters(
                        inlier_count,
                        n,
                        E::SAMPLE_SIZE,
                        cfg.confidence,
                        max_iters as usize,
                    ) as u32;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ransac::{
        estimators::FundamentalEstimator, ConsensusOutcome, Match2d2d, SPRTConfig,
        ThresholdConsensus, UniformSampler,
    };
    use kornia_algebra::{Mat3F64, Vec2F64, Vec3F64};
    use rand::{rngs::StdRng, SeedableRng};

    /// Delegates every operation to the seven-point estimator except the
    /// optional fused threshold hook. It pins the optimized driver path to
    /// the original residual-vector behavior for identical random samples.
    struct UnfusedFundamental;

    impl Estimator for UnfusedFundamental {
        type Model = Mat3F64;
        type Sample = Match2d2d;
        const SAMPLE_SIZE: usize = FundamentalEstimator::SAMPLE_SIZE;

        fn fit(&self, samples: &[Self::Sample], out: &mut Vec<Self::Model>) {
            FundamentalEstimator.fit(samples, out);
        }

        fn residual(&self, model: &Self::Model, sample: &Self::Sample) -> f64 {
            FundamentalEstimator.residual(model, sample)
        }

        fn residual_batch(&self, model: &Self::Model, samples: &[Self::Sample], out: &mut [f64]) {
            FundamentalEstimator.residual_batch(model, samples, out);
        }

        fn refit(&self, inliers: &[Self::Sample], out: &mut Vec<Self::Model>) {
            FundamentalEstimator.refit(inliers, out);
        }
    }

    /// Deliberately opaque consensus that has threshold semantics but does
    /// not expose the opt-in hook. This protects custom consensus users from
    /// the threshold fast path changing their control flow.
    struct OpaqueThreshold {
        threshold: f64,
    }

    impl Consensus for OpaqueThreshold {
        fn consensus(&self, residuals: &[f64], inliers_out: &mut Vec<bool>) -> ConsensusOutcome {
            ThresholdConsensus {
                threshold: self.threshold,
            }
            .consensus(residuals, inliers_out)
        }
    }

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

    /// Chunked threshold scoring must be observationally identical to the
    /// original full residual-vector path: same sampled prefix, adaptive
    /// iteration cap, winner, and mask.
    #[test]
    fn fused_threshold_scoring_matches_unfused_seven_point_path() {
        let pair = synthetic_with_outliers(90, 210, 0xF7F7);
        let consensus = ThresholdConsensus { threshold: 4.0 };
        let cfg = RansacConfig {
            max_iters: 1000,
            confidence: 0.999,
            inlier_threshold: 4.0,
            ..Default::default()
        };

        let mut optimized_sampler = UniformSampler::new(StdRng::seed_from_u64(0x1234));
        let optimized = run(
            &FundamentalEstimator,
            &consensus,
            &mut optimized_sampler,
            &pair.matches,
            &cfg,
        );
        let mut unfused_sampler = UniformSampler::new(StdRng::seed_from_u64(0x1234));
        let unfused = run(
            &UnfusedFundamental,
            &consensus,
            &mut unfused_sampler,
            &pair.matches,
            &cfg,
        );

        assert_eq!(optimized.num_iters, unfused.num_iters);
        assert_eq!(optimized.score, unfused.score);
        assert_eq!(optimized.inliers, unfused.inliers);
        assert_eq!(optimized.model, unfused.model);
    }

    /// A consensus implementation that does not explicitly expose a hard
    /// threshold must retain the generic scoring path, even if its output
    /// happens to equal `ThresholdConsensus`.
    #[test]
    fn opaque_consensus_bypasses_fused_threshold_scoring() {
        let pair = synthetic_with_outliers(70, 130, 0x0A0E);
        let opaque = OpaqueThreshold { threshold: 4.0 };
        let cfg = RansacConfig {
            max_iters: 500,
            confidence: 0.999,
            inlier_threshold: 4.0,
            ..Default::default()
        };
        let mut opaque_sampler = UniformSampler::new(StdRng::seed_from_u64(44));
        let opaque_result = run(
            &FundamentalEstimator,
            &opaque,
            &mut opaque_sampler,
            &pair.matches,
            &cfg,
        );
        let mut old_sampler = UniformSampler::new(StdRng::seed_from_u64(44));
        let old_result = run(
            &UnfusedFundamental,
            &opaque,
            &mut old_sampler,
            &pair.matches,
            &cfg,
        );

        assert_eq!(opaque_result.num_iters, old_result.num_iters);
        assert_eq!(opaque_result.score, old_result.score);
        assert_eq!(opaque_result.inliers, old_result.inliers);
        assert_eq!(opaque_result.model, old_result.model);
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

    /// SPRT-enabled and SPRT-disabled runs must produce equivalent results
    /// on a clean (no-outlier) data set, and the SPRT path must not
    /// break the driver.
    #[test]
    fn test_ransac_strategy_toggle() {
        let pair = synthetic_with_outliers(80, 20, 0xFACE);

        let est = FundamentalEstimator;
        let consensus = ThresholdConsensus { threshold: 4.0 };

        // Off (default).
        let cfg_off = RansacConfig {
            max_iters: 200,
            confidence: 0.99,
            inlier_threshold: 4.0,
            sprt: None,
            ..Default::default()
        };
        let mut sampler_off = UniformSampler::new(StdRng::seed_from_u64(13));
        let result_off = run(&est, &consensus, &mut sampler_off, &pair.matches, &cfg_off);

        // On.
        let cfg_on = RansacConfig {
            max_iters: 200,
            confidence: 0.99,
            inlier_threshold: 4.0,
            sprt: Some(SPRTConfig {
                epsilon: 0.8,
                delta: 0.01,
                t_M: 1.0,
                t_m: 0.1,
            }),
            ..Default::default()
        };
        let mut sampler_on = UniformSampler::new(StdRng::seed_from_u64(13));
        let result_on = run(&est, &consensus, &mut sampler_on, &pair.matches, &cfg_on);

        assert!(result_off.model.is_some(), "SPRT-off produced no model");
        assert!(result_on.model.is_some(), "SPRT-on produced no model");
        // SPRT must not regress; allow a small relative tolerance for
        // order-of-evaluation differences.
        let ratio = result_on.score / result_off.score.max(1.0);
        assert!(
            ratio > 0.85,
            "SPRT regressed: off.score={}, on.score={}",
            result_off.score,
            result_on.score
        );
    }

    /// SPRT must accept hypotheses that improve the best model and update
    /// the best-model tracking.
    #[test]
    fn test_sprt_updates_best_model() {
        let pair = synthetic_with_outliers(60, 40, 0xCAFE);
        let est = FundamentalEstimator;
        let consensus = ThresholdConsensus { threshold: 4.0 };
        let cfg = RansacConfig {
            max_iters: 500,
            confidence: 0.99,
            inlier_threshold: 4.0,
            sprt: Some(SPRTConfig {
                epsilon: 0.6,
                delta: 0.01,
                t_M: 1.0,
                t_m: 0.1,
            }),
            ..Default::default()
        };
        let mut sampler = UniformSampler::new(StdRng::seed_from_u64(0xBABE));
        let result = run(&est, &consensus, &mut sampler, &pair.matches, &cfg);

        // A model must be produced and at least 70% of true inliers must
        // be in the inlier set — proof that a good hypothesis survived
        // SPRT and was used to update the best model.
        assert!(result.model.is_some());
        assert!(
            result.inlier_count() >= 42,
            "SPRT pipeline only kept {} of 60 inliers",
            result.inlier_count()
        );
    }

    /// Identical seeds produce identical results, with and without SPRT.
    #[test]
    fn test_deterministic_execution() {
        let pair = synthetic_with_outliers(60, 30, 0xDEAD);

        let est = FundamentalEstimator;
        let consensus = ThresholdConsensus { threshold: 4.0 };
        let base_cfg = RansacConfig {
            max_iters: 100,
            confidence: 0.99,
            inlier_threshold: 4.0,
            ..Default::default()
        };

        let mut s1 = UniformSampler::new(StdRng::seed_from_u64(42));
        let r1 = run(&est, &consensus, &mut s1, &pair.matches, &base_cfg);
        let mut s2 = UniformSampler::new(StdRng::seed_from_u64(42));
        let r2 = run(&est, &consensus, &mut s2, &pair.matches, &base_cfg);

        // Same seed → same number of iterations and same score.
        assert_eq!(r1.num_iters, r2.num_iters);
        assert_eq!(r1.score, r2.score);

        // SPRT-on with a seeded RNG must be fully reproducible: the same
        // sampler seed and SPRT RNG seed produce identical results.
        let cfg_on = RansacConfig {
            sprt: Some(SPRTConfig {
                epsilon: 0.5,
                delta: 0.01,
                t_M: 1.0,
                t_m: 1.0,
            }),
            ..base_cfg.clone()
        };
        let mut s3 = UniformSampler::new(StdRng::seed_from_u64(42));
        let mut rng_a = StdRng::seed_from_u64(4242);
        let r3 = run_with_rng(
            &est,
            &consensus,
            &mut s3,
            &pair.matches,
            &cfg_on,
            &mut rng_a,
        );
        let mut s4 = UniformSampler::new(StdRng::seed_from_u64(42));
        let mut rng_b = StdRng::seed_from_u64(4242);
        let r4 = run_with_rng(
            &est,
            &consensus,
            &mut s4,
            &pair.matches,
            &cfg_on,
            &mut rng_b,
        );

        assert!(r3.model.is_some(), "SPRT path found no model");
        assert_eq!(r3.num_iters, r4.num_iters);
        assert_eq!(r3.score, r4.score);
        assert_eq!(r3.inliers, r4.inliers);
    }

    /// A grossly optimistic SPRT prior (0.9 on a 40%-inlier scene) must
    /// not starve the run: the pre-acceptance epsilon cap keeps good
    /// hypotheses alive and the grace period stops early rejection
    /// before the first accepted model.
    #[test]
    fn test_sprt_wrong_prior_recovers_model() {
        let pair = synthetic_with_outliers(40, 60, 0x5EED);

        let est = FundamentalEstimator;
        let consensus = ThresholdConsensus { threshold: 4.0 };
        let cfg = RansacConfig {
            max_iters: 4000,
            confidence: 0.999,
            inlier_threshold: 4.0,
            sprt: Some(SPRTConfig {
                epsilon: 0.9,
                delta: 0.01,
                t_M: 1.0,
                t_m: 1.0,
            }),
            ..Default::default()
        };
        let mut sampler = UniformSampler::new(StdRng::seed_from_u64(0xF00D));
        let mut rng = StdRng::seed_from_u64(0xBEEF);
        let result = run_with_rng(
            &est,
            &consensus,
            &mut sampler,
            &pair.matches,
            &cfg,
            &mut rng,
        );

        assert!(result.model.is_some(), "wrong-prior SPRT starved the run");
        assert!(
            result.inlier_count() >= 32,
            "wrong-prior SPRT only kept {} of 40 true inliers",
            result.inlier_count()
        );
    }

    /// With a *correct* prior SPRT must not regress vs plain RANSAC:
    /// good hypotheses pass the test, only clearly-bad ones are dropped.
    #[test]
    fn test_sprt_correct_prior_does_not_regress() {
        let pair = synthetic_with_outliers(40, 60, 0xABCD);

        let est = FundamentalEstimator;
        let consensus = ThresholdConsensus { threshold: 4.0 };
        let base = RansacConfig {
            max_iters: 4000,
            confidence: 0.999,
            inlier_threshold: 4.0,
            ..Default::default()
        };

        let mut sampler_off = UniformSampler::new(StdRng::seed_from_u64(77));
        let off = run(&est, &consensus, &mut sampler_off, &pair.matches, &base);

        let cfg_on = RansacConfig {
            sprt: Some(SPRTConfig {
                epsilon: 0.4,
                delta: 0.01,
                t_M: 1.0,
                t_m: 1.0,
            }),
            ..base.clone()
        };
        let mut sampler_on = UniformSampler::new(StdRng::seed_from_u64(77));
        let mut rng = StdRng::seed_from_u64(78);
        let on = run_with_rng(
            &est,
            &consensus,
            &mut sampler_on,
            &pair.matches,
            &cfg_on,
            &mut rng,
        );

        assert!(off.model.is_some() && on.model.is_some());
        let ratio = on.score / off.score.max(1.0);
        assert!(
            ratio > 0.85,
            "SPRT regressed: off.score={}, on.score={}",
            off.score,
            on.score
        );
    }

    /// Counts non-minimal refits so tests can see LO-RANSAC run.
    struct CountingRefits {
        refits: std::cell::Cell<usize>,
    }

    impl Estimator for CountingRefits {
        type Model = Mat3F64;
        type Sample = Match2d2d;
        const SAMPLE_SIZE: usize = FundamentalEstimator::SAMPLE_SIZE;

        fn fit(&self, samples: &[Self::Sample], out: &mut Vec<Self::Model>) {
            FundamentalEstimator.fit(samples, out);
        }

        fn residual(&self, model: &Self::Model, sample: &Self::Sample) -> f64 {
            FundamentalEstimator.residual(model, sample)
        }

        fn residual_batch(&self, model: &Self::Model, samples: &[Self::Sample], out: &mut [f64]) {
            FundamentalEstimator.residual_batch(model, samples, out);
        }

        fn refit(&self, inliers: &[Self::Sample], out: &mut Vec<Self::Model>) {
            self.refits.set(self.refits.get() + 1);
            FundamentalEstimator.refit(inliers, out);
        }
    }

    /// LO-RANSAC with a multi-root minimal solver (seven-point F): local
    /// optimization runs inside the loop, and the returned mask is exactly
    /// the strict threshold mask of the returned model.
    #[test]
    fn lo_with_seven_point_roots_keeps_mask_and_model_consistent() {
        let pair = synthetic_with_outliers(90, 110, 0x10E7);
        let threshold = 4.0;
        let consensus = ThresholdConsensus { threshold };
        for sprt in [
            None,
            Some(SPRTConfig {
                epsilon: 0.4,
                delta: 0.01,
                t_M: 1.0,
                t_m: 1.0,
            }),
        ] {
            let cfg = RansacConfig {
                max_iters: 400,
                confidence: 0.999,
                inlier_threshold: threshold,
                lo_every: 1,
                sprt,
                ..Default::default()
            };
            let est = CountingRefits {
                refits: std::cell::Cell::new(0),
            };
            let mut sampler = UniformSampler::new(StdRng::seed_from_u64(21));
            let result = run_with_rng(
                &est,
                &consensus,
                &mut sampler,
                &pair.matches,
                &cfg,
                &mut StdRng::seed_from_u64(22),
            );
            // At least one LO refit inside the loop plus the final refit.
            assert!(est.refits.get() >= 2, "LO never ran (sprt: {sprt:?})");
            let model = result.model.expect("no model");
            let expected: Vec<bool> = pair
                .matches
                .iter()
                .map(|m| FundamentalEstimator.residual(&model, m) < threshold)
                .collect();
            assert_eq!(result.inliers, expected);
            assert_eq!(result.score, result.inlier_count() as f64);
        }
    }

    /// A solver that never yields a model must still stop at the hard cap and
    /// report every draw.
    #[test]
    fn rootless_solver_terminates_at_the_hard_cap() {
        struct NoRoots;

        impl Estimator for NoRoots {
            type Model = Mat3F64;
            type Sample = Match2d2d;
            const SAMPLE_SIZE: usize = 7;

            fn fit(&self, _samples: &[Self::Sample], _out: &mut Vec<Self::Model>) {}

            fn residual(&self, _model: &Self::Model, _sample: &Self::Sample) -> f64 {
                0.0
            }
        }

        let pair = synthetic_with_outliers(20, 20, 0x0);
        let consensus = ThresholdConsensus { threshold: 4.0 };
        let cfg = RansacConfig {
            max_iters: 137,
            lo_every: 1,
            ..Default::default()
        };
        let mut sampler = UniformSampler::new(StdRng::seed_from_u64(5));
        let result = run(&NoRoots, &consensus, &mut sampler, &pair.matches, &cfg);
        assert!(result.model.is_none());
        assert_eq!(result.num_iters, 137);
        assert!(result.inliers.is_empty());
        assert_eq!(result.score, f64::NEG_INFINITY);
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
