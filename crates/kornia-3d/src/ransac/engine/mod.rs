//! The staged RANSAC loop shared by every robust estimator in this crate.
//!
//! Every RANSAC variant fits one loop; they differ only in which component
//! sits at each stage and when it runs (USAC, OpenCV `cv::usac`, PoseLib):
//!
//! ```text
//! while !termination.done(draws):
//!     sampler.sample(rng); draws += 1        rejected samples count as draws
//!     problem.check_sample                   e.g. collinear homography samples
//!     problem.fit_minimal → candidates       F7 ≤ 3, E5 ≤ 10, P3P ≤ 4 roots
//!     for each candidate:
//!         problem.check_model                cheap model checks
//!         verifier.verify                    statistical (SPRT), raw candidates only
//!         scorer.score_raw(incumbent)        exact; may prune a candidate that cannot win
//!         incumbent.consider(Strict)         the guarded replacement
//!     termination.after_draw; schedule.after_draw          generic LO
//! schedule.after_sampling                                  LO+ → DEGENSAC
//! schedule.final_fit                                       non-minimal refits
//! ```
//!
//! Components are generic parameters, so each preset monomorphizes its own
//! loop with no per-candidate dispatch. Derived candidates (local
//! optimization, repairs, final fits) never pass the verifier; schedules
//! offer them through [`StageContext::consider_derived`], which scores them
//! exhaustively and replaces the incumbent through [`Incumbent::consider`]
//! under an explicit [`Ties`] rule. The one other replacement path,
//! [`Incumbent::consider_keeping_mask`], exists only for two legacy final
//! fits that keep their pre-refit mask.
//!
//! This is phase A of #1171: each public entry point plugs in *legacy*
//! components that reproduce its previous loop exactly, including its RNG
//! stream, tie rules and draw accounting. Deliberate behaviour changes come
//! later, one at a time.

mod score;
mod sprt;
mod termination;

pub(crate) use score::{
    CountThenResidual, Incumbent, InlierCount, MaxScore, Objective, Score, Ties,
};
pub(crate) use sprt::SprtVerifier;
pub(crate) use termination::{adaptive_max_iters, CapBase, GenericCap, TwoViewCap};

use rand::Rng;

/// Geometry of one robust estimation problem, prepared for a single run.
///
/// Implementations own the run's data in whatever layout their solvers and
/// scoring kernels want (SoA coordinates, normalized points, ...), so
/// preparation happens once, before the loop.
pub(crate) trait Problem {
    /// The estimated model.
    type Model: Clone;

    /// Number of observations.
    fn num_observations(&self) -> usize;

    /// Size of a minimal sample.
    fn sample_size(&self) -> usize;

    /// Reject a degenerate sample before fitting. The draw still counts.
    #[inline]
    fn check_sample(&self, _sample: &[usize]) -> bool {
        true
    }

    /// Fit the candidates of a minimal sample. A degenerate sample yields
    /// none.
    fn fit_minimal(&mut self, sample: &[usize]);

    /// The candidates of the last [`Self::fit_minimal`] call.
    ///
    /// The problem owns this buffer, so solvers that return a fixed-size
    /// array (seven-point F, four-point H) are read without copying.
    fn candidates(&self) -> &[Self::Model];

    /// Reject a raw candidate of `sample` with a cheap check before it is
    /// verified or scored.
    #[inline]
    fn check_model(&self, _sample: &[usize], _model: &Self::Model) -> bool {
        true
    }
}

/// Per-observation residuals, for verifiers that visit points one at a time.
pub(crate) trait PointResidual: Problem {
    /// Residual of observation `index` under `model`.
    fn residual(&self, model: &Self::Model, index: usize) -> f64;
}

/// Draws minimal samples.
pub(crate) trait MinimalSampler<R> {
    /// Fill `out` with `out.len()` distinct indices in `0..population`.
    fn sample(&mut self, rng: &mut R, population: usize, out: &mut [usize]);
}

/// Uniform sampling without replacement from the run's RNG.
///
/// Returns the same indices, and leaves the RNG in the same state, as
/// `rand::seq::index::sample`.
pub(crate) struct UniformDraw;

impl<R: Rng> MinimalSampler<R> for UniformDraw {
    #[inline]
    fn sample(&mut self, rng: &mut R, population: usize, out: &mut [usize]) {
        crate::ransac::sample_distinct_indices(rng, population, out);
    }
}

/// A public [`crate::ransac::Sampler`], which carries its own RNG.
pub(crate) struct OwnRngSampler<'a, S>(pub &'a mut S);

impl<S: crate::ransac::Sampler, R> MinimalSampler<R> for OwnRngSampler<'_, S> {
    #[inline]
    fn sample(&mut self, _rng: &mut R, population: usize, out: &mut [usize]) {
        self.0.sample(population, out);
    }
}

/// What a verifier knows about the run.
pub(crate) struct VerifyContext {
    /// Draws made so far, including the current one.
    pub draws: usize,
    /// Support of the incumbent.
    pub best_support: usize,
}

/// Statistical early rejection of raw minimal candidates.
///
/// Unlike a [`Scorer`], a verifier may reject a good model, so it never sees
/// derived candidates.
pub(crate) trait Verifier<P: Problem, R> {
    /// Whether `model` passes. A verifier that writes every residual to
    /// [`Workspace::residuals`] sets [`Workspace::residuals_complete`].
    fn verify(
        &mut self,
        problem: &P,
        model: &P::Model,
        cx: &VerifyContext,
        ws: &mut Workspace,
        rng: &mut R,
    ) -> bool;
}

/// Accepts every candidate.
pub(crate) struct NoVerifier;

impl<P: Problem, R> Verifier<P, R> for NoVerifier {
    #[inline]
    fn verify(
        &mut self,
        _problem: &P,
        _model: &P::Model,
        _cx: &VerifyContext,
        _ws: &mut Workspace,
        _rng: &mut R,
    ) -> bool {
        true
    }
}

/// Exact scoring: owns the threshold semantics and the inlier mask.
pub(crate) trait Scorer<P: Problem> {
    /// The ranking objective.
    type Objective: Objective;

    /// Score a raw minimal candidate into [`Workspace::mask`].
    ///
    /// `bound` is the incumbent's score once a model exists. The scorer may
    /// return `None` only when the candidate provably cannot beat it under
    /// [`Ties::Strict`]; the mask is then unspecified. The default scores
    /// exhaustively.
    #[inline]
    fn score_raw(
        &mut self,
        problem: &P,
        model: &P::Model,
        _bound: Option<&Score<Self::Objective>>,
        ws: &mut Workspace,
    ) -> Option<Score<Self::Objective>> {
        Some(self.score_full(problem, model, ws))
    }

    /// Score a candidate exhaustively into [`Workspace::mask`].
    fn score_full(
        &mut self,
        problem: &P,
        model: &P::Model,
        ws: &mut Workspace,
    ) -> Score<Self::Objective>;
}

/// When the sampling loop stops.
pub(crate) trait Termination {
    /// Whether to stop before making draw `draws + 1`.
    fn done(&self, draws: usize) -> bool;

    /// A candidate with `support` inliers became the incumbent during draw
    /// `draws`.
    fn on_new_best(&mut self, support: usize, draws: usize);

    /// Draw `draws` finished; `improved` if it replaced the incumbent.
    #[inline]
    fn after_draw(&mut self, _draws: usize, _improved: bool, _best_support: usize) {}
}

/// When local optimization, degeneracy recovery and final fits run.
pub(crate) trait Schedule<P: Problem, Sc: Scorer<P>, T, R> {
    /// A raw minimal candidate became the incumbent.
    #[inline]
    fn on_raw_accept(&mut self) {}

    /// Called after every draw whose sample passed [`Problem::check_sample`].
    #[inline]
    fn after_draw(&mut self, _stage: &mut StageContext<'_, P, Sc, T, R>) {}

    /// Called once when sampling stops.
    #[inline]
    fn after_sampling(&mut self, _stage: &mut StageContext<'_, P, Sc, T, R>) {}

    /// Called last: non-minimal and non-linear refits.
    #[inline]
    fn final_fit(&mut self, _stage: &mut StageContext<'_, P, Sc, T, R>) {}
}

/// Plain RANSAC: no local optimization and no final fit.
pub(crate) struct NoSchedule;

impl<P: Problem, Sc: Scorer<P>, T, R> Schedule<P, Sc, T, R> for NoSchedule {}

/// Scratch buffers reused for every candidate.
pub(crate) struct Workspace {
    /// The candidate's inlier mask, swapped with the incumbent's on
    /// replacement.
    pub mask: Vec<bool>,
    /// Per-observation residuals, for scorers and verifiers that need them.
    pub residuals: Vec<f64>,
    /// Whether the verifier wrote every entry of [`Self::residuals`] for the
    /// current candidate.
    pub residuals_complete: bool,
}

impl Workspace {
    /// Wrap a candidate mask buffer and a residual buffer.
    pub(crate) fn new(mask: Vec<bool>, residuals: Vec<f64>) -> Self {
        Self {
            mask,
            residuals,
            residuals_complete: false,
        }
    }
}

/// The components of one run.
pub(crate) struct Components<S, V, Sc, Sch, T> {
    /// Draws minimal samples.
    pub sampler: S,
    /// Rejects raw candidates early.
    pub verifier: V,
    /// Scores candidates exactly.
    pub scorer: Sc,
    /// Places local optimization and final fits.
    pub schedule: Sch,
    /// Stops the loop.
    pub termination: T,
}

/// What a [`Schedule`] hook may touch.
///
/// The problem is read-only here; a schedule keeps its own scratch.
pub(crate) struct StageContext<'a, P: Problem, Sc: Scorer<P>, T, R> {
    /// The problem.
    pub problem: &'a P,
    /// The scorer.
    pub scorer: &'a mut Sc,
    /// The termination policy.
    pub termination: &'a mut T,
    /// The incumbent.
    pub best: &'a mut Incumbent<P::Model, Sc::Objective>,
    /// Shared scratch.
    pub ws: &'a mut Workspace,
    /// The run's RNG.
    pub rng: &'a mut R,
    /// Draws made so far.
    pub draws: usize,
}

impl<P: Problem, Sc: Scorer<P>, T: Termination, R> StageContext<'_, P, Sc, T, R> {
    /// Score a derived candidate exhaustively and offer it to the incumbent.
    ///
    /// Returns whether it replaced the incumbent.
    pub(crate) fn consider_derived(&mut self, model: &P::Model, ties: Ties) -> bool {
        let score = self.scorer.score_full(self.problem, model, self.ws);
        let accepted = self.best.consider(model, score, &mut self.ws.mask, ties);
        if accepted {
            self.termination.on_new_best(score.support, self.draws);
        }
        accepted
    }
}

/// Run the staged loop and return the number of draws made.
///
/// `best` is the initial (normally empty) incumbent and receives the result;
/// `ws` holds the scratch buffers the components expect.
pub(crate) fn run<P, S, V, Sc, Sch, T, R>(
    problem: &mut P,
    parts: &mut Components<S, V, Sc, Sch, T>,
    best: &mut Incumbent<P::Model, Sc::Objective>,
    ws: &mut Workspace,
    rng: &mut R,
) -> usize
where
    P: Problem,
    S: MinimalSampler<R>,
    V: Verifier<P, R>,
    Sc: Scorer<P>,
    Sch: Schedule<P, Sc, T, R>,
    T: Termination,
{
    let population = problem.num_observations();
    let mut sample = vec![0usize; problem.sample_size()];
    let mut draws = 0usize;

    while !parts.termination.done(draws) {
        parts.sampler.sample(rng, population, &mut sample);
        draws += 1;
        if !problem.check_sample(&sample) {
            continue;
        }
        problem.fit_minimal(&sample);

        let mut improved = false;
        for model in problem.candidates() {
            if !problem.check_model(&sample, model) {
                continue;
            }
            ws.residuals_complete = false;
            let cx = VerifyContext {
                draws,
                best_support: best.score.support,
            };
            if !parts.verifier.verify(problem, model, &cx, ws, rng) {
                continue;
            }
            let Some(score) = parts.scorer.score_raw(problem, model, best.bound(), ws) else {
                continue;
            };
            if best.consider(model, score, &mut ws.mask, Ties::Strict) {
                improved = true;
                parts.termination.on_new_best(score.support, draws);
                parts.schedule.on_raw_accept();
            }
        }

        parts
            .termination
            .after_draw(draws, improved, best.score.support);
        parts.schedule.after_draw(&mut StageContext {
            problem,
            scorer: &mut parts.scorer,
            termination: &mut parts.termination,
            best,
            ws,
            rng,
            draws,
        });
    }

    let mut stage = StageContext {
        problem,
        scorer: &mut parts.scorer,
        termination: &mut parts.termination,
        best,
        ws,
        rng,
        draws,
    };
    parts.schedule.after_sampling(&mut stage);
    parts.schedule.final_fit(&mut stage);
    draws
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};

    /// One-dimensional "line" problem: observations are values, a model is a
    /// value, and a minimal sample of one observation proposes it.
    struct Values {
        values: Vec<f64>,
        reject_samples: bool,
        fit_nothing: bool,
        fits: usize,
        candidate: Option<f64>,
    }

    impl Values {
        fn new(values: Vec<f64>) -> Self {
            Self {
                values,
                reject_samples: false,
                fit_nothing: false,
                fits: 0,
                candidate: None,
            }
        }
    }

    impl Problem for Values {
        type Model = f64;

        fn num_observations(&self) -> usize {
            self.values.len()
        }

        fn sample_size(&self) -> usize {
            1
        }

        fn check_sample(&self, _sample: &[usize]) -> bool {
            !self.reject_samples
        }

        fn fit_minimal(&mut self, sample: &[usize]) {
            self.fits += 1;
            self.candidate = (!self.fit_nothing).then(|| self.values[sample[0]]);
        }

        fn candidates(&self) -> &[f64] {
            self.candidate.as_slice()
        }
    }

    /// Strict `|x - m| < 0.5` inlier count.
    struct Count;

    impl Scorer<Values> for Count {
        type Objective = InlierCount;

        fn score_full(
            &mut self,
            problem: &Values,
            model: &f64,
            ws: &mut Workspace,
        ) -> Score<InlierCount> {
            let mut count = 0;
            for (keep, &x) in ws.mask.iter_mut().zip(&problem.values) {
                *keep = (x - model).abs() < 0.5;
                count += *keep as usize;
            }
            Score {
                objective: InlierCount(count),
                support: count,
            }
        }
    }

    /// Fixed draw budget that records what the engine told it.
    #[derive(Default)]
    struct Budget {
        max: usize,
        new_bests: Vec<(usize, usize)>,
        draws_seen: Vec<(usize, bool)>,
    }

    impl Termination for Budget {
        fn done(&self, draws: usize) -> bool {
            draws >= self.max
        }

        fn on_new_best(&mut self, support: usize, draws: usize) {
            self.new_bests.push((support, draws));
        }

        fn after_draw(&mut self, draws: usize, improved: bool, _best_support: usize) {
            self.draws_seen.push((draws, improved));
        }
    }

    /// Records hook order and offers one derived candidate per stage.
    #[derive(Default)]
    struct Recorder {
        log: Vec<&'static str>,
        derived: Option<f64>,
        accepted: Vec<bool>,
    }

    impl<R> Schedule<Values, Count, Budget, R> for Recorder {
        fn on_raw_accept(&mut self) {
            self.log.push("accept");
        }

        fn after_draw(&mut self, _stage: &mut StageContext<'_, Values, Count, Budget, R>) {
            self.log.push("draw");
        }

        fn after_sampling(&mut self, stage: &mut StageContext<'_, Values, Count, Budget, R>) {
            self.log.push("after_sampling");
            if let Some(model) = self.derived {
                self.accepted
                    .push(stage.consider_derived(&model, Ties::Strict));
            }
        }

        fn final_fit(&mut self, stage: &mut StageContext<'_, Values, Count, Budget, R>) {
            self.log.push("final_fit");
            if let Some(model) = self.derived {
                self.accepted
                    .push(stage.consider_derived(&model, Ties::AcceptTies));
            }
        }
    }

    struct RejectAll;

    impl<P: Problem, R> Verifier<P, R> for RejectAll {
        fn verify(
            &mut self,
            _: &P,
            _: &P::Model,
            _: &VerifyContext,
            _: &mut Workspace,
            _: &mut R,
        ) -> bool {
            false
        }
    }

    fn run_values<V: Verifier<Values, StdRng>>(
        problem: &mut Values,
        verifier: V,
        max: usize,
        derived: Option<f64>,
    ) -> (usize, Incumbent<f64, InlierCount>, Budget, Recorder) {
        let n = problem.num_observations();
        let mut parts = Components {
            sampler: UniformDraw,
            verifier,
            scorer: Count,
            schedule: Recorder {
                derived,
                ..Default::default()
            },
            termination: Budget {
                max,
                ..Default::default()
            },
        };
        let mut best = Incumbent::new(vec![false; n]);
        let mut ws = Workspace::new(vec![false; n], Vec::new());
        let draws = run(
            problem,
            &mut parts,
            &mut best,
            &mut ws,
            &mut StdRng::seed_from_u64(3),
        );
        (draws, best, parts.termination, parts.schedule)
    }

    #[test]
    fn rejected_samples_count_toward_the_hard_cap() {
        let mut problem = Values::new(vec![0.0, 0.1, 5.0]);
        problem.reject_samples = true;
        let (draws, best, budget, schedule) = run_values(&mut problem, NoVerifier, 37, None);
        assert_eq!(draws, 37);
        assert_eq!(problem.fits, 0);
        assert!(best.model.is_none());
        // Rejected samples skip the per-draw hooks.
        assert!(budget.draws_seen.is_empty());
        assert_eq!(schedule.log, ["after_sampling", "final_fit"]);
    }

    #[test]
    fn rootless_draws_count_toward_the_hard_cap() {
        let mut problem = Values::new(vec![0.0, 0.1, 5.0]);
        problem.fit_nothing = true;
        let (draws, best, budget, _) = run_values(&mut problem, NoVerifier, 23, None);
        assert_eq!(draws, 23);
        assert_eq!(problem.fits, 23);
        assert!(best.model.is_none());
        assert_eq!(budget.draws_seen.len(), 23);
        assert!(budget.draws_seen.iter().all(|&(_, improved)| !improved));
    }

    #[test]
    fn strict_raw_acceptance_and_hook_order() {
        let mut problem = Values::new(vec![0.0, 0.1, 0.2, 5.0]);
        let (draws, best, budget, schedule) = run_values(&mut problem, NoVerifier, 12, None);
        assert_eq!(draws, 12);
        // Any of the first three values explains all three: support 3.
        assert_eq!(best.score.support, 3);
        assert_eq!(best.inliers, [true, true, true, false]);
        // Only strict improvements are new bests, so supports increase.
        assert!(budget.new_bests.windows(2).all(|w| w[0].0 < w[1].0));
        assert_eq!(schedule.log.iter().filter(|&&e| e == "draw").count(), 12);
        assert_eq!(
            schedule.log.iter().filter(|&&e| e == "accept").count(),
            budget.new_bests.len()
        );
        assert_eq!(
            schedule.log[schedule.log.len() - 2..],
            ["after_sampling", "final_fit"]
        );
    }

    #[test]
    fn verifier_rejections_are_never_scored() {
        let mut problem = Values::new(vec![0.0, 0.1, 0.2]);
        let (draws, best, budget, _) = run_values(&mut problem, RejectAll, 9, None);
        assert_eq!(draws, 9);
        assert!(best.model.is_none());
        assert!(budget.new_bests.is_empty());
    }

    #[test]
    fn derived_candidates_bypass_the_verifier_and_follow_their_tie_rule() {
        // Every raw candidate is rejected; the derived one at 0.1 explains
        // the first three values. `after_sampling` offers it under Strict
        // (it beats the empty incumbent), `final_fit` again under AcceptTies
        // (it ties itself, so it is accepted again).
        let mut problem = Values::new(vec![0.0, 0.1, 0.2, 5.0]);
        let (_, best, budget, schedule) = run_values(&mut problem, RejectAll, 5, Some(0.1));
        assert_eq!(schedule.accepted, [true, true]);
        assert_eq!(best.model, Some(0.1));
        assert_eq!(best.score.support, 3);
        assert_eq!(budget.new_bests, [(3, 5), (3, 5)]);
    }
}
