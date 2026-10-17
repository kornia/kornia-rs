//! Robust two-view model estimation: the RANSAC entry points for the
//! fundamental, essential and homography models.
//!
//! Each entry point is a preset of the crate's staged RANSAC engine: a
//! [`Problem`] holding the prepared correspondences, an exact [`Scorer`] in
//! pixel units, the two-view adaptive cap and a [`Schedule`] for the optional
//! refit. Candidates are ranked by inlier count, then by the lower sum of
//! inlier residuals, with an inclusive `residual <= threshold²` test.

use super::kernels::{
    score_inliers_f, score_inliers_f_bounded_masked, score_inliers_f_upper_bound, score_inliers_h,
    split_xy,
};
use super::local_optimization::{degensac_recover_fundamental, lo_plus_fundamental};
use super::{RansacParams, RansacResult, TwoViewError};
use crate::pose::fundamental::fundamental_8point;
use crate::pose::fundamental_7pt::fundamental_7point_oriented_into;
use crate::pose::{essential_5pt, homography_4pt2d, homography_dlt};
use crate::ransac::engine::{
    self, CapBase, Components, CountThenResidual, Incumbent, NoSchedule, NoVerifier, Objective,
    Problem, Schedule, Score, Scorer, StageContext, Termination, Ties, TwoViewCap, UniformDraw,
    Workspace,
};
use kornia_algebra::{Mat3F64, Vec2F64, Vec3F64};
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Draws without a new best after which homography RANSAC stops.
const HOMOGRAPHY_STAGNATION_LIMIT: usize = 200;

#[inline]
pub(super) fn ransac_confidence(params: &RansacParams, default: f64) -> Result<f64, TwoViewError> {
    match params.confidence {
        Some(confidence) if confidence.is_finite() && confidence > 0.0 && confidence < 1.0 => {
            Ok(confidence)
        }
        Some(confidence) => Err(TwoViewError::InvalidConfidence { confidence }),
        None => Ok(default),
    }
}

/// The run's RNG: seeded when requested, otherwise from the thread RNG.
fn ransac_rng(params: &RansacParams) -> StdRng {
    match params.random_seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => {
            let mut thread_rng = rand::rng();
            StdRng::from_rng(&mut thread_rng)
        }
    }
}

/// Correspondences in both layouts: AoS for the solvers, SoA for the SIMD
/// scoring kernels.
pub(super) struct PointPairs<'a> {
    pub(super) x1: &'a [Vec2F64],
    pub(super) x2: &'a [Vec2F64],
    x1_x: Vec<f64>,
    x1_y: Vec<f64>,
    x2_x: Vec<f64>,
    x2_y: Vec<f64>,
}

impl<'a> PointPairs<'a> {
    /// Flatten `x1` and `x2`, which must have equal lengths, once per run.
    fn new(x1: &'a [Vec2F64], x2: &'a [Vec2F64]) -> Self {
        debug_assert_eq!(x1.len(), x2.len());
        let (x1_x, x1_y) = split_xy(x1);
        let (x2_x, x2_y) = split_xy(x2);
        Self {
            x1,
            x2,
            x1_x,
            x1_y,
            x2_x,
            x2_y,
        }
    }

    /// Number of correspondences.
    #[inline]
    pub(super) fn len(&self) -> usize {
        self.x1.len()
    }

    /// Sampson inliers of `f` at `thresh_sq`: sets their entries of the
    /// all-false `mask` and returns their count and residual sum.
    #[inline]
    pub(super) fn sampson(&self, f: &Mat3F64, thresh_sq: f64, mask: &mut [bool]) -> (usize, f64) {
        score_inliers_f(
            f, &self.x1_x, &self.x1_y, &self.x2_x, &self.x2_y, thresh_sq, mask,
        )
    }

    /// Transfer-error inliers of `h`, like [`Self::sampson`].
    #[inline]
    pub(super) fn transfer(&self, h: &Mat3F64, thresh_sq: f64, mask: &mut [bool]) -> (usize, f64) {
        score_inliers_h(
            h, &self.x1_x, &self.x1_y, &self.x2_x, &self.x2_y, thresh_sq, mask,
        )
    }
}

/// Run a two-view preset and apply the minimum-support gate.
fn run_two_view<P, Sc, Sch>(
    mut problem: P,
    scorer: Sc,
    schedule: Sch,
    termination: TwoViewCap,
    min_support: usize,
    rng: &mut StdRng,
) -> Result<RansacResult<Mat3F64>, TwoViewError>
where
    P: Problem<Model = Mat3F64>,
    Sc: Scorer<P, Objective = CountThenResidual>,
    Sch: Schedule<P, Sc, TwoViewCap, StdRng>,
{
    let n = problem.num_observations();
    let mut parts = Components {
        sampler: UniformDraw,
        verifier: NoVerifier,
        scorer,
        schedule,
        termination,
    };
    let mut best = Incumbent::new(vec![false; n]);
    let mut ws = Workspace::new(vec![false; n], Vec::new());
    engine::run(&mut problem, &mut parts, &mut best, &mut ws, rng);
    match best.model {
        Some(model) if best.score.support >= min_support => Ok(RansacResult {
            model,
            inliers: best.inliers,
            inlier_count: best.score.support,
            score: best.score.objective.residual,
        }),
        _ => Err(TwoViewError::RansacFailure),
    }
}

/// Whether a post-sampling refit may run: the schedule only refines a
/// winner that already passes the minimum-support gate.
#[inline]
fn refit_allowed<O>(refit: bool, best: &Incumbent<Mat3F64, O>, min_support: usize) -> bool {
    refit && best.model.is_some() && best.score.support >= min_support
}

/// A two-view score from a kernel's `(count, residual sum)`.
#[inline]
fn count_then_residual((count, residual): (usize, f64)) -> Score<CountThenResidual> {
    Score {
        objective: CountThenResidual { count, residual },
        support: count,
    }
}

/// Estimate a fundamental matrix with seven-point RANSAC hypotheses.
///
/// Every real solution of a minimal sample that satisfies the oriented
/// epipolar constraint on that sample is scored, as in DEGENSAC; the others
/// cannot come from points in front of both cameras. Optional local
/// refinement uses the eight-point solver on sets of at least eight inliers.
///
/// # Arguments
///
/// * `x1` - Pixel coordinates in the first image (at least seven points).
/// * `x2` - Corresponding pixel coordinates in the second image.
/// * `params` - Sampling budget, pixel threshold, minimum support, seed and refit flag.
///
/// # Returns
///
/// The best fundamental matrix, its inlier mask, count and Sampson score.
///
/// # Behaviour
///
/// - **Inliers:** Sampson distance `d ≤ threshold²` in squared pixels
///   (inclusive).
/// - **Ranking:** more inliers wins, then the lower sum of inlier distances.
/// - **Stopping:** after `max_iterations` draws, or once the adaptive bound
///   for sampling without replacement reaches `confidence` (default 0.9999).
///   A draw whose sample yields no consistent root still counts. Full support
///   stops after the current draw.
/// - **Minimum support:** the winner must have `min_inliers` inliers before
///   any refit, otherwise the call fails.
/// - **Refit:** with `refit`, LO+ threshold annealing and then DEGENSAC; each
///   only replaces the winner with a strictly better model, and the returned
///   mask, count and score are those of the returned model.
///
/// # Errors
///
/// Returns [`TwoViewError::InvalidInput`] for unequal lengths or fewer than
/// seven matches, [`TwoViewError::InvalidConfidence`] if `params.confidence` is
/// not finite and strictly between zero and one, or
/// [`TwoViewError::RansacFailure`] if no model meets the requested minimum
/// support.
///
/// # Example
///
/// ```
/// use kornia_3d::pose::{ransac_fundamental, RansacParams};
/// use kornia_algebra::Vec2F64;
/// let x1 = vec![Vec2F64::ZERO; 6];
/// assert!(ransac_fundamental(&x1, &x1, &RansacParams::default()).is_err());
/// ```
pub fn ransac_fundamental(
    x1: &[Vec2F64],
    x2: &[Vec2F64],
    params: &RansacParams,
) -> Result<RansacResult<Mat3F64>, TwoViewError> {
    ransac_fundamental_impl::<7>(x1, x2, params)
}

/// Estimate a fundamental matrix with the original eight-point RANSAC solver.
///
/// # Arguments
///
/// * `x1` - Pixel coordinates in the first image (at least eight points).
/// * `x2` - Corresponding pixel coordinates in the second image.
/// * `params` - Sampling budget, pixel threshold, minimum support, seed and refit flag.
///
/// # Returns
///
/// The best fundamental matrix, its inlier mask, count and Sampson score.
///
/// # Behaviour
///
/// - **Inliers:** Sampson distance `d ≤ threshold²` in squared pixels
///   (inclusive).
/// - **Ranking:** more inliers wins, then the lower sum of inlier distances.
/// - **Stopping:** after `max_iterations` draws, or once the adaptive bound
///   for sampling without replacement reaches `confidence` (default 0.9999).
///   A draw whose sample is degenerate still counts. Full support
///   stops after the current draw.
/// - **Minimum support:** the winner must have `min_inliers` inliers before
///   any refit, otherwise the call fails.
/// - **Refit:** with `refit`, LO+ threshold annealing and then DEGENSAC; each
///   only replaces the winner with a strictly better model, and the returned
///   mask, count and score are those of the returned model.
///
/// # Errors
///
/// Returns [`TwoViewError::InvalidInput`] for unequal lengths or fewer than
/// eight matches, [`TwoViewError::InvalidConfidence`] if `params.confidence` is
/// not finite and strictly between zero and one, or
/// [`TwoViewError::RansacFailure`] if no model meets the requested minimum
/// support.
///
/// # Example
///
/// ```
/// use kornia_3d::pose::{ransac_fundamental_8point, RansacParams};
/// use kornia_algebra::Vec2F64;
/// let x1 = vec![Vec2F64::ZERO; 7];
/// assert!(ransac_fundamental_8point(&x1, &x1, &RansacParams::default()).is_err());
/// ```
pub fn ransac_fundamental_8point(
    x1: &[Vec2F64],
    x2: &[Vec2F64],
    params: &RansacParams,
) -> Result<RansacResult<Mat3F64>, TwoViewError> {
    ransac_fundamental_impl::<8>(x1, x2, params)
}

fn ransac_fundamental_impl<const SAMPLE_SIZE: usize>(
    x1: &[Vec2F64],
    x2: &[Vec2F64],
    params: &RansacParams,
) -> Result<RansacResult<Mat3F64>, TwoViewError> {
    let confidence = ransac_confidence(params, 0.9999)?;
    if x1.len() != x2.len() || x1.len() < SAMPLE_SIZE {
        return Err(TwoViewError::InvalidInput {
            required: SAMPLE_SIZE,
        });
    }
    let mut rng = ransac_rng(params);
    let problem = FundamentalProblem::<SAMPLE_SIZE> {
        points: PointPairs::new(x1, x2),
        threshold: params.threshold,
        thresh_sq: params.threshold * params.threshold,
        roots: [Mat3F64::ZERO; 3],
        num_roots: 0,
    };
    // Confidence applies to minimal samples, rather than individual roots.
    let termination = TwoViewCap::new(
        params.max_iterations,
        CapBase::Current,
        x1.len(),
        SAMPLE_SIZE,
        confidence,
    );
    let schedule = FundamentalRefinement {
        refit: params.refit,
        min_support: params.min_inliers,
    };
    run_two_view(
        problem,
        SampsonScorer,
        schedule,
        termination,
        params.min_inliers,
        &mut rng,
    )
}

/// Seven- or eight-point fundamental matrix estimation in pixel space.
pub(super) struct FundamentalProblem<'a, const SAMPLE_SIZE: usize> {
    pub(super) points: PointPairs<'a>,
    /// Inlier threshold in pixels; LO+ anneals multiples of it.
    pub(super) threshold: f64,
    pub(super) thresh_sq: f64,
    /// Roots of the last minimal sample: up to three for seven points.
    roots: [Mat3F64; 3],
    num_roots: usize,
}

impl<const SAMPLE_SIZE: usize> Problem for FundamentalProblem<'_, SAMPLE_SIZE> {
    type Model = Mat3F64;

    #[inline]
    fn num_observations(&self) -> usize {
        self.points.len()
    }

    #[inline]
    fn sample_size(&self) -> usize {
        SAMPLE_SIZE
    }

    #[inline]
    fn fit_minimal(&mut self, sample: &[usize]) {
        let mut s1 = [Vec2F64::ZERO; SAMPLE_SIZE];
        let mut s2 = [Vec2F64::ZERO; SAMPLE_SIZE];
        for (i, &idx) in sample.iter().enumerate() {
            s1[i] = self.points.x1[idx];
            s2[i] = self.points.x2[idx];
        }
        self.num_roots = if SAMPLE_SIZE == 7 {
            fundamental_7point_oriented_into(&s1, &s2, &mut self.roots).unwrap_or(0)
        } else {
            match fundamental_8point(&s1, &s2) {
                Ok(f) => {
                    self.roots[0] = f;
                    1
                }
                Err(_) => 0,
            }
        };
    }

    #[inline]
    fn candidates(&self) -> &[Mat3F64] {
        &self.roots[..self.num_roots]
    }
}

/// Inclusive Sampson scoring. Seven-point roots are scored with pruning
/// against the incumbent; the eight-point path keeps its full scorer.
struct SampsonScorer;

impl<const SAMPLE_SIZE: usize> Scorer<FundamentalProblem<'_, SAMPLE_SIZE>> for SampsonScorer {
    type Objective = CountThenResidual;

    #[inline]
    fn score_raw(
        &mut self,
        problem: &FundamentalProblem<'_, SAMPLE_SIZE>,
        f: &Mat3F64,
        bound: Option<&Score<CountThenResidual>>,
        ws: &mut Workspace,
    ) -> Option<Score<CountThenResidual>> {
        if SAMPLE_SIZE != 7 {
            return Some(self.score_full(problem, f, ws));
        }
        let best = bound.map_or(CountThenResidual::worst(), |b| b.objective);
        let points = &problem.points;
        // A seven-point sample yields up to three roots, and most of them
        // cannot win. A division-free pass bounds each root's support from
        // above; roots that cannot even tie the incumbent are dropped before
        // any exact scoring or mask traffic.
        if best.count > 0
            && score_inliers_f_upper_bound(
                f,
                &points.x1_x,
                &points.x1_y,
                &points.x2_x,
                &points.x2_y,
                problem.thresh_sq,
                best.count - 1,
            )
            .is_none()
        {
            return None;
        }
        // Survivors usually win or tie, so score them exactly in fixed,
        // SIMD-aligned chunks while materializing the mask, stopping only
        // when the exact bound cannot beat the incumbent.
        ws.mask.fill(false);
        score_inliers_f_bounded_masked(
            f,
            &points.x1_x,
            &points.x1_y,
            &points.x2_x,
            &points.x2_y,
            problem.thresh_sq,
            &mut ws.mask,
            best.count,
            best.residual,
        )
        .map(count_then_residual)
    }

    #[inline]
    fn score_full(
        &mut self,
        problem: &FundamentalProblem<'_, SAMPLE_SIZE>,
        f: &Mat3F64,
        ws: &mut Workspace,
    ) -> Score<CountThenResidual> {
        ws.mask.fill(false);
        count_then_residual(problem.points.sampson(f, problem.thresh_sq, &mut ws.mask))
    }
}

/// The fundamental-matrix refit: LO+ threshold annealing, then DEGENSAC,
/// once after sampling.
struct FundamentalRefinement {
    refit: bool,
    min_support: usize,
}

impl<'a, const SAMPLE_SIZE: usize, Sc, T>
    Schedule<FundamentalProblem<'a, SAMPLE_SIZE>, Sc, T, StdRng> for FundamentalRefinement
where
    Sc: Scorer<FundamentalProblem<'a, SAMPLE_SIZE>, Objective = CountThenResidual>,
    T: Termination,
{
    fn after_sampling(
        &mut self,
        stage: &mut StageContext<'_, FundamentalProblem<'a, SAMPLE_SIZE>, Sc, T, StdRng>,
    ) {
        if !refit_allowed(self.refit, stage.best, self.min_support) {
            return;
        }
        lo_plus_fundamental(stage);
        // DEGENSAC safety net: if the LO+ winner sits on a dominant-plane
        // local optimum (Chum'04), recover the true F via [e']× H. No-op when
        // the scene is non-planar; the strict-improve gate guarantees we
        // never regress from the polished F.
        degensac_recover_fundamental(stage);
    }
}

/// Estimate an **essential matrix** with RANSAC using the Nistér 5-point
/// solver. Returns the result in **metric (camera) space** — i.e. the model
/// satisfies `x̂2ᵀ E x̂1 = 0` for normalized correspondences `x̂ = K⁻¹ x_h`.
///
/// Why prefer this over `ransac_fundamental` + `essential_from_fundamental`
/// when intrinsics are known:
/// - **Smaller sample size (5 vs 8)** → fewer iterations needed for the same
///   confidence. At 30% inlier rate, 5pt needs ~568 iters vs 8pt's ~4600 for
///   99% confidence (8× fewer draws).
/// - **No (σ, σ, 0) clipping**: 8pt → F → E projects onto the essential
///   manifold *after* decomposition, losing structure. 5pt stays on the
///   manifold throughout, giving 2-10× lower rotation error on small-motion
///   / narrow-baseline pairs (the SLAM bootstrap regime).
///
/// Each minimal sample produces up to 10 candidate Es; we score every
/// candidate via Sampson distance in **pixel** space (after mapping
/// `F = K2⁻ᵀ E K1⁻¹`) so the threshold semantics match `ransac_fundamental`.
///
/// `k1` / `k2` must be invertible upper-triangular intrinsics matrices.
///
/// # Arguments
///
/// * `x1` - Pixel coordinates in the first image (at least five points).
/// * `x2` - Corresponding pixel coordinates in the second image.
/// * `k1` - Intrinsics of the first camera.
/// * `k2` - Intrinsics of the second camera.
/// * `params` - Sampling budget, pixel threshold, minimum support, seed and
///   confidence. `refit` is ignored.
///
/// # Returns
///
/// The best essential matrix, its inlier mask, count and pixel Sampson score.
///
/// # Behaviour
///
/// - **Inliers:** Sampson distance of `F = K2⁻ᵀ E K1⁻¹` in squared pixels,
///   `d ≤ threshold²` (inclusive).
/// - **Ranking:** more inliers wins, then the lower sum of inlier distances.
/// - **Stopping:** after `max_iterations` draws, or once the adaptive bound
///   for sampling without replacement reaches `confidence` (default 0.9999).
///   Full support stops after the current draw.
/// - **Minimum support:** the winner must have `min_inliers` inliers.
/// - **Refit:** none.
///
/// # Errors
///
/// Returns [`TwoViewError::InvalidConfidence`] if `params.confidence` is not
/// finite and strictly between zero and one, [`TwoViewError::InvalidInput`]
/// for unequal lengths, fewer than five matches or a point mapped to infinity
/// by an inverse intrinsics matrix, or [`TwoViewError::RansacFailure`] if no
/// model meets the requested minimum support.
pub fn ransac_essential_5pt(
    x1: &[Vec2F64],
    x2: &[Vec2F64],
    k1: &Mat3F64,
    k2: &Mat3F64,
    params: &RansacParams,
) -> Result<RansacResult<Mat3F64>, TwoViewError> {
    let confidence = ransac_confidence(params, 0.9999)?;
    if x1.len() != x2.len() || x1.len() < 5 {
        return Err(TwoViewError::InvalidInput { required: 5 });
    }
    let mut rng = ransac_rng(params);
    let problem = EssentialProblem::new(x1, x2, k1, k2, params.threshold)?;
    // The default confidence is 0.9999, matching fundamental estimation;
    // callers can override it through `RansacParams::confidence`.
    let termination = TwoViewCap::new(
        params.max_iterations,
        CapBase::Configured,
        x1.len(),
        5,
        confidence,
    );
    run_two_view(
        problem,
        EssentialSampsonScorer,
        NoSchedule,
        termination,
        params.min_inliers,
        &mut rng,
    )
}

/// Five-point essential matrix estimation, scored in pixel space.
struct EssentialProblem<'a> {
    points: PointPairs<'a>,
    /// Calibrated coordinates `K⁻¹ x` for the five-point solver.
    x1n: Vec<Vec2F64>,
    x2n: Vec<Vec2F64>,
    k1_inv: Mat3F64,
    k2_inv_t: Mat3F64,
    thresh_sq: f64,
    /// Up to ten candidate Es of the last minimal sample.
    candidates: Vec<Mat3F64>,
}

impl<'a> EssentialProblem<'a> {
    fn new(
        x1: &'a [Vec2F64],
        x2: &'a [Vec2F64],
        k1: &Mat3F64,
        k2: &Mat3F64,
        threshold: f64,
    ) -> Result<Self, TwoViewError> {
        // The solver lives in calibrated space while scoring stays in pixel
        // space: every candidate E is lifted via F = K2⁻ᵀ E K1⁻¹.
        let k1_inv = k1.inverse();
        let k2_inv = k2.inverse();
        let mut x1n = Vec::with_capacity(x1.len());
        let mut x2n = Vec::with_capacity(x1.len());
        for (p1, p2) in x1.iter().zip(x2) {
            let h1 = k1_inv * Vec3F64::new(p1.x, p1.y, 1.0);
            let h2 = k2_inv * Vec3F64::new(p2.x, p2.y, 1.0);
            if h1.z.abs() < 1e-12 || h2.z.abs() < 1e-12 {
                return Err(TwoViewError::InvalidInput { required: 5 });
            }
            x1n.push(Vec2F64::new(h1.x / h1.z, h1.y / h1.z));
            x2n.push(Vec2F64::new(h2.x / h2.z, h2.y / h2.z));
        }
        Ok(Self {
            points: PointPairs::new(x1, x2),
            x1n,
            x2n,
            k1_inv,
            k2_inv_t: k2_inv.transpose(),
            thresh_sq: threshold * threshold,
            candidates: Vec::new(),
        })
    }
}

impl Problem for EssentialProblem<'_> {
    type Model = Mat3F64;

    #[inline]
    fn num_observations(&self) -> usize {
        self.points.len()
    }

    #[inline]
    fn sample_size(&self) -> usize {
        5
    }

    #[inline]
    fn fit_minimal(&mut self, sample: &[usize]) {
        let mut s1 = [Vec2F64::ZERO; 5];
        let mut s2 = [Vec2F64::ZERO; 5];
        for (i, &idx) in sample.iter().enumerate() {
            s1[i] = self.x1n[idx];
            s2[i] = self.x2n[idx];
        }
        self.candidates = essential_5pt(&s1, &s2);
    }

    #[inline]
    fn candidates(&self) -> &[Mat3F64] {
        &self.candidates
    }
}

/// Sampson scoring of `F = K2⁻ᵀ E K1⁻¹`, the same scorer as for F.
struct EssentialSampsonScorer;

impl Scorer<EssentialProblem<'_>> for EssentialSampsonScorer {
    type Objective = CountThenResidual;

    #[inline]
    fn score_full(
        &mut self,
        problem: &EssentialProblem<'_>,
        e: &Mat3F64,
        ws: &mut Workspace,
    ) -> Score<CountThenResidual> {
        let f = problem.k2_inv_t * *e * problem.k1_inv;
        ws.mask.fill(false);
        count_then_residual(problem.points.sampson(&f, problem.thresh_sq, &mut ws.mask))
    }
}

/// Estimate a homography with RANSAC using the 4-point solver.
///
/// # Arguments
///
/// * `x1` - Pixel coordinates in the first image (at least four points).
/// * `x2` - Corresponding pixel coordinates in the second image.
/// * `params` - Sampling budget, pixel threshold, minimum support, seed,
///   confidence and refit flag.
///
/// # Returns
///
/// The best homography, its inlier mask, count and transfer-error score.
///
/// # Behaviour
///
/// - **Inliers:** squared transfer error `‖H x1 − x2‖² ≤ threshold²` in
///   pixels (inclusive).
/// - **Ranking:** more inliers wins, then the lower sum of inlier errors.
/// - **Stopping:** after `max_iterations` draws, once the adaptive bound
///   reaches `confidence` (default 0.99: the homography mainly arbitrates
///   model selection in [`super::TwoViewEstimator`]), or after 200
///   consecutive draws without a new best. Samples with three collinear
///   points are rejected before fitting and still count as draws. Full
///   support stops after the current draw.
/// - **Minimum support:** the winner must have `min_inliers` inliers before
///   the refit, otherwise the call fails.
/// - **Refit:** with `refit`, a DLT over the inliers replaces the model when
///   it lowers the summed squared error over the *same* inliers. It is not
///   re-thresholded: `inliers` and `inlier_count` stay those of the
///   four-point winner, while `model` and `score` are the refit's.
///
/// # Errors
///
/// Returns [`TwoViewError::InvalidConfidence`] if `params.confidence` is not
/// finite and strictly between zero and one, [`TwoViewError::InvalidInput`]
/// for unequal lengths or fewer than four matches, or
/// [`TwoViewError::RansacFailure`] if no model meets the requested minimum
/// support.
pub fn ransac_homography(
    x1: &[Vec2F64],
    x2: &[Vec2F64],
    params: &RansacParams,
) -> Result<RansacResult<Mat3F64>, TwoViewError> {
    let confidence = ransac_confidence(params, 0.99)?;
    if x1.len() != x2.len() || x1.len() < 4 {
        return Err(TwoViewError::InvalidInput { required: 4 });
    }
    let mut rng = ransac_rng(params);
    let problem = HomographyProblem {
        points: PointPairs::new(x1, x2),
        thresh_sq: params.threshold * params.threshold,
        candidate: None,
    };
    // **Why p=0.99 instead of 0.999.** H here is a *model-selection oracle*,
    // not a final-product matrix — its only job is "is H_count meaningfully
    // larger than F_count?" If the scene is genuinely planar, H_count grows
    // fast and we converge in tens of iterations. If non-planar, H_count
    // never crosses the 0.8 × F_count gate no matter how long we run, so the
    // extra confidence buys us nothing but wall time. Lowering p to 0.99
    // shaves ~33% off the bound at low inlier ratios.
    //
    // **Stagnation early-exit.** If `HOMOGRAPHY_STAGNATION_LIMIT` draws pass
    // without improvement, the running best is almost certainly the local
    // optimum for this random seed and continued sampling is wasted work.
    // Belt-and-braces with the adaptive cap: adaptive shrinks fast at high
    // w; stagnation shrinks fast at low w (where adaptive can't tighten).
    let termination = TwoViewCap::new(
        params.max_iterations,
        CapBase::Configured,
        x1.len(),
        4,
        confidence,
    )
    .with_stagnation(HOMOGRAPHY_STAGNATION_LIMIT);
    let schedule = HomographyRefit {
        refit: params.refit,
        min_support: params.min_inliers,
    };
    run_two_view(
        problem,
        TransferScorer,
        schedule,
        termination,
        params.min_inliers,
        &mut rng,
    )
}

/// Four-point homography estimation in pixel space.
struct HomographyProblem<'a> {
    points: PointPairs<'a>,
    thresh_sq: f64,
    candidate: Option<Mat3F64>,
}

impl HomographyProblem<'_> {
    #[inline]
    fn minimal_sample(&self, sample: &[usize]) -> ([[f64; 2]; 4], [[f64; 2]; 4]) {
        let mut s1 = [[0.0; 2]; 4];
        let mut s2 = [[0.0; 2]; 4];
        for (i, &idx) in sample.iter().enumerate() {
            s1[i] = [self.points.x1[idx].x, self.points.x1[idx].y];
            s2[i] = [self.points.x2[idx].x, self.points.x2[idx].y];
        }
        (s1, s2)
    }
}

impl Problem for HomographyProblem<'_> {
    type Model = Mat3F64;

    #[inline]
    fn num_observations(&self) -> usize {
        self.points.len()
    }

    #[inline]
    fn sample_size(&self) -> usize {
        4
    }

    /// DEGENSAC-style collinearity check: a 4-point sample with 3+ collinear
    /// points produces a wildly-wrong H. Reject it before the solve.
    #[inline]
    fn check_sample(&self, sample: &[usize]) -> bool {
        let (s1, s2) = self.minimal_sample(sample);
        !(sample_is_degenerate(&s1) || sample_is_degenerate(&s2))
    }

    #[inline]
    fn fit_minimal(&mut self, sample: &[usize]) {
        let (s1, s2) = self.minimal_sample(sample);
        let mut h = [[0.0; 3]; 3];
        self.candidate = homography_4pt2d(&s1, &s2, &mut h).ok().map(|()| {
            Mat3F64::from_cols(
                Vec3F64::new(h[0][0], h[1][0], h[2][0]),
                Vec3F64::new(h[0][1], h[1][1], h[2][1]),
                Vec3F64::new(h[0][2], h[1][2], h[2][2]),
            )
        });
    }

    #[inline]
    fn candidates(&self) -> &[Mat3F64] {
        self.candidate.as_slice()
    }
}

/// Inclusive transfer-error scoring.
struct TransferScorer;

impl Scorer<HomographyProblem<'_>> for TransferScorer {
    type Objective = CountThenResidual;

    #[inline]
    fn score_full(
        &mut self,
        problem: &HomographyProblem<'_>,
        h: &Mat3F64,
        ws: &mut Workspace,
    ) -> Score<CountThenResidual> {
        ws.mask.fill(false);
        count_then_residual(problem.points.transfer(h, problem.thresh_sq, &mut ws.mask))
    }
}

/// The homography refit: a DLT over the final inliers.
struct HomographyRefit {
    refit: bool,
    min_support: usize,
}

impl<'a, Sc, T> Schedule<HomographyProblem<'a>, Sc, T, StdRng> for HomographyRefit
where
    Sc: Scorer<HomographyProblem<'a>, Objective = CountThenResidual>,
{
    /// The best 4-point sample passes many inliers, but those 4 points may
    /// not be a well-conditioned basis for H. A DLT across the full inlier
    /// set averages out that variance. The refit is kept only if the summed
    /// squared error over the same inliers improves, since the DLT may have
    /// overfit a borderline inlier.
    ///
    /// The refit is not rescored at the threshold: it keeps the pre-refit
    /// inlier mask and count (#1171 phase A preserves this stale mask).
    fn final_fit(&mut self, stage: &mut StageContext<'_, HomographyProblem<'a>, Sc, T, StdRng>) {
        let best = &mut *stage.best;
        if !refit_allowed(self.refit, best, self.min_support) || best.score.support < 4 {
            return;
        }
        let points = &stage.problem.points;
        let inlier_points = |pts: &[Vec2F64]| -> Vec<Vec2F64> {
            pts.iter()
                .zip(&best.inliers)
                .filter_map(|(p, &keep)| keep.then_some(*p))
                .collect()
        };
        let Ok(h_refit) = homography_dlt(&inlier_points(points.x1), &inlier_points(points.x2))
        else {
            return;
        };
        let mut refit_score = 0.0;
        for (i, &keep) in best.inliers.iter().enumerate() {
            if keep {
                refit_score += homography_reproj_error(&h_refit, &points.x1[i], &points.x2[i]);
            }
        }
        let support = best.score.support;
        let score = Score {
            objective: CountThenResidual {
                count: support,
                residual: refit_score,
            },
            support,
        };
        best.consider_keeping_mask(&h_refit, score, Ties::Strict);
    }
}

/// Twice the signed area of the triangle (a, b, c) — zero iff collinear.
#[inline]
fn triangle_area2(a: &[f64; 2], b: &[f64; 2], c: &[f64; 2]) -> f64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

/// True if any 3 of the 4 points are (near-)collinear.
///
/// Uses a scale-aware threshold: points spread over a 100-px patch and a 10-px
/// patch should both survive unless they're genuinely collinear. The threshold
/// scales with the sample's bounding-box area to avoid false positives at small
/// scales and false negatives at large scales.
pub(super) fn sample_is_degenerate(pts: &[[f64; 2]; 4]) -> bool {
    let mut xmin = pts[0][0];
    let mut xmax = pts[0][0];
    let mut ymin = pts[0][1];
    let mut ymax = pts[0][1];
    for p in pts.iter().skip(1) {
        if p[0] < xmin {
            xmin = p[0];
        }
        if p[0] > xmax {
            xmax = p[0];
        }
        if p[1] < ymin {
            ymin = p[1];
        }
        if p[1] > ymax {
            ymax = p[1];
        }
    }
    let span = (xmax - xmin).max(ymax - ymin).max(1.0);
    // Area threshold: 1% of the sample's bounding square. Well below any
    // non-degenerate sample but reliably nonzero for genuine triangles.
    let eps = 0.01 * span * span;
    let triples = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)];
    for (i, j, k) in triples {
        if triangle_area2(&pts[i], &pts[j], &pts[k]).abs() < eps {
            return true;
        }
    }
    false
}

/// Computes the squared reprojection error for mapping `x1` to `x2` via the homography `h`.
pub(super) fn homography_reproj_error(h: &Mat3F64, x1: &Vec2F64, x2: &Vec2F64) -> f64 {
    let x1h = Vec3F64::new(x1.x, x1.y, 1.0);
    let hx = *h * x1h;
    if hx.z.abs() < 1e-12 {
        return f64::INFINITY;
    }
    let u = hx.x / hx.z;
    let v = hx.y / hx.z;
    let dx = u - x2.x;
    let dy = v - x2.y;
    dx * dx + dy * dy
}
