//! # Two-View Initialization
//!
//! Recovers relative camera pose (R, t) and 3D structure from 2D correspondences.
//!
//! ## Pipeline
//!
//! ```text
//! Correspondences (pixel space)
//!     │
//!     │   ── rayon::join ──────────────────────────────────────────────
//!     ├─→ RANSAC + 8-point fundamental → F → enforce(σ,σ,0) → E (default)
//!     │   (NEON Sampson scoring in pixel space; LO+ refit on inliers)
//!     │   *or* 5-point Nistér → E (on-manifold by construction) if the
//!     │   builder is given an [`EssentialNister5ptSolver`] (translation-
//!     │   priority path)
//!     │
//!     └─→ RANSAC + 4-point → H → multiple (R,t,n) candidates
//!         (NEON H-reproj scoring; stagnation early-exit @ 200 iters)
//!     │   ────────────────────────────────────────────────────────────
//!     ▼
//! Model selection: H wins iff H_inliers > 0.8 × epipolar_inliers (planar scene)
//!     │
//!     ▼
//! Cheirality vote (4 candidates from E, ≥4 from H):
//!     ├─→ count_cheirality_fast — closed-form midpoint depths, no SVD
//!     └─→ winner-only triangulate_inliers — full 4×4 SVD per inlier
//!     │
//!     ▼
//! LM refinement (R, t) on Σ Sampson² over inliers, anneal-tight inlier set
//!     │
//!     ▼
//! (R, t_direction, 3D points)   ← translation scale is lost (quotient of SE(3))
//! ```
//!
//! The output translation is a **unit vector** (direction only). Scale is irrecoverable
//! from two views alone — this is the SE(3) → essential manifold quotient in action.
//!
//! ## Solver choice
//!
//! Three epipolar solvers ship with this crate; the builder picks one:
//! - [`Fundamental7ptSolver`] — seven-point pixel-space hypotheses,
//!   scoring all real roots and refining with the eight-point solver. Returns
//!   [`TwoViewModel::Fundamental`].
//! - [`Fundamental8ptSolver`] (default) — the original eight-point sampling strategy.
//! - [`EssentialNister5ptSolver`] — stays on the E manifold by construction (no
//!   σ-projection round-trip), preserving translation-direction accuracy at the
//!   cost of a slower per-sample polynomial solve. Returns
//!   [`TwoViewModel::Essential`].
//!
//! ```ignore
//! use kornia_3d::pose::{TwoViewEstimator, EssentialNister5ptSolver};
//!
//! // Default — 8-point fundamental + LM refinement.
//! let est = TwoViewEstimator::default();
//!
//! // Translation-priority — opt into the 5-point essential solver.
//! let est = TwoViewEstimator::builder()
//!     .epipolar_solver(EssentialNister5ptSolver::default())
//!     .build();
//! ```
//!
//! The variant returned in [`TwoViewResult::model`] reflects which solver ran, not
//! a downstream contract — both paths produce the same `(R, t, inliers)` shape.
//! Pose-only consumers should not branch on the variant.
//!
//! ## Performance
//!
//! The pipeline composes four independent wins over a serial scalar baseline:
//! parallel F+H RANSAC (`rayon::join`), NEON 2-lane f64 inner scorers (Sampson +
//! H-reproj on SoA-laid x/y arrays), a stagnation early-exit on H-RANSAC (which
//! can't tighten its adaptive cap on non-planar scenes), and a cheap-then-full
//! cheirality vote that replaces 4 × N SVDs with 1 × M SVDs (M = winner inliers).

#![allow(clippy::needless_range_loop)]

use crate::pose::fundamental::FundamentalError;
use crate::pose::lm_pose::{fundamental_from_rt, refine_pose_lm, LmPoseConfig};
use crate::pose::triangulation::{triangulate_inliers, TriangulateParams, TriangulationConfig};
use crate::pose::{
    decompose_essential, decompose_homography, enforce_essential_constraints,
    essential_from_fundamental, HomographyError,
};

use kornia_algebra::{Mat3F64, Vec2F64, Vec3F64};

/// Errors returned by two-view estimation utilities.
#[derive(thiserror::Error, Debug)]
pub enum TwoViewError {
    /// Input correspondences are invalid or insufficient.
    #[error("Need at least {required} correspondences and equal lengths")]
    InvalidInput {
        /// Minimum required correspondences for the chosen model.
        required: usize,
    },
    /// RANSAC failed to find a valid model.
    #[error("RANSAC failed to find a valid model")]
    RansacFailure,
    /// Requested RANSAC confidence is outside the open unit interval.
    #[error(
        "RANSAC confidence must be finite and strictly between zero and one, got {confidence}"
    )]
    InvalidConfidence {
        /// Invalid requested confidence.
        confidence: f64,
    },
    /// Two of the four E-decomposition candidates triangulate similar inlier
    /// counts (within `cheirality_ambiguity_max`), so the recovered pose is
    /// not uniquely determined — typical of pure-rotation, planar, or
    /// near-zero-parallax motion. Caller should request another frame pair
    /// rather than trust the winner.
    #[error(
        "cheirality ambiguous: second-best candidate has {second} of {best} inliers (ratio {ratio:.2} > {max_ratio:.2})"
    )]
    AmbiguousCheirality {
        /// Best candidate's cheirality-inlier count.
        best: usize,
        /// Runner-up's cheirality-inlier count.
        second: usize,
        /// Observed second/best ratio.
        ratio: f64,
        /// Configured maximum allowed ratio.
        max_ratio: f64,
    },
    /// SVD or other numerical decomposition failed (input may contain NaN/Inf).
    #[error("Numerical decomposition failed")]
    NumericalFailure,
    /// Fundamental estimation failed.
    #[error("Fundamental estimation error: {0}")]
    Fundamental(#[from] FundamentalError),
    /// Homography estimation failed.
    #[error("Homography estimation error: {0}")]
    Homography(#[from] HomographyError),
}

/// Parameters for RANSAC model estimation.
#[derive(Clone, Copy, Debug)]
pub struct RansacParams {
    /// Maximum number of RANSAC iterations.
    pub max_iterations: usize,
    /// Inlier threshold (pixel error). Compared against squared errors internally.
    pub threshold: f64,
    /// Minimum number of inliers required for acceptance.
    pub min_inliers: usize,
    /// Optional RNG seed for deterministic runs.
    pub random_seed: Option<u64>,
    /// Optional probability of sampling at least one all-inlier minimal set.
    /// When `None`, each estimator family preserves its historical default:
    /// 0.9999 for fundamental and essential estimation, 0.99 for homography.
    pub confidence: Option<f64>,
    /// If true, after the main RANSAC loop refit the model across ALL inliers
    /// using a least-squares solver (LO-RANSAC). The refit is kept only if it
    /// improves the inlier reprojection score. Fundamental and homography
    /// RANSAC honor this flag; default is `false` for backward compatibility.
    pub refit: bool,
}

impl Default for RansacParams {
    fn default() -> Self {
        Self {
            max_iterations: 2000,
            threshold: 1.0,
            min_inliers: 15,
            random_seed: Some(0),
            confidence: None,
            refit: false,
        }
    }
}

/// Result of a RANSAC model fit.
#[derive(Clone, Debug)]
pub struct RansacResult<M> {
    /// Estimated model.
    pub model: M,
    /// Per-point inlier mask.
    pub inliers: Vec<bool>,
    /// Total inlier count.
    pub inlier_count: usize,
    /// Sum of inlier errors (lower is better).
    pub score: f64,
}

/// Two-view model selected during estimation.
#[derive(Clone, Copy, Debug)]
pub enum TwoViewModel {
    /// Fundamental matrix model (pixel space).
    Fundamental(Mat3F64),
    /// Essential matrix model (metric/camera space) — emitted when the
    /// builder is configured with [`EssentialNister5ptSolver`].
    Essential(Mat3F64),
    /// Homography model (pixel space).
    Homography(Mat3F64),
}

/// Output of an [`EpipolarSolver`] — the inputs the downstream cheirality and
/// refinement stages need from any epipolar arm of the F-vs-H race.
pub struct EpipolarFit {
    /// Essential matrix used to decompose into 4 candidate `(R, t)` poses.
    /// For 8pt this is the σ-equalized lift of the F in `model`; for 5pt this
    /// equals the E in `model`.
    pub e: Mat3F64,
    /// Variant tag identifying which solver produced the fit. Carried into
    /// [`TwoViewResult::model`] so callers can inspect what ran without
    /// branching on solver type.
    pub model: TwoViewModel,
    /// Per-input-correspondence inlier mask.
    pub inliers: Vec<bool>,
    /// Cached inlier popcount.
    pub inlier_count: usize,
    /// Residual threshold the solver classified inliers at — fed to a
    /// downstream [`PoseRefiner`] for derived sub-thresholds (e.g. LO+ anneal).
    pub residual_threshold: f64,
}

/// Strategy for the epipolar arm of the F-vs-H race in [`TwoViewEstimator`].
///
/// Three implementations ship in this crate:
/// - [`Fundamental7ptSolver`] — seven-point F with eight-point refinement.
/// - [`Fundamental8ptSolver`] — pixel-space 8-point F + (σ, σ, 0) lift to E.
/// - [`EssentialNister5ptSolver`] — calibrated 5-point Nistér E (on-manifold).
///
/// Implement this trait to plug in a custom solver (e.g. USAC, MAGSAC).
pub trait EpipolarSolver: Send + Sync {
    /// Run the solver's RANSAC and return the essential matrix used for
    /// decomposition, plus the per-correspondence inlier mask.
    fn estimate(
        &self,
        x1: &[Vec2F64],
        x2: &[Vec2F64],
        k1: &Mat3F64,
        k2: &Mat3F64,
    ) -> Result<EpipolarFit, TwoViewError>;
}

/// Eight-point fundamental matrix solver. Pixel-space normalization and
/// σ-equalization on the F → E lift. Faster linear solve than the 5-point
/// path; cleaner rotation; translation absorbs some noise from the
/// `(σ, σ, 0)` clipping. Default solver in [`TwoViewEstimator::builder`].
#[derive(Clone, Debug, Default)]
pub struct Fundamental8ptSolver {
    /// RANSAC parameters for the F-fit.
    pub ransac: RansacParams,
}

impl EpipolarSolver for Fundamental8ptSolver {
    fn estimate(
        &self,
        x1: &[Vec2F64],
        x2: &[Vec2F64],
        k1: &Mat3F64,
        k2: &Mat3F64,
    ) -> Result<EpipolarFit, TwoViewError> {
        let res = ransac_fundamental_8point(x1, x2, &self.ransac)?;
        let f = res.model;
        let e_raw = essential_from_fundamental(&f, k1, k2);
        let e = enforce_essential_constraints(&e_raw).ok_or(TwoViewError::NumericalFailure)?;
        Ok(EpipolarFit {
            e,
            model: TwoViewModel::Fundamental(f),
            inliers: res.inliers,
            inlier_count: res.inlier_count,
            residual_threshold: self.ransac.threshold,
        })
    }
}

/// Seven-point fundamental matrix strategy with eight-point inlier refinement.
///
/// Opt in via [`TwoViewEstimatorBuilder::epipolar_solver`]. Hypotheses are fitted
/// in pixel space, and the resulting fundamental matrix is lifted to the
/// essential manifold using the supplied intrinsics.
#[derive(Clone, Debug, Default)]
pub struct Fundamental7ptSolver {
    /// RANSAC parameters for the fundamental fit.
    pub ransac: RansacParams,
}

impl EpipolarSolver for Fundamental7ptSolver {
    fn estimate(
        &self,
        x1: &[Vec2F64],
        x2: &[Vec2F64],
        k1: &Mat3F64,
        k2: &Mat3F64,
    ) -> Result<EpipolarFit, TwoViewError> {
        let res = ransac_fundamental(x1, x2, &self.ransac)?;
        let f = res.model;
        let e_raw = essential_from_fundamental(&f, k1, k2);
        let e = enforce_essential_constraints(&e_raw).ok_or(TwoViewError::NumericalFailure)?;
        Ok(EpipolarFit {
            e,
            model: TwoViewModel::Fundamental(f),
            inliers: res.inliers,
            inlier_count: res.inlier_count,
            residual_threshold: self.ransac.threshold,
        })
    }
}

/// Nistér five-point essential solver. Calibrated rays; on-manifold by
/// construction (no σ-clipping round-trip); preserves translation-direction
/// accuracy at the cost of a 10-degree polynomial solve per RANSAC sample.
/// Opt in via [`TwoViewEstimatorBuilder::epipolar_solver`].
#[derive(Clone, Debug, Default)]
pub struct EssentialNister5ptSolver {
    /// RANSAC parameters for the E-fit.
    pub ransac: RansacParams,
}

impl EpipolarSolver for EssentialNister5ptSolver {
    fn estimate(
        &self,
        x1: &[Vec2F64],
        x2: &[Vec2F64],
        k1: &Mat3F64,
        k2: &Mat3F64,
    ) -> Result<EpipolarFit, TwoViewError> {
        let res = ransac_essential_5pt(x1, x2, k1, k2, &self.ransac)?;
        let e = res.model;
        // 5pt produces an E that already satisfies the manifold constraints
        // exactly (algebraic by construction). Skip enforce_essential_constraints
        // to avoid an SVD round-trip that can only add noise.
        Ok(EpipolarFit {
            e,
            model: TwoViewModel::Essential(e),
            inliers: res.inliers,
            inlier_count: res.inlier_count,
            residual_threshold: self.ransac.threshold,
        })
    }
}

/// Inputs handed to a [`PoseRefiner`]. Bundled into a struct so the trait
/// signature stays stable as the pipeline evolves.
pub struct RefineContext<'a> {
    /// Full input correspondences in view 1 (pixel space).
    pub x1: &'a [Vec2F64],
    /// Full input correspondences in view 2 (pixel space).
    pub x2: &'a [Vec2F64],
    /// View-1 intrinsics.
    pub k1: &'a Mat3F64,
    /// View-2 intrinsics.
    pub k2: &'a Mat3F64,
    /// Per-input-correspondence inlier mask from the winning model's RANSAC.
    pub inliers: &'a [bool],
    /// Cached popcount of `inliers`.
    pub inlier_count: usize,
    /// Base inlier threshold the winning model used (for deriving annealed
    /// sub-thresholds).
    pub residual_threshold: f64,
    /// Tag identifying the winning model — refiners may want to behave
    /// differently for homography vs epipolar inlier sets.
    pub model: TwoViewModel,
}

/// Strategy for nonlinear `(R, t)` refinement after the cheirality vote.
///
/// Two implementations ship in this crate:
/// - [`LmRefiner`] — Levenberg-Marquardt on Σ Sampson² with optional
///   threshold-annealed polish (OpenCV USAC LO+ pattern).
/// - [`NoopRefiner`] — pass-through; useful for measuring raw F/E pipeline
///   accuracy or when the caller owns its own downstream refinement.
pub trait PoseRefiner: Send + Sync {
    /// Refine the pose. Implementations must never return a worse pose than
    /// they were given (LM has this property by construction; pass-through
    /// returns the input).
    fn refine(&self, r: Mat3F64, t: Vec3F64, ctx: &RefineContext<'_>) -> (Mat3F64, Vec3F64);
}

/// Levenberg-Marquardt refiner with optional threshold-annealed polish.
///
/// Pass 1 runs LM on the full RANSAC inlier set. Subsequent passes (one per
/// entry in `anneal_thresholds`) re-classify inliers at a tighter Sampson
/// threshold (`anneal_thresholds[i] * ctx.residual_threshold`) and re-run LM
/// on the cleaner subset. Annealing only fires for epipolar models —
/// homography inlier sets live in pixel-reproj space and use a different
/// threshold scale.
///
/// This is OpenCV USAC's LO+ inner loop and the largest accuracy lever between
/// kornia-rs and OpenCV USAC.
#[derive(Clone, Debug)]
pub struct LmRefiner {
    /// LM solver knobs (max iters, tolerances).
    pub config: LmPoseConfig,
    /// Multipliers applied to `ctx.residual_threshold` for the annealed
    /// polish passes. Empty disables annealing. Default `[0.5, 0.25]`.
    pub anneal_thresholds: Vec<f64>,
    /// Minimum inlier count required to admit an annealed pass — below this
    /// the residual set is too small to reliably constrain the 5-DOF problem.
    /// Default 30.
    pub anneal_min_inliers: usize,
}

impl Default for LmRefiner {
    fn default() -> Self {
        Self {
            config: LmPoseConfig::default(),
            anneal_thresholds: vec![0.5, 0.25],
            anneal_min_inliers: 30,
        }
    }
}

impl PoseRefiner for LmRefiner {
    fn refine(&self, r: Mat3F64, t: Vec3F64, ctx: &RefineContext<'_>) -> (Mat3F64, Vec3F64) {
        let n_inl = ctx.inlier_count;
        if n_inl < 6 {
            return (r, t);
        }
        let mut x1_inl: Vec<Vec2F64> = Vec::with_capacity(n_inl);
        let mut x2_inl: Vec<Vec2F64> = Vec::with_capacity(n_inl);
        // SoA companion for the NEON Sampson scorer used by the anneal passes.
        let mut x1i_x: Vec<f64> = Vec::with_capacity(n_inl);
        let mut x1i_y: Vec<f64> = Vec::with_capacity(n_inl);
        let mut x2i_x: Vec<f64> = Vec::with_capacity(n_inl);
        let mut x2i_y: Vec<f64> = Vec::with_capacity(n_inl);
        for (i, &is_inl) in ctx.inliers.iter().enumerate() {
            if is_inl {
                x1_inl.push(ctx.x1[i]);
                x2_inl.push(ctx.x2[i]);
                x1i_x.push(ctx.x1[i].x);
                x1i_y.push(ctx.x1[i].y);
                x2i_x.push(ctx.x2[i].x);
                x2i_y.push(ctx.x2[i].y);
            }
        }

        let (mut r_cur, mut t_cur) =
            refine_pose_lm(r, t, &x1_inl, &x2_inl, ctx.k1, ctx.k2, &self.config);

        let do_anneal = matches!(
            ctx.model,
            TwoViewModel::Fundamental(_) | TwoViewModel::Essential(_)
        ) && !self.anneal_thresholds.is_empty();
        if !do_anneal {
            return (r_cur, t_cur);
        }

        // Tighter passes start near-optimal — halve the iteration budget.
        let mut tight_cfg = self.config;
        tight_cfg.max_iters = (self.config.max_iters / 2).max(3);
        let k1_inv = ctx.k1.inverse();
        let k2_inv_t = ctx.k2.inverse().transpose();
        let mut tight_mask = vec![false; x1_inl.len()];
        let mut x1_tight: Vec<Vec2F64> = Vec::with_capacity(x1_inl.len());
        let mut x2_tight: Vec<Vec2F64> = Vec::with_capacity(x1_inl.len());
        let points = ScoringPoints::new(&x1i_x, &x1i_y, &x2i_x, &x2i_y);
        for &mult in &self.anneal_thresholds {
            let tight_t = ctx.residual_threshold * mult;
            let tight_t_sq = tight_t * tight_t;
            let f_cur = fundamental_from_rt(&r_cur, &t_cur, &k1_inv, &k2_inv_t);
            let (n_tight, _) = points.fundamental(&f_cur, tight_t_sq, &mut tight_mask);
            if n_tight < self.anneal_min_inliers {
                break;
            }
            x1_tight.clear();
            x2_tight.clear();
            for (k, &keep) in tight_mask.iter().enumerate() {
                if keep {
                    x1_tight.push(x1_inl[k]);
                    x2_tight.push(x2_inl[k]);
                }
            }
            let (r_new, t_new) = refine_pose_lm(
                r_cur, t_cur, &x1_tight, &x2_tight, ctx.k1, ctx.k2, &tight_cfg,
            );
            r_cur = r_new;
            t_cur = t_new;
        }
        (r_cur, t_cur)
    }
}

/// Identity refiner — returns the cheirality-vote pose unchanged.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoopRefiner;

impl PoseRefiner for NoopRefiner {
    fn refine(&self, r: Mat3F64, t: Vec3F64, _ctx: &RefineContext<'_>) -> (Mat3F64, Vec3F64) {
        (r, t)
    }
}

/// Configured two-view pose estimator. Build via [`TwoViewEstimator::builder`]
/// or take defaults via `TwoViewEstimator::default()`.
///
/// ```ignore
/// use kornia_3d::pose::{TwoViewEstimator, EssentialNister5ptSolver, NoopRefiner};
///
/// let est = TwoViewEstimator::builder()
///     .epipolar_solver(EssentialNister5ptSolver::default())
///     .refiner(NoopRefiner)
///     .build();
/// let result = est.estimate(&pts1, &pts2, &k1, &k2)?;
/// # Ok::<_, kornia_3d::pose::TwoViewError>(())
/// ```
pub struct TwoViewEstimator {
    epipolar: Box<dyn EpipolarSolver>,
    homography_ransac: RansacParams,
    homography_inlier_ratio: f64,
    triangulation: TriangulationConfig,
    refiner: Box<dyn PoseRefiner>,
}

impl Default for TwoViewEstimator {
    fn default() -> Self {
        TwoViewEstimator::builder().build()
    }
}

impl TwoViewEstimator {
    /// Start a builder with default settings: 8-point fundamental solver, LM
    /// refinement with `[0.5, 0.25]` annealing, default RANSAC + triangulation
    /// parameters.
    pub fn builder() -> TwoViewEstimatorBuilder {
        TwoViewEstimatorBuilder::default()
    }
}

/// Builder for [`TwoViewEstimator`]. Defaults documented on
/// [`TwoViewEstimator::builder`].
pub struct TwoViewEstimatorBuilder {
    epipolar: Box<dyn EpipolarSolver>,
    homography_ransac: RansacParams,
    homography_inlier_ratio: f64,
    triangulation: TriangulationConfig,
    refiner: Box<dyn PoseRefiner>,
}

impl Default for TwoViewEstimatorBuilder {
    fn default() -> Self {
        Self {
            epipolar: Box::new(Fundamental8ptSolver::default()),
            homography_ransac: RansacParams::default(),
            homography_inlier_ratio: 0.8,
            triangulation: TriangulationConfig::default(),
            refiner: Box::new(LmRefiner::default()),
        }
    }
}

impl TwoViewEstimatorBuilder {
    /// Plug in the epipolar solver for the F-vs-H race's epipolar arm.
    pub fn epipolar_solver(mut self, solver: impl EpipolarSolver + 'static) -> Self {
        self.epipolar = Box::new(solver);
        self
    }

    /// Plug in the post-cheirality pose refiner.
    pub fn refiner(mut self, refiner: impl PoseRefiner + 'static) -> Self {
        self.refiner = Box::new(refiner);
        self
    }

    /// Skip nonlinear refinement entirely. Equivalent to
    /// `.refiner(NoopRefiner)`.
    pub fn no_refinement(self) -> Self {
        self.refiner(NoopRefiner)
    }

    /// RANSAC parameters for the homography arm.
    pub fn homography_ransac(mut self, ransac: RansacParams) -> Self {
        self.homography_ransac = ransac;
        self
    }

    /// Inlier-ratio threshold for preferring H over the epipolar model. The
    /// homography wins iff `H_inliers > ratio * epipolar_inliers`.
    pub fn homography_inlier_ratio(mut self, ratio: f64) -> Self {
        self.homography_inlier_ratio = ratio;
        self
    }

    /// Triangulation-backed candidate-pose validation settings.
    pub fn triangulation(mut self, config: TriangulationConfig) -> Self {
        self.triangulation = config;
        self
    }

    /// Finalize the builder.
    pub fn build(self) -> TwoViewEstimator {
        TwoViewEstimator {
            epipolar: self.epipolar,
            homography_ransac: self.homography_ransac,
            homography_inlier_ratio: self.homography_inlier_ratio,
            triangulation: self.triangulation,
            refiner: self.refiner,
        }
    }
}

/// Output of two-view pose estimation.
#[derive(Clone, Debug)]
pub struct TwoViewResult {
    /// Selected model.
    pub model: TwoViewModel,
    /// Relative rotation from view1 to view2.
    pub rotation: Mat3F64,
    /// Relative translation direction from view1 to view2.
    pub translation: Vec3F64,
    /// Triangulated 3D points for inliers.
    pub points3d: Vec<Vec3F64>,
    /// Index into the input `x1`/`x2` arrays for each point in `points3d`.
    pub inlier_indices: Vec<usize>,
    /// Inlier mask from the selected model's RANSAC.
    pub inliers: Vec<bool>,
}

impl TwoViewResult {
    /// Median parallax angle in degrees between inlier bearing vectors.
    ///
    /// Converts each inlier point pair to normalized bearing vectors using the
    /// camera intrinsics, then computes the angle between them. Returns the
    /// median of these angles, or 0.0 if there are no valid inliers.
    pub fn median_parallax_deg(
        &self,
        x1: &[Vec2F64],
        x2: &[Vec2F64],
        camera: &crate::camera::PinholeCamera,
    ) -> f64 {
        let (fx, fy, cx, cy) = camera.intrinsics();
        let mut angles: Vec<f64> = self
            .inlier_indices
            .iter()
            .filter(|&&i| i < x1.len() && i < x2.len())
            .map(|&i| {
                let b1 = Vec3F64::new((x1[i].x - cx) / fx, (x1[i].y - cy) / fy, 1.0).normalize();
                let b2 = Vec3F64::new((x2[i].x - cx) / fx, (x2[i].y - cy) / fy, 1.0).normalize();
                b1.dot(b2).clamp(-1.0, 1.0).acos().to_degrees()
            })
            .collect();
        if angles.is_empty() {
            return 0.0;
        }
        let mid = angles.len() / 2;
        angles.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
        angles[mid]
    }
}

/// Fast cheirality counter — closed-form midpoint depths, no SVD, no allocs.
///
/// Replaces the 4×4 SVD inside `triangulate_inliers` for the 4-way (R, t)
/// candidate vote: three of four candidates are wrong (most points behind a
/// camera), and we only need SVD-quality 3D points for the *winner*.
///
/// `rays1`, `rays2_cam2` are pre-normalized direction vectors (call sites
/// hoist `K⁻¹·[x, 1]` and `.normalize()` once). Per-inlier work is then
/// just two dots and a 2×2 solve.
///
/// `min_parallax_sin2` is `sin²(min_parallax_deg)`; rays with `1 - b² <`
/// that threshold are dropped. This mirrors the parallax filter in
/// `triangulate_inliers` so the cheap counter and the full triangulator
/// rank candidates the same way — without it, a candidate with many
/// low-parallax points could win the cheap vote and lose the full pass.
fn count_cheirality_fast(
    rays1: &[Vec3F64],
    rays2_cam2: &[Vec3F64],
    inliers: &[bool],
    r: &Mat3F64,
    t: &Vec3F64,
    min_parallax_sin2: f64,
) -> usize {
    let r_t = r.transpose();
    let w = r_t * *t;
    let mut count = 0usize;
    for i in 0..rays1.len() {
        if !inliers[i] {
            continue;
        }
        let d1 = rays1[i];
        let d2 = r_t * rays2_cam2[i];
        let b = d1.dot(d2);
        let denom = 1.0 - b * b;
        if denom < min_parallax_sin2 {
            continue;
        }
        let d = d1.dot(w);
        let e = d2.dot(w);
        let s1 = (b * e - d) / denom;
        let s2 = (e - b * d) / denom;
        if s1 > 1e-8 && s2 > 1e-8 {
            count += 1;
        }
    }
    count
}

impl TwoViewEstimator {
    /// Estimate a two-view relative pose with model selection and
    /// triangulation. Full ORB-SLAM-style bootstrap: F + H RANSAC in parallel,
    /// model selection, 4-candidate cheirality vote, optional LM refinement.
    /// Returns `(R, t̂, inlier mask, 3D points)`. Translation is direction-only
    /// — monocular scale is unobservable.
    ///
    /// # Implementation outline
    ///
    /// 1. **Parallel epipolar + homography RANSAC** (`rayon::join`). Both
    ///    consume the same correspondences; runtime = max(F, H) instead of
    ///    sum. The epipolar arm runs whichever solver was registered on the
    ///    builder ([`Fundamental8ptSolver`] by default, or
    ///    [`EssentialNister5ptSolver`]).
    /// 2. **NEON-vectorized inner scorers**. Sampson (`score_inliers_f`) and
    ///    homography reprojection (`score_inliers_h`) run a 2-lane f64 path
    ///    on aarch64 over SoA `x_x[]` / `x_y[]` arrays — Vec2F64s are
    ///    flattened once outside the RANSAC loops.
    /// 3. **Stagnation early-exit** on H. The standard adaptive cap
    ///    `log(1-p)/log(1-w^s)` shrinks fast when w is high but stays at the
    ///    ceiling on non-planar scenes (where H's job is just to lose the
    ///    selection vote). A 200-iter no-improvement break shortcuts the
    ///    wasted work; confidence dropped to 0.99 since H is a model-
    ///    selection oracle, not a final-product matrix.
    /// 4. **LO+ for F** (Lebeda 2012). Outer-iterated expand-then-contract
    ///    schedule (4×, 3.4×, …, 1×) × 3 outer rounds; strict-improve gate at
    ///    the base threshold. **DEGENSAC** (Chum 2004) recovers from
    ///    dominant-plane degeneracy via F = [e′]ₓ H when ≥75% of F's inliers
    ///    are explained by a single plane.
    /// 5. **Two-stage cheirality vote**. `count_cheirality_fast` runs a
    ///    closed-form midpoint-depth check across 4 candidates without an
    ///    SVD or any allocation. The winning candidate alone gets the full
    ///    `triangulate_inliers` pass (4×4 SVD per point). Cheap counts also
    ///    drive the ambiguity-ratio guard so the test stays in one unit.
    /// 6. **Pose refinement** through the registered [`PoseRefiner`]
    ///    ([`LmRefiner`] by default; [`NoopRefiner`] to skip).
    pub fn estimate(
        &self,
        x1: &[Vec2F64],
        x2: &[Vec2F64],
        k1: &Mat3F64,
        k2: &Mat3F64,
    ) -> Result<TwoViewResult, TwoViewError> {
        // F-RANSAC and H-RANSAC are independent — same correspondences,
        // different models. `rayon::join` runs them on two cores so wall time
        // = max(F, H) instead of sum. On non-planar scenes H dominates (it
        // can't shrink its adaptive cap because the inlier ratio stays low),
        // so this win is real.
        let (epi_res, h_res) = rayon::join(
            || self.epipolar.estimate(x1, x2, k1, k2),
            || ransac_homography(x1, x2, &self.homography_ransac),
        );
        let epi = epi_res?;
        let res_h = h_res?;

        let use_h =
            (res_h.inlier_count as f64) > self.homography_inlier_ratio * (epi.inlier_count as f64);

        let k1_inv = k1.inverse();
        let k2_inv = k2.inverse();

        let tri_params = TriangulateParams {
            k1_inv: &k1_inv,
            k2_inv: &k2_inv,
            config: &self.triangulation,
        };

        let mut best_pose = None;
        let mut best_count = 0usize;
        let mut second_count = 0usize;
        let mut best_points = Vec::new();
        let mut best_indices = Vec::new();

        let normalize_ray = |k_inv: &Mat3F64, p: &Vec2F64| -> Vec3F64 {
            let r = *k_inv * Vec3F64::new(p.x, p.y, 1.0);
            r.normalize()
        };
        let rays1: Vec<Vec3F64> = x1.iter().map(|p| normalize_ray(&k1_inv, p)).collect();
        let rays2_cam2: Vec<Vec3F64> = x2.iter().map(|p| normalize_ray(&k2_inv, p)).collect();
        let min_parallax_sin2 = self
            .triangulation
            .min_parallax_deg
            .to_radians()
            .sin()
            .powi(2);

        let (poses, inliers, model): (Vec<(Mat3F64, Vec3F64)>, Vec<bool>, TwoViewModel) = if use_h {
            let h = res_h.model;
            (
                decompose_homography(&h, k1, k2),
                res_h.inliers,
                TwoViewModel::Homography(h),
            )
        } else {
            (
                decompose_essential(&epi.e)
                    .ok_or(TwoViewError::NumericalFailure)?
                    .to_vec(),
                epi.inliers,
                epi.model,
            )
        };

        // The ambiguity ratio (best/second) requires both counts to be
        // measured by the same predicate, so winner and runner-up both go
        // through the closed-form check. Mixing cheap (winner) with SVD
        // (runner-up) counts would bias the ratio: cheap ≥ SVD systematically
        // on degenerate parallax.
        let mut best_idx = None;
        for (idx, (r, t)) in poses.iter().enumerate() {
            let count =
                count_cheirality_fast(&rays1, &rays2_cam2, &inliers, r, t, min_parallax_sin2);
            if count >= self.triangulation.min_cheirality_count && count > best_count {
                second_count = best_count;
                best_count = count;
                best_idx = Some(idx);
            } else if count > second_count {
                second_count = count;
            }
        }
        if let Some(idx) = best_idx {
            let (r, t) = poses[idx];
            let (_full_count, points, indices) =
                triangulate_inliers(x1, x2, &inliers, &r, &t, &tri_params);
            best_pose = Some((r, t));
            best_points = points;
            best_indices = indices;
        }

        let (r, t) = match best_pose {
            Some(p) => p,
            None => return Err(TwoViewError::RansacFailure),
        };

        // Ambiguity guard: if the runner-up triangulates nearly as many
        // points as the winner, the decomposition is not uniquely determined
        // and committing to the winner would be a coin flip. Skip the check
        // when disabled (max == 1.0) or when best_count is zero.
        let ambiguity_max = self.triangulation.cheirality_ambiguity_max;
        if ambiguity_max < 1.0 && best_count > 0 {
            let ratio = second_count as f64 / best_count as f64;
            if ratio > ambiguity_max {
                return Err(TwoViewError::AmbiguousCheirality {
                    best: best_count,
                    second: second_count,
                    ratio,
                    max_ratio: ambiguity_max,
                });
            }
        }

        let inlier_count = if use_h {
            res_h.inlier_count
        } else {
            epi.inlier_count
        };
        let residual_threshold = if use_h {
            self.homography_ransac.threshold
        } else {
            epi.residual_threshold
        };
        let ctx = RefineContext {
            x1,
            x2,
            k1,
            k2,
            inliers: &inliers,
            inlier_count,
            residual_threshold,
            model,
        };
        let (r, t) = self.refiner.refine(r, t, &ctx);

        Ok(TwoViewResult {
            model,
            rotation: r,
            translation: t,
            points3d: best_points,
            inlier_indices: best_indices,
            inliers,
        })
    }
}

mod local_optimization;
mod robust;
mod scoring;

pub use robust::{
    ransac_essential_5pt, ransac_fundamental, ransac_fundamental_8point, ransac_homography,
};
use scoring::ScoringPoints;

#[cfg(test)]
mod tests;
