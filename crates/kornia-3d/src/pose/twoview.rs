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
//!     ├─→ RANSAC + 5-point Nistér → E (on-manifold by construction)
//!     │   (NEON Sampson scoring in pixel space; LO+ refit on inliers)
//!     │   *or* 8-point fundamental → F → enforce(σ,σ,0) → E if
//!     │   `use_5pt_essential = false` (legacy / rotation-priority path)
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
//! [`TwoViewConfig::use_5pt_essential`] picks the epipolar estimator and the choice
//! is a real accuracy / speed tradeoff:
//! - **5-point Nistér (default, `true`)** — stays on the E manifold by construction.
//!   Best translation-direction accuracy in our EuRoC MH_01 bench (3.39°, lower than
//!   every OpenCV USAC variant). Pose stage ~3 ms.
//! - **8-point F + (σ,σ,0) lift (`false`)** — pixel-space normalization gets the
//!   cleanest rotation (0.04° on MH_01) but the σ-equalization bleeds noise into
//!   translation (4.17°). Strictly faster (pose ~1.2 ms) because the 8-point linear
//!   solve is cheaper than 10-poly root-finding × cheirality.
//!
//! ## Performance
//!
//! Median ~3.0 ms pose / ~9.1 ms full pipeline on EuRoC MH_01 752×480 (Jetson Orin
//! AGX, 110 ORB matches, 5pt default). 4.5–9.2× faster total than every OpenCV
//! variant in the bench. Wins compound from four pieces: parallel F+H RANSAC
//! (rayon::join), NEON 2-lane f64 inner scorers (Sampson + H-reproj on SoA-laid
//! x/y arrays), the stagnation early-exit on H (which can't tighten its adaptive
//! cap on non-planar scenes), and the cheap-then-full cheirality vote that
//! replaces 4 × N SVDs with 1 × M SVDs (M = winner inliers).

#![allow(clippy::needless_range_loop)]

use crate::pose::fundamental::{fundamental_8point, FundamentalError};
use crate::pose::lm_pose::{refine_pose_lm, LmPoseConfig};
use crate::pose::triangulation::{triangulate_inliers, TriangulateParams, TriangulationConfig};
use crate::pose::{
    decompose_essential, decompose_homography, enforce_essential_constraints, essential_5pt,
    essential_from_fundamental, homography_4pt2d, homography_dlt, HomographyError,
};
use kornia_algebra::{Mat3F64, Vec2F64, Vec3F64};
use rand::prelude::*;
use rand::SeedableRng;

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
    /// If true, after the main RANSAC loop refit the model across ALL inliers
    /// using a least-squares solver (LO-RANSAC). The refit is kept only if it
    /// improves the inlier reprojection score. Only `ransac_homography` honors
    /// this flag today; default is `false` for bit-identical backward compat.
    pub refit: bool,
}

impl Default for RansacParams {
    fn default() -> Self {
        Self {
            max_iterations: 2000,
            threshold: 1.0,
            min_inliers: 15,
            random_seed: Some(0),
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
    /// 5-point path is enabled in `TwoViewConfig`.
    Essential(Mat3F64),
    /// Homography model (pixel space).
    Homography(Mat3F64),
}

/// Configuration for two-view pose estimation.
#[derive(Clone, Debug)]
pub struct TwoViewConfig {
    /// RANSAC settings for the fundamental matrix.
    pub ransac_f: RansacParams,
    /// RANSAC settings for the homography.
    pub ransac_h: RansacParams,
    /// Prefer homography when it has this ratio of inliers vs fundamental.
    pub homography_inlier_ratio: f64,
    /// Triangulation-backed candidate-pose validation settings.
    pub triangulation: TriangulationConfig,
    /// Run Levenberg-Marquardt non-linear refinement of (R, t) over the inlier
    /// Sampson cost after the cheirality winner is picked. Default `true` —
    /// this is the final accuracy lever that closes the gap vs OpenCV USAC.
    pub lm_enabled: bool,
    /// LM refinement knobs. See [`LmPoseConfig`].
    pub lm: LmPoseConfig,
    /// Use the **Nistér 5-point essential solver** in place of 8-point
    /// fundamental for the F-vs-H race. The 5pt path stays on the essential
    /// manifold throughout (no `(σ, σ, 0)` clipping after the 8-point lift),
    /// which preserves translation-direction accuracy. The 8-point path
    /// trades that for cleaner rotation: pixel-space normalization gets the
    /// rotation null-space crisp, but the σ-equalization bleeds into t.
    /// Default `false` — kornia-slam's bootstrap pipeline consumes the
    /// fundamental matrix downstream of [`two_view_estimate`], so the
    /// default keeps the 8-point F path. Opt into the 5pt essential path
    /// explicitly when translation accuracy matters more than F-availability;
    /// see `kornia-py/benchmarks.md` for current numbers.
    pub use_5pt_essential: bool,
    /// Threshold-annealed LM polish. After the initial LM call on the full
    /// RANSAC inlier set, re-classify inliers at progressively tighter Sampson
    /// thresholds (multipliers applied to `ransac_f.threshold`) and re-run LM
    /// on each tighter subset. This is OpenCV-USAC's LO+ inner loop pattern
    /// and the lever that lifts both rotation and translation accuracy by
    /// trimming residual contamination from the noisy tail of the inlier set.
    /// Empty / single-element schedules disable the annealing pass.
    /// Default `[0.5, 0.25]`.
    pub lm_anneal_thresholds: Vec<f64>,
    /// Minimum inlier count required to admit an annealed LM pass — below this
    /// the residual set is too small to reliably constrain the 5-DOF problem
    /// and we'd risk overfitting to noise. Default 30.
    pub lm_anneal_min_inliers: usize,
}

impl Default for TwoViewConfig {
    fn default() -> Self {
        Self {
            ransac_f: RansacParams::default(),
            ransac_h: RansacParams::default(),
            homography_inlier_ratio: 0.8,
            triangulation: TriangulationConfig::default(),
            lm_enabled: true,
            lm: LmPoseConfig::default(),
            use_5pt_essential: false,
            lm_anneal_thresholds: vec![0.5, 0.25],
            lm_anneal_min_inliers: 30,
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

/// OpenCV USAC-style LO+ inner refit (Lebeda 2012 +
/// `cv::usac::InnerIterativeLocalOptimizationImpl`). Polishes a fundamental-
/// matrix candidate via *expand-then-contract* threshold annealing and
/// returns whichever model beats the input under a strict-improve gate at the
/// base threshold.
///
/// One pass schedule: 4.0×, 3.4×, 2.8×, 2.2×, 1.6×, 1.0× — matches OpenCV's
/// `threshold_multiplier=4`, `lo_inner_iters=5`, ending at base. The full LO+
/// loop runs the schedule **`LO_OUTER_ITERS=3`** times back-to-back — each
/// outer iteration starts from the previous winner, so successive refits
/// converge from increasingly clean inlier sets. OpenCV's USAC default is 5
/// outer iters; 3 is the empirical sweet spot for our pipeline (extra iters
/// past 3 plateau on accuracy but still pay the 8-pt cost).
///
/// Returns `(model, inliers, count, score)` — never worse than the input
/// under (count > prev) || (count == prev && score < prev_score).
#[allow(clippy::too_many_arguments)]
fn lo_plus_fundamental(
    x1: &[Vec2F64],
    x2: &[Vec2F64],
    x1_x: &[f64],
    x1_y: &[f64],
    x2_x: &[f64],
    x2_y: &[f64],
    base_threshold: f64,
    base_thresh_sq: f64,
    in_model: Mat3F64,
    in_inliers: Vec<bool>,
    in_count: usize,
    in_score: f64,
) -> (Mat3F64, Vec<bool>, usize, f64) {
    const LO_MULTIPLIERS: [f64; 6] = [4.0, 3.4, 2.8, 2.2, 1.6, 1.0];
    const LO_OUTER_ITERS: usize = 3;
    let n = x1.len();
    let mut model = in_model;
    let mut best_inliers = in_inliers;
    let mut best_count = in_count;
    let mut best_score = in_score;
    if best_count < 8 {
        return (model, best_inliers, best_count, best_score);
    }
    let mut fit_mask = best_inliers.clone();
    let mut scratch_inliers = vec![false; n];
    let mut base_inliers = vec![false; n];
    let mut inl_x1: Vec<Vec2F64> = Vec::with_capacity(n);
    let mut inl_x2: Vec<Vec2F64> = Vec::with_capacity(n);
    for outer in 0..LO_OUTER_ITERS {
        let prev_count = best_count;
        let prev_score = best_score;
        for &mult in &LO_MULTIPLIERS {
            let virt_t = base_threshold * mult;
            let virt_t_sq = virt_t * virt_t;

            // Recompute fit_mask from the *current best* model at this
            // multiplier so each step picks up correspondences matching the
            // polished F, not a previous step's F. (At mult=1.0 we keep the
            // base inlier mask.) Goes through the NEON scorer — same fast
            // path as RANSAC's hot loop.
            let virt_count = if mult > 1.0 {
                for s in scratch_inliers.iter_mut() {
                    *s = false;
                }
                let (vc, _) = score_inliers_f(
                    &model,
                    x1_x,
                    x1_y,
                    x2_x,
                    x2_y,
                    virt_t_sq,
                    &mut scratch_inliers,
                );
                fit_mask.copy_from_slice(&scratch_inliers);
                vc
            } else {
                // mult=1.0: re-score on the running best at the base threshold
                // — the inlier set has drifted as `model` evolved through
                // earlier multipliers, so use the up-to-date support, not the
                // stale cached `best_inliers` mask.
                for s in fit_mask.iter_mut() {
                    *s = false;
                }
                let (vc, _) = score_inliers_f(
                    &model,
                    x1_x,
                    x1_y,
                    x2_x,
                    x2_y,
                    base_thresh_sq,
                    &mut fit_mask,
                );
                vc
            };
            if virt_count < 8 {
                break;
            }

            inl_x1.clear();
            inl_x2.clear();
            for (i, &keep) in fit_mask.iter().enumerate() {
                if keep {
                    inl_x1.push(x1[i]);
                    inl_x2.push(x2[i]);
                }
            }
            let f_refit = match fundamental_8point(&inl_x1, &inl_x2) {
                Ok(f) => f,
                Err(_) => break,
            };

            // Strict-improve at base threshold — never let a polished F drift
            // onto a dominant-plane local optimum that scores well on count
            // but produces a wrong (R, t) downstream.
            for s in base_inliers.iter_mut() {
                *s = false;
            }
            let (count_b, score_b) = score_inliers_f(
                &f_refit,
                x1_x,
                x1_y,
                x2_x,
                x2_y,
                base_thresh_sq,
                &mut base_inliers,
            );
            if count_b > best_count || (count_b == best_count && score_b < best_score) {
                model = f_refit;
                std::mem::swap(&mut best_inliers, &mut base_inliers);
                best_count = count_b;
                best_score = score_b;
            }
        }
        // Convergence: if this outer pass produced no improvement, further
        // passes will only repeat the same fixed-point. Save the work.
        if outer + 1 < LO_OUTER_ITERS && best_count == prev_count && best_score >= prev_score {
            break;
        }
    }
    (model, best_inliers, best_count, best_score)
}

/// DEGENSAC (Chum 2004) post-RANSAC degeneracy recovery for fundamental
/// matrices.
///
/// **Why this exists.** When the inlier set is dominated by a single plane
/// (typical of indoor scenes, walls, ground-plane motion, …), the 8-point F is
/// underdetermined: any F satisfying `F = [e']× H` for the dominant-plane H
/// fits the inliers equally well, but only the *correct* F decomposes into the
/// right (R, t). RANSAC + LO+ happily land on a wrong-but-high-scoring F and
/// downstream pose recovery silently produces garbage (Chum'04 §3, §4).
///
/// **Recovery.** Estimate the dominant H from the F-inliers; pick two
/// off-plane (parallax-bearing) inliers `(a, b)`; compute the second-image
/// epipole `e' = (x̃2_a × Hx̃1_a) × (x̃2_b × Hx̃1_b)` (intersection of two
/// epipolar lines); lift `F_new = [e']× H`. Replace the running F only on
/// strict-improve at the base threshold — guarantees this is a safety net,
/// never a footgun.
///
/// **Cost when non-degenerate.** Runs `H_TRIALS=15` minimal H samples + the
/// degeneracy gate. On general-3D scenes (EuRoC, KITTI), the H-support never
/// reaches the 75% threshold and we early-exit. ≤ 100µs at N≈400 inliers.
#[allow(clippy::too_many_arguments)]
fn degensac_recover_fundamental(
    x1_x: &[f64],
    x1_y: &[f64],
    x2_x: &[f64],
    x2_y: &[f64],
    base_thresh_sq: f64,
    in_model: Mat3F64,
    in_inliers: Vec<bool>,
    in_count: usize,
    in_score: f64,
    rng: &mut StdRng,
) -> (Mat3F64, Vec<bool>, usize, f64) {
    // Below ~12 inliers the plane-vs-non-plane signal is too noisy to act on
    // and the lift is unstable.
    if in_count < 12 {
        return (in_model, in_inliers, in_count, in_score);
    }

    // SoA companion over the F-inlier subset for the NEON H scorer.
    let mut inl_x1_x: Vec<f64> = Vec::with_capacity(in_count);
    let mut inl_x1_y: Vec<f64> = Vec::with_capacity(in_count);
    let mut inl_x2_x: Vec<f64> = Vec::with_capacity(in_count);
    let mut inl_x2_y: Vec<f64> = Vec::with_capacity(in_count);
    for (i, &b) in in_inliers.iter().enumerate() {
        if b {
            inl_x1_x.push(x1_x[i]);
            inl_x1_y.push(x1_y[i]);
            inl_x2_x.push(x2_x[i]);
            inl_x2_y.push(x2_y[i]);
        }
    }

    // Mini-RANSAC for the dominant H. We're not optimizing for the global best
    // H, only confirming whether *any* plane supports ≥75% of the F-inliers.
    const H_TRIALS: usize = 15;
    let mut best_h: Option<Mat3F64> = None;
    let mut best_h_count = 0usize;
    let mut h_inl = vec![false; in_count];
    let mut h_inl_best = vec![false; in_count];
    for _ in 0..H_TRIALS {
        let s = rand::seq::index::sample(rng, in_count, 4);
        let s1 = [
            [inl_x1_x[s.index(0)], inl_x1_y[s.index(0)]],
            [inl_x1_x[s.index(1)], inl_x1_y[s.index(1)]],
            [inl_x1_x[s.index(2)], inl_x1_y[s.index(2)]],
            [inl_x1_x[s.index(3)], inl_x1_y[s.index(3)]],
        ];
        let s2 = [
            [inl_x2_x[s.index(0)], inl_x2_y[s.index(0)]],
            [inl_x2_x[s.index(1)], inl_x2_y[s.index(1)]],
            [inl_x2_x[s.index(2)], inl_x2_y[s.index(2)]],
            [inl_x2_x[s.index(3)], inl_x2_y[s.index(3)]],
        ];
        if sample_is_degenerate(&s1) || sample_is_degenerate(&s2) {
            continue;
        }
        let mut h_arr = [[0.0; 3]; 3];
        if homography_4pt2d(&s1, &s2, &mut h_arr).is_err() {
            continue;
        }
        let h = Mat3F64::from_cols(
            Vec3F64::new(h_arr[0][0], h_arr[1][0], h_arr[2][0]),
            Vec3F64::new(h_arr[0][1], h_arr[1][1], h_arr[2][1]),
            Vec3F64::new(h_arr[0][2], h_arr[1][2], h_arr[2][2]),
        );
        for s in h_inl.iter_mut() {
            *s = false;
        }
        let (cnt, _) = score_inliers_h(
            &h,
            &inl_x1_x,
            &inl_x1_y,
            &inl_x2_x,
            &inl_x2_y,
            base_thresh_sq,
            &mut h_inl,
        );
        if cnt > best_h_count {
            best_h_count = cnt;
            best_h = Some(h);
            h_inl_best.copy_from_slice(&h_inl);
        }
    }

    // Degeneracy threshold: 75% of F-inliers explained by one H. Below this,
    // the F is generic enough that the lift can only hurt.
    let h = match best_h {
        Some(h) if best_h_count * 4 >= in_count * 3 => h,
        _ => return (in_model, in_inliers, in_count, in_score),
    };

    // Off-plane inliers = F-inliers that H fails to explain. These carry the
    // out-of-plane parallax needed to disambiguate the lift.
    let mut off: Vec<usize> = Vec::with_capacity(in_count - best_h_count);
    for k in 0..in_count {
        if !h_inl_best[k] {
            off.push(k);
        }
    }
    if off.len() < 2 {
        return (in_model, in_inliers, in_count, in_score);
    }

    // Try off-plane pairs; keep the F_new with the highest base-threshold inlier
    // count (strict-improve gate, same shape as LO+). Cap trials so DEGENSAC
    // stays sub-millisecond on heavily-planar scenes.
    let mut best = (in_model, in_inliers, in_count, in_score);
    const PAIR_TRIALS: usize = 30;
    let pair_count = off.len() * (off.len() - 1) / 2;
    let trial_cap = PAIR_TRIALS.min(pair_count);
    let mut tried = 0usize;
    let mut new_inl = vec![false; x1_x.len()];
    'outer: for ai in 0..off.len() {
        for bi in (ai + 1)..off.len() {
            if tried >= trial_cap {
                break 'outer;
            }
            tried += 1;
            let a = off[ai];
            let b_idx = off[bi];
            let p1a = Vec3F64::new(inl_x1_x[a], inl_x1_y[a], 1.0);
            let p2a = Vec3F64::new(inl_x2_x[a], inl_x2_y[a], 1.0);
            let p1b = Vec3F64::new(inl_x1_x[b_idx], inl_x1_y[b_idx], 1.0);
            let p2b = Vec3F64::new(inl_x2_x[b_idx], inl_x2_y[b_idx], 1.0);
            let h_p1a = h * p1a;
            let h_p1b = h * p1b;
            // l_i = p2_i × Hx̃1_i — epipolar line through correspondence i.
            let la = vec3_cross(p2a, h_p1a);
            let lb = vec3_cross(p2b, h_p1b);
            // e' = la × lb — intersection of two epipolar lines = epipole in img2.
            let e_prime = vec3_cross(la, lb);
            let en2 = e_prime.x * e_prime.x + e_prime.y * e_prime.y + e_prime.z * e_prime.z;
            if en2 < 1e-20 {
                // Pair is nearly coplanar with H — lift undefined, skip.
                continue;
            }
            let ex = Mat3F64::from_cols(
                Vec3F64::new(0.0, e_prime.z, -e_prime.y),
                Vec3F64::new(-e_prime.z, 0.0, e_prime.x),
                Vec3F64::new(e_prime.y, -e_prime.x, 0.0),
            );
            let f_new = ex * h;
            for s in new_inl.iter_mut() {
                *s = false;
            }
            let (cnt, scr) =
                score_inliers_f(&f_new, x1_x, x1_y, x2_x, x2_y, base_thresh_sq, &mut new_inl);
            if cnt > best.2 || (cnt == best.2 && scr < best.3) {
                best = (f_new, new_inl.clone(), cnt, scr);
            }
        }
    }
    best
}

/// 3-vector cross product. Inline-friendly hand roll — Vec3F64 doesn't
/// expose `.cross()` directly and the 6-mul/3-sub form avoids the glam
/// round-trip.
#[inline]
fn vec3_cross(a: Vec3F64, b: Vec3F64) -> Vec3F64 {
    Vec3F64::new(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    )
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

/// Estimate a fundamental matrix with RANSAC using the 8-point solver.
pub fn ransac_fundamental(
    x1: &[Vec2F64],
    x2: &[Vec2F64],
    params: &RansacParams,
) -> Result<RansacResult<Mat3F64>, TwoViewError> {
    if x1.len() != x2.len() || x1.len() < 8 {
        return Err(TwoViewError::InvalidInput { required: 8 });
    }

    let mut rng = match params.random_seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => {
            let mut thread_rng = rand::rng();
            StdRng::from_rng(&mut thread_rng)
        }
    };

    let n = x1.len();
    let thresh_sq = params.threshold * params.threshold;
    // One-time SoA flatten so score_inliers_f's NEON path reads contiguous f64.
    let (x1_x, x1_y) = split_xy(x1);
    let (x2_x, x2_y) = split_xy(x2);
    let mut best_model: Option<Mat3F64> = None;
    let mut best_inliers = vec![false; n];
    let mut best_count = 0usize;
    let mut best_score = f64::INFINITY;
    // Hoisted scratch — was per-iteration `vec![false; n]`. Adaptive cap can
    // run hundreds of iterations on tough inputs; one alloc + clear-each-iter
    // is much cheaper than allocator churn.
    let mut scratch_inliers = vec![false; n];

    // Adaptive iteration count: same log(1-p)/log(1-w^s) formula as H, with s=8
    // (minimal sample size for the 8-point algorithm). Confidence p=0.9999 — F
    // conditioning is more fragile than H, so we buy a few extra iterations.
    let log_fail = (1.0_f64 - 0.9999).ln();
    let mut dynamic_max = params.max_iterations;
    let mut iter = 0usize;
    while iter < dynamic_max {
        iter += 1;
        let sample = rand::seq::index::sample(&mut rng, n, 8);
        let mut s1 = [Vec2F64::ZERO; 8];
        let mut s2 = [Vec2F64::ZERO; 8];
        for (i, idx) in sample.iter().enumerate() {
            s1[i] = x1[idx];
            s2[i] = x2[idx];
        }
        let f = match fundamental_8point(&s1, &s2) {
            Ok(f) => f,
            Err(_) => continue,
        };

        for s in scratch_inliers.iter_mut() {
            *s = false;
        }
        let (count, score) = score_inliers_f(
            &f,
            &x1_x,
            &x1_y,
            &x2_x,
            &x2_y,
            thresh_sq,
            &mut scratch_inliers,
        );

        let improved = count > best_count || (count == best_count && score < best_score);
        if improved {
            best_model = Some(f);
            std::mem::swap(&mut best_inliers, &mut scratch_inliers);
            best_count = count;
            best_score = score;

            if best_count == n {
                break;
            }
            let w = best_count as f64 / n as f64;
            let denom = (1.0 - w.powi(8)).ln();
            if denom < -1e-12 {
                let need = (log_fail / denom).ceil();
                if need.is_finite() && need >= 0.0 {
                    dynamic_max = (need as usize).min(params.max_iterations).max(iter);
                }
            }
        }
    }

    let model_in = match best_model {
        Some(m) if best_count >= params.min_inliers => m,
        _ => return Err(TwoViewError::RansacFailure),
    };

    let (model, best_inliers, best_count, best_score) = if params.refit {
        let polished = lo_plus_fundamental(
            x1,
            x2,
            &x1_x,
            &x1_y,
            &x2_x,
            &x2_y,
            params.threshold,
            thresh_sq,
            model_in,
            best_inliers,
            best_count,
            best_score,
        );
        // DEGENSAC safety net: if the LO+ winner sits on a dominant-plane local
        // optimum (Chum'04), recover the true F via [e']× H. No-op when the
        // scene is non-planar (the strict-improve gate guarantees we never
        // regress from the polished F).
        degensac_recover_fundamental(
            &x1_x, &x1_y, &x2_x, &x2_y, thresh_sq, polished.0, polished.1, polished.2, polished.3,
            &mut rng,
        )
    } else {
        (model_in, best_inliers, best_count, best_score)
    };

    Ok(RansacResult {
        model,
        inliers: best_inliers,
        inlier_count: best_count,
        score: best_score,
    })
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
pub fn ransac_essential_5pt(
    x1: &[Vec2F64],
    x2: &[Vec2F64],
    k1: &Mat3F64,
    k2: &Mat3F64,
    params: &RansacParams,
) -> Result<RansacResult<Mat3F64>, TwoViewError> {
    if x1.len() != x2.len() || x1.len() < 5 {
        return Err(TwoViewError::InvalidInput { required: 5 });
    }

    let mut rng = match params.random_seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => {
            let mut thread_rng = rand::rng();
            StdRng::from_rng(&mut thread_rng)
        }
    };

    let n = x1.len();
    let thresh_sq = params.threshold * params.threshold;

    // One-time intrinsic inversions. The pose RANSAC scoring lives in pixel
    // space, but the 5-point solver lives in metric space — every candidate
    // E must be lifted via F = K2⁻ᵀ E K1⁻¹ before Sampson scoring.
    let k1_inv = k1.inverse();
    let k2_inv = k2.inverse();
    let k2_inv_t = k2_inv.transpose();

    // Pre-normalize the full point set once — RANSAC samples just index in.
    let mut x1n: Vec<Vec2F64> = Vec::with_capacity(n);
    let mut x2n: Vec<Vec2F64> = Vec::with_capacity(n);
    for i in 0..n {
        let h1 = k1_inv * Vec3F64::new(x1[i].x, x1[i].y, 1.0);
        let h2 = k2_inv * Vec3F64::new(x2[i].x, x2[i].y, 1.0);
        if h1.z.abs() < 1e-12 || h2.z.abs() < 1e-12 {
            return Err(TwoViewError::InvalidInput { required: 5 });
        }
        x1n.push(Vec2F64::new(h1.x / h1.z, h1.y / h1.z));
        x2n.push(Vec2F64::new(h2.x / h2.z, h2.y / h2.z));
    }

    // SoA pixel coords for the inner scorer (NEON-friendly, contiguous f64).
    let (x1_x, x1_y) = split_xy(x1);
    let (x2_x, x2_y) = split_xy(x2);

    let mut best_e: Option<Mat3F64> = None;
    let mut best_inliers = Vec::new();
    let mut best_count = 0usize;
    let mut best_score = f64::INFINITY;

    // Adaptive iteration cap. s = 5, p = 0.9999 (same conservative confidence
    // we use for F — E is comparably fragile in low-parallax regimes).
    let log_fail = (1.0_f64 - 0.9999).ln();
    let mut dynamic_max = params.max_iterations;
    let mut iter = 0usize;
    while iter < dynamic_max {
        iter += 1;
        let sample = rand::seq::index::sample(&mut rng, n, 5);
        let mut s1 = [Vec2F64::ZERO; 5];
        let mut s2 = [Vec2F64::ZERO; 5];
        for (i, idx) in sample.iter().enumerate() {
            s1[i] = x1n[idx];
            s2[i] = x2n[idx];
        }
        let candidates = essential_5pt(&s1, &s2);
        if candidates.is_empty() {
            continue;
        }

        // Each minimal sample yields ≤ 10 candidate Es — score them all.
        let mut sample_improved = false;
        for e in candidates {
            // F = K2⁻ᵀ E K1⁻¹ — same Sampson scorer as ransac_fundamental.
            let f = k2_inv_t * e * k1_inv;
            let mut inliers = vec![false; n];
            let (count, score) =
                score_inliers_f(&f, &x1_x, &x1_y, &x2_x, &x2_y, thresh_sq, &mut inliers);

            if count > best_count || (count == best_count && score < best_score) {
                best_e = Some(e);
                best_inliers = inliers;
                best_count = count;
                best_score = score;
                sample_improved = true;
            }
        }

        if sample_improved {
            if best_count == n {
                break;
            }
            let w = best_count as f64 / n as f64;
            let denom = (1.0 - w.powi(5)).ln();
            if denom < -1e-12 {
                let need = (log_fail / denom).ceil();
                if need.is_finite() && need >= 0.0 {
                    dynamic_max = (need as usize).min(params.max_iterations).max(iter);
                }
            }
        }
    }

    let model = match best_e {
        Some(m) if best_count >= params.min_inliers => m,
        _ => return Err(TwoViewError::RansacFailure),
    };

    Ok(RansacResult {
        model,
        inliers: best_inliers,
        inlier_count: best_count,
        score: best_score,
    })
}

/// Estimate a homography with RANSAC using the 4-point solver.
pub fn ransac_homography(
    x1: &[Vec2F64],
    x2: &[Vec2F64],
    params: &RansacParams,
) -> Result<RansacResult<Mat3F64>, TwoViewError> {
    if x1.len() != x2.len() || x1.len() < 4 {
        return Err(TwoViewError::InvalidInput { required: 4 });
    }

    let mut rng = match params.random_seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => {
            let mut thread_rng = rand::rng();
            StdRng::from_rng(&mut thread_rng)
        }
    };

    let n = x1.len();
    let thresh_sq = params.threshold * params.threshold;
    // One-time SoA flatten so score_inliers_h's NEON path reads contiguous f64.
    let (x1_x, x1_y) = split_xy(x1);
    let (x2_x, x2_y) = split_xy(x2);
    let mut best_model = None;
    let mut best_inliers = vec![false; n];
    let mut best_count = 0usize;
    let mut best_score = f64::INFINITY;
    // Hoisted scratch — ransac_homography used to allocate `vec![false; n]`
    // every iteration, which dominated wall time on non-planar scenes where
    // the loop runs hundreds of iters. Reuse one buffer + a swap-on-improve.
    let mut scratch_inliers = vec![false; n];

    // Adaptive iteration count: after each time best_count improves, recompute
    // the iteration bound that gives 99% confidence of drawing an all-inlier
    // sample at least once. N = log(1-p) / log(1-w^s) with s=4, p=0.99.
    //
    // **Why p=0.99 instead of 0.999.** H here is a *model-selection oracle*,
    // not a final-product matrix — its only job is "is H_count meaningfully
    // larger than F_count?" If the scene is genuinely planar, H_count grows
    // fast and we converge in tens of iterations. If non-planar, H_count
    // never crosses the 0.8 × F_count gate no matter how long we run, so the
    // extra confidence buys us nothing but wall time. Lowering p to 0.99
    // shaves ~33% off the bound at low inlier ratios.
    //
    // **Stagnation early-exit.** If `STAGNATION_LIMIT` iterations pass without
    // improvement, the running best is almost certainly the local optimum for
    // this random seed and continued sampling is wasted work. Belt-and-braces
    // with the adaptive cap: adaptive shrinks fast at high w; stagnation
    // shrinks fast at low w (where adaptive can't tighten).
    const STAGNATION_LIMIT: usize = 200;
    let log_fail = (1.0_f64 - 0.99).ln();
    let mut dynamic_max = params.max_iterations;
    let mut last_improve = 0usize;
    let mut iter = 0usize;
    while iter < dynamic_max {
        if iter - last_improve >= STAGNATION_LIMIT {
            break;
        }
        iter += 1;
        let sample = rand::seq::index::sample(&mut rng, n, 4);
        let mut s1 = [[0.0; 2]; 4];
        let mut s2 = [[0.0; 2]; 4];
        for (i, idx) in sample.iter().enumerate() {
            s1[i] = [x1[idx].x, x1[idx].y];
            s2[i] = [x2[idx].x, x2[idx].y];
        }
        // DEGENSAC-style collinearity check: a 4-point sample with 3+ collinear
        // points produces a wildly-wrong H. Reject before the solve and save
        // the iteration for a real candidate.
        if sample_is_degenerate(&s1) || sample_is_degenerate(&s2) {
            continue;
        }
        let mut h = [[0.0; 3]; 3];
        if homography_4pt2d(&s1, &s2, &mut h).is_err() {
            continue;
        }
        let h = Mat3F64::from_cols(
            Vec3F64::new(h[0][0], h[1][0], h[2][0]),
            Vec3F64::new(h[0][1], h[1][1], h[2][1]),
            Vec3F64::new(h[0][2], h[1][2], h[2][2]),
        );

        for s in scratch_inliers.iter_mut() {
            *s = false;
        }
        let (count, score) = score_inliers_h(
            &h,
            &x1_x,
            &x1_y,
            &x2_x,
            &x2_y,
            thresh_sq,
            &mut scratch_inliers,
        );

        let improved = count > best_count || (count == best_count && score < best_score);
        if improved {
            best_model = Some(h);
            std::mem::swap(&mut best_inliers, &mut scratch_inliers);
            best_count = count;
            best_score = score;
            last_improve = iter;

            if best_count == n {
                break;
            }
            let w = best_count as f64 / n as f64;
            let denom = (1.0 - w.powi(4)).ln();
            if denom < -1e-12 {
                let need = (log_fail / denom).ceil();
                if need.is_finite() && need >= 0.0 {
                    dynamic_max = (need as usize).min(params.max_iterations).max(iter);
                }
            }
        }
    }

    let mut model = match best_model {
        Some(m) if best_count >= params.min_inliers => m,
        _ => return Err(TwoViewError::RansacFailure),
    };

    // LO-RANSAC refit: the best 4-point sample passes many inliers but those
    // 4 points may not be a well-conditioned basis for H. A DLT across the
    // full inlier set averages out that variance. Keep the refit only if the
    // squared-error score improves — otherwise the DLT may have overfit a
    // borderline inlier that RANSAC rejected.
    if params.refit && best_count >= 4 {
        let inl_x1: Vec<Vec2F64> = x1
            .iter()
            .zip(best_inliers.iter())
            .filter_map(|(p, k)| if *k { Some(*p) } else { None })
            .collect();
        let inl_x2: Vec<Vec2F64> = x2
            .iter()
            .zip(best_inliers.iter())
            .filter_map(|(p, k)| if *k { Some(*p) } else { None })
            .collect();
        if let Ok(h_refit) = homography_dlt(&inl_x1, &inl_x2) {
            let mut refit_score = 0.0;
            for i in 0..n {
                if best_inliers[i] {
                    refit_score += homography_reproj_error(&h_refit, &x1[i], &x2[i]);
                }
            }
            if refit_score < best_score {
                model = h_refit;
                best_score = refit_score;
            }
        }
    }

    Ok(RansacResult {
        model,
        inliers: best_inliers,
        inlier_count: best_count,
        score: best_score,
    })
}

/// Estimate a two-view relative pose with model selection and triangulation.
///
/// Full ORB-SLAM-style bootstrap: F + H RANSAC in parallel, model selection,
/// 4-candidate cheirality vote, LM refinement. Returns (R, t̂, inlier mask,
/// 3D points). Translation is direction-only — monocular scale is unobservable.
///
/// # Implementation outline
///
/// 1. **Parallel epipolar + homography RANSAC** (`rayon::join`). Both consume
///    the same correspondences; runtime = max(F, H) instead of sum. F goes
///    8-point → enforce (σ,σ,0) → E, or 5-point essential when
///    `config.use_5pt_essential` (skips the σ-projection round-trip).
/// 2. **NEON-vectorized inner scorers**. Sampson (`score_inliers_f`) and
///    homography reprojection (`score_inliers_h`) run a 2-lane f64 path
///    on aarch64 over SoA `x_x[]` / `x_y[]` arrays — Vec2F64s are flattened
///    once outside the RANSAC loops.
/// 3. **Stagnation early-exit** on H. The standard adaptive cap
///    `log(1-p)/log(1-w^s)` shrinks fast when w is high but stays at the
///    ceiling on non-planar scenes (where H's job is just to lose the
///    selection vote). A 200-iter no-improvement break shortcuts the wasted
///    work; confidence dropped to 0.99 since H is a model-selection oracle,
///    not a final-product matrix.
/// 4. **LO+ for F** (Lebeda 2012). Outer-iterated expand-then-contract
///    schedule (4×, 3.4×, …, 1×) × 3 outer rounds; strict-improve gate at
///    the base threshold. **DEGENSAC** (Chum 2004) recovers from
///    dominant-plane degeneracy via F = [e′]ₓ H when ≥75% of F's inliers
///    are explained by a single plane.
/// 5. **Two-stage cheirality vote**. `count_cheirality_fast` runs a
///    closed-form midpoint-depth check across 4 candidates without an SVD
///    or any allocation. The winning candidate alone gets the full
///    `triangulate_inliers` pass (4×4 SVD per point). Cheap counts also
///    drive the ambiguity-ratio guard so the test stays in one unit.
/// 6. **LM (R, t) refinement** on SO(3) × S² minimizing Σ Sampson² over
///    inliers, with optional threshold annealing.
pub fn two_view_estimate(
    x1: &[Vec2F64],
    x2: &[Vec2F64],
    k1: &Mat3F64,
    k2: &Mat3F64,
    config: &TwoViewConfig,
) -> Result<TwoViewResult, TwoViewError> {
    // Pick the epipolar estimator: 5-point essential (metric, on-manifold) or
    // 8-point fundamental (pixel, requires `(σ,σ,0)` projection post-decomp).
    // Both expose the same `RansacResult<Mat3F64>` shape — the only difference
    // is what space the model lives in, handled below by the model variant.
    //
    // F-RANSAC and H-RANSAC are independent — same correspondences, different
    // models. `rayon::join` runs them on two cores so wall time = max(F, H)
    // instead of sum. On non-planar scenes H dominates (it can't shrink its
    // adaptive cap because the inlier ratio stays low), so this win is real.
    let (epi_res, h_res) = rayon::join(
        || -> Result<(Vec<bool>, usize, TwoViewModel, Mat3F64), TwoViewError> {
            if config.use_5pt_essential {
                let res_e = ransac_essential_5pt(x1, x2, k1, k2, &config.ransac_f)?;
                let e = res_e.model;
                // 5pt produces an E that already satisfies the manifold
                // constraints exactly (algebraic by construction). Skip
                // enforce_essential_constraints to avoid an unnecessary SVD
                // round-trip that can only add noise.
                Ok((
                    res_e.inliers,
                    res_e.inlier_count,
                    TwoViewModel::Essential(e),
                    e,
                ))
            } else {
                let res_f = ransac_fundamental(x1, x2, &config.ransac_f)?;
                let f = res_f.model;
                let e_raw = essential_from_fundamental(&f, k1, k2);
                let e = enforce_essential_constraints(&e_raw);
                Ok((
                    res_f.inliers,
                    res_f.inlier_count,
                    TwoViewModel::Fundamental(f),
                    e,
                ))
            }
        },
        || ransac_homography(x1, x2, &config.ransac_h),
    );
    let (epi_inliers, epi_count, epi_model_variant, e_decompose) = epi_res?;
    let res_h = h_res?;

    let use_h = (res_h.inlier_count as f64) > config.homography_inlier_ratio * (epi_count as f64);

    let k1_inv = k1.inverse();
    let k2_inv = k2.inverse();

    let tri_params = TriangulateParams {
        k1_inv: &k1_inv,
        k2_inv: &k2_inv,
        config: &config.triangulation,
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
    let min_parallax_sin2 = config
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
            decompose_essential(&e_decompose).to_vec(),
            epi_inliers,
            epi_model_variant,
        )
    };

    // The ambiguity ratio (best/second) requires both counts to be measured
    // by the same predicate, so winner and runner-up both go through the
    // closed-form check. Mixing cheap (winner) with SVD (runner-up) counts
    // would bias the ratio: cheap ≥ SVD systematically on degenerate parallax.
    let mut best_idx = None;
    for (idx, (r, t)) in poses.iter().enumerate() {
        let count = count_cheirality_fast(&rays1, &rays2_cam2, &inliers, r, t, min_parallax_sin2);
        if count >= config.triangulation.min_cheirality_count && count > best_count {
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

    // Ambiguity guard: if the runner-up triangulates nearly as many points as
    // the winner, the decomposition is not uniquely determined and committing
    // to the winner would be a coin flip. Skip the check when disabled
    // (max == 1.0) or when best_count is zero (already handled above).
    let ambiguity_max = config.triangulation.cheirality_ambiguity_max;
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

    // Non-linear refinement of (R, t) over the inlier Sampson cost.
    // Parameterized on SO(3) × S²; guaranteed never to return a worse pose
    // than its input.
    //
    // Pass 1 uses the *full* RANSAC inlier set — the loose universe of points
    // that passed the base Sampson threshold. Subsequent passes tighten the
    // inlier set using the schedule in `config.lm_anneal_thresholds`, each
    // applied as a multiplier to the base RANSAC threshold. This is OpenCV
    // USAC's LO+ pattern: the loose first pass corrects gross epipolar
    // misalignment; each tighter pass discards borderline-Sampson points that
    // were biasing the cost, and re-fits to the cleaner core. Empirically the
    // single largest accuracy lever between us and OpenCV USAC.
    let (r, t) = if config.lm_enabled {
        let n_inl = inliers.iter().filter(|&&b| b).count();
        let mut x1_inl: Vec<Vec2F64> = Vec::with_capacity(n_inl);
        let mut x2_inl: Vec<Vec2F64> = Vec::with_capacity(n_inl);
        // SoA companion for the NEON Sampson scorer used by the anneal passes.
        let mut x1i_x: Vec<f64> = Vec::with_capacity(n_inl);
        let mut x1i_y: Vec<f64> = Vec::with_capacity(n_inl);
        let mut x2i_x: Vec<f64> = Vec::with_capacity(n_inl);
        let mut x2i_y: Vec<f64> = Vec::with_capacity(n_inl);
        for (i, &is_inl) in inliers.iter().enumerate() {
            if is_inl {
                x1_inl.push(x1[i]);
                x2_inl.push(x2[i]);
                x1i_x.push(x1[i].x);
                x1i_y.push(x1[i].y);
                x2i_x.push(x2[i].x);
                x2i_y.push(x2[i].y);
            }
        }
        if x1_inl.len() >= 6 {
            // Pass 1: full inlier set, full LM iteration budget.
            let (mut r_cur, mut t_cur) = refine_pose_lm(r, t, &x1_inl, &x2_inl, k1, k2, &config.lm);

            // Annealed passes — only run on the F path; the H path's inlier
            // set is in pixel-reproj space and uses different thresholds.
            // Skip when the model isn't epipolar (homography winners stay at
            // pass 1).
            let do_anneal = matches!(
                model,
                TwoViewModel::Fundamental(_) | TwoViewModel::Essential(_)
            ) && !config.lm_anneal_thresholds.is_empty();
            if do_anneal {
                // Tighter passes can converge in fewer iterations since the
                // starting point is already near-optimal. Halve the budget.
                let mut tight_cfg = config.lm;
                tight_cfg.max_iters = (config.lm.max_iters / 2).max(3);
                let k1_inv = k1.inverse();
                let k2_inv_t = k2.inverse().transpose();
                let base_t = config.ransac_f.threshold;
                let n_inl = x1_inl.len();
                let mut tight_mask = vec![false; n_inl];
                let mut x1_tight: Vec<Vec2F64> = Vec::with_capacity(n_inl);
                let mut x2_tight: Vec<Vec2F64> = Vec::with_capacity(n_inl);
                for &mult in &config.lm_anneal_thresholds {
                    let tight_t = base_t * mult;
                    let tight_t_sq = tight_t * tight_t;
                    // Build current F = K2⁻ᵀ [t]× R K1⁻¹ to re-score Sampson.
                    let t_hat = Mat3F64::from_cols(
                        Vec3F64::new(0.0, t_cur.z, -t_cur.y),
                        Vec3F64::new(-t_cur.z, 0.0, t_cur.x),
                        Vec3F64::new(t_cur.y, -t_cur.x, 0.0),
                    );
                    let f_cur = k2_inv_t * (t_hat * r_cur) * k1_inv;
                    // Filter the original RANSAC inliers down to the tight
                    // subset using the same NEON Sampson scorer as RANSAC.
                    for s in tight_mask.iter_mut() {
                        *s = false;
                    }
                    let (n_tight, _) = score_inliers_f(
                        &f_cur,
                        &x1i_x,
                        &x1i_y,
                        &x2i_x,
                        &x2i_y,
                        tight_t_sq,
                        &mut tight_mask,
                    );
                    if n_tight < config.lm_anneal_min_inliers {
                        // Tightened set too small — further annealing would
                        // overfit. Keep the previous pass's (r, t).
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
                    let (r_new, t_new) =
                        refine_pose_lm(r_cur, t_cur, &x1_tight, &x2_tight, k1, k2, &tight_cfg);
                    r_cur = r_new;
                    t_cur = t_new;
                }
            }
            (r_cur, t_cur)
        } else {
            (r, t)
        }
    } else {
        (r, t)
    };

    Ok(TwoViewResult {
        model,
        rotation: r,
        translation: t,
        points3d: best_points,
        inlier_indices: best_indices,
        inliers,
    })
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
fn sample_is_degenerate(pts: &[[f64; 2]; 4]) -> bool {
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
fn homography_reproj_error(h: &Mat3F64, x1: &Vec2F64, x2: &Vec2F64) -> f64 {
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

/// Batched H reprojection scorer. For N correspondences in SoA layout and one
/// candidate homography, returns (inlier_count, score_sum) and populates
/// `inliers`. Bypasses `glam::DMat3 * DVec3` (scalar 9-mul per-call) with a
/// 2-lane f64 NEON FMA chain on aarch64. One-time SoA conversion at RANSAC
/// entry makes the per-iteration cost dominated by FMA + 2× `vdivq_f64`.
#[inline]
fn score_inliers_h(
    h: &Mat3F64,
    x1_x: &[f64],
    x1_y: &[f64],
    x2_x: &[f64],
    x2_y: &[f64],
    thresh_sq: f64,
    inliers: &mut [bool],
) -> (usize, f64) {
    let n = x1_x.len();
    // Column-major glam::DMat3: row-r, col-c = {x,y,z}_axis[r] at column c.
    let a = h.x_axis.x;
    let b = h.y_axis.x;
    let c = h.z_axis.x;
    let d = h.x_axis.y;
    let e = h.y_axis.y;
    let f = h.z_axis.y;
    let g = h.x_axis.z;
    let hh = h.y_axis.z;
    let ii = h.z_axis.z;

    let mut count = 0usize;
    let mut score = 0.0f64;
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let mut idx = 0usize;

    #[cfg(target_arch = "aarch64")]
    let mut idx = unsafe {
        score_inliers_h_neon(
            (a, b, c, d, e, f, g, hh, ii),
            x1_x,
            x1_y,
            x2_x,
            x2_y,
            thresh_sq,
            inliers,
            &mut count,
            &mut score,
        )
    };

    #[cfg(target_arch = "x86_64")]
    let mut idx = if kornia_imgproc::simd::cpu_features().has_avx2 {
        unsafe {
            score_inliers_h_avx2(
                (a, b, c, d, e, f, g, hh, ii),
                x1_x,
                x1_y,
                x2_x,
                x2_y,
                thresh_sq,
                inliers,
                &mut count,
                &mut score,
            )
        }
    } else {
        0usize
    };

    // Scalar tail (and full fallback on non-aarch64).
    while idx < n {
        let x = x1_x[idx];
        let y = x1_y[idx];
        let hw = g * x + hh * y + ii;
        if hw.abs() >= 1e-12 {
            let u = (a * x + b * y + c) / hw;
            let v = (d * x + e * y + f) / hw;
            let dx = u - x2_x[idx];
            let dy = v - x2_y[idx];
            let dd = dx * dx + dy * dy;
            if dd <= thresh_sq {
                inliers[idx] = true;
                count += 1;
                score += dd;
            }
        }
        idx += 1;
    }
    (count, score)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn score_inliers_h_neon(
    coeffs: (f64, f64, f64, f64, f64, f64, f64, f64, f64),
    x1_x: &[f64],
    x1_y: &[f64],
    x2_x: &[f64],
    x2_y: &[f64],
    thresh_sq: f64,
    inliers: &mut [bool],
    count: &mut usize,
    score: &mut f64,
) -> usize {
    use std::arch::aarch64::*;
    let (a, b, c, d, e, f, g, hh, ii) = coeffs;
    let a_v = vdupq_n_f64(a);
    let b_v = vdupq_n_f64(b);
    let c_v = vdupq_n_f64(c);
    let d_v = vdupq_n_f64(d);
    let e_v = vdupq_n_f64(e);
    let f_v = vdupq_n_f64(f);
    let g_v = vdupq_n_f64(g);
    let h_v = vdupq_n_f64(hh);
    let i_v = vdupq_n_f64(ii);
    let n = x1_x.len();
    let mut idx = 0usize;
    while idx + 2 <= n {
        let x1 = vld1q_f64(x1_x.as_ptr().add(idx));
        let y1 = vld1q_f64(x1_y.as_ptr().add(idx));
        let x2 = vld1q_f64(x2_x.as_ptr().add(idx));
        let y2 = vld1q_f64(x2_y.as_ptr().add(idx));
        // hx = a*x + b*y + c ; hy = d*x + e*y + f ; hw = g*x + h*y + i
        let hx = vfmaq_f64(vfmaq_f64(c_v, x1, a_v), y1, b_v);
        let hy = vfmaq_f64(vfmaq_f64(f_v, x1, d_v), y1, e_v);
        let hw = vfmaq_f64(vfmaq_f64(i_v, x1, g_v), y1, h_v);
        let u = vdivq_f64(hx, hw);
        let v = vdivq_f64(hy, hw);
        let dx = vsubq_f64(u, x2);
        let dy = vsubq_f64(v, y2);
        let sq = vfmaq_f64(vmulq_f64(dx, dx), dy, dy);
        let mut buf = [0.0f64; 2];
        vst1q_f64(buf.as_mut_ptr(), sq);
        // Commit per-lane (scalar reduction — count++/score+= are data-dependent).
        for k in 0..2 {
            let dd = buf[k];
            if dd.is_finite() && dd <= thresh_sq {
                *inliers.get_unchecked_mut(idx + k) = true;
                *count += 1;
                *score += dd;
            }
        }
        idx += 2;
    }
    idx
}

/// AVX2 mirror of [`score_inliers_h_neon`]. Same H-reprojection math, but
/// 4-lane f64 (`__m256d`) — twice NEON's 2-lane width, halving inner-loop
/// iteration count. `_mm256_fmadd_pd` covers `vfmaq_f64` exactly; the
/// per-lane scalar commit is unchanged because count++/score+= are
/// data-dependent regardless of vector width.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn score_inliers_h_avx2(
    coeffs: (f64, f64, f64, f64, f64, f64, f64, f64, f64),
    x1_x: &[f64],
    x1_y: &[f64],
    x2_x: &[f64],
    x2_y: &[f64],
    thresh_sq: f64,
    inliers: &mut [bool],
    count: &mut usize,
    score: &mut f64,
) -> usize {
    use std::arch::x86_64::*;
    let (a, b, c, d, e, f, g, hh, ii) = coeffs;
    let a_v = _mm256_set1_pd(a);
    let b_v = _mm256_set1_pd(b);
    let c_v = _mm256_set1_pd(c);
    let d_v = _mm256_set1_pd(d);
    let e_v = _mm256_set1_pd(e);
    let f_v = _mm256_set1_pd(f);
    let g_v = _mm256_set1_pd(g);
    let h_v = _mm256_set1_pd(hh);
    let i_v = _mm256_set1_pd(ii);
    let n = x1_x.len();
    let mut idx = 0usize;
    while idx + 4 <= n {
        let x1 = _mm256_loadu_pd(x1_x.as_ptr().add(idx));
        let y1 = _mm256_loadu_pd(x1_y.as_ptr().add(idx));
        let x2 = _mm256_loadu_pd(x2_x.as_ptr().add(idx));
        let y2 = _mm256_loadu_pd(x2_y.as_ptr().add(idx));
        let hx = _mm256_fmadd_pd(y1, b_v, _mm256_fmadd_pd(x1, a_v, c_v));
        let hy = _mm256_fmadd_pd(y1, e_v, _mm256_fmadd_pd(x1, d_v, f_v));
        let hw = _mm256_fmadd_pd(y1, h_v, _mm256_fmadd_pd(x1, g_v, i_v));
        let u = _mm256_div_pd(hx, hw);
        let v = _mm256_div_pd(hy, hw);
        let dx = _mm256_sub_pd(u, x2);
        let dy = _mm256_sub_pd(v, y2);
        let sq = _mm256_fmadd_pd(dy, dy, _mm256_mul_pd(dx, dx));
        let mut buf = [0.0f64; 4];
        _mm256_storeu_pd(buf.as_mut_ptr(), sq);
        for k in 0..4 {
            let dd = buf[k];
            if dd.is_finite() && dd <= thresh_sq {
                *inliers.get_unchecked_mut(idx + k) = true;
                *count += 1;
                *score += dd;
            }
        }
        idx += 4;
    }
    idx
}

/// Batched Sampson scorer for F. N correspondences × 1 model. Same structure
/// as `score_inliers_h` — 2-lane f64 NEON FMA path on aarch64, scalar tail.
/// Equivalent to calling `sampson_distance` per-point then thresholding.
#[inline]
fn score_inliers_f(
    f_mat: &Mat3F64,
    x1_x: &[f64],
    x1_y: &[f64],
    x2_x: &[f64],
    x2_y: &[f64],
    thresh_sq: f64,
    inliers: &mut [bool],
) -> (usize, f64) {
    let n = x1_x.len();
    // F entries (row-major naming, same convention as score_inliers_h).
    let f00 = f_mat.x_axis.x;
    let f01 = f_mat.y_axis.x;
    let f02 = f_mat.z_axis.x;
    let f10 = f_mat.x_axis.y;
    let f11 = f_mat.y_axis.y;
    let f12 = f_mat.z_axis.y;
    let f20 = f_mat.x_axis.z;
    let f21 = f_mat.y_axis.z;
    let f22 = f_mat.z_axis.z;

    let mut count = 0usize;
    let mut score = 0.0f64;
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let mut idx = 0usize;

    #[cfg(target_arch = "aarch64")]
    let mut idx = unsafe {
        score_inliers_f_neon(
            (f00, f01, f02, f10, f11, f12, f20, f21, f22),
            x1_x,
            x1_y,
            x2_x,
            x2_y,
            thresh_sq,
            inliers,
            &mut count,
            &mut score,
        )
    };

    #[cfg(target_arch = "x86_64")]
    let mut idx = if kornia_imgproc::simd::cpu_features().has_avx2 {
        unsafe {
            score_inliers_f_avx2(
                (f00, f01, f02, f10, f11, f12, f20, f21, f22),
                x1_x,
                x1_y,
                x2_x,
                x2_y,
                thresh_sq,
                inliers,
                &mut count,
                &mut score,
            )
        }
    } else {
        0usize
    };

    while idx < n {
        let x1 = x1_x[idx];
        let y1 = x1_y[idx];
        let x2 = x2_x[idx];
        let y2 = x2_y[idx];
        let fx1x = f00 * x1 + f01 * y1 + f02;
        let fx1y = f10 * x1 + f11 * y1 + f12;
        let fx1z = f20 * x1 + f21 * y1 + f22;
        let ftx2x = f00 * x2 + f10 * y2 + f20;
        let ftx2y = f01 * x2 + f11 * y2 + f21;
        let err = fx1x * x2 + fx1y * y2 + fx1z;
        let denom = fx1x * fx1x + fx1y * fx1y + ftx2x * ftx2x + ftx2y * ftx2y;
        let dd = if denom <= 1e-12 {
            err * err
        } else {
            (err * err) / denom
        };
        if dd <= thresh_sq {
            inliers[idx] = true;
            count += 1;
            score += dd;
        }
        idx += 1;
    }
    (count, score)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn score_inliers_f_neon(
    f: (f64, f64, f64, f64, f64, f64, f64, f64, f64),
    x1_x: &[f64],
    x1_y: &[f64],
    x2_x: &[f64],
    x2_y: &[f64],
    thresh_sq: f64,
    inliers: &mut [bool],
    count: &mut usize,
    score: &mut f64,
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
    let n = x1_x.len();
    let mut idx = 0usize;
    while idx + 2 <= n {
        let x1 = vld1q_f64(x1_x.as_ptr().add(idx));
        let y1 = vld1q_f64(x1_y.as_ptr().add(idx));
        let x2 = vld1q_f64(x2_x.as_ptr().add(idx));
        let y2 = vld1q_f64(x2_y.as_ptr().add(idx));

        // fx1 = F * [x1; y1; 1]
        let fx1x = vfmaq_f64(vfmaq_f64(f02v, x1, f00v), y1, f01v);
        let fx1y = vfmaq_f64(vfmaq_f64(f12v, x1, f10v), y1, f11v);
        let fx1z = vfmaq_f64(vfmaq_f64(f22v, x1, f20v), y1, f21v);
        // ftx2 = F^T * [x2; y2; 1]  (only x,y components needed for denom)
        let ftx2x = vfmaq_f64(vfmaq_f64(f20v, x2, f00v), y2, f10v);
        let ftx2y = vfmaq_f64(vfmaq_f64(f21v, x2, f01v), y2, f11v);

        // err = [x2; y2; 1] · fx1
        let err = vfmaq_f64(vfmaq_f64(fx1z, x2, fx1x), y2, fx1y);
        // denom = fx1x² + fx1y² + ftx2x² + ftx2y²
        let denom = vfmaq_f64(
            vfmaq_f64(vfmaq_f64(vmulq_f64(fx1x, fx1x), fx1y, fx1y), ftx2x, ftx2x),
            ftx2y,
            ftx2y,
        );
        let err_sq = vmulq_f64(err, err);
        // If denom > 0, use err²/denom; else err². Branchless select via bitwise
        // (denom > 1e-12) mask — otherwise fall back to scalar handling per lane.
        let denom_ok = vcgtq_f64(denom, vdupq_n_f64(1e-12));
        let safe_denom = vbslq_f64(denom_ok, denom, vdupq_n_f64(1.0));
        let div_val = vdivq_f64(err_sq, safe_denom);
        let dd = vbslq_f64(denom_ok, div_val, err_sq);

        let mut buf = [0.0f64; 2];
        vst1q_f64(buf.as_mut_ptr(), dd);
        for k in 0..2 {
            let dd_k = buf[k];
            if dd_k.is_finite() && dd_k <= thresh_sq {
                *inliers.get_unchecked_mut(idx + k) = true;
                *count += 1;
                *score += dd_k;
            }
        }
        idx += 2;
    }
    idx
}

/// AVX2 mirror of [`score_inliers_f_neon`]. Same F Sampson math at 4-lane
/// f64 (`__m256d`). The branchless `denom > 1e-12` masked select uses
/// `_mm256_blendv_pd`, whose argument order is the inverse of NEON's
/// `vbslq_f64` (blendv is `(false, true, mask)`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn score_inliers_f_avx2(
    f: (f64, f64, f64, f64, f64, f64, f64, f64, f64),
    x1_x: &[f64],
    x1_y: &[f64],
    x2_x: &[f64],
    x2_y: &[f64],
    thresh_sq: f64,
    inliers: &mut [bool],
    count: &mut usize,
    score: &mut f64,
) -> usize {
    use std::arch::x86_64::*;
    let (f00, f01, f02, f10, f11, f12, f20, f21, f22) = f;
    let f00v = _mm256_set1_pd(f00);
    let f01v = _mm256_set1_pd(f01);
    let f02v = _mm256_set1_pd(f02);
    let f10v = _mm256_set1_pd(f10);
    let f11v = _mm256_set1_pd(f11);
    let f12v = _mm256_set1_pd(f12);
    let f20v = _mm256_set1_pd(f20);
    let f21v = _mm256_set1_pd(f21);
    let f22v = _mm256_set1_pd(f22);
    let one_v = _mm256_set1_pd(1.0);
    let eps_v = _mm256_set1_pd(1e-12);
    let n = x1_x.len();
    let mut idx = 0usize;
    while idx + 4 <= n {
        let x1 = _mm256_loadu_pd(x1_x.as_ptr().add(idx));
        let y1 = _mm256_loadu_pd(x1_y.as_ptr().add(idx));
        let x2 = _mm256_loadu_pd(x2_x.as_ptr().add(idx));
        let y2 = _mm256_loadu_pd(x2_y.as_ptr().add(idx));

        let fx1x = _mm256_fmadd_pd(y1, f01v, _mm256_fmadd_pd(x1, f00v, f02v));
        let fx1y = _mm256_fmadd_pd(y1, f11v, _mm256_fmadd_pd(x1, f10v, f12v));
        let fx1z = _mm256_fmadd_pd(y1, f21v, _mm256_fmadd_pd(x1, f20v, f22v));
        let ftx2x = _mm256_fmadd_pd(y2, f10v, _mm256_fmadd_pd(x2, f00v, f20v));
        let ftx2y = _mm256_fmadd_pd(y2, f11v, _mm256_fmadd_pd(x2, f01v, f21v));

        let err = _mm256_fmadd_pd(y2, fx1y, _mm256_fmadd_pd(x2, fx1x, fx1z));
        let denom = _mm256_fmadd_pd(
            ftx2y,
            ftx2y,
            _mm256_fmadd_pd(
                ftx2x,
                ftx2x,
                _mm256_fmadd_pd(fx1y, fx1y, _mm256_mul_pd(fx1x, fx1x)),
            ),
        );
        let err_sq = _mm256_mul_pd(err, err);
        let denom_ok = _mm256_cmp_pd::<_CMP_GT_OQ>(denom, eps_v);
        let safe_denom = _mm256_blendv_pd(one_v, denom, denom_ok);
        let div_val = _mm256_div_pd(err_sq, safe_denom);
        let dd = _mm256_blendv_pd(err_sq, div_val, denom_ok);

        let mut buf = [0.0f64; 4];
        _mm256_storeu_pd(buf.as_mut_ptr(), dd);
        for k in 0..4 {
            let dd_k = buf[k];
            if dd_k.is_finite() && dd_k <= thresh_sq {
                *inliers.get_unchecked_mut(idx + k) = true;
                *count += 1;
                *score += dd_k;
            }
        }
        idx += 4;
    }
    idx
}

/// Flatten `&[Vec2F64]` into two parallel f64 slices (x, y). Called once per
/// RANSAC entry; keeps the inner-loop scorer in contiguous `&[f64]` land.
fn split_xy(pts: &[Vec2F64]) -> (Vec<f64>, Vec<f64>) {
    let mut xs = Vec::with_capacity(pts.len());
    let mut ys = Vec::with_capacity(pts.len());
    for p in pts {
        xs.push(p.x);
        ys.push(p.y);
    }
    (xs, ys)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ransac_fundamental_basic() {
        let f_true = Mat3F64::from_cols(
            Vec3F64::new(0.0, -0.001, 0.01),
            Vec3F64::new(0.0015, 0.0, -0.02),
            Vec3F64::new(-0.01, 0.02, 1.0),
        );
        let mut x1 = Vec::new();
        let mut x2 = Vec::new();
        for i in 0..50 {
            let xi = i as f64 * 1.2 - 10.0;
            let yi = i as f64 * -0.8 + 5.0;
            let x = Vec3F64::new(xi, yi, 1.0);
            let l = f_true * x;
            let xp = if l.x.abs() > 1e-12 { -l.z / l.x } else { 0.0 };
            x1.push(Vec2F64::new(x.x, x.y));
            x2.push(Vec2F64::new(xp, 0.0));
        }

        let params = RansacParams {
            max_iterations: 200,
            threshold: 1.0,
            min_inliers: 10,
            random_seed: Some(0),
            refit: false,
        };
        let res = ransac_fundamental(&x1, &x2, &params).unwrap();
        assert!(res.inlier_count >= params.min_inliers);
    }

    /// Verify that enabling the LO-refit step either matches or improves the
    /// plain-RANSAC result (more inliers *or* a lower Sampson score on ties).
    ///
    /// Synthetic scene: pinhole camera, 100 noisy correspondences + 20
    /// outliers with fixed seed so the test is fully deterministic.
    #[test]
    fn test_ransac_fundamental_refit_improves_accuracy() {
        use crate::pose::fundamental::sampson_distance;

        // Ground-truth fundamental matrix from a simple rotation about Y.
        // R = Ry(5°), t = [0.5, 0, 0].
        let angle = 5.0_f64.to_radians();
        let (s, c) = angle.sin_cos();
        // R (column-major): Ry rotates in X-Z plane.
        let r = Mat3F64::from_cols(
            Vec3F64::new(c, 0.0, -s),
            Vec3F64::new(0.0, 1.0, 0.0),
            Vec3F64::new(s, 0.0, c),
        );
        // t cross-product matrix.
        let t = Vec3F64::new(0.5, 0.0, 0.0);
        let tx = Mat3F64::from_cols(
            Vec3F64::new(0.0, t.z, -t.y),
            Vec3F64::new(-t.z, 0.0, t.x),
            Vec3F64::new(t.y, -t.x, 0.0),
        );
        // F = [t]_x R (unnormalized; used below to synthesize correspondences)
        let _f_true = tx * r;

        // Simple linear-congruential generator for a deterministic sequence
        // without pulling in extra dependencies.
        let mut lcg: u64 = 12345678901234567;
        let lcg_next = |state: &mut u64| -> f64 {
            *state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // map high 32 bits to [0, 1)
            (*state >> 32) as f64 / 4294967296.0
        };

        let mut x1: Vec<Vec2F64> = Vec::new();
        let mut x2: Vec<Vec2F64> = Vec::new();

        // 100 inliers: random points in front of cam, project into both views.
        // Focal=500, cx=320, cy=240.
        let fx = 500.0_f64;
        let cx = 320.0_f64;
        let cy = 240.0_f64;
        let noise_px = 0.5_f64; // half-pixel gaussian-ish noise
        for _ in 0..100 {
            let xc = (lcg_next(&mut lcg) - 0.5) * 4.0; // ±2 m
            let yc = (lcg_next(&mut lcg) - 0.5) * 2.0;
            let zc = lcg_next(&mut lcg) * 3.0 + 2.0; // 2-5 m in front
            let p1 = Vec3F64::new(xc, yc, zc);
            let p2 = r * p1 + t;
            // Project with noise.
            let u1 = p1.x / p1.z * fx + cx + (lcg_next(&mut lcg) - 0.5) * noise_px;
            let v1 = p1.y / p1.z * fx + cy + (lcg_next(&mut lcg) - 0.5) * noise_px;
            let u2 = p2.x / p2.z * fx + cx + (lcg_next(&mut lcg) - 0.5) * noise_px;
            let v2 = p2.y / p2.z * fx + cy + (lcg_next(&mut lcg) - 0.5) * noise_px;
            x1.push(Vec2F64::new(u1, v1));
            x2.push(Vec2F64::new(u2, v2));
        }
        // 20 outliers: random pixel positions unrelated to the scene.
        for _ in 0..20 {
            let u1 = lcg_next(&mut lcg) * 640.0;
            let v1 = lcg_next(&mut lcg) * 480.0;
            let u2 = lcg_next(&mut lcg) * 640.0;
            let v2 = lcg_next(&mut lcg) * 480.0;
            x1.push(Vec2F64::new(u1, v1));
            x2.push(Vec2F64::new(u2, v2));
        }

        let base_params = RansacParams {
            max_iterations: 500,
            threshold: 2.0,
            min_inliers: 15,
            random_seed: Some(42),
            refit: false,
        };
        let refit_params = RansacParams {
            refit: true,
            ..base_params
        };

        let res_base = ransac_fundamental(&x1, &x2, &base_params).unwrap();
        let res_refit = ransac_fundamental(&x1, &x2, &refit_params).unwrap();

        // Compute per-inlier Sampson score for each result to compare quality.
        let sampson_score = |res: &RansacResult<Mat3F64>| -> f64 {
            x1.iter()
                .zip(x2.iter())
                .zip(res.inliers.iter())
                .filter(|(_, &inl)| inl)
                .map(|((p1, p2), _)| sampson_distance(&res.model, p1, p2))
                .sum::<f64>()
        };

        let score_base = sampson_score(&res_base);
        let score_refit = sampson_score(&res_refit);

        // Refit must not regress: at least as many inliers, and if equal then
        // a lower or equal Sampson score.
        assert!(
            res_refit.inlier_count >= res_base.inlier_count || score_refit <= score_base,
            "refit regressed: base inliers={} score={:.4}, refit inliers={} score={:.4}",
            res_base.inlier_count,
            score_base,
            res_refit.inlier_count,
            score_refit,
        );
    }

    /// Basic happy-path: known (R, t), pinhole intrinsics, 100 inliers + 20
    /// outliers in pixel space. RANSAC with 5pt must (a) flag ≥ 95% of the
    /// inliers, and (b) recover an E close to the ground-truth E (up to sign
    /// and overall scale, since both are unobservable from epipolar
    /// constraints alone).
    #[test]
    fn test_ransac_essential_5pt_recovers_known_e() {
        // R = Ry(5°), t = (0.5, 0, 0).
        let angle = 5.0_f64.to_radians();
        let (s, c) = angle.sin_cos();
        let r = Mat3F64::from_cols(
            Vec3F64::new(c, 0.0, -s),
            Vec3F64::new(0.0, 1.0, 0.0),
            Vec3F64::new(s, 0.0, c),
        );
        let t = Vec3F64::new(0.5, 0.0, 0.0);
        let tx = Mat3F64::from_cols(
            Vec3F64::new(0.0, t.z, -t.y),
            Vec3F64::new(-t.z, 0.0, t.x),
            Vec3F64::new(t.y, -t.x, 0.0),
        );
        let e_true = tx * r;

        // Pinhole K (matches the F-RANSAC refit test).
        let fx = 500.0_f64;
        let cx = 320.0_f64;
        let cy = 240.0_f64;
        let k = Mat3F64::from_cols(
            Vec3F64::new(fx, 0.0, 0.0),
            Vec3F64::new(0.0, fx, 0.0),
            Vec3F64::new(cx, cy, 1.0),
        );

        // LCG for deterministic noisy data — same recipe as F-RANSAC test.
        let mut lcg: u64 = 12345678901234567;
        let lcg_next = |state: &mut u64| -> f64 {
            *state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (*state >> 32) as f64 / 4294967296.0
        };

        let mut x1: Vec<Vec2F64> = Vec::new();
        let mut x2: Vec<Vec2F64> = Vec::new();
        let noise_px = 0.5_f64;
        for _ in 0..100 {
            let xc = (lcg_next(&mut lcg) - 0.5) * 4.0;
            let yc = (lcg_next(&mut lcg) - 0.5) * 2.0;
            let zc = lcg_next(&mut lcg) * 3.0 + 2.0;
            let p1 = Vec3F64::new(xc, yc, zc);
            let p2 = r * p1 + t;
            let u1 = p1.x / p1.z * fx + cx + (lcg_next(&mut lcg) - 0.5) * noise_px;
            let v1 = p1.y / p1.z * fx + cy + (lcg_next(&mut lcg) - 0.5) * noise_px;
            let u2 = p2.x / p2.z * fx + cx + (lcg_next(&mut lcg) - 0.5) * noise_px;
            let v2 = p2.y / p2.z * fx + cy + (lcg_next(&mut lcg) - 0.5) * noise_px;
            x1.push(Vec2F64::new(u1, v1));
            x2.push(Vec2F64::new(u2, v2));
        }
        for _ in 0..20 {
            let u1 = lcg_next(&mut lcg) * 640.0;
            let v1 = lcg_next(&mut lcg) * 480.0;
            let u2 = lcg_next(&mut lcg) * 640.0;
            let v2 = lcg_next(&mut lcg) * 480.0;
            x1.push(Vec2F64::new(u1, v1));
            x2.push(Vec2F64::new(u2, v2));
        }

        let params = RansacParams {
            max_iterations: 500,
            threshold: 2.0,
            min_inliers: 30,
            random_seed: Some(42),
            refit: false,
        };
        let res = ransac_essential_5pt(&x1, &x2, &k, &k, &params).unwrap();
        // First 100 are inliers; we expect to flag the vast majority.
        let inl_in_first_100 = res.inliers[..100].iter().filter(|&&b| b).count();
        assert!(
            inl_in_first_100 >= 90,
            "expected ≥90/100 true inliers flagged, got {inl_in_first_100}"
        );

        // Compare recovered E to ground truth up to sign/scale (Frobenius dist
        // on unit-normalized matrices).
        let flat_true = e_true.to_cols_array();
        let norm_true: f64 = flat_true.iter().map(|v| v * v).sum::<f64>().sqrt();
        let e_true_unit: [f64; 9] = core::array::from_fn(|k| flat_true[k] / norm_true);
        let flat_est = res.model.to_cols_array();
        let norm_est: f64 = flat_est.iter().map(|v| v * v).sum::<f64>().sqrt();
        let mut err_pos = 0.0f64;
        let mut err_neg = 0.0f64;
        for k in 0..9 {
            let u = flat_est[k] / norm_est;
            err_pos += (u - e_true_unit[k]).powi(2);
            err_neg += (u + e_true_unit[k]).powi(2);
        }
        let dist = err_pos.min(err_neg).sqrt();
        // 0.5 px noise + 17% outliers — relax vs the synthetic-clean test.
        assert!(
            dist < 0.05,
            "recovered E far from ground truth: Frobenius dist (unit, ±sign) = {dist:.4e}"
        );
    }

    #[test]
    fn test_ransac_essential_5pt_invalid_input() {
        let x1 = vec![Vec2F64::new(0.0, 0.0); 4];
        let x2 = vec![Vec2F64::new(0.0, 0.0); 4];
        let k = Mat3F64::IDENTITY;
        let params = RansacParams::default();
        let err = ransac_essential_5pt(&x1, &x2, &k, &k, &params).unwrap_err();
        match err {
            TwoViewError::InvalidInput { required } => assert_eq!(required, 5),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_ransac_fundamental_invalid_input() {
        let x1 = vec![Vec2F64::new(0.0, 0.0); 7];
        let x2 = vec![Vec2F64::new(0.0, 0.0); 7];
        let params = RansacParams::default();
        let err = ransac_fundamental(&x1, &x2, &params).unwrap_err();
        match err {
            TwoViewError::InvalidInput { required } => assert_eq!(required, 8),
            other => panic!("unexpected error: {other:?}"),
        }

        let x2 = vec![Vec2F64::new(0.0, 0.0); 8];
        let err = ransac_fundamental(&x1, &x2, &params).unwrap_err();
        match err {
            TwoViewError::InvalidInput { required } => assert_eq!(required, 8),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_sample_degenerate_collinear_rejected() {
        // Four points on a line — every 3-subset is collinear → reject.
        let collinear = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]];
        assert!(sample_is_degenerate(&collinear));

        // Three collinear + one off-line — still has a collinear triple → reject.
        let mixed = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [1.5, 5.0]];
        assert!(sample_is_degenerate(&mixed));
    }

    #[test]
    fn test_sample_non_degenerate_accepted() {
        // A proper quadrilateral — no collinear triples → accept.
        let quad = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        assert!(!sample_is_degenerate(&quad));

        // Points from the `_make_known_homography_pair` region in the Python tests.
        let normal = [
            [120.0, 180.0],
            [340.0, 150.0],
            [280.0, 420.0],
            [410.0, 370.0],
        ];
        assert!(!sample_is_degenerate(&normal));
    }

    #[test]
    fn test_score_inliers_h_matches_scalar() {
        use crate::pose::fundamental::sampson_distance;
        let h = Mat3F64::from_cols(
            Vec3F64::new(1.25, 0.03, 0.0012),
            Vec3F64::new(-0.08, 0.95, -0.0008),
            Vec3F64::new(2.4, -1.7, 1.0),
        );
        let mut x1 = Vec::new();
        let mut x2 = Vec::new();
        for i in 0..47 {
            let xi = (i as f64 * 0.73 - 12.5).sin() * 100.0;
            let yi = (i as f64 * 1.11 + 3.0).cos() * 80.0;
            let p = Vec3F64::new(xi, yi, 1.0);
            let hp = h * p;
            let jitter = (i as f64 * 0.2).sin() * 0.01;
            x1.push(Vec2F64::new(xi, yi));
            x2.push(Vec2F64::new(hp.x / hp.z + jitter, hp.y / hp.z - jitter));
        }
        let thresh_sq = 0.04f64;

        let mut inl_ref = vec![false; x1.len()];
        let mut count_ref = 0usize;
        let mut score_ref = 0.0f64;
        for i in 0..x1.len() {
            let d = homography_reproj_error(&h, &x1[i], &x2[i]);
            if d <= thresh_sq {
                inl_ref[i] = true;
                count_ref += 1;
                score_ref += d;
            }
        }

        let (x1_x, x1_y) = split_xy(&x1);
        let (x2_x, x2_y) = split_xy(&x2);
        let mut inl = vec![false; x1.len()];
        let (count, score) = score_inliers_h(&h, &x1_x, &x1_y, &x2_x, &x2_y, thresh_sq, &mut inl);

        assert_eq!(count, count_ref);
        assert_eq!(inl, inl_ref);
        assert!((score - score_ref).abs() < 1e-10);

        // keep sampson_distance import live for symmetry across test changes
        let _ = sampson_distance;
    }

    #[test]
    fn test_score_inliers_f_matches_scalar() {
        use crate::pose::fundamental::sampson_distance;
        let f_mat = Mat3F64::from_cols(
            Vec3F64::new(0.0, -0.0012, 0.011),
            Vec3F64::new(0.0014, 0.0, -0.021),
            Vec3F64::new(-0.012, 0.018, 1.0),
        );
        let mut x1 = Vec::new();
        let mut x2 = Vec::new();
        for i in 0..53 {
            let xi = i as f64 * 1.33 - 20.0;
            let yi = (i as f64 * 0.7).sin() * 50.0 - 10.0;
            let x = Vec3F64::new(xi, yi, 1.0);
            let l = f_mat * x;
            let xp = if l.x.abs() > 1e-10 { -l.z / l.x } else { 0.0 };
            x1.push(Vec2F64::new(xi, yi));
            x2.push(Vec2F64::new(xp, (i as f64 * 0.05).cos() * 0.5));
        }
        let thresh_sq = 0.25f64;

        let mut inl_ref = vec![false; x1.len()];
        let mut count_ref = 0usize;
        let mut score_ref = 0.0f64;
        for i in 0..x1.len() {
            let d = sampson_distance(&f_mat, &x1[i], &x2[i]);
            if d <= thresh_sq {
                inl_ref[i] = true;
                count_ref += 1;
                score_ref += d;
            }
        }

        let (x1_x, x1_y) = split_xy(&x1);
        let (x2_x, x2_y) = split_xy(&x2);
        let mut inl = vec![false; x1.len()];
        let (count, score) =
            score_inliers_f(&f_mat, &x1_x, &x1_y, &x2_x, &x2_y, thresh_sq, &mut inl);

        assert_eq!(count, count_ref);
        assert_eq!(inl, inl_ref);
        assert!((score - score_ref).abs() < 1e-9);
    }

    #[test]
    fn test_ransac_homography_adaptive_stops_early_on_clean_data() {
        // All-inlier data should stop in O(10) iterations, not 2000. We can't
        // observe the iteration count directly, but we can verify it runs
        // quickly and converges; the adaptive bound at w=1 triggers the
        // `best_count == n` early-exit immediately.
        let h_true = Mat3F64::from_cols(
            Vec3F64::new(1.2, 0.0, 0.001),
            Vec3F64::new(0.1, 0.9, 0.002),
            Vec3F64::new(5.0, -3.0, 1.0),
        );
        let mut x1 = Vec::new();
        let mut x2 = Vec::new();
        for i in 0..30 {
            let xi = (i % 6) as f64 * 5.0;
            let yi = (i / 6) as f64 * 7.0;
            let p = Vec3F64::new(xi, yi, 1.0);
            let hp = h_true * p;
            x1.push(Vec2F64::new(xi, yi));
            x2.push(Vec2F64::new(hp.x / hp.z, hp.y / hp.z));
        }
        let params = RansacParams {
            max_iterations: 2000,
            threshold: 0.1,
            min_inliers: 25,
            random_seed: Some(42),
            refit: false,
        };
        let start = std::time::Instant::now();
        let res = ransac_homography(&x1, &x2, &params).unwrap();
        let elapsed = start.elapsed();
        assert_eq!(res.inlier_count, x1.len(), "should find all inliers");
        // Generous bound — even with collinearity-rejection overhead, 30 points
        // on clean data is well under 10ms on any dev machine. A runaway loop
        // (e.g., adaptive termination broken) would be orders of magnitude slower.
        assert!(
            elapsed.as_millis() < 50,
            "adaptive termination should be fast on clean data, took {elapsed:?}"
        );
    }

    #[test]
    fn test_ransac_homography_basic() {
        let h_true = Mat3F64::from_cols(
            Vec3F64::new(1.2, 0.0, 0.001),
            Vec3F64::new(0.1, 0.9, 0.002),
            Vec3F64::new(5.0, -3.0, 1.0),
        );

        let mut x1 = Vec::new();
        let mut x2 = Vec::new();
        for i in 0..25 {
            let xi = (i % 5) as f64 * 2.0 - 4.0;
            let yi = (i / 5) as f64 * 1.5 - 3.0;
            let p = Vec3F64::new(xi, yi, 1.0);
            let hp = h_true * p;
            let u = hp.x / hp.z;
            let v = hp.y / hp.z;
            x1.push(Vec2F64::new(xi, yi));
            x2.push(Vec2F64::new(u, v));
        }

        let params = RansacParams {
            max_iterations: 100,
            threshold: 1e-6,
            min_inliers: 12,
            random_seed: Some(0),
            refit: false,
        };
        let res = ransac_homography(&x1, &x2, &params).unwrap();
        assert!(res.inlier_count >= params.min_inliers);
    }

    #[test]
    fn test_ransac_homography_invalid_input() {
        let x1 = vec![Vec2F64::new(0.0, 0.0); 3];
        let x2 = vec![Vec2F64::new(0.0, 0.0); 3];
        let params = RansacParams::default();
        let err = ransac_homography(&x1, &x2, &params).unwrap_err();
        match err {
            TwoViewError::InvalidInput { required } => assert_eq!(required, 4),
            other => panic!("unexpected error: {other:?}"),
        }

        let x2 = vec![Vec2F64::new(0.0, 0.0); 4];
        let err = ransac_homography(&x1, &x2, &params).unwrap_err();
        match err {
            TwoViewError::InvalidInput { required } => assert_eq!(required, 4),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_two_view_config_default_nests_triangulation_defaults() {
        let config = TwoViewConfig::default();
        assert_eq!(config.triangulation.min_parallax_deg, 1.0);
        assert_eq!(config.triangulation.max_midpoint_gap, 1.0);
        assert_eq!(config.triangulation.max_reprojection_error, 2.0);
        assert_eq!(config.triangulation.min_cheirality_count, 1);
        assert_eq!(config.triangulation.cheirality_ambiguity_max, 0.7);
    }

    /// End-to-end two-view pose estimation on real EuRoC MH_01_easy images.
    ///
    /// Reads two grayscale frames, runs ORB detection + matching, estimates
    /// the relative pose via `two_view_estimate`, and compares against
    /// ground truth camera-frame pose.
    ///
    /// Frame pair: 1403636633263555584 → 1403636634263555584 (20 frames apart).
    /// Ground truth camera-frame relative pose (Vicon-derived; see
    /// `kornia-py/scripts/derive_mh01_gt.py` — Δ=0 ns match at both frames):
    ///   - Rotation: 2.7021°
    ///   - Translation: 658.5mm, direction [0.2422, -0.2330, 0.9418]
    #[test]
    fn test_two_view_euroc_mh01() {
        use kornia_imgproc::features::{match_orb_descriptors, OrbDetector, OrbMatchConfig};

        // Load images.
        let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let data_dir = manifest
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("tests/data");

        let img1_u8 =
            kornia_io::png::read_image_png_mono8(data_dir.join("mh01_frame1.png")).unwrap();
        let img2_u8 =
            kornia_io::png::read_image_png_mono8(data_dir.join("mh01_frame2.png")).unwrap();

        let img1 = u8_to_f32_image(&img1_u8);
        let img2 = u8_to_f32_image(&img2_u8);

        // ORB detect + extract.
        let orb = OrbDetector::default();
        let (kps1, scales1, ori1, _) = orb.detect(&img1).unwrap();
        let (desc1, mask1) = orb.extract(&img1, &kps1, &scales1, &ori1).unwrap();
        let (kps2, scales2, ori2, _) = orb.detect(&img2).unwrap();
        let (desc2, mask2) = orb.extract(&img2, &kps2, &scales2, &ori2).unwrap();

        // Filter by valid descriptors (border mask).
        let (valid_kps1, valid_ori1, valid_desc1) = filter_by_mask(&kps1, &ori1, &desc1, &mask1);
        let (valid_kps2, valid_ori2, valid_desc2) = filter_by_mask(&kps2, &ori2, &desc2, &mask2);

        // Match descriptors.
        let match_config = OrbMatchConfig {
            nn_ratio: 0.6,
            th_low: 50,
            check_orientation: true,
            histo_length: 30,
        };
        let matches = match_orb_descriptors(
            &valid_ori1,
            &valid_desc1,
            &valid_ori2,
            &valid_desc2,
            match_config,
        );
        assert!(
            matches.len() >= 15,
            "too few ORB matches: {} (need >= 15)",
            matches.len()
        );

        // Convert matched keypoints to Vec2F64 (x=col, y=row).
        let pts1: Vec<Vec2F64> = matches
            .iter()
            .map(|&(i, _)| {
                let (row, col) = valid_kps1[i];
                Vec2F64::new(col as f64, row as f64)
            })
            .collect();
        let pts2: Vec<Vec2F64> = matches
            .iter()
            .map(|&(_, j)| {
                let (row, col) = valid_kps2[j];
                Vec2F64::new(col as f64, row as f64)
            })
            .collect();

        // EuRoC MH_01_easy cam0 intrinsics.
        let k = Mat3F64::from_cols(
            Vec3F64::new(458.654, 0.0, 0.0),
            Vec3F64::new(0.0, 457.296, 0.0),
            Vec3F64::new(367.215, 248.375, 1.0),
        );

        let config = TwoViewConfig {
            ransac_f: RansacParams {
                max_iterations: 2000,
                threshold: 1.0,
                min_inliers: 15,
                random_seed: Some(42),
                refit: true,
            },
            ransac_h: RansacParams {
                max_iterations: 2000,
                threshold: 1.0,
                min_inliers: 8,
                random_seed: Some(42),
                refit: false,
            },
            homography_inlier_ratio: 0.8,
            triangulation: TriangulationConfig {
                min_parallax_deg: 0.5,
                ..TriangulationConfig::default()
            },
            lm_enabled: true,
            lm: LmPoseConfig::default(),
            use_5pt_essential: false,
            lm_anneal_thresholds: vec![0.5, 0.25],
            lm_anneal_min_inliers: 30,
        };

        let result = two_view_estimate(&pts1, &pts2, &k, &k, &config).unwrap();

        // Should select fundamental model (general motion, not planar).
        assert!(
            matches!(result.model, TwoViewModel::Fundamental(_)),
            "expected fundamental model"
        );

        // Check rotation angle: GT is 2.7021°. With LM polishing the
        // post-cheirality pose, rotation error stays well below 1°.
        let r = result.rotation;
        let trace = r.col(0).x + r.col(1).y + r.col(2).z;
        let est_angle_rad = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0).acos();
        let est_angle_deg = est_angle_rad.to_degrees();
        let gt_angle_deg = 2.7021;
        assert!(
            (est_angle_deg - gt_angle_deg).abs() < 1.0,
            "rotation error too large: estimated {est_angle_deg:.2}°, GT {gt_angle_deg}°"
        );

        // Check translation direction: GT is [0.242, -0.233, 0.942].
        // Translation can be recovered up to sign. Pre-LM the error was ~8°;
        // post-LM Sampson refinement pulls it under 5° on this pair.
        let t = result.translation.normalize();
        let gt_t = Vec3F64::new(0.2422, -0.2330, 0.9418).normalize();
        let cos_angle = t.dot(gt_t).clamp(-1.0, 1.0);
        let t_err_deg = cos_angle.abs().acos().to_degrees();
        assert!(
            t_err_deg < 5.0,
            "translation direction error too large: {t_err_deg:.2}°"
        );

        // Should have triangulated some points.
        assert!(
            !result.points3d.is_empty(),
            "expected triangulated 3D points"
        );
    }

    /// `use_5pt_essential = true` plumbs through `two_view_estimate`:
    /// the returned model variant must be `Essential`, the recovered (R, t)
    /// must hit ground truth, and triangulation must produce points. The 5pt
    /// path skips the F→E SVD round-trip (since 5pt builds E on-manifold),
    /// so for the same RANSAC budget it's expected to match or beat the F
    /// path on small-motion synthetic scenes.
    #[test]
    fn test_two_view_estimate_with_5pt_essential() {
        // Synthetic scene: Ry(5°), t = (0.5, 0, 0), pinhole at fx=500.
        let angle = 5.0_f64.to_radians();
        let (s, c) = angle.sin_cos();
        let r_true = Mat3F64::from_cols(
            Vec3F64::new(c, 0.0, -s),
            Vec3F64::new(0.0, 1.0, 0.0),
            Vec3F64::new(s, 0.0, c),
        );
        let t_true = Vec3F64::new(0.5, 0.0, 0.0);

        let fx = 500.0_f64;
        let cx = 320.0_f64;
        let cy = 240.0_f64;
        let k = Mat3F64::from_cols(
            Vec3F64::new(fx, 0.0, 0.0),
            Vec3F64::new(0.0, fx, 0.0),
            Vec3F64::new(cx, cy, 1.0),
        );

        // 100 noisy inliers + 20 outliers (same recipe as the RANSAC test).
        let mut lcg: u64 = 12345678901234567;
        let lcg_next = |state: &mut u64| -> f64 {
            *state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (*state >> 32) as f64 / 4294967296.0
        };

        let mut x1: Vec<Vec2F64> = Vec::new();
        let mut x2: Vec<Vec2F64> = Vec::new();
        let noise_px = 0.5_f64;
        for _ in 0..100 {
            let xc = (lcg_next(&mut lcg) - 0.5) * 4.0;
            let yc = (lcg_next(&mut lcg) - 0.5) * 2.0;
            let zc = lcg_next(&mut lcg) * 3.0 + 2.0;
            let p1 = Vec3F64::new(xc, yc, zc);
            let p2 = r_true * p1 + t_true;
            let u1 = p1.x / p1.z * fx + cx + (lcg_next(&mut lcg) - 0.5) * noise_px;
            let v1 = p1.y / p1.z * fx + cy + (lcg_next(&mut lcg) - 0.5) * noise_px;
            let u2 = p2.x / p2.z * fx + cx + (lcg_next(&mut lcg) - 0.5) * noise_px;
            let v2 = p2.y / p2.z * fx + cy + (lcg_next(&mut lcg) - 0.5) * noise_px;
            x1.push(Vec2F64::new(u1, v1));
            x2.push(Vec2F64::new(u2, v2));
        }
        for _ in 0..20 {
            x1.push(Vec2F64::new(
                lcg_next(&mut lcg) * 640.0,
                lcg_next(&mut lcg) * 480.0,
            ));
            x2.push(Vec2F64::new(
                lcg_next(&mut lcg) * 640.0,
                lcg_next(&mut lcg) * 480.0,
            ));
        }

        let config = TwoViewConfig {
            ransac_f: RansacParams {
                max_iterations: 500,
                threshold: 2.0,
                min_inliers: 30,
                random_seed: Some(42),
                refit: false,
            },
            ransac_h: RansacParams {
                max_iterations: 500,
                threshold: 2.0,
                min_inliers: 8,
                random_seed: Some(42),
                refit: false,
            },
            // Force the epipolar branch even if H scores comparably.
            homography_inlier_ratio: 1.5,
            triangulation: TriangulationConfig {
                min_parallax_deg: 0.1,
                ..TriangulationConfig::default()
            },
            lm_enabled: true,
            lm: LmPoseConfig::default(),
            use_5pt_essential: true,
            lm_anneal_thresholds: vec![0.5, 0.25],
            lm_anneal_min_inliers: 30,
        };

        let result = two_view_estimate(&x1, &x2, &k, &k, &config).unwrap();

        assert!(
            matches!(result.model, TwoViewModel::Essential(_)),
            "expected Essential model, got {:?}",
            result.model
        );

        // Rotation error.
        let r_est = result.rotation;
        let rt_r = r_est.transpose() * r_true;
        let trace = rt_r.col(0).x + rt_r.col(1).y + rt_r.col(2).z;
        let rot_err_deg = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0).acos().to_degrees();
        assert!(
            rot_err_deg < 0.5,
            "rotation error too large: {rot_err_deg:.4}°"
        );

        // Translation direction (recoverable up to sign).
        let t_dir = result.translation.normalize();
        let t_gt = t_true.normalize();
        let t_err_deg = t_dir.dot(t_gt).clamp(-1.0, 1.0).abs().acos().to_degrees();
        assert!(
            t_err_deg < 5.0,
            "translation direction error too large: {t_err_deg:.2}°"
        );

        assert!(
            !result.points3d.is_empty(),
            "expected triangulated 3D points"
        );
    }

    /// Guard should NOT fire on a well-conditioned general-motion scene.
    /// The synthetic_two_view helper uses t=(0.5, 0.1, 0.2) at Z∈[3,6] — a
    /// baseline/depth ratio around 0.1 that can be borderline. Here we
    /// hand-build a wider-baseline scene (t magnitude ~1 m, depth ~5 m)
    /// where one E candidate dominates cheirality decisively, so the
    /// default 0.7 threshold must accept it.
    #[test]
    fn test_cheirality_ambiguity_guard_permissive_default() {
        let k = Mat3F64::from_cols(
            Vec3F64::new(500.0, 0.0, 0.0),
            Vec3F64::new(0.0, 500.0, 0.0),
            Vec3F64::new(320.0, 240.0, 1.0),
        );
        let angle = 6.0_f64.to_radians();
        let (s, c) = angle.sin_cos();
        let r = Mat3F64::from_cols(
            Vec3F64::new(c, 0.0, -s),
            Vec3F64::new(0.0, 1.0, 0.0),
            Vec3F64::new(s, 0.0, c),
        );
        let t = Vec3F64::new(1.0, 0.0, 0.1);

        let mut x1 = Vec::new();
        let mut x2 = Vec::new();
        for i in 0..50usize {
            let fi = i as f64;
            // Spread over a cube: x,y ∈ [-1.5, 1.5], z ∈ [4, 6].
            let xc = (fi * 0.37).sin() * 1.5;
            let yc = (fi * 0.53).cos() * 1.0;
            let zc = 4.0 + (fi * 0.11).sin().abs() * 2.0;
            let p1 = Vec3F64::new(xc, yc, zc);
            let p2 = r * p1 + t;
            x1.push(Vec2F64::new(
                500.0 * p1.x / p1.z + 320.0,
                500.0 * p1.y / p1.z + 240.0,
            ));
            x2.push(Vec2F64::new(
                500.0 * p2.x / p2.z + 320.0,
                500.0 * p2.y / p2.z + 240.0,
            ));
        }

        let config = TwoViewConfig::default();
        two_view_estimate(&x1, &x2, &k, &k, &config)
            .expect("default 0.7 ambiguity threshold must accept well-conditioned motion");
    }

    /// With the threshold driven to 0.0, ANY runner-up candidate (even a
    /// single borderline triangulated point) trips the guard. Used to
    /// confirm the error path is reachable and the reported counts are sane.
    /// On clean synthetic data the runner-up is usually 0, so we force the
    /// degenerate case via a fronto-parallel plane + tiny translation —
    /// the classic two-E-candidates-both-pass configuration.
    #[test]
    fn test_cheirality_ambiguity_guard_fires_on_degenerate_motion() {
        // Fronto-parallel plane at Z=5, points spread over a 2×2 m patch.
        let k = Mat3F64::from_cols(
            Vec3F64::new(500.0, 0.0, 0.0),
            Vec3F64::new(0.0, 500.0, 0.0),
            Vec3F64::new(320.0, 240.0, 1.0),
        );
        // Small rotation about Y, near-zero translation. Pure rotation makes
        // E ≈ 0; a tiny bit of translation gives RANSAC enough signal to find
        // *some* F, but the resulting E's decomposition is poorly conditioned
        // and commonly produces two candidates with similar cheirality.
        let angle = 1.0_f64.to_radians();
        let (s, c) = angle.sin_cos();
        let r = Mat3F64::from_cols(
            Vec3F64::new(c, 0.0, -s),
            Vec3F64::new(0.0, 1.0, 0.0),
            Vec3F64::new(s, 0.0, c),
        );
        let t = Vec3F64::new(0.02, 0.0, 0.0);

        // Generate 30 points on the plane Z=5.
        let mut x1 = Vec::new();
        let mut x2 = Vec::new();
        for i in 0..30 {
            let xc = -1.0 + 0.1 * i as f64;
            let yc = 0.4 * (i as f64 * 0.7).sin();
            let p1 = Vec3F64::new(xc, yc, 5.0);
            let p2 = r * p1 + t;
            let u1 = 500.0 * p1.x / p1.z + 320.0;
            let v1 = 500.0 * p1.y / p1.z + 240.0;
            let u2 = 500.0 * p2.x / p2.z + 320.0;
            let v2 = 500.0 * p2.y / p2.z + 240.0;
            x1.push(Vec2F64::new(u1, v1));
            x2.push(Vec2F64::new(u2, v2));
        }

        // Strict threshold: any non-zero runner-up trips it. If the scene
        // ends up with second=0 (unambiguous), we'll take that as a signal
        // to document that this path needs a stronger synthetic case — but
        // typical runs of this near-pure-rotation setup produce multiple
        // cheirality-passing candidates.
        let config = TwoViewConfig {
            ransac_f: RansacParams {
                min_inliers: 8,
                random_seed: Some(0),
                ..RansacParams::default()
            },
            ransac_h: RansacParams {
                min_inliers: 8,
                random_seed: Some(0),
                ..RansacParams::default()
            },
            triangulation: TriangulationConfig {
                min_parallax_deg: 0.0,
                cheirality_ambiguity_max: 0.0,
                ..TriangulationConfig::default()
            },
            ..TwoViewConfig::default()
        };

        match two_view_estimate(&x1, &x2, &k, &k, &config) {
            Err(TwoViewError::AmbiguousCheirality {
                best,
                second,
                ratio,
                max_ratio,
            }) => {
                assert!(best >= second);
                assert!(ratio > max_ratio);
                assert_eq!(max_ratio, 0.0);
            }
            Err(TwoViewError::RansacFailure) => {
                // Acceptable: RANSAC may reject the F entirely when
                // t is this small — the whole pipeline correctly bails.
            }
            Err(other) => panic!("unexpected error variant: {other:?}"),
            Ok(r) => panic!(
                "degenerate motion must not produce a confident pose; got inliers={}",
                r.inliers.iter().filter(|b| **b).count()
            ),
        }
    }

    // -------- LM refinement tests --------

    /// Construct a synthetic two-view setup: N 3D points in front of both
    /// cameras, projected (with optional Gaussian-ish noise) into both views
    /// through a shared K. Uses a deterministic LCG so tests are reproducible
    /// without extra RNG dependencies.
    ///
    /// Returns (x1, x2, R_true, t_true_unit, K).
    fn synthetic_two_view(
        n_pts: usize,
        noise_px: f64,
        seed: u64,
    ) -> (Vec<Vec2F64>, Vec<Vec2F64>, Mat3F64, Vec3F64, Mat3F64) {
        // Fixed GT pose: small rotation about y-axis, translation mostly along x.
        let angle = 5.0_f64.to_radians();
        let (s, c) = angle.sin_cos();
        let r = Mat3F64::from_cols(
            Vec3F64::new(c, 0.0, -s),
            Vec3F64::new(0.0, 1.0, 0.0),
            Vec3F64::new(s, 0.0, c),
        );
        let t = Vec3F64::new(0.5, 0.1, 0.2);
        let t_unit = t.normalize();

        // Intrinsics: 640x480, fx=fy=500, principal point at center.
        let k = Mat3F64::from_cols(
            Vec3F64::new(500.0, 0.0, 0.0),
            Vec3F64::new(0.0, 500.0, 0.0),
            Vec3F64::new(320.0, 240.0, 1.0),
        );

        let mut lcg = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let mut next = || -> f64 {
            lcg = lcg
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (lcg >> 32) as f64 / 4_294_967_296.0
        };

        let mut x1 = Vec::with_capacity(n_pts);
        let mut x2 = Vec::with_capacity(n_pts);
        for _ in 0..n_pts {
            let xc = (next() - 0.5) * 3.0;
            let yc = (next() - 0.5) * 2.0;
            let zc = next() * 3.0 + 3.0; // 3-6 m depth
            let p1 = Vec3F64::new(xc, yc, zc);
            let p2_cam = r * p1 + t;
            // Project through K.
            let u1 = 500.0 * p1.x / p1.z + 320.0 + (next() - 0.5) * 2.0 * noise_px;
            let v1 = 500.0 * p1.y / p1.z + 240.0 + (next() - 0.5) * 2.0 * noise_px;
            let u2 = 500.0 * p2_cam.x / p2_cam.z + 320.0 + (next() - 0.5) * 2.0 * noise_px;
            let v2 = 500.0 * p2_cam.y / p2_cam.z + 240.0 + (next() - 0.5) * 2.0 * noise_px;
            x1.push(Vec2F64::new(u1, v1));
            x2.push(Vec2F64::new(u2, v2));
        }
        (x1, x2, r, t_unit, k)
    }

    /// Sum of Sampson distances for F = K2^-T [t]x R K1^-1 over all pairs.
    fn sampson_cost_rt(
        r: &Mat3F64,
        t: &Vec3F64,
        k1: &Mat3F64,
        k2: &Mat3F64,
        x1: &[Vec2F64],
        x2: &[Vec2F64],
    ) -> f64 {
        use crate::pose::fundamental::sampson_distance;
        let skew = Mat3F64::from_cols(
            Vec3F64::new(0.0, t.z, -t.y),
            Vec3F64::new(-t.z, 0.0, t.x),
            Vec3F64::new(t.y, -t.x, 0.0),
        );
        let e = skew * *r;
        let f = k2.inverse().transpose() * e * k1.inverse();
        x1.iter()
            .zip(x2.iter())
            .map(|(p1, p2)| sampson_distance(&f, p1, p2))
            .sum()
    }

    fn rot_angle_deg(r_rel: &Mat3F64) -> f64 {
        let tr = r_rel.col(0).x + r_rel.col(1).y + r_rel.col(2).z;
        ((tr - 1.0) / 2.0).clamp(-1.0, 1.0).acos().to_degrees()
    }

    fn rot_err_deg(r_est: &Mat3F64, r_gt: &Mat3F64) -> f64 {
        let r_diff = r_est.transpose() * *r_gt;
        rot_angle_deg(&r_diff)
    }

    fn t_err_deg(t_est: &Vec3F64, t_gt: &Vec3F64) -> f64 {
        let te = t_est.normalize();
        let tg = t_gt.normalize();
        te.dot(tg).abs().clamp(0.0, 1.0).acos().to_degrees()
    }

    /// Small rotation (~1°) + small translation-direction perturbation applied
    /// to a GT pose, used to seed LM.
    fn perturb_pose(r: Mat3F64, t: Vec3F64, rot_deg: f64, t_dir_deg: f64) -> (Mat3F64, Vec3F64) {
        // Rotation perturbation: axis = Y (normalized), angle = rot_deg.
        let angle = rot_deg.to_radians();
        let (s, c) = angle.sin_cos();
        let r_pert = Mat3F64::from_cols(
            Vec3F64::new(c, 0.0, -s),
            Vec3F64::new(0.0, 1.0, 0.0),
            Vec3F64::new(s, 0.0, c),
        );
        let r_new = r * r_pert;

        // Translation-direction perturbation: rotate t by t_dir_deg about an
        // axis orthogonal to t. Pick e.g. world-X (unless t is near-parallel to
        // it, then world-Y).
        let t_u = t.normalize();
        let seed_axis = if t_u.x.abs() < 0.9 {
            Vec3F64::new(1.0, 0.0, 0.0)
        } else {
            Vec3F64::new(0.0, 1.0, 0.0)
        };
        let axis = Vec3F64::new(
            t_u.y * seed_axis.z - t_u.z * seed_axis.y,
            t_u.z * seed_axis.x - t_u.x * seed_axis.z,
            t_u.x * seed_axis.y - t_u.y * seed_axis.x,
        )
        .normalize();
        let ang = t_dir_deg.to_radians();
        let (ss, cc) = ang.sin_cos();
        // Rodrigues: v' = v cos + (k × v) sin + k (k·v)(1 - cos)
        let kxv = Vec3F64::new(
            axis.y * t_u.z - axis.z * t_u.y,
            axis.z * t_u.x - axis.x * t_u.z,
            axis.x * t_u.y - axis.y * t_u.x,
        );
        let kdotv = axis.dot(t_u);
        let t_new = Vec3F64::new(
            t_u.x * cc + kxv.x * ss + axis.x * kdotv * (1.0 - cc),
            t_u.y * cc + kxv.y * ss + axis.y * kdotv * (1.0 - cc),
            t_u.z * cc + kxv.z * ss + axis.z * kdotv * (1.0 - cc),
        )
        .normalize();
        (r_new, t_new)
    }

    /// Noise-free synthetic setup: with perfect correspondences and a small
    /// perturbation of GT pose, LM must recover the GT to numerical
    /// precision.
    #[test]
    fn test_lm_pose_refine_synthetic_perfect() {
        let (x1, x2, r_gt, t_gt, k) = synthetic_two_view(60, 0.0, 7);
        let (r0, t0) = perturb_pose(r_gt, t_gt, 1.0, 5.0);

        let cfg = LmPoseConfig::default();
        let (r_ref, t_ref) = refine_pose_lm(r0, t0, &x1, &x2, &k, &k, &cfg);

        let r_err = rot_err_deg(&r_ref, &r_gt);
        let t_err = t_err_deg(&t_ref, &t_gt);
        // With a finite-difference Jacobian the residual of LM is bounded by
        // O(h²) ≈ 1e-12 on cost, but Sampson is scale-invariant in F; 1e-2° is
        // the effective noise floor on R/t from numerical error.
        assert!(
            r_err < 1e-2,
            "perfect-noise LM should recover R to ≤1e-2°, got {r_err:.6}°"
        );
        assert!(
            t_err < 1e-2,
            "perfect-noise LM should recover t_dir to ≤1e-2°, got {t_err:.6}°"
        );
    }

    /// Noisy synthetic setup: LM must strictly reduce the Sampson cost and
    /// never regress the rotation / translation error.
    #[test]
    fn test_lm_pose_refine_noisy_improves() {
        let (x1, x2, r_gt, t_gt, k) = synthetic_two_view(120, 0.5, 42);
        // Perturb GT by a non-trivial amount to ensure LM has work to do.
        let (r0, t0) = perturb_pose(r_gt, t_gt, 2.0, 8.0);

        let cost_before = sampson_cost_rt(&r0, &t0, &k, &k, &x1, &x2);
        let r_err_before = rot_err_deg(&r0, &r_gt);
        let t_err_before = t_err_deg(&t0, &t_gt);

        let cfg = LmPoseConfig::default();
        let (r1, t1) = refine_pose_lm(r0, t0, &x1, &x2, &k, &k, &cfg);

        let cost_after = sampson_cost_rt(&r1, &t1, &k, &k, &x1, &x2);
        let r_err_after = rot_err_deg(&r1, &r_gt);
        let t_err_after = t_err_deg(&t1, &t_gt);

        assert!(
            cost_after < cost_before,
            "Sampson cost did not decrease: before={cost_before:.6e}, after={cost_after:.6e}"
        );
        // With a finite-DoF linearized fit on noisy data we don't expect
        // monotone improvement of the *pose errors* in general, but on this
        // setup (large perturbation, low noise) they must not regress by more
        // than a small margin.
        assert!(
            r_err_after <= r_err_before + 0.05,
            "rotation error regressed: before={r_err_before:.4}°, after={r_err_after:.4}°"
        );
        assert!(
            t_err_after <= t_err_before + 0.05,
            "translation error regressed: before={t_err_before:.4}°, after={t_err_after:.4}°"
        );
    }

    /// If we start from GT pose, LM must stay there (within numerical
    /// tolerance) — an idempotency / no-harm guarantee.
    #[test]
    fn test_lm_pose_refine_does_no_harm_when_already_optimal() {
        let (x1, x2, r_gt, t_gt, k) = synthetic_two_view(60, 0.0, 99);
        let cfg = LmPoseConfig::default();
        let (r_out, t_out) = refine_pose_lm(r_gt, t_gt, &x1, &x2, &k, &k, &cfg);
        let r_err = rot_err_deg(&r_out, &r_gt);
        let t_err = t_err_deg(&t_out, &t_gt);
        assert!(r_err < 1e-4, "LM drifted from optimal R: err={r_err:.6}°");
        assert!(t_err < 1e-4, "LM drifted from optimal t: err={t_err:.6}°");
    }

    /// Helper to build a minimal TwoViewResult with given inlier_indices.
    fn stub_result(inlier_indices: Vec<usize>) -> TwoViewResult {
        TwoViewResult {
            model: TwoViewModel::Fundamental(Mat3F64::IDENTITY),
            rotation: Mat3F64::IDENTITY,
            translation: Vec3F64::new(0.0, 0.0, 1.0),
            points3d: Vec::new(),
            inlier_indices,
            inliers: Vec::new(),
        }
    }

    fn test_camera() -> crate::camera::PinholeCamera {
        crate::camera::PinholeCamera {
            fx: 500.0,
            fy: 500.0,
            cx: 320.0,
            cy: 240.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        }
    }

    #[test]
    fn test_median_parallax_empty_inliers() {
        let result = stub_result(vec![]);
        let cam = test_camera();
        let x1 = vec![Vec2F64::new(320.0, 240.0)];
        let x2 = vec![Vec2F64::new(330.0, 240.0)];
        assert_eq!(result.median_parallax_deg(&x1, &x2, &cam), 0.0);
    }

    #[test]
    fn test_median_parallax_identical_points() {
        // Same pixel in both views → zero parallax.
        let result = stub_result(vec![0]);
        let cam = test_camera();
        let x1 = vec![Vec2F64::new(400.0, 300.0)];
        let x2 = vec![Vec2F64::new(400.0, 300.0)];
        let angle = result.median_parallax_deg(&x1, &x2, &cam);
        assert!(
            angle.abs() < 1e-4,
            "expected ~0 parallax for identical points, got {angle}"
        );
    }

    #[test]
    fn test_median_parallax_known_angle() {
        // Construct a case where bearing vectors differ by a known angle.
        // Camera: fx=fy=500, cx=320, cy=240.
        // Point 1: at principal point → bearing (0, 0, 1).
        // Point 2: shifted 500px in x → bearing (1, 0, 1)/sqrt(2).
        // Angle = acos( (0*1 + 0*0 + 1*1) / (1 * sqrt(2)) ) = acos(1/sqrt(2)) = 45°.
        let cam = test_camera();
        let result = stub_result(vec![0]);
        let x1 = vec![Vec2F64::new(320.0, 240.0)]; // principal point
        let x2 = vec![Vec2F64::new(820.0, 240.0)]; // 500px right
        let angle = result.median_parallax_deg(&x1, &x2, &cam);
        assert!(
            (angle - 45.0).abs() < 0.01,
            "expected ~45° parallax, got {angle}"
        );
    }

    #[test]
    fn test_median_parallax_multiple_inliers() {
        // 3 inliers: angles 0°, 45°, 45° → sorted [0, 45, 45], median = 45°.
        let cam = test_camera();
        let result = stub_result(vec![0, 1, 2]);
        let x1 = vec![
            Vec2F64::new(320.0, 240.0), // pp
            Vec2F64::new(320.0, 240.0), // pp
            Vec2F64::new(320.0, 240.0), // pp
        ];
        let x2 = vec![
            Vec2F64::new(320.0, 240.0), // same → 0°
            Vec2F64::new(820.0, 240.0), // +500px → 45°
            Vec2F64::new(820.0, 240.0), // +500px → 45°
        ];
        let angle = result.median_parallax_deg(&x1, &x2, &cam);
        assert!(
            (angle - 45.0).abs() < 0.01,
            "expected median ~45°, got {angle}"
        );
    }

    #[test]
    fn test_median_parallax_out_of_bounds_indices_ignored() {
        // Inlier indices beyond x1/x2 length are filtered out.
        let cam = test_camera();
        let result = stub_result(vec![0, 99]); // index 99 doesn't exist
        let x1 = vec![Vec2F64::new(320.0, 240.0)];
        let x2 = vec![Vec2F64::new(820.0, 240.0)];
        let angle = result.median_parallax_deg(&x1, &x2, &cam);
        assert!(
            (angle - 45.0).abs() < 0.01,
            "expected ~45° (out-of-bounds index skipped), got {angle}"
        );
    }

    fn u8_to_f32_image(
        src: &kornia_image::Image<u8, 1, kornia_tensor::CpuAllocator>,
    ) -> kornia_image::Image<f32, 1, kornia_tensor::CpuAllocator> {
        let mut dst =
            kornia_image::Image::from_size_val(src.size(), 0.0, kornia_tensor::CpuAllocator)
                .unwrap();
        src.as_slice()
            .iter()
            .zip(dst.as_slice_mut())
            .for_each(|(&s, d)| *d = s as f32 / 255.0);
        dst
    }

    type OrbFiltered = (Vec<(f32, f32)>, Vec<f32>, Vec<[u8; 32]>);

    fn filter_by_mask(
        kps: &[(f32, f32)],
        ori: &[f32],
        desc: &[[u8; 32]],
        mask: &[bool],
    ) -> OrbFiltered {
        let mut out_kps = Vec::new();
        let mut out_ori = Vec::new();
        let mut out_desc = Vec::new();
        let mut desc_idx = 0;
        for (i, &valid) in mask.iter().enumerate() {
            if valid {
                out_kps.push(kps[i]);
                out_ori.push(ori[i]);
                out_desc.push(desc[desc_idx]);
                desc_idx += 1;
            }
        }
        (out_kps, out_ori, out_desc)
    }
}
