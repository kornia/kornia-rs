//! Post-sampling refinement of a fundamental-matrix winner: LO+ threshold
//! annealing and DEGENSAC dominant-plane recovery.

use super::kernels::score_inliers_h;
use super::ransac::{sample_is_degenerate, FundamentalProblem};
use crate::pose::fundamental::fundamental_8point;
use crate::pose::homography_4pt2d;
use crate::ransac::engine::{CountThenResidual, Scorer, StageContext, Termination, Ties};
use kornia_algebra::{Mat3F64, Vec2F64, Vec3F64};
use rand::Rng;

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
/// Every refit is a derived candidate: scored exhaustively at the base
/// threshold and offered to the incumbent under [`Ties::Strict`], so the
/// result is never worse than the input.
pub(super) fn lo_plus_fundamental<'p, const SAMPLE_SIZE: usize, Sc, T, R>(
    stage: &mut StageContext<'_, FundamentalProblem<'p, SAMPLE_SIZE>, Sc, T, R>,
) where
    Sc: Scorer<FundamentalProblem<'p, SAMPLE_SIZE>, Objective = CountThenResidual>,
    T: Termination,
{
    const LO_MULTIPLIERS: [f64; 6] = [4.0, 3.4, 2.8, 2.2, 1.6, 1.0];
    const LO_OUTER_ITERS: usize = 3;
    let Some(mut model) = stage.best.model else {
        return;
    };
    if stage.best.score.support < 8 {
        return;
    }
    let problem = stage.problem;
    let points = &problem.points;
    let base_threshold = problem.threshold;
    let n = points.len();
    let mut fit_mask = vec![false; n];
    let mut inl_x1: Vec<Vec2F64> = Vec::with_capacity(n);
    let mut inl_x2: Vec<Vec2F64> = Vec::with_capacity(n);
    for outer in 0..LO_OUTER_ITERS {
        let prev = stage.best.score.objective;
        for &mult in &LO_MULTIPLIERS {
            // Recompute the fit mask from the *current best* model at this
            // multiplier so each step picks up correspondences matching the
            // polished F, not a previous step's F. The last multiplier, 1.0,
            // re-scores the running best at the base threshold: the inlier
            // set has drifted as `model` evolved, so the cached incumbent
            // mask would be stale. (`threshold * 1.0` squares to exactly the
            // base threshold squared.)
            let virt_t = base_threshold * mult;
            fit_mask.fill(false);
            let (virt_count, _) = points.sampson(&model, virt_t * virt_t, &mut fit_mask);
            if virt_count < 8 {
                break;
            }

            inl_x1.clear();
            inl_x2.clear();
            for (i, &keep) in fit_mask.iter().enumerate() {
                if keep {
                    inl_x1.push(points.x1[i]);
                    inl_x2.push(points.x2[i]);
                }
            }
            let Ok(f_refit) = fundamental_8point(&inl_x1, &inl_x2) else {
                break;
            };

            // Strict-improve at base threshold — never let a polished F drift
            // onto a dominant-plane local optimum that scores well on count
            // but produces a wrong (R, t) downstream.
            if stage.consider_derived(&f_refit, Ties::Strict) {
                model = f_refit;
            }
        }
        // Convergence: if this outer pass produced no improvement, further
        // passes will only repeat the same fixed-point. Save the work.
        let current = stage.best.score.objective;
        if outer + 1 < LO_OUTER_ITERS
            && current.count == prev.count
            && current.residual >= prev.residual
        {
            break;
        }
    }
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
/// degeneracy gate. On general non-planar scenes the H-support never reaches
/// the 75% threshold and the routine early-exits.
///
/// Each lifted `F_new` is a derived candidate: scored exhaustively at the
/// base threshold and offered to the incumbent under [`Ties::Strict`]. The
/// run's RNG drives the homography samples.
pub(super) fn degensac_recover_fundamental<'p, const SAMPLE_SIZE: usize, Sc, T, R>(
    stage: &mut StageContext<'_, FundamentalProblem<'p, SAMPLE_SIZE>, Sc, T, R>,
) where
    Sc: Scorer<FundamentalProblem<'p, SAMPLE_SIZE>, Objective = CountThenResidual>,
    T: Termination,
    R: Rng,
{
    // Below ~12 inliers the plane-vs-non-plane signal is too noisy to act on
    // and the lift is unstable.
    let in_count = stage.best.score.support;
    if in_count < 12 {
        return;
    }

    // SoA companion over the F-inlier subset for the NEON H scorer.
    let mut inl_x1_x: Vec<f64> = Vec::with_capacity(in_count);
    let mut inl_x1_y: Vec<f64> = Vec::with_capacity(in_count);
    let mut inl_x2_x: Vec<f64> = Vec::with_capacity(in_count);
    let mut inl_x2_y: Vec<f64> = Vec::with_capacity(in_count);
    let problem = stage.problem;
    let base_thresh_sq = problem.thresh_sq;
    let points = &problem.points;
    for (i, &b) in stage.best.inliers.iter().enumerate() {
        if b {
            inl_x1_x.push(points.x1[i].x);
            inl_x1_y.push(points.x1[i].y);
            inl_x2_x.push(points.x2[i].x);
            inl_x2_y.push(points.x2[i].y);
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
        let s = rand::seq::index::sample(&mut *stage.rng, in_count, 4);
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
        _ => return,
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
        return;
    }

    // Try off-plane pairs; keep the F_new with the highest base-threshold inlier
    // count (strict-improve gate, same shape as LO+). Cap trials so DEGENSAC
    // stays sub-millisecond on heavily-planar scenes.
    const PAIR_TRIALS: usize = 30;
    let pair_count = off.len() * (off.len() - 1) / 2;
    let trial_cap = PAIR_TRIALS.min(pair_count);
    let mut tried = 0usize;
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
            stage.consider_derived(&f_new, Ties::Strict);
        }
    }
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
