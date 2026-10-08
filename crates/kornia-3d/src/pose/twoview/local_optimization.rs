//! Fundamental local optimization and dominant-plane recovery.

use super::scoring::{sample_is_degenerate, ScoringPoints};
use crate::pose::{fundamental_8point, homography_4pt2d};
use kornia_algebra::{Mat3F64, Vec2F64, Vec3F64};
use rand::rngs::StdRng;

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
pub(super) fn lo_plus_fundamental(
    x1: &[Vec2F64],
    x2: &[Vec2F64],
    points: &ScoringPoints<'_>,
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
                let (vc, _) = points.fundamental(&model, virt_t_sq, &mut scratch_inliers);
                fit_mask.copy_from_slice(&scratch_inliers);
                vc
            } else {
                // mult=1.0: re-score on the running best at the base threshold
                // — the inlier set has drifted as `model` evolved through
                // earlier multipliers, so use the up-to-date support, not the
                // stale cached `best_inliers` mask.
                let (vc, _) = points.fundamental(&model, base_thresh_sq, &mut fit_mask);
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
            let (count_b, score_b) =
                points.fundamental(&f_refit, base_thresh_sq, &mut base_inliers);
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
/// degeneracy gate. On general non-planar scenes the H-support never reaches
/// the 75% threshold and the routine early-exits.
#[allow(clippy::too_many_arguments)]
pub(super) fn degensac_recover_fundamental(
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

    let inlier_points = ScoringPoints::new(&inl_x1_x, &inl_x1_y, &inl_x2_x, &inl_x2_y);
    let points = ScoringPoints::new(x1_x, x1_y, x2_x, x2_y);

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
        let (cnt, _) = inlier_points.homography(&h, base_thresh_sq, &mut h_inl);
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
            let (cnt, scr) = points.fundamental(&f_new, base_thresh_sq, &mut new_inl);
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
