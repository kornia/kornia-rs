//! Robust two-view model estimation: the RANSAC entry points for the
//! fundamental, essential and homography models.

use super::kernels::{
    score_inliers_f, score_inliers_f_bounded_masked, score_inliers_f_upper_bound, score_inliers_h,
    split_xy,
};
use super::local_optimization::{degensac_recover_fundamental, lo_plus_fundamental};
use super::{RansacParams, RansacResult, TwoViewError};
use crate::pose::fundamental::fundamental_8point;
use crate::pose::fundamental_7pt::fundamental_7point_oriented_into;
use crate::pose::{essential_5pt, homography_4pt2d, homography_dlt};
use crate::ransac::{adaptive_max_iters, sample_distinct_indices};
use kornia_algebra::{Mat3F64, Vec2F64, Vec3F64};
use rand::prelude::*;
use rand::SeedableRng;
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

#[inline]
pub(super) fn adaptive_ransac_cap(
    inlier_count: usize,
    population: usize,
    sample_size: usize,
    confidence: f64,
    current_cap: usize,
    completed: usize,
) -> usize {
    adaptive_max_iters(
        inlier_count,
        population,
        sample_size,
        confidence,
        current_cap,
    )
    .max(completed)
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

    // Confidence applies to minimal samples, rather than individual roots.
    let mut dynamic_max = params.max_iterations;
    let mut iter = 0usize;
    while iter < dynamic_max {
        iter += 1;
        let mut sample = [0usize; SAMPLE_SIZE];
        sample_distinct_indices(&mut rng, n, &mut sample);
        let mut s1 = [Vec2F64::ZERO; SAMPLE_SIZE];
        let mut s2 = [Vec2F64::ZERO; SAMPLE_SIZE];
        for (i, &idx) in sample.iter().enumerate() {
            s1[i] = x1[idx];
            s2[i] = x2[idx];
        }
        let mut seven_models = [Mat3F64::ZERO; 3];
        let eight_model;
        let models: &[Mat3F64] = if SAMPLE_SIZE == 7 {
            let count = match fundamental_7point_oriented_into(&s1, &s2, &mut seven_models) {
                Ok(count) => count,
                Err(_) => continue,
            };
            &seven_models[..count]
        } else {
            match fundamental_8point(&s1, &s2) {
                Ok(f) => eight_model = [f],
                Err(_) => continue,
            }
            &eight_model
        };

        for &f in models {
            // A 7-point sample may yield up to three hypotheses, and most of
            // them cannot win. A division-free pass bounds each root's
            // support from above; roots that cannot even tie the incumbent
            // are dropped before any exact scoring or mask traffic. The
            // 8-point path deliberately keeps its original full scorer.
            if SAMPLE_SIZE == 7
                && best_count > 0
                && score_inliers_f_upper_bound(
                    &f,
                    &x1_x,
                    &x1_y,
                    &x2_x,
                    &x2_y,
                    thresh_sq,
                    best_count - 1,
                )
                .is_none()
            {
                continue;
            }
            // Survivors usually win or tie, so score them exactly in fixed,
            // SIMD-aligned chunks while materializing the mask, stopping only
            // when the exact bound cannot beat the current winner.
            scratch_inliers.fill(false);
            let scored = if SAMPLE_SIZE == 7 {
                score_inliers_f_bounded_masked(
                    &f,
                    &x1_x,
                    &x1_y,
                    &x2_x,
                    &x2_y,
                    thresh_sq,
                    &mut scratch_inliers,
                    best_count,
                    best_score,
                )
            } else {
                Some(score_inliers_f(
                    &f,
                    &x1_x,
                    &x1_y,
                    &x2_x,
                    &x2_y,
                    thresh_sq,
                    &mut scratch_inliers,
                ))
            };
            let Some((count, score)) = scored else {
                continue;
            };

            let improved = count > best_count || (count == best_count && score < best_score);
            if improved {
                best_model = Some(f);
                std::mem::swap(&mut best_inliers, &mut scratch_inliers);
                best_count = count;
                best_score = score;

                if best_count == n {
                    // Finish scoring this sample's roots before stopping.
                    dynamic_max = iter;
                    continue;
                }
                dynamic_max =
                    adaptive_ransac_cap(best_count, n, SAMPLE_SIZE, confidence, dynamic_max, iter);
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

    // Adaptive iteration cap. The default is 0.9999, matching fundamental
    // estimation; callers can override it through `RansacParams::confidence`.
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
            dynamic_max =
                adaptive_ransac_cap(best_count, n, 5, confidence, params.max_iterations, iter);
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
            dynamic_max =
                adaptive_ransac_cap(best_count, n, 4, confidence, params.max_iterations, iter);
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
