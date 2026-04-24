//! # Two-View Initialization
//!
//! Recovers relative camera pose (R, t) and 3D structure from 2D correspondences.
//!
//! ## Pipeline
//!
//! ```text
//! Correspondences (pixel space)
//!     │
//!     ├─→ RANSAC + 8-point → F (fundamental, pixel space)
//!     │                         │
//!     │                         ▼
//!     │                       E = K2ᵀ F K1 (essential, metric space)
//!     │                         │
//!     │                         ▼ enforce (σ,σ,0), decompose → 4 (R,t) candidates
//!     │
//!     └─→ RANSAC + 4-point → H (homography, pixel space)
//!                               │
//!                               ▼ decompose → multiple (R,t,n) candidates
//!     │
//!     ▼
//! Model selection: compare inlier ratios (H wins when scene is planar)
//!     │
//!     ▼
//! Cheirality check: triangulate with each candidate, pick the one where
//! points are in front of both cameras
//!     │
//!     ▼
//! (R, t_direction, 3D points)   ← translation scale is lost (quotient of SE(3))
//! ```
//!
//! The output translation is a **unit vector** (direction only). Scale is irrecoverable
//! from two views alone — this is the SE(3) → essential manifold quotient in action.

use crate::pose::fundamental::{fundamental_8point, FundamentalError};
use crate::pose::triangulation::{triangulate_inliers, TriangulateParams, TriangulationConfig};
use crate::pose::{
    decompose_essential, decompose_homography, enforce_essential_constraints,
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
    /// Fundamental matrix model.
    Fundamental(Mat3F64),
    /// Homography model.
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
}

impl Default for TwoViewConfig {
    fn default() -> Self {
        Self {
            ransac_f: RansacParams::default(),
            ransac_h: RansacParams::default(),
            homography_inlier_ratio: 0.8,
            triangulation: TriangulationConfig::default(),
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
    let mut best_model = None;
    let mut best_inliers = Vec::new();
    let mut best_count = 0usize;
    let mut best_score = f64::INFINITY;

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

        let mut inliers = vec![false; n];
        let (count, score) = score_inliers_f(
            &f, &x1_x, &x1_y, &x2_x, &x2_y, thresh_sq, &mut inliers,
        );

        let improved = count > best_count || (count == best_count && score < best_score);
        if improved {
            best_model = Some(f);
            best_inliers = inliers;
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

    let model = match best_model {
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
    let mut best_inliers = Vec::new();
    let mut best_count = 0usize;
    let mut best_score = f64::INFINITY;

    // Adaptive iteration count: after each time best_count improves, recompute
    // the iteration bound that gives 99.9% confidence of drawing an all-inlier
    // sample at least once. N = log(1-p) / log(1-w^s) with s=4, p=0.999.
    // At w=0.8 (the common case after descriptor filtering), N≈9 — vs the
    // hard-coded 2000. That's where most of the 47× gap vs OpenCV lives.
    let log_fail = (1.0_f64 - 0.999).ln();
    let mut dynamic_max = params.max_iterations;
    let mut iter = 0usize;
    while iter < dynamic_max {
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

        let mut inliers = vec![false; n];
        let (count, score) = score_inliers_h(
            &h, &x1_x, &x1_y, &x2_x, &x2_y, thresh_sq, &mut inliers,
        );

        let improved = count > best_count || (count == best_count && score < best_score);
        if improved {
            best_model = Some(h);
            best_inliers = inliers;
            best_count = count;
            best_score = score;

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
pub fn two_view_estimate(
    x1: &[Vec2F64],
    x2: &[Vec2F64],
    k1: &Mat3F64,
    k2: &Mat3F64,
    config: &TwoViewConfig,
) -> Result<TwoViewResult, TwoViewError> {
    let res_f = ransac_fundamental(x1, x2, &config.ransac_f)?;
    let res_h = ransac_homography(x1, x2, &config.ransac_h)?;

    let use_h =
        (res_h.inlier_count as f64) > config.homography_inlier_ratio * (res_f.inlier_count as f64);

    let k1_inv = k1.inverse();
    let k2_inv = k2.inverse();

    let tri_params = TriangulateParams {
        k1_inv: &k1_inv,
        k2_inv: &k2_inv,
        config: &config.triangulation,
    };

    let mut best_pose = None;
    let mut best_count = 0usize;
    let mut best_points = Vec::new();
    let mut best_indices = Vec::new();

    let (model, inliers) = if use_h {
        let h = res_h.model;
        let poses = decompose_homography(&h, k1, k2);
        for (r, t) in &poses {
            let (count, points, indices) =
                triangulate_inliers(x1, x2, &res_h.inliers, r, t, &tri_params);
            if count >= config.triangulation.min_cheirality_count && count > best_count {
                best_count = count;
                best_pose = Some((*r, *t));
                best_points = points;
                best_indices = indices;
            }
        }
        (TwoViewModel::Homography(h), res_h.inliers)
    } else {
        let f = res_f.model;
        let e = essential_from_fundamental(&f, k1, k2);
        let e = enforce_essential_constraints(&e);
        let poses = decompose_essential(&e);
        for (r, t) in &poses {
            let (count, points, indices) =
                triangulate_inliers(x1, x2, &res_f.inliers, r, t, &tri_params);
            if count >= config.triangulation.min_cheirality_count && count > best_count {
                best_count = count;
                best_pose = Some((*r, *t));
                best_points = points;
                best_indices = indices;
            }
        }
        (TwoViewModel::Fundamental(f), res_f.inliers)
    };

    let (r, t) = match best_pose {
        Some(p) => p,
        None => return Err(TwoViewError::RansacFailure),
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
    #[cfg(not(target_arch = "aarch64"))]
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
    #[cfg(not(target_arch = "aarch64"))]
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
        let err = vfmaq_f64(
            vfmaq_f64(fx1z, x2, fx1x),
            y2,
            fx1y,
        );
        // denom = fx1x² + fx1y² + ftx2x² + ftx2y²
        let denom = vfmaq_f64(
            vfmaq_f64(
                vfmaq_f64(vmulq_f64(fx1x, fx1x), fx1y, fx1y),
                ftx2x,
                ftx2x,
            ),
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
        let normal = [[120.0, 180.0], [340.0, 150.0], [280.0, 420.0], [410.0, 370.0]];
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
        let (count, score) = score_inliers_h(
            &h, &x1_x, &x1_y, &x2_x, &x2_y, thresh_sq, &mut inl,
        );

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
        let (count, score) = score_inliers_f(
            &f_mat, &x1_x, &x1_y, &x2_x, &x2_y, thresh_sq, &mut inl,
        );

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
                refit: false,
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
        };

        let result = two_view_estimate(&pts1, &pts2, &k, &k, &config).unwrap();

        // Should select fundamental model (general motion, not planar).
        assert!(
            matches!(result.model, TwoViewModel::Fundamental(_)),
            "expected fundamental model"
        );

        // Check rotation angle: GT is 2.7021°, allow 5° error.
        let r = result.rotation;
        let trace = r.col(0).x + r.col(1).y + r.col(2).z;
        let est_angle_rad = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0).acos();
        let est_angle_deg = est_angle_rad.to_degrees();
        let gt_angle_deg = 2.7021;
        assert!(
            (est_angle_deg - gt_angle_deg).abs() < 5.0,
            "rotation error too large: estimated {est_angle_deg:.2}°, GT {gt_angle_deg}°"
        );

        // Check translation direction: GT is [0.242, -0.233, 0.942].
        // Translation can be recovered up to sign, so check both directions.
        let t = result.translation.normalize();
        let gt_t = Vec3F64::new(0.2422, -0.2330, 0.9418).normalize();
        let cos_angle = t.dot(gt_t).clamp(-1.0, 1.0);
        let t_err_deg = cos_angle.abs().acos().to_degrees();
        assert!(
            t_err_deg < 15.0,
            "translation direction error too large: {t_err_deg:.2}°"
        );

        // Should have triangulated some points.
        assert!(
            !result.points3d.is_empty(),
            "expected triangulated 3D points"
        );
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
