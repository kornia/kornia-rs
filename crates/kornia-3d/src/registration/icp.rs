//! Projective point-to-plane ICP on RGBD pyramids.
//!
//! Estimates the rigid transform `T_target_source` (`p_tgt = R * p_src + t`)
//! between two depth frames given as [`RgbdPyramid`]s, coarse-to-fine.
//! Correspondences come from projective association (no kd-tree): each source
//! vertex is transformed by the current estimate, projected into the target
//! grid with the target level intrinsics, and matched to the nearest pixel's
//! vertex/normal.
//!
//! Per correspondence the point-to-plane residual is `r = n_t . (T*p_s - v_t)`
//! with Jacobian row `[(p' x n_t)^T, n_t^T]` where `p' = T*p_s` (left
//! perturbation `T <- exp([w, t]) * T`). Maps stay `f32`; `J^T W J` / `J^T W r`
//! are accumulated in `f64` and the 6x6 system is solved by Cholesky.

use super::rgbd::{is_valid_normal, is_valid_vertex, RgbdIcpError, RgbdLevel, RgbdPyramid};

/// Parameters of [`icp_projective_plane`].
#[derive(Debug, Clone)]
pub struct IcpPlaneCriteria {
    /// Gauss-Newton iterations per pyramid level, coarsest first. Levels
    /// beyond the list reuse the last entry; an empty list runs no iterations
    /// (the result is the evaluated initial guess).
    pub iters_per_level: Vec<usize>,
    /// Convergence threshold on the twist update norm `sqrt(|w|^2 + |dt|^2)`
    /// (rad / metres); a level stops early once the update falls below it.
    pub update_tolerance: f64,
    /// Reject correspondences further apart than this in metres.
    pub max_dist_m: f64,
    /// Reject correspondences whose (rotated) source and target normals
    /// disagree by more than this angle in radians.
    pub max_normal_angle_rad: f64,
    /// Huber loss scale in metres: residuals beyond it get weight `delta/|r|`.
    pub huber_delta_m: f64,
    /// Optional independent estimate of the rotation, e.g. an integrated gyro.
    ///
    /// Depth geometry does not always determine rotation: on a swept view the rotation block of
    /// the normal equations collapses and the solve returns rotations wrong by degrees per frame
    /// (measured: 1.4-4.4 deg against a 3.0 deg truth, eventually freezing at zero) while the
    /// residual and the inlier fraction stay perfect. A gyro measures exactly that quantity, and
    /// over a single frame interval it is roughly an order of magnitude more accurate than the
    /// degraded solve, so it can carry the rotation while the geometry carries the translation.
    ///
    /// Regularises, never overrides: it enters as three weighted rows, so ample geometry
    /// outvotes it and weak geometry defers to it.
    pub rotation_prior: Option<RotationPrior>,
}

/// An external rotation estimate for [`IcpPlaneCriteria::rotation_prior`].
#[derive(Debug, Clone, Copy)]
pub struct RotationPrior {
    /// Prior on the source-to-target rotation, same convention as
    /// [`IcpPlaneResult::rotation`].
    pub rotation: [[f64; 3]; 3],
    /// One-sigma expected error of `rotation`, in radians, OVER THE INTERVAL IT SPANS.
    ///
    /// Scale it with elapsed time rather than passing a per-frame constant: a prior accumulated
    /// across held frames covers a longer interval and has drifted further, and a fixed sigma
    /// would make the constraint overconfident precisely after a stall. For a consumer MEMS gyro,
    /// bias dominates — roughly 0.5 deg/s uncalibrated, so ~0.025 deg over a 50 ms frame.
    ///
    /// The weight is `1/sigma^2`, which puts a 0.05-0.1 deg sigma one to two orders of magnitude
    /// tighter than a collapsed rotation block, and negligible against a healthy one.
    ///
    /// Size it against the sensor's TOTAL error, systematic terms included — not its bias and
    /// noise alone. Two sweeps, measured:
    ///
    /// - varying the true error at fixed sigma: the prior is comfortably ahead at sigma, still
    ///   winning around 1.5x, and loses beyond roughly 2x
    /// - varying sigma at a fixed realistic error: accuracy improved monotonically as sigma
    ///   LOOSENED (0.66 mm at 0.2 deg/s, 0.52 at 2.0, 0.24 at 5.0) with the tracked-frame count
    ///   flat across the whole range
    ///
    /// The second is counter-intuitive and decides the tuning. The dominant error there was a
    /// fixed 3 deg gyro-to-camera misalignment — systematic, not random — so a tight sigma does
    /// not pull the solve toward truth, it pulls it toward a consistently wrong attitude, while a
    /// loose one supplies just enough weight to stop the rotation block collapsing and leaves the
    /// depth free to correct the rest. Tight weighting on a biased measurement buys you the bias.
    /// The flat frame count is the same fact from the other side: a collapsed block carries
    /// almost no information, so even a very loose prior dominates it.
    ///
    /// That ordering inverts once the systematic term is calibrated away. An unbiased gyro should
    /// favour a tighter sigma, so re-measure for that case rather than inheriting this guidance.
    pub sigma_rad: f64,
}

impl Default for IcpPlaneCriteria {
    fn default() -> Self {
        Self {
            iters_per_level: vec![10, 7, 5],
            update_tolerance: 1e-6,
            max_dist_m: 0.10,
            max_normal_angle_rad: 30.0_f64.to_radians(),
            huber_delta_m: 0.02,
            rotation_prior: None,
        }
    }
}

/// Result of [`icp_projective_plane`].
///
/// The transformation maps source-frame points into the target frame:
/// `p_tgt = rotation * p_src + translation`.
#[derive(Debug, Clone)]
pub struct IcpPlaneResult {
    /// Estimated rotation matrix (row-major).
    pub rotation: [[f64; 3]; 3],
    /// Estimated translation vector in metres.
    pub translation: [f64; 3],
    /// Root-mean-square point-to-plane residual (metres) of the gated
    /// correspondences at the finest level under the final transform.
    pub rmse: f64,
    /// Gated correspondences divided by *associated* ones at the finest level
    /// under the final transform: match quality alone. Deliberately not divided
    /// by all valid source pixels — that conflates quality with view overlap and
    /// collapses under camera motion, rejecting solves that converged perfectly.
    /// Pair it with [`Self::overlap_fraction`] and [`Self::num_associated`]: a
    /// small patch can match perfectly and still be too little to constrain a
    /// pose.
    pub inlier_fraction: f64,
    /// Source pixels that projected into the target frame onto valid geometry,
    /// before the distance and normal-angle gates. The observability figure: how
    /// much evidence the solve actually had.
    pub num_associated: usize,
    /// Associated pixels divided by valid source pixels: how much of the source
    /// view still overlaps the target. Falls as the camera moves away from the
    /// keyframe and is the signal to re-key, not to reject the solve.
    pub overlap_fraction: f64,
    /// Total Gauss-Newton iterations performed across all levels.
    pub iterations: usize,
    /// Conditioning of the rotation block: weakest pivot over strongest, in `0..=1`.
    pub rotation_conditioning: f64,
    /// Conditioning of the translation block: weakest pivot over strongest, in `0..=1`.
    ///
    /// This is the one that catches a camera sweeping along a wall. As the surface comes to
    /// dominate the view, translation along it stops being constrained while rotation stays
    /// perfectly determined — measured on a sweep: rotation exact to 0.01 deg, translation four
    /// times the truth, with [`Self::rmse`] at 0.4 mm and [`Self::inlier_fraction`] at 1.00
    /// throughout. Residual and inlier fraction cannot see it; they are computed over
    /// correspondences that all agree, on a surface that cannot pin the pose.
    ///
    /// Deliberately a ratio, not an absolute pivot: pivots accumulate over correspondences, so
    /// an absolute floor tracks the inlier count and the scene depth instead of the geometry.
    ///
    /// Calibrate a gate against the SLIDING regime, not against the degenerate extreme. The
    /// frames that corrupt a map sit orders of magnitude above a bare wall, so a threshold placed
    /// to catch outright degeneracy passes every pose that does damage — that mistake has already
    /// been made once here.
    ///
    /// Measured on a 320x180 depth camera sweeping a room at ~43 deg/s, 20 mm of true motion per
    /// frame. Calibrate against the ONSET of failure:
    ///
    /// | | conditioning | published motion |
    /// |---|---|---|
    /// | worst frame still exact | 1.7e-2 | error <= 0.1 mm |
    /// | first frame that slid | 7.2e-4 | 82.8 mm for 20 mm |
    ///
    /// A 24x separation with nothing in it, so a gate anywhere near the 3.5e-3 midpoint behaves
    /// the same. Take the threshold from that transition only: once a pose is wrong, every later
    /// frame's conditioning is measured against a corrupted estimate and is not independent
    /// evidence — averaging those in understates the separation.
    ///
    /// Gate on [`Self::observability`], the weaker block, not on this field alone. Measured on
    /// the same sweep, translation collapses first (7.2e-4 while rotation holds at 2.96e-2) and
    /// then the blocks SWAP: translation recovers to 1e-1 while rotation falls to 2e-3, with the
    /// published motion decaying to half of truth throughout. A gate on either block alone passes
    /// one half of that failure.
    ///
    pub translation_conditioning: f64,
    /// The weaker of the two blocks: a single scalar for callers that just need a gate.
    pub observability: f64,
}

/// Relative Cholesky pivot threshold of the degeneracy guard: while factoring
/// `A = J^T W J`, a squared pivot below `PIVOT_RTOL` times its *block's* max
/// diagonal (rotation rows 0-2 / translation rows 3-5 are thresholded
/// separately — rotational entries scale as `|p x n|^2` ~ depth² while
/// translational ones are O(1) from unit normals, so one global max would let
/// a collapsed translation pivot pass at large depth) means some twist
/// direction is (numerically) unobserved — e.g. a single plane leaves its two
/// in-plane translations and the in-plane rotation unconstrained, so three
/// pivots collapse to rounding noise — and the solve is rejected as
/// [`RgbdIcpError::SingularNormalEquations`] instead of returning a pose made
/// up along the null space.
const PIVOT_RTOL: f64 = 1e-8;

/// Projective point-to-plane ICP between two RGBD pyramids.
///
/// # Arguments
///
/// * `source` - Source frame pyramid.
/// * `target` - Target frame pyramid.
/// * `initial_rot` - Initial rotation from the source to the target frame.
/// * `initial_trans` - Initial translation from the source to the target frame.
/// * `criteria` - Gating, robustness and convergence parameters.
///
/// # Errors
///
/// [`RgbdIcpError::SingularNormalEquations`] when the geometry does not
/// constrain all six DoF (see [`PIVOT_RTOL`]);
/// [`RgbdIcpError::TooFewCorrespondences`] when fewer than 6 correspondences
/// survive gating (low overlap / bad initial guess);
/// [`RgbdIcpError::InvalidNumLevels`] if either pyramid is empty.
pub fn icp_projective_plane(
    source: &RgbdPyramid,
    target: &RgbdPyramid,
    initial_rot: [[f64; 3]; 3],
    initial_trans: [f64; 3],
    criteria: IcpPlaneCriteria,
) -> Result<IcpPlaneResult, RgbdIcpError> {
    let num_levels = source.levels.len().min(target.levels.len());
    if num_levels == 0 {
        return Err(RgbdIcpError::InvalidNumLevels(0));
    }

    let mut rotation = initial_rot;
    let mut translation = initial_trans;
    let mut iterations = 0;
    let mut rotation_conditioning = 0.0;
    let mut translation_conditioning = 0.0;

    // coarse-to-fine: level num_levels-1 down to 0 (levels[0] is finest)
    for (coarse_idx, level_idx) in (0..num_levels).rev().enumerate() {
        let iters = criteria
            .iters_per_level
            .get(coarse_idx)
            .or(criteria.iters_per_level.last())
            .copied()
            .unwrap_or(0);
        let src = &source.levels[level_idx];
        let tgt = &target.levels[level_idx];

        for _ in 0..iters {
            let mut eqs = accumulate_level(src, tgt, &rotation, &translation, &criteria);
            if eqs.num_inliers < 6 {
                return Err(RgbdIcpError::TooFewCorrespondences(eqs.num_inliers));
            }
            // Diagnostic first, on the DATA-only equations — see `block_conditioning`. A prior
            // must not be able to make weak geometry look strong to the caller.
            let (data_rot_cond, data_trans_cond) = block_conditioning(&eqs.a);
            if let Some(prior) = criteria.rotation_prior {
                // Three rows penalising the current rotation's deviation from the prior.
                //
                // The error must live in the frame the update acts in. The solver applies
                // `R <- exp(w) R`, a LEFT multiply, so the error is `e = log(R R_prior^T)`, also
                // on the left: then `R_new R_prior^T = exp(w) exp(e)` and the residual in `w` is
                // `e + w` to first order, with an identity Jacobian. Writing the body-frame error
                // `log(R_prior^T R)` here instead would agree only when the prior is near
                // identity and would otherwise pull toward the wrong attitude.
                //
                // Contribution is `1/sigma^2` on the rotation block's diagonal and `w e` on its
                // gradient. Only that block is touched — the prior says nothing about translation.
                let e = so3_log(&mat3_mul(&rotation, &mat3_transpose(&prior.rotation)));
                let w = 1.0 / (prior.sigma_rad * prior.sigma_rad).max(f64::MIN_POSITIVE);
                for (i, ei) in e.iter().enumerate() {
                    eqs.a[i][i] += w;
                    eqs.b[i] += w * ei;
                }
            }
            // solve A x = -b for the twist x = [w, dt]
            let neg_b = eqs.b.map(|v| -v);
            let (x, _, _) =
                cholesky_solve_6x6(&eqs.a, &neg_b).ok_or(RgbdIcpError::SingularNormalEquations)?;
            // Report the finest level's last solve: the geometry the pose actually rests on,
            // which is the data-only conditioning even when a prior carried the solve.
            rotation_conditioning = data_rot_cond;
            translation_conditioning = data_trans_cond;

            let omega = [x[0], x[1], x[2]];
            let dt = [x[3], x[4], x[5]];
            let r_delta = so3_exp(&omega);
            rotation = mat3_mul(&r_delta, &rotation);
            translation = [
                mat3_row_dot(&r_delta, 0, &translation) + dt[0],
                mat3_row_dot(&r_delta, 1, &translation) + dt[1],
                mat3_row_dot(&r_delta, 2, &translation) + dt[2],
            ];
            iterations += 1;

            let update_norm = x.iter().map(|v| v * v).sum::<f64>().sqrt();
            if update_norm < criteria.update_tolerance {
                break;
            }
        }
    }

    // final metrics: one association pass at the finest level, final pose
    let finest_src = &source.levels[0];
    let finest_tgt = &target.levels[0];
    let eqs = accumulate_level(finest_src, finest_tgt, &rotation, &translation, &criteria);
    let rmse = if eqs.num_inliers > 0 {
        (eqs.sum_sq_residual / eqs.num_inliers as f64).sqrt()
    } else {
        f64::INFINITY
    };
    let inlier_fraction = if eqs.num_associated > 0 {
        eqs.num_inliers as f64 / eqs.num_associated as f64
    } else {
        0.0
    };
    let overlap_fraction = if eqs.num_valid > 0 {
        eqs.num_associated as f64 / eqs.num_valid as f64
    } else {
        0.0
    };

    Ok(IcpPlaneResult {
        rotation,
        translation,
        rmse,
        inlier_fraction,
        num_associated: eqs.num_associated,
        rotation_conditioning,
        translation_conditioning,
        observability: rotation_conditioning.min(translation_conditioning),
        overlap_fraction,
        iterations,
    })
}

/// Accumulated normal equations of one association pass over a level.
struct NormalEquations {
    /// `J^T W J` (full symmetric 6x6).
    a: [[f64; 6]; 6],
    /// `J^T W r`.
    b: [f64; 6],
    /// Correspondences that survived gating.
    num_inliers: usize,
    /// Projected inside the target onto valid geometry (pre-gating).
    num_associated: usize,
    /// Source pixels with a valid vertex and normal.
    num_valid: usize,
    /// Unweighted sum of squared residuals over the inliers.
    sum_sq_residual: f64,
}

/// One projective-association pass: gate, weight and accumulate `J^T W J`,
/// `J^T W r` in f64.
fn accumulate_level(
    src: &RgbdLevel,
    tgt: &RgbdLevel,
    rotation: &[[f64; 3]; 3],
    translation: &[f64; 3],
    criteria: &IcpPlaneCriteria,
) -> NormalEquations {
    let mut eqs = NormalEquations {
        a: [[0.0; 6]; 6],
        b: [0.0; 6],
        num_inliers: 0,
        num_associated: 0,
        num_valid: 0,
        sum_sq_residual: 0.0,
    };
    let cos_max_angle = criteria.max_normal_angle_rad.cos();
    let max_dist_sq = criteria.max_dist_m * criteria.max_dist_m;
    let (tw, th) = (tgt.intrinsics.width, tgt.intrinsics.height);

    for (v_s, n_s) in src.vertices.iter().zip(src.normals.iter()) {
        if !is_valid_vertex(v_s) || !is_valid_normal(n_s) {
            continue;
        }
        eqs.num_valid += 1;

        let p_s = [v_s[0] as f64, v_s[1] as f64, v_s[2] as f64];
        // p' = T * p_s in the target frame
        let p = [
            mat3_row_dot(rotation, 0, &p_s) + translation[0],
            mat3_row_dot(rotation, 1, &p_s) + translation[1],
            mat3_row_dot(rotation, 2, &p_s) + translation[2],
        ];
        let Some(uv) = tgt.intrinsics.project(&p) else {
            continue;
        };
        let (u, v) = (uv[0].round(), uv[1].round());
        if u < 0.0 || v < 0.0 || u > (tw - 1) as f64 || v > (th - 1) as f64 {
            continue;
        }
        let idx = v as usize * tw + u as usize;
        let v_t = &tgt.vertices[idx];
        let n_t = &tgt.normals[idx];
        if !is_valid_vertex(v_t) || !is_valid_normal(n_t) {
            continue;
        }
        eqs.num_associated += 1;

        let diff = [
            p[0] - v_t[0] as f64,
            p[1] - v_t[1] as f64,
            p[2] - v_t[2] as f64,
        ];
        if diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2] > max_dist_sq {
            continue;
        }

        let n_t = [n_t[0] as f64, n_t[1] as f64, n_t[2] as f64];
        let n_s = [n_s[0] as f64, n_s[1] as f64, n_s[2] as f64];
        let n_s_rot = [
            mat3_row_dot(rotation, 0, &n_s),
            mat3_row_dot(rotation, 1, &n_s),
            mat3_row_dot(rotation, 2, &n_s),
        ];
        if n_s_rot[0] * n_t[0] + n_s_rot[1] * n_t[1] + n_s_rot[2] * n_t[2] < cos_max_angle {
            continue;
        }

        let r = n_t[0] * diff[0] + n_t[1] * diff[1] + n_t[2] * diff[2];
        // Jacobian row [(p' x n_t)^T, n_t^T]
        let j = [
            p[1] * n_t[2] - p[2] * n_t[1],
            p[2] * n_t[0] - p[0] * n_t[2],
            p[0] * n_t[1] - p[1] * n_t[0],
            n_t[0],
            n_t[1],
            n_t[2],
        ];
        let w = if r.abs() <= criteria.huber_delta_m {
            1.0
        } else {
            criteria.huber_delta_m / r.abs()
        };

        for i in 0..6 {
            eqs.b[i] += w * j[i] * r;
            for k in i..6 {
                eqs.a[i][k] += w * j[i] * j[k];
            }
        }
        eqs.num_inliers += 1;
        eqs.sum_sq_residual += r * r;
    }

    // mirror the accumulated upper triangle
    for i in 1..6 {
        for k in 0..i {
            eqs.a[i][k] = eqs.a[k][i];
        }
    }

    eqs
}

/// Solve the SPD system `A x = b` by Cholesky. Returns `None` when a squared
/// pivot falls below `PIVOT_RTOL` times its block's max diagonal (rotation /
/// translation thresholded separately — see [`PIVOT_RTOL`]) — the degeneracy
/// guard.
/// Solves `A x = b` and reports the weakest pivot relative to its block's max diagonal — the
/// observability of the least-constrained direction. A view that pins every DOF keeps this well
/// above [`PIVOT_RTOL`]; a wall filling the frame drives it toward zero along the sliding
/// direction, where the residual stays perfect while the pose drifts. Callers that must not act
/// on such a pose gate on it; the residual and the inlier fraction cannot see it.
/// Conditioning of each block WITHOUT solving: weakest pivot over strongest, `(rotation,
/// translation)`, or `(0, 0)` when the matrix does not factor at all.
///
/// Prior-free in its INPUTS, not in the pose. A prior changes the solve, so the final transform
/// differs, so the last association pass covers a slightly different correspondence set and the
/// conditioning is computed over that — measured drift of about 5% on the same frame. That is
/// benign for gating; evaluating at a fixed transform would be required to make it bit-stable.
///
/// Measured on the DATA-only normal equations, never on the prior-augmented ones. A rotation
/// prior lifts the rotation block, and because the blocks are coupled through elimination it
/// lifts the translation block's effective pivots too — measured, a 14x rise at a frame whose
/// translation error was unchanged at 3.5x the truth. Reporting that would let the regulariser
/// flatter the diagnostic and silently disarm the gate that catches unobservable translation.
/// The prior belongs in the solve; the diagnostic must keep describing the geometry.
fn block_conditioning(a: &[[f64; 6]; 6]) -> (f64, f64) {
    match cholesky_factor(a) {
        Some((_, rot, trans)) => (rot, trans),
        None => (0.0, 0.0),
    }
}

fn cholesky_solve_6x6(a: &[[f64; 6]; 6], b: &[f64; 6]) -> Option<([f64; 6], f64, f64)> {
    let (l, rot_cond, trans_cond) = cholesky_factor(a)?;

    // L y = b
    let mut y = [0.0; 6];
    for i in 0..6 {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[i][k] * y[k];
        }
        y[i] = sum / l[i][i];
    }
    // L^T x = y
    let mut x = [0.0; 6];
    for i in (0..6).rev() {
        let mut sum = y[i];
        for k in i + 1..6 {
            sum -= l[k][i] * x[k];
        }
        x[i] = sum / l[i][i];
    }
    Some((x, rot_cond, trans_cond))
}

fn cholesky_factor(a: &[[f64; 6]; 6]) -> Option<([[f64; 6]; 6], f64, f64)> {
    let max_diag_rot = (0..3).map(|i| a[i][i]).fold(0.0, f64::max);
    let max_diag_trans = (3..6).map(|i| a[i][i]).fold(0.0, f64::max);
    if max_diag_rot <= 0.0 || max_diag_trans <= 0.0 {
        return None;
    }

    // A = L L^T. Track each block's weakest and strongest pivot: their RATIO is the
    // conditioning. An absolute pivot is not comparable across scenes — pivots accumulate over
    // correspondences, so they grow with the inlier count and the depth, and a direction can be
    // orders of magnitude worse constrained than its neighbours while still being numerically
    // large. The ratio is scale-free and is what "one direction is unconstrained" means.
    let (mut min_rot, mut max_rot) = (f64::INFINITY, 0.0f64);
    let (mut min_trans, mut max_trans) = (f64::INFINITY, 0.0f64);
    let mut l = [[0.0; 6]; 6];
    for i in 0..6 {
        for j in 0..=i {
            let mut sum = a[i][j];
            for (lik, ljk) in l[i].iter().zip(l[j].iter()).take(j) {
                sum -= lik * ljk;
            }
            if i == j {
                let block_scale = if i < 3 { max_diag_rot } else { max_diag_trans };
                if i < 3 {
                    min_rot = min_rot.min(sum);
                    max_rot = max_rot.max(sum);
                } else {
                    min_trans = min_trans.min(sum);
                    max_trans = max_trans.max(sum);
                }
                if sum < PIVOT_RTOL * block_scale {
                    return None;
                }
                l[i][j] = sum.sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }

    let ratio = |lo: f64, hi: f64| {
        if hi > 0.0 {
            (lo / hi).clamp(0.0, 1.0)
        } else {
            0.0
        }
    };
    Some((l, ratio(min_rot, max_rot), ratio(min_trans, max_trans)))
}

/// Axis-angle of a rotation matrix: the inverse of [`so3_exp`].
fn so3_log(r: &[[f64; 3]; 3]) -> [f64; 3] {
    let trace = r[0][0] + r[1][1] + r[2][2];
    let cos = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0);
    let angle = cos.acos();
    // Near zero the axis is ill-defined but the vector is not: sin(angle)/angle -> 1, so the
    // off-diagonal difference IS the axis-angle to first order.
    let scale = if angle < 1e-8 {
        0.5
    } else if angle > std::f64::consts::PI - 1e-6 {
        // Near pi the same expression loses all precision; recover the axis from the symmetric
        // part, where the diagonal stays well conditioned, and restore the sign from the skew.
        let mut axis = [
            ((r[0][0] - cos) / (1.0 - cos)).max(0.0).sqrt(),
            ((r[1][1] - cos) / (1.0 - cos)).max(0.0).sqrt(),
            ((r[2][2] - cos) / (1.0 - cos)).max(0.0).sqrt(),
        ];
        let skew = [r[2][1] - r[1][2], r[0][2] - r[2][0], r[1][0] - r[0][1]];
        for k in 0..3 {
            if skew[k] < 0.0 {
                axis[k] = -axis[k];
            }
        }
        let n = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        return if n > 0.0 {
            [
                axis[0] / n * angle,
                axis[1] / n * angle,
                axis[2] / n * angle,
            ]
        } else {
            [0.0; 3]
        };
    } else {
        angle / (2.0 * angle.sin())
    };
    [
        scale * (r[2][1] - r[1][2]),
        scale * (r[0][2] - r[2][0]),
        scale * (r[1][0] - r[0][1]),
    ]
}

fn mat3_transpose(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut t = [[0.0; 3]; 3];
    for (i, row) in m.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            t[j][i] = v;
        }
    }
    t
}

#[inline]
fn mat3_row_dot(m: &[[f64; 3]; 3], row: usize, v: &[f64; 3]) -> f64 {
    m[row][0] * v[0] + m[row][1] * v[1] + m[row][2] * v[2]
}

fn mat3_mul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for (out_row, a_row) in out.iter_mut().zip(a.iter()) {
        for j in 0..3 {
            out_row[j] = a_row[0] * b[0][j] + a_row[1] * b[1][j] + a_row[2] * b[2][j];
        }
    }
    out
}

/// SO(3) exponential map (Rodrigues): `R = I + a [w]x + b [w]x^2` with
/// `a = sin(t)/t`, `b = (1 - cos(t))/t^2`, Taylor fallback near zero.
fn so3_exp(w: &[f64; 3]) -> [[f64; 3]; 3] {
    let theta2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
    let theta = theta2.sqrt();
    let (a, b) = if theta < 1e-9 {
        (1.0 - theta2 / 6.0, 0.5 - theta2 / 24.0)
    } else {
        (theta.sin() / theta, (1.0 - theta.cos()) / theta2)
    };
    // [w]x^2 = w w^T - theta^2 I
    [
        [
            1.0 + b * (w[0] * w[0] - theta2),
            -a * w[2] + b * w[0] * w[1],
            a * w[1] + b * w[0] * w[2],
        ],
        [
            a * w[2] + b * w[1] * w[0],
            1.0 + b * (w[1] * w[1] - theta2),
            -a * w[0] + b * w[1] * w[2],
        ],
        [
            -a * w[1] + b * w[2] * w[0],
            a * w[0] + b * w[2] * w[1],
            1.0 + b * (w[2] * w[2] - theta2),
        ],
    ]
}

#[cfg(test)]
mod tests {
    use super::super::rgbd::DepthIntrinsics;
    use super::super::synth::{render_depth_mm, Plane, Scene, Sphere};
    use super::*;
    use crate::transforms::axis_angle_to_rotation_matrix;

    const IDENTITY_ROT: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    fn test_intrinsics() -> DepthIntrinsics {
        DepthIntrinsics {
            fx: 200.0,
            fy: 200.0,
            cx: 159.5,
            cy: 89.5,
            width: 320,
            height: 180,
        }
    }

    /// Angle in degrees between two rotation matrices.
    fn rotation_error_deg(r_a: &[[f64; 3]; 3], r_b: &[[f64; 3]; 3]) -> f64 {
        // trace(R_a^T R_b)
        let mut trace = 0.0;
        for i in 0..3 {
            for k in 0..3 {
                trace += r_a[k][i] * r_b[k][i];
            }
        }
        ((trace - 1.0) / 2.0).clamp(-1.0, 1.0).acos().to_degrees()
    }

    fn translation_error(t_a: &[f64; 3], t_b: &[f64; 3]) -> f64 {
        ((t_a[0] - t_b[0]).powi(2) + (t_a[1] - t_b[1]).powi(2) + (t_a[2] - t_b[2]).powi(2)).sqrt()
    }

    /// Ground-truth `T_target_source` from the target camera pose `T_world_cam`
    /// (the source camera sits at the world origin):
    /// `R = R_wc^T`, `t = -R_wc^T t_wc`.
    fn gt_target_source(
        rot_world_cam: &[[f64; 3]; 3],
        t_world_cam: &[f64; 3],
    ) -> ([[f64; 3]; 3], [f64; 3]) {
        let mut rot = [[0.0; 3]; 3];
        for (i, rot_row) in rot.iter_mut().enumerate() {
            for (j, cell) in rot_row.iter_mut().enumerate() {
                *cell = rot_world_cam[j][i];
            }
        }
        let t = [
            -(rot[0][0] * t_world_cam[0] + rot[0][1] * t_world_cam[1] + rot[0][2] * t_world_cam[2]),
            -(rot[1][0] * t_world_cam[0] + rot[1][1] * t_world_cam[1] + rot[1][2] * t_world_cam[2]),
            -(rot[2][0] * t_world_cam[0] + rot[2][1] * t_world_cam[1] + rot[2][2] * t_world_cam[2]),
        ];
        (rot, t)
    }

    #[test]
    fn test_identity_self_icp() -> Result<(), Box<dyn std::error::Error>> {
        let intr = test_intrinsics();
        let depth = render_depth_mm(&Scene::corner_and_sphere(), &intr, &IDENTITY_ROT, &[0.0; 3]);
        let pyr = RgbdPyramid::from_depth_mm(&depth, &intr, 3)?;

        let result = icp_projective_plane(
            &pyr,
            &pyr,
            IDENTITY_ROT,
            [0.0; 3],
            IcpPlaneCriteria::default(),
        )?;

        assert!(
            rotation_error_deg(&result.rotation, &IDENTITY_ROT) < 0.01,
            "self-ICP rotation drifted: {:?}",
            result.rotation
        );
        let t_norm = translation_error(&result.translation, &[0.0; 3]);
        assert!(t_norm < 1e-4, "self-ICP translation drifted: {t_norm}");
        assert!(result.rmse < 1e-6, "self-ICP rmse: {}", result.rmse);
        assert!(
            result.inlier_fraction > 0.9,
            "self-ICP inlier fraction: {}",
            result.inlier_fraction
        );
        Ok(())
    }

    #[test]
    fn test_ground_truth_motion_grid() -> Result<(), Box<dyn std::error::Error>> {
        let intr = test_intrinsics();
        let scene = Scene::corner_and_sphere();
        let src_pyr = {
            let depth = render_depth_mm(&scene, &intr, &IDENTITY_ROT, &[0.0; 3]);
            RgbdPyramid::from_depth_mm(&depth, &intr, 3)?
        };

        // (axis, angle deg, translation) of the target camera pose T_world_cam
        let cases: &[([f64; 3], f64, [f64; 3])] = &[
            ([1.0, 0.0, 0.0], 2.0, [0.0, 0.0, 0.0]),
            ([0.0, 1.0, 0.0], -2.0, [0.0, 0.0, 0.0]),
            ([0.0, 0.0, 1.0], 2.0, [0.0, 0.0, 0.0]),
            ([1.0, 0.0, 0.0], 0.0, [0.03, 0.0, 0.0]),
            ([1.0, 0.0, 0.0], 0.0, [0.0, -0.03, 0.0]),
            ([1.0, 0.0, 0.0], 0.0, [0.0, 0.0, 0.03]),
            ([1.0, 1.0, 1.0], 2.0, [0.02, -0.02, 0.015]),
        ];

        for (axis, angle_deg, t_wc) in cases {
            let rot_wc = axis_angle_to_rotation_matrix(axis, angle_deg.to_radians())?;
            let tgt_depth = render_depth_mm(&scene, &intr, &rot_wc, t_wc);
            let tgt_pyr = RgbdPyramid::from_depth_mm(&tgt_depth, &intr, 3)?;
            let (rot_gt, t_gt) = gt_target_source(&rot_wc, t_wc);

            let result = icp_projective_plane(
                &src_pyr,
                &tgt_pyr,
                IDENTITY_ROT,
                [0.0; 3],
                IcpPlaneCriteria::default(),
            )?;

            let rot_err = rotation_error_deg(&result.rotation, &rot_gt);
            let t_err = translation_error(&result.translation, &t_gt);
            assert!(
                rot_err < 0.2,
                "case {axis:?}/{angle_deg} deg/{t_wc:?}: rotation error {rot_err} deg"
            );
            assert!(
                t_err < 5e-3,
                "case {axis:?}/{angle_deg} deg/{t_wc:?}: translation error {} mm",
                t_err * 1e3
            );
            assert!(
                result.inlier_fraction > 0.5,
                "case {axis:?}/{angle_deg} deg/{t_wc:?}: inlier fraction {}",
                result.inlier_fraction
            );
        }
        Ok(())
    }

    /// A partly-overlapping view must keep HIGH match quality while overlap falls.
    ///
    /// Regression guard for a live failure: `inlier_fraction` once divided by ALL valid source
    /// pixels, so it fell with view overlap rather than with match error. A moving camera then
    /// scored ~11% on solves that had converged to millimetres, every frame was rejected as
    /// "tracking lost", the pose froze and the map stopped growing. Quality and overlap are
    /// separate signals: low overlap means re-key, not reject.
    #[test]
    fn inlier_fraction_measures_quality_not_overlap() -> Result<(), Box<dyn std::error::Error>> {
        let intr = test_intrinsics();
        let scene = Scene::corner_and_sphere();
        let src_depth = render_depth_mm(&scene, &intr, &IDENTITY_ROT, &[0.0; 3]);
        let src_pyr = RgbdPyramid::from_depth_mm(&src_depth, &intr, 3)?;

        // Yaw far enough that a large part of the source view leaves the target frustum.
        let rot_wc = axis_angle_to_rotation_matrix(&[0.0, 1.0, 0.0], 18.0_f64.to_radians())?;
        let t_wc = [0.35, 0.0, 0.0];
        let tgt_depth = render_depth_mm(&scene, &intr, &rot_wc, &t_wc);
        let tgt_pyr = RgbdPyramid::from_depth_mm(&tgt_depth, &intr, 3)?;
        let (rot_gt, t_gt) = gt_target_source(&rot_wc, &t_wc);

        // Rotation comes from the gyro prior in the live node; translation does not.
        let result = icp_projective_plane(
            &src_pyr,
            &tgt_pyr,
            rot_gt,
            [0.0; 3],
            IcpPlaneCriteria::default(),
        )?;

        // The solve is genuinely good ...
        assert!(
            rotation_error_deg(&result.rotation, &rot_gt) < 0.5,
            "rotation error {} deg",
            rotation_error_deg(&result.rotation, &rot_gt)
        );
        assert!(
            translation_error(&result.translation, &t_gt) < 0.01,
            "translation error {} m",
            translation_error(&result.translation, &t_gt)
        );
        // ... so quality stays high, even though a chunk of the view is gone ...
        // ... so quality stays high even though half the view has left the frustum ...
        assert!(
            result.inlier_fraction > 0.9,
            "quality collapsed under partial overlap: {} (associated {})",
            result.inlier_fraction,
            result.num_associated
        );
        // ... the lost view registers as overlap instead, which is what should drive re-keying ...
        assert!(
            result.overlap_fraction < 0.7,
            "overlap should register the rotated-away view: {}",
            result.overlap_fraction
        );
        // ... and the two must not be the same number: the old inliers/valid metric (their
        // product) is what sank to ~11% on hardware and rejected converged solves.
        let inliers_over_valid = result.inlier_fraction * result.overlap_fraction;
        assert!(
            result.inlier_fraction > inliers_over_valid * 1.5,
            "quality {} tracks the old inliers/valid metric {} — the split is gone",
            result.inlier_fraction,
            inliers_over_valid
        );
        assert!(
            result.num_associated > 1000,
            "too little evidence to trust the pose: {}",
            result.num_associated
        );
        Ok(())
    }

    #[test]
    fn test_outlier_band_still_converges() -> Result<(), Box<dyn std::error::Error>> {
        let intr = test_intrinsics();
        let scene = Scene::corner_and_sphere();
        let src_pyr = {
            let depth = render_depth_mm(&scene, &intr, &IDENTITY_ROT, &[0.0; 3]);
            RgbdPyramid::from_depth_mm(&depth, &intr, 3)?
        };

        let rot_wc = axis_angle_to_rotation_matrix(&[0.0, 1.0, 0.0], 1.5f64.to_radians())?;
        let t_wc = [0.02, 0.0, 0.01];
        let mut tgt_depth = render_depth_mm(&scene, &intr, &rot_wc, &t_wc);
        // corrupt a 10% horizontal band with a +60 mm bias: inside the 100 mm
        // distance gate, so it exercises the Huber weighting rather than the gate
        let band_rows = intr.height / 10;
        for v in 80..80 + band_rows {
            for u in 0..intr.width {
                let d = &mut tgt_depth[v * intr.width + u];
                if *d != 0 {
                    *d += 60;
                }
            }
        }
        let tgt_pyr = RgbdPyramid::from_depth_mm(&tgt_depth, &intr, 3)?;
        let (rot_gt, t_gt) = gt_target_source(&rot_wc, &t_wc);

        let result = icp_projective_plane(
            &src_pyr,
            &tgt_pyr,
            IDENTITY_ROT,
            [0.0; 3],
            IcpPlaneCriteria::default(),
        )?;

        let rot_err = rotation_error_deg(&result.rotation, &rot_gt);
        let t_err = translation_error(&result.translation, &t_gt);
        assert!(rot_err < 0.2, "rotation error with outliers: {rot_err} deg");
        assert!(
            t_err < 5e-3,
            "translation error with outliers: {} mm",
            t_err * 1e3
        );
        Ok(())
    }

    /// Weak geometry must be visible in `observability`, because nothing else shows it.
    ///
    /// Guards a measured failure: on a near-degenerate view the tracker reported a perfect
    /// inlier fraction and a sub-millimetre residual while publishing ~3x the true motion,
    /// sliding along the surface. Residual and inlier fraction are blind to it — the pose moves
    /// through a direction the geometry does not penalise — so the caller needs the conditioning
    /// of the normal equations to refuse such a pose.
    #[test]
    fn observability_exposes_weak_geometry() -> Result<(), Box<dyn std::error::Error>> {
        let intr = test_intrinsics();
        // A wall with one small bump. The plane alone leaves three DoF free; the bump pins them,
        // but only just — the near-degenerate regime, not the exactly-singular one the guard
        // already rejects. (Two planes at any angle stay rank-deficient along their intersection,
        // so a "shallow wedge" is the wrong shape for this test.)
        let weak = Scene {
            planes: vec![Plane {
                normal: [0.0, 0.0, 1.0],
                d: 2.0,
            }],
            spheres: vec![Sphere {
                center: [0.0, 0.0, 1.9],
                radius: 0.06,
            }],
        };
        let slide = [0.02, 0.0, 0.0]; // along the surface: little residual to pay

        let mut measured: Vec<(f64, f64, f64)> = Vec::new();
        for scene in [weak, Scene::corner_and_sphere()] {
            let src = RgbdPyramid::from_depth_mm(
                &render_depth_mm(&scene, &intr, &IDENTITY_ROT, &[0.0; 3]),
                &intr,
                3,
            )?;
            let tgt = RgbdPyramid::from_depth_mm(
                &render_depth_mm(&scene, &intr, &IDENTITY_ROT, &slide),
                &intr,
                3,
            )?;
            let r = icp_projective_plane(
                &src,
                &tgt,
                IDENTITY_ROT,
                [0.0; 3],
                IcpPlaneCriteria::default(),
            )?;
            measured.push((r.observability, r.inlier_fraction, r.rmse));
        }
        let (weak_obs, weak_inl, weak_rmse) = measured[0];
        let (rich_obs, rich_inl, _) = measured[1];

        // Measured: 6.4e-7 on the weak scene against 8.8e-2 on the rich one — five orders of
        // magnitude at nearly identical association counts, so the margin is geometry, not scale.
        assert!(
            weak_obs < 1e-4 && rich_obs > 1e-3,
            "observability must separate the two geometries: weak {weak_obs:.3e}, rich {rich_obs:.3e}"
        );
        // The trap this field exists for: on the weak scene the numbers a caller would otherwise
        // trust are not merely acceptable, they are BETTER than on the well-conditioned one.
        assert!(
            weak_inl >= rich_inl && weak_rmse < 0.001,
            "expected the deceptive regime: weak inliers {weak_inl} (rich {rich_inl}), \
             weak rmse {weak_rmse}"
        );
        Ok(())
    }

    /// Conditioning must not move with the number of correspondences.
    ///
    /// This is why the field is a ratio and not a pivot magnitude. Pivots of `J^T W J` accumulate
    /// over correspondences, so an absolute floor tracks image resolution and scene depth rather
    /// than geometry: a gate tuned on one scene silently never fires on a denser or nearer one.
    /// Same geometry at half resolution — a quarter of the correspondences — must report the same
    /// conditioning.
    /// A rotation prior must not flatter the conditioning it is reported alongside.
    ///
    /// Measured regression: with a prior active, a frame whose translation was unobservable saw
    /// its reported translation conditioning rise 14x — over the gate that had correctly held it
    /// — while the published pose stayed wrong by 3.5x. The blocks are coupled through
    /// elimination, so lifting rotation lifts translation's effective pivots. Conditioning is
    /// therefore measured on the data-only equations while the solve uses the augmented ones.
    #[test]
    fn rotation_prior_does_not_inflate_reported_conditioning(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let intr = test_intrinsics();
        // Roll-degenerate: rotation unobservable, translation pinned by the sphere.
        let scene = Scene {
            planes: vec![Plane {
                normal: [0.0, 0.0, 1.0],
                d: 2.5,
            }],
            spheres: vec![Sphere {
                center: [0.0, 0.0, 1.8],
                radius: 0.55,
            }],
        };
        let true_rot = axis_angle_to_rotation_matrix(&[0.0, 0.0, 1.0], 3.0_f64.to_radians())?;
        let src = RgbdPyramid::from_depth_mm(
            &render_depth_mm(&scene, &intr, &IDENTITY_ROT, &[0.0; 3]),
            &intr,
            3,
        )?;
        let tgt = RgbdPyramid::from_depth_mm(
            &render_depth_mm(&scene, &intr, &true_rot, &[0.01, 0.0, 0.0]),
            &intr,
            3,
        )?;
        let (prior_gt, _) = gt_target_source(&true_rot, &[0.01, 0.0, 0.0]);

        let with = icp_projective_plane(
            &src,
            &tgt,
            IDENTITY_ROT,
            [0.0; 3],
            IcpPlaneCriteria {
                rotation_prior: Some(RotationPrior {
                    rotation: prior_gt,
                    sigma_rad: 0.1_f64.to_radians(),
                }),
                ..Default::default()
            },
        )?;

        // The prior rescues the solve — without it the guard rejects this scene outright — but
        // the rotation it rescued must still be REPORTED as unobservable, because it is.
        assert!(
            with.rotation_conditioning < 1e-3,
            "prior inflated reported rotation conditioning to {:.3e}; the geometry did not change",
            with.rotation_conditioning
        );
        Ok(())
    }

    #[test]
    fn conditioning_is_scale_free() -> Result<(), Box<dyn std::error::Error>> {
        let scene = Scene::corner_and_sphere();
        let full = test_intrinsics();
        let half = DepthIntrinsics {
            fx: full.fx / 2.0,
            fy: full.fy / 2.0,
            cx: (full.cx + 0.5) / 2.0 - 0.5,
            cy: (full.cy + 0.5) / 2.0 - 0.5,
            width: full.width / 2,
            height: full.height / 2,
        };
        let slide = [0.02, 0.0, 0.0];

        let mut solved = Vec::new();
        for intr in [&full, &half] {
            let src = RgbdPyramid::from_depth_mm(
                &render_depth_mm(&scene, intr, &IDENTITY_ROT, &[0.0; 3]),
                intr,
                2,
            )?;
            let tgt = RgbdPyramid::from_depth_mm(
                &render_depth_mm(&scene, intr, &IDENTITY_ROT, &slide),
                intr,
                2,
            )?;
            let r = icp_projective_plane(
                &src,
                &tgt,
                IDENTITY_ROT,
                [0.0; 3],
                IcpPlaneCriteria::default(),
            )?;
            solved.push(r);
        }
        let (a, b) = (&solved[0], &solved[1]);
        assert!(
            b.num_associated * 3 < a.num_associated,
            "half resolution should associate far fewer pixels: {} vs {}",
            b.num_associated,
            a.num_associated
        );
        for (name, lo, hi) in [
            ("rotation", a.rotation_conditioning, b.rotation_conditioning),
            (
                "translation",
                a.translation_conditioning,
                b.translation_conditioning,
            ),
        ] {
            let ratio = lo.max(hi) / lo.min(hi).max(f64::MIN_POSITIVE);
            assert!(
                ratio < 10.0,
                "{name} conditioning moved with correspondence count: {lo:.3e} vs {hi:.3e}"
            );
        }
        Ok(())
    }

    /// The rotation prior must rescue weak geometry WITHOUT damaging good geometry, and it must
    /// survive a prior that is itself wrong — a real gyro carries bias, noise and, on hardware
    /// with no IMU extrinsics, a fixed misalignment. A prior fed the exact answer proves nothing.
    #[test]
    fn rotation_prior_rescues_weak_geometry_without_harming_good(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let intr = test_intrinsics();
        // Rotation-weak, translation-strong: a sphere centred on the optical axis in front of a
        // fronto-parallel wall. Both are invariant under ROLL, so rotation about the viewing axis
        // is unobservable, while the sphere's normals span every direction and pin all three
        // translations. This isolates the mode the prior exists for — a scene that starves
        // translation as well would be beyond any rotation prior's reach.
        let weak = Scene {
            planes: vec![Plane {
                normal: [0.0, 0.0, 1.0],
                d: 2.5,
            }],
            spheres: vec![Sphere {
                center: [0.0, 0.0, 1.8],
                radius: 0.55,
            }],
        };
        let rich = Scene::corner_and_sphere();
        // Roll about the viewing axis — the unobservable direction on the weak scene.
        let true_rot = axis_angle_to_rotation_matrix(&[0.0, 0.0, 1.0], 3.0_f64.to_radians())?;
        let true_t = [0.01, 0.0, 0.0];

        // The prior is deliberately WRONG by 0.15 deg — well beyond a MEMS gyro's error over one
        // frame, so passing here means it tolerates worse than reality.
        let prior_err_deg: f64 = 0.15;
        let prior_rot = mat3_mul(
            &axis_angle_to_rotation_matrix(&[0.3, 0.6, 0.74], prior_err_deg.to_radians())?,
            &true_rot,
        );

        let mut summary = Vec::new();
        for scene in [&weak, &rich] {
            let src = RgbdPyramid::from_depth_mm(
                &render_depth_mm(scene, &intr, &IDENTITY_ROT, &[0.0; 3]),
                &intr,
                3,
            )?;
            let tgt = RgbdPyramid::from_depth_mm(
                &render_depth_mm(scene, &intr, &true_rot, &true_t),
                &intr,
                3,
            )?;
            let (rot_gt, _) = gt_target_source(&true_rot, &true_t);

            let solve = |prior: Option<RotationPrior>| {
                let mut c = IcpPlaneCriteria {
                    rotation_prior: prior,
                    ..Default::default()
                };
                c.iters_per_level = vec![10, 7, 5];
                icp_projective_plane(&src, &tgt, IDENTITY_ROT, [0.0; 3], c)
            };
            let (prior_gt, _) = gt_target_source(&prior_rot, &true_t);
            // Without the prior the weak scene is rejected outright by the degeneracy guard —
            // rotation is not merely poorly determined, it is unconstrained. Record that as an
            // infinite error so the two cases compare on one scale.
            let without = solve(None)
                .map(|r| rotation_error_deg(&r.rotation, &rot_gt))
                .unwrap_or(f64::INFINITY);
            let with = solve(Some(RotationPrior {
                rotation: prior_gt,
                sigma_rad: prior_err_deg.to_radians(),
            }))
            .map(|r| rotation_error_deg(&r.rotation, &rot_gt))
            .unwrap_or(f64::INFINITY);
            summary.push((without, with));
        }
        let (weak_without, weak_with) = summary[0];
        let (rich_without, rich_with) = summary[1];

        // Weak geometry: the prior must take over, landing near its own accuracy rather than the
        // solve's. Anything close to `prior_err_deg` means the constraint is carrying rotation.
        assert!(
            weak_without > 0.5 && weak_with < 0.3,
            "prior failed to rescue weak rotation: {weak_without} deg without, {weak_with:.3} with"
        );
        // Good geometry: the prior is wrong by 0.15 deg and must NOT drag the solve toward it.
        // Regularise, never override.
        assert!(
            rich_with < rich_without.max(0.05) * 3.0,
            "prior degraded well-conditioned geometry: {rich_without:.4} deg without, {rich_with:.4} with"
        );
        Ok(())
    }

    #[test]
    fn test_single_plane_is_degenerate() -> Result<(), Box<dyn std::error::Error>> {
        let intr = test_intrinsics();
        // one fronto-parallel plane: in-plane translation and in-plane rotation
        // are unobservable for point-to-plane ICP
        let scene = Scene {
            planes: vec![Plane {
                normal: [0.0, 0.0, 1.0],
                d: 2.0,
            }],
            spheres: vec![],
        };
        let depth = render_depth_mm(&scene, &intr, &IDENTITY_ROT, &[0.0; 3]);
        let pyr = RgbdPyramid::from_depth_mm(&depth, &intr, 3)?;

        let result = icp_projective_plane(
            &pyr,
            &pyr,
            IDENTITY_ROT,
            [0.0; 3],
            IcpPlaneCriteria::default(),
        );
        assert!(
            matches!(result, Err(RgbdIcpError::SingularNormalEquations)),
            "single plane must be rejected as degenerate, got {result:?}"
        );
        Ok(())
    }
}
