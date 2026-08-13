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
}

impl Default for IcpPlaneCriteria {
    fn default() -> Self {
        Self {
            iters_per_level: vec![10, 7, 5],
            update_tolerance: 1e-6,
            max_dist_m: 0.10,
            max_normal_angle_rad: 30.0_f64.to_radians(),
            huber_delta_m: 0.02,
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
    /// Weakest normal-equation pivot relative to its block, from the final solve: how well the
    /// geometry pins the least-constrained direction. Near zero means the pose can slide with no
    /// residual penalty — a wall filling the view yields a perfect [`Self::rmse`] and a perfect
    /// [`Self::inlier_fraction`] while the estimate drifts along the surface, so neither of those
    /// can detect it. Compare against [`PIVOT_RTOL`], which is only the hard-failure floor.
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
    let mut observability = 0.0;

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
            let eqs = accumulate_level(src, tgt, &rotation, &translation, &criteria);
            if eqs.num_inliers < 6 {
                return Err(RgbdIcpError::TooFewCorrespondences(eqs.num_inliers));
            }
            // solve A x = -b for the twist x = [w, dt]
            let neg_b = eqs.b.map(|v| -v);
            let (x, pivot) =
                cholesky_solve_6x6(&eqs.a, &neg_b).ok_or(RgbdIcpError::SingularNormalEquations)?;
            // Report the finest level's last solve: the geometry the pose actually rests on.
            observability = pivot;

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
        observability,
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
fn cholesky_solve_6x6(a: &[[f64; 6]; 6], b: &[f64; 6]) -> Option<([f64; 6], f64)> {
    let max_diag_rot = (0..3).map(|i| a[i][i]).fold(0.0, f64::max);
    let max_diag_trans = (3..6).map(|i| a[i][i]).fold(0.0, f64::max);
    if max_diag_rot <= 0.0 || max_diag_trans <= 0.0 {
        return None;
    }

    // A = L L^T
    let mut weakest_pivot = f64::INFINITY;
    let mut l = [[0.0; 6]; 6];
    for i in 0..6 {
        for j in 0..=i {
            let mut sum = a[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            if i == j {
                let block_scale = if i < 3 { max_diag_rot } else { max_diag_trans };
                weakest_pivot = weakest_pivot.min(sum / block_scale);
                if sum < PIVOT_RTOL * block_scale {
                    return None;
                }
                l[i][j] = sum.sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }

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
    Some((x, weakest_pivot))
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

        // Measured: 6.4e-7 on the weak scene against 6.5e-2 on the rich one — five orders of
        // magnitude, so the margin is not a tuning artefact.
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
