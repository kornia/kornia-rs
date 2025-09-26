//! Levenberg–Marquardt pose refinement for PnP solutions.

use crate::pnp::PnPError;
use glam::{Vec3, Vec3A};
use kornia_lie::so3::SO3;

// Levenberg–Marquardt defaults and numerical constants
const DEFAULT_MAX_ITERS: usize = 20;
const DEFAULT_EPS: f32 = 1e-6;
const DEFAULT_LAMBDA_INIT: f32 = 1e-3;
const DEFAULT_LAMBDA_MUL: f32 = 10.0;

// Problem sizes
const SOLVER_DIM: usize = 6; // [rx, ry, rz, tx, ty, tz]
const JTJ_SIZE: usize = SOLVER_DIM * SOLVER_DIM; // 36
const RESIDUAL_DIM: usize = 2; // [du, dv]

// Finite-difference steps
const FINITE_DIFF_ROT_STEP: f32 = 1e-4; // radians
const FINITE_DIFF_TRANS_BASE_STEP: f32 = 1e-4; // world units (scaled)

// Numerical thresholds
const NUMERICAL_EPS: f32 = 1e-12; // generic tiny epsilon for stability
const MIN_CORRESPONDENCES: usize = 3;
const MIN_TRANSLATION_SCALE: f32 = 1.0;

/// Parameters controlling the LM pose refinement.
#[derive(Debug, Clone)]
pub struct LMParams {
    /// Maximum number of LM iterations.
    pub max_iters: usize,
    /// Convergence threshold on squared reprojection error decrease.
    pub eps: f32,
    /// Initial damping factor (lambda).
    pub lambda_init: f32,
    /// Multiplicative factor to increase/decrease lambda.
    pub lambda_mul: f32,
    /// Finite-difference step for rotation parameters (radians).
    pub fd_rot_step: f32,
    /// Base finite-difference step for translation parameters (scaled by translation magnitude).
    pub fd_trans_base_step: f32,
}

impl Default for LMParams {
    fn default() -> Self {
        Self {
            max_iters: DEFAULT_MAX_ITERS,
            eps: DEFAULT_EPS,
            lambda_init: DEFAULT_LAMBDA_INIT,
            lambda_mul: DEFAULT_LAMBDA_MUL,
            fd_rot_step: FINITE_DIFF_ROT_STEP,
            fd_trans_base_step: FINITE_DIFF_TRANS_BASE_STEP,
        }
    }
}

/// Refine a pose (rvec, t) with Levenberg–Marquardt to minimize pixel reprojection error.
///
/// - `points_world`: World points (N,3)
/// - `points_image`: Pixel points (N,2)
/// - `k`: Intrinsics 3x3
/// - `rvec`: Initial axis-angle rotation (input/output)
/// - `t`: Initial translation (input/output)
///
/// Returns `(rmse, num_iters, converged)` and writes refined `rvec` and `t` in place.
pub fn refine_pose_lm(
    points_world: &[[f32; 3]],
    points_image: &[[f32; 2]],
    k: &[[f32; 3]; 3],
    rvec: &mut [f32; 3],
    t: &mut [f32; 3],
    params: &LMParams,
) -> Result<(f32, usize, bool), PnPError> {
    if points_world.len() != points_image.len() {
        return Err(PnPError::MismatchedArrayLengths {
            left_name: "world points",
            left_len: points_world.len(),
            right_name: "image points",
            right_len: points_image.len(),
        });
    }

    let n = points_world.len();
    if n < MIN_CORRESPONDENCES {
        return Err(PnPError::InsufficientCorrespondences {
            required: MIN_CORRESPONDENCES,
            actual: n,
        });
    }

    // Parameters vector x = [rx, ry, rz, tx, ty, tz]
    let mut x = [rvec[0], rvec[1], rvec[2], t[0], t[1], t[2]];

    // Precompute intrinsics vectors for fast projection
    let fx = k[0][0];
    let fy = k[1][1];
    let cx = k[0][2];
    let cy = k[1][2];
    let intr_x = Vec3::new(fx, 0.0, cx);
    let intr_y = Vec3::new(0.0, fy, cy);

    // Projection utility writing residuals in-place to avoid allocations
    let mut residuals = vec![0.0f32; RESIDUAL_DIM * n];
    let mut residuals_p = vec![0.0f32; RESIDUAL_DIM * n];
    let mut residuals_m = vec![0.0f32; RESIDUAL_DIM * n];

    let project_all_in_place = |x: &[f32; 6], out: &mut [f32]| -> f32 {
        let r_mat = SO3::exp(Vec3A::from_array([x[0], x[1], x[2]])).matrix();
        let t_vec = Vec3::new(x[3], x[4], x[5]);

        let mut sum_sq = 0.0f32;
        for (i, (pw_arr, &uv)) in points_world.iter().zip(points_image.iter()).enumerate() {
            let pw = Vec3::from_array(*pw_arr);
            let pc = r_mat * pw + t_vec;
            let inv_z = 1.0 / pc.z;
            let u_hat = intr_x.dot(pc) * inv_z;
            let v_hat = intr_y.dot(pc) * inv_z;
            let du = u_hat - uv[0];
            let dv = v_hat - uv[1];
            out[2 * i] = du;
            out[2 * i + 1] = dv;
            sum_sq += du.mul_add(du, dv * dv);
        }
        sum_sq
    };

    let mut lambda = params.lambda_init;
    let mut err_sq_base = project_all_in_place(&x, &mut residuals);

    let mut iters = 0usize;
    let mut converged = false;

    // Preallocate J, A and b once
    let mut j = vec![0.0f32; RESIDUAL_DIM * n * SOLVER_DIM];
    let mut a = [0.0f32; JTJ_SIZE];
    let mut b = [0.0f32; SOLVER_DIM];

    while iters < params.max_iters {
        iters += 1;
        // Reset accumulators
        j.fill(0.0);
        a.fill(0.0);
        b.fill(0.0);
        let t_scale = x[3]
            .abs()
            .max(x[4].abs())
            .max(x[5].abs())
            .max(MIN_TRANSLATION_SCALE);
        let h_trans = params.fd_trans_base_step * t_scale; // world units

        for k_idx in 0..SOLVER_DIM {
            // Central differences
            let h = if k_idx < 3 {
                params.fd_rot_step
            } else {
                h_trans
            };
            let mut x_plus = x;
            let mut x_minus = x;
            x_plus[k_idx] += h;
            x_minus[k_idx] -= h;
            let _ = project_all_in_place(&x_plus, &mut residuals_p);
            let _ = project_all_in_place(&x_minus, &mut residuals_m);
            for i in 0..(RESIDUAL_DIM * n) {
                j[i * SOLVER_DIM + k_idx] = (residuals_p[i] - residuals_m[i]) / (2.0 * h);
            }
        }

        // Build normal equations: (J^T J + lambda I) delta = -J^T r
        for r_i in 0..(RESIDUAL_DIM * n) {
            let r_val = residuals[r_i];
            for c in 0..SOLVER_DIM {
                let j_ic = j[r_i * SOLVER_DIM + c];
                b[c] += j_ic * r_val;
                for d in 0..SOLVER_DIM {
                    a[c * SOLVER_DIM + d] += j_ic * j[r_i * SOLVER_DIM + d];
                }
            }
        }
        // Damping
        for d in 0..SOLVER_DIM {
            a[d * SOLVER_DIM + d] += lambda;
        }

        // Solve A delta = -b
        let mut rhs = [0.0f32; SOLVER_DIM];
        for i in 0..SOLVER_DIM {
            rhs[i] = -b[i];
        }
        let mut a_mat = a;
        if let Some(delta) = solve_6x6(&mut a_mat, &mut rhs) {
            // Tentative update
            let mut x_new = x;
            for i in 0..SOLVER_DIM {
                x_new[i] += delta[i];
            }
            let err_sq_new = project_all_in_place(&x_new, &mut residuals_p);
            if err_sq_new < err_sq_base {
                // Accept step
                x = x_new;
                residuals.copy_from_slice(&residuals_p);
                if (err_sq_base - err_sq_new) < params.eps {
                    converged = true;
                    err_sq_base = err_sq_new;
                    break;
                }
                err_sq_base = err_sq_new;
                lambda = (lambda / params.lambda_mul).max(NUMERICAL_EPS);
            } else {
                // Reject step, increase damping
                lambda *= params.lambda_mul;
            }
        } else {
            // Singular system, increase damping
            lambda *= params.lambda_mul;
        }
    }

    // Write back results
    rvec.copy_from_slice(&x[0..3]);
    t.copy_from_slice(&x[3..6]);

    let rmse = (err_sq_base / (RESIDUAL_DIM as f32 * n as f32)).sqrt();
    Ok((rmse, iters, converged))
}

// Dense 6x6 solver using Gaussian elimination with partial pivoting.
fn solve_6x6(a: &mut [f32; JTJ_SIZE], b: &mut [f32; SOLVER_DIM]) -> Option<[f32; SOLVER_DIM]> {
    // Augment A|b in-place operations via indices
    // Perform elimination
    for i in 0..SOLVER_DIM {
        // Pivot
        let mut piv = i;
        let mut max_val = a[i * SOLVER_DIM + i].abs();
        for r in (i + 1)..SOLVER_DIM {
            let v = a[r * SOLVER_DIM + i].abs();
            if v > max_val {
                max_val = v;
                piv = r;
            }
        }
        if max_val < NUMERICAL_EPS {
            return None;
        }
        if piv != i {
            for c in i..SOLVER_DIM {
                a.swap(i * SOLVER_DIM + c, piv * SOLVER_DIM + c);
            }
            b.swap(i, piv);
        }
        // Normalize row i
        let diag = a[i * SOLVER_DIM + i];
        for c in i..SOLVER_DIM {
            a[i * SOLVER_DIM + c] /= diag;
        }
        b[i] /= diag;
        // Eliminate below
        for r in (i + 1)..SOLVER_DIM {
            let factor = a[r * SOLVER_DIM + i];
            if factor == 0.0 {
                continue;
            }
            for c in i..SOLVER_DIM {
                a[r * SOLVER_DIM + c] -= factor * a[i * SOLVER_DIM + c];
            }
            b[r] -= factor * b[i];
        }
    }
    // Back substitution
    for i in (0..SOLVER_DIM).rev() {
        for r in 0..i {
            let factor = a[r * SOLVER_DIM + i];
            if factor != 0.0 {
                a[r * SOLVER_DIM + i] = 0.0;
                b[r] -= factor * b[i];
            }
        }
    }
    Some(*b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pnp::PnPSolver;
    use crate::{EPnP, EPnPParams, PnPError};
    const TEST_RMSE_TOL: f32 = 1e-4;

    #[test]
    fn test_refine_lm_reduces_rmse() -> Result<(), PnPError> {
        let points_world: [[f32; 3]; 6] = [
            [0.0315, 0.03333, -0.10409],
            [-0.0315, 0.03333, -0.10409],
            [0.0, -0.00102, -0.12977],
            [0.02646, -0.03167, -0.1053],
            [-0.02646, -0.031667, -0.1053],
            [0.0, 0.04515, -0.11033],
        ];
        let points_image: [[f32; 2]; 6] = [
            [722.96466, 502.0828],
            [669.88837, 498.61877],
            [707.0025, 478.48975],
            [728.05634, 447.56918],
            [682.6069, 443.91776],
            [696.4414, 511.96442],
        ];
        let k: [[f32; 3]; 3] = [[800.0, 0.0, 640.0], [0.0, 800.0, 480.0], [0.0, 0.0, 1.0]];

        // Baseline EPnP
        let res_epnp = EPnP::solve(&points_world, &points_image, &k, &EPnPParams::default())?;
        let rmse0 = res_epnp.reproj_rmse.expect("EPnP should report RMSE");

        // EPnP + LM refinement
        let res_lm = EPnP::solve(
            &points_world,
            &points_image,
            &k,
            &EPnPParams {
                refine_lm: Some(LMParams::default()),
                ..Default::default()
            },
        )?;
        let rmse1 = res_lm.reproj_rmse.expect("LM should report RMSE");

        assert!(
            rmse1 <= rmse0 + TEST_RMSE_TOL,
            "LM RMSE should not be worse: {} vs {}",
            rmse1,
            rmse0
        );
        Ok(())
    }
}
