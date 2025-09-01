//! Levenberg–Marquardt pose refinement for PnP solutions.

use crate::pnp::PnPError;
use glam::{Mat3A, Vec3, Vec3A};

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
}

impl Default for LMParams {
    fn default() -> Self {
        Self {
            max_iters: 20,
            eps: 1e-6,
            lambda_init: 1e-3,
            lambda_mul: 10.0,
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
/// Returns updated `(rvec, t, rmse, num_iters, converged)`.
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
    if n < 3 {
        return Err(PnPError::InsufficientCorrespondences { required: 3, actual: n });
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

    // Helper closures
    let project_all = |x: &[f32; 6]| -> (Vec<f32>, f32) {
        let r = rodrigues(&[x[0], x[1], x[2]]);
        let t_vec = Vec3::new(x[3], x[4], x[5]);
        let r_mat = r;

        let mut residuals = Vec::with_capacity(2 * n);
        let mut sum_sq = 0.0f32;
        for (pw_arr, &uv) in points_world.iter().zip(points_image.iter()) {
            let pw = Vec3::from_array(*pw_arr);
            let pc = r_mat * pw + t_vec;
            let inv_z = 1.0 / pc.z;
            let u_hat = intr_x.dot(pc) * inv_z;
            let v_hat = intr_y.dot(pc) * inv_z;
            let du = u_hat - uv[0];
            let dv = v_hat - uv[1];
            residuals.push(du);
            residuals.push(dv);
            sum_sq += du.mul_add(du, dv * dv);
        }
        (residuals, sum_sq)
    };

    let mut lambda = params.lambda_init;
    let (mut r_base, mut err_sq_base) = project_all(&x);

    let mut iters = 0usize;
    let mut converged = false;

    while iters < params.max_iters {
        iters += 1;
        // Numerical Jacobian J (2N x 6)
        let mut j = vec![0.0f32; 2 * n * 6];
        let step_r = 1e-7f32.max((x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt() * 1e-7);
        let step_t = 1e-6f32.max((x[3] * x[3] + x[4] * x[4] + x[5] * x[5]).sqrt() * 1e-6);

        for k_idx in 0..6 {
            let mut x_pert = x;
            let h = if k_idx < 3 { step_r } else { step_t };
            x_pert[k_idx] += h;
            let (r_pert, _) = project_all(&x_pert);
            for i in 0..(2 * n) {
                // Column k_idx stored in j[i*6 + k_idx]
                j[i * 6 + k_idx] = (r_pert[i] - r_base[i]) / h;
            }
        }

        // Build normal equations: (J^T J + lambda I) delta = -J^T r
        let mut a = [0.0f32; 36];
        let mut b = [0.0f32; 6];
        for r_i in 0..(2 * n) {
            let r_val = r_base[r_i];
            for c in 0..6 {
                let j_ic = j[r_i * 6 + c];
                b[c] += j_ic * r_val;
                for d in 0..6 {
                    a[c * 6 + d] += j_ic * j[r_i * 6 + d];
                }
            }
        }
        // Damping
        for d in 0..6 {
            a[d * 6 + d] += lambda;
        }

        // Solve A delta = -b
        let mut rhs = [-b[0], -b[1], -b[2], -b[3], -b[4], -b[5]];
        let mut a_mat = a;
        if let Some(delta) = solve_6x6(&mut a_mat, &mut rhs) {
            // Tentative update
            let mut x_new = x;
            for i in 0..6 {
                x_new[i] += delta[i];
            }
            let (_r_new, err_sq_new) = project_all(&x_new);
            if err_sq_new < err_sq_base {
                // Accept step
                x = x_new;
                r_base = _r_new;
                if (err_sq_base - err_sq_new) < params.eps {
                    converged = true;
                    err_sq_base = err_sq_new;
                    break;
                }
                err_sq_base = err_sq_new;
                lambda = (lambda / params.lambda_mul).max(1e-12);
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
    rvec.copy_from_slice(&[x[0], x[1], x[2]]);
    t.copy_from_slice(&[x[3], x[4], x[5]]);

    let rmse = (err_sq_base / (2.0 * n as f32)).sqrt();
    Ok((rmse, iters, converged))
}

// Rodrigues' rotation formula: axis-angle (scaled axis) to rotation matrix.
fn rodrigues(rvec: &[f32; 3]) -> Mat3A {
    let rx = rvec[0] as f32;
    let ry = rvec[1] as f32;
    let rz = rvec[2] as f32;
    let theta2 = rx * rx + ry * ry + rz * rz;
    if theta2 < 1e-16 {
        // First-order approximation: R ≈ I + [w]_x
        let wx = skew(Vec3::new(rx, ry, rz));
        return Mat3A::IDENTITY + wx;
    }
    let theta = theta2.sqrt();
    let w = Vec3::new(rx / theta, ry / theta, rz / theta);
    let s = theta.sin();
    let c = theta.cos();
    let wx = skew(w);
    let wx2 = wx * wx;
    Mat3A::IDENTITY + wx * s + wx2 * (1.0 - c)
}

#[inline]
fn skew(w: Vec3) -> Mat3A {
    Mat3A::from_cols(
        Vec3A::new(0.0, w.z, -w.y),
        Vec3A::new(-w.z, 0.0, w.x),
        Vec3A::new(w.y, -w.x, 0.0),
    )
}

// Dense 6x6 solver using Gaussian elimination with partial pivoting.
fn solve_6x6(a: &mut [f32; 36], b: &mut [f32; 6]) -> Option<[f32; 6]> {
    // Augment A|b in-place operations via indices
    // Perform elimination
    for i in 0..6 {
        // Pivot
        let mut piv = i;
        let mut max_val = a[i * 6 + i].abs();
        for r in (i + 1)..6 {
            let v = a[r * 6 + i].abs();
            if v > max_val {
                max_val = v;
                piv = r;
            }
        }
        if max_val < 1e-12 {
            return None;
        }
        if piv != i {
            for c in i..6 {
                a.swap(i * 6 + c, piv * 6 + c);
            }
            b.swap(i, piv);
        }
        // Normalize row i
        let diag = a[i * 6 + i];
        for c in i..6 {
            a[i * 6 + c] /= diag;
        }
        b[i] /= diag;
        // Eliminate below
        for r in (i + 1)..6 {
            let factor = a[r * 6 + i];
            if factor == 0.0 {
                continue;
            }
            for c in i..6 {
                a[r * 6 + c] -= factor * a[i * 6 + c];
            }
            b[r] -= factor * b[i];
        }
    }
    // Back substitution
    for i in (0..6).rev() {
        for r in 0..i {
            let factor = a[r * 6 + i];
            if factor != 0.0 {
                a[r * 6 + i] = 0.0;
                b[r] -= factor * b[i];
            }
        }
    }
    Some(*b)
}


