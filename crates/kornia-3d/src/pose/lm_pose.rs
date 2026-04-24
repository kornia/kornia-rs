//! # Levenberg-Marquardt refinement of relative pose (R, t) from two-view
//!
//! Refines an initial (R, t) — typically the cheirality winner out of the four
//! essential-matrix decompositions — by minimizing the sum of squared Sampson
//! distances over the supplied inlier correspondences. This closes the accuracy
//! gap between the 8-point + linear-cheirality pipeline and OpenCV's USAC, which
//! carries out an equivalent non-linear polishing step.
//!
//! ## Parameterization
//!
//! The translation scale is unobservable from two views alone (see
//! [`crate::pose::essential`]), so the relative pose lives on the
//! **SO(3) × S²** manifold. We optimize over the 5-dimensional tangent:
//!
//! - `[δω_x, δω_y, δω_z]` — right-perturbation on SO(3): `R ⊕ δω = R · exp(δω)`.
//! - `[δt_1, δt_2]` — tangent displacement on the unit sphere at current `t`,
//!   expressed in an orthonormal basis `{b1, b2} ⊥ t`. The retraction is
//!   `t ⊕ δt = normalize(t + δt_1·b1 + δt_2·b2)`.
//!
//! This parameterization is singularity-free in a neighborhood of the current
//! estimate and removes the gauge freedoms (global rotation + translation-scale
//! ambiguity) that would otherwise make the normal equations rank-deficient.
//!
//! ## Jacobian
//!
//! Computed via **central finite differences** with step `h = 1e-6`. With only
//! 5 columns the Jacobian build is cheap — the inner loop rebuilds F from (R, t)
//! and evaluates N Sampson distances per column × 2 (central). For a typical
//! ~100-inlier MH_01 run this is ~100 µs, negligible versus the ~2 ms of the
//! enclosing 8-point RANSAC. Switching to analytical derivatives would save
//! <5% end-to-end; we'd rather keep the code readable.
//!
//! ## Safety net
//!
//! The refiner **never returns a worse pose than its input**. If no accepted
//! step ever lowers the cost, the function returns the input unchanged. This
//! matters because RANSAC's cheirality winner is already a decent local
//! optimum; a divergent LM call shouldn't regress it.

use crate::pose::fundamental::sampson_distance;
use faer::prelude::SpSolver;
use kornia_algebra::{Mat3F64, Vec2F64, Vec3F64};

/// Configuration for [`refine_pose_lm`].
///
/// Default values are tuned for bundle-level convergence on real data in 3–6
/// accepted iterations. Tightening tolerances or raising iteration caps yields
/// diminishing returns once the gradient falls below noise floor.
#[derive(Clone, Copy, Debug)]
pub struct LmPoseConfig {
    /// Hard cap on LM iterations (both accepted and rejected). Default 10.
    pub max_iters: usize,
    /// Initial Levenberg damping λ. Default 1e-3.
    pub initial_lambda: f64,
    /// Multiplier applied to λ on a rejected step (grow toward steepest descent).
    /// Default 10.0.
    pub lambda_up: f64,
    /// Multiplier applied to λ on an accepted step (shrink toward Gauss-Newton).
    /// Default 0.5.
    pub lambda_down: f64,
    /// Stop when `‖J^T r‖_∞ < gradient_tol`. Default 1e-9.
    pub gradient_tol: f64,
    /// Stop when `‖δ‖ < step_tol`. Default 1e-9.
    pub step_tol: f64,
    /// Stop when relative cost decrease `|Δcost|/|cost| < cost_tol`. Default 1e-12.
    pub cost_tol: f64,
}

impl Default for LmPoseConfig {
    fn default() -> Self {
        Self {
            max_iters: 10,
            initial_lambda: 1e-3,
            lambda_up: 10.0,
            lambda_down: 0.5,
            gradient_tol: 1e-9,
            step_tol: 1e-9,
            cost_tol: 1e-12,
        }
    }
}

/// Skew-symmetric cross-product matrix of a 3-vector.
///
/// `hat(v) * w = v × w` for any `w ∈ ℝ³`.
#[inline]
fn hat(v: Vec3F64) -> Mat3F64 {
    // Column-major for Mat3F64::from_cols.
    Mat3F64::from_cols(
        Vec3F64::new(0.0, v.z, -v.y),
        Vec3F64::new(-v.z, 0.0, v.x),
        Vec3F64::new(v.y, -v.x, 0.0),
    )
}

/// Closed-form Rodrigues exponential: so(3) → SO(3).
///
/// Small-angle guard (`θ < 1e-8`) falls back to the first-order expansion
/// `exp(ω̂) ≈ I + ω̂` to avoid 0/0 in the sin/(1−cos) terms.
#[inline]
fn so3_exp(w: Vec3F64) -> Mat3F64 {
    let theta_sq = w.x * w.x + w.y * w.y + w.z * w.z;
    let theta = theta_sq.sqrt();
    let wx = hat(w);
    if theta < 1e-8 {
        // First-order; higher-order O(θ²) terms negligible below this threshold.
        return Mat3F64::IDENTITY + wx;
    }
    let a = theta.sin() / theta;
    let b = (1.0 - theta.cos()) / theta_sq;
    // I + a·ω̂ + b·ω̂²
    let wx_sq = wx * wx;
    Mat3F64::IDENTITY + wx * a + wx_sq * b
}

#[inline]
fn cross(a: Vec3F64, b: Vec3F64) -> Vec3F64 {
    Vec3F64::new(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    )
}

/// Build an orthonormal basis `{b1, b2}` of the tangent plane to S² at `t`.
///
/// `t` is assumed unit-length. Uses a Gram-Schmidt step against the world axis
/// least aligned with `t` to keep the basis well-conditioned.
#[inline]
fn tangent_basis(t: Vec3F64) -> (Vec3F64, Vec3F64) {
    // Pick the world axis with the smallest |component| along t — cross products
    // with the *most perpendicular* axis are the best-conditioned.
    let ax = t.x.abs();
    let ay = t.y.abs();
    let az = t.z.abs();
    let seed = if ax <= ay && ax <= az {
        Vec3F64::new(1.0, 0.0, 0.0)
    } else if ay <= az {
        Vec3F64::new(0.0, 1.0, 0.0)
    } else {
        Vec3F64::new(0.0, 0.0, 1.0)
    };
    let b1 = cross(t, seed).normalize();
    let b2 = cross(t, b1); // already unit since t and b1 are orthonormal
    (b1, b2)
}

/// Build the fundamental matrix from (R, t) and camera intrinsics.
///
/// `E = [t]_× R`, `F = K2⁻ᵀ E K1⁻¹`. `t` is treated as a direction (scale
/// irrelevant for F up to a global factor).
#[inline]
fn fundamental_from_rt(r: &Mat3F64, t: &Vec3F64, k1_inv: &Mat3F64, k2_inv_t: &Mat3F64) -> Mat3F64 {
    let e = hat(*t) * *r;
    *k2_inv_t * e * *k1_inv
}

/// Sum of Sampson distances over the inlier set, given F.
#[inline]
fn sum_sampson(f: &Mat3F64, x1: &[Vec2F64], x2: &[Vec2F64]) -> f64 {
    let mut s = 0.0;
    for (p1, p2) in x1.iter().zip(x2.iter()) {
        s += sampson_distance(f, p1, p2);
    }
    s
}

/// Per-correspondence residual r_i = sqrt(sampson_distance). The sum of squares
/// of this residual equals the Sampson cost exactly.
#[inline]
fn residuals(f: &Mat3F64, x1: &[Vec2F64], x2: &[Vec2F64], out: &mut [f64]) {
    for (i, (p1, p2)) in x1.iter().zip(x2.iter()).enumerate() {
        let d = sampson_distance(f, p1, p2);
        out[i] = if d > 0.0 { d.sqrt() } else { 0.0 };
    }
}

/// Refine (R, t) by LM on the Sampson cost over `x1_inl` / `x2_inl` inliers.
///
/// Parameterized on SO(3) × S² — translation magnitude is unobservable from
/// two-view geometry alone. Returns the refined `(R, t_unit)` or the input
/// unchanged if the optimization failed to improve the cost (safety net — LM
/// must never regress the cheirality winner).
///
/// `k1` / `k2` are the intrinsic matrices; inliers are in pixel coordinates.
pub fn refine_pose_lm(
    r: Mat3F64,
    t: Vec3F64,
    x1_inl: &[Vec2F64],
    x2_inl: &[Vec2F64],
    k1: &Mat3F64,
    k2: &Mat3F64,
    cfg: &LmPoseConfig,
) -> (Mat3F64, Vec3F64) {
    let n = x1_inl.len();
    if n < 6 || x2_inl.len() != n {
        // 5 DOF — need at least 5 equations; take 6 for a tiny margin. Bail out
        // on shape mismatch.
        return (r, t);
    }

    // Precompute the K transforms that are fixed across iterations.
    let k1_inv = k1.inverse();
    let k2_inv_t = k2.inverse().transpose();

    // Working state: current best (R, t) and its cost. Invariant: these
    // always correspond to the lowest-cost pose seen so far (including the
    // input). Never return worse than this.
    let mut r_cur = r;
    let mut t_cur = t.normalize();
    let f_cur = fundamental_from_rt(&r_cur, &t_cur, &k1_inv, &k2_inv_t);
    let mut cost_cur = sum_sampson(&f_cur, x1_inl, x2_inl);
    let cost_initial = cost_cur;

    let mut lambda = cfg.initial_lambda;

    // Scratch buffers reused across iterations.
    let mut res = vec![0.0_f64; n];
    let mut res_pert = vec![0.0_f64; n];
    let mut res_minus = vec![0.0_f64; n];
    let mut jac = vec![0.0_f64; n * 5]; // column-major: jac[c*n + i] = J[i, c]

    for _iter in 0..cfg.max_iters {
        // 1. Evaluate residuals at current pose.
        let f_at_cur = fundamental_from_rt(&r_cur, &t_cur, &k1_inv, &k2_inv_t);
        residuals(&f_at_cur, x1_inl, x2_inl, &mut res);

        // 2. Build tangent basis at current t (S² retraction).
        let (b1, b2) = tangent_basis(t_cur);

        // 3. Finite-difference Jacobian (central scheme). Five columns —
        //    first three: SO(3) right-perturbation, last two: S² tangent.
        //    Step h = 1e-6: small enough that truncation error is < residual
        //    noise floor on pixel-scale Sampson costs, large enough that f64
        //    cancellation doesn't bite.
        let h_fd = 1e-6_f64;
        let two_h_inv = 1.0 / (2.0 * h_fd);
        for col in 0..5 {
            // +h perturbation.
            let (r_p, t_p) = apply_tangent_step(r_cur, t_cur, b1, b2, col, h_fd);
            let f_p = fundamental_from_rt(&r_p, &t_p, &k1_inv, &k2_inv_t);
            residuals(&f_p, x1_inl, x2_inl, &mut res_pert);
            // -h perturbation.
            let (r_m, t_m) = apply_tangent_step(r_cur, t_cur, b1, b2, col, -h_fd);
            let f_m = fundamental_from_rt(&r_m, &t_m, &k1_inv, &k2_inv_t);
            residuals(&f_m, x1_inl, x2_inl, &mut res_minus);
            let col_base = col * n;
            for i in 0..n {
                jac[col_base + i] = (res_pert[i] - res_minus[i]) * two_h_inv;
            }
        }

        // 4. Normal equations: H = JᵀJ (5×5), g = Jᵀr (5).
        let mut h_mat = [[0.0_f64; 5]; 5];
        let mut g_vec = [0.0_f64; 5];
        for c1 in 0..5 {
            let col1 = &jac[c1 * n..c1 * n + n];
            let mut g = 0.0;
            for i in 0..n {
                g += col1[i] * res[i];
            }
            g_vec[c1] = g;
            for c2 in c1..5 {
                let col2 = &jac[c2 * n..c2 * n + n];
                let mut h = 0.0;
                for i in 0..n {
                    h += col1[i] * col2[i];
                }
                h_mat[c1][c2] = h;
                h_mat[c2][c1] = h;
            }
        }

        // Gradient convergence test: ‖JᵀR‖_∞.
        let grad_inf = g_vec.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
        if grad_inf < cfg.gradient_tol {
            break;
        }

        // 5. Solve (H + λ·diag(H)) δ = -g via faer LU. Levenberg-style damping
        //    scales each diagonal by its own magnitude — more robust than a
        //    flat λ·I when the columns have wildly different scales (S² and
        //    SO(3) tangents often do).
        let mut a = faer::Mat::<f64>::zeros(5, 5);
        let mut rhs = faer::Mat::<f64>::zeros(5, 1);
        for r_i in 0..5 {
            for c_i in 0..5 {
                let mut v = h_mat[r_i][c_i];
                if r_i == c_i {
                    v += lambda * h_mat[r_i][r_i].abs().max(1e-12);
                }
                unsafe {
                    a.write_unchecked(r_i, c_i, v);
                }
            }
            unsafe {
                rhs.write_unchecked(r_i, 0, -g_vec[r_i]);
            }
        }
        let sol = a.partial_piv_lu().solve(&rhs);
        let delta = [
            sol.read(0, 0),
            sol.read(1, 0),
            sol.read(2, 0),
            sol.read(3, 0),
            sol.read(4, 0),
        ];
        if !delta.iter().all(|x: &f64| x.is_finite()) {
            // Singular system — bump λ and retry.
            lambda *= cfg.lambda_up;
            continue;
        }

        let step_norm = (delta.iter().map(|x| x * x).sum::<f64>()).sqrt();
        if step_norm < cfg.step_tol {
            break;
        }

        // 6. Retract and evaluate candidate cost.
        let (r_try, t_try) = retract(r_cur, t_cur, b1, b2, delta);
        let f_try = fundamental_from_rt(&r_try, &t_try, &k1_inv, &k2_inv_t);
        let cost_try = sum_sampson(&f_try, x1_inl, x2_inl);

        if cost_try.is_finite() && cost_try < cost_cur {
            // Accepted. Shrink λ; check relative cost tolerance.
            let rel_drop = (cost_cur - cost_try).abs() / cost_cur.abs().max(1e-300);
            r_cur = r_try;
            t_cur = t_try;
            cost_cur = cost_try;
            lambda *= cfg.lambda_down;
            if rel_drop < cfg.cost_tol {
                break;
            }
        } else {
            // Rejected — keep (r_cur, t_cur), bump λ toward steepest descent.
            lambda *= cfg.lambda_up;
            // Sanity cap — don't let λ blow up indefinitely.
            if lambda > 1e16 {
                break;
            }
        }
    }

    if cost_cur.is_finite() && cost_cur <= cost_initial {
        (r_cur, t_cur)
    } else {
        (r, t.normalize())
    }
}

/// Apply a step in the tangent-space coordinate `col` of magnitude `h`. Used to
/// build finite-difference Jacobian columns.
#[inline]
fn apply_tangent_step(
    r: Mat3F64,
    t: Vec3F64,
    b1: Vec3F64,
    b2: Vec3F64,
    col: usize,
    h: f64,
) -> (Mat3F64, Vec3F64) {
    let mut delta = [0.0_f64; 5];
    delta[col] = h;
    retract(r, t, b1, b2, delta)
}

/// Apply the full 5-D tangent step to (R, t) using the right-perturbation SO(3)
/// retraction and the S² re-normalize retraction.
#[inline]
fn retract(
    r: Mat3F64,
    t: Vec3F64,
    b1: Vec3F64,
    b2: Vec3F64,
    delta: [f64; 5],
) -> (Mat3F64, Vec3F64) {
    let omega = Vec3F64::new(delta[0], delta[1], delta[2]);
    let r_new = r * so3_exp(omega);
    // b1 * δ1 + b2 * δ2 — glam supports Vec * scalar.
    let b1s = Vec3F64::new(b1.x * delta[3], b1.y * delta[3], b1.z * delta[3]);
    let b2s = Vec3F64::new(b2.x * delta[4], b2.y * delta[4], b2.z * delta[4]);
    let t_raw = Vec3F64::new(t.x + b1s.x + b2s.x, t.y + b1s.y + b2s.y, t.z + b1s.z + b2s.z);
    let t_new = t_raw.normalize();
    (r_new, t_new)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_so3_exp_identity() {
        let m = so3_exp(Vec3F64::ZERO);
        let diff: [f64; 9] = (m - Mat3F64::IDENTITY).into();
        for d in diff {
            assert!(d.abs() < 1e-15);
        }
    }

    #[test]
    fn test_so3_exp_small_angle() {
        let w = Vec3F64::new(1e-10, -2e-10, 5e-11);
        let m = so3_exp(w);
        // Should be very close to I + hat(w).
        let expected = Mat3F64::IDENTITY + hat(w);
        let d: [f64; 9] = (m - expected).into();
        for v in d {
            assert!(v.abs() < 1e-18);
        }
    }

    #[test]
    fn test_tangent_basis_orthonormal() {
        let t = Vec3F64::new(0.2422, -0.2330, 0.9418).normalize();
        let (b1, b2) = tangent_basis(t);
        assert!(t.dot(b1).abs() < 1e-12);
        assert!(t.dot(b2).abs() < 1e-12);
        assert!(b1.dot(b2).abs() < 1e-12);
        assert!((b1.length() - 1.0).abs() < 1e-12);
        assert!((b2.length() - 1.0).abs() < 1e-12);
    }
}
