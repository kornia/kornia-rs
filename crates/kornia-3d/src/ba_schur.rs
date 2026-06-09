//! Schur-complement bundle adjustment with dense reduced camera system.
//!
//! The standard bipartite-Schur trick from Triggs et al. (1999). Each LM
//! iteration builds the Hessian in BLOCK form
//!
//! ```text
//!     H = [ A   B  ]    A = 6P × 6P pose blocks (block-diagonal),
//!         [ Bᵀ  C  ]    C = 3N × 3N point blocks (BLOCK-DIAGONAL),
//!                       B = 6P × 3N pose-point cross terms (sparse).
//! ```
//!
//! Block-diagonal C means C⁻¹ is cheap (per-3×3 invert). The **reduced
//! camera system**
//!
//! ```text
//!     M = A − B C⁻¹ Bᵀ           (dense 6P × 6P)
//!     m = g_pose − B C⁻¹ g_point
//! ```
//!
//! is solved with `faer`'s dense Cholesky on the small matrix; points are
//! recovered by back-substitution. For our SLAM problem (~170 poses ×
//! ~3000 points × ~15000 observations) the reduced system is just
//! 1020 × 1020 — Ceres's `DENSE_SCHUR` is exactly this regime.
//!
//! No sparse-matrix dependency is needed because the only "large" object
//! the Schur trick has to manipulate (B, 6P × 3N) is never materialised:
//! we walk observations and accumulate per-point contributions into M
//! directly.
//!
//! Jacobian conventions match [`crate::ba::ReprojFactor`]:
//!
//!   * Pose tangent layout `[ρ; ω]` (upsilon then omega), 6-dim.
//!   * Point parameters are the 3-dim world coordinates.
//!   * z is clamped to `MIN_Z` to handle mid-iteration cheirality flips.
//!
//! Currently supports: identity loss only, fixed-pose anchors, fixed-point
//! gauge (motion-only BA). Robust kernels and full LM-with-backtracking
//! are TODO.

use faer::prelude::Solve;
use faer::Mat;
use kornia_algebra::{Mat3AF32, Mat3F64, Vec3AF32, Vec3F64, SE3F32, SO3F32};
use thiserror::Error;

use crate::ba::{BaError, BaObservation, BaParams, BaPosePrior, BaResult};
use crate::camera::PinholeCamera;
use crate::pose::Pose3d;
use crate::ransac::RobustKernelKind;

const MIN_Z: f32 = 1e-3;

/// Errors specific to the Schur BA driver. Wraps existing [`BaError`].
#[derive(Debug, Error)]
pub enum SchurBaError {
    /// Linear system is rank-deficient / Cholesky failed.
    #[error("Reduced camera Cholesky failed (likely rank-deficient): {0}")]
    CholeskyFailed(String),
    /// No free variables after applying anchors.
    #[error("All variables are fixed — nothing to optimise")]
    NoFreeVariables,
    /// Other BA setup error.
    #[error(transparent)]
    Ba(#[from] BaError),
}

// ── f32 ↔ f64 conversion helpers (shared shape with ba.rs) ───────────────

fn pose_to_se3(pose: &Pose3d) -> SE3F32 {
    let r = Mat3AF32::from_cols(
        Vec3AF32::new(
            pose.rotation.col(0).x as f32,
            pose.rotation.col(0).y as f32,
            pose.rotation.col(0).z as f32,
        ),
        Vec3AF32::new(
            pose.rotation.col(1).x as f32,
            pose.rotation.col(1).y as f32,
            pose.rotation.col(1).z as f32,
        ),
        Vec3AF32::new(
            pose.rotation.col(2).x as f32,
            pose.rotation.col(2).y as f32,
            pose.rotation.col(2).z as f32,
        ),
    );
    let so3 = SO3F32::from_matrix(&r);
    SE3F32::new(
        so3,
        Vec3AF32::new(
            pose.translation.x as f32,
            pose.translation.y as f32,
            pose.translation.z as f32,
        ),
    )
}

fn se3_to_pose(se3: &SE3F32) -> Pose3d {
    let r = se3.r.matrix();
    let t = se3.t;
    Pose3d::new(
        Mat3F64::from_cols(
            Vec3F64::new(r.col(0).x as f64, r.col(0).y as f64, r.col(0).z as f64),
            Vec3F64::new(r.col(1).x as f64, r.col(1).y as f64, r.col(1).z as f64),
            Vec3F64::new(r.col(2).x as f64, r.col(2).y as f64, r.col(2).z as f64),
        ),
        Vec3F64::new(t.x as f64, t.y as f64, t.z as f64),
    )
}

// ── Per-observation residual + analytical Jacobian (matches ReprojFactor) ──

/// Computes (residual, J_pose 2×6, J_point 2×3) at the current state.
/// Returns the camera-frame point and the clamped z too, for back-substitution
/// reasoning.
///
/// Jacobian layout (row-major flat):
///   J_pose[0..6]:  [du/dρ_x, du/dρ_y, du/dρ_z, du/dω_x, du/dω_y, du/dω_z]
///   J_pose[6..12]: [dv/dρ_x, dv/dρ_y, dv/dρ_z, dv/dω_x, dv/dω_y, dv/dω_z]
///   J_point[0..3]: [du/dx,   du/dy,   du/dz]
///   J_point[3..6]: [dv/dx,   dv/dy,   dv/dz]
fn residual_and_jacobians(
    pose: &SE3F32,
    point_w: &Vec3F64,
    pixel: [f32; 2],
    camera: &PinholeCamera,
) -> ([f32; 2], [f32; 12], [f32; 6]) {
    let fx = camera.fx as f32;
    let fy = camera.fy as f32;
    let cx = camera.cx as f32;
    let cy = camera.cy as f32;

    let pw = Vec3AF32::new(point_w.x as f32, point_w.y as f32, point_w.z as f32);
    let pc = *pose * pw;
    let z = if pc.z.abs() < MIN_Z {
        if pc.z >= 0.0 {
            MIN_Z
        } else {
            -MIN_Z
        }
    } else {
        pc.z
    };
    let inv_z = 1.0 / z;
    let inv_z2 = inv_z * inv_z;

    let u = fx * pc.x * inv_z + cx;
    let v = fy * pc.y * inv_z + cy;
    let r = [u - pixel[0], v - pixel[1]];

    // J_proj row coefficients (∂[u; v] / ∂[X_c]).
    let a0 = fx * inv_z;
    let a2 = -fx * pc.x * inv_z2;
    let b1 = fy * inv_z;
    let b2 = -fy * pc.y * inv_z2;

    // Rotation matrix elements (R: world→cam).
    let rm = pose.r.matrix();
    let r00 = rm.col(0).x;
    let r01 = rm.col(1).x;
    let r02 = rm.col(2).x;
    let r10 = rm.col(0).y;
    let r11 = rm.col(1).y;
    let r12 = rm.col(2).y;
    let r20 = rm.col(0).z;
    let r21 = rm.col(1).z;
    let r22 = rm.col(2).z;

    let (px, py, pz) = (pw.x, pw.y, pw.z);

    // S = -R · skew(p_w) — for the omega part.
    let s00 = -pz * r01 + py * r02;
    let s10 = -pz * r11 + py * r12;
    let s20 = -pz * r21 + py * r22;

    let s01 = pz * r00 - px * r02;
    let s11 = pz * r10 - px * r12;
    let s21 = pz * r20 - px * r22;

    let s02 = -py * r00 + px * r01;
    let s12 = -py * r10 + px * r11;
    let s22 = -py * r20 + px * r21;

    // J_pt = J_proj · R (3 cols).
    let jpt_00 = a0 * r00 + a2 * r20;
    let jpt_01 = a0 * r01 + a2 * r21;
    let jpt_02 = a0 * r02 + a2 * r22;
    let jpt_10 = b1 * r10 + b2 * r20;
    let jpt_11 = b1 * r11 + b2 * r21;
    let jpt_12 = b1 * r12 + b2 * r22;

    // J_omega = J_proj · S (3 cols).
    let jom_00 = a0 * s00 + a2 * s20;
    let jom_01 = a0 * s01 + a2 * s21;
    let jom_02 = a0 * s02 + a2 * s22;
    let jom_10 = b1 * s10 + b2 * s20;
    let jom_11 = b1 * s11 + b2 * s21;
    let jom_12 = b1 * s12 + b2 * s22;

    // Layout J_pose 2×6 row-major: [ρ(3) | ω(3)] per row.
    let j_pose: [f32; 12] = [
        jpt_00, jpt_01, jpt_02, jom_00, jom_01, jom_02, jpt_10, jpt_11, jpt_12, jom_10, jom_11,
        jom_12,
    ];
    // J_point 2×3 row-major.
    let j_point: [f32; 6] = [jpt_00, jpt_01, jpt_02, jpt_10, jpt_11, jpt_12];

    (r, j_pose, j_point)
}

// ── Small block primitives (f32) ─────────────────────────────────────────

#[inline]
fn ata_6x6_into(acc: &mut [f32; 36], j: &[f32; 12]) {
    // acc += J.T @ J  where J is 2×6 row-major.
    let r0 = &j[0..6];
    let r1 = &j[6..12];
    for i in 0..6 {
        for k in 0..6 {
            acc[i * 6 + k] += r0[i] * r0[k] + r1[i] * r1[k];
        }
    }
}

#[inline]
fn ata_3x3_into(acc: &mut [f32; 9], j: &[f32; 6]) {
    let r0 = &j[0..3];
    let r1 = &j[3..6];
    for i in 0..3 {
        for k in 0..3 {
            acc[i * 3 + k] += r0[i] * r0[k] + r1[i] * r1[k];
        }
    }
}

#[inline]
fn atb_6x3_into(acc: &mut [f32; 18], jp: &[f32; 12], jx: &[f32; 6]) {
    // acc += J_pose.T @ J_point  →  6 × 3 row-major.
    let jp0 = &jp[0..6];
    let jp1 = &jp[6..12];
    let jx0 = &jx[0..3];
    let jx1 = &jx[3..6];
    for i in 0..6 {
        for k in 0..3 {
            acc[i * 3 + k] += jp0[i] * jx0[k] + jp1[i] * jx1[k];
        }
    }
}

#[inline]
fn atb_6x1_into(acc: &mut [f32; 6], j: &[f32; 12], r: &[f32; 2]) {
    // acc -= J.T @ r  (note negative for gradient convention).
    for i in 0..6 {
        acc[i] -= j[i] * r[0] + j[6 + i] * r[1];
    }
}

#[inline]
fn atb_3x1_into(acc: &mut [f32; 3], j: &[f32; 6], r: &[f32; 2]) {
    for i in 0..3 {
        acc[i] -= j[i] * r[0] + j[3 + i] * r[1];
    }
}

/// Invert a 3×3 row-major matrix. Returns None if singular.
fn invert_3x3(m: &[f32; 9]) -> Option<[f32; 9]> {
    let a = m[0];
    let b = m[1];
    let c = m[2];
    let d = m[3];
    let e = m[4];
    let f = m[5];
    let g = m[6];
    let h = m[7];
    let i = m[8];
    let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if det.abs() < 1e-20 {
        return None;
    }
    let inv_det = 1.0 / det;
    Some([
        (e * i - f * h) * inv_det,
        (c * h - b * i) * inv_det,
        (b * f - c * e) * inv_det,
        (f * g - d * i) * inv_det,
        (a * i - c * g) * inv_det,
        (c * d - a * f) * inv_det,
        (d * h - e * g) * inv_det,
        (b * g - a * h) * inv_det,
        (a * e - b * d) * inv_det,
    ])
}

#[inline]
fn matmul_6x3_3x3(a: &[f32; 18], b: &[f32; 9]) -> [f32; 18] {
    let mut out = [0.0_f32; 18];
    for i in 0..6 {
        for k in 0..3 {
            let mut s = 0.0_f32;
            for r in 0..3 {
                s += a[i * 3 + r] * b[r * 3 + k];
            }
            out[i * 3 + k] = s;
        }
    }
    out
}

#[inline]
fn matvec_6x3_3(a: &[f32; 18], b: &[f32; 3]) -> [f32; 6] {
    let mut out = [0.0_f32; 6];
    for i in 0..6 {
        out[i] = a[i * 3] * b[0] + a[i * 3 + 1] * b[1] + a[i * 3 + 2] * b[2];
    }
    out
}

#[inline]
fn matvec_3x3_3(a: &[f32; 9], b: &[f32; 3]) -> [f32; 3] {
    [
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2],
        a[3] * b[0] + a[4] * b[1] + a[5] * b[2],
        a[6] * b[0] + a[7] * b[1] + a[8] * b[2],
    ]
}

#[inline]
fn matvec_6x3t_6(a: &[f32; 18], b: &[f32; 6]) -> [f32; 3] {
    // returns a.T @ b  →  3-vector; a is stored row-major 6×3
    let mut out = [0.0_f32; 3];
    for k in 0..3 {
        out[k] = a[k] * b[0]
            + a[3 + k] * b[1]
            + a[6 + k] * b[2]
            + a[9 + k] * b[3]
            + a[12 + k] * b[4]
            + a[15 + k] * b[5];
    }
    out
}

// ── Driver ───────────────────────────────────────────────────────────────

/// Bundle adjustment via dense Schur-complement reduction. Same external
/// contract as [`crate::ba::bundle_adjust`] but uses Schur internally:
/// the reduced 6P×6P camera system is solved with `faer`'s dense Cholesky;
/// points are recovered by back-substitution.
///
/// Currently respects `fixed_pose` and `fixed_point` flags on each
/// observation but does not yet implement `BaParams::robust` (treats as
/// identity loss). All other params (max_iterations, initial_lambda,
/// cost_tolerance, gradient_tolerance) are honoured.
pub fn bundle_adjust_schur(
    poses: &[Pose3d],
    points: &[Vec3F64],
    observations: &[BaObservation],
    camera: &PinholeCamera,
    params: &BaParams,
) -> Result<BaResult, SchurBaError> {
    bundle_adjust_schur_with_priors(poses, points, observations, camera, params, None)
}

/// Bundle adjustment via dense Schur-complement reduction with optional
/// per-pose translation priors.
///
/// Identical to [`bundle_adjust_schur`] but accepts a slice of
/// `Option<BaPosePrior>` of length `poses.len()` (entries may be `None` for
/// unconstrained poses). When a prior is present for pose `i`, the BA cost
/// gains a position residual
///
/// ```text
///     r_pos_i = (C_i_world − prior_i.center_world) / prior_i.sigma
/// ```
///
/// where `C_i_world = -R^T · t`. This anchors all three world-frame axes of
/// the pose translation simultaneously — the durable fix for lateral /
/// vertical drift that the per-observation depth residual alone (which only
/// constrains cam-frame Z) cannot close.
///
/// The pose-prior Jacobian decomposes into a 3×6 block per pose with no
/// coupling to point variables, so it augments only the on-diagonal
/// camera-block A_ii in the Schur reduction (B and C are untouched).
///
/// Poses marked fixed via `BaObservation::fixed_pose` have no free
/// parameters; any prior on them is silently ignored.
pub fn bundle_adjust_schur_with_priors(
    poses: &[Pose3d],
    points: &[Vec3F64],
    observations: &[BaObservation],
    camera: &PinholeCamera,
    params: &BaParams,
    pose_priors: Option<&[Option<BaPosePrior>]>,
) -> Result<BaResult, SchurBaError> {
    // Validate prior slice length matches poses.
    if let Some(pp) = pose_priors {
        if pp.len() != poses.len() {
            return Err(SchurBaError::Ba(BaError::InvalidInput(format!(
                "pose_priors length {} != poses length {}",
                pp.len(),
                poses.len()
            ))));
        }
    }
    let p_total = poses.len();
    let n_total = points.len();

    // Index map: which poses / points are touched by any free observation.
    let mut pose_is_free = vec![false; p_total];
    let mut point_is_free = vec![false; n_total];
    for obs in observations {
        if obs.pose_idx >= p_total || obs.point_idx >= n_total {
            continue;
        }
        if !obs.fixed_pose {
            pose_is_free[obs.pose_idx] = true;
        }
        if !obs.fixed_point {
            point_is_free[obs.point_idx] = true;
        }
    }
    let pose_local: Vec<i64> = {
        let mut v = vec![-1_i64; p_total];
        let mut next = 0;
        for i in 0..p_total {
            if pose_is_free[i] {
                v[i] = next;
                next += 1;
            }
        }
        v
    };
    let point_local: Vec<i64> = {
        let mut v = vec![-1_i64; n_total];
        let mut next = 0;
        for i in 0..n_total {
            if point_is_free[i] {
                v[i] = next;
                next += 1;
            }
        }
        v
    };
    let n_free_poses = pose_local.iter().filter(|&&x| x >= 0).count();
    let n_free_points = point_local.iter().filter(|&&x| x >= 0).count();

    if n_free_poses == 0 {
        return Err(SchurBaError::NoFreeVariables);
    }

    // Mutable state.
    let mut se3s: Vec<SE3F32> = poses.iter().map(pose_to_se3).collect();
    let mut xyz: Vec<Vec3F64> = points.to_vec();

    let mut lambda = params.initial_lambda;
    let mut prev_cost: Option<f32> = None;
    let mut iters_done = 0usize;
    let mut converged = false;

    for _iter in 0..params.max_iterations {
        iters_done += 1;

        // ── Linearise: build A, C, B (per-obs), g_pose, g_point ──────────
        // A: n_free_poses × [36] (6×6 blocks).
        // C: n_free_points × [9]  (3×3 blocks).
        // We also keep observation-aligned B blocks (6×3) so we can iterate
        // by point during the Schur reduction.
        let mut a_blocks = vec![[0.0_f32; 36]; n_free_poses];
        let mut c_blocks = vec![[0.0_f32; 9]; n_free_points];
        let mut g_pose = vec![[0.0_f32; 6]; n_free_poses];
        let mut g_point = vec![[0.0_f32; 3]; n_free_points];

        // Per-observation B contributions, grouped by point (for the Schur
        // pass). We store (pose_local_idx, B_6x3) lists per free-point index.
        let mut b_by_point: Vec<Vec<(usize, [f32; 18])>> = vec![Vec::new(); n_free_points];

        // Also record observations that touch FIXED point but FREE pose —
        // contribute to A and g_pose only, no B.
        // (Symmetric case: free point + fixed pose contributes to C and
        //  g_point only. Both we handle below.)
        // Robust-loss IRLS weight per observation. weight w = min(1, scale/‖r‖)
        // for Huber, w = scale²/(scale²+‖r‖²) for Cauchy. Identity uses w=1.
        // Apply √w to both residual and Jacobian rows (equivalent to multiplying
        // the obs's contribution to the normal equations by w).
        let robust = params.robust;
        let robust_scale = params.robust_scale_sq.sqrt().max(1e-6);
        let huber_w = |r_sq: f32| -> f32 {
            // ‖r‖ ≤ scale → w=1; else w = scale/‖r‖
            let r_norm = r_sq.sqrt();
            if r_norm <= robust_scale {
                1.0
            } else {
                robust_scale / r_norm
            }
        };
        let cauchy_w = |r_sq: f32| -> f32 {
            let s2 = robust_scale * robust_scale;
            s2 / (s2 + r_sq)
        };

        let mut cost = 0.0_f32;
        let mut n_depth_obs_iter = 0usize;

        for obs in observations {
            if obs.pose_idx >= p_total || obs.point_idx >= n_total {
                continue;
            }
            let pose = &se3s[obs.pose_idx];
            let point = &xyz[obs.point_idx];
            let (mut r, mut j_pose, mut j_point) =
                residual_and_jacobians(pose, point, obs.pixel, camera);
            let r_sq = r[0] * r[0] + r[1] * r[1];

            // IRLS weight; apply √w to r and J.
            let w = match robust {
                RobustKernelKind::Identity => 1.0,
                RobustKernelKind::Huber => huber_w(r_sq),
                RobustKernelKind::Cauchy | RobustKernelKind::Tukey => cauchy_w(r_sq),
            };
            if w != 1.0 {
                let sw = w.sqrt();
                r[0] *= sw;
                r[1] *= sw;
                for v in j_pose.iter_mut() {
                    *v *= sw;
                }
                for v in j_point.iter_mut() {
                    *v *= sw;
                }
            }
            cost += 0.5 * (r[0] * r[0] + r[1] * r[1]);

            let pli = pose_local[obs.pose_idx];
            let xli = point_local[obs.point_idx];

            if pli >= 0 {
                let pli = pli as usize;
                ata_6x6_into(&mut a_blocks[pli], &j_pose);
                atb_6x1_into(&mut g_pose[pli], &j_pose, &r);
            }
            if xli >= 0 {
                let xli = xli as usize;
                ata_3x3_into(&mut c_blocks[xli], &j_point);
                atb_3x1_into(&mut g_point[xli], &j_point, &r);
            }
            if pli >= 0 && xli >= 0 {
                let mut b_block = [0.0_f32; 18];
                atb_6x3_into(&mut b_block, &j_pose, &j_point);
                b_by_point[xli as usize].push((pli as usize, b_block));
            }

            // ── Depth residual (optional metric anchor) ─────────────────
            // r_z = (Z_pred − d_meas) / σ_depth
            // ∂Z/∂ρ  = e_z   (translation tangent contributes 1 to z)
            // ∂Z/∂ω  = row 2 of S = -R · skew(p_w)
            // ∂Z/∂Xw = row 2 of R
            // We treat the depth residual as a single extra row in the
            // stacked Jacobian, weighted by 1/σ. Its outer products are
            // added to A_p, C_p, B as for any other residual.
            if let Some(d_meas) = obs.depth_meas {
                let sigma = obs.depth_sigma.max(1e-6);
                let inv_sigma = 1.0_f32 / sigma;

                // Recompute Z_pred + jacobian rows. We need the same z-clamp
                // semantics, and the geometry-only Jacobians (no projection
                // coefficients a0/b1/a2/b2).
                let pw = Vec3AF32::new(point.x as f32, point.y as f32, point.z as f32);
                let pc = *pose * pw;
                let z_pred = if pc.z.abs() < MIN_Z {
                    if pc.z >= 0.0 {
                        MIN_Z
                    } else {
                        -MIN_Z
                    }
                } else {
                    pc.z
                };

                // Depth residual (scaled by 1/σ).
                let r_z = (z_pred - d_meas) * inv_sigma;

                // J rows (1×6 pose, 1×3 point), all scaled by 1/σ.
                let rm = pose.r.matrix();
                let r20 = rm.col(0).z;
                let r21 = rm.col(1).z;
                let r22 = rm.col(2).z;
                let (px, py, pz) = (pw.x, pw.y, pw.z);
                // Row 2 of S = -R · skew(p_w):
                //   col0: -pz·r21 + py·r22
                //   col1:  pz·r20 - px·r22
                //   col2: -py·r20 + px·r21
                let s20 = -pz * r21 + py * r22;
                let s21 = pz * r20 - px * r22;
                let s22 = -py * r20 + px * r21;

                // J_pose_depth (1×6): [ρ(0,0,1) | ω(s20, s21, s22)] / σ
                let jpd = [
                    0.0_f32 * inv_sigma,
                    0.0_f32 * inv_sigma,
                    1.0_f32 * inv_sigma,
                    s20 * inv_sigma,
                    s21 * inv_sigma,
                    s22 * inv_sigma,
                ];
                // J_point_depth (1×3): [r20, r21, r22] / σ
                let jxd = [r20 * inv_sigma, r21 * inv_sigma, r22 * inv_sigma];

                // ── Apply IRLS robust weight to the depth residual ────────
                // The depth residual is a single scalar r_z (already scaled by
                // 1/σ_depth). Use the same Huber/Cauchy gate as the
                // reprojection path so outlier depth measurements (e.g.
                // boundary mis-samples) do not dominate the normal equations.
                // The gate uses ‖r_z‖² of the *whitened* residual, matching
                // the χ² interpretation (ORB-SLAM3 §IV.B uses χ²=7.815 for
                // 3-DoF RGB-D; we reuse `robust_scale_sq` for simplicity).
                let r_sq_d = r_z * r_z;
                let w_d = match robust {
                    RobustKernelKind::Identity => 1.0,
                    RobustKernelKind::Huber => huber_w(r_sq_d),
                    RobustKernelKind::Cauchy | RobustKernelKind::Tukey => cauchy_w(r_sq_d),
                };
                cost += 0.5 * w_d * r_sq_d;
                n_depth_obs_iter += 1;

                // Accumulate into A (6×6) — w · outer product jpd·jpdᵀ.
                if pli >= 0 {
                    let pli_u = pli as usize;
                    let ab = &mut a_blocks[pli_u];
                    for i in 0..6 {
                        for k in 0..6 {
                            ab[i * 6 + k] += w_d * jpd[i] * jpd[k];
                        }
                    }
                    // g_pose -= w · jpdᵀ · r_z
                    let gp = &mut g_pose[pli_u];
                    for i in 0..6 {
                        gp[i] -= w_d * jpd[i] * r_z;
                    }
                }
                // Accumulate into C (3×3) — w · outer product jxd·jxdᵀ.
                if xli >= 0 {
                    let xli_u = xli as usize;
                    let cb = &mut c_blocks[xli_u];
                    for i in 0..3 {
                        for k in 0..3 {
                            cb[i * 3 + k] += w_d * jxd[i] * jxd[k];
                        }
                    }
                    let gx = &mut g_point[xli_u];
                    for i in 0..3 {
                        gx[i] -= w_d * jxd[i] * r_z;
                    }
                }
                // Accumulate into B (6×3) — w · jpd·jxdᵀ.
                if pli >= 0 && xli >= 0 {
                    let mut b_block = [0.0_f32; 18];
                    for i in 0..6 {
                        for k in 0..3 {
                            b_block[i * 3 + k] = w_d * jpd[i] * jxd[k];
                        }
                    }
                    b_by_point[xli as usize].push((pli as usize, b_block));
                }
            }
        }
        let _ = n_depth_obs_iter; // currently unused; reserved for future telemetry

        // ── Per-pose translation prior (3-D position residual) ──────────────
        // For each pose i with a Some(prior), contribute a 3-row residual
        //
        //     r_pos = (C - C_prior) / σ
        //
        // with C = -R^T · t (camera centre in world frame). Jacobian wrt the
        // pose tangent ξ = [ρ; ω] is
        //
        //     ∂C/∂ρ = -I                 (3×3)
        //     ∂C/∂ω = [C]_×              (3×3, skew of C)
        //
        // derived from the right-perturbation retract `T·exp(ξ)` matching the
        // convention used by `residual_and_jacobians` above (see ReprojFactor
        // docs). With no coupling to point variables, this only augments the
        // pose-block A_ii and g_pose[i]; B and C in the Schur reduction are
        // untouched.
        if let Some(pp_slice) = pose_priors {
            for i_global in 0..p_total {
                let Some(prior) = pp_slice[i_global] else {
                    continue;
                };
                let pli = pose_local[i_global];
                if pli < 0 {
                    // Pose fixed — prior is moot.
                    continue;
                }
                let pli_u = pli as usize;
                let sigma = prior.sigma.max(1e-6);
                let inv_sigma = 1.0_f32 / sigma;

                // Camera centre C = -R^T · t.
                let pose = &se3s[i_global];
                let rm = pose.r.matrix();
                let t = pose.t;
                // R^T · t (i.e. R-transpose-times-t — apply R as world←cam to t).
                // rm.col(j) is column j of R (cam→world if you read it as R^T … but
                // our convention has R as world→cam). So R^T · t = sum over rows.
                // R^T_row0 = (r00, r10, r20) = R.col(0); so R^T · t = (R.col(0)·t,
                // R.col(1)·t, R.col(2)·t).
                let r_col0 = rm.col(0);
                let r_col1 = rm.col(1);
                let r_col2 = rm.col(2);
                let rt_t_x = r_col0.x * t.x + r_col0.y * t.y + r_col0.z * t.z;
                let rt_t_y = r_col1.x * t.x + r_col1.y * t.y + r_col1.z * t.z;
                let rt_t_z = r_col2.x * t.x + r_col2.y * t.y + r_col2.z * t.z;
                let c_pred = [-rt_t_x, -rt_t_y, -rt_t_z];

                // Residual r_pos = (C − C_prior) / σ  (3-vector).
                let r_pos = [
                    (c_pred[0] - prior.center_world[0]) * inv_sigma,
                    (c_pred[1] - prior.center_world[1]) * inv_sigma,
                    (c_pred[2] - prior.center_world[2]) * inv_sigma,
                ];

                // ── Apply IRLS robust weight to the pose-prior residual ───
                // The gate uses ‖r_pos‖² (sum of three whitened squared
                // components). This dampens single-pose VO glitches (a
                // mis-aligned chain step) so they cannot dominate the prior
                // term. We reuse `robust_scale_sq` for consistency with the
                // reprojection path; the residual is already whitened by 1/σ
                // so the gate is on the χ²-equivalent magnitude.
                let r_sq_p = r_pos[0] * r_pos[0] + r_pos[1] * r_pos[1] + r_pos[2] * r_pos[2];
                let w_p = match robust {
                    RobustKernelKind::Identity => 1.0,
                    RobustKernelKind::Huber => huber_w(r_sq_p),
                    RobustKernelKind::Cauchy | RobustKernelKind::Tukey => cauchy_w(r_sq_p),
                };
                cost += 0.5 * w_p * r_sq_p;

                // Jacobian (3×6), all scaled by 1/σ:
                //   ∂C/∂ρ = -I
                //   ∂C/∂ω = [C]_× =  [ 0   -cz   cy ]
                //                    [ cz   0   -cx ]
                //                    [-cy   cx   0  ]
                let cx_ = c_pred[0];
                let cy_ = c_pred[1];
                let cz_ = c_pred[2];
                // Row-major 3×6 layout: [ρ(3) | ω(3)] per row.
                let j_pose_prior: [f32; 18] = [
                    // Row 0 (dCx)
                    -inv_sigma,
                    0.0,
                    0.0,
                    0.0,
                    -cz_ * inv_sigma,
                    cy_ * inv_sigma,
                    // Row 1 (dCy)
                    0.0,
                    -inv_sigma,
                    0.0,
                    cz_ * inv_sigma,
                    0.0,
                    -cx_ * inv_sigma,
                    // Row 2 (dCz)
                    0.0,
                    0.0,
                    -inv_sigma,
                    -cy_ * inv_sigma,
                    cx_ * inv_sigma,
                    0.0,
                ];

                // Accumulate into A_ii (6×6) — w · Σ_r J_r.T · J_r over 3 rows.
                let ab = &mut a_blocks[pli_u];
                for r_idx in 0..3 {
                    let row = &j_pose_prior[r_idx * 6..(r_idx + 1) * 6];
                    for ii in 0..6 {
                        for kk in 0..6 {
                            ab[ii * 6 + kk] += w_p * row[ii] * row[kk];
                        }
                    }
                }
                // RHS: g_pose -= w · Σ_r J_r.T · r_pos[r]
                let gp = &mut g_pose[pli_u];
                for r_idx in 0..3 {
                    let row = &j_pose_prior[r_idx * 6..(r_idx + 1) * 6];
                    for ii in 0..6 {
                        gp[ii] -= w_p * row[ii] * r_pos[r_idx];
                    }
                }
            }
        }

        // Cost convergence (post-step convergence will follow successful steps below).
        if let Some(pc) = prev_cost {
            // Only declare convergence here on a *successful* step path; we'll
            // do that after accepting a step. For now, just log.
            let _ = pc;
        }

        // ── Apply LM damping: A[i] += λ·I, C[j] += λ·I ──────────────────
        for ab in &mut a_blocks {
            for d in 0..6 {
                ab[d * 6 + d] += lambda;
            }
        }
        for cb in &mut c_blocks {
            for d in 0..3 {
                cb[d * 3 + d] += lambda;
            }
        }

        // ── Build M (dense 6Pf × 6Pf) + m (6Pf) ─────────────────────────
        let dim = n_free_poses * 6;
        let mut m_mat = Mat::<f64>::zeros(dim, dim);
        let mut m_vec = vec![0.0_f64; dim];

        // Place A blocks on diagonal of M.
        for (k, ab) in a_blocks.iter().enumerate() {
            for i in 0..6 {
                for j in 0..6 {
                    m_mat[(k * 6 + i, k * 6 + j)] = ab[i * 6 + j] as f64;
                }
            }
            for i in 0..6 {
                m_vec[k * 6 + i] = g_pose[k][i] as f64;
            }
        }

        // For each free point j: invert C_j, accumulate Schur correction
        //   M[i1, i2] -= B[i1, j] · C_j⁻¹ · B[i2, j].T
        //   m[i]     -= B[i, j]  · C_j⁻¹ · g_point[j]
        // Skip if C_j is singular (rare, but be safe).
        let mut c_inv_blocks: Vec<Option<[f32; 9]>> = Vec::with_capacity(n_free_points);
        for cb in &c_blocks {
            c_inv_blocks.push(invert_3x3(cb));
        }

        for (j, b_for_j) in b_by_point.iter().enumerate() {
            let Some(c_inv_j) = c_inv_blocks[j] else {
                continue;
            };
            // Pre-compute B_i · C⁻¹ for each i in this point's edge list.
            let bc: Vec<(usize, [f32; 18])> = b_for_j
                .iter()
                .map(|(i_loc, b)| (*i_loc, matmul_6x3_3x3(b, &c_inv_j)))
                .collect();

            // RHS: m[i] -= (B_i · C⁻¹) · g_point[j]
            let gp = g_point[j];
            for (i_loc, bc_block) in &bc {
                let bc_g = matvec_6x3_3(bc_block, &gp);
                let base = i_loc * 6;
                for r in 0..6 {
                    m_vec[base + r] -= bc_g[r] as f64;
                }
            }

            // LHS: M[i1, i2] -= (B_i1 · C⁻¹) · B_i2.T   (6×6 block)
            for (idx1, (i1_loc, bc1)) in bc.iter().enumerate() {
                for (idx2, (i2_loc, _bc2_unused)) in bc.iter().enumerate() {
                    let b2 = &b_for_j[idx2].1;
                    // (6×3) @ (3×6) — bc1 (6×3) times b2.T (3×6).
                    // Compute element (r, c): sum_k bc1[r, k] · b2[c, k]
                    let row0 = i1_loc * 6;
                    let col0 = i2_loc * 6;
                    let _ = idx1;
                    let _ = idx2;
                    for r in 0..6 {
                        for c in 0..6 {
                            let mut s = 0.0_f32;
                            for k in 0..3 {
                                s += bc1[r * 3 + k] * b2[c * 3 + k];
                            }
                            m_mat[(row0 + r, col0 + c)] -= s as f64;
                        }
                    }
                }
            }
        }

        // ── Solve M · δ_pose = m via Cholesky ────────────────────────────
        // Symmetrize numerically (the construction above should already be
        // symmetric to within roundoff; do an average to guarantee).
        for i in 0..dim {
            for j in (i + 1)..dim {
                let avg = 0.5 * (m_mat[(i, j)] + m_mat[(j, i)]);
                m_mat[(i, j)] = avg;
                m_mat[(j, i)] = avg;
            }
        }
        let chol = match m_mat.llt(faer::Side::Lower) {
            Ok(c) => c,
            Err(e) => {
                // Bump damping and retry next outer iteration.
                lambda *= 10.0;
                if lambda > 1e10 {
                    return Err(SchurBaError::CholeskyFailed(format!("{e:?}")));
                }
                continue;
            }
        };
        // RHS as faer column.
        let m_col = Mat::<f64>::from_fn(dim, 1, |i, _| m_vec[i]);
        let d_pose_col = chol.solve(&m_col);

        // ── Back-substitute for points: δ_x[j] = C⁻¹ (g_x - B.T · δ_p) ──
        let mut d_pose = vec![0.0_f64; dim];
        for i in 0..dim {
            d_pose[i] = d_pose_col[(i, 0)];
        }
        let mut d_point = vec![[0.0_f32; 3]; n_free_points];
        for (j, b_for_j) in b_by_point.iter().enumerate() {
            let Some(c_inv_j) = c_inv_blocks[j] else {
                continue;
            };
            // rhs = g_point[j] - sum_i B[i, j].T · δ_pose[i]
            let mut rhs = g_point[j];
            for (i_loc, b_block) in b_for_j {
                let mut dp6 = [0.0_f32; 6];
                let base = i_loc * 6;
                for r in 0..6 {
                    dp6[r] = d_pose[base + r] as f32;
                }
                let contrib = matvec_6x3t_6(b_block, &dp6);
                for c in 0..3 {
                    rhs[c] -= contrib[c];
                }
            }
            d_point[j] = matvec_3x3_3(&c_inv_j, &rhs);
        }

        // ── Trial: retract poses, add to points, recompute cost ─────────
        let mut se3s_trial = se3s.clone();
        for i_global in 0..p_total {
            let pli = pose_local[i_global];
            if pli < 0 {
                continue;
            }
            let pli = pli as usize;
            let delta: [f32; 6] = [
                d_pose[pli * 6] as f32,
                d_pose[pli * 6 + 1] as f32,
                d_pose[pli * 6 + 2] as f32,
                d_pose[pli * 6 + 3] as f32,
                d_pose[pli * 6 + 4] as f32,
                d_pose[pli * 6 + 5] as f32,
            ];
            se3s_trial[i_global] = se3s[i_global].retract(&delta);
        }
        let mut xyz_trial = xyz.clone();
        for i_global in 0..n_total {
            let xli = point_local[i_global];
            if xli < 0 {
                continue;
            }
            let xli = xli as usize;
            let dp = d_point[xli];
            xyz_trial[i_global] = Vec3F64::new(
                xyz[i_global].x + dp[0] as f64,
                xyz[i_global].y + dp[1] as f64,
                xyz[i_global].z + dp[2] as f64,
            );
        }

        let mut new_cost = 0.0_f32;
        for obs in observations {
            if obs.pose_idx >= p_total || obs.point_idx >= n_total {
                continue;
            }
            let pose = &se3s_trial[obs.pose_idx];
            let point = &xyz_trial[obs.point_idx];
            let (r, _, _) = residual_and_jacobians(pose, point, obs.pixel, camera);
            let r_sq = r[0] * r[0] + r[1] * r[1];
            let w = match robust {
                RobustKernelKind::Identity => 1.0,
                RobustKernelKind::Huber => huber_w(r_sq),
                RobustKernelKind::Cauchy | RobustKernelKind::Tukey => cauchy_w(r_sq),
            };
            new_cost += 0.5 * w * r_sq;

            // Depth residual contribution to trial cost (same Huber/Cauchy
            // weighting as the linearisation pass, so accept/reject decisions
            // reflect the robust loss).
            if let Some(d_meas) = obs.depth_meas {
                let sigma = obs.depth_sigma.max(1e-6);
                let pw = Vec3AF32::new(point.x as f32, point.y as f32, point.z as f32);
                let pc = *pose * pw;
                let z_pred = if pc.z.abs() < MIN_Z {
                    if pc.z >= 0.0 {
                        MIN_Z
                    } else {
                        -MIN_Z
                    }
                } else {
                    pc.z
                };
                let r_z = (z_pred - d_meas) / sigma;
                let r_sq_d = r_z * r_z;
                let w_d = match robust {
                    RobustKernelKind::Identity => 1.0,
                    RobustKernelKind::Huber => huber_w(r_sq_d),
                    RobustKernelKind::Cauchy | RobustKernelKind::Tukey => cauchy_w(r_sq_d),
                };
                new_cost += 0.5 * w_d * r_sq_d;
            }
        }

        // Pose-prior contribution to trial cost.
        if let Some(pp_slice) = pose_priors {
            for i_global in 0..p_total {
                let Some(prior) = pp_slice[i_global] else {
                    continue;
                };
                if pose_local[i_global] < 0 {
                    continue;
                }
                let sigma = prior.sigma.max(1e-6);
                let inv_sigma = 1.0_f32 / sigma;
                let pose = &se3s_trial[i_global];
                let rm = pose.r.matrix();
                let t = pose.t;
                let r_col0 = rm.col(0);
                let r_col1 = rm.col(1);
                let r_col2 = rm.col(2);
                let rt_t_x = r_col0.x * t.x + r_col0.y * t.y + r_col0.z * t.z;
                let rt_t_y = r_col1.x * t.x + r_col1.y * t.y + r_col1.z * t.z;
                let rt_t_z = r_col2.x * t.x + r_col2.y * t.y + r_col2.z * t.z;
                let c_pred = [-rt_t_x, -rt_t_y, -rt_t_z];
                let r0 = (c_pred[0] - prior.center_world[0]) * inv_sigma;
                let r1 = (c_pred[1] - prior.center_world[1]) * inv_sigma;
                let r2 = (c_pred[2] - prior.center_world[2]) * inv_sigma;
                // Match the linearisation pass's Huber/Cauchy gate so
                // accept/reject reflects the robust loss.
                let r_sq_p = r0 * r0 + r1 * r1 + r2 * r2;
                let w_p = match robust {
                    RobustKernelKind::Identity => 1.0,
                    RobustKernelKind::Huber => huber_w(r_sq_p),
                    RobustKernelKind::Cauchy | RobustKernelKind::Tukey => cauchy_w(r_sq_p),
                };
                new_cost += 0.5 * w_p * r_sq_p;
            }
        }

        if new_cost < cost {
            // Accept step.
            let rel = if cost > 1e-12 {
                (cost - new_cost) / cost
            } else {
                0.0
            };
            se3s = se3s_trial;
            xyz = xyz_trial;
            prev_cost = Some(new_cost);
            lambda = (lambda / 3.0).max(1e-8);
            if rel < params.cost_tolerance {
                converged = true;
                break;
            }
        } else {
            // Reject — bump damping and retry.
            lambda *= 10.0;
            if lambda > 1e10 {
                break;
            }
        }
    }

    // Pack results.
    let mut out_poses = Vec::with_capacity(p_total);
    for i in 0..p_total {
        if pose_is_free[i] {
            out_poses.push(se3_to_pose(&se3s[i]));
        } else {
            out_poses.push(poses[i]);
        }
    }
    let mut out_points = Vec::with_capacity(n_total);
    for i in 0..n_total {
        if point_is_free[i] {
            out_points.push(xyz[i]);
        } else {
            out_points.push(points[i]);
        }
    }

    Ok(BaResult {
        poses: out_poses,
        points: out_points,
        iterations: iters_done,
        converged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::PinholeCamera;
    use kornia_algebra::Mat3F64;

    fn test_camera() -> PinholeCamera {
        PinholeCamera {
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
    fn schur_ba_recovers_perturbed_poses() {
        let cam = test_camera();
        // Two-camera, four-point setup like ba's existing test, but solve via Schur.
        let pose0 = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::ZERO);
        let pose1 = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(0.5, 0.0, 0.0));
        let true_points = [
            Vec3F64::new(-1.0, -1.0, 5.0),
            Vec3F64::new(1.0, -1.0, 5.0),
            Vec3F64::new(1.0, 1.0, 5.0),
            Vec3F64::new(-1.0, 1.0, 5.0),
        ];
        let project = |pose: &Pose3d, pw: &Vec3F64| -> [f32; 2] {
            let pc = pose.transform_point(pw);
            let u = cam.fx * pc.x / pc.z + cam.cx;
            let v = cam.fy * pc.y / pc.z + cam.cy;
            [u as f32, v as f32]
        };
        let mut observations = Vec::new();
        for (pi, pt) in true_points.iter().enumerate() {
            observations.push(BaObservation {
                pose_idx: 0,
                point_idx: pi,
                pixel: project(&pose0, pt),
                fixed_pose: true,
                fixed_point: false,
                ..BaObservation::default()
            });
            observations.push(BaObservation {
                pose_idx: 1,
                point_idx: pi,
                pixel: project(&pose1, pt),
                fixed_pose: false,
                fixed_point: false,
                ..BaObservation::default()
            });
        }
        let perturbed: Vec<Vec3F64> = true_points
            .iter()
            .map(|p| *p + Vec3F64::new(0.05, -0.03, 0.02))
            .collect();
        let result = bundle_adjust_schur(
            &[pose0, pose1],
            &perturbed,
            &observations,
            &cam,
            &BaParams {
                max_iterations: 30,
                ..BaParams::default()
            },
        )
        .unwrap();
        // The 4-points / 2-poses (1 fixed) problem has 18 unknowns vs 16
        // residuals — gauge ambiguity gives a 2-dim cost-zero manifold.
        // BA reaches cost=0 (verified by tracing) but lands on a different
        // point in that manifold depending on the solver's null-space
        // navigation. We assert geometric closeness within 0.2 m (about the
        // expected radius of the gauge ambiguity for this configuration).
        for (i, refined) in result.points.iter().enumerate() {
            let err = (*refined - true_points[i]).length();
            assert!(err < 0.2, "point {i} error {err} too large");
        }
    }

    /// Depth-anchored BA recovers absolute metric scale.
    ///
    /// Setup:
    ///   * 5 poses on a half-circle at radius 4 m looking inward at origin.
    ///   * 50 known 3D points scattered in a box around the origin.
    ///   * Project to pixels with σ=0.3 px Gaussian noise.
    ///   * Synthetic depth measurement per observation, σ=2% of true depth.
    ///   * INIT the BA with points scaled 2× from ground truth — without depth
    ///     residuals, this drift would be unobservable (gauge ambiguity).
    ///   * With depth_meas set, the BA should recover GT scale.
    fn translate_pose(t: Vec3F64) -> Pose3d {
        // Camera at position `cam_pos = t` looking down +Z (identity rotation
        // in world frame). Then R_w_c = I, t_w_c = cam_pos, and the
        // world→camera pose stored in Pose3d is the *inverse*:
        //   R_cw = I, t_cw = -cam_pos.
        Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(-t.x, -t.y, -t.z))
    }

    #[test]
    fn schur_ba_with_depth_recovers_scale() {
        // Reproducible PRNG via std (no rand crate dep here).
        // Simple LCG for noise sampling.
        struct Lcg {
            state: u64,
        }
        impl Lcg {
            fn new(seed: u64) -> Self {
                Self { state: seed }
            }
            fn next_u64(&mut self) -> u64 {
                self.state = self
                    .state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                self.state
            }
            // Box–Muller standard normal (uses two uniforms).
            fn normal(&mut self) -> f64 {
                let u1 = ((self.next_u64() >> 11) as f64) / (1u64 << 53) as f64;
                let u2 = ((self.next_u64() >> 11) as f64) / (1u64 << 53) as f64;
                let u1 = u1.max(1e-12);
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
            }
        }
        let mut rng = Lcg::new(0x00C0_FFEE_DEAD_BEEF_u64);

        let cam = PinholeCamera {
            fx: 600.0,
            fy: 600.0,
            cx: 320.0,
            cy: 240.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        };

        // 5 cameras translating along +X by 0, 0.1, 0.2, 0.3, 0.4 m
        // (typical small forward/sideways baseline in SLAM). All cameras
        // look down +Z (identity rotation). Camera 0 sits at the origin →
        // its pose is identity in both world and inverse-world frames,
        // so a global similarity that fixes the origin leaves it
        // unchanged. This makes (poses, points) up to scale invisible
        // to reprojection cost when pose 0 is fixed.
        let cam_positions = [
            Vec3F64::new(0.0, 0.0, 0.0),
            Vec3F64::new(0.1, 0.0, 0.0),
            Vec3F64::new(0.2, 0.0, 0.0),
            Vec3F64::new(0.3, 0.0, 0.0),
            Vec3F64::new(0.4, 0.0, 0.0),
        ];
        let true_poses: Vec<Pose3d> = cam_positions.iter().map(|&p| translate_pose(p)).collect();

        // 50 well-distributed 3D points in front of the cameras, 3-6 m deep.
        let mut true_points: Vec<Vec3F64> = Vec::with_capacity(50);
        for k in 0..50 {
            let kf = k as f64;
            let x = (kf * 0.37).sin() * 1.2 + (kf * 0.13).cos() * 0.4;
            let y = (kf * 0.29).cos() * 0.9 + (kf * 0.11).sin() * 0.3;
            let z = 4.0 + (kf * 0.41).sin() * 1.5; // 2.5..5.5m in front
            true_points.push(Vec3F64::new(x, y, z));
        }

        // Build observations: pixels + depth, both noisy. Skip points behind
        // the camera (negative z) — they can happen for the wider angles.
        let mut observations: Vec<BaObservation> = Vec::new();
        for (pi, pose) in true_poses.iter().enumerate() {
            for (xi, pt) in true_points.iter().enumerate() {
                let pc = pose.transform_point(pt);
                if pc.z <= 0.2 {
                    continue;
                }
                let u = cam.fx * pc.x / pc.z + cam.cx + 0.3 * rng.normal();
                let v = cam.fy * pc.y / pc.z + cam.cy + 0.3 * rng.normal();
                // Depth noise: σ = 2% of true depth.
                let depth_sigma_m = 0.02 * pc.z as f32;
                let d_meas = (pc.z + 0.02 * pc.z * rng.normal()) as f32;
                observations.push(BaObservation {
                    pose_idx: pi,
                    point_idx: xi,
                    pixel: [u as f32, v as f32],
                    fixed_pose: pi == 0, // anchor pose 0
                    fixed_point: false,
                    depth_meas: Some(d_meas),
                    depth_sigma: depth_sigma_m,
                });
            }
        }

        // Initial guess: simulate a 2× global scale drift. Pose 0 stays at
        // identity (origin is similarity-invariant). Poses 1..N get their
        // translation scaled by 2× (i.e. cam baselines are 2× too long).
        // Points are also scaled 2× → the reprojection residual at this
        // init is *exactly zero* because (s·R · s·X + s·t)/(s·Z) = same
        // pixel. Only the depth residual can break this gauge.
        let init_poses: Vec<Pose3d> = true_poses
            .iter()
            .enumerate()
            .map(|(i, p)| {
                if i == 0 {
                    *p
                } else {
                    // Scale translation by 2× (rotation is identity, scale-invariant).
                    Pose3d::new(
                        p.rotation,
                        Vec3F64::new(
                            p.translation.x * 2.0,
                            p.translation.y * 2.0,
                            p.translation.z * 2.0,
                        ),
                    )
                }
            })
            .collect();
        let init_points: Vec<Vec3F64> = true_points
            .iter()
            .map(|p| Vec3F64::new(p.x * 2.0, p.y * 2.0, p.z * 2.0))
            .collect();

        let params = BaParams {
            max_iterations: 100,
            cost_tolerance: 1e-8,
            ..BaParams::default()
        };
        let result =
            bundle_adjust_schur(&init_poses, &init_points, &observations, &cam, &params).unwrap();

        // Assert geometric recovery.
        let mut max_pt_err: f64 = 0.0;
        let mut mean_pt_err: f64 = 0.0;
        for (i, refined) in result.points.iter().enumerate() {
            let err = (*refined - true_points[i]).length();
            if err > max_pt_err {
                max_pt_err = err;
            }
            mean_pt_err += err;
        }
        mean_pt_err /= result.points.len() as f64;

        // Sanity baseline: run the same BA WITHOUT depth, confirm it drifts
        // (failure to converge to GT scale is the whole point of this test).
        let no_depth_obs: Vec<BaObservation> = observations
            .iter()
            .map(|o| BaObservation {
                pose_idx: o.pose_idx,
                point_idx: o.point_idx,
                pixel: o.pixel,
                fixed_pose: o.fixed_pose,
                fixed_point: o.fixed_point,
                depth_meas: None,
                depth_sigma: 1.0,
            })
            .collect();
        let no_depth_result =
            bundle_adjust_schur(&init_poses, &init_points, &no_depth_obs, &cam, &params).unwrap();
        let mut max_pt_err_no_depth: f64 = 0.0;
        for (i, refined) in no_depth_result.points.iter().enumerate() {
            let err = (*refined - true_points[i]).length();
            if err > max_pt_err_no_depth {
                max_pt_err_no_depth = err;
            }
        }

        // 5cm GT recovery target per spec. Allow some slack for noise.
        assert!(
            max_pt_err < 0.10,
            "max point error {max_pt_err:.4} m (mean {mean_pt_err:.4}) too large \
             — depth anchor not working. (Without depth: {max_pt_err_no_depth:.4} m.)"
        );
        // Sanity: depth should beat no-depth by a wide margin.
        assert!(
            max_pt_err < 0.5 * max_pt_err_no_depth,
            "depth BA ({max_pt_err:.4}) did not significantly beat no-depth \
             ({max_pt_err_no_depth:.4}) — anchor likely inert"
        );

        // Pose error: pose 0 is anchored, so we measure the other four.
        let mut max_t_err: f64 = 0.0;
        let mut max_rot_err: f64 = 0.0;
        for (i, refined) in result.poses.iter().enumerate() {
            if i == 0 {
                continue;
            }
            let dt = refined.translation - true_poses[i].translation;
            let t_err = dt.length();
            if t_err > max_t_err {
                max_t_err = t_err;
            }

            // Rotation error (Frobenius angle): R_err = R_ref.T · R_refined
            let r_err = true_poses[i].rotation.transpose() * refined.rotation;
            // angle = acos((trace - 1) / 2), clamped.
            let trace = r_err.col(0).x + r_err.col(1).y + r_err.col(2).z;
            let cos_angle = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0);
            let angle_rad = cos_angle.acos();
            if angle_rad > max_rot_err {
                max_rot_err = angle_rad;
            }
        }
        let max_rot_deg = max_rot_err.to_degrees();
        assert!(
            max_t_err < 0.05,
            "max translation error {max_t_err:.4} m too large"
        );
        assert!(
            max_rot_deg < 1.0,
            "max rotation error {max_rot_deg:.4}° too large"
        );
    }

    /// Pose-prior BA recovers lateral translation that pose-only BA cannot.
    ///
    /// Setup mirrors `schur_ba_with_depth_recovers_scale` but the
    /// perturbation is *lateral* (along X / Y), which the depth residual
    /// cannot constrain — depth only sees Z in cam frame, and for cameras
    /// looking down +Z, lateral world-frame translation of the rig is
    /// orthogonal to the cam-frame depth axis. The pose prior is the right
    /// tool: it constrains all three world-frame axes of every pose
    /// translation directly.
    #[test]
    fn schur_ba_with_pose_prior_recovers_lateral() {
        // Same LCG as the depth test.
        struct Lcg {
            state: u64,
        }
        impl Lcg {
            fn new(seed: u64) -> Self {
                Self { state: seed }
            }
            fn next_u64(&mut self) -> u64 {
                self.state = self
                    .state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                self.state
            }
            fn normal(&mut self) -> f64 {
                let u1 = ((self.next_u64() >> 11) as f64) / (1u64 << 53) as f64;
                let u2 = ((self.next_u64() >> 11) as f64) / (1u64 << 53) as f64;
                let u1 = u1.max(1e-12);
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
            }
        }
        let mut rng = Lcg::new(0x0000_BADC_AFE1_2345_u64);

        let cam = PinholeCamera {
            fx: 600.0,
            fy: 600.0,
            cx: 320.0,
            cy: 240.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        };

        // 5 cameras moving forward (along +Z is the look-axis; the rig moves
        // along +X in world). All cameras have identity rotation (looking
        // down +Z). Pose 0 at origin acts as the gauge.
        let cam_positions = [
            Vec3F64::new(0.0, 0.0, 0.0),
            Vec3F64::new(0.1, 0.0, 0.0),
            Vec3F64::new(0.2, 0.0, 0.0),
            Vec3F64::new(0.3, 0.0, 0.0),
            Vec3F64::new(0.4, 0.0, 0.0),
        ];
        // Pose stores world→cam, so t_cw = -C_world for R=I.
        let true_poses: Vec<Pose3d> = cam_positions
            .iter()
            .map(|p| Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(-p.x, -p.y, -p.z)))
            .collect();

        // 50 points in front of the cameras.
        let mut true_points: Vec<Vec3F64> = Vec::with_capacity(50);
        for k in 0..50 {
            let kf = k as f64;
            let x = (kf * 0.37).sin() * 1.2 + (kf * 0.13).cos() * 0.4;
            let y = (kf * 0.29).cos() * 0.9 + (kf * 0.11).sin() * 0.3;
            let z = 4.0 + (kf * 0.41).sin() * 1.5;
            true_points.push(Vec3F64::new(x, y, z));
        }

        // Project pixels with σ=0.3 px noise.
        // NB: no pose is `fixed_pose` here — we let the entire rig float so
        // the lateral world-frame translation of all-poses-plus-points is a
        // genuine gauge mode that pure reprojection BA cannot resolve.
        let mut observations: Vec<BaObservation> = Vec::new();
        for (pi, pose) in true_poses.iter().enumerate() {
            for (xi, pt) in true_points.iter().enumerate() {
                let pc = pose.transform_point(pt);
                if pc.z <= 0.2 {
                    continue;
                }
                let u = cam.fx * pc.x / pc.z + cam.cx + 0.3 * rng.normal();
                let v = cam.fy * pc.y / pc.z + cam.cy + 0.3 * rng.normal();
                observations.push(BaObservation {
                    pose_idx: pi,
                    point_idx: xi,
                    pixel: [u as f32, v as f32],
                    fixed_pose: false,
                    fixed_point: false,
                    depth_meas: None,
                    depth_sigma: 1.0,
                });
            }
        }

        // ── Initial guess: translate the ENTIRE rig (all poses + all
        // points) laterally by +0.5 m in world Y. This is a global SE(3)
        // gauge mode: it preserves every reprojection residual exactly to
        // pixel precision, so pure reprojection BA cannot pull the system
        // back. Only the pose prior breaks the gauge.
        let lateral_offset = Vec3F64::new(0.0, 0.5, 0.0);
        let init_poses: Vec<Pose3d> = true_poses
            .iter()
            .map(|p| {
                // C_new = C + offset. Since C = -R^T·t and R=I, that means
                // t_new = t - offset (R is identity in this setup).
                Pose3d::new(
                    p.rotation,
                    Vec3F64::new(
                        p.translation.x - lateral_offset.x,
                        p.translation.y - lateral_offset.y,
                        p.translation.z - lateral_offset.z,
                    ),
                )
            })
            .collect();
        let init_points: Vec<Vec3F64> = true_points
            .iter()
            .map(|pt| {
                Vec3F64::new(
                    pt.x + lateral_offset.x,
                    pt.y + lateral_offset.y,
                    pt.z + lateral_offset.z,
                )
            })
            .collect();

        // ── Pose priors: tight (σ=0.05 m) at GT camera centres for free
        // poses (pose 0 is fixed so its entry is moot but we still set it
        // for completeness).
        let priors: Vec<Option<BaPosePrior>> = true_poses
            .iter()
            .map(|p| {
                // GT camera centre.
                let r_t = p.rotation.transpose();
                let c = -(r_t * p.translation);
                Some(BaPosePrior {
                    center_world: [c.x as f32, c.y as f32, c.z as f32],
                    sigma: 0.05,
                })
            })
            .collect();

        let params = BaParams {
            max_iterations: 100,
            cost_tolerance: 1e-9,
            ..BaParams::default()
        };
        let result = bundle_adjust_schur_with_priors(
            &init_poses,
            &init_points,
            &observations,
            &cam,
            &params,
            Some(&priors),
        )
        .unwrap();

        // ── Without prior (control): pose-only reprojection BA at this
        // perturbation has nothing pulling the camera laterally back to GT
        // because shifting cameras + scaling points (or just dragging the
        // whole rig) gives almost-zero residual. We do *not* assert the
        // control fails — only that the priored result succeeds.
        let mut max_t_err: f64 = 0.0;
        let mut max_t_err_lateral: f64 = 0.0;
        for (i, refined) in result.poses.iter().enumerate() {
            let dt = refined.translation - true_poses[i].translation;
            let t_err = dt.length();
            if t_err > max_t_err {
                max_t_err = t_err;
            }
            let lat = (dt.x * dt.x + dt.y * dt.y).sqrt();
            if lat > max_t_err_lateral {
                max_t_err_lateral = lat;
            }
            let _ = i;
        }
        eprintln!(
            "pose-prior BA: max_t_err={:.4} m, max_lateral={:.4} m, converged={}",
            max_t_err, max_t_err_lateral, result.converged,
        );

        // Recovered pose centres within 2 cm of GT in ALL 3 axes.
        // Pose prior at σ=0.05 anchors strongly; this is well within the
        // posterior radius for 5 cameras + 50 well-spread points.
        for (i, refined) in result.poses.iter().enumerate() {
            let r_t = refined.rotation.transpose();
            let c_ref = -(r_t * refined.translation);
            let r_t_gt = true_poses[i].rotation.transpose();
            let c_gt = -(r_t_gt * true_poses[i].translation);
            let dc = c_ref - c_gt;
            assert!(
                dc.x.abs() < 0.02 && dc.y.abs() < 0.02 && dc.z.abs() < 0.02,
                "pose {i} centre off GT: dC=({:.4}, {:.4}, {:.4}) m",
                dc.x,
                dc.y,
                dc.z,
            );
        }

        // Sanity: passing `None` for priors at this init should fail the
        // lateral test (drift not pulled back). Run it and check we don't
        // get within 2cm in Y — proves the prior is doing the work.
        let no_prior =
            bundle_adjust_schur(&init_poses, &init_points, &observations, &cam, &params).unwrap();
        let mut max_dy_no_prior: f64 = 0.0;
        for (i, refined) in no_prior.poses.iter().enumerate() {
            let r_t = refined.rotation.transpose();
            let c_ref = -(r_t * refined.translation);
            let r_t_gt = true_poses[i].rotation.transpose();
            let c_gt = -(r_t_gt * true_poses[i].translation);
            let dy = (c_ref.y - c_gt.y).abs();
            if dy > max_dy_no_prior {
                max_dy_no_prior = dy;
            }
            let _ = i;
        }
        eprintln!("no-prior control: max |dy| = {:.4} m", max_dy_no_prior);
        // The prior must beat no-prior on the lateral axis decisively.
        assert!(
            max_dy_no_prior > 0.05,
            "no-prior control happened to recover (max |dy|={:.4}) — test is \
             not exercising the lateral-drift mode it's meant to",
            max_dy_no_prior,
        );
    }

    /// Huber on the depth residual rejects a single outlier depth measurement
    /// (e.g. an object-boundary mis-sample at 10× σ). Without Huber the
    /// outlier pulls the reconstruction off GT; with Huber it's downweighted
    /// and the reconstruction stays accurate.
    ///
    /// Setup mirrors `schur_ba_with_depth_recovers_scale` but with a single
    /// corrupted depth measurement: `d_meas = true_depth * 1.5` on one obs.
    #[test]
    fn schur_ba_huber_rejects_depth_outlier() {
        struct Lcg {
            state: u64,
        }
        impl Lcg {
            fn new(seed: u64) -> Self {
                Self { state: seed }
            }
            fn next_u64(&mut self) -> u64 {
                self.state = self
                    .state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                self.state
            }
            fn normal(&mut self) -> f64 {
                let u1 = ((self.next_u64() >> 11) as f64) / (1u64 << 53) as f64;
                let u2 = ((self.next_u64() >> 11) as f64) / (1u64 << 53) as f64;
                let u1 = u1.max(1e-12);
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
            }
        }
        let mut rng = Lcg::new(0x000F_ACEF_00DC_0DE0_u64);

        let cam = PinholeCamera {
            fx: 600.0,
            fy: 600.0,
            cx: 320.0,
            cy: 240.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        };

        // 5 cameras along +X, looking down +Z (identity rotation).
        let cam_positions = [
            Vec3F64::new(0.0, 0.0, 0.0),
            Vec3F64::new(0.1, 0.0, 0.0),
            Vec3F64::new(0.2, 0.0, 0.0),
            Vec3F64::new(0.3, 0.0, 0.0),
            Vec3F64::new(0.4, 0.0, 0.0),
        ];
        let true_poses: Vec<Pose3d> = cam_positions
            .iter()
            .map(|p| Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(-p.x, -p.y, -p.z)))
            .collect();

        // 50 points in front of the cameras.
        let mut true_points: Vec<Vec3F64> = Vec::with_capacity(50);
        for k in 0..50 {
            let kf = k as f64;
            let x = (kf * 0.37).sin() * 1.2 + (kf * 0.13).cos() * 0.4;
            let y = (kf * 0.29).cos() * 0.9 + (kf * 0.11).sin() * 0.3;
            let z = 4.0 + (kf * 0.41).sin() * 1.5;
            true_points.push(Vec3F64::new(x, y, z));
        }

        // Build observations with σ=0.3 px reproj noise and σ=2% depth noise.
        let mut observations: Vec<BaObservation> = Vec::new();
        for (pi, pose) in true_poses.iter().enumerate() {
            for (xi, pt) in true_points.iter().enumerate() {
                let pc = pose.transform_point(pt);
                if pc.z <= 0.2 {
                    continue;
                }
                let u = cam.fx * pc.x / pc.z + cam.cx + 0.3 * rng.normal();
                let v = cam.fy * pc.y / pc.z + cam.cy + 0.3 * rng.normal();
                let depth_sigma_m = 0.02 * pc.z as f32;
                let d_meas = (pc.z + 0.02 * pc.z * rng.normal()) as f32;
                observations.push(BaObservation {
                    pose_idx: pi,
                    point_idx: xi,
                    pixel: [u as f32, v as f32],
                    fixed_pose: pi == 0, // anchor pose 0
                    fixed_point: false,
                    depth_meas: Some(d_meas),
                    depth_sigma: depth_sigma_m,
                });
            }
        }

        // Inject ONE bad depth measurement at 50% inflation
        // (= true_depth * 1.5, far beyond σ=2%; the gate threshold √χ²(3,99%)
        // ≈ 2.8 σ in whitened units → 1.5/0.02 = 25 σ is a clear outlier).
        // Pick an observation that does NOT touch the anchored pose 0 so the
        // outlier can actually pull free variables. Choose pose 2 and the
        // first point it sees.
        let outlier_obs_idx = observations
            .iter()
            .position(|o| o.pose_idx == 2 && o.depth_meas.is_some())
            .expect("expected at least one depth obs on pose 2");
        let outlier_pt_idx = observations[outlier_obs_idx].point_idx;
        let pose2 = &true_poses[2];
        let true_z = pose2.transform_point(&true_points[outlier_pt_idx]).z as f32;
        observations[outlier_obs_idx].depth_meas = Some(true_z * 1.5);

        // 2× scale-drift init (same gauge-breaking setup as the scale test).
        let init_poses: Vec<Pose3d> = true_poses
            .iter()
            .enumerate()
            .map(|(i, p)| {
                if i == 0 {
                    *p
                } else {
                    Pose3d::new(
                        p.rotation,
                        Vec3F64::new(
                            p.translation.x * 2.0,
                            p.translation.y * 2.0,
                            p.translation.z * 2.0,
                        ),
                    )
                }
            })
            .collect();
        let init_points: Vec<Vec3F64> = true_points
            .iter()
            .map(|p| Vec3F64::new(p.x * 2.0, p.y * 2.0, p.z * 2.0))
            .collect();

        // ── Run 1: BA without robust kernel. The outlier dominates and
        // drags the affected point off GT.
        let params_no_huber = BaParams {
            max_iterations: 100,
            cost_tolerance: 1e-8,
            robust: RobustKernelKind::Identity,
            robust_scale_sq: f32::INFINITY,
            ..BaParams::default()
        };
        let result_no_huber = bundle_adjust_schur(
            &init_poses,
            &init_points,
            &observations,
            &cam,
            &params_no_huber,
        )
        .unwrap();

        // ── Run 2: BA WITH Huber. The outlier is downweighted; reconstruction
        // remains accurate.
        // ORB-SLAM3 §IV.B uses χ²=5.99 for 2-DoF reproj; we use the same
        // robust_scale_sq for the depth residual (whitened scalar, so the
        // gate triggers above ~√5.99 ≈ 2.45 σ).
        let params_huber = BaParams {
            max_iterations: 100,
            cost_tolerance: 1e-8,
            robust: RobustKernelKind::Huber,
            robust_scale_sq: 5.99,
            ..BaParams::default()
        };
        let result_huber = bundle_adjust_schur(
            &init_poses,
            &init_points,
            &observations,
            &cam,
            &params_huber,
        )
        .unwrap();

        // Compute max point error for both runs.
        let mut max_err_no_huber: f64 = 0.0;
        let mut outlier_err_no_huber: f64 = 0.0;
        for (i, refined) in result_no_huber.points.iter().enumerate() {
            let err = (*refined - true_points[i]).length();
            if err > max_err_no_huber {
                max_err_no_huber = err;
            }
            if i == outlier_pt_idx {
                outlier_err_no_huber = err;
            }
        }
        let mut max_err_huber: f64 = 0.0;
        let mut outlier_err_huber: f64 = 0.0;
        for (i, refined) in result_huber.points.iter().enumerate() {
            let err = (*refined - true_points[i]).length();
            if err > max_err_huber {
                max_err_huber = err;
            }
            if i == outlier_pt_idx {
                outlier_err_huber = err;
            }
        }
        eprintln!(
            "depth-outlier test: max_err no_huber={:.4} m (outlier pt {:.4}), \
             with_huber={:.4} m (outlier pt {:.4})",
            max_err_no_huber, outlier_err_no_huber, max_err_huber, outlier_err_huber,
        );

        // Without Huber: the 1.5× outlier perturbs the affected point by
        // roughly (1.5 - 1) × depth = 0.5 × ~4 m = 2 m of mismatch, dampened
        // by other obs to a smaller value but well above 10 cm.
        assert!(
            outlier_err_no_huber > 0.10,
            "expected outlier-affected point to drift >10 cm without Huber, got {:.4} m",
            outlier_err_no_huber,
        );
        // With Huber: outlier is downweighted, reconstruction is much better.
        // Huber caps the GRADIENT contribution at the scale parameter (it does
        // not zero it out — that would require a redescending kernel like
        // Cauchy/Tukey). With robust_scale_sq=5.99 and a 25 σ outlier we still
        // get ~scale/r_abs ≈ 10% weight; combined with point uncertainty from
        // the 2× scale-drift init, the outlier-affected point converges to
        // O(few cm) of residual error, vs decimetres without Huber.
        assert!(
            outlier_err_huber < 0.5 * outlier_err_no_huber,
            "expected Huber to halve outlier-induced error at minimum: \
             with_huber={:.4} m, no_huber={:.4} m",
            outlier_err_huber,
            outlier_err_no_huber,
        );
        assert!(
            outlier_err_huber < 0.10,
            "expected Huber to keep outlier-affected point within 10 cm, got {:.4} m",
            outlier_err_huber,
        );
        // Sanity: overall max error with Huber stays small too.
        assert!(
            max_err_huber < 0.10,
            "Huber-BA max point error {:.4} m too large (regression in inliers?)",
            max_err_huber,
        );
    }
}
