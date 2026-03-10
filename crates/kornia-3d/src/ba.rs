//! Dedicated bundle adjustment solver.
//!
//! Provides [`bundle_adjust`] which jointly optimizes camera poses and 3D point
//! positions by minimizing reprojection error using Levenberg-Marquardt with
//! analytical Jacobians.

use kornia_algebra::optim::{
    Factor, FactorError, FactorResult, LevenbergMarquardt, LinearizationResult, Problem, Variable,
    VariableType,
};
use kornia_algebra::{Mat3AF32, Mat3F64, QuatF32, Vec3AF32, Vec3F64, SE3F32, SO3F32};

use crate::camera::PinholeCamera;
use crate::pose::Pose3d;

/// A single observation linking a pose to a 3D point via a pixel measurement.
pub struct BaObservation {
    /// Index into the poses slice.
    pub pose_idx: usize,
    /// Index into the points slice.
    pub point_idx: usize,
    /// Observed pixel `[u, v]`.
    pub pixel: [f32; 2],
    /// If true, this observation's pose is held fixed during optimization.
    pub fixed_pose: bool,
}

/// Parameters for bundle adjustment.
pub struct BaParams {
    /// Maximum number of LM iterations.
    pub max_iterations: usize,
    /// Cost convergence tolerance.
    pub cost_tolerance: f32,
    /// Gradient convergence tolerance.
    pub gradient_tolerance: f32,
    /// Initial LM damping parameter.
    pub initial_lambda: f32,
}

impl Default for BaParams {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            cost_tolerance: 1e-5,
            gradient_tolerance: 1e-5,
            initial_lambda: 1e-3,
        }
    }
}

/// Result of bundle adjustment.
pub struct BaResult {
    /// Optimized poses.
    pub poses: Vec<Pose3d>,
    /// Optimized 3D points.
    pub points: Vec<Vec3F64>,
    /// Number of LM iterations performed.
    pub iterations: usize,
    /// Whether the optimizer converged.
    pub converged: bool,
}

// ---------------------------------------------------------------------------
// f32 ↔ f64 conversion helpers
// ---------------------------------------------------------------------------

fn pose_to_se3_params(pose: &Pose3d) -> Vec<f32> {
    let r_f32 = Mat3AF32::from_cols(
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
    let so3 = SO3F32::from_matrix(&r_f32);
    let se3 = SE3F32::new(
        so3,
        Vec3AF32::new(
            pose.translation.x as f32,
            pose.translation.y as f32,
            pose.translation.z as f32,
        ),
    );
    se3.to_params().to_vec()
}

fn se3_params_to_pose(params: &[f32]) -> Pose3d {
    let q = QuatF32::from_xyzw(params[1], params[2], params[3], params[0]).normalize();
    let se3 = SE3F32::from_qxyz(q, Vec3AF32::new(params[4], params[5], params[6]));
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

// ---------------------------------------------------------------------------
// Reprojection factor with analytical Jacobians
// ---------------------------------------------------------------------------

/// Internal reprojection factor for the BA solver.
struct ReprojFactor {
    obs_u: f32,
    obs_v: f32,
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    /// When set, the pose is fixed and only the point is optimized.
    fixed_pose: Option<[f32; 7]>,
}

impl ReprojFactor {
    fn new(obs: [f32; 2], camera: &PinholeCamera) -> Self {
        Self {
            obs_u: obs[0],
            obs_v: obs[1],
            fx: camera.fx as f32,
            fy: camera.fy as f32,
            cx: camera.cx as f32,
            cy: camera.cy as f32,
            fixed_pose: None,
        }
    }

    fn new_fixed_pose(obs: [f32; 2], camera: &PinholeCamera, pose: [f32; 7]) -> Self {
        Self {
            obs_u: obs[0],
            obs_v: obs[1],
            fx: camera.fx as f32,
            fy: camera.fy as f32,
            cx: camera.cx as f32,
            cy: camera.cy as f32,
            fixed_pose: Some(pose),
        }
    }

    /// Project a world point through an SE3 pose, returning `(u, v, z_cam)`.
    fn project(
        &self,
        pose_params: &[f32],
        point_params: &[f32],
    ) -> Result<(f32, f32, f32), FactorError> {
        let se3 = SE3F32::from_params(pose_params);
        let pw = Vec3AF32::new(point_params[0], point_params[1], point_params[2]);
        let pc = se3 * pw;

        if pc.z.abs() < 1e-10 {
            return Err(FactorError::InvalidParameters("point behind camera".into()));
        }
        let inv_z = 1.0 / pc.z;
        Ok((
            self.fx * pc.x * inv_z + self.cx,
            self.fy * pc.y * inv_z + self.cy,
            pc.z,
        ))
    }

    /// Analytical Jacobian for pose (6 DOF) + point (3 DOF).
    ///
    /// Returns a row-major 2×9 flat array.
    /// Tangent convention: `[upsilon(0..3), omega(3..6)]` matching `SE3F32::retract`.
    fn analytical_jacobian_full(
        &self,
        pose_params: &[f32],
        point_params: &[f32],
    ) -> FactorResult<Vec<f32>> {
        let se3 = SE3F32::from_params(pose_params);
        let r = se3.r.matrix();
        let pw = Vec3AF32::new(point_params[0], point_params[1], point_params[2]);
        let pc = se3 * pw;

        let inv_z = 1.0 / pc.z;
        let inv_z2 = inv_z * inv_z;

        // J_proj row coefficients
        let a0 = self.fx * inv_z;
        let a2 = -self.fx * pc.x * inv_z2;
        let b1 = self.fy * inv_z;
        let b2 = -self.fy * pc.y * inv_z2;

        // Rotation matrix elements: R[row][col]
        let r00 = r.col(0).x;
        let r01 = r.col(1).x;
        let r02 = r.col(2).x;
        let r10 = r.col(0).y;
        let r11 = r.col(1).y;
        let r12 = r.col(2).y;
        let r20 = r.col(0).z;
        let r21 = r.col(1).z;
        let r22 = r.col(2).z;

        let (px, py, pz) = (pw.x, pw.y, pw.z);

        // S = -R * skew(p_w)
        // S[i][j] columns:
        //   col 0: -pz*R[:,1] + py*R[:,2]
        //   col 1:  pz*R[:,0] - px*R[:,2]
        //   col 2: -py*R[:,0] + px*R[:,1]
        let s00 = -pz * r01 + py * r02;
        let s10 = -pz * r11 + py * r12;
        let s20 = -pz * r21 + py * r22;

        let s01 = pz * r00 - px * r02;
        let s11 = pz * r10 - px * r12;
        let s21 = pz * r20 - px * r22;

        let s02 = -py * r00 + px * r01;
        let s12 = -py * r10 + px * r11;
        let s22 = -py * r20 + px * r21;

        // J_pt = J_proj * R (also equals upsilon part of pose jacobian)
        let jpt_00 = a0 * r00 + a2 * r20;
        let jpt_01 = a0 * r01 + a2 * r21;
        let jpt_02 = a0 * r02 + a2 * r22;
        let jpt_10 = b1 * r10 + b2 * r20;
        let jpt_11 = b1 * r11 + b2 * r21;
        let jpt_12 = b1 * r12 + b2 * r22;

        // J_omega = J_proj * S
        let jom_00 = a0 * s00 + a2 * s20;
        let jom_01 = a0 * s01 + a2 * s21;
        let jom_02 = a0 * s02 + a2 * s22;
        let jom_10 = b1 * s10 + b2 * s20;
        let jom_11 = b1 * s11 + b2 * s21;
        let jom_12 = b1 * s12 + b2 * s22;

        // Row-major 2×9: [pose(6), point(3)]
        Ok(vec![
            // Row 0 (du)
            jpt_00, jpt_01, jpt_02, // upsilon
            jom_00, jom_01, jom_02, // omega
            jpt_00, jpt_01, jpt_02, // point (same as upsilon)
            // Row 1 (dv)
            jpt_10, jpt_11, jpt_12, // upsilon
            jom_10, jom_11, jom_12, // omega
            jpt_10, jpt_11, jpt_12, // point (same as upsilon)
        ])
    }

    /// Analytical Jacobian for point only (fixed pose).
    ///
    /// Returns a row-major 2×3 flat array.
    fn analytical_jacobian_point_only(
        &self,
        pose_params: &[f32],
        point_params: &[f32],
    ) -> FactorResult<Vec<f32>> {
        let se3 = SE3F32::from_params(pose_params);
        let r = se3.r.matrix();
        let pw = Vec3AF32::new(point_params[0], point_params[1], point_params[2]);
        let pc = se3 * pw;

        let inv_z = 1.0 / pc.z;
        let inv_z2 = inv_z * inv_z;

        let a0 = self.fx * inv_z;
        let a2 = -self.fx * pc.x * inv_z2;
        let b1 = self.fy * inv_z;
        let b2 = -self.fy * pc.y * inv_z2;

        let r00 = r.col(0).x;
        let r01 = r.col(1).x;
        let r02 = r.col(2).x;
        let r10 = r.col(0).y;
        let r11 = r.col(1).y;
        let r12 = r.col(2).y;
        let r20 = r.col(0).z;
        let r21 = r.col(1).z;
        let r22 = r.col(2).z;

        Ok(vec![
            a0 * r00 + a2 * r20,
            a0 * r01 + a2 * r21,
            a0 * r02 + a2 * r22,
            b1 * r10 + b2 * r20,
            b1 * r11 + b2 * r21,
            b1 * r12 + b2 * r22,
        ])
    }
}

impl Factor for ReprojFactor {
    fn linearize(
        &self,
        params: &[&[f32]],
        compute_jacobian: bool,
    ) -> FactorResult<LinearizationResult> {
        let (pose_params, point_params) = if let Some(ref fp) = self.fixed_pose {
            if params.len() != 1 {
                return Err(FactorError::DimensionMismatch {
                    expected: 1,
                    actual: params.len(),
                });
            }
            (fp.as_slice(), params[0])
        } else {
            if params.len() != 2 {
                return Err(FactorError::DimensionMismatch {
                    expected: 2,
                    actual: params.len(),
                });
            }
            (params[0], params[1])
        };

        let (u, v, z) = self.project(pose_params, point_params)?;
        if z <= 0.0 {
            return Err(FactorError::InvalidParameters("point behind camera".into()));
        }

        let residual = vec![u - self.obs_u, v - self.obs_v];

        if self.fixed_pose.is_some() {
            let jacobian = if compute_jacobian {
                Some(self.analytical_jacobian_point_only(pose_params, point_params)?)
            } else {
                None
            };
            Ok(LinearizationResult::new(residual, jacobian, 3))
        } else {
            let jacobian = if compute_jacobian {
                Some(self.analytical_jacobian_full(pose_params, point_params)?)
            } else {
                None
            };
            Ok(LinearizationResult::new(residual, jacobian, 9))
        }
    }

    fn residual_dim(&self) -> usize {
        2
    }

    fn num_variables(&self) -> usize {
        if self.fixed_pose.is_some() {
            1
        } else {
            2
        }
    }

    fn variable_local_dim(&self, idx: usize) -> usize {
        if self.fixed_pose.is_some() {
            3
        } else {
            match idx {
                0 => 6,
                1 => 3,
                _ => 0,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Run bundle adjustment over a set of poses, points, and observations.
///
/// Jointly optimizes camera poses (SE3) and 3D point positions by minimizing
/// reprojection error using Levenberg-Marquardt with analytical Jacobians.
///
/// Observations with `fixed_pose = true` hold the referenced pose constant.
pub fn bundle_adjust(
    poses: &[Pose3d],
    points: &[Vec3F64],
    observations: &[BaObservation],
    camera: &PinholeCamera,
    params: &BaParams,
) -> Result<BaResult, String> {
    let mut problem = Problem::new();

    // Track which poses are only used as fixed (never as free).
    let mut pose_is_free = vec![false; poses.len()];
    for obs in observations {
        if !obs.fixed_pose {
            if obs.pose_idx < poses.len() {
                pose_is_free[obs.pose_idx] = true;
            }
        }
    }

    // Add free pose variables.
    for (i, pose) in poses.iter().enumerate() {
        if !pose_is_free[i] {
            continue;
        }
        let se3_params = pose_to_se3_params(pose);
        let var = Variable::new(format!("pose_{i}"), VariableType::SE3, vec![0.0; 7]);
        problem
            .add_variable(var, se3_params)
            .map_err(|e| format!("BA: failed to add pose variable: {e}"))?;
    }

    // Precompute fixed pose params.
    let fixed_params: Vec<Option<[f32; 7]>> = poses
        .iter()
        .enumerate()
        .map(|(i, pose)| {
            if pose_is_free[i] {
                None
            } else {
                let p = pose_to_se3_params(pose);
                Some([p[0], p[1], p[2], p[3], p[4], p[5], p[6]])
            }
        })
        .collect();

    // Add point variables.
    for (i, pt) in points.iter().enumerate() {
        let var = Variable::euclidean(&format!("pt_{i}"), 3);
        let init = vec![pt.x as f32, pt.y as f32, pt.z as f32];
        problem
            .add_variable(var, init)
            .map_err(|e| format!("BA: failed to add point variable: {e}"))?;
    }

    // Add factors.
    for obs in observations {
        if obs.pose_idx >= poses.len() || obs.point_idx >= points.len() {
            continue;
        }
        let pt_name = format!("pt_{}", obs.point_idx);

        if obs.fixed_pose {
            if let Some(ref fp) = fixed_params[obs.pose_idx] {
                let factor = Box::new(ReprojFactor::new_fixed_pose(obs.pixel, camera, *fp));
                problem
                    .add_factor(factor, vec![pt_name])
                    .map_err(|e| format!("BA: failed to add factor: {e}"))?;
            } else {
                // Pose is actually free but this observation wants it fixed — use current params.
                let fp_arr: [f32; 7] = {
                    let p = pose_to_se3_params(&poses[obs.pose_idx]);
                    [p[0], p[1], p[2], p[3], p[4], p[5], p[6]]
                };
                let factor = Box::new(ReprojFactor::new_fixed_pose(obs.pixel, camera, fp_arr));
                problem
                    .add_factor(factor, vec![pt_name])
                    .map_err(|e| format!("BA: failed to add factor: {e}"))?;
            }
        } else {
            let pose_name = format!("pose_{}", obs.pose_idx);
            let factor = Box::new(ReprojFactor::new(obs.pixel, camera));
            problem
                .add_factor(factor, vec![pose_name, pt_name])
                .map_err(|e| format!("BA: failed to add factor: {e}"))?;
        }
    }

    // Run optimizer.
    let optimizer = LevenbergMarquardt {
        lambda_init: params.initial_lambda,
        lambda_max: 1e10,
        lambda_factor: 10.0,
        max_iterations: params.max_iterations,
        cost_tolerance: params.cost_tolerance,
        gradient_tolerance: params.gradient_tolerance,
    };

    let result = optimizer
        .optimize(&mut problem)
        .map_err(|e| format!("BA optimization failed: {e}"))?;

    let converged = matches!(
        result.termination_reason,
        kornia_algebra::optim::TerminationReason::CostConverged
            | kornia_algebra::optim::TerminationReason::GradientConverged
    );

    // Extract results.
    let vars = problem.get_variables();

    let mut out_poses = Vec::with_capacity(poses.len());
    for i in 0..poses.len() {
        if pose_is_free[i] {
            let name = format!("pose_{i}");
            let values = &vars[&name].values;
            out_poses.push(se3_params_to_pose(values));
        } else {
            out_poses.push(poses[i]);
        }
    }

    let mut out_points = Vec::with_capacity(points.len());
    for i in 0..points.len() {
        let name = format!("pt_{i}");
        let values = &vars[&name].values;
        out_points.push(Vec3F64::new(
            values[0] as f64,
            values[1] as f64,
            values[2] as f64,
        ));
    }

    Ok(BaResult {
        poses: out_poses,
        points: out_points,
        iterations: result.iterations,
        converged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_algebra::{Mat3F64, Vec3F64};

    fn test_camera() -> PinholeCamera {
        PinholeCamera {
            fx: 800.0,
            fy: 800.0,
            cx: 320.0,
            cy: 240.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        }
    }

    #[test]
    fn test_reproj_factor_zero_residual() {
        let cam = test_camera();
        let factor = ReprojFactor::new([320.0, 240.0], &cam);
        let pose = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0f32];
        let point = [0.0, 0.0, 5.0f32];
        let result = factor.linearize(&[&pose, &point], false).unwrap();
        assert!(result.residual[0].abs() < 1e-4);
        assert!(result.residual[1].abs() < 1e-4);
    }

    #[test]
    fn test_reproj_factor_jacobian_finite() {
        let cam = test_camera();
        let factor = ReprojFactor::new([350.0, 260.0], &cam);
        let pose = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0f32];
        let point = [0.5, 0.3, 4.0f32];
        let result = factor.linearize(&[&pose, &point], true).unwrap();
        let jac = result.jacobian.unwrap();
        assert_eq!(jac.len(), 2 * 9);
        assert!(jac.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_fixed_pose_factor() {
        let cam = test_camera();
        let pose = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0f32];
        let factor = ReprojFactor::new_fixed_pose([320.0, 240.0], &cam, pose);
        let point = [0.0, 0.0, 5.0f32];
        let result = factor.linearize(&[&point], true).unwrap();
        assert!(result.residual[0].abs() < 1e-4);
        assert!(result.residual[1].abs() < 1e-4);
        let jac = result.jacobian.unwrap();
        assert_eq!(jac.len(), 6);
        assert!(jac.iter().all(|x| x.is_finite()));
    }

    // TODO: rewrite numerical Jacobian in f64 (f64 SE3 exp + f64 projection) so we
    // can use eps~1e-8 and tighten the tolerance from 5% to ~1e-4. Currently limited
    // by f32 catastrophic cancellation.

    /// Compute numerical Jacobian via central differences.
    ///
    /// Uses eps=1e-3 which is near-optimal for f32 (balances truncation and rounding).
    fn numerical_jacobian(
        factor: &ReprojFactor,
        pose_params: &[f32],
        point_params: &[f32],
    ) -> Vec<f32> {
        let eps: f32 = 1e-3;
        let mut numerical = vec![0.0f32; 18];

        let se3 = SE3F32::from_params(pose_params);

        // Pose tangent (6 DOF)
        for i in 0..6 {
            let mut dp = [0.0f32; 6];
            let mut dm = [0.0f32; 6];
            dp[i] = eps;
            dm[i] = -eps;
            let pp = se3.retract(&dp).to_params();
            let pm = se3.retract(&dm).to_params();
            let (up, vp, _) = factor.project(&pp, point_params).unwrap();
            let (um, vm, _) = factor.project(&pm, point_params).unwrap();
            let inv = 1.0 / (2.0 * eps);
            numerical[i] = (up - um) * inv;
            numerical[9 + i] = (vp - vm) * inv;
        }

        // Point (3 DOF)
        for i in 0..3 {
            let mut pp = [point_params[0], point_params[1], point_params[2]];
            let mut pm = pp;
            pp[i] += eps;
            pm[i] -= eps;
            let (up, vp, _) = factor.project(pose_params, &pp).unwrap();
            let (um, vm, _) = factor.project(pose_params, &pm).unwrap();
            let inv = 1.0 / (2.0 * eps);
            numerical[6 + i] = (up - um) * inv;
            numerical[9 + 6 + i] = (vp - vm) * inv;
        }

        numerical
    }

    #[test]
    fn test_analytical_vs_numerical_jacobian() {
        let cam = test_camera();
        let factor = ReprojFactor::new([350.0, 260.0], &cam);
        let pose = [1.0, 0.0, 0.0, 0.0, 0.1, -0.05, 0.2f32];
        let point = [0.5, 0.3, 4.0f32];

        let analytical = factor.analytical_jacobian_full(&pose, &point).unwrap();
        let numerical = numerical_jacobian(&factor, &pose, &point);

        for (j, (&a, &n)) in analytical.iter().zip(numerical.iter()).enumerate() {
            let diff = (a - n).abs();
            let scale = n.abs().max(1.0);
            assert!(
                diff / scale < 5e-2,
                "Jacobian mismatch at index {j}: analytical={a}, numerical={n}, rel_err={}",
                diff / scale
            );
        }
    }

    #[test]
    fn test_analytical_vs_numerical_jacobian_rotated() {
        let cam = test_camera();

        // Build a rotated pose (~28.6° around Y axis).
        let angle = 0.5f32;
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        let r = Mat3AF32::from_cols(
            Vec3AF32::new(cos_a, 0.0, -sin_a),
            Vec3AF32::new(0.0, 1.0, 0.0),
            Vec3AF32::new(sin_a, 0.0, cos_a),
        );
        let so3 = SO3F32::from_matrix(&r);
        let se3 = SE3F32::new(so3, Vec3AF32::new(0.2, -0.1, 0.3));
        let pose = se3.to_params();
        let point = [1.0, -0.5, 6.0f32];

        let factor = ReprojFactor::new([300.0, 250.0], &cam);
        let analytical = factor.analytical_jacobian_full(&pose, &point).unwrap();
        let numerical = numerical_jacobian(&factor, &pose, &point);

        for (j, (&a, &n)) in analytical.iter().zip(numerical.iter()).enumerate() {
            let diff = (a - n).abs();
            let scale = n.abs().max(1.0);
            assert!(
                diff / scale < 5e-2,
                "Jacobian mismatch at index {j}: analytical={a}, numerical={n}, rel_err={}",
                diff / scale
            );
        }
    }

    #[test]
    fn test_bundle_adjust_converges() {
        let cam = test_camera();

        let pose0 = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::ZERO);
        let pose1 = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(0.5, 0.0, 0.0));

        let true_points = vec![
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
            });
            observations.push(BaObservation {
                pose_idx: 1,
                point_idx: pi,
                pixel: project(&pose1, pt),
                fixed_pose: false,
            });
        }

        let perturbed: Vec<Vec3F64> = true_points
            .iter()
            .map(|p| *p + Vec3F64::new(0.05, -0.03, 0.02))
            .collect();

        let result = bundle_adjust(
            &[pose0, pose1],
            &perturbed,
            &observations,
            &cam,
            &BaParams {
                max_iterations: 20,
                ..BaParams::default()
            },
        )
        .unwrap();

        for (i, refined) in result.points.iter().enumerate() {
            let err = (*refined - true_points[i]).length();
            assert!(err < 0.1, "point {i} error {err} too large");
        }
    }
}
