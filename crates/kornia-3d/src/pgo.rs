//! SE(3) pose graph optimization.
//!
//! Each constraint is a [`RelPoseFactor`] linking two SE(3) pose variables
//! `T_a`, `T_b` (world→cam) with a measured relative transform
//! `T_ab_meas` (cam_a → cam_b). The 6-dimensional residual is
//!
//!     r = log(T_ab_meas⁻¹ · T_b · T_a⁻¹)
//!
//! in `se(3)` tangent (right-perturbation convention, `[ρ; ω]` =
//! `[upsilon; omega]`).
//!
//! Jacobians are computed numerically (central differences in the SE(3)
//! tangent, evaluated via `SE3F32::retract`). For pose-graph problems the
//! residual evaluation is cheap and Jacobians end up well-conditioned —
//! analytical Jacobians (via the SE(3) adjoint) can be plugged in later
//! as an optimization.

use kornia_algebra::optim::{
    Factor, FactorError, FactorResult, LevenbergMarquardt, LinearizationResult, OptimizerError,
    Problem, ProblemError, RobustLoss, TerminationReason, Variable, VariableType,
};
use kornia_algebra::{Mat3F64, SE3F32, Vec3AF32, Vec3F64};
use std::sync::Arc;
use thiserror::Error;

use crate::pose::Pose3d;

const NUM_JACOBIAN_EPS: f32 = 1e-3;

/// Errors raised by [`pose_graph_optimize`].
#[derive(Debug, Error)]
pub enum PgoError {
    /// Problem setup error.
    #[error("Problem setup error: {0}")]
    Problem(#[from] ProblemError),
    /// Optimizer error.
    #[error("Optimizer error: {0}")]
    Optimizer(#[from] OptimizerError),
    /// Invalid input.
    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

/// One pose graph edge.
pub struct PgoEdge {
    /// Index of pose A in the poses slice.
    pub pose_a: usize,
    /// Index of pose B in the poses slice.
    pub pose_b: usize,
    /// Measured relative transform `T_ab` taking cam_a's frame to cam_b's
    /// frame (i.e. `T_b · T_a⁻¹` if `T_a`, `T_b` are world→cam).
    pub t_ab_meas: SE3F32,
    /// Edge weight (multiplies the residual). Use 1.0 for sequential
    /// odometry edges and ≤1 for less-trusted loop edges.
    pub weight: f32,
}

/// Parameters for [`pose_graph_optimize`].
pub struct PgoParams {
    /// Maximum LM iterations.
    pub max_iterations: usize,
    /// Cost convergence tolerance.
    pub cost_tolerance: f32,
    /// Gradient convergence tolerance.
    pub gradient_tolerance: f32,
    /// Initial LM damping (λ).
    pub initial_lambda: f32,
}

impl Default for PgoParams {
    fn default() -> Self {
        Self {
            max_iterations: 30,
            cost_tolerance: 1e-6,
            gradient_tolerance: 1e-6,
            initial_lambda: 1e-3,
        }
    }
}

/// Result of [`pose_graph_optimize`].
pub struct PgoResult {
    /// Optimized poses (world→cam).
    pub poses: Vec<Pose3d>,
    /// LM iterations performed.
    pub iterations: usize,
    /// Whether the optimizer converged.
    pub converged: bool,
}

// ───── f32 ↔ f64 conversion helpers (shared shape with ba.rs) ─────────────

fn pose_to_se3_params(pose: &Pose3d) -> Vec<f32> {
    use kornia_algebra::{Mat3AF32, SO3F32};
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
    use kornia_algebra::QuatF32;
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

// ───── Factor ─────────────────────────────────────────────────────────────

struct RelPoseFactor {
    t_ab_meas_inv: SE3F32,
    weight: f32,
    fixed_a: Option<[f32; 7]>,
    fixed_b: Option<[f32; 7]>,
    loss: Option<Arc<dyn RobustLoss>>,
}

impl RelPoseFactor {
    fn new(t_ab_meas: SE3F32, weight: f32) -> Self {
        Self {
            t_ab_meas_inv: t_ab_meas.inverse(),
            weight,
            fixed_a: None,
            fixed_b: None,
            loss: None,
        }
    }

    fn with_fixed_a(mut self, params: [f32; 7]) -> Self {
        self.fixed_a = Some(params);
        self
    }

    fn with_fixed_b(mut self, params: [f32; 7]) -> Self {
        self.fixed_b = Some(params);
        self
    }

    /// Compute the 6-d residual `weight · log(T_ab_meas⁻¹ · T_b · T_a⁻¹)`.
    fn residual_at(&self, params_a: &[f32], params_b: &[f32]) -> [f32; 6] {
        let t_a = SE3F32::from_params(params_a);
        let t_b = SE3F32::from_params(params_b);
        let t_err = self.t_ab_meas_inv * (t_b * t_a.inverse());
        let (rho, omega) = t_err.log();
        [
            self.weight * rho.x,
            self.weight * rho.y,
            self.weight * rho.z,
            self.weight * omega.x,
            self.weight * omega.y,
            self.weight * omega.z,
        ]
    }
}

impl Factor for RelPoseFactor {
    fn linearize(
        &self,
        params: &[&[f32]],
        compute_jacobian: bool,
    ) -> FactorResult<LinearizationResult> {
        // Resolve (params_a, params_b) accounting for fixed-pose variants.
        let (params_a, params_b, n_free): (&[f32], &[f32], usize) =
            match (&self.fixed_a, &self.fixed_b) {
                (None, None) => {
                    if params.len() != 2 {
                        return Err(FactorError::DimensionMismatch {
                            expected: 2,
                            actual: params.len(),
                        });
                    }
                    (params[0], params[1], 2)
                }
                (Some(fa), None) => {
                    if params.len() != 1 {
                        return Err(FactorError::DimensionMismatch {
                            expected: 1,
                            actual: params.len(),
                        });
                    }
                    (fa.as_slice(), params[0], 1)
                }
                (None, Some(fb)) => {
                    if params.len() != 1 {
                        return Err(FactorError::DimensionMismatch {
                            expected: 1,
                            actual: params.len(),
                        });
                    }
                    (params[0], fb.as_slice(), 1)
                }
                (Some(_), Some(_)) => {
                    // Both fixed → constant factor, contributes nothing.
                    return Err(FactorError::DimensionMismatch { expected: 1, actual: 0 });
                }
            };

        let r0 = self.residual_at(params_a, params_b);
        let residual = r0.to_vec();

        if !compute_jacobian {
            return Ok(LinearizationResult::new(residual, None, n_free * 6));
        }

        // Numerical Jacobian: central differences in the SE(3) tangent of each
        // free variable. 6 deltas per free pose, 6-dim residual → 6 × (n_free*6).
        let mut jacobian = vec![0.0_f32; 6 * n_free * 6];

        let perturb_a = self.fixed_a.is_none();
        let perturb_b = self.fixed_b.is_none();

        let t_a_base = SE3F32::from_params(params_a);
        let t_b_base = SE3F32::from_params(params_b);
        let mut col_start_b = 0;

        if perturb_a {
            for j in 0..6 {
                let mut dp = [0.0_f32; 6];
                let mut dm = [0.0_f32; 6];
                dp[j] = NUM_JACOBIAN_EPS;
                dm[j] = -NUM_JACOBIAN_EPS;
                let pa_plus = t_a_base.retract(&dp).to_params();
                let pa_minus = t_a_base.retract(&dm).to_params();
                let rp = self.residual_at(&pa_plus, params_b);
                let rm = self.residual_at(&pa_minus, params_b);
                let inv = 1.0 / (2.0 * NUM_JACOBIAN_EPS);
                for i in 0..6 {
                    jacobian[i * (n_free * 6) + j] = (rp[i] - rm[i]) * inv;
                }
            }
            col_start_b = 6;
        }
        if perturb_b {
            for j in 0..6 {
                let mut dp = [0.0_f32; 6];
                let mut dm = [0.0_f32; 6];
                dp[j] = NUM_JACOBIAN_EPS;
                dm[j] = -NUM_JACOBIAN_EPS;
                let pb_plus = t_b_base.retract(&dp).to_params();
                let pb_minus = t_b_base.retract(&dm).to_params();
                let rp = self.residual_at(params_a, &pb_plus);
                let rm = self.residual_at(params_a, &pb_minus);
                let inv = 1.0 / (2.0 * NUM_JACOBIAN_EPS);
                for i in 0..6 {
                    jacobian[i * (n_free * 6) + col_start_b + j] = (rp[i] - rm[i]) * inv;
                }
            }
        }

        Ok(LinearizationResult::new(
            residual,
            Some(jacobian),
            n_free * 6,
        ))
    }

    fn residual_dim(&self) -> usize {
        6
    }

    fn num_variables(&self) -> usize {
        match (&self.fixed_a, &self.fixed_b) {
            (None, None) => 2,
            (Some(_), None) | (None, Some(_)) => 1,
            (Some(_), Some(_)) => 0,
        }
    }

    fn variable_local_dim(&self, _idx: usize) -> usize {
        6
    }

    fn get_loss(&self) -> Option<&dyn RobustLoss> {
        self.loss.as_deref()
    }
}

// ───── Driver ─────────────────────────────────────────────────────────────

/// Solve an SE(3) pose graph. Each edge contributes a 6-d residual
/// `weight · log(T_ab_meas⁻¹ · T_b · T_a⁻¹)`. Poses listed in
/// `fixed_pose_indices` are held constant (use this to anchor the graph;
/// `[0]` is typical).
pub fn pose_graph_optimize(
    poses: &[Pose3d],
    edges: &[PgoEdge],
    fixed_pose_indices: &[usize],
    params: &PgoParams,
) -> Result<PgoResult, PgoError> {
    if poses.is_empty() {
        return Err(PgoError::InvalidInput("empty poses".into()));
    }
    if edges.is_empty() {
        return Err(PgoError::InvalidInput("empty edges".into()));
    }

    let fixed_set: std::collections::HashSet<usize> = fixed_pose_indices.iter().copied().collect();

    let mut problem = Problem::new();

    // Track which poses are free (touched by any edge that doesn't fix them).
    let mut pose_is_free = vec![false; poses.len()];
    for edge in edges {
        if !fixed_set.contains(&edge.pose_a) && edge.pose_a < poses.len() {
            pose_is_free[edge.pose_a] = true;
        }
        if !fixed_set.contains(&edge.pose_b) && edge.pose_b < poses.len() {
            pose_is_free[edge.pose_b] = true;
        }
    }

    // Add free pose variables.
    for (i, pose) in poses.iter().enumerate() {
        if !pose_is_free[i] {
            continue;
        }
        let se3_params = pose_to_se3_params(pose);
        let var = Variable::new(format!("pose_{i}"), VariableType::SE3, vec![0.0; 7]);
        problem.add_variable(var, se3_params)?;
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

    // Add factors.
    for edge in edges {
        if edge.pose_a >= poses.len() || edge.pose_b >= poses.len() {
            continue;
        }
        let a_free = pose_is_free[edge.pose_a];
        let b_free = pose_is_free[edge.pose_b];

        let mut factor = RelPoseFactor::new(edge.t_ab_meas, edge.weight);
        let mut vars: Vec<String> = Vec::with_capacity(2);

        match (a_free, b_free) {
            (true, true) => {
                vars.push(format!("pose_{}", edge.pose_a));
                vars.push(format!("pose_{}", edge.pose_b));
            }
            (false, true) => {
                let fp = fixed_params[edge.pose_a].unwrap();
                factor = factor.with_fixed_a(fp);
                vars.push(format!("pose_{}", edge.pose_b));
            }
            (true, false) => {
                let fp = fixed_params[edge.pose_b].unwrap();
                factor = factor.with_fixed_b(fp);
                vars.push(format!("pose_{}", edge.pose_a));
            }
            (false, false) => continue, // both fixed — skip
        }

        problem.add_factor(Box::new(factor), vars)?;
    }

    // Run LM.
    let optimizer = LevenbergMarquardt {
        lambda_init: params.initial_lambda,
        lambda_max: 1e10,
        lambda_factor: 10.0,
        max_iterations: params.max_iterations,
        cost_tolerance: params.cost_tolerance,
        gradient_tolerance: params.gradient_tolerance,
    };
    let result = optimizer.optimize(&mut problem)?;

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

    let converged = matches!(
        result.termination_reason,
        TerminationReason::CostConverged | TerminationReason::GradientConverged
    );

    Ok(PgoResult {
        poses: out_poses,
        iterations: result.iterations,
        converged,
    })
}
