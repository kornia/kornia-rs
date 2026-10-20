//! Levenberg-Marquardt pose refinement for PnP solutions.
//!
//! This module provides LM-based nonlinear refinement for camera pose estimates,
//! leveraging the optimization infrastructure from `kornia-algebra` using factor graph.

use crate::ransac::RobustKernelKind;
use kornia_algebra::optim::{CauchyLoss, HuberLoss, RobustLoss};
use kornia_algebra::optim::{
    Factor, FactorError, FactorResult, LevenbergMarquardt, LinearizationResult, Problem, Variable,
    VariableType,
};
use kornia_algebra::{Mat3AF32, QuatF32, Vec2F32, Vec3AF32, SE3F32, SO3F32};
use kornia_imgproc::calibration::{
    distortion::{distort_point_polynomial, PolynomialDistortion},
    CameraIntrinsic,
};
use std::sync::Arc;

use super::ops::compute_centroid;
use super::{PnPError, PnPResult};

const DEFAULT_MIN_Z_DEPTH: f32 = 1e-10;

/// Parameters controlling the LM pose refinement.
#[derive(Debug, Clone)]
pub struct LMRefineParams {
    /// Maximum number of LM iterations.
    pub max_iterations: usize,
    /// Convergence threshold on cost function change.
    pub cost_tolerance: f32,
    /// Convergence threshold on gradient norm.
    pub gradient_tolerance: f32,
    /// Initial damping factor (lambda).
    pub initial_lambda: f32,
    /// M-estimator kernel applied per-residual (pixel-space, unlike
    /// lightweight_vio's chi-square-whitened residuals — this crate's
    /// `ReprojectionFactor` residual is raw pixel error). `Identity`
    /// collapses to plain L2 (today's behavior, so this field is
    /// backward-compatible).
    pub robust: RobustKernelKind,
    /// Squared scale for `robust`; its square root is the linear delta fed
    /// to `HuberLoss::new`/`CauchyLoss::new`. Default `f32::INFINITY`
    /// collapses to L2 regardless of `robust`.
    pub robust_scale_sq: f32,
}

impl Default for LMRefineParams {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            cost_tolerance: 1e-6,
            gradient_tolerance: 1e-6,
            initial_lambda: 1e-3,
            robust: RobustKernelKind::Identity,
            robust_scale_sq: f32::INFINITY,
        }
    }
}

impl LMRefineParams {
    /// Create default refinement parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum iterations.
    pub fn with_max_iterations(self, max_iters: usize) -> Self {
        Self {
            max_iterations: max_iters,
            ..self
        }
    }

    /// Set cost tolerance.
    pub fn with_cost_tolerance(self, tol: f32) -> Self {
        Self {
            cost_tolerance: tol,
            ..self
        }
    }

    /// Set gradient tolerance.
    pub fn with_gradient_tolerance(self, tol: f32) -> Self {
        Self {
            gradient_tolerance: tol,
            ..self
        }
    }

    /// Set initial lambda.
    pub fn with_initial_lambda(self, lambda: f32) -> Self {
        Self {
            initial_lambda: lambda,
            ..self
        }
    }
}

/// A reprojection factor for PnP optimization.
///
/// This factor computes the residual between observed 2D image points and
/// the projection of 3D world points through the camera model.
///
/// Residual: `r = [u - u_hat, v - v_hat]`
/// where `(u_hat, v_hat) = project(R * P_world + t)`
pub struct ReprojectionFactor {
    /// 3D point in world coordinates.
    point_world: Vec3AF32,
    /// Observed 2D point in image coordinates.
    point_image: Vec2F32,
    /// Camera intrinsics: fx
    fx: f32,
    /// Camera intrinsics: fy
    fy: f32,
    /// Camera intrinsics: cx
    cx: f32,
    /// Camera intrinsics: cy
    cy: f32,
    /// Optional distortion coefficients (k1, k2, p1, p2, k3, k4, k5, k6)
    distortion_coeffs: Option<[f64; 8]>,
    /// Optional robust loss applied to this observation's residual by the optimizer
    /// via 'Factor::get_loss'.
    loss: Option<Arc<dyn RobustLoss>>,
}

impl ReprojectionFactor {
    /// Create a new reprojection factor.
    ///
    /// # Arguments
    ///
    /// * `point_world` - 3D point in world coordinates
    /// * `point_image` - Observed 2D point in image coordinates
    /// * `k` - Camera intrinsic matrix
    /// * `distortion` - Optional camera distortion model
    pub fn new(
        point_world: Vec3AF32,
        point_image: Vec2F32,
        k: &Mat3AF32,
        distortion: Option<&PolynomialDistortion>,
    ) -> Self {
        // Extract distortion coefficients if provided
        // PolynomialDistortion has: k1, k2, k3, k4, k5, k6, p1, p2
        let distortion_coeffs =
            distortion.map(|d| [d.k1, d.k2, d.k3, d.k4, d.k5, d.k6, d.p1, d.p2]);

        Self {
            point_world,
            point_image,
            fx: k.x_axis().x,
            fy: k.y_axis().y,
            cx: k.z_axis().x,
            cy: k.z_axis().y,
            distortion_coeffs,
            loss: None,
        }
    }

    /// Attach a shared robust loss; called per-factor once `LMRefineParams`
    /// has been translated into a concrete `RobustLoss` impl.
    pub fn with_loss(mut self, loss: Option<Arc<dyn RobustLoss>>) -> Self {
        self.loss = loss;
        self
    }

    /// Project a 3D point to 2D using the given pose (SE3 parameters as array).
    fn project(&self, pose_params: &[f32]) -> Result<(f32, f32, f32), FactorError> {
        if pose_params.len() < 7 {
            return Err(FactorError::DimensionMismatch {
                expected: 7,
                actual: pose_params.len(),
            });
        }

        // Normalize quaternion before creating SE3
        // pose_params is [qw, qx, qy, qz, tx, ty, tz]
        let q = QuatF32::from_xyzw(
            pose_params[1],
            pose_params[2],
            pose_params[3],
            pose_params[0],
        );
        let q_normalized = q.normalize();
        let t = Vec3AF32::new(pose_params[4], pose_params[5], pose_params[6]);

        // Create SE3 with normalized quaternion
        let camera_pose = SE3F32::from_qxyz(q_normalized, t);

        let pc = camera_pose * self.point_world; // camera-frame point

        let z = pc.z;
        if z.abs() < DEFAULT_MIN_Z_DEPTH {
            return Err(FactorError::InvalidParameters(
                "Point behind or too close to camera".to_string(),
            ));
        }

        let inv_z = 1.0 / z;
        let u_undist = self.fx * pc.x * inv_z + self.cx;
        let v_undist = self.fy * pc.y * inv_z + self.cy;

        // Apply distortion model if provided
        let (u_hat, v_hat) = if let Some(ref coeffs) = self.distortion_coeffs {
            let cam_intr = CameraIntrinsic {
                fx: self.fx as f64,
                fy: self.fy as f64,
                cx: self.cx as f64,
                cy: self.cy as f64,
            };
            let distortion = PolynomialDistortion {
                k1: coeffs[0],
                k2: coeffs[1],
                k3: coeffs[2],
                k4: coeffs[3],
                k5: coeffs[4],
                k6: coeffs[5],
                p1: coeffs[6],
                p2: coeffs[7],
            };
            let (ud, vd) =
                distort_point_polynomial(u_undist as f64, v_undist as f64, &cam_intr, &distortion);
            (ud as f32, vd as f32)
        } else {
            (u_undist, v_undist)
        };

        Ok((u_hat, v_hat, z))
    }

    /// Compute numerical Jacobian using central differences.
    fn numerical_jacobian(&self, pose_params: &[f32]) -> FactorResult<Vec<f32>> {
        const EPS: f32 = 1e-6;
        const DOF: usize = 6;

        let mut jacobian = vec![0.0f32; 2 * DOF]; // 2 residuals x 6 DOF

        let (_u0, _v0, _) = self.project(pose_params)?;

        // Compute Jacobian w.r.t. each tangent space dimension
        for i in 0..DOF {
            // Create perturbed poses using retraction
            let mut delta_plus = [0.0f32; 6];
            let mut delta_minus = [0.0f32; 6];
            delta_plus[i] = EPS;
            delta_minus[i] = -EPS;

            let pose = SE3F32::from_params(pose_params);
            let pose_plus = pose.retract(&delta_plus);
            let pose_minus = pose.retract(&delta_minus);

            let params_plus = pose_plus.to_params();
            let params_minus = pose_minus.to_params();

            let (u_plus, v_plus, _) = self.project(&params_plus)?;
            let (u_minus, v_minus, _) = self.project(&params_minus)?;

            // Central difference
            let inv_2eps = 1.0 / (2.0 * EPS);
            // Jacobian of residual = -(observed - projected) = projected - observed
            // d(residual)/d(param) = d(u_hat - u)/d(param) = d(u_hat)/d(param)
            jacobian[i] = (u_plus - u_minus) * inv_2eps; // d(u_hat)/d(delta_i)
            jacobian[DOF + i] = (v_plus - v_minus) * inv_2eps; // d(v_hat)/d(delta_i)
        }

        Ok(jacobian)
    }
}

impl Factor for ReprojectionFactor {
    fn linearize(
        &self,
        params: &[&[f32]],
        compute_jacobian: bool,
    ) -> FactorResult<LinearizationResult> {
        if params.len() != 1 {
            return Err(FactorError::DimensionMismatch {
                expected: 1,
                actual: params.len(),
            });
        }

        let pose_params = params[0];
        if pose_params.len() < 7 {
            return Err(FactorError::DimensionMismatch {
                expected: 7,
                actual: pose_params.len(),
            });
        }

        // Project point and compute residual
        let (u_hat, v_hat, z) = self.project(pose_params)?;

        // Check if point is behind camera
        if z <= 0.0 {
            return Err(FactorError::InvalidParameters(
                "Point behind camera".to_string(),
            ));
        }

        // Residual: predicted - observed (convention for minimization)
        let residual = vec![u_hat - self.point_image.x, v_hat - self.point_image.y];

        let jacobian = if compute_jacobian {
            Some(self.numerical_jacobian(pose_params)?)
        } else {
            None
        };

        Ok(LinearizationResult::new(residual, jacobian, 6))
    }

    fn get_loss(&self) -> Option<&dyn RobustLoss> {
        self.loss.as_deref()
    }

    fn residual_dim(&self) -> usize {
        2
    }

    fn num_variables(&self) -> usize {
        1
    }

    fn variable_local_dim(&self, _idx: usize) -> usize {
        6 // SE3 local tangent dimension
    }
}

/// Translate `LMRefineParams::robust`/`robust_scale_sq` into a concrete
/// `RobustLoss` shared across all reprojection factors, or `None` for plain L2.
fn build_robust_loss(params: &LMRefineParams) -> Result<Option<Arc<dyn RobustLoss>>, PnPError> {
    if params.robust == RobustKernelKind::Identity || !params.robust_scale_sq.is_finite() {
        return Ok(None);
    }

    let delta = params.robust_scale_sq.sqrt();
    let loss: Arc<dyn RobustLoss> = match params.robust {
        RobustKernelKind::Identity => unreachable!("handled above"),
        RobustKernelKind::Huber => Arc::new(HuberLoss::new(delta).map_err(|e| {
            PnPError::SvdFailed(format!("Invalid Huber robust loss parameters: {e}"))
        })?),
        RobustKernelKind::Cauchy => Arc::new(CauchyLoss::new(delta).map_err(|e| {
            PnPError::SvdFailed(format!("Invalid Cauchy robust loss parameters: {e}"))
        })?),
        RobustKernelKind::Tukey => {
            return Err(PnPError::SvdFailed(
                "Tukey robust kernel is not supported for LM pose refinement".to_string(),
            ));
        }
    };

    Ok(Some(loss))
}

/// The world frame the solver works in: `x' = scale · (x − centre)`, with `centre` the centroid
/// of the points and `scale` one over their median distance from the starting camera.
///
/// Reprojection does not change when the world and the camera move together by a similarity, but
/// the `f32` solver does: with large coordinates its finite-difference steps and pose updates are
/// lost in rounding. In this frame every number stays near 1. The centre is on the points, not on
/// the camera: about the camera, a small rotation and a sideways shift move a narrow view almost
/// the same way, and the finite differences cannot tell them apart.
struct WorkFrame {
    centre: Vec3AF32,
    scale: f32,
}

impl WorkFrame {
    /// The frame for the camera pose `(rotation, translation)` (world → camera), or `None` if a
    /// world point or the pose is not finite, or more than half of the points sit on the camera.
    fn new(points_world: &[Vec3AF32], rotation: Mat3AF32, translation: Vec3AF32) -> Option<Self> {
        let centre = compute_centroid(points_world);
        if !(centre.x.is_finite() && centre.y.is_finite() && centre.z.is_finite()) {
            return None;
        }
        let camera = -(rotation.transpose() * translation);
        let mut distances: Vec<f32> = points_world
            .iter()
            .map(|&p| (p - camera).length())
            .collect();
        let mid = distances.len() / 2;
        let (_, median, _) = distances.select_nth_unstable_by(mid, f32::total_cmp);
        // A non-finite start pose, or most points sitting on the start camera, has no scale.
        let scale = 1.0 / *median;
        (scale.is_finite() && scale > 0.0).then_some(Self { centre, scale })
    }

    /// A world point in this frame.
    fn point(&self, p: Vec3AF32) -> Vec3AF32 {
        (p - self.centre) * self.scale
    }

    /// The translation of a world → camera pose with this rotation, in this frame.
    fn translation_in(&self, rotation: Mat3AF32, translation: Vec3AF32) -> Vec3AF32 {
        (translation + rotation * self.centre) * self.scale
    }

    /// The translation of a pose in this frame, back in the world frame.
    fn translation_out(&self, rotation: Mat3AF32, translation: Vec3AF32) -> Vec3AF32 {
        translation / self.scale - rotation * self.centre
    }
}

/// Refine a PnP pose estimate using Levenberg-Marquardt optimization.
///
/// This function takes an initial pose estimate (typically from EPnP or similar)
/// and refines it by minimizing the reprojection error across all correspondences.
///
/// The solver works in a frame centred on the points and scaled by their median distance from the
/// initial camera, so large world coordinates (points far from the origin, a far camera, or
/// millimetre units) do not break its `f32` steps. The returned pose is in the original world
/// frame. The LM settings that depend on the pose units (`initial_lambda`, `gradient_tolerance`)
/// act in that frame.
///
/// # Arguments
///
/// * `points_world` - 3D points in world coordinates
/// * `points_image` - Corresponding 2D points in image coordinates
/// * `k` - Camera intrinsic matrix
/// * `initial_rotation` - Initial rotation estimate
/// * `initial_translation` - Initial translation estimate
/// * `distortion` - Optional camera distortion model
/// * `params` - LM refinement parameters
///
/// # Returns
///
/// Refined `PnPResult` with updated rotation, translation, and convergence info.
///
/// # Errors
///
/// Returns [`PnPError::MismatchedArrayLengths`] if the slices differ in length,
/// [`PnPError::InsufficientCorrespondences`] for fewer than 3 points, and [`PnPError::SvdFailed`]
/// if a world point or the initial pose is not finite, more than half of the points sit on the
/// initial camera, the robust loss is invalid or not supported, or the optimization fails.
pub fn refine_pose_lm(
    points_world: &[Vec3AF32],
    points_image: &[Vec2F32],
    k: &Mat3AF32,
    initial_rotation: &Mat3AF32,
    initial_translation: &Vec3AF32,
    distortion: Option<&PolynomialDistortion>,
    params: &LMRefineParams,
) -> Result<PnPResult, PnPError> {
    let n = points_world.len();
    if n != points_image.len() {
        return Err(PnPError::MismatchedArrayLengths {
            left_name: "world points",
            left_len: n,
            right_name: "image points",
            right_len: points_image.len(),
        });
    }

    if n < 3 {
        return Err(PnPError::InsufficientCorrespondences {
            required: 3,
            actual: n,
        });
    }

    // Convert initial pose to SE3 and then to parameter array, in the work frame. The quaternion
    // is normalised so the frame uses the same rotation as the factors (they normalise it too).
    let initial_so3 = SO3F32::from_quaternion(SO3F32::from_matrix(initial_rotation).q.normalize());
    let start_rotation = initial_so3.matrix();
    let frame =
        WorkFrame::new(points_world, start_rotation, *initial_translation).ok_or_else(|| {
            PnPError::SvdFailed(
                "non-finite world points or initial pose, or most points sit on the camera"
                    .to_string(),
            )
        })?;
    let points_work: Vec<Vec3AF32> = points_world.iter().map(|&p| frame.point(p)).collect();
    let initial_se3 = SE3F32::new(
        initial_so3,
        frame.translation_in(start_rotation, *initial_translation),
    );
    let initial_params = initial_se3.to_params().to_vec();

    // Create optimization problem
    let mut problem = Problem::new();

    // Add SE3 pose variable (7 DOF: quaternion + translation)
    let pose_var = Variable {
        name: "pose".to_string(),
        var_type: VariableType::SE3,
        values: vec![0.0; 7],
    };
    problem
        .add_variable(pose_var, initial_params)
        .map_err(|e| PnPError::SvdFailed(format!("Failed to add variable: {}", e)))?;

    // Add reprojection factors for each correspondence
    let loss = build_robust_loss(params)?;
    for (pw, pi) in points_work.iter().zip(points_image.iter()) {
        let factor =
            Box::new(ReprojectionFactor::new(*pw, *pi, k, distortion).with_loss(loss.clone()));
        problem
            .add_factor(factor, vec!["pose".to_string()])
            .map_err(|e| PnPError::SvdFailed(format!("Failed to add factor: {}", e)))?;
    }

    // Configure and run LM optimizer
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
        .map_err(|e| PnPError::SvdFailed(format!("Optimization failed: {}", e)))?;

    // Extract refined pose
    let pose_values = problem
        .get_variables()
        .get("pose")
        .ok_or_else(|| PnPError::SvdFailed("Pose variable not found in result".to_string()))?
        .values
        .clone();

    // Normalize quaternion before creating SE3 (optimizer may have made it non-normalized)
    // pose_values is [qw, qx, qy, qz, tx, ty, tz] from to_params()
    let q = QuatF32::from_xyzw(
        pose_values[1],
        pose_values[2],
        pose_values[3],
        pose_values[0],
    );
    let q_normalized = q.normalize();
    let t = Vec3AF32::new(pose_values[4], pose_values[5], pose_values[6]);

    let refined_se3 = SE3F32::from_qxyz(q_normalized, t);
    let refined_rotation = refined_se3.r.matrix();
    let refined_rvec = refined_se3.r.log();

    // Final RMSE in the work frame: the same pixels as in the world frame, without the rounding of
    // large world coordinates.
    let rmse = compute_rmse(
        &points_work,
        points_image,
        &refined_rotation,
        &refined_se3.t,
        k,
        distortion,
    )?;
    let refined_translation = frame.translation_out(refined_rotation, refined_se3.t);

    let converged = matches!(
        result.termination_reason,
        kornia_algebra::optim::TerminationReason::CostConverged
            | kornia_algebra::optim::TerminationReason::GradientConverged
    );

    Ok(PnPResult {
        rotation: refined_rotation,
        translation: refined_translation,
        rvec: refined_rvec,
        reproj_rmse: Some(rmse),
        num_iterations: Some(result.iterations),
        converged: Some(converged),
    })
}

/// Compute root-mean-square reprojection error.
fn compute_rmse(
    points_world: &[Vec3AF32],
    points_image: &[Vec2F32],
    rotation: &Mat3AF32,
    translation: &Vec3AF32,
    k: &Mat3AF32,
    distortion: Option<&PolynomialDistortion>,
) -> Result<f32, PnPError> {
    let fx = k.x_axis().x;
    let fy = k.y_axis().y;
    let cx = k.z_axis().x;
    let cy = k.z_axis().y;

    let mut sum_sq = 0.0f32;
    let mut valid_count = 0;

    // Prepare camera intrinsic for distortion if needed
    let cam_intr = if distortion.is_some() {
        Some(CameraIntrinsic {
            fx: fx as f64,
            fy: fy as f64,
            cx: cx as f64,
            cy: cy as f64,
        })
    } else {
        None
    };

    for (pw, pi) in points_world.iter().zip(points_image.iter()) {
        let pc = *rotation * *pw + *translation;

        if pc.z <= 0.0 {
            continue;
        }

        let inv_z = 1.0 / pc.z;
        let u_undist = fx * pc.x * inv_z + cx;
        let v_undist = fy * pc.y * inv_z + cy;

        // Apply distortion if provided
        let (u_hat, v_hat) = if let (Some(d), Some(intr)) = (distortion, cam_intr.as_ref()) {
            let (ud, vd) = distort_point_polynomial(u_undist as f64, v_undist as f64, intr, d);
            (ud as f32, vd as f32)
        } else {
            (u_undist, v_undist)
        };

        let du = u_hat - pi.x;
        let dv = v_hat - pi.y;
        sum_sq += du * du + dv * dv;
        valid_count += 1;
    }

    if valid_count == 0 {
        return Err(PnPError::SvdFailed(
            "No valid points for RMSE computation".to_string(),
        ));
    }

    Ok((sum_sq / valid_count as f32).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_algebra::optim::Factor;
    use kornia_algebra::{Mat3F64, Vec3F64, SO3F64};

    fn k_default() -> Mat3AF32 {
        Mat3AF32::from_cols(
            Vec3AF32::new(800.0, 0.0, 0.0),
            Vec3AF32::new(0.0, 800.0, 0.0),
            Vec3AF32::new(640.0, 480.0, 1.0),
        )
    }

    #[test]
    fn test_reprojection_factor_linearize() {
        let k = k_default();
        let point_world = Vec3AF32::new(0.0, 0.0, 1.0);
        let point_image = Vec2F32::new(640.0, 480.0);

        let factor = ReprojectionFactor::new(point_world, point_image, &k, None);
        let pose_params = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let params = vec![pose_params.as_slice()];

        let result = factor.linearize(&params, false).unwrap();
        assert_eq!(result.residual.len(), 2);
        assert!(result.residual[0].abs() < 1e-5);
        assert!(result.residual[1].abs() < 1e-5);
    }

    #[test]
    fn test_reprojection_factor_jacobian() {
        let k = k_default();
        let point_world = Vec3AF32::new(0.1, 0.1, 1.0);
        let point_image = Vec2F32::new(720.0, 560.0);

        let factor = ReprojectionFactor::new(point_world, point_image, &k, None);
        let pose_params = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let params = vec![pose_params.as_slice()];

        let result = factor.linearize(&params, true).unwrap();
        let jacobian = result.jacobian.unwrap();

        assert_eq!(jacobian.len(), 12); // 2 residuals x 6 tangent dimensions
        assert!(jacobian.iter().all(|x| x.is_finite()));
        let jacobian_norm: f32 = jacobian.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(jacobian_norm > 1e-6);
    }

    #[test]
    fn test_refine_pose_lm_convergence() -> Result<(), PnPError> {
        let points_world: [Vec3AF32; 6] = [
            Vec3AF32::new(0.0315, 0.03333, -0.10409),
            Vec3AF32::new(-0.0315, 0.03333, -0.10409),
            Vec3AF32::new(0.0, -0.00102, -0.12977),
            Vec3AF32::new(0.02646, -0.03167, -0.1053),
            Vec3AF32::new(-0.02646, -0.031667, -0.1053),
            Vec3AF32::new(0.0, 0.04515, -0.11033),
        ];

        let points_image: [Vec2F32; 6] = [
            Vec2F32::new(722.96466, 502.0828),
            Vec2F32::new(669.88837, 498.61877),
            Vec2F32::new(707.0025, 478.48975),
            Vec2F32::new(728.05634, 447.56918),
            Vec2F32::new(682.6069, 443.91776),
            Vec2F32::new(696.4414, 511.96442),
        ];

        let k = k_default();
        let initial_rotation = Mat3AF32::IDENTITY;
        let initial_translation = Vec3AF32::new(0.0, 0.0, 1.0);

        let params = LMRefineParams::default();
        let result = refine_pose_lm(
            &points_world,
            &points_image,
            &k,
            &initial_rotation,
            &initial_translation,
            None,
            &params,
        )?;

        assert!(result.converged.is_some());
        assert!(result.reproj_rmse.is_some());
        assert!(result.num_iterations.is_some());
        assert!(result.reproj_rmse.unwrap().is_finite());
        assert!(result.num_iterations.unwrap() > 0);

        Ok(())
    }

    // TEMP verification test (Claude): checks that `robust: Huber` actually
    // suppresses a gross outlier's influence on the LM solve, vs. plain L2.
    // Not part of the requested scaffold — delete or keep at your discretion.
    #[test]
    fn temp_huber_suppresses_outlier_influence() -> Result<(), PnPError> {
        let points_world: [Vec3AF32; 6] = [
            Vec3AF32::new(0.0315, 0.03333, -0.10409),
            Vec3AF32::new(-0.0315, 0.03333, -0.10409),
            Vec3AF32::new(0.0, -0.00102, -0.12977),
            Vec3AF32::new(0.02646, -0.03167, -0.1053),
            Vec3AF32::new(-0.02646, -0.031667, -0.1053),
            Vec3AF32::new(0.0, 0.04515, -0.11033),
        ];
        let points_image: [Vec2F32; 6] = [
            Vec2F32::new(722.96466, 502.0828),
            Vec2F32::new(669.88837, 498.61877),
            Vec2F32::new(707.0025, 478.48975),
            Vec2F32::new(728.05634, 447.56918),
            Vec2F32::new(682.6069, 443.91776),
            Vec2F32::new(696.4414, 511.96442),
        ];
        let k = k_default();
        let initial_rotation = Mat3AF32::IDENTITY;
        let initial_translation = Vec3AF32::new(0.0, 0.0, 1.0);

        // Baseline: LM on the 6 clean correspondences only.
        let clean = refine_pose_lm(
            &points_world,
            &points_image,
            &k,
            &initial_rotation,
            &initial_translation,
            None,
            &LMRefineParams::default(),
        )?;

        // Same 6 points plus one gross outlier (300px pixel error on an
        // otherwise-valid world point).
        let mut points_world_out = points_world.to_vec();
        let mut points_image_out = points_image.to_vec();
        points_world_out.push(points_world[0]);
        points_image_out.push(Vec2F32::new(
            points_image[0].x + 300.0,
            points_image[0].y - 300.0,
        ));

        let no_robust = refine_pose_lm(
            &points_world_out,
            &points_image_out,
            &k,
            &initial_rotation,
            &initial_translation,
            None,
            &LMRefineParams::default(),
        )?;

        let huber_params = LMRefineParams {
            robust: RobustKernelKind::Huber,
            robust_scale_sq: 25.0, // delta = 5px
            ..LMRefineParams::default()
        };
        let with_huber = refine_pose_lm(
            &points_world_out,
            &points_image_out,
            &k,
            &initial_rotation,
            &initial_translation,
            None,
            &huber_params,
        )?;

        // Score both outlier-contaminated solves against the CLEAN
        // correspondence set only, using the clean-only solve as ground
        // truth. A solve dragged toward the outlier will fit the clean data
        // worse than one that suppressed it.
        let rmse_no_robust_vs_clean = compute_rmse(
            &points_world,
            &points_image,
            &no_robust.rotation,
            &no_robust.translation,
            &k,
            None,
        )?;
        let rmse_huber_vs_clean = compute_rmse(
            &points_world,
            &points_image,
            &with_huber.rotation,
            &with_huber.translation,
            &k,
            None,
        )?;
        let rmse_clean_baseline = compute_rmse(
            &points_world,
            &points_image,
            &clean.rotation,
            &clean.translation,
            &k,
            None,
        )?;

        println!(
            "clean_baseline_rmse={rmse_clean_baseline:.4} \
             no_robust_vs_clean_rmse={rmse_no_robust_vs_clean:.4} \
             huber_vs_clean_rmse={rmse_huber_vs_clean:.4}"
        );

        // Huber should fit the clean data much closer to the outlier-free
        // baseline than plain L2 does.
        assert!(
            rmse_huber_vs_clean < rmse_no_robust_vs_clean,
            "Huber ({rmse_huber_vs_clean}) did not fit the clean set better than L2 ({rmse_no_robust_vs_clean})"
        );
        assert!(
            rmse_huber_vs_clean < 5.0,
            "Huber result ({rmse_huber_vs_clean}) still far from outlier-free baseline ({rmse_clean_baseline})"
        );

        Ok(())
    }

    #[test]
    fn test_refine_pose_lm_insufficient_correspondences() {
        let k = k_default();
        let points_world = vec![Vec3AF32::new(0.1, 0.0, 1.0)];
        let points_image = vec![Vec2F32::new(720.0, 480.0)];

        let result = refine_pose_lm(
            &points_world,
            &points_image,
            &k,
            &Mat3AF32::IDENTITY,
            &Vec3AF32::new(0.0, 0.0, 1.0),
            None,
            &LMRefineParams::default(),
        );

        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(PnPError::InsufficientCorrespondences { .. })
        ));
    }

    fn f32_vec(v: Vec3F64) -> Vec3AF32 {
        Vec3AF32::new(v.x as f32, v.y as f32, v.z as f32)
    }

    fn f64_vec(v: Vec3AF32) -> Vec3F64 {
        Vec3F64::new(v.x.into(), v.y.into(), v.z.into())
    }

    fn f32_mat(m: Mat3F64) -> Mat3AF32 {
        Mat3AF32::from_cols_array(&m.to_cols_array().map(|v| v as f32))
    }

    fn f64_mat(m: Mat3AF32) -> Mat3F64 {
        Mat3F64::from_cols_array(&m.to_cols_array().map(f64::from))
    }

    /// Point `i` (0..27) of the 3×3×3 grid filling [-1, 1]³.
    fn grid_point(i: u8) -> Vec3F64 {
        Vec3F64::new(
            f64::from(i % 3) - 1.0,
            f64::from(i / 3 % 3) - 1.0,
            f64::from(i / 9) - 1.0,
        )
    }

    /// A refined pose and its errors against the true pose.
    struct MovedSceneResult {
        pose: PnPResult,
        /// Rotation error in degrees.
        rotation_deg: f64,
        /// Camera-centre error relative to the depth.
        centre: f64,
    }

    /// Refines a pose that starts 0.5° and 1% of the depth away from the truth. The scene is a
    /// 3×3×3 grid filling a cube of half-size `size`, seen from `depth` units, and the whole scene
    /// is moved `offset` units from the world origin.
    fn refine_moved_scene(
        offset: f64,
        depth: f64,
        size: f64,
    ) -> Result<MovedSceneResult, PnPError> {
        let rotation = SO3F64::exp(Vec3F64::new(0.2, -0.4, 0.7)).matrix();
        let centre = Vec3F64::new(1.0, 0.5, 0.25).normalize() * offset;
        let translation = Vec3F64::new(0.0, 0.0, depth) - rotation * centre;
        let (mut world, mut image) = (Vec::new(), Vec::new());
        for i in 0..27u8 {
            let point = f32_vec(centre + grid_point(i) * size);
            // Project the stored f32 point, so the data is exact for the true pose.
            let pc = rotation * f64_vec(point) + translation;
            world.push(point);
            image.push(Vec2F32::new(
                (800.0 * pc.x / pc.z + 640.0) as f32,
                (800.0 * pc.y / pc.z + 480.0) as f32,
            ));
        }

        let camera_centre = -(rotation.transpose() * translation);
        let start_rotation =
            SO3F64::exp(Vec3F64::new(0.0, 0.5f64.to_radians(), 0.0)).matrix() * rotation;
        let start_centre =
            camera_centre + Vec3F64::new(0.3, -0.5, 0.8).normalize() * (0.01 * depth);
        let start_translation = -(start_rotation * start_centre);

        let result = refine_pose_lm(
            &world,
            &image,
            &k_default(),
            &f32_mat(start_rotation),
            &f32_vec(start_translation),
            None,
            &LMRefineParams::default(),
        )?;

        let estimate = f64_mat(result.rotation);
        let rotation_error_deg = SO3F64::from_matrix(&(estimate.transpose() * rotation))
            .log()
            .length()
            .to_degrees();
        let estimated_centre = -(estimate.transpose() * f64_vec(result.translation));
        let centre_error = (estimated_centre - camera_centre).length() / depth;
        Ok(MovedSceneResult {
            pose: result,
            rotation_deg: rotation_error_deg,
            centre: centre_error,
        })
    }

    /// The same scene and start, moved away from the world origin. Large world coordinates used to
    /// swallow the f32 pose update: of the 0.5° start error, 0.33° was left at 100 m and almost
    /// all of it at 10 km.
    #[test]
    fn refinement_does_not_depend_on_the_world_origin() -> Result<(), PnPError> {
        for offset in [0.0, 100.0, 1_000.0, 10_000.0] {
            let MovedSceneResult {
                pose: result,
                rotation_deg,
                centre,
            } = refine_moved_scene(offset, 6.0, 1.0)?;
            assert!(
                rotation_deg < 1e-3 && centre < 1e-3,
                "{offset} m from the origin: {rotation_deg}°, centre off by {centre} of the depth"
            );
            assert_eq!(result.converged, Some(true), "{offset} m from the origin");
            // The data is exact, so the reported reprojection error must be about zero too.
            let rmse = result.reproj_rmse.unwrap_or(f32::NAN);
            assert!(rmse < 1e-2, "{offset} m from the origin: RMSE {rmse} px");
        }
        Ok(())
    }

    /// A distant scene has the same problem even at the world origin, because the translation is
    /// large, and so does a scene in millimetres; a very close one must keep working.
    #[test]
    fn refinement_does_not_depend_on_the_scene_distance() -> Result<(), PnPError> {
        for (offset, depth, size) in [
            (0.0, 100.0, 10.0),
            (0.0, 300.0, 30.0),
            (10_000.0, 300.0, 30.0),
            (0.0, 6_000.0, 1_000.0),
            (0.0, 0.3, 0.05),
        ] {
            let MovedSceneResult {
                pose: result,
                rotation_deg,
                centre,
            } = refine_moved_scene(offset, depth, size)?;
            assert!(
                rotation_deg < 1e-3 && centre < 1e-3,
                "scene {depth} m away, {offset} m from the origin: {rotation_deg}°, \
                 centre off by {centre} of the depth"
            );
            assert_eq!(result.converged, Some(true), "scene {depth} m away");
            let rmse = result.reproj_rmse.unwrap_or(f32::NAN);
            assert!(rmse < 1e-2, "scene {depth} m away: RMSE {rmse} px");
        }
        Ok(())
    }

    /// The work frame must map the pose back to f32 rounding: with no LM iteration the start
    /// translation comes back, also 10 km from the origin and for a start rotation that is not quite
    /// orthonormal (the frame must use the normalised rotation the solver uses).
    #[test]
    fn zero_iterations_return_the_start_translation() -> Result<(), PnPError> {
        let rotation = SO3F64::exp(Vec3F64::new(0.2, -0.4, 0.7)).matrix();
        let centre = Vec3F64::new(1.0, 0.5, 0.25).normalize() * 10_000.0;
        let translation = Vec3F64::new(0.0, 0.0, 6.0) - rotation * centre;
        let world: Vec<Vec3AF32> = (0..27u8).map(|i| f32_vec(centre + grid_point(i))).collect();
        // Without iterations only the start pose matters, not the pixels.
        let image = vec![Vec2F32::new(640.0, 480.0); world.len()];
        for stretch in [1.0, 1.01] {
            let start = Mat3F64::from_cols_array(&rotation.to_cols_array().map(|v| v * stretch));
            let result = refine_pose_lm(
                &world,
                &image,
                &k_default(),
                &f32_mat(start),
                &f32_vec(translation),
                None,
                &LMRefineParams::default().with_max_iterations(0),
            )?;
            // |t| is about 10 km, where f32 steps are 1 mm.
            let moved = (f64_vec(result.translation) - translation).length();
            assert!(
                moved < 0.05,
                "stretch {stretch}: the translation moved by {moved} m"
            );
        }
        Ok(())
    }

    /// Input the work frame cannot be built from is an error: a non-finite world point, a
    /// non-finite start pose, or every point on the start camera.
    #[test]
    fn non_finite_or_degenerate_input_is_an_error() {
        let world: Vec<Vec3AF32> = (0..6)
            .map(|i| Vec3AF32::new(0.1 * i as f32, 0.05 * (i % 2) as f32, 5.0))
            .collect();
        let mut nan_point = world.clone();
        nan_point[2].x = f32::NAN;
        let on_camera = vec![Vec3AF32::ZERO; world.len()];
        let image = vec![Vec2F32::new(640.0, 480.0); world.len()];
        for (name, points, translation) in [
            ("NaN world point", &nan_point, Vec3AF32::ZERO),
            (
                "NaN start translation",
                &world,
                Vec3AF32::new(f32::NAN, 0.0, 0.0),
            ),
            ("points on the camera", &on_camera, Vec3AF32::ZERO),
        ] {
            let result = refine_pose_lm(
                points,
                &image,
                &k_default(),
                &Mat3AF32::IDENTITY,
                &translation,
                None,
                &LMRefineParams::default(),
            );
            assert!(matches!(result, Err(PnPError::SvdFailed(_))), "{name}");
        }
    }
}
