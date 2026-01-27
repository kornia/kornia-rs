//! Levenberg-Marquardt pose refinement for PnP solutions.
//!
//! This module provides LM-based nonlinear refinement for camera pose estimates,
//! leveraging the optimization infrastructure from `kornia-algebra` using factor graph.

use kornia_algebra::optim::{
    Factor, FactorError, FactorResult, LevenbergMarquardt, LinearizationResult, Problem, Variable,
    VariableType,
};
use kornia_algebra::{Mat3AF32, QuatF32, Vec2F32, Vec3AF32, SE3F32, SO3F32};
use kornia_imgproc::calibration::{
    distortion::{distort_point_polynomial, PolynomialDistortion},
    CameraIntrinsic,
};

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
}

impl Default for LMRefineParams {
    fn default() -> Self {
        Self {
            max_iterations: 20,
            cost_tolerance: 1e-6,
            gradient_tolerance: 1e-6,
            initial_lambda: 1e-3,
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
        }
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
        let se3 = SE3F32::from_qxyz(q_normalized, t);

        let pc = se3 * self.point_world; // camera-frame point

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

            let se3 = SE3F32::from_params(pose_params);
            let se3_plus = se3.retract(&delta_plus);
            let se3_minus = se3.retract(&delta_minus);

            let params_plus = se3_plus.to_params();
            let params_minus = se3_minus.to_params();

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

/// Refine a PnP pose estimate using Levenberg-Marquardt optimization.
///
/// This function takes an initial pose estimate (typically from EPnP or similar)
/// and refines it by minimizing the reprojection error across all correspondences.
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

    // Convert initial pose to SE3 and then to parameter array
    let initial_so3 = SO3F32::from_matrix(initial_rotation);
    let initial_se3 = SE3F32::new(initial_so3, *initial_translation);
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
    for (pw, pi) in points_world.iter().zip(points_image.iter()) {
        let factor = Box::new(ReprojectionFactor::new(*pw, *pi, k, distortion));
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
    let refined_translation = refined_se3.t;
    let refined_rvec = refined_se3.r.log();

    // Compute final RMSE
    let rmse = compute_rmse(
        points_world,
        points_image,
        &refined_rotation,
        &refined_translation,
        k,
        distortion,
    )?;

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
}
