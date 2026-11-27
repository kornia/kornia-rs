//! Pose-only Levenberg-Marquardt with an analytic Jacobian.
//!
//! Same objective and control flow as [`refine_pose_lm`](crate::pnp::refine_pose_lm) on
//! kornia-algebra's LM (Huber or L2 reprojection cost, the same damping,
//! tolerances and step rejection), but the 2x6 Jacobian of each residual is
//! closed-form and accumulated straight into a 6x6 normal matrix in `f64`. The
//! generic solver's numerical `f32` Jacobian projects each correspondence 12
//! more times and allocates per factor in every iteration; a tracker's
//! per-frame hot path needs the closed form.
//!
//! Departures from [`refine_pose_lm`](crate::pnp::refine_pose_lm): no distortion model (pixels must
//! be undistorted), no work-frame recentring, only the Identity and Huber
//! kernels, no allocation on a successful solve, and an optional `monotone`
//! rule for the last step.

// Fixed-size matrix loops read as the math they implement.
#![allow(clippy::needless_range_loop)]

use super::refine::{mat_to_f64, to_f64};
use super::{LMRefineParams, PnPError, PnPResult};
use crate::ransac::{RobustKernel, RobustKernelKind};
use kornia_algebra::{
    Mat3AF32, Mat3F64, Mat6F64, SMatF64, Vec2F32, Vec3AF32, Vec3F64, Vec6F64, SE3F64, SO3F32,
};

const LAMBDA_FACTOR: f64 = 10.0;
const LAMBDA_MIN: f64 = 1e-10;
const LAMBDA_MAX: f64 = 1e10;
const STEP_TOLERANCE: f64 = 1e-6;

/// The robust kernel and squared scale `params` select, or `None` for a kernel
/// this solver does not implement.
fn kernel_from_params(params: &LMRefineParams) -> Option<(RobustKernelKind, f64)> {
    if params.robust == RobustKernelKind::Identity || !params.robust_scale_sq.is_finite() {
        return Some((RobustKernelKind::Identity, 0.0));
    }
    match params.robust {
        RobustKernelKind::Huber if params.robust_scale_sq > 0.0 => {
            Some((RobustKernelKind::Huber, f64::from(params.robust_scale_sq)))
        }
        _ => None,
    }
}

/// `m` in `f32`, each entry rounded once.
fn mat_to_f32(m: Mat3F64) -> Mat3AF32 {
    Mat3AF32::from_cols(
        m.x_axis.as_vec3a().into(),
        m.y_axis.as_vec3a().into(),
        m.z_axis.as_vec3a().into(),
    )
}

/// `v` in `f32`, each entry rounded once.
fn to_f32(v: Vec3F64) -> Vec3AF32 {
    Vec3AF32::new(v.x as f32, v.y as f32, v.z as f32)
}

#[derive(Clone, Copy)]
struct Pose {
    r: Mat3F64,
    t: Vec3F64,
}

impl Pose {
    fn transform(&self, p: Vec3F64) -> Vec3F64 {
        self.r * p + self.t
    }

    /// `self * exp(delta)` with `delta = (upsilon, omega)`: kornia's SE3
    /// `retract`, so the damped steps follow the generic solver's path.
    fn right_update(&self, delta: &Vec6F64) -> Self {
        let upsilon = Vec3F64::new(delta[0], delta[1], delta[2]);
        let omega = Vec3F64::new(delta[3], delta[4], delta[5]);
        // Compose as matrices, not `SE3F64 *`: a quaternion product would move
        // the pinned output bits.
        let increment = SE3F64::exp(upsilon, omega);
        Self {
            r: self.r * increment.r.matrix(),
            t: self.t + self.r * increment.t,
        }
    }
}

/// The residuals, on the caller's `f32` slices: each point and pixel is widened
/// to `f64` (exactly) where it is used, so a solve allocates nothing.
struct Problem<'a> {
    points: &'a [Vec3AF32],
    pixels: &'a [Vec2F32],
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    kernel: (RobustKernelKind, f64),
}

impl Problem<'_> {
    /// Robust cost at `pose`; `None` when a point is not in front of the
    /// camera, which the generic solver treats as an infeasible step.
    fn cost(&self, pose: &Pose) -> Option<f64> {
        let (kernel, c_sq) = self.kernel;
        let mut total = 0.0;
        for (p, uv) in self.points.iter().zip(self.pixels) {
            let pc = pose.transform(to_f64(*p));
            let uv = [f64::from(uv.x), f64::from(uv.y)];
            if pc.z <= 0.0 {
                return None;
            }
            let ru = self.fx * pc.x / pc.z + self.cx - uv[0];
            let rv = self.fy * pc.y / pc.z + self.cy - uv[1];
            total += kernel.rho(ru * ru + rv * rv, c_sq);
        }
        Some(total)
    }

    /// Whether every point is strictly in front of the camera at `pose`
    /// (`false` for a non-finite depth).
    fn in_front(&self, pose: &Pose) -> bool {
        self.points
            .iter()
            .all(|p| pose.transform(to_f64(*p)).z > 0.0)
    }

    /// `pose` rounded to `f32`, checked again for depth: near `|t| = 2^24`
    /// the rounding can move a point that was in front onto or behind the
    /// camera.
    fn export(&self, pose: &Pose) -> Result<(Mat3AF32, Vec3AF32), PnPError> {
        let (rotation, translation) = (mat_to_f32(pose.r), to_f32(pose.t));
        let rounded = Pose {
            r: mat_to_f64(rotation),
            t: to_f64(translation),
        };
        if !self.in_front(&rounded) {
            return Err(PnPError::CheiralityCheckFailed);
        }
        Ok((rotation, translation))
    }

    /// Weighted normal equations `(J^T W J, J^T W r)` at `pose`.
    fn normal_equations(&self, pose: &Pose) -> (Mat6F64, Vec6F64) {
        let (kernel, c_sq) = self.kernel;
        let mut h = Mat6F64::zeros();
        let mut g = Vec6F64::zeros();
        for (p, uv) in self.points.iter().zip(self.pixels) {
            let p = to_f64(*p);
            let pc = pose.transform(p);
            let uv = [f64::from(uv.x), f64::from(uv.y)];
            let iz = 1.0 / pc.z;
            let r = [
                self.fx * pc.x * iz + self.cx - uv[0],
                self.fy * pc.y * iz + self.cy - uv[1],
            ];
            let w = kernel.weight(r[0] * r[0] + r[1] * r[1], c_sq);
            let jac = pose_jacobian_2x6(&pose.r, p, pc, self.fx, self.fy);
            let (ju, jv): ([f64; 6], [f64; 6]) = (
                std::array::from_fn(|i| jac[(0, i)]),
                std::array::from_fn(|i| jac[(1, i)]),
            );
            for i in 0..6 {
                g[i] += w * (ju[i] * r[0] + jv[i] * r[1]);
                for j in i..6 {
                    h[(i, j)] += w * (ju[i] * ju[j] + jv[i] * jv[j]);
                }
            }
        }
        for i in 0..6 {
            for j in 0..i {
                h[(i, j)] = h[(j, i)];
            }
        }
        (h, g)
    }
}

/// Rows `d(u)/d(p_c)` and `d(v)/d(p_c)` of a pinhole projection at the camera-frame
/// point `point_cam` (`z > 0`), in pixels per metre.
fn pinhole_point_jacobian(fx: f64, fy: f64, point_cam: Vec3F64) -> [Vec3F64; 2] {
    let iz = 1.0 / point_cam.z;
    [
        Vec3F64::new(fx * iz, 0.0, -fx * point_cam.x * iz * iz),
        Vec3F64::new(0.0, fy * iz, -fy * point_cam.y * iz * iz),
    ]
}

/// Pixel Jacobian with respect to a right perturbation `T_cw · exp(δ)`, `δ = (υ, ω)` (the
/// order of `SE3F64::retract`), of the world-to-camera pose with rotation `rotation_cw`,
/// at world point `point_world` whose camera-frame position is `point_cam`.
fn pose_jacobian_2x6(
    rotation_cw: &Mat3F64,
    point_world: Vec3F64,
    point_cam: Vec3F64,
    fx: f64,
    fy: f64,
) -> SMatF64<2, 6> {
    // d(pixel)/d(p_c) [R | -R [p]x] for p_c = R exp(delta) p + t.
    let rows = pinhole_point_jacobian(fx, fy, point_cam).map(|row| {
        let a = rotation_cw.transpose() * row;
        let b = point_world.cross(a);
        [a.x, a.y, a.z, b.x, b.y, b.z]
    });
    SMatF64::<2, 6>::from_fn(|r, c| rows[r][c])
}

/// The damped Gauss-Newton step `-(H + lambda I)^-1 g`; `None` when the damped
/// system is not positive definite or the step is not finite.
fn damped_step(h: &Mat6F64, g: &Vec6F64, lambda: f64) -> Option<Vec6F64> {
    let step = (h + Mat6F64::identity() * lambda).cholesky()?.solve(&(-g));
    step.iter().all(|v| v.is_finite()).then_some(step)
}

/// Whether [`refine_pose_lm_analytic`] implements the robust kernel of `params`.
///
/// # Arguments
///
/// * `params` - LM refinement parameters.
///
/// # Returns
///
/// `true` for the Identity kernel, for any kernel with a non-finite
/// `robust_scale_sq` (plain L2, as in [`super::refine_pose_lm`]), and for Huber
/// with a finite positive scale; `false` otherwise.
///
/// # Errors
///
/// This function is infallible.
pub fn refine_pose_lm_analytic_supports(params: &LMRefineParams) -> bool {
    kernel_from_params(params).is_some()
}

/// Refine a PnP pose by Levenberg-Marquardt with an analytic Jacobian.
///
/// Minimises the (Huber or L2) pinhole reprojection error over the
/// world-to-camera pose, like [`super::refine_pose_lm`] with the same damping
/// schedule and tolerances, but with the closed-form 2x6 Jacobian and normal
/// equations in `f64`. A successful solve does not allocate; an error may
/// (its message). The start rotation is projected onto SO(3) through its unit
/// quaternion first, as the generic solver does.
///
/// # Arguments
///
/// * `points_world` - 3D points in world coordinates.
/// * `points_image` - Corresponding undistorted pixels.
/// * `k` - Camera intrinsic matrix; only `fx`, `fy`, `cx`, `cy` are read.
/// * `rotation` - Initial world-to-camera rotation.
/// * `translation` - Initial world-to-camera translation.
/// * `params` - LM refinement parameters; see [`refine_pose_lm_analytic_supports`].
/// * `monotone` - When the cost change falls below `cost_tolerance`, adopt the
///   last step only if it does not raise the cost. `false` is the generic
///   solver's rule, which adopts it either way.
///
/// # Returns
///
/// The refined pose with `rvec`, `num_iterations` (LM iterations run) and
/// `converged` (a tolerance, not the iteration cap or the damping limit, ended
/// the solve). `reproj_rmse` is `None`.
///
/// # Errors
///
/// Returns [`PnPError::MismatchedArrayLengths`] if the slices differ in length,
/// [`PnPError::InsufficientCorrespondences`] for fewer than 3 points,
/// [`PnPError::InvalidInput`] if a world point, an image point, the initial
/// pose or `fx`, `fy`, `cx`, `cy` is not finite,
/// [`PnPError::CheiralityCheckFailed`] if a point is not in front of the
/// initial camera or of the refined pose rounded to `f32`, and
/// [`PnPError::SvdFailed`] if the robust kernel is not supported or a damped
/// system is not positive definite (the variant [`super::refine_pose_lm`]
/// reports both with).
///
/// # Example
///
/// ```rust
/// use kornia_3d::pnp::{refine_pose_lm_analytic, LMRefineParams};
/// use kornia_algebra::{Mat3AF32, Vec2F32, Vec3AF32};
///
/// let k = Mat3AF32::from_cols(
///     Vec3AF32::new(450.0, 0.0, 0.0),
///     Vec3AF32::new(0.0, 450.0, 0.0),
///     Vec3AF32::new(320.0, 240.0, 1.0),
/// );
/// let world = [
///     Vec3AF32::new(-0.5, -0.4, 2.0),
///     Vec3AF32::new(0.6, -0.3, 2.5),
///     Vec3AF32::new(0.1, 0.5, 3.0),
///     Vec3AF32::new(-0.3, 0.2, 1.5),
/// ];
/// // Seen from a camera at the origin.
/// let image: Vec<Vec2F32> = world
///     .iter()
///     .map(|p| Vec2F32::new(450.0 * p.x / p.z + 320.0, 450.0 * p.y / p.z + 240.0))
///     .collect();
/// let start = Vec3AF32::new(0.05, -0.02, 0.1);
/// let result = refine_pose_lm_analytic(
///     &world, &image, &k, &Mat3AF32::IDENTITY, &start, &LMRefineParams::default(), false,
/// )?;
/// assert!(result.translation.length() < 1e-4);
/// # Ok::<(), kornia_3d::pnp::PnPError>(())
/// ```
pub fn refine_pose_lm_analytic(
    points_world: &[Vec3AF32],
    points_image: &[Vec2F32],
    k: &Mat3AF32,
    rotation: &Mat3AF32,
    translation: &Vec3AF32,
    params: &LMRefineParams,
    monotone: bool,
) -> Result<PnPResult, PnPError> {
    if points_world.len() != points_image.len() {
        return Err(PnPError::MismatchedArrayLengths {
            left_name: "world points",
            left_len: points_world.len(),
            right_name: "image points",
            right_len: points_image.len(),
        });
    }
    if points_world.len() < 3 {
        return Err(PnPError::InsufficientCorrespondences {
            required: 3,
            actual: points_world.len(),
        });
    }
    // As the generic solver: a non-finite input is the caller's data, not a
    // numerical failure of the solve. NaN would otherwise pass `cost` (where
    // `NaN <= 0.0` is false) and surface as a failed damped step.
    let finite3 = |p: &Vec3AF32| p.x.is_finite() && p.y.is_finite() && p.z.is_finite();
    if let Some(i) = points_world.iter().position(|p| !finite3(p)) {
        return Err(PnPError::InvalidInput(format!(
            "world point {i} is not finite"
        )));
    }
    if let Some(i) = points_image
        .iter()
        .position(|p| !(p.x.is_finite() && p.y.is_finite()))
    {
        return Err(PnPError::InvalidInput(format!(
            "image point {i} is not finite"
        )));
    }
    if !(rotation.is_finite() && finite3(translation)) {
        return Err(PnPError::InvalidInput(
            "the initial pose is not finite".to_string(),
        ));
    }
    let (fx, fy, cx, cy) = (k.x_axis().x, k.y_axis().y, k.z_axis().x, k.z_axis().y);
    // Otherwise a NaN cost passes a zero-iteration solve as `Ok`.
    if !(fx.is_finite() && fy.is_finite() && cx.is_finite() && cy.is_finite()) {
        return Err(PnPError::InvalidInput(
            "the intrinsics are not finite".to_string(),
        ));
    }
    let kernel = kernel_from_params(params).ok_or_else(|| {
        PnPError::SvdFailed("robust kernel not supported by the analytic refine".to_string())
    })?;
    let problem = Problem {
        points: points_world,
        pixels: points_image,
        fx: fx as f64,
        fy: fy as f64,
        cx: cx as f64,
        cy: cy as f64,
        kernel,
    };
    // The generic solver starts from the start rotation's unit quaternion,
    // which projects a not-quite-orthonormal prediction back onto SO(3).
    // The quaternion of a scaled matrix is not unit: normalize it, or the
    // scale survives into the result and compounds through the motion model.
    let rotation = SO3F32::from_quaternion(SO3F32::from_matrix(rotation).q.normalize()).matrix();
    let mut pose = Pose {
        r: mat_to_f64(rotation),
        t: to_f64(*translation),
    };
    let mut cost = problem.cost(&pose).ok_or(PnPError::CheiralityCheckFailed)?;
    let mut lambda = params.initial_lambda as f64;
    let mut converged = false;
    let mut iterations = 0;
    for _ in 0..params.max_iterations {
        iterations += 1;
        let (h, g) = problem.normal_equations(&pose);
        if g.norm() < params.gradient_tolerance as f64 {
            converged = true;
            break;
        }
        let delta = damped_step(&h, &g, lambda).ok_or_else(|| {
            PnPError::SvdFailed("damped normal equations are not positive definite".to_string())
        })?;
        if delta.norm() < STEP_TOLERANCE {
            converged = true;
            break;
        }
        let candidate = pose.right_update(&delta);
        let new_cost = problem.cost(&candidate).unwrap_or(f64::INFINITY);
        let relative_change = if cost > 0.0 {
            (cost - new_cost).abs() / cost
        } else {
            (cost - new_cost).abs()
        };
        if relative_change < params.cost_tolerance as f64 {
            // The generic solver keeps this last step even uphill; `monotone`
            // keeps it only downhill, so a converged pose never costs more
            // than its start.
            if new_cost.is_finite() && (!monotone || new_cost <= cost) {
                pose = candidate;
            }
            converged = true;
            break;
        }
        if new_cost < cost {
            pose = candidate;
            cost = new_cost;
            lambda = (lambda / LAMBDA_FACTOR).max(LAMBDA_MIN);
        } else {
            lambda *= LAMBDA_FACTOR;
            if lambda > LAMBDA_MAX {
                break;
            }
        }
    }
    let (rotation, translation) = problem.export(&pose)?;
    Ok(PnPResult {
        rotation,
        translation,
        rvec: SO3F32::from_matrix(&rotation).log(),
        reproj_rmse: None,
        num_iterations: Some(iterations),
        converged: Some(converged),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pnp::refine_pose_lm;
    use kornia_algebra::SO3F64;

    /// Independent reference implementations used as oracles.
    mod oracle {
        fn skew(w: [f64; 3]) -> [[f64; 3]; 3] {
            [[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]]
        }

        fn identity_plus(a: f64, b: f64, w: [f64; 3]) -> [[f64; 3]; 3] {
            let k = skew(w);
            std::array::from_fn(|i| {
                std::array::from_fn(|j| {
                    let kk: f64 = (0..3).map(|l| k[i][l] * k[l][j]).sum();
                    f64::from(u8::from(i == j)) + a * k[i][j] + b * kk
                })
            })
        }

        pub fn so3_exp(w: [f64; 3]) -> [[f64; 3]; 3] {
            let theta2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
            let (a, b) = if theta2 < 1e-12 {
                (1.0 - theta2 / 6.0, 0.5 - theta2 / 24.0)
            } else {
                let theta = theta2.sqrt();
                (theta.sin() / theta, (1.0 - theta.cos()) / theta2)
            };
            identity_plus(a, b, w)
        }

        /// `V(w)` of the SE3 exponential, `t = V(w) upsilon`.
        pub fn se3_v(w: [f64; 3]) -> [[f64; 3]; 3] {
            let theta2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
            let (b, c) = if theta2 < 1e-12 {
                (0.5 - theta2 / 24.0, 1.0 / 6.0 - theta2 / 120.0)
            } else {
                let theta = theta2.sqrt();
                (
                    (1.0 - theta.cos()) / theta2,
                    (theta - theta.sin()) / (theta2 * theta),
                )
            };
            identity_plus(b, c, w)
        }

        /// Solves `a x = b` for a symmetric positive-definite 6x6 `a` (Cholesky).
        pub fn solve6(a: &[[f64; 6]; 6], b: &[f64; 6]) -> Option<[f64; 6]> {
            let mut l = [[0.0; 6]; 6];
            for i in 0..6 {
                for j in 0..=i {
                    let s: f64 = (0..j).map(|k| l[i][k] * l[j][k]).sum();
                    if i == j {
                        let d = a[i][i] - s;
                        if d <= 0.0 || !d.is_finite() {
                            return None;
                        }
                        l[i][i] = d.sqrt();
                    } else {
                        l[i][j] = (a[i][j] - s) / l[j][j];
                    }
                }
            }
            let mut y = [0.0; 6];
            for i in 0..6 {
                y[i] = (b[i] - (0..i).map(|k| l[i][k] * y[k]).sum::<f64>()) / l[i][i];
            }
            let mut x = [0.0; 6];
            for i in (0..6).rev() {
                x[i] = (y[i] - (i + 1..6).map(|k| l[k][i] * x[k]).sum::<f64>()) / l[i][i];
            }
            Some(x)
        }
    }

    /// Row `i`, column `j` of `m`.
    fn at(m: &Mat3F64, i: usize, j: usize) -> f64 {
        m.to_cols_array()[3 * j + i]
    }

    fn max_abs_difference(m: &Mat3F64, oracle: &[[f64; 3]; 3]) -> f64 {
        (0..9)
            .map(|n| (at(m, n / 3, n % 3) - oracle[n / 3][n % 3]).abs())
            .fold(0.0, f64::max)
    }

    fn rotation(w: [f64; 3]) -> Mat3F64 {
        SO3F64::exp(Vec3F64::from_array(w)).matrix()
    }

    fn scene(outlier: bool) -> (Vec<Vec3AF32>, Vec<Vec2F32>, Mat3AF32, Pose) {
        let k = Mat3AF32::from_cols(
            Vec3AF32::new(450.0, 0.0, 0.0),
            Vec3AF32::new(0.0, 450.0, 0.0),
            Vec3AF32::new(376.0, 240.0, 1.0),
        );
        let truth = Pose {
            r: rotation([0.02, -0.03, 0.01]),
            t: Vec3F64::new(0.05, -0.02, 0.1),
        };
        let mut world = Vec::new();
        let mut image = Vec::new();
        for i in 0..60 {
            let p = Vec3F64::new(
                ((i * 37) % 17) as f64 * 0.1 - 0.8,
                ((i * 11) % 13) as f64 * 0.08 - 0.5,
                1.0 + ((i * 7) % 5) as f64 * 0.3,
            );
            let pc = truth.transform(p);
            let u = 450.0 * pc.x / pc.z + 376.0;
            let v = 450.0 * pc.y / pc.z + 240.0;
            // One gross outlier for the robust kernel to discount.
            let (du, dv) = if outlier && i == 5 {
                (40.0, -30.0)
            } else {
                (0.0, 0.0)
            };
            world.push(Vec3AF32::new(p.x as f32, p.y as f32, p.z as f32));
            image.push(Vec2F32::new((u + du) as f32, (v + dv) as f32));
        }
        (world, image, k, truth)
    }

    fn params() -> LMRefineParams {
        LMRefineParams {
            robust: RobustKernelKind::Huber,
            robust_scale_sq: 25.0,
            ..LMRefineParams::default()
        }
    }

    #[test]
    fn the_jacobian_matches_central_differences() {
        let pose = Pose {
            r: rotation([0.3, -0.2, 0.1]),
            t: Vec3F64::new(0.2, -0.1, 0.5),
        };
        let point = [Vec3AF32::new(0.4, -0.3, 2.0)];
        let pixel = [Vec2F32::new(100.0, 80.0)];
        let problem = Problem {
            points: &point,
            pixels: &pixel,
            fx: 450.0,
            fy: 440.0,
            cx: 376.0,
            cy: 240.0,
            kernel: (RobustKernelKind::Identity, 0.0),
        };
        let residual = |pose: &Pose| {
            let pc = pose.transform(to_f64(point[0]));
            [
                450.0 * pc.x / pc.z + 376.0 - f64::from(pixel[0].x),
                440.0 * pc.y / pc.z + 240.0 - f64::from(pixel[0].y),
            ]
        };
        let r0 = residual(&pose);
        let (h, g) = problem.normal_equations(&pose);
        // With one residual, g = J^T r and H = J^T J; recover J by differences.
        for k in 0..6 {
            let mut d = Vec6F64::zeros();
            d[k] = 1e-6;
            let plus = residual(&pose.right_update(&d));
            d[k] = -1e-6;
            let minus = residual(&pose.right_update(&d));
            let jk = [(plus[0] - minus[0]) / 2e-6, (plus[1] - minus[1]) / 2e-6];
            let gk = jk[0] * r0[0] + jk[1] * r0[1];
            assert!(
                (gk - g[k]).abs() < 1e-3 * (1.0 + gk.abs()),
                "g[{k}] {} vs {gk}",
                g[k]
            );
            let hkk = jk[0] * jk[0] + jk[1] * jk[1];
            assert!(
                (hkk - h[(k, k)]).abs() < 1e-3 * (1.0 + hkk),
                "h[{k}] {} vs {hkk}",
                h[(k, k)]
            );
        }
    }

    /// J read directly, not through `JᵀJ`.
    #[test]
    fn pose_jacobian_2x6_matches_central_differences() {
        let pose = Pose {
            r: rotation([0.3, -0.2, 0.1]),
            t: Vec3F64::new(0.2, -0.1, 0.5),
        };
        let point = Vec3F64::new(0.4, -0.3, 2.0);
        let (fx, fy) = (450.0, 440.0);
        let pixel = |pose: &Pose| {
            let pc = pose.transform(point);
            [fx * pc.x / pc.z, fy * pc.y / pc.z]
        };
        let jacobian = pose_jacobian_2x6(&pose.r, point, pose.transform(point), fx, fy);
        for k in 0..6 {
            let mut d = Vec6F64::zeros();
            d[k] = 1e-6;
            let plus = pixel(&pose.right_update(&d));
            d[k] = -1e-6;
            let minus = pixel(&pose.right_update(&d));
            for row in 0..2 {
                let numeric = (plus[row] - minus[row]) / 2e-6;
                let analytic = jacobian[(row, k)];
                assert!(
                    (numeric - analytic).abs() < 1e-6 * (1.0 + analytic.abs()),
                    "J[{row}][{k}] {analytic} vs {numeric}"
                );
            }
        }
    }

    /// `Q diag(eigenvalues) Qᵀ` for a fixed orthogonal `Q`.
    fn symmetric_with(eigenvalues: [f64; 6]) -> Mat6F64 {
        let seed = Mat6F64::from_fn(|r, c| ((r * 5 + c * 11) as f64 * 0.29).cos());
        let q = seed.qr().q();
        q * Mat6F64::from_diagonal(&Vec6F64::from(eigenvalues)) * q.transpose()
    }

    fn rows(m: &Mat6F64) -> [[f64; 6]; 6] {
        std::array::from_fn(|r| std::array::from_fn(|c| m[(r, c)]))
    }

    #[test]
    fn the_cholesky_step_matches_the_local_solver() {
        let g = Vec6F64::new(0.3, -1.2, 0.7, 2.0, -0.4, 0.9);
        for kappa in [1.0f64, 1e2, 1e4] {
            let h = symmetric_with(std::array::from_fn(|i| kappa.powf(i as f64 / 5.0)));
            let new = damped_step(&h, &g, 0.0).unwrap();
            let old = Vec6F64::from(oracle::solve6(&rows(&h), &(-g).into()).unwrap());
            let relative = (new - old).norm() / old.norm();
            assert!(relative <= 1e-12, "kappa {kappa}: {relative:e}");
        }
    }

    #[test]
    fn a_system_that_is_not_positive_definite_is_refused_by_both_solvers() {
        let h = symmetric_with([1.0, 2.0, 3.0, 4.0, 5.0, -1.0]);
        let g = Vec6F64::new(0.3, -1.2, 0.7, 2.0, -0.4, 0.9);
        assert!(damped_step(&h, &g, 0.0).is_none());
        assert!(oracle::solve6(&rows(&h), &(-g).into()).is_none());
    }

    #[test]
    fn so3_exp_and_left_jacobian_match_the_local_bodies() {
        let axis = Vec3F64::new(0.3, -0.2, 0.9).normalize();
        for theta in [1e-7, 1e-5, 1e-3, 0.05, 0.5, 2.0, 3.0] {
            let w = axis * theta;
            let w_array = w.to_array();
            let exp = SO3F64::exp(w).matrix();
            let exp_error = max_abs_difference(&exp, &oracle::so3_exp(w_array));
            assert!(exp_error <= 1e-15, "exp at {theta}: {exp_error:e}");
            let jacobian = SO3F64::left_jacobian(w);
            let jacobian_error = max_abs_difference(&jacobian, &oracle::se3_v(w_array));
            let bound = 4e-16 / theta + 1e-15;
            assert!(
                jacobian_error <= bound,
                "left Jacobian at {theta}: {jacobian_error:e} > {bound:e}"
            );
        }
    }

    #[test]
    fn huber_matches_the_local_loss() {
        let (c_sq, delta) = (25.0f64, 5.0f64);
        for squared in [0.0f64, 1.0, 24.9, 25.0, 25.1, 400.0, 1e6] {
            let s = squared.sqrt();
            let old_rho = if s <= delta {
                squared
            } else {
                2.0 * delta * s - delta * delta
            };
            let old_weight = if squared <= delta * delta {
                1.0
            } else {
                delta / s
            };
            let rho = RobustKernelKind::Huber.rho(squared, c_sq);
            let weight = RobustKernelKind::Huber.weight(squared, c_sq);
            assert!(
                (rho - old_rho).abs() <= 1e-15 * old_rho.max(1.0),
                "{squared}"
            );
            assert!(
                (weight - old_weight).abs() <= 1e-15 * old_weight,
                "{squared}"
            );
        }
    }

    #[test]
    fn recovers_a_perturbed_pose() {
        let (world, image, k, truth) = scene(false);
        let start = Mat3AF32::IDENTITY;
        let refined = refine_pose_lm_analytic(
            &world,
            &image,
            &k,
            &start,
            &Vec3AF32::ZERO,
            &params(),
            false,
        )
        .unwrap();
        let (r, t) = (refined.rotation, refined.translation);
        assert_eq!(refined.converged, Some(true));
        for i in 0..3 {
            for j in 0..3 {
                let col = [r.x_axis(), r.y_axis(), r.z_axis()][j].to_array();
                assert!((col[i] as f64 - at(&truth.r, i, j)).abs() < 1e-4);
            }
        }
        assert!((t.x as f64 - truth.t.x).abs() < 1e-4);
        assert!((t.y as f64 - truth.t.y).abs() < 1e-4);
        assert!((t.z as f64 - truth.t.z).abs() < 1e-4);
    }

    #[test]
    fn agrees_with_the_generic_solver() {
        let (world, image, k, _) = scene(true);
        let start = Mat3AF32::IDENTITY;
        let refined = refine_pose_lm_analytic(
            &world,
            &image,
            &k,
            &start,
            &Vec3AF32::ZERO,
            &params(),
            false,
        )
        .unwrap();
        let (r, t) = (refined.rotation, refined.translation);
        let generic =
            refine_pose_lm(&world, &image, &k, &start, &Vec3AF32::ZERO, None, &params()).unwrap();
        assert!((t - generic.translation).length() < 1e-3);
        for (a, b) in [
            (r.x_axis(), generic.rotation.x_axis()),
            (r.y_axis(), generic.rotation.y_axis()),
            (r.z_axis(), generic.rotation.z_axis()),
        ] {
            assert!((a - b).length() < 1e-3);
        }
    }

    #[test]
    fn a_scaled_start_rotation_is_projected_onto_so3_like_the_generic_solver() {
        let (world, image, k, truth) = scene(false);
        // A prediction that drifted off SO(3): a rotation scaled by 0.9.
        let rotation = rotation([0.03, -0.01, 0.02]);
        let col = |j: usize| {
            Vec3AF32::new(
                0.9 * at(&rotation, 0, j) as f32,
                0.9 * at(&rotation, 1, j) as f32,
                0.9 * at(&rotation, 2, j) as f32,
            )
        };
        let scaled = Mat3AF32::from_cols(col(0), col(1), col(2));
        let refined = refine_pose_lm_analytic(
            &world,
            &image,
            &k,
            &scaled,
            &Vec3AF32::ZERO,
            &params(),
            false,
        )
        .unwrap();
        let t = refined.translation;
        assert!((t.x as f64 - truth.t.x).abs() < 1e-4);
        assert!((t.y as f64 - truth.t.y).abs() < 1e-4);
        assert!((t.z as f64 - truth.t.z).abs() < 1e-4);
        let r = refined.rotation;
        for column in [r.x_axis(), r.y_axis(), r.z_axis()] {
            assert!((column.length() - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn a_point_behind_the_start_pose_fails_like_the_generic_solver() {
        let (mut world, image, k, _) = scene(true);
        world[0] = Vec3AF32::new(0.0, 0.0, -1.0);
        let out = refine_pose_lm_analytic(
            &world,
            &image,
            &k,
            &Mat3AF32::IDENTITY,
            &Vec3AF32::ZERO,
            &params(),
            false,
        );
        assert!(matches!(out, Err(PnPError::CheiralityCheckFailed)));
    }

    #[test]
    fn unsupported_kernels_are_declined() {
        let (world, image, k, _) = scene(true);
        let cauchy = LMRefineParams {
            robust: RobustKernelKind::Cauchy,
            robust_scale_sq: 25.0,
            ..LMRefineParams::default()
        };
        let out = refine_pose_lm_analytic(
            &world,
            &image,
            &k,
            &Mat3AF32::IDENTITY,
            &Vec3AF32::ZERO,
            &cauchy,
            false,
        );
        assert!(!refine_pose_lm_analytic_supports(&cauchy));
        assert!(matches!(out, Err(PnPError::SvdFailed(_))));
    }

    #[test]
    fn mismatched_and_too_few_points_are_refused() {
        let (world, image, k, _) = scene(false);
        let refine = |world: &[Vec3AF32], image: &[Vec2F32]| {
            refine_pose_lm_analytic(
                world,
                image,
                &k,
                &Mat3AF32::IDENTITY,
                &Vec3AF32::ZERO,
                &params(),
                false,
            )
        };
        assert!(matches!(
            refine(&world, &image[1..]),
            Err(PnPError::MismatchedArrayLengths {
                left_len: 60,
                right_len: 59,
                ..
            })
        ));
        assert!(matches!(
            refine(&world[..2], &image[..2]),
            Err(PnPError::InsufficientCorrespondences {
                required: 3,
                actual: 2
            })
        ));
    }

    #[test]
    fn non_finite_input_is_invalid_like_the_generic_solver() {
        let (world, image, k, _) = scene(false);
        let refine = |world: &[Vec3AF32], image: &[Vec2F32], translation: Vec3AF32| {
            refine_pose_lm_analytic(
                world,
                image,
                &k,
                &Mat3AF32::IDENTITY,
                &translation,
                &params(),
                false,
            )
        };
        let mut bad_world = world.clone();
        bad_world[7].y = f32::NAN;
        let generic = refine_pose_lm(
            &bad_world,
            &image,
            &k,
            &Mat3AF32::IDENTITY,
            &Vec3AF32::ZERO,
            None,
            &params(),
        );
        let ours = refine(&bad_world, &image, Vec3AF32::ZERO);
        assert!(
            matches!(&ours, Err(PnPError::InvalidInput(r)) if r == "world point 7 is not finite"),
            "{ours:?}"
        );
        assert_eq!(format!("{ours:?}"), format!("{generic:?}"));
        let mut bad_image = image.clone();
        bad_image[3].x = f32::INFINITY;
        assert!(matches!(
            refine(&world, &bad_image, Vec3AF32::ZERO),
            Err(PnPError::InvalidInput(r)) if r == "image point 3 is not finite"
        ));
        assert!(matches!(
            refine(&world, &image, Vec3AF32::new(0.0, f32::NAN, 0.0)),
            Err(PnPError::InvalidInput(r)) if r == "the initial pose is not finite"
        ));
    }

    #[test]
    fn non_finite_intrinsics_are_invalid_even_without_iterations() {
        let (world, image, k, _) = scene(false);
        let k = Mat3AF32::from_cols(Vec3AF32::new(f32::NAN, 0.0, 0.0), k.y_axis(), k.z_axis());
        let no_iterations = LMRefineParams {
            max_iterations: 0,
            ..params()
        };
        let out = refine_pose_lm_analytic(
            &world,
            &image,
            &k,
            &Mat3AF32::IDENTITY,
            &Vec3AF32::ZERO,
            &no_iterations,
            false,
        );
        assert!(
            matches!(&out, Err(PnPError::InvalidInput(r)) if r == "the intrinsics are not finite"),
            "{out:?}"
        );
    }

    #[test]
    fn a_pose_that_rounds_onto_the_camera_is_refused() {
        // 2^24 - 1 is exact in f32; the 0.25 of the f64 depth is not.
        let points = [Vec3AF32::new(0.0, 0.0, -16_777_215.0); 3];
        let pixels = [Vec2F32::new(376.0, 240.0); 3];
        let problem = Problem {
            points: &points,
            pixels: &pixels,
            fx: 450.0,
            fy: 450.0,
            cx: 376.0,
            cy: 240.0,
            kernel: (RobustKernelKind::Identity, 0.0),
        };
        let pose = Pose {
            r: Mat3F64::IDENTITY,
            t: Vec3F64::new(0.0, 0.0, 16_777_215.25),
        };
        assert!(problem.in_front(&pose));
        assert!(matches!(
            problem.export(&pose),
            Err(PnPError::CheiralityCheckFailed)
        ));
    }

    #[test]
    fn the_iteration_cap_ends_the_solve_unconverged() {
        let (world, image, k, _) = scene(false);
        let capped = LMRefineParams {
            max_iterations: 1,
            ..LMRefineParams::default()
        };
        let result = refine_pose_lm_analytic(
            &world,
            &image,
            &k,
            &Mat3AF32::IDENTITY,
            &Vec3AF32::ZERO,
            &capped,
            false,
        )
        .unwrap();
        assert_eq!(result.num_iterations, Some(1));
        assert_eq!(result.converged, Some(false));
    }

    /// A start (Huber, the outlier) whose first damped step raises the cost by
    /// about 3 %, with `cost_tolerance` loose enough to stop on that step. The rule is
    /// the same at the default 1e-6; the loose tolerance makes the step visible.
    fn uphill_start() -> (Vec<Vec3AF32>, Vec<Vec2F32>, Mat3AF32, Pose, LMRefineParams) {
        let (world, image, k, _) = scene(true);
        let start = Pose {
            r: rotation([0.6, -0.1, 0.05]),
            t: Vec3F64::new(0.0, 0.0, 2.0),
        };
        let params = LMRefineParams {
            initial_lambda: 1e3,
            cost_tolerance: 0.05,
            ..params()
        };
        (world, image, k, start, params)
    }

    fn cost_at(world: &[Vec3AF32], image: &[Vec2F32], k: &Mat3AF32, pose: &Pose) -> f64 {
        let problem = Problem {
            points: world,
            pixels: image,
            fx: k.x_axis().x as f64,
            fy: k.y_axis().y as f64,
            cx: k.z_axis().x as f64,
            cy: k.z_axis().y as f64,
            kernel: kernel_from_params(&params()).unwrap(),
        };
        problem.cost(pose).unwrap()
    }

    /// The start cost and the cost of the refined pose, `monotone` as given.
    fn refined_cost(monotone: bool) -> (f64, f64) {
        let (world, image, k, start, params) = uphill_start();
        let refined = refine_pose_lm_analytic(
            &world,
            &image,
            &k,
            &mat_to_f32(start.r),
            &to_f32(start.t),
            &params,
            monotone,
        )
        .unwrap();
        assert_eq!(refined.converged, Some(true));
        let pose = Pose {
            r: mat_to_f64(refined.rotation),
            t: to_f64(refined.translation),
        };
        (
            cost_at(&world, &image, &k, &start),
            cost_at(&world, &image, &k, &pose),
        )
    }

    #[test]
    fn the_monotone_rule_refuses_an_uphill_step_inside_the_cost_tolerance() {
        let (start, refined) = refined_cost(true);
        assert!(refined <= start * (1.0 + 1e-6), "{start} -> {refined}");
    }

    #[test]
    fn without_the_monotone_rule_an_uphill_step_inside_the_cost_tolerance_is_kept() {
        let (start, refined) = refined_cost(false);
        let rise = refined / start - 1.0;
        assert!((0.01..0.05).contains(&rise), "{start} -> {refined}");
    }

    /// The `to_bits` of the refined rotation (column-major) and translation.
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    fn output_bits(outlier: bool, params: &LMRefineParams) -> ([u32; 9], [u32; 3]) {
        let (world, image, k, _) = scene(outlier);
        let refined = refine_pose_lm_analytic(
            &world,
            &image,
            &k,
            &Mat3AF32::IDENTITY,
            &Vec3AF32::ZERO,
            params,
            false,
        )
        .unwrap();
        (
            refined.rotation.to_cols_array().map(f32::to_bits),
            refined.translation.to_array().map(f32::to_bits),
        )
    }

    /// Pins the refined pose bit for bit, so a change to the arithmetic (such as
    /// where the points are converted to `f64`) cannot move it silently.
    /// Captured on x86_64 Linux (glibc libm). The f32 cast absorbs most f64
    /// last-ulp differences in `sin`/`cos`, but another libm (aarch64, a newer
    /// glibc) can still flip a bit, so it runs only on the capture platform;
    /// the tolerance tests cover the others.
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    #[test]
    fn output_bits_are_pinned() {
        assert_eq!(
            output_bits(false, &LMRefineParams::default()),
            (
                [
                    0x3f7fdf3c, 0x3c1ee317, 0x3cf68591, 0xbc28b761, 0x3f7fef9e, 0x3ca292b4,
                    0xbcf4e22f, 0xbca507c6, 0x3f7fd568,
                ],
                [0x3d4cccc9, 0xbca3d711, 0x3dcccccf],
            )
        );
        assert_eq!(
            output_bits(true, &params()),
            (
                [
                    0x3f7fdfce, 0x3c1a3dd9, 0x3cf4e252, 0xbc23d81d, 0x3f7ff03d, 0x3c9fe3e1,
                    0xbcf351e8, 0xbca242b1, 0x3f7fd639,
                ],
                [0x3d4cf2f1, 0xbca9563d, 0x3dc9cda7],
            )
        );
    }
}
