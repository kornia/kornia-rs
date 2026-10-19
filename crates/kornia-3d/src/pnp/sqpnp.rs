//! SQPnP: Perspective-n-Point by sequential quadratic programming on SO(3).
//!
//! Paper: [Terzakis & Lourakis, ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460460.pdf)
//! Reference: [OpenCV SQPnP implementation](https://github.com/opencv/opencv/blob/4.x/modules/calib3d/src/sqpnp.cpp)
//!
//! SQPnP minimises the object-space error `Σ ‖Q_i (R·M_i + t)‖²`, the distance of each world point
//! from the line of sight of its pixel, written as `rᵀ Ω r` with `r = vec(R)` (columns stacked).
//! SQP starts from the eigenvectors of `Ω` with the smallest eigenvalues and keeps only poses that
//! put the centroid of the points in front of the camera. It runs in `f64`.
//!
//! # Limits
//!
//! - With exactly 3 points it returns one of up to four exact solutions; use 4 or more.
//! - On noisy, nearly fronto-parallel planar scenes SQP can stop at `sqp_max_iterations`
//!   (`converged == Some(false)`) slightly short of the global minimum.
//! - Points close to the camera get little weight; set `refine_lm` when the pixel error matters.

use kornia_algebra::{Mat3AF32, Mat3F64, Vec2F32, Vec2F64, Vec3AF32, Vec3F64, SO3F32};
use kornia_imgproc::calibration::{
    distortion::{undistort_normalized_point_iter, PolynomialDistortion, TermCriteria},
    CameraIntrinsic,
};
use nalgebra::{Matrix3, SMatrix, SVector, SymmetricEigen, Vector3, Vector6};

use super::ops::rmse_px;
use super::refine::{refine_pose_lm, LMRefineParams};
use super::{PnPError, PnPResult, PnPSolver};
use crate::linalg::centroid;

/// Entries of `vec(R)`.
const VEC_R_LEN: usize = 9;
/// Orthonormality constraints on `R`.
const NUM_CONSTRAINTS: usize = 6;
/// Size of the KKT system `[Ω Hᵀ; H 0]`.
const KKT_LEN: usize = VEC_R_LEN + NUM_CONSTRAINTS;

type Mat9 = SMatrix<f64, VEC_R_LEN, VEC_R_LEN>;
type Vec9 = SVector<f64, VEC_R_LEN>;
type Mat39 = SMatrix<f64, 3, VEC_R_LEN>;
type Mat69 = SMatrix<f64, NUM_CONSTRAINTS, VEC_R_LEN>;
type MatKkt = SMatrix<f64, KKT_LEN, KKT_LEN>;
type VecKkt = SVector<f64, KKT_LEN>;

/// Minimum number of correspondences.
const MIN_CORRESPONDENCES: usize = 3;

/// `√3`: the norm of `vec(R)` for any rotation `R`.
const SQRT3: f64 = 1.732_050_807_568_877_2;

/// `‖vec(R)‖² = 3` for any rotation `R`.
const ROTATION_NORM_SQUARED: f64 = 3.0;

/// Iteration cap for the SVD and eigen decompositions (nalgebra's defaults never stop on NaN).
const MAX_DECOMPOSITION_ITERATIONS: usize = 1000;

/// Singular values below this are treated as zero in the least-squares KKT solve.
const KKT_SINGULAR_VALUE_TOLERANCE: f64 = 1e-12;

/// Relative tolerance below which two costs `rᵀ Ω r` count as equal.
const EQUAL_COST_RELATIVE_TOLERANCE: f64 = 1e-6;

/// `det(S) / (tr(S) / 3)³` below which all lines of sight count as parallel (`S = Σ Q_i`).
const PARALLEL_RAYS_TOLERANCE: f64 = 1e-10;

/// Undistortion stopping rule, far below the pixel noise.
const UNDISTORT_CRITERIA: TermCriteria = TermCriteria {
    max_iter: 20,
    eps: 1e-8,
};

/// Marker type representing the SQPnP algorithm.
pub struct SQPnP;

impl PnPSolver for SQPnP {
    type Param = SQPnPParams;

    fn solve(
        points_world: &[Vec3AF32],
        points_image: &[Vec2F32],
        k: &Mat3AF32,
        distortion: Option<&PolynomialDistortion>,
        params: &Self::Param,
    ) -> Result<PnPResult, PnPError> {
        solve_sqpnp(points_world, points_image, k, distortion, params)
    }
}

/// Parameters controlling the SQPnP solver. The defaults follow OpenCV, except `rank_tolerance`,
/// which is relative to the largest eigenvalue of `Ω` (OpenCV uses an absolute `1e-7`).
#[derive(Debug, Clone)]
pub struct SQPnPParams {
    /// An eigenvalue `λ` of `Ω` counts as zero when `λ ≤ rank_tolerance · λ_max`.
    pub rank_tolerance: f64,
    /// SQP stops when the squared norm of the step is below this value.
    pub sqp_squared_tolerance: f64,
    /// Maximum number of SQP iterations per starting rotation.
    pub sqp_max_iterations: usize,
    /// A start whose squared orthogonality error `‖X·Xᵀ − I‖²_F` is below this skips SQP.
    pub orthogonality_squared_error_threshold: f64,
    /// Optional LM refinement of the reprojection error after SQPnP.
    pub refine_lm: Option<LMRefineParams>,
}

impl Default for SQPnPParams {
    fn default() -> Self {
        Self {
            rank_tolerance: 1e-10,
            sqp_squared_tolerance: 1e-10,
            sqp_max_iterations: 15,
            orthogonality_squared_error_threshold: 1e-8,
            refine_lm: None,
        }
    }
}

/// Solve Perspective-n-Point with SQPnP.
///
/// # Arguments
///
/// * `points_world` - 3-D points in the world frame (`N ≥ 3`).
/// * `points_image` - Corresponding pixel coordinates.
/// * `k` - Camera intrinsics matrix `[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]` (no skew, as for the
///   other PnP solvers).
/// * `distortion` - Optional lens distortion; image points are undistorted before solving.
/// * `params` - Solver parameters.
///
/// # Returns
///
/// The world → camera pose and its reprojection RMSE in pixels. `num_iterations` and `converged`
/// are those of the SQP run that gave the returned rotation (0 and `true` when its start was
/// already a rotation), or of the LM run when `params.refine_lm` is set.
///
/// # Errors
///
/// * [`PnPError::MismatchedArrayLengths`] - `points_world` and `points_image` differ in length.
/// * [`PnPError::InsufficientCorrespondences`] - Fewer than 3 correspondences.
/// * [`PnPError::InvalidCameraMatrix`] - `K` is singular or not finite.
/// * [`PnPError::NonFinite`] - An input point or the reprojection error is not finite.
/// * [`PnPError::DegenerateConfiguration`] - All lines of sight are parallel or all world points
///   coincide.
/// * [`PnPError::SvdFailed`] - The eigen-decomposition of `Ω` does not converge.
/// * [`PnPError::CheiralityCheckFailed`] - No pose puts the points in front of the camera.
/// * With `params.refine_lm`, any error of [`refine_pose_lm`].
///
/// # Example
///
/// ```rust
/// use kornia_3d::pnp::{sqpnp::solve_sqpnp, SQPnPParams};
/// use kornia_algebra::{Mat3AF32, Vec2F32, Vec3AF32};
///
/// let k = Mat3AF32::from_cols_array(&[800.0, 0.0, 0.0, 0.0, 800.0, 0.0, 640.0, 480.0, 1.0]);
/// let world = [
///     Vec3AF32::new(-1.0, -0.8, -0.5),
///     Vec3AF32::new(0.9, -0.7, 0.3),
///     Vec3AF32::new(-0.6, 0.9, 0.8),
///     Vec3AF32::new(0.7, 0.6, -0.9),
/// ];
/// // The camera is 5 m in front of the points, with no rotation.
/// let image: Vec<Vec2F32> = world
///     .iter()
///     .map(|p| Vec2F32::new(800.0 * p.x / (p.z + 5.0) + 640.0, 800.0 * p.y / (p.z + 5.0) + 480.0))
///     .collect();
///
/// let pose = solve_sqpnp(&world, &image, &k, None, &SQPnPParams::default())?;
/// assert!((pose.translation.z - 5.0).abs() < 1e-3);
/// # Ok::<(), kornia_3d::pnp::PnPError>(())
/// ```
pub fn solve_sqpnp(
    points_world: &[Vec3AF32],
    points_image: &[Vec2F32],
    k: &Mat3AF32,
    distortion: Option<&PolynomialDistortion>,
    params: &SQPnPParams,
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
    if n < MIN_CORRESPONDENCES {
        return Err(PnPError::InsufficientCorrespondences {
            required: MIN_CORRESPONDENCES,
            actual: n,
        });
    }

    let sight = LineOfSight::new(*k, distortion.copied())?;
    // Centred world points: better conditioned Ω, and the centroid's depth is t'_z.
    let world_centroid = centroid(points_world);
    let objective = Objective::new(points_world, points_image, world_centroid, &sight)?;
    let best = minimize_on_so3(&objective, params)?;

    // t' is the translation for the centred points: R·(M − c) + t' = R·M + (t' − R·c).
    let rotation = Mat3F64::from_cols_array(&best.solution.r.into());
    let translation = objective.translation(&best.solution.r) - rotation * world_centroid;

    let rotation_f32 = rotation.as_mat3af32();
    let translation_f32 = translation.as_vec3af32();

    let pose = match params.refine_lm {
        Some(ref lm_params) => refine_pose_lm(
            points_world,
            points_image,
            k,
            &rotation_f32,
            &translation_f32,
            distortion,
            lm_params,
        )?,
        None => PnPResult {
            rotation: rotation_f32,
            translation: translation_f32,
            rvec: SO3F32::from_matrix(&rotation_f32).log(),
            reproj_rmse: Some(rmse_px(
                points_world,
                points_image,
                &rotation_f32,
                &translation_f32,
                k,
                distortion,
            )?),
            num_iterations: Some(best.solution.iterations),
            converged: Some(best.solution.converged),
        },
    };
    // Finite but extreme inputs (e.g. a pixel at 1e30) can overflow the reprojection.
    if pose.reproj_rmse.is_some_and(f32::is_finite) {
        Ok(pose)
    } else {
        Err(PnPError::NonFinite("reprojection error"))
    }
}

/// Maps pixels to lines of sight `(x, y, 1)` in normalised pinhole coordinates: `K⁻¹·(u, v, 1)`,
/// then undistorted when a distortion model is given.
struct LineOfSight {
    k_inv: Mat3F64,
    intrinsic: CameraIntrinsic,
    distortion: Option<PolynomialDistortion>,
}

impl LineOfSight {
    /// The map for the camera `k` and optional `distortion`.
    ///
    /// # Errors
    ///
    /// Returns [`PnPError::InvalidCameraMatrix`] if `K` is singular or not finite.
    fn new(k: Mat3AF32, distortion: Option<PolynomialDistortion>) -> Result<Self, PnPError> {
        let k = Mat3F64::from(k);
        let k_inv = k.inverse();
        if !k_inv.is_finite() {
            return Err(PnPError::InvalidCameraMatrix);
        }
        let intrinsic = CameraIntrinsic {
            fx: k.x_axis().x,
            fy: k.y_axis().y,
            cx: k.z_axis().x,
            cy: k.z_axis().y,
        };
        Ok(Self {
            k_inv,
            intrinsic,
            distortion,
        })
    }

    /// The direction `(x, y, 1)` of the line of sight through the pixel `uv`.
    ///
    /// # Errors
    ///
    /// Returns [`PnPError::NonFinite`] if the pixel maps to a non-finite coordinate.
    fn direction(&self, uv: Vec2F32) -> Result<Vec3F64, PnPError> {
        let uv = Vec2F64::from(uv);
        let h = self.k_inv * Vec3F64::new(uv.x, uv.y, 1.0);
        let (mut x, mut y) = (h.x / h.z, h.y / h.z);
        if let Some(d) = &self.distortion {
            let undistorted = undistort_normalized_point_iter(
                x,
                y,
                uv.x,
                uv.y,
                &self.intrinsic,
                d,
                UNDISTORT_CRITERIA,
            );
            (x, y) = (undistorted.x, undistorted.y);
        }
        if x.is_finite() && y.is_finite() {
            Ok(Vec3F64::new(x, y, 1.0))
        } else {
            Err(PnPError::NonFinite("image point"))
        }
    }
}

/// The SQPnP cost of a set of correspondences: `E(r) = rᵀ Ω r`, and the translation
/// `t' = P·r` that is optimal for the rotation `r` (and the centred world points).
struct Objective {
    omega: Mat9,
    p: Mat39,
}

impl Objective {
    /// Builds `Ω` and `P` from the correspondences, with the world points centred on `centroid`.
    ///
    /// With `A_i = M_iᵀ ⊗ I₃` (so that `R·M_i = A_i·r`), `S = Σ Q_i` and `B = Σ Q_i A_i`:
    /// `P = −S⁻¹ B` and `Ω = Σ A_iᵀ Q_i A_i − Bᵀ S⁻¹ B = Σ A_iᵀ Q_i A_i + Bᵀ P`.
    ///
    /// # Errors
    ///
    /// Returns [`PnPError::NonFinite`] if an image point maps to a non-finite coordinate,
    /// [`PnPError::DegenerateConfiguration`] if all lines of sight are parallel (every image point
    /// is the same), and [`PnPError::NonFinite`] if `Ω` is not finite.
    fn new(
        points_world: &[Vec3AF32],
        points_image: &[Vec2F32],
        centroid: Vec3F64,
        sight: &LineOfSight,
    ) -> Result<Self, PnPError> {
        // Block (j, l) of A_iᵀ Q_i A_i is M_i[j]·M_i[l]·Q_i and block j of Q_i A_i is M_i[j]·Q_i;
        // only the upper triangle (j ≤ l) of Ω is accumulated.
        let mut omega = Mat9::zeros();
        let mut s = Matrix3::<f64>::zeros();
        let mut b = Mat39::zeros();

        for (p, uv) in points_world.iter().zip(points_image) {
            let m = (Vec3F64::from(*p) - centroid).to_array();
            let v = Vector3::from(sight.direction(*uv)?);
            let q = Matrix3::identity() - (v * v.transpose()) / v.norm_squared();

            s += q;
            for j in 0..3 {
                let mut b_block = b.fixed_view_mut::<3, 3>(0, 3 * j);
                b_block += q * m[j];
                for l in j..3 {
                    let mut omega_block = omega.fixed_view_mut::<3, 3>(3 * j, 3 * l);
                    omega_block += q * (m[j] * m[l]);
                }
            }
        }

        // Parallel lines of sight make S singular, but rounding can leave det(S) just above zero.
        let parallel = PnPError::DegenerateConfiguration("all lines of sight are parallel");
        if s.determinant() <= PARALLEL_RAYS_TOLERANCE * (s.trace() / 3.0).powi(3) {
            return Err(parallel);
        }
        let s_inv = s.try_inverse().ok_or(parallel)?;
        let p = -(s_inv * b);
        // Ω += Bᵀ·P, accumulated in place.
        omega.gemm_tr(1.0, &b, &p, 1.0);
        // Mirror the upper triangle so Ω is exactly symmetric.
        omega.fill_lower_triangle_with_upper_triangle();

        if omega.iter().all(|v| v.is_finite()) {
            Ok(Self { omega, p })
        } else {
            Err(PnPError::NonFinite("world point"))
        }
    }

    /// The optimal translation `t' = P·r` for the rotation `r` (and the centred world points).
    fn translation(&self, r: &Vec9) -> Vec3F64 {
        Vec3F64::from(self.p * r)
    }

    /// `rᵀ Ω r`.
    fn cost(&self, r: &Vec9) -> f64 {
        r.dot(&(self.omega * r))
    }

    /// Camera-frame depth of the centroid of the world points under `r`. The points are centred,
    /// so it is `t'_z = (P·r)_z`.
    fn centroid_depth(&self, r: &Vec9) -> f64 {
        self.p.row(2).tr_dot(r)
    }
}

/// A rotation reached from one starting point.
struct SqpSolution {
    /// `vec(R)`, columns stacked.
    r: Vec9,
    /// SQP iterations run; 0 when the start was already a rotation.
    iterations: usize,
    /// Whether SQP's step got below `sqp_squared_tolerance`; `true` when no SQP was needed.
    converged: bool,
}

/// The best rotation found so far, with its cost `rᵀ Ω r`.
struct Candidate {
    solution: SqpSolution,
    cost: f64,
}

/// Minimises `rᵀ Ω r` over SO(3), keeping only rotations that put the centroid in front of the
/// camera.
///
/// # Errors
///
/// Returns [`PnPError::SvdFailed`] if the eigen-decomposition of `Ω` does not converge,
/// [`PnPError::DegenerateConfiguration`] if `Ω` is all zeros, and
/// [`PnPError::CheiralityCheckFailed`] if no candidate passes the depth check.
fn minimize_on_so3(objective: &Objective, params: &SQPnPParams) -> Result<Candidate, PnPError> {
    let eig = SymmetricEigen::try_new(objective.omega, f64::EPSILON, MAX_DECOMPOSITION_ITERATIONS)
        .ok_or_else(|| PnPError::SvdFailed("eigen-decomposition did not converge".to_string()))?;
    let mut order: [usize; VEC_R_LEN] = std::array::from_fn(|i| i);
    order.sort_by(|&a, &b| eig.eigenvalues[a].total_cmp(&eig.eigenvalues[b]));
    let eigenvalue = |k: usize| eig.eigenvalues[order[k]].max(0.0);

    let lambda_max = eigenvalue(VEC_R_LEN - 1);
    if lambda_max <= 0.0 {
        return Err(PnPError::DegenerateConfiguration(
            "all world points coincide",
        ));
    }
    let num_null = (0..VEC_R_LEN)
        .filter(|&k| eigenvalue(k) <= params.rank_tolerance * lambda_max)
        .count();

    let mut search = Search {
        best: None,
        lowest_rejected_cost: f64::INFINITY,
    };
    let try_eigenvector = |k: usize, search: &mut Search| {
        let x: Vec9 = eig.eigenvectors.column(order[k]) * SQRT3;
        if orthogonality_squared_error(&x) < params.orthogonality_squared_error_threshold {
            // Already close to a rotation up to sign; project it so the result is orthonormal.
            if let Some(r) = project_to_so3(&with_positive_det(x)) {
                search.consider(
                    objective,
                    SqpSolution {
                        r,
                        iterations: 0,
                        converged: true,
                    },
                );
            }
        } else {
            for start in [x, -x] {
                if let Some(solution) =
                    project_to_so3(&start).and_then(|r0| sqp(&objective.omega, r0, params))
                {
                    search.consider(objective, solution);
                }
            }
        }
    };

    // As in the paper, every null vector is a start.
    for k in 0..num_null {
        try_eigenvector(k, &mut search);
    }
    // Stop at the paper's bound `best ≤ 3·λ_k`, unless a depth-rejected rotation had a lower cost.
    let cost_floor = ROTATION_NORM_SQUARED * params.rank_tolerance * lambda_max;
    for k in num_null..VEC_R_LEN {
        let best_cost = search.best.as_ref().map_or(f64::INFINITY, |c| c.cost);
        let rejected_is_lower = best_cost
            > search.lowest_rejected_cost * (1.0 + EQUAL_COST_RELATIVE_TOLERANCE) + cost_floor;
        if best_cost <= ROTATION_NORM_SQUARED * eigenvalue(k) && !rejected_is_lower {
            break;
        }
        try_eigenvector(k, &mut search);
    }

    search.best.ok_or(PnPError::CheiralityCheckFailed)
}

/// State of the search over the eigenvectors of `Ω`.
struct Search {
    /// Lowest-cost rotation that puts the centroid in front of the camera.
    best: Option<Candidate>,
    /// Lowest `rᵀ Ω r` among rotations rejected by the depth check.
    lowest_rejected_cost: f64,
}

impl Search {
    /// Keeps `solution` as the best candidate if it puts the centroid in front of the camera and
    /// has the lowest `objective` cost so far.
    fn consider(&mut self, objective: &Objective, solution: SqpSolution) {
        let cost = objective.cost(&solution.r);
        if objective.centroid_depth(&solution.r) <= 0.0 {
            self.lowest_rejected_cost = self.lowest_rejected_cost.min(cost);
            return;
        }
        if self.best.as_ref().is_none_or(|b| cost < b.cost) {
            self.best = Some(Candidate { solution, cost });
        }
    }
}

/// Runs SQP from the rotation `r0`, then projects the result onto SO(3). Returns `None` if that
/// projection fails (e.g. on non-finite values).
///
/// Each iteration solves the KKT system of `min (r + δ)ᵀ Ω (r + δ)` subject to the linearised
/// orthonormality constraints `h(r) + H(r)·δ = 0`.
fn sqp(omega: &Mat9, r0: Vec9, params: &SQPnPParams) -> Option<SqpSolution> {
    let mut r = r0;
    let mut iterations = 0;
    let mut converged = false;
    // [Ω Hᵀ; H 0]: the Ω block is written once, the H blocks every iteration.
    let mut kkt = MatKkt::zeros();
    kkt.fixed_view_mut::<VEC_R_LEN, VEC_R_LEN>(0, 0)
        .copy_from(omega);

    while iterations < params.sqp_max_iterations {
        iterations += 1;
        let constraints = Constraints::new(&r);
        kkt.fixed_view_mut::<NUM_CONSTRAINTS, VEC_R_LEN>(VEC_R_LEN, 0)
            .copy_from(&constraints.jacobian);
        kkt.fixed_view_mut::<VEC_R_LEN, NUM_CONSTRAINTS>(0, VEC_R_LEN)
            .tr_copy_from(&constraints.jacobian);

        // rhs = (−Ω·r, −h), written in place.
        let mut rhs = VecKkt::zeros();
        rhs.fixed_rows_mut::<VEC_R_LEN>(0)
            .gemv(-1.0, omega, &r, 0.0);
        rhs.fixed_rows_mut::<NUM_CONSTRAINTS>(VEC_R_LEN)
            .axpy(-1.0, &constraints.h, 0.0);

        // LU, or a least-squares SVD solve when the KKT matrix is numerically singular.
        let solution = kkt
            .lu()
            .solve(&rhs)
            .filter(|s| s.iter().all(|v| v.is_finite()))
            .or_else(|| {
                kkt.try_svd(true, true, f64::EPSILON, MAX_DECOMPOSITION_ITERATIONS)?
                    .solve(&rhs, KKT_SINGULAR_VALUE_TOLERANCE)
                    .ok()
            });
        let Some(solution) = solution else {
            break;
        };
        let delta = solution.fixed_rows::<VEC_R_LEN>(0);
        r += delta;
        if delta.norm_squared() < params.sqp_squared_tolerance {
            converged = true;
            break;
        }
    }

    Some(SqpSolution {
        r: project_to_so3(&r)?,
        iterations,
        converged,
    })
}

/// The six orthonormality constraints `h(r) = 0` on the rows `r₁, r₂, r₃` of `R`, as in the paper
/// and OpenCV, `h = (‖r₁‖² − 1, ‖r₂‖² − 1, ‖r₃‖² − 1, r₁·r₂, r₂·r₃, r₁·r₃)`, and their Jacobian
/// `∂h/∂r`.
struct Constraints {
    h: Vector6<f64>,
    jacobian: Mat69,
}

impl Constraints {
    fn new(r: &Vec9) -> Self {
        let m = Mat3F64::from_cols_array(&(*r).into());
        let (r1, r2, r3) = (m.row(0), m.row(1), m.row(2));

        let h = Vector6::new(
            r1.length_squared() - 1.0,
            r2.length_squared() - 1.0,
            r3.length_squared() - 1.0,
            r1.dot(r2),
            r2.dot(r3),
            r1.dot(r3),
        );

        // Entry (i, j) of R is r[3j + i] (columns stacked).
        let mut jacobian = Mat69::zeros();
        for j in 0..3 {
            let (i1, i2, i3) = (3 * j, 3 * j + 1, 3 * j + 2);
            jacobian[(0, i1)] = 2.0 * r1[j];
            jacobian[(1, i2)] = 2.0 * r2[j];
            jacobian[(2, i3)] = 2.0 * r3[j];
            // d(r1·r2)
            jacobian[(3, i1)] = r2[j];
            jacobian[(3, i2)] = r1[j];
            // d(r2·r3)
            jacobian[(4, i2)] = r3[j];
            jacobian[(4, i3)] = r2[j];
            // d(r1·r3)
            jacobian[(5, i1)] = r3[j];
            jacobian[(5, i3)] = r1[j];
        }
        Self { h, jacobian }
    }
}

/// `x` or `−x`, whichever is `vec(X)` of a matrix with `det(X) ≥ 0`.
///
/// An eigenvector of `Ω` is only defined up to sign, and for an orthogonal `X` negating it flips
/// `det` between `−1` (a reflection) and `+1` (a rotation).
fn with_positive_det(x: Vec9) -> Vec9 {
    if Mat3F64::from_cols_array(&x.into()).determinant() < 0.0 {
        -x
    } else {
        x
    }
}

/// Squared Frobenius norm of `X·Xᵀ − I` for `vec(X) = x`.
fn orthogonality_squared_error(x: &Vec9) -> f64 {
    let m = Mat3F64::from_cols_array(&(*x).into());
    let e = m * m.transpose() - Mat3F64::IDENTITY;
    (0..3).map(|i| e.col(i).length_squared()).sum()
}

/// `vec(R)` of the rotation closest to `X` (`vec(X) = x`) in the Frobenius norm, or `None` if `x`
/// is not finite or the SVD does not converge.
fn project_to_so3(x: &Vec9) -> Option<Vec9> {
    let r = Mat3F64::from_cols_array(&(*x).into()).nearest_rotation()?;
    Some(Vec9::from(r.to_cols_array()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pnp::{solve_pnp, solve_pnp_ransac, PnPMethod, RansacParams};
    use crate::test_utils::{
        k_default, k_matrix, random_rotation, rotation_error_deg, translation_error, Scene,
        SceneConfig,
    };
    use kornia_algebra::SO3F64;
    use kornia_imgproc::calibration::distortion::distort_point_polynomial;
    use rand::{rngs::StdRng, RngExt, SeedableRng};

    /// Rotation error, in degrees, below which a noise-free pose counts as exact.
    const EXACT_DEG: f64 = 2e-3;

    /// Reprojection RMSE, in pixels, below which a noise-free pose counts as exact.
    const EXACT_RMSE_PX: f32 = 1e-2;

    /// `vec(m)`: the columns of `m` stacked.
    fn vec9(m: Mat3F64) -> Vec9 {
        Vec9::from(m.to_cols_array())
    }

    /// `n` points on a 2 m square seen from 5 m, at most 1.2 rad away from facing it.
    fn marker(n: usize, noise_px: f64) -> SceneConfig {
        SceneConfig {
            max_angle: 1.2,
            ..SceneConfig::cube(n, 5.0, noise_px).planar()
        }
    }

    /// Five valid world points, for tests that break one input at a time.
    fn valid_world() -> Vec<Vec3AF32> {
        vec![
            Vec3AF32::new(0.0, 0.0, 0.0),
            Vec3AF32::new(1.0, 0.0, 0.0),
            Vec3AF32::new(0.0, 1.0, 0.0),
            Vec3AF32::new(0.0, 0.0, 1.0),
            Vec3AF32::new(1.0, 1.0, 1.0),
        ]
    }

    /// Five finite pixels paired with [`valid_world`].
    fn valid_image() -> Vec<Vec2F32> {
        vec![
            Vec2F32::new(600.0, 400.0),
            Vec2F32::new(700.0, 420.0),
            Vec2F32::new(650.0, 500.0),
            Vec2F32::new(620.0, 450.0),
            Vec2F32::new(690.0, 470.0),
        ]
    }

    /// `Σ ‖Q_i (R·(M_i − c) + t)‖²` with `Q_i x = x − v (v·x) / (v·v)`, the part of `x` off the
    /// line of sight `v` of pixel `i`.
    fn object_space_error(
        scene: &Scene,
        sight: &LineOfSight,
        c: Vec3F64,
        r: Mat3F64,
        t: Vec3F64,
    ) -> Result<f64, PnPError> {
        let mut sum = 0.0;
        for (&p, uv) in scene.world.iter().zip(&scene.image) {
            let v = sight.direction(*uv)?;
            let x = r * (Vec3F64::from(p) - c) + t;
            let off_line = x - v * (v.dot(x) / v.dot(v));
            sum += off_line.dot(off_line);
        }
        Ok(sum)
    }

    /// Default parameters with LM refinement.
    fn with_lm() -> SQPnPParams {
        SQPnPParams {
            refine_lm: Some(LMRefineParams::default()),
            ..Default::default()
        }
    }

    fn solve_with(scene: &Scene, params: &SQPnPParams) -> Result<PnPResult, PnPError> {
        solve_sqpnp(&scene.world, &scene.image, &scene.k, None, params)
    }

    fn solve(scene: &Scene) -> Result<PnPResult, PnPError> {
        solve_with(scene, &SQPnPParams::default())
    }

    #[test]
    fn noise_free_poses_are_exact() -> Result<(), PnPError> {
        let k = k_default();
        let mut rng = StdRng::seed_from_u64(1);
        for n in [4, 5, 6, 10, 50, 200] {
            for _ in 0..10 {
                let depth = rng.random_range(3.0..10.0);
                let scene = Scene::generate(&mut rng, k, &SceneConfig::cube(n, depth, 0.0));
                let pose = solve(&scene)?;
                let rot_err = rotation_error_deg(pose.rotation, scene.rotation);
                let t_err = translation_error(pose.translation, scene.translation)
                    / scene.translation.length();
                assert!(rot_err < EXACT_DEG, "n={n}: rotation error {rot_err} deg");
                assert!(t_err < 1e-4, "n={n}: relative translation error {t_err}");
                assert!(
                    pose.reproj_rmse.is_some_and(|e| e < EXACT_RMSE_PX),
                    "n={n}: {pose:?}"
                );
            }
        }
        Ok(())
    }

    /// P3P has up to four exact solutions; SQPnP must return one that reprojects exactly.
    #[test]
    fn three_points_reproject_exactly() -> Result<(), PnPError> {
        let k = k_default();
        let mut rng = StdRng::seed_from_u64(2);
        for _ in 0..20 {
            let scene = Scene::generate(&mut rng, k, &SceneConfig::cube(3, 5.0, 0.0));
            let pose = solve(&scene)?;
            assert!(
                pose.reproj_rmse.is_some_and(|e| e < EXACT_RMSE_PX),
                "{pose:?}"
            );
        }
        Ok(())
    }

    /// Coplanar points give Ω extra null vectors; the pose must still be exact.
    #[test]
    fn planar_points_are_exact() -> Result<(), PnPError> {
        let k = k_default();
        let mut rng = StdRng::seed_from_u64(3);
        for n in [4, 6, 20] {
            for _ in 0..10 {
                let scene = Scene::generate(&mut rng, k, &marker(n, 0.0));
                let pose = solve(&scene)?;
                let rot_err = rotation_error_deg(pose.rotation, scene.rotation);
                assert!(
                    rot_err < EXACT_DEG,
                    "planar n={n}: rotation error {rot_err} deg"
                );
                assert!(
                    translation_error(pose.translation, scene.translation) < 1e-3,
                    "planar n={n}"
                );
            }
        }
        Ok(())
    }

    /// A 20 cm object 6 m away (fx = 600), a long-range scene (cf. #1075): SQPnP must stay exact.
    #[test]
    fn small_object_far_away_is_exact() -> Result<(), PnPError> {
        let k = k_matrix(600.0, 600.0, 320.0, 240.0);
        let mut rng = StdRng::seed_from_u64(4);
        for _ in 0..10 {
            let config = SceneConfig {
                half_size: 0.1,
                ..SceneConfig::cube(50, 6.0, 0.0)
            };
            let scene = Scene::generate(&mut rng, k, &config);
            let pose = solve(&scene)?;
            assert!(
                rotation_error_deg(pose.rotation, scene.rotation) < 0.01,
                "{pose:?}"
            );
            assert!(
                translation_error(pose.translation, scene.translation) < 1e-3,
                "{pose:?}"
            );
        }
        Ok(())
    }

    /// With 1 px noise SQPnP's object-space optimum reprojects within a few percent of the
    /// image-space optimum (LM started from the true pose), and its LM-refined pose reaches it.
    #[test]
    fn noisy_poses_are_accurate() -> Result<(), Box<dyn std::error::Error>> {
        let k = k_default();
        let mut rng = StdRng::seed_from_u64(5);
        for _ in 0..20 {
            let scene = Scene::generate(&mut rng, k, &SceneConfig::cube(100, 6.0, 1.0));
            let pose = solve(&scene)?;
            let reference = refine_pose_lm(
                &scene.world,
                &scene.image,
                &k,
                &scene.rotation.as_mat3af32(),
                &scene.translation.as_vec3af32(),
                None,
                &LMRefineParams::default(),
            )?;
            let best = reference.reproj_rmse.ok_or("LM returned no RMSE")?;
            let rmse = pose.reproj_rmse.ok_or("SQPnP returned no RMSE")?;
            assert!(
                rmse <= best * 1.05 + 1e-3,
                "SQPnP rmse {rmse} vs optimum {best}"
            );
            assert!(rotation_error_deg(pose.rotation, scene.rotation) < 0.5);

            let params = with_lm();
            let refined = solve_with(&scene, &params)?;
            let refined_rmse = refined.reproj_rmse.ok_or("SQPnP + LM returned no RMSE")?;
            assert!(
                refined_rmse <= best * 1.001 + 1e-4,
                "{refined_rmse} vs {best}"
            );
        }
        Ok(())
    }

    /// Scaling the world (and translation) and moving its origin must not change the rotation:
    /// the points are centred and the rank test is relative.
    #[test]
    fn invariant_to_world_scale_and_origin() -> Result<(), PnPError> {
        let k = k_default();
        let mut rng = StdRng::seed_from_u64(6);
        let scene = Scene::generate(&mut rng, k, &SceneConfig::cube(30, 5.0, 0.5));
        let pose = solve(&scene)?;
        for (scale, shift) in [(1e-3, 0.0), (1e3, 0.0), (1.0, 100.0)] {
            // M' = s·M + d projects to the same pixels with the translation s·t − R·d.
            let d = Vec3F64::new(shift, -shift, shift);
            let world: Vec<Vec3AF32> = scene
                .world
                .iter()
                .map(|&p| (Vec3F64::from(p) * scale + d).as_vec3af32())
                .collect();
            let moved = solve_sqpnp(&world, &scene.image, &k, None, &SQPnPParams::default())?;
            let delta = rotation_error_deg(moved.rotation, Mat3F64::from(pose.rotation));
            assert!(
                delta < EXACT_DEG,
                "scale {scale}, shift {shift}: rotation changed by {delta} deg"
            );
        }
        Ok(())
    }

    /// Uses fx ≠ fy, so reading one focal length for the other fails it.
    #[test]
    fn distortion_is_undistorted_before_solving() -> Result<(), PnPError> {
        let k = k_matrix(800.0, 760.0, 640.0, 480.0);
        let intrinsic = CameraIntrinsic {
            fx: 800.0,
            fy: 760.0,
            cx: 640.0,
            cy: 480.0,
        };
        let distortion = PolynomialDistortion {
            k1: -0.4,
            k2: 0.08,
            p1: 1e-3,
            p2: -5e-4,
            ..Default::default()
        };
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..10 {
            // 3 m away the points reach the image corners, where the distortion is strongest.
            let mut scene = Scene::generate(&mut rng, k, &SceneConfig::cube(60, 3.0, 0.0));
            for uv in &mut scene.image {
                let p = Vec2F64::from(*uv);
                let (u, v) = distort_point_polynomial(p.x, p.y, &intrinsic, &distortion);
                *uv = Vec2F32::new(u as f32, v as f32);
            }

            let with_model = solve_sqpnp(
                &scene.world,
                &scene.image,
                &k,
                Some(&distortion),
                &SQPnPParams::default(),
            )?;
            assert!(rotation_error_deg(with_model.rotation, scene.rotation) < EXACT_DEG);
            assert!(
                with_model.reproj_rmse.is_some_and(|e| e < EXACT_RMSE_PX),
                "{with_model:?}"
            );

            // Ignoring the distortion leaves pixel-sized residuals.
            let without_model = solve(&scene)?;
            assert!(
                without_model.reproj_rmse.is_some_and(|e| e > 1.0),
                "{without_model:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn rejects_invalid_input() {
        let k = k_default();
        let (world, image) = (valid_world(), valid_image());
        let params = SQPnPParams::default();
        let solve_with_k = |world: &[Vec3AF32], image: &[Vec2F32], k: &Mat3AF32| {
            solve_sqpnp(world, image, k, None, &params)
        };

        assert!(matches!(
            solve_with_k(&world[..2], &image[..2], &k),
            Err(PnPError::InsufficientCorrespondences {
                required: 3,
                actual: 2
            })
        ));
        assert!(matches!(
            solve_with_k(&world, &image[..3], &k),
            Err(PnPError::MismatchedArrayLengths { .. })
        ));
        // All world points coincide: centred, they are all zero, and so is Ω.
        assert!(matches!(
            solve_with_k(&[world[4]; 5], &image, &k),
            Err(PnPError::DegenerateConfiguration(
                "all world points coincide"
            ))
        ));
        // Every line of sight is the same ray. Off the principal point, rounding leaves
        // S = Σ Q_i invertible in floating point.
        for pixel in [Vec2F32::new(640.0, 480.0), Vec2F32::new(700.0, 433.0)] {
            assert!(
                matches!(
                    solve_with_k(&world, &[pixel; 5], &k),
                    Err(PnPError::DegenerateConfiguration(
                        "all lines of sight are parallel"
                    ))
                ),
                "{pixel:?}"
            );
        }
        let singular = Mat3AF32::from_cols_array(&[0.0; 9]);
        assert!(matches!(
            solve_with_k(&world, &image, &singular),
            Err(PnPError::InvalidCameraMatrix)
        ));
        let mut not_finite = k;
        not_finite.x_axis.x = f32::NAN;
        assert!(matches!(
            solve_with_k(&world, &image, &not_finite),
            Err(PnPError::InvalidCameraMatrix)
        ));
        // Collinear points leave the rotation about their line free; SQPnP still returns a pose.
        let collinear: Vec<Vec3AF32> = (0..6).map(|i| Vec3AF32::new(i as f32, 0.0, 0.0)).collect();
        let pixels: Vec<Vec2F32> = (0..6)
            .map(|i| Vec2F32::new(600.0 + 20.0 * i as f32, 480.0))
            .collect();
        let pose = solve_with_k(&collinear, &pixels, &k);
        assert!(
            pose.as_ref().is_ok_and(|p| p.rotation.is_finite()
                && p.translation.to_array().into_iter().all(f32::is_finite)),
            "{pose:?}"
        );
    }

    /// Close to the camera the object-space cost (3D distance to the line of sight) and the
    /// pixel error have different minima: SQPnP must win on the first, `refine_lm` on the second.
    #[test]
    fn object_space_optimum_and_lm_refinement() -> Result<(), PnPError> {
        let k = k_default();
        let sight = LineOfSight::new(k, None)?;
        let lm = with_lm();
        let mut rng = StdRng::seed_from_u64(9);
        for _ in 0..10 {
            // Points as close as ~0.3 m with 3 px noise.
            let scene = Scene::generate(&mut rng, k, &SceneConfig::cube(50, 2.0, 3.0));
            let cost = |pose: &PnPResult| {
                let (r, t) = (
                    Mat3F64::from(pose.rotation),
                    Vec3F64::from(pose.translation),
                );
                object_space_error(&scene, &sight, Vec3F64::ZERO, r, t)
            };
            let plain = solve(&scene)?;
            let refined = solve_with(&scene, &lm)?;
            assert!(cost(&plain)? <= cost(&refined)?);
            let (a, b) = (plain.reproj_rmse, refined.reproj_rmse);
            assert!(
                b.zip(a).is_some_and(|(b, a)| b < a - 1e-3),
                "{b:?} vs {a:?}"
            );
        }
        Ok(())
    }

    /// Planar poses have a twin of equal cost behind the camera; the early stop must still find
    /// the same minimum as running SQP from every eigenvector of Ω.
    #[test]
    fn noisy_planar_search_matches_every_start() -> Result<(), PnPError> {
        let k = k_default();
        let params = SQPnPParams::default();
        let sight = LineOfSight::new(k, None)?;
        let mut rng = StdRng::seed_from_u64(84);
        let mut checked = 0;
        for _ in 0..50 {
            let scene = Scene::generate(&mut rng, k, &marker(6, 2.0));
            let objective =
                Objective::new(&scene.world, &scene.image, centroid(&scene.world), &sight)?;
            let found = minimize_on_so3(&objective, &params)?;

            let eig = SymmetricEigen::new(objective.omega);
            let mut every = f64::INFINITY;
            for e in eig.eigenvectors.column_iter() {
                let x: Vec9 = e * SQRT3;
                for start in [x, -x] {
                    let Some(solution) =
                        project_to_so3(&start).and_then(|r0| sqp(&objective.omega, r0, &params))
                    else {
                        continue;
                    };
                    if objective.centroid_depth(&solution.r) > 0.0 {
                        every = every.min(objective.cost(&solution.r));
                    }
                }
            }
            // The guarantee holds only when SQP converged.
            if found.solution.converged {
                assert!(
                    found.cost <= every * (1.0 + 1e-6),
                    "{} vs {every}",
                    found.cost
                );
                checked += 1;
            }
        }
        assert!(checked >= 40, "only {checked} of 50 scenes converged");
        Ok(())
    }

    /// With almost no noise the null vector of Ω is accepted without SQP, but it is only close
    /// to orthonormal (up to ~1e-4); the returned rotation must still be exact to f32 precision.
    #[test]
    fn fast_path_rotation_is_orthonormal() -> Result<(), PnPError> {
        let k = k_default();
        let mut rng = StdRng::seed_from_u64(11);
        for _ in 0..20 {
            let scene = Scene::generate(&mut rng, k, &SceneConfig::cube(100, 6.0, 0.001));
            let pose = solve(&scene)?;
            assert_eq!(pose.num_iterations, Some(0), "fast path not taken");
            let r = Mat3F64::from(pose.rotation);
            let off = (r.transpose() * r - Mat3F64::IDENTITY).to_cols_array();
            assert!(off.iter().all(|v| v.abs() < 1e-6), "RᵀR − I = {off:?}");
        }
        Ok(())
    }

    /// `converged` must report whether SQP's step got below `sqp_squared_tolerance`: true with the
    /// default iteration cap, false when the cap stops SQP after one step.
    #[test]
    fn reports_whether_sqp_converged() -> Result<(), PnPError> {
        let k = k_default();
        let mut rng = StdRng::seed_from_u64(12);
        let one_step = SQPnPParams {
            sqp_max_iterations: 1,
            ..Default::default()
        };
        for _ in 0..10 {
            // With 2 px noise on 6 points the null vector of Ω is not a rotation, so SQP runs.
            let scene = Scene::generate(&mut rng, k, &SceneConfig::cube(6, 5.0, 2.0));
            let pose = solve(&scene)?;
            assert!(pose.num_iterations.is_some_and(|i| i > 0), "{pose:?}");
            assert_eq!(pose.converged, Some(true), "{pose:?}");

            let capped = solve_with(&scene, &one_step)?;
            assert_eq!(capped.num_iterations, Some(1), "{capped:?}");
            assert_eq!(capped.converged, Some(false), "{capped:?}");
        }
        Ok(())
    }

    /// `rᵀ Ω r` must be the object-space error `Σ ‖Q_i (R·(M_i − c) + P·r)‖²` of the rotation `r`
    /// with its optimal translation `P·r`, and `P·r` must minimise it over all translations.
    #[test]
    fn objective_matches_the_object_space_error() -> Result<(), PnPError> {
        let k = k_default();
        let mut rng = StdRng::seed_from_u64(13);
        let scene = Scene::generate(&mut rng, k, &SceneConfig::cube(20, 5.0, 2.0));
        let c = centroid(&scene.world);
        let sight = LineOfSight::new(k, None)?;
        let objective = Objective::new(&scene.world, &scene.image, c, &sight)?;
        let error = |r, t| object_space_error(&scene, &sight, c, r, t);
        for _ in 0..5 {
            let rotation = random_rotation(&mut rng, std::f64::consts::PI);
            let r = vec9(rotation);
            let t = objective.translation(&r);
            let direct = error(rotation, t)?;
            assert!(
                (objective.cost(&r) - direct).abs() <= 1e-9 * direct.max(1.0),
                "rᵀΩr = {} vs {direct}",
                objective.cost(&r)
            );
            // Moving the translation away from P·r can only increase the error.
            for step in [
                Vec3F64::new(1.0, 0.0, 0.0),
                Vec3F64::new(0.0, 1.0, 0.0),
                Vec3F64::new(0.0, 0.0, 1.0),
            ] {
                assert!(error(rotation, t + step * 1e-3)? > direct);
                assert!(error(rotation, t - step * 1e-3)? > direct);
            }
        }
        Ok(())
    }

    /// The orthonormality constraints vanish on a rotation, and their Jacobian matches central
    /// differences at an arbitrary (non-rotation) point.
    #[test]
    fn constraints_vanish_on_rotations_and_match_finite_differences() {
        let mut rng = StdRng::seed_from_u64(14);
        let rotation = vec9(random_rotation(&mut rng, std::f64::consts::PI));
        assert!(Constraints::new(&rotation).h.norm() < 1e-12);

        let r = Vec9::from_fn(|i, _| (i as f64 * 0.7).sin());
        let analytic = Constraints::new(&r).jacobian;
        let eps = 1e-6;
        for j in 0..9 {
            let mut step = Vec9::zeros();
            step[j] = eps;
            let numeric =
                (Constraints::new(&(r + step)).h - Constraints::new(&(r - step)).h) / (2.0 * eps);
            assert!(
                (analytic.column(j) - numeric).norm() < 1e-8,
                "column {j}: {} vs {numeric}",
                analytic.column(j)
            );
        }
    }

    /// With `Ω = 0` the KKT matrix is singular, so the LU solve fails and SQP must take the
    /// least-squares step instead. At a rotation that step is zero: SQP converges at once and
    /// returns the start unchanged.
    #[test]
    fn sqp_takes_a_least_squares_step_on_a_singular_system() {
        let rotation = SO3F64::exp(Vec3F64::new(0.3, -0.5, 0.8).normalize() * 0.7).matrix();
        let r0 = vec9(rotation);
        let solution = sqp(&Mat9::zeros(), r0, &SQPnPParams::default());
        assert!(
            solution.is_some_and(|s| s.iterations == 1 && s.converged && (s.r - r0).norm() < 1e-9),
            "SQP did not stop at the start"
        );
    }

    /// `‖X·Xᵀ − I‖²_F` for `X = diag(1, 1, 2)` is `(4 − 1)² = 9`, all of it in the third column.
    #[test]
    fn orthogonality_error_counts_every_column() {
        let x = vec9(Mat3F64::from_diagonal(Vec3F64::new(1.0, 1.0, 2.0)));
        assert_eq!(orthogonality_squared_error(&x), 9.0);
        assert_eq!(orthogonality_squared_error(&vec9(Mat3F64::IDENTITY)), 0.0);
    }

    #[test]
    fn eigenvector_sign_is_fixed_to_a_rotation() {
        let rotation = SO3F64::exp(Vec3F64::new(0.3, -0.7, 1.1)).matrix();
        let r = vec9(rotation);
        assert_eq!(with_positive_det(r), r);
        assert_eq!(with_positive_det(-r), r);
    }

    #[test]
    fn rejects_non_finite_values() {
        // nalgebra's unbounded SVD never converges on NaN; these calls must still return.
        let nan = Vec9::from_element(f64::NAN);
        assert!(project_to_so3(&nan).is_none());
        assert!(sqp(&Mat9::identity(), nan, &SQPnPParams::default()).is_none());

        let k = k_default();
        let (mut world, mut image) = (valid_world(), valid_image());
        let plain = SQPnPParams::default();
        let with_lm = with_lm();
        world[1].x = f32::NAN;
        assert!(matches!(
            solve_sqpnp(&world, &image, &k, None, &plain),
            Err(PnPError::NonFinite("world point"))
        ));
        world[1].x = 1.0;
        image[2].y = f32::INFINITY;
        assert!(matches!(
            solve_sqpnp(&world, &image, &k, None, &plain),
            Err(PnPError::NonFinite("image point"))
        ));
        // Finite but extreme values overflow the reprojection error; with LM, `refine_pose_lm`
        // reports its own failure first.
        image[2].y = 1e30;
        assert!(matches!(
            solve_sqpnp(&world, &image, &k, None, &plain),
            Err(PnPError::NonFinite("reprojection error"))
        ));
        assert!(matches!(
            solve_sqpnp(&world, &image, &k, None, &with_lm),
            Err(PnPError::SvdFailed(_))
        ));
        image[2].y = 500.0;
        world[0] = Vec3AF32::new(3e38, 3e38, 3e38);
        for params in [&plain, &with_lm] {
            assert!(matches!(
                solve_sqpnp(&world, &image, &k, None, params),
                Err(PnPError::NonFinite("reprojection error"))
            ));
        }
    }

    #[test]
    fn dispatched_through_solve_pnp_and_ransac() -> Result<(), Box<dyn std::error::Error>> {
        let k = k_default();
        let mut rng = StdRng::seed_from_u64(8);
        let mut scene = Scene::generate(&mut rng, k, &SceneConfig::cube(40, 5.0, 0.0));

        let direct = solve(&scene)?;
        let dispatched = solve_pnp(
            &scene.world,
            &scene.image,
            &k,
            None,
            PnPMethod::SQPnPDefault,
        )?;
        assert_eq!(direct.rotation, dispatched.rotation);

        // `PnPMethod::SQPnP` must pass its parameters on: a 1-step cap on a noisy scene shows.
        let noisy = Scene::generate(
            &mut StdRng::seed_from_u64(12),
            k,
            &SceneConfig::cube(6, 5.0, 2.0),
        );
        let one_step = SQPnPParams {
            sqp_max_iterations: 1,
            ..Default::default()
        };
        let capped = solve_pnp(
            &noisy.world,
            &noisy.image,
            &k,
            None,
            PnPMethod::SQPnP(one_step),
        )?;
        assert_eq!(
            (capped.num_iterations, capped.converged),
            (Some(1), Some(false))
        );

        // 40 inliers plus 40 outliers.
        scene.add_outliers(&mut rng, 40);
        let params = RansacParams {
            max_iterations: 2000,
            reproj_threshold_px: 2.0,
            random_seed: Some(1),
            ..Default::default()
        };
        let result = solve_pnp_ransac(
            &scene.world,
            &scene.image,
            &k,
            None,
            PnPMethod::SQPnPDefault,
            &params,
        )?;
        let true_inliers = result.inliers.iter().filter(|&&i| i < 40).count();
        assert!(true_inliers >= 38, "found {true_inliers} of 40 inliers");
        assert!(rotation_error_deg(result.pose.rotation, scene.rotation) < 0.05);
        Ok(())
    }
}
