//! SQPnP: Perspective-n-Point by sequential quadratic programming on SO(3).
//!
//! Paper: [Terzakis & Lourakis, "A Consistently Fast and Globally Optimal Solution to the
//! Perspective-n-Point Problem", ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460460.pdf)
//!
//! Reference: [OpenCV SQPnP implementation](https://github.com/opencv/opencv/blob/4.x/modules/calib3d/src/sqpnp.cpp)
//! and the authors' [C++ implementation](https://github.com/terzakig/sqpnp).
//!
//! # Method
//!
//! Let `r = vec(R)` be the columns of the rotation stacked into a 9-vector (the order nalgebra and
//! glam store a 3×3 matrix in, so `r` and `R` are the same nine numbers) and `v_i = (x_i, y_i, 1)`
//! the normalised (undistorted) image point `i`. SQPnP minimises the object-space error
//!
//! ```text
//! E(R, t) = Σ_i ‖Q_i (R·M_i + t)‖²,   Q_i = I − v_i v_iᵀ / (v_iᵀ v_i),
//! ```
//!
//! the squared distance of each transformed world point `M_i` from the line of sight of its
//! image point. For a fixed rotation the optimal translation is linear in `r`, `t = P·r`, so the
//! cost reduces to the quadratic form `E(r) = rᵀ Ω r` with a 9×9 positive semi-definite `Ω`.
//!
//! `rᵀ Ω r` is then minimised over SO(3): SQP is started from the rotations nearest to `±√3·e`
//! for the eigenvectors `e` of `Ω` with the smallest eigenvalues (`‖vec(R)‖² = 3` for every
//! rotation). Following the paper, further eigenvectors `e_k` are tried while the best cost
//! found so far is above `3·λ_k`; below that bound they cannot lead to a better minimum.
//!
//! The bound is on the minimum over all of SO(3), but only rotations that put the centroid of the
//! world points in front of the camera are accepted. For planar points every pose has a twin of
//! equal cost behind the camera (rotated by π about the plane normal, translation negated), so
//! when a candidate rejected by this depth check has a lower cost (by more than 1e-6, relative) the
//! search does not stop early.
//! The paper's bound alone can return a local minimum in that case; OpenCV checks the depth of
//! the centroid or of most points, and this module checks the centroid only.
//!
//! # Limits
//!
//! - The result is the global minimum of `E` when SQP converges. On noisy, nearly fronto-parallel
//!   planar scenes `Ω` is ill-conditioned and SQP can stop at `sqp_max_iterations` (reported as
//!   `converged == Some(false)`) slightly short of it.
//! - With exactly 3 points the problem has up to four exact solutions; SQPnP returns one of them,
//!   which is often not the true pose. Use 4 or more points for a unique pose.
//! - `E` measures 3-D distance to the line of sight, so points very close to the camera get little
//!   weight. When the pixel error matters, set `refine_lm`.
//!
//! # Precision and conditioning
//!
//! All internal computation is in `f64`; inputs and the returned pose use the crate's `f32`
//! types. World points are centred on their centroid before `Ω` is formed, and the rank test on
//! `Ω`'s eigenvalues is relative to the largest one, so the result does not depend on the
//! world-frame origin or scale.
//!
//! Points, `K`, the lines of sight and the returned pose are kornia-algebra types, converted with
//! its `From` impls. `Ω` (9×9), `B` and `P` (3×9), the 15×15 KKT system and the 9-vector `r` are
//! nalgebra matrices, since kornia-algebra has no types of those sizes; `S` (3×3) is inverted
//! there too, because `P = −S⁻¹·B` is formed there.
//!
//! # Distortion
//!
//! With `distortion = Some(..)` the image points are undistorted to normalised pinhole
//! coordinates first (`undistort_normalized_point_iter`), the pose is solved on those, and the
//! reported reprojection RMSE (and the optional LM refinement) use the distortion model.

use kornia_algebra::{Mat3AF32, Mat3F64, Vec2F32, Vec2F64, Vec3AF32, Vec3F64, SO3F32};
use kornia_imgproc::calibration::{
    distortion::{undistort_normalized_point_iter, PolynomialDistortion, TermCriteria},
    CameraIntrinsic,
};
use nalgebra::{Const, Matrix3, SMatrix, SVector, SymmetricEigen, Vector3};

use super::epnp::rmse_px;
use super::refine::{refine_pose_lm, LMRefineParams};
use super::{PnPError, PnPResult, PnPSolver};

type Mat9 = SMatrix<f64, 9, 9>;
type Vec9 = SVector<f64, 9>;
type Mat39 = SMatrix<f64, 3, 9>;
type Mat69 = SMatrix<f64, 6, 9>;
type Mat15 = SMatrix<f64, 15, 15>;
type Vec15 = SVector<f64, 15>;

/// Minimum number of correspondences SQPnP accepts. With exactly 3 the problem has up to four
/// exact solutions and SQPnP returns one of them.
const MIN_CORRESPONDENCES: usize = 3;

/// `√3`: the norm of `vec(R)` for any rotation `R`.
const SQRT3: f64 = 1.732_050_807_568_877_2;

/// Iteration cap for the SVD and eigen decompositions. nalgebra's default (`svd`,
/// `SymmetricEigen::new`) iterates until convergence, which never happens on non-finite input.
const MAX_DECOMPOSITION_ITERATIONS: usize = 1000;

/// Relative difference below which two costs `rᵀ Ω r` count as equal. A planar pose and its twin
/// behind the camera have the same cost up to round-off. OpenCV likewise treats accepted
/// solutions within `1e-6` as equal (`EQUAL_SQUARED_ERRORS_DIFF`, absolute there); here it is
/// relative, so it does not depend on the scale of the world points.
const EQUAL_COST_RELATIVE_TOLERANCE: f64 = 1e-6;

/// Stopping rule for undistorting each image point. The pose is solved on the undistorted points,
/// so the iteration has to converge far below the pixel noise. Measured on 50 noise-free scenes
/// 3 m away with `k1 = −0.4`: with the library default (5 iterations, `eps = 1e-2`) the solved
/// pose reprojects with up to 0.2 px RMSE, with this rule under 0.001 px.
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

/// Parameters controlling the SQPnP solver.
///
/// The defaults follow the OpenCV implementation, except the rank tolerance: OpenCV uses `1e-7`
/// as an absolute bound, here it is `1e-10` relative to the largest eigenvalue of `Ω`, so it does
/// not depend on the scale of the world points (see the module docs).
#[derive(Debug, Clone)]
pub struct SQPnPParams {
    /// An eigenvalue `λ` of `Ω` counts as zero (part of its null space) when
    /// `λ ≤ rank_tolerance · λ_max`. A cost `rᵀ Ω r` below `3 · rank_tolerance · λ_max` likewise
    /// counts as zero when the search compares costs (`‖vec(R)‖² = 3`).
    pub rank_tolerance: f64,
    /// SQP stops when the squared norm of the step is below this value.
    pub sqp_squared_tolerance: f64,
    /// Maximum number of SQP iterations per starting rotation.
    pub sqp_max_iterations: usize,
    /// A starting point `√3·e` whose squared orthogonality error `‖X·Xᵀ − I‖²_F` is below this
    /// value is already a rotation and is used without running SQP.
    pub orthogonality_squared_error_threshold: f64,
    /// Optional LM refinement of the reprojection error after SQPnP. SQPnP is optimal for its
    /// object-space cost; LM then minimises the image-space reprojection error instead.
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
/// * `k` - Camera intrinsics matrix.
/// * `distortion` - Optional lens distortion; image points are undistorted before solving.
/// * `params` - Solver parameters.
///
/// With exactly 3 points one of up to four exact solutions is returned; see the module docs.
///
/// # Returns
///
/// The world → camera pose (`R`, `t`, Rodrigues `rvec`) and its reprojection RMSE in pixels.
/// `num_iterations` and `converged` describe the SQP run that produced the pose, or the LM run
/// when `params.refine_lm` is set.
///
/// # Errors
///
/// Returns [`PnPError::MismatchedArrayLengths`] if the slices differ in length,
/// [`PnPError::InsufficientCorrespondences`] if `N < 3`, [`PnPError::SvdFailed`] if `K` is
/// singular, an input is not finite, the correspondences are degenerate (e.g. all image points
/// coincide) or the reprojection error overflows, and [`PnPError::CheiralityCheckFailed`] if
/// every candidate pose puts the centroid of the points behind the camera. With
/// `params.refine_lm` set, any error of [`refine_pose_lm`] is returned as well.
///
/// # Example
///
/// ```rust
/// use kornia_3d::pnp::{solve_sqpnp, SQPnPParams};
/// use kornia_algebra::{Mat3AF32, Vec2F32, Vec3AF32};
///
/// let k = Mat3AF32::from_cols_array(&[800.0, 0.0, 0.0, 0.0, 800.0, 0.0, 640.0, 480.0, 1.0]);
/// // Camera 5 m in front of the points, no rotation.
/// let world = [
///     Vec3AF32::new(-1.0, -0.8, -0.5),
///     Vec3AF32::new(0.9, -0.7, 0.3),
///     Vec3AF32::new(-0.6, 0.9, 0.8),
///     Vec3AF32::new(0.7, 0.6, -0.9),
///     Vec3AF32::new(0.1, -0.2, 1.0),
/// ];
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

    let sight = LineOfSight::new(k, distortion)?;
    // Centre the world points: it improves the conditioning of Ω and puts the centroid at the
    // origin, so its camera-frame depth is simply t'_z.
    let centroid = centroid(points_world);
    let objective = Objective::new(points_world, points_image, centroid, &sight)?;
    let best = minimize_on_so3(&objective, params)?;

    // t' is the translation for the centred points: R·(M − c) + t' = R·M + (t' − R·c).
    let rotation = mat(best.solution.r);
    let translation = objective.translation(&best.solution.r) - rotation * Vector3::from(centroid);
    if !translation.iter().all(|v| v.is_finite()) {
        return Err(PnPError::SvdFailed(
            "SQPnP: non-finite translation".to_string(),
        ));
    }

    let rotation_f32 = Mat3F64::from(rotation).as_mat3af32();
    let translation_f32 = Vec3F64::from(translation).as_vec3af32();

    if let Some(ref lm_params) = params.refine_lm {
        return refine_pose_lm(
            points_world,
            points_image,
            k,
            &rotation_f32,
            &translation_f32,
            distortion,
            lm_params,
        );
    }

    let rmse = rmse_px(
        points_world,
        points_image,
        &rotation_f32,
        &translation_f32,
        k,
        distortion,
    )?;
    // Finite but extreme inputs (e.g. a pixel at 1e30) can still overflow the reprojection.
    if !rmse.is_finite() {
        return Err(PnPError::SvdFailed(
            "SQPnP: non-finite reprojection error".to_string(),
        ));
    }

    Ok(PnPResult {
        rotation: rotation_f32,
        translation: translation_f32,
        rvec: SO3F32::from_matrix(&rotation_f32).log(),
        reproj_rmse: Some(rmse),
        num_iterations: Some(best.solution.iterations),
        converged: Some(best.solution.converged),
    })
}

/// The centroid of the world points, in `f64`.
fn centroid(points_world: &[Vec3AF32]) -> Vec3F64 {
    points_world
        .iter()
        .fold(Vec3F64::ZERO, |sum, &p| sum + Vec3F64::from(p))
        / points_world.len() as f64
}

/// Maps pixels to lines of sight `(x, y, 1)` in normalised pinhole coordinates `K⁻¹·(u, v, 1)`,
/// undistorting them first when a distortion model is given.
struct LineOfSight<'a> {
    k_inv: Mat3F64,
    intrinsic: CameraIntrinsic,
    distortion: Option<&'a PolynomialDistortion>,
}

impl<'a> LineOfSight<'a> {
    /// # Errors
    ///
    /// Returns [`PnPError::SvdFailed`] if `K` is singular or not finite.
    fn new(k: &Mat3AF32, distortion: Option<&'a PolynomialDistortion>) -> Result<Self, PnPError> {
        let k = Mat3F64::from(*k);
        let det = k.determinant();
        if !(det.is_finite() && det != 0.0) {
            return Err(PnPError::SvdFailed(
                "SQPnP: camera matrix K is singular or not finite".to_string(),
            ));
        }
        let intrinsic = CameraIntrinsic {
            fx: k.x_axis().x,
            fy: k.y_axis().y,
            cx: k.z_axis().x,
            cy: k.z_axis().y,
        };
        Ok(Self {
            k_inv: k.inverse(),
            intrinsic,
            distortion,
        })
    }

    /// # Errors
    ///
    /// Returns [`PnPError::SvdFailed`] if the pixel maps to a non-finite coordinate.
    fn of(&self, uv: &Vec2F32) -> Result<Vec3F64, PnPError> {
        let uv = Vec2F64::from(*uv);
        let h = self.k_inv * Vec3F64::new(uv.x, uv.y, 1.0);
        let (mut x, mut y) = (h.x / h.z, h.y / h.z);
        if let Some(d) = self.distortion {
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
            Err(PnPError::SvdFailed(
                "SQPnP: image point maps to a non-finite normalised coordinate".to_string(),
            ))
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
    /// Returns [`PnPError::SvdFailed`] if `S` is singular, i.e. all lines of sight are parallel
    /// (every image point is the same), or if an image point maps to a non-finite coordinate.
    fn new(
        points_world: &[Vec3AF32],
        points_image: &[Vec2F32],
        centroid: Vec3F64,
        sight: &LineOfSight,
    ) -> Result<Self, PnPError> {
        // Block j of A_i is M_i[j]·I₃, so block (j, l) of A_iᵀ Q_i A_i is M_i[j]·M_i[l]·Q_i and
        // block j of Q_i A_i is M_i[j]·Q_i. Only the blocks with j ≤ l are accumulated here; the
        // lower triangle of Ω is filled in from the upper one below.
        let mut omega = Mat9::zeros();
        let mut s = Matrix3::<f64>::zeros();
        let mut b = Mat39::zeros();

        for (p, uv) in points_world.iter().zip(points_image) {
            let m = (Vec3F64::from(*p) - centroid).to_array();
            let v = Vector3::from(sight.of(uv)?);
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

        let s_inv = s.try_inverse().ok_or_else(|| {
            PnPError::SvdFailed(
                "SQPnP: degenerate correspondences, all lines of sight are parallel".to_string(),
            )
        })?;
        let p = -(s_inv * b);
        // Ω += Bᵀ·P, accumulated in place.
        omega.gemm_tr(1.0, &b, &p, 1.0);
        // Ω is symmetric: mirroring the upper triangle gives the symmetric eigensolver and the KKT
        // system an exactly symmetric matrix.
        omega.fill_lower_triangle_with_upper_triangle();

        if omega.iter().all(|v| v.is_finite()) {
            Ok(Self { omega, p })
        } else {
            Err(PnPError::SvdFailed(
                "SQPnP: non-finite entries in Omega".to_string(),
            ))
        }
    }

    /// The optimal translation `t' = P·r` for the rotation `r` (and the centred world points).
    fn translation(&self, r: &Vec9) -> Vector3<f64> {
        self.p * r
    }

    /// `rᵀ Ω r`.
    fn cost(&self, r: &Vec9) -> f64 {
        r.dot(&(self.omega * r))
    }

    /// Camera-frame depth of the centroid of the world points under `r`. The points are centred,
    /// so it is `t'_z = (P·r)_z`.
    fn centroid_depth(&self, r: &Vec9) -> f64 {
        self.p.row(2).transpose().dot(r)
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
/// Returns [`PnPError::SvdFailed`] if `Ω` is all zeros, and [`PnPError::CheiralityCheckFailed`]
/// if no candidate passes the depth check.
fn minimize_on_so3(objective: &Objective, params: &SQPnPParams) -> Result<Candidate, PnPError> {
    let eig = SymmetricEigen::try_new(objective.omega, f64::EPSILON, MAX_DECOMPOSITION_ITERATIONS)
        .ok_or_else(|| {
            PnPError::SvdFailed("SQPnP: eigen-decomposition of Omega did not converge".to_string())
        })?;
    let mut order: [usize; 9] = core::array::from_fn(|i| i);
    order.sort_by(|&a, &b| eig.eigenvalues[a].total_cmp(&eig.eigenvalues[b]));
    let eigenvalue = |k: usize| eig.eigenvalues[order[k]].max(0.0);

    let lambda_max = eigenvalue(8);
    if !(lambda_max.is_finite() && lambda_max > 0.0) {
        return Err(PnPError::SvdFailed(
            "SQPnP: degenerate correspondences, Omega is zero".to_string(),
        ));
    }
    let num_null = (0..9)
        .filter(|&k| eigenvalue(k) <= params.rank_tolerance * lambda_max)
        .count()
        .max(1);

    let mut search = Search {
        objective,
        best: None,
        lowest_rejected_cost: f64::INFINITY,
    };
    let try_eigenvector = |k: usize, search: &mut Search| {
        let x: Vec9 = eig.eigenvectors.column(order[k]) * SQRT3;
        if orthogonality_squared_error(&x) < params.orthogonality_squared_error_threshold {
            // Already close to a rotation up to sign; project it so the result is orthonormal.
            if let Some(r) = nearest_rotation(&with_positive_det(x)) {
                search.consider(SqpSolution {
                    r,
                    iterations: 0,
                    converged: true,
                });
            }
        } else {
            for start in [x, -x] {
                if let Some(solution) =
                    nearest_rotation(&start).and_then(|r0| sqp(&objective.omega, r0, params))
                {
                    search.consider(solution);
                }
            }
        }
    };

    // As in the paper, every null vector is a start. (The bound below would reach them anyway
    // unless the best cost is already about zero, so this loop sets the order, not the result.)
    for k in 0..num_null {
        try_eigenvector(k, &mut search);
    }
    // The paper's bound: for k beyond the null space, a start from e_k can only improve on the
    // lowest cost over SO(3) if that cost is above 3·λ_k. The bound is on the unconstrained
    // minimum, so it only ends the search when that minimum passed the depth check. For planar
    // points the true pose has a twin of equal cost behind the camera (rotated by π about the
    // plane normal, translation negated). While a rejected rotation has a lower cost than the
    // best accepted one, keep searching; a twin of equal cost (within 1e-6, relative) does not
    // count.
    // `cost_floor` is the size of a cost that is numerically zero (noise-free points).
    let cost_floor = 3.0 * params.rank_tolerance * lambda_max;
    for k in num_null..9 {
        let best_cost = search.best.as_ref().map_or(f64::INFINITY, |c| c.cost);
        let rejected_is_lower = best_cost
            > search.lowest_rejected_cost * (1.0 + EQUAL_COST_RELATIVE_TOLERANCE) + cost_floor;
        if best_cost <= 3.0 * eigenvalue(k) && !rejected_is_lower {
            break;
        }
        try_eigenvector(k, &mut search);
    }

    search.best.ok_or(PnPError::CheiralityCheckFailed)
}

/// State of the search over the eigenvectors of `Ω`.
struct Search<'a> {
    objective: &'a Objective,
    /// Lowest-cost rotation that puts the centroid in front of the camera.
    best: Option<Candidate>,
    /// Lowest `rᵀ Ω r` among rotations rejected by the depth check.
    lowest_rejected_cost: f64,
}

impl Search<'_> {
    /// Keeps `solution` as the best candidate if it puts the centroid in front of the camera and
    /// has the lowest cost so far. Its rotation comes from `nearest_rotation`, so it is finite.
    fn consider(&mut self, solution: SqpSolution) {
        let cost = self.objective.cost(&solution.r);
        if self.objective.centroid_depth(&solution.r) <= 0.0 {
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
    let mut kkt = Mat15::zeros();
    kkt.fixed_view_mut::<9, 9>(0, 0).copy_from(omega);

    while iterations < params.sqp_max_iterations {
        iterations += 1;
        let constraints = Constraints::at(&r);
        kkt.fixed_view_mut::<6, 9>(9, 0)
            .copy_from(&constraints.jacobian);
        kkt.fixed_view_mut::<9, 6>(0, 9)
            .tr_copy_from(&constraints.jacobian);

        // rhs = (−Ω·r, −h), written in place.
        let mut rhs = Vec15::zeros();
        rhs.fixed_rows_mut::<9>(0).gemv(-1.0, omega, &r, 0.0);
        rhs.fixed_rows_mut::<6>(9).axpy(-1.0, &constraints.h, 0.0);

        // LU is enough away from degeneracies; fall back to a least-squares SVD solve when the
        // KKT matrix is (numerically) singular. Both factorise a copy of `kkt` (a stack value),
        // since its Ω block is reused by the next iteration.
        let solution = kkt
            .lu()
            .solve(&rhs)
            .filter(|s| s.iter().all(|v| v.is_finite()))
            .or_else(|| {
                kkt.try_svd(true, true, f64::EPSILON, MAX_DECOMPOSITION_ITERATIONS)?
                    .solve(&rhs, 1e-12)
                    .ok()
            });
        let Some(solution) = solution else {
            break;
        };
        let delta = solution.fixed_rows::<9>(0);
        r += delta;
        if delta.norm_squared() < params.sqp_squared_tolerance {
            converged = true;
            break;
        }
    }

    Some(SqpSolution {
        r: nearest_rotation(&r)?,
        iterations,
        converged,
    })
}

/// The six orthonormality constraints `h(r) = 0` of the rows `r₁, r₂, r₃` of `R`, as in the paper
/// and OpenCV, `h = (‖r₁‖² − 1, ‖r₂‖² − 1, ‖r₃‖² − 1, r₁·r₂, r₂·r₃, r₁·r₃)`, and their Jacobian
/// `∂h/∂r`. On SO(3) column constraints would be equivalent, but away from it they linearise
/// differently and SQP would take other steps than the paper and OpenCV.
struct Constraints {
    h: SVector<f64, 6>,
    jacobian: Mat69,
}

impl Constraints {
    fn at(r: &Vec9) -> Self {
        let m = mat(*r);
        let (r1, r2, r3) = (m.row(0), m.row(1), m.row(2));

        let h = SVector::<f64, 6>::from([
            r1.norm_squared() - 1.0,
            r2.norm_squared() - 1.0,
            r3.norm_squared() - 1.0,
            r1.dot(&r2),
            r2.dot(&r3),
            r1.dot(&r3),
        ]);

        // Entry (i, j) of R is r[3j + i] (columns stacked).
        let mut jacobian = Mat69::zeros();
        for j in 0..3 {
            let (row1, row2, row3) = (3 * j, 3 * j + 1, 3 * j + 2);
            jacobian[(0, row1)] = 2.0 * r1[j];
            jacobian[(1, row2)] = 2.0 * r2[j];
            jacobian[(2, row3)] = 2.0 * r3[j];
            // d(r1·r2)
            jacobian[(3, row1)] = r2[j];
            jacobian[(3, row2)] = r1[j];
            // d(r2·r3)
            jacobian[(4, row2)] = r3[j];
            jacobian[(4, row3)] = r2[j];
            // d(r1·r3)
            jacobian[(5, row1)] = r3[j];
            jacobian[(5, row3)] = r1[j];
        }
        Self { h, jacobian }
    }
}

/// `vec(R)` read back as the 3×3 matrix it stores, with columns `r[0..3]`, `r[3..6]`, `r[6..9]`.
/// nalgebra keeps a 3×3 matrix in this column order, so no element changes place.
fn mat(r: Vec9) -> Matrix3<f64> {
    r.reshape_generic(Const::<3>, Const::<3>)
}

/// `x` or `−x`, whichever has `det(mat(x)) ≥ 0`.
///
/// An eigenvector of `Ω` is only defined up to sign, and for an orthogonal `mat(x)` negating it
/// flips `det` between `−1` (a reflection) and `+1` (a rotation).
fn with_positive_det(x: Vec9) -> Vec9 {
    if mat(x).determinant() < 0.0 {
        -x
    } else {
        x
    }
}

/// Squared Frobenius norm of `X·Xᵀ − I` for `X = mat(x)`.
fn orthogonality_squared_error(x: &Vec9) -> f64 {
    let m = mat(*x);
    (m * m.transpose() - Matrix3::identity()).norm_squared()
}

/// The rotation closest to `mat(x)` in the Frobenius norm (`Mat3F64::nearest_rotation`), as
/// `vec(R)`, or `None` if `x` is not finite or the SVD fails.
fn nearest_rotation(x: &Vec9) -> Option<Vec9> {
    let rotation = Mat3F64::from(mat(*x)).nearest_rotation()?;
    let r = Matrix3::from(rotation).reshape_generic(Const::<9>, Const::<1>);
    r.iter().all(|v| v.is_finite()).then_some(r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pnp::{solve_pnp, solve_pnp_ransac, EPnPParams, PnPMethod, RansacParams};
    use crate::test_utils::{
        k_matrix, random_rotation, rotation_error_deg, translation_error, Scene,
    };
    use kornia_algebra::SO3F64;
    use kornia_imgproc::calibration::distortion::distort_point_polynomial;
    use rand::{rngs::StdRng, RngExt, SeedableRng};

    fn solve(scene: &Scene) -> Result<PnPResult, PnPError> {
        solve_sqpnp(
            &scene.world,
            &scene.image,
            &scene.k,
            None,
            &SQPnPParams::default(),
        )
    }

    /// `vec(R)`: the inverse of [`mat`].
    fn vec(m: Matrix3<f64>) -> Vec9 {
        m.reshape_generic(Const::<9>, Const::<1>)
    }

    #[test]
    fn noise_free_poses_are_exact() -> Result<(), PnPError> {
        let k = k_matrix(800.0, 800.0, 640.0, 480.0);
        let mut rng = StdRng::seed_from_u64(1);
        for n in [4, 5, 6, 10, 50, 200] {
            for _ in 0..10 {
                let depth = rng.random_range(3.0..10.0);
                let scene = Scene::random(&mut rng, k, n, 1.0, depth, 0.0);
                let pose = solve(&scene)?;
                let rot_err = rotation_error_deg(&pose.rotation, &scene.rotation);
                let t_err = translation_error(pose.translation, scene.translation)
                    / scene.translation.length();
                assert!(rot_err < 2e-3, "n={n}: rotation error {rot_err} deg");
                assert!(t_err < 1e-4, "n={n}: relative translation error {t_err}");
                assert!(
                    pose.reproj_rmse.is_some_and(|e| e < 1e-2),
                    "n={n}: {pose:?}"
                );
                if n >= 10 {
                    // The null vector of Ω is already ±vec(R): the sign fix alone must give the
                    // pose, without SQP. (With 6 points the f32 rounding of the pixels leaves it
                    // just above the orthogonality threshold in about 1 scene in 20.)
                    assert_eq!(pose.num_iterations, Some(0), "n={n}");
                }
            }
        }
        Ok(())
    }

    /// P3P has up to four exact solutions; SQPnP must return one that reprojects exactly.
    #[test]
    fn three_points_reproject_exactly() -> Result<(), PnPError> {
        let k = k_matrix(800.0, 800.0, 640.0, 480.0);
        let mut rng = StdRng::seed_from_u64(2);
        for _ in 0..20 {
            let scene = Scene::random(&mut rng, k, 3, 1.0, 5.0, 0.0);
            let pose = solve(&scene)?;
            assert!(pose.reproj_rmse.is_some_and(|e| e < 1e-2), "{pose:?}");
        }
        Ok(())
    }

    /// Coplanar points give Ω extra null vectors; the pose must still be exact.
    #[test]
    fn planar_points_are_exact() -> Result<(), PnPError> {
        let k = k_matrix(800.0, 800.0, 640.0, 480.0);
        let mut rng = StdRng::seed_from_u64(3);
        for n in [4, 6, 20] {
            for _ in 0..10 {
                let scene = Scene::random_planar(&mut rng, k, n, 0.0);
                let pose = solve(&scene)?;
                let rot_err = rotation_error_deg(&pose.rotation, &scene.rotation);
                assert!(rot_err < 2e-3, "planar n={n}: rotation error {rot_err} deg");
                assert!(
                    translation_error(pose.translation, scene.translation) < 1e-3,
                    "planar n={n}"
                );
            }
        }
        Ok(())
    }

    /// A 20 cm object 6 m away (fx = 600), the long-range case of #1075: SQPnP must stay exact.
    #[test]
    fn small_object_far_away_is_exact() -> Result<(), PnPError> {
        let k = k_matrix(600.0, 600.0, 320.0, 240.0);
        let mut rng = StdRng::seed_from_u64(4);
        for _ in 0..10 {
            let world = (0..50)
                .map(|_| {
                    Vec3F64::new(
                        rng.random_range(-0.1..0.1),
                        rng.random_range(-0.1..0.1),
                        rng.random_range(-0.1..0.1),
                    )
                })
                .collect();
            let rotation = random_rotation(&mut rng, std::f64::consts::PI);
            let translation = Vec3F64::new(0.1, -0.05, 6.0);
            let scene = Scene::project(&mut rng, k, world, rotation, translation, 0.0);
            let pose = solve(&scene)?;
            assert!(
                rotation_error_deg(&pose.rotation, &scene.rotation) < 0.01,
                "{pose:?}"
            );
            assert!(
                translation_error(pose.translation, scene.translation) < 1e-3,
                "{pose:?}"
            );
        }
        Ok(())
    }

    /// The EPnP unit test data (`test_solve_epnp`) was generated from a pose with translation
    /// (0.05, −0.04, 1.0) and no noise; SQPnP must recover it.
    #[test]
    fn recovers_the_epnp_unit_test_pose() -> Result<(), PnPError> {
        let world = [
            Vec3AF32::new(0.0315, 0.03333, -0.10409),
            Vec3AF32::new(-0.0315, 0.03333, -0.10409),
            Vec3AF32::new(0.0, -0.00102, -0.12977),
            Vec3AF32::new(0.02646, -0.03167, -0.1053),
            Vec3AF32::new(-0.02646, -0.031667, -0.1053),
            Vec3AF32::new(0.0, 0.04515, -0.11033),
        ];
        let image = [
            Vec2F32::new(722.96466, 502.0828),
            Vec2F32::new(669.88837, 498.61877),
            Vec2F32::new(707.0025, 478.48975),
            Vec2F32::new(728.05634, 447.56918),
            Vec2F32::new(682.6069, 443.91776),
            Vec2F32::new(696.4414, 511.96442),
        ];
        let k = k_matrix(800.0, 800.0, 640.0, 480.0);
        let pose = solve_sqpnp(&world, &image, &k, None, &SQPnPParams::default())?;
        let t = pose.translation;
        assert!((t.x - 0.05).abs() < 1e-3 && (t.y + 0.04).abs() < 1e-3 && (t.z - 1.0).abs() < 1e-3);
        assert!(pose.reproj_rmse.is_some_and(|e| e < 1e-2), "{pose:?}");
        Ok(())
    }

    /// With 1 px noise SQPnP's object-space optimum reprojects within a few percent of the
    /// image-space optimum, and its LM-refined pose reaches that optimum.
    #[test]
    fn noisy_poses_are_accurate() -> Result<(), PnPError> {
        let k = k_matrix(800.0, 800.0, 640.0, 480.0);
        let mut rng = StdRng::seed_from_u64(5);
        for _ in 0..20 {
            let scene = Scene::random(&mut rng, k, 100, 1.0, 6.0, 1.0);
            let pose = solve(&scene)?;
            let reference = solve_pnp(
                &scene.world,
                &scene.image,
                &k,
                None,
                PnPMethod::EPnP(EPnPParams {
                    refine_lm: Some(LMRefineParams::default()),
                    ..Default::default()
                }),
            )?;
            let best = reference.reproj_rmse.unwrap_or(f32::INFINITY);
            let rmse = pose.reproj_rmse.unwrap_or(f32::INFINITY);
            assert!(
                rmse <= best * 1.05 + 1e-3,
                "SQPnP rmse {rmse} vs optimum {best}"
            );
            assert!(rotation_error_deg(&pose.rotation, &scene.rotation) < 0.5);

            let params = SQPnPParams {
                refine_lm: Some(LMRefineParams::default()),
                ..Default::default()
            };
            let refined = solve_sqpnp(&scene.world, &scene.image, &k, None, &params)?;
            let refined_rmse = refined.reproj_rmse.unwrap_or(f32::INFINITY);
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
        let k = k_matrix(800.0, 800.0, 640.0, 480.0);
        let mut rng = StdRng::seed_from_u64(6);
        let scene = Scene::random(&mut rng, k, 30, 1.0, 5.0, 0.5);
        let pose = solve(&scene)?;
        for (scale, shift) in [(1e-3, 0.0), (1e3, 0.0), (1.0, 100.0)] {
            // M' = s·M + d and t' = s·t − R·d project to the same pixels.
            let d = Vec3F64::new(shift, -shift, shift);
            let world: Vec<Vec3AF32> = scene
                .world
                .iter()
                .map(|&p| (Vec3F64::from(p) * scale + d).as_vec3af32())
                .collect();
            let moved = solve_sqpnp(&world, &scene.image, &k, None, &SQPnPParams::default())?;
            let delta = rotation_error_deg(&moved.rotation, &Mat3F64::from(pose.rotation));
            assert!(
                delta < 2e-3,
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
        // Strong enough that the library's default undistortion stopping rule fails the 0.01 px
        // bound below (see `UNDISTORT_CRITERIA`).
        let distortion = PolynomialDistortion {
            k1: -0.4,
            k2: 0.08,
            k3: 0.0,
            k4: 0.0,
            k5: 0.0,
            k6: 0.0,
            p1: 1e-3,
            p2: -5e-4,
        };
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..10 {
            // 3 m away the points reach the image corners, where the distortion is strongest.
            let mut scene = Scene::random(&mut rng, k, 60, 1.0, 3.0, 0.0);
            for uv in &mut scene.image {
                let (u, v) =
                    distort_point_polynomial(uv.x as f64, uv.y as f64, &intrinsic, &distortion);
                *uv = Vec2F32::new(u as f32, v as f32);
            }

            let with_model = solve_sqpnp(
                &scene.world,
                &scene.image,
                &k,
                Some(&distortion),
                &SQPnPParams::default(),
            )?;
            assert!(rotation_error_deg(&with_model.rotation, &scene.rotation) < 2e-3);
            assert!(
                with_model.reproj_rmse.is_some_and(|e| e < 1e-2),
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
    fn rejects_invalid_input_without_panicking() {
        let k = k_matrix(800.0, 800.0, 640.0, 480.0);
        let world = [
            Vec3AF32::new(0.0, 0.0, 0.0),
            Vec3AF32::new(1.0, 0.0, 0.0),
            Vec3AF32::new(0.0, 1.0, 0.0),
            Vec3AF32::new(0.0, 0.0, 1.0),
        ];
        let same_pixel = [Vec2F32::new(640.0, 480.0); 4];
        let params = SQPnPParams::default();

        assert!(matches!(
            solve_sqpnp(&world[..2], &same_pixel[..2], &k, None, &params),
            Err(PnPError::InsufficientCorrespondences {
                required: 3,
                actual: 2
            })
        ));
        assert!(matches!(
            solve_sqpnp(&world, &same_pixel[..3], &k, None, &params),
            Err(PnPError::MismatchedArrayLengths { .. })
        ));
        // Every line of sight is the same ray: the translation cannot be recovered.
        assert!(matches!(
            solve_sqpnp(&world, &same_pixel, &k, None, &params),
            Err(PnPError::SvdFailed(_))
        ));
        // Singular camera matrix.
        let singular = Mat3AF32::from_cols_array(&[0.0; 9]);
        let image = [
            Vec2F32::new(600.0, 400.0),
            Vec2F32::new(700.0, 420.0),
            Vec2F32::new(650.0, 500.0),
            Vec2F32::new(620.0, 450.0),
        ];
        assert!(matches!(
            solve_sqpnp(&world, &image, &singular, None, &params),
            Err(PnPError::SvdFailed(message)) if message.contains("singular")
        ));
        // Collinear world points are degenerate; any result is fine as long as it does not panic.
        let collinear: Vec<Vec3AF32> = (0..6).map(|i| Vec3AF32::new(i as f32, 0.0, 0.0)).collect();
        let pixels: Vec<Vec2F32> = (0..6)
            .map(|i| Vec2F32::new(600.0 + 20.0 * i as f32, 480.0))
            .collect();
        let _ = solve_sqpnp(&collinear, &pixels, &k, None, &params);
    }

    /// Close to the camera the object-space cost (3D distance to the line of sight) and the
    /// pixel error have different minima: SQPnP must win on the first, `refine_lm` on the second.
    #[test]
    fn object_space_optimum_and_lm_refinement() -> Result<(), PnPError> {
        let k = k_matrix(800.0, 800.0, 640.0, 480.0);
        let k_inv = Mat3F64::from(k).inverse();
        let lm = SQPnPParams {
            refine_lm: Some(LMRefineParams::default()),
            ..Default::default()
        };
        let mut rng = StdRng::seed_from_u64(5);
        for _ in 0..10 {
            // Points as close as ~0.3 m with 3 px noise.
            let scene = Scene::random(&mut rng, k, 50, 1.0, 2.0, 3.0);
            let object_space_cost = |pose: &PnPResult| -> f64 {
                let r = Mat3F64::from(pose.rotation);
                let t = Vec3F64::from(pose.translation);
                scene
                    .world
                    .iter()
                    .zip(&scene.image)
                    .map(|(&m, u)| {
                        let p = Vec2F64::from(*u);
                        let v = Vector3::from(k_inv * Vec3F64::new(p.x, p.y, 1.0));
                        let q = Matrix3::identity() - v * v.transpose() / v.norm_squared();
                        (q * Vector3::from(r * Vec3F64::from(m) + t)).norm_squared()
                    })
                    .sum()
            };
            let plain = solve(&scene)?;
            let refined = solve_sqpnp(&scene.world, &scene.image, &k, None, &lm)?;
            assert!(object_space_cost(&plain) <= object_space_cost(&refined));
            let (a, b) = (plain.reproj_rmse, refined.reproj_rmse);
            assert!(
                b.zip(a).is_some_and(|(b, a)| b < a - 1e-3),
                "{b:?} vs {a:?}"
            );
        }
        Ok(())
    }

    /// For planar points every pose has a twin of equal cost behind the camera. When the search
    /// lands on the twin first, the early stop must not settle for a worse local minimum: the
    /// result must match running SQP from every eigenvector of Ω; without the depth-aware stop
    /// this test fails.
    #[test]
    fn noisy_planar_search_matches_every_start() -> Result<(), PnPError> {
        let k = k_matrix(800.0, 800.0, 640.0, 480.0);
        let params = SQPnPParams::default();
        let mut rng = StdRng::seed_from_u64(84);
        for _ in 0..200 {
            let scene = Scene::random_planar(&mut rng, k, 6, 2.0);
            let centroid = centroid(&scene.world);
            let sight = LineOfSight::new(&k, None)?;
            let objective = Objective::new(&scene.world, &scene.image, centroid, &sight)?;
            let found = minimize_on_so3(&objective, &params)?;

            let eig = SymmetricEigen::new(objective.omega);
            let mut every = f64::INFINITY;
            for e in eig.eigenvectors.column_iter() {
                let x: Vec9 = e * SQRT3;
                for start in [x, -x] {
                    let Some(solution) =
                        nearest_rotation(&start).and_then(|r0| sqp(&objective.omega, r0, &params))
                    else {
                        continue;
                    };
                    if objective.centroid_depth(&solution.r) > 0.0 {
                        every = every.min(objective.cost(&solution.r));
                    }
                }
            }
            // The guarantee assumes SQP converged. On these ill-conditioned scenes it sometimes
            // stops at its iteration cap instead, and then the result can fall slightly short.
            if found.solution.converged {
                assert!(
                    found.cost <= every * (1.0 + 1e-6),
                    "{} vs {every}",
                    found.cost
                );
            }
        }
        Ok(())
    }

    /// With almost no noise the null vector of Ω is accepted without SQP, but it is only close
    /// to orthonormal (up to ~1e-4); the returned rotation must still be exact to f32 precision.
    #[test]
    fn fast_path_rotation_is_orthonormal() -> Result<(), PnPError> {
        let k = k_matrix(800.0, 800.0, 640.0, 480.0);
        let mut rng = StdRng::seed_from_u64(11);
        for _ in 0..20 {
            let scene = Scene::random(&mut rng, k, 100, 1.0, 6.0, 0.001);
            let pose = solve(&scene)?;
            assert_eq!(pose.num_iterations, Some(0), "fast path not taken");
            let r = Matrix3::from(Mat3F64::from(pose.rotation));
            let error = (r.transpose() * r - Matrix3::identity()).norm();
            assert!(error < 1e-6, "‖RᵀR − I‖ = {error}");
        }
        Ok(())
    }

    /// `converged` must report whether SQP's step got below `sqp_squared_tolerance`: true with the
    /// default iteration cap, false when the cap stops SQP after one step.
    #[test]
    fn reports_whether_sqp_converged() -> Result<(), PnPError> {
        let k = k_matrix(800.0, 800.0, 640.0, 480.0);
        let mut rng = StdRng::seed_from_u64(12);
        let one_step = SQPnPParams {
            sqp_max_iterations: 1,
            ..Default::default()
        };
        for _ in 0..10 {
            // With 2 px noise on 6 points the null vector of Ω is not a rotation, so SQP runs.
            let scene = Scene::random(&mut rng, k, 6, 1.0, 5.0, 2.0);
            let pose = solve(&scene)?;
            assert!(pose.num_iterations.is_some_and(|i| i > 0), "{pose:?}");
            assert_eq!(pose.converged, Some(true), "{pose:?}");

            let capped = solve_sqpnp(&scene.world, &scene.image, &k, None, &one_step)?;
            assert_eq!(capped.num_iterations, Some(1), "{capped:?}");
            assert_eq!(capped.converged, Some(false), "{capped:?}");
        }
        Ok(())
    }

    #[test]
    fn centroid_is_the_mean_of_the_points() {
        let points = [
            Vec3AF32::new(1.0, 2.0, 3.0),
            Vec3AF32::new(-1.0, 0.0, 5.0),
            Vec3AF32::new(3.0, 1.0, -2.0),
        ];
        assert_eq!(centroid(&points).to_array(), [1.0, 1.0, 2.0]);
    }

    /// `vec(R)` stacks the columns of `R`, the order nalgebra keeps a 3×3 matrix in.
    #[test]
    fn vec_stacks_columns() {
        let r = Vec9::from_column_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]);
        let m = Matrix3::new(1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 10.0);
        assert_eq!(mat(r), m);
        assert_eq!(vec(m), r);
    }

    /// `rᵀ Ω r` must be the object-space error `Σ ‖Q_i (R·(M_i − c) + P·r)‖²` of the rotation `r`
    /// with its optimal translation `P·r`, and `P·r` must minimise it over all translations.
    #[test]
    fn objective_matches_the_object_space_error() -> Result<(), PnPError> {
        let k = k_matrix(800.0, 800.0, 640.0, 480.0);
        let mut rng = StdRng::seed_from_u64(13);
        let scene = Scene::random(&mut rng, k, 20, 1.0, 5.0, 2.0);
        let centroid = centroid(&scene.world);
        let sight = LineOfSight::new(&k, None)?;
        let objective = Objective::new(&scene.world, &scene.image, centroid, &sight)?;

        let error = |r: &Matrix3<f64>, t: &Vector3<f64>| -> Result<f64, PnPError> {
            let mut sum = 0.0;
            for (&p, uv) in scene.world.iter().zip(&scene.image) {
                let v = Vector3::from(sight.of(uv)?);
                let q = Matrix3::identity() - v * v.transpose() / v.norm_squared();
                let m = Vector3::from(Vec3F64::from(p) - centroid);
                sum += (q * (r * m + t)).norm_squared();
            }
            Ok(sum)
        };
        for _ in 0..5 {
            let rotation = Matrix3::from(random_rotation(&mut rng, std::f64::consts::PI));
            let r = vec(rotation);
            let t = objective.translation(&r);
            let direct = error(&rotation, &t)?;
            assert!(
                (objective.cost(&r) - direct).abs() <= 1e-9 * direct.max(1.0),
                "rᵀΩr = {} vs {direct}",
                objective.cost(&r)
            );
            // Moving the translation away from P·r can only increase the error.
            for step in [Vector3::x(), Vector3::y(), Vector3::z()] {
                assert!(error(&rotation, &(t + step * 1e-3))? > direct);
                assert!(error(&rotation, &(t - step * 1e-3))? > direct);
            }
        }
        Ok(())
    }

    /// The orthonormality constraints vanish on a rotation, and their Jacobian matches central
    /// differences at an arbitrary (non-rotation) point.
    #[test]
    fn constraints_vanish_on_rotations_and_match_finite_differences() {
        let mut rng = StdRng::seed_from_u64(14);
        let rotation = vec(Matrix3::from(random_rotation(
            &mut rng,
            std::f64::consts::PI,
        )));
        assert!(Constraints::at(&rotation).h.norm() < 1e-12);

        let r = Vec9::from_fn(|i, _| (i as f64 * 0.7).sin());
        let analytic = Constraints::at(&r).jacobian;
        let eps = 1e-6;
        for j in 0..9 {
            let mut step = Vec9::zeros();
            step[j] = eps;
            let numeric =
                (Constraints::at(&(r + step)).h - Constraints::at(&(r - step)).h) / (2.0 * eps);
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
        let r0 = vec(Matrix3::from(rotation));
        let solution = sqp(&Mat9::zeros(), r0, &SQPnPParams::default());
        assert!(
            solution.is_some_and(|s| s.iterations == 1 && s.converged && (s.r - r0).norm() < 1e-9),
            "SQP did not stop at the start"
        );
    }

    #[test]
    fn eigenvector_sign_is_fixed_to_a_rotation() {
        let rotation = SO3F64::exp(Vec3F64::new(0.3, -0.7, 1.1)).matrix();
        let r = vec(Matrix3::from(rotation));
        assert_eq!(with_positive_det(r), r);
        assert_eq!(with_positive_det(-r), r);
    }

    #[test]
    fn non_finite_values_terminate() {
        // nalgebra's unbounded SVD never converges on NaN, so before the iteration caps these
        // calls hung forever instead of returning.
        let nan = Vec9::from_element(f64::NAN);
        assert!(nearest_rotation(&nan).is_none());
        assert!(sqp(&Mat9::identity(), nan, &SQPnPParams::default()).is_none());

        let k = k_matrix(800.0, 800.0, 640.0, 480.0);
        let mut world = vec![
            Vec3AF32::new(0.0, 0.0, 0.0),
            Vec3AF32::new(1.0, 0.0, 0.0),
            Vec3AF32::new(0.0, 1.0, 0.0),
            Vec3AF32::new(0.0, 0.0, 1.0),
            Vec3AF32::new(1.0, 1.0, 1.0),
        ];
        let mut image = vec![
            Vec2F32::new(600.0, 400.0),
            Vec2F32::new(700.0, 420.0),
            Vec2F32::new(650.0, 500.0),
            Vec2F32::new(620.0, 450.0),
            Vec2F32::new(690.0, 470.0),
        ];
        let params = SQPnPParams::default();
        world[1].x = f32::NAN;
        assert!(matches!(
            solve_sqpnp(&world, &image, &k, None, &params),
            Err(PnPError::SvdFailed(_))
        ));
        world[1].x = 1.0;
        image[2].y = f32::INFINITY;
        assert!(matches!(
            solve_sqpnp(&world, &image, &k, None, &params),
            Err(PnPError::SvdFailed(_))
        ));
        // Finite but extreme values overflow the reprojection error.
        image[2].y = 1e30;
        assert!(matches!(
            solve_sqpnp(&world, &image, &k, None, &params),
            Err(PnPError::SvdFailed(_))
        ));
        image[2].y = 500.0;
        world[0] = Vec3AF32::new(3e38, 3e38, 3e38);
        assert!(matches!(
            solve_sqpnp(&world, &image, &k, None, &params),
            Err(PnPError::SvdFailed(_))
        ));
    }

    #[test]
    fn dispatched_through_solve_pnp_and_ransac() -> Result<(), Box<dyn std::error::Error>> {
        let k = k_matrix(800.0, 800.0, 640.0, 480.0);
        let mut rng = StdRng::seed_from_u64(8);
        let scene = Scene::random(&mut rng, k, 40, 1.0, 5.0, 0.0);

        let direct = solve(&scene)?;
        let dispatched = solve_pnp(
            &scene.world,
            &scene.image,
            &k,
            None,
            PnPMethod::SQPnPDefault,
        )?;
        assert_eq!(direct.rotation, dispatched.rotation);

        // 40 inliers plus 40 outliers.
        let mut world = scene.world.clone();
        let mut image = scene.image.clone();
        for _ in 0..40 {
            world.push(Vec3AF32::new(
                rng.random_range(-1.0..1.0) as f32,
                rng.random_range(-1.0..1.0) as f32,
                rng.random_range(-1.0..1.0) as f32,
            ));
            image.push(Vec2F32::new(
                rng.random_range(0.0..1280.0) as f32,
                rng.random_range(0.0..960.0) as f32,
            ));
        }
        let params = RansacParams {
            max_iterations: 2000,
            reproj_threshold_px: 2.0,
            random_seed: Some(1),
            ..Default::default()
        };
        let result = solve_pnp_ransac(&world, &image, &k, None, PnPMethod::SQPnPDefault, &params)?;
        let true_inliers = result.inliers.iter().filter(|&&i| i < 40).count();
        assert!(true_inliers >= 38, "found {true_inliers} of 40 inliers");
        assert!(rotation_error_deg(&result.pose.rotation, &scene.rotation) < 0.05);
        Ok(())
    }
}
