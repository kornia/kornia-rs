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
//! Let `r = vec(R)` be the rotation's rows stacked into a 9-vector and `v_i = (x_i, y_i, 1)` the
//! normalised (undistorted) image point `i`. SQPnP minimises the object-space error
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
//! when a lower-cost candidate was rejected by this depth check the search does not stop early.
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
//! # Distortion
//!
//! With `distortion = Some(..)` the image points are undistorted to normalised pinhole
//! coordinates first (`undistort_normalized_point_iter`), the pose is solved on those, and the
//! reported reprojection RMSE (and the optional LM refinement) use the distortion model.

use kornia_algebra::{Mat3AF32, Vec2F32, Vec3AF32, SO3F32};
use kornia_imgproc::calibration::{
    distortion::{undistort_normalized_point_iter, PolynomialDistortion, TermCriteria},
    CameraIntrinsic,
};
use nalgebra::{Matrix3, SMatrix, SVector, SymmetricEigen, Vector3};

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
/// The defaults follow the OpenCV implementation, except that the rank tolerance is relative to
/// the largest eigenvalue of `Ω` (see the module docs).
#[derive(Debug, Clone)]
pub struct SQPnPParams {
    /// An eigenvalue `λ` of `Ω` counts as zero (part of its null space) when
    /// `λ ≤ rank_tolerance · λ_max`.
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
    let centroid = points_world.iter().map(to_f64).sum::<Vector3<f64>>() / n as f64;
    let (omega, p_mat) = build_omega(points_world, points_image, &centroid, &sight)?;
    let best = minimize_on_so3(&omega, &p_mat, params)?;

    // t' is the translation for the centred points: R·(M − c) + t' = R·M + (t' − R·c).
    let rotation = rotation_from_vec(&best.r);
    let translation = p_mat * best.r - rotation * centroid;
    if !translation.iter().all(|v| v.is_finite()) {
        return Err(PnPError::SvdFailed(
            "SQPnP: non-finite translation".to_string(),
        ));
    }

    let rotation_f32 = Mat3AF32::from_cols(
        Vec3AF32::new(
            rotation[(0, 0)] as f32,
            rotation[(1, 0)] as f32,
            rotation[(2, 0)] as f32,
        ),
        Vec3AF32::new(
            rotation[(0, 1)] as f32,
            rotation[(1, 1)] as f32,
            rotation[(2, 1)] as f32,
        ),
        Vec3AF32::new(
            rotation[(0, 2)] as f32,
            rotation[(1, 2)] as f32,
            rotation[(2, 2)] as f32,
        ),
    );
    let translation_f32 = Vec3AF32::new(
        translation.x as f32,
        translation.y as f32,
        translation.z as f32,
    );

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
        num_iterations: Some(best.iterations),
        converged: Some(best.converged),
    })
}

/// Maps a world point to `f64`.
fn to_f64(p: &Vec3AF32) -> Vector3<f64> {
    Vector3::new(p.x as f64, p.y as f64, p.z as f64)
}

/// Maps pixels to lines of sight `(x, y, 1)` in normalised pinhole coordinates `K⁻¹·(u, v, 1)`,
/// undistorting them first when a distortion model is given.
struct LineOfSight<'a> {
    k_inv: Matrix3<f64>,
    intrinsic: CameraIntrinsic,
    distortion: Option<&'a PolynomialDistortion>,
}

impl<'a> LineOfSight<'a> {
    /// # Errors
    ///
    /// Returns [`PnPError::SvdFailed`] if `K` is singular.
    fn new(k: &Mat3AF32, distortion: Option<&'a PolynomialDistortion>) -> Result<Self, PnPError> {
        let k64 = Matrix3::from_column_slice(&k.0.to_cols_array().map(|v| v as f64));
        let k_inv = k64
            .try_inverse()
            .ok_or_else(|| PnPError::SvdFailed("SQPnP: camera matrix K is singular".to_string()))?;
        let intrinsic = CameraIntrinsic {
            fx: k64[(0, 0)],
            fy: k64[(1, 1)],
            cx: k64[(0, 2)],
            cy: k64[(1, 2)],
        };
        Ok(Self {
            k_inv,
            intrinsic,
            distortion,
        })
    }

    /// # Errors
    ///
    /// Returns [`PnPError::SvdFailed`] if the pixel maps to a non-finite coordinate.
    fn of(&self, uv: &Vec2F32) -> Result<Vector3<f64>, PnPError> {
        let (u, v) = (uv.x as f64, uv.y as f64);
        let h = self.k_inv * Vector3::new(u, v, 1.0);
        let (mut x, mut y) = (h.x / h.z, h.y / h.z);
        if let Some(d) = self.distortion {
            let criteria = TermCriteria {
                max_iter: 20,
                eps: 1e-8,
            };
            let undistorted =
                undistort_normalized_point_iter(x, y, u, v, &self.intrinsic, d, criteria);
            (x, y) = (undistorted.x, undistorted.y);
        }
        if x.is_finite() && y.is_finite() {
            Ok(Vector3::new(x, y, 1.0))
        } else {
            Err(PnPError::SvdFailed(
                "SQPnP: image point maps to a non-finite normalised coordinate".to_string(),
            ))
        }
    }
}

/// Builds `Ω` and `P` such that `E(r) = rᵀ Ω r` and the optimal translation is `t = P·r`.
///
/// With `A_i = I₃ ⊗ M_iᵀ` (so that `R·M_i = A_i·r`), `S = Σ Q_i` and `B = Σ Q_i A_i`:
/// `P = −S⁻¹ B` and `Ω = Σ A_iᵀ Q_i A_i − Bᵀ S⁻¹ B = Σ A_iᵀ Q_i A_i + Bᵀ P`.
///
/// # Errors
///
/// Returns [`PnPError::SvdFailed`] if `S` is singular, i.e. all lines of sight are parallel
/// (every image point is the same), or if an image point maps to a non-finite coordinate.
fn build_omega(
    points_world: &[Vec3AF32],
    points_image: &[Vec2F32],
    centroid: &Vector3<f64>,
    sight: &LineOfSight,
) -> Result<(Mat9, Mat39), PnPError> {
    // Σ A_iᵀ Q_i A_i is the Kronecker sum Σ Q_i ⊗ (M_i M_iᵀ): block (a, c) is Σ Q_i[a][c]·M_i M_iᵀ.
    // Only the upper blocks are accumulated here; the lower ones are mirrored below.
    let mut omega = Mat9::zeros();
    let mut s = Matrix3::<f64>::zeros();
    let mut b = Mat39::zeros();

    for (p, uv) in points_world.iter().zip(points_image) {
        let m = to_f64(p) - centroid;
        let v = sight.of(uv)?;
        let q = Matrix3::identity() - (v * v.transpose()) / v.norm_squared();
        let mmt = m * m.transpose();

        s += q;
        for a in 0..3 {
            for c in a..3 {
                let mut block = omega.fixed_view_mut::<3, 3>(3 * a, 3 * c);
                block += mmt * q[(a, c)];
            }
            // B[p, 3a + j] += Q[p][a] · M[j]
            for p in 0..3 {
                let qpa = q[(p, a)];
                for j in 0..3 {
                    b[(p, 3 * a + j)] += qpa * m[j];
                }
            }
        }
    }

    let s_inv = s.try_inverse().ok_or_else(|| {
        PnPError::SvdFailed(
            "SQPnP: degenerate correspondences, all lines of sight are parallel".to_string(),
        )
    })?;
    let p_mat = -(s_inv * b);

    // Q is symmetric and so is M Mᵀ, so block (c, a) equals block (a, c).
    for a in 0..3 {
        for c in a + 1..3 {
            let upper = omega.fixed_view::<3, 3>(3 * a, 3 * c).into_owned();
            omega.fixed_view_mut::<3, 3>(3 * c, 3 * a).copy_from(&upper);
        }
    }
    omega += b.transpose() * p_mat;
    // Remove the round-off asymmetry so the symmetric eigensolver sees an exactly symmetric matrix.
    let omega = (omega + omega.transpose()) * 0.5;

    if omega.iter().all(|v| v.is_finite()) {
        Ok((omega, p_mat))
    } else {
        Err(PnPError::SvdFailed(
            "SQPnP: non-finite entries in Omega".to_string(),
        ))
    }
}

/// A rotation candidate found by SQPnP.
struct Candidate {
    /// `vec(R)`, rows stacked.
    r: Vec9,
    /// `rᵀ Ω r`.
    cost: f64,
    iterations: usize,
    converged: bool,
}

/// Minimises `rᵀ Ω r` over SO(3), keeping only rotations that put the centroid in front of the
/// camera.
///
/// # Errors
///
/// Returns [`PnPError::SvdFailed`] if `Ω` is all zeros, and [`PnPError::CheiralityCheckFailed`]
/// if no candidate passes the depth check.
fn minimize_on_so3(
    omega: &Mat9,
    p_mat: &Mat39,
    params: &SQPnPParams,
) -> Result<Candidate, PnPError> {
    let eig = SymmetricEigen::try_new(*omega, f64::EPSILON, MAX_DECOMPOSITION_ITERATIONS)
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
        omega,
        p_mat,
        best: None,
        lowest_rejected_cost: f64::INFINITY,
    };
    let try_eigenvector = |k: usize, search: &mut Search| {
        let x: Vec9 = eig.eigenvectors.column(order[k]) * SQRT3;
        if orthogonality_squared_error(&x) < params.orthogonality_squared_error_threshold {
            // Already close to a rotation up to sign; project it so the result is orthonormal.
            if let Some(r) = nearest_rotation(&with_positive_det(x)) {
                search.consider(r, 0, true);
            }
        } else {
            for start in [x, -x] {
                if let Some(r0) = nearest_rotation(&start) {
                    let (r, iterations, converged) = sqp(omega, r0, params);
                    search.consider(r, iterations, converged);
                }
            }
        }
    };

    for k in 0..num_null {
        try_eigenvector(k, &mut search);
    }
    // The paper's bound: for k beyond the null space, a start from e_k can only improve on the
    // lowest cost over SO(3) if that cost is above 3·λ_k. The bound is on the unconstrained
    // minimum, so it only ends the search when that minimum passed the depth check. For planar
    // points the true pose has a twin of equal cost behind the camera (rotated by π about the
    // plane normal, translation negated); when SQP lands on the twin, keep searching.
    for k in num_null..9 {
        let best_cost = search.best.as_ref().map_or(f64::INFINITY, |c| c.cost);
        if best_cost <= 3.0 * eigenvalue(k) && best_cost <= search.lowest_rejected_cost {
            break;
        }
        try_eigenvector(k, &mut search);
    }

    search.best.ok_or(PnPError::CheiralityCheckFailed)
}

/// State of the search over the eigenvectors of `Ω`.
struct Search<'a> {
    omega: &'a Mat9,
    p_mat: &'a Mat39,
    /// Lowest-cost rotation that puts the centroid in front of the camera.
    best: Option<Candidate>,
    /// Lowest `rᵀ Ω r` among rotations rejected by the depth check.
    lowest_rejected_cost: f64,
}

impl Search<'_> {
    /// Keeps `r` as the best candidate if it is finite, puts the centroid in front of the camera
    /// and has the lowest cost so far.
    fn consider(&mut self, r: Vec9, iterations: usize, converged: bool) {
        if !r.iter().all(|v| v.is_finite()) {
            return;
        }
        let cost = (r.transpose() * self.omega * r)[0];
        // World points are centred, so the centroid's camera-frame depth is t'_z = (P·r)_z.
        let centroid_depth = (self.p_mat.row(2) * r)[0];
        if centroid_depth <= 0.0 {
            self.lowest_rejected_cost = self.lowest_rejected_cost.min(cost);
            return;
        }
        if self.best.as_ref().is_none_or(|b| cost < b.cost) {
            self.best = Some(Candidate {
                r,
                cost,
                iterations,
                converged,
            });
        }
    }
}

/// Runs SQP from the rotation `r0` and returns the nearest rotation to the result, the number of
/// iterations and whether the step size converged.
///
/// Each iteration solves the KKT system of `min (r + δ)ᵀ Ω (r + δ)` subject to the linearised
/// orthonormality constraints `h(r) + H(r)·δ = 0`.
fn sqp(omega: &Mat9, r0: Vec9, params: &SQPnPParams) -> (Vec9, usize, bool) {
    let mut r = r0;
    let mut iterations = 0;
    let mut converged = false;
    let mut kkt = Mat15::zeros();
    kkt.fixed_view_mut::<9, 9>(0, 0).copy_from(omega);

    while iterations < params.sqp_max_iterations {
        iterations += 1;
        let (h, jacobian) = orthonormality_constraints(&r);
        kkt.fixed_view_mut::<6, 9>(9, 0).copy_from(&jacobian);
        kkt.fixed_view_mut::<9, 6>(0, 9)
            .copy_from(&jacobian.transpose());

        let mut rhs = Vec15::zeros();
        rhs.fixed_rows_mut::<9>(0).copy_from(&(-(omega * r)));
        rhs.fixed_rows_mut::<6>(9).copy_from(&(-h));

        // LU is enough away from degeneracies; fall back to a least-squares SVD solve when the
        // KKT matrix is (numerically) singular.
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
        let delta = solution.fixed_rows::<9>(0).into_owned();
        r += delta;
        if delta.norm_squared() < params.sqp_squared_tolerance {
            converged = true;
            break;
        }
    }

    (nearest_rotation(&r).unwrap_or(r), iterations, converged)
}

/// The six orthonormality constraints of the rows `r₁, r₂, r₃` of `R` and their Jacobian:
/// `h = (‖r₁‖² − 1, ‖r₂‖² − 1, ‖r₃‖² − 1, r₁·r₂, r₂·r₃, r₁·r₃)`.
fn orthonormality_constraints(r: &Vec9) -> (SVector<f64, 6>, Mat69) {
    let row = |i: usize| Vector3::new(r[3 * i], r[3 * i + 1], r[3 * i + 2]);
    let (r1, r2, r3) = (row(0), row(1), row(2));

    let h = SVector::<f64, 6>::from([
        r1.norm_squared() - 1.0,
        r2.norm_squared() - 1.0,
        r3.norm_squared() - 1.0,
        r1.dot(&r2),
        r2.dot(&r3),
        r1.dot(&r3),
    ]);

    let mut jacobian = Mat69::zeros();
    for j in 0..3 {
        jacobian[(0, j)] = 2.0 * r1[j];
        jacobian[(1, 3 + j)] = 2.0 * r2[j];
        jacobian[(2, 6 + j)] = 2.0 * r3[j];
        // d(r1·r2)
        jacobian[(3, j)] = r2[j];
        jacobian[(3, 3 + j)] = r1[j];
        // d(r2·r3)
        jacobian[(4, 3 + j)] = r3[j];
        jacobian[(4, 6 + j)] = r2[j];
        // d(r1·r3)
        jacobian[(5, j)] = r3[j];
        jacobian[(5, 6 + j)] = r1[j];
    }
    (h, jacobian)
}

/// `vec(R)` (rows stacked) back to a 3×3 matrix.
fn rotation_from_vec(r: &Vec9) -> Matrix3<f64> {
    Matrix3::new(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8])
}

/// `x` or `−x`, whichever has `det(mat(x)) ≥ 0`.
///
/// An eigenvector of `Ω` is only defined up to sign, and for an orthogonal `mat(x)` negating it
/// flips `det` between `−1` (a reflection) and `+1` (a rotation).
fn with_positive_det(x: Vec9) -> Vec9 {
    if rotation_from_vec(&x).determinant() < 0.0 {
        -x
    } else {
        x
    }
}

/// Squared Frobenius norm of `X·Xᵀ − I` for `X = mat(x)`.
fn orthogonality_squared_error(x: &Vec9) -> f64 {
    let m = rotation_from_vec(x);
    (m * m.transpose() - Matrix3::identity()).norm_squared()
}

/// The rotation closest to `mat(x)` in the Frobenius norm, `U·diag(1, 1, det(U·Vᵀ))·Vᵀ`.
fn nearest_rotation(x: &Vec9) -> Option<Vec9> {
    if !x.iter().all(|v| v.is_finite()) {
        return None;
    }
    let svd =
        rotation_from_vec(x).try_svd(true, true, f64::EPSILON, MAX_DECOMPOSITION_ITERATIONS)?;
    let (u, v_t) = (svd.u?, svd.v_t?);
    let mut d = Matrix3::<f64>::identity();
    if (u * v_t).determinant() < 0.0 {
        d[(2, 2)] = -1.0;
    }
    let rot = u * d * v_t;
    let r = Vec9::from_row_slice(&[
        rot[(0, 0)],
        rot[(0, 1)],
        rot[(0, 2)],
        rot[(1, 0)],
        rot[(1, 1)],
        rot[(1, 2)],
        rot[(2, 0)],
        rot[(2, 1)],
        rot[(2, 2)],
    ]);
    r.iter().all(|v| v.is_finite()).then_some(r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pnp::{solve_pnp, solve_pnp_ransac, EPnPParams, PnPMethod, RansacParams};
    use kornia_imgproc::calibration::distortion::distort_point_polynomial;

    /// Small deterministic generator so the tests need no extra dependency.
    struct Lcg(u64);

    impl Lcg {
        fn uniform(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }

        fn range(&mut self, lo: f64, hi: f64) -> f64 {
            lo + (hi - lo) * self.uniform()
        }

        fn gauss(&mut self) -> f64 {
            let (u1, u2) = (self.uniform().max(1e-12), self.uniform());
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        }
    }

    /// A ground-truth pose and the correspondences it generates.
    struct Scene {
        rotation: Matrix3<f64>,
        translation: Vector3<f64>,
        world: Vec<Vec3AF32>,
        image: Vec<Vec2F32>,
    }

    fn k_matrix(f: f32, cx: f32, cy: f32) -> Mat3AF32 {
        Mat3AF32::from_cols_array(&[f, 0.0, 0.0, 0.0, f, 0.0, cx, cy, 1.0])
    }

    /// Rotation by `angle` radians about the (normalised) `axis`.
    fn axis_angle(axis: Vector3<f64>, angle: f64) -> Matrix3<f64> {
        nalgebra::Rotation3::from_axis_angle(&nalgebra::Unit::new_normalize(axis), angle)
            .into_inner()
    }

    fn random_rotation(rng: &mut Lcg) -> Matrix3<f64> {
        let axis = Vector3::new(rng.gauss(), rng.gauss(), rng.gauss());
        axis_angle(axis, rng.range(0.0, std::f64::consts::PI))
    }

    /// Projects `world` with the pose and pinhole `(f, cx, cy)`, adding `noise_px` Gaussian noise.
    fn make_scene(
        rng: &mut Lcg,
        world: Vec<Vector3<f64>>,
        rotation: Matrix3<f64>,
        translation: Vector3<f64>,
        (f, cx, cy): (f64, f64, f64),
        noise_px: f64,
    ) -> Scene {
        let image = world
            .iter()
            .map(|m| {
                let c = rotation * m + translation;
                Vec2F32::new(
                    (f * c.x / c.z + cx + noise_px * rng.gauss()) as f32,
                    (f * c.y / c.z + cy + noise_px * rng.gauss()) as f32,
                )
            })
            .collect();
        let world = world
            .iter()
            .map(|m| Vec3AF32::new(m.x as f32, m.y as f32, m.z as f32))
            .collect();
        Scene {
            rotation,
            translation,
            world,
            image,
        }
    }

    /// `n` random points in a `size`-wide box around the origin, seen from `depth` metres.
    fn random_scene(rng: &mut Lcg, n: usize, size: f64, depth: f64, noise_px: f64) -> Scene {
        let world = (0..n)
            .map(|_| {
                Vector3::new(
                    rng.range(-size, size),
                    rng.range(-size, size),
                    rng.range(-size, size),
                )
            })
            .collect();
        let rotation = random_rotation(rng);
        let translation = Vector3::new(rng.range(-0.5, 0.5), rng.range(-0.5, 0.5), depth);
        make_scene(
            rng,
            world,
            rotation,
            translation,
            (800.0, 640.0, 480.0),
            noise_px,
        )
    }

    /// Angle between the estimated and true rotation. `2·asin(‖R₁ − R₂‖_F / 2√2)` stays
    /// accurate for tiny angles, unlike `acos((tr(R₁ᵀR₂) − 1) / 2)` on f32 matrices.
    fn rotation_error_deg(pose: &PnPResult, truth: &Matrix3<f64>) -> f64 {
        let c = pose.rotation.0.to_cols_array().map(|v| v as f64);
        let estimate = Matrix3::from_column_slice(&c);
        let chord = (estimate - truth).norm() / (2.0 * std::f64::consts::SQRT_2);
        (2.0 * chord.clamp(0.0, 1.0).asin()).to_degrees()
    }

    fn translation_error(pose: &PnPResult, truth: &Vector3<f64>) -> f64 {
        let t = pose.translation;
        (Vector3::new(t.x as f64, t.y as f64, t.z as f64) - truth).norm()
    }

    fn solve(scene: &Scene, k: &Mat3AF32) -> Result<PnPResult, PnPError> {
        solve_sqpnp(&scene.world, &scene.image, k, None, &SQPnPParams::default())
    }

    #[test]
    fn noise_free_poses_are_exact() -> Result<(), PnPError> {
        let k = k_matrix(800.0, 640.0, 480.0);
        let mut rng = Lcg(1);
        for n in [4, 5, 6, 10, 50, 200] {
            for _ in 0..10 {
                let depth = rng.range(3.0, 10.0);
                let scene = random_scene(&mut rng, n, 1.0, depth, 0.0);
                let pose = solve(&scene, &k)?;
                let rot_err = rotation_error_deg(&pose, &scene.rotation);
                let t_err = translation_error(&pose, &scene.translation) / scene.translation.norm();
                assert!(rot_err < 2e-3, "n={n}: rotation error {rot_err} deg");
                assert!(t_err < 1e-4, "n={n}: relative translation error {t_err}");
                assert!(
                    pose.reproj_rmse.is_some_and(|e| e < 1e-2),
                    "n={n}: {pose:?}"
                );
                if n >= 6 {
                    // The null vector of Ω is already ±vec(R): the sign fix alone must give the
                    // pose, without falling back to SQP from other eigenvectors.
                    assert_eq!(pose.num_iterations, Some(0), "n={n}");
                }
            }
        }
        Ok(())
    }

    /// P3P has up to four exact solutions; SQPnP must return one that reprojects exactly.
    #[test]
    fn three_points_reproject_exactly() -> Result<(), PnPError> {
        let k = k_matrix(800.0, 640.0, 480.0);
        let mut rng = Lcg(2);
        for _ in 0..20 {
            let scene = random_scene(&mut rng, 3, 1.0, 5.0, 0.0);
            let pose = solve(&scene, &k)?;
            assert!(pose.reproj_rmse.is_some_and(|e| e < 1e-2), "{pose:?}");
        }
        Ok(())
    }

    /// Coplanar points give Ω extra null vectors; the pose must still be exact.
    #[test]
    fn planar_points_are_exact() -> Result<(), PnPError> {
        let k = k_matrix(800.0, 640.0, 480.0);
        let mut rng = Lcg(3);
        for n in [4, 6, 20] {
            for _ in 0..10 {
                let world = (0..n)
                    .map(|_| Vector3::new(rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), 0.0))
                    .collect();
                let rotation = axis_angle(
                    Vector3::new(rng.gauss(), rng.gauss(), rng.gauss()),
                    rng.range(0.0, 1.2),
                );
                let translation = Vector3::new(0.1, -0.2, 5.0);
                let scene = make_scene(
                    &mut rng,
                    world,
                    rotation,
                    translation,
                    (800.0, 640.0, 480.0),
                    0.0,
                );
                let pose = solve(&scene, &k)?;
                let rot_err = rotation_error_deg(&pose, &scene.rotation);
                assert!(rot_err < 2e-3, "planar n={n}: rotation error {rot_err} deg");
                assert!(
                    translation_error(&pose, &scene.translation) < 1e-3,
                    "planar n={n}"
                );
            }
        }
        Ok(())
    }

    /// A 20 cm object 6 m away (fx = 600): the regime in which EPnP's control points are poorly
    /// conditioned (#1075). SQPnP has no control points and stays exact.
    #[test]
    fn small_object_far_away_is_exact() -> Result<(), PnPError> {
        let k = k_matrix(600.0, 320.0, 240.0);
        let mut rng = Lcg(4);
        for _ in 0..10 {
            let world = (0..50)
                .map(|_| {
                    Vector3::new(
                        rng.range(-0.1, 0.1),
                        rng.range(-0.1, 0.1),
                        rng.range(-0.1, 0.1),
                    )
                })
                .collect();
            let rotation = random_rotation(&mut rng);
            let translation = Vector3::new(0.1, -0.05, 6.0);
            let scene = make_scene(
                &mut rng,
                world,
                rotation,
                translation,
                (600.0, 320.0, 240.0),
                0.0,
            );
            let pose = solve(&scene, &k)?;
            assert!(
                rotation_error_deg(&pose, &scene.rotation) < 0.01,
                "{pose:?}"
            );
            assert!(
                translation_error(&pose, &scene.translation) < 1e-3,
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
        let k = k_matrix(800.0, 640.0, 480.0);
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
        let k = k_matrix(800.0, 640.0, 480.0);
        let mut rng = Lcg(5);
        for _ in 0..20 {
            let scene = random_scene(&mut rng, 100, 1.0, 6.0, 1.0);
            let pose = solve(&scene, &k)?;
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
            assert!(rotation_error_deg(&pose, &scene.rotation) < 0.5);

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
        let k = k_matrix(800.0, 640.0, 480.0);
        let mut rng = Lcg(6);
        let scene = random_scene(&mut rng, 30, 1.0, 5.0, 0.5);
        let pose = solve(&scene, &k)?;
        for (scale, shift) in [(1e-3, 0.0), (1e3, 0.0), (1.0, 100.0)] {
            // M' = s·M + d and t' = s·t − R·d project to the same pixels.
            let d = Vector3::new(shift, -shift, shift);
            let world: Vec<Vec3AF32> = scene
                .world
                .iter()
                .map(|p| {
                    Vec3AF32::new(
                        (p.x as f64 * scale + d.x) as f32,
                        (p.y as f64 * scale + d.y) as f32,
                        (p.z as f64 * scale + d.z) as f32,
                    )
                })
                .collect();
            let moved = solve_sqpnp(&world, &scene.image, &k, None, &SQPnPParams::default())?;
            let c = pose.rotation.0.to_cols_array().map(|v| v as f64);
            let delta = rotation_error_deg(&moved, &Matrix3::from_column_slice(&c));
            assert!(
                delta < 2e-3,
                "scale {scale}, shift {shift}: rotation changed by {delta} deg"
            );
        }
        Ok(())
    }

    #[test]
    fn distortion_is_undistorted_before_solving() -> Result<(), PnPError> {
        let k = k_matrix(800.0, 640.0, 480.0);
        let intrinsic = CameraIntrinsic {
            fx: 800.0,
            fy: 800.0,
            cx: 640.0,
            cy: 480.0,
        };
        let distortion = PolynomialDistortion {
            k1: -0.25,
            k2: 0.08,
            k3: 0.0,
            k4: 0.0,
            k5: 0.0,
            k6: 0.0,
            p1: 1e-3,
            p2: -5e-4,
        };
        let mut rng = Lcg(7);
        let mut scene = random_scene(&mut rng, 60, 1.0, 4.0, 0.0);
        let truth = scene.rotation;
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
        assert!(rotation_error_deg(&with_model, &truth) < 2e-3);
        assert!(
            with_model.reproj_rmse.is_some_and(|e| e < 1e-2),
            "{with_model:?}"
        );

        // Ignoring the distortion leaves pixel-sized residuals.
        let without_model = solve(&scene, &k)?;
        assert!(
            without_model.reproj_rmse.is_some_and(|e| e > 1.0),
            "{without_model:?}"
        );
        Ok(())
    }

    #[test]
    fn rejects_invalid_input_without_panicking() {
        let k = k_matrix(800.0, 640.0, 480.0);
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
        assert!(solve_sqpnp(&world, &same_pixel, &k, None, &params).is_err());
        // Singular camera matrix.
        let singular = Mat3AF32::from_cols_array(&[0.0; 9]);
        let image = [
            Vec2F32::new(600.0, 400.0),
            Vec2F32::new(700.0, 420.0),
            Vec2F32::new(650.0, 500.0),
            Vec2F32::new(620.0, 450.0),
        ];
        assert!(solve_sqpnp(&world, &image, &singular, None, &params).is_err());
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
        let k = k_matrix(800.0, 640.0, 480.0);
        let lm = SQPnPParams {
            refine_lm: Some(LMRefineParams::default()),
            ..Default::default()
        };
        let mut rng = Lcg(5);
        for _ in 0..10 {
            // Points as close as ~0.3 m with 3 px noise.
            let scene = random_scene(&mut rng, 50, 1.0, 2.0, 3.0);
            let object_space_cost = |pose: &PnPResult| -> f64 {
                let c = pose.rotation.to_cols_array().map(|v| v as f64);
                let r = Matrix3::from_column_slice(&c);
                let t = Vector3::new(
                    pose.translation.x as f64,
                    pose.translation.y as f64,
                    pose.translation.z as f64,
                );
                scene
                    .world
                    .iter()
                    .zip(&scene.image)
                    .map(|(m, u)| {
                        let v = Vector3::new(
                            (u.x as f64 - 640.0) / 800.0,
                            (u.y as f64 - 480.0) / 800.0,
                            1.0,
                        );
                        let q = Matrix3::identity() - v * v.transpose() / v.norm_squared();
                        let m = Vector3::new(m.x as f64, m.y as f64, m.z as f64);
                        (q * (r * m + t)).norm_squared()
                    })
                    .sum()
            };
            let plain = solve(&scene, &k)?;
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
    /// result must match running SQP from every eigenvector of Ω. Without the depth-aware stop,
    /// one of these scenes ends 69% above the best start.
    #[test]
    fn noisy_planar_search_matches_every_start() -> Result<(), PnPError> {
        let k = k_matrix(800.0, 640.0, 480.0);
        let params = SQPnPParams::default();
        let mut rng = Lcg(84);
        for _ in 0..200 {
            let world = (0..6)
                .map(|_| Vector3::new(rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), 0.0))
                .collect();
            let rotation = axis_angle(
                Vector3::new(rng.gauss(), rng.gauss(), rng.gauss()),
                rng.range(0.0, 1.2),
            );
            let translation = Vector3::new(0.1, -0.2, 5.0);
            let scene = make_scene(
                &mut rng,
                world,
                rotation,
                translation,
                (800.0, 640.0, 480.0),
                2.0,
            );

            let centroid = scene.world.iter().map(to_f64).sum::<Vector3<f64>>() / 6.0;
            let sight = LineOfSight::new(&k, None)?;
            let (omega, p_mat) = build_omega(&scene.world, &scene.image, &centroid, &sight)?;
            let found = minimize_on_so3(&omega, &p_mat, &params)?;

            let eig = SymmetricEigen::new(omega);
            let mut every = f64::INFINITY;
            for e in eig.eigenvectors.column_iter() {
                let x: Vec9 = e * SQRT3;
                for start in [x, -x] {
                    if let Some(r0) = nearest_rotation(&start) {
                        let (r, _, _) = sqp(&omega, r0, &params);
                        if (p_mat.row(2) * r)[0] > 0.0 {
                            every = every.min((r.transpose() * omega * r)[0]);
                        }
                    }
                }
            }
            // The guarantee assumes SQP converged. On these ill-conditioned scenes it sometimes
            // stops at its iteration cap instead, and then the result can fall slightly short.
            if found.converged {
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
        let k = k_matrix(800.0, 640.0, 480.0);
        let mut rng = Lcg(11);
        for _ in 0..20 {
            let scene = random_scene(&mut rng, 100, 1.0, 6.0, 0.001);
            let pose = solve(&scene, &k)?;
            assert_eq!(pose.num_iterations, Some(0), "fast path not taken");
            let c = pose.rotation.0.to_cols_array().map(|v| v as f64);
            let r = Matrix3::from_column_slice(&c);
            let error = (r.transpose() * r - Matrix3::identity()).norm();
            assert!(error < 1e-6, "‖RᵀR − I‖ = {error}");
        }
        Ok(())
    }

    #[test]
    fn eigenvector_sign_is_fixed_to_a_rotation() {
        let rotation = nalgebra::Rotation3::from_euler_angles(0.3, -0.7, 1.1);
        let r = Vec9::from_row_slice(rotation.matrix().transpose().as_slice());
        assert_eq!(with_positive_det(r), r);
        assert_eq!(with_positive_det(-r), r);
    }

    #[test]
    fn non_finite_values_terminate() {
        // nalgebra's unbounded SVD never converges on NaN, so before the iteration caps these
        // calls hung forever instead of returning.
        let nan = Vec9::from_element(f64::NAN);
        assert!(nearest_rotation(&nan).is_none());
        let (_, iterations, converged) = sqp(&Mat9::identity(), nan, &SQPnPParams::default());
        assert!(iterations <= SQPnPParams::default().sqp_max_iterations && !converged);

        let k = k_matrix(800.0, 640.0, 480.0);
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
        assert!(solve_sqpnp(&world, &image, &k, None, &params).is_err());
        world[1].x = 1.0;
        image[2].y = f32::INFINITY;
        assert!(solve_sqpnp(&world, &image, &k, None, &params).is_err());
        // Finite but extreme values overflow the reprojection error.
        image[2].y = 1e30;
        assert!(solve_sqpnp(&world, &image, &k, None, &params).is_err());
        image[2].y = 500.0;
        world[0] = Vec3AF32::new(3e38, 3e38, 3e38);
        assert!(solve_sqpnp(&world, &image, &k, None, &params).is_err());
    }

    #[test]
    fn dispatched_through_solve_pnp_and_ransac() -> Result<(), Box<dyn std::error::Error>> {
        let k = k_matrix(800.0, 640.0, 480.0);
        let mut rng = Lcg(8);
        let scene = random_scene(&mut rng, 40, 1.0, 5.0, 0.0);

        let direct = solve(&scene, &k)?;
        let dispatched = solve_pnp(
            &scene.world,
            &scene.image,
            &k,
            None,
            PnPMethod::SQPnPDefault,
        )?;
        assert_eq!(
            direct.rotation.0.to_cols_array(),
            dispatched.rotation.0.to_cols_array()
        );

        // 40 inliers plus 40 outliers.
        let mut world = scene.world.clone();
        let mut image = scene.image.clone();
        for _ in 0..40 {
            world.push(Vec3AF32::new(
                rng.range(-1.0, 1.0) as f32,
                rng.range(-1.0, 1.0) as f32,
                rng.range(-1.0, 1.0) as f32,
            ));
            image.push(Vec2F32::new(
                rng.range(0.0, 1280.0) as f32,
                rng.range(0.0, 960.0) as f32,
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
        assert!(rotation_error_deg(&result.pose, &scene.rotation) < 0.05);
        Ok(())
    }
}
