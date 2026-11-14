//! # SO(3) — The Special Orthogonal Group in 3D
//!
//! SO(3) is the group of 3D rotations: 3×3 orthogonal matrices with determinant +1.
//! It has 3 degrees of freedom and is topologically **RP³** (real projective 3-space).
//!
//! ## Internal representation: unit quaternions (SU(2))
//!
//! [`SO3F32`] stores rotations as unit quaternions. Crucially, a unit quaternion is an
//! element of **SU(2)**, which is the **double cover** of SO(3) — not SO(3) itself.
//! The naming `SO3F32` reflects what the type *represents* (a 3D rotation), but the
//! internal storage carries strictly more structure than SO(3).
//!
//! The double cover means:
//! - Every rotation R ∈ SO(3) corresponds to **two** quaternions: `q` and `-q`.
//! - SU(2) is topologically S³ (the 3-sphere, simply connected).
//! - SO(3) is topologically RP³ = S³/{±1} (antipodal points identified, not simply connected).
//! - A 360° rotation takes `q → -q` in SU(2) (different point). A 720° rotation
//!   takes `q → -q → q` (back to start). Objects sensitive to this are **spinors**.
//!
//! ## The exp/log maps
//!
//! The Lie algebra **so(3)** is the space of 3×3 skew-symmetric matrices, isomorphic
//! to R³ via the hat operator. A vector `v ∈ R³` represents an axis-angle rotation:
//! the direction is the rotation axis, the magnitude is the angle in radians.
//!
//! - `exp(v)`: axis-angle vector → unit quaternion. This is `q = (cos(θ/2), sin(θ/2) · axis)`.
//! - `log()`: unit quaternion → axis-angle vector. Inverse of exp.
//!
//! Note the **half-angle**: this is the SU(2) → SO(3) covering map in action. A full
//! rotation of θ = 2π gives `q = (cos(π), sin(π) · axis) = (-1, 0, 0, 0) = -I`,
//! which is the antipode of the identity in SU(2), but maps to the identity in SO(3).
//!
//! ## Jacobians
//!
//! The left and right Jacobians describe how the exponential map distorts
//! infinitesimal tangent vectors. They are essential for optimization on the
//! rotation manifold (Levenberg-Marquardt, Gauss-Newton). They use Taylor series
//! approximations at small angles (θ² < 1e-2 in f32, θ < 1e-10 in f64) to maintain
//! numerical stability and prevent catastrophic cancellation.

use crate::{
    param::{Param, ParamError},
    Mat3AF32, Mat3F64, Mat4F32, Mat4F64, QuatF32, QuatF64, Vec3AF32, Vec3F64,
};
use rand::RngExt;
const SMALL_ANGLE_EPSILON: f32 = 1.0e-8;
const SMALL_ANGLE_EPSILON_F64: f64 = 1.0e-10;
/// Threshold on θ² below which Taylor expansion is used for f32 Jacobians.
/// In f32, `1.0 - cos(θ)` cancels to 0.0 for θ < 1e-4. Below θ² = 1e-2 (θ < 0.1 rad),
/// 4th-order Taylor series avoids cancellation with truncation error (< 1e-9) well below f32::EPSILON.
const JACOBIAN_TAYLOR_THRESHOLD_F32: f32 = 1.0e-2;

/// A 3D rotation, stored as a unit quaternion.
///
/// Internally an element of SU(2) ≅ S³, projected to SO(3) ≅ RP³ via the 2:1
/// covering map. See the [module-level documentation](self) for details.
///
/// # Important
///
/// - `q` and `-q` represent the **same rotation**. When comparing SO3F32 values,
///   both signs must be checked.
/// - [`from_matrix`](SO3F32::from_matrix) chooses one of the two possible quaternions.
///   There is no globally continuous way to make this choice — discontinuities
///   (quaternion sign flips) are unavoidable over the full rotation group.
/// - After repeated multiplications, call `q.normalize()` to prevent drift off S³.
///
/// Note regarding `PartialEq`:
///   Quaternions form a double cover for SO3, meaning `q` and `-q` represent the
///   same rotation. However, `PartialEq` performs an exact, member-wise comparison
///   and will return `false` for `q` and `-q`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SO3F32 {
    pub q: QuatF32,
}

impl SO3F32 {
    pub const IDENTITY: Self = Self {
        q: QuatF32::IDENTITY,
    };

    /// Create a new SO3F32 from a quaternion.
    /// NOTE: quaternion should be normalized
    #[inline]
    pub fn new(quat: QuatF32) -> Self {
        Self { q: quat }
    }

    pub fn from_array(arr: [f32; 4]) -> Self {
        Self {
            q: QuatF32::from_array(arr),
        }
    }

    pub fn to_array(&self) -> [f32; 4] {
        self.q.to_array()
    }

    /// Create a new SO3F32 from a quaternion.
    /// NOTE: quaternion should be normalized
    #[inline]
    pub fn from_quaternion(quat: QuatF32) -> Self {
        Self::new(quat)
    }

    pub fn from_matrix(mat: &Mat3AF32) -> Self {
        Self {
            q: QuatF32::from_mat3a(mat),
        }
    }

    pub fn from_matrix4(mat: &Mat4F32) -> Self {
        Self {
            q: QuatF32::from_mat4(mat),
        }
    }

    pub fn from_random() -> Self {
        let mut rng = rand::rng();

        let r1: f32 = rng.random();
        let r2: f32 = rng.random();
        let r3: f32 = rng.random();

        // Correct uniform random quaternion generation (Shoemake method)
        let one_minus_r1_sqrt = (1.0 - r1).sqrt();
        let r1_sqrt = r1.sqrt();

        let w = one_minus_r1_sqrt * (2.0 * std::f32::consts::PI * r2).cos();
        let x = one_minus_r1_sqrt * (2.0 * std::f32::consts::PI * r2).sin();
        let y = r1_sqrt * (2.0 * std::f32::consts::PI * r3).cos();
        let z = r1_sqrt * (2.0 * std::f32::consts::PI * r3).sin();

        Self {
            q: QuatF32::from_xyzw(x, y, z, w).normalize(),
        }
    }

    #[inline]
    pub fn rplus(&self, tau: Vec3AF32) -> Self {
        *self * SO3F32::exp(tau)
    }

    #[inline]
    pub fn rminus(&self, other: &Self) -> Vec3AF32 {
        (self.inverse() * *other).log()
    }

    #[inline]
    pub fn lplus(tau: Vec3AF32, x: &Self) -> Self {
        SO3F32::exp(tau) * *x
    }

    #[inline]
    pub fn lminus(y: &Self, x: &Self) -> Vec3AF32 {
        (*y * x.inverse()).log()
    }

    pub fn matrix(&self) -> Mat3AF32 {
        Mat3AF32::from_quat(self.q)
    }

    pub fn adjoint(&self) -> Mat3AF32 {
        self.matrix()
    }

    pub fn inverse(&self) -> Self {
        Self {
            q: self.q.inverse(),
        }
    }

    pub fn exp(v: Vec3AF32) -> Self {
        let theta_sq = v.dot(v);
        let theta = theta_sq.sqrt();
        let theta_half: f32 = 0.5 * theta;

        let (w, b) = if theta < SMALL_ANGLE_EPSILON {
            // using the taylor series expansion of cos(x/2) and sin(x/2)/x around 0
            (1.0 - theta_sq / 8.0, 0.5 - theta_sq / 48.0)
        } else {
            (theta_half.cos(), theta_half.sin() / theta)
        };

        let xyz = b * v;

        Self {
            q: QuatF32::from_xyzw(xyz.x, xyz.y, xyz.z, w),
        }
    }

    /// Lie group -> Lie algebra
    pub fn log(&self) -> Vec3AF32 {
        let mut w = self.q.w;
        let mut vec = Vec3AF32::new(self.q.x, self.q.y, self.q.z);

        if w < 0.0 {
            w = -w;
            vec = -vec;
        }

        let theta_sq = vec.dot(vec);
        let theta = theta_sq.sqrt();

        if theta > SMALL_ANGLE_EPSILON {
            // Use atan2 rather than acos(w): near identity w rounds to 1.0 and
            // acos(1.0) = 0 collapses the result, whereas atan2 stays accurate.
            let half_theta = theta.atan2(w);
            let scale = 2.0 * half_theta / theta;
            vec * scale
        } else {
            // Small-angle approximation (Taylor series)
            let scale = 2.0 / w;
            vec * scale
        }
    }

    /// Vector space -> Lie algebra
    pub fn hat(v: Vec3AF32) -> Mat3AF32 {
        let (a, b, c) = (v.x, v.y, v.z);
        Mat3AF32::from_cols_array(&[0.0, c, -b, -c, 0.0, a, b, -a, 0.0])
    }

    /// Lie algebra -> vector space
    pub fn vee(omega: Mat3AF32) -> Vec3AF32 {
        let a = omega.y_axis.z;
        let b = omega.z_axis.x;
        let c = omega.x_axis.y;
        Vec3AF32::new(a, b, c)
    }

    pub fn vee4(omega: Mat4F32) -> Vec3AF32 {
        let a = omega.y_axis.z;
        let b = omega.z_axis.x;
        let c = omega.x_axis.y;
        Vec3AF32::new(a, b, c)
    }

    /// Left Jacobian of SO(3) at `v`.
    pub fn left_jacobian(v: Vec3AF32) -> Mat3AF32 {
        let skew = Self::hat(v);
        let (a, b) = jacobian_coefficients_f32(v.dot(v));
        Mat3AF32::IDENTITY + a * skew + b * (skew * skew)
    }

    /// Right Jacobian of SO(3) at `v`.
    pub fn right_jacobian(v: Vec3AF32) -> Mat3AF32 {
        let skew = Self::hat(v);
        let (a, b) = jacobian_coefficients_f32(v.dot(v));
        Mat3AF32::IDENTITY - a * skew + b * (skew * skew)
    }
}

/// `(1 − cos θ)/θ²` and `(θ − sin θ)/θ³`, the coefficients of the SO(3)
/// Jacobians. Below 0.1 rad (theta2 < 1e-2) in f32, `1 − cos θ` cancels
/// drastically to 0.0, so their Taylor series is used there.
#[inline]
pub(crate) fn jacobian_coefficients_f32(theta2: f32) -> (f32, f32) {
    if theta2 < JACOBIAN_TAYLOR_THRESHOLD_F32 {
        let theta4 = theta2 * theta2;
        (
            0.5 - theta2 / 24.0 + theta4 / 720.0,
            (1.0 / 6.0) - theta2 / 120.0 + theta4 / 5040.0,
        )
    } else {
        let theta = theta2.sqrt();
        (
            (1.0 - theta.cos()) / theta2,
            (theta - theta.sin()) / (theta2 * theta),
        )
    }
}

impl std::ops::Mul<SO3F32> for SO3F32 {
    type Output = SO3F32;

    fn mul(self, rhs: Self) -> Self::Output {
        Self { q: self.q * rhs.q }
    }
}

impl Param for SO3F32 {
    const GLOBAL_SIZE: usize = 4;
    const LOCAL_SIZE: usize = 3;

    #[inline]
    fn plus(x: &[f32], delta: &[f32], out: &mut [f32]) -> Result<(), ParamError> {
        if x.len() < Self::GLOBAL_SIZE {
            return Err(ParamError::WrongGlobalSize {
                expected: Self::GLOBAL_SIZE,
                got: x.len(),
            });
        }
        if delta.len() < Self::LOCAL_SIZE {
            return Err(ParamError::WrongLocalSize {
                expected: Self::LOCAL_SIZE,
                got: delta.len(),
            });
        }
        if out.len() < Self::GLOBAL_SIZE {
            return Err(ParamError::WrongOutSize {
                expected: Self::GLOBAL_SIZE,
                got: out.len(),
            });
        }

        let q = QuatF32::from_array([x[0], x[1], x[2], x[3]]).normalize();
        let so3 = SO3F32::new(q);
        let tau = Vec3AF32::new(delta[0], delta[1], delta[2]);
        let so3_plus = so3.rplus(tau);
        out[..Self::GLOBAL_SIZE].copy_from_slice(&so3_plus.to_array());
        Ok(())
    }
}

impl std::ops::MulAssign<SO3F32> for SO3F32 {
    #[inline]
    fn mul_assign(&mut self, rhs: SO3F32) {
        *self = *self * rhs;
    }
}

impl std::ops::Mul<Vec3AF32> for SO3F32 {
    type Output = Vec3AF32;

    fn mul(self, rhs: Vec3AF32) -> Self::Output {
        let result = self.q.mul_vec3([rhs.x, rhs.y, rhs.z]);
        Vec3AF32::new(result[0], result[1], result[2])
    }
}

#[cfg(feature = "approx")]
impl approx::AbsDiffEq for SO3F32 {
    type Epsilon = <QuatF32 as approx::AbsDiffEq>::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        <QuatF32 as approx::AbsDiffEq>::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.q.abs_diff_eq(&other.q, epsilon)
    }
}

#[cfg(feature = "approx")]
impl approx::RelativeEq for SO3F32 {
    #[inline]
    fn default_max_relative() -> Self::Epsilon {
        <QuatF32 as approx::RelativeEq>::default_max_relative()
    }

    #[inline]
    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.q.relative_eq(&other.q, epsilon, max_relative)
    }
}

/// Errors that can occur when converting a 3×3 matrix to an [`SO3F64`] rotation.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum RotationError {
    /// The supplied tolerance is negative or non-finite (NaN, infinity).
    #[error("tolerance must be non-negative and finite, got {tol}")]
    InvalidTolerance {
        /// The invalid tolerance value.
        tol: f64,
    },

    /// The matrix contains non-finite entries (NaN, infinity) or intermediate
    /// calculations produced non-finite values.
    #[error("matrix entries and intermediate calculations must be finite")]
    NonFinite,

    /// The matrix is not orthonormal (`Rᵀ R ≠ I`).
    #[error("matrix is not orthonormal: max |(RᵀR - I)_ij| = {max_error:.6e}")]
    NotOrthonormal {
        /// The maximum absolute entry of `Rᵀ R - I`.
        max_error: f64,
    },

    /// The matrix does not have a positive determinant (e.g. reflection with det ≤ 0).
    #[error("matrix is not a proper rotation: det = {det:.6e} (expected > 0)")]
    NotProper {
        /// The determinant of the matrix.
        det: f64,
    },

    /// The matrix projection onto SO(3) failed due to zero-length or degenerate column vectors.
    #[error("matrix projection to SO(3) failed due to zero or linearly dependent column vectors")]
    DegenerateProjection,
}

/// Shared internal validator for rotation matrices.
///
/// Validates that a 3x3 matrix is an element of SO(3) within tolerance `tol`:
/// 1. Rejects non-finite or negative `tol`.
/// 2. Rejects matrices with non-finite entries.
/// 3. Computes `Rᵀ R` and verifies `max |(RᵀR - I)_ij| <= tol`.
/// 4. Verifies `det(R) > 0.0` (proper rotation / right-handedness).
pub(crate) fn validate_rotation_matrix(mat: &Mat3F64, tol: f64) -> Result<(), RotationError> {
    if !tol.is_finite() || tol < 0.0 {
        return Err(RotationError::InvalidTolerance { tol });
    }

    if !mat.is_finite() {
        return Err(RotationError::NonFinite);
    }

    let rtr = mat.transpose() * (*mat);
    let diff = rtr - Mat3F64::IDENTITY;

    // Explicitly verify that intermediate calculations for RᵀR - I are finite
    // before computing max_error.
    if !diff.is_finite() {
        return Err(RotationError::NonFinite);
    }

    let max_err = diff
        .to_cols_array()
        .iter()
        .fold(0.0_f64, |acc, &x| acc.max(x.abs()));

    if !max_err.is_finite() {
        return Err(RotationError::NonFinite);
    }

    if max_err > tol {
        return Err(RotationError::NotOrthonormal { max_error: max_err });
    }

    let det = mat.determinant();
    if !det.is_finite() {
        return Err(RotationError::NonFinite);
    }

    if det <= 0.0 {
        return Err(RotationError::NotProper { det });
    }

    Ok(())
}

/// Normalizes a 3D vector using scale-first normalization to avoid underflow
/// and overflow during norm calculation.
///
/// Returns `Err(RotationError::NonFinite)` if any vector component is non-finite (NaN, infinity).
/// Returns `Err(RotationError::DegenerateProjection)` if the vector has zero length or underflows
/// to zero.
fn scale_first_normalize(v: glam::DVec3) -> Result<glam::DVec3, RotationError> {
    if !v.is_finite() {
        return Err(RotationError::NonFinite);
    }

    let max_c = v.x.abs().max(v.y.abs()).max(v.z.abs());
    if max_c == 0.0 {
        return Err(RotationError::DegenerateProjection);
    }

    let scaled = v / max_c;
    let norm_sq = scaled.length_squared();
    if !norm_sq.is_finite() {
        return Err(RotationError::NonFinite);
    }
    if norm_sq == 0.0 {
        return Err(RotationError::DegenerateProjection);
    }

    let norm = norm_sq.sqrt();
    let unit = scaled / norm;
    if !unit.is_finite() {
        return Err(RotationError::NonFinite);
    }
    if !unit.is_normalized() {
        return Err(RotationError::DegenerateProjection);
    }

    Ok(unit)
}

/// Orthonormalizes a 3x3 matrix using Gram-Schmidt process with a right-handed
/// cross product.
///
/// This projects an approximate rotation matrix (already validated to be in SO(3)
/// within tolerance) to an element of SO(3). Employs scale-first normalization
/// to prevent underflow with subnormal components, ensuring columns are unit length
/// within standard floating-point precision (`is_normalized() == true`) and mutually
/// orthogonal before passing to `glam::DQuat::from_mat3`, preventing `glam_assert` failures.
///
/// Returns `Err(RotationError::NonFinite)` if columns or intermediate vectors are non-finite.
/// Returns `Err(RotationError::DegenerateProjection)` if columns or Gram-Schmidt residuals
/// are zero or linearly dependent.
pub(crate) fn project_to_so3(mat: &Mat3F64) -> Result<glam::DMat3, RotationError> {
    if !mat.is_finite() {
        return Err(RotationError::NonFinite);
    }

    let x = mat.0.x_axis;
    let y = mat.0.y_axis;

    let u0 = scale_first_normalize(x)?;
    let r1 = y - u0 * u0.dot(y);
    let u1 = scale_first_normalize(r1)?;
    let u2 = u0.cross(u1);

    if !u2.is_finite() {
        return Err(RotationError::NonFinite);
    }
    if !u2.is_normalized() {
        return Err(RotationError::DegenerateProjection);
    }

    Ok(glam::DMat3::from_cols(u0, u1, u2))
}

/// A 3D rotation in double precision, stored as a unit quaternion.
///
/// The `f64` counterpart of [`SO3F32`]: same SU(2) → SO(3) double-cover
/// semantics (`q` and `-q` are the same rotation), same exp/log/hat/vee and
/// Jacobian conventions, computed in double precision. See the
/// [module-level documentation](self) and [`SO3F32`] for the full discussion.
///
/// Unlike [`SO3F32`], this type does not implement [`Param`]: the optimizer's
/// parameter interface is currently `f32`-only.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SO3F64 {
    /// The underlying unit quaternion.
    pub q: QuatF64,
}

impl SO3F64 {
    /// The identity rotation.
    pub const IDENTITY: Self = Self {
        q: QuatF64::IDENTITY,
    };

    /// Create a new SO3F64 from a quaternion.
    /// NOTE: quaternion should be normalized
    #[inline]
    pub fn new(quat: QuatF64) -> Self {
        Self { q: quat }
    }

    /// Create from a `[x, y, z, w]` quaternion array.
    pub fn from_array(arr: [f64; 4]) -> Self {
        Self {
            q: QuatF64::from_array(arr),
        }
    }

    /// Convert to a `[x, y, z, w]` quaternion array.
    pub fn to_array(&self) -> [f64; 4] {
        self.q.to_array()
    }

    /// Create a new SO3F64 from a quaternion.
    /// NOTE: quaternion should be normalized
    #[inline]
    pub fn from_quaternion(quat: QuatF64) -> Self {
        Self::new(quat)
    }

    /// Create from a 3x3 rotation matrix (chooses one of the two quaternions).
    pub fn from_matrix(mat: &Mat3F64) -> Self {
        Self {
            q: QuatF64::from_mat3(mat),
        }
    }

    /// Attempt to construct an `SO3F64` rotation from a 3×3 matrix, validating
    /// orthonormality (`Rᵀ R ≈ I`) and proper handedness (`det(R) > 0`).
    ///
    /// Matrices with non-positive determinants (`det ≤ 0.0`, including reflections)
    /// are strictly rejected as improper (`RotationError::NotProper`), even when a large
    /// tolerance is provided. Invalid tolerances (negative, NaN, infinity) return
    /// [`RotationError::InvalidTolerance`], and non-finite matrix entries return
    /// [`RotationError::NonFinite`].
    ///
    /// When the input matrix is accepted within tolerance, it is orthonormalized
    /// via scale-first Gram-Schmidt projection before quaternion conversion, ensuring
    /// normalized columns within floating-point precision and numerical robustness
    /// without panicking under assertion checks.
    ///
    /// # Arguments
    ///
    /// * `mat` - Candidate 3×3 rotation matrix.
    /// * `tol` - Non-negative, finite tolerance for the orthonormality check.
    ///
    /// # Returns
    ///
    /// `Ok(SO3F64)` if `mat` is a valid rotation matrix within `tol`.
    ///
    /// # Errors
    ///
    /// Returns [`RotationError::InvalidTolerance`] if `tol` is negative or non-finite.
    /// Returns [`RotationError::NonFinite`] if `mat` has non-finite entries or intermediate
    /// calculations produce non-finite values.
    /// Returns [`RotationError::NotOrthonormal`] if the maximum absolute entry of `Rᵀ R - I`
    /// exceeds `tol`.
    /// Returns [`RotationError::NotProper`] if the matrix has a non-positive determinant (`det ≤ 0.0`).
    /// Returns [`RotationError::DegenerateProjection`] if the matrix columns or Gram-Schmidt residuals
    /// are degenerate (zero-length or collinear), preventing projection to SO(3).
    ///
    /// # Example
    ///
    /// ```rust
    /// use kornia_algebra::{Mat3F64, SO3F64, RotationError};
    ///
    /// // Valid identity matrix
    /// let r = SO3F64::try_from_matrix(&Mat3F64::IDENTITY, 1e-6);
    /// assert!(r.is_ok());
    ///
    /// // Reflection matrix (det = -1) rejected, even with a large tolerance
    /// let reflection = Mat3F64::from_rows_array(&[
    ///     -1.0, 0.0, 0.0,
    ///      0.0, 1.0, 0.0,
    ///      0.0, 0.0, 1.0,
    /// ]);
    /// assert!(matches!(
    ///     SO3F64::try_from_matrix(&reflection, 1e-6),
    ///     Err(RotationError::NotProper { .. })
    /// ));
    /// assert!(matches!(
    ///     SO3F64::try_from_matrix(&reflection, 2.0),
    ///     Err(RotationError::NotProper { .. })
    /// ));
    /// ```
    pub fn try_from_matrix(mat: &Mat3F64, tol: f64) -> Result<Self, RotationError> {
        validate_rotation_matrix(mat, tol)?;
        let ortho = project_to_so3(mat)?;
        Ok(Self {
            q: QuatF64(glam::DQuat::from_mat3(&ortho).normalize()),
        })
    }

    /// Create from the rotation block of a 4x4 homogeneous matrix.
    pub fn from_matrix4(mat: &Mat4F64) -> Self {
        Self {
            q: QuatF64::from_mat4(mat),
        }
    }

    /// Uniformly-random rotation (Shoemake's method).
    pub fn from_random() -> Self {
        let mut rng = rand::rng();

        let r1: f64 = rng.random();
        let r2: f64 = rng.random();
        let r3: f64 = rng.random();

        // Correct uniform random quaternion generation (Shoemake method)
        let one_minus_r1_sqrt = (1.0 - r1).sqrt();
        let r1_sqrt = r1.sqrt();

        let w = one_minus_r1_sqrt * (2.0 * std::f64::consts::PI * r2).cos();
        let x = one_minus_r1_sqrt * (2.0 * std::f64::consts::PI * r2).sin();
        let y = r1_sqrt * (2.0 * std::f64::consts::PI * r3).cos();
        let z = r1_sqrt * (2.0 * std::f64::consts::PI * r3).sin();

        Self {
            q: QuatF64::from_xyzw(x, y, z, w).normalize(),
        }
    }

    /// Right-plus retraction: `self ∘ exp(tau)`.
    #[inline]
    pub fn rplus(&self, tau: Vec3F64) -> Self {
        *self * SO3F64::exp(tau)
    }

    /// Right-minus: `log(self⁻¹ ∘ other)`.
    #[inline]
    pub fn rminus(&self, other: &Self) -> Vec3F64 {
        (self.inverse() * *other).log()
    }

    /// Left-plus retraction: `exp(tau) ∘ x`.
    #[inline]
    pub fn lplus(tau: Vec3F64, x: &Self) -> Self {
        SO3F64::exp(tau) * *x
    }

    /// Left-minus: `log(y ∘ x⁻¹)`.
    #[inline]
    pub fn lminus(y: &Self, x: &Self) -> Vec3F64 {
        (*y * x.inverse()).log()
    }

    /// The 3x3 rotation matrix.
    pub fn matrix(&self) -> Mat3F64 {
        Mat3F64::from_quat(self.q)
    }

    /// Adjoint representation (equal to the rotation matrix for SO(3)).
    pub fn adjoint(&self) -> Mat3F64 {
        self.matrix()
    }

    /// Group inverse (quaternion conjugate of a unit quaternion).
    pub fn inverse(&self) -> Self {
        Self {
            q: self.q.inverse(),
        }
    }

    /// Exponential map so(3) → SO(3): axis-angle vector → rotation.
    pub fn exp(v: Vec3F64) -> Self {
        let theta_sq = v.dot(v);
        let theta = theta_sq.sqrt();
        let theta_half: f64 = 0.5 * theta;

        let (w, b) = if theta < SMALL_ANGLE_EPSILON_F64 {
            // using the taylor series expansion of cos(x/2) and sin(x/2)/x around 0
            (1.0 - theta_sq / 8.0, 0.5 - theta_sq / 48.0)
        } else {
            (theta_half.cos(), theta_half.sin() / theta)
        };

        let xyz = b * v;

        Self {
            q: QuatF64::from_xyzw(xyz.x, xyz.y, xyz.z, w),
        }
    }

    /// Logarithmic map SO(3) → so(3): rotation → axis-angle vector.
    pub fn log(&self) -> Vec3F64 {
        let mut w = self.q.w;
        let mut vec = Vec3F64::new(self.q.x, self.q.y, self.q.z);

        if w < 0.0 {
            w = -w;
            vec = -vec;
        }

        let theta_sq = vec.dot(vec);
        let theta = theta_sq.sqrt();

        if theta > SMALL_ANGLE_EPSILON_F64 {
            // `vec = sin(θ/2)·axis` and `w = cos(θ/2)`, so the half-angle is
            // `atan2(|vec|, w)`. We use atan2 rather than `acos(w)`: near the
            // identity `w` rounds to 1.0 and `acos(1.0) = 0` collapses the
            // result, whereas atan2 stays accurate for small `|vec|`.
            let half_theta = theta.atan2(w);
            let scale = 2.0 * half_theta / theta;
            vec * scale
        } else {
            // Small-angle approximation (Taylor series)
            let scale = 2.0 / w;
            vec * scale
        }
    }

    /// Hat operator R³ → so(3): vector → skew-symmetric matrix.
    pub fn hat(v: Vec3F64) -> Mat3F64 {
        let (a, b, c) = (v.x, v.y, v.z);
        Mat3F64::from_cols_array(&[0.0, c, -b, -c, 0.0, a, b, -a, 0.0])
    }

    /// Vee operator so(3) → R³: skew-symmetric matrix → vector (inverse of [`hat`](Self::hat)).
    pub fn vee(omega: Mat3F64) -> Vec3F64 {
        let a = omega.y_axis().z;
        let b = omega.z_axis().x;
        let c = omega.x_axis().y;
        Vec3F64::new(a, b, c)
    }

    /// Vee operator reading the rotation block of a 4x4 matrix.
    pub fn vee4(omega: Mat4F64) -> Vec3F64 {
        let a = omega.y_axis().z;
        let b = omega.z_axis().x;
        let c = omega.x_axis().y;
        Vec3F64::new(a, b, c)
    }

    /// Left Jacobian of SO(3) at `v`.
    pub fn left_jacobian(v: Vec3F64) -> Mat3F64 {
        let skew = Self::hat(v);
        let (a, b) = jacobian_coefficients_f64(v.dot(v));
        Mat3F64::IDENTITY + a * skew + b * (skew * skew)
    }

    /// Right Jacobian of SO(3) at `v`.
    pub fn right_jacobian(v: Vec3F64) -> Mat3F64 {
        let skew = Self::hat(v);
        let (a, b) = jacobian_coefficients_f64(v.dot(v));
        Mat3F64::IDENTITY - a * skew + b * (skew * skew)
    }
}

/// `(1 − cos θ)/θ²` and `(θ − sin θ)/θ³`, the coefficients of the SO(3)
/// Jacobians. Below 0.01 rad `1 − cos θ` cancels (an absolute error of about
/// `1e-16/θ` in the Jacobian), so their Taylor series is used there.
fn jacobian_coefficients_f64(theta2: f64) -> (f64, f64) {
    if theta2 < 1.0e-4 {
        let theta4 = theta2 * theta2;
        (
            0.5 - theta2 / 24.0 + theta4 / 720.0,
            1.0 / 6.0 - theta2 / 120.0 + theta4 / 5040.0,
        )
    } else {
        let theta = theta2.sqrt();
        (
            (1.0 - theta.cos()) / theta2,
            (theta - theta.sin()) / (theta2 * theta),
        )
    }
}

impl std::ops::Mul<SO3F64> for SO3F64 {
    type Output = SO3F64;

    fn mul(self, rhs: Self) -> Self::Output {
        Self { q: self.q * rhs.q }
    }
}

impl std::ops::MulAssign<SO3F64> for SO3F64 {
    #[inline]
    fn mul_assign(&mut self, rhs: SO3F64) {
        *self = *self * rhs;
    }
}

impl std::ops::Mul<Vec3F64> for SO3F64 {
    type Output = Vec3F64;

    fn mul(self, rhs: Vec3F64) -> Self::Output {
        let result = self.q.mul_vec3([rhs.x, rhs.y, rhs.z]);
        Vec3F64::new(result[0], result[1], result[2])
    }
}

#[cfg(feature = "approx")]
impl approx::AbsDiffEq for SO3F64 {
    type Epsilon = <QuatF64 as approx::AbsDiffEq>::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        <QuatF64 as approx::AbsDiffEq>::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.q.abs_diff_eq(&other.q, epsilon)
    }
}

#[cfg(feature = "approx")]
impl approx::RelativeEq for SO3F64 {
    #[inline]
    fn default_max_relative() -> Self::Epsilon {
        <QuatF64 as approx::RelativeEq>::default_max_relative()
    }

    #[inline]
    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.q.relative_eq(&other.q, epsilon, max_relative)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    const EPSILON: f32 = 1e-6;
    // Roundtrip tests (rplus/rminus, lplus/lminus) compose exp + log on random inputs, which
    // accumulates ~1 ULP per cos/sin/atan2 call and can land right at EPSILON under --release
    // FMA fusion. Use a slightly looser tolerance for those to avoid intermittent f32 boundary flakes.
    const ROUNDTRIP_EPSILON: f32 = 5e-6;

    #[test]
    fn test_identity() {
        let s = SO3F32::IDENTITY;
        let q_expected = QuatF32::from_xyzw(0.0, 0.0, 0.0, 1.0);
        assert_relative_eq!(s.q.x, q_expected.x, epsilon = EPSILON);
        assert_relative_eq!(s.q.y, q_expected.y, epsilon = EPSILON);
        assert_relative_eq!(s.q.z, q_expected.z, epsilon = EPSILON);
        assert_relative_eq!(s.q.w, q_expected.w, epsilon = EPSILON);
    }

    #[test]
    fn test_from_quaternion() {
        let q = QuatF32::from_xyzw(4.0, -2.0, 1.0, 3.5).normalize();
        let s = SO3F32::from_quaternion(q);
        assert_relative_eq!(s.q.x, q.x, epsilon = EPSILON);
        assert_relative_eq!(s.q.y, q.y, epsilon = EPSILON);
        assert_relative_eq!(s.q.z, q.z, epsilon = EPSILON);
        assert_relative_eq!(s.q.w, q.w, epsilon = EPSILON);
    }

    #[test]
    fn test_from_matrix() {
        let mat = Mat3AF32::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.6, 0.8, 0.0, -0.8, 0.6]);
        let s = SO3F32::from_matrix(&mat);

        let q_expected = QuatF32::from_xyzw(0.5, 0.0, 0.0, 1.0).normalize();
        assert_relative_eq!(s.q.x, q_expected.x, epsilon = EPSILON);
        assert_relative_eq!(s.q.y, q_expected.y, epsilon = EPSILON);
        assert_relative_eq!(s.q.z, q_expected.z, epsilon = EPSILON);
        assert_relative_eq!(s.q.w, q_expected.w, epsilon = EPSILON);
    }

    #[test]
    fn test_unit_norm() {
        // Test that quaternions maintain unit norm
        let q1 = QuatF32::from_xyzw(1.0, 2.0, 3.0, 4.0).normalize();
        let q2 = QuatF32::from_xyzw(-1.0, 0.5, -2.0, 1.5).normalize();
        let s1 = SO3F32::from_quaternion(q1);
        let s2 = SO3F32::from_quaternion(q2);
        let s3 = s1 * s2;
        let s4 = s1.inverse();
        let s5 = s2.inverse();
        let s6 = s3.inverse();

        assert_relative_eq!(s1.q.length(), 1.0, epsilon = EPSILON);
        assert_relative_eq!(s2.q.length(), 1.0, epsilon = EPSILON);
        assert_relative_eq!(s3.q.length(), 1.0, epsilon = EPSILON);
        assert_relative_eq!(s4.q.length(), 1.0, epsilon = EPSILON);
        assert_relative_eq!(s5.q.length(), 1.0, epsilon = EPSILON);
        assert_relative_eq!(s6.q.length(), 1.0, epsilon = EPSILON);
    }

    #[test]
    fn test_so3_rplus_rminus_roundtrip() {
        let x = SO3F32::from_random();
        let tau = Vec3AF32::new(0.4, -0.2, 0.7);
        let y = x.rplus(tau); // X ⊕ τ → Y
        let diff = x.rminus(&y); // Y ⊖ X → τ
        assert_relative_eq!(diff.x, tau.x, epsilon = ROUNDTRIP_EPSILON);
        assert_relative_eq!(diff.y, tau.y, epsilon = ROUNDTRIP_EPSILON);
        assert_relative_eq!(diff.z, tau.z, epsilon = ROUNDTRIP_EPSILON);
    }

    #[test]
    fn test_so3_lplus_lminus_consistency() {
        let x = SO3F32::from_random();
        let tau = Vec3AF32::new(-0.3, 1.1, 0.2);
        let y = SO3F32::lplus(tau, &x); // τ ⊕ X → Y
        let diff = SO3F32::lminus(&y, &x); // Y ⊖ X → τ
        assert_relative_eq!(diff.x, tau.x, epsilon = ROUNDTRIP_EPSILON);
        assert_relative_eq!(diff.y, tau.y, epsilon = ROUNDTRIP_EPSILON);
        assert_relative_eq!(diff.z, tau.z, epsilon = ROUNDTRIP_EPSILON);
    }

    #[test]
    fn test_exp() {
        // Test exp of zero vector is identity
        let v = Vec3AF32::from_array([0.0, 0.0, 0.0]);
        let s = SO3F32::exp(v);

        let q_expected = QuatF32::from_xyzw(0.0, 0.0, 0.0, 1.0);
        assert_relative_eq!(s.q.x, q_expected.x, epsilon = EPSILON);
        assert_relative_eq!(s.q.y, q_expected.y, epsilon = EPSILON);
        assert_relative_eq!(s.q.z, q_expected.z, epsilon = EPSILON);
        assert_relative_eq!(s.q.w, q_expected.w, epsilon = EPSILON);
    }

    #[test]
    fn test_log() {
        // Test 1: log of identity quaternion should be zero vector
        {
            let so3 = SO3F32::IDENTITY;
            let log = so3.log();

            let log_expected = Vec3AF32::new(0.0, 0.0, 0.0);
            assert_relative_eq!(log.x, log_expected.x, epsilon = EPSILON);
            assert_relative_eq!(log.y, log_expected.y, epsilon = EPSILON);
            assert_relative_eq!(log.z, log_expected.z, epsilon = EPSILON);
        }

        // Test 2: exp-log consistency
        {
            let so3 = SO3F32::exp(Vec3AF32::new(1.0, 0.0, 0.0));
            let log = so3.log();

            let log_expected = Vec3AF32::new(1.0, 0.0, 0.0);
            assert_relative_eq!(log.x, log_expected.x, epsilon = EPSILON);
            assert_relative_eq!(log.y, log_expected.y, epsilon = EPSILON);
            assert_relative_eq!(log.z, log_expected.z, epsilon = EPSILON);
        }
    }

    #[test]
    fn test_exp_log() {
        // Test exp-log consistency for various vectors
        let test_vectors = [
            Vec3AF32::new(0.1, 0.2, 0.3),
            Vec3AF32::new(1.0, 0.0, 0.0),
            Vec3AF32::new(0.0, 1.0, 0.0),
            Vec3AF32::new(0.0, 0.0, 1.0),
            Vec3AF32::new(-0.5, 0.3, -0.2),
        ];

        for v in test_vectors.iter() {
            let s = SO3F32::exp(*v);
            let log_result = s.log();
            assert_relative_eq!(log_result.x, v.x, epsilon = 1e-5);
            assert_relative_eq!(log_result.y, v.y, epsilon = 1e-5);
            assert_relative_eq!(log_result.z, v.z, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_hat() {
        let v = Vec3AF32::new(1.0, 2.0, 3.0);
        let hat_v = SO3F32::hat(v);

        // Check skew-symmetric matrix structure
        assert_relative_eq!(hat_v.x_axis.x, 0.0, epsilon = EPSILON);
        assert_relative_eq!(hat_v.x_axis.y, 3.0, epsilon = EPSILON);
        assert_relative_eq!(hat_v.x_axis.z, -2.0, epsilon = EPSILON);
        assert_relative_eq!(hat_v.y_axis.x, -3.0, epsilon = EPSILON);
        assert_relative_eq!(hat_v.y_axis.y, 0.0, epsilon = EPSILON);
        assert_relative_eq!(hat_v.y_axis.z, 1.0, epsilon = EPSILON);
        assert_relative_eq!(hat_v.z_axis.x, 2.0, epsilon = EPSILON);
        assert_relative_eq!(hat_v.z_axis.y, -1.0, epsilon = EPSILON);
        assert_relative_eq!(hat_v.z_axis.z, 0.0, epsilon = EPSILON);
    }

    #[test]
    fn test_vee() {
        // Test vee operation
        let omega = SO3F32::hat(Vec3AF32::new(1.0, 2.0, 3.0));
        let v = SO3F32::vee(omega);

        let v_expected = Vec3AF32::new(1.0, 2.0, 3.0);
        assert_relative_eq!(v.x, v_expected.x, epsilon = EPSILON);
        assert_relative_eq!(v.y, v_expected.y, epsilon = EPSILON);
        assert_relative_eq!(v.z, v_expected.z, epsilon = EPSILON);
    }

    #[test]
    fn test_hat_vee() {
        // Test hat-vee consistency
        let test_vectors = [
            Vec3AF32::new(1.0, 2.0, 3.0),
            Vec3AF32::new(-0.5, 0.0, 1.5),
            Vec3AF32::new(0.1, -0.2, 0.3),
        ];

        for v in test_vectors.iter() {
            let omega = SO3F32::hat(*v);
            let v_recovered = SO3F32::vee(omega);
            assert_relative_eq!(v_recovered.x, v.x, epsilon = EPSILON);
            assert_relative_eq!(v_recovered.y, v.y, epsilon = EPSILON);
            assert_relative_eq!(v_recovered.z, v.z, epsilon = EPSILON);
        }
    }

    #[test]
    fn test_matrix() {
        // Test rotation matrix properties
        let q = QuatF32::from_xyzw(0.1, 0.2, 0.3, 0.9).normalize();
        let so3 = SO3F32::from_quaternion(q);
        let r = so3.matrix();

        // Test that it's orthogonal (R^T * R = I)
        let identity = r.transpose() * r;
        let identity_expected = Mat3AF32::IDENTITY;

        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(
                    identity.col(i)[j],
                    identity_expected.col(i)[j],
                    epsilon = 1e-5
                );
            }
        }

        // Test determinant is 1
        let det = r.determinant();
        assert_relative_eq!(det, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_mul() {
        // Test multiplication with identity
        let q1 = QuatF32::IDENTITY;
        let q2 = QuatF32::from_xyzw(0.1, 0.2, 0.3, 0.9).normalize();
        let s1 = SO3F32::from_quaternion(q1);
        let s2 = SO3F32::from_quaternion(q2);

        let result = s1 * s2;
        assert_relative_eq!(result.q.x, s2.q.x, epsilon = EPSILON);
        assert_relative_eq!(result.q.y, s2.q.y, epsilon = EPSILON);
        assert_relative_eq!(result.q.z, s2.q.z, epsilon = EPSILON);
        assert_relative_eq!(result.q.w, s2.q.w, epsilon = EPSILON);

        // Test inverse property: s * s^-1 = identity
        let s_inv = s2.inverse();
        let identity_result = s2 * s_inv;
        assert_relative_eq!(identity_result.q.x, s1.q.x, epsilon = 1e-5);
        assert_relative_eq!(identity_result.q.y, s1.q.y, epsilon = 1e-5);
        assert_relative_eq!(identity_result.q.z, s1.q.z, epsilon = 1e-5);
        assert_relative_eq!(identity_result.q.w.abs(), s1.q.w.abs(), epsilon = 1e-5);
    }

    #[test]
    fn test_mul_assign() {
        let mut s = SO3F32::IDENTITY;
        let s2 = SO3F32::from_random();
        s *= s2;
        assert_relative_eq!(s.q.x, s2.q.x, epsilon = EPSILON);
        assert_relative_eq!(s.q.y, s2.q.y, epsilon = EPSILON);
        assert_relative_eq!(s.q.z, s2.q.z, epsilon = EPSILON);
        assert_relative_eq!(s.q.w, s2.q.w, epsilon = EPSILON);

        let mut s3 = SO3F32::from_random();
        let original_s3 = s3;
        let s4 = SO3F32::from_random();
        s3 *= s4;
        let expected = original_s3 * s4;
        assert_relative_eq!(s3.q.x, expected.q.x, epsilon = EPSILON);
        assert_relative_eq!(s3.q.y, expected.q.y, epsilon = EPSILON);
        assert_relative_eq!(s3.q.z, expected.q.z, epsilon = EPSILON);
        assert_relative_eq!(s3.q.w, expected.q.w, epsilon = EPSILON);
    }

    #[test]
    fn test_mul_vec() {
        // Test rotation of vectors
        let s1 = SO3F32::IDENTITY;
        let t = Vec3AF32::new(1.0, 2.0, 3.0);

        // Identity rotation should not change the vector
        let result = s1 * t;
        assert_relative_eq!(result.x, t.x, epsilon = EPSILON);
        assert_relative_eq!(result.y, t.y, epsilon = EPSILON);
        assert_relative_eq!(result.z, t.z, epsilon = EPSILON);

        // Test that rotation preserves vector length
        let q = QuatF32::from_xyzw(0.1, 0.2, 0.3, 0.9).normalize();
        let s2 = SO3F32::from_quaternion(q);
        let rotated = s2 * t;
        assert_relative_eq!(rotated.length(), t.length(), epsilon = 1e-5);
    }

    #[test]
    fn test_inverse() {
        // Test inverse properties
        let q = QuatF32::from_xyzw(0.1, 0.2, 0.3, 0.9).normalize();
        let so3 = SO3F32::from_quaternion(q);

        // Test double inverse
        let double_inv = so3.inverse().inverse();
        assert_relative_eq!(double_inv.q.x, so3.q.x, epsilon = 1e-5);
        assert_relative_eq!(double_inv.q.y, so3.q.y, epsilon = 1e-5);
        assert_relative_eq!(double_inv.q.z, so3.q.z, epsilon = 1e-5);
        assert_relative_eq!(double_inv.q.w.abs(), so3.q.w.abs(), epsilon = 1e-5);

        // Test matrix inverse
        let matrix_inv = so3.inverse().matrix();
        let matrix_expected = so3.matrix().transpose();

        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(
                    matrix_inv.col(i)[j],
                    matrix_expected.col(i)[j],
                    epsilon = 1e-5
                );
            }
        }
    }

    #[test]
    fn test_adjoint() {
        // Test adjoint properties
        let q1 = QuatF32::from_xyzw(0.1, 0.2, 0.3, 0.9).normalize();
        let q2 = QuatF32::from_xyzw(-0.2, 0.1, 0.4, 0.8).normalize();
        let x = SO3F32::from_quaternion(q1);
        let y = SO3F32::from_quaternion(q2);

        // Test: x^-1.adjoint() = x.adjoint()^-1
        let left = x.inverse().adjoint();
        let right = x.adjoint().inverse();

        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(left.col(i)[j], right.col(i)[j], epsilon = 1e-5);
            }
        }

        // Test: (x * y).adjoint() = x.adjoint() * y.adjoint()
        let left_mult = (x * y).adjoint();
        let right_mult = x.adjoint() * y.adjoint();

        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(left_mult.col(i)[j], right_mult.col(i)[j], epsilon = 1e-5);
            }
        }
    }

    #[test]
    fn test_left_jacobian() {
        let test_vectors = [
            Vec3AF32::new(0.1, 0.2, 0.3),
            Vec3AF32::new(0.01, 0.02, 0.03), // Small angles
            Vec3AF32::new(1.0, 0.0, 0.0),
            Vec3AF32::new(0.0, 1.0, 0.0),
            Vec3AF32::new(0.0, 0.0, 1.0),
        ];

        for v in test_vectors.iter() {
            let jl = SO3F32::left_jacobian(*v);

            // Test that Jacobian is finite
            assert!(jl.is_finite());

            // Test basic property: Jl * v = v (approximately)
            let result = jl * *v;
            assert_relative_eq!(result.x, v.x, epsilon = 1e-4);
            assert_relative_eq!(result.y, v.y, epsilon = 1e-4);
            assert_relative_eq!(result.z, v.z, epsilon = 1e-4);
        }
    }

    #[test]
    fn test_right_jacobian() {
        let test_vectors = [
            Vec3AF32::new(0.1, 0.2, 0.3),
            Vec3AF32::new(0.01, 0.02, 0.03), // Small angles
            Vec3AF32::new(-0.1, 0.3, 0.2),
        ];

        for v in test_vectors.iter() {
            let jr = SO3F32::right_jacobian(*v);

            // Test that Jacobian is finite
            assert!(jr.is_finite());

            // Test basic property: Jr * v = v (approximately)
            let result = jr * *v;
            assert_relative_eq!(result.x, v.x, epsilon = 1e-4);
            assert_relative_eq!(result.y, v.y, epsilon = 1e-4);
            assert_relative_eq!(result.z, v.z, epsilon = 1e-4);
        }
    }

    #[test]
    fn test_right_left_jacobian() {
        // Test relationship between left and right Jacobians
        let test_vectors = [
            Vec3AF32::new(0.1, 0.2, 0.3),
            Vec3AF32::new(-0.05, 0.15, -0.1),
        ];

        for v in test_vectors.iter() {
            let jr = SO3F32::right_jacobian(*v);
            let jl = SO3F32::left_jacobian(*v);

            // Test: Jl = Jr^T (approximately for small angles)
            let jr_transpose = jr.transpose();

            for i in 0..3 {
                for j in 0..3 {
                    assert_relative_eq!(
                        jl.col(i)[j],
                        jr_transpose.col(i)[j],
                        epsilon = 0.1 // Looser tolerance as this is approximate
                    );
                }
            }
        }
    }

    /// At small angles both Jacobians equal their series `I ± ½ω× + ⅙ω×²` to
    /// single precision; without the Taylor series, `(1 − cos θ)/θ²` cancelled
    /// completely to 0.0 in f32 between 1e-8 and 1e-4 rad.
    #[test]
    fn small_angle_jacobians_match_series() {
        for scale in [1e-7f32, 1e-5, 1e-4, 1e-3] {
            let v = Vec3AF32::new(0.6, -0.8, 0.3) * scale;
            let skew = SO3F32::hat(v);
            let quadratic = (1.0 / 6.0) * (skew * skew);
            let left_expected = Mat3AF32::IDENTITY + 0.5 * skew + quadratic;
            let right_expected = Mat3AF32::IDENTITY - 0.5 * skew + quadratic;
            let left_actual = SO3F32::left_jacobian(v);
            let right_actual = SO3F32::right_jacobian(v);

            for i in 0..3 {
                for j in 0..3 {
                    assert_relative_eq!(
                        left_actual.col(i)[j],
                        left_expected.col(i)[j],
                        epsilon = 1e-6
                    );
                    assert_relative_eq!(
                        right_actual.col(i)[j],
                        right_expected.col(i)[j],
                        epsilon = 1e-6
                    );
                }
            }
        }
    }

    /// Tests smooth continuity and branch selection of `jacobian_coefficients_f32`
    /// across the series threshold (theta² = 1e-2, theta ≈ 0.1 rad).
    #[test]
    fn jacobian_coefficients_continuity() {
        let theta2_below = JACOBIAN_TAYLOR_THRESHOLD_F32 * 0.99;
        let theta2_above = JACOBIAN_TAYLOR_THRESHOLD_F32 * 1.01;

        let (a_below, b_below) = jacobian_coefficients_f32(theta2_below);
        let (a_above, b_above) = jacobian_coefficients_f32(theta2_above);

        // Independent double-precision reference
        let ref_coef = |t2: f32| -> (f32, f32) {
            let t2_f64 = t2 as f64;
            let t_f64 = t2_f64.sqrt();
            (
                ((1.0 - t_f64.cos()) / t2_f64) as f32,
                ((t_f64 - t_f64.sin()) / (t2_f64 * t_f64)) as f32,
            )
        };

        let (a_ref_below, b_ref_below) = ref_coef(theta2_below);
        let (a_ref_above, b_ref_above) = ref_coef(theta2_above);

        // Verify series branch below threshold matches reference
        assert_relative_eq!(a_below, a_ref_below, epsilon = 1e-6);
        assert_relative_eq!(b_below, b_ref_below, epsilon = 1e-6);

        // Verify closed-form branch above threshold matches reference
        assert_relative_eq!(a_above, a_ref_above, epsilon = 5e-6);
        assert_relative_eq!(b_above, b_ref_above, epsilon = 5e-6);

        // Verify continuity across the threshold matches the reference step
        assert_relative_eq!(a_below - a_above, a_ref_below - a_ref_above, epsilon = 6e-6);
        assert_relative_eq!(b_below - b_above, b_ref_below - b_ref_above, epsilon = 6e-6);
    }

    #[test]
    fn test_random() {
        // Test that random rotations are valid
        for _ in 0..10 {
            let so3 = SO3F32::from_random();

            // Test unit quaternion
            assert_relative_eq!(so3.q.length(), 1.0, epsilon = 1e-5);

            // Test that inverse works
            let identity = so3.inverse() * so3;
            assert_relative_eq!(identity.q.x, 0.0, epsilon = 1e-5);
            assert_relative_eq!(identity.q.y, 0.0, epsilon = 1e-5);
            assert_relative_eq!(identity.q.z, 0.0, epsilon = 1e-5);
            assert_relative_eq!(identity.q.w.abs(), 1.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_orthogonal_matrix() {
        // Test that rotation matrices are orthogonal
        let test_quaternions = [
            QuatF32::IDENTITY,
            QuatF32::from_xyzw(0.1, 0.2, 0.3, 0.9).normalize(),
            QuatF32::from_xyzw(-0.5, 0.3, -0.2, 0.7).normalize(),
        ];

        for q in test_quaternions.iter() {
            let so3 = SO3F32::from_quaternion(*q);
            let r = so3.matrix();
            let r_inv = so3.inverse().matrix();

            // Test R * R^T = I
            let identity = r * r.transpose();
            let expected_identity = Mat3AF32::IDENTITY;

            for i in 0..3 {
                for j in 0..3 {
                    assert_relative_eq!(
                        identity.col(i)[j],
                        expected_identity.col(i)[j],
                        epsilon = 1e-5
                    );
                }
            }

            // Test R^-1 = R^T
            let r_transpose = r.transpose();
            for i in 0..3 {
                for j in 0..3 {
                    assert_relative_eq!(r_inv.col(i)[j], r_transpose.col(i)[j], epsilon = 1e-5);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests_f64 {
    use super::*;

    const EPS: f64 = 1e-12;

    fn assert_mat_eq(a: &Mat3F64, b: &Mat3F64, tol: f64) {
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (a.col(i)[j] - b.col(i)[j]).abs() < tol,
                    "mismatch at col {i} row {j}: {} vs {}",
                    a.col(i)[j],
                    b.col(i)[j]
                );
            }
        }
    }

    #[test]
    fn exp_zero_is_identity() {
        let r = SO3F64::exp(Vec3F64::ZERO).matrix();
        assert_mat_eq(&r, &Mat3F64::IDENTITY, EPS);
    }

    #[test]
    fn exp_pi_around_z() {
        // 180° about z: x -> -x, y -> -y, z -> z.
        let r = SO3F64::exp(Vec3F64::new(0.0, 0.0, std::f64::consts::PI)).matrix();
        let expected = Mat3F64::from_cols_array(&[-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0]);
        assert_mat_eq(&r, &expected, 1e-12);
    }

    #[test]
    fn exp_log_roundtrip() {
        let test_vectors = [
            Vec3F64::new(0.1, 0.2, 0.3),
            Vec3F64::new(1.0, 0.0, 0.0),
            Vec3F64::new(0.0, 0.0, 1.0),
            Vec3F64::new(-0.5, 0.3, -0.2),
            Vec3F64::new(1e-9, -2e-9, 3e-9), // small-angle branch
        ];
        for v in test_vectors.iter() {
            let back = SO3F64::exp(*v).log();
            assert!((back - *v).length() < 1e-12, "got {back:?} from {v:?}");
        }
    }

    #[test]
    fn hat_is_skew_symmetric() {
        let v = Vec3F64::new(1.0, 2.0, 3.0);
        let h = SO3F64::hat(v);
        let sum = h + h.transpose();
        assert_mat_eq(&sum, &Mat3F64::from_cols_array(&[0.0; 9]), EPS);
        // vee undoes hat.
        let back = SO3F64::vee(h);
        assert!((back - v).length() < EPS);
    }

    #[test]
    fn matrix_is_orthogonal_with_unit_det() {
        let r = SO3F64::exp(Vec3F64::new(0.3, -0.7, 0.2)).matrix();
        assert_mat_eq(&(r.transpose() * r), &Mat3F64::IDENTITY, 1e-12);
        assert!((r.determinant() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn right_jacobian_finite_difference() {
        // exp(omega + delta) ≈ exp(omega) · exp(J_r(omega) · delta).
        let omega = Vec3F64::new(0.3, -0.2, 0.5);
        let jr = SO3F64::right_jacobian(omega);
        let r = SO3F64::exp(omega).matrix();
        let eps = 1e-6;

        for i in 0..3 {
            let mut d = [0.0; 3];
            d[i] = eps;
            let dv = Vec3F64::new(d[0], d[1], d[2]);

            let lhs = SO3F64::exp(omega + dv).matrix();
            let jr_delta = jr * dv;
            let rhs = r * SO3F64::exp(jr_delta).matrix();
            assert_mat_eq(&lhs, &rhs, 1e-9);
        }
    }

    /// At small angles both Jacobians equal their series `I ± ½ω× + ⅙ω×²` to
    /// the last digit; the closed form `(1 − cos θ)/θ²` was off by ~1e-16/θ.
    #[test]
    fn small_angle_jacobians_match_series() {
        for scale in [1e-9, 1e-7, 1e-5, 1e-3] {
            let v = Vec3F64::new(0.6, -0.8, 0.3) * scale;
            let skew = SO3F64::hat(v);
            let quadratic = (1.0 / 6.0) * (skew * skew);
            let left = Mat3F64::IDENTITY + 0.5 * skew + quadratic;
            let right = Mat3F64::IDENTITY - 0.5 * skew + quadratic;
            assert_mat_eq(&SO3F64::left_jacobian(v), &left, 1e-15 + scale.powi(3));
            assert_mat_eq(&SO3F64::right_jacobian(v), &right, 1e-15 + scale.powi(3));
        }
    }

    #[test]
    fn left_jacobian_is_right_jacobian_transpose() {
        let v = Vec3F64::new(0.1, 0.2, 0.3);
        let jl = SO3F64::left_jacobian(v);
        let jr = SO3F64::right_jacobian(v);
        assert_mat_eq(&jl, &jr.transpose(), 1e-12);
    }

    #[test]
    fn mul_and_inverse() {
        let x = SO3F64::from_random();
        let identity = (x.inverse() * x).matrix();
        assert_mat_eq(&identity, &Mat3F64::IDENTITY, 1e-12);
    }

    #[test]
    fn from_matrix_roundtrip() {
        let r = SO3F64::exp(Vec3F64::new(0.4, -0.1, 0.8));
        let r2 = SO3F64::from_matrix(&r.matrix());
        assert_mat_eq(&r.matrix(), &r2.matrix(), 1e-12);
    }

    #[test]
    fn try_from_matrix_identity() {
        let r = SO3F64::try_from_matrix(&Mat3F64::IDENTITY, 1e-12).unwrap();
        assert_mat_eq(&r.matrix(), &Mat3F64::IDENTITY, 1e-12);
    }

    #[test]
    fn try_from_matrix_roundtrip() {
        let original = SO3F64::exp(Vec3F64::new(0.3, -0.5, 0.2));
        let mat = original.matrix();
        let recovered = SO3F64::try_from_matrix(&mat, 1e-12).unwrap();
        assert_mat_eq(&recovered.matrix(), &mat, 1e-12);
    }

    #[test]
    fn try_from_matrix_known_rotation() {
        // Analytical reference: +90° rotation around Z axis.
        // Quaternion convention: [x, y, z, w] = [0, 0, sin(45°), cos(45°)] = [0, 0, 1/√2, 1/√2].
        // Cross-checked against standard SO(3) exponential map: exp([0, 0, π/2]^)
        // and Eigen's Geometry module documentation (Eigen::AngleAxisd and Eigen::Quaterniond):
        // https://libeigen.gitlab.io/eigen/docs-master/classEigen_1_1AngleAxis.html
        // https://libeigen.gitlab.io/eigen/docs-master/classEigen_1_1Quaternion.html
        // Eigen::AngleAxisd(pi/2, Vector3d::UnitZ()) analytically maps to
        // Quaterniond(cos(theta/2), axis.x * sin(theta/2), axis.y * sin(theta/2), axis.z * sin(theta/2)),
        // which gives [x, y, z, w] = [0, 0, 1/sqrt(2), 1/sqrt(2)].
        let expected_val = std::f64::consts::FRAC_1_SQRT_2;
        let mat = Mat3F64::from_rows_array(&[0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        let r = SO3F64::try_from_matrix(&mat, 1e-12).unwrap();
        assert_mat_eq(&r.matrix(), &mat, 1e-12);

        let q = r.q;
        assert!(q.x.abs() < 1e-12);
        assert!(q.y.abs() < 1e-12);
        assert!((q.z.abs() - expected_val).abs() < 1e-12);
        assert!((q.w.abs() - expected_val).abs() < 1e-12);
    }

    #[test]
    fn try_from_matrix_rejects_reflection_with_not_proper() {
        let reflection = Mat3F64::from_rows_array(&[-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let err = SO3F64::try_from_matrix(&reflection, 1e-6).unwrap_err();
        match err {
            RotationError::NotProper { det } => {
                assert!((det - (-1.0)).abs() < 1e-12);
            }
            _ => panic!("expected NotProper, got {err:?}"),
        }

        // Even with a large tolerance (e.g. tol = 2.0 or 10.0 where |det - 1| = 2.0 <= tol),
        // non-positive determinants must still be rejected with RotationError::NotProper.
        for large_tol in [2.0, 10.0] {
            let err_large = SO3F64::try_from_matrix(&reflection, large_tol).unwrap_err();
            match err_large {
                RotationError::NotProper { det } => {
                    assert!((det - (-1.0)).abs() < 1e-12);
                }
                _ => panic!("expected NotProper for tol = {large_tol}, got {err_large:?}"),
            }
        }
    }

    #[test]
    fn try_from_matrix_rejects_non_orthonormal() {
        // Scaled identity
        let scaled = Mat3F64::from_rows_array(&[2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0]);
        let err = SO3F64::try_from_matrix(&scaled, 1e-6).unwrap_err();
        match err {
            RotationError::NotOrthonormal { max_error } => {
                // R^T R - I = 4 I - I = 3 I -> max_error = 3.0
                assert!((max_error - 3.0).abs() < 1e-12);
            }
            _ => panic!("expected NotOrthonormal, got {err:?}"),
        }

        // Shear
        let shear = Mat3F64::from_rows_array(&[1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let err = SO3F64::try_from_matrix(&shear, 1e-6).unwrap_err();
        assert!(matches!(err, RotationError::NotOrthonormal { .. }));
    }

    #[test]
    fn try_from_matrix_rejects_non_finite() {
        let with_nan =
            Mat3F64::from_rows_array(&[f64::NAN, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let err = SO3F64::try_from_matrix(&with_nan, 1e-6).unwrap_err();
        assert_eq!(err, RotationError::NonFinite);

        let with_inf =
            Mat3F64::from_rows_array(&[f64::INFINITY, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let err = SO3F64::try_from_matrix(&with_inf, 1e-6).unwrap_err();
        assert_eq!(err, RotationError::NonFinite);
    }

    #[test]
    fn try_from_matrix_rejects_invalid_tolerance() {
        let err_neg = SO3F64::try_from_matrix(&Mat3F64::IDENTITY, -1e-6).unwrap_err();
        assert_eq!(err_neg, RotationError::InvalidTolerance { tol: -1e-6 });

        let err_nan = SO3F64::try_from_matrix(&Mat3F64::IDENTITY, f64::NAN).unwrap_err();
        match err_nan {
            RotationError::InvalidTolerance { tol } => assert!(tol.is_nan()),
            _ => panic!("expected InvalidTolerance for NaN, got {err_nan:?}"),
        }

        let err_inf = SO3F64::try_from_matrix(&Mat3F64::IDENTITY, f64::INFINITY).unwrap_err();
        assert_eq!(
            err_inf,
            RotationError::InvalidTolerance { tol: f64::INFINITY }
        );
    }

    #[test]
    fn try_from_matrix_tolerance_boundary() {
        let mut perturbed = Mat3F64::IDENTITY.to_cols_array();
        perturbed[0] += 1e-7;
        let m_perturbed = Mat3F64::from_cols_array(&perturbed);

        assert!(SO3F64::try_from_matrix(&m_perturbed, 1e-6).is_ok());
        assert!(SO3F64::try_from_matrix(&m_perturbed, 1e-8).is_err());
    }

    #[test]
    fn try_from_matrix_rejects_overflow_intermediate() {
        // Finite entries that overflow during RᵀR (max_error becomes +Inf or NaN):
        let large_val = 1e200_f64;
        let large_mat = Mat3F64::from_rows_array(&[
            large_val, 0.0, 0.0, 0.0, large_val, 0.0, 0.0, 0.0, large_val,
        ]);
        assert!(large_mat.is_finite());
        let err = SO3F64::try_from_matrix(&large_mat, 1e-6).unwrap_err();
        assert_eq!(err, RotationError::NonFinite);

        // Matrix with f64::MAX:
        let max_mat = Mat3F64::from_rows_array(&[f64::MAX, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        assert!(max_mat.is_finite());
        let err_max = SO3F64::try_from_matrix(&max_mat, 1e-6).unwrap_err();
        assert_eq!(err_max, RotationError::NonFinite);

        // Finite matrix whose RᵀR off-diagonal dot-product produces +Inf - Inf = NaN:
        let nan_prod_mat =
            Mat3F64::from_rows_array(&[1e200, -1e200, 0.0, 1e200, 1e200, 0.0, 0.0, 0.0, 1.0]);
        assert!(nan_prod_mat.is_finite());
        let err_nan_prod = SO3F64::try_from_matrix(&nan_prod_mat, 1e-6).unwrap_err();
        assert_eq!(err_nan_prod, RotationError::NonFinite);
    }

    #[test]
    fn try_from_matrix_perturbed_scale_passes_and_quaternion_is_unit() {
        // Small scale perturbation where RᵀR - I is within tolerance and det > 0.
        let s = 1.0 + 4.9e-7;
        let scaled_perturbed = Mat3F64::from_rows_array(&[s, 0.0, 0.0, 0.0, s, 0.0, 0.0, 0.0, s]);
        let so3 = SO3F64::try_from_matrix(&scaled_perturbed, 1e-6).unwrap();
        let q = so3.q;
        let len = (q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w).sqrt();
        assert!(
            (len - 1.0).abs() < 1e-12,
            "quaternion length {} is not unit",
            len
        );
    }

    #[test]
    fn try_from_matrix_approximate_scale_does_not_panic_under_glam_assert() {
        // diag(1.01, 1.01, 1.01) has RᵀR = diag(1.0201, 1.0201, 1.0201).
        // max |RᵀR - I| = 0.0201 <= 0.04.
        // Its determinant is 1.01³ ≈ 1.0303 > 0.
        // It must pass validation at tol = 0.04 and project to a valid unit quaternion
        // without panicking under glam assertions.
        let m = Mat3F64::from_rows_array(&[1.01, 0.0, 0.0, 0.0, 1.01, 0.0, 0.0, 0.0, 1.01]);
        let so3 = SO3F64::try_from_matrix(&m, 0.04).expect("should accept scale within tol");
        let q = so3.q;
        let len = (q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w).sqrt();
        assert!(
            (len - 1.0).abs() < 1e-12,
            "quaternion length {} is not unit",
            len
        );

        // Verify inverse behavior: SO3 * SO3.inverse() == Identity
        let inv = so3.inverse();
        let prod = so3 * inv;
        let diff = (prod.q.w.abs() - 1.0).abs();
        assert!(diff < 1e-12, "inverse identity diff: {}", diff);
    }

    #[test]
    fn try_from_matrix_subnormal_column_does_not_panic_under_glam_assert() {
        // Subnormal diagonal element: 1e-320 has (1e-320)² = 0.0 (underflow in standard dot product).
        // At tol = 1.0, RᵀR - I = diag(-1.0, 0.0, 0.0), so max_error = 1.0 <= tol.
        // Determinant is 1e-320 > 0.0.
        // Scale-first normalization avoids underflow during projection, producing
        // a valid unit quaternion without panicking under glam assertions.
        let mat = Mat3F64::from_rows_array(&[1e-320, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let res = SO3F64::try_from_matrix(&mat, 1.0);
        let so3 = res.expect("scale-first normalization should succeed for subnormal column");
        let q = so3.q;
        assert!(q.x.is_finite() && q.y.is_finite() && q.z.is_finite() && q.w.is_finite());
        let len = (q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w).sqrt();
        assert!(
            (len - 1.0).abs() < 1e-12,
            "quaternion length {len} is not unit"
        );

        // Verify inverse property
        let inv = so3.inverse();
        let prod = so3 * inv;
        let diff = (prod.q.w.abs() - 1.0).abs();
        assert!(diff < 1e-12, "inverse identity diff: {diff}");
    }

    #[test]
    fn project_to_so3_distinguishes_degenerate_from_non_finite() {
        // 1. Zero column vector is DegenerateProjection, not NonFinite
        let zero_col_mat = Mat3F64::from_cols(
            Vec3F64::ZERO,
            Vec3F64::new(0.0, 1.0, 0.0),
            Vec3F64::new(0.0, 0.0, 1.0),
        );
        let err_zero = project_to_so3(&zero_col_mat).unwrap_err();
        assert_eq!(err_zero, RotationError::DegenerateProjection);

        // 2. Collinear columns (Gram-Schmidt residual zero) is DegenerateProjection
        let collinear_mat = Mat3F64::from_cols(
            Vec3F64::new(1.0, 0.0, 0.0),
            Vec3F64::new(1.0, 0.0, 0.0), // y is parallel to x
            Vec3F64::new(0.0, 0.0, 1.0),
        );
        let err_collinear = project_to_so3(&collinear_mat).unwrap_err();
        assert_eq!(err_collinear, RotationError::DegenerateProjection);

        // 3. Non-finite column is NonFinite
        let nan_col_mat = Mat3F64::from_cols(
            Vec3F64::new(f64::NAN, 0.0, 0.0),
            Vec3F64::new(0.0, 1.0, 0.0),
            Vec3F64::new(0.0, 0.0, 1.0),
        );
        let err_nan = project_to_so3(&nan_col_mat).unwrap_err();
        assert_eq!(err_nan, RotationError::NonFinite);

        let inf_col_mat = Mat3F64::from_cols(
            Vec3F64::new(1.0, 0.0, 0.0),
            Vec3F64::new(0.0, f64::INFINITY, 0.0),
            Vec3F64::new(0.0, 0.0, 1.0),
        );
        let err_inf = project_to_so3(&inf_col_mat).unwrap_err();
        assert_eq!(err_inf, RotationError::NonFinite);

        // 4. Non-finite in third column is NonFinite
        let nan_col3_mat = Mat3F64::from_cols(
            Vec3F64::new(1.0, 0.0, 0.0),
            Vec3F64::new(0.0, 1.0, 0.0),
            Vec3F64::new(0.0, 0.0, f64::NAN),
        );
        let err_nan3 = project_to_so3(&nan_col3_mat).unwrap_err();
        assert_eq!(err_nan3, RotationError::NonFinite);

        let inf_col3_mat = Mat3F64::from_cols(
            Vec3F64::new(1.0, 0.0, 0.0),
            Vec3F64::new(0.0, 1.0, 0.0),
            Vec3F64::new(0.0, 0.0, f64::INFINITY),
        );
        let err_inf3 = project_to_so3(&inf_col3_mat).unwrap_err();
        assert_eq!(err_inf3, RotationError::NonFinite);
    }
}
