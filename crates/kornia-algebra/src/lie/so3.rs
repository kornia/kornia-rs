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
//! rotation manifold (Levenberg-Marquardt, Gauss-Newton). Both use Taylor series
//! approximations for small angles (θ < 1e-8) to maintain numerical stability.

use crate::{
    param::{Param, ParamError},
    Mat3AF32, Mat4F32, QuatF32, Vec3AF32,
};
use rand::Rng;
const SMALL_ANGLE_EPSILON: f32 = 1.0e-8;

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
#[derive(Debug, Clone, Copy)]
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
            let half_theta = w.acos();
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

    pub fn left_jacobian(v: Vec3AF32) -> Mat3AF32 {
        let skew = Self::hat(v);
        let theta = v.dot(v).sqrt();
        let ident = Mat3AF32::IDENTITY;

        ident
            + ((1.0 - theta.cos()) / theta.powi(2)) * skew
            + ((theta - theta.sin()) / theta.powi(3)) * (skew * skew)
    }

    pub fn right_jacobian(v: Vec3AF32) -> Mat3AF32 {
        let skew = Self::hat(v);
        let theta = v.dot(v).sqrt();
        let ident = Mat3AF32::IDENTITY;

        ident - ((1.0 - theta.cos()) / theta.powi(2)) * skew
            + ((theta - theta.sin()) / theta.powi(3)) * (skew * skew)
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    const EPSILON: f32 = 1e-6;

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
        assert_relative_eq!(diff.x, tau.x, epsilon = EPSILON);
        assert_relative_eq!(diff.y, tau.y, epsilon = EPSILON);
        assert_relative_eq!(diff.z, tau.z, epsilon = EPSILON);
    }

    #[test]
    fn test_so3_lplus_lminus_consistency() {
        let x = SO3F32::from_random();
        let tau = Vec3AF32::new(-0.3, 1.1, 0.2);
        let y = SO3F32::lplus(tau, &x); // τ ⊕ X → Y
        let diff = SO3F32::lminus(&y, &x); // Y ⊖ X → τ
        assert_relative_eq!(diff.x, tau.x, epsilon = EPSILON);
        assert_relative_eq!(diff.y, tau.y, epsilon = EPSILON);
        assert_relative_eq!(diff.z, tau.z, epsilon = EPSILON);
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
