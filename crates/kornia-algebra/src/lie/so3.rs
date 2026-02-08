use crate::{
    param::{Param, ParamError},
    Mat3AF32, Mat4F32, QuatF32, Vec3AF32,
};
use rand::Rng;
const SMALL_ANGLE_EPSILON: f32 = 1.0e-8;

#[derive(Debug, Clone, Copy)]
pub struct SO3F32 {
    pub q: QuatF32,
}

impl SO3F32 {
    pub const IDENTITY: Self = Self {
        q: QuatF32::IDENTITY,
    };

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
        let v: glam::Vec3A = v.into();
        Self {
            q: QuatF32(glam::Quat::from_scaled_axis(v.into())),
        }
    }

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
            let scale = 2.0 / w;
            vec * scale
        }
    }

    pub fn hat(v: Vec3AF32) -> Mat3AF32 {
        let (a, b, c) = (v.x, v.y, v.z);
        Mat3AF32::from_cols_array(&[0.0, c, -b, -c, 0.0, a, b, -a, 0.0])
    }

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
        let q = self.q.0;
        let v: glam::Vec3A = rhs.into();
        let result = q * v;
        result.into()
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
        let y = x.rplus(tau);
        let diff = x.rminus(&y);
        assert_relative_eq!(diff.x, tau.x, epsilon = EPSILON);
        assert_relative_eq!(diff.y, tau.y, epsilon = EPSILON);
        assert_relative_eq!(diff.z, tau.z, epsilon = EPSILON);
    }

    #[test]
    fn test_so3_lplus_lminus_consistency() {
        let x = SO3F32::from_random();
        let tau = Vec3AF32::new(-0.3, 1.1, 0.2);
        let y = SO3F32::lplus(tau, &x);
        let diff = SO3F32::lminus(&y, &x);
        assert_relative_eq!(diff.x, tau.x, epsilon = EPSILON);
        assert_relative_eq!(diff.y, tau.y, epsilon = EPSILON);
        assert_relative_eq!(diff.z, tau.z, epsilon = EPSILON);
    }

    #[test]
    fn test_exp() {
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
        {
            let so3 = SO3F32::IDENTITY;
            let log = so3.log();

            let log_expected = Vec3AF32::new(0.0, 0.0, 0.0);
            assert_relative_eq!(log.x, log_expected.x, epsilon = EPSILON);
            assert_relative_eq!(log.y, log_expected.y, epsilon = EPSILON);
            assert_relative_eq!(log.z, log_expected.z, epsilon = EPSILON);
        }

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
        let omega = SO3F32::hat(Vec3AF32::new(1.0, 2.0, 3.0));
        let v = SO3F32::vee(omega);

        let v_expected = Vec3AF32::new(1.0, 2.0, 3.0);
        assert_relative_eq!(v.x, v_expected.x, epsilon = EPSILON);
        assert_relative_eq!(v.y, v_expected.y, epsilon = EPSILON);
        assert_relative_eq!(v.z, v_expected.z, epsilon = EPSILON);
    }

    #[test]
    fn test_hat_vee() {
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
        let q = QuatF32::from_xyzw(0.1, 0.2, 0.3, 0.9).normalize();
        let so3 = SO3F32::from_quaternion(q);
        let r = so3.matrix();

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

        let det = r.determinant();
        assert_relative_eq!(det, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_mul() {
        let q1 = QuatF32::IDENTITY;
        let q2 = QuatF32::from_xyzw(0.1, 0.2, 0.3, 0.9).normalize();
        let s1 = SO3F32::from_quaternion(q1);
        let s2 = SO3F32::from_quaternion(q2);

        let result = s1 * s2;
        assert_relative_eq!(result.q.x, s2.q.x, epsilon = EPSILON);
        assert_relative_eq!(result.q.y, s2.q.y, epsilon = EPSILON);
        assert_relative_eq!(result.q.z, s2.q.z, epsilon = EPSILON);
        assert_relative_eq!(result.q.w, s2.q.w, epsilon = EPSILON);

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
        let s1 = SO3F32::IDENTITY;
        let t = Vec3AF32::new(1.0, 2.0, 3.0);

        let result = s1 * t;
        assert_relative_eq!(result.x, t.x, epsilon = EPSILON);
        assert_relative_eq!(result.y, t.y, epsilon = EPSILON);
        assert_relative_eq!(result.z, t.z, epsilon = EPSILON);

        let q = QuatF32::from_xyzw(0.1, 0.2, 0.3, 0.9).normalize();
        let s2 = SO3F32::from_quaternion(q);
        let rotated = s2 * t;
        assert_relative_eq!(rotated.length(), t.length(), epsilon = 1e-5);
    }

    #[test]
    fn test_inverse() {
        let q = QuatF32::from_xyzw(0.1, 0.2, 0.3, 0.9).normalize();
        let so3 = SO3F32::from_quaternion(q);

        let double_inv = so3.inverse().inverse();
        assert_relative_eq!(double_inv.q.x, so3.q.x, epsilon = 1e-5);
        assert_relative_eq!(double_inv.q.y, so3.q.y, epsilon = 1e-5);
        assert_relative_eq!(double_inv.q.z, so3.q.z, epsilon = 1e-5);
        assert_relative_eq!(double_inv.q.w.abs(), so3.q.w.abs(), epsilon = 1e-5);

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
        let q1 = QuatF32::from_xyzw(0.1, 0.2, 0.3, 0.9).normalize();
        let q2 = QuatF32::from_xyzw(-0.2, 0.1, 0.4, 0.8).normalize();
        let x = SO3F32::from_quaternion(q1);
        let y = SO3F32::from_quaternion(q2);

        let left = x.inverse().adjoint();
        let right = x.adjoint().inverse();

        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(left.col(i)[j], right.col(i)[j], epsilon = 1e-5);
            }
        }

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
            Vec3AF32::new(0.01, 0.02, 0.03),
            Vec3AF32::new(1.0, 0.0, 0.0),
            Vec3AF32::new(0.0, 1.0, 0.0),
            Vec3AF32::new(0.0, 0.0, 1.0),
        ];

        for v in test_vectors.iter() {
            let jl = SO3F32::left_jacobian(*v);

            assert!(jl.is_finite());

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
            Vec3AF32::new(0.01, 0.02, 0.03),
            Vec3AF32::new(-0.1, 0.3, 0.2),
        ];

        for v in test_vectors.iter() {
            let jr = SO3F32::right_jacobian(*v);

            assert!(jr.is_finite());

            let result = jr * *v;
            assert_relative_eq!(result.x, v.x, epsilon = 1e-4);
            assert_relative_eq!(result.y, v.y, epsilon = 1e-4);
            assert_relative_eq!(result.z, v.z, epsilon = 1e-4);
        }
    }

    #[test]
    fn test_right_left_jacobian() {
        let test_vectors = [
            Vec3AF32::new(0.1, 0.2, 0.3),
            Vec3AF32::new(-0.05, 0.15, -0.1),
        ];

        for v in test_vectors.iter() {
            let jr = SO3F32::right_jacobian(*v);
            let jl = SO3F32::left_jacobian(*v);

            let jr_transpose = jr.transpose();

            for i in 0..3 {
                for j in 0..3 {
                    assert_relative_eq!(jl.col(i)[j], jr_transpose.col(i)[j], epsilon = 0.1);
                }
            }
        }
    }

    #[test]
    fn test_random() {
        for _ in 0..10 {
            let so3 = SO3F32::from_random();

            assert_relative_eq!(so3.q.length(), 1.0, epsilon = 1e-5);

            let identity = so3.inverse() * so3;
            assert_relative_eq!(identity.q.x, 0.0, epsilon = 1e-5);
            assert_relative_eq!(identity.q.y, 0.0, epsilon = 1e-5);
            assert_relative_eq!(identity.q.z, 0.0, epsilon = 1e-5);
            assert_relative_eq!(identity.q.w.abs(), 1.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_orthogonal_matrix() {
        let test_quaternions = [
            QuatF32::IDENTITY,
            QuatF32::from_xyzw(0.1, 0.2, 0.3, 0.9).normalize(),
            QuatF32::from_xyzw(-0.5, 0.3, -0.2, 0.7).normalize(),
        ];

        for q in test_quaternions.iter() {
            let so3 = SO3F32::from_quaternion(*q);
            let r = so3.matrix();
            let r_inv = so3.inverse().matrix();

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

            let r_transpose = r.transpose();
            for i in 0..3 {
                for j in 0..3 {
                    assert_relative_eq!(r_inv.col(i)[j], r_transpose.col(i)[j], epsilon = 1e-5);
                }
            }
        }
    }
}
