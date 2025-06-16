use glam::{Affine3A, Mat3A, Mat4, Quat, Vec3A};
use rand::Rng;

#[derive(Debug, Clone, Copy)]
pub struct SO3 {
    pub q: Quat,
}

impl SO3 {
    pub const IDENTITY: Self = Self { q: Quat::IDENTITY };

    // NOTE: quatenrion should be normalized
    pub fn from_quaternion(quat: Quat) -> Self {
        Self { q: quat }
    }

    pub fn from_matrix(mat: &Mat3A) -> Self {
        Self {
            q: Quat::from_mat3a(mat),
        }
    }

    pub fn from_matrix4(mat: &Mat4) -> Self {
        Self {
            q: Quat::from_mat4(mat),
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
            q: Quat::from_xyzw(x, y, z, w).normalize(), // Ensure normalization
        }
    }

    pub fn matrix(&self) -> Mat3A {
        Affine3A::from_quat(self.q).matrix3
    }

    pub fn adjoint(&self) -> Mat3A {
        self.matrix()
    }

    pub fn inverse(&self) -> Self {
        Self {
            q: self.q.inverse(),
        }
    }

    /// Lie algebra -> Lie group
    pub fn exp(v: Vec3A) -> Self {
        let theta = v.dot(v).sqrt();
        let theta_half = 0.5 * theta;

        let (w, b) = if theta != 0.0 {
            (theta_half.cos(), theta_half.sin() / theta)
        } else {
            (1.0, 0.0)
        };

        let xyz = b * v;

        Self {
            q: Quat::from_xyzw(xyz.x, xyz.y, xyz.z, w),
        }
    }

    /// Lie group -> Lie algebra
    pub fn log(&self) -> Vec3A {
        let real = self.q.w;
        let vec = Vec3A::new(self.q.x, self.q.y, self.q.z);

        let theta = vec.dot(vec).sqrt();

        if theta != 0.0 {
            2.0 * vec * real.acos() / theta
        } else {
            2.0 * vec / real
        }
    }

    /// Vector space -> Lie algebra
    pub fn hat(v: Vec3A) -> Mat3A {
        let (a, b, c) = (v.x, v.y, v.z);
        Mat3A::from_cols_array(&[0.0, c, -b, -c, 0.0, a, b, -a, 0.0])
    }

    /// Lie algebra -> vector space
    pub fn vee(omega: Mat3A) -> Vec3A {
        let a = omega.y_axis.z;
        let b = omega.z_axis.x;
        let c = omega.x_axis.y;
        Vec3A::new(a, b, c)
    }

    pub fn vee4(omega: Mat4) -> Vec3A {
        let a = omega.y_axis.z;
        let b = omega.z_axis.x;
        let c = omega.x_axis.y;
        Vec3A::new(a, b, c)
    }

    pub fn left_jacobian(v: Vec3A) -> Mat3A {
        let skew = Self::hat(v);
        let theta = v.dot(v).sqrt();
        let ident = Mat3A::IDENTITY;

        ident
            + ((1.0 - theta.cos()) / theta.powi(2)) * skew
            + ((theta - theta.sin()) / theta.powi(3)) * (skew * skew)
    }

    pub fn right_jacobian(v: Vec3A) -> Mat3A {
        let skew = Self::hat(v);
        let theta = v.dot(v).sqrt();
        let ident = Mat3A::IDENTITY;

        ident - ((1.0 - theta.cos()) / theta.powi(2)) * skew
            + ((theta - theta.sin()) / theta.powi(3)) * (skew * skew)
    }
}

impl std::ops::Mul<SO3> for SO3 {
    type Output = SO3;

    fn mul(self, rhs: Self) -> Self::Output {
        Self { q: self.q * rhs.q }
    }
}

impl std::ops::Mul<Vec3A> for SO3 {
    type Output = Vec3A;

    fn mul(self, rhs: Vec3A) -> Self::Output {
        let quat = Quat::from_xyzw(rhs.x, rhs.y, rhs.z, 0.0);
        let out = self.q * quat * self.q.conjugate();
        Vec3A::new(out.x, out.y, out.z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    const EPSILON: f32 = 1e-6;

    #[test]
    fn test_identity() {
        let s = SO3::IDENTITY;
        let q_expected = Quat::from_xyzw(0.0, 0.0, 0.0, 1.0);
        assert_relative_eq!(s.q.x, q_expected.x, epsilon = EPSILON);
        assert_relative_eq!(s.q.y, q_expected.y, epsilon = EPSILON);
        assert_relative_eq!(s.q.z, q_expected.z, epsilon = EPSILON);
        assert_relative_eq!(s.q.w, q_expected.w, epsilon = EPSILON);
    }

    #[test]
    fn test_from_quaternion() {
        let q = Quat::from_xyzw(4.0, -2.0, 1.0, 3.5).normalize();
        let s = SO3::from_quaternion(q);
        assert_relative_eq!(s.q.x, q.x, epsilon = EPSILON);
        assert_relative_eq!(s.q.y, q.y, epsilon = EPSILON);
        assert_relative_eq!(s.q.z, q.z, epsilon = EPSILON);
        assert_relative_eq!(s.q.w, q.w, epsilon = EPSILON);
    }

    #[test]
    fn test_from_matrix() {
        let mat = Mat3A::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.6, 0.8, 0.0, -0.8, 0.6]);
        let s = SO3::from_matrix(&mat);

        let q_expected = Quat::from_xyzw(0.5, 0.0, 0.0, 1.0).normalize();
        assert_relative_eq!(s.q.x, q_expected.x, epsilon = EPSILON);
        assert_relative_eq!(s.q.y, q_expected.y, epsilon = EPSILON);
        assert_relative_eq!(s.q.z, q_expected.z, epsilon = EPSILON);
        assert_relative_eq!(s.q.w, q_expected.w, epsilon = EPSILON);
    }

    #[test]
    fn test_unit_norm() {
        // Test that quaternions maintain unit norm
        let q1 = Quat::from_xyzw(1.0, 2.0, 3.0, 4.0).normalize();
        let q2 = Quat::from_xyzw(-1.0, 0.5, -2.0, 1.5).normalize();
        let s1 = SO3::from_quaternion(q1);
        let s2 = SO3::from_quaternion(q2);
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
    fn test_exp() {
        // Test exp of zero vector is identity
        let v = Vec3A::from_array([0.0, 0.0, 0.0]);
        let s = SO3::exp(v);

        let q_expected = Quat::from_xyzw(0.0, 0.0, 0.0, 1.0);
        assert_relative_eq!(s.q.x, q_expected.x, epsilon = EPSILON);
        assert_relative_eq!(s.q.y, q_expected.y, epsilon = EPSILON);
        assert_relative_eq!(s.q.z, q_expected.z, epsilon = EPSILON);
        assert_relative_eq!(s.q.w, q_expected.w, epsilon = EPSILON);
    }

    #[test]
    fn test_log() {
        // Test 1: log of identity quaternion should be zero vector
        {
            let so3 = SO3::IDENTITY;
            let log = so3.log();

            let log_expected = Vec3A::new(0.0, 0.0, 0.0);
            assert_relative_eq!(log.x, log_expected.x, epsilon = EPSILON);
            assert_relative_eq!(log.y, log_expected.y, epsilon = EPSILON);
            assert_relative_eq!(log.z, log_expected.z, epsilon = EPSILON);
        }

        // Test 2: exp-log consistency
        {
            let so3 = SO3::exp(Vec3A::new(1.0, 0.0, 0.0));
            let log = so3.log();

            let log_expected = Vec3A::new(1.0, 0.0, 0.0);
            assert_relative_eq!(log.x, log_expected.x, epsilon = EPSILON);
            assert_relative_eq!(log.y, log_expected.y, epsilon = EPSILON);
            assert_relative_eq!(log.z, log_expected.z, epsilon = EPSILON);
        }
    }

    #[test]
    fn test_exp_log() {
        // Test exp-log consistency for various vectors
        let test_vectors = [
            Vec3A::new(0.1, 0.2, 0.3),
            Vec3A::new(1.0, 0.0, 0.0),
            Vec3A::new(0.0, 1.0, 0.0),
            Vec3A::new(0.0, 0.0, 1.0),
            Vec3A::new(-0.5, 0.3, -0.2),
        ];

        for v in test_vectors.iter() {
            let s = SO3::exp(*v);
            let log_result = s.log();
            assert_relative_eq!(log_result.x, v.x, epsilon = 1e-5);
            assert_relative_eq!(log_result.y, v.y, epsilon = 1e-5);
            assert_relative_eq!(log_result.z, v.z, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_hat() {
        let v = Vec3A::new(1.0, 2.0, 3.0);
        let hat_v = SO3::hat(v);

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
        let omega = SO3::hat(Vec3A::new(1.0, 2.0, 3.0));
        let v = SO3::vee(omega);

        let v_expected = Vec3A::new(1.0, 2.0, 3.0);
        assert_relative_eq!(v.x, v_expected.x, epsilon = EPSILON);
        assert_relative_eq!(v.y, v_expected.y, epsilon = EPSILON);
        assert_relative_eq!(v.z, v_expected.z, epsilon = EPSILON);
    }

    #[test]
    fn test_hat_vee() {
        // Test hat-vee consistency
        let test_vectors = [
            Vec3A::new(1.0, 2.0, 3.0),
            Vec3A::new(-0.5, 0.0, 1.5),
            Vec3A::new(0.1, -0.2, 0.3),
        ];

        for v in test_vectors.iter() {
            let omega = SO3::hat(*v);
            let v_recovered = SO3::vee(omega);
            assert_relative_eq!(v_recovered.x, v.x, epsilon = EPSILON);
            assert_relative_eq!(v_recovered.y, v.y, epsilon = EPSILON);
            assert_relative_eq!(v_recovered.z, v.z, epsilon = EPSILON);
        }
    }

    #[test]
    fn test_matrix() {
        // Test rotation matrix properties
        let q = Quat::from_xyzw(0.1, 0.2, 0.3, 0.9).normalize();
        let so3 = SO3::from_quaternion(q);
        let r = so3.matrix();

        // Test that it's orthogonal (R^T * R = I)
        let identity = r.transpose() * r;
        let identity_expected = Mat3A::IDENTITY;

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
        let q1 = Quat::IDENTITY;
        let q2 = Quat::from_xyzw(0.1, 0.2, 0.3, 0.9).normalize();
        let s1 = SO3::from_quaternion(q1);
        let s2 = SO3::from_quaternion(q2);

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
    fn test_mul_vec() {
        // Test rotation of vectors
        let s1 = SO3::IDENTITY;
        let t = Vec3A::new(1.0, 2.0, 3.0);

        // Identity rotation should not change the vector
        let result = s1 * t;
        assert_relative_eq!(result.x, t.x, epsilon = EPSILON);
        assert_relative_eq!(result.y, t.y, epsilon = EPSILON);
        assert_relative_eq!(result.z, t.z, epsilon = EPSILON);

        // Test that rotation preserves vector length
        let q = Quat::from_xyzw(0.1, 0.2, 0.3, 0.9).normalize();
        let s2 = SO3::from_quaternion(q);
        let rotated = s2 * t;
        assert_relative_eq!(rotated.length(), t.length(), epsilon = 1e-5);
    }

    #[test]
    fn test_inverse() {
        // Test inverse properties
        let q = Quat::from_xyzw(0.1, 0.2, 0.3, 0.9).normalize();
        let so3 = SO3::from_quaternion(q);

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
        let q1 = Quat::from_xyzw(0.1, 0.2, 0.3, 0.9).normalize();
        let q2 = Quat::from_xyzw(-0.2, 0.1, 0.4, 0.8).normalize();
        let x = SO3::from_quaternion(q1);
        let y = SO3::from_quaternion(q2);

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
            Vec3A::new(0.1, 0.2, 0.3),
            Vec3A::new(0.01, 0.02, 0.03), // Small angles
            Vec3A::new(1.0, 0.0, 0.0),
            Vec3A::new(0.0, 1.0, 0.0),
            Vec3A::new(0.0, 0.0, 1.0),
        ];

        for v in test_vectors.iter() {
            let jl = SO3::left_jacobian(*v);

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
            Vec3A::new(0.1, 0.2, 0.3),
            Vec3A::new(0.01, 0.02, 0.03), // Small angles
            Vec3A::new(-0.1, 0.3, 0.2),
        ];

        for v in test_vectors.iter() {
            let jr = SO3::right_jacobian(*v);

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
        let test_vectors = [Vec3A::new(0.1, 0.2, 0.3), Vec3A::new(-0.05, 0.15, -0.1)];

        for v in test_vectors.iter() {
            let jr = SO3::right_jacobian(*v);
            let jl = SO3::left_jacobian(*v);

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
            let so3 = SO3::from_random();

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
            Quat::IDENTITY,
            Quat::from_xyzw(0.1, 0.2, 0.3, 0.9).normalize(),
            Quat::from_xyzw(-0.5, 0.3, -0.2, 0.7).normalize(),
        ];

        for q in test_quaternions.iter() {
            let so3 = SO3::from_quaternion(*q);
            let r = so3.matrix();
            let r_inv = so3.inverse().matrix();

            // Test R * R^T = I
            let identity = r * r.transpose();
            let expected_identity = Mat3A::IDENTITY;

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
