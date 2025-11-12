use crate::{Mat2, Mat3A, Vec2};
use rand::Rng;
use std::f32::consts::TAU; // <-- Added this import

#[derive(Debug, Clone, Copy)]
pub struct SO2 {
    /// representing complex number [real, imaginary]
    pub z: Vec2,
}

impl SO2 {
    pub const IDENTITY: Self = Self {
        z: Vec2(glam::Vec2::new(1.0, 0.0)),
    };

    pub fn new(z: Vec2) -> Self {
        Self { z }
    }

    pub fn from_matrix(mat: Mat2) -> Self {
        Self {
            z: Vec2::new(mat.x_axis.x, mat.x_axis.y),
        }
    }

    pub fn from_matrix3a(mat: Mat3A) -> Self {
        Self {
            z: Vec2::new(mat.x_axis.x, mat.x_axis.y),
        }
    }

    pub fn from_random() -> Self {
        let mut rng = rand::rng();
        let theta: f32 = rng.random_range(0.0..TAU); // Sample a uniform angle
        Self::exp(theta)
    }

    #[inline]
    pub fn rplus(&self, dtheta: f32) -> Self {
        *self * SO2::exp(dtheta)
    }

    #[inline]
    pub fn rminus(&self, other: &Self) -> f32 {
        (self.inverse() * *other).log()
    }

    #[inline]
    pub fn lplus(dtheta: f32, x: &Self) -> Self {
        SO2::exp(dtheta) * *x
    }

    #[inline]
    pub fn lminus(y: &Self, x: &Self) -> f32 {
        (*y * x.inverse()).log()
    }

    pub fn matrix(&self) -> Mat2 {
        Mat2::from_cols_array(&[self.z.x, self.z.y, -self.z.y, self.z.x])
    }

    /// inverting the complex number z (represented as a 2D vector)
    /// assumes unit norm
    pub fn inverse(&self) -> Self {
        Self {
            z: Vec2::new(self.z.x, -self.z.y),
        }
    }

    pub fn adjoint(&self) -> f32 {
        1.0f32
    }

    pub fn exp(theta: f32) -> Self {
        Self {
            z: Vec2::new(theta.cos(), theta.sin()),
        }
    }

    pub fn log(&self) -> f32 {
        self.z.y.atan2(self.z.x)
    }

    pub fn hat(theta: f32) -> Mat2 {
        Mat2::from_cols_array(&[0.0, theta, -theta, 0.0])
    }

    pub fn vee(omega: Mat2) -> f32 {
        omega.x_axis.y
    }

    /// Left Jacobian of SO(2).
    /// For SO(2), the Lie group is commutative, J_l(theta) = 1.
    pub fn left_jacobian() -> f32 {
        1.0
    }

    /// Right Jacobian of SO(2).
    /// For SO(2), the Lie group is commutative, J_r(theta) = 1.
    pub fn right_jacobian() -> f32 {
        1.0
    }
}

impl std::ops::Mul<Vec2> for SO2 {
    type Output = Vec2;

    fn mul(self, rhs: Vec2) -> Self::Output {
        Vec2::new(
            self.z.x * rhs.x - self.z.y * rhs.y,
            self.z.y * rhs.x + self.z.x * rhs.y,
        )
    }
}

impl std::ops::Mul<SO2> for SO2 {
    type Output = SO2;

    fn mul(self, other: SO2) -> Self::Output {
        // Complex number multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        let real = self.z.x * other.z.x - self.z.y * other.z.y;
        let imag = self.z.x * other.z.y + self.z.y * other.z.x;
        SO2::new(Vec2::new(real, imag))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    const EPSILON: f32 = 1e-6;

    fn make_random_so2() -> SO2 {
        SO2::from_random()
    }

    #[test]
    fn test_identity() {
        let identity = SO2::IDENTITY;
        assert_eq!(identity.z, Vec2::new(1.0, 0.0));
    }

    #[test]
    fn test_new() {
        let z = Vec2::new(0.5, 0.5);
        let so2 = SO2::new(z);
        assert_eq!(so2.z, z);
    }

    #[test]
    fn test_from_matrix() {
        // Test with identity matrix
        let mat = Mat2::IDENTITY;
        let so2 = SO2::from_matrix(mat);
        assert_relative_eq!(so2.z.x, 1.0);
        assert_relative_eq!(so2.z.y, 0.0);

        // Test with specific rotation matrix
        // Mat2::from_cols_array(&[0.6, 0.8, -0.8, 0.6]) creates:
        // [0.6  -0.8]
        // [0.8   0.6]
        // x_axis = [0.6, 0.8], y_axis = [-0.8, 0.6]
        let mat = Mat2::from_cols_array(&[0.6, 0.8, -0.8, 0.6]);
        let so2 = SO2::from_matrix(mat);
        assert_relative_eq!(so2.z.x, 0.6);
        assert_relative_eq!(so2.z.y, 0.8);
    }

    #[test]
    fn test_rplus_rminus_roundtrip() {
        let x = make_random_so2();
        let dtheta = 0.42_f32; // some increment
        let y = x.rplus(dtheta); // X ⊕ τ → Y
        let diff = x.rminus(&y); // Y ⊖ X → τ
        assert_relative_eq!(diff, dtheta, epsilon = EPSILON);
    }

    #[test]
    fn test_lplus_lminus_consistency() {
        let x = make_random_so2();
        let dtheta = -1.1_f32; // another increment
        let y = SO2::lplus(dtheta, &x); // τ ⊕ X → Y
        let diff = SO2::lminus(&y, &x); // Y ⊖ X → τ
        assert_relative_eq!(diff, dtheta, epsilon = EPSILON);
    }

    #[test]
    fn test_matrix() {
        let so2 = SO2::new(Vec2::new(0.6, 0.8));
        let expected = Mat2::from_cols_array(&[0.6, 0.8, -0.8, 0.6]);
        let actual = so2.matrix();

        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(actual.col(i)[j], expected.col(i)[j]);
            }
        }
    }

    #[test]
    fn test_inverse() {
        // Test with identity
        let so2 = SO2::IDENTITY;
        let inv = so2.inverse();
        let expected = Vec2::new(1.0, 0.0);
        assert_relative_eq!(inv.z.x, expected.x);
        assert_relative_eq!(inv.z.y, expected.y);

        // Test with specific rotation
        let so2 = SO2::new(Vec2::new(0.6, 0.8));
        let inv = so2.inverse();
        let expected = Vec2::new(0.6, -0.8);
        assert_relative_eq!(inv.z.x, expected.x);
        assert_relative_eq!(inv.z.y, expected.y);

        // Test inverse property: so2 * so2.inverse() = identity
        let so2 = make_random_so2();
        let inv = so2.inverse();
        let result = so2 * inv;
        assert_relative_eq!(result.z.x, SO2::IDENTITY.z.x, epsilon = EPSILON);
        assert_relative_eq!(result.z.y, SO2::IDENTITY.z.y, epsilon = EPSILON);
    }

    #[test]
    fn test_adjoint() {
        let so2_id = SO2::IDENTITY;
        assert_relative_eq!(so2_id.adjoint(), 1.0f32);
        let so2_rand = SO2::from_random();
        assert_relative_eq!(so2_rand.adjoint(), 1.0f32);
    }

    #[test]
    fn test_exp() {
        let theta = std::f32::consts::PI / 4.0;
        let so2 = SO2::exp(theta);
        assert_relative_eq!(so2.z.x, theta.cos(), epsilon = EPSILON);
        assert_relative_eq!(so2.z.y, theta.sin(), epsilon = EPSILON);

        // Test exp(0) = identity
        let so2_identity = SO2::exp(0.0);
        assert_relative_eq!(so2_identity.z.x, 1.0);
        assert_relative_eq!(so2_identity.z.y, 0.0);
    }

    #[test]
    fn test_log() {
        let so2 = SO2::new(Vec2::new(0.6, 0.8));
        let theta = so2.log();
        assert_relative_eq!(theta, 0.9273, epsilon = 1e-4);

        // Test log(identity) = 0
        let identity_log = SO2::IDENTITY.log();
        assert_relative_eq!(identity_log, 0.0);
    }

    #[test]
    fn test_exp_log_roundtrip() {
        let theta = 0.5;
        let so2 = SO2::exp(theta);
        let log_theta = so2.log();
        assert_relative_eq!(log_theta, theta, epsilon = EPSILON);

        // Test with different angles
        let angles = [
            0.0,
            std::f32::consts::PI / 6.0,
            std::f32::consts::PI / 4.0,
            std::f32::consts::PI / 2.0,
        ];
        for &angle in &angles {
            let so2 = SO2::exp(angle);
            let recovered_angle = so2.log();
            assert_relative_eq!(recovered_angle, angle, epsilon = EPSILON);
        }
    }

    #[test]
    fn test_hat() {
        let theta = 0.5;
        let expected = Mat2::from_cols_array(&[0.0, 0.5, -0.5, 0.0]);
        let actual = SO2::hat(theta);

        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(actual.col(i)[j], expected.col(i)[j]);
            }
        }

        // Test hat structure
        let hat_matrix = SO2::hat(theta);
        assert_relative_eq!(hat_matrix.x_axis.x, 0.0);
        assert_relative_eq!(hat_matrix.y_axis.y, 0.0);
        assert_relative_eq!(hat_matrix.x_axis.y, -hat_matrix.y_axis.x);
    }

    #[test]
    fn test_vee() {
        let omega = Mat2::from_cols_array(&[0.0, 0.5, 0.5, 0.0]);
        let theta = SO2::vee(omega);
        assert_relative_eq!(theta, 0.5);
    }

    #[test]
    fn test_hat_vee_roundtrip() {
        let theta = 0.5;
        let hat_matrix = SO2::hat(theta);
        let vee_theta = SO2::vee(hat_matrix);
        assert_relative_eq!(vee_theta, theta);

        // Test with different values
        let values = [0.0, 0.1, -0.3, 1.0, -2.5];
        for &val in &values {
            let hat = SO2::hat(val);
            let vee = SO2::vee(hat);
            assert_relative_eq!(vee, val, epsilon = EPSILON);
        }
    }

    #[test]
    fn test_mul_so2() {
        let s1 = SO2::IDENTITY;
        let s2 = SO2::new(Vec2::from(glam::Vec2::new(0.9, 0.1).normalize()));

        // Test identity * s2 = s2
        let s1_pose_s2 = s1 * s2;
        assert_relative_eq!(s1_pose_s2.z.x, s2.z.x);
        assert_relative_eq!(s1_pose_s2.z.y, s2.z.y);

        // Test s2 * s2.inverse() = identity
        let s2_pose_s2_inv = s2 * s2.inverse();
        assert_relative_eq!(s2_pose_s2_inv.z.x, s1.z.x, epsilon = EPSILON);
        assert_relative_eq!(s2_pose_s2_inv.z.y, s1.z.y, epsilon = EPSILON);
    }

    #[test]
    fn test_mul_vec2() {
        let s1 = SO2::IDENTITY;
        let s2 = SO2::new(Vec2::from(glam::Vec2::new(0.9, 0.1).normalize()));
        let t1 = Vec2::new(1.0, 2.0);
        let t2 = Vec2::new(3.0, 4.0);

        // Test identity rotation doesn't change vector
        let t1_pose_s1 = s1 * t1;
        assert_relative_eq!(t1_pose_s1.x, t1.x);
        assert_relative_eq!(t1_pose_s1.y, t1.y);

        // Test specific rotation
        let t2_pose_s2 = s2 * t2;
        let expected_x = s2.z.x * t2.x - s2.z.y * t2.y;
        let expected_y = s2.z.y * t2.x + s2.z.x * t2.y;
        assert_relative_eq!(t2_pose_s2.x, expected_x);
        assert_relative_eq!(t2_pose_s2.y, expected_y);
    }

    #[test]
    fn test_matrix_vector_consistency() {
        let theta = std::f32::consts::PI / 3.0;
        let so2 = SO2::exp(theta);
        let t = Vec2::new(1.0, 2.0);

        // Test that matrix multiplication gives same result as vector multiplication
        let result1 = so2 * t;
        let matrix = so2.matrix();
        let result2 = Vec2::new(
            matrix.x_axis.x * t.x + matrix.y_axis.x * t.y,
            matrix.x_axis.y * t.x + matrix.y_axis.y * t.y,
        );

        assert_relative_eq!(result1.x, result2.x, epsilon = EPSILON);
        assert_relative_eq!(result1.y, result2.y, epsilon = EPSILON);
    }

    #[test]
    fn test_from_matrix_matrix_roundtrip() {
        let so2 = make_random_so2();
        let matrix = so2.matrix();
        let so2_reconstructed = SO2::from_matrix(matrix);

        // Check that we get back the same rotation (up to normalization)
        let norm_original = so2.z.length();
        let norm_reconstructed = so2_reconstructed.z.length();

        assert_relative_eq!(
            so2.z.x / norm_original,
            so2_reconstructed.z.x / norm_reconstructed,
            epsilon = EPSILON
        );
        assert_relative_eq!(
            so2.z.y / norm_original,
            so2_reconstructed.z.y / norm_reconstructed,
            epsilon = EPSILON
        );
    }

    #[test]
    fn test_composition_associativity() {
        let s1 = make_random_so2();
        let s2 = make_random_so2();
        let s3 = make_random_so2();

        // Test (s1 * s2) * s3 = s1 * (s2 * s3)
        let left_assoc = (s1 * s2) * s3;
        let right_assoc = s1 * (s2 * s3);

        assert_relative_eq!(left_assoc.z.x, right_assoc.z.x, epsilon = EPSILON);
        assert_relative_eq!(left_assoc.z.y, right_assoc.z.y, epsilon = EPSILON);
    }

    #[test]
    fn test_from_random() {
        let so2 = SO2::from_random();

        // Test that inverse property holds
        let inv = so2.inverse();
        let result = so2 * inv;
        assert_relative_eq!(result.z.x, SO2::IDENTITY.z.x, epsilon = EPSILON);
        assert_relative_eq!(result.z.y, SO2::IDENTITY.z.y, epsilon = EPSILON);
    }

    #[test]
    fn test_rotation_properties() {
        let theta = std::f32::consts::PI / 4.0;
        let so2 = SO2::exp(theta);

        // Test that rotation preserves vector length
        let v = Vec2::new(3.0, 4.0);
        let rotated_v = so2 * v;
        assert_relative_eq!(v.length(), rotated_v.length(), epsilon = EPSILON);

        // Test that rotation matrix is orthogonal (det = 1)
        let matrix = so2.matrix();
        let det = matrix.determinant();
        assert_relative_eq!(det, 1.0, epsilon = EPSILON);

        // Test that rotation matrix is orthogonal (R^T * R = I)
        let transpose = matrix.transpose();
        let product = Mat2::from(glam::Mat2::from(transpose) * glam::Mat2::from(matrix));
        assert_relative_eq!(product.x_axis.x, 1.0, epsilon = EPSILON);
        assert_relative_eq!(product.y_axis.y, 1.0, epsilon = EPSILON);
        assert_relative_eq!(product.x_axis.y, 0.0, epsilon = EPSILON);
        assert_relative_eq!(product.y_axis.x, 0.0, epsilon = EPSILON);
    }

    #[test]
    fn test_specific_angles() {
        // Test 90 degree rotation
        let so2_90 = SO2::exp(std::f32::consts::PI / 2.0);
        let v = Vec2::new(1.0, 0.0);
        let rotated = so2_90 * v;
        assert_relative_eq!(rotated.x, 0.0, epsilon = EPSILON);
        assert_relative_eq!(rotated.y, 1.0, epsilon = EPSILON);

        // Test 180 degree rotation
        let so2_180 = SO2::exp(std::f32::consts::PI);
        let rotated_180 = so2_180 * v;
        assert_relative_eq!(rotated_180.x, -1.0, epsilon = EPSILON);
        assert_relative_eq!(rotated_180.y, 0.0, epsilon = EPSILON);

        // Test 270 degree rotation
        let so2_270 = SO2::exp(3.0 * std::f32::consts::PI / 2.0);
        let rotated_270 = so2_270 * v;
        assert_relative_eq!(rotated_270.x, 0.0, epsilon = EPSILON);
        assert_relative_eq!(rotated_270.y, -1.0, epsilon = EPSILON);
    }

    #[test]
    fn test_basic_verification() {
        // Test that basic operations work
        let so2 = SO2::exp(std::f32::consts::PI / 4.0);

        // Test matrix roundtrip
        let matrix = so2.matrix();
        let so2_from_matrix = SO2::from_matrix(matrix);
        assert_relative_eq!(so2.z.x, so2_from_matrix.z.x, epsilon = EPSILON);
        assert_relative_eq!(so2.z.y, so2_from_matrix.z.y, epsilon = EPSILON);

        // Test inverse
        let inv = so2.inverse();
        let result = so2 * inv;
        assert_relative_eq!(result.z.x, 1.0, epsilon = EPSILON);
        assert_relative_eq!(result.z.y, 0.0, epsilon = EPSILON);

        // Test random generation
        let random_so2 = SO2::from_random();
        let norm = random_so2.z.length();
        assert_relative_eq!(norm, 1.0, epsilon = EPSILON);
    }
}
