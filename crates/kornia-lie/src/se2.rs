use crate::so2::SO2;
use glam::{Mat2, Mat3A, Vec2, Vec3A};
use rand::Rng;

#[derive(Debug, Clone, Copy)]
pub struct SE2 {
    pub r: SO2,
    pub t: Vec2,
}

impl SE2 {
    pub const IDENTITY: Self = Self {
        r: SO2::IDENTITY,
        t: Vec2::from_array([0.0, 0.0]),
    };

    pub fn new(r: SO2, t: Vec2) -> Self {
        Self { r, t }
    }

    pub fn from_matrix(mat: Mat3A) -> Self {
        Self {
            r: SO2::from_matrix3a(mat),
            t: Vec2::new(mat.z_axis.x, mat.z_axis.y),
        }
    }

    pub fn from_random() -> Self {
        let mut rng = rand::rng();

        let r1: f32 = rng.random();
        let r2: f32 = rng.random();

        Self {
            r: SO2::from_random(),
            t: Vec2::new(r1, r2),
        }
    }

    pub fn matrix(&self) -> Mat3A {
        let r = self.r.matrix();
        Mat3A::from_cols_array(&[
            r.x_axis.x, r.x_axis.y, 0.0, //
            r.y_axis.x, r.y_axis.y, 0.0, //
            self.t.x, self.t.y, 1.0, //
        ])
    }

    pub fn inverse(&self) -> Self {
        let r_inv = self.r.inverse();
        Self {
            r: r_inv,
            t: r_inv * (-self.t),
        }
    }

    pub fn adjoint(&self) -> Mat3A {
        let mut mat = self.matrix();
        mat.z_axis.x = self.t.y; // matrix[0, 2] = t.y
        mat.z_axis.y = -self.t.x; // matrix[1, 2] = -t.x
        mat
    }

    pub fn exp(upsilon: Vec2, theta: f32) -> Self {
        let so2 = SO2::exp(theta);

        Self {
            r: so2,
            t: {
                let (a, b) = if theta != 0.0 {
                    (so2.z.y / theta, (1.0 - so2.z.x) / theta)
                } else {
                    (0.0, 0.0)
                };
                Vec2::new(a * upsilon.x - b * upsilon.y, b * upsilon.x + a * upsilon.y)
            },
        }
    }

    pub fn log(&self) -> (Vec2, f32) {
        let theta = self.r.log();
        let half_theta = 0.5 * theta;
        let denom = self.r.z.x - 1.0;
        let a = if denom != 0.0 {
            -(half_theta * self.r.z.y) / denom
        } else {
            0.0
        };
        let v_inv = Mat2::from_cols_array(&[a, -half_theta, half_theta, a]);
        let upsilon = v_inv * self.t;

        (upsilon, theta)
    }

    pub fn hat(upsilon: Vec2, theta: f32) -> Mat3A {
        let hat_theta = SO2::hat(theta);
        Mat3A::from_cols(
            hat_theta.x_axis.extend(0.0).into(),
            hat_theta.y_axis.extend(0.0).into(),
            Vec3A::new(upsilon.x, upsilon.y, 0.0),
        )
    }

    pub fn vee(omega: Mat3A) -> (Vec2, f32) {
        (
            Vec2::new(omega.z_axis.x, omega.z_axis.y),
            SO2::vee(Mat2::from_cols_array(&[
                omega.x_axis.x, // [0, 0]
                omega.x_axis.y, // [1, 0]
                omega.y_axis.x, // [0, 1]
                omega.y_axis.y, // [1, 1]
            ])),
        )
    }

    #[inline]
    fn absc(theta: f32) -> (f32, f32, f32, f32) {
        if theta.abs() < 1e-6 {
            let t2 = theta * theta;
            let s = 1.0 - t2 / 6.0;
            let c = 0.0; // (1 - cos θ) / θ ≈ 0 for small θ
            let a = 0.5 - t2 / 24.0;
            let b = 0.5 - t2 / 24.0;
            (a, b, s, c)
        } else {
            let s = theta.sin() / theta;
            let c = (1.0 - theta.cos()) / theta;
            let a = c / theta;
            let b = s / theta;
            (a, b, s, c)
        }
    }

    pub fn right_jacobian(v: Vec2, theta: f32) -> Mat3A {
        let (_a, _b, s, c) = Self::absc(theta);
        let p1 = v.x;
        let p2 = v.y;

        let (third_col_x, third_col_y) = if theta.abs() < 1e-6 {
            // Limit as theta -> 0: [p2, -p1, 1]
            (p2, -p1)
        } else {
            let theta_sq = theta * theta;
            let sin_t = theta.sin();
            let cos_t = theta.cos();
            (
                (theta * p1 + p2 * (cos_t - 1.0) - p1 * sin_t) / theta_sq,
                (p1 * (sin_t - theta) + theta * p2 + p2 * cos_t) / theta_sq,
            )
        };

        Mat3A::from_cols(
            Vec3A::new(s, c, 0.0),
            Vec3A::new(-c, s, 0.0),
            Vec3A::new(third_col_x, third_col_y, 1.0),
        )
    }

    pub fn left_jacobian(v: Vec2, theta: f32) -> Mat3A {
        let (_a, _b, s, c) = Self::absc(theta);
        let p1 = v.x;
        let p2 = v.y;

        let (third_col_x, third_col_y) = if theta.abs() < 1e-6 {
            // Limit as theta -> 0: [p2, -p1, 1]
            (p2, -p1)
        } else {
            let theta_sq = theta * theta;
            let sin_t = theta.sin();
            let cos_t = theta.cos();
            (
                (theta * p1 + p2 * (cos_t - 1.0) + p1 * sin_t) / theta_sq,
                (-p1 * (sin_t - theta) + theta * p2 - p2 * cos_t) / theta_sq,
            )
        };
        Mat3A::from_cols(
            Vec3A::new(s, c, 0.0),
            Vec3A::new(-c, s, 0.0),
            Vec3A::new(third_col_x, third_col_y, 1.0),
        )
    }
}

impl std::ops::Mul<SE2> for SE2 {
    type Output = SE2;

    fn mul(self, other: SE2) -> SE2 {
        SE2::new(self.r * other.r, self.r * other.t + self.t)
    }
}

impl std::ops::Mul<Vec2> for SE2 {
    type Output = Vec2;

    fn mul(self, rhs: Vec2) -> Self::Output {
        self.r * rhs + self.t
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    const EPSILON: f32 = 1e-6;

    fn make_random_se2() -> SE2 {
        SE2::from_random()
    }

    fn make_random_vec2() -> Vec2 {
        let mut rng = rand::rng();
        Vec2::new(rng.random(), rng.random())
    }

    fn make_random_vec3() -> (Vec2, f32) {
        let mut rng = rand::rng();
        (Vec2::new(rng.random(), rng.random()), rng.random())
    }

    #[test]
    fn test_identity() {
        let identity = SE2::IDENTITY;
        assert_eq!(identity.r.z, SO2::IDENTITY.z);
        assert_eq!(identity.t, Vec2::ZERO);
    }

    #[test]
    fn test_new() {
        let rotation = SO2::exp(0.5);
        let translation = Vec2::new(1.0, 2.0);
        let se2 = SE2::new(rotation, translation);
        assert_eq!(se2.t, translation);
        assert_eq!(se2.r.z, rotation.z);
    }

    #[test]
    fn test_from_matrix() {
        // Test with identity matrix
        let mat = Mat3A::IDENTITY;
        let se2 = SE2::from_matrix(mat);
        assert_eq!(se2.t, Vec2::new(0.0, 0.0));
        assert_eq!(se2.r.matrix(), SO2::IDENTITY.matrix());

        // Test with specific transformation matrix
        let theta = std::f32::consts::PI / 4.0;
        let so2 = SO2::exp(theta);
        let translation = Vec2::new(2.0, 3.0);
        let se2_original = SE2::new(so2, translation);
        let matrix = se2_original.matrix();
        let se2_reconstructed = SE2::from_matrix(matrix);

        assert_relative_eq!(
            se2_original.r.z.x,
            se2_reconstructed.r.z.x,
            epsilon = EPSILON
        );
        assert_relative_eq!(
            se2_original.r.z.y,
            se2_reconstructed.r.z.y,
            epsilon = EPSILON
        );
        assert_relative_eq!(se2_original.t.x, se2_reconstructed.t.x, epsilon = EPSILON);
        assert_relative_eq!(se2_original.t.y, se2_reconstructed.t.y, epsilon = EPSILON);
    }

    #[test]
    fn test_matrix() {
        let se2 = SE2::new(SO2::exp(0.5), Vec2::new(1.0, 2.0));
        let mat = se2.matrix();

        // Check translation components
        assert_eq!(mat.z_axis.x, 1.0);
        assert_eq!(mat.z_axis.y, 2.0);
        assert_eq!(mat.z_axis.z, 1.0);

        // Check rotation part
        let rotation_part =
            Mat2::from_cols_array(&[mat.x_axis.x, mat.x_axis.y, mat.y_axis.x, mat.y_axis.y]);
        let expected_rotation = se2.r.matrix();

        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(
                    rotation_part.col(i)[j],
                    expected_rotation.col(i)[j],
                    epsilon = EPSILON
                );
            }
        }

        // Check bottom row (z-components should be [0, 0, 1])
        assert_eq!(mat.x_axis.z, 0.0);
        assert_eq!(mat.y_axis.z, 0.0);
        assert_eq!(mat.z_axis.z, 1.0);
    }

    #[test]
    fn test_inverse() {
        let se2 = SE2::new(SO2::exp(0.5), Vec2::new(1.0, 2.0));
        let inv = se2.inverse();

        // Test inverse properties
        let ri = se2.r.inverse();
        let expected_t = ri * (-se2.t);
        assert_relative_eq!(inv.t.x, expected_t.x, epsilon = EPSILON);
        assert_relative_eq!(inv.t.y, expected_t.y, epsilon = EPSILON);
        assert_eq!(inv.r.z, se2.r.inverse().z);

        // Test that se2 * se2.inverse() = identity
        let result = se2 * inv;
        assert_relative_eq!(result.r.z.x, SE2::IDENTITY.r.z.x, epsilon = EPSILON);
        assert_relative_eq!(result.r.z.y, SE2::IDENTITY.r.z.y, epsilon = EPSILON);
        assert_relative_eq!(result.t.x, SE2::IDENTITY.t.x, epsilon = EPSILON);
        assert_relative_eq!(result.t.y, SE2::IDENTITY.t.y, epsilon = EPSILON);

        // Test inverse of inverse
        let inv_inv = inv.inverse();
        assert_relative_eq!(inv_inv.r.z.x, se2.r.z.x, epsilon = EPSILON);
        assert_relative_eq!(inv_inv.r.z.y, se2.r.z.y, epsilon = EPSILON);
        assert_relative_eq!(inv_inv.t.x, se2.t.x, epsilon = EPSILON);
        assert_relative_eq!(inv_inv.t.y, se2.t.y, epsilon = EPSILON);
    }

    #[test]
    fn test_mul_se2() {
        // Test identity multiplication
        let s1 = SE2::IDENTITY;
        let s2 = make_random_se2();
        let s1_pose_s2 = s1 * s2;
        assert_relative_eq!(s1_pose_s2.r.z.x, s2.r.z.x, epsilon = EPSILON);
        assert_relative_eq!(s1_pose_s2.r.z.y, s2.r.z.y, epsilon = EPSILON);
        assert_relative_eq!(s1_pose_s2.t.x, s2.t.x, epsilon = EPSILON);
        assert_relative_eq!(s1_pose_s2.t.y, s2.t.y, epsilon = EPSILON);

        // Test inverse multiplication
        let s2_pose_s2_inv = s2 * s2.inverse();
        assert_relative_eq!(s2_pose_s2_inv.r.z.x, SE2::IDENTITY.r.z.x, epsilon = EPSILON);
        assert_relative_eq!(s2_pose_s2_inv.r.z.y, SE2::IDENTITY.r.z.y, epsilon = EPSILON);
        assert_relative_eq!(s2_pose_s2_inv.t.x, SE2::IDENTITY.t.x, epsilon = EPSILON);
        assert_relative_eq!(s2_pose_s2_inv.t.y, SE2::IDENTITY.t.y, epsilon = EPSILON);

        // Test composition formula: (R1, t1) * (R2, t2) = (R1*R2, R1*t2 + t1)
        let s3 = make_random_se2();
        let s4 = make_random_se2();
        let composed = s3 * s4;
        let expected_r = s3.r * s4.r;
        let expected_t = s3.r * s4.t + s3.t;

        assert_relative_eq!(composed.r.z.x, expected_r.z.x, epsilon = EPSILON);
        assert_relative_eq!(composed.r.z.y, expected_r.z.y, epsilon = EPSILON);
        assert_relative_eq!(composed.t.x, expected_t.x, epsilon = EPSILON);
        assert_relative_eq!(composed.t.y, expected_t.y, epsilon = EPSILON);
    }

    #[test]
    fn test_mul_vec2() {
        // Test identity transformation
        let s1 = SE2::IDENTITY;
        let v = Vec2::new(1.0, 2.0);
        let result = s1 * v;
        assert_relative_eq!(result.x, v.x, epsilon = EPSILON);
        assert_relative_eq!(result.y, v.y, epsilon = EPSILON);

        // Test transformation formula: SE2 * v = R * v + t
        let s2 = make_random_se2();
        let v2 = make_random_vec2();
        let result2 = s2 * v2;
        let expected = s2.r * v2 + s2.t;

        assert_relative_eq!(result2.x, expected.x, epsilon = EPSILON);
        assert_relative_eq!(result2.y, expected.y, epsilon = EPSILON);
    }

    #[test]
    fn test_exp() {
        // Test with specific values
        let upsilon = Vec2::new(1.0, 1.0);
        let theta = 1.0;
        let se2 = SE2::exp(upsilon, theta);

        assert_relative_eq!(se2.r.z.x, 0.5403, epsilon = 1e-3);
        assert_relative_eq!(se2.r.z.y, 0.8415, epsilon = 1e-3);
        assert_relative_eq!(se2.t.x, 0.3818, epsilon = 1e-3);
        assert_relative_eq!(se2.t.y, 1.3012, epsilon = 1e-3);

        // Test with zero rotation
        let upsilon_zero = Vec2::new(2.0, 3.0);
        let theta_zero = 0.0;
        let se2_zero = SE2::exp(upsilon_zero, theta_zero);

        assert_relative_eq!(se2_zero.r.z.x, 1.0, epsilon = EPSILON);
        assert_relative_eq!(se2_zero.r.z.y, 0.0, epsilon = EPSILON);
        assert_relative_eq!(se2_zero.t.x, 0.0, epsilon = EPSILON);
        assert_relative_eq!(se2_zero.t.y, 0.0, epsilon = EPSILON);

        // Test exp(0) = identity
        let se2_identity = SE2::exp(Vec2::ZERO, 0.0);
        assert_relative_eq!(se2_identity.r.z.x, SE2::IDENTITY.r.z.x, epsilon = EPSILON);
        assert_relative_eq!(se2_identity.r.z.y, SE2::IDENTITY.r.z.y, epsilon = EPSILON);
        assert_relative_eq!(se2_identity.t.x, SE2::IDENTITY.t.x, epsilon = EPSILON);
        assert_relative_eq!(se2_identity.t.y, SE2::IDENTITY.t.y, epsilon = EPSILON);
    }

    #[test]
    fn test_log() {
        // Test with specific values
        let upsilon = Vec2::new(1.0, 1.0);
        let theta = 1.0;
        let se2 = SE2::exp(upsilon, theta);
        let (log_t, log_theta) = se2.log();

        assert_relative_eq!(log_t.x, upsilon.x, epsilon = 1e-3);
        assert_relative_eq!(log_t.y, upsilon.y, epsilon = 1e-3);
        assert_relative_eq!(log_theta, theta, epsilon = 1e-3);

        // Test with another set of values
        let upsilon2 = Vec2::new(0.5 / 0.707_106_77, -0.5 / 0.707_106_77);
        let theta2 = 0.3;
        let se2_2 = SE2::exp(upsilon2, theta2);
        let (log_t2, log_theta2) = se2_2.log();

        assert_relative_eq!(log_t2.x, upsilon2.x, epsilon = 1e-5);
        assert_relative_eq!(log_t2.y, upsilon2.y, epsilon = 1e-5);
        assert_relative_eq!(log_theta2, theta2, epsilon = 1e-5);

        // Test log(identity) = 0
        let (log_identity_t, log_identity_theta) = SE2::IDENTITY.log();
        assert_relative_eq!(log_identity_t.x, 0.0, epsilon = EPSILON);
        assert_relative_eq!(log_identity_t.y, 0.0, epsilon = EPSILON);
        assert_relative_eq!(log_identity_theta, 0.0, epsilon = EPSILON);
    }

    #[test]
    fn test_exp_log_roundtrip() {
        // Test multiple random values
        for _ in 0..10 {
            let se2 = make_random_se2();
            let (upsilon, theta) = se2.log();
            let se2_exp = SE2::exp(upsilon, theta);
            let (log_upsilon, log_theta) = se2_exp.log();

            assert_relative_eq!(log_upsilon.x, upsilon.x, epsilon = EPSILON);
            assert_relative_eq!(log_upsilon.y, upsilon.y, epsilon = EPSILON);
            assert_relative_eq!(log_theta, theta, epsilon = EPSILON);
        }

        // Test specific values
        let test_cases = [
            (Vec2::new(0.0, 0.0), 0.0),
            (Vec2::new(1.0, 0.0), 0.5),
            (Vec2::new(0.0, 1.0), -0.3),
            (Vec2::new(2.0, -1.5), 1.2),
        ];

        for (upsilon, theta) in test_cases {
            let se2 = SE2::exp(upsilon, theta);
            let (log_upsilon, log_theta) = se2.log();

            assert_relative_eq!(log_upsilon.x, upsilon.x, epsilon = EPSILON);
            assert_relative_eq!(log_upsilon.y, upsilon.y, epsilon = EPSILON);
            assert_relative_eq!(log_theta, theta, epsilon = EPSILON);
        }
    }

    #[test]
    fn test_hat() {
        let upsilon = Vec2::new(1.0, 2.0);
        let theta = 0.5;
        let hat_matrix = SE2::hat(upsilon, theta);

        // Check structure: should be 3x3 matrix with specific form
        // [hat(theta)  upsilon]
        // [0     0     0     ]
        let so2_hat = SO2::hat(theta);

        // Check rotation part (top-left 2x2)
        assert_relative_eq!(hat_matrix.x_axis.x, so2_hat.x_axis.x, epsilon = EPSILON);
        assert_relative_eq!(hat_matrix.x_axis.y, so2_hat.x_axis.y, epsilon = EPSILON);
        assert_relative_eq!(hat_matrix.y_axis.x, so2_hat.y_axis.x, epsilon = EPSILON);
        assert_relative_eq!(hat_matrix.y_axis.y, so2_hat.y_axis.y, epsilon = EPSILON);

        // Check translation part (third column)
        assert_relative_eq!(hat_matrix.z_axis.x, upsilon.x, epsilon = EPSILON);
        assert_relative_eq!(hat_matrix.z_axis.y, upsilon.y, epsilon = EPSILON);

        // Check bottom row (should be [0, 0, 0] for the z components)
        assert_relative_eq!(hat_matrix.x_axis.z, 0.0, epsilon = EPSILON);
        assert_relative_eq!(hat_matrix.y_axis.z, 0.0, epsilon = EPSILON);
        assert_relative_eq!(hat_matrix.z_axis.z, 0.0, epsilon = EPSILON);
    }

    #[test]
    fn test_vee() {
        // Create a test matrix in the correct form
        let upsilon = Vec2::new(1.5, -2.3);
        let theta = 0.7;
        let omega = SE2::hat(upsilon, theta);
        let (vee_upsilon, vee_theta) = SE2::vee(omega);

        assert_relative_eq!(vee_upsilon.x, upsilon.x, epsilon = EPSILON);
        assert_relative_eq!(vee_upsilon.y, upsilon.y, epsilon = EPSILON);
        assert_relative_eq!(vee_theta, theta, epsilon = EPSILON);
    }

    #[test]
    fn test_hat_vee_roundtrip() {
        // Test with specific values
        let upsilon = Vec2::new(1.0 / 2.236_068, 2.0 / 2.236_068);
        let theta = 0.3;
        let hat_matrix = SE2::hat(upsilon, theta);
        let (vee_t, vee_theta) = SE2::vee(hat_matrix);

        assert_relative_eq!(vee_t.x, upsilon.x, epsilon = 1e-5);
        assert_relative_eq!(vee_t.y, upsilon.y, epsilon = 1e-5);
        assert_relative_eq!(vee_theta, theta, epsilon = 1e-5);

        // Test with multiple random values
        for _ in 0..10 {
            let (rand_upsilon, rand_theta) = make_random_vec3();
            let hat = SE2::hat(rand_upsilon, rand_theta);
            let (vee_upsilon, vee_theta) = SE2::vee(hat);

            assert_relative_eq!(vee_upsilon.x, rand_upsilon.x, epsilon = EPSILON);
            assert_relative_eq!(vee_upsilon.y, rand_upsilon.y, epsilon = EPSILON);
            assert_relative_eq!(vee_theta, rand_theta, epsilon = EPSILON);
        }
    }

    #[test]
    fn test_from_random() {
        let se2 = SE2::from_random();

        // Test that the rotation part is properly normalized
        let norm = se2.r.z.length();
        assert_relative_eq!(norm, 1.0, epsilon = EPSILON);

        // Test that inverse property holds
        let inv = se2.inverse();
        let result = se2 * inv;
        assert_relative_eq!(result.r.z.x, SE2::IDENTITY.r.z.x, epsilon = EPSILON);
        assert_relative_eq!(result.r.z.y, SE2::IDENTITY.r.z.y, epsilon = EPSILON);
        assert_relative_eq!(result.t.x, SE2::IDENTITY.t.x, epsilon = EPSILON);
        assert_relative_eq!(result.t.y, SE2::IDENTITY.t.y, epsilon = EPSILON);
    }

    #[test]
    fn test_adjoint() {
        // Test adjoint properties
        let x = make_random_se2();

        // Check that the adjoint has the correct structure
        let adj = x.adjoint();
        let matrix = x.matrix();

        // The adjoint should have the same rotation part as the matrix
        assert_relative_eq!(adj.x_axis.x, matrix.x_axis.x, epsilon = EPSILON);
        assert_relative_eq!(adj.x_axis.y, matrix.x_axis.y, epsilon = EPSILON);
        assert_relative_eq!(adj.y_axis.x, matrix.y_axis.x, epsilon = EPSILON);
        assert_relative_eq!(adj.y_axis.y, matrix.y_axis.y, epsilon = EPSILON);

        // But different translation part: [t.y, -t.x] instead of [t.x, t.y]
        assert_relative_eq!(adj.z_axis.x, x.t.y, epsilon = EPSILON);
        assert_relative_eq!(adj.z_axis.y, -x.t.x, epsilon = EPSILON);
    }

    #[test]
    fn test_matrix_vector_consistency() {
        let se2 = make_random_se2();
        let v = make_random_vec2();

        // Test that SE2 * v equals matrix multiplication
        let result1 = se2 * v;
        let matrix = se2.matrix();
        let v_homogeneous = Vec3A::new(v.x, v.y, 1.0);
        let result2_homogeneous = matrix * v_homogeneous;
        let result2 = Vec2::new(result2_homogeneous.x, result2_homogeneous.y);

        assert_relative_eq!(result1.x, result2.x, epsilon = EPSILON);
        assert_relative_eq!(result1.y, result2.y, epsilon = EPSILON);
    }

    #[test]
    fn test_composition_associativity() {
        let s1 = make_random_se2();
        let s2 = make_random_se2();
        let s3 = make_random_se2();

        // Test (s1 * s2) * s3 = s1 * (s2 * s3)
        let left_assoc = (s1 * s2) * s3;
        let right_assoc = s1 * (s2 * s3);

        assert_relative_eq!(left_assoc.r.z.x, right_assoc.r.z.x, epsilon = EPSILON);
        assert_relative_eq!(left_assoc.r.z.y, right_assoc.r.z.y, epsilon = EPSILON);
        assert_relative_eq!(left_assoc.t.x, right_assoc.t.x, epsilon = EPSILON);
        assert_relative_eq!(left_assoc.t.y, right_assoc.t.y, epsilon = EPSILON);
    }

    #[test]
    fn test_from_matrix_matrix_roundtrip() {
        let se2 = make_random_se2();
        let matrix = se2.matrix();
        let se2_reconstructed = SE2::from_matrix(matrix);

        // Check that we get back the same transformation
        assert_relative_eq!(se2.r.z.x, se2_reconstructed.r.z.x, epsilon = EPSILON);
        assert_relative_eq!(se2.r.z.y, se2_reconstructed.r.z.y, epsilon = EPSILON);
        assert_relative_eq!(se2.t.x, se2_reconstructed.t.x, epsilon = EPSILON);
        assert_relative_eq!(se2.t.y, se2_reconstructed.t.y, epsilon = EPSILON);
    }

    #[test]
    fn test_specific_transformations() {
        // Test pure translation
        let translation_only = SE2::new(SO2::IDENTITY, Vec2::new(3.0, 4.0));
        let point = Vec2::new(1.0, 2.0);
        let transformed = translation_only * point;
        assert_relative_eq!(transformed.x, 4.0, epsilon = EPSILON);
        assert_relative_eq!(transformed.y, 6.0, epsilon = EPSILON);

        // Test pure rotation (90 degrees)
        let rotation_only = SE2::new(SO2::exp(std::f32::consts::PI / 2.0), Vec2::ZERO);
        let point2 = Vec2::new(1.0, 0.0);
        let rotated = rotation_only * point2;
        assert_relative_eq!(rotated.x, 0.0, epsilon = EPSILON);
        assert_relative_eq!(rotated.y, 1.0, epsilon = EPSILON);

        // Test combined transformation
        let combined = SE2::new(SO2::exp(std::f32::consts::PI / 2.0), Vec2::new(1.0, 1.0));
        let transformed_combined = combined * point2;
        assert_relative_eq!(transformed_combined.x, 1.0, epsilon = EPSILON);
        assert_relative_eq!(transformed_combined.y, 2.0, epsilon = EPSILON);
    }

    #[test]
    fn test_identity_properties() {
        let se2 = make_random_se2();

        // Test left identity
        let left_result = SE2::IDENTITY * se2;
        assert_relative_eq!(left_result.r.z.x, se2.r.z.x, epsilon = EPSILON);
        assert_relative_eq!(left_result.r.z.y, se2.r.z.y, epsilon = EPSILON);
        assert_relative_eq!(left_result.t.x, se2.t.x, epsilon = EPSILON);
        assert_relative_eq!(left_result.t.y, se2.t.y, epsilon = EPSILON);

        // Test right identity
        let right_result = se2 * SE2::IDENTITY;
        assert_relative_eq!(right_result.r.z.x, se2.r.z.x, epsilon = EPSILON);
        assert_relative_eq!(right_result.r.z.y, se2.r.z.y, epsilon = EPSILON);
        assert_relative_eq!(right_result.t.x, se2.t.x, epsilon = EPSILON);
        assert_relative_eq!(right_result.t.y, se2.t.y, epsilon = EPSILON);
    }

    #[test]
    fn test_left_jacobian() {
        // Test case 1: v = (1.0, 2.0), theta = 0.5
        let v = Vec2::new(1.0, 2.0);
        let theta = 0.5;
        let jl = SE2::left_jacobian(v, theta);

        let sin_t = theta.sin();
        let cos_t = theta.cos();
        let theta_sq = theta * theta;
        let third_col_x = (theta * v.x + v.y * (cos_t - 1.0) + v.x * sin_t) / theta_sq;
        let third_col_y = (-v.x * (sin_t - theta) + theta * v.y - v.y * cos_t) / theta_sq;
        let expected_jl = Mat3A::from_cols(
            Vec3A::new(sin_t / theta, (1.0 - cos_t) / theta, 0.0),
            Vec3A::new(-(1.0 - cos_t) / theta, sin_t / theta, 0.0),
            Vec3A::new(third_col_x, third_col_y, 1.0),
        );

        for col in 0..3 {
            for row in 0..3 {
                assert_relative_eq!(
                    jl.col(col)[row],
                    expected_jl.col(col)[row],
                    epsilon = 1e-5
                );
            }
        }

        // Test case 2: v = (1.0, 2.0), theta = 0.0 (small-angle case)
        let v = Vec2::new(1.0, 2.0);
        let theta = 0.0;
        let jl = SE2::left_jacobian(v, theta);

        let expected_jl = Mat3A::from_cols(
            Vec3A::new(1.0, 0.0, 0.0),
            Vec3A::new(0.0, 1.0, 0.0),
            Vec3A::new(v.y, -v.x, 1.0), // [p2, -p1, 1]
        );

        for col in 0..3 {
            for row in 0..3 {
                assert_relative_eq!(
                    jl.col(col)[row],
                    expected_jl.col(col)[row],
                    epsilon = 1e-5
                );
            }
        }
    }

    #[test]
    fn test_right_jacobian() {
        // Test case 1: v = (1.0, 2.0), theta = 0.5
        let v = Vec2::new(1.0, 2.0);
        let theta = 0.5;
        let jr = SE2::right_jacobian(v, theta);

        let sin_t = theta.sin();
        let cos_t = theta.cos();
        let theta_sq = theta * theta;
        let third_col_x = (theta * v.x + v.y * (cos_t - 1.0) - v.x * sin_t) / theta_sq;
        let third_col_y = (v.x * (sin_t - theta) + theta * v.y + v.y * cos_t) / theta_sq;
        let expected_jr = Mat3A::from_cols(
            Vec3A::new(sin_t / theta, (1.0 - cos_t) / theta, 0.0),
            Vec3A::new(-(1.0 - cos_t) / theta, sin_t / theta, 0.0),
            Vec3A::new(third_col_x, third_col_y, 1.0),
        );

        for col in 0..3 {
            for row in 0..3 {
                assert_relative_eq!(
                    jr.col(col)[row],
                    expected_jr.col(col)[row],
                    epsilon = 1e-5
                );
            }
        }

        // Test case 2: v = (1.0, 2.0), theta = 0.0 (small-angle case)
        let v = Vec2::new(1.0, 2.0);
        let theta = 0.0;
        let jr = SE2::right_jacobian(v, theta);

        let expected_jr = Mat3A::from_cols(
            Vec3A::new(1.0, 0.0, 0.0),
            Vec3A::new(0.0, 1.0, 0.0),
            Vec3A::new(v.y, -v.x, 1.0), // [p2, -p1, 1]
        );

        for col in 0..3 {
            for row in 0..3 {
                assert_relative_eq!(
                    jr.col(col)[row],
                    expected_jr.col(col)[row],
                    epsilon = 1e-5
                );
            }
        }
    }
}
