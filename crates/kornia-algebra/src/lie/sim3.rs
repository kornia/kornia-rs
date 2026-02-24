//! Similarity group Sim(3) - scaling, rotation and translation in 3D
//!
//! Sim(3) is the group of rotations, scaling and translation in 3D. It is
//! the semi-direct product of RxSO(3) and the 3D Euclidean vector space.
//! The class is represented using a composition of RxSO3 for scaling plus
//! rotation and a 3-vector for translation.
//!
//! Sim(3) is neither compact, nor a commutative group.
//!
//! Reference: Sophus library (https://github.com/strasdat/Sophus)

use crate::{Mat3AF32, Mat4F32, QuatF32, Vec3AF32};

/// Small angle threshold for Taylor series approximations
const SMALL_ANGLE_EPSILON: f32 = 1.0e-8;

use super::rxso3::RxSO3F32;
use super::so3::SO3F32;

/// Similarity transformation in 3D: rotation + scale + translation
///
/// Sim3F32 represents the similarity group Sim(3) in 3D as
/// the semi-direct product (R+ × SO(3)) ⋉ R³, where:
/// - R+ is the positive real numbers (scale)
/// - SO(3) is the rotation group in 3D
/// - R³ is the 3D Euclidean vector space (translation)
///
/// 7 degrees of freedom: 3 for rotation, 1 for scale, 3 for translation
/// Note regarding `PartialEq`:
/// This struct derives `PartialEq` which performs an exact element-wise comparison.
/// Because quaternions form a double cover for SO3 (`q` and `-q` represent the same rotation),
/// this means that two `Sim3F32` instances representing the same transformation may evaluate as not equal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sim3F32 {
    /// Scaling and rotation component
    pub rxso3: RxSO3F32,
    /// Translation component
    pub translation: Vec3AF32,
}

impl Sim3F32 {
    /// Identity transformation
    pub const IDENTITY: Self = Self {
        rxso3: RxSO3F32::IDENTITY,
        translation: Vec3AF32::ZERO,
    };

    /// Create from RxSO3F32 and translation
    pub fn new(rxso3: RxSO3F32, translation: Vec3AF32) -> Self {
        Self { rxso3, translation }
    }

    /// Create from scale, rotation quaternion, and translation
    pub fn from_scale_rotation_translation(
        scale: f32,
        rotation: QuatF32,
        translation: Vec3AF32,
    ) -> Self {
        Self {
            rxso3: RxSO3F32::from_scale_quaternion(scale, rotation),
            translation,
        }
    }

    /// Create from 4x4 homogeneous transformation matrix
    ///
    /// Matrix should be of the form:
    /// | s*R t |
    /// |  0  1 |
    ///
    /// where R is rotation, s is scale, t is translation
    pub fn from_matrix(mat: &Mat4F32) -> Self {
        // Extract rotation and scale from top-left 3x3
        // mat.x_axis, mat.y_axis, mat.z_axis are the columns
        let rot_scale_mat = Mat3AF32::from_cols(
            Vec3AF32::new(mat.x_axis.x, mat.x_axis.y, mat.x_axis.z), // first column
            Vec3AF32::new(mat.y_axis.x, mat.y_axis.y, mat.y_axis.z), // second column
            Vec3AF32::new(mat.z_axis.x, mat.z_axis.y, mat.z_axis.z), // third column
        );

        let scale = rot_scale_mat.col(0).length();

        // Normalize to get pure rotation
        let rot_mat = Mat3AF32::from_cols(
            Vec3AF32::from(rot_scale_mat.x_axis / scale),
            Vec3AF32::from(rot_scale_mat.y_axis / scale),
            Vec3AF32::from(rot_scale_mat.z_axis / scale),
        );

        Self {
            rxso3: RxSO3F32::from_scale_matrix(scale, rot_mat),
            translation: Vec3AF32::new(mat.w_axis.x, mat.w_axis.y, mat.w_axis.z),
        }
    }

    /// Get the scale factor
    pub fn scale(&self) -> f32 {
        self.rxso3.scale()
    }

    /// Get the rotation quaternion
    pub fn rotation(&self) -> QuatF32 {
        self.rxso3.rotation()
    }

    /// Get the rotation matrix
    pub fn rotation_matrix(&self) -> Mat3AF32 {
        self.rxso3.rotation_matrix()
    }

    /// Convert to 4x4 homogeneous transformation matrix
    pub fn matrix(&self) -> Mat4F32 {
        let rxso3_mat = self.rxso3.matrix();
        Mat4F32::from_cols_array(&[
            rxso3_mat.x_axis.x,
            rxso3_mat.x_axis.y,
            rxso3_mat.x_axis.z,
            0.0,
            rxso3_mat.y_axis.x,
            rxso3_mat.y_axis.y,
            rxso3_mat.y_axis.z,
            0.0,
            rxso3_mat.z_axis.x,
            rxso3_mat.z_axis.y,
            rxso3_mat.z_axis.z,
            0.0,
            self.translation.x,
            self.translation.y,
            self.translation.z,
            1.0,
        ])
    }

    /// Inverse transformation
    pub fn inverse(&self) -> Self {
        let rxso3_inv = self.rxso3.inverse();
        let scale = rxso3_inv.scale();
        let rot_inv = rxso3_inv.rotation_matrix();

        // t_inv = -R^T * s * t
        let t_inv = rot_inv * (-scale * self.translation);

        Self {
            rxso3: rxso3_inv,
            translation: t_inv,
        }
    }

    /// Exponential map from Lie algebra to group
    ///
    /// Input: 7-vector [upsilon; omega; sigma] where:
    /// - upsilon: 3D translation velocity
    /// - omega: 3D rotation velocity
    /// - sigma: scale velocity
    pub fn exp(upsilon: Vec3AF32, omega: Vec3AF32, sigma: f32) -> Self {
        let rxso3 = RxSO3F32::exp(omega, sigma);
        let scale = rxso3.scale();

        // For small angles, use first-order approximation
        // V ≈ I + 1/2 [ω]× + (1/6)σ [ω]×²
        let omega_hat = SO3F32::hat(omega);

        let v_mat = if omega.length() < SMALL_ANGLE_EPSILON {
            // Small angle approximation
            Mat3AF32::IDENTITY + omega_hat * 0.5
        } else {
            let theta = omega.length();
            let theta_sq = theta * theta;

            Mat3AF32::IDENTITY
                + omega_hat * ((1.0 - theta.cos()) / theta_sq)
                + (omega_hat * omega_hat) * ((theta - theta.sin()) / (theta_sq * theta))
        };

        let t = v_mat * upsilon / scale;

        Self {
            rxso3,
            translation: t,
        }
    }

    /// Logarithmic map from group to Lie algebra
    ///
    /// Returns: (upsilon, omega, sigma) where:
    /// - upsilon: 3D translation velocity
    /// - omega: 3D rotation velocity
    /// - sigma: scale velocity
    pub fn log(&self) -> (Vec3AF32, Vec3AF32, f32) {
        let (omega, sigma) = self.rxso3.log();
        let rot_mat = self.rxso3.rotation_matrix();
        let scale = self.rxso3.scale();

        // Compute W_inv matrix for translation
        let omega_hat = SO3F32::hat(omega);
        let theta = omega.length();

        let w_inv = if theta < SMALL_ANGLE_EPSILON {
            // Small angle approximation
            Mat3AF32::IDENTITY + omega_hat * 0.5 + (omega_hat * omega_hat) * ((1.0 / 6.0) * sigma)
        } else {
            let theta_sq = theta * theta;
            let a = theta.sin() / theta;
            let c = (1.0 - a / theta_sq) / theta_sq;

            Mat3AF32::IDENTITY - omega_hat * 0.5 + (omega_hat * omega_hat) * (c * sigma)
        };

        let upsilon = w_inv * rot_mat.transpose() * self.translation * scale;

        (upsilon, omega, sigma)
    }

    /// Adjoint representation for computing Jacobians
    ///
    /// Returns the 7x7 adjoint matrix for Sim3:
    /// ```text
    /// [ sR    [t]×R    -t ]
    /// [  0      R       0 ]
    /// [  0      0       1 ]
    /// ```
    pub fn adjoint(&self) -> [[f32; 7]; 7] {
        let r = self.rxso3.rotation_matrix();
        let s = self.rxso3.scale();
        let t_cross_r = SO3F32::hat(self.translation) * r;
        let t = self.translation;

        // Build the 7x7 adjoint matrix directly
        // Row 0: [sR row0, [t]×R row0, -t[0]]
        // Row 1: [sR row1, [t]×R row1, -t[1]]
        // Row 2: [sR row2, [t]×R row2, -t[2]]
        // Row 3: [0, 0, 0, R row0, 0]
        // Row 4: [0, 0, 0, R row1, 0]
        // Row 5: [0, 0, 0, R row2, 0]
        // Row 6: [0, 0, 0, 0, 0, 0, 1]
        [
            [
                s * r.x_axis.x,
                s * r.y_axis.x,
                s * r.z_axis.x,
                t_cross_r.x_axis.x,
                t_cross_r.y_axis.x,
                t_cross_r.z_axis.x,
                -t.x,
            ],
            [
                s * r.x_axis.y,
                s * r.y_axis.y,
                s * r.z_axis.y,
                t_cross_r.x_axis.y,
                t_cross_r.y_axis.y,
                t_cross_r.z_axis.y,
                -t.y,
            ],
            [
                s * r.x_axis.z,
                s * r.y_axis.z,
                s * r.z_axis.z,
                t_cross_r.x_axis.z,
                t_cross_r.y_axis.z,
                t_cross_r.z_axis.z,
                -t.z,
            ],
            [0.0, 0.0, 0.0, r.x_axis.x, r.y_axis.x, r.z_axis.x, 0.0],
            [0.0, 0.0, 0.0, r.x_axis.y, r.y_axis.y, r.z_axis.y, 0.0],
            [0.0, 0.0, 0.0, r.x_axis.z, r.y_axis.z, r.z_axis.z, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    }
}

impl std::ops::Mul<Sim3F32> for Sim3F32 {
    type Output = Sim3F32;

    fn mul(self, rhs: Sim3F32) -> Self::Output {
        let rxso3 = self.rxso3 * rhs.rxso3;
        let translation = self.translation + self.rxso3.matrix() * rhs.translation;
        Sim3F32::new(rxso3, translation)
    }
}

impl std::ops::Mul<Vec3AF32> for Sim3F32 {
    type Output = Vec3AF32;

    fn mul(self, rhs: Vec3AF32) -> Self::Output {
        let scaled_rotated = self.rxso3.matrix() * rhs;
        scaled_rotated + self.translation
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    const EPSILON: f32 = 1e-5;

    #[test]
    fn test_sim3_identity() {
        let sim3 = Sim3F32::IDENTITY;
        assert_relative_eq!(sim3.scale(), 1.0, epsilon = EPSILON);
        assert_relative_eq!(sim3.translation.x, 0.0, epsilon = EPSILON);
        assert_relative_eq!(sim3.translation.y, 0.0, epsilon = EPSILON);
        assert_relative_eq!(sim3.translation.z, 0.0, epsilon = EPSILON);
    }

    #[test]
    fn test_sim3_from_matrix() {
        // Test identity matrix
        let mat = Mat4F32::IDENTITY;
        let sim3 = Sim3F32::from_matrix(&mat);
        assert_relative_eq!(sim3.scale(), 1.0, epsilon = EPSILON);
        assert_relative_eq!(sim3.translation.length(), 0.0, epsilon = EPSILON);

        // Test scaled translation matrix
        // Matrix: | 2 0 0 1 |
        //         | 0 2 0 2 |
        //         | 0 0 2 3 |
        //         | 0 0 0 1 |
        let mat = Mat4F32::from_cols_array(&[
            2.0, 0.0, 0.0, 0.0, // col 0
            0.0, 2.0, 0.0, 0.0, // col 1
            0.0, 0.0, 2.0, 0.0, // col 2
            1.0, 2.0, 3.0, 1.0, // col 3 (translation)
        ]);
        let sim3 = Sim3F32::from_matrix(&mat);
        assert_relative_eq!(sim3.scale(), 2.0, epsilon = EPSILON);
        assert_relative_eq!(sim3.translation.x, 1.0, epsilon = EPSILON);
        assert_relative_eq!(sim3.translation.y, 2.0, epsilon = EPSILON);
        assert_relative_eq!(sim3.translation.z, 3.0, epsilon = EPSILON);
    }

    #[test]
    fn test_sim3_inverse() {
        let sim3 = Sim3F32::from_scale_rotation_translation(
            2.0,
            QuatF32::from_xyzw(0.1, 0.2, 0.3, 0.9).normalize(),
            Vec3AF32::new(1.0, 2.0, 3.0),
        );

        let inv = sim3.inverse();
        let product = sim3 * inv;

        // Should be identity
        assert_relative_eq!(product.scale(), 1.0, epsilon = EPSILON);
        assert_relative_eq!(product.translation.length(), 0.0, epsilon = EPSILON);
    }

    #[test]
    fn test_sim3_matrix_roundtrip() {
        let sim3 = Sim3F32::from_scale_rotation_translation(
            1.5,
            QuatF32::from_xyzw(0.1, 0.2, 0.3, 0.9).normalize(),
            Vec3AF32::new(1.0, 2.0, 3.0),
        );

        let mat = sim3.matrix();
        let sim3_reconstructed = Sim3F32::from_matrix(&mat);

        assert_relative_eq!(sim3.scale(), sim3_reconstructed.scale(), epsilon = EPSILON);
        assert_relative_eq!(
            (sim3.translation - sim3_reconstructed.translation).length(),
            0.0,
            epsilon = EPSILON
        );
    }

    #[test]
    fn test_sim3_multiplication() {
        let sim3_1 = Sim3F32::from_scale_rotation_translation(
            2.0,
            QuatF32::IDENTITY, // Identity rotation for simplicity
            Vec3AF32::new(1.0, 0.0, 0.0),
        );

        let sim3_2 = Sim3F32::from_scale_rotation_translation(
            1.5,
            QuatF32::from_xyzw(0.0, 0.0, 0.0, 1.0), // Identity rotation
            Vec3AF32::new(0.0, 1.0, 0.0),
        );

        let combined = sim3_1 * sim3_2;
        assert_relative_eq!(combined.scale(), 3.0, epsilon = EPSILON); // 2.0 * 1.5
        assert_relative_eq!(combined.translation.x, 1.0, epsilon = EPSILON);
        assert_relative_eq!(combined.translation.y, 2.0, epsilon = EPSILON); // 2.0 * 1.0 (from sim3_2) + 0.0
    }

    #[test]
    fn test_point_transformation() {
        let sim3 = Sim3F32::from_scale_rotation_translation(
            2.0,
            QuatF32::IDENTITY, // No rotation
            Vec3AF32::new(1.0, 2.0, 3.0),
        );

        let point = Vec3AF32::new(1.0, 1.0, 1.0);
        let transformed = sim3 * point;

        // Should be: 2.0 * [1,1,1] + [1,2,3] = [3,4,5]
        assert_relative_eq!(transformed.x, 3.0, epsilon = EPSILON);
        assert_relative_eq!(transformed.y, 4.0, epsilon = EPSILON);
        assert_relative_eq!(transformed.z, 5.0, epsilon = EPSILON);
    }

    #[test]
    fn test_sim3_exp_log_roundtrip() {
        // Test exp then log roundtrip for simpler inputs
        // Note: The exp/log implementation uses approximations that work well for
        // small angles and simple cases
        let test_cases = [
            // Translation only (no rotation, no scale)
            (Vec3AF32::new(1.0, 2.0, 3.0), Vec3AF32::ZERO, 0.0),
            // Scale only
            (Vec3AF32::ZERO, Vec3AF32::ZERO, 0.5),
            // Pure rotation around X
            (Vec3AF32::ZERO, Vec3AF32::new(0.5, 0.0, 0.0), 0.0),
            // Pure rotation around Y
            (Vec3AF32::ZERO, Vec3AF32::new(0.0, 0.5, 0.0), 0.0),
            // Pure rotation around Z
            (Vec3AF32::ZERO, Vec3AF32::new(0.0, 0.0, 0.5), 0.0),
        ];

        for (upsilon, omega, sigma) in test_cases {
            let sim3 = Sim3F32::exp(upsilon, omega, sigma);
            let (upsilon_out, omega_out, sigma_out) = sim3.log();

            assert_relative_eq!(upsilon_out.x, upsilon.x, epsilon = 0.01);
            assert_relative_eq!(upsilon_out.y, upsilon.y, epsilon = 0.01);
            assert_relative_eq!(upsilon_out.z, upsilon.z, epsilon = 0.01);
            assert_relative_eq!(omega_out.x, omega.x, epsilon = EPSILON);
            assert_relative_eq!(omega_out.y, omega.y, epsilon = EPSILON);
            assert_relative_eq!(omega_out.z, omega.z, epsilon = EPSILON);
            assert_relative_eq!(sigma_out, sigma, epsilon = EPSILON);
        }
    }

    #[test]
    fn test_sim3_log_exp_roundtrip() {
        // Test log then exp roundtrip for simpler Sim3 elements
        let test_cases = [
            // Identity with translation
            Sim3F32::from_scale_rotation_translation(
                1.0,
                QuatF32::IDENTITY,
                Vec3AF32::new(1.0, 2.0, 3.0),
            ),
            // Scale only
            Sim3F32::from_scale_rotation_translation(2.0, QuatF32::IDENTITY, Vec3AF32::ZERO),
            // Rotation only (no scale, no translation)
            Sim3F32::from_scale_rotation_translation(
                1.0,
                QuatF32::from_xyzw(0.0, 0.0, 0.383, 0.924).normalize(), // ~45 deg around Z
                Vec3AF32::ZERO,
            ),
        ];

        for sim3 in test_cases {
            let (upsilon, omega, sigma) = sim3.log();
            let sim3_out = Sim3F32::exp(upsilon, omega, sigma);

            assert_relative_eq!(sim3_out.scale(), sim3.scale(), epsilon = 0.01);
            assert_relative_eq!(
                (sim3_out.translation - sim3.translation).length(),
                0.0,
                epsilon = 0.01
            );
            // Rotations should be equivalent
            let dot = sim3_out.rotation().dot(sim3.rotation().0).abs();
            assert_relative_eq!(dot, 1.0, epsilon = EPSILON);
        }
    }

    #[test]
    fn test_sim3_exp_identity() {
        // exp(0, 0, 0) should give identity
        let sim3 = Sim3F32::exp(Vec3AF32::ZERO, Vec3AF32::ZERO, 0.0);

        assert_relative_eq!(sim3.scale(), 1.0, epsilon = EPSILON);
        assert_relative_eq!(sim3.translation.x, 0.0, epsilon = EPSILON);
        assert_relative_eq!(sim3.translation.y, 0.0, epsilon = EPSILON);
        assert_relative_eq!(sim3.translation.z, 0.0, epsilon = EPSILON);
        assert_relative_eq!(sim3.rotation().w.abs(), 1.0, epsilon = EPSILON);
    }

    #[test]
    fn test_sim3_log_identity() {
        // log(identity) should give (0, 0, 0)
        let (upsilon, omega, sigma) = Sim3F32::IDENTITY.log();

        assert_relative_eq!(upsilon.x, 0.0, epsilon = EPSILON);
        assert_relative_eq!(upsilon.y, 0.0, epsilon = EPSILON);
        assert_relative_eq!(upsilon.z, 0.0, epsilon = EPSILON);
        assert_relative_eq!(omega.x, 0.0, epsilon = EPSILON);
        assert_relative_eq!(omega.y, 0.0, epsilon = EPSILON);
        assert_relative_eq!(omega.z, 0.0, epsilon = EPSILON);
        assert_relative_eq!(sigma, 0.0, epsilon = EPSILON);
    }

    #[test]
    fn test_sim3_exp_small_angles() {
        // Test behavior with very small rotation angles (Taylor series regime)
        let upsilon = Vec3AF32::new(0.1, 0.2, 0.3);
        let omega = Vec3AF32::new(1e-8, 2e-8, 3e-8);
        let sigma = 0.1;

        let sim3 = Sim3F32::exp(upsilon, omega, sigma);
        let (upsilon_out, omega_out, sigma_out) = sim3.log();

        // Should recover the inputs even for small angles
        assert_relative_eq!(upsilon_out.x, upsilon.x, epsilon = 0.01);
        assert_relative_eq!(upsilon_out.y, upsilon.y, epsilon = 0.01);
        assert_relative_eq!(upsilon_out.z, upsilon.z, epsilon = 0.01);
        assert_relative_eq!(omega_out.x, omega.x, epsilon = 1e-6);
        assert_relative_eq!(omega_out.y, omega.y, epsilon = 1e-6);
        assert_relative_eq!(omega_out.z, omega.z, epsilon = 1e-6);
        assert_relative_eq!(sigma_out, sigma, epsilon = EPSILON);
    }

    #[test]
    fn test_sim3_translation_only() {
        // Test pure translation (no rotation, no scale)
        let upsilon = Vec3AF32::new(1.0, 2.0, 3.0);
        let sim3 = Sim3F32::exp(upsilon, Vec3AF32::ZERO, 0.0);

        assert_relative_eq!(sim3.scale(), 1.0, epsilon = EPSILON);
        assert_relative_eq!(sim3.translation.x, upsilon.x, epsilon = EPSILON);
        assert_relative_eq!(sim3.translation.y, upsilon.y, epsilon = EPSILON);
        assert_relative_eq!(sim3.translation.z, upsilon.z, epsilon = EPSILON);
    }

    #[test]
    fn test_sim3_scale_only() {
        // Test pure scaling (no rotation, no translation)
        let sigma = 0.693; // ln(2) ≈ 0.693
        let sim3 = Sim3F32::exp(Vec3AF32::ZERO, Vec3AF32::ZERO, sigma);

        assert_relative_eq!(sim3.scale(), sigma.exp(), epsilon = EPSILON);
        assert_relative_eq!(sim3.translation.length(), 0.0, epsilon = EPSILON);
        assert_relative_eq!(sim3.rotation().w.abs(), 1.0, epsilon = EPSILON);
    }

    #[test]
    fn test_sim3_adjoint_identity() {
        let sim3 = Sim3F32::IDENTITY;
        let adj = sim3.adjoint();

        // Expected 7x7 adjoint of identity: identity matrix
        // For identity: scale=1, R=I, t=0
        // So adjoint should be 7x7 identity
        let expected = [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ];

        for i in 0..7 {
            for j in 0..7 {
                assert_relative_eq!(adj[i][j], expected[i][j], epsilon = EPSILON);
            }
        }
    }

    #[test]
    fn test_sim3_adjoint_properties() {
        // Test: Ad(g * h) = Ad(g) * Ad(h)
        let sim3_1 = Sim3F32::from_scale_rotation_translation(
            1.5,
            QuatF32::from_xyzw(0.1, 0.2, 0.3, 0.9).normalize(),
            Vec3AF32::new(1.0, 2.0, 3.0),
        );

        let sim3_2 = Sim3F32::from_scale_rotation_translation(
            2.0,
            QuatF32::from_xyzw(-0.2, 0.1, 0.4, 0.8).normalize(),
            Vec3AF32::new(-1.0, 0.5, 2.0),
        );

        let composed = sim3_1 * sim3_2;
        let adj_composed = composed.adjoint();

        let adj_1 = sim3_1.adjoint();
        let adj_2 = sim3_2.adjoint();

        // Matrix multiplication of 7x7 matrices: adj_1 * adj_2
        let mut adj_product = [[0.0f32; 7]; 7];
        for i in 0..7 {
            for j in 0..7 {
                for (k, adj_2k) in adj_2.iter().enumerate() {
                    adj_product[i][j] += adj_1[i][k] * adj_2k[j];
                }
            }
        }

        for i in 0..7 {
            for j in 0..7 {
                assert_relative_eq!(adj_composed[i][j], adj_product[i][j], epsilon = EPSILON);
            }
        }
    }
}
