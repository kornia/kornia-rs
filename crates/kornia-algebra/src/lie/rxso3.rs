//! RxSO3 - Direct product of positive scalars and rotations in 3D
//!
//! RxSO3 represents R+ × SO3, the direct product of positive scalars and rotations.
//! It combines scaling and rotation operations, used as a component of Sim3.
//!
//! Reference: Sophus library (https://github.com/strasdat/Sophus)

use crate::{Mat3AF32, QuatF32, Vec3AF32};

/// Minimum allowed scale to prevent numerical issues
const MIN_SCALE: f32 = 1.0e-6;

use super::so3::SO3F32;

/// Scaling and rotation component: R₊ × SO(3)
///
/// RxSO3F32 represents the direct product of positive real numbers (scaling)
/// and 3D rotations. Internally stored as a quaternion where:
/// - Rotation: normalized part of quaternion
/// - Scale: ||quaternion||²
///
/// Note regarding `PartialEq`:
///   Quaternions form a double cover for SO3, meaning `q` and `-q` represent the
///   same rotation. However, `PartialEq` performs an exact, member-wise comparison
///   and will return `false` for `q` and `-q`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RxSO3F32 {
    /// Quaternion representing rotation, scale = ||q||²
    pub quaternion: QuatF32,
}

impl RxSO3F32 {
    /// Identity: no rotation, unit scale
    pub const IDENTITY: Self = Self {
        quaternion: QuatF32::IDENTITY,
    };

    /// Create from scale and quaternion
    pub fn from_scale_quaternion(scale: f32, rotation: QuatF32) -> Self {
        // Normalize quaternion and incorporate scale into its norm
        let rot_norm = rotation.length();
        let combined_norm = (scale * rot_norm).sqrt();
        let q_normalized = rotation.0 / rot_norm;
        let q_scaled = q_normalized * combined_norm;

        Self {
            quaternion: QuatF32::from(q_scaled),
        }
    }

    /// Create from scale and rotation matrix
    pub fn from_scale_matrix(scale: f32, rotation: Mat3AF32) -> Self {
        let q = QuatF32::from_mat3a(&rotation);
        Self::from_scale_quaternion(scale, q)
    }

    /// Get the scale factor
    pub fn scale(&self) -> f32 {
        self.quaternion.length_squared()
    }

    /// Get the rotation quaternion (normalized)
    pub fn rotation(&self) -> QuatF32 {
        QuatF32::from(self.quaternion.0 / self.quaternion.length())
    }

    /// Get the rotation matrix
    pub fn rotation_matrix(&self) -> Mat3AF32 {
        Mat3AF32::from_quat(self.rotation())
    }

    /// Get the 3x3 transformation matrix (scale * rotation)
    pub fn matrix(&self) -> Mat3AF32 {
        let scale = self.scale();
        let rot_mat = self.rotation_matrix();
        rot_mat * scale
    }

    /// Inverse operation
    pub fn inverse(&self) -> Self {
        // For RxSO3, inverse is (1/scale) * R^T
        // Which corresponds to conjugate quaternion divided by norm squared
        let norm_sq = self.quaternion.length_squared();
        Self {
            quaternion: QuatF32::from(self.quaternion.conjugate().0 / norm_sq),
        }
    }

    /// Exponential map for RxSO3
    pub fn exp(omega: Vec3AF32, sigma: f32) -> Self {
        let so3 = SO3F32::exp(omega);
        let rot_q = so3.q;

        // Scale factor from exponential: exp(sigma)
        let scale_factor = sigma.exp();

        Self::from_scale_quaternion(scale_factor, rot_q)
    }

    /// Logarithmic map for RxSO3
    /// Returns (omega, sigma) where omega is rotation vector, sigma is log scale
    pub fn log(&self) -> (Vec3AF32, f32) {
        let scale = self.scale();
        let rotation_q = self.rotation();

        let so3 = SO3F32::from_quaternion(rotation_q);
        let omega = so3.log();

        // Log of scale: log(s)
        let sigma = scale.ln();

        (omega, sigma)
    }
}

/// RxSO3F32 * RxSO3F32 composition
impl std::ops::Mul for RxSO3F32 {
    type Output = RxSO3F32;

    fn mul(self, rhs: RxSO3F32) -> Self::Output {
        // Quaternion multiplication with saturation to avoid scale becoming too small
        let result_q = self.quaternion * rhs.quaternion;

        if result_q.length_squared() < MIN_SCALE {
            // Saturation: keep minimum scale
            let current_norm = result_q.length();
            let target_norm = MIN_SCALE.sqrt();
            Self {
                quaternion: QuatF32::from(result_q.0 * (target_norm / current_norm)),
            }
        } else {
            Self {
                quaternion: result_q,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    const EPSILON: f32 = 1e-5;

    #[test]
    fn test_rxso3_identity() {
        let rxso3 = RxSO3F32::IDENTITY;
        assert_relative_eq!(rxso3.scale(), 1.0, epsilon = EPSILON);
        assert_relative_eq!(rxso3.rotation().length(), 1.0, epsilon = EPSILON);
    }

    #[test]
    fn test_rxso3_from_scale_quaternion() {
        let scale = 2.0;
        let rotation = QuatF32::from_xyzw(0.1, 0.2, 0.3, 0.9).normalize();
        let rxso3 = RxSO3F32::from_scale_quaternion(scale, rotation);

        assert_relative_eq!(rxso3.scale(), scale, epsilon = EPSILON);
        assert_relative_eq!(rxso3.rotation().dot(rotation.0), 1.0, epsilon = EPSILON);
    }

    #[test]
    fn test_rxso3_inverse() {
        let rxso3 = RxSO3F32::from_scale_quaternion(
            2.0,
            QuatF32::from_xyzw(0.1, 0.2, 0.3, 0.9).normalize(),
        );
        let inv = rxso3.inverse();

        assert_relative_eq!(inv.scale(), 1.0 / rxso3.scale(), epsilon = EPSILON);
    }

    #[test]
    fn test_rxso3_multiplication() {
        let rxso3_1 = RxSO3F32::from_scale_quaternion(2.0, QuatF32::IDENTITY);
        let rxso3_2 = RxSO3F32::from_scale_quaternion(1.5, QuatF32::IDENTITY);

        let combined = rxso3_1 * rxso3_2;
        assert_relative_eq!(combined.scale(), 3.0, epsilon = EPSILON);
    }

    #[test]
    fn test_rxso3_exp_log_roundtrip() {
        // Test exp then log roundtrip for various inputs
        let test_cases = [
            (Vec3AF32::new(0.1, 0.2, 0.3), 0.5),   // General case
            (Vec3AF32::new(1.0, 0.0, 0.0), 0.0),   // Rotation only around X
            (Vec3AF32::new(0.0, 1.0, 0.0), 0.0),   // Rotation only around Y
            (Vec3AF32::new(0.0, 0.0, 1.0), 0.0),   // Rotation only around Z
            (Vec3AF32::new(0.0, 0.0, 0.0), 1.0),   // Scale only
            (Vec3AF32::new(-0.2, 0.1, 0.4), -0.3), // Mixed with negative scale log
        ];

        for (omega, sigma) in test_cases {
            let rxso3 = RxSO3F32::exp(omega, sigma);
            let (omega_out, sigma_out) = rxso3.log();

            assert_relative_eq!(omega_out.x, omega.x, epsilon = EPSILON);
            assert_relative_eq!(omega_out.y, omega.y, epsilon = EPSILON);
            assert_relative_eq!(omega_out.z, omega.z, epsilon = EPSILON);
            assert_relative_eq!(sigma_out, sigma, epsilon = EPSILON);
        }
    }

    #[test]
    fn test_rxso3_log_exp_roundtrip() {
        // Test log then exp roundtrip for various RxSO3 elements
        let test_cases = [
            RxSO3F32::from_scale_quaternion(2.0, QuatF32::IDENTITY),
            RxSO3F32::from_scale_quaternion(
                1.5,
                QuatF32::from_xyzw(0.1, 0.2, 0.3, 0.9).normalize(),
            ),
            RxSO3F32::from_scale_quaternion(
                0.5,
                QuatF32::from_xyzw(-0.1, 0.3, -0.2, 0.8).normalize(),
            ),
        ];

        for rxso3 in test_cases {
            let (omega, sigma) = rxso3.log();
            let rxso3_out = RxSO3F32::exp(omega, sigma);

            assert_relative_eq!(rxso3_out.scale(), rxso3.scale(), epsilon = EPSILON);
            // Rotations should be equivalent (dot product close to 1 or -1)
            let dot = rxso3_out.rotation().dot(rxso3.rotation().0).abs();
            assert_relative_eq!(dot, 1.0, epsilon = EPSILON);
        }
    }

    #[test]
    fn test_rxso3_exp_identity() {
        // exp(0, 0) should give identity
        let rxso3 = RxSO3F32::exp(Vec3AF32::ZERO, 0.0);

        assert_relative_eq!(rxso3.scale(), 1.0, epsilon = EPSILON);
        assert_relative_eq!(rxso3.rotation().w, 1.0, epsilon = EPSILON);
        assert_relative_eq!(rxso3.rotation().x, 0.0, epsilon = EPSILON);
        assert_relative_eq!(rxso3.rotation().y, 0.0, epsilon = EPSILON);
        assert_relative_eq!(rxso3.rotation().z, 0.0, epsilon = EPSILON);
    }

    #[test]
    fn test_rxso3_log_identity() {
        // log(identity) should give (0, 0)
        let (omega, sigma) = RxSO3F32::IDENTITY.log();

        assert_relative_eq!(omega.x, 0.0, epsilon = EPSILON);
        assert_relative_eq!(omega.y, 0.0, epsilon = EPSILON);
        assert_relative_eq!(omega.z, 0.0, epsilon = EPSILON);
        assert_relative_eq!(sigma, 0.0, epsilon = EPSILON);
    }

    #[test]
    fn test_rxso3_exp_small_angles() {
        // Test behavior with very small rotation angles (Taylor series regime)
        let omega = Vec3AF32::new(1e-8, 2e-8, 3e-8);
        let sigma = 0.1;

        let rxso3 = RxSO3F32::exp(omega, sigma);
        let (omega_out, sigma_out) = rxso3.log();

        // Should recover the inputs even for small angles
        assert_relative_eq!(omega_out.x, omega.x, epsilon = 1e-6);
        assert_relative_eq!(omega_out.y, omega.y, epsilon = 1e-6);
        assert_relative_eq!(omega_out.z, omega.z, epsilon = 1e-6);
        assert_relative_eq!(sigma_out, sigma, epsilon = EPSILON);
    }

    #[test]
    fn test_rxso3_scale_only() {
        // Test pure scaling (no rotation)
        let sigma = 0.693; // ln(2) ≈ 0.693
        let rxso3 = RxSO3F32::exp(Vec3AF32::ZERO, sigma);

        assert_relative_eq!(rxso3.scale(), sigma.exp(), epsilon = EPSILON);
        // Rotation should be identity
        assert_relative_eq!(rxso3.rotation().w.abs(), 1.0, epsilon = EPSILON);
    }
}
