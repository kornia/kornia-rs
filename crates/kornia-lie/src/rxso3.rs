//! RxSO3 - Direct product of positive scalars and rotations in 3D
//!
//! RxSO3 represents R+ × SO3, the direct product of positive scalars and rotations.
//! It combines scaling and rotation operations, used as a component of Sim3.
//!
//! Reference: Sophus library (https://github.com/strasdat/Sophus)

use glam::{Mat3A, Quat};

/// Scaling and rotation component: R₊ × SO(3)
///
/// RxSO3 represents the direct product of positive real numbers (scaling)
/// and 3D rotations. Internally stored as a quaternion where:
/// - Rotation: normalized part of quaternion
/// - Scale: ||quaternion||²
#[derive(Debug, Clone, Copy)]
pub struct RxSO3 {
    /// Quaternion representing rotation, scale = ||q||²
    pub quaternion: Quat,
}

impl RxSO3 {
    /// Identity: no rotation, unit scale
    pub const IDENTITY: Self = Self {
        quaternion: Quat::IDENTITY,
    };

    /// Create from scale and quaternion
    pub fn from_scale_quaternion(scale: f32, rotation: Quat) -> Self {
        // Normalize quaternion and incorporate scale into its norm
        let rot_norm = rotation.length();
        let combined_norm = (scale * rot_norm).sqrt();
        let q_normalized = rotation / rot_norm;
        let q_scaled = q_normalized * combined_norm;

        Self {
            quaternion: q_scaled,
        }
    }

    /// Create from scale and rotation matrix
    pub fn from_scale_matrix(scale: f32, rotation: Mat3A) -> Self {
        let q = Quat::from_mat3a(&rotation);
        Self::from_scale_quaternion(scale, q)
    }

    /// Get the scale factor
    pub fn scale(&self) -> f32 {
        self.quaternion.length_squared()
    }

    /// Get the rotation quaternion (normalized)
    pub fn rotation(&self) -> Quat {
        self.quaternion / self.quaternion.length()
    }

    /// Get the rotation matrix
    pub fn rotation_matrix(&self) -> Mat3A {
        use glam::Affine3A;
        Affine3A::from_quat(self.rotation()).matrix3
    }

    /// Get the 3x3 transformation matrix (scale * rotation)
    pub fn matrix(&self) -> Mat3A {
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
            quaternion: self.quaternion.conjugate() / norm_sq,
        }
    }

    /// Exponential map for RxSO3
    pub fn exp(omega: glam::Vec3A, sigma: f32) -> Self {
        use crate::so3::SO3;
        let so3 = SO3::exp(omega);
        let rot_q = so3.q;

        // Scale factor from exponential: exp(sigma)
        let scale_factor = sigma.exp();

        Self::from_scale_quaternion(scale_factor, rot_q)
    }

    /// Logarithmic map for RxSO3
    /// Returns (omega, sigma) where omega is rotation vector, sigma is log scale
    pub fn log(&self) -> (glam::Vec3A, f32) {
        use crate::so3::SO3;
        let scale = self.scale();
        let rotation_q = self.rotation();

        let so3 = SO3::from_quaternion(rotation_q);
        let omega = so3.log();

        // Log of scale: log(s)
        let sigma = scale.ln();

        (omega, sigma)
    }
}

/// RxSO3 * RxSO3 composition
impl std::ops::Mul for RxSO3 {
    type Output = RxSO3;

    fn mul(self, rhs: RxSO3) -> Self::Output {
        // Quaternion multiplication with saturation to avoid scale becoming too small
        let result_q = self.quaternion * rhs.quaternion;
        let min_scale = 1e-6; // Minimum allowed scale

        if result_q.length_squared() < min_scale {
            // Saturation: keep minimum scale
            let current_norm = result_q.length();
            let target_norm = min_scale.sqrt();
            Self {
                quaternion: result_q * (target_norm / current_norm),
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
        let rxso3 = RxSO3::IDENTITY;
        assert_relative_eq!(rxso3.scale(), 1.0, epsilon = EPSILON);
        assert_relative_eq!(rxso3.rotation().length(), 1.0, epsilon = EPSILON);
    }

    #[test]
    fn test_rxso3_from_scale_quaternion() {
        let scale = 2.0;
        let rotation = Quat::from_xyzw(0.1, 0.2, 0.3, 0.9).normalize();
        let rxso3 = RxSO3::from_scale_quaternion(scale, rotation);

        assert_relative_eq!(rxso3.scale(), scale, epsilon = EPSILON);
        assert_relative_eq!(rxso3.rotation().dot(rotation), 1.0, epsilon = EPSILON);
    }

    #[test]
    fn test_rxso3_inverse() {
        let rxso3 = RxSO3::from_scale_quaternion(2.0, Quat::from_xyzw(0.1, 0.2, 0.3, 0.9).normalize());
        let inv = rxso3.inverse();

        assert_relative_eq!(inv.scale(), 1.0 / rxso3.scale(), epsilon = EPSILON);
    }

    #[test]
    fn test_rxso3_multiplication() {
        let rxso3_1 = RxSO3::from_scale_quaternion(2.0, Quat::IDENTITY);
        let rxso3_2 = RxSO3::from_scale_quaternion(1.5, Quat::IDENTITY);

        let combined = rxso3_1 * rxso3_2;
        assert_relative_eq!(combined.scale(), 3.0, epsilon = EPSILON);
    }
}
