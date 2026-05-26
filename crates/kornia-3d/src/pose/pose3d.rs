//! Pose representation utilities.

use kornia_algebra::{Mat3F64, Vec3F64};

/// A rigid 3D pose that maps points from world frame to camera frame.
///
/// The transform follows:
/// `p_cam = R_cw * p_world + t_cw`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pose3d {
    /// Rotation from world frame to camera frame.
    pub rotation: Mat3F64,
    /// Translation from world frame to camera frame.
    pub translation: Vec3F64,
}

impl Pose3d {
    /// Identity world-to-camera transform.
    pub const IDENTITY: Self = Self {
        rotation: Mat3F64::IDENTITY,
        translation: Vec3F64::ZERO,
    };

    /// Create a pose from rotation and translation.
    pub const fn new(rotation: Mat3F64, translation: Vec3F64) -> Self {
        Self {
            rotation,
            translation,
        }
    }

    /// Alias constructor from `(R, t)`.
    pub const fn from_rt(rotation: Mat3F64, translation: Vec3F64) -> Self {
        Self::new(rotation, translation)
    }

    /// Convert to `(R, t)`.
    pub fn to_rt(&self) -> (Mat3F64, Vec3F64) {
        (self.rotation, self.translation)
    }

    /// Inverse transform (camera-to-world).
    pub fn inverse(&self) -> Self {
        let r_inv = self.rotation.transpose();
        Self {
            rotation: r_inv,
            translation: -(r_inv * self.translation),
        }
    }

    /// Compose two world-to-camera transforms.
    ///
    /// Returns `self ∘ rhs`:
    /// applying `rhs` first, then `self`.
    pub fn compose(&self, rhs: &Self) -> Self {
        Self {
            rotation: self.rotation * rhs.rotation,
            translation: self.rotation * rhs.translation + self.translation,
        }
    }

    /// Relative transform from `from` to `to`.
    pub fn between(from: &Self, to: &Self) -> Self {
        to.compose(&from.inverse())
    }

    /// Transform a world-frame point into camera frame.
    pub fn transform_point(&self, point_world: &Vec3F64) -> Vec3F64 {
        self.rotation * *point_world + self.translation
    }

    /// Transform a camera-frame point into world frame.
    pub fn untransform_point(&self, point_cam: &Vec3F64) -> Vec3F64 {
        self.inverse().transform_point(point_cam)
    }
}

impl Default for Pose3d {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl std::ops::Mul<Vec3F64> for Pose3d {
    type Output = Vec3F64;

    fn mul(self, rhs: Vec3F64) -> Self::Output {
        self.transform_point(&rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::Pose3d;
    use kornia_algebra::{Mat3F64, Vec3F64};

    #[test]
    fn test_identity() {
        let pose = Pose3d::IDENTITY;
        let point = Vec3F64::new(1.0, 2.0, 3.0);
        assert_eq!(pose.transform_point(&point), point);
    }

    #[test]
    fn test_inverse_roundtrip() {
        let pose = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(1.0, -2.0, 0.5));
        let inv = pose.inverse();
        let point = Vec3F64::new(0.5, 0.25, 5.0);
        let cam = pose.transform_point(&point);
        let world = inv.transform_point(&cam);
        assert!((world - point).length() < 1e-10);
    }

    #[test]
    fn test_between_matches_delta_formula() {
        let prev = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(1.0, 0.0, 0.0));
        let next = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(1.5, -0.5, 0.2));
        let delta = Pose3d::between(&prev, &next);
        let recomposed = delta.compose(&prev);
        assert!((recomposed.translation - next.translation).length() < 1e-10);
        assert_eq!(recomposed.rotation, next.rotation);
    }
}
