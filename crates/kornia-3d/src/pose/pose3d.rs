//! Pose representation utilities.

use kornia_algebra::linalg::svd::svd3_f64;
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
    ///
    /// Assumes a rigid pose: the rotation is inverted by its transpose, which
    /// equals the inverse only when `RᵀR = I`. A rotation with scale `s`
    /// comes back with scale `s` instead of `1/s`, so a drifted pose composed
    /// with its inverse is not the identity. Call [`Pose3d::orthonormalized`]
    /// first on a pose whose [`Pose3d::rotation_defect`] is not negligible.
    pub fn inverse(&self) -> Self {
        let r_inv = self.rotation.transpose();
        Self {
            rotation: r_inv,
            translation: -(r_inv * self.translation),
        }
    }

    /// Camera centre in the world frame, `C = -Rᵀ t`.
    ///
    /// Equals `self.inverse().translation` bit for bit, so it carries the same
    /// rigid-pose assumption as [`Pose3d::inverse`].
    ///
    /// # Arguments
    ///
    /// None; reads `self`, a world-to-camera pose.
    ///
    /// # Returns
    ///
    /// The position of the camera origin in world coordinates.
    ///
    /// # Errors
    ///
    /// This method is infallible.
    ///
    /// # Example
    ///
    /// ```rust
    /// use kornia_3d::pose::Pose3d;
    /// use kornia_algebra::{Mat3F64, Vec3F64};
    ///
    /// // A camera at world (1, 2, 3) looking along +z.
    /// let pose = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(-1.0, -2.0, -3.0));
    /// assert_eq!(pose.camera_centre(), Vec3F64::new(1.0, 2.0, 3.0));
    /// ```
    pub fn camera_centre(&self) -> Vec3F64 {
        self.inverse().translation
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

    /// Largest absolute entry of `RᵀR - I`, a measure of how far the rotation
    /// is from orthonormal.
    ///
    /// Repeated composition in floating point lets a rotation drift off
    /// `SO(3)`; this reports by how much. It does not check the sign of
    /// `det R`: a reflection is orthonormal and reports `0.0`.
    ///
    /// # Returns
    ///
    /// `0.0` for an exact rotation, about `2ε` for a rotation scaled by
    /// `1 + ε`, and `f64::INFINITY` when the rotation has a non-finite entry.
    ///
    /// # Example
    ///
    /// ```
    /// use kornia_3d::pose::Pose3d;
    /// use kornia_algebra::{Mat3F64, Vec3F64};
    ///
    /// let pose = Pose3d::new(Mat3F64::IDENTITY * 1.001, Vec3F64::ZERO);
    /// assert!((pose.rotation_defect() - 2.001e-3).abs() < 1e-12);
    /// ```
    pub fn rotation_defect(&self) -> f64 {
        let residual =
            (self.rotation.transpose() * self.rotation - Mat3F64::IDENTITY).to_cols_array();
        if !residual.iter().all(|v| v.is_finite()) {
            return f64::INFINITY;
        }
        residual.iter().fold(0.0, |acc, v| acc.max(v.abs()))
    }

    /// The pose with its rotation projected back onto `SO(3)`; the
    /// translation is kept.
    ///
    /// The rotation is replaced by the nearest proper rotation in the
    /// Frobenius norm, `U·diag(1, 1, det(U·Vᵀ))·Vᵀ` from the SVD `R = U·Σ·Vᵀ`
    /// (polar projection). A rigid pose comes back unchanged up to rounding,
    /// `s·R` comes back as `R` for any `s > 0`, and a reflection comes back as
    /// the nearest rotation.
    ///
    /// # Returns
    ///
    /// A pose whose [`Pose3d::rotation_defect`] is at rounding level.
    /// A non-finite rotation gives a non-finite result.
    ///
    /// # Example
    ///
    /// ```
    /// use kornia_3d::pose::Pose3d;
    /// use kornia_algebra::{Vec3F64, SO3F64};
    ///
    /// let r = SO3F64::exp(Vec3F64::new(0.3, -0.2, 0.9)).matrix();
    /// let drifted = Pose3d::new(r * 1.001, Vec3F64::new(1.0, 2.0, 3.0));
    /// let rigid = drifted.orthonormalized();
    /// assert!(rigid.rotation_defect() < 1e-12);
    /// assert_eq!(rigid.translation, drifted.translation);
    /// ```
    pub fn orthonormalized(&self) -> Self {
        if !self.rotation.to_cols_array().iter().all(|v| v.is_finite()) {
            return *self;
        }
        let svd = svd3_f64(&self.rotation);
        let (u, v) = (svd.u(), svd.v());
        let sign = (u.determinant() * v.determinant()).signum();
        let rotation = *u * Mat3F64::from_diagonal(Vec3F64::new(1.0, 1.0, sign)) * v.transpose();
        Self::new(rotation, self.translation)
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
    use kornia_algebra::{Mat3F64, Vec3F64, SO3F64};

    #[test]
    fn test_identity() {
        let pose = Pose3d::IDENTITY;
        let point = Vec3F64::new(1.0, 2.0, 3.0);
        assert_eq!(pose.transform_point(&point), point);
    }

    #[test]
    fn camera_centre_inverts_the_world_to_camera_translation() {
        let rotation = SO3F64::exp(Vec3F64::new(0.3, -0.2, 0.9)).matrix();
        let centre = Vec3F64::new(4.0, -1.5, 12.0);
        let pose = Pose3d::new(rotation, -(rotation * centre));
        assert!((pose.camera_centre() - centre).length() < 1e-12);
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

    fn rotation() -> Mat3F64 {
        SO3F64::exp(Vec3F64::new(0.3, -0.2, 0.9)).matrix()
    }

    #[test]
    fn test_rotation_defect() {
        assert_eq!(Pose3d::IDENTITY.rotation_defect(), 0.0);
        let pose = Pose3d::new(rotation(), Vec3F64::ZERO);
        assert!(pose.rotation_defect() < 1e-14);
        // A reflection is orthonormal: the defect does not see `det R = -1`.
        let mirror = Mat3F64::from_diagonal(Vec3F64::new(1.0, 1.0, -1.0));
        assert_eq!(Pose3d::new(mirror, Vec3F64::ZERO).rotation_defect(), 0.0);
        // (1.001)^2 - 1 on the diagonal of RᵀR.
        let scaled = Pose3d::new(rotation() * 1.001, Vec3F64::ZERO).rotation_defect();
        assert!((scaled - 2.001e-3).abs() < 1e-9, "{scaled}");
        let mut broken = pose;
        broken.rotation.x_axis.x = f64::NAN;
        assert_eq!(broken.rotation_defect(), f64::INFINITY);
    }

    #[test]
    fn test_orthonormalized_restores_a_scaled_rotation() {
        let t = Vec3F64::new(4.0, -1.5, 12.0);
        for scale in [0.999, 1.001, 1.0 + 1e-7] {
            let pose = Pose3d::new(rotation() * scale, t).orthonormalized();
            assert!(pose.rotation_defect() < 1e-12, "scale {scale}");
            assert_eq!(pose.translation, t);
            let angle = SO3F64::from_matrix(&(rotation().transpose() * pose.rotation))
                .log()
                .length();
            assert!(angle < 1e-12, "scale {scale}: {angle}");
        }
    }

    #[test]
    fn test_orthonormalized_maps_a_reflection_to_a_rotation() {
        let mirror = Mat3F64::from_diagonal(Vec3F64::new(1.0, 1.0, -1.0));
        let pose = Pose3d::new(rotation() * mirror, Vec3F64::ZERO).orthonormalized();
        assert!(pose.rotation_defect() < 1e-12);
        assert!((pose.rotation.determinant() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_orthonormalized_passes_a_non_finite_rotation_through() {
        let mut pose = Pose3d::new(rotation(), Vec3F64::ZERO);
        pose.rotation.x_axis.x = f64::NAN;
        assert_eq!(pose.orthonormalized().rotation_defect(), f64::INFINITY);
    }

    #[test]
    fn test_orthonormalized_keeps_a_rigid_pose() {
        let pose = Pose3d::new(rotation(), Vec3F64::new(4.0, -1.5, 12.0));
        let rigid = pose.orthonormalized();
        assert_eq!(rigid.translation, pose.translation);
        for (a, b) in rigid
            .rotation
            .to_cols_array()
            .iter()
            .zip(pose.rotation.to_cols_array())
        {
            assert!((a - b).abs() <= 1e-14, "{a} vs {b}");
        }
    }

    #[test]
    fn test_inverse_of_a_scaled_pose_is_not_its_inverse() {
        // Documents the rigid assumption of `inverse`: the scale survives the
        // transpose, so pose ∘ pose⁻¹ carries scale² until orthonormalized.
        let pose = Pose3d::new(rotation() * 1.001, Vec3F64::new(1.0, 2.0, 3.0));
        assert!(pose.compose(&pose.inverse()).rotation_defect() > 1e-3);
        let rigid = pose.orthonormalized();
        assert!(rigid.compose(&rigid.inverse()).rotation_defect() < 1e-12);
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
