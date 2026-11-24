//! Pose representation utilities.

use kornia_algebra::linalg::svd::svd3_f64;
use kornia_algebra::{Mat3F64, Mat4F64, QuatF64, Vec3F64, Vec4F64, SE3F64, SO3F64};

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

    /// Create a pose from a 4x4 transformation matrix.
    ///
    /// Extracts the 3x3 linear rotation block and 3x1 translation vector,
    /// ignoring the homogeneous bottom row `[0.0, 0.0, 0.0, 1.0]`.
    ///
    /// # Arguments
    ///
    /// * `m` - Homogeneous 4x4 transformation matrix mapping points from world frame to camera frame.
    ///
    /// # Returns
    ///
    /// A new [`Pose3d`] with rotation extracted from top-left 3x3 and translation from top-right 3x1.
    ///
    /// # Errors
    ///
    /// This method is infallible.
    ///
    /// # Example
    ///
    /// ```
    /// use kornia_3d::pose::Pose3d;
    /// use kornia_algebra::Mat4F64;
    ///
    /// let m = Mat4F64::IDENTITY;
    /// let pose = Pose3d::from_matrix(&m);
    /// assert_eq!(pose, Pose3d::IDENTITY);
    /// ```
    pub fn from_matrix(m: &Mat4F64) -> Self {
        let (cx, cy, cz, ct) = (m.x_axis(), m.y_axis(), m.z_axis(), m.w_axis());
        let rotation = Mat3F64::from_cols(
            Vec3F64::new(cx.x, cx.y, cx.z),
            Vec3F64::new(cy.x, cy.y, cy.z),
            Vec3F64::new(cz.x, cz.y, cz.z),
        );
        let translation = Vec3F64::new(ct.x, ct.y, ct.z);
        Self::new(rotation, translation)
    }

    /// Convert to a homogeneous 4x4 transformation matrix.
    ///
    /// Top-left 3x3 is rotation, top-right 3x1 is translation, and bottom row is `[0.0, 0.0, 0.0, 1.0]`.
    ///
    /// # Returns
    ///
    /// Homogeneous 4x4 transformation matrix representation of this pose.
    ///
    /// # Errors
    ///
    /// This method is infallible.
    ///
    /// # Example
    ///
    /// ```
    /// use kornia_3d::pose::Pose3d;
    /// use kornia_algebra::Mat4F64;
    ///
    /// let pose = Pose3d::IDENTITY;
    /// assert_eq!(pose.to_matrix(), Mat4F64::IDENTITY);
    /// ```
    pub fn to_matrix(&self) -> Mat4F64 {
        let (rx, ry, rz) = (
            self.rotation.x_axis(),
            self.rotation.y_axis(),
            self.rotation.z_axis(),
        );
        let t = self.translation;
        Mat4F64::from_cols(
            Vec4F64::new(rx.x, rx.y, rx.z, 0.0),
            Vec4F64::new(ry.x, ry.y, ry.z, 0.0),
            Vec4F64::new(rz.x, rz.y, rz.z, 0.0),
            Vec4F64::new(t.x, t.y, t.z, 1.0),
        )
    }

    /// Create a pose from a quaternion and a translation vector.
    ///
    /// The input quaternion is explicitly normalized before constructing the rotation
    /// matrix to ensure the rigid orthogonality invariant (`RᵀR = I`) is preserved.
    ///
    /// If `q` is non-finite or degenerate (zero norm), the resulting rotation matrix
    /// contains non-finite entries, which can be checked via [`Pose3d::rotation_defect`].
    ///
    /// # Arguments
    ///
    /// * `q` - Rotation quaternion mapping vectors from world to camera frame. Will be normalized.
    /// * `t` - Translation vector from world to camera frame.
    ///
    /// # Returns
    ///
    /// A new [`Pose3d`] representing the rigid transformation.
    ///
    /// # Errors
    ///
    /// This method is infallible. If `q` has zero or non-finite norm, the resulting
    /// pose contains non-finite entries.
    ///
    /// # Example
    ///
    /// ```
    /// use kornia_3d::pose::Pose3d;
    /// use kornia_algebra::{QuatF64, Vec3F64};
    ///
    /// let q = QuatF64::from_xyzw(0.0, 0.0, 0.0, 1.0);
    /// let t = Vec3F64::new(1.0, 2.0, 3.0);
    /// let pose = Pose3d::from_quat_translation(q, t);
    /// assert_eq!(pose.translation, t);
    /// assert!(pose.rotation_defect() < 1e-12);
    /// ```
    pub fn from_quat_translation(q: QuatF64, t: Vec3F64) -> Self {
        let max_abs = q.x.abs().max(q.y.abs()).max(q.z.abs()).max(q.w.abs());
        let q_scaled = if max_abs > 0.0 && max_abs.is_finite() {
            QuatF64::from_xyzw(q.x / max_abs, q.y / max_abs, q.z / max_abs, q.w / max_abs)
        } else {
            q
        };
        Self::new(Mat3F64::from_quat(q_scaled.normalize()), t)
    }

    /// Extract the pose as a unit quaternion and a translation vector `(q, t)`.
    ///
    /// # Returns
    ///
    /// A tuple `(q, t)` of type `(QuatF64, Vec3F64)`.
    ///
    /// # Preconditions
    ///
    /// `self.rotation` must be a valid proper rotation matrix with negligible defect
    /// (`RᵀR = I`, `det R = 1`). If the pose has accumulated numerical drift or scaling,
    /// callers should invoke [`Pose3d::orthonormalized`] first before extracting the quaternion.
    ///
    /// # Errors
    ///
    /// This method is infallible.
    ///
    /// # Example
    ///
    /// ```
    /// use kornia_3d::pose::Pose3d;
    /// use kornia_algebra::{QuatF64, Vec3F64};
    ///
    /// let pose = Pose3d::IDENTITY;
    /// let (q, t) = pose.to_quat_translation();
    /// assert_eq!(q, QuatF64::from_xyzw(0.0, 0.0, 0.0, 1.0));
    /// assert_eq!(t, Vec3F64::ZERO);
    /// ```
    pub fn to_quat_translation(&self) -> (QuatF64, Vec3F64) {
        (QuatF64::from_mat3(&self.rotation), self.translation)
    }

    /// Heading (yaw) angle in radians about the +Z axis, in `(-π, π]`.
    ///
    /// Computed via `atan2(R[1][0], R[0][0])` conforming to the standard robotics navigation
    /// convention (REP 103 / ISO 8855) where +X is forward and +Z is upward.
    ///
    /// # Returns
    ///
    /// Heading angle in radians about the +Z axis in the interval `(-π, π]`.
    ///
    /// # Errors
    ///
    /// This method is infallible.
    ///
    /// # Example
    ///
    /// ```
    /// use kornia_3d::pose::Pose3d;
    /// use std::f64::consts::FRAC_PI_2;
    /// use kornia_algebra::{Mat3F64, Vec3F64};
    ///
    /// // 90 degree rotation about +Z maps +X (1, 0, 0) to +Y (0, 1, 0)
    /// let r_z_90 = Mat3F64::from_cols(
    ///     Vec3F64::new(0.0, 1.0, 0.0),
    ///     Vec3F64::new(-1.0, 0.0, 0.0),
    ///     Vec3F64::new(0.0, 0.0, 1.0),
    /// );
    /// let pose = Pose3d::new(r_z_90, Vec3F64::ZERO);
    /// assert!((pose.yaw() - FRAC_PI_2).abs() < 1e-12);
    /// ```
    pub fn yaw(&self) -> f64 {
        self.rotation.x_axis.y.atan2(self.rotation.x_axis.x)
    }

    /// Project the pose onto the horizontal ground plane (X-Y plane).
    ///
    /// Sets the rotation to a pure planar rotation about +Z by [`Pose3d::yaw`],
    /// and drops the vertical component of translation by setting `translation.z = 0.0`.
    ///
    /// This operation is idempotent: `pose.planar().planar() == pose.planar()`.
    ///
    /// # Returns
    ///
    /// A new projected [`Pose3d`] on the ground plane.
    ///
    /// # Errors
    ///
    /// This method is infallible.
    ///
    /// # Example
    ///
    /// ```
    /// use kornia_3d::pose::Pose3d;
    /// use kornia_algebra::{Mat3F64, Vec3F64};
    ///
    /// let pose = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(1.0, 2.0, 5.0));
    /// let planar = pose.planar();
    /// assert_eq!(planar.translation, Vec3F64::new(1.0, 2.0, 0.0));
    /// assert_eq!(planar.yaw(), 0.0);
    /// ```
    pub fn planar(&self) -> Self {
        let yaw = self.yaw();
        let (sin_y, cos_y) = yaw.sin_cos();
        let rotation = Mat3F64::from_cols(
            Vec3F64::new(cos_y, sin_y, 0.0),
            Vec3F64::new(-sin_y, cos_y, 0.0),
            Vec3F64::new(0.0, 0.0, 1.0),
        );
        let translation = Vec3F64::new(self.translation.x, self.translation.y, 0.0);
        Self::new(rotation, translation)
    }

    /// Shortest geodesic rotation angle from the identity rotation in radians, in `[0, π]`.
    ///
    /// Uses the numerically stable quaternion formulation:
    /// `2.0 * atan2(||v||, |w|)`
    /// where `v` is the vector part and `w` is the scalar part of the extracted quaternion.
    ///
    /// This formulation handles the SU(2) double-cover (`q` and `-q` yield identical angles)
    /// and avoids the catastrophic cancellation and domain errors of `acos((tr(R) - 1) / 2)`
    /// near `θ ≈ 0`.
    ///
    /// # Returns
    ///
    /// The geodesic rotation distance from identity in `[0, π]` radians.
    ///
    /// # Preconditions
    ///
    /// `self.rotation` must be a valid proper rotation matrix (`RᵀR = I`, `det R = 1`).
    /// If numerical drift has accumulated, call [`Pose3d::orthonormalized`] first.
    ///
    /// # Errors
    ///
    /// This method is infallible.
    ///
    /// # Example
    ///
    /// ```
    /// use kornia_3d::pose::Pose3d;
    ///
    /// let pose = Pose3d::IDENTITY;
    /// assert_eq!(pose.rotation_angle(), 0.0);
    /// ```
    pub fn rotation_angle(&self) -> f64 {
        let q = QuatF64::from_mat3(&self.rotation);
        2.0 * q.xyz().length().atan2(q.w.abs())
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

impl From<glam::DAffine3> for Pose3d {
    /// Convert from `glam::DAffine3` to `Pose3d`.
    #[inline]
    fn from(affine: glam::DAffine3) -> Self {
        Self {
            rotation: Mat3F64::from(affine.matrix3),
            translation: Vec3F64::from(affine.translation),
        }
    }
}

impl From<Pose3d> for glam::DAffine3 {
    /// Convert from `Pose3d` to `glam::DAffine3`.
    #[inline]
    fn from(pose: Pose3d) -> Self {
        Self::from_mat3_translation(pose.rotation.into(), pose.translation.into())
    }
}

impl From<SE3F64> for Pose3d {
    /// Convert from `kornia_algebra::SE3F64` to `Pose3d`.
    #[inline]
    fn from(se3: SE3F64) -> Self {
        Self::new(se3.r.matrix(), se3.t)
    }
}

impl From<Pose3d> for SE3F64 {
    /// Convert from `Pose3d` to `kornia_algebra::SE3F64`.
    ///
    /// # Preconditions
    ///
    /// Assumes `pose.rotation` is a proper rotation matrix with negligible defect.
    /// Call [`Pose3d::orthonormalized`] first if the pose has accumulated numerical drift.
    #[inline]
    fn from(pose: Pose3d) -> Self {
        Self::new(SO3F64::from_matrix(&pose.rotation), pose.translation)
    }
}

impl From<Pose3d> for Mat4F64 {
    /// Convert from `Pose3d` to a homogeneous 4x4 matrix `Mat4F64`.
    #[inline]
    fn from(pose: Pose3d) -> Self {
        pose.to_matrix()
    }
}

#[cfg(test)]
mod tests {
    use super::Pose3d;
    use kornia_algebra::{Mat3F64, QuatF64, Vec3F64, SE3F64, SO3F64};
    use std::f64::consts::{FRAC_PI_2, PI};

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

    #[test]
    fn test_glam_daffine3_roundtrip() {
        let pose = Pose3d::new(rotation(), Vec3F64::new(1.5, -2.5, 3.5));
        let affine: glam::DAffine3 = pose.into();
        let recovered = Pose3d::from(affine);

        assert_eq!(recovered.translation, pose.translation);
        for (a, b) in recovered
            .rotation
            .to_cols_array()
            .iter()
            .zip(pose.rotation.to_cols_array())
        {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn test_matrix_roundtrip() {
        let pose = Pose3d::new(rotation(), Vec3F64::new(-4.0, 5.0, 6.0));
        let mat = pose.to_matrix();
        let recovered = Pose3d::from_matrix(&mat);

        assert_eq!(recovered.translation, pose.translation);
        for (a, b) in recovered
            .rotation
            .to_cols_array()
            .iter()
            .zip(pose.rotation.to_cols_array())
        {
            assert!((a - b).abs() < 1e-15);
        }

        // Check homogeneous bottom row
        let cols = mat.to_cols_array();
        assert_eq!(cols[3], 0.0);
        assert_eq!(cols[7], 0.0);
        assert_eq!(cols[11], 0.0);
        assert_eq!(cols[15], 1.0);
    }

    #[test]
    fn test_se3f64_roundtrip() {
        let pose = Pose3d::new(rotation(), Vec3F64::new(2.0, -1.0, 0.5));
        let se3: SE3F64 = pose.into();
        let recovered = Pose3d::from(se3);

        assert_eq!(recovered.translation, pose.translation);
        for (a, b) in recovered
            .rotation
            .to_cols_array()
            .iter()
            .zip(pose.rotation.to_cols_array())
        {
            assert!((a - b).abs() < 1e-14);
        }

        let pt = Vec3F64::new(1.0, 2.0, 3.0);
        assert!((pose.transform_point(&pt) - (se3 * pt)).length() < 1e-14);
    }

    #[test]
    fn test_quat_translation_roundtrip() {
        let pose = Pose3d::new(rotation(), Vec3F64::new(0.1, -0.2, 0.3));
        let (q, t) = pose.to_quat_translation();
        assert_eq!(t, pose.translation);

        let reconstructed = Pose3d::from_quat_translation(q, t);
        assert_eq!(reconstructed.translation, pose.translation);
        for (a, b) in reconstructed
            .rotation
            .to_cols_array()
            .iter()
            .zip(pose.rotation.to_cols_array())
        {
            assert!((a - b).abs() < 1e-14);
        }
    }

    #[test]
    fn test_from_quat_translation_normalizes_unnormalized_input() {
        let unnormalized_q = QuatF64::from_xyzw(0.0, 0.0, 3.0, 4.0); // norm = 5.0
        let t = Vec3F64::new(1.0, 2.0, 3.0);
        let pose = Pose3d::from_quat_translation(unnormalized_q, t);

        assert_eq!(pose.translation, t);
        assert!(pose.rotation_defect() < 1e-12);
        assert!((pose.rotation.determinant() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_yaw_ground_truth() {
        // Identity has zero yaw
        assert_eq!(Pose3d::IDENTITY.yaw(), 0.0);

        // Yaw +90 degrees (pi/2)
        let r_90 = SO3F64::exp(Vec3F64::new(0.0, 0.0, FRAC_PI_2)).matrix();
        let p_90 = Pose3d::new(r_90, Vec3F64::ZERO);
        assert!((p_90.yaw() - FRAC_PI_2).abs() < 1e-14);

        // Yaw -90 degrees (-pi/2)
        let r_neg_90 = SO3F64::exp(Vec3F64::new(0.0, 0.0, -FRAC_PI_2)).matrix();
        let p_neg_90 = Pose3d::new(r_neg_90, Vec3F64::ZERO);
        assert!((p_neg_90.yaw() - (-FRAC_PI_2)).abs() < 1e-14);

        // Yaw +180 degrees (pi)
        let r_180 = SO3F64::exp(Vec3F64::new(0.0, 0.0, PI)).matrix();
        let p_180 = Pose3d::new(r_180, Vec3F64::ZERO);
        assert!((p_180.yaw().abs() - PI).abs() < 1e-14);
    }

    #[test]
    fn test_yaw_with_roll_and_pitch() {
        let yaw_expected = 0.75;
        let r_z = SO3F64::exp(Vec3F64::new(0.0, 0.0, yaw_expected)).matrix();
        let r_y = SO3F64::exp(Vec3F64::new(0.0, 0.2, 0.0)).matrix(); // pitch
        let r_x = SO3F64::exp(Vec3F64::new(-0.3, 0.0, 0.0)).matrix(); // roll
        let r_combined = r_z * r_y * r_x;

        let pose = Pose3d::new(r_combined, Vec3F64::ZERO);
        assert!((pose.yaw() - yaw_expected).abs() < 1e-14);
    }

    #[test]
    fn test_planar_projection() {
        let pose = Pose3d::new(rotation(), Vec3F64::new(10.0, -20.0, 30.0));
        let planar = pose.planar();

        // Translation z is zeroed out, x and y are preserved
        assert_eq!(planar.translation.x, 10.0);
        assert_eq!(planar.translation.y, -20.0);
        assert_eq!(planar.translation.z, 0.0);

        // Yaw is preserved
        assert!((planar.yaw() - pose.yaw()).abs() < 1e-14);

        // Planar rotation has zero roll and pitch
        let r = planar.rotation;
        assert_eq!(r.x_axis.z, 0.0);
        assert_eq!(r.y_axis.z, 0.0);
        assert_eq!(r.z_axis.x, 0.0);
        assert_eq!(r.z_axis.y, 0.0);
        assert_eq!(r.z_axis.z, 1.0);

        // Idempotency: planar().planar() == planar()
        let idempotent = planar.planar();
        assert_eq!(idempotent, planar);
    }

    #[test]
    fn test_rotation_angle() {
        // Identity has angle 0
        assert_eq!(Pose3d::IDENTITY.rotation_angle(), 0.0);

        // Known 90-degree rotation about X
        let r_x_90 = SO3F64::exp(Vec3F64::new(FRAC_PI_2, 0.0, 0.0)).matrix();
        let p_x_90 = Pose3d::new(r_x_90, Vec3F64::ZERO);
        assert!((p_x_90.rotation_angle() - FRAC_PI_2).abs() < 1e-14);

        // Known 180-degree rotation about Y
        let r_y_180 = SO3F64::exp(Vec3F64::new(0.0, PI, 0.0)).matrix();
        let p_y_180 = Pose3d::new(r_y_180, Vec3F64::ZERO);
        assert!((p_y_180.rotation_angle() - PI).abs() < 1e-14);

        // Near-zero angle numerical accuracy (avoids acos catastrophic cancellation)
        let eps = 1e-9;
        let r_eps = SO3F64::exp(Vec3F64::new(0.0, 0.0, eps)).matrix();
        let p_eps = Pose3d::new(r_eps, Vec3F64::ZERO);
        assert!((p_eps.rotation_angle() - eps).abs() < 1e-15);

        // Near-pi angle numerical accuracy
        let near_pi = PI - 1e-8;
        let r_near_pi = SO3F64::exp(Vec3F64::new(0.0, 0.0, near_pi)).matrix();
        let p_near_pi = Pose3d::new(r_near_pi, Vec3F64::ZERO);
        assert!((p_near_pi.rotation_angle() - near_pi).abs() < 1e-12);
    }

    #[test]
    fn test_rotation_angle_double_cover() {
        // Construct two quaternions differing only by sign
        let q_pos = QuatF64::from_xyzw(0.0, 0.0, FRAC_PI_2.sin(), FRAC_PI_2.cos());
        let q_neg = QuatF64::from_xyzw(0.0, 0.0, -FRAC_PI_2.sin(), -FRAC_PI_2.cos());

        let p_pos = Pose3d::from_quat_translation(q_pos, Vec3F64::ZERO);
        let p_neg = Pose3d::from_quat_translation(q_neg, Vec3F64::ZERO);

        assert!((p_pos.rotation_angle() - p_neg.rotation_angle()).abs() < 1e-15);
    }

    #[test]
    fn test_drifted_rotation_orthonormalized_before_extraction() {
        let drifted = Pose3d::new(Mat3F64::IDENTITY * 1.05, Vec3F64::new(1.0, 2.0, 3.0));
        assert!(drifted.rotation_defect() > 0.05);
        let rigid = drifted.orthonormalized();
        assert!(rigid.rotation_defect() < 1e-12);
        let (q, t) = rigid.to_quat_translation();
        assert!((q.length() - 1.0).abs() < 1e-12);
        assert_eq!(t, Vec3F64::new(1.0, 2.0, 3.0));
        let se3: SE3F64 = rigid.into();
        assert_eq!(se3.t, Vec3F64::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_from_quat_translation_zero_or_non_finite() {
        let zero_q = QuatF64::from_xyzw(0.0, 0.0, 0.0, 0.0);
        let pose = Pose3d::from_quat_translation(zero_q, Vec3F64::ZERO);
        assert!(pose.rotation_defect().is_infinite());

        let nan_q = QuatF64::from_xyzw(f64::NAN, 0.0, 0.0, 1.0);
        let pose_nan = Pose3d::from_quat_translation(nan_q, Vec3F64::ZERO);
        assert!(pose_nan.rotation_defect().is_infinite());

        let inf_q = QuatF64::from_xyzw(f64::INFINITY, 0.0, 0.0, 1.0);
        let pose_inf = Pose3d::from_quat_translation(inf_q, Vec3F64::ZERO);
        assert!(pose_inf.rotation_defect().is_infinite());
    }

    #[test]
    fn test_from_quat_translation_large_finite_does_not_overflow() {
        // 180 degree rotation about X represented with large magnitude components (1e200)
        // Without scaling before normalization, dot(q, q) overflows to +inf, producing zero quat.
        let q = QuatF64::from_xyzw(1e200, 0.0, 0.0, 0.0);
        let pose = Pose3d::from_quat_translation(q, Vec3F64::ZERO);
        let pt = Vec3F64::new(0.0, 1.0, 0.0);
        let transformed = pose.transform_point(&pt);
        // 180 deg about X rotates (0, 1, 0) to (0, -1, 0)
        assert!((transformed.x - 0.0).abs() < 1e-12);
        assert!((transformed.y - (-1.0)).abs() < 1e-12);
        assert!((transformed.z - 0.0).abs() < 1e-12);
    }
}
