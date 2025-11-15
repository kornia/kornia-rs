//! Quaternion (single precision).

/// Quaternion (single precision).
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(transparent)]
pub struct Quat(pub glam::Quat);

impl Quat {
    /// Identity quaternion.
    pub const IDENTITY: Self = Self(glam::Quat::IDENTITY);

    /// Create a new quaternion from x, y, z, w components.
    #[inline]
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self(glam::Quat::from_xyzw(x, y, z, w))
    }

    /// Create a quaternion from x, y, z, w components.
    #[inline]
    pub fn from_xyzw(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self(glam::Quat::from_xyzw(x, y, z, w))
    }

    /// Create a quaternion from a Mat3A matrix.
    #[inline]
    pub fn from_mat3a(mat: &crate::Mat3A) -> Self {
        Self(glam::Quat::from_mat3a(&glam::Mat3A::from(*mat)))
    }

    /// Create a quaternion from a Mat4 matrix.
    #[inline]
    pub fn from_mat4(mat: &crate::Mat4) -> Self {
        Self(glam::Quat::from_mat4(&glam::Mat4::from(*mat)))
    }
}

impl std::ops::Deref for Quat {
    type Target = glam::Quat;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for Quat {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<glam::Quat> for Quat {
    #[inline]
    fn from(q: glam::Quat) -> Self {
        Self(q)
    }
}

impl From<Quat> for glam::Quat {
    #[inline]
    fn from(q: Quat) -> Self {
        q.0
    }
}

// Quaternion multiplication
impl std::ops::Mul<Quat> for Quat {
    type Output = Quat;

    #[inline]
    fn mul(self, rhs: Quat) -> Self::Output {
        Quat::from(self.0 * rhs.0)
    }
}
