//! 4D vector (single precision).

use super::vec3::Vec3F32;
use std::ops::{Deref, DerefMut};

/// 4D vector (single precision).
///
/// This is a newtype wrapper around `glam::Vec4` to provide a unified
/// algebraic type system for kornia-rs.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(transparent)]
pub struct Vec4F32(pub glam::Vec4);

impl Vec4F32 {
    /// Create a new Vec4 from x, y, z, and w components.
    #[inline]
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self(glam::Vec4::new(x, y, z, w))
    }

    /// Create a Vec4 from an array.
    #[inline]
    pub fn from_array(arr: [f32; 4]) -> Self {
        Self(glam::Vec4::from_array(arr))
    }

    /// Convert to array.
    #[inline]
    pub fn to_array(self) -> [f32; 4] {
        self.0.to_array()
    }

    /// Zero vector.
    pub const ZERO: Self = Self(glam::Vec4::ZERO);

    /// Get the x component.
    #[inline]
    pub fn x(self) -> f32 {
        self.0.x
    }

    /// Get the y component.
    #[inline]
    pub fn y(self) -> f32 {
        self.0.y
    }

    /// Get the z component.
    #[inline]
    pub fn z(self) -> f32 {
        self.0.z
    }

    /// Get the w component.
    #[inline]
    pub fn w(self) -> f32 {
        self.0.w
    }
}

impl Deref for Vec4F32 {
    type Target = glam::Vec4;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Vec4F32 {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<glam::Vec4> for Vec4F32 {
    #[inline]
    fn from(v: glam::Vec4) -> Self {
        Self(v)
    }
}

impl From<Vec4F32> for glam::Vec4 {
    #[inline]
    fn from(v: Vec4F32) -> Self {
        v.0
    }
}

impl From<[f32; 4]> for Vec4F32 {
    #[inline]
    fn from(arr: [f32; 4]) -> Self {
        Self::from_array(arr)
    }
}

impl From<Vec4F32> for [f32; 4] {
    #[inline]
    fn from(v: Vec4F32) -> Self {
        v.to_array()
    }
}

impl From<Vec3F32> for Vec4F32 {
    #[inline]
    fn from(v: Vec3F32) -> Self {
        Self(glam::Vec4::new(v.x, v.y, v.z, 0.0))
    }
}

// Arithmetic operations
impl std::ops::Add for Vec4F32 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl std::ops::Sub for Vec4F32 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl std::ops::Mul<f32> for Vec4F32 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        Self(self.0 * rhs)
    }
}

impl std::ops::Div<f32> for Vec4F32 {
    type Output = Self;

    #[inline]
    fn div(self, rhs: f32) -> Self::Output {
        Self(self.0 / rhs)
    }
}

impl std::ops::Neg for Vec4F32 {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec4_basic() {
        let v = Vec4F32::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(v.x, 1.0);
        assert_eq!(v.y, 2.0);
        assert_eq!(v.z, 3.0);
        assert_eq!(v.w, 4.0);
    }

    #[test]
    fn test_vec4_from_array() {
        let v = Vec4F32::from_array([1.0, 2.0, 3.0, 4.0]);
        assert_eq!(v.to_array(), [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_vec4_conversion() {
        let v = Vec4F32::new(1.0, 2.0, 3.0, 4.0);
        let glam_v: glam::Vec4 = v.into();
        let back: Vec4F32 = glam_v.into();
        assert_eq!(v, back);
    }
}
