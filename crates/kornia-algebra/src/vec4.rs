//! 4D vector (single precision).

use crate::vec3::Vec3;
use std::ops::{Deref, DerefMut};

/// 4D vector (single precision).
///
/// This is a newtype wrapper around `glam::Vec4` to provide a unified
/// algebraic type system for kornia-rs.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(transparent)]
pub struct Vec4(pub glam::Vec4);

impl Vec4 {
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
}

impl Deref for Vec4 {
    type Target = glam::Vec4;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Vec4 {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<glam::Vec4> for Vec4 {
    #[inline]
    fn from(v: glam::Vec4) -> Self {
        Self(v)
    }
}

impl From<Vec4> for glam::Vec4 {
    #[inline]
    fn from(v: Vec4) -> Self {
        v.0
    }
}

impl From<Vec3> for Vec4 {
    #[inline]
    fn from(v: Vec3) -> Self {
        Self(glam::Vec4::new(v.x, v.y, v.z, 0.0))
    }
}

// Arithmetic operations
impl std::ops::Add for Vec4 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl std::ops::Sub for Vec4 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl std::ops::Mul<f32> for Vec4 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        Self(self.0 * rhs)
    }
}

impl std::ops::Div<f32> for Vec4 {
    type Output = Self;

    #[inline]
    fn div(self, rhs: f32) -> Self::Output {
        Self(self.0 / rhs)
    }
}

impl std::ops::Neg for Vec4 {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}
