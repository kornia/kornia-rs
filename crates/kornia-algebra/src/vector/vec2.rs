//! 2D vector (single precision).

use std::ops::{Deref, DerefMut};

/// 2D vector (single precision).
///
/// This is a newtype wrapper around `glam::Vec2` to provide a unified
/// algebraic type system for kornia-rs.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(transparent)]
pub struct Vec2F32(pub glam::Vec2);

impl Vec2F32 {
    /// Create a new Vec2 from x and y components.
    #[inline]
    pub fn new(x: f32, y: f32) -> Self {
        Self(glam::Vec2::new(x, y))
    }

    /// Create a Vec2 from an array.
    #[inline]
    pub fn from_array(arr: [f32; 2]) -> Self {
        Self(glam::Vec2::from_array(arr))
    }

    /// Convert to array.
    #[inline]
    pub fn to_array(self) -> [f32; 2] {
        self.0.to_array()
    }

    /// Zero vector.
    pub const ZERO: Self = Self(glam::Vec2::ZERO);

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
}

impl Deref for Vec2F32 {
    type Target = glam::Vec2;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Vec2F32 {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<glam::Vec2> for Vec2F32 {
    #[inline]
    fn from(v: glam::Vec2) -> Self {
        Self(v)
    }
}

impl From<Vec2F32> for glam::Vec2 {
    #[inline]
    fn from(v: Vec2F32) -> Self {
        v.0
    }
}

impl From<[f32; 2]> for Vec2F32 {
    #[inline]
    fn from(arr: [f32; 2]) -> Self {
        Self::from_array(arr)
    }
}

impl From<Vec2F32> for [f32; 2] {
    #[inline]
    fn from(v: Vec2F32) -> Self {
        v.to_array()
    }
}

// Arithmetic operations
impl std::ops::Add for Vec2F32 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl std::ops::Sub for Vec2F32 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl std::ops::Mul<f32> for Vec2F32 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        Self(self.0 * rhs)
    }
}

impl std::ops::Div<f32> for Vec2F32 {
    type Output = Self;

    #[inline]
    fn div(self, rhs: f32) -> Self::Output {
        Self(self.0 / rhs)
    }
}

impl std::ops::Neg for Vec2F32 {
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
    fn test_vec2_basic() {
        let v = Vec2F32::new(1.0, 2.0);
        assert_eq!(v.x, 1.0);
        assert_eq!(v.y, 2.0);
    }

    #[test]
    fn test_vec2_from_array() {
        let v = Vec2F32::from_array([1.0, 2.0]);
        assert_eq!(v.to_array(), [1.0, 2.0]);
    }

    #[test]
    fn test_vec2_arithmetic() {
        let v1 = Vec2F32::new(1.0, 2.0);
        let v2 = Vec2F32::new(3.0, 4.0);
        assert_eq!(v1 + v2, Vec2F32::new(4.0, 6.0));
        assert_eq!(v1 * 2.0, Vec2F32::new(2.0, 4.0));
    }
}
