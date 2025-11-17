//! 3D vector (aligned, single precision).

use super::vec3::Vec3F32;
use std::ops::{Deref, DerefMut};

/// 3D vector (aligned, single precision).
///
/// This is a newtype wrapper around `glam::Vec3A` for SIMD-optimized operations.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(transparent)]
pub struct Vec3AF32(pub glam::Vec3A);

impl Vec3AF32 {
    /// Create a new Vec3AF32 from x, y, and z components.
    #[inline]
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self(glam::Vec3A::new(x, y, z))
    }

    /// Create a Vec3AF32 from an array.
    #[inline]
    pub fn from_array(arr: [f32; 3]) -> Self {
        Self(glam::Vec3A::from_array(arr))
    }

    /// Convert to array.
    #[inline]
    pub fn to_array(self) -> [f32; 3] {
        self.0.to_array()
    }

    /// Zero vector.
    pub const ZERO: Self = Self(glam::Vec3A::ZERO);

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
}

impl Deref for Vec3AF32 {
    type Target = glam::Vec3A;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Vec3AF32 {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<glam::Vec3A> for Vec3AF32 {
    #[inline]
    fn from(v: glam::Vec3A) -> Self {
        Self(v)
    }
}

impl From<Vec3AF32> for glam::Vec3A {
    #[inline]
    fn from(v: Vec3AF32) -> Self {
        v.0
    }
}

impl From<Vec3F32> for Vec3AF32 {
    #[inline]
    fn from(v: Vec3F32) -> Self {
        Self(glam::Vec3A::from(glam::Vec3::from(v)))
    }
}

impl From<Vec3AF32> for Vec3F32 {
    #[inline]
    fn from(v: Vec3AF32) -> Self {
        Self::from(glam::Vec3::from(glam::Vec3A::from(v)))
    }
}

// Arithmetic operations
impl std::ops::Add for Vec3AF32 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl std::ops::Sub for Vec3AF32 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl std::ops::Mul<f32> for Vec3AF32 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        Self(self.0 * rhs)
    }
}

impl std::ops::Div<f32> for Vec3AF32 {
    type Output = Self;

    #[inline]
    fn div(self, rhs: f32) -> Self::Output {
        Self(self.0 / rhs)
    }
}

impl std::ops::Neg for Vec3AF32 {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

// Scalar multiplication (reverse)
impl std::ops::Mul<Vec3AF32> for f32 {
    type Output = Vec3AF32;

    #[inline]
    fn mul(self, rhs: Vec3AF32) -> Self::Output {
        Vec3AF32::from(self * rhs.0)
    }
}
