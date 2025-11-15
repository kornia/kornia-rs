//! 3x3 matrix (aligned, single precision).

use crate::vec3a::Vec3A;
use std::ops::{Deref, DerefMut};

/// 3x3 matrix (aligned, single precision).
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(transparent)]
pub struct Mat3A(pub glam::Mat3A);

impl Mat3A {
    /// Create a new Mat3A from column vectors.
    #[inline]
    pub fn from_cols(x_axis: Vec3A, y_axis: Vec3A, z_axis: Vec3A) -> Self {
        Self(glam::Mat3A::from_cols(
            glam::Vec3A::from(x_axis),
            glam::Vec3A::from(y_axis),
            glam::Vec3A::from(z_axis),
        ))
    }

    /// Create a new Mat3A from a column-major array.
    #[inline]
    pub fn from_cols_array(arr: &[f32; 9]) -> Self {
        Self(glam::Mat3A::from_cols_array(arr))
    }

    /// Identity matrix.
    pub const IDENTITY: Self = Self(glam::Mat3A::IDENTITY);
}

impl Deref for Mat3A {
    type Target = glam::Mat3A;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Mat3A {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<glam::Mat3A> for Mat3A {
    #[inline]
    fn from(m: glam::Mat3A) -> Self {
        Self(m)
    }
}

impl From<Mat3A> for glam::Mat3A {
    #[inline]
    fn from(m: Mat3A) -> Self {
        m.0
    }
}

// Matrix-vector multiplication
impl std::ops::Mul<Vec3A> for Mat3A {
    type Output = Vec3A;

    #[inline]
    fn mul(self, rhs: Vec3A) -> Self::Output {
        Vec3A::from(self.0 * glam::Vec3A::from(rhs))
    }
}

// Matrix-matrix multiplication
impl std::ops::Mul<Mat3A> for Mat3A {
    type Output = Mat3A;

    #[inline]
    fn mul(self, rhs: Mat3A) -> Self::Output {
        Mat3A::from(self.0 * rhs.0)
    }
}

// Scalar multiplication
impl std::ops::Mul<f32> for Mat3A {
    type Output = Mat3A;

    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        Mat3A::from(self.0 * rhs)
    }
}

// Scalar multiplication (reverse)
impl std::ops::Mul<Mat3A> for f32 {
    type Output = Mat3A;

    #[inline]
    fn mul(self, rhs: Mat3A) -> Self::Output {
        Mat3A::from(self * rhs.0)
    }
}

// Matrix addition
impl std::ops::Add<Mat3A> for Mat3A {
    type Output = Mat3A;

    #[inline]
    fn add(self, rhs: Mat3A) -> Self::Output {
        Mat3A::from(self.0 + rhs.0)
    }
}

// Matrix subtraction
impl std::ops::Sub<Mat3A> for Mat3A {
    type Output = Mat3A;

    #[inline]
    fn sub(self, rhs: Mat3A) -> Self::Output {
        Mat3A::from(self.0 - rhs.0)
    }
}

impl Mat3A {
    /// Transpose the matrix.
    #[inline]
    pub fn transpose(self) -> Self {
        Mat3A::from(self.0.transpose())
    }

    /// Get the inverse of the matrix.
    #[inline]
    pub fn inverse(self) -> Self {
        Mat3A::from(self.0.inverse())
    }

    /// Get the determinant of the matrix.
    #[inline]
    pub fn determinant(self) -> f32 {
        self.0.determinant()
    }

    /// Check if all elements are finite.
    #[inline]
    pub fn is_finite(self) -> bool {
        self.0.is_finite()
    }
}
