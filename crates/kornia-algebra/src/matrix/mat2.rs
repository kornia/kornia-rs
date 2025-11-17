//! 2x2 matrix (single precision).

use crate::Vec2F32;
use std::ops::{Deref, DerefMut};

/// 2x2 matrix (single precision).
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(transparent)]
pub struct Mat2(pub glam::Mat2);

impl Mat2 {
    /// Create a new Mat2 from column vectors.
    #[inline]
    pub fn from_cols(x_axis: Vec2F32, y_axis: Vec2F32) -> Self {
        Self(glam::Mat2::from_cols(
            glam::Vec2::from(x_axis),
            glam::Vec2::from(y_axis),
        ))
    }

    /// Create a new Mat2 from a column-major array.
    #[inline]
    pub fn from_cols_array(arr: &[f32; 4]) -> Self {
        Self(glam::Mat2::from_cols_array(arr))
    }

    /// Identity matrix.
    pub const IDENTITY: Self = Self(glam::Mat2::IDENTITY);
}

impl Deref for Mat2 {
    type Target = glam::Mat2;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Mat2 {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<glam::Mat2> for Mat2 {
    #[inline]
    fn from(m: glam::Mat2) -> Self {
        Self(m)
    }
}

impl From<Mat2> for glam::Mat2 {
    #[inline]
    fn from(m: Mat2) -> Self {
        m.0
    }
}

// Matrix-matrix multiplication
impl std::ops::Mul<Mat2> for Mat2 {
    type Output = Mat2;

    #[inline]
    fn mul(self, rhs: Mat2) -> Self::Output {
        Mat2::from(self.0 * rhs.0)
    }
}

// Matrix-vector multiplication
impl std::ops::Mul<Vec2F32> for Mat2 {
    type Output = Vec2F32;

    #[inline]
    fn mul(self, rhs: Vec2F32) -> Self::Output {
        Vec2F32::from(self.0 * glam::Vec2::from(rhs))
    }
}

impl Mat2 {
    /// Transpose the matrix.
    #[inline]
    pub fn transpose(self) -> Self {
        Mat2::from(self.0.transpose())
    }
}
