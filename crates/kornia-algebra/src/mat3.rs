//! 3x3 matrix (single precision).

use crate::vec3::Vec3;
use std::ops::{Deref, DerefMut};

/// 3x3 matrix (single precision).
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(transparent)]
pub struct Mat3(pub glam::Mat3);

impl Mat3 {
    /// Create a new Mat3 from column vectors.
    #[inline]
    pub fn from_cols(x_axis: Vec3, y_axis: Vec3, z_axis: Vec3) -> Self {
        Self(glam::Mat3::from_cols(
            glam::Vec3::from(x_axis),
            glam::Vec3::from(y_axis),
            glam::Vec3::from(z_axis),
        ))
    }

    /// Create a new Mat3 from a column-major array.
    #[inline]
    pub fn from_cols_array(arr: &[f32; 9]) -> Self {
        Self(glam::Mat3::from_cols_array(arr))
    }

    /// Identity matrix.
    pub const IDENTITY: Self = Self(glam::Mat3::IDENTITY);
}

impl Deref for Mat3 {
    type Target = glam::Mat3;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Mat3 {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<glam::Mat3> for Mat3 {
    #[inline]
    fn from(m: glam::Mat3) -> Self {
        Self(m)
    }
}

impl From<Mat3> for glam::Mat3 {
    #[inline]
    fn from(m: Mat3) -> Self {
        m.0
    }
}

// Matrix-matrix multiplication
impl std::ops::Mul<Mat3> for Mat3 {
    type Output = Mat3;

    #[inline]
    fn mul(self, rhs: Mat3) -> Self::Output {
        Mat3::from(self.0 * rhs.0)
    }
}

// Matrix-vector multiplication
impl std::ops::Mul<Vec3> for Mat3 {
    type Output = Vec3;

    #[inline]
    fn mul(self, rhs: Vec3) -> Self::Output {
        Vec3::from(self.0 * glam::Vec3::from(rhs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mat3_mul_vec3() {
        let m = Mat3::IDENTITY;
        let v = Vec3::new(1.0, 2.0, 3.0);
        let result = m * v;
        assert_eq!(result, v);
    }
}
