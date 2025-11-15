//! 4x4 matrix (single precision).

use crate::{vec3::Vec3, vec4::Vec4};
use std::ops::{Deref, DerefMut};

/// 4x4 matrix (single precision).
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(transparent)]
pub struct Mat4(pub glam::Mat4);

impl Mat4 {
    /// Create a new Mat4 from column vectors.
    #[inline]
    pub fn from_cols(x_axis: Vec4, y_axis: Vec4, z_axis: Vec4, w_axis: Vec4) -> Self {
        Self(glam::Mat4::from_cols(
            glam::Vec4::from(x_axis),
            glam::Vec4::from(y_axis),
            glam::Vec4::from(z_axis),
            glam::Vec4::from(w_axis),
        ))
    }

    /// Create a new Mat4 from Vec3 column vectors (w component defaults to 0 for x/y/z, 1 for w).
    #[inline]
    pub fn from_cols_vec3(x_axis: Vec3, y_axis: Vec3, z_axis: Vec3, w_axis: Vec3) -> Self {
        Self(glam::Mat4::from_cols(
            glam::Vec4::new(x_axis.x, x_axis.y, x_axis.z, 0.0),
            glam::Vec4::new(y_axis.x, y_axis.y, y_axis.z, 0.0),
            glam::Vec4::new(z_axis.x, z_axis.y, z_axis.z, 0.0),
            glam::Vec4::new(w_axis.x, w_axis.y, w_axis.z, 1.0),
        ))
    }

    /// Create a new Mat4 from a column-major array.
    #[inline]
    pub fn from_cols_array(arr: &[f32; 16]) -> Self {
        Self(glam::Mat4::from_cols_array(arr))
    }

    /// Identity matrix.
    pub const IDENTITY: Self = Self(glam::Mat4::IDENTITY);
}

impl Deref for Mat4 {
    type Target = glam::Mat4;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Mat4 {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<glam::Mat4> for Mat4 {
    #[inline]
    fn from(m: glam::Mat4) -> Self {
        Self(m)
    }
}

impl From<Mat4> for glam::Mat4 {
    #[inline]
    fn from(m: Mat4) -> Self {
        m.0
    }
}

// Matrix-matrix multiplication
impl std::ops::Mul<Mat4> for Mat4 {
    type Output = Mat4;

    #[inline]
    fn mul(self, rhs: Mat4) -> Self::Output {
        Mat4::from(self.0 * rhs.0)
    }
}

// Matrix-vector multiplication
impl std::ops::Mul<Vec4> for Mat4 {
    type Output = Vec4;

    #[inline]
    fn mul(self, rhs: Vec4) -> Self::Output {
        Vec4::from(self.0 * glam::Vec4::from(rhs))
    }
}

impl Mat4 {
    /// Multiply Mat4 by Vec3 (treats Vec3 as homogeneous with w=1).
    #[inline]
    pub fn mul_vec3(self, rhs: Vec3) -> Vec3 {
        let v4 = glam::Vec4::new(rhs.x, rhs.y, rhs.z, 1.0);
        let result = self.0 * v4;
        Vec3::new(result.x, result.y, result.z)
    }
}
