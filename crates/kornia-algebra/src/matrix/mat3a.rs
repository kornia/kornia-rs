//! 3x3 matrix (aligned, single precision).

use crate::Vec3AF32;

// 3x3 matrix (aligned, single precision).
define_matrix_type!(
    Mat3AF32,
    glam::Mat3A,
    [f32; 9],
    Vec3AF32,
    glam::Vec3A,
    [x_axis, y_axis, z_axis]
);

// Scalar multiplication
impl std::ops::Mul<f32> for Mat3AF32 {
    type Output = Mat3AF32;

    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        Mat3AF32::from(self.0 * rhs)
    }
}

// Scalar multiplication (reverse)
impl std::ops::Mul<Mat3AF32> for f32 {
    type Output = Mat3AF32;

    #[inline]
    fn mul(self, rhs: Mat3AF32) -> Self::Output {
        Mat3AF32::from(self * rhs.0)
    }
}

// Matrix addition
impl std::ops::Add<Mat3AF32> for Mat3AF32 {
    type Output = Mat3AF32;

    #[inline]
    fn add(self, rhs: Mat3AF32) -> Self::Output {
        Mat3AF32::from(self.0 + rhs.0)
    }
}

// Matrix subtraction
impl std::ops::Sub<Mat3AF32> for Mat3AF32 {
    type Output = Mat3AF32;

    #[inline]
    fn sub(self, rhs: Mat3AF32) -> Self::Output {
        Mat3AF32::from(self.0 - rhs.0)
    }
}

impl Mat3AF32 {
    /// Transpose the matrix.
    #[inline]
    pub fn transpose(self) -> Self {
        Mat3AF32::from(self.0.transpose())
    }

    /// Get the inverse of the matrix.
    #[inline]
    pub fn inverse(self) -> Self {
        Mat3AF32::from(self.0.inverse())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mat3af32_mul_vec3af32() {
        let m = Mat3AF32::IDENTITY;
        let v = Vec3AF32::new(1.0, 2.0, 3.0);
        let result = m * v;
        assert_eq!(result, v);
    }
}
