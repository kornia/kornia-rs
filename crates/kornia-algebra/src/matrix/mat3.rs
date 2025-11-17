//! 3x3 matrix (single precision).

use crate::{Vec3F32, Vec3F64};

// 3x3 matrix (single and double precision).
define_matrix_type!(
    Mat3F32,
    glam::Mat3,
    [f32; 9],
    Vec3F32,
    glam::Vec3,
    [x_axis, y_axis, z_axis]
);

define_matrix_type!(
    Mat3F64,
    glam::DMat3,
    [f64; 9],
    Vec3F64,
    glam::DVec3,
    [x_axis, y_axis, z_axis]
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mat3f32_mul_vec3() {
        let m = Mat3F32::IDENTITY;
        let v = Vec3F32::new(1.0, 2.0, 3.0);
        let result = m * v;
        assert_eq!(result, v);
    }

    #[test]
    fn test_mat3f64_mul_vec3() {
        let m = Mat3F64::IDENTITY;
        let v = Vec3F64::new(1.0, 2.0, 3.0);
        let result = m * v;
        assert_eq!(result, v);
    }
}
