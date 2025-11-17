//! 4x4 matrix (single precision).

use crate::{Vec4F32, Vec4F64};

// 4x4 matrix (single and double precision).
define_matrix_type!(
    Mat4F32,
    glam::Mat4,
    [f32; 16],
    Vec4F32,
    glam::Vec4,
    [x_axis, y_axis, z_axis, w_axis]
);

define_matrix_type!(
    Mat4F64,
    glam::DMat4,
    [f64; 16],
    Vec4F64,
    glam::DVec4,
    [x_axis, y_axis, z_axis, w_axis]
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mat4f32_mul_vec4() {
        let m = Mat4F32::IDENTITY;
        let v = Vec4F32::new(1.0, 2.0, 3.0, 4.0);
        let result = m * v;
        assert_eq!(result, v);
    }

    #[test]
    fn test_mat4f64_mul_vec4() {
        let m = Mat4F64::IDENTITY;
        let v = Vec4F64::new(1.0, 2.0, 3.0, 4.0);
        let result = m * v;
        assert_eq!(result, v);
    }
}
