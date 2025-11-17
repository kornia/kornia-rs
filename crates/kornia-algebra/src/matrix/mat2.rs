//! 2x2 matrix (single precision).

use crate::{Vec2F32, Vec2F64};

// 2x2 matrix (single and double precision).
define_matrix_type!(
    Mat2F32,
    glam::Mat2,
    [f32; 4],
    Vec2F32,
    glam::Vec2,
    [x_axis, y_axis]
);

define_matrix_type!(
    Mat2F64,
    glam::DMat2,
    [f64; 4],
    Vec2F64,
    glam::DVec2,
    [x_axis, y_axis]
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mat2_mul_vec2() {
        let m = Mat2F32::IDENTITY;
        let v = Vec2F32::new(1.0, 2.0);
        let result = m * v;
        assert_eq!(result, v);
    }

    #[test]
    fn test_mat2f64_mul_vec2() {
        let m = Mat2F64::IDENTITY;
        let v = Vec2F64::new(1.0, 2.0);
        let result = m * v;
        assert_eq!(result, v);
    }
}
