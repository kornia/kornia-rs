//! 2D vector types (single and double precision).
define_vector_type!(Vec2F32, glam::Vec2, f32, [f32; 2], [x, y]);

define_vector_type!(Vec2F64, glam::DVec2, f64, [f64; 2], [x, y]);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec2f32_basic() {
        let v = Vec2F32::new(1.0, 2.0);
        assert_eq!(v.x, 1.0);
        assert_eq!(v.y, 2.0);
    }

    #[test]
    fn test_vec2f32_from_array() {
        let v = Vec2F32::from_array([1.0, 2.0]);
        assert_eq!(v.to_array(), [1.0, 2.0]);
    }

    #[test]
    fn test_vec2f32_arithmetic() {
        let v1 = Vec2F32::new(1.0, 2.0);
        let v2 = Vec2F32::new(3.0, 4.0);
        assert_eq!(v1 + v2, Vec2F32::new(4.0, 6.0));
        assert_eq!(v1 * 2.0, Vec2F32::new(2.0, 4.0));
    }

    #[test]
    fn test_vec2f64_basic() {
        let v = Vec2F64::new(1.0, 2.0);
        assert_eq!(v.x, 1.0);
        assert_eq!(v.y, 2.0);
    }

    #[test]
    fn test_vec2f64_from_array() {
        let v = Vec2F64::from_array([1.0, 2.0]);
        assert_eq!(v.to_array(), [1.0, 2.0]);
    }

    #[test]
    fn test_vec2f64_arithmetic() {
        let v1 = Vec2F64::new(1.0, 2.0);
        let v2 = Vec2F64::new(3.0, 4.0);
        assert_eq!(v1 + v2, Vec2F64::new(4.0, 6.0));
        assert_eq!(v1 * 2.0, Vec2F64::new(2.0, 4.0));
    }
}
