//! 4D vector types (single and double precision).
define_vector_type!(Vec4F32, glam::Vec4, f32, [f32; 4], [x, y, z, w]);

define_vector_type!(Vec4F64, glam::DVec4, f64, [f64; 4], [x, y, z, w]);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec4f32_basic() {
        let v = Vec4F32::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(v.x, 1.0);
        assert_eq!(v.y, 2.0);
        assert_eq!(v.z, 3.0);
        assert_eq!(v.w, 4.0);
    }

    #[test]
    fn test_vec4f32_from_array() {
        let v = Vec4F32::from_array([1.0, 2.0, 3.0, 4.0]);
        assert_eq!(v.to_array(), [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_vec4f32_conversion() {
        let v = Vec4F32::new(1.0, 2.0, 3.0, 4.0);
        let glam_v: glam::Vec4 = v.into();
        let back: Vec4F32 = glam_v.into();
        assert_eq!(v, back);
    }

    #[test]
    fn test_vec4f64_basic() {
        let v = Vec4F64::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(v.x, 1.0);
        assert_eq!(v.y, 2.0);
        assert_eq!(v.z, 3.0);
        assert_eq!(v.w, 4.0);
    }

    #[test]
    fn test_vec4f64_from_array() {
        let v = Vec4F64::from_array([1.0, 2.0, 3.0, 4.0]);
        assert_eq!(v.to_array(), [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_vec4f64_conversion() {
        let v = Vec4F64::new(1.0, 2.0, 3.0, 4.0);
        let glam_v: glam::DVec4 = v.into();
        let back: Vec4F64 = glam_v.into();
        assert_eq!(v, back);
    }
}
