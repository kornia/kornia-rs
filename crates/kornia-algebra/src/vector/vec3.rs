//! 3D vector types (single and double precision).
define_vector_type!(Vec3F32, glam::Vec3, f32, [f32; 3], [x, y, z]);

define_vector_type!(Vec3F64, glam::DVec3, f64, [f64; 3], [x, y, z]);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec3f32_basic() {
        let v = Vec3F32::new(1.0, 2.0, 3.0);
        assert_eq!(v.x, 1.0);
        assert_eq!(v.y, 2.0);
        assert_eq!(v.z, 3.0);
    }

    #[test]
    fn test_vec3f32_from_array() {
        let v = Vec3F32::from_array([1.0, 2.0, 3.0]);
        assert_eq!(v.to_array(), [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_vec3f32_conversion() {
        let v = Vec3F32::new(1.0, 2.0, 3.0);
        let glam_v: glam::Vec3 = v.into();
        let back: Vec3F32 = glam_v.into();
        assert_eq!(v, back);
    }

    #[test]
    fn test_vec3f64_basic() {
        let v = Vec3F64::new(1.0, 2.0, 3.0);
        assert_eq!(v.x, 1.0);
        assert_eq!(v.y, 2.0);
        assert_eq!(v.z, 3.0);
    }

    #[test]
    fn test_vec3f64_from_array() {
        let v = Vec3F64::from_array([1.0, 2.0, 3.0]);
        assert_eq!(v.to_array(), [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_vec3f64_conversion() {
        let v = Vec3F64::new(1.0, 2.0, 3.0);
        let glam_v: glam::DVec3 = v.into();
        let back: Vec3F64 = glam_v.into();
        assert_eq!(v, back);
    }
}
