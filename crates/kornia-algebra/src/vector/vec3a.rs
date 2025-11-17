//! 3D vector (aligned, single precision).
define_vector_type!(Vec3AF32, glam::Vec3A, f32, [f32; 3], [x, y, z]);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec3af32_basic() {
        let v = Vec3AF32::new(1.0, 2.0, 3.0);
        assert_eq!(v.x, 1.0);
        assert_eq!(v.y, 2.0);
        assert_eq!(v.z, 3.0);
    }

    #[test]
    fn test_vec3af32_from_array() {
        let v = Vec3AF32::from_array([1.0, 2.0, 3.0]);
        assert_eq!(v.to_array(), [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_vec3af32_arithmetic() {
        let v1 = Vec3AF32::new(1.0, 2.0, 3.0);
        let v2 = Vec3AF32::new(4.0, 5.0, 6.0);
        assert_eq!(v1 + v2, Vec3AF32::new(5.0, 7.0, 9.0));
        assert_eq!(v1 * 2.0, Vec3AF32::new(2.0, 4.0, 6.0));
    }

    #[test]
    fn test_vec3af32_conversion() {
        let v = Vec3AF32::new(1.0, 2.0, 3.0);
        let glam_v: glam::Vec3A = v.into();
        let back: Vec3AF32 = glam_v.into();
        assert_eq!(v, back);
    }
}
