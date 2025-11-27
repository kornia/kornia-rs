//! Quaternion types.
//!
//! This module provides quaternion types for kornia-rs as thin newtype
//! wrappers over `glam` quaternions.

use crate::{Mat3AF32, Mat3F64, Mat4F32, Mat4F64};

macro_rules! define_quat_type {
    (
        $(#[$meta:meta])*
        $name:ident,
        $glam_type:ty,
        $scalar:ty,
        $mat3_type:ty,
        $mat4_type:ty,
        $from_mat3:ident
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq)]
        #[repr(transparent)]
        pub struct $name(pub $glam_type);

        impl $name {
            /// Identity quaternion.
            pub const IDENTITY: Self = Self(<$glam_type>::IDENTITY);

            /// Create a new quaternion from x, y, z, w components.
            #[inline]
            pub fn new(x: $scalar, y: $scalar, z: $scalar, w: $scalar) -> Self {
                Self(<$glam_type>::from_xyzw(x, y, z, w))
            }

            /// Create a quaternion from x, y, z, w components.
            #[inline]
            pub fn from_xyzw(x: $scalar, y: $scalar, z: $scalar, w: $scalar) -> Self {
                Self(<$glam_type>::from_xyzw(x, y, z, w))
            }

            /// Create a quaternion from an array of 4 components.
            #[inline]
            pub fn from_array(arr: [$scalar; 4]) -> Self {
                Self(<$glam_type>::from_array(arr))
            }

            /// Convert the quaternion to an array of 4 components.
            #[inline]
            pub fn to_array(&self) -> [$scalar; 4] {
                self.0.to_array()
            }

            /// Create a quaternion from a 3x3 rotation matrix.
            #[inline]
            pub fn $from_mat3(mat: &$mat3_type) -> Self {
                Self(<$glam_type>::$from_mat3(&mat.0))
            }

            /// Create a quaternion from a 4x4 matrix.
            #[inline]
            pub fn from_mat4(mat: &$mat4_type) -> Self {
                Self(<$glam_type>::from_mat4(&mat.0))
            }
        }

        impl std::ops::Deref for $name {
            type Target = $glam_type;

            #[inline]
            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }

        impl std::ops::DerefMut for $name {
            #[inline]
            fn deref_mut(&mut self) -> &mut Self::Target {
                &mut self.0
            }
        }

        // Conversions to and from the underlying glam type.
        impl From<$glam_type> for $name {
            #[inline]
            fn from(q: $glam_type) -> Self {
                Self(q)
            }
        }

        impl From<$name> for $glam_type {
            #[inline]
            fn from(q: $name) -> Self {
                q.0
            }
        }

        // Quaternion-quaternion multiplication.
        impl std::ops::Mul<$name> for $name {
            type Output = $name;

            #[inline]
            fn mul(self, rhs: $name) -> Self::Output {
                $name::from(self.0 * rhs.0)
            }
        }

        impl std::ops::MulAssign<$name> for $name {
            #[inline]
            fn mul_assign(&mut self, rhs: $name) {
                self.0 *= rhs.0;
            }
        }
    };
}

// Quaternion (single precision, `f32`).
define_quat_type!(QuatF32, glam::Quat, f32, Mat3AF32, Mat4F32, from_mat3a);

// Quaternion (double precision, `f64`).
define_quat_type!(QuatF64, glam::DQuat, f64, Mat3F64, Mat4F64, from_mat3);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quatf32_identity() {
        let quat = QuatF32::IDENTITY;
        assert_eq!(quat, QuatF32::from_xyzw(0.0, 0.0, 0.0, 1.0));
    }

    #[test]
    fn test_quatf32_new() {
        let quat = QuatF32::new(0.0, 0.0, 0.0, 1.0);
        assert_eq!(quat, QuatF32::from_xyzw(0.0, 0.0, 0.0, 1.0));
    }

    #[test]
    fn test_quatf32_from_xyzw() {
        let quat = QuatF32::from_xyzw(0.0, 0.0, 0.0, 1.0);
        assert_eq!(quat, QuatF32::from_xyzw(0.0, 0.0, 0.0, 1.0));
    }

    #[test]
    fn test_quatf32_from_array() {
        let quat = QuatF32::from_array([0.0, 0.0, 0.0, 1.0]);
        assert_eq!(quat, QuatF32::from_xyzw(0.0, 0.0, 0.0, 1.0));
    }

    #[test]
    fn test_quat64_identity() {
        let quat = QuatF64::IDENTITY;
        assert_eq!(quat, QuatF64::from_xyzw(0.0, 0.0, 0.0, 1.0));
    }

    #[test]
    fn test_quat64_new() {
        let quat = QuatF64::new(0.0, 0.0, 0.0, 1.0);
        assert_eq!(quat, QuatF64::from_xyzw(0.0, 0.0, 0.0, 1.0));
    }

    #[test]
    fn test_quat64_from_xyzw() {
        let quat = QuatF64::from_xyzw(0.0, 0.0, 0.0, 1.0);
        assert_eq!(quat, QuatF64::from_xyzw(0.0, 0.0, 0.0, 1.0));
    }

    #[test]
    fn test_quat64_from_array() {
        let quat = QuatF64::from_array([0.0, 0.0, 0.0, 1.0]);
        assert_eq!(quat, QuatF64::from_xyzw(0.0, 0.0, 0.0, 1.0));
    }

    #[test]
    fn test_quat64_to_array() {
        let quat = QuatF64::from_xyzw(0.0, 0.0, 0.0, 1.0);
        let array = quat.to_array();
        assert_eq!(array, [0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_quatf32_mul_assign() {
        let mut q = QuatF32::IDENTITY;
        let q2 = QuatF32::from_xyzw(0.0, 0.0, 0.0, 1.0);
        q *= q2;
        assert_eq!(q, QuatF32::IDENTITY);
    }
}
