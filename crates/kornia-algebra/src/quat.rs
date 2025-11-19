//! Quaternion types.
//!
//! This module provides quaternion types for kornia-rs as thin newtype
//! wrappers over `glam` quaternions.

macro_rules! define_quat_type {
    ($(#[$meta:meta])* $name:ident, $glam_type:ty, $scalar:ty) => {
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
    };
}

// Quaternion (single precision, `f32`).
define_quat_type!(QuatF32, glam::Quat, f32);

// Quaternion (double precision, `f64`).
define_quat_type!(QuatF64, glam::DQuat, f64);

impl QuatF32 {
    /// Create a quaternion from a `Mat3AF32` matrix.
    #[inline]
    pub fn from_mat3a(mat: &crate::Mat3AF32) -> Self {
        Self(glam::Quat::from_mat3a(&glam::Mat3A::from(*mat)))
    }

    /// Create a quaternion from a `Mat4` matrix.
    #[inline]
    pub fn from_mat4(mat: &crate::Mat4) -> Self {
        Self(glam::Quat::from_mat4(&glam::Mat4::from(*mat)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quatf32_from_mat3a() {
        let mat = crate::Mat3AF32::IDENTITY;
        let quat = QuatF32::from_mat3a(&mat);
        // Just ensure the conversion compiles and produces a finite quaternion.
        let q_glam: glam::Quat = quat.into();
        assert!(q_glam.is_finite());
    }
    #[test]
    fn test_quatf32_from_mat4() {
        let mat = crate::Mat4F32::IDENTITY;
        let quat = QuatF32::from_mat4(&mat);
        let q_glam: glam::Quat = quat.into();
        assert!(q_glam.is_finite());
    }
}
