//! # Quaternion types
//!
//! Thin newtype wrappers over `glam` quaternions.
//!
//! ## Mathematical context
//!
//! The set of **unit quaternions** (quaternions with |q| = 1) forms a Lie group that is
//! isomorphic to **SU(2)**, the group of 2×2 unitary matrices with determinant 1.
//! Topologically, the unit quaternions form **S³**, the 3-sphere in R⁴.
//!
//! This is a stronger statement than "quaternions represent rotations":
//! - **Unit quaternions ≅ SU(2)** — isomorphic as Lie groups (same topology AND same
//!   group multiplication). They are the same mathematical object.
//! - **SU(2) ≠ SO(3)** — SU(2) is the **double cover** of SO(3). The quotient map
//!   SU(2) → SO(3) sends both `q` and `-q` to the same rotation matrix. SO(3) is
//!   topologically RP³ (the 3-sphere with antipodal points identified).
//!
//! The three levels of equivalence:
//!
//! | Level | Meaning | Unit quaternions ≅ SU(2)? |
//! |-------|---------|---------------------------|
//! | Topological | Same shape (homeomorphic) | Yes — both S³ |
//! | Smooth | Same calculus (diffeomorphic) | Yes |
//! | Lie group | Same shape AND same multiplication | Yes |
//!
//! ## Practical consequences
//!
//! - **`q` and `-q` are distinct quaternions that represent the same rotation.**
//!   This is the 2:1 nature of the double cover, not a numerical artifact.
//! - **SLERP is smooth** because it operates on S³ (simply connected), not on
//!   RP³ = SO(3) (which has a non-contractible loop at 360°).
//! - **`from_mat3` must choose a sign** — given a rotation matrix, there are two
//!   quaternions that produce it. No continuous global section exists.
//! - **Normalize after repeated multiplications** — floating point drift pushes
//!   quaternions off S³ into the surrounding R⁴. Call [`QuatF32::normalize`] to
//!   project back onto the unit sphere and stay in SU(2).
//! - **Quaternion multiplication IS the SU(2) group operation.** The `Mul` impl
//!   below is not just "convenient math" — it is the Lie group composition law.

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

            /// Normalize the quaternion to unit length.
            #[inline]
            pub fn normalize(self) -> Self {
                Self(self.0.normalize())
            }

            /// Get the inverse of the quaternion.
            #[inline]
            pub fn inverse(self) -> Self {
                Self(self.0.inverse())
            }

            /// Get the conjugate of the quaternion.
            #[inline]
            pub fn conjugate(self) -> Self {
                Self(self.0.conjugate())
            }

            /// Get the length (magnitude) of the quaternion.
            #[inline]
            pub fn length(self) -> $scalar {
                self.0.length()
            }

            /// Rotate a 3D vector by this quaternion.
            /// This computes q * v * q^-1 where v is treated as a pure quaternion.
            #[inline]
            pub fn mul_vec3(self, v: [$scalar; 3]) -> [$scalar; 3] {
                let result = self.0.mul_vec3(<_>::from(v));
                result.into()
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

        #[cfg(feature = "approx")]
        impl approx::AbsDiffEq for $name {
            type Epsilon = <$scalar as approx::AbsDiffEq>::Epsilon;

            #[inline]
            fn default_epsilon() -> Self::Epsilon {
                <$scalar as approx::AbsDiffEq>::default_epsilon()
            }

            #[inline]
            fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
                let a = self.to_array();
                let b = other.to_array();
                a.iter()
                    .zip(b.iter())
                    .all(|(ai, bi)| <$scalar as approx::AbsDiffEq>::abs_diff_eq(ai, bi, epsilon))
            }
        }

        #[cfg(feature = "approx")]
        impl approx::RelativeEq for $name {
            #[inline]
            fn default_max_relative() -> Self::Epsilon {
                <$scalar as approx::RelativeEq>::default_max_relative()
            }

            #[inline]
            fn relative_eq(
                &self,
                other: &Self,
                epsilon: Self::Epsilon,
                max_relative: Self::Epsilon,
            ) -> bool {
                let a = self.to_array();
                let b = other.to_array();
                a.iter().zip(b.iter()).all(|(ai, bi)| {
                    <$scalar as approx::RelativeEq>::relative_eq(ai, bi, epsilon, max_relative)
                })
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
