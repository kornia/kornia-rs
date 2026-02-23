//! Macro to define a vector type.
//!
//! # Arguments
//!
//! * `name` - The name of the vector type.
//! * `glam_type` - The underlying glam type.
//! * `scalar` - The scalar type.
//! * `array` - The array type.
//! * `fields` - The fields of the vector.
//!
macro_rules! define_vector_type {
    ($(#[$meta:meta])* $name:ident, $glam_type:ty, $scalar:ty, $array:ty, [$($field:ident),+]) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Default)]
        pub struct $name {
            $(pub $field: $scalar),+
        }

        impl $name {
            /// Create a new vector from its components.
            #[inline]
            pub fn new($($field: $scalar),+) -> Self {
                Self { $($field),+ }
            }

            /// Create a vector from an array.
            #[inline]
            pub fn from_array(arr: $array) -> Self {
                let [$($field),+] = arr;
                Self { $($field),+ }
            }

            /// Convert the vector to an array.
            #[inline]
            pub fn to_array(self) -> $array {
                [$(self.$field),+]
            }

            /// Zero vector.
            pub const ZERO: Self = Self {
                $($field: 0.0 as $scalar),+
            };

            /// Euclidean length (magnitude) of the vector.
            #[inline]
            pub fn length(self) -> $scalar {
                let v: $glam_type = self.into();
                v.length()
            }

            /// Dot product between two vectors.
            #[inline]
            pub fn dot(self, rhs: Self) -> $scalar {
                let a: $glam_type = self.into();
                let b: $glam_type = rhs.into();
                a.dot(b)
            }

            /// Normalize the vector to unit length.
            #[inline]
            pub fn normalize(self) -> Self {
                let v: $glam_type = self.into();
                Self::from(v.normalize())
            }
        }

        // Conversions to and from the underlying glam type.
        impl From<$glam_type> for $name {
            #[inline]
            fn from(v: $glam_type) -> Self {
                Self {
                    $($field: v.$field),+
                }
            }
        }

        impl From<$name> for $glam_type {
            #[inline]
            fn from(v: $name) -> Self {
                <$glam_type>::new($(v.$field),+)
            }
        }

        // Conversions to and from arrays.
        impl From<$array> for $name {
            #[inline]
            fn from(arr: $array) -> Self {
                Self::from_array(arr)
            }
        }

        impl From<$name> for $array {
            #[inline]
            fn from(v: $name) -> Self {
                v.to_array()
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
                let a: $array = (*self).to_array();
                let b: $array = (*other).to_array();
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
                let a: $array = (*self).to_array();
                let b: $array = (*other).to_array();
                a.iter().zip(b.iter()).all(|(ai, bi)| {
                    <$scalar as approx::RelativeEq>::relative_eq(ai, bi, epsilon, max_relative)
                })
            }
        }

        // Arithmetic operations implemented via glam.
        impl std::ops::Add for $name {
            type Output = Self;

            #[inline]
            fn add(self, rhs: Self) -> Self::Output {
                let a: $glam_type = self.into();
                let b: $glam_type = rhs.into();
                Self::from(a + b)
            }
        }

        impl std::ops::Sub for $name {
            type Output = Self;

            #[inline]
            fn sub(self, rhs: Self) -> Self::Output {
                let a: $glam_type = self.into();
                let b: $glam_type = rhs.into();
                Self::from(a - b)
            }
        }

        impl std::ops::Mul<$scalar> for $name {
            type Output = Self;

            #[inline]
            fn mul(self, rhs: $scalar) -> Self::Output {
                let a: $glam_type = self.into();
                Self::from(a * rhs)
            }
        }

        impl std::ops::Mul<$name> for $scalar {
            type Output = $name;

            #[inline]
            fn mul(self, rhs: $name) -> Self::Output {
                let b: $glam_type = rhs.into();
                $name::from(self * b)
            }
        }

        impl std::ops::Div<$scalar> for $name {
            type Output = Self;

            #[inline]
            fn div(self, rhs: $scalar) -> Self::Output {
                let a: $glam_type = self.into();
                Self::from(a / rhs)
            }
        }

        impl std::ops::Neg for $name {
            type Output = Self;

            #[inline]
            fn neg(self) -> Self::Output {
                let a: $glam_type = self.into();
                Self::from(-a)
            }
        }

        impl std::ops::AddAssign for $name {
            #[inline]
            fn add_assign(&mut self, rhs: Self) {
                *self = *self + rhs;
            }
        }

        impl std::ops::SubAssign for $name {
            #[inline]
            fn sub_assign(&mut self, rhs: Self) {
                *self = *self - rhs;
            }
        }

        impl std::ops::MulAssign<$scalar> for $name {
            #[inline]
            fn mul_assign(&mut self, rhs: $scalar) {
                *self = *self * rhs;
            }
        }

        impl std::ops::DivAssign<$scalar> for $name {
            #[inline]
            fn div_assign(&mut self, rhs: $scalar) {
                *self = *self / rhs;
            }
        }
    };
}

// 2D vector types (single and double precision).
define_vector_type!(Vec2F32, glam::Vec2, f32, [f32; 2], [x, y]);
define_vector_type!(Vec2F64, glam::DVec2, f64, [f64; 2], [x, y]);

// 3D vector types (single and double precision).
define_vector_type!(Vec3F32, glam::Vec3, f32, [f32; 3], [x, y, z]);
define_vector_type!(Vec3F64, glam::DVec3, f64, [f64; 3], [x, y, z]);

// 3D vector (aligned, single precision).
define_vector_type!(Vec3AF32, glam::Vec3A, f32, [f32; 3], [x, y, z]);

// 4D vector types (single and double precision).
define_vector_type!(Vec4F32, glam::Vec4, f32, [f32; 4], [x, y, z, w]);
define_vector_type!(Vec4F64, glam::DVec4, f64, [f64; 4], [x, y, z, w]);

// Dynamic-sized vectors from nalgebra (for optimization and large-scale operations)
/// Dynamic-sized vector with f32 elements.
pub type DVecF32 = nalgebra::DVector<f32>;

/// Dynamic-sized vector with f64 elements.
pub type DVecF64 = nalgebra::DVector<f64>;

#[cfg(test)]
mod tests {
    use super::*;

    // Vec2 tests
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

    // Vec3 tests
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

    // Vec3A tests
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

    // Vec4 tests
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

    #[test]
    fn test_vec2f32_assign_ops() {
        let mut v = Vec2F32::new(1.0, 2.0);
        v += Vec2F32::new(3.0, 4.0);
        assert_eq!(v, Vec2F32::new(4.0, 6.0));

        v -= Vec2F32::new(1.0, 1.0);
        assert_eq!(v, Vec2F32::new(3.0, 5.0));

        v *= 2.0;
        assert_eq!(v, Vec2F32::new(6.0, 10.0));

        v /= 2.0;
        assert_eq!(v, Vec2F32::new(3.0, 5.0));
    }
}
