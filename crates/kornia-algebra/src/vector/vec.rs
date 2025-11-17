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
        #[derive(Debug, Clone, Copy, PartialEq)]
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
    };
}
