//! Macro to define a matrix type.
//!
//! we provide a small `macro_rules!` helper so we can easily support multiple backends and
//! precisions without copy-pasting boilerplate.
//!
//! The generated types are thin `#[repr(transparent)]` newtypes over the
//! corresponding `glam` matrix type and delegate arithmetic to `glam`.
//!
//! # Arguments
//!
//! * `name`        - The name of the matrix type.
//! * `glam_type`   - The underlying `glam` matrix type.
//! * `array`       - The column-major array type (e.g. `[f32; 4]` for 2x2).
//! * `vec_type`    - The public vector type used for columns and mat-vec mul.
//! * `glam_vec`    - The underlying `glam` vector type.
//! * `cols`        - The column parameters (e.g. `[x_axis, y_axis]`).
//!
macro_rules! define_matrix_type {
    (
        $(#[$meta:meta])*
        $name:ident,
        $glam_type:ty,
        $array:ty,
        $vec_type:ty,
        $glam_vec:ty,
        [$($col:ident),+]
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq)]
        #[repr(transparent)]
        pub struct $name(pub $glam_type);

        impl $name {
            /// Create a new matrix from column vectors.
            #[inline]
            pub fn from_cols($($col: $vec_type),+) -> Self {
                Self(<$glam_type>::from_cols(
                    $(<$glam_vec>::from($col)),+
                ))
            }

            /// Create a new matrix from a column-major array.
            #[inline]
            pub fn from_cols_array(arr: &$array) -> Self {
                Self(<$glam_type>::from_cols_array(arr))
            }

            /// Identity matrix.
            pub const IDENTITY: Self = Self(<$glam_type>::IDENTITY);
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
            fn from(m: $glam_type) -> Self {
                Self(m)
            }
        }

        impl From<$name> for $glam_type {
            #[inline]
            fn from(m: $name) -> Self {
                m.0
            }
        }

        // Conversions to and from column-major arrays.
        impl From<$array> for $name {
            #[inline]
            fn from(arr: $array) -> Self {
                Self(<$glam_type>::from_cols_array(&arr))
            }
        }

        impl From<$name> for $array {
            #[inline]
            fn from(m: $name) -> Self {
                m.to_cols_array()
            }
        }

        // Matrix-matrix multiplication.
        impl std::ops::Mul<$name> for $name {
            type Output = $name;

            #[inline]
            fn mul(self, rhs: $name) -> Self::Output {
                $name::from(self.0 * rhs.0)
            }
        }

        // Matrix-vector multiplication.
        impl std::ops::Mul<$vec_type> for $name {
            type Output = $vec_type;

            #[inline]
            fn mul(self, rhs: $vec_type) -> Self::Output {
                <$vec_type>::from(self.0 * <$glam_vec>::from(rhs))
            }
        }
    };
}
