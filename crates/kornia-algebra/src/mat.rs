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
//! * `scalar`      - The scalar type (e.g. `f32` or `f64`).
//! * `array`       - The column-major array type (e.g. `[f32; 4]` for 2x2).
//! * `vec_type`    - The public vector type used for columns and mat-vec mul.
//! * `glam_vec`    - The underlying `glam` vector type.
//! * `cols`        - The column parameters (e.g. `[x_axis, y_axis]`).
//!

use crate::{QuatF32, QuatF64, Vec2F32, Vec2F64, Vec3AF32, Vec3F32, Vec3F64, Vec4F32, Vec4F64};

macro_rules! define_matrix_type {
    (
        $(#[$meta:meta])*
        $name:ident,
        $glam_type:ty,
        $scalar:ty,
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

            /// Zero matrix.
            pub const ZERO: Self = Self(<$glam_type>::ZERO);

            /// Transpose the matrix.
            #[inline]
            pub fn transpose(self) -> Self {
                Self::from(self.0.transpose())
            }

            /// Get the inverse of the matrix.
            #[inline]
            pub fn inverse(self) -> Self {
                Self::from(self.0.inverse())
            }

            /// Get the determinant of the matrix.
            #[inline]
            pub fn determinant(self) -> $scalar {
                self.0.determinant()
            }

            /// Check if all elements are finite.
            #[inline]
            pub fn is_finite(self) -> bool {
                self.0.is_finite()
            }

            /// Create a diagonal matrix from a vector.
            #[inline]
            pub fn from_diagonal(diagonal: $vec_type) -> Self {
                Self(<$glam_type>::from_diagonal(<$glam_vec>::from(diagonal).into()))
            }

            $(
            /// Get the $col column vector.
            #[inline]
            pub fn $col(&self) -> $vec_type {
                <$vec_type>::from(self.0.$col)
            }
            )+
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

        #[cfg(feature = "approx")]
        impl approx::AbsDiffEq for $name {
            type Epsilon = <$scalar as approx::AbsDiffEq>::Epsilon;

            #[inline]
            fn default_epsilon() -> Self::Epsilon {
                <$scalar as approx::AbsDiffEq>::default_epsilon()
            }

            #[inline]
            fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
                let a: $array = (*self).into();
                let b: $array = (*other).into();
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
                let a: $array = (*self).into();
                let b: $array = (*other).into();
                a.iter().zip(b.iter()).all(|(ai, bi)| {
                    <$scalar as approx::RelativeEq>::relative_eq(ai, bi, epsilon, max_relative)
                })
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

        // Matrix addition
        impl std::ops::Add<$name> for $name {
            type Output = $name;

            #[inline]
            fn add(self, rhs: $name) -> Self::Output {
                $name::from(self.0 + rhs.0)
            }
        }

        // Matrix subtraction
        impl std::ops::Sub<$name> for $name {
            type Output = $name;

            #[inline]
            fn sub(self, rhs: $name) -> Self::Output {
                $name::from(self.0 - rhs.0)
            }
        }

        // Scalar multiplication
        impl std::ops::Mul<$scalar> for $name {
            type Output = $name;

            #[inline]
            fn mul(self, rhs: $scalar) -> Self::Output {
                $name::from(self.0 * rhs)
            }
        }

        // Scalar multiplication (reverse)
        impl std::ops::Mul<$name> for $scalar {
            type Output = $name;

            #[inline]
            fn mul(self, rhs: $name) -> Self::Output {
                $name::from(self * rhs.0)
            }
        }

        // Matrix addition assignment
        impl std::ops::AddAssign<$name> for $name {
            #[inline]
            fn add_assign(&mut self, rhs: $name) {
                self.0 += rhs.0;
            }
        }

        // Matrix subtraction assignment
        impl std::ops::SubAssign<$name> for $name {
            #[inline]
            fn sub_assign(&mut self, rhs: $name) {
                self.0 -= rhs.0;
            }
        }

        // Matrix multiplication assignment
        impl std::ops::MulAssign<$name> for $name {
            #[inline]
            fn mul_assign(&mut self, rhs: $name) {
                self.0 *= rhs.0;
            }
        }

        // Scalar multiplication assignment
        impl std::ops::MulAssign<$scalar> for $name {
            #[inline]
            fn mul_assign(&mut self, rhs: $scalar) {
                self.0 *= rhs;
            }
        }
    };
}

// 2x2 matrix (single and double precision).
define_matrix_type!(
    Mat2F32,
    glam::Mat2,
    f32,
    [f32; 4],
    Vec2F32,
    glam::Vec2,
    [x_axis, y_axis]
);

define_matrix_type!(
    Mat2F64,
    glam::DMat2,
    f64,
    [f64; 4],
    Vec2F64,
    glam::DVec2,
    [x_axis, y_axis]
);

// 3x3 matrix (single and double precision).
define_matrix_type!(
    Mat3F32,
    glam::Mat3,
    f32,
    [f32; 9],
    Vec3F32,
    glam::Vec3,
    [x_axis, y_axis, z_axis]
);

define_matrix_type!(
    Mat3F64,
    glam::DMat3,
    f64,
    [f64; 9],
    Vec3F64,
    glam::DVec3,
    [x_axis, y_axis, z_axis]
);

impl Mat3F64 {
    /// Create a 3x3 rotation matrix from a quaternion.
    #[inline]
    pub fn from_quat(quat: QuatF64) -> Self {
        Self(glam::DMat3::from_quat(quat.0))
    }

    /// Create a matrix from a row-major flat array.
    ///
    /// The input is interpreted as:
    ///
    /// ```text
    /// m[0] m[1] m[2]   row 0
    /// m[3] m[4] m[5]   row 1
    /// m[6] m[7] m[8]   row 2
    /// ```
    ///
    /// This is the conventional mathematical layout (and the layout used by
    /// NumPy, Eigen, and most textbooks). Internally the matrix is stored in
    /// the column-major order that `glam` requires, so the constructor
    /// transposes the input.
    ///
    /// # Arguments
    ///
    /// * `m` - Row-major flat array `[R00, R01, R02, R10, R11, R12, R20, R21, R22]`.
    ///
    /// # Returns
    ///
    /// A `Mat3F64` whose `(i, j)` entry equals `m[i * 3 + j]`.
    ///
    /// # Errors
    ///
    /// This method does not return an error.
    ///
    /// # Example
    ///
    /// ```rust
    /// use kornia_algebra::Mat3F64;
    ///
    /// // Identity supplied row-major.
    /// let m = Mat3F64::from_rows_array(&[
    ///     1.0, 0.0, 0.0,
    ///     0.0, 1.0, 0.0,
    ///     0.0, 0.0, 1.0,
    /// ]);
    /// assert_eq!(m, Mat3F64::IDENTITY);
    /// ```
    #[inline]
    pub fn from_rows_array(m: &[f64; 9]) -> Self {
        // glam DMat3::from_cols_array expects column-major order:
        // [col0_row0, col0_row1, col0_row2, col1_row0, ...].
        // A row-major input m[row * 3 + col] becomes col-major as m[col * 3 + row].
        Self(glam::DMat3::from_cols_array(&[
            m[0], m[3], m[6], // column 0: (R00, R10, R20)
            m[1], m[4], m[7], // column 1: (R01, R11, R21)
            m[2], m[5], m[8], // column 2: (R02, R12, R22)
        ]))
    }

    /// Return `true` if this matrix is a valid rotation matrix within `tol`.
    ///
    /// A matrix R is a rotation matrix (element of SO(3)) if and only if:
    ///
    /// 1. **Orthonormality**: `Rᵀ R ≈ I` — every column is a unit vector and
    ///    all columns are mutually orthogonal.
    /// 2. **Proper handedness**: `det(R) ≈ +1` — the matrix preserves orientation.
    ///
    /// Both conditions are checked against `tol`. Specifically:
    ///
    /// - The maximum absolute entry of `Rᵀ R − I` must be `≤ tol`.
    /// - `det(R)` must be strictly positive (`det(R) > 0`), and `|det(R) − 1|` must be `≤ tol`.
    ///   Matrices with non-positive determinants (such as reflections where `det(R) = -1`)
    ///   are always rejected, even if a large tolerance would otherwise encompass them.
    ///
    /// Tolerance `tol` must be finite and non-negative. Non-finite tolerances (NaN, +Inf, -Inf)
    /// and negative tolerances are rejected immediately. Returns `false` for any non-finite entry,
    /// invalid tolerance, or if intermediate calculations produce non-finite values.
    ///
    /// # Arguments
    ///
    /// * `self` - The matrix to test.
    /// * `tol`  - Non-negative, finite tolerance; `1e-6` is a reasonable default for
    ///   `f64` matrices rounded through intermediate `f32` arithmetic. Non-finite tolerances
    ///   (such as `f64::NAN` or `f64::INFINITY`) are rejected.
    ///
    /// # Returns
    ///
    /// `true` if `self` is a rotation matrix to within `tol`, `false` otherwise.
    ///
    /// # Errors
    ///
    /// This method does not return an error. It returns `false` for
    /// invalid tolerances (negative, non-finite) or non-finite matrix values.
    ///
    /// # Example
    ///
    /// ```rust
    /// use kornia_algebra::Mat3F64;
    ///
    /// assert!(Mat3F64::IDENTITY.is_rotation(1e-10));
    ///
    /// // A reflection has det = -1, so it is not a rotation, even with a large tolerance.
    /// let reflection = Mat3F64::from_rows_array(&[
    ///     -1.0, 0.0, 0.0,
    ///      0.0, 1.0, 0.0,
    ///      0.0, 0.0, 1.0,
    /// ]);
    /// assert!(!reflection.is_rotation(1e-10));
    /// assert!(!reflection.is_rotation(2.0));
    /// ```
    pub fn is_rotation(&self, tol: f64) -> bool {
        // Reject non-finite entries or invalid tolerance early.
        if !self.is_finite() || !tol.is_finite() || tol < 0.0 {
            return false;
        }
        // Check orthonormality: max |entry of (RᵀR - I)| <= tol.
        let ata = self.transpose() * *self;
        let diff = ata - Mat3F64::IDENTITY;
        let cols_arr = diff.to_cols_array();
        let max_err = cols_arr.iter().fold(0.0_f64, |acc, &v| {
            if v.is_nan() || acc.is_nan() {
                f64::NAN
            } else {
                acc.max(v.abs())
            }
        });
        if !max_err.is_finite() || max_err > tol {
            return false;
        }
        // Check proper handedness: det > 0 and |det - 1| <= tol.
        let det = self.determinant();
        if !det.is_finite() || det <= 0.0 || (det - 1.0_f64).abs() > tol {
            return false;
        }
        true
    }
}

// 3x3 matrix (aligned, single precision).
define_matrix_type!(
    Mat3AF32,
    glam::Mat3A,
    f32,
    [f32; 9],
    Vec3AF32,
    glam::Vec3A,
    [x_axis, y_axis, z_axis]
);

impl Mat3AF32 {
    /// Create a 3x3 rotation matrix from a quaternion.
    #[inline]
    pub fn from_quat(quat: QuatF32) -> Self {
        Self(glam::Mat3A::from_quat(quat.0))
    }
}

// 4x4 matrix (single and double precision).
define_matrix_type!(
    Mat4F32,
    glam::Mat4,
    f32,
    [f32; 16],
    Vec4F32,
    glam::Vec4,
    [x_axis, y_axis, z_axis, w_axis]
);

define_matrix_type!(
    Mat4F64,
    glam::DMat4,
    f64,
    [f64; 16],
    Vec4F64,
    glam::DVec4,
    [x_axis, y_axis, z_axis, w_axis]
);

// Dynamic-sized matrices from nalgebra (for optimization and large-scale operations)
/// Dynamic-sized matrix with f32 elements.
pub type DMatF32 = nalgebra::DMatrix<f32>;

/// Dynamic-sized matrix with f64 elements.
pub type DMatF64 = nalgebra::DMatrix<f64>;

// Fixed 6x6 from nalgebra: glam stops at 4x4. Six is the tangent size of SE(3) and of a
// rotation-plus-translation pose, so normal equations of a pose solve live here.
/// 6x6 matrix with f64 elements, e.g. the `J^T W J` normal matrix of a 6-DoF pose solve.
///
/// A stack value (no allocation). Factorisations are nalgebra's: `cholesky()`,
/// `symmetric_eigen()`.
pub type Mat6F64 = nalgebra::Matrix6<f64>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vec6F64;

    #[test]
    fn mat6_cholesky_solves_spd_system() {
        // A = M^T M + I is symmetric positive definite
        let m = Mat6F64::from_fn(|i, k| ((i * 7 + k * 3) % 5) as f64 - 2.0);
        let a = m.transpose() * m + Mat6F64::identity();
        let x = Vec6F64::new(1.0, -2.0, 0.5, 3.0, -0.25, 4.0);
        let b = a * x;
        let solved = a.cholesky().map(|c| c.solve(&b)).unwrap();
        assert!((solved - x).amax() < 1e-12);
    }

    // Mat2 tests
    #[test]
    fn test_mat2f32_from_cols() {
        let col1 = Vec2F32::new(1.0, 2.0);
        let col2 = Vec2F32::new(3.0, 4.0);
        let m = Mat2F32::from_cols(col1, col2);
        assert_eq!(m.x_axis(), Vec2F32::new(1.0, 2.0));
        assert_eq!(m.y_axis(), Vec2F32::new(3.0, 4.0));
    }

    #[test]
    fn test_mat2f32_from_cols_array() {
        let arr = [1.0, 2.0, 3.0, 4.0];
        let m = Mat2F32::from_cols_array(&arr);
        assert_eq!(m.x_axis(), Vec2F32::new(1.0, 2.0));
        assert_eq!(m.y_axis(), Vec2F32::new(3.0, 4.0));
    }

    #[test]
    fn test_mat2f32_identity() {
        let m = Mat2F32::IDENTITY;
        assert_eq!(m.x_axis(), Vec2F32::new(1.0, 0.0));
        assert_eq!(m.y_axis(), Vec2F32::new(0.0, 1.0));
    }

    #[test]
    fn test_mat2f32_transpose() {
        let m = Mat2F32::from_cols_array(&[1.0, 2.0, 3.0, 4.0]);
        let mt = m.transpose();
        assert_eq!(mt.x_axis(), Vec2F32::new(1.0, 3.0));
        assert_eq!(mt.y_axis(), Vec2F32::new(2.0, 4.0));
    }

    #[test]
    fn test_mat2f32_inverse() {
        let m = Mat2F32::from_cols_array(&[1.0, 0.0, 0.0, 2.0]); // diagonal matrix
        let inv = m.inverse();
        assert_eq!(inv.x_axis(), Vec2F32::new(1.0, 0.0));
        assert_eq!(inv.y_axis(), Vec2F32::new(0.0, 0.5));
    }

    #[test]
    fn test_mat2f32_determinant() {
        let m = Mat2F32::from_cols_array(&[1.0, 2.0, 3.0, 4.0]);
        // det = 1*4 - 3*2 = 4 - 6 = -2
        assert_eq!(m.determinant(), -2.0);
    }

    #[test]
    fn test_mat2f32_conversions() {
        let arr = [1.0, 2.0, 3.0, 4.0];
        let m: Mat2F32 = arr.into();
        assert_eq!(m.x_axis(), Vec2F32::new(1.0, 2.0));
        assert_eq!(m.y_axis(), Vec2F32::new(3.0, 4.0));

        let m_glam: glam::Mat2 = m.into();
        assert_eq!(m_glam.x_axis, glam::Vec2::new(1.0, 2.0));
        assert_eq!(m_glam.y_axis, glam::Vec2::new(3.0, 4.0));

        let m_back: Mat2F32 = m_glam.into();
        assert_eq!(m_back, m);

        let arr_back: [f32; 4] = m.into();
        assert_eq!(arr_back, arr);
    }

    #[test]
    fn test_mat2f32_mul_vec2() {
        let m = Mat2F32::IDENTITY;
        let v = Vec2F32::new(1.0, 2.0);
        let result = m * v;
        assert_eq!(result, v);
    }

    #[test]
    fn test_mat2f64_mul_vec2() {
        let m = Mat2F64::IDENTITY;
        let v = Vec2F64::new(1.0, 2.0);
        let result = m * v;
        assert_eq!(result, v);
    }

    #[test]
    fn test_mat2f64_from_cols() {
        let col1 = Vec2F64::new(1.0, 2.0);
        let col2 = Vec2F64::new(3.0, 4.0);
        let m = Mat2F64::from_cols(col1, col2);
        assert_eq!(m.x_axis(), Vec2F64::new(1.0, 2.0));
        assert_eq!(m.y_axis(), Vec2F64::new(3.0, 4.0));
    }

    #[test]
    fn test_mat2f64_from_cols_array() {
        let arr = [1.0, 2.0, 3.0, 4.0];
        let m = Mat2F64::from_cols_array(&arr);
        assert_eq!(m.x_axis(), Vec2F64::new(1.0, 2.0));
        assert_eq!(m.y_axis(), Vec2F64::new(3.0, 4.0));
    }

    #[test]
    fn test_mat2f64_identity() {
        let m = Mat2F64::IDENTITY;
        assert_eq!(m.x_axis(), Vec2F64::new(1.0, 0.0));
        assert_eq!(m.y_axis(), Vec2F64::new(0.0, 1.0));
    }

    #[test]
    fn test_mat2f64_transpose() {
        let m = Mat2F64::from_cols_array(&[1.0, 2.0, 3.0, 4.0]);
        let mt = m.transpose();
        assert_eq!(mt.x_axis(), Vec2F64::new(1.0, 3.0));
        assert_eq!(mt.y_axis(), Vec2F64::new(2.0, 4.0));
    }

    #[test]
    fn test_mat2f64_inverse() {
        let m = Mat2F64::from_cols_array(&[1.0, 0.0, 0.0, 2.0]); // diagonal matrix
        let inv = m.inverse();
        assert_eq!(inv.x_axis(), Vec2F64::new(1.0, 0.0));
        assert_eq!(inv.y_axis(), Vec2F64::new(0.0, 0.5));
    }

    #[test]
    fn test_mat2f64_determinant() {
        let m = Mat2F64::from_cols_array(&[1.0, 2.0, 3.0, 4.0]);
        // det = 1*4 - 3*2 = 4 - 6 = -2
        assert_eq!(m.determinant(), -2.0);
    }

    #[test]
    fn test_mat2f64_conversions() {
        let arr = [1.0, 2.0, 3.0, 4.0];
        let m: Mat2F64 = arr.into();
        assert_eq!(m.x_axis(), Vec2F64::new(1.0, 2.0));
        assert_eq!(m.y_axis(), Vec2F64::new(3.0, 4.0));

        let m_glam: glam::DMat2 = m.into();
        assert_eq!(m_glam.x_axis, glam::DVec2::new(1.0, 2.0));
        assert_eq!(m_glam.y_axis, glam::DVec2::new(3.0, 4.0));

        let m_back: Mat2F64 = m_glam.into();
        assert_eq!(m_back, m);

        let arr_back: [f64; 4] = m.into();
        assert_eq!(arr_back, arr);
    }

    // Mat3 tests
    #[test]
    fn test_mat3f32_from_cols() {
        let col1 = Vec3F32::new(1.0, 2.0, 3.0);
        let col2 = Vec3F32::new(4.0, 5.0, 6.0);
        let col3 = Vec3F32::new(7.0, 8.0, 9.0);
        let m = Mat3F32::from_cols(col1, col2, col3);
        assert_eq!(m.x_axis(), Vec3F32::new(1.0, 2.0, 3.0));
        assert_eq!(m.y_axis(), Vec3F32::new(4.0, 5.0, 6.0));
        assert_eq!(m.z_axis(), Vec3F32::new(7.0, 8.0, 9.0));
    }

    #[test]
    fn test_mat3f32_from_cols_array() {
        let arr = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let m = Mat3F32::from_cols_array(&arr);
        assert_eq!(m.x_axis(), Vec3F32::new(1.0, 2.0, 3.0));
        assert_eq!(m.y_axis(), Vec3F32::new(4.0, 5.0, 6.0));
        assert_eq!(m.z_axis(), Vec3F32::new(7.0, 8.0, 9.0));
    }

    #[test]
    fn test_mat3f32_identity() {
        let m = Mat3F32::IDENTITY;
        assert_eq!(m.x_axis(), Vec3F32::new(1.0, 0.0, 0.0));
        assert_eq!(m.y_axis(), Vec3F32::new(0.0, 1.0, 0.0));
        assert_eq!(m.z_axis(), Vec3F32::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn test_mat3f32_transpose() {
        let m = Mat3F32::from_cols_array(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let mt = m.transpose();
        assert_eq!(mt.x_axis(), Vec3F32::new(1.0, 4.0, 7.0));
        assert_eq!(mt.y_axis(), Vec3F32::new(2.0, 5.0, 8.0));
        assert_eq!(mt.z_axis(), Vec3F32::new(3.0, 6.0, 9.0));
    }

    #[test]
    fn test_mat3f32_inverse() {
        let m = Mat3F32::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.5]);
        let inv = m.inverse();
        assert_eq!(inv.x_axis(), Vec3F32::new(1.0, 0.0, 0.0));
        assert_eq!(inv.y_axis(), Vec3F32::new(0.0, 0.5, 0.0));
        assert_eq!(inv.z_axis(), Vec3F32::new(0.0, 0.0, 2.0));
    }

    #[test]
    fn test_mat3f32_determinant() {
        let m = Mat3F32::from_cols_array(&[2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0]);
        assert_eq!(m.determinant(), 24.0);
    }

    #[test]
    fn test_mat3f32_conversions() {
        let arr = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let m: Mat3F32 = arr.into();
        assert_eq!(m.x_axis(), Vec3F32::new(1.0, 2.0, 3.0));

        let m_glam: glam::Mat3 = m.into();
        assert_eq!(m_glam.x_axis, glam::Vec3::new(1.0, 2.0, 3.0));

        let m_back: Mat3F32 = m_glam.into();
        assert_eq!(m_back, m);

        let arr_back: [f32; 9] = m.into();
        assert_eq!(arr_back, arr);
    }

    #[test]
    fn test_mat3f32_mul_vec3() {
        let m = Mat3F32::IDENTITY;
        let v = Vec3F32::new(1.0, 2.0, 3.0);
        let result = m * v;
        assert_eq!(result, v);
    }

    #[test]
    fn test_mat3f64_from_cols() {
        let col1 = Vec3F64::new(1.0, 2.0, 3.0);
        let col2 = Vec3F64::new(4.0, 5.0, 6.0);
        let col3 = Vec3F64::new(7.0, 8.0, 9.0);
        let m = Mat3F64::from_cols(col1, col2, col3);
        assert_eq!(m.x_axis(), Vec3F64::new(1.0, 2.0, 3.0));
        assert_eq!(m.y_axis(), Vec3F64::new(4.0, 5.0, 6.0));
        assert_eq!(m.z_axis(), Vec3F64::new(7.0, 8.0, 9.0));
    }

    #[test]
    fn test_mat3f64_from_cols_array() {
        let arr = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let m = Mat3F64::from_cols_array(&arr);
        assert_eq!(m.x_axis(), Vec3F64::new(1.0, 2.0, 3.0));
        assert_eq!(m.y_axis(), Vec3F64::new(4.0, 5.0, 6.0));
        assert_eq!(m.z_axis(), Vec3F64::new(7.0, 8.0, 9.0));
    }

    #[test]
    fn test_mat3f64_from_rows_array() {
        // Row-major input:
        // [ 1.0, 2.0, 3.0 ]  row 0
        // [ 4.0, 5.0, 6.0 ]  row 1
        // [ 7.0, 8.0, 9.0 ]  row 2
        let row_major = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let m = Mat3F64::from_rows_array(&row_major);

        // Columns are the transposes of rows:
        assert_eq!(m.x_axis(), Vec3F64::new(1.0, 4.0, 7.0));
        assert_eq!(m.y_axis(), Vec3F64::new(2.0, 5.0, 8.0));
        assert_eq!(m.z_axis(), Vec3F64::new(3.0, 6.0, 9.0));

        // Matches column-major array of transposed entries:
        let col_major = [1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0];
        let m_col = Mat3F64::from_cols_array(&col_major);
        assert_eq!(m, m_col);

        // from_rows_array is the transpose of from_cols_array on identical flat buffers:
        let m_direct_cols = Mat3F64::from_cols_array(&row_major);
        assert_eq!(m, m_direct_cols.transpose());
    }

    #[test]
    fn test_mat3f64_is_rotation() {
        // 1. Identity is a rotation.
        assert!(Mat3F64::IDENTITY.is_rotation(1e-12));

        // 2. Exact 90-degree rotations.
        let rot_z90 = Mat3F64::from_rows_array(&[0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        assert!(rot_z90.is_rotation(1e-12));

        let rot_x90 = Mat3F64::from_rows_array(&[1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0]);
        assert!(rot_x90.is_rotation(1e-12));

        let rot_y90 = Mat3F64::from_rows_array(&[0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0]);
        assert!(rot_y90.is_rotation(1e-12));

        // 3. Small perturbation within vs outside tolerance.
        let mut perturbed = Mat3F64::IDENTITY.to_cols_array();
        perturbed[0] += 1e-7;
        let m_perturbed = Mat3F64::from_cols_array(&perturbed);
        assert!(m_perturbed.is_rotation(1e-6));
        assert!(!m_perturbed.is_rotation(1e-8));

        // 4. Reflection (orthogonal, det = -1).
        let reflection = Mat3F64::from_rows_array(&[-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        assert!(!reflection.is_rotation(1e-6));
        // Large tolerance must not accept reflection with non-positive determinant:
        assert!(!reflection.is_rotation(2.0));
        assert!(!reflection.is_rotation(10.0));

        // 5. Scaling (det != 1, R^T R != I).
        let scaled = Mat3F64::from_rows_array(&[2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0]);
        assert!(!scaled.is_rotation(1e-6));

        // 6. Shear (non-orthogonal).
        let shear = Mat3F64::from_rows_array(&[1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        assert!(!shear.is_rotation(1e-6));

        // 7. Non-finite values.
        let with_nan =
            Mat3F64::from_rows_array(&[f64::NAN, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        assert!(!with_nan.is_rotation(1e-6));

        let with_inf =
            Mat3F64::from_rows_array(&[f64::INFINITY, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        assert!(!with_inf.is_rotation(1e-6));

        // 8. Invalid tolerance (negative, NaN, or infinite).
        assert!(!Mat3F64::IDENTITY.is_rotation(-1e-6));
        assert!(!Mat3F64::IDENTITY.is_rotation(f64::NAN));
        assert!(!Mat3F64::IDENTITY.is_rotation(f64::INFINITY));
    }

    #[test]
    fn test_mat3f64_is_rotation_overflow() {
        // Finite entries that overflow during RᵀR:
        let large_val = 1e200_f64;
        let large_mat = Mat3F64::from_rows_array(&[
            large_val, 0.0, 0.0, 0.0, large_val, 0.0, 0.0, 0.0, large_val,
        ]);
        assert!(large_mat.is_finite());
        assert!(!large_mat.is_rotation(1e-6));

        // Matrix with f64::MAX:
        let max_mat = Mat3F64::from_rows_array(&[f64::MAX, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        assert!(max_mat.is_finite());
        assert!(!max_mat.is_rotation(1e-6));

        // Finite matrix whose RᵀR off-diagonal dot-product produces +Inf - Inf = NaN:
        let nan_prod_mat =
            Mat3F64::from_rows_array(&[1e200, -1e200, 0.0, 1e200, 1e200, 0.0, 0.0, 0.0, 1.0]);
        assert!(nan_prod_mat.is_finite());
        assert!(!nan_prod_mat.is_rotation(1e-6));
    }

    #[test]
    fn test_mat3f64_identity() {
        let m = Mat3F64::IDENTITY;
        assert_eq!(m.x_axis(), Vec3F64::new(1.0, 0.0, 0.0));
        assert_eq!(m.y_axis(), Vec3F64::new(0.0, 1.0, 0.0));
        assert_eq!(m.z_axis(), Vec3F64::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn test_mat3f64_transpose() {
        let m = Mat3F64::from_cols_array(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let mt = m.transpose();
        assert_eq!(mt.x_axis(), Vec3F64::new(1.0, 4.0, 7.0));
        assert_eq!(mt.y_axis(), Vec3F64::new(2.0, 5.0, 8.0));
        assert_eq!(mt.z_axis(), Vec3F64::new(3.0, 6.0, 9.0));
    }

    #[test]
    fn test_mat3f64_inverse() {
        let m = Mat3F64::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.5]);
        let inv = m.inverse();
        assert_eq!(inv.x_axis(), Vec3F64::new(1.0, 0.0, 0.0));
        assert_eq!(inv.y_axis(), Vec3F64::new(0.0, 0.5, 0.0));
        assert_eq!(inv.z_axis(), Vec3F64::new(0.0, 0.0, 2.0));
    }

    #[test]
    fn test_mat3f64_determinant() {
        let m = Mat3F64::from_cols_array(&[2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0]);
        assert_eq!(m.determinant(), 24.0);
    }

    #[test]
    fn test_mat3f64_conversions() {
        let arr = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let m: Mat3F64 = arr.into();
        assert_eq!(m.x_axis(), Vec3F64::new(1.0, 2.0, 3.0));

        let m_glam: glam::DMat3 = m.into();
        assert_eq!(m_glam.x_axis, glam::DVec3::new(1.0, 2.0, 3.0));

        let m_back: Mat3F64 = m_glam.into();
        assert_eq!(m_back, m);

        let arr_back: [f64; 9] = m.into();
        assert_eq!(arr_back, arr);
    }

    #[test]
    fn test_mat3f64_mul_vec3() {
        let m = Mat3F64::IDENTITY;
        let v = Vec3F64::new(1.0, 2.0, 3.0);
        let result = m * v;
        assert_eq!(result, v);
    }

    #[test]
    fn test_inner_field_access() {
        let m = Mat3F32::IDENTITY;
        // This confirms the inner field .0 is accessible
        let _inner = m.0;
        assert_eq!(m.0, glam::Mat3::IDENTITY);
    }

    // Mat3A tests
    #[test]
    fn test_mat3af32_from_cols() {
        let col1 = Vec3AF32::new(1.0, 2.0, 3.0);
        let col2 = Vec3AF32::new(4.0, 5.0, 6.0);
        let col3 = Vec3AF32::new(7.0, 8.0, 9.0);
        let m = Mat3AF32::from_cols(col1, col2, col3);
        assert_eq!(m.x_axis(), Vec3AF32::new(1.0, 2.0, 3.0));
        assert_eq!(m.y_axis(), Vec3AF32::new(4.0, 5.0, 6.0));
        assert_eq!(m.z_axis(), Vec3AF32::new(7.0, 8.0, 9.0));
    }

    #[test]
    fn test_mat3af32_from_cols_array() {
        let arr = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let m = Mat3AF32::from_cols_array(&arr);
        assert_eq!(m.x_axis(), Vec3AF32::new(1.0, 2.0, 3.0));
        assert_eq!(m.y_axis(), Vec3AF32::new(4.0, 5.0, 6.0));
        assert_eq!(m.z_axis(), Vec3AF32::new(7.0, 8.0, 9.0));
    }

    #[test]
    fn test_mat3af32_identity() {
        let m = Mat3AF32::IDENTITY;
        assert_eq!(m.x_axis(), Vec3AF32::new(1.0, 0.0, 0.0));
        assert_eq!(m.y_axis(), Vec3AF32::new(0.0, 1.0, 0.0));
        assert_eq!(m.z_axis(), Vec3AF32::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn test_mat3af32_transpose() {
        let m = Mat3AF32::from_cols_array(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let mt = m.transpose();
        assert_eq!(mt.x_axis(), Vec3AF32::new(1.0, 4.0, 7.0));
        assert_eq!(mt.y_axis(), Vec3AF32::new(2.0, 5.0, 8.0));
        assert_eq!(mt.z_axis(), Vec3AF32::new(3.0, 6.0, 9.0));
    }

    #[test]
    fn test_mat3af32_inverse() {
        let m = Mat3AF32::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.5]);
        let inv = m.inverse();
        assert_eq!(inv.x_axis(), Vec3AF32::new(1.0, 0.0, 0.0));
        assert_eq!(inv.y_axis(), Vec3AF32::new(0.0, 0.5, 0.0));
        assert_eq!(inv.z_axis(), Vec3AF32::new(0.0, 0.0, 2.0));
    }

    #[test]
    fn test_mat3af32_determinant() {
        let m = Mat3AF32::from_cols_array(&[2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0]);
        assert_eq!(m.determinant(), 24.0);
    }

    #[test]
    fn test_mat3af32_conversions() {
        let arr = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let m: Mat3AF32 = arr.into();
        assert_eq!(m.x_axis(), Vec3AF32::new(1.0, 2.0, 3.0));

        let m_glam: glam::Mat3A = m.into();
        assert_eq!(m_glam.x_axis, glam::Vec3A::new(1.0, 2.0, 3.0));

        let m_back: Mat3AF32 = m_glam.into();
        assert_eq!(m_back, m);

        let arr_back: [f32; 9] = m.into();
        assert_eq!(arr_back, arr);
    }

    #[test]
    fn test_mat3af32_mul_vec3af32() {
        let m = Mat3AF32::IDENTITY;
        let v = Vec3AF32::new(1.0, 2.0, 3.0);
        let result = m * v;
        assert_eq!(result, v);
    }

    // Mat4 tests
    #[test]
    fn test_mat4f32_from_cols() {
        let col1 = Vec4F32::new(1.0, 2.0, 3.0, 4.0);
        let col2 = Vec4F32::new(5.0, 6.0, 7.0, 8.0);
        let col3 = Vec4F32::new(9.0, 10.0, 11.0, 12.0);
        let col4 = Vec4F32::new(13.0, 14.0, 15.0, 16.0);
        let m = Mat4F32::from_cols(col1, col2, col3, col4);
        assert_eq!(m.x_axis(), Vec4F32::new(1.0, 2.0, 3.0, 4.0));
        assert_eq!(m.y_axis(), Vec4F32::new(5.0, 6.0, 7.0, 8.0));
        assert_eq!(m.z_axis(), Vec4F32::new(9.0, 10.0, 11.0, 12.0));
        assert_eq!(m.w_axis(), Vec4F32::new(13.0, 14.0, 15.0, 16.0));
    }

    #[test]
    fn test_mat4f32_from_cols_array() {
        let arr = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let m = Mat4F32::from_cols_array(&arr);
        assert_eq!(m.x_axis(), Vec4F32::new(1.0, 2.0, 3.0, 4.0));
        assert_eq!(m.y_axis(), Vec4F32::new(5.0, 6.0, 7.0, 8.0));
        assert_eq!(m.z_axis(), Vec4F32::new(9.0, 10.0, 11.0, 12.0));
        assert_eq!(m.w_axis(), Vec4F32::new(13.0, 14.0, 15.0, 16.0));
    }

    #[test]
    fn test_mat4f32_identity() {
        let m = Mat4F32::IDENTITY;
        assert_eq!(m.x_axis(), Vec4F32::new(1.0, 0.0, 0.0, 0.0));
        assert_eq!(m.y_axis(), Vec4F32::new(0.0, 1.0, 0.0, 0.0));
        assert_eq!(m.z_axis(), Vec4F32::new(0.0, 0.0, 1.0, 0.0));
        assert_eq!(m.w_axis(), Vec4F32::new(0.0, 0.0, 0.0, 1.0));
    }

    #[test]
    fn test_mat4f32_transpose() {
        let m = Mat4F32::from_cols_array(&[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ]);
        let mt = m.transpose();
        assert_eq!(mt.x_axis(), Vec4F32::new(1.0, 5.0, 9.0, 13.0));
        assert_eq!(mt.y_axis(), Vec4F32::new(2.0, 6.0, 10.0, 14.0));
        assert_eq!(mt.z_axis(), Vec4F32::new(3.0, 7.0, 11.0, 15.0));
        assert_eq!(mt.w_axis(), Vec4F32::new(4.0, 8.0, 12.0, 16.0));
    }

    #[test]
    fn test_mat4f32_inverse() {
        let m = Mat4F32::from_cols_array(&[
            1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.5,
        ]); // diagonal matrix
        let inv = m.inverse();
        assert_eq!(inv.x_axis(), Vec4F32::new(1.0, 0.0, 0.0, 0.0));
        assert_eq!(inv.y_axis(), Vec4F32::new(0.0, 0.5, 0.0, 0.0));
        assert_eq!(inv.z_axis(), Vec4F32::new(0.0, 0.0, 0.25, 0.0));
        assert_eq!(inv.w_axis(), Vec4F32::new(0.0, 0.0, 0.0, 2.0));
    }

    #[test]
    fn test_mat4f32_determinant() {
        let m = Mat4F32::from_cols_array(&[
            2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 5.0,
        ]);
        assert_eq!(m.determinant(), 120.0);
    }

    #[test]
    fn test_mat4f32_conversions() {
        let arr = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let m: Mat4F32 = arr.into();
        assert_eq!(m.x_axis(), Vec4F32::new(1.0, 2.0, 3.0, 4.0));

        let m_glam: glam::Mat4 = m.into();
        assert_eq!(m_glam.x_axis, glam::Vec4::new(1.0, 2.0, 3.0, 4.0));

        let m_back: Mat4F32 = m_glam.into();
        assert_eq!(m_back, m);

        let arr_back: [f32; 16] = m.into();
        assert_eq!(arr_back, arr);
    }

    #[test]
    fn test_mat4f32_mul_vec4() {
        let m = Mat4F32::IDENTITY;
        let v = Vec4F32::new(1.0, 2.0, 3.0, 4.0);
        let result = m * v;
        assert_eq!(result, v);
    }

    #[test]
    fn test_mat4f64_from_cols() {
        let col1 = Vec4F64::new(1.0, 2.0, 3.0, 4.0);
        let col2 = Vec4F64::new(5.0, 6.0, 7.0, 8.0);
        let col3 = Vec4F64::new(9.0, 10.0, 11.0, 12.0);
        let col4 = Vec4F64::new(13.0, 14.0, 15.0, 16.0);
        let m = Mat4F64::from_cols(col1, col2, col3, col4);
        assert_eq!(m.x_axis(), Vec4F64::new(1.0, 2.0, 3.0, 4.0));
        assert_eq!(m.y_axis(), Vec4F64::new(5.0, 6.0, 7.0, 8.0));
        assert_eq!(m.z_axis(), Vec4F64::new(9.0, 10.0, 11.0, 12.0));
        assert_eq!(m.w_axis(), Vec4F64::new(13.0, 14.0, 15.0, 16.0));
    }

    #[test]
    fn test_mat4f64_from_cols_array() {
        let arr = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let m = Mat4F64::from_cols_array(&arr);
        assert_eq!(m.x_axis(), Vec4F64::new(1.0, 2.0, 3.0, 4.0));
        assert_eq!(m.y_axis(), Vec4F64::new(5.0, 6.0, 7.0, 8.0));
        assert_eq!(m.z_axis(), Vec4F64::new(9.0, 10.0, 11.0, 12.0));
        assert_eq!(m.w_axis(), Vec4F64::new(13.0, 14.0, 15.0, 16.0));
    }

    #[test]
    fn test_mat4f64_identity() {
        let m = Mat4F64::IDENTITY;
        assert_eq!(m.x_axis(), Vec4F64::new(1.0, 0.0, 0.0, 0.0));
        assert_eq!(m.y_axis(), Vec4F64::new(0.0, 1.0, 0.0, 0.0));
        assert_eq!(m.z_axis(), Vec4F64::new(0.0, 0.0, 1.0, 0.0));
        assert_eq!(m.w_axis(), Vec4F64::new(0.0, 0.0, 0.0, 1.0));
    }

    #[test]
    fn test_mat4f64_transpose() {
        let m = Mat4F64::from_cols_array(&[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ]);
        let mt = m.transpose();
        assert_eq!(mt.x_axis(), Vec4F64::new(1.0, 5.0, 9.0, 13.0));
        assert_eq!(mt.y_axis(), Vec4F64::new(2.0, 6.0, 10.0, 14.0));
        assert_eq!(mt.z_axis(), Vec4F64::new(3.0, 7.0, 11.0, 15.0));
        assert_eq!(mt.w_axis(), Vec4F64::new(4.0, 8.0, 12.0, 16.0));
    }

    #[test]
    fn test_mat4f64_inverse() {
        let m = Mat4F64::from_cols_array(&[
            1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.5,
        ]); // diagonal matrix
        let inv = m.inverse();
        assert_eq!(inv.x_axis(), Vec4F64::new(1.0, 0.0, 0.0, 0.0));
        assert_eq!(inv.y_axis(), Vec4F64::new(0.0, 0.5, 0.0, 0.0));
        assert_eq!(inv.z_axis(), Vec4F64::new(0.0, 0.0, 0.25, 0.0));
        assert_eq!(inv.w_axis(), Vec4F64::new(0.0, 0.0, 0.0, 2.0));
    }

    #[test]
    fn test_mat4f64_determinant() {
        let m = Mat4F64::from_cols_array(&[
            2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 5.0,
        ]);
        assert_eq!(m.determinant(), 120.0);
    }

    #[test]
    fn test_mat4f64_conversions() {
        let arr = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let m: Mat4F64 = arr.into();
        assert_eq!(m.x_axis(), Vec4F64::new(1.0, 2.0, 3.0, 4.0));

        let m_glam: glam::DMat4 = m.into();
        assert_eq!(m_glam.x_axis, glam::DVec4::new(1.0, 2.0, 3.0, 4.0));

        let m_back: Mat4F64 = m_glam.into();
        assert_eq!(m_back, m);

        let arr_back: [f64; 16] = m.into();
        assert_eq!(arr_back, arr);
    }

    #[test]
    fn test_mat4f64_mul_vec4() {
        let m = Mat4F64::IDENTITY;
        let v = Vec4F64::new(1.0, 2.0, 3.0, 4.0);
        let result = m * v;
        assert_eq!(result, v);
    }

    #[test]
    fn test_mat2f32_assign_ops() {
        let mut m = Mat2F32::IDENTITY;
        m += Mat2F32::IDENTITY;
        assert_eq!(m.x_axis(), Vec2F32::new(2.0, 0.0));

        m -= Mat2F32::IDENTITY;
        assert_eq!(m, Mat2F32::IDENTITY);

        m *= 2.0;
        assert_eq!(m.x_axis(), Vec2F32::new(2.0, 0.0));

        let mut m2 = Mat2F32::IDENTITY;
        m2 *= Mat2F32::IDENTITY;
        assert_eq!(m2, Mat2F32::IDENTITY);
    }

    #[test]
    fn test_mat3f32_from_diagonal() {
        let diag = Vec3F32::new(1.0, 2.0, 3.0);
        let m = Mat3F32::from_diagonal(diag);
        assert_eq!(m.x_axis(), Vec3F32::new(1.0, 0.0, 0.0));
        assert_eq!(m.y_axis(), Vec3F32::new(0.0, 2.0, 0.0));
        assert_eq!(m.z_axis(), Vec3F32::new(0.0, 0.0, 3.0));
    }
}
