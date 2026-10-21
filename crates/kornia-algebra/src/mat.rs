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

    /// Round each element to `f32`.
    ///
    /// # Arguments
    ///
    /// * `self` - Matrix whose elements are rounded to the nearest representable `f32`.
    ///
    /// # Returns
    ///
    /// The `f32` matrix; values outside the `f32` range become `±inf`.
    ///
    /// # Errors
    ///
    /// This conversion is infallible; it preserves NaNs and may overflow to infinity.
    #[inline]
    pub fn as_mat3af32(self) -> Mat3AF32 {
        Mat3AF32(glam::Mat3A::from(self.0.as_mat3()))
    }

    /// Compute a singular value decomposition with a bounded nalgebra backend.
    ///
    /// # Arguments
    ///
    /// * `self` - Matrix to decompose, borrowed without an intermediate array copy.
    ///
    /// # Returns
    ///
    /// `U`, diagonal `S` with non-negative singular values in descending order, and `V`,
    /// satisfying `self = U * S * V.transpose()`.
    ///
    /// # Errors
    ///
    /// Returns `None` for non-finite inputs, unrepresentable singular values, or failure
    /// to converge within the backend's iteration limit.
    ///
    /// # Example
    ///
    /// ```rust
    /// use kornia_algebra::{Mat3F64, Vec3F64};
    /// let matrix = Mat3F64::from_diagonal(Vec3F64::new(3.0, 2.0, 1.0));
    /// let svd = matrix.try_svd().ok_or("SVD failed")?;
    /// assert_eq!(*svd.u() * *svd.s() * svd.v().transpose(), matrix);
    /// # Ok::<(), &str>(())
    /// ```
    pub fn try_svd(&self) -> Option<crate::linalg::svd::SVD3SetF64> {
        crate::linalg::svd::try_svd3_f64(self)
    }

    /// Project the matrix onto the nearest rotation in the Frobenius norm.
    ///
    /// For the SVD `M = U·Σ·Vᵀ` this is `U·diag(1, 1, det(U·Vᵀ))·Vᵀ`. When `U·Vᵀ` is a
    /// reflection, negating a column of `U·Vᵀ` instead gives a rotation that is in general not the
    /// nearest.
    ///
    /// # Arguments
    ///
    /// * `self` - Matrix to project onto SO(3).
    ///
    /// # Returns
    ///
    /// The nearest rotation, or `None` if the matrix has a non-finite entry or its SVD does not
    /// converge. The zero matrix, to which every rotation is equally near, gives the identity;
    /// below rank 2 the answer is not unique either.
    ///
    /// # Errors
    ///
    /// Returns `None` for non-finite input or if the SVD fails to converge.
    ///
    /// # Example
    ///
    /// ```rust
    /// use kornia_algebra::Mat3F64;
    ///
    /// // A rotation with a little noise on one entry.
    /// let noisy = Mat3F64::from_cols_array(&[1.0, 0.0, 0.0, 0.01, 1.0, 0.0, 0.0, 0.0, 1.0]);
    /// let r = noisy.nearest_rotation().ok_or("no rotation")?;
    /// assert!((r.determinant() - 1.0).abs() < 1e-12);
    /// # Ok::<(), &str>(())
    /// ```
    pub fn nearest_rotation(self) -> Option<Self> {
        crate::linalg::svd::nearest_rotation_f64(&self)
    }
}

// Conversions between `Mat3F64` and nalgebra's `Matrix3<f64>`, through the `[[f64; 3]; 3]` array of
// columns both store.
impl From<nalgebra::Matrix3<f64>> for Mat3F64 {
    #[inline]
    fn from(m: nalgebra::Matrix3<f64>) -> Self {
        Self(glam::DMat3::from_cols_array_2d(&m.into()))
    }
}

impl From<Mat3F64> for nalgebra::Matrix3<f64> {
    #[inline]
    fn from(m: Mat3F64) -> Self {
        m.0.to_cols_array_2d().into()
    }
}

// Both layouts are contiguous column-major f64 values. The view borrows glam's array;
// decompositions still require their own writable workspace.
impl<'a> From<&'a Mat3F64> for nalgebra::MatrixView3<'a, f64> {
    #[inline]
    fn from(matrix: &'a Mat3F64) -> Self {
        Self::from_slice(matrix.0.as_ref())
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

impl From<Mat3AF32> for Mat3F64 {
    #[inline]
    fn from(m: Mat3AF32) -> Self {
        Self(m.0.as_dmat3())
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SO3F64;

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

    /// Every element must keep its row and column through both conversions.
    #[test]
    fn test_mat3f64_nalgebra_conversions() {
        // Not symmetric, so a transposed copy changes the matrix.
        let na = nalgebra::Matrix3::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0);
        let m = Mat3F64::from(na);
        for r in 0..3 {
            for c in 0..3 {
                assert_eq!(m.col(c)[r], na[(r, c)], "row {r}, column {c}");
            }
        }
        assert_eq!(nalgebra::Matrix3::<f64>::from(m), na);
    }

    #[test]
    fn test_mat3f64_nalgebra_view_borrows_column_storage() {
        let matrix = Mat3F64::from_cols_array(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]);
        let view = nalgebra::MatrixView3::from(&matrix);
        let storage: &[f64; 9] = matrix.0.as_ref();
        assert_eq!(view.as_ptr(), storage.as_ptr());
        assert_eq!(view[(0, 1)], 4.0);
        assert_eq!(view[(1, 0)], 2.0);
    }

    #[test]
    fn test_mat3f64_try_svd_reconstructs_scaled_and_rank_deficient_matrices(
    ) -> Result<(), &'static str> {
        let left = SO3F64::exp(Vec3F64::new(0.3, -0.5, 0.8)).matrix();
        let right = SO3F64::exp(Vec3F64::new(-0.6, 0.2, 0.4)).matrix();
        for diagonal in [Vec3F64::new(1.0, 3.0, -2.0), Vec3F64::new(0.0, 2.0, 1.0)] {
            let matrix = left * Mat3F64::from_diagonal(diagonal) * right;
            for scale in [1.0, 1e300, 1e-300] {
                let svd = (matrix * scale).try_svd().ok_or("SVD failed")?;
                let normalized_s = Mat3F64(svd.s().0 / scale);
                let reconstructed = *svd.u() * normalized_s * svd.v().transpose();
                assert!(reconstructed.abs_diff_eq(matrix.0, 1e-12));
                assert!((svd.u().transpose() * *svd.u()).abs_diff_eq(Mat3F64::IDENTITY.0, 1e-12));
                assert!((svd.v().transpose() * *svd.v()).abs_diff_eq(Mat3F64::IDENTITY.0, 1e-12));
                assert!(normalized_s.x_axis.x >= normalized_s.y_axis.y);
                assert!(normalized_s.y_axis.y >= normalized_s.z_axis.z);
                assert!(normalized_s.z_axis.z >= 0.0);
            }
        }
        assert_eq!(
            *Mat3F64::ZERO.try_svd().ok_or("zero SVD failed")?.s(),
            Mat3F64::ZERO
        );
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(Mat3F64::from_cols_array(&[value; 9]).try_svd().is_none());
        }
        assert!(Mat3F64::from_cols_array(&[f64::MAX; 9]).try_svd().is_none());
        Ok(())
    }

    #[test]
    fn test_mat3f64_from_mat3af32() {
        // Not symmetric; 0.1 is not exact in binary, so the f64 must be the f32 value itself.
        let m = Mat3AF32::from_cols_array(&[0.1, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]);
        let wide = Mat3F64::from(m);
        assert_eq!(wide.to_cols_array(), m.to_cols_array().map(f64::from));
        assert_eq!(wide.as_mat3af32(), m);
    }

    #[test]
    fn test_mat3f64_nearest_rotation_keeps_a_rotation() -> Result<(), &'static str> {
        let rotation = SO3F64::exp(Vec3F64::new(0.3, -0.5, 0.8)).matrix();
        let nearest = rotation.nearest_rotation().ok_or("no rotation")?;
        assert!(nearest.abs_diff_eq(rotation.0, 1e-12));
        assert!(Mat3F64::from_cols_array(&[f64::NAN; 9])
            .nearest_rotation()
            .is_none());
        Ok(())
    }

    /// `M = A·diag(3, 2, −1)·B` (A, B rotations) has `U·Vᵀ = A·diag(1, 1, −1)·B`, a reflection.
    /// Its nearest rotation is `A·B`; negating the third column of `U·Vᵀ` gives a rotation that
    /// is farther from `M`.
    #[test]
    fn test_mat3f64_nearest_rotation_of_a_reflection() -> Result<(), &'static str> {
        let a = SO3F64::exp(Vec3F64::new(0.3, -0.5, 0.8)).matrix();
        let b = SO3F64::exp(Vec3F64::new(-0.6, 0.2, 0.4)).matrix();
        let diag = |x, y, z| Mat3F64::from_diagonal(Vec3F64::new(x, y, z));
        let m = a * diag(3.0, 2.0, -1.0) * b;
        let nearest = m.nearest_rotation().ok_or("no rotation")?;
        assert!(nearest.abs_diff_eq((a * b).0, 1e-9));

        let mut negated = a * diag(1.0, 1.0, -1.0) * b;
        negated.z_axis = -negated.z_axis;
        assert!((negated.determinant() - 1.0).abs() < 1e-9);
        let distance_to_m = |x: Mat3F64| (x - m).to_cols_array().map(|v| v * v).iter().sum::<f64>();
        assert!(distance_to_m(a * b) < distance_to_m(negated) - 0.1);
        Ok(())
    }

    /// Scaling does not change the nearest rotation, also far from 1, where `MᵀM` would overflow
    /// or underflow without the internal rescaling; the zero matrix gives the identity.
    #[test]
    fn test_mat3f64_nearest_rotation_is_scale_invariant() -> Result<(), &'static str> {
        let base = Mat3F64::from_cols_array(&[0.1, 1.0, 0.0, 1.0, 0.2, 0.0, 0.0, 0.0, -0.5]);
        let expected = base.nearest_rotation().ok_or("no rotation")?;
        for s in [1e155, 1e-170, 1e300, 1e-300] {
            let got = (base * s).nearest_rotation().ok_or("no rotation")?;
            assert!(got.abs_diff_eq(expected.0, 1e-12), "scale {s:e}");
        }
        assert_eq!(Mat3F64::ZERO.nearest_rotation(), Some(Mat3F64::IDENTITY));
        Ok(())
    }

    #[test]
    fn test_nearest_rotation_does_not_require_representable_singular_values(
    ) -> Result<(), &'static str> {
        // This rank-one matrix has a singular value of 3 * f64::MAX. Projection depends
        // only on singular vectors and must remain defined even though a full SVD cannot.
        let matrix = Mat3F64::from_cols_array(&[f64::MAX; 9]);
        assert!(matrix.try_svd().is_none());
        let rotation = matrix.nearest_rotation().ok_or("projection failed")?;
        assert!((rotation.transpose() * rotation).abs_diff_eq(Mat3F64::IDENTITY.0, 1e-12));
        assert!((rotation.determinant() - 1.0).abs() < 1e-12);
        Ok(())
    }

    /// `R` is the rotation nearest to `M` only if `Rᵀ·M` is symmetric (polar decomposition).
    #[test]
    fn test_mat3f64_nearest_rotation_is_exact() -> Result<(), &'static str> {
        // `svd3_f64`'s projection leaves Rᵀ·M asymmetric by about 2e-5 on this matrix.
        let m = Mat3F64::from_cols_array(&[-0.2, 0.6, -0.6, -0.3, 0.6, -0.3, 0.0, 0.9, -0.8]);
        let r = m.nearest_rotation().ok_or("no rotation")?;
        let p = r.transpose() * m;
        assert!(p.abs_diff_eq(p.transpose().0, 1e-12));
        Ok(())
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
