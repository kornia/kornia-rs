// Reference: https://github.com/wi-re/tbtSVD/blob/master/source/SVD.h
use glam::{DMat3, DQuat, DVec3, Mat3, Quat, Vec3};

// Define the macro that writes the code for us
macro_rules! impl_svd_algorithm {
    (
        // Arguments passed to the macro
        $module_name:ident,      // Name of the module to put this in (e.g., f32_impl)
        $scalar:ty,              // The float type (f32 or f64)
        $mat:ty,                 // The matrix type (Mat3 or DMat3)
        $vec:ty,                 // The vector type (Vec3 or DVec3)
        $quat:ty,                // The quaternion type (Quat or DQuat)
        $gamma:expr,             // Constants...
        $cstar:expr,
        $sstar:expr,
        $epsilon:expr,
        $mask_ty:ty,             // Integer type for bit manipulation (i32 or i64)
        $uint_ty:ty,             // Unsigned int type (u32 or u64)
        $sign_mask:expr,         // Bitmask for sign
        $identity_fn:expr,       // Function to get identity matrix
        $zero_fn:expr,           // Function to get zero matrix
        $from_cols_fn:expr,      // Function to create matrix from cols
        $from_diag_fn:expr       // Function to create matrix from diagonal
    ) => {
        // We wrap the implementation in a module to avoid naming conflicts
        /// Internal implementation details.
        pub mod $module_name {
            use super::*;

            const GAMMA: $scalar = $gamma;
            const CSTAR: $scalar = $cstar;
            const SSTAR: $scalar = $sstar;
            const EPSILON: $scalar = $epsilon;
            const MAX_SWEEPS: usize = 4;

            #[derive(Debug, Clone)]
            /// A simple symmetric 3x3 Matrix class
            struct Symmetric3x3 {
                /// The element at row 0, column 0, first diagonal element.
                m_00: $scalar,
                /// The element at row 1, column 0. equivalent to `m_01`.
                m_10: $scalar,
                /// The element at row 1, column 1, the second diagonal element.
                m_11: $scalar,
                /// The element at row 2, column 0. equivalent to `m_02`.
                m_20: $scalar,
                /// The element at row 2, column 1. equivalent to `m_12`.
                m_21: $scalar,
                /// The element at row 2, column 2, the third diagonal element.
                m_22: $scalar,
            }

            impl Symmetric3x3 {
                fn from_mat(mat: &$mat) -> Self {
                    Symmetric3x3 {
                        m_00: mat.x_axis.x,
                        m_10: mat.y_axis.x,
                        m_11: mat.y_axis.y,
                        m_20: mat.x_axis.z,
                        m_21: mat.y_axis.z,
                        m_22: mat.z_axis.z,
                    }
                }
            }

            #[derive(Debug)]
            struct Givens {
                cos_theta: $scalar,
                sin_theta: $scalar,
            }

            #[derive(Debug)]
            struct QR3 {
                q: $mat,
                r: $mat,
            }

            #[derive(Debug)]
            /// Helper struct to store 3 Matrices to avoid OUT parameters on functions
            pub struct SVD3Set {
                /// The matrix of left singular vectors.
                pub u: $mat,
                /// The diagonal matrix of singular values.
                pub s: $mat,
                /// The matrix of right singular vectors.
                pub v: $mat,
            }

            impl SVD3Set {
                /// Get the left singular vectors matrix.
                #[inline]
                pub fn u(&self) -> &$mat {
                    &self.u
                }
                /// Get the diagonal matrix of singular values.
                #[inline]
                pub fn s(&self) -> &$mat {
                    &self.s
                }
                /// Get the right singular vectors matrix.
                #[inline]
                pub fn v(&self) -> &$mat {
                    &self.v
                }
            }

            #[inline(always)]
            fn approximate_givens_parameters(
                s_pp: $scalar,
                s_qq: $scalar,
                s_pq: $scalar,
            ) -> Givens {
                let cos_theta_val = 2.0 * (s_pp - s_qq);
                let sin_theta_val = s_pq;
                let cos_theta2 = cos_theta_val * cos_theta_val;
                let sin_theta2 = sin_theta_val * sin_theta_val;

                if GAMMA * sin_theta2 < cos_theta2 {
                    let w = 1.0 / ((cos_theta2 + sin_theta2).sqrt());
                    Givens {
                        cos_theta: w * cos_theta_val,
                        sin_theta: w * sin_theta_val,
                    }
                } else {
                    Givens {
                        cos_theta: CSTAR,
                        sin_theta: SSTAR,
                    }
                }
            }

            #[inline(always)]
            fn conjugate_xy(s: &mut Symmetric3x3, q: &mut $quat) {
                let mut g = approximate_givens_parameters(s.m_00, s.m_11, s.m_10);
                let cos_theta2 = g.cos_theta * g.cos_theta;
                let sin_theta2 = g.sin_theta * g.sin_theta;
                let scale = 1.0 / (cos_theta2 + sin_theta2);
                let a = (cos_theta2 - sin_theta2) * scale;
                let b = 2.0 * g.sin_theta * g.cos_theta * scale;

                let s00 = s.m_00;
                let s10 = s.m_10;
                let s11 = s.m_11;
                let s20 = s.m_20;
                let s21 = s.m_21;

                s.m_00 = a * (a * s00 + b * s10) + b * (a * s10 + b * s11);
                s.m_10 = a * (-b * s00 + a * s10) + b * (-b * s10 + a * s11);
                s.m_11 = -b * (-b * s00 + a * s10) + a * (-b * s10 + a * s11);
                s.m_20 = a * s20 + b * s21;
                s.m_21 = -b * s20 + a * s21;

                let tmp_x = q.x * g.sin_theta;
                let tmp_y = q.y * g.sin_theta;
                let tmp_z = q.z * g.sin_theta;
                g.sin_theta *= q.w;

                q.z = q.z * g.cos_theta + g.sin_theta;
                q.w = q.w * g.cos_theta - tmp_z;
                q.x = q.x * g.cos_theta + tmp_y;
                q.y = q.y * g.cos_theta - tmp_x;
            }

            #[inline(always)]
            fn conjugate_yz(s: &mut Symmetric3x3, q: &mut $quat) {
                let mut g = approximate_givens_parameters(s.m_11, s.m_22, s.m_21);
                let cos_theta2 = g.cos_theta * g.cos_theta;
                let sin_theta2 = g.sin_theta * g.sin_theta;
                let scale = 1.0 / (cos_theta2 + sin_theta2);
                let a = (cos_theta2 - sin_theta2) * scale;
                let b = 2.0 * g.sin_theta * g.cos_theta * scale;

                let s11 = s.m_11;
                let s21 = s.m_21;
                let s22 = s.m_22;
                let s10 = s.m_10;
                let s20 = s.m_20;

                s.m_11 = a * (a * s11 + b * s21) + b * (a * s21 + b * s22);
                s.m_21 = a * (-b * s11 + a * s21) + b * (-b * s21 + a * s22);
                s.m_22 = -b * (-b * s11 + a * s21) + a * (-b * s21 + a * s22);
                s.m_10 = a * s10 + b * s20;
                s.m_20 = -b * s10 + a * s20;

                let tmp_x = q.x * g.sin_theta;
                let tmp_y = q.y * g.sin_theta;
                let tmp_z = q.z * g.sin_theta;
                g.sin_theta *= q.w;

                q.x = q.x * g.cos_theta + g.sin_theta;
                q.w = q.w * g.cos_theta - tmp_x;
                q.y = q.y * g.cos_theta + tmp_z;
                q.z = q.z * g.cos_theta - tmp_y;
            }

            #[inline(always)]
            fn conjugate_xz(s: &mut Symmetric3x3, q: &mut $quat) {
                let mut g = approximate_givens_parameters(s.m_00, s.m_22, s.m_20);
                let cos_theta2 = g.cos_theta * g.cos_theta;
                let sin_theta2 = g.sin_theta * g.sin_theta;
                let scale = 1.0 / (cos_theta2 + sin_theta2);
                let a = (cos_theta2 - sin_theta2) * scale;
                let b = 2.0 * g.sin_theta * g.cos_theta * scale;

                let s00 = s.m_00;
                let s20 = s.m_20;
                let s22 = s.m_22;
                let s10 = s.m_10;
                let s21 = s.m_21;

                s.m_00 = a * (a * s00 + b * s20) + b * (a * s20 + b * s22);
                s.m_20 = a * (-b * s00 + a * s20) + b * (-b * s20 + a * s22);
                s.m_22 = -b * (-b * s00 + a * s20) + a * (-b * s20 + a * s22);
                s.m_10 = a * s10 + b * s21;
                s.m_21 = -b * s10 + a * s21;

                let tmp_x = q.x * g.sin_theta;
                let tmp_y = q.y * g.sin_theta;
                let tmp_z = q.z * g.sin_theta;
                g.sin_theta *= q.w;

                q.y = q.y * g.cos_theta + g.sin_theta;
                q.w = q.w * g.cos_theta - tmp_y;
                q.z = q.z * g.cos_theta + tmp_x;
                q.x = q.x * g.cos_theta - tmp_z;
            }

            fn jacobi_eigenanalysis(mut s: Symmetric3x3) -> $mat {
                let mut q = <$quat>::from_xyzw(0.0, 0.0, 0.0, 1.0);
                for _i in 0..MAX_SWEEPS {
                    conjugate_xy(&mut s, &mut q);
                    conjugate_yz(&mut s, &mut q);
                    conjugate_xz(&mut s, &mut q);

                    let off_diag_norm_sq = s.m_10 * s.m_10 + s.m_20 * s.m_20 + s.m_21 * s.m_21;
                    if off_diag_norm_sq < EPSILON {
                        break;
                    }
                }
                <$mat>::from_quat(q)
            }

            #[inline(always)]
            fn cond_swap(c: bool, x: &mut $scalar, y: &mut $scalar) {
                let z = *x;
                if c {
                    *x = *y;
                    *y = z;
                }
            }

            #[inline(always)]
            fn cond_swap_vec(c: bool, x: &mut $vec, y: &mut $vec) {
                let z = *x;
                if c {
                    *x = *y;
                    *y = z;
                }
            }

            #[inline(always)]
            fn cond_negate_vec(c: bool, v: &mut $vec) {
                let mask = (0 as $mask_ty).wrapping_sub(c as $mask_ty) as $uint_ty;
                let sign_mask = $sign_mask;
                let neg_mask = mask & sign_mask;
                *v = <$vec>::new(
                    <$scalar>::from_bits(v.x.to_bits() ^ neg_mask),
                    <$scalar>::from_bits(v.y.to_bits() ^ neg_mask),
                    <$scalar>::from_bits(v.z.to_bits() ^ neg_mask),
                );
            }

            #[inline(always)]
            fn sort_singular_values(b: &mut $mat, v: &mut $mat) {
                let mut rho1 = b.x_axis.length_squared();
                let mut rho2 = b.y_axis.length_squared();
                let mut rho3 = b.z_axis.length_squared();

                let c1 = rho1 < rho2;
                cond_swap(c1, &mut rho1, &mut rho2);
                cond_swap_vec(c1, &mut b.x_axis, &mut b.y_axis);
                cond_swap_vec(c1, &mut v.x_axis, &mut v.y_axis);
                cond_negate_vec(c1, &mut b.y_axis);
                cond_negate_vec(c1, &mut v.y_axis);

                let c2 = rho1 < rho3;
                cond_swap(c2, &mut rho1, &mut rho3);
                cond_swap_vec(c2, &mut b.x_axis, &mut b.z_axis);
                cond_swap_vec(c2, &mut v.x_axis, &mut v.z_axis);
                cond_negate_vec(c2, &mut b.z_axis);
                cond_negate_vec(c2, &mut v.z_axis);

                let c3 = rho2 < rho3;
                cond_swap_vec(c3, &mut b.y_axis, &mut b.z_axis);
                cond_swap_vec(c3, &mut v.y_axis, &mut v.z_axis);
                cond_negate_vec(c3, &mut b.z_axis);
                cond_negate_vec(c3, &mut v.z_axis);
            }

            #[inline(always)]
            fn qr_givens_quaternion(a1: $scalar, a2: $scalar) -> Givens {
                let rho = (a1 * a1 + a2 * a2).sqrt();
                let mut g = Givens {
                    cos_theta: a1.abs() + <$scalar>::max(rho, EPSILON),
                    sin_theta: if rho > EPSILON { a2 } else { 0.0 },
                };
                let b = a1 < 0.0;
                cond_swap(b, &mut g.sin_theta, &mut g.cos_theta);
                let w = (g.cos_theta * g.cos_theta + g.sin_theta * g.sin_theta)
                    .sqrt()
                    .recip();
                g.cos_theta *= w;
                g.sin_theta *= w;
                g
            }

            fn qr_decomposition(b_mat: &mut $mat) -> QR3 {
                let g1 = qr_givens_quaternion(b_mat.x_axis.x, b_mat.x_axis.y);
                let a1 = -2.0 * g1.sin_theta * g1.sin_theta + 1.0;
                let b1 = 2.0 * g1.cos_theta * g1.sin_theta;

                let c0 = b_mat.x_axis.x;
                let c1 = b_mat.x_axis.y;
                b_mat.x_axis.x = a1 * c0 + b1 * c1;
                b_mat.x_axis.y = -b1 * c0 + a1 * c1;
                let c0 = b_mat.y_axis.x;
                let c1 = b_mat.y_axis.y;
                b_mat.y_axis.x = a1 * c0 + b1 * c1;
                b_mat.y_axis.y = -b1 * c0 + a1 * c1;
                let c0 = b_mat.z_axis.x;
                let c1 = b_mat.z_axis.y;
                b_mat.z_axis.x = a1 * c0 + b1 * c1;
                b_mat.z_axis.y = -b1 * c0 + a1 * c1;

                let g2 = qr_givens_quaternion(b_mat.x_axis.x, b_mat.x_axis.z);
                let a2 = -2.0 * g2.sin_theta * g2.sin_theta + 1.0;
                let b2 = 2.0 * g2.cos_theta * g2.sin_theta;

                let c0 = b_mat.x_axis.x;
                let c2 = b_mat.x_axis.z;
                b_mat.x_axis.x = a2 * c0 + b2 * c2;
                b_mat.x_axis.z = -b2 * c0 + a2 * c2;
                let c0 = b_mat.y_axis.x;
                let c2 = b_mat.y_axis.z;
                b_mat.y_axis.x = a2 * c0 + b2 * c2;
                b_mat.y_axis.z = -b2 * c0 + a2 * c2;
                let c0 = b_mat.z_axis.x;
                let c2 = b_mat.z_axis.z;
                b_mat.z_axis.x = a2 * c0 + b2 * c2;
                b_mat.z_axis.z = -b2 * c0 + a2 * c2;

                let g3 = qr_givens_quaternion(b_mat.y_axis.y, b_mat.y_axis.z);
                let a3 = -2.0 * g3.sin_theta * g3.sin_theta + 1.0;
                let b3 = 2.0 * g3.cos_theta * g3.sin_theta;

                let c1 = b_mat.x_axis.y;
                let c2 = b_mat.x_axis.z;
                b_mat.x_axis.y = a3 * c1 + b3 * c2;
                b_mat.x_axis.z = -b3 * c1 + a3 * c2;
                let c1 = b_mat.y_axis.y;
                let c2 = b_mat.y_axis.z;
                b_mat.y_axis.y = a3 * c1 + b3 * c2;
                b_mat.y_axis.z = -b3 * c1 + a3 * c2;
                let c1 = b_mat.z_axis.y;
                let c2 = b_mat.z_axis.z;
                b_mat.z_axis.y = a3 * c1 + b3 * c2;
                b_mat.z_axis.z = -b3 * c1 + a3 * c2;

                let r = *b_mat;

                let q1 = <$mat>::from_cols(
                    <$vec>::new(a1, b1, 0.0),
                    <$vec>::new(-b1, a1, 0.0),
                    <$vec>::Z,
                );
                let q2 = <$mat>::from_cols(
                    <$vec>::new(a2, 0.0, b2),
                    <$vec>::Y,
                    <$vec>::new(-b2, 0.0, a2),
                );
                let q3 = <$mat>::from_cols(
                    <$vec>::X,
                    <$vec>::new(0.0, a3, b3),
                    <$vec>::new(0.0, -b3, a3),
                );

                let q = q1 * q2 * q3;
                QR3 { q, r }
            }

            /// Wrapping function used to contain all of the required sub calls
            pub fn svd(a: &$mat) -> SVD3Set {
                let mut v =
                    jacobi_eigenanalysis(Symmetric3x3::from_mat(&(a.transpose().mul_mat3(a))));
                let mut b = a.mul_mat3(&v);
                sort_singular_values(&mut b, &mut v);
                let qr = qr_decomposition(&mut b);
                let mut u = qr.q;
                let mut s = qr.r;

                let cond_x = s.x_axis.x < 0.0;
                let cond_y = s.y_axis.y < 0.0;
                let cond_z = s.z_axis.z < 0.0;

                cond_negate_vec(cond_x, &mut u.x_axis);
                cond_negate_vec(cond_y, &mut u.y_axis);
                cond_negate_vec(cond_z, &mut u.z_axis);

                s.x_axis.x = s.x_axis.x.abs();
                s.y_axis.y = s.y_axis.y.abs();
                s.z_axis.z = s.z_axis.z.abs();

                SVD3Set { u, s, v }
            }
        }
    };
}

// Generate the f32 implementation
impl_svd_algorithm!(
    impl_f32,
    f32,
    Mat3,
    Vec3,
    Quat,
    5.828_427_3,
    0.923_879_5,
    0.382_683_43,
    1e-6,
    i32,
    u32,
    0x8000_0000u32,
    Mat3::IDENTITY,
    Mat3::ZERO,
    Mat3::from_cols,
    Mat3::from_diagonal
);

// Generate the f64 implementation
impl_svd_algorithm!(
    impl_f64,
    f64,
    DMat3,
    DVec3,
    DQuat,
    5.828_427_124_746_19,
    0.923_879_532_511_286_7,
    0.382_683_432_365_089_8,
    1e-12,
    i64,
    u64,
    0x8000_0000_0000_0000u64,
    DMat3::IDENTITY,
    DMat3::ZERO,
    DMat3::from_cols,
    DMat3::from_diagonal
);

// Re-export the main types and functions to keep the API compatible
pub use impl_f32::svd as svd3;
pub use impl_f32::SVD3Set;

pub use impl_f64::svd as svd3_f64;
pub use impl_f64::SVD3Set as SVD3SetF64;

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{DMat3, DVec3, Mat3, Vec3};

    fn verify_svd_properties(a: &Mat3, svd: &SVD3Set, epsilon: f32) {
        let u = svd.u;
        let s = svd.s;
        let v = svd.v;

        let reconstruction = u * s * v.transpose();
        let reconstruction_epsilon = 1e-4;
        assert!(
            a.abs_diff_eq(reconstruction, reconstruction_epsilon),
            "Reconstruction failed: A != U*S*V.T"
        );

        let u_t_u = u.transpose() * u;
        assert!(
            Mat3::IDENTITY.abs_diff_eq(u_t_u, epsilon),
            "U is not orthogonal"
        );

        let v_t_v = v.transpose() * v;
        assert!(
            Mat3::IDENTITY.abs_diff_eq(v_t_v, epsilon),
            "V is not orthogonal"
        );
    }

    fn verify_svd_properties_f64(a: &DMat3, svd: &SVD3SetF64, epsilon: f64) {
        let u = svd.u;
        let s = svd.s;
        let v = svd.v;

        let reconstruction = u * s * v.transpose();
        let reconstruction_epsilon = 1e-10;
        assert!(
            a.abs_diff_eq(reconstruction, reconstruction_epsilon),
            "Reconstruction failed"
        );

        let u_t_u = u.transpose() * u;
        assert!(
            DMat3::IDENTITY.abs_diff_eq(u_t_u, epsilon),
            "U is not orthogonal"
        );

        let v_t_v = v.transpose() * v;
        assert!(
            DMat3::IDENTITY.abs_diff_eq(v_t_v, epsilon),
            "V is not orthogonal"
        );
    }

    #[test]
    fn test_svd3_1_diagonal_sorted() {
        let a = Mat3::from_diagonal(Vec3::new(3.0, 2.0, 1.0));
        let svd_result = svd3(&a);
        verify_svd_properties(&a, &svd_result, 1e-6);
    }

    #[test]
    fn test_svd3_diagonal_unsorted() {
        let a = Mat3::from_diagonal(Vec3::new(2.0, 3.0, 1.0));
        let svd_result = svd3(&a);
        verify_svd_properties(&a, &svd_result, 1e-6);
        let s_diag = Vec3::new(
            svd_result.s.x_axis.x,
            svd_result.s.y_axis.y,
            svd_result.s.z_axis.z,
        );
        assert!(s_diag.abs_diff_eq(Vec3::new(3.0, 2.0, 1.0), 1e-6));
    }

    #[test]
    #[cfg(not(target_arch = "aarch64"))]
    fn test_svd3_f64_diagonal_sorted() {
        let a = DMat3::from_diagonal(DVec3::new(3.0, 2.0, 1.0));
        let svd_result = svd3_f64(&a);
        verify_svd_properties_f64(&a, &svd_result, 1e-12);
    }

    #[test]
    #[cfg(not(target_arch = "aarch64"))]
    fn test_svd3_f64_general_full_rank() {
        let a = DMat3::from_cols(
            DVec3::new(1.0, 4.0, 7.0),
            DVec3::new(2.0, 5.0, 8.0),
            DVec3::new(3.0, 6.0, 10.0),
        );
        let svd_result = svd3_f64(&a);
        verify_svd_properties_f64(&a, &svd_result, 1e-12);

        let s_diag = DVec3::new(
            svd_result.s.x_axis.x,
            svd_result.s.y_axis.y,
            svd_result.s.z_axis.z,
        );
        assert!(s_diag.min_element() > 1e-12);
    }
}
