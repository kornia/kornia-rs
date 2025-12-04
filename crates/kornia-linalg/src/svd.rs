// Reference: https://github.com/wi-re/tbtSVD/blob/master/source/SVD.h
use kornia_algebra::{Mat3F32, Mat3F64};

// =================================================================================
//  f32 Implementation
// =================================================================================

mod impl_f32 {
    use glam::{Mat3, Quat, Vec3};

    const GAMMA: f32 = 5.828_427_3;
    const CSTAR: f32 = 0.923_879_5;
    const SSTAR: f32 = 0.382_683_43;
    const EPSILON: f32 = 1e-6;
    const MAX_SWEEPS: usize = 4;

    #[derive(Debug, Clone)]
    /// A simple symmetric 3x3 Matrix class
    struct Symmetric3x3 {
        /// The element at row 0, column 0.
        m_00: f32,
        /// The element at row 1, column 0.
        m_10: f32,
        /// The element at row 1, column 1.
        m_11: f32,
        /// The element at row 2, column 0.
        m_20: f32,
        /// The element at row 2, column 1.
        m_21: f32,
        /// The element at row 2, column 2.
        m_22: f32,
    }

    impl Symmetric3x3 {
        fn from_mat(mat: &Mat3) -> Self {
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
        cos_theta: f32,
        sin_theta: f32,
    }

    #[derive(Debug)]
    struct QR3 {
        q: Mat3,
        r: Mat3,
    }

    #[derive(Debug)]
    /// Helper struct to store 3 Matrices to avoid OUT parameters on functions
    pub struct SVD3Set {
        /// The matrix of left singular vectors.
        pub u: Mat3,
        /// The diagonal matrix of singular values.
        pub s: Mat3,
        /// The matrix of right singular vectors.
        pub v: Mat3,
    }

    #[allow(dead_code)]
    impl SVD3Set {
        /// Get the left singular vectors matrix.
        #[inline]
        pub fn u(&self) -> &Mat3 {
            &self.u
        }
        /// Get the diagonal matrix of singular values.
        #[inline]
        pub fn s(&self) -> &Mat3 {
            &self.s
        }
        /// Get the right singular vectors matrix.
        #[inline]
        pub fn v(&self) -> &Mat3 {
            &self.v
        }
    }

    #[inline(always)]
    fn approximate_givens_parameters(s_pp: f32, s_qq: f32, s_pq: f32) -> Givens {
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
    fn conjugate_xy(s: &mut Symmetric3x3, q: &mut Quat) {
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
    fn conjugate_yz(s: &mut Symmetric3x3, q: &mut Quat) {
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
    fn conjugate_xz(s: &mut Symmetric3x3, q: &mut Quat) {
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

    fn jacobi_eigenanalysis(mut s: Symmetric3x3) -> Mat3 {
        let mut q = Quat::from_xyzw(0.0, 0.0, 0.0, 1.0);
        for _i in 0..MAX_SWEEPS {
            conjugate_xy(&mut s, &mut q);
            conjugate_yz(&mut s, &mut q);
            conjugate_xz(&mut s, &mut q);

            let off_diag_norm_sq = s.m_10 * s.m_10 + s.m_20 * s.m_20 + s.m_21 * s.m_21;
            if off_diag_norm_sq < EPSILON {
                break;
            }
        }
        Mat3::from_quat(q)
    }

    #[inline(always)]
    fn cond_swap(c: bool, x: &mut f32, y: &mut f32) {
        let z = *x;
        if c {
            *x = *y;
            *y = z;
        }
    }

    #[inline(always)]
    fn cond_swap_vec(c: bool, x: &mut Vec3, y: &mut Vec3) {
        let z = *x;
        if c {
            *x = *y;
            *y = z;
        }
    }

    #[inline(always)]
    fn cond_negate_vec(c: bool, v: &mut Vec3) {
        let mask = 0_i32.wrapping_sub(c as i32) as u32;
        let sign_mask = 0x8000_0000u32;
        let neg_mask = mask & sign_mask;
        *v = Vec3::new(
            f32::from_bits(v.x.to_bits() ^ neg_mask),
            f32::from_bits(v.y.to_bits() ^ neg_mask),
            f32::from_bits(v.z.to_bits() ^ neg_mask),
        );
    }

    #[inline(always)]
    fn sort_singular_values(b: &mut Mat3, v: &mut Mat3) {
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
    fn qr_givens_quaternion(a1: f32, a2: f32) -> Givens {
        let rho = (a1 * a1 + a2 * a2).sqrt();
        let mut g = Givens {
            cos_theta: a1.abs() + f32::max(rho, EPSILON),
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

    fn qr_decomposition(b_mat: &mut Mat3) -> QR3 {
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

        let q1 = Mat3::from_cols(Vec3::new(a1, b1, 0.0), Vec3::new(-b1, a1, 0.0), Vec3::Z);
        let q2 = Mat3::from_cols(Vec3::new(a2, 0.0, b2), Vec3::Y, Vec3::new(-b2, 0.0, a2));
        let q3 = Mat3::from_cols(Vec3::X, Vec3::new(0.0, a3, b3), Vec3::new(0.0, -b3, a3));

        let q = q1 * q2 * q3;
        QR3 { q, r }
    }

    /// Internal SVD implementation for f32 (glam types)
    pub fn svd_inner(a: &Mat3) -> SVD3Set {
        let mut v = jacobi_eigenanalysis(Symmetric3x3::from_mat(&(a.transpose() * (*a))));
        let mut b = *a * v;
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

// =================================================================================
//  f64 Implementation
// =================================================================================

mod impl_f64 {
    use glam::{DMat3, DQuat, DVec3};

    const GAMMA: f64 = 5.828_427_124_746_19; // 3 + 2 * sqrt(2)
    const CSTAR: f64 = 0.923_879_532_511_286_7; // cos(pi/8)
    const SSTAR: f64 = 0.382_683_432_365_089_8; // sin(pi/8)
    const EPSILON: f64 = 1e-12;
    const MAX_SWEEPS: usize = 4;

    #[derive(Debug, Clone)]
    /// A simple symmetric 3x3 Matrix class (f64)
    struct Symmetric3x3 {
        /// The element at row 0, column 0.
        m_00: f64,
        /// The element at row 1, column 0.
        m_10: f64,
        /// The element at row 1, column 1.
        m_11: f64,
        /// The element at row 2, column 0.
        m_20: f64,
        /// The element at row 2, column 1.
        m_21: f64,
        /// The element at row 2, column 2.
        m_22: f64,
    }

    impl Symmetric3x3 {
        fn from_mat(mat: &DMat3) -> Self {
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
        cos_theta: f64,
        sin_theta: f64,
    }

    #[derive(Debug)]
    struct QR3 {
        q: DMat3,
        r: DMat3,
    }

    #[derive(Debug)]
    /// Helper struct to store 3 Matrices to avoid OUT parameters on functions
    pub struct SVD3Set {
        /// The matrix of left singular vectors.
        pub u: DMat3,
        /// The diagonal matrix of singular values.
        pub s: DMat3,
        /// The matrix of right singular vectors.
        pub v: DMat3,
    }

    #[allow(dead_code)]
    impl SVD3Set {
        /// Get the left singular vectors matrix.
        #[inline]
        pub fn u(&self) -> &DMat3 {
            &self.u
        }
        /// Get the diagonal matrix of singular values.
        #[inline]
        pub fn s(&self) -> &DMat3 {
            &self.s
        }
        /// Get the right singular vectors matrix.
        #[inline]
        pub fn v(&self) -> &DMat3 {
            &self.v
        }
    }

    #[inline(always)]
    fn approximate_givens_parameters(s_pp: f64, s_qq: f64, s_pq: f64) -> Givens {
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
    fn conjugate_xy(s: &mut Symmetric3x3, q: &mut DQuat) {
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
    fn conjugate_yz(s: &mut Symmetric3x3, q: &mut DQuat) {
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
    fn conjugate_xz(s: &mut Symmetric3x3, q: &mut DQuat) {
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

    fn jacobi_eigenanalysis(mut s: Symmetric3x3) -> DMat3 {
        let mut q = DQuat::from_xyzw(0.0, 0.0, 0.0, 1.0);
        for _i in 0..MAX_SWEEPS {
            conjugate_xy(&mut s, &mut q);
            conjugate_yz(&mut s, &mut q);
            conjugate_xz(&mut s, &mut q);

            let off_diag_norm_sq = s.m_10 * s.m_10 + s.m_20 * s.m_20 + s.m_21 * s.m_21;
            if off_diag_norm_sq < EPSILON {
                break;
            }
        }
        DMat3::from_quat(q)
    }

    #[inline(always)]
    fn cond_swap(c: bool, x: &mut f64, y: &mut f64) {
        let z = *x;
        if c {
            *x = *y;
            *y = z;
        }
    }

    #[inline(always)]
    fn cond_swap_vec(c: bool, x: &mut DVec3, y: &mut DVec3) {
        let z = *x;
        if c {
            *x = *y;
            *y = z;
        }
    }

    #[inline(always)]
    fn cond_negate_vec(c: bool, v: &mut DVec3) {
        // FIX: Use 0_i64 to avoid clippy::unnecessary_cast
        let mask = 0_i64.wrapping_sub(c as i64) as u64;
        let sign_mask = 0x8000_0000_0000_0000u64;
        let neg_mask = mask & sign_mask;
        *v = DVec3::new(
            f64::from_bits(v.x.to_bits() ^ neg_mask),
            f64::from_bits(v.y.to_bits() ^ neg_mask),
            f64::from_bits(v.z.to_bits() ^ neg_mask),
        );
    }

    #[inline(always)]
    fn sort_singular_values(b: &mut DMat3, v: &mut DMat3) {
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
    fn qr_givens_quaternion(a1: f64, a2: f64) -> Givens {
        let rho = (a1 * a1 + a2 * a2).sqrt();
        let mut g = Givens {
            cos_theta: a1.abs() + f64::max(rho, EPSILON),
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

    fn qr_decomposition(b_mat: &mut DMat3) -> QR3 {
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

        let q1 = DMat3::from_cols(DVec3::new(a1, b1, 0.0), DVec3::new(-b1, a1, 0.0), DVec3::Z);
        let q2 = DMat3::from_cols(DVec3::new(a2, 0.0, b2), DVec3::Y, DVec3::new(-b2, 0.0, a2));
        let q3 = DMat3::from_cols(DVec3::X, DVec3::new(0.0, a3, b3), DVec3::new(0.0, -b3, a3));

        let q = q1 * q2 * q3;
        QR3 { q, r }
    }

    /// Internal SVD implementation for f64 (glam types)
    pub fn svd_inner(a: &DMat3) -> SVD3Set {
        let mut v = jacobi_eigenanalysis(Symmetric3x3::from_mat(&(a.transpose() * (*a))));
        let mut b = *a * v;
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

// =================================================================================
//  Public Wrappers
// =================================================================================

/// Helper struct to store 3 Matrices to avoid OUT parameters on functions (f32)
#[derive(Debug)]
pub struct SVD3Set {
    /// The matrix of left singular vectors.
    pub u: Mat3F32,
    /// The diagonal matrix of singular values.
    pub s: Mat3F32,
    /// The matrix of right singular vectors.
    pub v: Mat3F32,
}

#[allow(dead_code)]
impl SVD3Set {
    /// Get the left singular vectors matrix.
    #[inline]
    pub fn u(&self) -> &Mat3F32 {
        &self.u
    }
    /// Get the diagonal matrix of singular values.
    #[inline]
    pub fn s(&self) -> &Mat3F32 {
        &self.s
    }
    /// Get the right singular vectors matrix.
    #[inline]
    pub fn v(&self) -> &Mat3F32 {
        &self.v
    }
}

/// Helper struct to store 3 Matrices to avoid OUT parameters on functions (f64)
#[derive(Debug)]
pub struct SVD3SetF64 {
    /// The matrix of left singular vectors.
    pub u: Mat3F64,
    /// The diagonal matrix of singular values.
    pub s: Mat3F64,
    /// The matrix of right singular vectors.
    pub v: Mat3F64,
}

#[allow(dead_code)] // These are part of the public API, used by future consumers
impl SVD3SetF64 {
    /// Get the left singular vectors matrix.
    #[inline]
    pub fn u(&self) -> &Mat3F64 {
        &self.u
    }
    /// Get the diagonal matrix of singular values.
    #[inline]
    pub fn s(&self) -> &Mat3F64 {
        &self.s
    }
    /// Get the right singular vectors matrix.
    #[inline]
    pub fn v(&self) -> &Mat3F64 {
        &self.v
    }
}

/// SVD for f32 matrices.
///
/// Wraps internal glam implementation.
pub fn svd3(a: &Mat3F32) -> SVD3Set {
    let a_inner: glam::Mat3 = (*a).into();
    let res = impl_f32::svd_inner(&a_inner);
    SVD3Set {
        u: res.u.into(),
        s: res.s.into(),
        v: res.v.into(),
    }
}

/// SVD for f64 matrices.
///
/// Wraps internal glam implementation.
pub fn svd3_f64(a: &Mat3F64) -> SVD3SetF64 {
    let a_inner: glam::DMat3 = (*a).into();
    let res = impl_f64::svd_inner(&a_inner);
    SVD3SetF64 {
        u: res.u.into(),
        s: res.s.into(),
        v: res.v.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_algebra::{Mat3F32, Mat3F64, Vec3F32, Vec3F64};

    fn verify_svd_properties(a: &Mat3F32, svd: &SVD3Set, epsilon: f32) {
        let u: glam::Mat3 = svd.u.into();
        let s: glam::Mat3 = svd.s.into();
        let v: glam::Mat3 = svd.v.into();
        let a_glam: glam::Mat3 = (*a).into();

        let reconstruction = u * s * v.transpose();
        let reconstruction_epsilon = 1e-4;

        let diff = a_glam - reconstruction;
        let max_diff = diff
            .x_axis
            .abs()
            .max_element()
            .max(diff.y_axis.abs().max_element())
            .max(diff.z_axis.abs().max_element());

        assert!(
            max_diff < reconstruction_epsilon,
            "Reconstruction failed: A != U*S*V.T"
        );

        let u_t_u = u.transpose() * u;
        let diff_u = glam::Mat3::IDENTITY - u_t_u;
        let max_diff_u = diff_u
            .x_axis
            .abs()
            .max_element()
            .max(diff_u.y_axis.abs().max_element())
            .max(diff_u.z_axis.abs().max_element());

        assert!(max_diff_u < epsilon, "U is not orthogonal");

        let v_t_v = v.transpose() * v;
        let diff_v = glam::Mat3::IDENTITY - v_t_v;
        let max_diff_v = diff_v
            .x_axis
            .abs()
            .max_element()
            .max(diff_v.y_axis.abs().max_element())
            .max(diff_v.z_axis.abs().max_element());

        assert!(max_diff_v < epsilon, "V is not orthogonal");
    }

    fn verify_svd_properties_f64(a: &Mat3F64, svd: &SVD3SetF64, epsilon: f64) {
        let u: glam::DMat3 = svd.u.into();
        let s: glam::DMat3 = svd.s.into();
        let v: glam::DMat3 = svd.v.into();
        let a_glam: glam::DMat3 = (*a).into();

        let reconstruction = u * s * v.transpose();
        let reconstruction_epsilon = 1e-10;

        let diff = a_glam - reconstruction;
        let max_diff = diff
            .x_axis
            .abs()
            .max_element()
            .max(diff.y_axis.abs().max_element())
            .max(diff.z_axis.abs().max_element());

        assert!(max_diff < reconstruction_epsilon, "Reconstruction failed");

        let u_t_u = u.transpose() * u;
        let diff_u = glam::DMat3::IDENTITY - u_t_u;
        let max_diff_u = diff_u
            .x_axis
            .abs()
            .max_element()
            .max(diff_u.y_axis.abs().max_element())
            .max(diff_u.z_axis.abs().max_element());

        assert!(max_diff_u < epsilon, "U is not orthogonal");

        let v_t_v = v.transpose() * v;
        let diff_v = glam::DMat3::IDENTITY - v_t_v;
        let max_diff_v = diff_v
            .x_axis
            .abs()
            .max_element()
            .max(diff_v.y_axis.abs().max_element())
            .max(diff_v.z_axis.abs().max_element());

        assert!(max_diff_v < epsilon, "V is not orthogonal");
    }

    #[test]
    fn test_svd3_1_diagonal_sorted() {
        let a = Mat3F32::from_diagonal(Vec3F32::new(3.0, 2.0, 1.0));
        let svd_result = svd3(&a);
        verify_svd_properties(&a, &svd_result, 1e-6);
    }

    #[test]
    fn test_svd3_diagonal_unsorted() {
        let a = Mat3F32::from_diagonal(Vec3F32::new(2.0, 3.0, 1.0));
        let svd_result = svd3(&a);
        verify_svd_properties(&a, &svd_result, 1e-6);

        // Convert to glam to access components easily
        let s_diag: glam::Mat3 = svd_result.s.into();
        let s_vec = glam::Vec3::new(s_diag.x_axis.x, s_diag.y_axis.y, s_diag.z_axis.z);
        assert!((s_vec - glam::Vec3::new(3.0, 2.0, 1.0)).length() < 1e-6);
    }

    #[test]
    #[cfg(not(target_arch = "aarch64"))]
    fn test_svd3_f64_diagonal_sorted() {
        let a = Mat3F64::from_diagonal(Vec3F64::new(3.0, 2.0, 1.0));
        let svd_result = svd3_f64(&a);
        verify_svd_properties_f64(&a, &svd_result, 1e-12);
    }

    #[test]
    #[cfg(not(target_arch = "aarch64"))]
    fn test_svd3_f64_general_full_rank() {
        let a = Mat3F64::from_cols(
            Vec3F64::new(1.0, 4.0, 7.0),
            Vec3F64::new(2.0, 5.0, 8.0),
            Vec3F64::new(3.0, 6.0, 10.0),
        );
        let svd_result = svd3_f64(&a);
        verify_svd_properties_f64(&a, &svd_result, 1e-12);

        let s_diag: glam::DMat3 = svd_result.s.into();
        let s_vec = glam::DVec3::new(s_diag.x_axis.x, s_diag.y_axis.y, s_diag.z_axis.z);
        assert!(s_vec.min_element() > 1e-12);
    }
}
