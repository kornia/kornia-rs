// Reference: https://github.com/wi-re/tbtSVD/blob/master/source/SVD.h
use crate::{Mat3AF32, Mat3F32, Mat3F64, QuatF32, QuatF64, Vec3F32, Vec3F64};

// Re-export the SVD result types for public use
pub use impl_f32::{sort_singular_values, svd3 as svd3_f32, SVD3Set as SVD3SetF32};
pub use impl_f64::{
    sort_singular_values as sort_singular_values_f64, svd3 as svd3_f64, SVD3Set as SVD3SetF64,
};

mod impl_f32 {
    use super::*;

    const GAMMA: f32 = 5.828_427_3;
    const CSTAR: f32 = 0.923_879_5;
    const SSTAR: f32 = 0.382_683_43;
    const SVD3_EPSILON: f32 = 1e-6;
    const JACOBI_TOLERANCE: f32 = 1e-6;
    const MAX_SWEEPS: usize = 4;

    #[derive(Debug, Clone)]
    /// A simple symmetric 3x3 Matrix class (contains no storage for (0, 1) (0, 2) and (1, 2)
    struct Symmetric3x3 {
        /// The element at row 0, column 0, first diagonal element.
        m_00: f32,
        /// The element at row 1, column 0. equivalent to `m_01`.
        m_10: f32,
        /// The element at row 1, column 1, the second diagonal element.
        m_11: f32,
        /// The element at row 2, column 0. equivalent to `m_02`.
        m_20: f32,
        /// The element at row 2, column 1. equivalent to `m_12`.
        m_21: f32,
        /// The element at row 2, column 2, the third diagonal element.
        m_22: f32,
    }

    impl Symmetric3x3 {
        /// Constructor from a regular Mat3x3 (assuming Mat3x3 exists)
        fn from_mat3x3(mat: &Mat3F32) -> Self {
            let x_axis = mat.x_axis();
            let y_axis = mat.y_axis();
            let z_axis = mat.z_axis();
            Symmetric3x3 {
                m_00: x_axis.x,
                m_10: y_axis.x,
                m_11: y_axis.y,
                m_20: x_axis.z,
                m_21: y_axis.z,
                m_22: z_axis.z,
            }
        }
    }

    #[derive(Debug)]
    /// Helper struct to store 2 floats to avoid OUT parameters on functions
    struct Givens {
        /// The cosine of the angle in the Givens rotation.
        cos_theta: f32,
        /// The sine of the angle in the Givens rotation.
        sin_theta: f32,
    }

    #[derive(Debug)]
    /// Helper struct to store 2 Matrices to avoid OUT parameters on functions
    struct QR3 {
        /// The orthogonal matrix Q from the QR decomposition.
        q: Mat3F32,
        /// The upper triangular matrix R from the QR decomposition.
        r: Mat3F32,
    }

    #[derive(Debug)]
    /// Helper struct to store 3 Matrices to avoid OUT parameters on functions
    pub struct SVD3Set {
        /// The matrix of left singular vectors.
        u: Mat3F32,
        /// The diagonal matrix of singular values.
        s: Mat3F32,
        /// The matrix of right singular vectors.
        v: Mat3F32,
    }

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

    /// For an explanation of the math see http://pages.cs.wisc.edu/~sifakis/papers/SVD_TR1690.pdf
    /// Computing the Singular Value Decomposition of 3 x 3 matrices with minimal branching and elementary floating point operations
    /// See Algorithm 2 in reference. Given a matrix A this function returns the givens quaternion (x and w component, y and z are 0)
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
    fn conjugate_xy(s: &mut Symmetric3x3, q: &mut QuatF32) {
        // Compute Givens rotation parameters
        let mut g = approximate_givens_parameters(s.m_00, s.m_11, s.m_10);
        let cos_theta2 = g.cos_theta * g.cos_theta;
        let sin_theta2 = g.sin_theta * g.sin_theta;
        let scale = 1.0 / (cos_theta2 + sin_theta2);
        let a = (cos_theta2 - sin_theta2) * scale;
        let b = 2.0 * g.sin_theta * g.cos_theta * scale;

        // cache original matrix elements
        let s00 = s.m_00;
        let s10 = s.m_10;
        let s11 = s.m_11;
        let s20 = s.m_20;
        let s21 = s.m_21;

        // Perform the matrix conjugation
        s.m_00 = a * (a * s00 + b * s10) + b * (a * s10 + b * s11);
        s.m_10 = a * (-b * s00 + a * s10) + b * (-b * s10 + a * s11);
        s.m_11 = -b * (-b * s00 + a * s10) + a * (-b * s10 + a * s11);
        s.m_20 = a * s20 + b * s21;
        s.m_21 = -b * s20 + a * s21;

        // Update the cumulative rotation quaternion using named fields
        let q_arr = q.to_array();
        let tmp_x = q_arr[0] * g.sin_theta;
        let tmp_y = q_arr[1] * g.sin_theta;
        let tmp_z = q_arr[2] * g.sin_theta;
        g.sin_theta *= q_arr[3];

        *q = QuatF32::from_xyzw(
            q_arr[0] * g.cos_theta + tmp_y,
            q_arr[1] * g.cos_theta - tmp_x,
            q_arr[2] * g.cos_theta + g.sin_theta,
            q_arr[3] * g.cos_theta - tmp_z,
        );
    }

    #[inline(always)]
    fn conjugate_yz(s: &mut Symmetric3x3, q: &mut QuatF32) {
        // Compute Givens rotation parameters
        let mut g = approximate_givens_parameters(s.m_11, s.m_22, s.m_21);
        let cos_theta2 = g.cos_theta * g.cos_theta;
        let sin_theta2 = g.sin_theta * g.sin_theta;
        let scale = 1.0 / (cos_theta2 + sin_theta2);
        let a = (cos_theta2 - sin_theta2) * scale;
        let b = 2.0 * g.sin_theta * g.cos_theta * scale;

        // cache original matrix elements
        let s11 = s.m_11;
        let s21 = s.m_21;
        let s22 = s.m_22;
        let s10 = s.m_10;
        let s20 = s.m_20;

        // Perform the matrix conjugation
        s.m_11 = a * (a * s11 + b * s21) + b * (a * s21 + b * s22);
        s.m_21 = a * (-b * s11 + a * s21) + b * (-b * s21 + a * s22);
        s.m_22 = -b * (-b * s11 + a * s21) + a * (-b * s21 + a * s22);
        s.m_10 = a * s10 + b * s20;
        s.m_20 = -b * s10 + a * s20;

        // Update the cumulative rotation quaternion using named fields
        let q_arr = q.to_array();
        let tmp_x = q_arr[0] * g.sin_theta;
        let tmp_y = q_arr[1] * g.sin_theta;
        let tmp_z = q_arr[2] * g.sin_theta;
        g.sin_theta *= q_arr[3];

        *q = QuatF32::from_xyzw(
            q_arr[0] * g.cos_theta + g.sin_theta,
            q_arr[1] * g.cos_theta + tmp_z,
            q_arr[2] * g.cos_theta - tmp_y,
            q_arr[3] * g.cos_theta - tmp_x,
        );
    }

    #[inline(always)]
    fn conjugate_xz(s: &mut Symmetric3x3, q: &mut QuatF32) {
        // Compute Givens rotation parameters
        let mut g = approximate_givens_parameters(s.m_00, s.m_22, s.m_20);
        let cos_theta2 = g.cos_theta * g.cos_theta;
        let sin_theta2 = g.sin_theta * g.sin_theta;
        let scale = 1.0 / (cos_theta2 + sin_theta2);
        let a = (cos_theta2 - sin_theta2) * scale;
        let b = 2.0 * g.sin_theta * g.cos_theta * scale;

        // cache original matrix elements
        let s00 = s.m_00;
        let s20 = s.m_20;
        let s22 = s.m_22;
        let s10 = s.m_10;
        let s21 = s.m_21;

        // Perform the matrix conjugation
        s.m_00 = a * (a * s00 + b * s20) + b * (a * s20 + b * s22);
        s.m_20 = a * (-b * s00 + a * s20) + b * (-b * s20 + a * s22);
        s.m_22 = -b * (-b * s00 + a * s20) + a * (-b * s20 + a * s22);
        s.m_10 = a * s10 + b * s21;
        s.m_21 = -b * s10 + a * s21;

        // Update the cumulative rotation quaternion
        let q_arr = q.to_array();
        let tmp_x = q_arr[0] * g.sin_theta;
        let tmp_y = q_arr[1] * g.sin_theta;
        let tmp_z = q_arr[2] * g.sin_theta;
        g.sin_theta *= q_arr[3];

        *q = QuatF32::from_xyzw(
            q_arr[0] * g.cos_theta - tmp_z,
            q_arr[1] * g.cos_theta + g.sin_theta,
            q_arr[2] * g.cos_theta + tmp_x,
            q_arr[3] * g.cos_theta - tmp_y,
        );
    }

    fn jacobi_eigenanalysis(mut s: Symmetric3x3) -> Mat3F32 {
        let mut q = QuatF32::from_xyzw(0.0, 0.0, 0.0, 1.0);
        for _i in 0..MAX_SWEEPS {
            conjugate_xy(&mut s, &mut q);
            conjugate_yz(&mut s, &mut q);
            conjugate_xz(&mut s, &mut q);

            let off_diag_norm_sq = s.m_10 * s.m_10 + s.m_20 * s.m_20 + s.m_21 * s.m_21;
            if off_diag_norm_sq < JACOBI_TOLERANCE {
                break;
            }
        }
        // Convert quaternion to matrix using Mat3AF32 which has from_quat, then convert to Mat3F32
        let mat3a = Mat3AF32::from_quat(q);
        let x_axis_a = mat3a.x_axis();
        let y_axis_a = mat3a.y_axis();
        let z_axis_a = mat3a.z_axis();
        Mat3F32::from_cols(
            Vec3F32::new(x_axis_a.x, x_axis_a.y, x_axis_a.z),
            Vec3F32::new(y_axis_a.x, y_axis_a.y, y_axis_a.z),
            Vec3F32::new(z_axis_a.x, z_axis_a.y, z_axis_a.z),
        )
    }

    /// Helper function used to swap X with Y and Y with X if c == true
    #[inline(always)]
    fn cond_swap(c: bool, x: &mut f32, y: &mut f32) {
        let z = *x;
        if c {
            *x = *y;
            *y = z;
        }
    }

    /// Helper function to conditionally swap two Vec3s
    #[inline(always)]
    fn cond_swap_vec3(c: bool, x: &mut Vec3F32, y: &mut Vec3F32) {
        let z = *x;
        if c {
            *x = *y;
            *y = z;
        }
    }

    /// Helper function to conditionally negate a Vec3
    #[inline(always)]
    fn cond_negate_vec3(c: bool, v: &mut Vec3F32) {
        let mask = 0_i32.wrapping_sub(c as i32) as u32;
        let sign_mask = 0x8000_0000u32;

        let neg_mask = mask & sign_mask;
        *v = Vec3F32::new(
            f32::from_bits(v.x.to_bits() ^ neg_mask),
            f32::from_bits(v.y.to_bits() ^ neg_mask),
            f32::from_bits(v.z.to_bits() ^ neg_mask),
        );
    }

    /// Sorts the singular values in descending order and adjusts the corresponding singular vectors accordingly
    #[inline(always)]
    pub fn sort_singular_values(b: &mut Mat3F32, v: &mut Mat3F32) {
        let mut b_x = b.x_axis();
        let mut b_y = b.y_axis();
        let mut b_z = b.z_axis();
        let mut v_x = v.x_axis();
        let mut v_y = v.y_axis();
        let mut v_z = v.z_axis();
        let mut rho1 = b_x.length() * b_x.length();
        let mut rho2 = b_y.length() * b_y.length();
        let mut rho3 = b_z.length() * b_z.length();

        // First comparison (rho1, rho2)
        let c1 = rho1 < rho2;
        cond_swap(c1, &mut rho1, &mut rho2);
        cond_swap_vec3(c1, &mut b_x, &mut b_y);
        cond_swap_vec3(c1, &mut v_x, &mut v_y);
        cond_negate_vec3(c1, &mut b_y);
        cond_negate_vec3(c1, &mut v_y);

        // Second comparison (rho1, rho3)
        let c2 = rho1 < rho3;
        cond_swap(c2, &mut rho1, &mut rho3);
        cond_swap_vec3(c2, &mut b_x, &mut b_z);
        cond_swap_vec3(c2, &mut v_x, &mut v_z);
        cond_negate_vec3(c2, &mut b_z);
        cond_negate_vec3(c2, &mut v_z);

        // Third comparison (rho2, rho3)
        let c3 = rho2 < rho3;
        cond_swap_vec3(c3, &mut b_y, &mut b_z);
        cond_swap_vec3(c3, &mut v_y, &mut v_z);
        cond_negate_vec3(c3, &mut b_z);
        cond_negate_vec3(c3, &mut v_z);

        // Update matrices with swapped vectors
        *b = Mat3F32::from_cols(b_x, b_y, b_z);
        *v = Mat3F32::from_cols(v_x, v_y, v_z);
    }

    /// Implementation of Algorithm 4
    #[inline(always)]
    fn qr_givens_quaternion(a1: f32, a2: f32) -> Givens {
        let epsilon = SVD3_EPSILON;
        let rho = (a1 * a1 + a2 * a2).sqrt();

        let mut g = Givens {
            cos_theta: a1.abs() + f32::max(rho, epsilon),
            sin_theta: if rho > epsilon { a2 } else { 0.0 },
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

    /// Implements a QR decomposition of a Matrix using Givens rotations
    fn qr_decomposition(b_mat: &mut Mat3F32) -> QR3 {
        let mut x_axis = b_mat.x_axis();
        let mut y_axis = b_mat.y_axis();
        let mut z_axis = b_mat.z_axis();
        let g1 = qr_givens_quaternion(x_axis.x, x_axis.y);
        let rot_c = -2.0 * g1.sin_theta * g1.sin_theta + 1.0; // cos(2*theta)
        let rot_s = 2.0 * g1.cos_theta * g1.sin_theta; // sin(2*theta)

        // Apply Q1.T to B (row-wise operation on rows 0 and 1)
        let val_row0 = x_axis.x;
        let val_row1 = x_axis.y;
        x_axis.x = rot_c * val_row0 + rot_s * val_row1;
        x_axis.y = -rot_s * val_row0 + rot_c * val_row1;
        let val_row0 = y_axis.x;
        let val_row1 = y_axis.y;
        y_axis.x = rot_c * val_row0 + rot_s * val_row1;
        y_axis.y = -rot_s * val_row0 + rot_c * val_row1;
        let val_row0 = z_axis.x;
        let val_row1 = z_axis.y;
        z_axis.x = rot_c * val_row0 + rot_s * val_row1;
        z_axis.y = -rot_s * val_row0 + rot_c * val_row1;

        // --- Second Givens rotation to zero out b[2][0] (affects rows 0 and 2) ---
        let g2 = qr_givens_quaternion(x_axis.x, x_axis.z);
        let rot_c = -2.0 * g2.sin_theta * g2.sin_theta + 1.0;
        let rot_s = 2.0 * g2.cos_theta * g2.sin_theta;

        // Apply Q2.T to (Q1.T * B)
        let val_row0 = x_axis.x;
        let val_row2 = x_axis.z;
        x_axis.x = rot_c * val_row0 + rot_s * val_row2;
        x_axis.z = -rot_s * val_row0 + rot_c * val_row2;
        let val_row0 = y_axis.x;
        let val_row2 = y_axis.z;
        y_axis.x = rot_c * val_row0 + rot_s * val_row2;
        y_axis.z = -rot_s * val_row0 + rot_c * val_row2;
        let val_row0 = z_axis.x;
        let val_row2 = z_axis.z;
        z_axis.x = rot_c * val_row0 + rot_s * val_row2;
        z_axis.z = -rot_s * val_row0 + rot_c * val_row2;

        // --- Third Givens rotation to zero out b[2][1] (affects rows 1 and 2) ---
        let g3 = qr_givens_quaternion(y_axis.y, y_axis.z);
        let rot_c = -2.0 * g3.sin_theta * g3.sin_theta + 1.0;
        let rot_s = 2.0 * g3.cos_theta * g3.sin_theta;

        // Apply Q3.T to (Q2.T * Q1.T * B)
        let val_row1 = x_axis.y;
        let val_row2 = x_axis.z;
        x_axis.y = rot_c * val_row1 + rot_s * val_row2;
        x_axis.z = -rot_s * val_row1 + rot_c * val_row2;
        let val_row1 = y_axis.y;
        let val_row2 = y_axis.z;
        y_axis.y = rot_c * val_row1 + rot_s * val_row2;
        y_axis.z = -rot_s * val_row1 + rot_c * val_row2;
        let val_row1 = z_axis.y;
        let val_row2 = z_axis.z;
        z_axis.y = rot_c * val_row1 + rot_s * val_row2;
        z_axis.z = -rot_s * val_row1 + rot_c * val_row2;

        let r = Mat3F32::from_cols(x_axis, y_axis, z_axis);
        *b_mat = r;

        // --- Construct Q = Q1 * Q2 * Q3 ---
        let rot_c1 = -2.0 * g1.sin_theta * g1.sin_theta + 1.0;
        let rot_s1 = 2.0 * g1.cos_theta * g1.sin_theta;
        let rot_c2 = -2.0 * g2.sin_theta * g2.sin_theta + 1.0;
        let rot_s2 = 2.0 * g2.cos_theta * g2.sin_theta;
        let rot_c3 = -2.0 * g3.sin_theta * g3.sin_theta + 1.0;
        let rot_s3 = 2.0 * g3.cos_theta * g3.sin_theta;

        let q1 = Mat3F32::from_cols(
            Vec3F32::new(rot_c1, rot_s1, 0.0),
            Vec3F32::new(-rot_s1, rot_c1, 0.0),
            Vec3F32::new(0.0, 0.0, 1.0),
        );

        let q2 = Mat3F32::from_cols(
            Vec3F32::new(rot_c2, 0.0, rot_s2),
            Vec3F32::new(0.0, 1.0, 0.0),
            Vec3F32::new(-rot_s2, 0.0, rot_c2),
        );

        let q3 = Mat3F32::from_cols(
            Vec3F32::new(1.0, 0.0, 0.0),
            Vec3F32::new(0.0, rot_c3, rot_s3),
            Vec3F32::new(0.0, -rot_s3, rot_c3),
        );

        let q = q1 * q2 * q3;
        QR3 { q, r }
    }

    pub fn svd3(a: &Mat3F32) -> SVD3Set {
        let at_a = a.transpose() * *a;
        let mut v = jacobi_eigenanalysis(Symmetric3x3::from_mat3x3(&at_a));
        // Compute B = A * V
        let mut b = *a * v;

        // Sort the singular values
        sort_singular_values(&mut b, &mut v);

        // Perform QR decomposition on B to get Q and R
        let qr = qr_decomposition(&mut b);

        let mut u = qr.q;
        let mut s = qr.r;

        let mut s_x = s.x_axis();
        let mut s_y = s.y_axis();
        let mut s_z = s.z_axis();
        let cond_x = s_x.x < 0.0;
        let cond_y = s_y.y < 0.0;
        let cond_z = s_z.z < 0.0;

        let mut u_x = u.x_axis();
        let mut u_y = u.y_axis();
        let mut u_z = u.z_axis();
        cond_negate_vec3(cond_x, &mut u_x);
        cond_negate_vec3(cond_y, &mut u_y);
        cond_negate_vec3(cond_z, &mut u_z);
        u = Mat3F32::from_cols(u_x, u_y, u_z);

        s_x.x = s_x.x.abs();
        s_y.y = s_y.y.abs();
        s_z.z = s_z.z.abs();
        s = Mat3F32::from_cols(s_x, s_y, s_z);

        // Return the SVD result
        SVD3Set { u, s, v }
    }
}

mod impl_f64 {
    use super::*;

    const EPSILON: f64 = 1e-15;
    const JACOBI_TOLERANCE: f64 = 1e-12;
    // Standard Jacobi rotation requires more sweeps for f64 Exact Givens
    // to ensure convergence without hitting the limit prematurely.
    const MAX_SWEEPS: usize = 20;

    #[derive(Debug, Clone, Copy)]
    struct Symmetric3x3 {
        m_00: f64,
        m_10: f64,
        m_20: f64,
        m_11: f64,
        m_21: f64,
        m_22: f64,
    }

    impl Symmetric3x3 {
        fn from_mat3x3(m: &Mat3F64) -> Self {
            let x_axis = m.x_axis;
            let y_axis = m.y_axis;
            let z_axis = m.z_axis;
            Self {
                m_00: x_axis.dot(x_axis),
                m_10: x_axis.dot(y_axis),
                m_20: x_axis.dot(z_axis),
                m_11: y_axis.dot(y_axis),
                m_21: y_axis.dot(z_axis),
                m_22: z_axis.dot(z_axis),
            }
        }
    }

    #[inline(always)]
    fn exact_givens(app: f64, aqq: f64, apq: f64) -> (f64, f64) {
        if apq.abs() < EPSILON {
            return (1.0, 0.0);
        }
        let tau = (aqq - app) / (2.0 * apq);
        let t = if tau >= 0.0 {
            1.0 / (tau + tau.hypot(1.0))
        } else {
            -1.0 / (-tau + tau.hypot(1.0))
        };
        let c = 1.0 / t.hypot(1.0);
        let s = c * t;
        (c, s)
    }

    #[derive(Debug, Clone)]
    pub struct SVD3Set {
        u: Mat3F64,
        s: Mat3F64,
        v: Mat3F64,
    }

    impl SVD3Set {
        #[inline]
        pub fn u(&self) -> &Mat3F64 {
            &self.u
        }
        #[inline]
        pub fn s(&self) -> &Mat3F64 {
            &self.s
        }
        #[inline]
        pub fn v(&self) -> &Mat3F64 {
            &self.v
        }
    }

    /// Helper to compute the right singular vectors (V) using Exact Givens sweeps
    #[inline]
    fn compute_v_quat(s_mat: &mut Symmetric3x3) -> QuatF64 {
        let mut q_accum = QuatF64::IDENTITY;

        for _sweep in 0..MAX_SWEEPS {
            {
                let (c, si) = exact_givens(s_mat.m_00, s_mat.m_11, s_mat.m_10);
                let s00 = s_mat.m_00;
                let s01 = s_mat.m_10;
                let s11 = s_mat.m_11;
                let s02 = s_mat.m_20;
                let s12 = s_mat.m_21;
                s_mat.m_00 = c * c * s00 - 2.0 * c * si * s01 + si * si * s11;
                s_mat.m_11 = si * si * s00 + 2.0 * c * si * s01 + c * c * s11;
                s_mat.m_10 = 0.0;
                s_mat.m_20 = c * s02 - si * s12;
                s_mat.m_21 = si * s02 + c * s12;
                let ch = ((1.0 + c) / 2.0).sqrt();
                let sh = -si / (2.0 * ch);
                let q = q_accum.to_array();
                q_accum = QuatF64::from_xyzw(
                    q[0] * ch + q[1] * sh,
                    q[1] * ch - q[0] * sh,
                    q[2] * ch + q[3] * sh,
                    q[3] * ch - q[2] * sh,
                );
            }
            {
                let (c, si) = exact_givens(s_mat.m_11, s_mat.m_22, s_mat.m_21);
                let s11 = s_mat.m_11;
                let s12 = s_mat.m_21;
                let s22 = s_mat.m_22;
                let s01 = s_mat.m_10;
                let s02 = s_mat.m_20;
                s_mat.m_11 = c * c * s11 - 2.0 * c * si * s12 + si * si * s22;
                s_mat.m_22 = si * si * s11 + 2.0 * c * si * s12 + c * c * s22;
                s_mat.m_21 = 0.0;
                s_mat.m_10 = c * s01 - si * s02;
                s_mat.m_20 = si * s01 + c * s02;
                let ch = ((1.0 + c) / 2.0).sqrt();
                let sh = -si / (2.0 * ch);
                let q = q_accum.to_array();
                q_accum = QuatF64::from_xyzw(
                    q[0] * ch + q[3] * sh,
                    q[1] * ch + q[2] * sh,
                    q[2] * ch - q[1] * sh,
                    q[3] * ch - q[0] * sh,
                );
            }
            {
                let (c, si) = exact_givens(s_mat.m_00, s_mat.m_22, s_mat.m_20);
                let s00 = s_mat.m_00;
                let s02 = s_mat.m_20;
                let s22 = s_mat.m_22;
                let s01 = s_mat.m_10;
                let s12 = s_mat.m_21;
                s_mat.m_00 = c * c * s00 - 2.0 * c * si * s02 + si * si * s22;
                s_mat.m_22 = si * si * s00 + 2.0 * c * si * s02 + c * c * s22;
                s_mat.m_20 = 0.0;
                s_mat.m_10 = c * s01 - si * s12;
                s_mat.m_21 = si * s01 + c * s12;
                let ch = ((1.0 + c) / 2.0).sqrt();
                let sh = si / (2.0 * ch);
                let q = q_accum.to_array();
                q_accum = QuatF64::from_xyzw(
                    q[0] * ch - q[2] * sh,
                    q[1] * ch + q[3] * sh,
                    q[2] * ch + q[0] * sh,
                    q[3] * ch - q[1] * sh,
                );
            }

            // Normalize quaternion once per sweep to balance orthogonality and performance
            let len = q_accum.length();
            debug_assert!(
                (len - 1.0).abs() < 0.1,
                "Quaternion drift detected in compute_v_quat: length={}",
                len
            );
            q_accum = q_accum.normalize();

            let off_diag_sq =
                s_mat.m_10 * s_mat.m_10 + s_mat.m_20 * s_mat.m_20 + s_mat.m_21 * s_mat.m_21;
            if off_diag_sq < JACOBI_TOLERANCE {
                break;
            }
        }

        q_accum
    }

    #[derive(Debug)]
    struct GivensF64 {
        cos_theta: f64,
        sin_theta: f64,
    }

    #[derive(Debug)]
    struct QR3F64 {
        q: Mat3F64,
        r: Mat3F64,
    }

    #[inline(always)]
    fn qr_givens_quaternion(a1: f64, a2: f64) -> GivensF64 {
        let rho = (a1 * a1 + a2 * a2).sqrt();
        let mut g = GivensF64 {
            cos_theta: a1.abs() + f64::max(rho, EPSILON),
            sin_theta: if rho > EPSILON { a2 } else { 0.0 },
        };

        if a1 < 0.0 {
            std::mem::swap(&mut g.sin_theta, &mut g.cos_theta);
        }

        let w = (g.cos_theta * g.cos_theta + g.sin_theta * g.sin_theta)
            .sqrt()
            .recip();
        g.cos_theta *= w;
        g.sin_theta *= w;
        g
    }

    fn qr_decomposition(b_mat: &mut Mat3F64) -> QR3F64 {
        let mut x_axis = b_mat.x_axis;
        let mut y_axis = b_mat.y_axis;
        let mut z_axis = b_mat.z_axis;

        // First Givens rotation to zero out b[1][0]
        let g1 = qr_givens_quaternion(x_axis.x, x_axis.y);
        let rot_c = -2.0 * g1.sin_theta * g1.sin_theta + 1.0;
        let rot_s = 2.0 * g1.cos_theta * g1.sin_theta;

        let val_row0 = x_axis.x;
        let val_row1 = x_axis.y;
        x_axis.x = rot_c * val_row0 + rot_s * val_row1;
        x_axis.y = -rot_s * val_row0 + rot_c * val_row1;
        let val_row0 = y_axis.x;
        let val_row1 = y_axis.y;
        y_axis.x = rot_c * val_row0 + rot_s * val_row1;
        y_axis.y = -rot_s * val_row0 + rot_c * val_row1;
        let val_row0 = z_axis.x;
        let val_row1 = z_axis.y;
        z_axis.x = rot_c * val_row0 + rot_s * val_row1;
        z_axis.y = -rot_s * val_row0 + rot_c * val_row1;

        // Second Givens rotation to zero out b[2][0]
        let g2 = qr_givens_quaternion(x_axis.x, x_axis.z);
        let rot_c = -2.0 * g2.sin_theta * g2.sin_theta + 1.0;
        let rot_s = 2.0 * g2.cos_theta * g2.sin_theta;

        let val_row0 = x_axis.x;
        let val_row2 = x_axis.z;
        x_axis.x = rot_c * val_row0 + rot_s * val_row2;
        x_axis.z = -rot_s * val_row0 + rot_c * val_row2;
        let val_row0 = y_axis.x;
        let val_row2 = y_axis.z;
        y_axis.x = rot_c * val_row0 + rot_s * val_row2;
        y_axis.z = -rot_s * val_row0 + rot_c * val_row2;
        let val_row0 = z_axis.x;
        let val_row2 = z_axis.z;
        z_axis.x = rot_c * val_row0 + rot_s * val_row2;
        z_axis.z = -rot_s * val_row0 + rot_c * val_row2;

        // Third Givens rotation to zero out b[2][1]
        let g3 = qr_givens_quaternion(y_axis.y, y_axis.z);
        let rot_c = -2.0 * g3.sin_theta * g3.sin_theta + 1.0;
        let rot_s = 2.0 * g3.cos_theta * g3.sin_theta;

        let val_row1 = x_axis.y;
        let val_row2 = x_axis.z;
        x_axis.y = rot_c * val_row1 + rot_s * val_row2;
        x_axis.z = -rot_s * val_row1 + rot_c * val_row2;
        let val_row1 = y_axis.y;
        let val_row2 = y_axis.z;
        y_axis.y = rot_c * val_row1 + rot_s * val_row2;
        y_axis.z = -rot_s * val_row1 + rot_c * val_row2;
        let val_row1 = z_axis.y;
        let val_row2 = z_axis.z;
        z_axis.y = rot_c * val_row1 + rot_s * val_row2;
        z_axis.z = -rot_s * val_row1 + rot_c * val_row2;

        let r = Mat3F64::from_cols(x_axis.into(), y_axis.into(), z_axis.into());
        *b_mat = r;

        let rot_c1 = -2.0 * g1.sin_theta * g1.sin_theta + 1.0;
        let rot_s1 = 2.0 * g1.cos_theta * g1.sin_theta;
        let rot_c2 = -2.0 * g2.sin_theta * g2.sin_theta + 1.0;
        let rot_s2 = 2.0 * g2.cos_theta * g2.sin_theta;
        let rot_c3 = -2.0 * g3.sin_theta * g3.sin_theta + 1.0;
        let rot_s3 = 2.0 * g3.cos_theta * g3.sin_theta;

        let q1 = Mat3F64::from_cols(
            Vec3F64::new(rot_c1, rot_s1, 0.0),
            Vec3F64::new(-rot_s1, rot_c1, 0.0),
            Vec3F64::new(0.0, 0.0, 1.0),
        );
        let q2 = Mat3F64::from_cols(
            Vec3F64::new(rot_c2, 0.0, rot_s2),
            Vec3F64::new(0.0, 1.0, 0.0),
            Vec3F64::new(-rot_s2, 0.0, rot_c2),
        );
        let q3 = Mat3F64::from_cols(
            Vec3F64::new(1.0, 0.0, 0.0),
            Vec3F64::new(0.0, rot_c3, rot_s3),
            Vec3F64::new(0.0, -rot_s3, rot_c3),
        );

        QR3F64 { q: q1 * q2 * q3, r }
    }

    pub fn sort_singular_values(b: &mut Mat3F64, v: &mut Mat3F64) {
        let mut b_x = b.x_axis;
        let mut b_y = b.y_axis;
        let mut b_z = b.z_axis;
        let mut v_x = v.x_axis;
        let mut v_y = v.y_axis;
        let mut v_z = v.z_axis;
        let mut rho1 = b_x.dot(b_x);
        let mut rho2 = b_y.dot(b_y);
        let mut rho3 = b_z.dot(b_z);

        if rho1 < rho2 {
            std::mem::swap(&mut rho1, &mut rho2);
            std::mem::swap(&mut b_x, &mut b_y);
            std::mem::swap(&mut v_x, &mut v_y);
            b_y = -b_y;
            v_y = -v_y;
        }
        if rho1 < rho3 {
            std::mem::swap(&mut rho1, &mut rho3);
            std::mem::swap(&mut b_x, &mut b_z);
            std::mem::swap(&mut v_x, &mut v_z);
            b_z = -b_z;
            v_z = -v_z;
        }
        if rho2 < rho3 {
            std::mem::swap(&mut b_y, &mut b_z);
            std::mem::swap(&mut v_y, &mut v_z);
            b_z = -b_z;
            v_z = -v_z;
        }
        *b = Mat3F64::from_cols(b_x.into(), b_y.into(), b_z.into());
        *v = Mat3F64::from_cols(v_x.into(), v_y.into(), v_z.into());
    }

    pub fn svd3(mat: &Mat3F64) -> SVD3Set {
        let mut s_mat = Symmetric3x3::from_mat3x3(&(mat.transpose() * *mat));
        let q = compute_v_quat(&mut s_mat);
        let mut v_mat = Mat3F64::from_quat(q);

        let mut b = *mat * v_mat;
        sort_singular_values(&mut b, &mut v_mat);

        // Use QR decomposition of B to obtain an orthonormal U and upper-triangular S,
        // instead of inferring U directly from B's columns via ad-hoc normalization.
        let qr = qr_decomposition(&mut b);

        let mut u = qr.q;
        let mut s = qr.r;

        let s_x = s.x_axis;
        let s_y = s.y_axis;
        let s_z = s.z_axis;
        let mut u_x = u.x_axis;
        let mut u_y = u.y_axis;
        let mut u_z = u.z_axis;

        if s_x.x < 0.0 {
            u_x = -u_x;
        }
        if s_y.y < 0.0 {
            u_y = -u_y;
        }
        if s_z.z < 0.0 {
            u_z = -u_z;
        }

        u = Mat3F64::from_cols(u_x.into(), u_y.into(), u_z.into());
        s = Mat3F64::from_cols(
            Vec3F64::new(s_x.x.abs(), s_y.x, s_z.x),
            Vec3F64::new(s_x.y, s_y.y.abs(), s_z.y),
            Vec3F64::new(s_x.z, s_y.z, s_z.z.abs()),
        );

        SVD3Set { u, s, v: v_mat }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Mat3F32, Mat3F64, Vec3F32, Vec3F64};

    const TEST_EPSILON_F32: f32 = 1e-5;
    const TEST_EPSILON_F32_STRICT: f32 = 1e-6;
    const TEST_EPSILON_F64: f64 = 1e-10;

    fn verify_svd_properties_f32(a: &Mat3F32, svd: &SVD3SetF32, epsilon: f32) {
        let u = *svd.u();
        let s = *svd.s();
        let v = *svd.v();
        let reconstruction = u * s * v.transpose();
        // Check reconstruction
        let diff = *a - reconstruction;
        assert!(
            diff.x_axis().length() < epsilon,
            "Reconstruction error (x): {:?}",
            diff
        );
        assert!(
            diff.y_axis().length() < epsilon,
            "Reconstruction error (y): {:?}",
            diff
        );
        assert!(
            diff.z_axis().length() < epsilon,
            "Reconstruction error (z): {:?}",
            diff
        );

        // Check orthogonality U
        let u_t_u = u.transpose() * u;
        let diff_u = u_t_u - Mat3F32::IDENTITY;
        assert!(
            diff_u.x_axis().length() < epsilon,
            "U orthogonality error (x)"
        );
        assert!(
            diff_u.y_axis().length() < epsilon,
            "U orthogonality error (y)"
        );
        assert!(
            diff_u.z_axis().length() < epsilon,
            "U orthogonality error (z)"
        );

        // Check orthogonality V
        let v_t_v = v.transpose() * v;
        let diff_v = v_t_v - Mat3F32::IDENTITY;
        assert!(
            diff_v.x_axis().length() < epsilon,
            "V orthogonality error (x)"
        );
        assert!(
            diff_v.y_axis().length() < epsilon,
            "V orthogonality error (y)"
        );
        assert!(
            diff_v.z_axis().length() < epsilon,
            "V orthogonality error (z)"
        );
    }

    fn verify_svd_properties_f64(a: &Mat3F64, svd: &SVD3SetF64, epsilon: f64) {
        let u = *svd.u();
        let s = *svd.s();
        let v = *svd.v();
        let reconstruction = u * s * v.transpose();
        // Check reconstruction
        let diff = *a - reconstruction;
        assert!(
            diff.x_axis().length() < epsilon,
            "Reconstruction error (x): {:?}",
            diff
        );
        assert!(
            diff.y_axis().length() < epsilon,
            "Reconstruction error (y): {:?}",
            diff
        );
        assert!(
            diff.z_axis().length() < epsilon,
            "Reconstruction error (z): {:?}",
            diff
        );

        // Check orthogonality U
        let u_t_u = u.transpose() * u;
        let diff_u = u_t_u - Mat3F64::IDENTITY;
        assert!(
            diff_u.x_axis().length() < epsilon,
            "U orthogonality error (x)"
        );
        assert!(
            diff_u.y_axis().length() < epsilon,
            "U orthogonality error (y)"
        );
        assert!(
            diff_u.z_axis().length() < epsilon,
            "U orthogonality error (z)"
        );

        // Check orthogonality V
        let v_t_v = v.transpose() * v;
        let diff_v = v_t_v - Mat3F64::IDENTITY;
        assert!(
            diff_v.x_axis().length() < epsilon,
            "V orthogonality error (x)"
        );
        assert!(
            diff_v.y_axis().length() < epsilon,
            "V orthogonality error (y)"
        );
        assert!(
            diff_v.z_axis().length() < epsilon,
            "V orthogonality error (z)"
        );
    }

    #[test]
    fn test_svd3_f32_simple() {
        let a = Mat3F32::from_diagonal(Vec3F32::new(3.0, 2.0, 1.0));
        let svd = svd3_f32(&a);
        verify_svd_properties_f32(&a, &svd, TEST_EPSILON_F32);
    }

    #[test]
    fn test_svd3_f32_zero() {
        let a = Mat3F32::from_cols(Vec3F32::ZERO, Vec3F32::ZERO, Vec3F32::ZERO);
        let svd_result = svd3_f32(&a);
        verify_svd_properties_f32(&a, &svd_result, TEST_EPSILON_F32_STRICT);
        let s_x = svd_result.s().x_axis();
        assert!(s_x.x.abs() < TEST_EPSILON_F32_STRICT);
    }

    #[test]
    fn test_svd3_f32_identity() {
        let a = Mat3F32::IDENTITY;
        let svd_result = svd3_f32(&a);
        verify_svd_properties_f32(&a, &svd_result, TEST_EPSILON_F32_STRICT);
    }

    #[test]
    fn test_svd3_f32_rotation_matrix() {
        let cos = std::f32::consts::FRAC_PI_4.cos();
        let sin = std::f32::consts::FRAC_PI_4.sin();
        let a = Mat3F32::from_cols(
            Vec3F32::new(cos, 0.0, sin),
            Vec3F32::new(0.0, 1.0, 0.0),
            Vec3F32::new(-sin, 0.0, cos),
        );
        let svd_result = svd3_f32(&a);
        verify_svd_properties_f32(&a, &svd_result, TEST_EPSILON_F32_STRICT);
    }

    #[test]
    fn test_svd3_f64_simple() {
        let a = Mat3F64::from_diagonal(Vec3F64::new(3.0, 2.0, 1.0));
        let svd = svd3_f64(&a);
        verify_svd_properties_f64(&a, &svd, TEST_EPSILON_F64);
    }

    #[test]
    fn test_svd3_f64_zero() {
        let a = Mat3F64::from_cols(Vec3F64::ZERO, Vec3F64::ZERO, Vec3F64::ZERO);
        let svd_result = svd3_f64(&a);
        verify_svd_properties_f64(&a, &svd_result, TEST_EPSILON_F64);
        let s_x = svd_result.s().x_axis();
        assert!(s_x.x.abs() < TEST_EPSILON_F64);
    }

    #[test]
    fn test_svd3_f64_identity() {
        let a = Mat3F64::IDENTITY;
        let svd_result = svd3_f64(&a);
        verify_svd_properties_f64(&a, &svd_result, TEST_EPSILON_F64);
    }

    #[test]
    fn test_svd3_f64_random_rotation() {
        // Test with a known rotation matrix (should have singular values 1,1,1)
        let ang = std::f64::consts::PI / 4.0;
        let cos = ang.cos();
        let sin = ang.sin();
        let a = Mat3F64::from_cols(
            Vec3F64::new(cos, sin, 0.0),
            Vec3F64::new(-sin, cos, 0.0),
            Vec3F64::new(0.0, 0.0, 1.0),
        );
        let svd = svd3_f64(&a);
        verify_svd_properties_f64(&a, &svd, TEST_EPSILON_F64);
        let s = svd.s();
        let s_vec = Vec3F64::new(s.x_axis().x, s.y_axis().y, s.z_axis().z);
        assert!((s_vec.x - 1.0).abs() < TEST_EPSILON_F64);
        assert!((s_vec.y - 1.0).abs() < TEST_EPSILON_F64);
        assert!((s_vec.z - 1.0).abs() < TEST_EPSILON_F64);
    }

    #[test]
    fn test_svd3_f64_degenerate_essential_matrix() {
        // Essential matrix E with repeated singular values (sigma, sigma, 0)
        let e = Mat3F64::from_cols(
            Vec3F64::new(0.0, -2.552, 0.0),
            Vec3F64::new(1.708, 0.0, -8.540),
            Vec3F64::new(0.0, 8.327, 0.0),
        );
        let svd = svd3_f64(&e);

        // Verify full SVD properties: orthogonality and reconstruction
        verify_svd_properties_f64(&e, &svd, TEST_EPSILON_F64);

        let s = svd.s();
        let sigma1 = s.x_axis.x;
        let sigma2 = s.y_axis.y;
        let sigma3 = s.z_axis.z;

        let diff = (sigma1 - sigma2).abs();
        assert!(
            diff < 1e-3,
            "FAILURE: Singular values should be equal! Diff: {}",
            diff
        );
        assert!(
            sigma3.abs() < 1e-3,
            "FAILURE: Third singular value should be zero! Got: {}",
            sigma3
        );
        assert!(
            (sigma1 - 8.709).abs() < 1e-2,
            "FAILURE: Incorrect magnitude. Expected ~8.709"
        );
    }
}
