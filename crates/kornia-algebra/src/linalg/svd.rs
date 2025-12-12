// Reference: https://github.com/wi-re/tbtSVD/blob/master/source/SVD.h
use crate::{Mat3AF32, Mat3F32, QuatF32, Vec3F32};
const GAMMA: f32 = 5.828_427_3;
const CSTAR: f32 = 0.923_879_5;
const SSTAR: f32 = 0.382_683_43;
const SVD3_EPSILON: f32 = 1e-6;
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

    // Calculate rotation matrix elements 'a' and 'b'
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

    // Calculate rotation matrix elements 'a' and 'b'
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
        if off_diag_norm_sq < 1e-6 {
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
    let a1 = -2.0 * g1.sin_theta * g1.sin_theta + 1.0; // a1 = cos(theta1)
    let b1 = 2.0 * g1.cos_theta * g1.sin_theta; // b1 = sin(theta1)

    // Apply Q1.T to B (row-wise operation)
    let c0 = x_axis.x;
    let c1 = x_axis.y;
    x_axis.x = a1 * c0 + b1 * c1;
    x_axis.y = -b1 * c0 + a1 * c1;
    let c0 = y_axis.x;
    let c1 = y_axis.y;
    y_axis.x = a1 * c0 + b1 * c1;
    y_axis.y = -b1 * c0 + a1 * c1;
    let c0 = z_axis.x;
    let c1 = z_axis.y;
    z_axis.x = a1 * c0 + b1 * c1;
    z_axis.y = -b1 * c0 + a1 * c1;

    // --- Second Givens rotation to zero out b[2][0] (affects rows 0 and 2) ---
    let g2 = qr_givens_quaternion(x_axis.x, x_axis.z);
    let a2 = -2.0 * g2.sin_theta * g2.sin_theta + 1.0; // a2 = cos(theta2)
    let b2 = 2.0 * g2.cos_theta * g2.sin_theta; // b2 = sin(theta2)

    // Apply Q2.T to (Q1.T * B)
    let c0 = x_axis.x;
    let c2 = x_axis.z;
    x_axis.x = a2 * c0 + b2 * c2;
    x_axis.z = -b2 * c0 + a2 * c2;
    let c0 = y_axis.x;
    let c2 = y_axis.z;
    y_axis.x = a2 * c0 + b2 * c2;
    y_axis.z = -b2 * c0 + a2 * c2;
    let c0 = z_axis.x;
    let c2 = z_axis.z;
    z_axis.x = a2 * c0 + b2 * c2;
    z_axis.z = -b2 * c0 + a2 * c2;

    // --- Third Givens rotation to zero out b[2][1] (affects rows 1 and 2) ---
    let g3 = qr_givens_quaternion(y_axis.y, y_axis.z);
    let a3 = -2.0 * g3.sin_theta * g3.sin_theta + 1.0; // a3 = cos(theta3)
    let b3 = 2.0 * g3.cos_theta * g3.sin_theta; // b3 = sin(theta3)

    // Apply Q3.T to (Q2.T * Q1.T * B)
    let c1 = x_axis.y;
    let c2 = x_axis.z;
    x_axis.y = a3 * c1 + b3 * c2;
    x_axis.z = -b3 * c1 + a3 * c2;
    let c1 = y_axis.y;
    let c2 = y_axis.z;
    y_axis.y = a3 * c1 + b3 * c2;
    y_axis.z = -b3 * c1 + a3 * c2;
    let c1 = z_axis.y;
    let c2 = z_axis.z;
    z_axis.y = a3 * c1 + b3 * c2;
    z_axis.z = -b3 * c1 + a3 * c2;

    let r = Mat3F32::from_cols(x_axis, y_axis, z_axis);
    *b_mat = r;

    // --- Construct Q = Q1 * Q2 * Q3 ---
    // Q1 = (Q1.T).T
    let q1 = Mat3F32::from_cols(
        Vec3F32::new(a1, b1, 0.0),
        Vec3F32::new(-b1, a1, 0.0),
        Vec3F32::new(0.0, 0.0, 1.0),
    );

    // Q2 = (Q2.T).T
    let q2 = Mat3F32::from_cols(
        Vec3F32::new(a2, 0.0, b2),
        Vec3F32::new(0.0, 1.0, 0.0),
        Vec3F32::new(-b2, 0.0, a2),
    );

    // Q3 = (Q3.T).T
    let q3 = Mat3F32::from_cols(
        Vec3F32::new(1.0, 0.0, 0.0),
        Vec3F32::new(0.0, a3, b3),
        Vec3F32::new(0.0, -b3, a3),
    );

    let q = q1 * q2 * q3;
    QR3 { q, r }
}

/// Wrapping function used to contain all of the required sub calls
pub fn svd3(a: &Mat3F32) -> SVD3Set {
    // Compute the eigenvectors of A^T * A, which is V in SVD (right singular vectors)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Mat3F32, Vec3F32};

    /// Helper function to validate all critical SVD properties
    fn verify_svd_properties(a: &Mat3F32, svd: &SVD3Set, epsilon: f32) {
        let u = svd.u;
        let s = svd.s;
        let v = svd.v;

        // Property 1: Reconstruction (A = U * S * V.T)
        let reconstruction = u * s * v.transpose();
        let reconstruction_epsilon = 1e-4;
        // Compare element by element since we don't have abs_diff_eq on Mat3F32
        let a_x = a.x_axis();
        let a_y = a.y_axis();
        let a_z = a.z_axis();
        let r_x = reconstruction.x_axis();
        let r_y = reconstruction.y_axis();
        let r_z = reconstruction.z_axis();
        assert!(
            (a_x.x - r_x.x).abs() < reconstruction_epsilon
                && (a_x.y - r_x.y).abs() < reconstruction_epsilon
                && (a_x.z - r_x.z).abs() < reconstruction_epsilon
                && (a_y.x - r_y.x).abs() < reconstruction_epsilon
                && (a_y.y - r_y.y).abs() < reconstruction_epsilon
                && (a_y.z - r_y.z).abs() < reconstruction_epsilon
                && (a_z.x - r_z.x).abs() < reconstruction_epsilon
                && (a_z.y - r_z.y).abs() < reconstruction_epsilon
                && (a_z.z - r_z.z).abs() < reconstruction_epsilon,
            "Reconstruction failed: A != U*S*V.T"
        );

        // Property 2: U is Orthogonal (U.T * U = I)
        let u_t_u = u.transpose() * u;
        let u_t_u_x = u_t_u.x_axis();
        let u_t_u_y = u_t_u.y_axis();
        let u_t_u_z = u_t_u.z_axis();
        let identity = Mat3F32::IDENTITY;
        let id_x = identity.x_axis();
        let id_y = identity.y_axis();
        let id_z = identity.z_axis();
        assert!(
            (u_t_u_x.x - id_x.x).abs() < epsilon
                && (u_t_u_x.y - id_x.y).abs() < epsilon
                && (u_t_u_x.z - id_x.z).abs() < epsilon
                && (u_t_u_y.x - id_y.x).abs() < epsilon
                && (u_t_u_y.y - id_y.y).abs() < epsilon
                && (u_t_u_y.z - id_y.z).abs() < epsilon
                && (u_t_u_z.x - id_z.x).abs() < epsilon
                && (u_t_u_z.y - id_z.y).abs() < epsilon
                && (u_t_u_z.z - id_z.z).abs() < epsilon,
            "U is not orthogonal: U.T*U != I"
        );

        // Property 3: V is Orthogonal (V.T * V = I)
        let v_t_v = v.transpose() * v;
        let v_t_v_x = v_t_v.x_axis();
        let v_t_v_y = v_t_v.y_axis();
        let v_t_v_z = v_t_v.z_axis();
        assert!(
            (v_t_v_x.x - id_x.x).abs() < epsilon
                && (v_t_v_x.y - id_x.y).abs() < epsilon
                && (v_t_v_x.z - id_x.z).abs() < epsilon
                && (v_t_v_y.x - id_y.x).abs() < epsilon
                && (v_t_v_y.y - id_y.y).abs() < epsilon
                && (v_t_v_y.z - id_y.z).abs() < epsilon
                && (v_t_v_z.x - id_z.x).abs() < epsilon
                && (v_t_v_z.y - id_z.y).abs() < epsilon
                && (v_t_v_z.z - id_z.z).abs() < epsilon,
            "V is not orthogonal: V.T*V != I"
        );

        // Property 4: S is Diagonal
        let s_x = s.x_axis();
        let s_y = s.y_axis();
        let s_z = s.z_axis();
        let s_diag = Vec3F32::new(s_x.x, s_y.y, s_z.z);
        assert!(
            s_diag.x >= 0.0 && s_diag.y >= 0.0 && s_diag.z >= 0.0,
            "Singular values are not non-negative: ({}, {}, {})",
            s_diag.x,
            s_diag.y,
            s_diag.z
        );
        assert!(
            s_diag.x >= s_diag.y - epsilon && s_diag.y >= s_diag.z - epsilon,
            "Singular values are not sorted: ({}, {}, {})",
            s_diag.x,
            s_diag.y,
            s_diag.z
        );
    }

    #[test]
    fn test_svd3_1_diagonal_sorted() {
        let a = Mat3F32::from_diagonal(Vec3F32::new(3.0, 2.0, 1.0));
        let svd_result = svd3(&a);
        verify_svd_properties(&a, &svd_result, SVD3_EPSILON);

        let s_x = svd_result.s.x_axis();
        let s_y = svd_result.s.y_axis();
        let s_z = svd_result.s.z_axis();
        let s_diag = Vec3F32::new(s_x.x, s_y.y, s_z.z);
        let expected = Vec3F32::new(3.0, 2.0, 1.0);
        assert!((s_diag.x - expected.x).abs() < SVD3_EPSILON);
        assert!((s_diag.y - expected.y).abs() < SVD3_EPSILON);
        assert!((s_diag.z - expected.z).abs() < SVD3_EPSILON);
    }

    #[test]
    fn test_svd3_2_zero() {
        let a = Mat3F32::from_cols(Vec3F32::ZERO, Vec3F32::ZERO, Vec3F32::ZERO);
        let svd_result = svd3(&a);
        verify_svd_properties(&a, &svd_result, SVD3_EPSILON);
        let s_x = svd_result.s.x_axis();
        let s_y = svd_result.s.y_axis();
        let s_z = svd_result.s.z_axis();
        assert!(
            s_x.x.abs() < SVD3_EPSILON && s_x.y.abs() < SVD3_EPSILON && s_x.z.abs() < SVD3_EPSILON
        );
        assert!(
            s_y.x.abs() < SVD3_EPSILON && s_y.y.abs() < SVD3_EPSILON && s_y.z.abs() < SVD3_EPSILON
        );
        assert!(
            s_z.x.abs() < SVD3_EPSILON && s_z.y.abs() < SVD3_EPSILON && s_z.z.abs() < SVD3_EPSILON
        );
    }

    #[test]
    fn test_svd3_3_identity() {
        let a = Mat3F32::IDENTITY;
        let svd_result = svd3(&a);
        verify_svd_properties(&a, &svd_result, SVD3_EPSILON);
        let s_x = svd_result.s.x_axis();
        let s_y = svd_result.s.y_axis();
        let s_z = svd_result.s.z_axis();
        let identity = Mat3F32::IDENTITY;
        let id_x = identity.x_axis();
        let id_y = identity.y_axis();
        let id_z = identity.z_axis();
        assert!(
            (s_x.x - id_x.x).abs() < SVD3_EPSILON
                && (s_x.y - id_x.y).abs() < SVD3_EPSILON
                && (s_x.z - id_x.z).abs() < SVD3_EPSILON
        );
        assert!(
            (s_y.x - id_y.x).abs() < SVD3_EPSILON
                && (s_y.y - id_y.y).abs() < SVD3_EPSILON
                && (s_y.z - id_y.z).abs() < SVD3_EPSILON
        );
        assert!(
            (s_z.x - id_z.x).abs() < SVD3_EPSILON
                && (s_z.y - id_z.y).abs() < SVD3_EPSILON
                && (s_z.z - id_z.z).abs() < SVD3_EPSILON
        );
    }

    #[test]
    fn test_svd3_4_singular_rank1() {
        let a = Mat3F32::from_cols(
            Vec3F32::new(1.0, 2.0, 3.0),
            Vec3F32::new(2.0, 4.0, 6.0),
            Vec3F32::new(3.0, 6.0, 9.0),
        );
        let svd_result = svd3(&a);
        verify_svd_properties(&a, &svd_result, SVD3_EPSILON);

        let s_x = svd_result.s.x_axis();
        let s_y = svd_result.s.y_axis();
        let s_z = svd_result.s.z_axis();
        let s_diag = Vec3F32::new(s_x.x, s_y.y, s_z.z);
        assert!(s_diag.x > SVD3_EPSILON);
        assert!(s_diag.y.abs() < SVD3_EPSILON);
        assert!(s_diag.z.abs() < SVD3_EPSILON);
    }

    #[test]
    fn test_svd3_diagonal_unsorted() {
        let a = Mat3F32::from_diagonal(Vec3F32::new(2.0, 3.0, 1.0));
        let svd_result = svd3(&a);
        verify_svd_properties(&a, &svd_result, SVD3_EPSILON);
        let s_x = svd_result.s.x_axis();
        let s_y = svd_result.s.y_axis();
        let s_z = svd_result.s.z_axis();
        let s_diag = Vec3F32::new(s_x.x, s_y.y, s_z.z);
        let expected = Vec3F32::new(3.0, 2.0, 1.0);
        assert!((s_diag.x - expected.x).abs() < SVD3_EPSILON);
        assert!((s_diag.y - expected.y).abs() < SVD3_EPSILON);
        assert!((s_diag.z - expected.z).abs() < SVD3_EPSILON);
    }

    #[test]
    fn test_svd3_rotation_matrix() {
        // Create a rotation matrix around Y axis
        let cos = std::f32::consts::FRAC_PI_4.cos();
        let sin = std::f32::consts::FRAC_PI_4.sin();
        let a = Mat3F32::from_cols(
            Vec3F32::new(cos, 0.0, sin),
            Vec3F32::new(0.0, 1.0, 0.0),
            Vec3F32::new(-sin, 0.0, cos),
        );
        let svd_result = svd3(&a);
        verify_svd_properties(&a, &svd_result, SVD3_EPSILON);

        let s_x = svd_result.s.x_axis();
        let s_y = svd_result.s.y_axis();
        let s_z = svd_result.s.z_axis();
        let s_diag = Vec3F32::new(s_x.x, s_y.y, s_z.z);
        let expected = Vec3F32::new(1.0, 1.0, 1.0);
        assert!((s_diag.x - expected.x).abs() < SVD3_EPSILON);
        assert!((s_diag.y - expected.y).abs() < SVD3_EPSILON);
        assert!((s_diag.z - expected.z).abs() < SVD3_EPSILON);
    }

    #[test]
    fn test_svd3_reflection_matrix() {
        let a = Mat3F32::from_diagonal(Vec3F32::new(1.0, -1.0, 1.0));
        let svd_result = svd3(&a);
        verify_svd_properties(&a, &svd_result, SVD3_EPSILON);

        let s_x = svd_result.s.x_axis();
        let s_y = svd_result.s.y_axis();
        let s_z = svd_result.s.z_axis();
        let s_diag = Vec3F32::new(s_x.x, s_y.y, s_z.z);
        let expected = Vec3F32::new(1.0, 1.0, 1.0);
        assert!((s_diag.x - expected.x).abs() < SVD3_EPSILON);
        assert!((s_diag.y - expected.y).abs() < SVD3_EPSILON);
        assert!((s_diag.z - expected.z).abs() < SVD3_EPSILON);
    }

    #[test]
    fn test_svd3_general_full_rank() {
        let a = Mat3F32::from_cols(
            Vec3F32::new(1.0, 4.0, 7.0),
            Vec3F32::new(2.0, 5.0, 8.0),
            Vec3F32::new(3.0, 6.0, 10.0),
        );
        let svd_result = svd3(&a);
        verify_svd_properties(&a, &svd_result, SVD3_EPSILON);
        let s_x = svd_result.s.x_axis();
        let s_y = svd_result.s.y_axis();
        let s_z = svd_result.s.z_axis();
        let s_diag = Vec3F32::new(s_x.x, s_y.y, s_z.z);
        let min_val = s_diag.x.min(s_diag.y).min(s_diag.z);
        assert!(min_val > SVD3_EPSILON);
    }

    #[test]
    fn test_svd3_singular_rank2() {
        let a = Mat3F32::from_cols(
            Vec3F32::new(1.0, 2.0, 3.0),
            Vec3F32::new(4.0, 5.0, 6.0),
            Vec3F32::new(5.0, 7.0, 9.0), // c0 + c1
        );
        let svd_result = svd3(&a);
        verify_svd_properties(&a, &svd_result, SVD3_EPSILON);

        let s_x = svd_result.s.x_axis();
        let s_y = svd_result.s.y_axis();
        let s_z = svd_result.s.z_axis();
        let s_diag = Vec3F32::new(s_x.x, s_y.y, s_z.z);
        assert!(s_diag.x > SVD3_EPSILON);
        assert!(s_diag.y > SVD3_EPSILON);
        assert!(s_diag.z.abs() < SVD3_EPSILON);
    }
}
