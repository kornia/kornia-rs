//! Fast 3×3 Singular Value Decomposition (SVD) implementation.
//!
//! This module provides an efficient, branch-minimizing algorithm for computing
//! the singular value decomposition of 3×3 matrices, commonly needed in computer
//! vision applications such as pose estimation, point cloud registration, and
//! camera calibration.
//!
//! # Mathematical Background
//!
//! For any matrix A ∈ ℝ³ˣ³, the SVD decomposes it into three matrices:
//!
//! ```text
//! A = U Σ Vᵀ
//! ```
//!
//! where:
//! * U ∈ ℝ³ˣ³ is an orthogonal matrix (left singular vectors)
//! * Σ ∈ ℝ³ˣ³ is a diagonal matrix of singular values (σ₁ ≥ σ₂ ≥ σ₃ ≥ 0)
//! * V ∈ ℝ³ˣ³ is an orthogonal matrix (right singular vectors)
//!
//! # Implementation Details
//!
//! This implementation uses:
//! * Jacobi eigenvalue algorithm for symmetric matrices
//! * QR decomposition using Givens rotations
//! * Minimal branching for SIMD-friendly performance
//!
//! # Example
//!
//! ```
//! use glam::Mat3;
//! use kornia_linalg::svd::svd3;
//!
//! let matrix = Mat3::from_cols_array(&[
//!     1.0, 0.0, 0.0,
//!     0.0, 2.0, 0.0,
//!     0.0, 0.0, 3.0,
//! ]);
//!
//! let svd_result = svd3(&matrix);
//! let u = svd_result.u();
//! let s = svd_result.s();
//! let v = svd_result.v();
//! ```
//!
//! # References
//!
//! * McAdams, Selle, Tamstorf, Teran, and Sifakis (2011).
//!   "Computing the Singular Value Decomposition of 3x3 matrices with minimal
//!   branching and elementary floating point operations."
//!   University of Wisconsin-Madison Technical Report TR1690.
//!
//! # See also
//!
//! * [`crate::rigid`] for using SVD in rigid body transformations

// Reference: https://github.com/wi-re/tbtSVD/blob/master/source/SVD.h
use glam::{Mat3, Quat, Vec3};
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
    fn from_mat3x3(mat: &Mat3) -> Self {
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
    q: Mat3,

    /// The upper triangular matrix R from the QR decomposition.
    r: Mat3,
}

#[derive(Debug)]
/// Helper struct to store 3 Matrices to avoid OUT parameters on functions
pub struct SVD3Set {
    /// The matrix of left singular vectors.
    u: Mat3,

    /// The diagonal matrix of singular values.
    s: Mat3,

    /// The matrix of right singular vectors.
    v: Mat3,
}

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
fn conjugate_xy(s: &mut Symmetric3x3, q: &mut Quat) {
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
        if off_diag_norm_sq < 1e-6 {
            break;
        }
    }
    Mat3::from_quat(q)
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
fn cond_swap_vec3(c: bool, x: &mut Vec3, y: &mut Vec3) {
    let z = *x;
    if c {
        *x = *y;
        *y = z;
    }
}

/// Helper function to conditionally negate a Vec3
#[inline(always)]
fn cond_negate_vec3(c: bool, v: &mut Vec3) {
    let mask = 0_i32.wrapping_sub(c as i32) as u32;
    let sign_mask = 0x8000_0000u32;

    let neg_mask = mask & sign_mask;
    *v = Vec3::new(
        f32::from_bits(v.x.to_bits() ^ neg_mask),
        f32::from_bits(v.y.to_bits() ^ neg_mask),
        f32::from_bits(v.z.to_bits() ^ neg_mask),
    );
}

/// Sorts the singular values in descending order and adjusts the corresponding singular vectors accordingly
#[inline(always)]
pub fn sort_singular_values(b: &mut Mat3, v: &mut Mat3) {
    let mut rho1 = b.x_axis.length_squared();
    let mut rho2 = b.y_axis.length_squared();
    let mut rho3 = b.z_axis.length_squared();

    // First comparison (rho1, rho2)
    let c1 = rho1 < rho2;
    cond_swap(c1, &mut rho1, &mut rho2);
    cond_swap_vec3(c1, &mut b.x_axis, &mut b.y_axis);
    cond_swap_vec3(c1, &mut v.x_axis, &mut v.y_axis);
    cond_negate_vec3(c1, &mut b.y_axis);
    cond_negate_vec3(c1, &mut v.y_axis);

    // Second comparison (rho1, rho3)
    let c2 = rho1 < rho3;
    cond_swap(c2, &mut rho1, &mut rho3);
    cond_swap_vec3(c2, &mut b.x_axis, &mut b.z_axis);
    cond_swap_vec3(c2, &mut v.x_axis, &mut v.z_axis);
    cond_negate_vec3(c2, &mut b.z_axis);
    cond_negate_vec3(c2, &mut v.z_axis);

    // Third comparison (rho2, rho3)
    let c3 = rho2 < rho3;
    cond_swap_vec3(c3, &mut b.y_axis, &mut b.z_axis);
    cond_swap_vec3(c3, &mut v.y_axis, &mut v.z_axis);
    cond_negate_vec3(c3, &mut b.z_axis);
    cond_negate_vec3(c3, &mut v.z_axis);
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
fn qr_decomposition(b_mat: &mut Mat3) -> QR3 {
    let g1 = qr_givens_quaternion(b_mat.x_axis.x, b_mat.x_axis.y);
    let a1 = -2.0 * g1.sin_theta * g1.sin_theta + 1.0; // a1 = cos(theta1)
    let b1 = 2.0 * g1.cos_theta * g1.sin_theta; // b1 = sin(theta1)

    // Apply Q1.T to B (row-wise operation)
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

    // --- Second Givens rotation to zero out b[2][0] (affects rows 0 and 2) ---
    let g2 = qr_givens_quaternion(b_mat.x_axis.x, b_mat.x_axis.z);
    let a2 = -2.0 * g2.sin_theta * g2.sin_theta + 1.0; // a2 = cos(theta2)
    let b2 = 2.0 * g2.cos_theta * g2.sin_theta; // b2 = sin(theta2)

    // Apply Q2.T to (Q1.T * B)
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

    // --- Third Givens rotation to zero out b[2][1] (affects rows 1 and 2) ---
    let g3 = qr_givens_quaternion(b_mat.y_axis.y, b_mat.y_axis.z);
    let a3 = -2.0 * g3.sin_theta * g3.sin_theta + 1.0; // a3 = cos(theta3)
    let b3 = 2.0 * g3.cos_theta * g3.sin_theta; // b3 = sin(theta3)

    // Apply Q3.T to (Q2.T * Q1.T * B)
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

    // --- Construct Q = Q1 * Q2 * Q3 ---
    // Q1 = (Q1.T).T
    let q1 = Mat3::from_cols(Vec3::new(a1, b1, 0.0), Vec3::new(-b1, a1, 0.0), Vec3::Z);

    // Q2 = (Q2.T).T
    let q2 = Mat3::from_cols(Vec3::new(a2, 0.0, b2), Vec3::Y, Vec3::new(-b2, 0.0, a2));

    // Q3 = (Q3.T).T
    let q3 = Mat3::from_cols(Vec3::X, Vec3::new(0.0, a3, b3), Vec3::new(0.0, -b3, a3));

    let q = q1 * q2 * q3;
    QR3 { q, r }
}

/// Wrapping function used to contain all of the required sub calls
pub fn svd3(a: &Mat3) -> SVD3Set {
    // Compute the eigenvectors of A^T * A, which is V in SVD (right singular vectors)
    let mut v = jacobi_eigenanalysis(Symmetric3x3::from_mat3x3(&(a.transpose().mul_mat3(a))));
    // Compute B = A * V
    let mut b = a.mul_mat3(&v);

    // Sort the singular values
    sort_singular_values(&mut b, &mut v);

    // Perform QR decomposition on B to get Q and R
    let qr = qr_decomposition(&mut b);

    let mut u = qr.q;
    let mut s = qr.r;

    let cond_x = s.x_axis.x < 0.0;
    let cond_y = s.y_axis.y < 0.0;
    let cond_z = s.z_axis.z < 0.0;

    cond_negate_vec3(cond_x, &mut u.x_axis);
    cond_negate_vec3(cond_y, &mut u.y_axis);
    cond_negate_vec3(cond_z, &mut u.z_axis);

    s.x_axis.x = s.x_axis.x.abs();
    s.y_axis.y = s.y_axis.y.abs();
    s.z_axis.z = s.z_axis.z.abs();

    // Return the SVD result
    SVD3Set { u, s, v }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Mat3, Vec3};

    /// Helper function to validate all critical SVD properties
    fn verify_svd_properties(a: &Mat3, svd: &SVD3Set, epsilon: f32) {
        let u = svd.u;
        let s = svd.s;
        let v = svd.v;

        // Property 1: Reconstruction (A = U * S * V.T)
        let reconstruction = u * s * v.transpose();
        let reconstruction_epsilon = 1e-4;
        assert!(
            a.abs_diff_eq(reconstruction, reconstruction_epsilon),
            "Reconstruction failed: A != U*S*V.T\nA:\n{}\nReconstruction:\n{}",
            a,
            reconstruction
        );

        // Property 2: U is Orthogonal (U.T * U = I)
        let u_t_u = u.transpose() * u;
        assert!(
            Mat3::IDENTITY.abs_diff_eq(u_t_u, epsilon),
            "U is not orthogonal: U.T*U != I\nU.T*U:\n{}",
            u_t_u
        );

        // Property 3: V is Orthogonal (V.T * V = I)
        let v_t_v = v.transpose() * v;
        assert!(
            Mat3::IDENTITY.abs_diff_eq(v_t_v, epsilon),
            "V is not orthogonal: V.T*V != I\nV.T*V:\n{}",
            v_t_v
        );

        // Property 4: S is Diagonal
        let s_diag = Vec3::new(s.x_axis.x, s.y_axis.y, s.z_axis.z);
        assert!(
            s_diag.x >= 0.0 && s_diag.y >= 0.0 && s_diag.z >= 0.0,
            "Singular values are not non-negative: {:?}",
            s_diag
        );
        assert!(
            s_diag.x >= s_diag.y - epsilon && s_diag.y >= s_diag.z - epsilon,
            "Singular values are not sorted: {:?}",
            s_diag
        );
    }

    #[test]
    fn test_svd3_1_diagonal_sorted() {
        let a = Mat3::from_diagonal(Vec3::new(3.0, 2.0, 1.0));
        let svd_result = svd3(&a);
        verify_svd_properties(&a, &svd_result, SVD3_EPSILON);

        let s_diag = Vec3::new(
            svd_result.s.x_axis.x,
            svd_result.s.y_axis.y,
            svd_result.s.z_axis.z,
        );
        assert!(s_diag.abs_diff_eq(Vec3::new(3.0, 2.0, 1.0), SVD3_EPSILON));
    }

    #[test]
    fn test_svd3_2_zero() {
        let a = Mat3::ZERO;
        let svd_result = svd3(&a);
        verify_svd_properties(&a, &svd_result, SVD3_EPSILON);
        assert!(svd_result.s.abs_diff_eq(Mat3::ZERO, SVD3_EPSILON));
    }

    #[test]
    fn test_svd3_3_identity() {
        let a = Mat3::IDENTITY;
        let svd_result = svd3(&a);
        verify_svd_properties(&a, &svd_result, SVD3_EPSILON);
        assert!(svd_result.s.abs_diff_eq(Mat3::IDENTITY, SVD3_EPSILON));
    }

    #[test]
    fn test_svd3_4_singular_rank1() {
        let a = Mat3 {
            x_axis: Vec3::new(1.0, 2.0, 3.0),
            y_axis: Vec3::new(2.0, 4.0, 6.0),
            z_axis: Vec3::new(3.0, 6.0, 9.0),
        };
        let svd_result = svd3(&a);
        verify_svd_properties(&a, &svd_result, SVD3_EPSILON);

        let s_diag = Vec3::new(
            svd_result.s.x_axis.x,
            svd_result.s.y_axis.y,
            svd_result.s.z_axis.z,
        );
        assert!(s_diag.x > SVD3_EPSILON);
        assert!(s_diag.y.abs() < SVD3_EPSILON);
        assert!(s_diag.z.abs() < SVD3_EPSILON);
    }

    #[test]
    fn test_svd3_diagonal_unsorted() {
        let a = Mat3::from_diagonal(Vec3::new(2.0, 3.0, 1.0));
        let svd_result = svd3(&a);
        verify_svd_properties(&a, &svd_result, SVD3_EPSILON);
        let s_diag = Vec3::new(
            svd_result.s.x_axis.x,
            svd_result.s.y_axis.y,
            svd_result.s.z_axis.z,
        );
        assert!(s_diag.abs_diff_eq(Vec3::new(3.0, 2.0, 1.0), SVD3_EPSILON));
    }

    #[test]
    fn test_svd3_rotation_matrix() {
        let a = Mat3::from_rotation_y(std::f32::consts::FRAC_PI_4);
        let svd_result = svd3(&a);
        verify_svd_properties(&a, &svd_result, SVD3_EPSILON);

        let s_diag = Vec3::new(
            svd_result.s.x_axis.x,
            svd_result.s.y_axis.y,
            svd_result.s.z_axis.z,
        );
        assert!(s_diag.abs_diff_eq(Vec3::ONE, SVD3_EPSILON));
    }

    #[test]
    fn test_svd3_reflection_matrix() {
        let a = Mat3::from_diagonal(Vec3::new(1.0, -1.0, 1.0));
        let svd_result = svd3(&a);
        verify_svd_properties(&a, &svd_result, SVD3_EPSILON);

        let s_diag = Vec3::new(
            svd_result.s.x_axis.x,
            svd_result.s.y_axis.y,
            svd_result.s.z_axis.z,
        );
        assert!(s_diag.abs_diff_eq(Vec3::ONE, SVD3_EPSILON));
    }

    #[test]
    fn test_svd3_general_full_rank() {
        let a = Mat3::from_cols(
            Vec3::new(1.0, 4.0, 7.0),
            Vec3::new(2.0, 5.0, 8.0),
            Vec3::new(3.0, 6.0, 10.0),
        );
        let svd_result = svd3(&a);
        verify_svd_properties(&a, &svd_result, SVD3_EPSILON);
        let s_diag = Vec3::new(
            svd_result.s.x_axis.x,
            svd_result.s.y_axis.y,
            svd_result.s.z_axis.z,
        );
        assert!(s_diag.min_element() > SVD3_EPSILON);
    }

    #[test]
    fn test_svd3_singular_rank2() {
        let a = Mat3::from_cols(
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(4.0, 5.0, 6.0),
            Vec3::new(5.0, 7.0, 9.0), // c0 + c1
        );
        let svd_result = svd3(&a);
        verify_svd_properties(&a, &svd_result, SVD3_EPSILON);

        let s_diag = Vec3::new(
            svd_result.s.x_axis.x,
            svd_result.s.y_axis.y,
            svd_result.s.z_axis.z,
        );
        assert!(s_diag.x > SVD3_EPSILON);
        assert!(s_diag.y > SVD3_EPSILON);
        assert!(s_diag.z.abs() < SVD3_EPSILON);
    }
}
