// Reference: https://github.com/wi-re/tbtSVD/blob/master/source/SVD.h
use glam::{Mat3, Quat, Vec3};
const GAMMA: f32 = 5.828_427_3;
const CSTAR: f32 = 0.923_879_5;
const SSTAR: f32 = 0.382_683_43;
const SVD3_EPSILON: f32 = 1e-6;
const MAX_SWEEPS: usize = 5;

#[derive(Debug, Clone)]
/// A simple symmetric 3x3 Matrix class (contains no storage for (0, 1) (0, 2) and (1, 2)
struct Symmetric3x3 {
    /// The element at row 0, column 0 of the matrix, typically the first diagonal element.
    m_00: f32,

    /// The element at row 1, column 0 of the matrix. Since this is a symmetric matrix, it is equivalent to `m_01`.
    m_10: f32,

    /// The element at row 1, column 1 of the matrix, the second diagonal element.
    m_11: f32,

    /// The element at row 2, column 0 of the matrix. Since this is a symmetric matrix, it is equivalent to `m_02`.
    m_20: f32,

    /// The element at row 2, column 1 of the matrix. Since this is a symmetric matrix, it is equivalent to `m_12`.
    m_21: f32,

    /// The element at row 2, column 2 of the matrix, the third diagonal element.
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
    ch: f32,

    /// The sine of the angle in the Givens rotation.
    sh: f32,
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
/// this implementation dosent follow that paper exactly, as the compelete qr decomposion is redundant for svd calculation
#[inline(always)]
fn approximate_givens_parameters(s_pp: f32, s_qq: f32, s_pq: f32) -> Givens {
    let ch_val = 2.0 * (s_pp - s_qq);
    let sh_val = s_pq;
    let ch2 = ch_val * ch_val;
    let sh2 = sh_val * sh_val;

    if GAMMA * sh2 < ch2 {
        let w = 1.0 / ((ch2 + sh2).sqrt());
        Givens {
            ch: w * ch_val,
            sh: w * sh_val,
        }
    } else {
        Givens {
            ch: CSTAR,
            sh: SSTAR,
        }
    }
}

#[inline(always)]
fn conjugate_xy(s: &mut Symmetric3x3, q: &mut Quat) {
    // Compute Givens rotation parameters
    let mut g = approximate_givens_parameters(s.m_00, s.m_11, s.m_10);

    let ch2 = g.ch * g.ch;
    let sh2 = g.sh * g.sh;
    let scale = 1.0 / (ch2 + sh2);
    let a = (ch2 - sh2) * scale;
    let b = 2.0 * g.sh * g.ch * scale;

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

    let tmp_x = q.x * g.sh;
    let tmp_y = q.y * g.sh;
    let tmp_z = q.z * g.sh;
    g.sh *= q.w;

    q.z = q.z * g.ch + g.sh;
    q.w = q.w * g.ch - tmp_z;
    q.x = q.x * g.ch + tmp_y;
    q.y = q.y * g.ch - tmp_x;
}

#[inline(always)]
fn conjugate_yz(s: &mut Symmetric3x3, q: &mut Quat) {
    // Compute Givens rotation parameters
    let mut g = approximate_givens_parameters(s.m_11, s.m_22, s.m_21);

    // Calculate rotation matrix elements 'a' and 'b'
    let ch2 = g.ch * g.ch;
    let sh2 = g.sh * g.sh;
    let scale = 1.0 / (ch2 + sh2);
    let a = (ch2 - sh2) * scale;
    let b = 2.0 * g.sh * g.ch * scale;

    // Cache original matrix elements
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

    // Update the cumulative rotation
    let tmp_x = q.x * g.sh;
    let tmp_y = q.y * g.sh;
    let tmp_z = q.z * g.sh;
    g.sh *= q.w;

    q.x = q.x * g.ch + g.sh;
    q.w = q.w * g.ch - tmp_x;
    q.y = q.y * g.ch + tmp_z;
    q.z = q.z * g.ch - tmp_y;
}

#[inline(always)]
fn conjugate_xz(s: &mut Symmetric3x3, q: &mut Quat) {
    // Compute Givens rotation parameters
    let mut g = approximate_givens_parameters(s.m_00, s.m_22, s.m_20);

    // Calculate rotation matrix elements 'a' and 'b'
    let ch2 = g.ch * g.ch;
    let sh2 = g.sh * g.sh;
    let scale = 1.0 / (ch2 + sh2);
    let a = (ch2 - sh2) * scale;
    let b = 2.0 * g.sh * g.ch * scale;

    // Cache original matrix elements
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

    // Update the cumulative rotation
    let tmp_x = q.x * g.sh;
    let tmp_y = q.y * g.sh;
    let tmp_z = q.z * g.sh;
    g.sh *= q.w;

    q.y = q.y * g.ch + g.sh;
    q.w = q.w * g.ch - tmp_y;
    q.z = q.z * g.ch + tmp_x;
    q.x = q.x * g.ch - tmp_z;
}

#[inline(always)]
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
    if c {
        *v = -*v;
    }
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

/// Wrapping function used to contain all of the required sub calls
pub fn svd3(a: &Mat3) -> SVD3Set {
    // Compute the eigenvectors of A^T * A, which is V in SVD (right singular vectors)
    let mut v = jacobi_eigenanalysis(Symmetric3x3::from_mat3x3(&(a.transpose().mul_mat3(a))));
    // Compute B = A * V
    let mut b = a.mul_mat3(&v);

    // Sort the singular values
    sort_singular_values(&mut b, &mut v);

    let s1 = b.x_axis.length();
    let s2 = b.y_axis.length();
    let s3 = b.z_axis.length();

    // Create the diagonal singular value matrix S
    let mut s = Mat3::from_diagonal(Vec3::new(s1, s2, s3));

    // Calculate inverse-length for normalization.
    // Handle potential division by zero if s_i is tiny.
    let s1_inv = if s1.abs() < SVD3_EPSILON {
        0.0
    } else {
        1.0 / s1
    };
    let s2_inv = if s2.abs() < SVD3_EPSILON {
        0.0
    } else {
        1.0 / s2
    };
    let s3_inv = if s3.abs() < SVD3_EPSILON {
        0.0
    } else {
        1.0 / s3
    };

    // Get U by normalizing the columns of B (U_col_i = B_col_i / s_i).
    let mut u = Mat3::from_cols(b.x_axis * s1_inv, b.y_axis * s2_inv, b.z_axis * s3_inv);

    // Ensure U is a proper rotation (det(U) = +1).
    // If not, flip the sign of the column associated with the smallest singular value.
    if u.determinant() < 0.0 {
        u.z_axis = -u.z_axis;
        s.z_axis.z = -s.z_axis.z; // Also flip the singular value
    }

    // Return the SVD result
    SVD3Set { u, s, v }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::*;

    #[test]
    fn test_svd3_1() {
        // Define a simple 3x3 matrix A
        let a = Mat3 {
            x_axis: Vec3::new(1.0, 0.0, 0.0),
            y_axis: Vec3::new(0.0, 2.0, 0.0),
            z_axis: Vec3::new(0.0, 0.0, 3.0),
        };

        // Perform SVD on matrix A
        let svd_result = svd3(&a);
        let _ = a.abs_diff_eq(
            svd_result
                .u
                .mul_mat3(&(svd_result.s.mul_mat3(&svd_result.v.transpose()))),
            SVD3_EPSILON,
        );
    }

    #[test]
    fn test_svd3_2() {
        // Define a Zero Matrix 3x3 matrix A
        let a = Mat3 {
            x_axis: Vec3::new(0.0, 0.0, 0.0),
            y_axis: Vec3::new(0.0, 0.0, 0.0),
            z_axis: Vec3::new(0.0, 0.0, 0.0),
        };

        // Perform SVD on matrix A
        let svd_result = svd3(&a);
        let _ = a.abs_diff_eq(
            svd_result
                .u
                .mul_mat3(&(svd_result.s.mul_mat3(&svd_result.v.transpose()))),
            SVD3_EPSILON,
        );
    }

    #[test]
    fn test_svd3_3() {
        // Define a Identity Matrix 3x3 matrix A
        let a = Mat3 {
            x_axis: Vec3::new(1.0, 0.0, 0.0),
            y_axis: Vec3::new(0.0, 1.0, 0.0),
            z_axis: Vec3::new(0.0, 0.0, 1.0),
        };

        // Perform SVD on matrix A
        let svd_result = svd3(&a);
        let _ = a.abs_diff_eq(
            svd_result
                .u
                .mul_mat3(&(svd_result.s.mul_mat3(&svd_result.v.transpose()))),
            SVD3_EPSILON,
        );
    }

    #[test]
    fn test_svd3_4() {
        // Define a Singular Matrix 3x3 matrix A
        let a = Mat3 {
            x_axis: Vec3::new(1.0, 2.0, 3.0),
            y_axis: Vec3::new(2.0, 4.0, 6.0),
            z_axis: Vec3::new(3.0, 6.0, 9.0),
        };

        // Perform SVD on matrix A
        let svd_result = svd3(&a);
        let _ = a.abs_diff_eq(
            svd_result
                .u
                .mul_mat3(&(svd_result.s.mul_mat3(&svd_result.v.transpose()))),
            SVD3_EPSILON,
        );
    }
}
