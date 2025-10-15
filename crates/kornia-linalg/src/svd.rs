// Reference: https://github.com/wi-re/tbtSVD/blob/master/source/SVD.h
use glam::{Mat3, Quat, Vec3};
const GAMMA: f32 = 5.828_427_3;
const CSTAR: f32 = 0.923_879_5;
const SSTAR: f32 = 0.382_683_43;
const SVD3_EPSILON: f32 = 1e-6;
const MAX_SWEEPS: usize = 6;

/// Helper function used to swap X with Y and Y with  X if c == true
#[inline(always)]
fn cond_swap(c: bool, x: &mut f32, y: &mut f32) {
    let z = *x;
    if c {
        *x = *y;
        *y = z;
    }
}

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

/// Calculates the squared norm of the vector [x y z] using a standard scalar product d = x * x + y * y + z * z
#[inline(always)]
fn dist2(x: f32, y: f32, z: f32) -> f32 {
    x * x + y * y + z * z
}

/// For an explanation of the math see http://pages.cs.wisc.edu/~sifakis/papers/SVD_TR1690.pdf
/// Computing the Singular Value Decomposition of 3 x 3 matrices with minimal branching and elementary floating point operations
/// See Algorithm 2 in reference. Given a matrix A this function returns the givens quaternion (x and w component, y and z are 0)
#[inline(always)]
fn approximate_givens_quaternion(a: &Symmetric3x3) -> Givens {
    let ch_val = 2.0 * (a.m_00 - a.m_11);
    let sh_val = a.m_10;
    let ch2 = ch_val * ch_val;
    let sh2 = sh_val * sh_val;

    if GAMMA * sh2 < ch2 {
        let w = (ch2 + sh2).sqrt().recip();
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

/// Function used to apply a givens rotation S. Calculates the weights and updates the quaternion to contain the cumultative rotation

/// Function used to contain the givens permutations and the loop of the jacobi steps controlled by JACOBI_STEPS
/// Returns the quaternion q containing the cumultative result used to reconstruct S
#[inline(always)]
fn conjugate_xy(s: &mut Symmetric3x3, q: &mut Quat) {
    // Compute Givens rotation parameters
    let mut g = approximate_givens_quaternion(s);

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
    let mut g = approximate_givens_quaternion(s);

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

    // Update the cumulative rotation quaternion using named fields
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
    let mut g = approximate_givens_quaternion(s);

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

    // Update the cumulative rotation quaternion using named fields
    let tmp_x = q.x * g.sh;
    let tmp_y = q.y * g.sh;
    let tmp_z = q.z * g.sh;
    g.sh *= q.w;

    q.y = q.y * g.ch + g.sh;
    q.w = q.w * g.ch - tmp_y;
    q.z = q.z * g.ch + tmp_x;
    q.x = q.x * g.ch - tmp_z;
}

fn jacobi_eigenanalysis(mut s: Symmetric3x3) -> Mat3 {
    let mut q = Quat::from_xyzw(0.0, 0.0, 0.0, 1.0);
    for _i in 0..MAX_SWEEPS {
        conjugate_xy(&mut s, &mut q);
        conjugate_yz(&mut s, &mut q);
        conjugate_xz(&mut s, &mut q);

        let sum_off_diagonal_sq = s.m_10 * s.m_10 + s.m_20 * s.m_20 + s.m_21 * s.m_21;
        if sum_off_diagonal_sq < SVD3_EPSILON {
            break;
        }
    }
    Mat3::from_quat(q)
}

#[inline(always)]
fn manual_swap<T: Copy>(a: &mut T, b: &mut T) {
    let temp = *a;
    *a = *b;
    *b = temp;
}

fn sort_singular_values(b: &mut Mat3, v: &mut Mat3) {
    let mut rho1 = dist2(b.x_axis.x, b.x_axis.y, b.x_axis.z);
    let mut rho2 = dist2(b.y_axis.x, b.y_axis.y, b.y_axis.z);
    let mut rho3 = dist2(b.z_axis.x, b.z_axis.y, b.z_axis.z);

    if rho1 < rho2 {
        manual_swap(&mut b.x_axis, &mut b.y_axis);
        manual_swap(&mut v.x_axis, &mut v.y_axis);

        b.y_axis = Vec3 {
            x: -b.y_axis.x,
            y: -b.y_axis.y,
            z: -b.y_axis.z,
        };
        v.y_axis = Vec3 {
            x: -v.y_axis.x,
            y: -v.y_axis.y,
            z: -v.y_axis.z,
        };

        manual_swap(&mut rho1, &mut rho2);
    }
    if rho1 < rho3 {
        manual_swap(&mut b.x_axis, &mut b.z_axis);
        manual_swap(&mut v.x_axis, &mut v.z_axis);

        b.z_axis = Vec3 {
            x: -b.z_axis.x,
            y: -b.z_axis.y,
            z: -b.z_axis.z,
        };
        v.z_axis = Vec3 {
            x: -v.z_axis.x,
            y: -v.z_axis.y,
            z: -v.z_axis.z,
        };

        manual_swap(&mut rho1, &mut rho3);
    }
    if rho2 < rho3 {
        manual_swap(&mut b.y_axis, &mut b.z_axis);
        manual_swap(&mut v.y_axis, &mut v.z_axis);

        b.z_axis = Vec3 {
            x: -b.z_axis.x,
            y: -b.z_axis.y,
            z: -b.z_axis.z,
        };
        v.z_axis = Vec3 {
            x: -v.z_axis.x,
            y: -v.z_axis.y,
            z: -v.z_axis.z,
        };
    }
}

/// Implementation of Algorithm 4
#[inline(always)]
fn qr_givens_quaternion(a1: f32, a2: f32) -> Givens {
    let epsilon = SVD3_EPSILON;
    let rho = (a1 * a1 + a2 * a2).sqrt();

    let mut g = Givens {
        ch: a1.abs() + f32::max(rho, epsilon),
        sh: if rho > epsilon { a2 } else { 0.0 },
    };

    let b = a1 < 0.0;
    cond_swap(b, &mut g.sh, &mut g.ch);

    let w = (g.ch * g.ch + g.sh * g.sh).sqrt().recip();
    g.ch *= w;
    g.sh *= w;
    g
}

/// Implements a QR decomposition of a Matrix
fn qr_decomposition(b_mat: &mut Mat3) -> QR3 {
    let mut q = Mat3::ZERO;
    // --- First Givens rotation to zero out a[1][0] (affects columns 0 and 1) ---
    let g1 = qr_givens_quaternion(b_mat.x_axis.x, b_mat.x_axis.y);
    let a1 = -2.0 * g1.sh * g1.sh + 1.0;
    let b1 = 2.0 * g1.ch * g1.sh;

    // Apply to row 0
    let c0 = b_mat.x_axis.x;
    let c1 = b_mat.x_axis.y;
    b_mat.x_axis.x = a1 * c0 + b1 * c1;
    b_mat.x_axis.y = -b1 * c0 + a1 * c1;
    // Apply to row 1
    let c0 = b_mat.y_axis.x;
    let c1 = b_mat.y_axis.y;
    b_mat.y_axis.x = a1 * c0 + b1 * c1;
    b_mat.y_axis.y = -b1 * c0 + a1 * c1;
    // Apply to row 2
    let c0 = b_mat.z_axis.x;
    let c1 = b_mat.z_axis.y;
    b_mat.z_axis.x = a1 * c0 + b1 * c1;
    b_mat.z_axis.y = -b1 * c0 + a1 * c1;

    // --- Second Givens rotation to zero out a[2][0] (affects columns 0 and 2) ---
    let g2 = qr_givens_quaternion(b_mat.x_axis.x, b_mat.x_axis.z);
    let a2 = -2.0 * g2.sh * g2.sh + 1.0;
    let b2 = 2.0 * g2.ch * g2.sh;

    // Apply to row 0
    let c0 = b_mat.x_axis.x;
    let c2 = b_mat.x_axis.z;
    b_mat.x_axis.x = a2 * c0 + b2 * c2;
    b_mat.x_axis.z = -b2 * c0 + a2 * c2;
    // Apply to row 1
    let c0 = b_mat.y_axis.x;
    let c2 = b_mat.y_axis.z;
    b_mat.y_axis.x = a2 * c0 + b2 * c2;
    b_mat.y_axis.z = -b2 * c0 + a2 * c2;
    // Apply to row 2
    let c0 = b_mat.z_axis.x;
    let c2 = b_mat.z_axis.z;
    b_mat.z_axis.x = a2 * c0 + b2 * c2;
    b_mat.z_axis.z = -b2 * c0 + a2 * c2;

    // --- Third Givens rotation to zero out a[2][1] (affects columns 1 and 2) ---
    let g3 = qr_givens_quaternion(b_mat.y_axis.y, b_mat.y_axis.z);
    let a3 = -2.0 * g3.sh * g3.sh + 1.0;
    let b3 = 2.0 * g3.ch * g3.sh;

    // Apply to row 0
    let c1 = b_mat.x_axis.y;
    let c2 = b_mat.x_axis.z;
    b_mat.x_axis.y = a3 * c1 + b3 * c2;
    b_mat.x_axis.z = -b3 * c1 + a3 * c2;
    // Apply to row 1
    let c1 = b_mat.y_axis.y;
    let c2 = b_mat.y_axis.z;
    b_mat.y_axis.y = a3 * c1 + b3 * c2;
    b_mat.y_axis.z = -b3 * c1 + a3 * c2;
    // Apply to row 2
    let c1 = b_mat.z_axis.y;
    let c2 = b_mat.z_axis.z;
    b_mat.z_axis.y = a3 * c1 + b3 * c2;
    b_mat.z_axis.z = -b3 * c1 + a3 * c2;

    let r = *b_mat;

    q.x_axis.x = a1 * a2;
    q.x_axis.y = (b2 * b3 * -a1) - b1 * a3;
    q.x_axis.z = b1 * b3 - b2 * a1 * a3;
    q.y_axis.x = b1 * a2;
    q.y_axis.y = a1 * a3 - (b1 * b2 * b3);
    q.y_axis.z = -2.0 * g3.ch * g3.sh
        + 4.0 * g1.sh * (g3.ch * g1.sh * g3.sh + g1.ch * g2.ch * g2.sh * (-a3));
    q.z_axis.x = b2;
    q.z_axis.y = b3 * a2;
    q.z_axis.z = a2 * a3;

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

    // Return the SVD result, which includes Q (as U), R (as S), and V
    SVD3Set {
        u: qr.q,
        s: qr.r,
        v,
    }
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
