// Reference: https://github.com/wi-re/tbtSVD/blob/master/source/SVD.h
use glam::{Mat3, Quat};
use std::ops::{Index, IndexMut};
const GAMMA: f32 = 5.828_427_3;
const CSTAR: f32 = 0.923_879_5;
const SSTAR: f32 = 0.382_683_43;
const SVD3_EPSILON: f32 = 1e-6;
const JACOBI_STEPS: u8 = 6;
const RSQRT1_STEPS: u8 = 6;

/// Standard CPU division.
fn fdiv(x: f32, y: f32) -> f32 {
    x / y
}

/// Calculates the reciprocal square root of x using a fast approximation.
fn rsqrt(x: f32) -> f32 {
    let mut i: i32 = x.to_bits() as i32;
    i = 0x5F375A86_i32.wrapping_sub(i >> 1);
    let y = f32::from_bits(i as u32);
    y * (1.5 - (x * 0.5 * y * y))
}

/// Uses RSQRT1_STEPS to offer a higher precision alternative
fn rsqrt1(x: f32) -> f32 {
    let xhalf = -0.5 * x;
    let i: i32 = x.to_bits() as i32;
    let i = 0x5f37599e_i32.wrapping_sub(i >> 1);
    let mut x: f32 = f32::from_bits(i as u32);

    for _ in 0..RSQRT1_STEPS {
        x = x * (1.5 + xhalf * x * x);
    }

    x
}

/// Calculates the square root of x using 1.f/rsqrt1(x)to give a square root with controllable and consistent precision.
fn accurate_sqrt(x: f32) -> f32 {
    fdiv(1.0, rsqrt1(x))
}

/// Helper function used to swap X with Y and Y with  X if c == true
fn cond_swap(c: bool, x: &mut f32, y: &mut f32) {
    let z = *x;
    if c {
        *x = *y;
        *y = z;
    }
}

// Helper function to swap X and Y and swap Y with -X if c is true
fn cond_neg_swap(c: bool, x: &mut f32, y: &mut f32) {
    let z = -(*x);
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
fn dist2(x: f32, y: f32, z: f32) -> f32 {
    x * x + y * y + z * z
}

/// For an explanation of the math see http://pages.cs.wisc.edu/~sifakis/papers/SVD_TR1690.pdf
/// Computing the Singular Value Decomposition of 3 x 3 matrices with minimal branching and elementary floating point operations
/// See Algorithm 2 in reference. Given a matrix A this function returns the givens quaternion (x and w component, y and z are 0)
fn approximate_givens_quaternion(a: &Symmetric3x3) -> Givens {
    let g = Givens {
        ch: 2.0 * (a.m_00 - a.m_11),
        sh: a.m_10,
    };
    let ch2 = g.ch * g.ch;
    let sh2 = g.sh * g.sh;
    let mut b = GAMMA * sh2 < ch2;
    let w = rsqrt(ch2 + sh2);

    if w.is_nan() {
        // Checking for NaN
        b = false;
    }

    Givens {
        ch: if b { w * g.ch } else { CSTAR },
        sh: if b { w * g.sh } else { SSTAR },
    }
}

#[derive(Debug)]
/// A wrapper around the `glam::Quat` type that allows dynamic indexing into its components.
///
/// This struct provides custom indexing behavior for quaternion components (`x`, `y`, `z`, and `w`),
/// enabling access and mutation using an index (e.g., `q[0]`, `q[1]`, etc.). It implements both
/// the `Index` and `IndexMut` traits to allow for immutable and mutable access to the quaternion's components.
struct IndexedQuat(Quat);

impl IndexedQuat {
    fn new(q: Quat) -> Self {
        IndexedQuat(q)
    }

    fn to_quat(&self) -> Quat {
        self.0
    }
}

impl Index<usize> for IndexedQuat {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.0.x,
            1 => &self.0.y,
            2 => &self.0.z,
            3 => &self.0.w,
            _ => panic!("Index out of bounds for Quaternion: {}", index),
        }
    }
}

impl IndexMut<usize> for IndexedQuat {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.0.x,
            1 => &mut self.0.y,
            2 => &mut self.0.z,
            3 => &mut self.0.w,
            _ => panic!("Index out of bounds for Quaternion: {}", index),
        }
    }
}

/// Function used to apply a givens rotation S. Calculates the weights and updates the quaternion to contain the cumultative rotation
fn jacobi_conjugation(x: usize, y: usize, z: usize, s: &mut Symmetric3x3, q: &mut IndexedQuat) {
    // Compute the Givens rotation (approximated)
    let mut g = approximate_givens_quaternion(s);
    // Scale and calculate intermediate values
    let ch2 = g.ch * g.ch;
    let sh2 = g.sh * g.sh;
    let scale = 1.0 / (ch2 + sh2);
    let a = (ch2 - sh2) * scale;
    let b = 2.0 * g.sh * g.ch * scale;

    // Create a copy of the matrix to avoid modifying the original during calculations
    let mut _s = s.clone();

    // Perform conjugation: S = Q'*S*Q
    s.m_00 = a * (a * _s.m_00 + b * _s.m_10) + b * (a * _s.m_10 + b * _s.m_11);
    s.m_10 = a * (-b * _s.m_00 + a * _s.m_10) + b * (-b * _s.m_10 + a * _s.m_11);
    s.m_11 = -b * (-b * _s.m_00 + a * _s.m_10) + a * (-b * _s.m_10 + a * _s.m_11);
    s.m_20 = a * _s.m_20 + b * _s.m_21;
    s.m_21 = -b * _s.m_20 + a * _s.m_21;
    s.m_22 = _s.m_22;

    // Update cumulative rotation qV
    let mut tmp = [0.0, 0.0, 0.0];
    tmp[0] = q[0] * g.sh;
    tmp[1] = q[1] * g.sh;
    tmp[2] = q[2] * g.sh;
    g.sh *= q[3];

    // (x, y, z) corresponds to (0,1,2), (1,2,0), (2,0,1) for (p, q) = (0,1), (1,2), (0,2)
    q[z] = q[z] * g.ch + g.sh;
    q[3] = q[3] * g.ch - tmp[z]; // w
    q[x] = q[x] * g.ch + tmp[y];
    q[y] = q[y] * g.ch - tmp[x];

    // Re-arrange matrix for next iteration
    _s.m_00 = s.m_11;
    _s.m_10 = s.m_21;
    _s.m_11 = s.m_22;
    _s.m_20 = s.m_10;
    _s.m_21 = s.m_20;
    _s.m_22 = s.m_00;

    s.m_00 = _s.m_00;
    s.m_10 = _s.m_10;
    s.m_11 = _s.m_11;
    s.m_20 = _s.m_20;
    s.m_21 = _s.m_21;
    s.m_22 = _s.m_22;
}

/// Function used to contain the givens permutations and the loop of the jacobi steps controlled by JACOBI_STEPS
/// Returns the quaternion q containing the cumultative result used to reconstruct S
fn jacobi_eigenanalysis(mut s: Symmetric3x3) -> Mat3 {
    let mut q = IndexedQuat::new(Quat::from_xyzw(0.0, 0.0, 0.0, 1.0));
    for _i in 0..JACOBI_STEPS {
        jacobi_conjugation(0, 1, 2, &mut s, &mut q);
        jacobi_conjugation(1, 2, 0, &mut s, &mut q);
        jacobi_conjugation(2, 0, 1, &mut s, &mut q);
    }

    Mat3::from_quat(q.to_quat())
}

/// Implementation of Algorithm 3
fn sort_singular_values(b: &mut Mat3, v: &mut Mat3) {
    let mut rho1 = dist2(b.x_axis.x, b.x_axis.y, b.x_axis.z);
    let mut rho2 = dist2(b.y_axis.x, b.y_axis.y, b.y_axis.z);
    let mut rho3 = dist2(b.z_axis.x, b.z_axis.y, b.z_axis.z);

    let mut c = rho1 < rho2;
    cond_neg_swap(c, &mut b.x_axis.x, &mut b.y_axis.x);
    cond_neg_swap(c, &mut v.x_axis.x, &mut v.y_axis.x);
    cond_neg_swap(c, &mut b.x_axis.y, &mut b.y_axis.y);
    cond_neg_swap(c, &mut v.x_axis.y, &mut v.y_axis.y);
    cond_neg_swap(c, &mut b.x_axis.z, &mut b.y_axis.z);
    cond_neg_swap(c, &mut v.x_axis.z, &mut v.y_axis.z);
    cond_swap(c, &mut rho1, &mut rho2);

    c = rho1 < rho3;
    cond_neg_swap(c, &mut b.x_axis.x, &mut b.z_axis.x);
    cond_neg_swap(c, &mut v.x_axis.x, &mut v.z_axis.x);
    cond_neg_swap(c, &mut b.x_axis.y, &mut b.z_axis.y);
    cond_neg_swap(c, &mut v.x_axis.y, &mut v.z_axis.y);
    cond_neg_swap(c, &mut b.x_axis.z, &mut b.z_axis.z);
    cond_neg_swap(c, &mut v.x_axis.z, &mut v.z_axis.z);
    cond_swap(c, &mut rho1, &mut rho3);

    c = rho2 < rho3;
    cond_neg_swap(c, &mut b.y_axis.x, &mut b.z_axis.x);
    cond_neg_swap(c, &mut v.y_axis.x, &mut v.z_axis.x);
    cond_neg_swap(c, &mut b.y_axis.y, &mut b.z_axis.y);
    cond_neg_swap(c, &mut v.y_axis.y, &mut v.z_axis.y);
    cond_neg_swap(c, &mut b.y_axis.z, &mut b.z_axis.z);
    cond_neg_swap(c, &mut v.y_axis.z, &mut v.z_axis.z);
}

/// Implementation of Algorithm 4
fn qr_givens_quaternion(a1: f32, a2: f32) -> Givens {
    let epsilon = SVD3_EPSILON;
    let rho = accurate_sqrt(a1 * a1 + a2 * a2);

    let mut g = Givens {
        ch: a1.abs() + f32::max(rho, epsilon),
        sh: if rho > epsilon { a2 } else { 0.0 },
    };

    let b = a1 < 0.0;
    cond_swap(b, &mut g.sh, &mut g.ch);

    let w = rsqrt(g.ch * g.ch + g.sh * g.sh);
    g.ch *= w;
    g.sh *= w;
    g
}

/// Implements a QR decomposition of a Matrix
fn qr_decomposition(b_mat: &mut Mat3) -> QR3 {
    let mut q = Mat3::ZERO;
    let mut r = Mat3::ZERO;

    // First Givens rotation (ch, 0, 0, sh)
    let g1 = qr_givens_quaternion(b_mat.x_axis.x, b_mat.x_axis.y);
    let mut a = -2.0 * g1.sh * g1.sh + 1.0;
    let mut b = 2.0 * g1.ch * g1.sh;

    // Apply B = Q' * B
    r.x_axis.x = a * b_mat.x_axis.x + b * b_mat.x_axis.y;
    r.y_axis.x = a * b_mat.y_axis.x + b * b_mat.y_axis.y;
    r.z_axis.x = a * b_mat.z_axis.x + b * b_mat.z_axis.y;
    r.x_axis.y = -b * b_mat.x_axis.x + a * b_mat.x_axis.y;
    r.y_axis.y = -b * b_mat.y_axis.x + a * b_mat.y_axis.y;
    r.z_axis.y = -b * b_mat.z_axis.x + a * b_mat.z_axis.y;
    r.x_axis.z = b_mat.x_axis.z;
    r.y_axis.z = b_mat.y_axis.z;
    r.z_axis.z = b_mat.z_axis.z;

    // Second Givens rotation (ch, 0, -sh, 0)
    let g2 = qr_givens_quaternion(r.x_axis.x, r.x_axis.z);
    a = -2.0 * g2.sh * g2.sh + 1.0;
    b = 2.0 * g2.ch * g2.sh;

    // Apply B = Q' * B
    b_mat.x_axis.x = a * r.x_axis.x + b * r.x_axis.z;
    b_mat.y_axis.x = a * r.y_axis.x + b * r.y_axis.z;
    b_mat.z_axis.x = a * r.z_axis.x + b * r.z_axis.z;
    b_mat.x_axis.y = r.x_axis.y;
    b_mat.y_axis.y = r.y_axis.y;
    b_mat.z_axis.y = r.z_axis.y;
    b_mat.x_axis.z = -b * r.x_axis.x + a * r.x_axis.z;
    b_mat.y_axis.z = -b * r.y_axis.x + a * r.y_axis.z;
    b_mat.z_axis.z = -b * r.z_axis.x + a * r.z_axis.z;

    // Third Givens rotation (ch, sh, 0, 0)
    let g3 = qr_givens_quaternion(b_mat.y_axis.y, b_mat.y_axis.z);
    a = -2.0 * g3.sh * g3.sh + 1.0;
    b = 2.0 * g3.ch * g3.sh;

    // R is now set to desired value
    r.x_axis.x = b_mat.x_axis.x;
    r.y_axis.x = b_mat.y_axis.x;
    r.z_axis.x = b_mat.z_axis.x;
    r.x_axis.y = a * b_mat.x_axis.y + b * b_mat.x_axis.z;
    r.y_axis.y = a * b_mat.y_axis.y + b * b_mat.y_axis.z;
    r.z_axis.y = a * b_mat.z_axis.y + b * b_mat.z_axis.z;
    r.x_axis.z = -b * b_mat.x_axis.y + a * b_mat.x_axis.z;
    r.y_axis.z = -b * b_mat.y_axis.y + a * b_mat.y_axis.z;
    r.z_axis.z = -b * b_mat.z_axis.y + a * b_mat.z_axis.z;

    // Construct the cumulative rotation Q = Q1 * Q2 * Q3
    let sh12 = 2.0 * (g1.sh * g1.sh - 0.5);
    let sh22 = 2.0 * (g2.sh * g2.sh - 0.5);
    let sh32 = 2.0 * (g3.sh * g3.sh - 0.5);

    q.x_axis.x = sh12 * sh22;
    q.x_axis.y = 4.0 * g2.ch * g3.ch * sh12 * g2.sh * g3.sh + 2.0 * g1.ch * g1.sh * sh32;
    q.x_axis.z = 4.0 * g1.ch * g3.ch * g1.sh * g3.sh - 2.0 * g2.ch * sh12 * g2.sh * sh32;

    q.y_axis.x = -2.0 * g1.ch * g1.sh * sh22;
    q.y_axis.y = -8.0 * g1.ch * g2.ch * g3.ch * g1.sh * g2.sh * g3.sh + sh12 * sh32;
    q.y_axis.z =
        -2.0 * g3.ch * g3.sh + 4.0 * g1.sh * (g3.ch * g1.sh * g3.sh + g1.ch * g2.ch * g2.sh * sh32);

    q.z_axis.x = 2.0 * g2.ch * g2.sh;
    q.z_axis.y = -2.0 * g3.ch * sh22 * g3.sh;
    q.z_axis.z = sh22 * sh32;

    QR3 { q, r }
}

/// Wrapping function used to contain all of the required sub calls
pub fn svd3(a: &Mat3) -> SVD3Set {
    // Compute the eigenvectors of A^T * A, which is V in SVD (Singular Vectors)
    let v = jacobi_eigenanalysis(Symmetric3x3::from_mat3x3(&(a.transpose().mul_mat3(a))));
    // Compute B = A * V
    let mut b = a.mul_mat3(&v);

    // Sort the singular values
    sort_singular_values(&mut b, &mut v.clone());

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
