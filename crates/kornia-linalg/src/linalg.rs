use std::ops::{IndexMut, Index};

const GEMMA:f32 = 5.828427124; 
const CSTAR:f32 = 0.923879532;
const SSTAR:f32 = 0.3826834323;
const SVD_EPSILON: f32 = 1e-6;
const JACOBI_STEPS: u32 = 12;
const RSQRT_STEPS: u32 = 4;
const RSQRT1_STEPS: u32 = 6;


/// Calculates the result of x / y. Required as the accurate square root function otherwise uses a reciprocal approximation when using optimizations on a GPU which can lead to slightly different results. If non exact matching results are acceptable a simple division can be used.
pub fn fdiv(x:f32, y:f32) -> f32 {
    return x / y;
}

/// Calculates the reciprocal square root of x using a fast approximation.
pub fn rsqrt(x: f32) -> f32 {
    let mut xhalf = -0.5 * x;
    let mut i = x.to_bits() as i32; // Convert float to raw bits
    i = 0x5f375a82 - (i >> 1); // Magic constant and bit manipulation
    let mut x = f32::from_bits(i as u32); // Convert bits back to float

    for _ in 0..RSQRT_STEPS {
        x = x * (x * x * xhalf + 1.5);
    }

    x
}

/// See rsqrt. Uses RSQRT1_STEPS to offer a higher precision alternative
pub fn rsqrt1(mut x: f32) -> f32 {
    let xhalf = -0.5 * x;
    let mut i = x.to_bits() as i32;
    i = 0x5f37599e - (i >> 1);
    x = f32::from_bits(i as u32);

    for _ in 0..RSQRT1_STEPS {
        x = x * (x * x * xhalf + 1.5);
    }

    return x;
}

/// Calculates the square root of x using 1.f/rsqrt1(x) to give a square root with controllable and consistent precision.
pub fn accurate_sqrt(mut x:f32) -> f32 {
    return fdiv(1.0, rsqrt(x));
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

/// Helper function used to convert quaternion to matrix
pub fn quaternion_to_matrix(q: &Quaternion) -> Mat3x3 {
    let w = q.w;
    let x = q.x;
    let y = q.y;
    let z = q.z;
    
    Mat3x3 {
        m_00: 1.0 - 2.0 * (y * y + z * z),
        m_01: 2.0 * (x * y - w * z),
        m_02: 2.0 * (x * z + w * y),
        
        m_10: 2.0 * (x * y + w * z),
        m_11: 1.0 - 2.0 * (x * x + z * z),
        m_12: 2.0 * (y * z - w * x),
        
        m_20: 2.0 * (x * z - w * y),
        m_21: 2.0 * (y * z + w * x),
        m_22: 1.0 - 2.0 * (x * x + y * y),
    }
}


#[derive(Debug,Clone)]
/// Helper class to contain a quaternion. Could be replaced with float4 (CUDA based type) but this might lead to unintended conversions when using the supplied matrices
pub struct Quaternion {
    /// The `x` component of the quaternion.
    /// Represents the vector component of the quaternion along the x-axis.
    pub x: f32,
    
    /// The `y` component of the quaternion.
    /// Represents the vector component of the quaternion along the y-axis.
    pub y: f32,

    /// The `z` component of the quaternion.
    /// Represents the vector component of the quaternion along the z-axis.
    pub z: f32,

    /// The `w` component of the quaternion.
    /// The scalar (real) part of the quaternion, representing the cosine of half the rotation angle.
    pub w: f32,
}

impl Quaternion {
    /// Create a new Quaternion from values
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Quaternion { x, y, z, w }
    }
}

impl Default for Quaternion {
    fn default() -> Self {
        Quaternion {
            x: 0.0,  // default x value
            y: 0.0,  // default y value
            z: 0.0,  // default z value
            w: 1.0,  // default w value (usually 1 for identity quaternion)
        }
    }
}

impl Index<usize> for Quaternion {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            3 => &self.w,
            _ => panic!("Index out of bounds for Quaternion"),
        }
    }
}


impl IndexMut<usize> for Quaternion {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            3 => &mut self.w,
            _ => panic!("Index out of bounds for Quaternion"),
        }
    }
}


#[derive(Debug, Clone, Copy)]
/// A simple 3x3 Matrix class
pub struct Mat3x3 {
  /// The element at row 0, column 0 of the matrix.
  pub m_00: f32,

  /// The element at row 0, column 1 of the matrix.
  pub m_01: f32,

  /// The element at row 0, column 2 of the matrix.
  pub m_02: f32,

  /// The element at row 1, column 0 of the matrix.
  pub m_10: f32,

  /// The element at row 1, column 1 of the matrix.
  pub m_11: f32,

  /// The element at row 1, column 2 of the matrix.
  pub m_12: f32,

  /// The element at row 2, column 0 of the matrix.
  pub m_20: f32,

  /// The element at row 2, column 1 of the matrix.
  pub m_21: f32,

  /// The element at row 2, column 2 of the matrix.
  pub m_22: f32,
}

impl Mat3x3 {
    /// Constructor to initialize matrix with given values
    pub fn new(a11: f32, a12: f32, a13: f32, a21: f32, a22: f32, a23: f32, a31: f32, a32: f32, a33: f32) -> Self {
        Mat3x3 {
            m_00: a11, m_01: a12, m_02: a13,
            m_10: a21, m_11: a22, m_12: a23,
            m_20: a31, m_21: a32, m_22: a33,
        }
    }

    /// Zero Matrix
    pub fn zero() -> Self {
        Mat3x3 {
            m_00: 0.0, m_01: 0.0, m_02: 0.0,
            m_10: 0.0, m_11: 0.0, m_12: 0.0,
            m_20: 0.0, m_21: 0.0, m_22: 0.0,
        }
    }

    /// identity Matrix
    pub fn identity() -> Self {
        Mat3x3 {
            m_00: 1.0, m_01: 0.0, m_02: 0.0,
            m_10: 0.0, m_11: 1.0, m_12: 0.0,
            m_20: 0.0, m_21: 0.0, m_22: 1.0,
        }
    }

    /// Matrix from pointer (assuming a flat array with a certain stride)
    pub fn from_ptr(ptr: &[f32], i: usize, offset: usize) -> Self {
        Mat3x3 {
            m_00: ptr[i + 0 * offset], m_01: ptr[i + 1 * offset], m_02: ptr[i + 2 * offset],
            m_10: ptr[i + 3 * offset], m_11: ptr[i + 4 * offset], m_12: ptr[i + 5 * offset],
            m_20: ptr[i + 6 * offset], m_21: ptr[i + 7 * offset], m_22: ptr[i + 8 * offset],
        }
    }

    /// Determinant of the matrix
    pub fn det(&self) -> f32 {
        self.m_00 * (self.m_11 * self.m_22 - self.m_12 * self.m_21) 
        - self.m_01 * (self.m_10 * self.m_22 - self.m_12 * self.m_20)
        + self.m_02 * (self.m_10 * self.m_21 - self.m_11 * self.m_20)
    }

    /// Convert matrix to pointer (store in flat array with stride)
    pub fn to_ptr(&self, ptr: &mut [f32], i: usize, offset: usize) {
        ptr[i + 0 * offset] = self.m_00;
        ptr[i + 1 * offset] = self.m_01;
        ptr[i + 2 * offset] = self.m_02;
        ptr[i + 3 * offset] = self.m_10;
        ptr[i + 4 * offset] = self.m_11;
        ptr[i + 5 * offset] = self.m_12;
        ptr[i + 6 * offset] = self.m_20;
        ptr[i + 7 * offset] = self.m_21;
        ptr[i + 8 * offset] = self.m_22;
    }

    /// Transpose of the matrix
    pub fn transpose(&self) -> Self {
        Mat3x3 {
            m_00: self.m_00, m_10: self.m_01, m_20: self.m_02,
            m_01: self.m_10, m_11: self.m_11, m_21: self.m_12,
            m_02: self.m_20, m_12: self.m_21, m_22: self.m_22,
        }
    }

    /// Matrix multiplication with scalar
    pub fn mul_scalar(&self, o: f32) -> Self {
        Mat3x3 {
            m_00: self.m_00 * o, m_01: self.m_01 * o, m_02: self.m_02 * o,
            m_10: self.m_10 * o, m_11: self.m_11 * o, m_12: self.m_12 * o,
            m_20: self.m_20 * o, m_21: self.m_21 * o, m_22: self.m_22 * o,
        }
    }

    /// In-place matrix multiplication with scalar
    pub fn mul_scalar_in_place(&mut self, o: f32) {
        self.m_00 *= o; self.m_01 *= o; self.m_02 *= o;
        self.m_10 *= o; self.m_11 *= o; self.m_12 *= o;
        self.m_20 *= o; self.m_21 *= o; self.m_22 *= o;
    }

    /// Matrix subtraction
    pub fn sub(&self, o: &Mat3x3) -> Self {
        Mat3x3 {
            m_00: self.m_00 - o.m_00, m_01: self.m_01 - o.m_01, m_02: self.m_02 - o.m_02,
            m_10: self.m_10 - o.m_10, m_11: self.m_11 - o.m_11, m_12: self.m_12 - o.m_12,
            m_20: self.m_20 - o.m_20, m_21: self.m_21 - o.m_21, m_22: self.m_22 - o.m_22,
        }
    }

    /// Matrix multiplication
    pub fn mul(&self, o: &Mat3x3) -> Mat3x3 {
        Mat3x3 {
            m_00: self.m_00 * o.m_00 + self.m_01 * o.m_10 + self.m_02 * o.m_20,
            m_01: self.m_00 * o.m_01 + self.m_01 * o.m_11 + self.m_02 * o.m_21,
            m_02: self.m_00 * o.m_02 + self.m_01 * o.m_12 + self.m_02 * o.m_22,
            m_10: self.m_10 * o.m_00 + self.m_11 * o.m_10 + self.m_12 * o.m_20,
            m_11: self.m_10 * o.m_01 + self.m_11 * o.m_11 + self.m_12 * o.m_21,
            m_12: self.m_10 * o.m_02 + self.m_11 * o.m_12 + self.m_12 * o.m_22,
            m_20: self.m_20 * o.m_00 + self.m_21 * o.m_10 + self.m_22 * o.m_20,
            m_21: self.m_20 * o.m_01 + self.m_21 * o.m_11 + self.m_22 * o.m_21,
            m_22: self.m_20 * o.m_02 + self.m_21 * o.m_12 + self.m_22 * o.m_22,
        }
    }
}


impl Default for Mat3x3 {
    // Default implementation: Identity Matrix
    fn default() -> Self {
        Mat3x3 {
            m_00: 1.0, m_01: 0.0, m_02: 0.0,
            m_10: 0.0, m_11: 1.0, m_12: 0.0,
            m_20: 0.0, m_21: 0.0, m_22: 1.0,
        }
    }
}

// Implement the Index trait for Mat3x3
impl Index<(usize, usize)> for Mat3x3 {
    type Output = f32;

    fn index(&self, idx: (usize, usize)) -> &Self::Output {
        match idx {
            (0, 0) => &self.m_00,
            (0, 1) => &self.m_01,
            (0, 2) => &self.m_02,
            (1, 0) => &self.m_10,
            (1, 1) => &self.m_11,
            (1, 2) => &self.m_12,
            (2, 0) => &self.m_20,
            (2, 1) => &self.m_21,
            (2, 2) => &self.m_22,
            _ => panic!("Index out of bounds for Mat3x3"),
        }
    }
}




#[derive(Debug, Clone, Copy)]
/// A simple symmetric 3x3 Matrix class (contains no storage for (0, 1) (0, 2) and (1, 2)
pub struct Symmetric3x3 {
    /// The element at row 0, column 0 of the matrix, typically the first diagonal element.
    pub m_00: f32,

    /// The element at row 1, column 0 of the matrix. Since this is a symmetric matrix, it is equivalent to `m_01`.
    pub m_10: f32,

    /// The element at row 1, column 1 of the matrix, the second diagonal element.
    pub m_11: f32,

    /// The element at row 2, column 0 of the matrix. Since this is a symmetric matrix, it is equivalent to `m_02`.
    pub m_20: f32,

    /// The element at row 2, column 1 of the matrix. Since this is a symmetric matrix, it is equivalent to `m_12`.
    pub m_21: f32,

    /// The element at row 2, column 2 of the matrix, the third diagonal element.
    pub m_22: f32,
}


impl Symmetric3x3 {
    /// Constructor to initialize the symmetric matrix with given values
    pub fn new(a11: f32, a21: f32, a22: f32, a31: f32, a32: f32, a33: f32) -> Self {
        Symmetric3x3 {
            m_00: a11,
            m_10: a21,
            m_11: a22,
            m_20: a31,
            m_21: a32,
            m_22: a33,
        }
    }

    /// Constructor from a regular Mat3x3 (assuming Mat3x3 exists)
    pub fn from_mat3x3(mat: &Mat3x3) -> Self {
        Symmetric3x3 {
            m_00: mat.m_00,
            m_10: mat.m_10,
            m_11: mat.m_11,
            m_20: mat.m_20,
            m_21: mat.m_21,
            m_22: mat.m_22,
        }
    }
}


#[derive(Debug, Clone, Copy)]
/// Helper struct to store 2 floats to avoid OUT parameters on functions
pub struct Givens {
    /// The cosine of the angle in the Givens rotation.
    pub ch: f32, 

    /// The sine of the angle in the Givens rotation.
    pub sh: f32,
}

impl Givens {
    /// Constructor with default values for ch and sh
    pub fn new(ch: f32, sh: f32) -> Self {
        Givens { ch, sh }
    }

    /// Constructor with default CSTAR and SSTAR values
    pub fn default() -> Self {
        Givens {
            ch: CSTAR,
            sh: SSTAR,
        }
    }
}


#[derive(Debug, Clone, Copy)]
/// Helper struct to store 2 Matrices to avoid OUT parameters on functions
pub struct QR {
    /// The orthogonal matrix Q from the QR decomposition.
    pub Q: Mat3x3,

    /// The upper triangular matrix R from the QR decomposition.
    pub R: Mat3x3,
}


#[derive(Debug, Clone, Copy)]
/// Helper struct to store 3 Matrices to avoid OUT parameters on functions
pub struct SVDSet {
   /// The matrix of left singular vectors.
   pub U: Mat3x3,

   /// The diagonal matrix of singular values.
   pub S: Mat3x3,

   /// The matrix of right singular vectors.
   pub V: Mat3x3,
}

/// Calculates the squared norm of the vector [x y z] using a standard scalar product d = x * x + y *y + z * z
pub fn dist2(x:f32,y:f32,z:f32) -> f32 {
    x * x + (y *y + z * z)
}

/// For an explanation of the math see http://pages.cs.wisc.edu/~sifakis/papers/SVD_TR1690.pdf 
/// Computing the Singular Value Decomposition of 3 x 3 matrices with minimal branching and elementary floating point operations
/// See Algorithm 2 in reference. Given a matrix A this function returns the givens quaternion (x and w component, y and z are 0)
pub fn approximate_givens_quaternion(A: &Symmetric3x3) -> Givens {
    let g = Givens {
        ch: 2.0 * (A.m_00 - A.m_11),
        sh: A.m_10,
    };

    let mut b = GEMMA * g.sh * g.sh < g.ch * g.ch;
    let w = rsqrt(g.ch * g.ch + g.sh * g.sh);

    if w != w { // Checking for NaN
        b = false;
    }

    Givens {
        ch: if b { w * g.ch } else { CSTAR },
        sh: if b { w * g.sh } else { SSTAR },
    }
}

/// Function used to apply a givens rotation S. Calculates the weights and updates the quaternion to contain the cumultative rotation
pub fn jacobi_conjugation(x: usize, y: usize, z: usize, S: &mut Symmetric3x3, q: &mut Quaternion) {
    // Compute the Givens rotation (approximated)
    let mut g = approximate_givens_quaternion(S);
    // Scale and calculate intermediate values
    let scale = 1.0 / (g.ch * g.ch + g.sh * g.sh);
    let a = (g.ch * g.ch - g.sh * g.sh) * scale;
    let b = 2.0 * g.sh * g.ch * scale;

    // Create a copy of the matrix to avoid modifying the original during calculations
    let mut _S = S.clone();

    // Perform conjugation: S = Q'*S*Q
    S.m_00 = a * (a * _S.m_00 + b * _S.m_10) + b * (a * _S.m_10 + b * _S.m_11);
    S.m_10 = a * (-b * _S.m_00 + a * _S.m_10) + b * (-b * _S.m_10 + a * _S.m_11);
    S.m_11 = -b * (-b * _S.m_00 + a * _S.m_10) + a * (-b * _S.m_10 + a * _S.m_11);
    S.m_20 = a * _S.m_20 + b * _S.m_21;
    S.m_21 = -b * _S.m_20 + a * _S.m_21;
    S.m_22 = _S.m_22;

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
    _S.m_00 = S.m_11;
    _S.m_10 = S.m_21;
    _S.m_11 = S.m_22;
    _S.m_20 = S.m_10;
    _S.m_21 = S.m_20;
    _S.m_22 = S.m_00;

    S.m_00 = _S.m_00;
    S.m_10 = _S.m_10;
    S.m_11 = _S.m_11;
    S.m_20 = _S.m_20;
    S.m_21 = _S.m_21;
    S.m_22 = _S.m_22;
}

/// Function used to contain the givens permutations and the loop of the jacobi steps controlled by JACOBI_STEPS
/// Returns the quaternion q containing the cumultative result used to reconstruct S
pub fn jacobi_eigenanalysis(mut S: Symmetric3x3) -> Mat3x3 {
    let mut q = Quaternion::default();
    for _i in 0..JACOBI_STEPS {
        jacobi_conjugation(0, 1, 2, &mut S, &mut q);
        jacobi_conjugation(1, 2, 0, &mut S, &mut q);
        jacobi_conjugation(2, 0, 1, &mut S, &mut q);
    }
    print!("{:?}",q);
    return quaternion_to_matrix(&q);
}

/// Implementation of Algorithm 3
pub fn sort_singular_values(B: &mut Mat3x3, V: &mut Mat3x3) {
    let mut rho1 = dist2(B.m_00, B.m_10, B.m_20);
    let mut rho2 = dist2(B.m_01, B.m_11, B.m_21);
    let mut rho3 = dist2(B.m_02, B.m_12, B.m_22);

    let mut c = rho1 < rho2;
    cond_neg_swap(c, &mut B.m_00, &mut B.m_01);
    cond_neg_swap(c, &mut V.m_00, &mut V.m_01);
    cond_neg_swap(c, &mut B.m_10, &mut B.m_11);
    cond_neg_swap(c, &mut V.m_10, &mut V.m_11);
    cond_neg_swap(c, &mut B.m_20, &mut B.m_21);
    cond_neg_swap(c, &mut V.m_20, &mut V.m_21);
    cond_swap(c, &mut rho1, &mut rho2);

    c = rho1 < rho3;
    cond_neg_swap(c, &mut B.m_00, &mut B.m_02);
    cond_neg_swap(c, &mut V.m_00, &mut V.m_02);
    cond_neg_swap(c, &mut B.m_10, &mut B.m_12);
    cond_neg_swap(c, &mut V.m_10, &mut V.m_12);
    cond_neg_swap(c, &mut B.m_20, &mut B.m_22);
    cond_neg_swap(c, &mut V.m_20, &mut V.m_22);
    cond_swap(c, &mut rho1, &mut rho3);

    c = rho2 < rho3;
    cond_neg_swap(c, &mut B.m_01, &mut B.m_02);
    cond_neg_swap(c, &mut V.m_01, &mut V.m_02);
    cond_neg_swap(c, &mut B.m_11, &mut B.m_12);
    cond_neg_swap(c, &mut V.m_11, &mut V.m_12);
    cond_neg_swap(c, &mut B.m_21, &mut B.m_22);
    cond_neg_swap(c, &mut V.m_21, &mut V.m_22);
}


/// Implementation of Algorithm 4
pub fn qr_givens_quaternion(a1: f32, a2: f32) -> Givens {
    // a1 = pivot point on diagonal
    // a2 = lower triangular entry we want to annihilate
    let epsilon = SVD_EPSILON ; // Assuming _SVD_EPSILON is defined elsewhere
    let rho = accurate_sqrt(a1 * a1 + a2 * a2);

    let mut g = Givens {
        ch: (a1.abs() + (f32::max(rho, epsilon))),
        sh: if rho > epsilon { a2 } else { 0.0 },
    };

    let b = a1 < 0.0;
    cond_swap(b, &mut g.sh, &mut g.ch);

    let w = (g.ch * g.ch + g.sh * g.sh).sqrt();
    g.ch *= w;
    g.sh *= w;

    return g
}

/// Implements a QR decomposition of a Matrix
pub fn qr_decomposition(B: &mut Mat3x3) -> QR {
    let mut Q = Mat3x3::zero();
    let mut R = Mat3x3::zero();

    // First Givens rotation (ch, 0, 0, sh)
    let g1 = qr_givens_quaternion(B.m_00, B.m_10);
    let mut a = -2.0 * g1.sh * g1.sh + 1.0;
    let mut b = 2.0 * g1.ch * g1.sh;
    
    // Apply B = Q' * B
    R.m_00 = a * B.m_00 + b * B.m_10;
    R.m_01 = a * B.m_01 + b * B.m_11;
    R.m_02 = a * B.m_02 + b * B.m_12;
    R.m_10 = -b * B.m_00 + a * B.m_10;
    R.m_11 = -b * B.m_01 + a * B.m_11;
    R.m_12 = -b * B.m_02 + a * B.m_12;
    R.m_20 = B.m_20;
    R.m_21 = B.m_21;
    R.m_22 = B.m_22;

    // Second Givens rotation (ch, 0, -sh, 0)
    let g2 = qr_givens_quaternion(R.m_00, R.m_20);
    a = -2.0 * g2.sh * g2.sh + 1.0;
    b = 2.0 * g2.ch * g2.sh;
    
    // Apply B = Q' * B
    B.m_00 = a * R.m_00 + b * R.m_20;
    B.m_01 = a * R.m_01 + b * R.m_21;
    B.m_02 = a * R.m_02 + b * R.m_22;
    B.m_10 = R.m_10;
    B.m_11 = R.m_11;
    B.m_12 = R.m_12;
    B.m_20 = -b * R.m_00 + a * R.m_20;
    B.m_21 = -b * R.m_01 + a * R.m_21;
    B.m_22 = -b * R.m_02 + a * R.m_22;

    // Third Givens rotation (ch, sh, 0, 0)
    let g3 = qr_givens_quaternion(B.m_11, B.m_21);
    a = -2.0 * g3.sh * g3.sh + 1.0;
    b = 2.0 * g3.ch * g3.sh;
    
    // R is now set to desired value
    R.m_00 = B.m_00;
    R.m_01 = B.m_01;
    R.m_02 = B.m_02;
    R.m_10 = a * B.m_10 + b * B.m_20;
    R.m_11 = a * B.m_11 + b * B.m_21;
    R.m_12 = a * B.m_12 + b * B.m_22;
    R.m_20 = -b * B.m_10 + a * B.m_20;
    R.m_21 = -b * B.m_11 + a * B.m_21;
    R.m_22 = -b * B.m_12 + a * B.m_22;

    // Construct the cumulative rotation Q = Q1 * Q2 * Q3
    let sh12 = 2.0 * (g1.sh * g1.sh - 0.5);
    let sh22 = 2.0 * (g2.sh * g2.sh - 0.5);
    let sh32 = 2.0 * (g3.sh * g3.sh - 0.5);

    Q.m_00 = sh12 * sh22;
    Q.m_01 = 4.0 * g2.ch * g3.ch * sh12 * g2.sh * g3.sh + 2.0 * g1.ch * g1.sh * sh32;
    Q.m_02 = 4.0 * g1.ch * g3.ch * g1.sh * g3.sh - 2.0 * g2.ch * sh12 * g2.sh * sh32;
    
    Q.m_10 = -2.0 * g1.ch * g1.sh * sh22;
    Q.m_11 = -8.0 * g1.ch * g2.ch * g3.ch * g1.sh * g2.sh * g3.sh + sh12 * sh32;
    Q.m_12 = -2.0 * g3.ch * g3.sh + 4.0 * g1.sh * (g3.ch * g1.sh * g3.sh + g1.ch * g2.ch * g2.sh * sh32);
    
    Q.m_20 = 2.0 * g2.ch * g2.sh;
    Q.m_21 = -2.0 * g3.ch * sh22 * g3.sh;
    Q.m_22 = sh22 * sh32;

    QR { Q, R }
}


/// Wrapping function used to contain all of the required sub calls
pub fn svd(A: Mat3x3) -> SVDSet {
    // Compute the eigenvectors of A^T * A, which is V in SVD (Singular Vectors)
    let V =  jacobi_eigenanalysis(Symmetric3x3::from_mat3x3(&(A.transpose().mul(&A))));
    print!("{:?}",V);
    // Compute B = A * V
    let mut B = A.mul(&V);
    
    // Sort the singular values
    sort_singular_values(&mut B, &mut V.clone());

    // Perform QR decomposition on B to get Q and R
    let qr = qr_decomposition(&mut B);
    
    // Reset MXCSR register (if needed)

    // Return the SVD result, which includes Q (as U), R (as S), and V
    SVDSet {
        U: qr.Q,
        S: qr.R,
        V,
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_svd() {
        // Define a simple 3x3 matrix A
        let A = Mat3x3 {
            m_00: 2.0, m_01: 0.0, m_02: 0.0,
            m_10: 2.0, m_11: 1.0, m_12: 0.0,
            m_20: 0.0, m_21: -2.0, m_22: 0.0,
        };

        // Perform SVD on matrix A
        let svd_result = svd(A);
        println!("SVD Result : {:?}", svd_result);
        // Check if the shapes of U, S, and V are correct (3x3 matrices)
        assert_eq!(svd_result.S.m_00, 3.0, "S[0][0] is wrong");  // Singular value (approximate)
        assert_eq!(svd_result.V.m_00, 0.8944271, "V[0][0] is wrong");  // Singular vector (approximate)

        // Optionally, check if the singular values are sorted (since it's typical for SVD results to have descending singular values)
        let singular_values = vec![
            svd_result.S.m_00, svd_result.S.m_11, svd_result.S.m_22
        ];
        assert!(singular_values[0] >= singular_values[1] && singular_values[1] >= singular_values[2], 
            "Singular values are not sorted properly");

        // Check if A * V is close to B (since B = A * V is computed in the function)
        let A_mul_V = A.mul(&svd_result.V);
        for i in 0..3 {
            for j in 0..3 {
                assert!((A_mul_V[(i, j)] - svd_result.S[(i, j)]).abs() < 1e-5, 
                    "A * V does not match the S matrix after decomposition");
            }
        }
    }
}
