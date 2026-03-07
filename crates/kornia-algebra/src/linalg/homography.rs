use crate::{Mat3F32, Mat3F64};
use nalgebra::{SMatrix, SVector};

/// Computes the homography matrix from four 2D point correspondences using Gaussian elimination.
/// This matches the behavior of the solver historically used in kornia-apriltag.
/// 
/// # Arguments
///
/// * `c` - A 4x4 array where each row contains the coordinates of a correspondence:
///   [x, y, x', y'] where (x, y) maps to (x', y').
///
/// # Returns
///
/// An `Option<Mat3F32>` containing the 3x3 homography matrix if successful, or `None` if the matrix is singular.
pub fn dlt_gauss_elim_f32(c: &[[f32; 4]; 4]) -> Option<Mat3F32> {
    #[rustfmt::skip]
    let mut a = [
        c[0][0], c[0][1], 1.0, 0.0,     0.0,     0.0, -c[0][0]*c[0][2], -c[0][1]*c[0][2], c[0][2],
        0.0,     0.0,     0.0, c[0][0], c[0][1], 1.0, -c[0][0]*c[0][3], -c[0][1]*c[0][3], c[0][3],
        c[1][0], c[1][1], 1.0, 0.0,     0.0,     0.0, -c[1][0]*c[1][2], -c[1][1]*c[1][2], c[1][2],
        0.0,     0.0,     0.0, c[1][0], c[1][1], 1.0, -c[1][0]*c[1][3], -c[1][1]*c[1][3], c[1][3],
        c[2][0], c[2][1], 1.0, 0.0,     0.0,     0.0, -c[2][0]*c[2][2], -c[2][1]*c[2][2], c[2][2],
        0.0,     0.0,     0.0, c[2][0], c[2][1], 1.0, -c[2][0]*c[2][3], -c[2][1]*c[2][3], c[2][3],
        c[3][0], c[3][1], 1.0, 0.0,     0.0,     0.0, -c[3][0]*c[3][2], -c[3][1]*c[3][2], c[3][2],
        0.0,     0.0,     0.0, c[3][0], c[3][1], 1.0, -c[3][0]*c[3][3], -c[3][1]*c[3][3], c[3][3],
    ];

    const EPSILON: f32 = 1e-10;

    // Eliminate
    for col in 0..8 {
        // Find best row to swap with
        let mut max_val = 0.0;
        let mut max_val_idx = -1;

        for row in col..8 {
            let val = a[row * 9 + col].abs();
            if val > max_val {
                max_val = val;
                max_val_idx = row as isize;
            }
        }

        if max_val_idx < 0 {
            return None;
        }

        let max_val_idx = max_val_idx as usize;

        if max_val < EPSILON {
            // Matrix is singular
            return None;
        }

        // Swap to get best row
        if max_val_idx != col {
            for i in col..9 {
                a.swap(col * 9 + i, max_val_idx * 9 + i);
            }
        }

        // Do eliminate
        for i in (col + 1)..8 {
            let f = a[i * 9 + col] / a[col * 9 + col];
            a[i * 9 + col] = 0.0;
            for j in (col + 1)..9 {
                a[i * 9 + j] -= f * a[col * 9 + j];
            }
        }
    }

    // Back solve
    for col in (0..8).rev() {
        let mut sum = 0.0;
        for i in (col + 1)..8 {
            sum += a[col * 9 + i] * a[i * 9 + 8];
        }
        a[col * 9 + 8] = (a[col * 9 + 8] - sum) / a[col * 9 + col];
    }

    // Variables solve as: h11, h12, h13, h21, h22, h23, h31, h32. h33 is 1.0.
    // glam::Mat3 is column-major: [h11, h21, h31, h12, h22, h32, h13, h23, h33]
    Some(Mat3F32::from_cols_array(&[
        a[8], a[35], a[62], // col 0
        a[17], a[44], a[71], // col 1
        a[26], a[53], 1.0, // col 2
    ]))
}

/// Compute the homography matrix from four 2d point correspondences using SVD.
///
/// * `x1` - The source 2d points with shape (4, 2).
/// * `x2` - The destination 2d points with shape (4, 2).
///
/// Returns the 3x3 homography matrix.
pub fn dlt_svd_f64(x1: &[[f64; 2]; 4], x2: &[[f64; 2]; 4]) -> Option<Mat3F64> {
    // construct matrix A (9x9) to get a full 9x9 V matrix from SVD
    let mut mat_a = SMatrix::<f64, 9, 9>::zeros();
    for i in 0..4 {
        let (x1_i, x2_i) = (x1[i], x2[i]);
        mat_a[(2 * i, 0)] = x1_i[0];
        mat_a[(2 * i, 1)] = x1_i[1];
        mat_a[(2 * i, 2)] = 1.0;
        mat_a[(2 * i, 6)] = -x2_i[0] * x1_i[0];
        mat_a[(2 * i, 7)] = -x2_i[0] * x1_i[1];
        mat_a[(2 * i, 8)] = -x2_i[0];

        mat_a[(2 * i + 1, 3)] = x1_i[0];
        mat_a[(2 * i + 1, 4)] = x1_i[1];
        mat_a[(2 * i + 1, 5)] = 1.0;
        mat_a[(2 * i + 1, 6)] = -x2_i[1] * x1_i[0];
        mat_a[(2 * i + 1, 7)] = -x2_i[1] * x1_i[1];
        mat_a[(2 * i + 1, 8)] = -x2_i[1];
    }
    // The 9th row is naturally zeroed.

    // svd solver
    let svd = mat_a.svd(true, true);
    // take the 9th row of v_t (which is the last column of v)
    let v_t = svd.v_t?;
    let h = v_t.row(8);

    Some(Mat3F64::from_cols_array(&[
        h[0], h[3], h[6], // col 0
        h[1], h[4], h[7], // col 1
        h[2], h[5], h[8], // col 2
    ]))
}

/// Compute the homography matrix from four 3d point correspondences using LU decomposition.
/// 
/// Note: The points supplied are often assumed to be structured such that the matrix is solvable.
///
/// # Arguments
///
/// * `x1` - The source 3d points with shape (4, 3).
/// * `x2` - The destination 3d points with shape (4, 3).
///
/// # Returns
///
/// The output homography matrix from src to dst.
pub fn dlt_lu_f64(x1: &[[f64; 3]; 4], x2: &[[f64; 3]; 4]) -> Option<Mat3F64> {
    let mut mat_a = SMatrix::<f64, 8, 8>::zeros();
    let mut vec_b = SVector::<f64, 8>::zeros();

    for i in 0..4 {
        let (x1_0, x1_1, x1_2) = (x1[i][0], x1[i][1], x1[i][2]);
        let (x2_0, x2_1, x2_2) = (x2[i][0], x2[i][1], x2[i][2]);

        mat_a[(2 * i, 0)] = x2_2 * x1_0;
        mat_a[(2 * i, 1)] = x2_2 * x1_1;
        mat_a[(2 * i, 2)] = x2_2 * x1_2;
        mat_a[(2 * i, 6)] = -x2_0 * x1_0;
        mat_a[(2 * i, 7)] = -x2_0 * x1_1;
        vec_b[2 * i] = x2_0 * x1_2;

        mat_a[(2 * i + 1, 3)] = x2_2 * x1_0;
        mat_a[(2 * i + 1, 4)] = x2_2 * x1_1;
        mat_a[(2 * i + 1, 5)] = x2_2 * x1_2;
        mat_a[(2 * i + 1, 6)] = -x2_1 * x1_0;
        mat_a[(2 * i + 1, 7)] = -x2_1 * x1_1;
        vec_b[2 * i + 1] = x2_1 * x1_2;
    }

    let h_mat = mat_a.lu().solve(&vec_b)?;

    // column-major array
    Some(Mat3F64::from_cols_array(&[
        h_mat[0], h_mat[3], h_mat[6], // col 0
        h_mat[1], h_mat[4], h_mat[7], // col 1
        h_mat[2], h_mat[5], 1.0,      // col 2
    ]))
}
