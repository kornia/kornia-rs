use crate::{Mat3F32, Mat3F64, Vec2F32, Vec2F64, Vec3F64};
use nalgebra::{SMatrix, SVector};

/// Error type for homography estimation.
#[derive(thiserror::Error, Debug, Clone, Copy, PartialEq)]
pub enum HomographyError {
    /// Homography matrix is singular or near-singular.
    #[error("Homography determinant too small (near-singular matrix)")]
    SingularMatrix,

    /// Cheirality check failed.
    #[error("Cheirality check failed")]
    CheiralityCheckFailed,
}

/// Threshold for determinant check to avoid singular matrices.
const DETERMINANT_THRESHOLD: f64 = 1e-8;

/// Computes the homography matrix from four 2D point correspondences using Gaussian elimination.
///
/// This matches the behavior of the solver historically used in kornia-apriltag.
///
/// # Arguments
///
/// * `x1` - The source 2D points.
/// * `x2` - The destination 2D points.
///
/// # Returns
///
/// A `Result<Mat3F32, HomographyError>` containing the 3x3 homography matrix if successful.
///
/// # Errors
///
/// Returns [`HomographyError::SingularMatrix`] if the system is unsolvable or the resulting matrix is singular.
///
/// # Example
///
/// ```rust
/// use kornia_algebra::{Vec2F32, Mat3F32};
/// use kornia_algebra::linalg::homography::{homography_2d_dlt_gauss_elim_f32};
///
/// let x1 = [Vec2F32::new(0.0, 0.0), Vec2F32::new(1.0, 0.0), Vec2F32::new(1.0, 1.0), Vec2F32::new(0.0, 1.0)];
/// let x2 = [Vec2F32::new(1.0, 1.0), Vec2F32::new(2.0, 1.0), Vec2F32::new(2.0, 2.0), Vec2F32::new(1.0, 2.0)];
/// let h = homography_2d_dlt_gauss_elim_f32(&x1, &x2).unwrap();
/// ```
pub fn homography_2d_dlt_gauss_elim_f32(
    x1: &[Vec2F32; 4],
    x2: &[Vec2F32; 4],
) -> Result<Mat3F32, HomographyError> {
    #[rustfmt::skip]
    let mut a = [
        x1[0].x, x1[0].y, 1.0, 0.0,     0.0,     0.0, -x1[0].x*x2[0].x, -x1[0].y*x2[0].x, x2[0].x,
        0.0,     0.0,     0.0, x1[0].x, x1[0].y, 1.0, -x1[0].x*x2[0].y, -x1[0].y*x2[0].y, x2[0].y,
        x1[1].x, x1[1].y, 1.0, 0.0,     0.0,     0.0, -x1[1].x*x2[1].x, -x1[1].y*x2[1].x, x2[1].x,
        0.0,     0.0,     0.0, x1[1].x, x1[1].y, 1.0, -x1[1].x*x2[1].y, -x1[1].y*x2[1].y, x2[1].y,
        x1[2].x, x1[2].y, 1.0, 0.0,     0.0,     0.0, -x1[2].x*x2[2].x, -x1[2].y*x2[2].x, x2[2].x,
        0.0,     0.0,     0.0, x1[2].x, x1[2].y, 1.0, -x1[2].x*x2[2].y, -x1[2].y*x2[2].y, x2[2].y,
        x1[3].x, x1[3].y, 1.0, 0.0,     0.0,     0.0, -x1[3].x*x2[3].x, -x1[3].y*x2[3].x, x2[3].x,
        0.0,     0.0,     0.0, x1[3].x, x1[3].y, 1.0, -x1[3].x*x2[3].y, -x1[3].y*x2[3].y, x2[3].y,
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
            return Err(HomographyError::SingularMatrix);
        }

        let max_val_idx = max_val_idx as usize;

        if max_val < EPSILON {
            // Matrix is singular
            return Err(HomographyError::SingularMatrix);
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
    let h = Mat3F32::from_cols_array(&[
        a[8], a[35], a[62], // col 0
        a[17], a[44], a[71], // col 1
        a[26], a[53], 1.0, // col 2
    ]);

    if h.determinant().abs() < DETERMINANT_THRESHOLD as f32 {
        return Err(HomographyError::SingularMatrix);
    }

    Ok(h)
}

/// Computes the homography matrix from four 2D point correspondences using Direct Linear Transform (DLT) and SVD.
///
/// # Arguments
///
/// * `x1` - The source 2D points.
/// * `x2` - The destination 2D points.
///
/// # Returns
///
/// A `Result<Mat3F64, HomographyError>` containing the 3x3 homography matrix if successful.
///
/// # Errors
///
/// Returns [`HomographyError::SingularMatrix`] if the points are collinear or the system is otherwise degenerate.
///
/// # Example
///
/// ```rust
/// use kornia_algebra::{Vec2F64, Mat3F64};
/// use kornia_algebra::linalg::homography::{homography_2d_dlt_svd_f64};
///
/// let x1 = [Vec2F64::new(0.0, 0.0), Vec2F64::new(1.0, 0.0), Vec2F64::new(1.0, 1.0), Vec2F64::new(0.0, 1.0)];
/// let x2 = [Vec2F64::new(0.0, 0.0), Vec2F64::new(1.0, 0.0), Vec2F64::new(1.0, 1.0), Vec2F64::new(0.0, 1.0)];
/// let h = homography_2d_dlt_svd_f64(&x1, &x2).unwrap();
/// ```
pub fn homography_2d_dlt_svd_f64(
    x1: &[Vec2F64; 4],
    x2: &[Vec2F64; 4],
) -> Result<Mat3F64, HomographyError> {
    // construct matrix A (9x9) to get a full 9x9 V matrix from SVD
    let mut mat_a = SMatrix::<f64, 9, 9>::zeros();
    for i in 0..4 {
        let (x1_i, x2_i) = (x1[i], x2[i]);
        mat_a[(2 * i, 0)] = x1_i.x;
        mat_a[(2 * i, 1)] = x1_i.y;
        mat_a[(2 * i, 2)] = 1.0;
        mat_a[(2 * i, 6)] = -x2_i.x * x1_i.x;
        mat_a[(2 * i, 7)] = -x2_i.x * x1_i.y;
        mat_a[(2 * i, 8)] = -x2_i.x;

        mat_a[(2 * i + 1, 3)] = x1_i.x;
        mat_a[(2 * i + 1, 4)] = x1_i.y;
        mat_a[(2 * i + 1, 5)] = 1.0;
        mat_a[(2 * i + 1, 6)] = -x2_i.y * x1_i.x;
        mat_a[(2 * i + 1, 7)] = -x2_i.y * x1_i.y;
        mat_a[(2 * i + 1, 8)] = -x2_i.y;
    }
    // The 9th row is naturally zeroed.

    // svd solver
    let svd = mat_a.svd(true, true);
    // take the 9th row of v_t (which is the last column of v)
    let v_t = svd.v_t.ok_or(HomographyError::SingularMatrix)?;
    let h_vec = v_t.row(8);

    let h = Mat3F64::from_cols_array(&[
        h_vec[0], h_vec[3], h_vec[6], // col 0
        h_vec[1], h_vec[4], h_vec[7], // col 1
        h_vec[2], h_vec[5], h_vec[8], // col 2
    ]);

    if h.determinant().abs() < DETERMINANT_THRESHOLD {
        return Err(HomographyError::SingularMatrix);
    }

    Ok(h)
}

/// Compute the homography matrix from four 3D point correspondences using LU decomposition.
///
/// The homography matrix $H$ is computed such that $x_2 \sim H x_1$, where $\sim$ denotes projective equality.
///
/// # Arguments
///
/// * `x1` - The source 3D points.
/// * `x2` - The destination 3D points.
/// * `check_cheirality` - Whether to check the cheirality condition (points must be in front of the camera).
///
/// # Returns
///
/// A `Result<Mat3F64, HomographyError>` containing the 3x3 homography matrix if successful.
///
/// # Errors
///
/// Returns [`HomographyError::SingularMatrix`] if the linear system is unsolvable.
/// Returns [`HomographyError::CheiralityCheckFailed`] if `check_cheirality` is true and the constraint is violated.
///
/// # Example
///
/// ```rust
/// use kornia_algebra::{Vec3F64, Mat3F64};
/// use kornia_algebra::linalg::homography::{homography_3d_dlt_lu_f64};
///
/// let x1 = [Vec3F64::new(0.0, 0.0, 1.0), Vec3F64::new(1.0, 0.0, 1.0), Vec3F64::new(1.0, 1.0, 1.0), Vec3F64::new(0.0, 1.0, 1.0)];
/// let x2 = [Vec3F64::new(0.0, 0.0, 1.0), Vec3F64::new(1.0, 0.0, 1.0), Vec3F64::new(1.0, 1.0, 1.0), Vec3F64::new(0.0, 1.0, 1.0)];
/// let h = homography_3d_dlt_lu_f64(&x1, &x2, true).unwrap();
/// ```
pub fn homography_3d_dlt_lu_f64(
    x1: &[Vec3F64; 4],
    x2: &[Vec3F64; 4],
    check_cheirality: bool,
) -> Result<Mat3F64, HomographyError> {
    if check_cheirality {
        let p01 = x1[0].cross(x1[1]);
        let q01 = x2[0].cross(x2[1]);

        if (p01.dot(x1[2]) * q01.dot(x2[2])) < 0.0 {
            return Err(HomographyError::CheiralityCheckFailed);
        }

        if p01.dot(x1[3]) * q01.dot(x2[3]) < 0.0 {
            return Err(HomographyError::CheiralityCheckFailed);
        }

        let p23 = x1[2].cross(x1[3]);
        let q23 = x2[2].cross(x2[3]);

        if (p23.dot(x1[0]) * q23.dot(x2[0])) < 0.0 {
            return Err(HomographyError::CheiralityCheckFailed);
        }

        if (p23.dot(x1[1]) * q23.dot(x2[1])) < 0.0 {
            return Err(HomographyError::CheiralityCheckFailed);
        }
    }

    let mut mat_a = SMatrix::<f64, 8, 8>::zeros();
    let mut vec_b = SVector::<f64, 8>::zeros();

    for i in 0..4 {
        let (x1_i, x2_i) = (x1[i], x2[i]);

        mat_a[(2 * i, 0)] = x2_i.z * x1_i.x;
        mat_a[(2 * i, 1)] = x2_i.z * x1_i.y;
        mat_a[(2 * i, 2)] = x2_i.z * x1_i.z;
        mat_a[(2 * i, 6)] = -x2_i.x * x1_i.x;
        mat_a[(2 * i, 7)] = -x2_i.x * x1_i.y;
        vec_b[2 * i] = x2_i.x * x1_i.z;

        mat_a[(2 * i + 1, 3)] = x2_i.z * x1_i.x;
        mat_a[(2 * i + 1, 4)] = x2_i.z * x1_i.y;
        mat_a[(2 * i + 1, 5)] = x2_i.z * x1_i.z;
        mat_a[(2 * i + 1, 6)] = -x2_i.y * x1_i.x;
        mat_a[(2 * i + 1, 7)] = -x2_i.y * x1_i.y;
        vec_b[2 * i + 1] = x2_i.y * x1_i.z;
    }

    let h_mat = mat_a.lu().solve(&vec_b).ok_or(HomographyError::SingularMatrix)?;

    // column-major array
    let h = Mat3F64::from_cols_array(&[
        h_mat[0], h_mat[3], h_mat[6], // col 0
        h_mat[1], h_mat[4], h_mat[7], // col 1
        h_mat[2], h_mat[5], 1.0, // col 2
    ]);

    if h.determinant().abs() < DETERMINANT_THRESHOLD {
        return Err(HomographyError::SingularMatrix);
    }

    Ok(h)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_homography_2d_dlt_gauss_elim_f32() -> Result<(), Box<dyn std::error::Error>> {
        let x1 = [
            Vec2F32::new(-1.0, -1.0),
            Vec2F32::new(1.0, -1.0),
            Vec2F32::new(1.0, 1.0),
            Vec2F32::new(-1.0, 1.0),
        ];
        let x2 = [
            Vec2F32::new(27.0, 3.0),
            Vec2F32::new(27.0, 27.0),
            Vec2F32::new(3.0, 27.0),
            Vec2F32::new(3.0, 3.0),
        ];

        let h = homography_2d_dlt_gauss_elim_f32(&x1, &x2)?;
        // glam::Mat3 is column-major: [h11, h21, h31, h12, h22, h32, h13, h23, h33]
        let expected = Mat3F32::from_cols_array(&[
            -0.0, 12.0, -0.0, // col 0
            -12.0, -0.0, 0.0, // col 1
            15.0, 15.0, 1.0, // col 2
        ]);

        assert_eq!(h, expected);
        Ok(())
    }

    #[test]
    fn test_homography_2d_dlt_svd_f64_identity() -> Result<(), Box<dyn std::error::Error>> {
        let x1 = [
            Vec2F64::new(0.0, 0.0),
            Vec2F64::new(1.0, 0.0),
            Vec2F64::new(0.0, 1.0),
            Vec2F64::new(1.0, 1.0),
        ];
        let x2 = [
            Vec2F64::new(0.0, 0.0),
            Vec2F64::new(1.0, 0.0),
            Vec2F64::new(0.0, 1.0),
            Vec2F64::new(1.0, 1.0),
        ];

        let mut h = homography_2d_dlt_svd_f64(&x1, &x2)?;
        h.0 /= h.z_axis.z;

        let h_arr = h.to_cols_array();
        let exp_arr = Mat3F64::IDENTITY.to_cols_array();
        for i in 0..9 {
            assert_relative_eq!(h_arr[i], exp_arr[i], epsilon = 1e-6);
        }
        Ok(())
    }

    #[test]
    fn test_homography_2d_dlt_svd_f64_transform() -> Result<(), Box<dyn std::error::Error>> {
        let x1 = [
            Vec2F64::new(0.0, 0.0),
            Vec2F64::new(1.0, 0.0),
            Vec2F64::new(0.0, 1.0),
            Vec2F64::new(1.0, 1.0),
        ];
        let (tx, ty) = (1.0, 1.0);
        let expected_h = Mat3F64::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, tx, ty, 1.0]);

        let mut x2 = [Vec2F64::ZERO; 4];
        for i in 0..4 {
            let p = expected_h * Vec3F64::new(x1[i].x, x1[i].y, 1.0);
            x2[i] = Vec2F64::new(p.x / p.z, p.y / p.z);
        }

        let mut h = homography_2d_dlt_svd_f64(&x1, &x2)?;
        h.0 /= h.z_axis.z;

        let h_arr = h.to_cols_array();
        let exp_arr = expected_h.to_cols_array();
        for i in 0..9 {
            assert_relative_eq!(h_arr[i], exp_arr[i], epsilon = 1e-6);
        }
        Ok(())
    }

    #[test]
    fn test_homography_3d_dlt_lu_f64_identity() -> Result<(), Box<dyn std::error::Error>> {
        let x1 = [
            Vec3F64::new(0.0, 0.0, 1.0),
            Vec3F64::new(1.0, 0.0, 1.0),
            Vec3F64::new(0.0, 1.0, 1.0),
            Vec3F64::new(1.0, 1.0, 1.0),
        ];
        let x2 = [
            Vec3F64::new(0.0, 0.0, 1.0),
            Vec3F64::new(1.0, 0.0, 1.0),
            Vec3F64::new(0.0, 1.0, 1.0),
            Vec3F64::new(1.0, 1.0, 1.0),
        ];

        let mut h = homography_3d_dlt_lu_f64(&x1, &x2, true)?;
        h.0 /= h.z_axis.z;

        let h_arr = h.to_cols_array();
        let exp_arr = Mat3F64::IDENTITY.to_cols_array();
        for i in 0..9 {
            assert_relative_eq!(h_arr[i], exp_arr[i], epsilon = 1e-6);
        }
        Ok(())
    }

    #[test]
    fn test_homography_3d_dlt_lu_f64_transform() -> Result<(), Box<dyn std::error::Error>> {
        let x1 = [
            Vec3F64::new(0.0, 0.0, 1.0),
            Vec3F64::new(1.0, 0.0, 1.0),
            Vec3F64::new(0.0, 1.0, 1.0),
            Vec3F64::new(1.0, 1.0, 1.0),
        ];

        let (tx, ty) = (1.0, 1.0);
        let expected_h = Mat3F64::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, tx, ty, 1.0]);

        let mut x2 = [Vec3F64::ZERO; 4];
        for i in 0..4 {
            x2[i] = expected_h * x1[i];
        }

        let mut h = homography_3d_dlt_lu_f64(&x1, &x2, true)?;
        h.0 /= h.z_axis.z;

        let h_arr = h.to_cols_array();
        let exp_arr = expected_h.to_cols_array();
        for i in 0..9 {
            assert_relative_eq!(h_arr[i], exp_arr[i], epsilon = 1e-6);
        }
        Ok(())
    }
}
