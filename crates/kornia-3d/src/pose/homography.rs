use crate::linalg;
use faer::prelude::SpSolver;
use kornia_algebra::{linalg::svd::svd3_f64, Mat3F64, Vec3F64};

/// Error type for homography estimation.
#[derive(thiserror::Error, Debug)]
pub enum HomographyError {
    /// Homography matrix is singular or near-singular.
    #[error("Homography determinant too small (near-singular matrix)")]
    SingularMatrix,

    /// Cheirality constraint violated.
    #[error("Cheirality check failed")]
    CheiralityCheckFailed,
}

/// Compute the homography matrix from four 2d point correspondences.
///
/// * `x1` - The source 2d points with shape (4, 2).
/// * `x2` - The destination 2d points with shape (4, 2).
/// * `homo` - The output homography matrix from src to dst with shape (3, 3).
pub fn homography_4pt2d(
    x1: &[[f64; 2]; 4],
    x2: &[[f64; 2]; 4],
    homo: &mut [[f64; 3]; 3],
) -> Result<(), HomographyError> {
    // construct matrix A
    let mut mat_a = faer::Mat::<f64>::zeros(8, 9);
    for i in 0..4 {
        let (x1_i, x2_i) = (x1[i], x2[i]);
        unsafe {
            mat_a.write_unchecked(2 * i, 0, x1_i[0]);
            mat_a.write_unchecked(2 * i, 1, x1_i[1]);
            mat_a.write_unchecked(2 * i, 2, 1.0);
            mat_a.write_unchecked(2 * i, 6, -x2_i[0] * x1_i[0]);
            mat_a.write_unchecked(2 * i, 7, -x2_i[0] * x1_i[1]);
            mat_a.write_unchecked(2 * i, 8, -x2_i[0]);

            mat_a.write_unchecked(2 * i + 1, 3, x1_i[0]);
            mat_a.write_unchecked(2 * i + 1, 4, x1_i[1]);
            mat_a.write_unchecked(2 * i + 1, 5, 1.0);
            mat_a.write_unchecked(2 * i + 1, 6, -x2_i[1] * x1_i[0]);
            mat_a.write_unchecked(2 * i + 1, 7, -x2_i[1] * x1_i[1]);
            mat_a.write_unchecked(2 * i + 1, 8, -x2_i[1]);
        }
    }

    // solve -> h_mat: 8x1 and take the smallest singular value
    let svd = mat_a.svd();
    let h = svd.v().col(8);

    // copy to homography matrix
    homo[0] = [h[0], h[1], h[2]];
    homo[1] = [h[3], h[4], h[5]];
    homo[2] = [h[6], h[7], h[8]];

    // normalize the homography matrix
    linalg::normalize_mat33_inplace(homo);

    if linalg::det_mat33(homo).abs() < 1e-8 {
        return Err(HomographyError::SingularMatrix);
    }

    Ok(())
}

/// Compute the homography matrix from four 3d point correspondences.
///
/// Inspired by: <https://github.com/PoseLib/PoseLib/blob/56d158f744d3561b0b70174e6d8ca9a7fc9bd9c1/PoseLib/solvers/homography_4pt.cc#L73C4-L76C20>
///
/// The homography matrix is computed by solving the linear system of equations.
///
/// # Arguments
///
/// * `x1` - The source 3d points with shape (4, 3).
/// * `x2` - The destination 3d points with shape (4, 3).
/// * `homo` - The output homography matrix from src to dst with shape (3, 3).
/// * `check_cheirality` - Whether to check the cheirality condition.
pub fn homography_4pt3d(
    x1: &[[f64; 3]; 4],
    x2: &[[f64; 3]; 4],
    homo: &mut [[f64; 3]; 3],
    check_cheirality: bool,
) -> Result<(), HomographyError> {
    if check_cheirality {
        let mut p = [0.0; 3];
        linalg::cross_vec3(&x1[0], &x1[1], &mut p);

        let mut q = [0.0; 3];
        linalg::cross_vec3(&x2[0], &x2[1], &mut q);

        if (linalg::dot_product3(&p, &x1[2]) * linalg::dot_product3(&q, &x2[2])) < 0.0 {
            return Err(HomographyError::CheiralityCheckFailed);
        }

        if linalg::dot_product3(&p, &x1[3]) * linalg::dot_product3(&q, &x2[3]) < 0.0 {
            return Err(HomographyError::CheiralityCheckFailed);
        }

        linalg::cross_vec3(&x1[2], &x1[3], &mut p);
        linalg::cross_vec3(&x2[2], &x2[3], &mut q);

        if (linalg::dot_product3(&p, &x1[0]) * linalg::dot_product3(&q, &x2[0])) < 0.0 {
            return Err(HomographyError::CheiralityCheckFailed);
        }

        if (linalg::dot_product3(&p, &x1[1]) * linalg::dot_product3(&q, &x2[1])) < 0.0 {
            return Err(HomographyError::CheiralityCheckFailed);
        }
    }

    let mut m_mat = faer::Mat::<f64>::zeros(8, 9);
    for i in 0..4 {
        let (x1_0, x1_1, x1_2) = (x1[i][0], x1[i][1], x1[i][2]);
        let (x2_0, x2_1, x2_2) = (x2[i][0], x2[i][1], x2[i][2]);
        unsafe {
            m_mat.write_unchecked(2 * i, 0, x2_2 * x1_0);
            m_mat.write_unchecked(2 * i, 1, x2_2 * x1_1);
            m_mat.write_unchecked(2 * i, 2, x2_2 * x1_2);
            m_mat.write_unchecked(2 * i, 6, -x2_0 * x1_0);
            m_mat.write_unchecked(2 * i, 7, -x2_0 * x1_1);
            m_mat.write_unchecked(2 * i, 8, -x2_0 * x1_2);

            m_mat.write_unchecked(2 * i + 1, 3, x2_2 * x1_0);
            m_mat.write_unchecked(2 * i + 1, 4, x2_2 * x1_1);
            m_mat.write_unchecked(2 * i + 1, 5, x2_2 * x1_2);
            m_mat.write_unchecked(2 * i + 1, 6, -x2_1 * x1_0);
            m_mat.write_unchecked(2 * i + 1, 7, -x2_1 * x1_1);
            m_mat.write_unchecked(2 * i + 1, 8, -x2_1 * x1_2);
        }
    }

    // solve -> h_mat: 8x1
    let h_mat = m_mat
        .submatrix(0, 0, 8, 8)
        .partial_piv_lu()
        .solve(-m_mat.submatrix(0, 8, 8, 1));
    let h = h_mat.col(0);

    // copy to homography matrix
    // NOTE: it contains 8 elements as transposed
    homo[0] = [h[0], h[1], h[2]];
    homo[1] = [h[3], h[4], h[5]];
    homo[2] = [h[6], h[7], 1.0];

    let det = linalg::det_mat33(homo);
    if det.abs() < 1e-8 {
        return Err(HomographyError::SingularMatrix);
    }

    Ok(())
}

/// Decompose a homography into candidate (R, t) pairs using a full 8-solution method.
///
/// This mirrors the Faugeras et al. motion-from-homography decomposition and returns
/// up to 8 candidates (some can be invalid if singular values are near-degenerate).
pub fn decompose_homography(h: &Mat3F64, k1: &Mat3F64, k2: &Mat3F64) -> Vec<(Mat3F64, Vec3F64)> {
    let a = k2.inverse() * *h * *k1;
    let svd = svd3_f64(&a);
    let u = *svd.u();
    let v = *svd.v();
    let vt = v.transpose();
    let w = svd.s();

    let s = u.determinant() * vt.determinant();
    let d1 = w.x_axis.x;
    let d2 = w.y_axis.y;
    let d3 = w.z_axis.z;

    if d1 / d2 < 1.00001 || d2 / d3 < 1.00001 {
        return Vec::new();
    }

    let aux1 = ((d1 * d1 - d2 * d2) / (d1 * d1 - d3 * d3)).sqrt();
    let aux3 = ((d2 * d2 - d3 * d3) / (d1 * d1 - d3 * d3)).sqrt();
    let x1 = [aux1, aux1, -aux1, -aux1];
    let x3 = [aux3, -aux3, aux3, -aux3];

    let aux_stheta = ((d1 * d1 - d2 * d2) * (d2 * d2 - d3 * d3)).sqrt() / ((d1 + d3) * d2);
    let ctheta = (d2 * d2 + d1 * d3) / ((d1 + d3) * d2);
    let stheta = [aux_stheta, -aux_stheta, -aux_stheta, aux_stheta];

    let mut out = Vec::with_capacity(8);
    for i in 0..4 {
        let rp = Mat3F64::from_cols(
            Vec3F64::new(ctheta, 0.0, stheta[i]),
            Vec3F64::new(0.0, 1.0, 0.0),
            Vec3F64::new(-stheta[i], 0.0, ctheta),
        );
        let r = (u * rp * vt) * s;

        let mut tp = Vec3F64::new(x1[i], 0.0, -x3[i]);
        tp *= d1 - d3;
        let mut t = u * tp;
        let t_norm = t.length();
        if t_norm > 1e-12 {
            t /= t_norm;
            out.push((r, t));
        }
    }

    let aux_sphi = ((d1 * d1 - d2 * d2) * (d2 * d2 - d3 * d3)).sqrt() / ((d1 - d3) * d2);
    let cphi = (d1 * d3 - d2 * d2) / ((d1 - d3) * d2);
    let sphi = [aux_sphi, -aux_sphi, -aux_sphi, aux_sphi];

    for i in 0..4 {
        let rp = Mat3F64::from_cols(
            Vec3F64::new(cphi, 0.0, sphi[i]),
            Vec3F64::new(0.0, -1.0, 0.0),
            Vec3F64::new(sphi[i], 0.0, -cphi),
        );
        let r = (u * rp * vt) * s;

        let mut tp = Vec3F64::new(x1[i], 0.0, x3[i]);
        tp *= d1 + d3;
        let mut t = u * tp;
        let t_norm = t.length();
        if t_norm > 1e-12 {
            t /= t_norm;
            out.push((r, t));
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_homography_4pt2d_identity() -> Result<(), HomographyError> {
        let x1 = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let x2 = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let expected = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let mut homo = [[0.0; 3]; 3];
        homography_4pt2d(&x1, &x2, &mut homo)?;

        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(homo[i][j], expected[i][j], epsilon = 1e-6);
            }
        }
        Ok(())
    }

    #[test]
    fn test_homography_4pt2d_transform() -> Result<(), HomographyError> {
        let x1 = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let (tx, ty) = (1.0, 1.0);
        let expected = [[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]];
        let mut x2 = [[0.0; 2]; 4];
        for i in 0..4 {
            let x1_i = [x1[i][0], x1[i][1], 1.0];
            let mut x2_i = [0.0; 3];
            linalg::mat33_mul_vec3(&expected, &x1_i, &mut x2_i);
            x2[i] = [x2_i[0], x2_i[1]];
        }
        let mut homo = [[0.0; 3]; 3];
        homography_4pt2d(&x1, &x2, &mut homo)?;

        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(homo[i][j], expected[i][j], epsilon = 1e-6);
            }
        }
        Ok(())
    }

    #[test]
    fn test_homography_4pt3d_identity() -> Result<(), HomographyError> {
        let x1 = [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ];
        let x2 = [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ];
        let mut homo = [[0.0; 3]; 3];
        let homo_expected = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        homography_4pt3d(&x1, &x2, &mut homo, true)?;
        assert_eq!(homo, homo_expected);
        Ok(())
    }

    #[test]
    fn test_homography_4pt3d_transform() -> Result<(), HomographyError> {
        let x1 = [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ];

        let (tx, ty) = (1.0, 1.0);
        let homo_trans = [[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]];

        let mut x2 = [[0.0; 3]; 4];
        for i in 0..4 {
            linalg::mat33_mul_vec3(&homo_trans, &x1[i], &mut x2[i]);
        }

        let mut homo = [[0.0; 3]; 3];
        homography_4pt3d(&x1, &x2, &mut homo, true)?;
        assert_eq!(homo, homo_trans);

        Ok(())
    }
}
