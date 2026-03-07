use crate::linalg;
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
    if let Some(h) = kornia_algebra::linalg::homography::homography_2d_dlt_svd_f64(x1, x2) {
        // copy to homography matrix
        homo[0] = [h.x_axis().x, h.y_axis().x, h.z_axis().x];
        homo[1] = [h.x_axis().y, h.y_axis().y, h.z_axis().y];
        homo[2] = [h.x_axis().z, h.y_axis().z, h.z_axis().z];

        // normalize the homography matrix
        linalg::normalize_mat33_inplace(homo);

        if linalg::det_mat33(homo).abs() < 1e-8 {
            return Err(HomographyError::SingularMatrix);
        }

        Ok(())
    } else {
        Err(HomographyError::SingularMatrix)
    }
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

    if let Some(h) = kornia_algebra::linalg::homography::homography_3d_dlt_lu_f64(x1, x2) {
        // copy to homography matrix
        homo[0] = [h.x_axis().x, h.y_axis().x, h.z_axis().x];
        homo[1] = [h.x_axis().y, h.y_axis().y, h.z_axis().y];
        homo[2] = [h.x_axis().z, h.y_axis().z, h.z_axis().z];

        let det = linalg::det_mat33(homo);
        if det.abs() < 1e-8 {
            return Err(HomographyError::SingularMatrix);
        }

        Ok(())
    } else {
        Err(HomographyError::SingularMatrix)
    }
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
