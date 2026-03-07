pub use kornia_algebra::linalg::homography::HomographyError;
use kornia_algebra::{Mat3F64, Vec2F64, Vec3F64};

/// Compute the homography matrix from four 2D point correspondences using SVD.
///
/// # Arguments
///
/// * `x1` - The source 2D points.
/// * `x2` - The destination 2D points.
/// * `homo` - Pre-allocated 3x3 homography matrix.
///
/// # Returns
///
/// `Ok(())` on success, or a [`HomographyError`] if the matrix is singular.
///
/// # Errors
///
/// Returns [`HomographyError::SingularMatrix`] if the points are collinear or the system is degenerate.
///
/// # Example
///
/// ```rust
/// use kornia_algebra::{Vec2F64, Mat3F64};
/// use kornia_3d::pose::homography_4pt2d;
///
/// let x1 = [Vec2F64::new(0.0, 0.0), Vec2F64::new(1.0, 0.0), Vec2F64::new(1.0, 1.0), Vec2F64::new(0.0, 1.0)];
/// let x2 = [Vec2F64::new(0.0, 0.0), Vec2F64::new(1.0, 0.0), Vec2F64::new(1.0, 1.0), Vec2F64::new(0.0, 1.0)];
/// let mut h = Mat3F64::IDENTITY;
/// homography_4pt2d(&x1, &x2, &mut h).unwrap();
/// ```
pub fn homography_4pt2d(
    x1: &[Vec2F64; 4],
    x2: &[Vec2F64; 4],
    homo: &mut Mat3F64,
) -> Result<(), HomographyError> {
    *homo = kornia_algebra::linalg::homography::homography_2d_dlt_svd_f64(x1, x2)?;
    Ok(())
}

/// Compute the homography matrix from four 3D point correspondences using LU decomposition.
///
/// Inspired by: <https://github.com/PoseLib/PoseLib/blob/56d158f744d3561b0b70174e6d8ca9a7fc9bd9c1/PoseLib/solvers/homography_4pt.cc#L73C4-L76C20>
///
/// # Arguments
///
/// * `x1` - The source 3D points.
/// * `x2` - The destination 3D points.
/// * `homo` - Pre-allocated 3x3 homography matrix.
/// * `check_cheirality` - Whether to check the cheirality condition.
///
/// # Returns
///
/// `Ok(())` on success, or a [`HomographyError`] if singular or cheirality check fails.
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
/// use kornia_3d::pose::homography_4pt3d;
///
/// let x1 = [Vec3F64::new(0.0, 0.0, 1.0), Vec3F64::new(1.0, 0.0, 1.0), Vec3F64::new(1.0, 1.0, 1.0), Vec3F64::new(0.0, 1.0, 1.0)];
/// let x2 = [Vec3F64::new(0.0, 0.0, 1.0), Vec3F64::new(1.0, 0.0, 1.0), Vec3F64::new(1.0, 1.0, 1.0), Vec3F64::new(0.0, 1.0, 1.0)];
/// let mut h = Mat3F64::IDENTITY;
/// homography_4pt3d(&x1, &x2, &mut h, true).unwrap();
/// ```
pub fn homography_4pt3d(
    x1: &[Vec3F64; 4],
    x2: &[Vec3F64; 4],
    homo: &mut Mat3F64,
    check_cheirality: bool,
) -> Result<(), HomographyError> {
    *homo = kornia_algebra::linalg::homography::homography_3d_dlt_lu_f64(x1, x2, check_cheirality)?;
    Ok(())
}

/// Decompose a homography into candidate (R, t) pairs using a full 8-solution method.
///
/// This mirrors the Faugeras et al. motion-from-homography decomposition and returns
/// up to 8 candidates (some can be invalid if singular values are near-degenerate).
pub fn decompose_homography(h: &Mat3F64, k1: &Mat3F64, k2: &Mat3F64) -> Vec<(Mat3F64, Vec3F64)> {
    let a = k2.inverse() * *h * *k1;
    let svd = kornia_algebra::linalg::svd::svd3_f64(&a);
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
