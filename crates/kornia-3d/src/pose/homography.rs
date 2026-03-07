pub use kornia_algebra::linalg::homography::HomographyError;
use kornia_algebra::{Mat3F64, Vec3F64};

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
