use crate::linalg;
use faer::prelude::SpSolver;
use kornia_algebra::{linalg::svd::svd3_f64, Mat3F64, Vec2F64, Vec3F64};
use nalgebra::SMatrix;

/// Error type for homography estimation.
#[derive(thiserror::Error, Debug)]
pub enum HomographyError {
    /// Homography matrix is singular or near-singular.
    #[error("Homography determinant too small (near-singular matrix)")]
    SingularMatrix,

    /// Cheirality constraint violated.
    #[error("Cheirality check failed")]
    CheiralityCheckFailed,

    /// Input correspondences are invalid or insufficient.
    #[error("Need at least {required} correspondences with equal lengths")]
    InvalidInput {
        /// Minimum required correspondences.
        required: usize,
    },
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
    // Fix h[2][2]=1 and solve the resulting 8×8 system via LU — a minimal 4-point
    // sample has a 1-D null space, so fixing one scale gives an exactly-determined
    // linear system. ~10× cheaper than the full 8×9 SVD for the same H.
    // This is the hot path inside `ransac_homography`.
    let mut mat_a = faer::Mat::<f64>::zeros(8, 8);
    let mut rhs = faer::Mat::<f64>::zeros(8, 1);
    for i in 0..4 {
        let (u, v) = (x1[i][0], x1[i][1]);
        let (up, vp) = (x2[i][0], x2[i][1]);
        unsafe {
            mat_a.write_unchecked(2 * i, 0, u);
            mat_a.write_unchecked(2 * i, 1, v);
            mat_a.write_unchecked(2 * i, 2, 1.0);
            mat_a.write_unchecked(2 * i, 6, -up * u);
            mat_a.write_unchecked(2 * i, 7, -up * v);
            rhs.write_unchecked(2 * i, 0, up);

            mat_a.write_unchecked(2 * i + 1, 3, u);
            mat_a.write_unchecked(2 * i + 1, 4, v);
            mat_a.write_unchecked(2 * i + 1, 5, 1.0);
            mat_a.write_unchecked(2 * i + 1, 6, -vp * u);
            mat_a.write_unchecked(2 * i + 1, 7, -vp * v);
            rhs.write_unchecked(2 * i + 1, 0, vp);
        }
    }

    let sol = mat_a.partial_piv_lu().solve(&rhs);
    let h = sol.col(0);

    homo[0] = [h[0], h[1], h[2]];
    homo[1] = [h[3], h[4], h[5]];
    homo[2] = [h[6], h[7], 1.0];

    if linalg::det_mat33(homo).abs() < 1e-8 {
        return Err(HomographyError::SingularMatrix);
    }

    Ok(())
}

/// Hartley-normalize a point set: translate centroid to origin, scale so the mean
/// distance to origin is sqrt(2). Returns the normalized points and the 3x3 transform
/// T such that `p_normalized = T * p_homogeneous`.
fn hartley_normalize(pts: &[Vec2F64]) -> (Vec<[f64; 2]>, Mat3F64) {
    let n = pts.len() as f64;
    let cx = pts.iter().map(|p| p.x).sum::<f64>() / n;
    let cy = pts.iter().map(|p| p.y).sum::<f64>() / n;
    let mean_dist = pts
        .iter()
        .map(|p| ((p.x - cx).powi(2) + (p.y - cy).powi(2)).sqrt())
        .sum::<f64>()
        / n;
    // If all points coincide, skip scaling — let the DLT error downstream.
    let s = if mean_dist > 1e-12 {
        std::f64::consts::SQRT_2 / mean_dist
    } else {
        1.0
    };
    let normed: Vec<[f64; 2]> = pts
        .iter()
        .map(|p| [(p.x - cx) * s, (p.y - cy) * s])
        .collect();
    // T = [[s, 0, -s*cx], [0, s, -s*cy], [0, 0, 1]] — column-major for Mat3F64.
    let t = Mat3F64::from_cols(
        Vec3F64::new(s, 0.0, 0.0),
        Vec3F64::new(0.0, s, 0.0),
        Vec3F64::new(-s * cx, -s * cy, 1.0),
    );
    (normed, t)
}

/// Compute the homography matrix from N≥4 2D point correspondences using the
/// Direct Linear Transform (DLT) with Hartley normalization.
///
/// Equivalent to `cv2.findHomography(src, dst, method=0)` — pure least-squares,
/// no outlier rejection. Use `ransac_homography` when the correspondences contain
/// outliers.
///
/// # Arguments
///
/// * `x1` - Source 2D points (len ≥ 4).
/// * `x2` - Destination 2D points (must match `x1.len()`).
///
/// # Returns
///
/// The 3x3 homography mapping `x1 → x2` (normalized so H[2][2] ≈ 1 when non-degenerate).
pub fn homography_dlt(x1: &[Vec2F64], x2: &[Vec2F64]) -> Result<Mat3F64, HomographyError> {
    if x1.len() != x2.len() || x1.len() < 4 {
        return Err(HomographyError::InvalidInput { required: 4 });
    }
    let n = x1.len();

    // Hartley normalization — raw pixel coords make the DLT system ill-conditioned
    // (entries in M scale as x² → 1e10 range for 1080p). Normalizing brings the
    // condition number back to sensible levels and is standard practice
    // (Hartley 1997, "In defense of the 8-point algorithm" — same trick applies here).
    let (x1n, t1) = hartley_normalize(x1);
    let (x2n, t2) = hartley_normalize(x2);

    // DLT system: 2N rows × 9 cols, minimize ‖A h‖ subject to ‖h‖ = 1.
    //
    // Instead of SVD on the 2N×9 A, accumulate M = AᵀA (9×9 symmetric) via
    // streaming rank-1 outer products and pick the eigenvector of M's smallest
    // eigenvalue. Equivalent because the right-singular vector of σ_min(A) is
    // the eigenvector of λ_min(AᵀA). Hartley normalization keeps κ(A) ≈ 10²,
    // so κ(AᵀA) ≈ 10⁴ — well within f64 headroom. O(81·N + 9³) vs O(N·81 + N²·9)
    // for the SVD; drops the LO-RANSAC refit from ~1ms to ≪0.1ms at N≈200.
    let mut m = [[0.0f64; 9]; 9];
    for i in 0..n {
        let (u, v) = (x1n[i][0], x1n[i][1]);
        let (up, vp) = (x2n[i][0], x2n[i][1]);
        let r0 = [u, v, 1.0, 0.0, 0.0, 0.0, -up * u, -up * v, -up];
        let r1 = [0.0, 0.0, 0.0, u, v, 1.0, -vp * u, -vp * v, -vp];
        for a in 0..9 {
            let r0a = r0[a];
            let r1a = r1[a];
            for b in a..9 {
                m[a][b] += r0a * r0[b] + r1a * r1[b];
            }
        }
    }
    for a in 0..9 {
        for b in 0..a {
            m[a][b] = m[b][a];
        }
    }

    let mtm = SMatrix::<f64, 9, 9>::from_fn(|r, c| m[r][c]);
    let eig = mtm.symmetric_eigen();
    let mut min_idx = 0usize;
    let mut min_val = eig.eigenvalues[0];
    for i in 1..9 {
        if eig.eigenvalues[i] < min_val {
            min_val = eig.eigenvalues[i];
            min_idx = i;
        }
    }
    let hc = eig.eigenvectors.column(min_idx);
    let h = [
        hc[0], hc[1], hc[2], hc[3], hc[4], hc[5], hc[6], hc[7], hc[8],
    ];

    // H_normalized is in the normalized frame: x2n = Hn * x1n.
    // Undo normalization: H = T2⁻¹ · Hn · T1.
    let hn = Mat3F64::from_cols(
        Vec3F64::new(h[0], h[3], h[6]),
        Vec3F64::new(h[1], h[4], h[7]),
        Vec3F64::new(h[2], h[5], h[8]),
    );
    let h_pixel = t2.inverse() * hn * t1;

    // Normalize so H[2][2] = 1 (matches the 4pt2d solver output convention).
    // cols[c * 3 + r] = H[r][c] (column-major glam storage).
    let cols = h_pixel.to_cols_array();
    let mut h_out = [[0.0; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            h_out[r][c] = cols[c * 3 + r];
        }
    }
    linalg::normalize_mat33_inplace(&mut h_out);
    if linalg::det_mat33(&h_out).abs() < 1e-8 {
        return Err(HomographyError::SingularMatrix);
    }

    Ok(Mat3F64::from_cols(
        Vec3F64::new(h_out[0][0], h_out[1][0], h_out[2][0]),
        Vec3F64::new(h_out[0][1], h_out[1][1], h_out[2][1]),
        Vec3F64::new(h_out[0][2], h_out[1][2], h_out[2][2]),
    ))
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
    fn test_homography_dlt_recovers_known_h() -> Result<(), HomographyError> {
        // Random-looking H (rotation + translation + mild perspective).
        let h_true = [
            [1.1, -0.2, 30.0],
            [0.15, 0.95, -10.0],
            [1e-4, 2e-4, 1.0],
        ];
        let x1_raw = [
            [10.0, 20.0],
            [200.0, 50.0],
            [50.0, 300.0],
            [400.0, 350.0],
            [150.0, 150.0],
            [350.0, 100.0],
            [80.0, 400.0],
            [450.0, 250.0],
        ];
        let mut x1 = Vec::with_capacity(8);
        let mut x2 = Vec::with_capacity(8);
        for p in &x1_raw {
            let u = [p[0], p[1], 1.0];
            let mut v = [0.0; 3];
            linalg::mat33_mul_vec3(&h_true, &u, &mut v);
            x1.push(Vec2F64::new(p[0], p[1]));
            x2.push(Vec2F64::new(v[0] / v[2], v[1] / v[2]));
        }
        let h_est = homography_dlt(&x1, &x2)?;
        // Reproject x1 through h_est, compare to x2 in pixel space.
        let cols = h_est.to_cols_array();
        let h = [
            [cols[0], cols[3], cols[6]],
            [cols[1], cols[4], cols[7]],
            [cols[2], cols[5], cols[8]],
        ];
        for (p1, p2) in x1.iter().zip(x2.iter()) {
            let u = [p1.x, p1.y, 1.0];
            let mut v = [0.0; 3];
            linalg::mat33_mul_vec3(&h, &u, &mut v);
            let (ex, ey) = (v[0] / v[2], v[1] / v[2]);
            assert_relative_eq!(ex, p2.x, epsilon = 1e-6);
            assert_relative_eq!(ey, p2.y, epsilon = 1e-6);
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
