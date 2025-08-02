//! Efficient Perspective-n-Point (EPnP) solver
//! Paper: https://www.tugraz.at/fileadmin/user_upload/Institute/ICG/Images/team_lepetit/publications/lepetit_ijcv08.pdf
//! Reference: https://github.com/opencv/opencv/blob/4.x/modules/calib3d/src/epnp.cpp

use crate::ops::{compute_centroid, gauss_newton};
use crate::types::{NumericTol, PnPError, PnPResult, PnPSolver};
use glam::{Mat3, Mat3A, Vec3};
use kornia_lie::so3::SO3;
use kornia_linalg::rigid::umeyama;
use kornia_linalg::svd::svd3;
use nalgebra::{DMatrix, DVector, Vector4};

/// Marker type representing the Efficient PnP algorithm.
pub struct EPnP;

impl PnPSolver for EPnP {
    type Param = EPNPParams;

    fn solve(
        points_world: &[[f32; 3]],
        points_image: &[[f32; 2]],
        k: &[[f32; 3]; 3],
        params: &Self::Param,
    ) -> Result<PnPResult, PnPError> {
        solve_epnp(points_world, points_image, k, params)
    }
}

/// Parameters controlling the EPnP solver.
#[derive(Debug, Clone, Default)]
pub struct EPNPParams {
    /// Shared numeric tolerances.
    pub tol: NumericTol,
}

/// Solve Perspective-n-Point (EPnP).
///
/// # Arguments
/// * `points_world` – 3-D coordinates in the world frame, shape *(N,3)* with `N≥4`.
/// * `points_image` – Corresponding pixel coordinates, shape *(N,2)*.
/// * `k` – Camera intrinsics matrix.
///
/// # Returns
/// Tuple *(R, t, rvec)* where
/// * `R` – 3×3 rotation **world → camera**,
/// * `t` – 3-vector translation,
/// * `rvec` – Rodrigues axis-angle representation of `R`.
pub fn solve_epnp(
    points_world: &[[f32; 3]],
    points_image: &[[f32; 2]],
    k: &[[f32; 3]; 3],
    params: &EPNPParams,
) -> Result<PnPResult, PnPError> {
    let n = points_world.len();
    if n != points_image.len() {
        return Err(PnPError::MismatchedArrayLengths(n, points_image.len()));
    }
    if n < 4 {
        return Err(PnPError::InsufficientCorrespondences {
            required: 4,
            actual: n,
        });
    }

    let cw = select_control_points(points_world);

    let alphas = compute_barycentric(points_world, &cw, params.tol.eps);

    // Build the 2N×12 design matrix M
    let m_rows = build_m(&alphas, points_image, k);

    let m_flat: Vec<f32> = m_rows.iter().flat_map(|row| row.iter()).cloned().collect();
    let m_mat = DMatrix::<f32>::from_row_slice(2 * n, 12, &m_flat);

    // Null-space of M (4 right-singular vectors associated with smallest singular values)
    let svd = m_mat.svd(true, true);
    let v_t = match svd.v_t {
        Some(v) => v,
        None => return Err(PnPError::SvdFailed("Failed to compute V^T".to_string())),
    };
    let cols = 12;
    let start_col = cols - 4;
    let null4 = v_t.rows(start_col, 4).transpose(); // shape 12×4

    // Build helper matrices for beta initialisation
    let l = build_l6x10(&null4);
    let rho = rho_ctrlpts(&cw);

    // Convert L and rho to nalgebra types once.
    let rho_vec = DVector::<f32>::from_column_slice(&rho);

    let mut betas: Vec<[f32; 4]> = Vec::new();

    betas.extend(
        [
            estimate_beta([0, 1, 3, 6], &l, &rho_vec, params.tol.svd),
            estimate_beta([0, 1, 2], &l, &rho_vec, params.tol.svd),
            estimate_beta([0, 1, 2, 3, 4], &l, &rho_vec, params.tol.svd),
        ]
        .into_iter()
        .flatten(),
    );

    let betas_refined: Vec<[f32; 4]> = betas
        .iter()
        .map(|&b| gauss_newton(b, &null4, &rho))
        .collect();

    let mut best_err = f32::INFINITY;
    let mut best_r = [[1.0; 3]; 3];
    let mut best_t = [0.0; 3];

    for bet in &betas_refined {
        let (r_c, t_c) = pose_from_betas(bet, &null4, &cw, &alphas);
        let err = rmse_px(points_world, points_image, &r_c, &t_c, k);
        if err < best_err {
            best_err = err;
            best_r = r_c;
            best_t = t_c;
        }
    }

    let mat = Mat3A::from_cols_array(&[
        best_r[0][0],
        best_r[1][0],
        best_r[2][0],
        best_r[0][1],
        best_r[1][1],
        best_r[2][1],
        best_r[0][2],
        best_r[1][2],
        best_r[2][2],
    ]);
    let rvec_f32 = SO3::from_matrix(&mat).log();
    let rvec = [rvec_f32.x, rvec_f32.y, rvec_f32.z];

    Ok(PnPResult {
        rotation: best_r,
        translation: best_t,
        rvec,
        reproj_rmse: Some(best_err),
        num_iterations: None,
        converged: Some(true),
    })
}

/// Compute pose (R, t) from a set of betas using the null-space vectors.
fn pose_from_betas(
    betas: &[f32; 4],
    null4: &DMatrix<f32>, // 12×4 matrix (V)
    cw: &[[f32; 3]; 4],   // control points in world frame
    alphas: &[[f32; 4]],  // barycentric coordinates for each world point
) -> ([[f32; 3]; 3], [f32; 3]) {
    let beta_vec = Vector4::from_column_slice(betas);
    let cc_flat = null4 * beta_vec; // 12×1 vector

    let mut cc: [[f32; 3]; 4] = [[0.0; 3]; 4];
    for i in 0..4 {
        cc[i][0] = cc_flat[3 * i];
        cc[i][1] = cc_flat[3 * i + 1];
        cc[i][2] = cc_flat[3 * i + 2];
    }

    let a0 = alphas[0];
    let mut pc0 = [0.0; 3];
    for j in 0..4 {
        pc0[0] += a0[j] * cc[j][0];
        pc0[1] += a0[j] * cc[j][1];
        pc0[2] += a0[j] * cc[j][2];
    }

    if pc0[2] < 0.0 {
        for pt in &mut cc {
            pt[0] *= -1.0;
            pt[1] *= -1.0;
            pt[2] *= -1.0;
        }
    }

    // Convert arrays to Vec3 for consistency with glam usage
    let cw_vec3: Vec<Vec3> = cw.iter().map(|p| Vec3::new(p[0], p[1], p[2])).collect();
    let cc_vec3: Vec<Vec3> = cc.iter().map(|p| Vec3::new(p[0], p[1], p[2])).collect();

    let (r, t, _s) = umeyama(&cw_vec3, &cc_vec3);

    (r, t)
}

/// Root-mean-square reprojection error in pixels.
fn rmse_px(
    points_world: &[[f32; 3]],
    points_image: &[[f32; 2]],
    r: &[[f32; 3]; 3],
    t: &[f32; 3],
    k: &[[f32; 3]; 3],
) -> f32 {
    assert_eq!(points_world.len(), points_image.len());

    let fx = k[0][0];
    let fy = k[1][1];
    let cx = k[0][2];
    let cy = k[1][2];

    let mut sum_sq = 0.0;
    let n = points_world.len() as f32;

    for (p, &img) in points_world.iter().zip(points_image.iter()) {
        // Camera-frame coordinates: Pc = R * Pw + t
        let x_c = r[0][0] * p[0] + r[0][1] * p[1] + r[0][2] * p[2] + t[0];
        let y_c = r[1][0] * p[0] + r[1][1] * p[1] + r[1][2] * p[2] + t[1];
        let z_c = r[2][0] * p[0] + r[2][1] * p[1] + r[2][2] * p[2] + t[2];

        let inv_z = 1.0 / z_c;
        let u_hat = fx * x_c * inv_z + cx;
        let v_hat = fy * y_c * inv_z + cy;

        let du = u_hat - img[0];
        let dv = v_hat - img[1];
        sum_sq += du * du + dv * dv;
    }

    (sum_sq / n).sqrt()
}

fn select_control_points(points_world: &[[f32; 3]]) -> [[f32; 3]; 4] {
    let n = points_world.len();
    let c = compute_centroid(points_world);

    // Compute covariance using glam for consistency
    let mut cov_mat = Mat3::ZERO;
    for p in points_world {
        let diff = Vec3::new(p[0] - c[0], p[1] - c[1], p[2] - c[2]);
        let outer_product = Mat3::from_cols(
            Vec3::new(diff.x * diff.x, diff.y * diff.x, diff.z * diff.x),
            Vec3::new(diff.x * diff.y, diff.y * diff.y, diff.z * diff.y),
            Vec3::new(diff.x * diff.z, diff.y * diff.z, diff.z * diff.z),
        );
        cov_mat += outer_product;
    }
    cov_mat *= 1.0 / n as f32;

    let svd = svd3(&cov_mat);
    let v = svd.v();
    let s = svd.s(); // diagonal matrix of singular values (eigenvalues)

    let s_diag = [s.x_axis.x, s.y_axis.y, s.z_axis.z];
    let mut axes_sig: Vec<(f32, Vec3)> = vec![
        (s_diag[0].sqrt(), v.x_axis),
        (s_diag[1].sqrt(), v.y_axis),
        (s_diag[2].sqrt(), v.z_axis),
    ];
    axes_sig.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    // Assemble control points: centroid + principal-axis displacements
    let mut cw = [[0.0; 3]; 4];
    cw[0] = c;

    for (i, (sigma, axis)) in axes_sig.iter().enumerate() {
        cw[i + 1][0] = c[0] + sigma * axis.x;
        cw[i + 1][1] = c[1] + sigma * axis.y;
        cw[i + 1][2] = c[2] + sigma * axis.z;
    }

    cw
}

/// Compute barycentric coordinates of world-space points with respect to the
/// 4 control points returned by `select_control_points`.
///
/// # Arguments
/// * `points_world` – World points *(N,3)*.
/// * `cw` – Control points *(4,3)*.
/// * `eps` – Threshold that decides whether the control-point tetrahedron is
///   degenerate. If the determinant of the 3-by-3 matrix built from `cw` is
///   smaller than `eps`, a Moore–Penrose pseudo-inverse is used instead of the exact inverse.
///
/// # Returns
/// Vector of length *N* where each element is `[α0, α1, α2, α3]` such that
/// `α0 + α1 + α2 + α3 = 1` and `pw_i = Σ αj Cw_j`.
fn compute_barycentric(points_world: &[[f32; 3]], cw: &[[f32; 3]; 4], eps: f32) -> Vec<[f32; 4]> {
    // Build B = [C1 - C0, C2 - C0, C3 - C0].
    let c0 = Vec3::new(cw[0][0], cw[0][1], cw[0][2]);
    let d1 = Vec3::new(cw[1][0] - c0.x, cw[1][1] - c0.y, cw[1][2] - c0.z);
    let d2 = Vec3::new(cw[2][0] - c0.x, cw[2][1] - c0.y, cw[2][2] - c0.z);
    let d3 = Vec3::new(cw[3][0] - c0.x, cw[3][1] - c0.y, cw[3][2] - c0.z);

    let b = Mat3::from_cols(d1, d2, d3);

    // Invert or pseudo-invert B.
    let b_inv = if b.determinant().abs() > eps {
        // Safe to invert.
        b.inverse()
    } else {
        // Moore–Penrose pseudo-inverse: B⁺ = V Σ⁺ Uᵀ
        let svd = svd3(&b);
        let u = *svd.u();
        let v_mat = *svd.v();
        let s_mat = *svd.s();
        let s_diag = [s_mat.x_axis.x, s_mat.y_axis.y, s_mat.z_axis.z];
        let inv_diag = Vec3::new(
            if s_diag[0].abs() > eps { 1.0 / s_diag[0] } else { 0.0 },
            if s_diag[1].abs() > eps { 1.0 / s_diag[1] } else { 0.0 },
            if s_diag[2].abs() > eps { 1.0 / s_diag[2] } else { 0.0 },
        );
        let sigma_inv = Mat3::from_diagonal(inv_diag);
        v_mat * sigma_inv * u.transpose()
    };

    // Compute barycentric coordinates.
    points_world
        .iter()
        .map(|p| {
            let diff = Vec3::new(p[0] - c0.x, p[1] - c0.y, p[2] - c0.z);
            let lamb = b_inv * diff;
            [1.0 - (lamb.x + lamb.y + lamb.z), lamb.x, lamb.y, lamb.z]
        })
        .collect()
}

/// Construct the 2N×12 design matrix **M** used by EPnP.
///
/// * `alphas` – Barycentric coordinates for each world point, produced by
///   [`compute_barycentric`]; shape *(N,4)*.
/// * `points_image`     – Pixel coordinates for each correspondence; shape *(N,2)*.
/// * `k`      – Camera intrinsics 3×3 matrix.
///
/// The output is a vector of length `2*N` where each element is the 12-vector
/// corresponding to a row of **M**.
fn build_m(alphas: &[[f32; 4]], points_image: &[[f32; 2]], k: &[[f32; 3]; 3]) -> Vec<[f32; 12]> {
    assert_eq!(
        alphas.len(),
        points_image.len(),
        "alphas and uv must have the same length"
    );
    let n = alphas.len();

    let fu = k[0][0];
    let fv = k[1][1];
    let uc = k[0][2];
    let vc = k[1][2];

    // Pre-allocate 2N rows of zeros.
    let mut m = vec![[0.0f32; 12]; 2 * n];

    for (i, (a, &points_image_i)) in alphas.iter().zip(points_image.iter()).enumerate() {
        let u = points_image_i[0];
        let v = points_image_i[1];

        let row_x = 2 * i;
        let row_y = row_x + 1;

        for (j, &alpha) in a.iter().enumerate() {
            let base = 3 * j;
            m[row_x][base] = alpha * fu;
            m[row_x][base + 2] = alpha * (uc - u);
            m[row_y][base + 1] = alpha * fv;
            m[row_y][base + 2] = alpha * (vc - v);
        }
    }

    m
}

/// Build the 6×10 matrix **L** used in EPnP from the 4-dimensional null-space matrix `V` (shape 12×4).
fn build_l6x10(null4: &DMatrix<f32>) -> [[f32; 10]; 6] {
    // Re-ordered column indices (reverse order).
    let col_order = [3usize, 2, 1, 0];

    // v[i] is 4×3 matrix => Vec<[Vec3;4]>
    let mut v_cp: Vec<[Vec3; 4]> = Vec::with_capacity(4);

    for &c in &col_order {
        let col = null4.column(c);
        let mut blocks = [Vec3::ZERO; 4];
        for k in 0..4 {
            blocks[k] = Vec3::new(col[3 * k], col[3 * k + 1], col[3 * k + 2]);
        }
        v_cp.push(blocks);
    }

    // Differences between control-point vectors for each null-space component.
    let dv_arr: Vec<Vec<Vec3>> = (0..4)
        .map(|i| {
            CP_PAIRS
                .iter()
                .map(|&(a, b)| v_cp[i][a] - v_cp[i][b])
                .collect::<Vec<_>>()
        })
        .collect();

    let mut l = [[0.0f32; 10]; 6];
    for (j, _) in dv_arr[0].iter().enumerate() {
        l[j][0] = dv_arr[0][j].dot(dv_arr[0][j]);
        l[j][1] = 2.0 * dv_arr[0][j].dot(dv_arr[1][j]);
        l[j][2] = dv_arr[1][j].dot(dv_arr[1][j]);
        l[j][3] = 2.0 * dv_arr[0][j].dot(dv_arr[2][j]);
        l[j][4] = 2.0 * dv_arr[1][j].dot(dv_arr[2][j]);
        l[j][5] = dv_arr[2][j].dot(dv_arr[2][j]);
        l[j][6] = 2.0 * dv_arr[0][j].dot(dv_arr[3][j]);
        l[j][7] = 2.0 * dv_arr[1][j].dot(dv_arr[3][j]);
        l[j][8] = 2.0 * dv_arr[2][j].dot(dv_arr[3][j]);
        l[j][9] = dv_arr[3][j].dot(dv_arr[3][j]);
    }

    l
}

/// Extracts a 6×k `DMatrix` by picking the specified columns from the 6×10 `L` matrix.
fn l_submatrix(l: &[[f32; 10]; 6], cols: &[usize]) -> DMatrix<f32> {
    let data: Vec<f32> = cols
        .iter()
        .flat_map(|&c| (0..6).map(move |r| l[r][c]))
        .collect();
    DMatrix::<f32>::from_column_slice(6, cols.len(), &data)
}

/// Solve for a beta vector given a column subset of the 6×10 L matrix.
/// Returns `None` if the least-squares solve fails.
fn estimate_beta<const K: usize>(
    cols: [usize; K],
    l: &[[f32; 10]; 6],
    rho: &DVector<f32>,
    tol_svd: f32,
) -> Option<[f32; 4]> {
    let l_sub = l_submatrix(l, &cols);
    let sol = l_sub.svd(true, true).solve(rho, tol_svd).ok()?;
    let x = sol.column(0);

    match K {
        4 => Some([
            x[0].abs().sqrt().copysign(1.0), // sign handled below
            x[1] / x[0].abs().sqrt(),
            x[2] / x[0].abs().sqrt(),
            x[3] / x[0].abs().sqrt(),
        ])
        .map(|mut b| {
            if x[0] < 0.0 {
                for v in &mut b {
                    *v = -*v;
                }
            }
            b
        }),
        3 => {
            let mut beta = [0.0; 4];
            if x[0] < 0.0 {
                beta[0] = (-x[0]).sqrt();
                beta[2] = if x[2] > 0.0 { 0.0 } else { (-x[2]).sqrt() };
            } else {
                beta[0] = x[0].sqrt();
                beta[2] = if x[2] < 0.0 { 0.0 } else { x[2].sqrt() };
            }
            if x[1] < 0.0 {
                beta[0] = -beta[0];
            }
            Some(beta)
        }
        5 => {
            let mut beta = [0.0; 4];
            if x[0] < 0.0 {
                beta[0] = (-x[0]).sqrt();
                beta[1] = if x[2] > 0.0 { 0.0 } else { (-x[2]).sqrt() };
                beta[2] = x[3] / (-x[0]).sqrt();
            } else {
                beta[0] = x[0].sqrt();
                beta[1] = if x[2] < 0.0 { 0.0 } else { x[2].sqrt() };
                beta[2] = x[3] / x[0].sqrt();
            }
            if x[1] < 0.0 {
                beta[0] = -beta[0];
            }
            Some(beta)
        }
        _ => None,
    }
}

const CP_PAIRS: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

/// Compute the six squared distances (ρ vector) between the 4 control points.
fn rho_ctrlpts(cw: &[[f32; 3]; 4]) -> [f32; 6] {
    CP_PAIRS.map(|(i, j)| {
        cw[i]
            .iter()
            .zip(cw[j].iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f32>()
    })
}

#[cfg(test)]
mod solve_epnp_tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_solve_epnp() {
        // Hardcoded test data verified with OpenCV
        let points_world: [[f32; 3]; 6] = [
            [0.0315, 0.03333, -0.10409],
            [-0.0315, 0.03333, -0.10409],
            [0.0, -0.00102, -0.12977],
            [0.02646, -0.03167, -0.1053],
            [-0.02646, -0.031667, -0.1053],
            [0.0, 0.04515, -0.11033],
        ];

        // Image points (uv)
        let points_image: [[f32; 2]; 6] = [
            [722.96466, 502.0828],
            [669.88837, 498.61877],
            [707.0025, 478.48975],
            [728.05634, 447.56918],
            [682.6069, 443.91776],
            [696.4414, 511.96442],
        ];

        let k: [[f32; 3]; 3] = [[800.0, 0.0, 640.0], [0.0, 800.0, 480.0], [0.0, 0.0, 1.0]];

        let cw = select_control_points(&points_world);

        let alphas = compute_barycentric(&points_world, &cw, EPNPParams::default().tol.eps);

        for (p, alpha) in points_world.iter().zip(alphas.iter()) {
            let mut recon = [0.0; 3];
            for j in 0..4 {
                recon[0] += alpha[j] * cw[j][0];
                recon[1] += alpha[j] * cw[j][1];
                recon[2] += alpha[j] * cw[j][2];
            }
            for k in 0..3 {
                assert_relative_eq!(recon[k], p[k], epsilon = 1e-6);
            }

            assert_relative_eq!(alpha.iter().sum::<f32>(), 1.0, epsilon = 1e-9);
        }

        let m = build_m(&alphas, &points_image, &k);
        assert_eq!(m.len(), 2 * points_world.len());
        for row in &m {
            assert_eq!(row.len(), 12);
        }

        let fu = k[0][0];
        let fv = k[1][1];
        let uc = k[0][2];
        let vc = k[1][2];

        let u0 = points_image[0][0];
        let v0 = points_image[0][1];

        let mut expected_x = [0.0; 12];
        let mut expected_y = [0.0; 12];
        for j in 0..4 {
            let base = 3 * j;
            expected_x[base] = alphas[0][j] * fu;
            expected_x[base + 2] = alphas[0][j] * (uc - u0);
            expected_y[base + 1] = alphas[0][j] * fv;
            expected_y[base + 2] = alphas[0][j] * (vc - v0);
        }

        for k in 0..12 {
            assert_relative_eq!(m[0][k], expected_x[k], epsilon = 1e-9);
            assert_relative_eq!(m[1][k], expected_y[k], epsilon = 1e-9);
        }

        let result = EPnP::solve(&points_world, &points_image, &k, &EPNPParams::default())
            .expect("EPnP::solve should succeed");
        let r = result.rotation;
        let t = result.translation;
        let rvec = result.rvec;

        assert_relative_eq!(r[0][0], 0.6965054, epsilon = 1e-2);
        assert_relative_eq!(r[0][1], 0.07230615, epsilon = 1e-2);
        assert_relative_eq!(r[0][2], -0.71389916, epsilon = 1e-2);
        assert_relative_eq!(r[1][0], 0.2240602, epsilon = 1e-2);
        assert_relative_eq!(r[1][1], 0.92324643, epsilon = 1e-2);
        assert_relative_eq!(r[1][2], 0.31211066, epsilon = 1e-2);
        assert_relative_eq!(r[2][0], 0.6816724, epsilon = 1e-2);

        assert_relative_eq!(t[0], -0.00861299, epsilon = 1e-2);
        assert_relative_eq!(t[1], 0.02666388, epsilon = 1e-2);
        assert_relative_eq!(t[2], 1.014955, epsilon = 1e-2);

        assert_relative_eq!(rvec[0], -0.39580156, epsilon = 1e-2);
        assert_relative_eq!(rvec[1], -0.8011695, epsilon = 1e-2);
        assert_relative_eq!(rvec[2], 0.08711894, epsilon = 1e-2);
    }
}
