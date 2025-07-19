//! Efficient Perspective-n-Point (EPnP) solver
//! Paper: https://www.tugraz.at/fileadmin/user_upload/Institute/ICG/Images/team_lepetit/publications/lepetit_ijcv08.pdf
//! Reference: https://github.com/opencv/opencv/blob/4.x/modules/calib3d/src/epnp.cpp

use crate::ops::{compute_centroid, gauss_newton};
use crate::types::NumericTol;
use crate::types::PnPResult;
use crate::types::PnPSolver;
use glam::{Mat3, Mat3A, Vec3};
use kornia_lie::so3::SO3;
use kornia_linalg::rigid::umeyama;
use kornia_linalg::svd::svd3;
use nalgebra::{DMatrix, DVector, Vector3, Vector4};

/// Marker type representing the Efficient PnP algorithm.
pub struct EPnP;

impl PnPSolver for EPnP {
    type Param = EPNPParams;

    fn solve(
        world: &[[f64; 3]],
        image: &[[f64; 2]],
        k: &[[f64; 3]; 3],
        params: &Self::Param,
    ) -> Result<PnPResult, &'static str> {
        solve_epnp(world, image, k, params)
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
/// * `world_pts` – 3-D coordinates in the world frame, shape *(N,3)* with `N≥4`.
/// * `image_pts` – Corresponding pixel coordinates, shape *(N,2)*.
/// * `k` – Camera intrinsics matrix.
///
/// # Returns
/// Tuple *(R, t, rvec)* where
/// * `R` – 3×3 rotation **world → camera**,
/// * `t` – 3-vector translation,
/// * `rvec` – Rodrigues axis-angle representation of `R`.
pub fn solve_epnp(
    world_pts: &[[f64; 3]],
    image_pts: &[[f64; 2]],
    k: &[[f64; 3]; 3],
    params: &EPNPParams,
) -> Result<PnPResult, &'static str> {
    let n = world_pts.len();
    if n != image_pts.len() || n < 4 {
        return Err("EPnP requires ≥4 2D–3D correspondences");
    }

    // 1. Control points in world frame
    let cw = select_control_points(world_pts);

    let alphas = compute_barycentric(world_pts, &cw, params.tol.eps);

    // 3. Build the 2N×12 design matrix M
    let m_rows = build_m(&alphas, image_pts, k);

    // Flatten into a single Vec<f64> row-major for nalgebra
    let m_flat: Vec<f64> = m_rows.iter().flat_map(|row| row.iter()).cloned().collect();
    let m_mat = DMatrix::<f64>::from_row_slice(2 * n, 12, &m_flat);

    // 4. Null-space of M (4 right-singular vectors associated with smallest singular values)
    let svd = m_mat.svd(true, true);
    let v_t = svd
        .v_t
        .expect("SVD should return V^T since full_matrices=true");
    let cols = 12;
    let start_col = cols - 4;
    let null4 = v_t.rows(start_col, 4).transpose(); // shape 12×4

    // 5. Build helper matrices for beta initialisation
    let l = build_l6x10(&null4);
    let rho = rho_ctrlpts(&cw);

    // Convert L and rho to nalgebra types once.
    let rho_vec = DVector::<f64>::from_column_slice(&rho);

    let mut betas: Vec<[f64; 4]> = Vec::new();

    // -- Approx-1 (4 unknowns, columns 0,1,3,6)
    {
        let cols = [0, 1, 3, 6];
        let sub_data: Vec<f64> = cols
            .iter()
            .flat_map(|&c| (0..6).map(move |r| l[r][c]))
            .collect();
        let l_sub = DMatrix::<f64>::from_column_slice(6, 4, &sub_data);
        if let Ok(b4_mat) = l_sub.svd(true, true).solve(&rho_vec, params.tol.svd) {
            let b4_vec = b4_mat.column(0);
            let beta1 = if b4_vec[0] < 0.0 {
                [
                    (-b4_vec[0]).sqrt(),
                    -b4_vec[1] / (-b4_vec[0]).sqrt(),
                    -b4_vec[2] / (-b4_vec[0]).sqrt(),
                    -b4_vec[3] / (-b4_vec[0]).sqrt(),
                ]
            } else {
                [
                    b4_vec[0].sqrt(),
                    b4_vec[1] / b4_vec[0].sqrt(),
                    b4_vec[2] / b4_vec[0].sqrt(),
                    b4_vec[3] / b4_vec[0].sqrt(),
                ]
            };
            betas.push(beta1);
        }
    }

    // -- Approx-2 (3 unknowns, columns 0,1,2)
    {
        let cols = [0, 1, 2];
        let sub_data: Vec<f64> = cols
            .iter()
            .flat_map(|&c| (0..6).map(move |r| l[r][c]))
            .collect();
        let l_sub = DMatrix::<f64>::from_column_slice(6, 3, &sub_data);
        if let Ok(b3_mat) = l_sub.svd(true, true).solve(&rho_vec, params.tol.svd) {
            let b3_vec = b3_mat.column(0);
            let mut beta2 = [0.0; 4];
            if b3_vec[0] < 0.0 {
                beta2[0] = (-b3_vec[0]).sqrt();
                beta2[1] = 0.0;
                beta2[2] = if b3_vec[2] > 0.0 {
                    0.0
                } else {
                    (-b3_vec[2]).sqrt()
                };
            } else {
                beta2[0] = b3_vec[0].sqrt();
                beta2[1] = 0.0;
                beta2[2] = if b3_vec[2] < 0.0 {
                    0.0
                } else {
                    b3_vec[2].sqrt()
                };
            }
            if b3_vec[1] < 0.0 {
                beta2[0] = -beta2[0];
            }
            betas.push(beta2);
        }
    }

    // -- Approx-3 (5 unknowns, columns 0,1,2,3,4)
    {
        let cols = [0, 1, 2, 3, 4];
        let sub_data: Vec<f64> = cols
            .iter()
            .flat_map(|&c| (0..6).map(move |r| l[r][c]))
            .collect();
        let l_sub = DMatrix::<f64>::from_column_slice(6, 5, &sub_data);
        if let Ok(b5_mat) = l_sub.svd(true, true).solve(&rho_vec, params.tol.svd) {
            let b5_vec = b5_mat.column(0);
            let mut beta3 = [0.0; 4];
            if b5_vec[0] < 0.0 {
                beta3[0] = (-b5_vec[0]).sqrt();
                beta3[1] = if b5_vec[2] > 0.0 {
                    0.0
                } else {
                    (-b5_vec[2]).sqrt()
                };
                beta3[2] = b5_vec[3] / (-b5_vec[0]).sqrt();
            } else {
                beta3[0] = b5_vec[0].sqrt();
                beta3[1] = if b5_vec[2] < 0.0 {
                    0.0
                } else {
                    b5_vec[2].sqrt()
                };
                beta3[2] = b5_vec[3] / b5_vec[0].sqrt();
            }
            if b5_vec[1] < 0.0 {
                beta3[0] = -beta3[0];
            }
            betas.push(beta3);
        }
    }

    // ------------------------------------------------------------------
    // 6. Gauss–Newton refinement of each beta candidate
    // ------------------------------------------------------------------
    let betas_refined: Vec<[f64; 4]> = betas
        .iter()
        .map(|&b| gauss_newton(b, &null4, &rho))
        .collect();

    // ------------------------------------------------------------------
    // 7. Evaluate reprojection error and pick best candidate
    // ------------------------------------------------------------------
    let mut best_err = f64::INFINITY;
    let mut best_r = [[1.0; 3]; 3];
    let mut best_t = [0.0; 3];

    for bet in &betas_refined {
        let (r_c, t_c) = pose_from_betas(bet, &null4, &cw, &alphas);
        let err = rmse_px(world_pts, image_pts, &r_c, &t_c, k);
        if err < best_err {
            best_err = err;
            best_r = r_c;
            best_t = t_c;
        }
    }

    let mat = Mat3A::from_cols_array(&[
        best_r[0][0] as f32,
        best_r[1][0] as f32,
        best_r[2][0] as f32,
        best_r[0][1] as f32,
        best_r[1][1] as f32,
        best_r[2][1] as f32,
        best_r[0][2] as f32,
        best_r[1][2] as f32,
        best_r[2][2] as f32,
    ]);
    let rvec_f32 = SO3::from_matrix(&mat).log();
    let rvec = [rvec_f32.x as f64, rvec_f32.y as f64, rvec_f32.z as f64];

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
    betas: &[f64; 4],
    null4: &DMatrix<f64>, // 12×4 matrix (V)
    cw: &[[f64; 3]; 4],   // control points in world frame
    alphas: &[[f64; 4]],  // barycentric coordinates for each world point
) -> ([[f64; 3]; 3], [f64; 3]) {
    // 1. Compute control points in camera frame: Cc = V * betas
    let beta_vec = Vector4::from_column_slice(betas);
    let cc_flat = null4 * beta_vec; // 12×1 vector

    // Reshape to 4×3 (row-major order: consecutive triplets are x,y,z per point)
    let mut cc: [[f64; 3]; 4] = [[0.0; 3]; 4];
    for i in 0..4 {
        cc[i][0] = cc_flat[3 * i];
        cc[i][1] = cc_flat[3 * i + 1];
        cc[i][2] = cc_flat[3 * i + 2];
    }

    // 2. Reconstruct first point in camera frame for sign check
    let a0 = alphas[0];
    let mut pc0 = [0.0; 3];
    for j in 0..4 {
        pc0[0] += a0[j] * cc[j][0];
        pc0[1] += a0[j] * cc[j][1];
        pc0[2] += a0[j] * cc[j][2];
    }

    // 3. If Z negative, flip sign of all camera-frame control points
    if pc0[2] < 0.0 {
        for pt in &mut cc {
            pt[0] *= -1.0;
            pt[1] *= -1.0;
            pt[2] *= -1.0;
        }
    }

    // 4. Estimate R and t via Umeyama between control points
    let (r, t, _s) = umeyama(cw, &cc);
    (r, t)
}

/// Root-mean-square reprojection error in pixels.
fn rmse_px(
    pw: &[[f64; 3]],
    uv: &[[f64; 2]],
    r: &[[f64; 3]; 3],
    t: &[f64; 3],
    k: &[[f64; 3]; 3],
) -> f64 {
    assert_eq!(pw.len(), uv.len());

    let fx = k[0][0];
    let fy = k[1][1];
    let cx = k[0][2];
    let cy = k[1][2];

    let mut sum_sq = 0.0;
    let n = pw.len() as f64;

    for (p, &img) in pw.iter().zip(uv.iter()) {
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

fn select_control_points(pw: &[[f64; 3]]) -> [[f64; 3]; 4] {
    let n = pw.len();
    let c = compute_centroid(pw);

    // ------------------------------------------------------------------
    // 1. Compute the 3×3 covariance matrix of demeaned points.
    // ------------------------------------------------------------------
    let mut cov = [[0.0f64; 3]; 3];
    for p in pw {
        let dx = p[0] - c[0];
        let dy = p[1] - c[1];
        let dz = p[2] - c[2];

        cov[0][0] += dx * dx;
        cov[0][1] += dx * dy;
        cov[0][2] += dx * dz;

        cov[1][0] += dy * dx;
        cov[1][1] += dy * dy;
        cov[1][2] += dy * dz;

        cov[2][0] += dz * dx;
        cov[2][1] += dz * dy;
        cov[2][2] += dz * dz;
    }

    let inv_n = 1.0 / n as f64;
    for row in &mut cov {
        for val in row {
            *val *= inv_n;
        }
    }

    // Convert covariance matrix to glam::Mat3<f32> column-major order.
    let cov_mat = Mat3::from_cols(
        Vec3::new(cov[0][0] as f32, cov[1][0] as f32, cov[2][0] as f32),
        Vec3::new(cov[0][1] as f32, cov[1][1] as f32, cov[2][1] as f32),
        Vec3::new(cov[0][2] as f32, cov[1][2] as f32, cov[2][2] as f32),
    );

    let svd = svd3(&cov_mat);
    let v = svd.v();
    let s = svd.s(); // diagonal matrix of singular values (eigenvalues)

    let mut axes_sig: Vec<(f64, Vec3)> = vec![
        ((s.x_axis.x as f64).sqrt(), v.x_axis),
        ((s.y_axis.y as f64).sqrt(), v.y_axis),
        ((s.z_axis.z as f64).sqrt(), v.z_axis),
    ];
    axes_sig.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    // ------------------------------------------------------------------
    // 3. Assemble control points: centroid + principal-axis displacements
    //    using sigmas sorted in descending order.
    // ------------------------------------------------------------------
    let mut cw = [[0.0; 3]; 4];
    cw[0] = c;

    for (i, (sigma, axis)) in axes_sig.iter().enumerate() {
        cw[i + 1][0] = c[0] + sigma * axis.x as f64;
        cw[i + 1][1] = c[1] + sigma * axis.y as f64;
        cw[i + 1][2] = c[2] + sigma * axis.z as f64;
    }

    cw
}

/// Compute barycentric coordinates of world-space points with respect to the
/// 4 control points returned by `select_control_points`.
///
/// # Arguments
/// * `pw` – World points *(N,3)*.
/// * `cw` – Control points *(4,3)*.
/// * `eps` – Threshold that decides whether the control-point tetrahedron is
///   degenerate. If the determinant of the 3-by-3 matrix built from `cw` is
///   smaller than `eps`, a Moore–Penrose pseudo-inverse is used instead of the exact inverse.
///
/// # Returns
/// Vector of length *N* where each element is `[α0, α1, α2, α3]` such that
/// `α0 + α1 + α2 + α3 = 1` and `pw_i = Σ αj Cw_j`.
fn compute_barycentric(pw: &[[f64; 3]], cw: &[[f64; 3]; 4], eps: f64) -> Vec<[f64; 4]> {
    // ------------------------------------------------------------------
    // 1. Build the 3×3 matrix B = [C1-C0, C2-C0, C3-C0].  Each column is a
    //    displacement vector from the first control point C0.
    // ------------------------------------------------------------------
    let c0 = cw[0];
    let d1 = [cw[1][0] - c0[0], cw[1][1] - c0[1], cw[1][2] - c0[2]];
    let d2 = [cw[2][0] - c0[0], cw[2][1] - c0[1], cw[2][2] - c0[2]];
    let d3 = [cw[3][0] - c0[0], cw[3][1] - c0[1], cw[3][2] - c0[2]];

    // Convert to `glam::Mat3<f32>` column-major layout.
    let b_mat = Mat3::from_cols(
        Vec3::new(d1[0] as f32, d1[1] as f32, d1[2] as f32),
        Vec3::new(d2[0] as f32, d2[1] as f32, d2[2] as f32),
        Vec3::new(d3[0] as f32, d3[1] as f32, d3[2] as f32),
    );

    // ------------------------------------------------------------------
    // 2. Invert B (or obtain its pseudo-inverse if nearly singular).
    // ------------------------------------------------------------------
    let det = b_mat.determinant() as f64;
    let b_inv: Mat3 = if det.abs() > eps {
        b_mat.inverse()
    } else {
        // Moore–Penrose pseudo-inverse via SVD: B⁺ = V Σ⁺ Uᵀ
        let svd = svd3(&b_mat);
        let u = svd.u();
        let s = svd.s();
        let v = svd.v();

        // Invert singular values with thresholding.
        let mut sigma_inv = Mat3::ZERO;
        for (i, sigma) in [s.x_axis.x, s.y_axis.y, s.z_axis.z].iter().enumerate() {
            let val = sigma.abs();
            if val > eps as f32 {
                match i {
                    0 => sigma_inv.x_axis.x = 1.0 / val,
                    1 => sigma_inv.y_axis.y = 1.0 / val,
                    2 => sigma_inv.z_axis.z = 1.0 / val,
                    _ => unreachable!(),
                }
            }
        }

        // B⁺ = V * Σ⁺ * Uᵀ
        v.mul_mat3(&sigma_inv).mul_mat3(&u.transpose())
    };

    // Convert `b_inv` to f64 for downstream computations.
    let b_inv_f64: [[f64; 3]; 3] = [
        [
            b_inv.x_axis.x as f64,
            b_inv.y_axis.x as f64,
            b_inv.z_axis.x as f64,
        ],
        [
            b_inv.x_axis.y as f64,
            b_inv.y_axis.y as f64,
            b_inv.z_axis.y as f64,
        ],
        [
            b_inv.x_axis.z as f64,
            b_inv.y_axis.z as f64,
            b_inv.z_axis.z as f64,
        ],
    ];

    // ------------------------------------------------------------------
    // 3. Compute barycentric coordinates α such that p = Σ αj Cw_j.
    // ------------------------------------------------------------------
    let mut alphas = vec![[0.0; 4]; pw.len()];
    for (i, p) in pw.iter().enumerate() {
        let diff = [p[0] - c0[0], p[1] - c0[1], p[2] - c0[2]];

        // λ = B_inv * (p - C0)
        let mut lamb = [0.0; 3];
        for row in 0..3 {
            lamb[row] = b_inv_f64[row][0] * diff[0]
                + b_inv_f64[row][1] * diff[1]
                + b_inv_f64[row][2] * diff[2];
        }

        alphas[i][1] = lamb[0];
        alphas[i][2] = lamb[1];
        alphas[i][3] = lamb[2];
        alphas[i][0] = 1.0 - lamb.iter().sum::<f64>();
    }

    alphas
}

/// Construct the 2N×12 design matrix **M** used by EPnP.
///
/// * `alphas` – Barycentric coordinates for each world point, produced by
///   [`compute_barycentric`]; shape *(N,4)*.
/// * `uv`     – Pixel coordinates for each correspondence; shape *(N,2)*.
/// * `k`      – Camera intrinsics 3×3 matrix.
///
/// The output is a vector of length `2*N` where each element is the 12-vector
/// corresponding to a row of **M**.
fn build_m(alphas: &[[f64; 4]], uv: &[[f64; 2]], k: &[[f64; 3]; 3]) -> Vec<[f64; 12]> {
    assert_eq!(
        alphas.len(),
        uv.len(),
        "alphas and uv must have the same length"
    );
    let n = alphas.len();

    let fu = k[0][0];
    let fv = k[1][1];
    let uc = k[0][2];
    let vc = k[1][2];

    // Pre-allocate 2N rows of zeros.
    let mut m = vec![[0.0f64; 12]; 2 * n];

    for (i, (a, &uv_i)) in alphas.iter().zip(uv.iter()).enumerate() {
        let u = uv_i[0];
        let v = uv_i[1];

        let row_x = 2 * i;
        let row_y = row_x + 1;

        for (j, &alpha) in a.iter().enumerate() {
            let base = 3 * j;
            // x-row
            m[row_x][base] = alpha * fu;
            // base+1 remains 0
            m[row_x][base + 2] = alpha * (uc - u);
            // y-row
            m[row_y][base + 1] = alpha * fv;
            m[row_y][base + 2] = alpha * (vc - v);
        }
    }

    m
}

/// Build the 6×10 matrix **L** used in EPnP from the 4-dimensional null-space matrix `V` (shape 12×4).
fn build_l6x10(null4: &DMatrix<f64>) -> [[f64; 10]; 6] {
    // Re-ordered column indices (reverse order).
    let col_order = [3usize, 2, 1, 0];

    // v[i] is 4×3 matrix => Vec<[Vector3;4]>
    let mut v_cp: Vec<[Vector3<f64>; 4]> = Vec::with_capacity(4);

    for &c in &col_order {
        let col = null4.column(c);
        let mut blocks = [Vector3::zeros(); 4];
        for k in 0..4 {
            blocks[k] = Vector3::new(col[3 * k], col[3 * k + 1], col[3 * k + 2]);
        }
        v_cp.push(blocks);
    }

    let pairs = [(0usize, 1usize), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

    let mut dv_arr: Vec<Vec<Vector3<f64>>> = vec![vec![Vector3::zeros(); 6]; 4];
    for i in 0..4 {
        for (j, &(a, b)) in pairs.iter().enumerate() {
            dv_arr[i][j] = v_cp[i][a] - v_cp[i][b];
        }
    }

    let mut l = [[0.0f64; 10]; 6];
    for (j, _) in dv_arr[0].iter().enumerate() {
        l[j][0] = dv_arr[0][j].dot(&dv_arr[0][j]);
        l[j][1] = 2.0 * dv_arr[0][j].dot(&dv_arr[1][j]);
        l[j][2] = dv_arr[1][j].dot(&dv_arr[1][j]);
        l[j][3] = 2.0 * dv_arr[0][j].dot(&dv_arr[2][j]);
        l[j][4] = 2.0 * dv_arr[1][j].dot(&dv_arr[2][j]);
        l[j][5] = dv_arr[2][j].dot(&dv_arr[2][j]);
        l[j][6] = 2.0 * dv_arr[0][j].dot(&dv_arr[3][j]);
        l[j][7] = 2.0 * dv_arr[1][j].dot(&dv_arr[3][j]);
        l[j][8] = 2.0 * dv_arr[2][j].dot(&dv_arr[3][j]);
        l[j][9] = dv_arr[3][j].dot(&dv_arr[3][j]);
    }

    l
}

/// Compute the six squared distances (ρ vector) between the 4 control points.
fn rho_ctrlpts(cw: &[[f64; 3]; 4]) -> [f64; 6] {
    let pairs = [(0usize, 1usize), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    let mut rho = [0.0f64; 6];
    for (idx, &(i, j)) in pairs.iter().enumerate() {
        let dx = cw[i][0] - cw[j][0];
        let dy = cw[i][1] - cw[j][1];
        let dz = cw[i][2] - cw[j][2];
        rho[idx] = dx * dx + dy * dy + dz * dz;
    }
    rho
}

#[cfg(test)]
mod solve_epnp_tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_solve_epnp() {
        // World points (same sample as before)
        let pw: [[f64; 3]; 6] = [
            [0.0315, 0.03333, -0.10409],
            [-0.0315, 0.03333, -0.10409],
            [0.0, -0.00102, -0.12977],
            [0.02646, -0.03167, -0.1053],
            [-0.02646, -0.031667, -0.1053],
            [0.0, 0.04515, -0.11033],
        ];

        // Image points (uv)
        let uv: [[f64; 2]; 6] = [
            [722.96465987, 502.08278077],
            [669.88838745, 498.61877868],
            [707.00251568, 478.48975973],
            [728.05635561, 447.56919481],
            [682.60688321, 443.91774467],
            [696.44137826, 511.96442904],
        ];

        // Camera intrinsics
        let k: [[f64; 3]; 3] = [[800.0, 0.0, 640.0], [0.0, 800.0, 480.0], [0.0, 0.0, 1.0]];

        // 1. Control points
        let cw = select_control_points(&pw);

        // 2. Barycentric coordinates
        let alphas = compute_barycentric(&pw, &cw, EPNPParams::default().tol.eps);

        // Basic sanity checks on barycentric coordinates
        for (p, alpha) in pw.iter().zip(alphas.iter()) {
            // reconstruction
            let mut recon = [0.0; 3];
            for j in 0..4 {
                recon[0] += alpha[j] * cw[j][0];
                recon[1] += alpha[j] * cw[j][1];
                recon[2] += alpha[j] * cw[j][2];
            }
            for k in 0..3 {
                assert_relative_eq!(recon[k], p[k], epsilon = 1e-6);
            }

            assert_relative_eq!(alpha.iter().sum::<f64>(), 1.0, epsilon = 1e-9);
        }

        // 3. Design matrix M
        let m = build_m(&alphas, &uv, &k);
        assert_eq!(m.len(), 2 * pw.len());
        for row in &m {
            assert_eq!(row.len(), 12);
        }

        // Verify first correspondence’s rows explicitly (numerical values)
        let fu = k[0][0];
        let fv = k[1][1];
        let uc = k[0][2];
        let vc = k[1][2];

        let u0 = uv[0][0];
        let v0 = uv[0][1];

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

        // 4. Ensure `solve_epnp` runs without error (placeholder implementation)
        let result =
            EPnP::solve(&pw, &uv, &k, &EPNPParams::default()).expect("EPnP::solve should succeed");
        let r = result.rotation;
        let t = result.translation;
        let rvec = result.rvec;

        assert_relative_eq!(r[0][0], 0.69650543, epsilon = 1e-2);
        assert_relative_eq!(r[0][1], 0.07230615, epsilon = 1e-2);
        assert_relative_eq!(r[0][2], -0.71389916, epsilon = 1e-2);
        assert_relative_eq!(r[1][0], 0.22406019, epsilon = 1e-2);
        assert_relative_eq!(r[1][1], 0.92324643, epsilon = 1e-2);
        assert_relative_eq!(r[1][2], 0.31211066, epsilon = 1e-2);
        assert_relative_eq!(r[2][0], 0.68167237, epsilon = 1e-2);

        assert_relative_eq!(t[0], -0.00861299, epsilon = 1e-2);
        assert_relative_eq!(t[1], 0.02666388, epsilon = 1e-2);
        assert_relative_eq!(t[2], 1.01495503, epsilon = 1e-2);

        assert_relative_eq!(rvec[0], -0.39580156, epsilon = 1e-2);
        assert_relative_eq!(rvec[1], -0.80116952, epsilon = 1e-2);
        assert_relative_eq!(rvec[2], 0.08711894, epsilon = 1e-2);
    }
}
