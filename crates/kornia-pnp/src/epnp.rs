//! Efficient Perspective-n-Point (EPnP) solver
//! Paper: [Lepetit et al., IJCV 2009](https://www.tugraz.at/fileadmin/user_upload/Institute/ICG/Images/team_lepetit/publications/lepetit_ijcv08.pdf)
//! Reference: [OpenCV EPnP implementation](https://github.com/opencv/opencv/blob/4.x/modules/calib3d/src/epnp.cpp)

use crate::ops::{compute_centroid, gauss_newton, intrinsics_as_vectors, pose_to_rt};
use crate::pnp::{NumericTol, PnPError, PnPResult, PnPSolver};
use kornia_algebra::{Mat3AF32, Mat3F32, Vec3F32, SO3F32};
use kornia_imgproc::calibration::{
    distortion::{distort_point_polynomial, PolynomialDistortion},
    CameraIntrinsic,
};
use kornia_linalg::rigid::umeyama;
use kornia_linalg::svd::svd3;
use nalgebra::{DMatrix, DVector, Vector4};

/// Marker type representing the Efficient PnP algorithm.
pub struct EPnP;

impl PnPSolver for EPnP {
    type Param = EPnPParams;

    fn solve(
        points_world: &[[f32; 3]],
        points_image: &[[f32; 2]],
        k: &[[f32; 3]; 3],
        distortion: Option<&PolynomialDistortion>,
        params: &Self::Param,
    ) -> Result<PnPResult, PnPError> {
        solve_epnp(points_world, points_image, k, distortion, params)
    }
}

/// Parameters controlling the EPnP solver.
#[derive(Debug, Clone, Default)]
pub struct EPnPParams {
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
/// Tuple (`R`, `t`, `rvec`) where
/// - `R`: 3×3 rotation, mapping from world → camera
/// - `t`: 3-vector translation
/// - `rvec`: Rodrigues axis-angle representation of `R`
pub fn solve_epnp(
    points_world: &[[f32; 3]],
    points_image: &[[f32; 2]],
    k: &[[f32; 3]; 3],
    distortion: Option<&PolynomialDistortion>,
    params: &EPnPParams,
) -> Result<PnPResult, PnPError> {
    let n = points_world.len();
    if n != points_image.len() {
        return Err(PnPError::MismatchedArrayLengths {
            left_name: "world points",
            left_len: n,
            right_name: "image points",
            right_len: points_image.len(),
        });
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
    let m_rows = build_m(&alphas, points_image, k)?;

    let m_flat: Vec<f32> = m_rows.iter().flat_map(|row| row.iter()).cloned().collect();
    let m_mat = DMatrix::<f32>::from_row_slice(2 * n, 12, &m_flat);

    // Null-space of M via eigen decomposition of MtM (12×12)
    // TODO: mtm is always symmetric; look into more efficient multiplication for this case.
    let mtm = m_mat.transpose() * &m_mat; // 12×12
    let eig = mtm.symmetric_eigen();

    let eigenvalues = eig.eigenvalues;
    let eigenvectors = eig.eigenvectors;

    let mut value_index_pairs: Vec<(f32, usize)> = eigenvalues
        .iter()
        .cloned()
        .enumerate()
        .map(|(index, value)| (value.abs(), index))
        .collect();

    value_index_pairs.sort_by(|a, b| a.0.total_cmp(&b.0));

    let null4 = DMatrix::from_columns(&[
        eigenvectors.column(value_index_pairs[3].1),
        eigenvectors.column(value_index_pairs[2].1),
        eigenvectors.column(value_index_pairs[1].1),
        eigenvectors.column(value_index_pairs[0].1),
    ]);

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
        let (r_c, t_c) = pose_from_betas(bet, &null4, &cw, &alphas)?;
        let err = rmse_px(points_world, points_image, &r_c, &t_c, k, distortion)?;
        if err < best_err {
            best_err = err;
            best_r = r_c;
            best_t = t_c;
        }
    }

    let mat = Mat3AF32::from_cols_array(&[
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
    let rvec_f32 = SO3F32::from_matrix(&mat).log();
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
) -> Result<([[f32; 3]; 3], [f32; 3]), PnPError> {
    let beta_vec = Vector4::from_column_slice(betas);
    let cc_flat = null4 * beta_vec; // 12×1 vector

    let mut cc: [[f32; 3]; 4] = [[0.0; 3]; 4];
    for i in 0..4 {
        cc[i][0] = cc_flat[3 * i];
        cc[i][1] = cc_flat[3 * i + 1];
        cc[i][2] = cc_flat[3 * i + 2];
    }

    let a0 = alphas[0];
    let mut pc0_vec = Vec3F32::ZERO;
    for j in 0..4 {
        pc0_vec += Vec3F32::from_array(cc[j]) * a0[j];
    }

    if pc0_vec.z < 0.0 {
        for pt in &mut cc {
            pt[0] *= -1.0;
            pt[1] *= -1.0;
            pt[2] *= -1.0;
        }
    }

    // Convert arrays to Vec3 for consistency with glam usage
    let cw_vec3: Vec<glam::Vec3> = cw
        .iter()
        .map(|p| glam::Vec3::new(p[0], p[1], p[2]))
        .collect();
    let cc_vec3: Vec<glam::Vec3> = cc
        .iter()
        .map(|p| glam::Vec3::new(p[0], p[1], p[2]))
        .collect();

    let (r, t, _s) = umeyama(&cw_vec3, &cc_vec3).map_err(|e| PnPError::SvdFailed(e.to_string()))?;

    Ok((r, t))
}

/// Root-mean-square reprojection error in pixels.
fn rmse_px(
    points_world: &[[f32; 3]],
    points_image: &[[f32; 2]],
    r: &[[f32; 3]; 3],
    t: &[f32; 3],
    k: &[[f32; 3]; 3],
    distortion: Option<&PolynomialDistortion>,
) -> Result<f32, PnPError> {
    if points_world.len() != points_image.len() {
        return Err(PnPError::MismatchedArrayLengths {
            left_name: "world points",
            left_len: points_world.len(),
            right_name: "image points",
            right_len: points_image.len(),
        });
    }

    let fx = k[0][0];
    let fy = k[1][1];
    let cx = k[0][2];
    let cy = k[1][2];

    let (r_mat, t_vec) = pose_to_rt(r, t);
    let (intr_x, intr_y) = intrinsics_as_vectors(k);

    let mut sum_sq = 0.0;
    let n = points_world.len() as f32;

    // Prepare camera intrinsic for distortion if needed
    let cam_intr = CameraIntrinsic {
        fx: fx as f64,
        fy: fy as f64,
        cx: cx as f64,
        cy: cy as f64,
    };

    for (pw_arr, &uv) in points_world.iter().zip(points_image.iter()) {
        let pw = Vec3F32::from_array(*pw_arr);
        let pc = r_mat * pw + t_vec; // camera-frame point

        let inv_z = 1.0 / pc.z;
        let u_undist = intr_x.dot(pc) * inv_z; // (fx * x + cx * z) / z
        let v_undist = intr_y.dot(pc) * inv_z; // (fy * y + cy * z) / z

        // Apply distortion model if provided
        let (u_hat, v_hat) = if let Some(d) = distortion {
            let (ud, vd) = distort_point_polynomial(u_undist as f64, v_undist as f64, &cam_intr, d);
            (ud as f32, vd as f32)
        } else {
            (u_undist, v_undist)
        };

        let du = u_hat - uv[0];
        let dv = v_hat - uv[1];
        sum_sq += du.mul_add(du, dv * dv); // FMA where available
    }

    Ok((sum_sq / n).sqrt())
}

fn select_control_points(points_world: &[[f32; 3]]) -> [[f32; 3]; 4] {
    let n = points_world.len();
    let c = compute_centroid(points_world);

    // Compute covariance using glam for consistency
    let mut cov_mat = Mat3F32::from_cols_array(&[0.0; 9]);
    for p in points_world {
        let diff = Vec3F32::new(p[0] - c[0], p[1] - c[1], p[2] - c[2]);
        // Outer product diff * diffᵀ via column scaling
        let outer_product = Mat3F32::from_cols(diff * diff.x, diff * diff.y, diff * diff.z);
        cov_mat += outer_product;
    }
    cov_mat *= 1.0 / n as f32;

    let svd = svd3(&cov_mat);
    let v = svd.v();
    let s = svd.s(); // diagonal matrix of singular values (eigenvalues)

    let s_diag = [s.x_axis.x, s.y_axis.y, s.z_axis.z];
    let mut axes_sig: Vec<(f32, Vec3F32)> = vec![
        (s_diag[0].sqrt(), Vec3F32::from(v.x_axis)),
        (s_diag[1].sqrt(), Vec3F32::from(v.y_axis)),
        (s_diag[2].sqrt(), Vec3F32::from(v.z_axis)),
    ];
    axes_sig.sort_by(|a, b| b.0.total_cmp(&a.0));

    let c_vec = Vec3F32::from_array(c);
    let mut cw = [[0.0; 3]; 4];
    cw[0] = c;

    for (i, (sigma, axis)) in axes_sig.iter().enumerate() {
        let cp = c_vec + *axis * *sigma;
        cw[i + 1] = cp.to_array();
    }

    cw
}

/// Compute barycentric coordinates of world-space points with respect to the
/// 4 control points returned by `select_control_points`.
///
/// # Arguments
/// - `points_world`: World points, shape `(N, 3)`.
/// - `cw`: Control points, shape `(4, 3)`.
/// - `eps`: Degeneracy threshold for the control-point tetrahedron. If `det(B) < eps`,
///   a Moore–Penrose pseudo-inverse is used instead of the exact inverse.
///
/// # Returns
/// `Vec<[f32; 4]>` of length `N`. For each point, the weights `[a0, a1, a2, a3]` satisfy
/// `a0 + a1 + a2 + a3 = 1` and `pw_i = sum_j(a_j * Cw_j)`.
fn compute_barycentric(points_world: &[[f32; 3]], cw: &[[f32; 3]; 4], eps: f32) -> Vec<[f32; 4]> {
    // Build B = [C1 - C0, C2 - C0, C3 - C0].
    let c0 = Vec3F32::new(cw[0][0], cw[0][1], cw[0][2]);
    let d1 = Vec3F32::new(cw[1][0] - c0.x, cw[1][1] - c0.y, cw[1][2] - c0.z);
    let d2 = Vec3F32::new(cw[2][0] - c0.x, cw[2][1] - c0.y, cw[2][2] - c0.z);
    let d3 = Vec3F32::new(cw[3][0] - c0.x, cw[3][1] - c0.y, cw[3][2] - c0.z);

    let b = Mat3F32::from_cols(d1, d2, d3);

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
        let inv_diag = Vec3F32::new(
            if s_diag[0].abs() > eps {
                1.0 / s_diag[0]
            } else {
                0.0
            },
            if s_diag[1].abs() > eps {
                1.0 / s_diag[1]
            } else {
                0.0
            },
            if s_diag[2].abs() > eps {
                1.0 / s_diag[2]
            } else {
                0.0
            },
        );
        let sigma_inv = Mat3F32::from_diagonal(inv_diag);
        v_mat * sigma_inv * u.transpose()
    };

    // Compute barycentric coordinates.
    points_world
        .iter()
        .map(|p| {
            let diff = Vec3F32::from_array(*p) - c0;
            let lamb = b_inv * diff;
            [1.0 - (lamb.x + lamb.y + lamb.z), lamb.x, lamb.y, lamb.z]
        })
        .collect()
}

/// Construct the 2N x 12 design matrix `M` used by EPnP.
///
/// # Arguments
/// - `alphas`: Barycentric coordinates for each world point, produced by [`compute_barycentric`]; shape `(N, 4)`.
/// - `points_image`: Pixel coordinates for each correspondence; shape `(N, 2)`.
/// - `k`: Camera intrinsics 3 x 3 matrix.
///
/// # Returns
/// `Result<Vec<[f32; 12]>, PnPError>`: a vector of length `2*N` where each element is the 12-vector
/// corresponding to a row of `M` (two rows per correspondence). Returns an error if the input slices differ in length.
fn build_m(
    alphas: &[[f32; 4]],
    points_image: &[[f32; 2]],
    k: &[[f32; 3]; 3],
) -> Result<Vec<[f32; 12]>, PnPError> {
    if alphas.len() != points_image.len() {
        return Err(PnPError::MismatchedArrayLengths {
            left_name: "barycentric alphas",
            left_len: alphas.len(),
            right_name: "image points",
            right_len: points_image.len(),
        });
    }
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

    Ok(m)
}

/// Build the 6×10 matrix **L** used in EPnP from the 4-dimensional null-space matrix `V` (shape 12×4).
fn build_l6x10(null4: &DMatrix<f32>) -> [[f32; 10]; 6] {
    // Re-ordered column indices (reverse order).
    let mut l = [[0.0f32; 10]; 6];

    let c3 = null4.column(3);
    let c2 = null4.column(2);
    let c1 = null4.column(1);
    let c0 = null4.column(0);

    let cols: [&[f32]; 4] = [c3.as_slice(), c2.as_slice(), c1.as_slice(), c0.as_slice()];

    for (j, &(a, b)) in CP_PAIRS.iter().enumerate() {
        let mut d = [[0.0; 3]; 4];

        for (k, col) in cols.iter().enumerate() {
            let base_a = 3 * a;
            let base_b = 3 * b;
            d[k][0] = col[base_a] - col[base_b];
            d[k][1] = col[base_a + 1] - col[base_b + 1];
            d[k][2] = col[base_a + 2] - col[base_b + 2];
        }

        #[inline(always)]
        fn dot(a: &[f32; 3], b: &[f32; 3]) -> f32 {
            a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
        }

        l[j] = [
            dot(&d[0], &d[0]),
            2.0 * dot(&d[0], &d[1]),
            dot(&d[1], &d[1]),
            2.0 * dot(&d[0], &d[2]),
            2.0 * dot(&d[1], &d[2]),
            dot(&d[2], &d[2]),
            2.0 * dot(&d[0], &d[3]),
            2.0 * dot(&d[1], &d[3]),
            2.0 * dot(&d[2], &d[3]),
            dot(&d[3], &d[3]),
        ];
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
        let diff = Vec3F32::from_array(cw[i]) - Vec3F32::from_array(cw[j]);
        diff.dot(diff)
    })
}

#[cfg(test)]
mod solve_epnp_tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_solve_epnp() -> Result<(), PnPError> {
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

        let alphas = compute_barycentric(&points_world, &cw, EPnPParams::default().tol.eps);

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

        let m = build_m(&alphas, &points_image, &k)?;
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

        #[allow(clippy::needless_range_loop)]
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

        let result = EPnP::solve(
            &points_world,
            &points_image,
            &k,
            None,
            &EPnPParams::default(),
        )?;
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
        Ok(())
    }
}
