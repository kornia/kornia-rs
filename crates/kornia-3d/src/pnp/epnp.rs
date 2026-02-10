//! Efficient Perspective-n-Point (EPnP) solver
//! Paper: [Lepetit et al., IJCV 2009](https://www.tugraz.at/fileadmin/user_upload/Institute/ICG/Images/team_lepetit/publications/lepetit_ijcv08.pdf)
//! Reference: [OpenCV EPnP implementation](https://github.com/opencv/opencv/blob/4.x/modules/calib3d/src/epnp.cpp)

use super::ops::{compute_centroid, gauss_newton, intrinsics_as_vectors};
use super::refine::{refine_pose_lm, LMRefineParams};
use super::{NumericTol, PnPError, PnPResult, PnPSolver};
use kornia_algebra::linalg::rigid::umeyama;
use kornia_algebra::linalg::svd::svd3_f32;
use kornia_algebra::{Mat3AF32, Mat3F32, Vec2F32, Vec3AF32, Vec3F32, SO3F32};
use kornia_imgproc::calibration::{
    distortion::{distort_point_polynomial, PolynomialDistortion},
    CameraIntrinsic,
};
use nalgebra::{DMatrix, DVector, Vector4};

/// Marker type representing the Efficient PnP algorithm.
pub struct EPnP;

impl PnPSolver for EPnP {
    type Param = EPnPParams;

    fn solve(
        points_world: &[Vec3AF32],
        points_image: &[Vec2F32],
        k: &Mat3AF32,
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
    /// Optional LM refinement parameters. If `Some`, the pose will be refined
    /// after the initial EPnP solution.
    pub refine_lm: Option<LMRefineParams>,
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
    points_world: &[Vec3AF32],
    points_image: &[Vec2F32],
    k: &Mat3AF32,
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
    let mut best_r = Mat3AF32::IDENTITY;
    let mut best_t = Vec3AF32::ZERO;

    for bet in &betas_refined {
        let (r_c, t_c) = pose_from_betas(bet, &null4, &cw, &alphas)?;
        let err = rmse_px(points_world, points_image, &r_c, &t_c, k, distortion)?;
        if err < best_err {
            best_err = err;
            best_r = r_c;
            best_t = t_c;
        }
    }

    let rvec = SO3F32::from_matrix(&best_r).log();

    // Optionally refine pose using LM optimization
    if let Some(ref lm_params) = params.refine_lm {
        return refine_pose_lm(
            points_world,
            points_image,
            k,
            &best_r,
            &best_t,
            distortion,
            lm_params,
        );
    }

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
    cw: &[Vec3AF32; 4],   // control points in world frame
    alphas: &[[f32; 4]],  // barycentric coordinates for each world point
) -> Result<(Mat3AF32, Vec3AF32), PnPError> {
    let beta_vec = Vector4::from_column_slice(betas);
    let cc_flat = null4 * beta_vec; // 12×1 vector

    let mut cc: [Vec3AF32; 4] = [Vec3AF32::ZERO; 4];
    for i in 0..4 {
        cc[i] = Vec3AF32::new(cc_flat[3 * i], cc_flat[3 * i + 1], cc_flat[3 * i + 2]);
    }

    let a0 = alphas[0];
    let mut pc0_vec = Vec3AF32::ZERO;
    for j in 0..4 {
        pc0_vec += cc[j] * a0[j];
    }

    if pc0_vec.z < 0.0 {
        for pt in &mut cc {
            *pt *= -1.0;
        }
    }

    let (r, t, _s) = umeyama(cw, &cc).map_err(|e| PnPError::SvdFailed(e.to_string()))?;
    Ok((r, t))
}

/// Root-mean-square reprojection error in pixels.
fn rmse_px(
    points_world: &[Vec3AF32],
    points_image: &[Vec2F32],
    r: &Mat3AF32,
    t: &Vec3AF32,
    k: &Mat3AF32,
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

    let (intr_x, intr_y) = intrinsics_as_vectors(k);
    let fx = k.x_axis().x;
    let fy = k.y_axis().y;
    let cx = k.z_axis().x;
    let cy = k.z_axis().y;

    let mut sum_sq = 0.0;
    let n = points_world.len() as f32;

    // Prepare camera intrinsic for distortion if needed
    let cam_intr = CameraIntrinsic {
        fx: fx as f64,
        fy: fy as f64,
        cx: cx as f64,
        cy: cy as f64,
    };

    for (&pw, &uv) in points_world.iter().zip(points_image.iter()) {
        let pc = *r * pw + *t; // camera-frame point

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

        let du = u_hat - uv.x;
        let dv = v_hat - uv.y;
        sum_sq += du.mul_add(du, dv * dv); // FMA where available
    }

    Ok((sum_sq / n).sqrt())
}

fn select_control_points(points_world: &[Vec3AF32]) -> [Vec3AF32; 4] {
    let n = points_world.len();
    let c = compute_centroid(points_world);

    // Compute covariance using glam for consistency
    let mut cov_mat = Mat3F32::from_cols_array(&[0.0; 9]);
    for p in points_world {
        // svd3 currently expects Mat3F32; keep covariance math on Vec3F32.
        let diff = Vec3F32::new(p.x - c.x, p.y - c.y, p.z - c.z);
        // Outer product diff * diffᵀ via column scaling
        let outer_product = Mat3F32::from_cols(diff * diff.x, diff * diff.y, diff * diff.z);
        cov_mat += outer_product;
    }
    cov_mat *= 1.0 / n as f32;

    let svd = svd3_f32(&cov_mat);
    let v = svd.v();
    let s = svd.s(); // diagonal matrix of singular values (eigenvalues)

    let s_x = s.x_axis();
    let s_y = s.y_axis();
    let s_z = s.z_axis();
    let s_diag = [s_x.x, s_y.y, s_z.z];
    let v_x = v.x_axis();
    let v_y = v.y_axis();
    let v_z = v.z_axis();
    let mut axes_sig: Vec<(f32, Vec3AF32)> = vec![
        (s_diag[0].sqrt(), Vec3AF32::new(v_x.x, v_x.y, v_x.z)),
        (s_diag[1].sqrt(), Vec3AF32::new(v_y.x, v_y.y, v_y.z)),
        (s_diag[2].sqrt(), Vec3AF32::new(v_z.x, v_z.y, v_z.z)),
    ];
    axes_sig.sort_by(|a, b| b.0.total_cmp(&a.0));

    let mut cw = [Vec3AF32::ZERO; 4];
    cw[0] = c;

    for (i, (sigma, axis)) in axes_sig.iter().enumerate() {
        let cp = c + *axis * *sigma;
        cw[i + 1] = cp;
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
fn compute_barycentric(points_world: &[Vec3AF32], cw: &[Vec3AF32; 4], eps: f32) -> Vec<[f32; 4]> {
    // Build B = [C1 - C0, C2 - C0, C3 - C0].
    // Keep the linear algebra here on Mat3F32 for simplicity (svd3 and friends are Mat3F32-based).
    let c0 = Vec3F32::new(cw[0].x, cw[0].y, cw[0].z);
    let d1 = Vec3F32::new(cw[1].x - cw[0].x, cw[1].y - cw[0].y, cw[1].z - cw[0].z);
    let d2 = Vec3F32::new(cw[2].x - cw[0].x, cw[2].y - cw[0].y, cw[2].z - cw[0].z);
    let d3 = Vec3F32::new(cw[3].x - cw[0].x, cw[3].y - cw[0].y, cw[3].z - cw[0].z);

    let b = Mat3F32::from_cols(d1, d2, d3);

    // Invert or pseudo-invert B.
    let b_inv = if b.determinant().abs() > eps {
        // Safe to invert.
        b.inverse()
    } else {
        // Moore–Penrose pseudo-inverse: B⁺ = V Σ⁺ Uᵀ
        let svd = svd3_f32(&b);
        let u = *svd.u();
        let v_mat = *svd.v();
        let s_mat = *svd.s();
        let s_x = s_mat.x_axis();
        let s_y = s_mat.y_axis();
        let s_z = s_mat.z_axis();
        let s_diag = [s_x.x, s_y.y, s_z.z];
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
        .map(|&p| {
            let diff = Vec3F32::new(p.x - c0.x, p.y - c0.y, p.z - c0.z);
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
    points_image: &[Vec2F32],
    k: &Mat3AF32,
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

    let fu = k.x_axis().x;
    let fv = k.y_axis().y;
    let uc = k.z_axis().x;
    let vc = k.z_axis().y;

    // Pre-allocate 2N rows of zeros.
    let mut m = vec![[0.0f32; 12]; 2 * n];

    for (i, (a, &points_image_i)) in alphas.iter().zip(points_image.iter()).enumerate() {
        let u = points_image_i.x;
        let v = points_image_i.y;

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
fn rho_ctrlpts(cw: &[Vec3AF32; 4]) -> [f32; 6] {
    CP_PAIRS.map(|(i, j)| {
        let dx = cw[i].x - cw[j].x;
        let dy = cw[i].y - cw[j].y;
        let dz = cw[i].z - cw[j].z;
        dx.mul_add(dx, dy.mul_add(dy, dz * dz))
    })
}

#[cfg(test)]
mod solve_epnp_tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_solve_epnp() -> Result<(), PnPError> {
        // Hardcoded test data verified with OpenCV
        let points_world: [Vec3AF32; 6] = [
            Vec3AF32::new(0.0315, 0.03333, -0.10409),
            Vec3AF32::new(-0.0315, 0.03333, -0.10409),
            Vec3AF32::new(0.0, -0.00102, -0.12977),
            Vec3AF32::new(0.02646, -0.03167, -0.1053),
            Vec3AF32::new(-0.02646, -0.031667, -0.1053),
            Vec3AF32::new(0.0, 0.04515, -0.11033),
        ];

        // Image points (uv)
        let points_image: [Vec2F32; 6] = [
            Vec2F32::new(722.96466, 502.0828),
            Vec2F32::new(669.88837, 498.61877),
            Vec2F32::new(707.0025, 478.48975),
            Vec2F32::new(728.05634, 447.56918),
            Vec2F32::new(682.6069, 443.91776),
            Vec2F32::new(696.4414, 511.96442),
        ];

        let k = Mat3AF32::from_cols(
            Vec3AF32::new(800.0, 0.0, 0.0),
            Vec3AF32::new(0.0, 800.0, 0.0),
            Vec3AF32::new(640.0, 480.0, 1.0),
        );

        let cw = select_control_points(&points_world);

        let alphas = compute_barycentric(&points_world, &cw, EPnPParams::default().tol.eps);

        for (p, alpha) in points_world.iter().zip(alphas.iter()) {
            let mut recon = Vec3AF32::ZERO;
            for j in 0..4 {
                recon += cw[j] * alpha[j];
            }
            for k in 0..3 {
                assert_relative_eq!(recon.to_array()[k], p.to_array()[k], epsilon = 1e-6);
            }

            assert_relative_eq!(alpha.iter().sum::<f32>(), 1.0, epsilon = 1e-9);
        }

        let m = build_m(&alphas, &points_image, &k)?;
        assert_eq!(m.len(), 2 * points_world.len());
        for row in &m {
            assert_eq!(row.len(), 12);
        }

        let fu = k.x_axis().x;
        let fv = k.y_axis().y;
        let uc = k.z_axis().x;
        let vc = k.z_axis().y;

        let u0 = points_image[0].x;
        let v0 = points_image[0].y;

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

        // Mat3F32 stores columns; r[row][col] == column(col)[row].
        // Rotation Matrix (Column-Major)
        // Row 0
        assert_relative_eq!(r.x_axis().x, 0.63441813, epsilon = 1e-2);
        assert_relative_eq!(r.y_axis().x, -0.35425714, epsilon = 1e-2);
        assert_relative_eq!(r.z_axis().x, 0.6870339, epsilon = 1e-2);

        // Row 1
        assert_relative_eq!(r.x_axis().y, -0.12940092, epsilon = 1e-2);
        assert_relative_eq!(r.y_axis().y, 0.8275856, epsilon = 1e-2);
        assert_relative_eq!(r.z_axis().y, 0.546221, epsilon = 1e-2);

        // Row 2 (checking first col)
        assert_relative_eq!(r.x_axis().z, -0.76208216, epsilon = 1e-2);

        // Translation
        assert_relative_eq!(t.x, 0.15193805, epsilon = 1e-2);
        assert_relative_eq!(t.y, 0.057428963, epsilon = 1e-2);
        assert_relative_eq!(t.z, 0.9616908, epsilon = 1e-2);

        // Rodrigues Vector
        assert_relative_eq!(rvec.x, -0.6012375, epsilon = 1e-2);
        assert_relative_eq!(rvec.y, 0.8875437, epsilon = 1e-2);
        assert_relative_eq!(rvec.z, 0.13771825, epsilon = 1e-2);

        Ok(())
    }
}
