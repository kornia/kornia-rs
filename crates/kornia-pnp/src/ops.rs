#![allow(clippy::op_ref)]
use glam::{Mat3, Vec3};
use nalgebra::{DMatrix, Matrix3x4, Matrix4, SMatrix, SVector, Vector3, Vector4};

/// Compute the centroid of a set of points.
pub(crate) fn compute_centroid(pts: &[[f32; 3]]) -> [f32; 3] {
    let n = pts.len() as f32;
    let sum = pts.iter().fold(Vec3::ZERO, |acc, &p| acc + Vec3::from(p));

    let centroid = sum / n;
    [centroid.x, centroid.y, centroid.z]
}

/// Construct compact intrinsics vectors used for fast projection.
pub(crate) fn intrinsics_as_vectors(k: &[[f32; 3]; 3]) -> (Vec3, Vec3) {
    let fx = k[0][0];
    let fy = k[1][1];
    let cx = k[0][2];
    let cy = k[1][2];
    (Vec3::new(fx, 0.0, cx), Vec3::new(0.0, fy, cy))
}

/// Convert array-form pose to glam matrices/vectors.
pub(crate) fn pose_to_rt(r: &[[f32; 3]; 3], t: &[f32; 3]) -> (Mat3, Vec3) {
    let r_mat = Mat3::from_cols(
        Vec3::new(r[0][0], r[1][0], r[2][0]),
        Vec3::new(r[0][1], r[1][1], r[2][1]),
        Vec3::new(r[0][2], r[1][2], r[2][2]),
    );
    let t_vec = Vec3::new(t[0], t[1], t[2]);
    (r_mat, t_vec)
}

/// Compute squared reprojection error for a single correspondence.
/// If `skip_if_behind` is true, returns `None` for points with non-positive depth.
pub(crate) fn project_sq_error(
    world_point: &[f32; 3],
    image_point: &[f32; 2],
    r_mat: &Mat3,
    t_vec: &Vec3,
    intr_x: &Vec3,
    intr_y: &Vec3,
    skip_if_behind: bool,
) -> Option<f32> {
    let pw = Vec3::from_array(*world_point);
    let pc = *r_mat * pw + *t_vec;
    if skip_if_behind && pc.z <= 0.0 {
        return None;
    }
    let inv_z = 1.0 / pc.z;
    let u_hat = intr_x.dot(pc) * inv_z;
    let v_hat = intr_y.dot(pc) * inv_z;
    let du = u_hat - image_point[0];
    let dv = v_hat - image_point[1];
    Some(du.mul_add(du, dv * dv))
}

const EPSILON: f32 = 1e-10;
const NUM_CONTROL_POINTS: usize = 4;
const MAX_ITERATIONS: usize = 6;
const PAIRS: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
const NUM_PAIRS: usize = PAIRS.len(); // 6

/// Solves the linear system A * x = b for a 4x4 symmetric positive-definite matrix A
/// using an unrolled Cholesky decomposition.
///
/// # Arguments
/// * `a` - A reference to a 4x4 matrix, assumed to be symmetric positive-definite.
/// * `b` - A reference to a 4x4 vector.
///
/// # Returns
/// * `Some(Vector4<f32>)` containing the solution vector `x` if `A` is positive-definite.
/// * `None` if the decomposition fails (i.e., `A` is not positive-definite).
#[inline(always)]
pub fn solve_4x4_cholesky(a: &Matrix4<f32>, b: &Vector4<f32>) -> Option<Vector4<f32>> {
    // --- Cholesky Decomposition (L * L^T = A) ---
    // First column of L
    let l_11 = a.m11.sqrt();
    if l_11 < EPSILON {
        return None;
    }
    let l_21 = a.m21 / l_11;
    let l_31 = a.m31 / l_11;
    let l_41 = a.m41 / l_11;

    // Second column of L
    let l_22_sq = a.m22 - l_21 * l_21;
    if l_22_sq < EPSILON {
        return None;
    }
    let l_22 = l_22_sq.sqrt();
    let l_32 = (a.m32 - l_31 * l_21) / l_22;
    let l_42 = (a.m42 - l_41 * l_21) / l_22;

    // Third column of L
    let l_33_sq = a.m33 - l_31 * l_31 - l_32 * l_32;
    if l_33_sq < EPSILON {
        return None;
    }
    let l_33 = l_33_sq.sqrt();
    let l_43 = (a.m43 - l_41 * l_31 - l_42 * l_32) / l_33;

    // Fourth column of L
    let l_44_sq = a.m44 - l_41 * l_41 - l_42 * l_42 - l_43 * l_43;
    if l_44_sq < EPSILON {
        return None;
    }
    let l_44 = l_44_sq.sqrt();

    // --- Solve L * y = b (Forward substitution) ---
    let inv_l11 = 1.0 / l_11;
    let inv_l22 = 1.0 / l_22;
    let inv_l33 = 1.0 / l_33;
    let inv_l44 = 1.0 / l_44;

    let y1 = b[0] * inv_l11;
    let y2 = (b[1] - l_21 * y1) * inv_l22;
    let y3 = (b[2] - (l_31 * y1 + l_32 * y2)) * inv_l33;
    let y4 = (b[3] - (l_41 * y1 + l_42 * y2 + l_43 * y3)) * inv_l44;

    // --- Solve L^T * x = y (Backward substitution) ---
    let x4 = y4 * inv_l44;
    let x3 = (y3 - l_43 * x4) * inv_l33;
    let x2 = (y2 - (l_32 * x3 + l_42 * x4)) * inv_l22;
    let x1 = (y1 - (l_21 * x2 + l_31 * x3 + l_41 * x4)) * inv_l11;
    Some(Vector4::new(x1, x2, x3, x4))
}

/// Performs optimization using the Gauss-Newton algorithm.
pub(crate) fn gauss_newton(beta_init: [f32; 4], null4: &DMatrix<f32>, rho: &[f32; 6]) -> [f32; 4] {
    const DAMPING: f32 = 1e-9;
    const STOP_EPS: f32 = 1e-8;

    let mut bet = Vector4::from(beta_init);
    let rho_vec = SVector::<f32, NUM_PAIRS>::from_row_slice(rho);

    for _ in 0..MAX_ITERATIONS {
        let mut vs = [Vector3::zeros(); NUM_CONTROL_POINTS];

        for (i, v) in vs.iter_mut().enumerate() {
            let m: Matrix3x4<f32> = null4.fixed_view::<3, 4>(i * 3, 0).into();
            *v = m * bet;
        }

        let mut f = SVector::<f32, NUM_PAIRS>::zeros();
        let mut j = SMatrix::<f32, NUM_PAIRS, NUM_CONTROL_POINTS>::zeros();

        for (r, &(i, jj)) in PAIRS.iter().enumerate() {
            let diff = vs[i] - vs[jj];
            f[r] = diff.norm_squared();

            let rows_i = null4.fixed_rows::<3>(i * 3);
            let rows_jj = null4.fixed_rows::<3>(jj * 3);

            for k in 0..NUM_CONTROL_POINTS {
                let d_col_i = rows_i.column(k);
                let d_col_jj = rows_jj.column(k);
                let d_col = d_col_i - d_col_jj;
                j[(r, k)] = 2.0 * diff.dot(&d_col);
            }
        }

        f -= rho_vec;

        let a = Matrix4::from(j.transpose() * j);
        let b = Vector4::from(j.transpose() * f);

        a.diagonal().add_scalar_mut(DAMPING);

        if let Some(delta) = solve_4x4_cholesky(&a, &b) {
            bet -= delta;
            if delta.norm() < STOP_EPS {
                break; // Converged
            }
        } else {
            // Cholesky failed, matrix is not positive-definite.
            break;
        }
    }

    bet.into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_centroid() {
        let pts = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let c = compute_centroid(&pts);
        assert_eq!(c, [4.0, 5.0, 6.0]);
    }
}

#[cfg(test)]
mod gauss_newton_tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::{DMatrix, Vector3, Vector4};

    fn setup_test_data() -> (DMatrix<f32>, [f32; 6], [f32; 4]) {
        #[rustfmt::skip]
        let null4 = DMatrix::from_row_slice(12, 4, &[
            0.1, 0.5, 0.2, 0.8, 0.4, 0.3, 0.6, 0.1, 0.7, 0.9, 0.3, 0.2,
            0.2, 0.1, 0.8, 0.5, 0.5, 0.4, 0.2, 0.9, 0.8, 0.7, 0.5, 0.3,
            0.3, 0.6, 0.9, 0.1, 0.6, 0.2, 0.4, 0.7, 0.9, 0.5, 0.7, 0.4,
            0.1, 0.8, 0.1, 0.6, 0.4, 0.3, 0.5, 0.2, 0.7, 0.6, 0.8, 0.9,
        ]);

        let beta_true = [0.5, -0.2, 0.8, 0.1];
        let beta_vec = Vector4::from(beta_true);

        let mut vs = [Vector3::zeros(); 4];
        for i in 0..4 {
            let m = null4.view((i * 3, 0), (3, 4));
            let v_dynamic = m * beta_vec;
            vs[i] = Vector3::new(v_dynamic[0], v_dynamic[1], v_dynamic[2]);
        }

        const PAIRS: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
        let mut rho = [0.0; 6];
        for (idx, &(i, j)) in PAIRS.iter().enumerate() {
            rho[idx] = (vs[i] - vs[j]).norm_squared();
        }

        (null4, rho, beta_true)
    }

    #[test]
    fn test_gauss_newton() {
        let (null4, rho, beta_true) = setup_test_data();
        let beta_init = [0.4, -0.1, 0.7, 0.2];
        let result = gauss_newton(beta_init, &null4, &rho);

        for i in 0..4 {
            assert_relative_eq!(result[i], beta_true[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_solve_4x4_cholesky_valid() {
        // Create a known symmetric positive-definite matrix
        // A = [ 5, -1,  0,  0]
        //     [-1,  5, -1,  0]
        //     [ 0, -1,  5, -1]
        //     [ 0,  0, -1,  5]
        let a = Matrix4::new(
            5.0, -1.0, 0.0, 0.0, -1.0, 5.0, -1.0, 0.0, 0.0, -1.0, 5.0, -1.0, 0.0, 0.0, -1.0, 5.0,
        );
        let b = Vector4::new(1.0, 2.0, 3.0, 4.0);

        // Known solution x for A*x = b
        let expected_x = Vector4::new(0.3303085, 0.6515426, 0.92740476, 0.98548114);
        if let Some(x) = solve_4x4_cholesky(&a, &b) {
            assert_relative_eq!(x, expected_x, epsilon = 1e-4);
        } else {
            panic!("Cholesky decomposition failed for a valid matrix.");
        }
    }

    #[test]
    fn test_solve_4x4_cholesky_non_positive_definite() {
        // Not positive-definite (m22 = -1.0)
        let a_non_pd = Matrix4::new(
            4.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, // This row makes it non-PD
            1.0, -1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 6.0,
        );

        let b = Vector4::new(1.0, 2.0, 3.0, 4.0);

        let result = solve_4x4_cholesky(&a_non_pd, &b);
        assert!(
            result.is_none(),
            "Solver should return None for non-positive-definite matrix."
        );
    }

    #[test]
    fn test_solve_4x4_cholesky_zero_on_diagonal() {
        // Not positive-definite (m11 = 0.0)
        let a_zero_diag = Matrix4::new(
            0.0, 1.0, 1.0, 1.0, // This row makes it non-PD
            1.0, 3.0, -1.0, 1.0, 1.0, -1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 6.0,
        );

        let b = Vector4::new(1.0, 2.0, 3.0, 4.0);

        let result = solve_4x4_cholesky(&a_zero_diag, &b);
        assert!(
            result.is_none(),
            "Solver should return None for matrix with zero on diagonal."
        );
    }
}
