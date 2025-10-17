#![allow(clippy::op_ref)]
use glam::{Mat3, Vec3};
use nalgebra::{DMatrix, Vector3, Vector4, Matrix4};

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

#[inline(always)]
pub fn solve_4x4_cholesky(a: &Matrix4<f32>, b: &Vector4<f32>) -> Option<Vector4<f32>> {
    let l11 = a.m11.sqrt();
    if l11 == 0.0 { return None; }
    let l21 = a.m21 / l11;
    let l31 = a.m31 / l11;
    let l41 = a.m41 / l11;
    let l22_sq = a.m22 - l21 * l21;
    if l22_sq <= 0.0 { return None; }
    let l22 = l22_sq.sqrt();
    let l32 = (a.m32 - l31 * l21) / l22;
    let l42 = (a.m42 - l41 * l21) / l22;
    let l33_sq = a.m33 - l31 * l31 - l32 * l32;
    if l33_sq <= 0.0 { return None; }
    let l33 = l33_sq.sqrt();
    let l43 = (a.m43 - l41 * l31 - l42 * l32) / l33;
    let l44_sq = a.m44 - l41 * l41 - l42 * l42 - l43 * l43;
    if l44_sq <= 0.0 { return None; }
    let l44 = l44_sq.sqrt();
    let inv_l11 = 1.0 / l11;
    let inv_l22 = 1.0 / l22;
    let inv_l33 = 1.0 / l33;
    let inv_l44 = 1.0 / l44;
    let y1 = b[0] * inv_l11;
    let y2 = (b[1] - l21 * y1) * inv_l22;
    let y3 = (b[2] - (l31 * y1 + l32 * y2)) * inv_l33;
    let y4 = (b[3] - (l41 * y1 + l42 * y2 + l43 * y3)) * inv_l44;
    let x4 = y4 * inv_l44;
    let x3 = (y3 - l43 * x4) * inv_l33;
    let x2 = (y2 - (l32 * x3 + l42 * x4)) * inv_l22;
    let x1 = (y1 - (l21 * x2 + l31 * x3 + l41 * x4)) * inv_l11;
    Some(Vector4::new(x1, x2, x3, x4))
}

pub(crate) fn gauss_newton(beta_init: [f32; 4], null4: &DMatrix<f32>, rho: &[f32; 6]) -> [f32; 4] {
    const PAIRS: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    const DAMPING: f32 = 1e-9;
    const EPS: f32 = 1e-8;

    let mut bet = Vector4::from(beta_init);

    for _ in 0..6 {
        let mut vs = [Vector3::zeros(); 4];
        for i in 0..4 {
            let m = null4.view((i * 3, 0), (3, 4));
            vs[i] = Vector3::new(
                m[(0, 0)] * bet.x + m[(0, 1)] * bet.y + m[(0, 2)] * bet.z + m[(0, 3)] * bet.w,
                m[(1, 0)] * bet.x + m[(1, 1)] * bet.y + m[(1, 2)] * bet.z + m[(1, 3)] * bet.w,
                m[(2, 0)] * bet.x + m[(2, 1)] * bet.y + m[(2, 2)] * bet.z + m[(2, 3)] * bet.w,
            );
        }

        let mut f = [0.0f32; 6];
        let mut j = [[0.0f32; 4]; 6];
        for (r, &(i, jj)) in PAIRS.iter().enumerate() {
            let dx = vs[i].x - vs[jj].x;
            let dy = vs[i].y - vs[jj].y;
            let dz = vs[i].z - vs[jj].z;
            f[r] = dx * dx + dy * dy + dz * dz - rho[r];
            for k in 0..4 {
                let di0 = null4[(i * 3, k)] - null4[(jj * 3, k)];
                let di1 = null4[(i * 3 + 1, k)] - null4[(jj * 3 + 1, k)];
                let di2 = null4[(i * 3 + 2, k)] - null4[(jj * 3 + 2, k)];
                j[r][k] = 2.0 * (di0 * dx + di1 * dy + di2 * dz);
            }
        }

        let mut a = Matrix4::zeros();
        let mut b = Vector4::zeros();
        for r in 0..6 {
            for i in 0..4 {
                b[i] += j[r][i] * f[r];
                for k in 0..4 {
                    a[(i, k)] += j[r][i] * j[r][k];
                }
            }
        }

        a[(0, 0)] += DAMPING;
        a[(1, 1)] += DAMPING;
        a[(2, 2)] += DAMPING;
        a[(3, 3)] += DAMPING;

        if let Some(delta) = solve_4x4_cholesky(&a, &b) {
            bet -= delta;
            if delta.norm() < EPS {
                break;
            }
        } else {
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
}