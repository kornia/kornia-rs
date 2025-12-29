#![allow(clippy::op_ref)]

use kornia_algebra::{Mat3AF32, Vec2F32, Vec3AF32};
use nalgebra::{DMatrix, DVector, Vector4};

/// Compute the centroid of a set of points.
pub(crate) fn compute_centroid(pts: &[Vec3AF32]) -> Vec3AF32 {
    let n = pts.len() as f32;
    let sum = pts.iter().copied().fold(Vec3AF32::ZERO, |acc, p| acc + p);
    sum / n
}

/// Construct compact intrinsics vectors used for fast projection.
pub(crate) fn intrinsics_as_vectors(k: &Mat3AF32) -> (Vec3AF32, Vec3AF32) {
    // For pinhole intrinsics K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
    // columns are: x_axis=(fx,0,0), y_axis=(0,fy,0), z_axis=(cx,cy,1).
    let fx = k.x_axis().x;
    let fy = k.y_axis().y;
    let cx = k.z_axis().x;
    let cy = k.z_axis().y;
    (Vec3AF32::new(fx, 0.0, cx), Vec3AF32::new(0.0, fy, cy))
}

/// Compute squared reprojection error for a single correspondence.
/// If `skip_if_behind` is true, returns `None` for points with non-positive depth.
pub(crate) fn project_sq_error(
    world_point: &Vec3AF32,
    image_point: &Vec2F32,
    r_mat: &Mat3AF32,
    t_vec: &Vec3AF32,
    intr_x: &Vec3AF32,
    intr_y: &Vec3AF32,
    skip_if_behind: bool,
) -> Option<f32> {
    let pc = *r_mat * *world_point + *t_vec;
    if skip_if_behind && pc.z <= 0.0 {
        return None;
    }
    let inv_z = 1.0 / pc.z;
    let u_hat = intr_x.dot(pc) * inv_z;
    let v_hat = intr_y.dot(pc) * inv_z;
    let du = u_hat - image_point.x;
    let dv = v_hat - image_point.y;
    Some(du.mul_add(du, dv * dv))
}

pub(crate) fn gauss_newton(beta_init: [f32; 4], null4: &DMatrix<f32>, rho: &[f32; 6]) -> [f32; 4] {
    const PAIRS: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

    let mut bet = Vector4::from_column_slice(&beta_init);

    for _ in 0..6 {
        let mut f_vec = DVector::<f32>::zeros(6);
        let mut j_mat = DMatrix::<f32>::zeros(6, 4);

        for (r, (i, j)) in PAIRS.iter().enumerate() {
            // Vi = (null4 block rows 3*i..3*i+3) * bet
            let block_i = null4.view((*i * 3, 0), (3, 4));
            let block_j = null4.view((*j * 3, 0), (3, 4));

            let vi = &block_i * &bet; // 3×1 nalgebra vec
            let vj = &block_j * &bet;

            let diff_vec = Vec3AF32::new(vi[0] - vj[0], vi[1] - vj[1], vi[2] - vj[2]);

            f_vec[r] = diff_vec.dot(diff_vec) - rho[r];

            for k in 0..4 {
                let vi_k = block_i.column(k);
                let vj_k = block_j.column(k);
                let col_diff =
                    Vec3AF32::new(vi_k[0] - vj_k[0], vi_k[1] - vj_k[1], vi_k[2] - vj_k[2]);
                let deriv = col_diff.dot(diff_vec) * 2.0;
                j_mat[(r, k)] = deriv;
            }
        }

        let jt = j_mat.transpose();
        let a = &jt * &j_mat + DMatrix::<f32>::identity(4, 4) * 1e-9;
        let b = &jt * f_vec;

        if let Some(delta) = a.lu().solve(&b) {
            let norm_val = delta.norm();
            bet -= &delta;
            if norm_val < 1e-8 {
                break;
            }
        } else {
            break;
        }
    }

    [bet[0], bet[1], bet[2], bet[3]]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_centroid() {
        let pts = [
            Vec3AF32::new(1.0, 2.0, 3.0),
            Vec3AF32::new(4.0, 5.0, 6.0),
            Vec3AF32::new(7.0, 8.0, 9.0),
        ];
        let c = compute_centroid(&pts);
        assert_eq!(c, Vec3AF32::new(4.0, 5.0, 6.0));
    }
}

// TODO: redo this test
#[cfg(test)]
mod gauss_newton_tests {
    use super::*;

    #[test]
    fn test_gauss_newton() {
        let beta_init = [1.0, 2.0, 3.0, 4.0];
        let null4 = DMatrix::<f32>::zeros(12, 4);
        let rho = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let result = gauss_newton(beta_init, &null4, &rho);
        assert_eq!(result, [1.0, 2.0, 3.0, 4.0]);
    }
}
