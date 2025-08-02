#![allow(clippy::op_ref)]
use glam::Vec3;
use nalgebra::{DMatrix, DVector, Vector4};

/// Compute the centroid of a set of points.
pub(crate) fn compute_centroid(pts: &[[f32; 3]]) -> [f32; 3] {
    let n = pts.len() as f32;
    let mut c = [0.0; 3];
    for p in pts {
        c[0] += p[0];
        c[1] += p[1];
        c[2] += p[2];
    }
    c[0] /= n;
    c[1] /= n;
    c[2] /= n;
    c
}

//TODO: Checkout faer for this
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

            let diff_vec = Vec3::new(vi[0] - vj[0], vi[1] - vj[1], vi[2] - vj[2]);

            f_vec[r] = diff_vec.length_squared() - rho[r];

            for k in 0..4 {
                let vi_k = block_i.column(k);
                let vj_k = block_j.column(k);
                let col_diff = Vec3::new(vi_k[0] - vj_k[0], vi_k[1] - vj_k[1], vi_k[2] - vj_k[2]);
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
        let pts = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let c = compute_centroid(&pts);
        assert_eq!(c, [4.0, 5.0, 6.0]);
    }
}

//TODO: redo this test
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
