use nalgebra::{DMatrix, DVector, Vector4};

/// Compute the centroid of a set of points.
pub(crate) fn compute_centroid(pts: &[[f64; 3]]) -> [f64; 3] {
    let n = pts.len() as f64;
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
pub(crate) fn gauss_newton(
    beta_init: [f64; 4],
    null4: &DMatrix<f64>,
    rho: &[f64; 6],
) -> [f64; 4] {
    const PAIRS: [(usize, usize); 6] = [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
    ];

    let mut bet = Vector4::from_column_slice(&beta_init);

    for _ in 0..6 {
        let mut f_vec = DVector::<f64>::zeros(6);
        let mut j_mat = DMatrix::<f64>::zeros(6, 4);

        for (r, (i, j)) in PAIRS.iter().enumerate() {
            // Vi = (null4 block rows 3*i..3*i+3) * bet
            let block_i = null4.view((*i * 3, 0), (3, 4));
            let block_j = null4.view((*j * 3, 0), (3, 4));

            let vi = &block_i * &bet;
            let vj = &block_j * &bet;
            let diff = &vi - &vj; // 3-vector

            f_vec[r] = diff.dot(&diff) - rho[r];

            for k in 0..4 {
                let vi_k = block_i.column(k);
                let vj_k = block_j.column(k);
                let deriv = (&vi_k - &vj_k).dot(&diff) * 2.0;
                j_mat[(r, k)] = deriv;
            }
        }

        let jt = j_mat.transpose();
        let a = &jt * &j_mat + DMatrix::<f64>::identity(4, 4) * 1e-9;
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
        let pts = vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let c = compute_centroid(&pts);
        assert_eq!(c, [2.5, 3.5, 4.5]);
    }
}

//TODO: redo this test
#[cfg(test)]
mod gauss_newton_tests {
    use super::*;

    #[test]
    fn test_gauss_newton() {
        let beta_init = [1.0, 2.0, 3.0, 4.0];
        let null4 = DMatrix::<f64>::zeros(12, 4);
        let rho = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let result = gauss_newton(beta_init, &null4, &rho);
        assert_eq!(result, [1.0, 2.0, 3.0, 4.0]);
    }
}