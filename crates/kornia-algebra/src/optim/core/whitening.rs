//! Utilities for pre-whitening residuals and Jacobians by an information matrix.
//!
//! In factor graph optimization, the cost for a factor is `eᵀ Ω e` where Ω is the
//! information matrix (inverse covariance). Rather than modifying the solver to
//! handle per-factor information matrices, we pre-whiten: decompose Ω = LᵀL via
//! Cholesky, then return `L·e` and `L·J` from `linearize`. The solver then
//! computes `(LJ)ᵀ(LJ) = JᵀLᵀLJ = JᵀΩJ` and `(LJ)ᵀ(Le) = JᵀΩe` automatically.
//!
//! This is the standard approach used by Ceres, GTSAM, and g2o.

/// Whiten a residual and Jacobian by a scalar information weight.
///
/// For isotropic information `Ω = w·I`, the Cholesky factor is `L = √w·I`,
/// so whitening is just scaling by `√w`.
///
/// This is the common case for visual reprojection factors where `w = 1/σ²`
/// comes from the image pyramid octave.
pub fn whiten_scalar(residual: &mut [f32], jacobian: &mut [f32], information_weight: f32) {
    let sqrt_w = information_weight.sqrt();
    for r in residual.iter_mut() {
        *r *= sqrt_w;
    }
    for j in jacobian.iter_mut() {
        *j *= sqrt_w;
    }
}

/// Whiten a residual and Jacobian by a dense information matrix via Cholesky.
///
/// Given information matrix Ω (dim × dim), computes L from Ω = LᵀL, then
/// applies `residual ← L·residual` and `jacobian ← L·jacobian`.
///
/// The Jacobian is stored row-major with shape (residual_dim × total_local_dim).
///
/// Returns `false` if the Cholesky decomposition fails (matrix not positive definite).
pub fn whiten_matrix(
    residual: &mut [f32],
    jacobian: &mut [f32],
    information: &[f32],
    dim: usize,
    jacobian_cols: usize,
) -> bool {
    // Cholesky decomposition: Ω = LᵀL (upper triangular L)
    // We compute L such that LᵀL = Ω, then apply L to residual and Jacobian rows.
    //
    // We use the lower-triangular convention: Ω = L Lᵀ, then whiten with Lᵀ.
    // Actually for whitening we need: eᵀΩe = eᵀLᵀLe = (Le)ᵀ(Le)
    // So we need L from Ω = LᵀL (upper Cholesky), or equivalently Lᵀ from lower Cholesky.

    let mut l = vec![0.0f32; dim * dim];

    // Lower Cholesky: L such that L·Lᵀ = Ω
    for j in 0..dim {
        let mut sum = 0.0f32;
        for k in 0..j {
            sum += l[j * dim + k] * l[j * dim + k];
        }
        let diag = information[j * dim + j] - sum;
        if diag <= 0.0 {
            return false;
        }
        l[j * dim + j] = diag.sqrt();

        for i in (j + 1)..dim {
            let mut sum = 0.0f32;
            for k in 0..j {
                sum += l[i * dim + k] * l[j * dim + k];
            }
            l[i * dim + j] = (information[i * dim + j] - sum) / l[j * dim + j];
        }
    }

    // We have L such that L·Lᵀ = Ω.
    // We need to apply Lᵀ (upper triangular) to get whitened = Lᵀ · original.
    // Because (Lᵀ·e)ᵀ(Lᵀ·e) = eᵀ·L·Lᵀ·e = eᵀ·Ω·e. ✓

    // Whiten residual: r' = Lᵀ · r
    let mut whitened_r = vec![0.0f32; dim];
    for i in 0..dim {
        let mut val = 0.0f32;
        // Lᵀ[i,j] = L[j,i], nonzero for j >= i
        for j in i..dim {
            val += l[j * dim + i] * residual[j];
        }
        whitened_r[i] = val;
    }
    residual[..dim].copy_from_slice(&whitened_r);

    // Whiten Jacobian rows: J'[i, :] = Σ_j Lᵀ[i,j] · J[j, :]
    let mut whitened_j = vec![0.0f32; dim * jacobian_cols];
    for i in 0..dim {
        for j in i..dim {
            let lt_ij = l[j * dim + i]; // Lᵀ[i,j] = L[j,i]
            for c in 0..jacobian_cols {
                whitened_j[i * jacobian_cols + c] += lt_ij * jacobian[j * jacobian_cols + c];
            }
        }
    }
    jacobian[..dim * jacobian_cols].copy_from_slice(&whitened_j);

    true
}

/// Compute the inverse of a symmetric positive-definite matrix (covariance → information).
///
/// Uses Cholesky decomposition followed by back-substitution.
/// Returns `None` if the matrix is not positive definite.
pub fn invert_spd(matrix: &[f32], dim: usize) -> Option<Vec<f32>> {
    // Lower Cholesky: L such that L·Lᵀ = matrix
    let mut l = vec![0.0f32; dim * dim];

    for j in 0..dim {
        let mut sum = 0.0f32;
        for k in 0..j {
            sum += l[j * dim + k] * l[j * dim + k];
        }
        let diag = matrix[j * dim + j] - sum;
        if diag <= 0.0 {
            return None;
        }
        l[j * dim + j] = diag.sqrt();

        for i in (j + 1)..dim {
            let mut sum = 0.0f32;
            for k in 0..j {
                sum += l[i * dim + k] * l[j * dim + k];
            }
            l[i * dim + j] = (matrix[i * dim + j] - sum) / l[j * dim + j];
        }
    }

    // Invert L (lower triangular)
    let mut l_inv = vec![0.0f32; dim * dim];
    for i in 0..dim {
        l_inv[i * dim + i] = 1.0 / l[i * dim + i];
        for j in (i + 1)..dim {
            let mut sum = 0.0f32;
            for k in i..j {
                sum += l[j * dim + k] * l_inv[k * dim + i];
            }
            l_inv[j * dim + i] = -sum / l[j * dim + j];
        }
    }

    // Ω = (LLᵀ)⁻¹ = L⁻ᵀ L⁻¹
    let mut inv = vec![0.0f32; dim * dim];
    for i in 0..dim {
        for j in 0..=i {
            let mut sum = 0.0f32;
            for k in i..dim {
                sum += l_inv[k * dim + i] * l_inv[k * dim + j];
            }
            inv[i * dim + j] = sum;
            inv[j * dim + i] = sum;
        }
    }

    Some(inv)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_whitening_scales_by_sqrt() {
        let mut residual = vec![2.0f32, 3.0];
        let mut jacobian = vec![1.0f32, 0.0, 0.0, 1.0];
        let w = 4.0; // sqrt(4) = 2

        whiten_scalar(&mut residual, &mut jacobian, w);

        assert!((residual[0] - 4.0).abs() < 1e-6);
        assert!((residual[1] - 6.0).abs() < 1e-6);
        assert!((jacobian[0] - 2.0).abs() < 1e-6);
        assert!((jacobian[3] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn matrix_whitening_reproduces_mahalanobis() {
        // 2x2 information matrix: Ω = [[4, 1], [1, 2]]
        let info = vec![4.0f32, 1.0, 1.0, 2.0];
        let e_orig = vec![1.0f32, 2.0];

        let mut e = e_orig.clone();
        // Dummy 2x1 jacobian
        let mut j = vec![1.0f32, 0.0];

        assert!(whiten_matrix(&mut e, &mut j, &info, 2, 1));

        // Check: whitened eᵀe should equal eᵀΩe
        let whitened_sq: f32 = e.iter().map(|x| x * x).sum();
        // eᵀΩe = [1,2]·[[4,1],[1,2]]·[1,2]ᵀ = 1*4+1*1 + 2*1+2*2 = 5+6 = 12? no
        // = 1*(4*1+1*2) + 2*(1*1+2*2) = 1*6 + 2*5 = 6+10 = 16? no wait
        // eᵀΩe = e[0]*(Ω[0,0]*e[0]+Ω[0,1]*e[1]) + e[1]*(Ω[1,0]*e[0]+Ω[1,1]*e[1])
        //       = 1*(4+2) + 2*(1+4) = 6 + 10 = 16
        let expected: f32 = {
            let oe0 = info[0] * e_orig[0] + info[1] * e_orig[1];
            let oe1 = info[2] * e_orig[0] + info[3] * e_orig[1];
            e_orig[0] * oe0 + e_orig[1] * oe1
        };

        assert!(
            (whitened_sq - expected).abs() < 1e-4,
            "whitened {whitened_sq} != expected {expected}"
        );
    }

    #[test]
    fn invert_spd_identity() {
        let id = vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let inv = invert_spd(&id, 3).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (inv[i * 3 + j] - expected).abs() < 1e-6,
                    "inv[{i},{j}] = {} != {expected}",
                    inv[i * 3 + j]
                );
            }
        }
    }

    #[test]
    fn invert_spd_roundtrip() {
        // Ω = [[4, 1], [1, 2]], Σ = Ω⁻¹, then Ω·Σ = I
        let omega = vec![4.0f32, 1.0, 1.0, 2.0];
        let sigma = invert_spd(&omega, 2).unwrap();

        // Check Ω·Σ ≈ I
        for i in 0..2 {
            for j in 0..2 {
                let mut sum = 0.0f32;
                for k in 0..2 {
                    sum += omega[i * 2 + k] * sigma[k * 2 + j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (sum - expected).abs() < 1e-5,
                    "product[{i},{j}] = {sum} != {expected}"
                );
            }
        }
    }

    #[test]
    fn invert_spd_fails_on_non_positive_definite() {
        let bad = vec![1.0f32, 2.0, 2.0, 1.0]; // eigenvalues: 3, -1
        assert!(invert_spd(&bad, 2).is_none());
    }
}
