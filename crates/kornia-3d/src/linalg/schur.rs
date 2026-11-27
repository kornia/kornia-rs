//! Schur complement of a quadratic: the marginal prior left when variables are
//! eliminated.
//!
//! For the quadratic `½ xᵀHx + bᵀx` over `keep ∪ marg` (`b` the gradient at
//! `x = 0`), minimizing over the `marg` variables leaves
//! `½ x_kᵀH'x_k + b'ᵀx_k + c` with `H' = H_kk − H_km H_mm⁻¹ H_mk`,
//! `b' = b_k − H_km H_mm⁻¹ b_m` and `c = −½ b_mᵀ H_mm⁻¹ b_m`: the prior a sliding
//! window keeps after marginalizing old states (Sibley et al., "Sliding window
//! filter with application to planetary landing", JFR 2010; Leutenegger et al.,
//! OKVIS, IJRR 2015).
//!
//! `H_mm` is factored by Cholesky. When it is not positive definite, its
//! pseudo-inverse from a symmetric eigendecomposition is used, with
//! eigenvalues at or below `n · ε · λ_max` cut. Basalt always pseudo-inverts
//! (`marg_helper.cpp`); this keeps the cheaper and exact Cholesky path when it
//! applies. When `b_m` has a component outside the range of a singular `H_mm`,
//! the quadratic is unbounded below in the eliminated variables; the
//! pseudo-inverse then returns the prior of the projection of `b_m` onto that
//! range, not a minimum. Runs on faer, beside [`dense_llt`](crate::linalg::dense_llt), rather than on the
//! kornia-algebra (nalgebra) types other new solvers use: its callers already hold faer
//! matrices, and a conversion would change the bits of their priors.

use faer::prelude::Solve;
use faer::{Mat, Side};

/// Errors of [`schur_complement`].
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum SchurError {
    /// `h` is not square, or `b` or a vector argument does not match its size.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch {
        /// The expected length.
        expected: usize,
        /// The offending length.
        got: usize,
    },
    /// An index is out of range, or appears twice across `keep` and `marg`.
    #[error("invalid index {index} for a system of size {size}")]
    InvalidIndex {
        /// The offending index.
        index: usize,
        /// The size of the system.
        size: usize,
    },
    /// The eigendecomposition of a singular `H_mm` did not converge.
    #[error("eigendecomposition of the marginal block did not converge")]
    EigenFailed,
    /// The result holds a non-finite value.
    #[error("the Schur complement is not finite")]
    NonFinite,
}

/// The prior left when variables are eliminated from `½ xᵀHx + bᵀx`.
#[derive(Debug, Clone)]
pub struct SchurComplement {
    /// `H' = H_kk − H_km H_mm⁻¹ H_mk`, symmetrized, over the `keep` indices in
    /// the order given.
    pub h: Mat<f64>,
    /// `b' = b_k − H_km H_mm⁻¹ b_m`.
    pub b: Vec<f64>,
    /// `c = −½ b_mᵀ H_mm⁻¹ b_m`, the minimum over the eliminated variables at
    /// `x_k = 0` (see [`Self::pseudo_inverse`] for the singular case).
    pub constant: f64,
    /// `H_mm` was not positive definite, so its pseudo-inverse was used. Each
    /// right-hand side is then projected onto the range of `H_mm`: when `b_m`
    /// or `H_mk` reaches outside it, the original quadratic has no minimum over
    /// the eliminated variables and the prior is that of the projection.
    pub pseudo_inverse: bool,
    /// `H_mm` had an eigenvalue below `−n · ε · λ_max`: it is indefinite, not
    /// only rank-deficient. Those directions were dropped with the
    /// near-zero ones, so the prior is not the marginal of a convex quadratic;
    /// a Hessian assembled from residuals is never indefinite, so this points
    /// at a sign or assembly error upstream.
    pub indefinite: bool,
}

impl SchurComplement {
    /// Gradient of the prior at `d`, `b' + H' d`.
    ///
    /// # Arguments
    ///
    /// * `d` - A point in the kept variables, one entry per `keep` index.
    ///
    /// # Returns
    ///
    /// The gradient, one entry per kept variable.
    ///
    /// # Errors
    ///
    /// [`SchurError::DimensionMismatch`] when `d` does not have one entry per
    /// kept variable, or `h` is not `b.len()` square.
    pub fn gradient(&self, d: &[f64]) -> Result<Vec<f64>, SchurError> {
        self.check_len(d)?;
        Ok((0..self.b.len())
            .map(|r| self.b[r] + (0..d.len()).map(|c| self.h[(r, c)] * d[c]).sum::<f64>())
            .collect())
    }

    /// Value of the prior at `d`, `½ dᵀH'd + b'ᵀd + c`.
    ///
    /// # Arguments
    ///
    /// * `d` - A point in the kept variables, one entry per `keep` index.
    ///
    /// # Returns
    ///
    /// The minimum of the original quadratic over the eliminated variables,
    /// with the kept variables at `d` (the projected value when
    /// [`Self::pseudo_inverse`] is set and no minimum exists).
    ///
    /// # Errors
    ///
    /// [`SchurError::DimensionMismatch`] when `d` does not have one entry per
    /// kept variable, or `h` is not `b.len()` square.
    pub fn cost(&self, d: &[f64]) -> Result<f64, SchurError> {
        self.check_len(d)?;
        let n = d.len();
        let quadratic: f64 = (0..n)
            .map(|r| d[r] * (0..n).map(|c| self.h[(r, c)] * d[c]).sum::<f64>())
            .sum();
        let linear: f64 = (0..n).map(|r| self.b[r] * d[r]).sum();
        Ok(0.5 * quadratic + linear + self.constant)
    }

    fn check_len(&self, d: &[f64]) -> Result<(), SchurError> {
        let n = self.b.len();
        // The fields are public, so `h` can be swapped independently of `b`.
        for got in [self.h.nrows(), self.h.ncols(), d.len()] {
            if got != n {
                return Err(SchurError::DimensionMismatch { expected: n, got });
            }
        }
        Ok(())
    }
}

/// Eliminates the variables `marg` from `½ xᵀHx + bᵀx`, keeping `keep`.
///
/// # Arguments
///
/// * `h` - Symmetric `n`×`n` Hessian; only `keep` and `marg` rows and columns are read.
/// * `b` - Gradient at `x = 0`, length `n`.
/// * `keep` - Indices of the variables to keep, in the order of the result.
/// * `marg` - Indices of the variables to eliminate.
///
/// # Returns
///
/// The [`SchurComplement`] over `keep`. With `marg` empty it is the `keep`
/// block of `h` and `b`, unchanged, with `constant = 0`. A singular `H_mm` is
/// pseudo-inverted and flagged in [`SchurComplement::pseudo_inverse`], and an
/// indefinite one also in [`SchurComplement::indefinite`].
///
/// # Errors
///
/// [`SchurError::DimensionMismatch`] when `h` is not square or `b` does not
/// match it; [`SchurError::InvalidIndex`] for an index `>= n` or one that
/// appears twice across `keep` and `marg`; [`SchurError::EigenFailed`] when a
/// singular `H_mm` cannot be decomposed; [`SchurError::NonFinite`] when the
/// result is not finite.
///
/// # Example
///
/// ```rust
/// use faer::Mat;
/// use kornia_3d::linalg::schur::schur_complement;
///
/// // ½ xᵀHx + bᵀx over (x0, x1); eliminate x1.
/// let h = Mat::from_fn(2, 2, |r, c| [[4.0, 1.0], [1.0, 2.0]][r][c]);
/// let b = [1.0, -1.0];
/// let prior = schur_complement(&h, &b, &[0], &[1])?;
/// assert!((prior.h[(0, 0)] - 3.5).abs() < 1e-12); // 4 − 1·½·1
/// assert!((prior.b[0] - 1.5).abs() < 1e-12); // 1 − 1·½·(−1)
/// assert!((prior.constant + 0.25).abs() < 1e-12); // −½ · 1 · ½
/// # Ok::<(), kornia_3d::linalg::schur::SchurError>(())
/// ```
pub fn schur_complement(
    h: &Mat<f64>,
    b: &[f64],
    keep: &[usize],
    marg: &[usize],
) -> Result<SchurComplement, SchurError> {
    let n = h.nrows();
    if h.ncols() != n {
        return Err(SchurError::DimensionMismatch {
            expected: n,
            got: h.ncols(),
        });
    }
    if b.len() != n {
        return Err(SchurError::DimensionMismatch {
            expected: n,
            got: b.len(),
        });
    }
    let mut seen = vec![false; n];
    for &index in keep.iter().chain(marg) {
        if index >= n || seen[index] {
            return Err(SchurError::InvalidIndex { index, size: n });
        }
        seen[index] = true;
    }

    let (nk, nm) = (keep.len(), marg.len());
    if nm == 0 {
        return finite(SchurComplement {
            h: Mat::<f64>::from_fn(nk, nk, |r, c| h[(keep[r], keep[c])]),
            b: keep.iter().map(|&k| b[k]).collect(),
            constant: 0.0,
            pseudo_inverse: false,
            indefinite: false,
        });
    }
    let h_mm = Mat::<f64>::from_fn(nm, nm, |r, c| h[(marg[r], marg[c])]);
    // The eigen path can return NaN eigenvalues that the cut then drops,
    // hiding a non-finite H_mm from the final check.
    if (0..nm).any(|c| h_mm.col_as_slice(c).iter().any(|v| !v.is_finite())) {
        return Err(SchurError::NonFinite);
    }
    // [H_mk | b_m]: one solve gives H_mm⁻¹ H_mk and H_mm⁻¹ b_m.
    let rhs = Mat::<f64>::from_fn(nm, nk + 1, |r, c| match c {
        c if c == nk => b[marg[r]],
        c => h[(marg[r], keep[c])],
    });
    let (x, pseudo_inverse, indefinite) = match h_mm.llt(Side::Lower) {
        Ok(chol) => (chol.solve(&rhs), false, false),
        Err(_) => {
            let (inverse, indefinite) = pseudo_inverse_of(&h_mm)?;
            (inverse * &rhs, true, indefinite)
        }
    };
    let correction = |r: usize, c: usize| {
        (0..nm)
            .map(|k| h[(keep[r], marg[k])] * x[(k, c)])
            .sum::<f64>()
    };
    let mut out = Mat::<f64>::from_fn(nk, nk, |r, c| h[(keep[r], keep[c])] - correction(r, c));
    for r in 0..nk {
        for c in (r + 1)..nk {
            // Halve before adding: the sum of two finite entries can overflow.
            let avg = 0.5 * out[(r, c)] + 0.5 * out[(c, r)];
            out[(r, c)] = avg;
            out[(c, r)] = avg;
        }
    }
    let b_out: Vec<f64> = (0..nk).map(|r| b[keep[r]] - correction(r, nk)).collect();
    let constant = -0.5 * (0..nm).map(|k| b[marg[k]] * x[(k, nk)]).sum::<f64>();
    finite(SchurComplement {
        h: out,
        b: b_out,
        constant,
        pseudo_inverse,
        indefinite,
    })
}

/// `prior`, or [`SchurError::NonFinite`] when it holds a non-finite value.
fn finite(prior: SchurComplement) -> Result<SchurComplement, SchurError> {
    let nk = prior.b.len();
    let finite = (0..nk).all(|c| prior.h.col_as_slice(c).iter().all(|v| v.is_finite()))
        && prior.b.iter().all(|v| v.is_finite())
        && prior.constant.is_finite();
    if finite {
        Ok(prior)
    } else {
        Err(SchurError::NonFinite)
    }
}

/// Pseudo-inverse of the symmetric `a`, eigenvalues at or below
/// `n · ε · λ_max` cut, and whether one was below `−n · ε · λ_max`.
fn pseudo_inverse_of(a: &Mat<f64>) -> Result<(Mat<f64>, bool), SchurError> {
    let n = a.nrows();
    let eigen = a
        .self_adjoint_eigen(Side::Lower)
        .map_err(|_| SchurError::EigenFailed)?;
    let values = eigen.S().column_vector();
    let vectors = eigen.U();
    let largest = (0..n).map(|i| values[i]).fold(0.0f64, f64::max);
    let cut = n as f64 * f64::EPSILON * largest;
    let indefinite = (0..n).any(|i| values[i] < -cut);
    let mut out = Mat::<f64>::zeros(n, n);
    for i in (0..n).filter(|&i| values[i] > cut) {
        let inv = 1.0 / values[i];
        for c in 0..n {
            for r in 0..n {
                out[(r, c)] += vectors[(r, i)] * inv * vectors[(c, i)];
            }
        }
    }
    Ok((out, indefinite))
}

#[cfg(test)]
mod tests {
    use super::*;

    const N: usize = 9;

    /// `BᵀB + I` from a fixed LCG: symmetric positive definite.
    fn spd(n: usize, seed: u64) -> Mat<f64> {
        let mut state = seed;
        let b = Mat::<f64>::from_fn(n, n, |_, _| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        });
        let mut a = b.transpose() * &b;
        for i in 0..n {
            a[(i, i)] += 1.0;
        }
        a
    }

    fn gradient() -> Vec<f64> {
        (0..N).map(|i| (i as f64 * 0.7).sin()).collect()
    }

    /// `½ xᵀHx + bᵀx`.
    fn full_cost(h: &Mat<f64>, b: &[f64], x: &[f64]) -> f64 {
        let n = x.len();
        let quadratic: f64 = (0..n)
            .map(|r| x[r] * (0..n).map(|c| h[(r, c)] * x[c]).sum::<f64>())
            .sum();
        0.5 * quadratic + (0..n).map(|r| b[r] * x[r]).sum::<f64>()
    }

    #[test]
    fn schur_matches_the_full_solve() {
        let h = spd(N, 7);
        let b = gradient();
        let keep = [7, 1, 4, 0];
        let marg = [2, 3, 5, 6, 8];
        let prior = schur_complement(&h, &b, &keep, &marg).unwrap();
        assert!(!prior.pseudo_inverse);

        // Joint minimum: H x = −b.
        let neg_b = Mat::<f64>::from_fn(N, 1, |r, _| -b[r]);
        let x = h.llt(Side::Lower).unwrap().solve(&neg_b);
        // Reduced minimum: H' x_k = −b'.
        let neg_b_k = Mat::<f64>::from_fn(keep.len(), 1, |r, _| -prior.b[r]);
        let x_k = prior.h.llt(Side::Lower).unwrap().solve(&neg_b_k);
        for (r, &k) in keep.iter().enumerate() {
            assert!((x_k[(r, 0)] - x[(k, 0)]).abs() < 1e-10, "x[{k}]");
        }
        // The prior's minimum is the joint minimum, so the constant is right.
        let x_full: Vec<f64> = (0..N).map(|r| x[(r, 0)]).collect();
        let x_kept: Vec<f64> = (0..keep.len()).map(|r| x_k[(r, 0)]).collect();
        let reduced = prior.cost(&x_kept).unwrap();
        assert!((reduced - full_cost(&h, &b, &x_full)).abs() < 1e-10);
    }

    #[test]
    fn a_singular_marginal_block_uses_the_pseudo_inverse() {
        // x2 and x3 enter only through x2 + x3: H_mm has rank 1.
        let mut h = Mat::<f64>::zeros(4, 4);
        let a = [[3.0, 0.5], [0.5, 2.0]];
        for r in 0..2 {
            for c in 0..2 {
                h[(r, c)] = a[r][c];
            }
        }
        let coupling = [0.4, -0.3];
        for r in 0..2 {
            for m in 2..4 {
                h[(r, m)] = coupling[r];
                h[(m, r)] = coupling[r];
            }
        }
        for m in 2..4 {
            for n in 2..4 {
                h[(m, n)] = 1.0;
            }
        }
        let b = [0.2, -0.1, 0.6, 0.6];
        let prior = schur_complement(&h, &b, &[0, 1], &[2, 3]).unwrap();
        assert!(prior.pseudo_inverse);
        // Equivalent to one variable s = x2 + x3 with H_ss = 4, H_ks = 2·coupling, b_s = 1.2:
        // the pseudo-inverse of [[1, 1], [1, 1]] is [[¼, ¼], [¼, ¼]].
        for r in 0..2 {
            for c in 0..2 {
                let expected = a[r][c] - 0.25 * (2.0 * coupling[r]) * (2.0 * coupling[c]);
                assert!((prior.h[(r, c)] - expected).abs() < 1e-12);
            }
            let expected_b = b[r] - 0.25 * (2.0 * coupling[r]) * (2.0 * 0.6);
            assert!((prior.b[r] - expected_b).abs() < 1e-12);
        }
        assert!((prior.constant + 0.5 * 0.25 * 1.2 * 1.2).abs() < 1e-12);
        assert!(!prior.indefinite, "rank-deficient, not indefinite");
    }

    #[test]
    fn an_indefinite_marginal_block_is_flagged() {
        let mut h = spd(3, 19);
        // Flip the sign of the marginal block: every eigenvalue below zero.
        for r in 1..3 {
            for c in 1..3 {
                h[(r, c)] = -h[(r, c)];
            }
        }
        let prior = schur_complement(&h, &[0.1, 0.2, 0.3], &[0], &[1, 2]).unwrap();
        assert!(prior.pseudo_inverse && prior.indefinite);
    }

    #[test]
    fn a_non_finite_passthrough_is_refused() {
        let mut h = spd(3, 23);
        h[(0, 0)] = f64::NAN;
        assert_eq!(
            schur_complement(&h, &[0.0; 3], &[0, 1], &[]).unwrap_err(),
            SchurError::NonFinite
        );
        assert_eq!(
            schur_complement(&spd(3, 23), &[f64::INFINITY, 0.0, 0.0], &[0], &[]).unwrap_err(),
            SchurError::NonFinite
        );
    }

    #[test]
    fn a_non_finite_marginal_block_is_refused() {
        // Cholesky fails on the −1 pivot; the eigen path must not drop the NaN.
        let mut h = Mat::<f64>::zeros(3, 3);
        h[(0, 0)] = 2.0;
        h[(1, 1)] = -1.0;
        h[(2, 2)] = f64::NAN;
        assert_eq!(
            schur_complement(&h, &[0.0; 3], &[0], &[1, 2]).unwrap_err(),
            SchurError::NonFinite
        );
    }

    #[test]
    fn symmetrizing_large_finite_entries_does_not_overflow() {
        let rows = [
            [1.5e308, 1e308, 0.0],
            [1e308, 1.5e308, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let h = Mat::<f64>::from_fn(3, 3, |r, c| rows[r][c]);
        let prior = schur_complement(&h, &[0.0; 3], &[0, 1], &[2]).unwrap();
        assert_eq!(prior.h[(0, 1)], 1e308);
        assert_eq!(prior.h[(1, 0)], 1e308);
    }

    #[test]
    fn a_reshaped_prior_is_refused() {
        let mut prior = schur_complement(&spd(3, 29), &[0.1; 3], &[0], &[1, 2]).unwrap();
        prior.h = Mat::zeros(0, 0);
        let mismatch = SchurError::DimensionMismatch {
            expected: 1,
            got: 0,
        };
        assert_eq!(prior.cost(&[0.0]).unwrap_err(), mismatch);
        assert_eq!(prior.gradient(&[0.0]).unwrap_err(), mismatch);
    }

    #[test]
    fn no_marginal_indices_is_a_passthrough() {
        let h = spd(N, 11);
        let b = gradient();
        let keep = [3, 0, 8];
        let prior = schur_complement(&h, &b, &keep, &[]).unwrap();
        for (r, &kr) in keep.iter().enumerate() {
            for (c, &kc) in keep.iter().enumerate() {
                assert_eq!(prior.h[(r, c)].to_bits(), h[(kr, kc)].to_bits());
            }
            assert_eq!(prior.b[r].to_bits(), b[kr].to_bits());
        }
        assert_eq!(prior.constant, 0.0);
        assert!(!prior.pseudo_inverse);
    }

    #[test]
    fn cost_gradient_is_its_derivative() {
        let h = spd(N, 13);
        let b = gradient();
        let prior = schur_complement(&h, &b, &[0, 2, 4, 6], &[1, 3, 5, 7, 8]).unwrap();
        let d = [0.3, -0.2, 0.5, 0.1];
        let g = prior.gradient(&d).unwrap();
        let step = 1e-6;
        for i in 0..d.len() {
            let mut plus = d;
            plus[i] += step;
            let mut minus = d;
            minus[i] -= step;
            let numeric = (prior.cost(&plus).unwrap() - prior.cost(&minus).unwrap()) / (2.0 * step);
            assert!(
                (numeric - g[i]).abs() < 1e-8,
                "g[{i}] {} vs {numeric}",
                g[i]
            );
        }
        assert!(matches!(
            prior.gradient(&[0.0; 3]),
            Err(SchurError::DimensionMismatch {
                expected: 4,
                got: 3
            })
        ));
    }

    #[test]
    fn invalid_indices_are_refused() {
        let h = spd(4, 17);
        let b = [0.0; 4];
        assert_eq!(
            schur_complement(&h, &b, &[0, 4], &[1]).unwrap_err(),
            SchurError::InvalidIndex { index: 4, size: 4 }
        );
        assert_eq!(
            schur_complement(&h, &b, &[0, 1], &[1, 2]).unwrap_err(),
            SchurError::InvalidIndex { index: 1, size: 4 }
        );
        assert_eq!(
            schur_complement(&h, &b, &[0, 0], &[]).unwrap_err(),
            SchurError::InvalidIndex { index: 0, size: 4 }
        );
        assert_eq!(
            schur_complement(&h, &[0.0; 3], &[0], &[1]).unwrap_err(),
            SchurError::DimensionMismatch {
                expected: 4,
                got: 3
            }
        );
    }
}
