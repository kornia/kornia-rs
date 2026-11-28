//! A reusable dense Cholesky factor and solve on an explicit [`faer::Par`].
//!
//! faer's `Mat::llt` and `Llt::solve` always read the process-wide
//! parallelism and allocate the factor and its scratch on every call.
//! [`DenseLlt`](crate::linalg::dense_llt::DenseLlt) calls the same low-level routines
//! (`faer::linalg::cholesky::llt`) with the same inputs (a zero-upper copy of
//! the lower triangle, right-hand sides held in a `Mat`), so on the global
//! parallelism its result is faer's bit for bit. It keeps the factor, its
//! scratch and a solve column across calls, for solvers that factor a system
//! of one size every iteration.

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt::factor::{cholesky_in_place, cholesky_in_place_scratch, LltError};
use faer::linalg::cholesky::llt::solve::{solve_in_place_scratch, solve_in_place_with_conj};
use faer::{Conj, Mat, Par};

/// Errors of [`DenseLlt`].
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum DenseLltError {
    /// The matrix is not positive definite: a pivot was not positive.
    #[error("matrix is not positive definite (pivot {index})")]
    NotPositiveDefinite {
        /// Index of the first non-positive pivot.
        index: usize,
    },
    /// A matrix or vector does not match the factor's dimension.
    #[error("dimension mismatch: expected {expected} rows, got {got}")]
    DimensionMismatch {
        /// The factor's dimension.
        expected: usize,
        /// The offending number of rows (or columns, for a non-square matrix).
        got: usize,
    },
    /// A solve ran before a successful [`DenseLlt::factor`], or after a failed one.
    #[error("no valid factor: call factor successfully before solving")]
    NotFactored,
}

/// A reused lower Cholesky factor `A = L Lᵀ` of a `dim`×`dim` matrix.
///
/// # Example
///
/// ```rust
/// use kornia_3d::linalg::dense_llt::DenseLlt;
/// use faer::{Mat, Par};
///
/// let a = Mat::from_fn(3, 3, |i, j| if i == j { 4.0 } else { 1.0 });
/// let mut llt = DenseLlt::new(3, Par::Seq);
/// llt.factor(&a)?;
/// let x = llt.solve_column(&[6.0, 6.0, 6.0])?;
/// assert!(x.iter().all(|v| (v - 1.0).abs() < 1e-12));
/// # Ok::<(), kornia_3d::linalg::dense_llt::DenseLltError>(())
/// ```
pub struct DenseLlt {
    par: Par,
    l: Mat<f64>,
    /// `l` holds a complete factor: set by a successful `factor`, cleared by a
    /// failed one, which can leave `l` half overwritten.
    factored: bool,
    column: Mat<f64>,
    scratch: MemBuffer,
}

impl DenseLlt {
    /// Allocates the factor, a solve column and the factorization scratch.
    ///
    /// # Arguments
    ///
    /// * `dim` - Dimension of the matrices to factor.
    /// * `par` - Parallelism of the factor and the solves, e.g. `Par::Seq` or
    ///   `faer::get_global_parallelism()`.
    ///
    /// # Returns
    ///
    /// An unfactored `DenseLlt`; call [`DenseLlt::factor`] before any solve.
    ///
    /// # Errors
    ///
    /// This function is infallible.
    pub fn new(dim: usize, par: Par) -> Self {
        Self {
            par,
            l: Mat::zeros(dim, dim),
            factored: false,
            column: Mat::zeros(dim, 1),
            scratch: MemBuffer::new(cholesky_in_place_scratch::<f64>(
                dim,
                par,
                Default::default(),
            )),
        }
    }

    /// The dimension this factor was built for.
    ///
    /// # Returns
    ///
    /// The number of rows of the factored matrix.
    ///
    /// # Errors
    ///
    /// This function is infallible.
    pub fn dim(&self) -> usize {
        self.l.nrows()
    }

    /// Factors the lower triangle of `a`; the upper triangle is not read.
    ///
    /// # Arguments
    ///
    /// * `a` - Symmetric positive-definite `dim`×`dim` matrix.
    ///
    /// # Returns
    ///
    /// `Ok(())` once the factor is ready for solves.
    ///
    /// # Errors
    ///
    /// [`DenseLltError::DimensionMismatch`] when `a` is not `dim`×`dim`;
    /// [`DenseLltError::NotPositiveDefinite`] when a pivot is not positive.
    /// After an error the factor is invalid, and solves return
    /// [`DenseLltError::NotFactored`] until a factor succeeds.
    pub fn factor(&mut self, a: &Mat<f64>) -> Result<(), DenseLltError> {
        self.factored = false;
        self.check_rows(a.nrows())?;
        self.check_rows(a.ncols())?;
        // As `Llt::new`: a zeroed matrix holding only the lower triangle.
        self.l.fill(0.0);
        self.l.copy_from_triangular_lower(a);
        cholesky_in_place(
            self.l.as_mut(),
            Default::default(),
            self.par,
            MemStack::new(&mut self.scratch),
            Default::default(),
        )
        .map_err(|error| match error {
            LltError::NonPositivePivot { index } => DenseLltError::NotPositiveDefinite { index },
        })?;
        self.factored = true;
        Ok(())
    }

    /// Overwrites `rhs` with `A⁻¹ rhs`.
    ///
    /// # Arguments
    ///
    /// * `rhs` - `dim`×`k` right-hand sides, replaced by the solutions.
    ///
    /// # Returns
    ///
    /// `Ok(())` once `rhs` holds the solutions.
    ///
    /// # Errors
    ///
    /// [`DenseLltError::NotFactored`] without a successful [`DenseLlt::factor`];
    /// [`DenseLltError::DimensionMismatch`] when `rhs` does not have `dim` rows.
    pub fn solve_in_place(&self, rhs: &mut Mat<f64>) -> Result<(), DenseLltError> {
        self.check_factored()?;
        self.check_rows(rhs.nrows())?;
        solve(&self.l, self.par, rhs);
        Ok(())
    }

    /// `A⁻¹ b`, in a reused column.
    ///
    /// The column stays a `Mat`: faer's kernels can depend on alignment, and
    /// faer's own solve runs on a `Mat`, so the bits match it.
    ///
    /// # Arguments
    ///
    /// * `b` - Right-hand side of length `dim`.
    ///
    /// # Returns
    ///
    /// The solution, borrowed from the reused column until the next call.
    ///
    /// # Errors
    ///
    /// [`DenseLltError::NotFactored`] without a successful [`DenseLlt::factor`];
    /// [`DenseLltError::DimensionMismatch`] when `b` does not have `dim` entries.
    pub fn solve_column(&mut self, b: &[f64]) -> Result<&[f64], DenseLltError> {
        self.check_factored()?;
        self.check_rows(b.len())?;
        self.column.col_as_slice_mut(0).copy_from_slice(b);
        solve(&self.l, self.par, &mut self.column);
        Ok(self.column.col_as_slice(0))
    }

    fn check_factored(&self) -> Result<(), DenseLltError> {
        if self.factored {
            Ok(())
        } else {
            Err(DenseLltError::NotFactored)
        }
    }

    fn check_rows(&self, got: usize) -> Result<(), DenseLltError> {
        if got == self.dim() {
            Ok(())
        } else {
            Err(DenseLltError::DimensionMismatch {
                expected: self.dim(),
                got,
            })
        }
    }
}

/// As faer's `SolveCore for Llt`, on `par`.
fn solve(l: &Mat<f64>, par: Par, rhs: &mut Mat<f64>) {
    // Zero-size: no heap allocation.
    let mut mem = MemBuffer::new(solve_in_place_scratch::<f64>(l.nrows(), rhs.ncols(), par));
    solve_in_place_with_conj(
        l.as_ref(),
        Conj::No,
        rhs.as_mut(),
        par,
        MemStack::new(&mut mem),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::prelude::Solve;
    use faer::Side;

    const N: usize = 150;

    /// `BᵀB + N·I` from a fixed LCG: symmetric positive definite.
    fn spd() -> Mat<f64> {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let b = Mat::<f64>::from_fn(N, N, |_, _| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        });
        let mut a = b.transpose() * &b;
        for i in 0..N {
            a[(i, i)] += N as f64;
        }
        a
    }

    fn rhs_column() -> Mat<f64> {
        Mat::from_fn(N, 1, |i, _| (i as f64 * 0.37).sin())
    }

    fn unit_block() -> Mat<f64> {
        let mut unit = Mat::zeros(N, 15);
        for c in 0..15 {
            unit[(60 + c, c)] = 1.0;
        }
        unit
    }

    fn bits(m: &Mat<f64>) -> Vec<u64> {
        (0..m.ncols())
            .flat_map(|c| m.col_as_slice(c).iter().map(|v| v.to_bits()))
            .collect()
    }

    fn explicit(a: &Mat<f64>, par: Par) -> (Vec<u64>, Vec<u64>) {
        let mut dense = DenseLlt::new(N, par);
        dense.factor(a).unwrap();
        let column = dense
            .solve_column(rhs_column().col_as_slice(0))
            .unwrap()
            .to_vec();
        let mut unit = unit_block();
        dense.solve_in_place(&mut unit).unwrap();
        (column.iter().map(|v| v.to_bits()).collect(), bits(&unit))
    }

    #[test]
    fn the_explicit_solve_is_faers_solve_bit_for_bit() {
        let a = spd();
        let chol = a.llt(Side::Lower).unwrap();
        let faer_column = bits(&chol.solve(&rhs_column()));
        let faer_unit = bits(&chol.solve(&unit_block()));
        let (column, unit) = explicit(&a, faer::get_global_parallelism());
        assert_eq!(column, faer_column);
        assert_eq!(unit, faer_unit);
    }

    /// Four threads, not the pool size, so a one-thread pool still asks for a
    /// threaded factor. faer may keep a 150×150 factor on one thread below its
    /// parallel cutover; then this pins only that the threaded entry point does
    /// not change the bits.
    #[test]
    fn sequential_and_threaded_factor_to_the_same_bits() {
        let a = spd();
        assert_eq!(explicit(&a, Par::Seq), explicit(&a, Par::rayon(4)));
    }

    #[test]
    fn solves_without_a_valid_factor_are_refused() {
        let mut dense = DenseLlt::new(3, Par::Seq);
        assert_eq!(
            dense.solve_column(&[1.0, 1.0, 1.0]).map(|_| ()),
            Err(DenseLltError::NotFactored)
        );
        assert_eq!(
            dense.solve_in_place(&mut Mat::zeros(3, 1)),
            Err(DenseLltError::NotFactored)
        );
        dense.factor(&Mat::identity(3, 3)).unwrap();
        assert!(dense.solve_column(&[1.0, 1.0, 1.0]).is_ok());
        let indefinite = Mat::from_fn(3, 3, |i, j| if i == j { -1.0 } else { 0.0 });
        assert!(dense.factor(&indefinite).is_err());
        assert_eq!(
            dense.solve_column(&[1.0, 1.0, 1.0]).map(|_| ()),
            Err(DenseLltError::NotFactored)
        );
        assert_eq!(
            dense.factor(&Mat::identity(2, 2)),
            Err(DenseLltError::DimensionMismatch {
                expected: 3,
                got: 2
            })
        );
        assert_eq!(
            dense.solve_in_place(&mut Mat::zeros(3, 1)),
            Err(DenseLltError::NotFactored)
        );
    }

    #[test]
    fn a_matrix_that_is_not_positive_definite_is_refused() {
        let a = spd();
        let mut indefinite = a.clone();
        indefinite[(N / 2, N / 2)] = -1.0;
        let mut dense = DenseLlt::new(N, Par::Seq);
        assert!(matches!(
            dense.factor(&indefinite),
            Err(DenseLltError::NotPositiveDefinite { .. })
        ));
        dense.factor(&a).unwrap();
        let column: Vec<u64> = dense
            .solve_column(rhs_column().col_as_slice(0))
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        assert_eq!(column, explicit(&a, Par::Seq).0);
    }

    #[test]
    fn mismatched_dimensions_are_refused() {
        let mut dense = DenseLlt::new(3, Par::Seq);
        let mismatch = Err(DenseLltError::DimensionMismatch {
            expected: 3,
            got: 2,
        });
        assert_eq!(dense.factor(&Mat::identity(2, 2)), mismatch);
        assert_eq!(dense.factor(&Mat::identity(3, 2)), mismatch);
        dense.factor(&Mat::identity(3, 3)).unwrap();
        assert_eq!(dense.solve_column(&[1.0, 2.0]).map(|_| ()), mismatch);
        assert_eq!(dense.solve_in_place(&mut Mat::zeros(2, 1)), mismatch);
    }
}
