//! Factor trait for factor graph optimization
//!
//! Factors represent constraints or measurements in a factor graph. Each factor
//! computes a residual (error) and optionally a Jacobian with respect to the
//! connected variables.
//!
//! # References
//!
//! - [apex-solver](https://github.com/amin-abouee/apex-solver): A Rust-based library
//!   for efficient non-linear least squares optimization with factor graph support
//! - [ceres-solver](https://github.com/ceres-solver/ceres-solver): A C++ library for modeling and solving large, complicated optimization problems.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum FactorError {
    /// Invalid dimension mismatch
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// Jacobian computation failed
    #[error("Jacobian computation failed: {0}")]
    JacobianFailed(String),

    /// Invalid parameter values
    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),

    /// Numerical instability detected
    #[error("Numerical instability: {0}")]
    NumericalInstability(String),
}

/// Result type for factor operations
pub type FactorResult<T> = Result<T, FactorError>;

/// Output of factor linearization
#[derive(Debug, Clone)]
pub struct LinearizationResult {
    /// Residual vector (error)
    pub residual: Vec<f32>,
    /// Jacobian matrix (row-major, flattened)
    /// Shape: (residual_dim, total_local_dim)
    pub jacobian: Option<Vec<f32>>,
    /// Total local dimension (number of Jacobian columns)
    pub total_local_dim: usize,
}

impl LinearizationResult {
    /// Create a new linearization result
    pub fn new(residual: Vec<f32>, jacobian: Option<Vec<f32>>, total_local_dim: usize) -> Self {
        Self {
            residual,
            jacobian,
            total_local_dim,
        }
    }

    /// Get the residual dimension
    pub fn residual_dim(&self) -> usize {
        self.residual.len()
    }

    /// Get a specific Jacobian element (row-major order)
    pub fn jacobian_element(&self, row: usize, col: usize) -> Option<f32> {
        self.jacobian
            .as_ref()
            .map(|j| j[row * self.total_local_dim + col])
    }
}

/// Trait for factor (constraint) implementations in factor graph optimization.
///
/// A factor represents a measurement or constraint connecting one or more variables.
/// It computes the residual (error) and Jacobian for the current variable values,
/// which are used by the optimizer to minimize the total cost.
///
/// # Implementing Custom Factors
///
/// To create a custom factor:
/// 1. Implement this trait
/// 2. Define the residual function `r(x)` (how to compute error from variable values)
/// 3. Compute the Jacobian `J = ∂r/∂x` (analytically or numerically)
/// 4. Return the residual dimension
///
/// # Thread Safety
///
/// Factors must be `Send + Sync` to enable parallel residual/Jacobian evaluation.
pub trait Factor: Send + Sync {
    /// Compute the residual and optionally the Jacobian at the given parameter values.
    ///
    /// # Arguments
    ///
    /// * `params` - Slice of variable values (one slice per connected variable, global storage)
    /// * `compute_jacobian` - Whether to compute the Jacobian matrix
    ///
    /// # Returns
    ///
    /// `LinearizationResult` containing:
    /// - `residual`: N-dimensional error vector
    /// - `jacobian`: N × M flattened matrix where M is the total local DOF of all variables
    fn linearize(
        &self,
        params: &[&[f32]],
        compute_jacobian: bool,
    ) -> FactorResult<LinearizationResult>;

    /// Get the dimension of the residual vector.
    ///
    /// # Returns
    ///
    /// Number of elements in the residual vector (number of constraints)
    fn residual_dim(&self) -> usize;

    /// Get the number of variables this factor connects.
    fn num_variables(&self) -> usize;

    /// Get the local (tangent) dimension of a specific connected variable.
    ///
    /// # Arguments
    ///
    /// * `idx` - Index of the variable (0 to num_variables()-1)
    fn variable_local_dim(&self, idx: usize) -> usize;

    /// Get the total local (tangent) dimension of all connected variables.
    fn total_local_dim(&self) -> usize {
        (0..self.num_variables())
            .map(|i| self.variable_local_dim(i))
            .sum()
    }
}

/// A simple prior factor that penalizes deviation from a target value.
///
/// Residual: r = x - target
#[derive(Debug, Clone)]
pub struct PriorFactor {
    /// Target value
    pub target: Vec<f32>,
}

impl PriorFactor {
    /// Create a new prior factor
    pub fn new(target: Vec<f32>) -> Self {
        Self { target }
    }
}

impl Factor for PriorFactor {
    fn linearize(
        &self,
        params: &[&[f32]],
        compute_jacobian: bool,
    ) -> FactorResult<LinearizationResult> {
        if params.len() != 1 {
            return Err(FactorError::DimensionMismatch {
                expected: 1,
                actual: params.len(),
            });
        }

        let x = params[0];
        if x.len() != self.target.len() {
            return Err(FactorError::DimensionMismatch {
                expected: self.target.len(),
                actual: x.len(),
            });
        }

        // Residual: x - target
        let residual: Vec<f32> = x.iter().zip(&self.target).map(|(xi, ti)| xi - ti).collect();

        // Jacobian is identity matrix
        let jacobian = if compute_jacobian {
            let n = x.len();
            let mut jac = vec![0.0f32; n * n];
            for i in 0..n {
                jac[i * n + i] = 1.0;
            }
            Some(jac)
        } else {
            None
        };

        Ok(LinearizationResult::new(residual, jacobian, x.len()))
    }

    fn residual_dim(&self) -> usize {
        self.target.len()
    }

    fn num_variables(&self) -> usize {
        1
    }

    fn variable_local_dim(&self, _idx: usize) -> usize {
        self.target.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prior_factor() {
        let factor = PriorFactor::new(vec![1.0, 2.0, 3.0]);
        let params = vec![1.5f32, 2.5, 3.5];
        let params_slice: &[f32] = &params;

        let result = factor.linearize(&[params_slice], true).unwrap();

        // Check residual
        assert_eq!(result.residual.len(), 3);
        assert!((result.residual[0] - 0.5).abs() < 1e-6);
        assert!((result.residual[1] - 0.5).abs() < 1e-6);
        assert!((result.residual[2] - 0.5).abs() < 1e-6);

        // Check Jacobian is identity
        let jac = result.jacobian.unwrap();
        assert_eq!(jac.len(), 9);
        assert!((jac[0] - 1.0).abs() < 1e-6); // (0,0)
        assert!((jac[4] - 1.0).abs() < 1e-6); // (1,1)
        assert!((jac[8] - 1.0).abs() < 1e-6); // (2,2)
    }

    #[test]
    fn test_factor_dimensions() {
        let factor = PriorFactor::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(factor.residual_dim(), 3);
        assert_eq!(factor.num_variables(), 1);
        assert_eq!(factor.variable_local_dim(0), 3);
        assert_eq!(factor.total_local_dim(), 3);
    }
}
