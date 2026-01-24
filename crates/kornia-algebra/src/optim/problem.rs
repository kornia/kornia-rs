//! Problem struct for factor graph optimization
//!
//! A Problem contains variables (parameters to optimize) and factors (constraints/measurements).

use std::collections::HashMap;

use thiserror::Error;

use super::factor::{Factor, FactorError};
use super::variable::Variable;

/// Errors that can occur when working with optimization problems.
#[derive(Debug, Error)]
pub enum ProblemError {
    /// Variable with this name already exists
    #[error("Variable '{name}' already exists")]
    DuplicateVariable { name: String },
    /// Variable with this name was not found
    #[error("Variable '{name}' not found")]
    VariableNotFound { name: String },
    /// Dimension mismatch between expected and actual values
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    /// Factor evaluation failed
    #[error("Factor evaluation failed: {0}")]
    FactorEvaluation(#[from] FactorError),
}

/// An optimization problem containing variables and factors.
#[derive(Default)]
pub struct Problem {
    /// Variables in the problem, indexed by name
    variables: HashMap<String, Variable>,
    /// Factors in the problem, each with the names of variables it connects
    factors: Vec<(Box<dyn Factor>, Vec<String>)>,
}

impl Problem {
    /// Create a new empty problem.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a variable to the problem.
    ///
    /// # Arguments
    ///
    /// * `var` - The variable to add
    /// * `initial_values` - Initial parameter values for the variable
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - A variable with the same name already exists
    /// - The initial values dimension doesn't match the variable's dimension
    pub fn add_variable(
        &mut self,
        mut var: Variable,
        initial_values: Vec<f32>,
    ) -> Result<(), ProblemError> {
        if self.variables.contains_key(&var.name) {
            return Err(ProblemError::DuplicateVariable {
                name: var.name.clone(),
            });
        }
        if initial_values.len() != var.dim() {
            return Err(ProblemError::DimensionMismatch {
                expected: var.dim(),
                actual: initial_values.len(),
            });
        }
        var.values = initial_values;
        self.variables.insert(var.name.clone(), var);
        Ok(())
    }

    /// Add a factor to the problem.
    ///
    /// # Arguments
    ///
    /// * `factor` - The factor to add
    /// * `var_names` - Names of variables this factor connects to
    ///
    /// # Errors
    ///
    /// Returns an error if any variable name doesn't exist.
    pub fn add_factor(
        &mut self,
        factor: Box<dyn Factor>,
        var_names: Vec<String>,
    ) -> Result<(), ProblemError> {
        // Validate all variable names exist
        for name in &var_names {
            if !self.variables.contains_key(name) {
                return Err(ProblemError::VariableNotFound { name: name.clone() });
            }
        }
        self.factors.push((factor, var_names));
        Ok(())
    }

    /// Get all variables in the problem.
    pub fn get_variables(&self) -> &HashMap<String, Variable> {
        &self.variables
    }

    /// Get mutable access to all variables in the problem.
    pub fn get_variables_mut(&mut self) -> &mut HashMap<String, Variable> {
        &mut self.variables
    }

    /// Get all factors in the problem.
    pub fn get_factors(&self) -> &[(Box<dyn Factor>, Vec<String>)] {
        &self.factors
    }

    /// Compute the total squared cost (sum of squared residuals) for the current variable values.
    ///
    /// # Errors
    ///
    /// Returns an error if any factor evaluation fails.
    pub fn compute_total_cost(&self) -> Result<f32, ProblemError> {
        let mut total_cost = 0.0;

        for (factor, var_names) in &self.factors {
            // Get parameter slices for this factor's variables
            let mut params = Vec::new();
            for name in var_names {
                let var = self
                    .variables
                    .get(name)
                    .ok_or_else(|| ProblemError::VariableNotFound { name: name.clone() })?;
                params.push(var.values.as_slice());
            }

            // Linearize the factor (only need residual, not Jacobian)
            let result = factor.linearize(&params, false)?;

            // Accumulate squared residuals
            for r in &result.residual {
                total_cost += r * r;
            }
        }

        Ok(total_cost)
    }
}
