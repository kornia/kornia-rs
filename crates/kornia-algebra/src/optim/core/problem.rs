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
        if initial_values.len() != var.global_dim() {
            return Err(ProblemError::DimensionMismatch {
                expected: var.global_dim(),
                actual: initial_values.len(),
            });
        }
        var.values = initial_values;
        self.variables.insert(var.name.clone(), var);
        Ok(())
    }

    pub fn add_factor(
        &mut self,
        factor: Box<dyn Factor>,
        var_names: Vec<String>,
    ) -> Result<(), ProblemError> {
        for name in &var_names {
            if !self.variables.contains_key(name) {
                return Err(ProblemError::VariableNotFound { name: name.clone() });
            }
        }
        self.factors.push((factor, var_names));
        Ok(())
    }

    pub fn get_variables(&self) -> &HashMap<String, Variable> {
        &self.variables
    }

    pub fn get_variables_mut(&mut self) -> &mut HashMap<String, Variable> {
        &mut self.variables
    }

    pub fn get_factors(&self) -> &[(Box<dyn Factor>, Vec<String>)] {
        &self.factors
    }

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optim::PriorFactor;

    #[test]
    fn test_add_and_get_variable() {
        let mut problem = Problem::new();
        let var = Variable::euclidean("R", 2);
        assert!(problem.add_variable(var.clone(), vec![1.0, 2.0]).is_ok());

        let vars = problem.get_variables();
        assert!(vars.contains_key("R"));
        assert_eq!(vars["R"].values, vec![1.0, 2.0]);
        assert_eq!(vars["R"].name, "R");
    }

    #[test]
    fn test_add_variable_duplicate_should_fail() {
        let mut problem = Problem::new();
        let var = Variable::euclidean("R", 2);
        assert!(problem.add_variable(var.clone(), vec![1.0, 2.0]).is_ok());
        // Add same variable again with correct dim should fail due to duplicate name
        assert!(problem.add_variable(var.clone(), vec![1.0, 2.0]).is_err());
    }

    #[test]
    fn test_add_variable_wrong_dim_should_fail() {
        let mut problem = Problem::new();
        // Vec is length 3 but variable is dim 2, should fail
        let var = Variable::euclidean("R", 2);
        assert!(problem.add_variable(var, vec![1.0, 2.0, 3.0]).is_err());
    }

    #[test]
    fn test_add_and_get_factor() {
        let mut problem = Problem::new();
        problem
            .add_variable(Variable::euclidean("x", 2), vec![1.0, 2.0])
            .unwrap();
        let factor = PriorFactor::new(vec![1.5, 2.5]);
        // Add factor referencing the variable
        assert!(problem
            .add_factor(Box::new(factor), vec!["x".to_string()])
            .is_ok());
        let factors = problem.get_factors();
        assert_eq!(factors.len(), 1);
        assert_eq!(factors[0].1, vec!["x".to_string()]);
    }

    #[test]
    fn test_compute_total_cost() {
        let mut problem = Problem::new();
        problem
            .add_variable(Variable::euclidean("foo", 2), vec![3.0, -2.0])
            .unwrap();
        let prior = PriorFactor::new(vec![4.0, -5.0]);
        problem
            .add_factor(Box::new(prior), vec!["foo".to_string()])
            .unwrap();

        // Cost is (3-4)^2 + (-2+5)^2 = 1^2 + 3^2 = 10
        let cost = problem.compute_total_cost().unwrap();
        assert!((cost - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_compute_total_cost_with_multiple_factors() {
        let mut problem = Problem::new();
        problem
            .add_variable(Variable::euclidean("x", 1), vec![2.0])
            .unwrap();
        problem
            .add_variable(Variable::euclidean("y", 1), vec![-1.0])
            .unwrap();
        let fx = PriorFactor::new(vec![3.0]); // (2-3)^2 = 1
        let fy = PriorFactor::new(vec![0.0]); // (-1-0)^2 = 1
        problem
            .add_factor(Box::new(fx), vec!["x".to_string()])
            .unwrap();
        problem
            .add_factor(Box::new(fy), vec!["y".to_string()])
            .unwrap();

        let cost = problem.compute_total_cost().unwrap();
        assert!((cost - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_compute_total_cost_error_on_missing_variable() {
        let mut problem = Problem::new();
        let prior = PriorFactor::new(vec![1.0, 1.0]);
        // reference to a variable not present
        let res = problem.add_factor(Box::new(prior), vec!["not_present".to_string()]);
        assert!(res.is_err());
        if let Err(ProblemError::VariableNotFound { name }) = res {
            assert_eq!(name, "not_present");
        } else {
            panic!("Expected VariableNotFound error");
        }
    }
}
