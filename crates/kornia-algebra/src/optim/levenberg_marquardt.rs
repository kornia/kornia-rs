//! Levenberg-Marquardt optimizer for non-linear least squares optimization
//!
//! The Levenberg-Marquardt algorithm is a trust-region method that combines
//! the advantages of gradient descent and Gauss-Newton methods. It solves
//! the damped normal equations: (J^T J + λI) δ = -J^T r

use std::collections::HashMap;

use crate::{DMatF32, DVecF32};
use thiserror::Error;

use super::factor::FactorError;
use super::problem::{Problem, ProblemError};

/// Errors that can occur during optimization.
#[derive(Debug, Error)]
pub enum OptimizerError {
    /// Problem-related error
    #[error("Problem error: {0}")]
    Problem(#[from] ProblemError),

    /// Factor evaluation failed
    #[error("Factor evaluation failed: {0}")]
    Factor(#[from] FactorError),

    /// Linear system solve failed (singular matrix)
    #[error("Linear system solve failed: {0}")]
    SolveFailed(String),

    /// Maximum iterations reached
    #[error("Maximum iterations ({0}) reached")]
    MaxIterations(usize),

    /// Numerical instability detected
    #[error("Numerical instability: {0}")]
    NumericalInstability(String),
}

/// Result of an optimization run.
#[derive(Debug, Clone)]
pub struct OptimizerResult {
    /// Final cost (sum of squared residuals)
    pub final_cost: f32,
    /// Number of iterations performed
    pub iterations: usize,
    /// Reason for termination
    pub termination_reason: TerminationReason,
}

/// Reason why the optimizer terminated.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TerminationReason {
    /// Converged: cost change below tolerance
    CostConverged,
    /// Converged: gradient norm below tolerance
    GradientConverged,
    /// Maximum iterations reached
    MaxIterations,
    /// Lambda exceeded maximum (likely numerical issues)
    LambdaMaxExceeded,
}

/// Levenberg-Marquardt optimizer configuration.
#[derive(Debug, Clone)]
pub struct LevenbergMarquardt {
    /// Initial damping parameter
    pub lambda_init: f32,
    /// Maximum damping parameter
    pub lambda_max: f32,
    /// Factor for lambda adaptation
    pub lambda_factor: f32,
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence threshold for cost change
    pub cost_tolerance: f32,
    /// Convergence threshold for gradient norm
    pub gradient_tolerance: f32,
}

impl Default for LevenbergMarquardt {
    fn default() -> Self {
        Self {
            lambda_init: 1e-3,
            lambda_max: 1e10,
            lambda_factor: 10.0,
            max_iterations: 50,
            cost_tolerance: 1e-6,
            gradient_tolerance: 1e-6,
        }
    }
}

impl LevenbergMarquardt {
    /// Minimum step norm threshold. Steps smaller than this are considered zero.
    const STEP_SIZE_TOLERANCE: f32 = 1e-12;

    /// Create a new optimizer with default parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Optimize the given problem.
    ///
    /// # Arguments
    ///
    /// * `problem` - The optimization problem to solve
    ///
    /// # Returns
    ///
    /// Optimization result containing final cost, iterations, and termination reason.
    ///
    /// # Errors
    ///
    /// Returns an error if optimization fails due to numerical issues, singular matrices,
    /// or factor evaluation failures.
    pub fn optimize(&self, problem: &mut Problem) -> Result<OptimizerResult, OptimizerError> {
        let variables = problem.get_variables();
        let factors = problem.get_factors();

        if variables.is_empty() {
            return Err(OptimizerError::NumericalInstability(
                "No variables in problem".to_string(),
            ));
        }

        if factors.is_empty() {
            return Err(OptimizerError::NumericalInstability(
                "No factors in problem".to_string(),
            ));
        }

        // Build variable index mapping
        let mut var_names: Vec<String> = variables.keys().cloned().collect();
        var_names.sort(); // Ensure consistent ordering
        let var_index_map: HashMap<String, usize> = var_names
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();

        // Compute total parameter dimension
        let total_dim: usize = var_names.iter().map(|name| variables[name].dim()).sum();

        if total_dim == 0 {
            return Err(OptimizerError::NumericalInstability(
                "Total parameter dimension is zero".to_string(),
            ));
        }

        // Compute initial cost
        let mut current_cost = problem.compute_total_cost()?;

        let mut lambda = self.lambda_init;
        let mut iterations = 0;

        loop {
            if iterations >= self.max_iterations {
                return Ok(OptimizerResult {
                    final_cost: current_cost,
                    iterations,
                    termination_reason: TerminationReason::MaxIterations,
                });
            }

            // Build normal equations: J^T J and J^T r
            let (jtj, jtr) =
                self.build_normal_equations(problem, &var_names, &var_index_map, total_dim)?;

            // Check gradient convergence
            let gradient_norm = jtr.norm();
            if gradient_norm < self.gradient_tolerance {
                return Ok(OptimizerResult {
                    final_cost: current_cost,
                    iterations,
                    termination_reason: TerminationReason::GradientConverged,
                });
            }

            // Solve damped system: (J^T J + λI) δ = -J^T r
            let delta = self.solve_damped_system(&jtj, &jtr, lambda, total_dim)?;

            // Compute step size
            let step_norm = delta.norm();
            if step_norm < Self::STEP_SIZE_TOLERANCE {
                // Step is essentially zero, consider converged
                return Ok(OptimizerResult {
                    final_cost: current_cost,
                    iterations,
                    termination_reason: TerminationReason::CostConverged,
                });
            }

            // Apply step and compute new cost
            let new_cost = self.apply_step(problem, &var_names, &delta, &var_index_map)?;

            // Compute cost change
            let cost_change = current_cost - new_cost;
            let relative_cost_change = if current_cost > 0.0 {
                cost_change.abs() / current_cost
            } else {
                cost_change.abs()
            };

            // Check cost convergence
            if relative_cost_change < self.cost_tolerance {
                return Ok(OptimizerResult {
                    final_cost: new_cost,
                    iterations: iterations + 1,
                    termination_reason: TerminationReason::CostConverged,
                });
            }

            // Adapt lambda based on cost improvement
            if cost_change > 0.0 {
                // Step improved cost: accept it and decrease lambda
                current_cost = new_cost;
                lambda = (lambda / self.lambda_factor).max(1e-10);
                iterations += 1;
            } else {
                // Step increased cost: reject it and increase lambda
                // Revert the step
                self.revert_step(problem, &var_names, &delta, &var_index_map)?;
                lambda *= self.lambda_factor;
                iterations += 1;

                if lambda > self.lambda_max {
                    return Ok(OptimizerResult {
                        final_cost: current_cost,
                        iterations,
                        termination_reason: TerminationReason::LambdaMaxExceeded,
                    });
                }
            }
        }
    }

    /// Build the normal equations J^T J and J^T r from all factors.
    fn build_normal_equations(
        &self,
        problem: &Problem,
        var_names: &[String],
        var_index_map: &HashMap<String, usize>,
        total_dim: usize,
    ) -> Result<(DMatF32, DVecF32), OptimizerError> {
        let variables = problem.get_variables();
        let factors = problem.get_factors();

        // Initialize J^T J (Hessian approximation) and J^T r (negative gradient)
        let mut jtj = DMatF32::zeros(total_dim, total_dim);
        let mut jtr = DVecF32::zeros(total_dim);

        // Accumulate contributions from each factor
        for (factor, factor_var_names) in factors {
            // Get parameter slices for this factor's variables
            let mut params = Vec::new();
            for name in factor_var_names {
                let var = variables
                    .get(name)
                    .ok_or_else(|| ProblemError::VariableNotFound { name: name.clone() })?;
                params.push(var.values.as_slice());
            }

            // Linearize the factor
            let result = factor.linearize(&params, true)?;

            // Build the full Jacobian block for this factor
            // The factor's Jacobian is w.r.t. its connected variables only
            // We need to map it to the full parameter space
            if let Some(jacobian) = &result.jacobian {
                let residual_dim = result.residual_dim();
                let jacobian_cols = result.jacobian_cols;

                // Compute J^T J and J^T r for this factor
                // Factor Jacobian shape: (residual_dim, jacobian_cols)
                // where jacobian_cols is the total DOF of connected variables

                // Map factor variable indices to global parameter indices
                let mut global_col_starts = Vec::new();
                let mut global_col_dims = Vec::new();

                for var_name in factor_var_names {
                    let var_idx = var_index_map[var_name];
                    let mut global_col_start = 0;
                    for i in 0..var_idx {
                        global_col_start += variables[&var_names[i]].dim();
                    }
                    let var_dim = variables[var_name].dim();
                    global_col_starts.push(global_col_start);
                    global_col_dims.push(var_dim);
                }

                // Factor Jacobian columns are organized sequentially for connected variables
                // Column layout: [var0_dim0, var0_dim1, ..., var0_dimN, var1_dim0, ...]
                let mut factor_col_offset = 0;

                // Accumulate J^T J block by block
                for (local_i, _var_name_i) in factor_var_names.iter().enumerate() {
                    let global_start_i = global_col_starts[local_i];
                    let dim_i = global_col_dims[local_i];

                    for (local_j, _var_name_j) in factor_var_names.iter().enumerate() {
                        let global_start_j = global_col_starts[local_j];
                        let dim_j = global_col_dims[local_j];

                        // Compute block: J_i^T J_j
                        // Factor Jacobian columns for var_i start at factor_col_offset
                        // Factor Jacobian columns for var_j start at their offset
                        let factor_col_offset_j: usize = global_col_dims.iter().take(local_j).sum();

                        for row in 0..residual_dim {
                            for di in 0..dim_i {
                                let jac_i_val =
                                    jacobian[row * jacobian_cols + factor_col_offset + di];
                                for dj in 0..dim_j {
                                    let jac_j_val =
                                        jacobian[row * jacobian_cols + factor_col_offset_j + dj];
                                    jtj[(global_start_i + di, global_start_j + dj)] +=
                                        jac_i_val * jac_j_val;
                                }
                            }
                        }
                    }
                    factor_col_offset += dim_i;
                }

                // Accumulate J^T r
                factor_col_offset = 0;
                for (local_i, _var_name_i) in factor_var_names.iter().enumerate() {
                    let global_start_i = global_col_starts[local_i];
                    let dim_i = global_col_dims[local_i];

                    for row in 0..residual_dim {
                        let residual_val = result.residual[row];
                        for di in 0..dim_i {
                            let jac_val = jacobian[row * jacobian_cols + factor_col_offset + di];
                            jtr[global_start_i + di] += jac_val * residual_val;
                        }
                    }
                    factor_col_offset += dim_i;
                }
            } else {
                return Err(OptimizerError::Factor(FactorError::JacobianFailed(
                    "Jacobian required for optimization".to_string(),
                )));
            }
        }

        Ok((jtj, jtr))
    }

    /// Solve the damped system (J^T J + λI) δ = -J^T r.
    fn solve_damped_system(
        &self,
        jtj: &DMatF32,
        jtr: &DVecF32,
        lambda: f32,
        dim: usize,
    ) -> Result<DVecF32, OptimizerError> {
        // Build damped Hessian: H = J^T J + λI
        let mut h = jtj.clone();
        for i in 0..dim {
            h[(i, i)] += lambda;
        }

        // Solve H δ = -J^T r
        let rhs = -jtr;
        let lu = h.lu();
        let delta = lu
            .solve(&rhs)
            .ok_or_else(|| OptimizerError::SolveFailed("LU solve failed".to_string()))?;

        Ok(delta)
    }

    /// Apply a step to the problem variables.
    fn apply_step(
        &self,
        problem: &mut Problem,
        var_names: &[String],
        delta: &DVecF32,
        _var_index_map: &HashMap<String, usize>,
    ) -> Result<f32, OptimizerError> {
        let variables = problem.get_variables_mut();
        let mut param_offset = 0;

        for var_name in var_names {
            let var =
                variables
                    .get_mut(var_name)
                    .ok_or_else(|| ProblemError::VariableNotFound {
                        name: var_name.clone(),
                    })?;

            let dim = var.dim();
            for i in 0..dim {
                var.values[i] += delta[param_offset + i];
            }
            param_offset += dim;
        }

        problem.compute_total_cost().map_err(OptimizerError::from)
    }

    /// Revert a step (undo the last update).
    fn revert_step(
        &self,
        problem: &mut Problem,
        var_names: &[String],
        delta: &DVecF32,
        _var_index_map: &HashMap<String, usize>,
    ) -> Result<(), OptimizerError> {
        let variables = problem.get_variables_mut();
        let mut param_offset = 0;

        for var_name in var_names {
            let var =
                variables
                    .get_mut(var_name)
                    .ok_or_else(|| ProblemError::VariableNotFound {
                        name: var_name.clone(),
                    })?;

            let dim = var.dim();
            for i in 0..dim {
                var.values[i] -= delta[param_offset + i];
            }
            param_offset += dim;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::{PriorFactor, Problem, Variable};
    use super::*;

    #[test]
    fn test_simple_1d_optimization() {
        // Minimize (x - 5)^2
        let mut problem = Problem::new();
        let var = Variable::euclidean("x", 1);
        problem.add_variable(var, vec![0.0]).unwrap();

        let factor = PriorFactor::new(vec![5.0]);
        problem
            .add_factor(Box::new(factor), vec!["x".to_string()])
            .unwrap();

        let optimizer = LevenbergMarquardt::new();
        let result = optimizer.optimize(&mut problem).unwrap();

        assert!(result.iterations > 0);
        assert!(result.final_cost < 1e-6);
        assert!(
            result.termination_reason == TerminationReason::CostConverged
                || result.termination_reason == TerminationReason::GradientConverged
        );

        let x = problem.get_variables().get("x").unwrap();
        assert!((x.values[0] - 5.0).abs() < 1e-3);
    }

    #[test]
    fn test_multi_variable_optimization() {
        // Minimize (x - 1)^2 + (y - 2)^2 + (z - 3)^2
        let mut problem = Problem::new();
        problem
            .add_variable(Variable::euclidean("x", 1), vec![0.0])
            .unwrap();
        problem
            .add_variable(Variable::euclidean("y", 1), vec![0.0])
            .unwrap();
        problem
            .add_variable(Variable::euclidean("z", 1), vec![0.0])
            .unwrap();

        problem
            .add_factor(Box::new(PriorFactor::new(vec![1.0])), vec!["x".to_string()])
            .unwrap();
        problem
            .add_factor(Box::new(PriorFactor::new(vec![2.0])), vec!["y".to_string()])
            .unwrap();
        problem
            .add_factor(Box::new(PriorFactor::new(vec![3.0])), vec!["z".to_string()])
            .unwrap();

        let optimizer = LevenbergMarquardt::new();
        let result = optimizer.optimize(&mut problem).unwrap();

        assert!(result.final_cost < 1e-6);

        let vars = problem.get_variables();
        assert!((vars["x"].values[0] - 1.0).abs() < 1e-3);
        assert!((vars["y"].values[0] - 2.0).abs() < 1e-3);
        assert!((vars["z"].values[0] - 3.0).abs() < 1e-3);
    }

    #[test]
    fn test_optimizer_default() {
        let optimizer = LevenbergMarquardt::default();
        assert_eq!(optimizer.lambda_init, 1e-3);
        assert_eq!(optimizer.lambda_max, 1e10);
        assert_eq!(optimizer.lambda_factor, 10.0);
        assert_eq!(optimizer.max_iterations, 50);
    }
}
