//! Levenberg-Marquardt optimizer for non-linear least squares optimization
//!
//! The Levenberg-Marquardt algorithm is a trust-region method that combines
//! the advantages of gradient descent and Gauss-Newton methods. It solves
//! the damped normal equations: (J^T J + λI) δ = -J^T r

use crate::param::ParamError;
use crate::{DMatF32, DVecF32};
use thiserror::Error;

use super::linear_system::{LinearSystemBuilder, VariableLayout};
use crate::optim::core::{FactorError, Problem, ProblemError};

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

    /// Parameter update failed
    #[error("Parameter update failed: {0}")]
    Param(#[from] ParamError),
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
    /// Optimization stopped early by user callback
    Interrupted,
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

/// Snapshot of the optimizer state for callbacks.
#[derive(Debug, Clone, Copy)]
pub struct OptimizerState {
    pub iteration: usize,
    pub cost: f32,
    pub lambda: f32,
    pub last_step_accepted: Option<bool>,
}

impl LevenbergMarquardt {
    /// Minimum step norm threshold. Steps smaller than this are considered zero.
    const STEP_SIZE_TOLERANCE: f32 = 1e-12;

    pub fn optimize(&self, problem: &mut Problem) -> Result<OptimizerResult, OptimizerError> {
        self.optimize_with_callback(problem, |_problem, _state| true)
    }

    pub fn optimize_with_callback<F>(
        &self,
        problem: &mut Problem,
        mut callback: F,
    ) -> Result<OptimizerResult, OptimizerError>
    where
        F: FnMut(&Problem, &OptimizerState) -> bool,
    {
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

        let layout = VariableLayout::from_problem(problem)?;
        let var_names = &layout.var_names;

        if layout.total_local_dim == 0 {
            return Err(OptimizerError::NumericalInstability(
                "Total parameter dimension is zero".to_string(),
            ));
        }

        // Compute initial cost
        let mut current_cost = problem.compute_total_cost()?;

        let mut lambda = self.lambda_init;
        let mut iterations = 0;

        let init_state = OptimizerState {
            iteration: 0,
            cost: current_cost,
            lambda,
            last_step_accepted: None,
        };
        if !callback(problem, &init_state) {
            return Ok(OptimizerResult {
                final_cost: current_cost,
                iterations,
                termination_reason: TerminationReason::Interrupted,
            });
        }

        loop {
            if iterations >= self.max_iterations {
                return Ok(OptimizerResult {
                    final_cost: current_cost,
                    iterations,
                    termination_reason: TerminationReason::MaxIterations,
                });
            }

            // Build normal equations: J^T J and J^T r
            let (jtj, jtr) = LinearSystemBuilder::build(problem, &layout)?;

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
            let delta = self.solve_damped_system(jtj, &jtr, lambda, layout.total_local_dim)?;

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

            // Apply step and compute new cost (also returns snapshot for reverting)
            let (new_cost, snapshot) = self.apply_step(problem, var_names, &delta)?;

            let cost_change = current_cost - new_cost;
            let relative_cost_change = if current_cost > 0.0 {
                cost_change.abs() / current_cost
            } else {
                cost_change.abs()
            };

            if relative_cost_change < self.cost_tolerance {
                return Ok(OptimizerResult {
                    final_cost: new_cost,
                    iterations: iterations + 1,
                    termination_reason: TerminationReason::CostConverged,
                });
            }

            let mut last_step_accepted = false;
            if cost_change > 0.0 {
                // Step improved cost: accept it and decrease lambda
                current_cost = new_cost;
                lambda = (lambda / self.lambda_factor).max(1e-10);
                last_step_accepted = true;
                iterations += 1;
            } else {
                // Step increased cost: reject it and increase lambda
                self.revert_step(problem, var_names, snapshot)?;
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

            let state = OptimizerState {
                iteration: iterations,
                cost: current_cost,
                lambda,
                last_step_accepted: Some(last_step_accepted),
            };
            if !callback(problem, &state) {
                return Ok(OptimizerResult {
                    final_cost: current_cost,
                    iterations,
                    termination_reason: TerminationReason::Interrupted,
                });
            }
        }
    }

    /// Solve the damped system (J^T J + λI) δ = -J^T r.
    fn solve_damped_system(
        &self,
        mut jtj: DMatF32,
        jtr: &DVecF32,
        lambda: f32,
        dim: usize,
    ) -> Result<DVecF32, OptimizerError> {
        // Build damped Hessian in-place: H = J^T J + λI
        // Add lambda to diagonal elements
        for i in 0..dim {
            jtj[(i, i)] += lambda;
        }

        // Solve H δ = -J^T r
        let rhs = -jtr;
        let lu = jtj.lu();
        let delta = lu
            .solve(&rhs)
            .ok_or_else(|| OptimizerError::SolveFailed("LU solve failed".to_string()))?;

        Ok(delta)
    }

    fn apply_step(
        &self,
        problem: &mut Problem,
        var_names: &[String],
        delta: &DVecF32,
    ) -> Result<(f32, Vec<Vec<f32>>), OptimizerError> {
        let variables = problem.get_variables_mut();
        let mut param_offset = 0;

        // Snapshot current values so we can revert safely on manifolds.
        let mut snapshot: Vec<Vec<f32>> = Vec::with_capacity(var_names.len());

        for var_name in var_names {
            let var =
                variables
                    .get_mut(var_name)
                    .ok_or_else(|| ProblemError::VariableNotFound {
                        name: var_name.clone(),
                    })?;

            snapshot.push(var.values.clone());

            let local = var.local_dim();
            let delta_block = &delta.as_slice()[param_offset..param_offset + local];

            var.var_type.apply_plus(&mut var.values, delta_block)?;

            param_offset += local;
        }

        let cost = problem.compute_total_cost().map_err(OptimizerError::from)?;
        Ok((cost, snapshot))
    }

    /// Revert the step to the previous values.
    fn revert_step(
        &self,
        problem: &mut Problem,
        var_names: &[String],
        snapshot: Vec<Vec<f32>>,
    ) -> Result<(), OptimizerError> {
        let variables = problem.get_variables_mut();

        debug_assert_eq!(var_names.len(), snapshot.len());

        for (var_name, old_vals) in var_names.iter().zip(snapshot.into_iter()) {
            let var =
                variables
                    .get_mut(var_name)
                    .ok_or_else(|| ProblemError::VariableNotFound {
                        name: var_name.clone(),
                    })?;
            var.values = old_vals;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optim::{PriorFactor, Problem, Variable};

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

        let optimizer = LevenbergMarquardt::default();
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

        let optimizer = LevenbergMarquardt::default();
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
