use std::collections::HashMap;

use crate::{DMatF32, DVecF32};

use super::OptimizerError;
use crate::optim::core::FactorError;
use crate::optim::core::{Problem, ProblemError};

/// Precomputed variable ordering and dimension layout for optimization.
#[derive(Debug, Clone)]
pub struct VariableLayout {
    pub var_names: Vec<String>,
    pub var_index_map: HashMap<String, usize>,
    pub global_starts: Vec<usize>,
    pub local_dims: Vec<usize>,
    pub total_local_dim: usize,
}

impl VariableLayout {
    pub fn from_problem(problem: &Problem) -> Result<Self, ProblemError> {
        let variables = problem.get_variables();
        let mut var_names: Vec<String> = variables.keys().cloned().collect();
        var_names.sort();

        let var_index_map: HashMap<String, usize> = var_names
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();

        let mut total_local_dim: usize = 0;
        let mut local_dims: Vec<usize> = Vec::with_capacity(var_names.len());
        let mut global_starts: Vec<usize> = Vec::with_capacity(var_names.len());

        for name in &var_names {
            global_starts.push(total_local_dim);
            let dim = variables[name].local_dim();
            local_dims.push(dim);
            total_local_dim += dim;
        }

        Ok(Self {
            var_names,
            var_index_map,
            global_starts,
            local_dims,
            total_local_dim,
        })
    }
}

/// Builds normal equations from factors for a given layout.
pub struct LinearSystemBuilder;

impl LinearSystemBuilder {
    pub fn build(
        problem: &Problem,
        layout: &VariableLayout,
    ) -> Result<(DMatF32, DVecF32), OptimizerError> {
        let variables = problem.get_variables();
        let factors = problem.get_factors();

        let mut jtj = DMatF32::zeros(layout.total_local_dim, layout.total_local_dim);
        let mut jtr = DVecF32::zeros(layout.total_local_dim);

        for (factor, factor_var_names) in factors {
            let mut params = Vec::new();
            for name in factor_var_names {
                let var = variables
                    .get(name)
                    .ok_or_else(|| ProblemError::VariableNotFound { name: name.clone() })?;
                params.push(var.values.as_slice());
            }

            let result = factor.linearize(&params, true)?;

            if let Some(jacobian) = &result.jacobian {
                let residual_dim = result.residual_dim();
                let total_local_dim = result.total_local_dim;
                let expected_cols: usize = factor_var_names
                    .iter()
                    .map(|name| layout.local_dims[layout.var_index_map[name]])
                    .sum();
                if total_local_dim != expected_cols {
                    return Err(OptimizerError::Factor(FactorError::DimensionMismatch {
                        expected: expected_cols,
                        actual: total_local_dim,
                    }));
                }

                let mut mapping: Vec<(usize, usize, usize)> =
                    Vec::with_capacity(factor_var_names.len());
                let mut factor_col_offset = 0;
                for var_name in factor_var_names {
                    let var_idx = layout.var_index_map[var_name];
                    let global_start = layout.global_starts[var_idx];
                    let dim = layout.local_dims[var_idx];
                    mapping.push((global_start, dim, factor_col_offset));
                    factor_col_offset += dim;
                }

                // Apply robust loss scaling if factor provides one
                let mut scaled_residual = None;
                let mut scaled_jacobian = None;

                if let Some(loss) = factor.get_loss() {
                    let squared_norm: f32 = result.residual.iter().map(|r| r * r).sum();
                    let weight = loss.weight(squared_norm);
                    let sqrt_weight = weight.sqrt();

                    let mut scaled_res = result.residual.clone();
                    let mut scaled_jac = jacobian.clone();

                    // Scale residual and Jacobian by sqrt(weight)
                    for r in &mut scaled_res {
                        *r *= sqrt_weight;
                    }
                    for j in &mut scaled_jac {
                        *j *= sqrt_weight;
                    }

                    scaled_residual = Some(scaled_res);
                    scaled_jacobian = Some(scaled_jac);
                }

                let residual_ref = scaled_residual.as_ref().unwrap_or(&result.residual);
                let jacobian_ref = scaled_jacobian.as_ref().unwrap_or(jacobian);

                for (global_start_i, dim_i, factor_col_offset_i) in &mapping {
                    for (global_start_j, dim_j, factor_col_offset_j) in &mapping {
                        for row in 0..residual_dim {
                            for di in 0..*dim_i {
                                let jac_i_val =
                                    jacobian_ref[row * total_local_dim + factor_col_offset_i + di];
                                for dj in 0..*dim_j {
                                    let jac_j_val = jacobian_ref
                                        [row * total_local_dim + factor_col_offset_j + dj];
                                    jtj[(global_start_i + di, global_start_j + dj)] +=
                                        jac_i_val * jac_j_val;
                                }
                            }
                        }
                    }
                }

                for (global_start_i, dim_i, factor_col_offset_i) in &mapping {
                    for row in 0..residual_dim {
                        let residual_val = residual_ref[row];
                        for di in 0..*dim_i {
                            let jac_val =
                                jacobian_ref[row * total_local_dim + factor_col_offset_i + di];
                            jtr[global_start_i + di] += jac_val * residual_val;
                        }
                    }
                }
            } else {
                return Err(OptimizerError::Factor(FactorError::JacobianFailed(
                    "Jacobian required for optimization".to_string(),
                )));
            }
        }

        Ok((jtj, jtr))
    }
}
