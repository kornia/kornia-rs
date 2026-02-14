//! Unit tests validating L2 least squares baseline.

#[cfg(test)]
mod tests {
    use crate::optim::{Factor, FactorError, LinearizationResult, Problem, Variable};
    use std::f32::consts::PI;

    /// Simple 2D line fitting factor: y = mx + b
    struct LineResidualFactor {
        x: f32,
        observed_y: f32,
    }

    impl LineResidualFactor {
        fn new(x: f32, observed_y: f32) -> Self {
            Self { x, observed_y }
        }
    }

    impl Factor for LineResidualFactor {
        fn linearize(
            &self,
            params: &[&[f32]],
            compute_jacobian: bool,
        ) -> Result<LinearizationResult, FactorError> {
            if params.len() != 1 {
                return Err(FactorError::DimensionMismatch {
                    expected: 1,
                    actual: params.len(),
                });
            }

            let line_params = params[0];
            if line_params.len() != 2 {
                return Err(FactorError::DimensionMismatch {
                    expected: 2,
                    actual: line_params.len(),
                });
            }

            let m = line_params[0]; // slope
            let b = line_params[1]; // intercept

            // Predict: y_pred = m * x + b
            let y_pred = m * self.x + b;

            // Residual: observed - predicted
            let residual = vec![self.observed_y - y_pred];

            // Jacobian: [∂r/∂m, ∂r/∂b] = [-x, -1]
            let jacobian = if compute_jacobian {
                Some(vec![-self.x, -1.0])
            } else {
                None
            };

            Ok(LinearizationResult::new(residual, jacobian, 2))
        }

        fn residual_dim(&self) -> usize {
            1
        }

        fn num_variables(&self) -> usize {
            1
        }

        fn variable_local_dim(&self, _idx: usize) -> usize {
            2
        }
    }

    /// Generate synthetic 2D points on y = 2x + 1 with Gaussian noise
    /// and 30% outliers. Uses numerically stable Box-Muller transform.
    fn generate_noisy_line_data(
        num_points: usize,
        noise_std: f32,
        outlier_fraction: f32,
    ) -> Vec<(f32, f32)> {
        let mut data = Vec::new();
        let mut rng_state = 12345u64;

        // Simple LCG random number generator
        let mut lcg = || {
            rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
            (rng_state >> 32) as f32 / u32::MAX as f32
        };

        let num_outliers = (num_points as f32 * outlier_fraction).ceil() as usize;

        for i in 0..num_points {
            let x = (i as f32 - num_points as f32 / 2.0) / num_points as f32 * 10.0;
            let true_y = 2.0 * x + 1.0;
            let is_outlier = i < num_outliers;

            let y = if is_outlier {
                // Strong outlier: random offset
                let outlier_offset = (lcg() - 0.5) * 20.0;
                true_y + outlier_offset
            } else {
                // Gaussian noise via Box-Muller transform (numerically stable)
                let u1 = (lcg() + 1e-8).min(1.0 - 1e-8); // Clamp to avoid ln(0)
                let u2 = lcg();
                let gauss = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
                true_y + gauss * noise_std
            };

            data.push((x, y));
        }

        data
    }

    #[test]
    fn test_line_fitting_l2_baseline_no_outliers() {
        // Fit a line to clean data (no outliers)
        let data = generate_noisy_line_data(20, 0.1, 0.0);

        let mut problem = Problem::new();
        problem
            .add_variable(Variable::euclidean("line", 2), vec![0.0, 0.0])
            .unwrap();

        for (x, y) in &data {
            let factor = LineResidualFactor::new(*x, *y);
            problem
                .add_factor(Box::new(factor), vec!["line".to_string()])
                .unwrap();
        }

        let optimizer = crate::optim::LevenbergMarquardt::default();
        let result = optimizer.optimize(&mut problem).unwrap();

        let line_vars = problem.get_variables();
        let m = line_vars["line"].values[0];
        let b = line_vars["line"].values[1];

        // Assertions
        assert!(
            result.iterations > 0,
            "Optimization should complete at least one iteration"
        );
        assert!(
            result.final_cost < 1.0,
            "Final L2 cost should be small for clean data, got {:.6e}",
            result.final_cost
        );
        assert!(
            (m - 2.0).abs() < 0.1,
            "Slope should converge to ~2.0, got {:.4}",
            m
        );
        assert!(
            (b - 1.0).abs() < 0.1,
            "Intercept should converge to ~1.0, got {:.4}",
            b
        );
    }

    #[test]
    fn test_line_fitting_l2_baseline_with_outliers() {
        // Fit a line to data with 30% strong outliers
        // L2 minimization will be influenced by outliers (expected behavior for L2)
        // This test establishes the baseline cost to compare against robust losses later
        let clean_data = generate_noisy_line_data(30, 0.1, 0.0);
        let outlier_data = generate_noisy_line_data(30, 0.1, 0.3);

        // Compute clean baseline
        let mut clean_problem = Problem::new();
        clean_problem
            .add_variable(Variable::euclidean("line", 2), vec![0.0, 0.0])
            .unwrap();

        for (x, y) in &clean_data {
            let factor = LineResidualFactor::new(*x, *y);
            clean_problem
                .add_factor(Box::new(factor), vec!["line".to_string()])
                .unwrap();
        }

        let optimizer = crate::optim::LevenbergMarquardt::default();
        let clean_result = optimizer.optimize(&mut clean_problem).unwrap();
        let clean_cost = clean_result.final_cost;

        // Optimize with outliers
        let mut outlier_problem = Problem::new();
        outlier_problem
            .add_variable(Variable::euclidean("line", 2), vec![0.0, 0.0])
            .unwrap();

        for (x, y) in &outlier_data {
            let factor = LineResidualFactor::new(*x, *y);
            outlier_problem
                .add_factor(Box::new(factor), vec!["line".to_string()])
                .unwrap();
        }

        let outlier_result = optimizer.optimize(&mut outlier_problem).unwrap();

        let outlier_vars = outlier_problem.get_variables();
        let _m = outlier_vars["line"].values[0];
        let _b = outlier_vars["line"].values[1];

        // Assertions
        assert!(
            outlier_result.iterations > 0,
            "Optimization should complete at least one iteration"
        );
        // L2 cost with outliers should be noticeably higher than clean case
        assert!(
            outlier_result.final_cost > clean_cost,
            "L2 cost with outliers ({:.6e}) should exceed clean case ({:.6e})",
            outlier_result.final_cost,
            clean_cost
        );
        // Convergence should still occur
        assert!(
            matches!(
                outlier_result.termination_reason,
                crate::optim::TerminationReason::CostConverged
                    | crate::optim::TerminationReason::GradientConverged
            ),
            "Should converge despite outliers"
        );
    }

    #[test]
    fn test_residual_jacobian_assembly() {
        // Validate that residuals and Jacobians are correctly assembled
        let data = vec![(0.0, 1.0), (1.0, 3.0), (2.0, 5.0)];

        let mut problem = Problem::new();
        problem
            .add_variable(Variable::euclidean("line", 2), vec![0.5, 0.5])
            .unwrap();

        for (x, y) in &data {
            let factor = LineResidualFactor::new(*x, *y);
            problem
                .add_factor(Box::new(factor), vec!["line".to_string()])
                .unwrap();
        }

        // Compute initial cost to validate assembly
        let initial_cost = problem.compute_total_cost().unwrap();

        // Expected residuals for (m=0.5, b=0.5):
        // Point (0, 1): r = 1 - (0.5*0 + 0.5) = 0.5
        // Point (1, 3): r = 3 - (0.5*1 + 0.5) = 2.0
        // Point (2, 5): r = 5 - (0.5*2 + 0.5) = 3.5
        // Cost = (0.5^2 + 2.0^2 + 3.5^2) = 0.25 + 4.0 + 12.25 = 16.5
        let expected_cost = 16.5;

        assert!(
            (initial_cost - expected_cost).abs() < 1e-5,
            "Residual assembly mismatch: expected {:.6e}, got {:.6e}",
            expected_cost,
            initial_cost
        );

        // Optimize to true parameters (m=2, b=1)
        let optimizer = crate::optim::LevenbergMarquardt::default();
        let result = optimizer.optimize(&mut problem).unwrap();

        let line_vars = problem.get_variables();
        let m = line_vars["line"].values[0];
        let b = line_vars["line"].values[1];

        assert!((m - 2.0).abs() < 1e-3, "Slope should be ~2.0, got {}", m);
        assert!(
            (b - 1.0).abs() < 1e-3,
            "Intercept should be ~1.0, got {}",
            b
        );
        assert!(result.final_cost < 1e-6, "Final cost should be near zero");
    }

    #[test]
    fn test_convergence_with_increasing_noise() {
        // Test convergence across different noise levels
        for noise_std in &[0.01, 0.1, 0.5] {
            let data = generate_noisy_line_data(50, *noise_std, 0.0);

            let mut problem = Problem::new();
            problem
                .add_variable(Variable::euclidean("line", 2), vec![0.0, 0.0])
                .unwrap();

            for (x, y) in &data {
                let factor = LineResidualFactor::new(*x, *y);
                problem
                    .add_factor(Box::new(factor), vec!["line".to_string()])
                    .unwrap();
            }

            let optimizer = crate::optim::LevenbergMarquardt::default();
            let result = optimizer.optimize(&mut problem).unwrap();

            assert!(result.iterations > 0);
            assert!(
                matches!(
                    result.termination_reason,
                    crate::optim::TerminationReason::CostConverged
                        | crate::optim::TerminationReason::GradientConverged
                ),
                "Should converge for noise level {}",
                noise_std
            );
        }
    }

    #[test]
    fn test_normal_equations_structure() {
        // Validate that normal equations (J^T J and J^T r) are properly formed
        let data = vec![(1.0, 2.0), (2.0, 4.0), (3.0, 6.0)];

        let mut problem = Problem::new();
        problem
            .add_variable(Variable::euclidean("line", 2), vec![1.0, 0.0])
            .unwrap();

        for (x, y) in &data {
            let factor = LineResidualFactor::new(*x, *y);
            problem
                .add_factor(Box::new(factor), vec!["line".to_string()])
                .unwrap();
        }

        let optimizer = crate::optim::LevenbergMarquardt::default();
        let result = optimizer.optimize(&mut problem).unwrap();

        let line_vars = problem.get_variables();
        let m = line_vars["line"].values[0];
        let b = line_vars["line"].values[1];

        // For perfect linear relationship y = 2x
        assert!((m - 2.0).abs() < 1e-4);
        assert!(b.abs() < 1e-4);
        // With perfect data, cost should be near zero
        assert!(result.final_cost < 1e-8);
    }
}
