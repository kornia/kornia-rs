//! Integration tests validating robust loss improvements on outlier data.

#[cfg(test)]
mod tests {
    use crate::optim::{
        CauchyLoss, Factor, FactorError, HuberLoss, IdentityLoss, LinearizationResult,
        OptimizerError, Problem, RobustLoss, Variable,
    };
    use std::f32::consts::PI;

    struct LineFactorWithLoss {
        x: f32,
        observed_y: f32,
        loss: Option<Box<dyn RobustLoss>>,
    }

    impl LineFactorWithLoss {
        fn with_identity_loss(x: f32, observed_y: f32) -> Self {
            Self {
                x,
                observed_y,
                loss: Some(Box::new(IdentityLoss)),
            }
        }

        fn with_huber_loss(x: f32, observed_y: f32, delta: f32) -> Result<Self, FactorError> {
            Ok(Self {
                x,
                observed_y,
                loss: Some(Box::new(HuberLoss::new(delta)?)),
            })
        }

        fn with_cauchy_loss(x: f32, observed_y: f32, scale: f32) -> Result<Self, FactorError> {
            Ok(Self {
                x,
                observed_y,
                loss: Some(Box::new(CauchyLoss::new(scale)?)),
            })
        }
    }

    impl Factor for LineFactorWithLoss {
        fn linearize(
            &self,
            params: &[&[f32]],
            compute_jacobian: bool,
        ) -> Result<LinearizationResult, FactorError> {
            if params.len() != 1 || params[0].len() != 2 {
                return Err(FactorError::DimensionMismatch {
                    expected: 2,
                    actual: params.len(),
                });
            }

            let m = params[0][0];
            let b = params[0][1];
            let y_pred = m * self.x + b;
            let residual = vec![self.observed_y - y_pred];

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

        fn get_loss(&self) -> Option<&dyn RobustLoss> {
            self.loss.as_ref().map(|b| b.as_ref())
        }
    }

    /// Generate synthetic line data: y = 2x + 1 with Gaussian noise and outliers.
    /// Uses fixed RNG seed (42) for deterministic testing.
    fn generate_outlier_line_data(
        num_points: usize,
        noise_std: f32,
        outlier_fraction: f32,
    ) -> Vec<(f32, f32)> {
        let mut data = Vec::new();
        let mut rng_state = 42u64;

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
                true_y + (lcg() - 0.5) * 20.0
            } else {
                let u1 = (lcg() + 1e-8).min(1.0 - 1e-8);
                let u2 = lcg();
                let gauss = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
                true_y + gauss * noise_std
            };

            data.push((x, y));
        }

        data
    }

    #[test]
    fn test_robust_losses_improve_outlier_estimation() -> Result<(), OptimizerError> {
        let data = generate_outlier_line_data(50, 0.1, 0.3);
        let optimizer = crate::optim::LevenbergMarquardt::default();

        // L2 baseline
        let mut l2_problem = Problem::new();
        l2_problem.add_variable(Variable::euclidean("line", 2), vec![0.0, 0.0])?;
        for (x, y) in &data {
            l2_problem.add_factor(
                Box::new(LineFactorWithLoss::with_identity_loss(*x, *y)),
                vec!["line".to_string()],
            )?;
        }
        let l2_result = optimizer.optimize(&mut l2_problem)?;
        let l2_m = l2_problem.get_variables()["line"].values[0];
        let l2_b = l2_problem.get_variables()["line"].values[1];
        let l2_error = (l2_m - 2.0).abs() + (l2_b - 1.0).abs();

        // Huber
        let mut huber_problem = Problem::new();
        huber_problem.add_variable(Variable::euclidean("line", 2), vec![0.0, 0.0])?;
        for (x, y) in &data {
            huber_problem.add_factor(
                Box::new(LineFactorWithLoss::with_huber_loss(*x, *y, 0.25)?),
                vec!["line".to_string()],
            )?;
        }
        let huber_result = optimizer.optimize(&mut huber_problem)?;
        let huber_m = huber_problem.get_variables()["line"].values[0];
        let huber_b = huber_problem.get_variables()["line"].values[1];
        let huber_error = (huber_m - 2.0).abs() + (huber_b - 1.0).abs();

        // Cauchy
        let mut cauchy_problem = Problem::new();
        cauchy_problem.add_variable(Variable::euclidean("line", 2), vec![0.0, 0.0])?;
        for (x, y) in &data {
            cauchy_problem.add_factor(
                Box::new(LineFactorWithLoss::with_cauchy_loss(*x, *y, 0.25)?),
                vec!["line".to_string()],
            )?;
        }
        let cauchy_result = optimizer.optimize(&mut cauchy_problem)?;
        let cauchy_m = cauchy_problem.get_variables()["line"].values[0];
        let cauchy_b = cauchy_problem.get_variables()["line"].values[1];
        let cauchy_error = (cauchy_m - 2.0).abs() + (cauchy_b - 1.0).abs();

        assert!(l2_result.iterations > 0);
        assert!(huber_result.iterations > 0);
        assert!(cauchy_result.iterations > 0);

        assert!(
            huber_error < l2_error,
            "Huber should outperform L2 with outliers"
        );
        assert!(
            cauchy_error < l2_error,
            "Cauchy should outperform L2 with outliers"
        );

        Ok(())
    }

    #[test]
    fn test_robust_losses_do_not_degrade_clean_data() -> Result<(), OptimizerError> {
        let data = generate_outlier_line_data(50, 0.1, 0.0);
        let optimizer = crate::optim::LevenbergMarquardt::default();
        let epsilon = 0.05;

        // L2 baseline
        let mut l2_problem = Problem::new();
        l2_problem.add_variable(Variable::euclidean("line", 2), vec![0.0, 0.0])?;
        for (x, y) in &data {
            l2_problem.add_factor(
                Box::new(LineFactorWithLoss::with_identity_loss(*x, *y)),
                vec!["line".to_string()],
            )?;
        }
        let l2_result = optimizer.optimize(&mut l2_problem)?;
        let l2_m = l2_problem.get_variables()["line"].values[0];
        let l2_b = l2_problem.get_variables()["line"].values[1];
        let l2_error = (l2_m - 2.0).abs() + (l2_b - 1.0).abs();

        // Huber
        let mut huber_problem = Problem::new();
        huber_problem.add_variable(Variable::euclidean("line", 2), vec![0.0, 0.0])?;
        for (x, y) in &data {
            huber_problem.add_factor(
                Box::new(LineFactorWithLoss::with_huber_loss(*x, *y, 0.25)?),
                vec!["line".to_string()],
            )?;
        }
        let huber_result = optimizer.optimize(&mut huber_problem)?;
        let huber_m = huber_problem.get_variables()["line"].values[0];
        let huber_b = huber_problem.get_variables()["line"].values[1];
        let huber_error = (huber_m - 2.0).abs() + (huber_b - 1.0).abs();

        // Cauchy
        let mut cauchy_problem = Problem::new();
        cauchy_problem.add_variable(Variable::euclidean("line", 2), vec![0.0, 0.0])?;
        for (x, y) in &data {
            cauchy_problem.add_factor(
                Box::new(LineFactorWithLoss::with_cauchy_loss(*x, *y, 0.25)?),
                vec!["line".to_string()],
            )?;
        }
        let cauchy_result = optimizer.optimize(&mut cauchy_problem)?;
        let cauchy_m = cauchy_problem.get_variables()["line"].values[0];
        let cauchy_b = cauchy_problem.get_variables()["line"].values[1];
        let cauchy_error = (cauchy_m - 2.0).abs() + (cauchy_b - 1.0).abs();

        assert!(l2_result.iterations > 0);
        assert!(huber_result.iterations > 0);
        assert!(cauchy_result.iterations > 0);

        assert!(
            (huber_error - l2_error).abs() < epsilon,
            "Huber should match L2 on clean data"
        );
        assert!(
            (cauchy_error - l2_error).abs() < epsilon,
            "Cauchy should match L2 on clean data"
        );

        Ok(())
    }
}
