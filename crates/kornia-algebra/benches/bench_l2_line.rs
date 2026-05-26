//! Criterion benchmark for L2 least squares baseline.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use kornia_algebra::optim::{Factor, FactorError, LinearizationResult, Problem, Variable};
use std::f32::consts::PI;
use std::hint::black_box;

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

        let m = line_params[0];
        let b = line_params[1];
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
}

/// Generate synthetic 2D line fitting data
fn generate_line_data(
    num_points: usize,
    noise_std: f32,
    outlier_fraction: f32,
    seed: u64,
) -> Vec<(f32, f32)> {
    let mut data = Vec::new();
    let mut rng_state = seed;

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
            let outlier_offset = (lcg() - 0.5) * 20.0;
            true_y + outlier_offset
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

fn l2_baseline_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("l2_line_fitting");
    group.sample_size(10);

    // Benchmark scenarios
    let scenarios = vec![
        ("clean_small", 30, 0.1, 0.0),
        ("clean_large", 100, 0.1, 0.0),
        ("outliers_30_small", 30, 0.1, 0.3),
        ("outliers_30_large", 100, 0.1, 0.3),
    ];

    let optimizer = kornia_algebra::optim::LevenbergMarquardt::default();

    for (scenario_name, num_points, noise_std, outlier_fraction) in scenarios {
        let data = generate_line_data(num_points, noise_std, outlier_fraction, 12345);

        let benchmark_id = BenchmarkId::from_parameter(format!(
            "{}_n{}_out{}",
            scenario_name,
            num_points,
            (outlier_fraction * 100.0) as u32
        ));

        group.bench_with_input(
            benchmark_id,
            &(scenario_name, num_points, noise_std, outlier_fraction),
            |b, &(_scenario_name, _num_points, _noise_std, _outlier_fraction)| {
                let data_clone = data.clone();
                b.iter(|| {
                    let data = black_box(data_clone.clone());

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

                    let result = optimizer.optimize(&mut problem).unwrap();

                    result.final_cost
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, l2_baseline_benchmarks);
criterion_main!(benches);
