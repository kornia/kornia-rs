//! Curve fitting benchmark: double exponential decay via Levenberg-Marquardt.
//!
//! Models y(t) = c1·exp(−t/τ1) + c2·exp(−t/τ2) and fits {τ1, c1, τ2, c2}
//! from synthetic data.
//!
//! NOTE: A VarPro comparison benchmark was removed because the VarPro solver
//! is not yet implemented. Add it back here once the integration is complete.

use criterion::{criterion_group, criterion_main, Criterion};
use kornia_algebra::optim::core::{Factor, FactorResult, LinearizationResult, Problem, Variable};
use kornia_algebra::optim::solvers::LevenbergMarquardt;
use nalgebra as na;

fn exponential_decay(t: &na::DVector<f64>, tau: f64, c: f64) -> na::DVector<f64> {
    t.map(|t_val| c * (-t_val / tau).exp())
}

fn generate_data() -> (na::DVector<f64>, na::DVector<f64>) {
    let t = na::DVector::from_vec((0..10).map(|i| i as f64).collect());
    let y = exponential_decay(&t, 2.0, 5.0) + exponential_decay(&t, 0.5, 2.0);
    (t, y)
}

struct DoubleExpFactor {
    t: f64,
    y: f64,
}

impl Factor for DoubleExpFactor {
    fn linearize(
        &self,
        params: &[&[f32]],
        compute_jacobian: bool,
    ) -> FactorResult<LinearizationResult> {
        let p = params[0];
        let tau1 = p[0] as f64;
        let c1   = p[1] as f64;
        let tau2 = p[2] as f64;
        let c2   = p[3] as f64;

        let exp1 = (-self.t / tau1).exp();
        let exp2 = (-self.t / tau2).exp();
        let residual = (c1 * exp1 + c2 * exp2 - self.y) as f32;

        if !compute_jacobian {
            return Ok(LinearizationResult::new(vec![residual], None, 4));
        }

        // Analytical Jacobian
        let jacobian = vec![
            (c1 * exp1 * (self.t / (tau1 * tau1))) as f32,
            exp1 as f32,
            (c2 * exp2 * (self.t / (tau2 * tau2))) as f32,
            exp2 as f32,
        ];

        Ok(LinearizationResult::new(vec![residual], Some(jacobian), 4))
    }

    fn residual_dim(&self) -> usize { 1 }
    fn num_variables(&self) -> usize { 1 }
    fn variable_local_dim(&self, _idx: usize) -> usize { 4 }
}

fn solve_double_exp() -> Result<(), Box<dyn std::error::Error>> {
    let (t_vec, y_vec) = generate_data();
    let mut problem = Problem::new();

    problem.add_variable(
        Variable::euclidean("params", 4),
        vec![1.0f32, 1.0, 1.0, 1.0],
    )?;

    for (t, y) in t_vec.iter().zip(y_vec.iter()) {
        problem.add_factor(
            Box::new(DoubleExpFactor { t: *t, y: *y }),
            vec!["params".to_string()],
        )?;
    }

    let optimizer = LevenbergMarquardt::default();
    let _ = optimizer.optimize(&mut problem)?;
    Ok(())
}

fn bench_curve_fitting(c: &mut Criterion) {
    let mut group = c.benchmark_group("curve_fitting");
    group.bench_function("double_exp_lm", |b| {
        b.iter(|| solve_double_exp().unwrap())
    });
    group.finish();
}

criterion_group!(benches, bench_curve_fitting);
criterion_main!(benches);