use criterion::{criterion_group, criterion_main, Criterion};
use kornia_algebra::optim::core::{Factor, FactorResult, LinearizationResult, Problem, Variable};
use kornia_algebra::optim::solvers::LevenbergMarquardt;
use nalgebra as na;
use varpro::prelude::*;
use varpro::solvers::levmar::{LevMarProblemBuilder, LevMarSolver};

// --- DATA GENERATION ---
fn exponential_decay(t: &na::DVector<f64>, tau: f64, c: f64) -> na::DVector<f64> {
    t.map(|t_val| c * (-t_val / tau).exp())
}

fn generate_data() -> (na::DVector<f64>, na::DVector<f64>) {
    let t = na::DVector::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    let y = exponential_decay(&t, 2.0, 5.0) + exponential_decay(&t, 0.5, 2.0);
    // No noise for benchmark purity
    (t, y)
}

// --- KORNIA ALGEBRA IMPLEMENTATION ---
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
        // Param layout: [tau1, c1, tau2, c2]
        // Actually, let's keep it simple: params[0] = [tau1, c1, tau2, c2] ?
        // No, usually variables are separate. Let's say one variable of dim 4.
        
        // params[0] is [tau1, c1, tau2, c2]
        let p = params[0];
        let tau1 = p[0] as f64;
        let c1 = p[1] as f64;
        let tau2 = p[2] as f64;
        let c2 = p[3] as f64;

        let exp1 = (-self.t / tau1).exp();
        let exp2 = (-self.t / tau2).exp();
        
        let y_pred = c1 * exp1 + c2 * exp2;
        let residual = y_pred - self.y;

        if !compute_jacobian {
            return Ok(LinearizationResult::new(vec![residual as f32], None, 4));
        }

        // Jacobians
        // d/dtau1 = c1 * exp1 * (t / tau1^2)
        // d/dc1 = exp1
        // d/dtau2 = c2 * exp2 * (t / tau2^2)
        // d/dc2 = exp2
        
        let d_tau1 = c1 * exp1 * (self.t / (tau1 * tau1));
        let d_c1 = exp1;
        let d_tau2 = c2 * exp2 * (self.t / (tau2 * tau2));
        let d_c2 = exp2;

        let jacobian = vec![
            d_tau1 as f32, d_c1 as f32, d_tau2 as f32, d_c2 as f32
        ];

        Ok(LinearizationResult::new(
            vec![residual as f32],
            Some(jacobian),
             4,
        ))
    }
    
    fn residual_dim(&self) -> usize { 1 }
    fn num_variables(&self) -> usize { 1 }
    fn variable_local_dim(&self, _idx: usize) -> usize { 4 }
}

fn solve_kornia() -> Result<(), Box<dyn std::error::Error>> {
    let (t_vec, y_vec) = generate_data();
    let mut problem = Problem::new();
    
    // Initial guess: tau1, c1, tau2, c2
    let initial_guess = vec![1.0f32, 1.0, 1.0, 1.0];
    problem.add_variable(Variable::euclidean("params", 4), initial_guess)?;
    
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


// --- VARPRO IMPLEMENTATION ---
// Separable model: f(t, alpha, c) = c1 * exp(-t/tau1) + c2 * exp(-t/tau2)
// Basis functions: phi1 = exp(-t/tau1), phi2 = exp(-t/tau2)
// Nonlinear params (alpha): tau1, tau2
// Linear params (c): c1, c2

fn solve_varpro() -> Result<(), Box<dyn std::error::Error>> {
    let (t, y) = generate_data();

    // Basis functions
    let model = SeparableModelBuilder::<f64>::new(vec!["tau1", "tau2"])
        .function(&["tau1"], move |x: &na::DVector<f64>, tau1: f64| {
            x.map(|t| (-t / tau1).exp())
        })
        .partial_deriv("tau1", move |x: &na::DVector<f64>, tau1: f64| {
            x.map(|t| (-t / tau1).exp() * (t / (tau1 * tau1)))
        })
        .function(&["tau2"], move |x: &na::DVector<f64>, tau2: f64| {
            x.map(|t| (-t / tau2).exp())
        })
        .partial_deriv("tau2", move |x: &na::DVector<f64>, tau2: f64| {
             x.map(|t| (-t / tau2).exp() * (t / (tau2 * tau2)))
        })
        .independent_variable(t.clone())
        .initial_parameters(vec![1.0, 1.0]) // Initial guess for tau1, tau2
        .build()
        .unwrap();

    let problem = LevMarProblemBuilder::new(model)
        .observations(y)
        .build()
        .unwrap();
        
    let _fit_result = LevMarSolver::new().fit(problem).unwrap();
    
    // VarPro solves for linear parameters (c1, c2) automatically
    // let alpha = solved_problem.params();
    // let c = solved_problem.linear_coefficients().unwrap();
    
    Ok(())
}

fn bench_curve_fitting(c: &mut Criterion) {
    let mut group = c.benchmark_group("curve_fitting");
    
    group.bench_function("double_exp_kornia", |b| {
        b.iter(|| {
             solve_kornia().unwrap();
        })
    });
    
    group.bench_function("double_exp_varpro", |b| {
        b.iter(|| {
             solve_varpro().unwrap();
        })
    });
}

criterion_group!(benches, bench_curve_fitting);
criterion_main!(benches);
