use criterion::{criterion_group, criterion_main, Criterion};
use kornia_algebra::optim::core::{Factor, FactorResult, LinearizationResult, Problem, Variable};
use kornia_algebra::optim::solvers::LevenbergMarquardt;

struct RosenbrockFactor {
    a: f32,
    b: f32,
}

impl RosenbrockFactor {
    fn new(a: f32, b: f32) -> Self {
        Self { a, b }
    }
}

impl Factor for RosenbrockFactor {
    fn linearize(
        &self,
        params: &[&[f32]],
        compute_jacobian: bool,
    ) -> FactorResult<LinearizationResult> {
        let x = params[0][0];
        let y = params[0][1];

        let sqrt_b = self.b.sqrt();

        let r1 = self.a - x;
        let r2 = sqrt_b * (y - x * x);
        let residual = vec![r1, r2];

        let jacobian = if compute_jacobian {
            vec![-1.0, 0.0, -2.0 * x * sqrt_b, sqrt_b]
        } else {
            Vec::new()
        };

        Ok(LinearizationResult::new(
            residual,
            if compute_jacobian {
                Some(jacobian)
            } else {
                None
            },
            2,
        ))
    }

    fn residual_dim(&self) -> usize {
        2
    }

    fn num_variables(&self) -> usize {
        1
    }

    fn variable_local_dim(&self, _idx: usize) -> usize {
        2
    }
}

fn solve_rosenbrock() {
    let mut problem = Problem::new();

    problem
        .add_variable(Variable::euclidean("x", 2), vec![-1.2, 1.0])
        .unwrap();

    let factor = RosenbrockFactor::new(1.0, 100.0);
    problem
        .add_factor(Box::new(factor), vec!["x".to_string()])
        .unwrap();

    let optimizer = LevenbergMarquardt::default();
    let _ = optimizer.optimize(&mut problem).unwrap();
}

fn bench_rosenbrock_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimization");

    group.bench_function("rosenbrock_solve", |b| {
        b.iter(|| {
            solve_rosenbrock();
        })
    });
}

criterion_group!(benches, bench_rosenbrock_optimization);
criterion_main!(benches);
