// Benchmark for comparing with apex-solver

#[cfg(feature = "bench_external")]
mod internal {
    use apex_solver::{
        core::problem::Problem, factors::Factor, manifold::ManifoldType,
        optimizer::LevenbergMarquardt,
    };
    use criterion::Criterion;
    use std::collections::HashMap;

    // Reprojection factor implementation
    struct ReprojectionFactor {
        observed_u: f64,
        observed_v: f64,
    }

    impl ReprojectionFactor {
        pub fn new(u: f64, v: f64) -> Self {
            Self {
                observed_u: u,
                observed_v: v,
            }
        }
    }

    impl Factor for ReprojectionFactor {
        fn linearize(
            &self,
            params: &[nalgebra::DVector<f64>], // [camera_se3, point_rn]
            compute_jacobian: bool,
        ) -> (nalgebra::DVector<f64>, Option<nalgebra::DMatrix<f64>>) {
            let cam_vec = &params[0];
            let point_vec = &params[1];

            // Dummy residual logic for benchmarking
            let residual = nalgebra::DVector::from_vec(vec![
                cam_vec[0] - point_vec[0] - self.observed_u,
                cam_vec[1] - point_vec[1] - self.observed_v,
            ]);

            let jacobian = if compute_jacobian {
                let mut j = nalgebra::DMatrix::zeros(2, 9);
                j[(0, 0)] = 1.0;
                j[(1, 1)] = 1.0;
                j[(0, 6)] = -1.0;
                Some(j)
            } else {
                None
            };

            (residual, jacobian)
        }

        fn get_dimension(&self) -> usize {
            2
        }
    }

    fn solve_apex_ba() {
        let mut problem = Problem::new();
        let num_cams = 10;
        let num_points = 50;
        let mut initial_values = HashMap::new();

        for i in 0..num_cams {
            let key = format!("c{}", i);
            let val = nalgebra::DVector::from_vec(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
            initial_values.insert(key, (ManifoldType::SE3, val));
        }

        for j in 0..num_points {
            let key = format!("p{}", j);
            let val = nalgebra::DVector::from_vec(vec![1.0, 1.0, 5.0]);
            initial_values.insert(key, (ManifoldType::RN, val));
        }

        for i in 0..num_cams {
            for j in 0..num_points {
                let factor = Box::new(ReprojectionFactor::new(0.0, 0.0));
                problem.add_residual_block(&[&format!("c{}", i), &format!("p{}", j)], factor, None);
            }
        }

        let mut solver = LevenbergMarquardt::new();
        let _result = solver.optimize(&problem, &initial_values);
    }

    pub fn bench_apex(c: &mut Criterion) {
        let mut group = c.benchmark_group("bundle_adjustment_apex");
        group.bench_function("ba_apex_10_50", |b| {
            b.iter(|| {
                solve_apex_ba();
            })
        });
    }
}

use criterion::{criterion_group, criterion_main, Criterion};

#[cfg(feature = "bench_external")]
criterion_group!(benches, internal::bench_apex);

#[cfg(not(feature = "bench_external"))]
fn bench_apex_skip(_c: &mut Criterion) {}

#[cfg(not(feature = "bench_external"))]
criterion_group!(benches, bench_apex_skip);

criterion_main!(benches);
