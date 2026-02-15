use apex_solver::{
    core::problem::Problem,
    factors::Factor,
    manifold::{se3::SE3, rn::Rn, ManifoldType},
    optimizer::{LevenbergMarquardt, Solver},
};
use nalgebra_033;
use criterion::{criterion_group, criterion_main, Criterion};
use rand::Rng;
use std::collections::HashMap;

// Reprojection factor implementation
struct ReprojectionFactor {
    observed_u: f64,
    observed_v: f64,
}

impl ReprojectionFactor {
    pub fn new(u: f64, v: f64) -> Self {
        Self { observed_u: u, observed_v: v }
    }
}

impl Factor for ReprojectionFactor {
    fn linearize(
        &self,
        params: &[nalgebra_033::DVector<f64>], // [camera_se3, point_rn]
        compute_jacobian: bool,
    ) -> (nalgebra_033::DVector<f64>, Option<nalgebra_033::DMatrix<f64>>) {
        // Param 0: Camera SE3 (7 params: tx, ty, tz, qw, qx, qy, qz)
        // Param 1: Point Rn (3 params: x, y, z)
        
        let cam_vec = &params[0];
        let point_vec = &params[1]; 
        
        // Dummy residual logic for benchmarking
        let residual = nalgebra_033::DVector::from_vec(vec![
            cam_vec[0] - point_vec[0] - self.observed_u,
            cam_vec[1] - point_vec[1] - self.observed_v,
        ]);
        
        let jacobian = if compute_jacobian {
             // 2 rows (residual), 9 columns (6 for SE3 tangent + 3 for Point)
             let mut j = nalgebra_033::DMatrix::zeros(2, 9);
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

    // 1. Create Data
    // 10 cameras, 50 points
    let num_cams = 10;
    let num_points = 50;
    
    // 2. Initialize Variables
    let mut initial_values = HashMap::new();
    
    for i in 0..num_cams {
        let key = format!("c{}", i);
        // SE3: [tx, ty, tz, qw, qx, qy, qz]
        let val = nalgebra_033::DVector::from_vec(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]); 
        initial_values.insert(key, (ManifoldType::SE3, val));
    }
    
    for j in 0..num_points {
        let key = format!("p{}", j);
        let val = nalgebra_033::DVector::from_vec(vec![1.0, 1.0, 5.0]);
        initial_values.insert(key, (ManifoldType::RN, val));
    }
    
    // 3. Add Factors
    for i in 0..num_cams {
        for j in 0..num_points {
             let factor = Box::new(ReprojectionFactor::new(0.0, 0.0));
             problem.add_residual_block(
                &[&format!("c{}", i), &format!("p{}", j)], 
                factor, 
                None
             );
        }
    }
    
    // 5. Solve
    let mut solver = LevenbergMarquardt::new();
    let _result = solver.optimize(&problem, &initial_values);
}

fn bench_apex(c: &mut Criterion) {
    let mut group = c.benchmark_group("bundle_adjustment_apex");
    group.bench_function("ba_apex_10_50", |b| {
        b.iter(|| {
             solve_apex_ba();
        })
    });
}

criterion_group!(benches, bench_apex);
criterion_main!(benches);
