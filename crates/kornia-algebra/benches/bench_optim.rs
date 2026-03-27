//! Optimization benchmarks for the Levenberg-Marquardt solver.
//!
//! Tests solver performance on pose graph optimization — a core SLAM problem
//! where camera poses are connected by odometry measurements.

use criterion::{criterion_group, criterion_main, Criterion};
use kornia_algebra::optim::core::{Factor, FactorResult, LinearizationResult, Problem, Variable};
use kornia_algebra::optim::solvers::LevenbergMarquardt;
use kornia_algebra::Vec2F32;
use rand::Rng;

/// Relative pose measurement between two consecutive 2-D positions.
struct PoseGraphFactor {
    measured_relative: Vec2F32,
    information_weight: f32,
}

impl PoseGraphFactor {
    fn new(measured_relative: Vec2F32, information_weight: f32) -> Self {
        Self { measured_relative, information_weight }
    }
}

impl Factor for PoseGraphFactor {
    fn linearize(
        &self,
        params: &[&[f32]],
        compute_jacobian: bool,
    ) -> FactorResult<LinearizationResult> {
        if params.len() != 2 {
            return Err(kornia_algebra::optim::FactorError::DimensionMismatch {
                expected: 2,
                actual: params.len(),
            });
        }

        let pose1 = params[0]; // [x1, y1]
        let pose2 = params[1]; // [x2, y2]

        let dx = pose2[0] - pose1[0];
        let dy = pose2[1] - pose1[1];

        // Residual: weighted difference between measured and predicted relative pose
        let w = self.information_weight;
        let residual = vec![
            w * (dx - self.measured_relative.x),
            w * (dy - self.measured_relative.y),
        ];

        if !compute_jacobian {
            return Ok(LinearizationResult::new(residual, None, 4));
        }

        // Jacobian (2 × 4, row-major)
        let jacobian = vec![
            -w, 0.0,  w, 0.0,
             0.0, -w, 0.0,  w,
        ];

        Ok(LinearizationResult::new(residual, Some(jacobian), 4))
    }

    fn residual_dim(&self) -> usize { 2 }
    fn num_variables(&self) -> usize { 2 }
    fn variable_local_dim(&self, _idx: usize) -> usize { 2 }
}

/// Chain of `num_poses` 2-D camera poses connected by odometry measurements.
fn solve_pose_graph(num_poses: usize) -> Result<(), Box<dyn std::error::Error>> {
    let mut problem = Problem::new();
    let mut rng = rand::rng();

    // Ground-truth: straight-line trajectory with small lateral jitter
    let mut true_poses = Vec::with_capacity(num_poses);
    let mut cur = Vec2F32::new(0.0, 0.0);
    for _ in 0..num_poses {
        true_poses.push(cur);
        cur.x += 0.3 + rng.random::<f32>() * 0.1 - 0.05;
        cur.y += rng.random::<f32>() * 0.05 - 0.025;
    }

    // Perturbed initial estimates
    for (i, &pose) in true_poses.iter().enumerate() {
        let init = vec![
            pose.x + rng.random::<f32>() * 0.04 - 0.02,
            pose.y + rng.random::<f32>() * 0.04 - 0.02,
        ];
        problem.add_variable(Variable::euclidean(&format!("pose_{}", i), 2), init)?;
    }

    // Odometry factors between consecutive poses
    for i in 0..num_poses - 1 {
        let meas = Vec2F32::new(
            true_poses[i + 1].x - true_poses[i].x,
            true_poses[i + 1].y - true_poses[i].y,
        );
        problem.add_factor(
            Box::new(PoseGraphFactor::new(meas, 1.0)),
            vec![format!("pose_{}", i), format!("pose_{}", i + 1)],
        )?;
    }

    let mut optimizer = LevenbergMarquardt::default();
    optimizer.max_iterations = 5;
    let _ = optimizer.optimize(&mut problem)?;
    Ok(())
}

fn bench_pose_graph_small(c: &mut Criterion) {
    let mut group = c.benchmark_group("pose_graph");
    group.sample_size(20);
    group.bench_function("small_10_poses", |b| {
        b.iter(|| solve_pose_graph(10).unwrap())
    });
    group.finish();
}

fn bench_pose_graph_medium(c: &mut Criterion) {
    let mut group = c.benchmark_group("pose_graph");
    group.sample_size(15);
    group.bench_function("medium_50_poses", |b| {
        b.iter(|| solve_pose_graph(50).unwrap())
    });
    group.finish();
}

criterion_group!(benches, bench_pose_graph_small, bench_pose_graph_medium);
criterion_main!(benches);