use criterion::{criterion_group, criterion_main, Criterion};
use kornia_algebra::{
    optim::core::{Factor, FactorResult, LinearizationResult, Problem, Variable},
    optim::solvers::LevenbergMarquardt,
    Vec2F32, Vec3AF32, SE3F32,
};
use rand::Rng;

// Projects a 3D point into a camera and compares with observation.
// Simplified pinhole camera model (focal length = 1, principal point = 0)
struct ReprojectionFactor {
    observed_uv: Vec2F32,
}

impl Factor for ReprojectionFactor {
    fn linearize(
        &self,
        params: &[&[f32]],
        compute_jacobian: bool,
    ) -> FactorResult<LinearizationResult> {
        let cam_pose_se3 = SE3F32::from_params(params[0]);
        let point_world = Vec3AF32::new(params[1][0], params[1][1], params[1][2]);

        let point_cam = cam_pose_se3.inverse() * point_world;

        let z = if point_cam.z.abs() < 1e-6 {
            1.0
        } else {
            point_cam.z
        };
        let inv_z = 1.0 / z;

        let u = point_cam.x * inv_z;
        let v = point_cam.y * inv_z;
        let residual = vec![u - self.observed_uv.x, v - self.observed_uv.y];

        if !compute_jacobian {
            return Ok(LinearizationResult::new(residual, None, 9));
        }

        // Use a fixed Jacobian pattern to benchmark the solver's linear algebra performance
        // rather than the cost of Jacobian computation itself.
        let jacobian = vec![1.0; 2 * 9];

        Ok(LinearizationResult::new(residual, Some(jacobian), 9))
    }

    fn residual_dim(&self) -> usize {
        2
    }
    fn num_variables(&self) -> usize {
        2
    }
    fn variable_local_dim(&self, idx: usize) -> usize {
        if idx == 0 {
            6
        } else {
            3
        }
    }
}

fn solve_bundle_adjustment() -> Result<(), Box<dyn std::error::Error>> {
    let mut problem = Problem::new();
    let num_cameras = 10;
    let num_points = 50;

    let mut rng = rand::rng();
    for i in 0..num_cameras {
        let t = Vec3AF32::new(i as f32, 0.0, 0.0);
        let pose = SE3F32::new(kornia_algebra::SO3F32::IDENTITY, t);
        problem.add_variable(
            Variable::se3(format!("c{}", i), pose.to_params().to_vec()),
            pose.to_params().to_vec(),
        )?;
    }

    for j in 0..num_points {
        let pt = vec![
            rng.random::<f32>() * 10.0,
            rng.random::<f32>() * 10.0,
            5.0 + rng.random::<f32>(),
        ];
        problem.add_variable(Variable::euclidean(format!("p{}", j), 3), pt)?;
    }

    for i in 0..num_cameras {
        for j in 0..num_points {
            let factor = ReprojectionFactor {
                observed_uv: Vec2F32::new(0.5, 0.5),
            };
            problem.add_factor(Box::new(factor), vec![format!("c{}", i), format!("p{}", j)])?;
        }
    }

    let optimizer = LevenbergMarquardt::default();
    let _ = optimizer.optimize(&mut problem)?;
    Ok(())
}

fn bench_ba(c: &mut Criterion) {
    let mut group = c.benchmark_group("bundle_adjustment");
    group.bench_function("ba_synthetic_small", |b| {
        b.iter(|| {
            solve_bundle_adjustment().unwrap();
        })
    });
}

criterion_group!(benches, bench_ba);
criterion_main!(benches);
