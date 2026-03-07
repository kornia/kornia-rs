use criterion::{criterion_group, criterion_main, Criterion};
use kornia_algebra::{
    optim::core::{Factor, FactorResult, LinearizationResult, Problem, Variable},
    optim::solvers::LevenbergMarquardt,
    SE3F32, Vec2F32, Vec3AF32,
};
use rand::Rng;

/// Reprojection residual factor for bundle adjustment.
///
/// Residual: r = [X_c/Z_c − u_obs, Y_c/Z_c − v_obs]
/// where (X_c, Y_c, Z_c) = T_cw * p_world  (normalized image plane, unit focal length).
struct ReprojectionFactor {
    observed_uv: Vec2F32,
}

impl Factor for ReprojectionFactor {
    fn linearize(
        &self,
        params: &[&[f32]],
        compute_jacobian: bool,
    ) -> FactorResult<LinearizationResult> {
        let cam_pose_se3 = SE3F32::from_params(params[0]); // T_wc (world frame of camera)
        let point_world = Vec3AF32::new(params[1][0], params[1][1], params[1][2]);

        // Transform point into camera frame: p_c = T_cw * p_world = T_wc^{-1} * p_world
        let point_cam = cam_pose_se3.inverse() * point_world;

        let z = if point_cam.z.abs() < 1e-6 { 1e-6 } else { point_cam.z };
        let inv_z = 1.0 / z;
        let inv_z2 = inv_z * inv_z;

        let x = point_cam.x;
        let y = point_cam.y;

        let residual = vec![x * inv_z - self.observed_uv.x, y * inv_z - self.observed_uv.y];

        if !compute_jacobian {
            return Ok(LinearizationResult::new(residual, None, 9));
        }

        let j_se3_row0 = [
            x * y * inv_z2,
            -(1.0 + x * x * inv_z2),
            y * inv_z,
            -inv_z,
            0.0,
            x * inv_z2,
        ];
        let j_se3_row1 = [
            1.0 + y * y * inv_z2,
            -x * y * inv_z2,
            -x * inv_z,
            0.0,
            -inv_z,
            y * inv_z2,
        ];

     
        let qw = cam_pose_se3.r.q.w;
        let qx = cam_pose_se3.r.q.x;
        let qy = cam_pose_se3.r.q.y;
        let qz = cam_pose_se3.r.q.z;

        // Rows of R^T (= columns of R)
        let rt_row0 = [1.0 - 2.0*(qy*qy + qz*qz),  2.0*(qx*qy + qw*qz),  2.0*(qx*qz - qw*qy)];
        let rt_row1 = [2.0*(qx*qy - qw*qz),  1.0 - 2.0*(qx*qx + qz*qz),  2.0*(qy*qz + qw*qx)];
        let rt_row2 = [2.0*(qx*qz + qw*qy),  2.0*(qy*qz - qw*qx),  1.0 - 2.0*(qx*qx + qy*qy)];

        // J_proj · R^T: J_proj = [[1/z, 0, −x/z²], [0, 1/z, −y/z²]]
        let j_pt_row0 = [
            inv_z * rt_row0[0] - x * inv_z2 * rt_row2[0], // ∂u/∂p_x
            inv_z * rt_row0[1] - x * inv_z2 * rt_row2[1], // ∂u/∂p_y
            inv_z * rt_row0[2] - x * inv_z2 * rt_row2[2], // ∂u/∂p_z
        ];
        let j_pt_row1 = [
            inv_z * rt_row1[0] - y * inv_z2 * rt_row2[0], // ∂v/∂p_x
            inv_z * rt_row1[1] - y * inv_z2 * rt_row2[1], // ∂v/∂p_y
            inv_z * rt_row1[2] - y * inv_z2 * rt_row2[2], // ∂v/∂p_z
        ];

        // Assemble [J_se3 | J_point] row-major, shape (2, 9)
        let mut jacobian = vec![0.0f32; 2 * 9];
        jacobian[0..6].copy_from_slice(&j_se3_row0);
        jacobian[6..9].copy_from_slice(&j_pt_row0);
        jacobian[9..15].copy_from_slice(&j_se3_row1);
        jacobian[15..18].copy_from_slice(&j_pt_row1);

        Ok(LinearizationResult::new(residual, Some(jacobian), 9))
    }

    fn residual_dim(&self) -> usize { 2 }
    fn num_variables(&self) -> usize { 2 }
    fn variable_local_dim(&self, idx: usize) -> usize {
        if idx == 0 { 6 } else { 3 }
    }
}

/// Synthetic small bundle adjustment: 10 cameras, 50 3-D points, full visibility.
fn solve_bundle_adjustment() -> Result<(), Box<dyn std::error::Error>> {
    let mut problem = Problem::new();
    let num_cameras = 10;
    let num_points = 50;

    let mut rng = rand::rng();

    for i in 0..num_cameras {
        let t = Vec3AF32::new(i as f32 * 0.5, 0.0, 0.0);
        let pose = SE3F32::new(kornia_algebra::SO3F32::IDENTITY, t);
        problem.add_variable(
            Variable::se3(format!("c{}", i), pose.to_params().to_vec()),
            pose.to_params().to_vec(),
        )?;
    }

    for j in 0..num_points {
        let pt = vec![
            rng.random::<f32>() * 4.0 - 2.0, // x ∈ [−2, 2]
            rng.random::<f32>() * 4.0 - 2.0, // y ∈ [−2, 2]
            4.0 + rng.random::<f32>(),        // z > 0 (in front of all cameras)
        ];
        problem.add_variable(Variable::euclidean(format!("p{}", j), 3), pt)?;
    }

    // Full-visibility graph: every camera observes every point
    for i in 0..num_cameras {
        for j in 0..num_points {
            let factor = ReprojectionFactor {
                observed_uv: Vec2F32::new(0.0, 0.0), // centred observation
            };
            problem.add_factor(
                Box::new(factor),
                vec![format!("c{}", i), format!("p{}", j)],
            )?;
        }
    }

    let optimizer = LevenbergMarquardt::default();
    let _ = optimizer.optimize(&mut problem)?;
    Ok(())
}

fn bench_ba(c: &mut Criterion) {
    let mut group = c.benchmark_group("bundle_adjustment");
    group.measurement_time(std::time::Duration::from_secs(10));
    group.sample_size(20);
    group.bench_function("ba_synthetic_10cams_50pts", |b| {
        b.iter(|| solve_bundle_adjustment().unwrap())
    });
    group.finish();
}

criterion_group!(benches, bench_ba);
criterion_main!(benches);