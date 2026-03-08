//! Bundle adjustment: finite-difference Jacobians vs analytical Jacobians.
//!
//! Both variants solve the same synthetic problem (10 cameras, 50 points) using
//! kornia-algebra's Levenberg-Marquardt solver. The only difference is how the
//! Jacobian is computed inside the factor:
//!
//!   - `fd`         : finite differences (ε = 1e-6), no hand-derived math required.
//!   - `analytical` : closed-form Jacobian (see bench_bundle_adjustment.rs).
//!
//! This benchmark answers a practical question for solver developers:
//! "Is it worth deriving the analytical Jacobian, or is finite-differencing fast enough?"

use criterion::{criterion_group, criterion_main, Criterion};
use kornia_algebra::{
    optim::core::{Factor, FactorResult, LinearizationResult, Problem, Variable},
    optim::solvers::LevenbergMarquardt,
    SE3F32, Vec2F32, Vec3AF32,
};
use rand::Rng;

// Shared projection helper 

/// Projects `point_world` through `pose` (T_wc) onto the normalised image plane.
/// Returns `(x/z, y/z)` in camera frame.
fn project(pose: &SE3F32, point_world: &Vec3AF32) -> Option<(f32, f32)> {
    let p = pose.inverse() * *point_world;
    if p.z.abs() < 1e-6 {
        return None;
    }
    Some((p.x / p.z, p.y / p.z))
}

// Factor: finite-difference Jacobian 

struct ReprojectionFD {
    observed_uv: Vec2F32,
}

impl Factor for ReprojectionFD {
    fn linearize(
        &self,
        params: &[&[f32]],
        compute_jacobian: bool,
    ) -> FactorResult<LinearizationResult> {
        let pose = SE3F32::from_params(params[0]);
        let pw = Vec3AF32::new(params[1][0], params[1][1], params[1][2]);

        let (u, v) = project(&pose, &pw).unwrap_or((1e6, 1e6));
        let residual = vec![u - self.observed_uv.x, v - self.observed_uv.y];

        if !compute_jacobian {
            return Ok(LinearizationResult::new(residual, None, 9));
        }

        // Finite-difference Jacobian (2 × 9, row-major).
        
        const EPS: f32 = 1e-5;
        let mut jac = vec![0.0f32; 2 * 9];

        // Translation DOFs (columns 0, 1, 2): perturb T by adding δt to translation
        for k in 0..3 {
            let mut delta_t = kornia_algebra::Vec3AF32::ZERO;
            match k {
                0 => delta_t.x = EPS,
                1 => delta_t.y = EPS,
                _ => delta_t.z = EPS,
            }
            let pose_p = SE3F32::new(pose.r, pose.t + delta_t);
            let (up, vp) = project(&pose_p, &pw).unwrap_or((u, v));
            jac[k]     = (up - u) / EPS; // row 0
            jac[9 + k] = (vp - v) / EPS; // row 1
        }

        // Rotation DOFs (columns 3, 4, 5): right-perturb rotation via SO3::exp
        for k in 0..3 {
            let mut omega = kornia_algebra::Vec3AF32::ZERO;
            match k {
                0 => omega.x = EPS,
                1 => omega.y = EPS,
                _ => omega.z = EPS,
            }
            let delta_r = kornia_algebra::SO3F32::exp(omega);
            let pose_p = SE3F32::new(pose.r * delta_r, pose.t);
            let (up, vp) = project(&pose_p, &pw).unwrap_or((u, v));
            jac[3 + k]      = (up - u) / EPS; // row 0
            jac[9 + 3 + k]  = (vp - v) / EPS; // row 1
        }

        // Point DOFs (columns 6, 7, 8): perturb p_world directly
        for k in 0..3 {
            let mut pt = [params[1][0], params[1][1], params[1][2]];
            pt[k] += EPS;
            let pw_p = Vec3AF32::new(pt[0], pt[1], pt[2]);
            let (up, vp) = project(&pose, &pw_p).unwrap_or((u, v));
            jac[6 + k]     = (up - u) / EPS; // row 0
            jac[9 + 6 + k] = (vp - v) / EPS; // row 1
        }

        Ok(LinearizationResult::new(residual, Some(jac), 9))
    }

    fn residual_dim(&self) -> usize { 2 }
    fn num_variables(&self) -> usize { 2 }
    fn variable_local_dim(&self, idx: usize) -> usize { if idx == 0 { 6 } else { 3 } }
}

// Factor: analytical Jacobian 

struct ReprojectionAnalytical {
    observed_uv: Vec2F32,
}

impl Factor for ReprojectionAnalytical {
    fn linearize(
        &self,
        params: &[&[f32]],
        compute_jacobian: bool,
    ) -> FactorResult<LinearizationResult> {
        let pose = SE3F32::from_params(params[0]);
        let pw = Vec3AF32::new(params[1][0], params[1][1], params[1][2]);
        let pc = pose.inverse() * pw;

        let z = if pc.z.abs() < 1e-6 { 1e-6 } else { pc.z };
        let inv_z = 1.0 / z;
        let inv_z2 = inv_z * inv_z;
        let x = pc.x;
        let y = pc.y;

        let residual = vec![x * inv_z - self.observed_uv.x, y * inv_z - self.observed_uv.y];

        if !compute_jacobian {
            return Ok(LinearizationResult::new(residual, None, 9));
        }

        let j_se3_row0 = [
            -inv_z,                      
            0.0,                         
            x * inv_z2,                  
            x * y * inv_z2,            
            -(1.0 + x * x * inv_z2),    
            y * inv_z,                   
        ];
        let j_se3_row1 = [
            0.0,                         
            -inv_z,                      
            y * inv_z2,                  
            1.0 + y * y * inv_z2,      
            -x * y * inv_z2,             
            -x * inv_z,                  
        ];

        // Point Jacobian: ∂r/∂p_world = J_proj · R^T  (2×3)
        let qw = pose.r.q.w; let qx = pose.r.q.x;
        let qy = pose.r.q.y; let qz = pose.r.q.z;
        let rt = [
            [1.0-2.0*(qy*qy+qz*qz), 2.0*(qx*qy+qw*qz), 2.0*(qx*qz-qw*qy)],
            [2.0*(qx*qy-qw*qz), 1.0-2.0*(qx*qx+qz*qz), 2.0*(qy*qz+qw*qx)],
            [2.0*(qx*qz+qw*qy), 2.0*(qy*qz-qw*qx), 1.0-2.0*(qx*qx+qy*qy)],
        ];
        let j_pt_row0 = [
            inv_z*rt[0][0] - x*inv_z2*rt[2][0],
            inv_z*rt[0][1] - x*inv_z2*rt[2][1],
            inv_z*rt[0][2] - x*inv_z2*rt[2][2],
        ];
        let j_pt_row1 = [
            inv_z*rt[1][0] - y*inv_z2*rt[2][0],
            inv_z*rt[1][1] - y*inv_z2*rt[2][1],
            inv_z*rt[1][2] - y*inv_z2*rt[2][2],
        ];

        let mut jacobian = vec![0.0f32; 2 * 9];
        jacobian[0..6].copy_from_slice(&j_se3_row0);
        jacobian[6..9].copy_from_slice(&j_pt_row0);
        jacobian[9..15].copy_from_slice(&j_se3_row1);
        jacobian[15..18].copy_from_slice(&j_pt_row1);

        Ok(LinearizationResult::new(residual, Some(jacobian), 9))
    }

    fn residual_dim(&self) -> usize { 2 }
    fn num_variables(&self) -> usize { 2 }
    fn variable_local_dim(&self, idx: usize) -> usize { if idx == 0 { 6 } else { 3 } }
}

// Problem builder 

fn build_problem<F>(make_factor: F) -> Result<Problem, Box<dyn std::error::Error>>
where
    F: Fn() -> Box<dyn Factor>,
{
    let mut problem = Problem::new();
    let mut rng = rand::rng();

    for i in 0..10usize {
        let t = Vec3AF32::new(i as f32 * 0.5, 0.0, 0.0);
        let pose = SE3F32::new(kornia_algebra::SO3F32::IDENTITY, t);
        problem.add_variable(
            Variable::se3(format!("c{}", i), pose.to_params().to_vec()),
            pose.to_params().to_vec(),
        )?;
    }
    for j in 0..50usize {
        let pt = vec![
            rng.random::<f32>() * 4.0 - 2.0,
            rng.random::<f32>() * 4.0 - 2.0,
            4.0 + rng.random::<f32>(),
        ];
        problem.add_variable(Variable::euclidean(format!("p{}", j), 3), pt)?;
    }
    for i in 0..10usize {
        for j in 0..50usize {
            problem.add_factor(
                make_factor(),
                vec![format!("c{}", i), format!("p{}", j)],
            )?;
        }
    }
    Ok(problem)
}

// Benchmarks 

fn bench_fd_vs_analytical(c: &mut Criterion) {
    let mut group = c.benchmark_group("bundle_adjustment_jacobian");
    group.measurement_time(std::time::Duration::from_secs(15));
    group.sample_size(20);

    group.bench_function("finite_difference", |b| {
        b.iter(|| {
            let mut problem = build_problem(|| {
                Box::new(ReprojectionFD { observed_uv: Vec2F32::new(0.0, 0.0) })
            }).unwrap();
            let _ = LevenbergMarquardt::default().optimize(&mut problem).unwrap();
        })
    });

    group.bench_function("analytical", |b| {
        b.iter(|| {
            let mut problem = build_problem(|| {
                Box::new(ReprojectionAnalytical { observed_uv: Vec2F32::new(0.0, 0.0) })
            }).unwrap();
            let _ = LevenbergMarquardt::default().optimize(&mut problem).unwrap();
        })
    });

    group.finish();
}

criterion_group!(benches, bench_fd_vs_analytical);
criterion_main!(benches);