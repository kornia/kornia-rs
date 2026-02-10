use criterion::{criterion_group, criterion_main, Criterion};
use kornia_algebra::{
    optim::{
        core::{Factor, FactorResult, LinearizationResult, Problem, Variable},
        solvers::LevenbergMarquardt,
    },
    Vec3AF32, SE2F32,
};
use rand::Rng;
use levenberg_marquardt::LeastSquaresProblem;
use nalgebra as na;

struct BetweenFactorSE2 {
    measurement: SE2F32,
    noise_std: f32,
}

impl BetweenFactorSE2 {
    fn new(measurement: SE2F32, noise_std: f32) -> Self {
        Self {
            measurement,
            noise_std,
        }
    }

    fn error(&self, t_i: &SE2F32, t_j: &SE2F32) -> Vec3AF32 {
        // r = Log(Z_ij^-1 * T_i^-1 * T_j)
        let diff = self.measurement.inverse() * t_i.inverse() * *t_j;
        // Weighting: r' = r / sigma
        diff.log() / self.noise_std
    }
}

impl Factor for BetweenFactorSE2 {
    fn linearize(
        &self,
        params: &[&[f32]],
        compute_jacobian: bool,
    ) -> FactorResult<LinearizationResult> {
        let t_i = SE2F32::from_array([params[0][0], params[0][1], params[0][2], params[0][3]]);
        let t_j = SE2F32::from_array([params[1][0], params[1][1], params[1][2], params[1][3]]);

        let residual = self.error(&t_i, &t_j);
        let residual_vec = vec![residual.x, residual.y, residual.z];

        if !compute_jacobian {
            return Ok(LinearizationResult::new(residual_vec, None, 4));
        }

        // Numerical differentiation
        let eps = 1e-5;
        let mut jacobian = Vec::with_capacity(6 * 6); // 3 (residual) * (3 (Ti) + 3 (Tj))

        // Jacobian w.r.t T_i (3 params)
        for k in 0..3 {
            let mut delta = Vec3AF32::ZERO;
            if k == 0 {
                delta.x = eps;
            }
            if k == 1 {
                delta.y = eps;
            }
            if k == 2 {
                delta.z = eps;
            }

            let t_i_pert = t_i.rplus(delta);
            let residual_pert = self.error(&t_i_pert, &t_j);

            jacobian.push((residual_pert.x - residual.x) / eps);
            jacobian.push((residual_pert.y - residual.y) / eps);
            jacobian.push((residual_pert.z - residual.z) / eps);
        }

        // Jacobian w.r.t T_j (3 params)
        for k in 0..3 {
            let mut delta = Vec3AF32::ZERO;
            if k == 0 {
                delta.x = eps;
            }
            if k == 1 {
                delta.y = eps;
            }
            if k == 2 {
                delta.z = eps;
            }

            let t_j_pert = t_j.rplus(delta);
            let residual_pert = self.error(&t_i, &t_j_pert);

            jacobian.push((residual_pert.x - residual.x) / eps);
            jacobian.push((residual_pert.y - residual.y) / eps);
            jacobian.push((residual_pert.z - residual.z) / eps);
        }

        // Reorder for kornia optimizer layout: [J_i_col0, J_j_col0, J_i_col1, J_j_col1 ...]
        // We assume it's simply concatenated Jacobians of variables.


        let mut full_jacobian = vec![0.0; 3 * 6]; // 3 rows, 6 columns

        // Fill Jacobian
        for (col_idx, _) in (0..6).enumerate() {
            let mut delta = Vec3AF32::ZERO;
            let is_i = col_idx < 3;
            let local_k = col_idx % 3;

            if local_k == 0 {
                delta.x = eps;
            }
            if local_k == 1 {
                delta.y = eps;
            }
            if local_k == 2 {
                delta.z = eps;
            }

            let r_pert = if is_i {
                self.error(&t_i.rplus(delta), &t_j)
            } else {
                self.error(&t_i, &t_j.rplus(delta))
            };

            full_jacobian[0 * 6 + col_idx] = (r_pert.x - residual.x) / eps;
            full_jacobian[1 * 6 + col_idx] = (r_pert.y - residual.y) / eps;
            full_jacobian[2 * 6 + col_idx] = (r_pert.z - residual.z) / eps;
        }

        Ok(LinearizationResult::new(
            residual_vec,
            Some(full_jacobian),
            6, // total local dim: 3 (Ti) + 3 (Tj)
        ))
    }

    fn residual_dim(&self) -> usize {
        3
    }

    fn num_variables(&self) -> usize {
        2
    }

    fn variable_local_dim(&self, _idx: usize) -> usize {
        3
    }
}

fn create_pgo_problem() -> Result<Problem, Box<dyn std::error::Error>> {
    let mut problem = Problem::new();
    let num_poses = 5;
    let _radius = 5.0;

    // Add variables (poses in a circle)
    for i in 0..num_poses {
        // Initial guess with some noise
        let mut rng = rand::rng();
        let noise_t = Vec3AF32::new(
            rng.random::<f32>() * 0.1,
            rng.random::<f32>() * 0.1,
            rng.random::<f32>() * 0.1,
        );
        let pose = SE2F32::from_angle_translation(
            (i as f32) * 2.0 * std::f32::consts::PI / (num_poses as f32),
            kornia_algebra::Vec2F32::new(0.0, 0.0), // simplified
        )
        .rplus(noise_t);

        let pose_vec = pose.to_array().to_vec();
        problem.add_variable(
            Variable::se2(format!("x{}", i).as_str(), pose_vec.clone()),
            pose_vec,
        )?;
    }

    // Add factors (odometry constraints)
    for i in 0..num_poses {
        let j = (i + 1) % num_poses;
        // Perfect measurement for circle
        let measurement = SE2F32::from_angle_translation(
            2.0 * std::f32::consts::PI / (num_poses as f32),
            kornia_algebra::Vec2F32::ZERO,
        );

        let factor = BetweenFactorSE2::new(measurement, 0.1);
        problem.add_factor(Box::new(factor), vec![format!("x{}", i), format!("x{}", j)])?;
    }

    // Fix first pose by adding a prior? Or just let it float (gauge ambiguity).
    // LM should handle it with damping, but better to add a prior.
    // For simplicity of benchmark, we skip prior, LM damping handles rank deficiency.

    Ok(problem)
}

fn solve_pgo() -> Result<(), Box<dyn std::error::Error>> {
    let mut problem = create_pgo_problem()?;
    let optimizer = LevenbergMarquardt::default();
    let _ = optimizer.optimize(&mut problem)?;
    Ok(())
}

fn bench_pgo_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimization");

    group.bench_function("se2_pgo_solve_5_nodes", |b| {
        b.iter(|| {
            if let Err(e) = solve_pgo() {
                panic!("Benchmark failed: {}", e);
            }
        })
    });
}


#[derive(Clone)]
struct PgoProblemLM {
    params: na::DVector<f32>,
    measurements: Vec<(usize, usize, SE2F32, f32)>,
    num_poses: usize,
}

impl LeastSquaresProblem<f32, na::Dyn, na::Dyn> for PgoProblemLM {
    type ParameterStorage = na::VecStorage<f32, na::Dyn, na::Const<1>>;
    type ResidualStorage = na::VecStorage<f32, na::Dyn, na::Const<1>>;
    type JacobianStorage = na::VecStorage<f32, na::Dyn, na::Dyn>;

    fn set_params(&mut self, params: &na::DVector<f32>) {
        self.params = params.clone();
    }

    fn params(&self) -> na::DVector<f32> {
        self.params.clone()
    }

    fn residuals(&self) -> Option<na::DVector<f32>> {
        let mut residuals = Vec::with_capacity(3 * self.measurements.len());
        for (i, j, measurement, noise_std) in &self.measurements {
            let idx_i = i * 3;
            let idx_j = j * 3;

            let ti = SE2F32::from_angle_translation(
                self.params[idx_i + 2],
                kornia_algebra::Vec2F32::new(self.params[idx_i], self.params[idx_i + 1]),
            );

            let tj = SE2F32::from_angle_translation(
                self.params[idx_j + 2],
                kornia_algebra::Vec2F32::new(self.params[idx_j], self.params[idx_j + 1]),
            );

            // Using exactly the same error function
            let diff = measurement.inverse() * ti.inverse() * tj;
            let error = diff.log() / *noise_std;

            residuals.push(error.x);
            residuals.push(error.y);
            residuals.push(error.z);
        }
        Some(na::DVector::from_vec(residuals))
    }

    fn jacobian(&self) -> Option<na::DMatrix<f32>> {
        // Numerical differentiation
        let eps = 1e-4;
        let residuals = self.residuals()?;
        let n = self.params.len();
        let m = residuals.len();
        let mut jacobian = na::DMatrix::zeros(m, n);

        for col in 0..n {
            let mut params_pert = self.params.clone();
            params_pert[col] += eps;

            // Create a temporary problem just to compute residuals
            let mut temp_prob = self.clone();
            temp_prob.params = params_pert;
            let res_pert = temp_prob.residuals()?;

            for row in 0..m {
                jacobian[(row, col)] = (res_pert[row] - residuals[row]) / eps;
            }
        }
        Some(jacobian)
    }
}

fn solve_pgo_lm_crate() -> Result<(), Box<dyn std::error::Error>> {
    let num_poses = 5;
    let _radius = 5.0;
    
    // Initial guess
    let mut params_vec = Vec::with_capacity(num_poses * 3);
    for i in 0..num_poses {
        let mut rng = rand::rng();
        let noise_t = Vec3AF32::new(
             rng.random::<f32>() * 0.1,
             rng.random::<f32>() * 0.1,
             rng.random::<f32>() * 0.1
        );
        let pose = SE2F32::from_angle_translation(
            (i as f32) * 2.0 * std::f32::consts::PI / (num_poses as f32),
            kornia_algebra::Vec2F32::new(0.0, 0.0)
        ).rplus(noise_t);
        
        let theta = pose.r.log();
        params_vec.push(pose.t.x);
        params_vec.push(pose.t.y);
        params_vec.push(theta);
    }
    
    // Measurements
    let mut measurements = Vec::new();
     for i in 0..num_poses {
        let j = (i + 1) % num_poses;
        let measurement = SE2F32::from_angle_translation(
            2.0 * std::f32::consts::PI / (num_poses as f32),
            kornia_algebra::Vec2F32::ZERO,
        );
        measurements.push((i, j, measurement, 0.1));
    }
    
    let problem = PgoProblemLM {
        params: na::DVector::from_vec(params_vec),
        measurements,
        num_poses,
    };
    
    let (result, report) = levenberg_marquardt::LevenbergMarquardt::new().minimize(problem);
    
    if !report.termination.was_successful() {
         return Err("Optimization failed".into());
    }
    
    Ok(())
}

fn bench_pgo_optimization_lm(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimization");
    group.bench_function("se2_pgo_solve_5_nodes_lm_crate", |b| {
        b.iter(|| {
             if let Err(e) = solve_pgo_lm_crate() {
                 panic!("Benchmark failed: {}", e);
             }
        })
    });
}

criterion_group!(benches, bench_pgo_optimization, bench_pgo_optimization_lm);
criterion_main!(benches);
