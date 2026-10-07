use kornia_3d::{
    pose::fundamental_8point,
    ransac::{
        estimators::{Fundamental8PointEstimator, FundamentalEstimator},
        run, Estimator, Match2d2d, RansacConfig, Sampler, ThresholdConsensus, UniformSampler,
    },
};
use kornia_algebra::{Mat3F64, Vec2F64, Vec3F64};
use rand::{rngs::StdRng, RngExt, SeedableRng};
use serde_json::json;
use std::time::Instant;
fn scene(n: usize, inlier_ratio: f64) -> (Vec<Vec2F64>, Vec<Vec2F64>, Vec<Match2d2d>) {
    let mut rng = StdRng::seed_from_u64(42);
    let mut a = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    for i in 0..n {
        let p = Vec3F64::new(
            rng.random_range(-2.0..2.0),
            rng.random_range(-1.5..1.5),
            rng.random_range(3.0..8.0),
        );
        let angle = 0.1_f64;
        let q = Vec3F64::new(
            angle.cos() * p.x + angle.sin() * p.z + 0.5,
            p.y + 0.1,
            -angle.sin() * p.x + angle.cos() * p.z + 0.2,
        );
        a.push(Vec2F64::new(
            800.0 * p.x / p.z + 320.0,
            800.0 * p.y / p.z + 240.0,
        ));
        b.push(if (i as f64) < n as f64 * inlier_ratio {
            Vec2F64::new(800.0 * q.x / q.z + 320.0, 800.0 * q.y / q.z + 240.0)
        } else {
            Vec2F64::new(rng.random_range(0.0..640.0), rng.random_range(0.0..480.0))
        });
    }
    let matches = a
        .iter()
        .zip(&b)
        .map(|(&x1, &x2)| Match2d2d::new(x1, x2))
        .collect();
    (a, b, matches)
}

struct AuditSampler {
    inner: UniformSampler<StdRng>,
    inlier_n: usize,
    all_inlier_draws: u32,
}
impl Sampler for AuditSampler {
    fn sample(&mut self, n: usize, out: &mut [usize]) {
        self.inner.sample(n, out);
        self.all_inlier_draws += out.iter().all(|&i| i < self.inlier_n) as u32;
    }
}
fn budget(n: usize, inliers: usize, k: usize, confidence: f64) -> u32 {
    let p = (0..k)
        .map(|j| (inliers - j) as f64 / (n - j) as f64)
        .product::<f64>();
    ((1.0 - confidence).ln() / (-p).ln_1p()).ceil() as u32
}
fn evaluate<E: Estimator<Model = Mat3F64, Sample = Match2d2d>>(
    e: &E,
    samples: &[Match2d2d],
    truth: &Mat3F64,
    inliers: usize,
    seed: u64,
    mode: &str,
) -> serde_json::Value {
    let k = E::SAMPLE_SIZE;
    let cap = match mode {
        "fixed_1000" => 1000,
        "matched_99_percent" => budget(samples.len(), inliers, k, 0.99),
        "adaptive_99_9_percent" => 5_000_000,
        _ => unreachable!(),
    };
    // The oracle-budget experiment forces the exact number of draws; a
    // near-certain internal target keeps adaptive bounds above either cap.
    let cfg = RansacConfig {
        max_iters: cap,
        confidence: if mode == "adaptive_99_9_percent" {
            0.999
        } else {
            1.0
        },
        ..Default::default()
    };
    let mut sampler = AuditSampler {
        inner: UniformSampler::new(StdRng::seed_from_u64(seed)),
        inlier_n: inliers,
        all_inlier_draws: 0,
    };
    let start = Instant::now();
    let result = run(
        e,
        &ThresholdConsensus { threshold: 1.0 },
        &mut sampler,
        samples,
        &cfg,
    );
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    let recovered = result.inliers[..inliers].iter().filter(|&&v| v).count();
    let norm = (0..3)
        .flat_map(|r| (0..3).map(move |c| (r, c)))
        .map(|(r, c)| truth.to_cols_array()[c * 3 + r].powi(2))
        .sum::<f64>()
        .sqrt();
    let distance = result.model.map(|model| {
        let norm_m = (0..3)
            .flat_map(|r| (0..3).map(move |c| (r, c)))
            .map(|(r, c)| model.to_cols_array()[c * 3 + r].powi(2))
            .sum::<f64>()
            .sqrt();
        let pos = (0..3)
            .flat_map(|r| (0..3).map(move |c| (r, c)))
            .map(|(r, c)| {
                (truth.to_cols_array()[c * 3 + r] / norm
                    - model.to_cols_array()[c * 3 + r] / norm_m)
                    .powi(2)
            })
            .sum::<f64>()
            .sqrt();
        let neg = (0..3)
            .flat_map(|r| (0..3).map(move |c| (r, c)))
            .map(|(r, c)| {
                (truth.to_cols_array()[c * 3 + r] / norm
                    + model.to_cols_array()[c * 3 + r] / norm_m)
                    .powi(2)
            })
            .sum::<f64>()
            .sqrt();
        pos.min(neg)
    });
    let p = (0..k)
        .map(|j| (inliers - j) as f64 / (samples.len() - j) as f64)
        .product::<f64>();
    json!({"solver":k,"mode":mode,"seed":seed,"elapsed_ms":elapsed,"cap":cap,"draws":result.num_iters,"all_inlier_draws":sampler.all_inlier_draws,"true_inliers_recovered":recovered,"support":result.inlier_count(),"projective_distance":distance,"success":recovered==inliers,"oracle_all_inlier_probability":-((result.num_iters as f64)*(-p).ln_1p()).exp_m1()})
}
fn main() {
    let (a, b, matches) = scene(500, 0.2);
    let truth = fundamental_8point(&a[..100], &b[..100]).unwrap();
    let mut rows = Vec::new();
    for mode in ["fixed_1000", "matched_99_percent", "adaptive_99_9_percent"] {
        for seed in 0..5 {
            if seed % 2 == 0 {
                rows.push(evaluate(
                    &FundamentalEstimator,
                    &matches,
                    &truth,
                    100,
                    seed,
                    mode,
                ));
                rows.push(evaluate(
                    &Fundamental8PointEstimator,
                    &matches,
                    &truth,
                    100,
                    seed,
                    mode,
                ));
            } else {
                rows.push(evaluate(
                    &Fundamental8PointEstimator,
                    &matches,
                    &truth,
                    100,
                    seed,
                    mode,
                ));
                rows.push(evaluate(
                    &FundamentalEstimator,
                    &matches,
                    &truth,
                    100,
                    seed,
                    mode,
                ));
            }
            eprintln!("{mode}: seed {seed} complete");
        }
    }
    println!("{}",serde_json::to_string_pretty(&json!({"n":500,"true_inliers":100,"threshold_squared_px":1.0,"seed_count":5,"scene_seed":42,"rows":rows})).unwrap());
}
