//! Focused seven-point versus eight-point solver and RANSAC benchmarks.
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use kornia_3d::{
    pose::{
        fundamental_7point, fundamental_8point, ransac_fundamental, ransac_fundamental_8point,
        RansacParams,
    },
    ransac::{
        estimators::{Fundamental8PointEstimator, FundamentalEstimator},
        run, Estimator, Match2d2d, RansacConfig, ThresholdConsensus, UniformSampler,
    },
};
use kornia_algebra::{Vec2F64, Vec3F64};
use rand::{rngs::StdRng, RngExt, SeedableRng};

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

fn run_estimator<E: Estimator<Model = kornia_algebra::Mat3F64, Sample = Match2d2d>>(
    estimator: &E,
    samples: &[Match2d2d],
    cfg: &RansacConfig,
    expected_inliers: Option<usize>,
) {
    let mut sampler = UniformSampler::new(StdRng::seed_from_u64(0));
    let result = run(
        estimator,
        &ThresholdConsensus { threshold: 1.0 },
        &mut sampler,
        samples,
        cfg,
    );
    if let Some(n) = expected_inliers {
        assert_eq!(result.num_iters, cfg.max_iters);
        assert!(result.inliers[..n].iter().all(|&is_inlier| is_inlier));
    }
    std::hint::black_box(result);
}

// Exact probability for the uniform sampler's distinct indices. These oracle
// budgets compare equal sampling success, rather than equal failed-run cost.
fn confidence_budget(n: usize, inliers: usize, sample_size: usize) -> RansacConfig {
    let p = (0..sample_size)
        .map(|j| (inliers - j) as f64 / (n - j) as f64)
        .product::<f64>();
    RansacConfig {
        max_iters: ((1.0_f64 - 0.999_999).ln() / (-p).ln_1p()).ceil() as u32,
        // Keep adaptive stopping above the oracle cap. Each solver consumes
        // exactly its 99.9999% budget, checked after every invocation.
        confidence: 1.0,
        ..Default::default()
    }
}

fn bench_fundamental(c: &mut Criterion) {
    let (a, b, _) = scene(8, 1.0);
    assert!(!fundamental_7point(&a[..7], &b[..7]).unwrap().is_empty());
    assert!(fundamental_8point(&a, &b).is_ok());
    let mut group = c.benchmark_group("fundamental_minimal");
    group.bench_function("7point", |bench| {
        bench.iter(|| {
            std::hint::black_box(
                fundamental_7point(std::hint::black_box(&a[..7]), std::hint::black_box(&b[..7]))
                    .unwrap(),
            );
        })
    });
    group.bench_function("8point", |bench| {
        bench.iter(|| {
            std::hint::black_box(
                fundamental_8point(std::hint::black_box(&a), std::hint::black_box(&b)).unwrap(),
            );
        })
    });
    group.finish();

    // This group measures fixed-budget work; it does not compare reliability.
    let mut group = c.benchmark_group("fundamental_ransac_fixed_budget");
    let cfg = RansacConfig {
        max_iters: 1000,
        confidence: 0.999_999,
        ..Default::default()
    };
    for ratio in [0.2, 0.5, 0.8] {
        let (_, _, matches) = scene(500, ratio);
        group.bench_with_input(
            BenchmarkId::new("7point", ratio),
            &matches,
            |bench, samples| {
                bench.iter(|| {
                    run_estimator(
                        &FundamentalEstimator,
                        std::hint::black_box(samples),
                        &cfg,
                        None,
                    )
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("8point", ratio),
            &matches,
            |bench, samples| {
                bench.iter(|| {
                    run_estimator(
                        &Fundamental8PointEstimator,
                        std::hint::black_box(samples),
                        &cfg,
                        None,
                    )
                });
            },
        );
    }
    group.finish();

    // Public two-view path has independent scoring and count/score tie-breaking.
    let mut group = c.benchmark_group("fundamental_twoview_fixed_budget");
    for ratio in [0.2, 0.5, 0.8] {
        let (a, b, _) = scene(500, ratio);
        let params = RansacParams {
            max_iterations: 1000,
            threshold: 1.0,
            confidence: Some(0.999_999),
            ..Default::default()
        };
        group.bench_function(BenchmarkId::new("7point", ratio), |bench| {
            bench.iter(|| {
                std::hint::black_box(
                    ransac_fundamental(std::hint::black_box(&a), std::hint::black_box(&b), &params)
                        .unwrap(),
                )
            });
        });
        group.bench_function(BenchmarkId::new("8point", ratio), |bench| {
            bench.iter(|| {
                std::hint::black_box(
                    ransac_fundamental_8point(
                        std::hint::black_box(&a),
                        std::hint::black_box(&b),
                        &params,
                    )
                    .unwrap(),
                )
            });
        });
    }
    group.finish();

    let mut group = c.benchmark_group("fundamental_ransac_equal_confidence");
    group.sample_size(10);
    for ratio in [0.2, 0.5, 0.8] {
        let (_, _, matches) = scene(500, ratio);
        let inliers = (500.0 * ratio) as usize;
        let cfg7 = confidence_budget(500, inliers, 7);
        let cfg8 = confidence_budget(500, inliers, 8);
        group.bench_with_input(
            BenchmarkId::new("7point", ratio),
            &matches,
            |bench, samples| {
                bench.iter(|| {
                    run_estimator(
                        &FundamentalEstimator,
                        std::hint::black_box(samples),
                        &cfg7,
                        Some(inliers),
                    )
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("8point", ratio),
            &matches,
            |bench, samples| {
                bench.iter(|| {
                    run_estimator(
                        &Fundamental8PointEstimator,
                        std::hint::black_box(samples),
                        &cfg8,
                        Some(inliers),
                    )
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_fundamental);
criterion_main!(benches);
