//! End-to-end `umeyama_f64` / `umeyama_f64_robust` benchmarks over point
//! counts, weight modes, and outlier fractions.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use kornia_algebra::linalg::rigid::{umeyama_f64, umeyama_f64_robust, RobustUmeyamaConfig};
use kornia_algebra::{Mat3F64, Vec3F64};

/// Cheap deterministic pseudo-random value in `[-0.5, 0.5)` (no rng dependency).
fn pseudo(i: usize) -> f64 {
    let x = (i.wrapping_mul(1_664_525) ^ i).wrapping_add(1_013_904_223) as u64;
    (x % 1000) as f64 / 1000.0 - 0.5
}

fn bench_umeyama(c: &mut Criterion) {
    let mut group = c.benchmark_group("umeyama_f64");

    for n in [8usize, 50, 200] {
        let src: Vec<Vec3F64> = (0..n)
            .map(|i| Vec3F64::new(pseudo(3 * i), pseudo(3 * i + 1), pseudo(3 * i + 2)))
            .collect();
        // A rigid translation-only transform: the fit is exact and well-conditioned.
        let dst: Vec<Vec3F64> = src
            .iter()
            .map(|p| *p + Vec3F64::new(1.0, 2.0, 3.0))
            .collect();
        let weights: Vec<f64> = (0..n).map(|i| (i as f64 % 3.0) + 1.0).collect();

        group.bench_function(BenchmarkId::new("unweighted", n), |b| {
            b.iter(|| {
                std::hint::black_box(umeyama_f64(
                    std::hint::black_box(&src),
                    std::hint::black_box(&dst),
                    None,
                    true,
                ))
            });
        });
        group.bench_function(BenchmarkId::new("weighted", n), |b| {
            b.iter(|| {
                std::hint::black_box(umeyama_f64(
                    std::hint::black_box(&src),
                    std::hint::black_box(&dst),
                    Some(std::hint::black_box(&weights)),
                    true,
                ))
            });
        });
    }

    group.finish();
}

/// Robust `umeyama_f64_robust` benchmark: clean data (fast path), 10% and
/// 40% gross outliers (IRLS + MAD rejection), with the plain unweighted
/// `umeyama_f64` as the overhead baseline.
fn bench_umeyama_robust(c: &mut Criterion) {
    let mut group = c.benchmark_group("umeyama_f64_robust");
    let cfg = RobustUmeyamaConfig::default();
    let r_true = Mat3F64::from_cols_array(&[0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
    let t_true = Vec3F64::new(0.5, -0.3, 2.0);
    let s_true = 1.7;

    // Clean data: the seed fit plus a possible MAD refit at the SVD noise
    // floor (the all-ones fast path skips reweighted solves); ~4-5x the cost
    // of `umeyama_f64` on this machine.
    for n in [20usize, 100, 1000] {
        let src: Vec<Vec3F64> = (0..n)
            .map(|i| Vec3F64::new(pseudo(3 * i), pseudo(3 * i + 1), pseudo(3 * i + 2)))
            .collect();
        let dst: Vec<Vec3F64> = src
            .iter()
            .map(|p| (r_true * *p) * s_true + t_true)
            .collect();
        group.bench_function(BenchmarkId::new("clean", n), |b| {
            b.iter(|| {
                std::hint::black_box(umeyama_f64_robust(
                    std::hint::black_box(&src),
                    std::hint::black_box(&dst),
                    std::hint::black_box(&cfg),
                    std::hint::black_box(true),
                ))
            });
        });
    }

    // 10% gross outliers: realistic misdecode rate, extra IRLS iterations.
    {
        let n = 100;
        let src: Vec<Vec3F64> = (0..n)
            .map(|i| Vec3F64::new(pseudo(3 * i), pseudo(3 * i + 1), pseudo(3 * i + 2)))
            .collect();
        let mut dst: Vec<Vec3F64> = src
            .iter()
            .map(|p| (r_true * *p) * s_true + t_true)
            .collect();
        for i in (0..n).step_by(10) {
            dst[i] += Vec3F64::new(50.0, -40.0, 30.0);
        }
        group.bench_function(BenchmarkId::new("outliers_10pct", n), |b| {
            b.iter(|| {
                std::hint::black_box(umeyama_f64_robust(
                    std::hint::black_box(&src),
                    std::hint::black_box(&dst),
                    std::hint::black_box(&cfg),
                    std::hint::black_box(true),
                ))
            });
        });
    }

    // 40% gross outliers (40 of 100, spread): worst case — full IRLS budget
    // + MAD rejection + refit.
    {
        let n = 100;
        let src: Vec<Vec3F64> = (0..n)
            .map(|i| Vec3F64::new(pseudo(3 * i + 7), pseudo(3 * i + 8), pseudo(3 * i + 9)))
            .collect();
        let mut dst: Vec<Vec3F64> = src
            .iter()
            .map(|p| (r_true * *p) * s_true + t_true)
            .collect();
        for i in (0..n).step_by(5).chain((2..n).step_by(5)) {
            dst[i] += Vec3F64::new(30.0, 40.0, -50.0);
        }
        group.bench_function(BenchmarkId::new("outliers_40pct", n), |b| {
            b.iter(|| {
                std::hint::black_box(umeyama_f64_robust(
                    std::hint::black_box(&src),
                    std::hint::black_box(&dst),
                    std::hint::black_box(&cfg),
                    std::hint::black_box(true),
                ))
            });
        });
    }

    // Baseline: the plain unweighted closed form on the clean n=100 set.
    {
        let n = 100;
        let src: Vec<Vec3F64> = (0..n)
            .map(|i| Vec3F64::new(pseudo(3 * i), pseudo(3 * i + 1), pseudo(3 * i + 2)))
            .collect();
        let dst: Vec<Vec3F64> = src
            .iter()
            .map(|p| (r_true * *p) * s_true + t_true)
            .collect();
        group.bench_function(BenchmarkId::new("baseline_unweighted", n), |b| {
            b.iter(|| {
                std::hint::black_box(umeyama_f64(
                    std::hint::black_box(&src),
                    std::hint::black_box(&dst),
                    None,
                    true,
                ))
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_umeyama, bench_umeyama_robust);
criterion_main!(benches);
