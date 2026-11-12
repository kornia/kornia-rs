//! End-to-end `umeyama_f64` benchmark over point counts and weight modes.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use kornia_algebra::linalg::rigid::umeyama_f64;
use kornia_algebra::Vec3F64;

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

criterion_group!(benches, bench_umeyama);
criterion_main!(benches);
