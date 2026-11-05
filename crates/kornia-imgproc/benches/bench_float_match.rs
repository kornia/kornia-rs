use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use kornia_imgproc::features::{
    match_descriptors_f32, match_smnn_f32, sift_match_descriptors, sift_match_descriptors_scalar,
};

const SIFT_DIM: usize = 128;
const COSINE_DIM: usize = 64;

fn descriptors(rows: usize, dim: usize, mut seed: u64) -> Vec<f32> {
    (0..rows * dim)
        .map(|_| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) % 256) as f32
        })
        .collect()
}

fn unit_rows<const D: usize>(flat: &[f32]) -> Vec<[f32; D]> {
    flat.as_chunks::<D>()
        .0
        .iter()
        .map(|row| {
            let norm = row
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
                .sqrt()
                .max(f32::MIN_POSITIVE);
            row.map(|x| x / norm)
        })
        .collect()
}

fn bench_float_match(c: &mut Criterion) {
    for threads in [1, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let mut group = c.benchmark_group(format!("float_match/{threads}_threads"));
        group.sample_size(10);
        for count in [512, 1024, 4096] {
            let sift1 = descriptors(count, SIFT_DIM, 1);
            let sift2 = descriptors(count, SIFT_DIM, 2);
            group.bench_with_input(BenchmarkId::new("match_smnn_f32", count), &count, |b, _| {
                b.iter(|| {
                    pool.install(|| {
                        black_box(match_smnn_f32(
                            black_box(&sift1),
                            black_box(&sift2),
                            SIFT_DIM,
                            0.95,
                        ))
                    })
                })
            });
            group.bench_with_input(
                BenchmarkId::new("sift_cross_check", count),
                &count,
                |b, _| {
                    b.iter(|| {
                        pool.install(|| {
                            black_box(sift_match_descriptors(
                                black_box(&sift1),
                                count,
                                black_box(&sift2),
                                count,
                                0.95,
                                true,
                            ))
                        })
                    })
                },
            );
            if count <= 1024 {
                group.bench_with_input(
                    BenchmarkId::new("sift_cross_check_scalar", count),
                    &count,
                    |b, _| {
                        b.iter(|| {
                            pool.install(|| {
                                black_box(sift_match_descriptors_scalar(
                                    black_box(&sift1),
                                    count,
                                    black_box(&sift2),
                                    count,
                                    0.95,
                                    true,
                                ))
                            })
                        })
                    },
                );
            }
            let cosine1 = unit_rows::<COSINE_DIM>(&descriptors(count, COSINE_DIM, 3));
            let cosine2 = unit_rows::<COSINE_DIM>(&descriptors(count, COSINE_DIM, 4));
            group.bench_with_input(
                BenchmarkId::new("cosine_mutual_ratio", count),
                &count,
                |b, _| {
                    b.iter(|| {
                        pool.install(|| {
                            black_box(match_descriptors_f32(
                                black_box(&cosine1),
                                black_box(&cosine2),
                                None,
                                true,
                                Some(0.95),
                            ))
                        })
                    })
                },
            );
        }
        group.finish();
    }
}

criterion_group!(benches, bench_float_match);
criterion_main!(benches);
