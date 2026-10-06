use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use kornia_imgproc::features::match_descriptors;

fn descriptors(count: usize, mut seed: u64) -> Vec<[u8; 32]> {
    (0..count)
        .map(|_| {
            let mut descriptor = [0; 32];
            for byte in &mut descriptor {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                *byte = seed as u8;
            }
            descriptor
        })
        .collect()
}

fn bench_hamming_match(c: &mut Criterion) {
    for threads in [1, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let mut group = c.benchmark_group(format!("hamming_match/{threads}_threads"));
        for count in [512, 1024, 4096] {
            let queries = descriptors(count, 0x1234_5678);
            let candidates = descriptors(count, 0x9876_5432);
            for cross_check in [false, true] {
                let direction = if cross_check { "mutual" } else { "forward" };
                group.bench_with_input(BenchmarkId::new(direction, count), &count, |b, _| {
                    pool.install(|| {
                        b.iter(|| {
                            std::hint::black_box(match_descriptors(
                                std::hint::black_box(&queries),
                                std::hint::black_box(&candidates),
                                None,
                                cross_check,
                                None,
                            ))
                        });
                    });
                });
            }
        }
        group.finish();
    }
}

criterion_group!(benches, bench_hamming_match);
criterion_main!(benches);
