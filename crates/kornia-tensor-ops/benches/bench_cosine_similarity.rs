use criterion::{black_box, criterion_group, criterion_main, Criterion};

use kornia_tensor::{CpuAllocator, Tensor};
use kornia_tensor_ops::ops::{cosine_similarity, cosine_similarity_optimized};
use rand::Rng;

fn bench_cosine_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_similarity");
    let mut rng = rand::rng();

    // Create random data for different data types
    let size = 1536;

    // Float32 data
    let a: Vec<f32> = (0..size).map(|_| rng.random::<f32>()).collect();
    let b: Vec<f32> = (0..size).map(|_| rng.random::<f32>()).collect();
    let a_tensor =
        Tensor::<f32, 1, CpuAllocator>::from_shape_slice([size], &a, CpuAllocator).unwrap();
    let b_tensor =
        Tensor::<f32, 1, CpuAllocator>::from_shape_slice([size], &b, CpuAllocator).unwrap();

    // Benchmark with float32
    group.bench_function("cosine_similarity", |bencher| {
        bencher.iter(|| black_box(cosine_similarity(&a_tensor, &b_tensor).unwrap()))
    });

    group.bench_function("cosine_similarity_optimized", |bencher| {
        bencher
            .iter(|| black_box(cosine_similarity_optimized(&a_tensor, &b_tensor).unwrap()))
    });

    group.finish();
}

criterion_group!(benches, bench_cosine_similarity);
criterion_main!(benches);
