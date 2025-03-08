use criterion::{black_box, criterion_group, criterion_main, Criterion};

use kornia_tensor::{CpuAllocator, Tensor};
use kornia_tensor_ops::ops::cosine_similarity;
use rand::Rng;

fn bench_cosine_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_similarity");
    let mut rng = rand::rng();

    let test_sizes = vec![8, 128, 1024, 16384];

    for size in test_sizes {
        let a: Vec<f32> = (0..size).map(|_| rng.random::<f32>()).collect();
        let b: Vec<f32> = (0..size).map(|_| rng.random::<f32>()).collect();
        let a_tensor =
            Tensor::<f32, 1, CpuAllocator>::from_shape_slice([size], &a, CpuAllocator).unwrap();
        let b_tensor =
            Tensor::<f32, 1, CpuAllocator>::from_shape_slice([size], &b, CpuAllocator).unwrap();

        group.bench_function(&format!("f32_size_{}", size), |bencher| {
            bencher.iter(|| black_box(cosine_similarity(&a_tensor, &b_tensor).unwrap()))
        });
    }

    group.finish();
}

criterion_group!(benches, bench_cosine_similarity);
criterion_main!(benches);
