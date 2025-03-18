use criterion::{black_box, criterion_group, criterion_main, Criterion};

use kornia_tensor::{CpuAllocator, Tensor};
use kornia_tensor_ops::TensorOps;
use rand::Rng;

fn bench_dot_product1(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_product1");
    let mut rng = rand::rng();

    let test_sizes = vec![8, 128, 1024, 16384];

    for size in test_sizes.clone() {
        let a: Vec<f32> = (0..size).map(|_| rng.random::<f32>()).collect();
        let b: Vec<f32> = (0..size).map(|_| rng.random::<f32>()).collect();
        let a_tensor =
            Tensor::<f32, 1, CpuAllocator>::from_shape_slice([size], &a, CpuAllocator).unwrap();
        let b_tensor =
            Tensor::<f32, 1, CpuAllocator>::from_shape_slice([size], &b, CpuAllocator).unwrap();

        group.bench_function(format!("f32_size_{}", size), |bencher| {
            bencher.iter(|| {
                black_box(
                    Tensor::<f32, 1, CpuAllocator>::dot_product1(&a_tensor, &b_tensor).unwrap(),
                )
            })
        });
    }

    for size in test_sizes.clone() {
        let a: Vec<i8> = (0..size).map(|_| rng.random::<i8>()).collect();
        let b: Vec<i8> = (0..size).map(|_| rng.random::<i8>()).collect();
        let a_tensor =
            Tensor::<i8, 1, CpuAllocator>::from_shape_slice([size], &a, CpuAllocator).unwrap();
        let b_tensor =
            Tensor::<i8, 1, CpuAllocator>::from_shape_slice([size], &b, CpuAllocator).unwrap();

        group.bench_function(format!("i8_size_{}", size), |bencher| {
            bencher.iter(|| {
                black_box(
                    Tensor::<i8, 1, CpuAllocator>::dot_product1(&a_tensor, &b_tensor).unwrap(),
                )
            })
        });
    }

    group.finish();
}

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

        group.bench_function(format!("f32_size_{}", size), |bencher| {
            bencher.iter(|| black_box(Tensor::cosine_similarity(&a_tensor, &b_tensor).unwrap()))
        });
    }

    group.finish();
}

criterion_group!(benches, bench_dot_product1, bench_cosine_similarity);
criterion_main!(benches);
