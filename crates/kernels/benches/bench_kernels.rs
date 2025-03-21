use criterion::{black_box, criterion_group, criterion_main, Criterion};
use kernels::ops::{cosine_similarity_float_kernel, dot_product1_kernel};
use rand::Rng;

fn bench_product1_float_kernel(c: &mut Criterion) {
    let mut group = c.benchmark_group("product1_float_kernel");
    let mut rng = rand::rng();

    let test_sizes = vec![8, 128, 1024, 16384];

    for size in test_sizes.clone() {
        let a: Vec<f32> = (0..size).map(|_| rng.random::<f32>()).collect();
        let b: Vec<f32> = (0..size).map(|_| rng.random::<f32>()).collect();

        group.bench_function(format!("f32_size_{}", size), |bencher| {
            bencher.iter(|| black_box(dot_product1_kernel(&a, &b).unwrap()))
        });
    }

    for size in test_sizes.clone() {
        let a: Vec<i8> = (0..size).map(|_| rng.random::<i8>()).collect();
        let b: Vec<i8> = (0..size).map(|_| rng.random::<i8>()).collect();

        group.bench_function(format!("i8_size_{}", size), |bencher| {
            bencher.iter(|| black_box(dot_product1_kernel(&a, &b).unwrap()))
        });
    }

    group.finish();
}

fn bench_cosine_similarity_kernel(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_similarity_kernel");
    let mut rng = rand::rng();

    let test_sizes = vec![8, 128, 1024, 16384];

    for size in test_sizes.clone() {
        let a: Vec<f32> = (0..size).map(|_| rng.random::<f32>()).collect();
        let b: Vec<f32> = (0..size).map(|_| rng.random::<f32>()).collect();

        group.bench_function(format!("f32_size_{}", size), |bencher| {
            bencher.iter(|| black_box(cosine_similarity_float_kernel(&a, &b).unwrap()))
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_product1_float_kernel,
    bench_cosine_similarity_kernel
);
criterion_main!(benches);
