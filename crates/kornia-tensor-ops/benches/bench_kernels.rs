//! Benchmark suite for tensor operation kernels.
//!
//! # Limitations & Design Decisions
//!
//! - **Integer types (i8, i16, etc.)**: Only floating-point (f32) benchmarks are included.
//!   Integer dot product operations can easily overflow during multiplication (e.g., 50 * 50 = 2500
//!   exceeds i8's max value of 127). While the kernel uses saturating arithmetic to handle
//!   overflow gracefully, benchmarking with random integer values is not representative of
//!   real-world use cases.
//!
//! - **i8 specifically excluded**: Dot product on i8 vectors isn't a realistic use case for
//!   image processing (which is kornia's primary domain). Image data typically uses f32
//!   or normalized u8 values, not raw signed integers.
//!
//! - **Future considerations**: If integer benchmarks are needed in the future, consider:
//!   1. Using controlled test data with known bounds
//!   2. Using larger integer types (i32, i64) with appropriate test ranges
//!   3. Benchmarking the saturating vs non-saturating behavior separately

use criterion::{criterion_group, criterion_main, Criterion};
use kornia_tensor_ops::kernels::{cosine_similarity_float_kernel, dot_product1_kernel};
use rand::Rng;

/// Benchmarks for the dot product1 kernel.
///
/// Only benchmarks f32 data type. Integer types (i8, i16, etc.) are intentionally excluded
/// because:
/// 1. Random integer multiplication easily causes overflow (e.g., 50*50 = 2500 > i8 max of 127)
/// 2. Dot product on signed integers isn't a realistic use case for image processing
/// 3. The kernel uses saturating arithmetic, making random-value benchmarks unrepresentative
///
/// See module-level documentation for more details on design decisions.
fn bench_product1_float_kernel(c: &mut Criterion) {
    let mut group = c.benchmark_group("product1_float_kernel");
    let mut rng = rand::rng();

    let test_sizes = vec![8, 128, 1024, 16384];

    for size in test_sizes.clone() {
        let a: Vec<f32> = (0..size).map(|_| rng.random::<f32>()).collect();
        let b: Vec<f32> = (0..size).map(|_| rng.random::<f32>()).collect();

        group.bench_function(format!("f32_size_{size}"), |bencher| {
            bencher.iter(|| std::hint::black_box(dot_product1_kernel(&a, &b).unwrap()))
        });
    }

    group.finish();
}

/// Benchmarks for the cosine similarity kernel.
///
/// Only benchmarks f32 data type. Integer types are excluded for the same reasons
/// as the dot product kernel - see module-level documentation for details.
///
/// Cosine similarity computes the cosine of the angle between two vectors, which is
/// primarily useful for floating-point data in image processing contexts.
fn bench_cosine_similarity_kernel(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_similarity_kernel");
    let mut rng = rand::rng();

    let test_sizes = vec![8, 128, 1024, 16384];

    for size in test_sizes.clone() {
        let a: Vec<f32> = (0..size).map(|_| rng.random::<f32>()).collect();
        let b: Vec<f32> = (0..size).map(|_| rng.random::<f32>()).collect();

        group.bench_function(format!("f32_size_{size}"), |bencher| {
            bencher.iter(|| std::hint::black_box(cosine_similarity_float_kernel(&a, &b).unwrap()))
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
