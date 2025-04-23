//! Benchmarks for SmolVLM model implementations
//!
//! This benchmark compares the performance of different SmolVLM backends
//! (ONNX Runtime and Candle) on various hardware configurations.

use std::path::Path;
use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use kornia_models::smolvlm::{run_benchmark, SmolVLMModel, SmolVLMVariant};

#[cfg(all(feature = "onnx", feature = "candle"))]
use kornia_models::smolvlm::compare_backends;

/// Benchmark SmolVLM with the Candle backend
#[cfg(feature = "candle")]
fn bench_candle(c: &mut Criterion) {
    let mut group = c.benchmark_group("SmolVLM-Candle");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(10);

    let model_path = Path::new("models/smolvlm");
    let image_path = Path::new("test_image.jpg");
    let prompt = "Describe this image:";

    // Skip benchmark if model or image doesn't exist
    if !model_path.exists() || !image_path.exists() {
        println!("Skipping benchmark: model or test image not found");
        return;
    }

    // Benchmark small model on CPU
    group.bench_function("small-cpu", |b| {
        b.iter(|| {
            let result = run_benchmark(
                black_box(SmolVLMModel::Candle),
                black_box(SmolVLMVariant::Small),
                black_box(true), // use_cpu = true
                black_box(model_path),
                black_box(image_path),
                black_box(prompt),
            );
            black_box(result)
        })
    });

    // Benchmark small model on GPU (if available)
    if cfg!(target_os = "linux") && std::process::Command::new("nvidia-smi").status().is_ok() {
        group.bench_function("small-gpu", |b| {
            b.iter(|| {
                let result = run_benchmark(
                    black_box(SmolVLMModel::Candle),
                    black_box(SmolVLMVariant::Small),
                    black_box(false), // use_cpu = false
                    black_box(model_path),
                    black_box(image_path),
                    black_box(prompt),
                );
                black_box(result)
            })
        });
    }

    group.finish();
}

/// Benchmark SmolVLM with the ONNX backend
#[cfg(feature = "onnx")]
fn bench_onnx(c: &mut Criterion) {
    let mut group = c.benchmark_group("SmolVLM-ONNX");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(10);

    let model_path = Path::new("models/smolvlm");
    let image_path = Path::new("test_image.jpg");
    let prompt = "Describe this image:";

    // Skip benchmark if model or image doesn't exist
    if !model_path.exists() || !image_path.exists() {
        println!("Skipping benchmark: model or test image not found");
        return;
    }

    // Benchmark small model on CPU
    group.bench_function("small-cpu", |b| {
        b.iter(|| {
            let result = run_benchmark(
                black_box(SmolVLMModel::Onnx),
                black_box(SmolVLMVariant::Small),
                black_box(true), // use_cpu = true
                black_box(model_path),
                black_box(image_path),
                black_box(prompt),
            );
            black_box(result)
        })
    });

    // Benchmark small model on GPU (if available)
    if cfg!(target_os = "linux") && std::process::Command::new("nvidia-smi").status().is_ok() {
        group.bench_function("small-gpu", |b| {
            b.iter(|| {
                let result = run_benchmark(
                    black_box(SmolVLMModel::Onnx),
                    black_box(SmolVLMVariant::Small),
                    black_box(false), // use_cpu = false
                    black_box(model_path),
                    black_box(image_path),
                    black_box(prompt),
                );
                black_box(result)
            })
        });
    }

    group.finish();
}

/// Benchmark comparing both backends directly
#[cfg(all(feature = "onnx", feature = "candle"))]
fn bench_compare(c: &mut Criterion) {
    let mut group = c.benchmark_group("SmolVLM-Compare");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(10);

    let model_path = Path::new("models/smolvlm");
    let image_path = Path::new("test_image.jpg");
    let prompt = "Describe this image:";

    // Skip benchmark if model or image doesn't exist
    if !model_path.exists() || !image_path.exists() {
        println!("Skipping benchmark: model or test image not found");
        return;
    }

    // Compare backends on CPU
    group.bench_function("small-cpu", |b| {
        b.iter(|| {
            let result = compare_backends(
                black_box(SmolVLMVariant::Small),
                black_box(true), // use_cpu = true
                black_box(model_path),
                black_box(image_path),
                black_box(prompt),
            );
            black_box(result)
        })
    });

    group.finish();
}

#[cfg(feature = "candle")]
criterion_group!(benches_candle, bench_candle);

#[cfg(feature = "onnx")]
criterion_group!(benches_onnx, bench_onnx);

#[cfg(all(feature = "onnx", feature = "candle"))]
criterion_group!(benches_compare, bench_compare);

#[cfg(all(feature = "onnx", feature = "candle"))]
criterion_main!(benches_candle, benches_onnx, benches_compare);

#[cfg(all(feature = "candle", not(feature = "onnx")))]
criterion_main!(benches_candle);

#[cfg(all(feature = "onnx", not(feature = "candle")))]
criterion_main!(benches_onnx);

#[cfg(not(any(feature = "onnx", feature = "candle")))]
fn main() {
    println!("No backend features enabled. Enable 'onnx' or 'candle' features to run benchmarks.");
}
