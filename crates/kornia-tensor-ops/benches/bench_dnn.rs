use candle_core::{Device, Module, Tensor};
use candle_nn::Linear;
use criterion::{criterion_group, criterion_main, Criterion};
use kornia_tensor_ops::dnn::{linear_layer_gemm, linear_layer_iter_simd, linear_layer_sequential};
use rand::random;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

fn bench_linear_layer(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear_layer");

    // Large sizes similar to Gemma transformer dimensions
    // batch_size 32, seq_len 512, input_dim 3072 (Gemma 7B), output_dim 3072
    const BATCH_SIZE: usize = 2;
    const SEQ_LEN: usize = 32;
    const INPUT_DIM: usize = 64;
    const OUTPUT_DIM: usize = 64;

    // Use Vec for heap allocation to avoid stack overflow
    let src: Vec<f32> = (0..BATCH_SIZE * SEQ_LEN * INPUT_DIM)
        .map(|_| random::<f32>())
        .collect();

    let weight: Vec<f32> = (0..OUTPUT_DIM * INPUT_DIM)
        .map(|_| random::<f32>())
        .collect();

    let bias: Vec<f32> = (0..OUTPUT_DIM).map(|_| random::<f32>()).collect();

    let mut dst: Vec<f32> = vec![0.0; BATCH_SIZE * SEQ_LEN * OUTPUT_DIM];

    group.bench_function("linear_layer_iter_sequential", |bencher| {
        bencher.iter(|| {
            std::hint::black_box(linear_layer_sequential(
                &src, &weight, &bias, &mut dst, BATCH_SIZE, SEQ_LEN, INPUT_DIM, OUTPUT_DIM,
            ));
        });
    });

    group.bench_function("linear_layer_iter_simd", |bencher| {
        bencher.iter(|| {
            std::hint::black_box(linear_layer_iter_simd(
                &src, &weight, &bias, &mut dst, SEQ_LEN, INPUT_DIM, OUTPUT_DIM,
            ));
        });
    });

    // Add Candle linear layer benchmark
    let device = Device::Cpu;

    // Create input tensor
    let src_tensor =
        Tensor::from_vec(src.clone(), (BATCH_SIZE, SEQ_LEN, INPUT_DIM), &device).unwrap();

    // Create weight and bias tensors for initialization
    let weight_tensor = Tensor::from_vec(weight.clone(), (OUTPUT_DIM, INPUT_DIM), &device).unwrap();
    let bias_tensor = Tensor::from_vec(bias.clone(), OUTPUT_DIM, &device).unwrap();

    // Create linear layer with our weights and bias
    let linear = Linear::new(weight_tensor, Some(bias_tensor));

    group.bench_function("linear_layer_candle", |bencher| {
        bencher.iter(|| {
            std::hint::black_box(linear.forward(&src_tensor).unwrap());
        });
    });

    group.bench_function("linear_layer_gemm", |bencher| {
        bencher.iter(|| {
            std::hint::black_box(linear_layer_gemm(
                &src, &weight, &bias, &mut dst, BATCH_SIZE, SEQ_LEN, INPUT_DIM, OUTPUT_DIM,
            ));
        });
    });

    group.finish();
}

criterion_group!(benches, bench_linear_layer);
criterion_main!(benches);
