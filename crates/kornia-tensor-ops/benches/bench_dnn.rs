use criterion::{criterion_group, criterion_main, Criterion};
use kornia_tensor_ops::dnn::{
    linear_layer_gemm, linear_layer_sequential, linear_layer_simd_sequential,
};
use rand::random;
use std::hint::black_box;

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

    group.bench_function("linear_layer_sequential", |bencher| {
        bencher.iter(|| {
            linear_layer_sequential(
                &src, &weight, &bias, &mut dst, BATCH_SIZE, SEQ_LEN, INPUT_DIM, OUTPUT_DIM,
            );
            black_box(());
        });
    });

    group.bench_function("linear_layer_simd_sequential", |bencher| {
        bencher.iter(|| {
            linear_layer_simd_sequential(
                &src, &weight, &bias, &mut dst, SEQ_LEN, INPUT_DIM, OUTPUT_DIM,
            );
            black_box(());
        });
    });

    group.bench_function("linear_layer_gemm", |bencher| {
        bencher.iter(|| {
            linear_layer_gemm(
                &src, &weight, &bias, &mut dst, BATCH_SIZE, SEQ_LEN, INPUT_DIM, OUTPUT_DIM,
            );
            black_box(());
        });
    });

    // candle
    //use candle_nn::Module;
    //let linear_layer = candle_nn::Linear::new(
    //    candle_core::Tensor::from_vec(
    //        weight.clone(),
    //        (OUTPUT_DIM, INPUT_DIM),
    //        &candle_core::Device::Cpu,
    //    )
    //    .unwrap(),
    //    Some(
    //        candle_core::Tensor::from_vec(bias.clone(), (OUTPUT_DIM,), &candle_core::Device::Cpu)
    //            .unwrap(),
    //    ),
    //);
    //let input = candle_core::Tensor::from_vec(
    //    src.clone(),
    //    (BATCH_SIZE, SEQ_LEN, INPUT_DIM),
    //    &candle_core::Device::Cpu,
    //)
    //.unwrap();
    //group.bench_function("linear_layer_candle", |bencher| {
    //    bencher.iter(|| {
    //        black_box(linear_layer.forward(&input).unwrap());
    //    });
    //});

    group.finish();
}

criterion_group!(benches, bench_linear_layer);
criterion_main!(benches);
