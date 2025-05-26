use criterion::{black_box, criterion_group, criterion_main, Criterion};
use kornia_tensor_ops::dnn::{
    linear_layer_iter_output_flat, linear_layer_iter_output_flat_parallel,
    linear_layer_sequential_flat,
};
use rand::random;

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

    group.bench_function("linear_layer_iter_output_flat", |bencher| {
        bencher.iter(|| {
            black_box(linear_layer_iter_output_flat::<INPUT_DIM, OUTPUT_DIM>(
                &src, &weight, &bias, &mut dst, BATCH_SIZE, SEQ_LEN,
            ));
        });
    });

    group.bench_function("linear_layer_iter_output_flat_parallel", |bencher| {
        bencher.iter(|| {
            black_box(linear_layer_iter_output_flat_parallel::<
                INPUT_DIM,
                OUTPUT_DIM,
            >(
                &src, &weight, &bias, &mut dst, BATCH_SIZE, SEQ_LEN
            ));
        });
    });

    group.bench_function("linear_layer_sequential_flat", |bencher| {
        bencher.iter(|| {
            black_box(linear_layer_sequential_flat::<INPUT_DIM, OUTPUT_DIM>(
                &src, &weight, &bias, &mut dst, BATCH_SIZE, SEQ_LEN,
            ));
        });
    });

    group.finish();
}

criterion_group!(benches, bench_linear_layer);
criterion_main!(benches);
