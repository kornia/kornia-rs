use criterion::{criterion_group, criterion_main, Criterion};
#[cfg(feature = "mkl")]
use kornia_nn::linear::linear_layer_gemm;
use rand::random;
use std::hint::black_box;

/// Perform a linear layer operation on batched input in sequential manner.
/// Input shape: [B, T, D] -> Output shape: [B, T, N]
/// where B=batch_size, T=sequence_length, D=input_features, N=output_features
#[allow(clippy::too_many_arguments)]
pub fn linear_layer_sequential(
    src: &[f32],     // Shape: [B, T, D] flattened
    weight: &[f32],  // Shape: [N, D] flattened (transposed for efficiency)
    bias: &[f32],    // Shape: [N]
    dst: &mut [f32], // Shape: [B, T, N] flattened
    batch_size: usize,
    seq_len: usize,
    input_dim: usize,
    output_dim: usize,
) {
    assert_eq!(
        src.len(),
        batch_size * seq_len * input_dim,
        "Input size mismatch"
    );
    assert_eq!(
        dst.len(),
        batch_size * seq_len * output_dim,
        "Output size mismatch"
    );
    assert_eq!(weight.len(), output_dim * input_dim, "Weight size mismatch");
    assert_eq!(bias.len(), output_dim, "Bias size mismatch");

    for b in 0..batch_size {
        for t in 0..seq_len {
            for n in 0..output_dim {
                let mut sum = 0.0;
                for d in 0..input_dim {
                    sum += src[b * seq_len * input_dim + t * input_dim + d]
                        * weight[d * output_dim + n];
                }
                dst[b * seq_len * output_dim + t * output_dim + n] = sum + bias[n];
            }
        }
    }
}

/// SIMD-optimized linear layer using wide SIMD with sequential processing
#[allow(clippy::too_many_arguments)]
pub fn linear_layer_wide_simd(
    src: &[f32],     // Shape: [B, T, D] flattened
    weight: &[f32],  // Shape: [N, D] flattened (transposed for efficiency)
    bias: &[f32],    // Shape: [N]
    dst: &mut [f32], // Shape: [B, T, N] flattened
    seq_len: usize,
    input_dim: usize,
    output_dim: usize,
) {
    const LANES: usize = 4;

    dst.chunks_exact_mut(seq_len * output_dim)
        .zip(src.chunks_exact(seq_len * input_dim))
        .for_each(|(dst_batch, src_batch)| {
            dst_batch
                .chunks_exact_mut(output_dim)
                .zip(src_batch.chunks_exact(input_dim))
                .for_each(|(dst_timestep, src_timestep)| {
                    for n in 0..output_dim {
                        let weight_row = &weight[n * input_dim..(n + 1) * input_dim];
                        let mut sum_simd = wide::f32x4::splat(0.0);

                        src_timestep
                            .chunks_exact(LANES)
                            .zip(weight_row.chunks_exact(LANES))
                            .for_each(|(src_chunk, weight_chunk)| {
                                let src_vec = wide::f32x4::new(src_chunk.try_into().unwrap());
                                let weight_vec = wide::f32x4::new(weight_chunk.try_into().unwrap());
                                sum_simd = src_vec.mul_add(weight_vec, sum_simd);
                            });

                        // Handle remainder elements
                        let scalar_sum = src_timestep
                            .chunks_exact(LANES)
                            .remainder()
                            .iter()
                            .zip(weight_row.chunks_exact(LANES).remainder().iter())
                            .map(|(s, w)| s * w)
                            .sum::<f32>();

                        dst_timestep[n] = sum_simd.reduce_add() + scalar_sum + bias[n];
                    }
                });
        });
}

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

    group.bench_function("linear_layer_wide_simd", |bencher| {
        bencher.iter(|| {
            linear_layer_wide_simd(
                &src, &weight, &bias, &mut dst, SEQ_LEN, INPUT_DIM, OUTPUT_DIM,
            );
            black_box(());
        });
    });

    #[cfg(feature = "mkl")]
    group.bench_function("linear_layer_gemm", |bencher| {
        bencher.iter(|| {
            linear_layer_gemm(
                &src, &weight, &bias, &mut dst, BATCH_SIZE, SEQ_LEN, INPUT_DIM, OUTPUT_DIM,
            );
            black_box(());
        });
    });

    // candle
    use candle_nn::Module;
    let linear_layer = candle_nn::Linear::new(
        candle_core::Tensor::from_vec(
            weight.clone(),
            (INPUT_DIM, OUTPUT_DIM),
            &candle_core::Device::Cpu,
        )
        .unwrap(),
        Some(
            candle_core::Tensor::from_vec(bias.clone(), (OUTPUT_DIM,), &candle_core::Device::Cpu)
                .unwrap(),
        ),
    );
    let input = candle_core::Tensor::from_vec(
        src.clone(),
        (BATCH_SIZE, SEQ_LEN, INPUT_DIM),
        &candle_core::Device::Cpu,
    )
    .unwrap();
    group.bench_function("linear_layer_candle", |bencher| {
        bencher.iter(|| {
            black_box(linear_layer.forward(&input).unwrap());
        });
    });

    group.finish();
}

criterion_group!(benches, bench_linear_layer);
criterion_main!(benches);
