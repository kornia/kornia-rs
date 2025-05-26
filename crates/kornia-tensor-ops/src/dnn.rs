use rayon::prelude::*;

#[cfg(feature = "portable_simd")]
use std::simd::f32x8;
#[cfg(feature = "portable_simd")]
use std::simd::num::SimdFloat;

/// Perform a linear layer operation.
pub fn linear_layer_sequential<const D: usize, const N: usize, const B: usize, const T: usize>(
    src: &[[[f32; D]; T]; B],
    weight: &[[f32; N]; D],
    bias: &[f32; N],
    dst: &mut [[[f32; N]; T]; B],
) {
    for b in 0..B {
        for t in 0..T {
            for n in 0..N {
                let mut sum = 0.0;
                for d in 0..D {
                    sum += src[b][t][d] * weight[d][n];
                }
                dst[b][t][n] = sum + bias[n];
            }
        }
    }
}

/// Perform a linear layer operation on batched input in sequential manner.
/// Input shape: [B, T, D] -> Output shape: [B, T, N]
/// where B=batch_size, T=sequence_length, D=input_features, N=output_features
pub fn linear_layer_sequential_flat<const D: usize, const N: usize>(
    src: &[f32],     // Shape: [B, T, D] flattened
    weight: &[f32],  // Shape: [N, D] flattened (transposed for efficiency)
    bias: &[f32],    // Shape: [N]
    dst: &mut [f32], // Shape: [B, T, N] flattened
    batch_size: usize,
    seq_len: usize,
) {
    for b in 0..batch_size {
        for t in 0..seq_len {
            for n in 0..N {
                let mut sum = 0.0;
                for d in 0..D {
                    sum += src[b * seq_len * D + t * D + d] * weight[d * N + n];
                }
                dst[b * seq_len * N + t * N + n] = sum + bias[n];
            }
        }
    }
}

/// Perform a linear layer operation on batched input.
/// Input shape: [B, T, D] -> Output shape: [B, T, N]
/// where B=batch_size, T=sequence_length, D=input_features, N=output_features
pub fn linear_layer_iter_output_flat<const D: usize, const N: usize>(
    src: &[f32],     // Shape: [B, T, D] flattened
    weight: &[f32],  // Shape: [N, D] flattened (transposed for efficiency)
    bias: &[f32],    // Shape: [N]
    dst: &mut [f32], // Shape: [B, T, N] flattened
    batch_size: usize,
    seq_len: usize,
) {
    // Validate input dimensions
    assert_eq!(src.len(), batch_size * seq_len * D, "Input size mismatch");
    assert_eq!(dst.len(), batch_size * seq_len * N, "Output size mismatch");
    assert_eq!(weight.len(), N * D, "Weight size mismatch");
    assert_eq!(bias.len(), N, "Bias size mismatch");

    // Process each batch
    dst.chunks_exact_mut(seq_len * N) // Each batch has seq_len * N outputs
        .zip(src.chunks_exact(seq_len * D)) // Each batch has seq_len * D inputs
        .for_each(|(dst_batch, src_batch)| {
            // Process each timestep in the sequence
            dst_batch
                .chunks_exact_mut(N) // Each timestep has N outputs
                .zip(src_batch.chunks_exact(D)) // Each timestep has D inputs
                .for_each(|(dst_timestep, src_timestep)| {
                    // Compute output for each feature
                    dst_timestep
                        .iter_mut()
                        .zip(weight.chunks_exact(D))
                        .zip(bias.iter())
                        .for_each(|((dst_val, weight_row), bias_val)| {
                            let mut sum = 0.0;
                            for i in 0..D {
                                sum += src_timestep[i] * weight_row[i];
                            }
                            *dst_val = sum + bias_val;
                        });
                });
        });
}

/// Perform a linear layer operation on batched input in parallel.
/// Input shape: [B, T, D] -> Output shape: [B, T, N]
/// where B=batch_size, T=sequence_length, D=input_features, N=output_features
pub fn linear_layer_iter_output_flat_parallel<const D: usize, const N: usize>(
    src: &[f32],
    weight: &[f32],
    bias: &[f32],
    dst: &mut [f32],
    batch_size: usize,
    seq_len: usize,
) {
    // Parallel over batches
    dst.par_chunks_exact_mut(seq_len * N)
        .zip(src.par_chunks_exact(seq_len * D))
        .for_each(|(dst_batch, src_batch)| {
            dst_batch
                .par_chunks_exact_mut(N)
                .zip(src_batch.par_chunks_exact(D))
                .for_each(|(dst_timestep, src_timestep)| {
                    dst_timestep
                        .par_iter_mut()
                        .zip(weight.par_chunks_exact(D))
                        .zip(bias.par_iter())
                        .for_each(|((dst_val, weight_row), bias_val)| {
                            let mut sum = 0.0;
                            for i in 0..D {
                                sum += src_timestep[i] * weight_row[i];
                            }
                            *dst_val = sum + bias_val;
                        });
                });
        });
}

/// SIMD-optimized linear layer using portable SIMD
#[cfg(feature = "portable_simd")]
pub fn linear_layer_simd<const D: usize, const N: usize>(
    src: &[f32],
    weight: &[f32],
    bias: &[f32],
    dst: &mut [f32],
    batch_size: usize,
    seq_len: usize,
) {
    const LANES: usize = 8;

    for b in 0..batch_size {
        for t in 0..seq_len {
            for n in 0..N {
                let mut sum_simd = f32x8::splat(0.0);
                let mut scalar_sum = 0.0;

                let src_base = b * seq_len * D + t * D;
                let weight_base = n * D;

                // Process 8 elements at a time
                let simd_chunks = D / LANES;
                for chunk in 0..simd_chunks {
                    let offset = chunk * LANES;

                    let src_vec = f32x8::from_slice(&src[src_base + offset..]);
                    let weight_vec = f32x8::from_slice(&weight[weight_base + offset..]);

                    sum_simd += src_vec * weight_vec;
                }

                // Handle remaining elements
                for d in (simd_chunks * LANES)..D {
                    scalar_sum += src[src_base + d] * weight[weight_base + d];
                }

                dst[b * seq_len * N + t * N + n] = sum_simd.reduce_sum() + scalar_sum + bias[n];
            }
        }
    }
}

/// Parallel processing with SIMD optimization - FIXED
#[cfg(feature = "portable_simd")]
pub fn linear_layer_parallel_simd<const D: usize, const N: usize>(
    src: &[f32],
    weight: &[f32],
    bias: &[f32],
    dst: &mut [f32],
    batch_size: usize,
    seq_len: usize,
) {
    const LANES: usize = 8;

    // Parallel over batches
    dst.par_chunks_exact_mut(seq_len * N)
        .zip(src.par_chunks_exact(seq_len * D))
        .for_each(|(dst_batch, src_batch)| {
            // Sequential over timesteps within each batch
            for t in 0..seq_len {
                let src_timestep = &src_batch[t * D..(t + 1) * D];
                let dst_timestep = &mut dst_batch[t * N..(t + 1) * N];

                // Sequential over output features (or could be parallel if N is large)
                for n in 0..N {
                    let mut sum_simd = f32x8::splat(0.0);
                    let weight_row = &weight[n * D..(n + 1) * D];

                    let simd_chunks = D / LANES;
                    for chunk in 0..simd_chunks {
                        let offset = chunk * LANES;
                        let src_vec = f32x8::from_slice(&src_timestep[offset..]);
                        let weight_vec = f32x8::from_slice(&weight_row[offset..]);
                        sum_simd += src_vec * weight_vec;
                    }

                    let mut scalar_sum = 0.0;
                    for d in (simd_chunks * LANES)..D {
                        scalar_sum += src_timestep[d] * weight_row[d];
                    }

                    dst_timestep[n] = sum_simd.reduce_sum() + scalar_sum + bias[n];
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_linear_layer_sequential() {
        // [1, 1, 3] - batch=1, seq_len=1, input_dim=3
        let src = [[[1.0, 2.0, 3.0]]];
        // [3, 2] - 3 inputs, 2 outputs (transposed to match weight[n][d] indexing)
        let weight = [[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]];
        // [2] - 2 output features
        let bias = [0.1, 0.2];

        let mut dst = [[[0.0, 0.0]]];
        linear_layer_sequential(&src, &weight, &bias, &mut dst);

        let expected = [[[1.5, 3.4]]];
        for (actual, expected) in dst.iter().zip(expected.iter()) {
            for (actual_seq, expected_seq) in actual.iter().zip(expected.iter()) {
                for (a, e) in actual_seq.iter().zip(expected_seq.iter()) {
                    assert_relative_eq!(a, e, epsilon = 1e-6);
                }
            }
        }
    }

    #[test]
    fn test_linear_layer_iter_output_flat_batch_seq_len_2() {
        // [3, 2, 4] - batch=3, seq_len=2, input_dim=4
        let src = [
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
            [[9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]],
            [[17.0, 18.0, 19.0, 20.0], [21.0, 22.0, 23.0, 24.0]],
        ];
        // [2, 4] - 2 outputs, 4 inputs
        let weight = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]];
        // [2] - 2 output features
        let bias = [0.1, 0.2];
        // [3, 2, 2] - batch=3, seq_len=2, output_dim=2
        let mut dst = [
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ];

        linear_layer_iter_output_flat::<4, 2>(
            src.as_flattened().as_flattened(),
            weight.as_flattened(),
            &bias,
            dst.as_flattened_mut().as_flattened_mut(),
            3,
            2,
        );

        // Calculate expected values manually for verification
        let expected = [
            [[3.1, 7.2], [7.1, 17.6]],    // batch 0
            [[11.1, 28.0], [15.1, 38.4]], // batch 1
            [[19.1, 48.8], [23.1, 59.2]], // batch 2
        ];

        for (actual_batch, expected_batch) in dst.iter().zip(expected.iter()) {
            for (actual_seq, expected_seq) in actual_batch.iter().zip(expected_batch.iter()) {
                for (a, e) in actual_seq.iter().zip(expected_seq.iter()) {
                    assert_relative_eq!(a, e, epsilon = 1e-6);
                }
            }
        }
    }
}
