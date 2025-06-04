/// Perform a linear layer operation on batched input in sequential manner.
/// Input shape: [B, T, D] -> Output shape: [B, T, N]
/// where B=batch_size, T=sequence_length, D=input_features, N=output_features
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

use rayon::prelude::*;

/// SIMD-optimized linear layer using wide SIMD with iterator
pub fn linear_layer_iter_simd(
    src: &[f32],     // Shape: [B, T, D] flattened
    weight: &[f32],  // Shape: [N, D] flattened (transposed for efficiency)
    bias: &[f32],    // Shape: [N]
    dst: &mut [f32], // Shape: [B, T, N] flattened
    seq_len: usize,
    input_dim: usize,
    output_dim: usize,
) {
    const LANES: usize = 8;

    dst.par_chunks_exact_mut(seq_len * output_dim)
        .zip(src.par_chunks_exact(seq_len * input_dim))
        .for_each(|(dst_batch, src_batch)| {
            dst_batch
                .par_chunks_exact_mut(output_dim)
                .zip(src_batch.par_chunks_exact(input_dim))
                .for_each(|(dst_timestep, src_timestep)| {
                    for n in 0..output_dim {
                        let weight_row = &weight[n * input_dim..(n + 1) * input_dim];
                        let mut sum_simd = wide::f32x8::splat(0.0);

                        src_timestep
                            .chunks_exact(LANES)
                            .zip(weight_row.chunks_exact(LANES))
                            .for_each(|(src_chunk, weight_chunk)| {
                                let src_vec = wide::f32x8::new(src_chunk.try_into().unwrap());
                                let weight_vec = wide::f32x8::new(weight_chunk.try_into().unwrap());
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

/// GEMM-based linear layer using gemm crate
pub fn linear_layer_gemm(
    src: &[f32],
    weight: &[f32],
    bias: &[f32],
    dst: &mut [f32],
    batch_size: usize,
    seq_len: usize,
    input_dim: usize,
    output_dim: usize,
) {
    let m = batch_size * seq_len;
    let n = output_dim;
    let k = input_dim;

    // Initialize dst with bias using iterator
    dst.chunks_exact_mut(output_dim).for_each(|chunk| {
        chunk.iter_mut().zip(bias.iter()).for_each(|(d, &b)| *d = b);
    });

    unsafe {
        gemm::gemm(
            m, // Number of rows in src and dst
            n, // Number of columns in dst
            k, // Number of columns in src
            dst.as_mut_ptr(),
            n as isize, // dst row stride (row-major)
            1,          // dst column stride
            true,       // read_dst = true (we have bias)
            src.as_ptr(),
            k as isize, // src row stride (row-major)
            1,          // src column stride
            weight.as_ptr(),
            1,                  // weight row stride (treat as column-major)
            input_dim as isize, // weight column stride (treat as column-major)
            1.0,                // alpha = 1.0 (keep bias)
            1.0,                // beta = 1.0 (add matrix mult)
            false,
            false,
            false,
            gemm::Parallelism::None,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_linear_layer_gemm() {
        // [1, 1, 3] - batch=1, seq_len=1, input_dim=3
        //let src = [[[1.0, 2.0, 3.0]]];
        // [2, 3] - 2 outputs, 3 inputs (correct format for weight[n][d] indexing)
        //let weight = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]];
        // [2] - 2 output features
        //let bias = [0.1, 0.2];

        //let mut dst = [[[0.0, 0.0]]];

        let src: &[f32] = &[1.0, 2.0, 3.0]; // 1 row of 3 elements
        let weight: &[f32] = &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]; // 2 rows of 3 elements
        let bias: &[f32] = &[0.1, 0.2];
        let mut dst = [0.0, 0.0]; // 1 row of 2 output dims

        //linear_layer_iter_simd(
        linear_layer_gemm(src, weight, &bias, &mut dst, 1, 1, 3, 2);

        // from pytorch
        let expected = [1.5, 3.4];
        assert_relative_eq!(dst.as_slice(), expected.as_slice());
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

        linear_layer_iter_simd(
            src.as_flattened().as_flattened(),
            weight.as_flattened(),
            &bias,
            dst.as_flattened_mut().as_flattened_mut(),
            2,
            4,
            2,
        );

        // from pytorch
        let expected = [
            [[3.1, 7.2], [7.1, 17.6]],    // batch 0
            [[11.1, 28.0], [15.1, 38.4]], // batch 1
            [[19.1, 48.8], [23.1, 59.2]], // batch 2
        ];

        for (actual_batch, expected_batch) in dst.iter().zip(expected.iter()) {
            for (actual_seq, expected_seq) in actual_batch.iter().zip(expected_batch.iter()) {
                for (a, e) in actual_seq.iter().zip(expected_seq.iter()) {
                    assert_relative_eq!(a, e);
                }
            }
        }
    }
}
