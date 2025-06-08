/// Linear layer implementation.
///
/// Linear layer implemented using `matrixmultiply::sgemm`.
///
/// # Arguments
///
/// * `src` - Input tensor of shape `[B, T, D]`
/// * `weight` - Weight tensor of shape `[N, D]`
/// * `bias` - Bias tensor of shape `[N]`
/// * `dst` - Output tensor of shape `[B, T, N]`
/// * `batch_size` - Batch size
/// * `seq_len` - Sequence length
/// * `input_dim` - Input dimension
/// * `output_dim` - Output dimension
///
/// # Example
///
/// ```
/// use kornia_nn::linear::linear_layer_gemm;
///
/// let src = [[[1.0, 2.0, 3.0]]];
/// let weight = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]];
/// let bias = [0.1, 0.2];
///
/// let mut dst = [[[0.0, 0.0]]];
///
/// linear_layer_gemm(
///     src.as_flattened().as_flattened(),
///     weight.as_flattened(),
///     &bias,
///     dst.as_flattened_mut().as_flattened_mut(),
///     1,
///     1,
///     3,
///     2,
/// );
///
/// assert_eq!(dst, [[[1.5000001, 3.4]]]);
/// ```
#[allow(clippy::too_many_arguments)]
pub fn linear_layer_gemm(
    src: &[f32],     // Shape: [B, T, D] flattened
    weight: &[f32],  // Shape: [N, D] flattened (row-major format)
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
        "Input size mismatch",
    );
    assert_eq!(
        dst.len(),
        batch_size * seq_len * output_dim,
        "Output size mismatch",
    );
    assert_eq!(weight.len(), output_dim * input_dim, "Weight size mismatch");
    assert_eq!(bias.len(), output_dim, "Bias size mismatch");

    let m = batch_size * seq_len; // total rows
    let k = input_dim;
    let n = output_dim;

    // 1. Set bias for each output row
    for output_row in dst.chunks_exact_mut(output_dim) {
        output_row.copy_from_slice(bias);
    }

    // 2. Perform GEMM for the whole batch: dst = src * weight^T + bias
    //    (beta = 1.0, so GEMM adds to the bias-initialized output)
    unsafe {
        matrixmultiply::sgemm(
            /* m */ m,
            /* k */ k,
            /* n */ n,
            /* alpha */ 1.0,
            /* a */ src.as_ptr(),
            /* rsa */ k as isize,
            /* csa */ 1,
            /* b */ weight.as_ptr(),
            /* rsb */ 1,
            /* csb */ k as isize,
            /* beta */ 1.0,
            /* c */ dst.as_mut_ptr(),
            /* rsc */ n as isize,
            /* csc */ 1,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_linear_layer_simd_wide() {
        // [1, 1, 3] - batch=1, seq_len=1, input_dim=3
        let src = [[[1.0, 2.0, 3.0]]];
        // [2, 3] - 2 outputs, 3 inputs (correct format for weight[n][d] indexing)
        let weight = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]];
        // [2] - 2 output features
        let bias = [0.1, 0.2];

        let mut dst = [[[0.0, 0.0]]];

        linear_layer_gemm(
            src.as_flattened().as_flattened(),
            weight.as_flattened(),
            &bias,
            dst.as_flattened_mut().as_flattened_mut(),
            1,
            1,
            3,
            2,
        );

        // from pytorch
        let expected = [[[1.5, 3.4]]];
        for (actual, expected) in dst.iter().zip(expected.iter()) {
            for (actual_seq, expected_seq) in actual.iter().zip(expected.iter()) {
                for (a, e) in actual_seq.iter().zip(expected_seq.iter()) {
                    assert_relative_eq!(a, e);
                }
            }
        }
    }

    #[test]
    fn test_linear_layer_gemm_batch_size_2() {
        let src = [[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]];
        let weight = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]];
        let bias = [0.1, 0.2];

        let mut dst_gemm = [[[0.0, 0.0], [0.0, 0.0]]];
        linear_layer_gemm(
            src.as_flattened().as_flattened(),
            weight.as_flattened(),
            &bias,
            dst_gemm.as_flattened_mut().as_flattened_mut(),
            2,
            1,
            3,
            2,
        );

        // from pytorch
        let expected = [[[1.5, 3.4], [3.3, 7.9]]];
        for (actual, expected) in dst_gemm.iter().zip(expected.iter()) {
            for (actual_seq, expected_seq) in actual.iter().zip(expected.iter()) {
                for (a, e) in actual_seq.iter().zip(expected_seq.iter()) {
                    assert_relative_eq!(a, e);
                }
            }
        }
    }
}
