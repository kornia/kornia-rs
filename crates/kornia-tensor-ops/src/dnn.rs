/// Perform a linear layer operation.
///
/// # Arguments
///
/// * `src` - The input tensor.
/// * `weight` - The weight tensor.
/// * `bias` - The bias tensor.
/// * `dst` - The output tensor.
///
/// # Example
///
/// ```
/// use kornia_tensor_ops::dnn::linear_layer;
///
/// let src = [1.0, 2.0, 3.0];
/// let weight = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]];
/// let bias = [0.1, 0.2];
/// let mut dst = [0.0, 0.0];
///
/// linear_layer(&src, &weight, &bias, &mut dst);
///
pub fn linear_layer_sequential<const IN: usize, const OUT: usize>(
    src: &[f32; IN],
    weight: &[[f32; IN]; OUT],
    bias: &[f32; OUT],
    dst: &mut [f32; OUT],
) {
    for i in 0..OUT {
        let mut sum = 0.0;
        for j in 0..IN {
            sum += src[j] * weight[i][j];
        }
        dst[i] = sum + bias[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_linear_layer_sequential() {
        let src = [1.0, 2.0, 3.0];
        let weight = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]];
        let bias = [0.1, 0.2];

        let mut dst = [0.0, 0.0];
        linear_layer_sequential(&src, &weight, &bias, &mut dst);

        let expected = [1.5, 3.4];
        for (x, y) in dst.iter().zip(expected.iter()) {
            assert_relative_eq!(x, y, epsilon = 1e-6);
        }
    }
}
