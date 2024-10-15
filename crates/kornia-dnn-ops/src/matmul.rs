/// Matrix multiplication forward pass
///
/// dst = bias + src * weight
///
/// b == batch size
/// c == in_features
/// n == out_features
pub fn matmul_forward_cpu(
    dst: &mut [f32],
    src: &[f32],
    weight: &[f32],
    bias: &[f32],
    b: usize,
    c: usize,
    n: usize,
) {
    for b in 0..b {
        for n in 0..n {
            for c in 0..c {
                dst[b * n * c + n * c + c] = bias[c] + src[b * c + c] * weight[c * n + n];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_forward_cpu() {
        let b = 1;
        let c = 2;
        let n = 3;

        let mut dst = vec![0.0; b * n * c];
        let src = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let weight = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let bias = vec![1.0, 2.0, 3.0, 4.0];

        matmul_forward_cpu(&mut dst, &src, &weight, &bias, b, c, n);

        assert_eq!(
            dst,
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0
            ]
        );
    }
}
