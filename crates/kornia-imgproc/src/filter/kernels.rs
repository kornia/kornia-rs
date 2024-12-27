/// Create a box blur kernel.
///
/// # Arguments
///
/// * `kernel_size` - The size of the kernel.
///
/// # Returns
///
/// A vector of the kernel.
pub fn box_blur_kernel_1d(kernel_size: usize) -> Vec<f32> {
    let kernel = vec![1.0 / kernel_size as f32; kernel_size];
    kernel
}

/// Create a gaussian blur kernel.
///
/// # Arguments
///
/// * `kernel_size` - The size of the kernel.
/// * `sigma` - The sigma of the gaussian kernel.
///
/// # Returns
///
/// A vector of the kernel.
pub fn gaussian_kernel_1d(kernel_size: usize, sigma: f32) -> Vec<f32> {
    let mut kernel = Vec::with_capacity(kernel_size);

    let mean = (kernel_size - 1) as f32 / 2.0;
    let sigma_sq = sigma * sigma;

    // compute the kernel
    for i in 0..kernel_size {
        let x = i as f32 - mean;
        kernel.push((-(x * x) / (2.0 * sigma_sq)).exp());
    }

    // normalize the kernel
    let norm = kernel.iter().sum::<f32>();
    kernel.iter_mut().for_each(|k| *k /= norm);
    kernel
}

/// Create a sobel kernel.
///
/// # Arguments
///
/// * `kernel_size` - The size of the kernel.
///
/// # Returns
///
/// A vector of the kernel.
pub fn sobel_kernel_1d(kernel_size: usize) -> (Vec<f32>, Vec<f32>) {
    let (kernel_x, kernel_y) = match kernel_size {
        3 => (vec![-1.0, 0.0, 1.0], vec![1.0, 2.0, 1.0]),
        5 => (
            vec![1.0, 4.0, 6.0, 4.0, 1.0],
            vec![-1.0, -2.0, 0.0, 2.0, 1.0],
        ),
        _ => panic!("Invalid kernel size for sobel kernel"),
    };
    (kernel_x, kernel_y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sobel_kernel_1d() {
        let kernel = sobel_kernel_1d(3);
        assert_eq!(kernel.0, vec![-1.0, 0.0, 1.0]);
        assert_eq!(kernel.1, vec![1.0, 2.0, 1.0]);

        let kernel = sobel_kernel_1d(5);
        assert_eq!(kernel.0, vec![1.0, 4.0, 6.0, 4.0, 1.0]);
        assert_eq!(kernel.1, vec![-1.0, -2.0, 0.0, 2.0, 1.0]);
    }

    #[test]
    fn test_gaussian_kernel_1d() {
        let kernel = gaussian_kernel_1d(5, 0.5);

        let expected = [
            0.00026386508,
            0.10645077,
            0.78657067,
            0.10645077,
            0.00026386508,
        ];

        for (i, &k) in kernel.iter().enumerate() {
            assert_eq!(k, expected[i]);
        }
    }
}
