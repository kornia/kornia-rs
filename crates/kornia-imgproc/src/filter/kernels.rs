use kornia_image::ImageError;

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

/// Compute the derivative of a Gaussian kernel.
///
/// This calculates the first derivative of a Gaussian function: -x/σ² * exp(-x²/2σ²)
///
/// # Arguments
///
/// * `kernel_size` - The size of the kernel (must be odd).
/// * `sigma` - The standard deviation of the Gaussian.
///
/// # Returns
///
/// A 1D Gaussian derivative kernel with zero mean (no L1/L2 normalization).
fn gaussian_derivative_1d(kernel_size: usize, sigma: f32) -> Vec<f32> {
    // The derivative of a Gaussian is an odd function, so we construct the
    // kernel explicitly as anti-symmetric around a zero center tap. This
    // guarantees exact zero-mean without needing a post-hoc mean subtraction.
    let mut kernel = vec![0.0f32; kernel_size];
    let center = kernel_size / 2;
    let sigma_sq = sigma * sigma;

    for i in 1..=center {
        let x = i as f32;
        // Derivative of Gaussian: -x/σ² * exp(-x²/2σ²)
        let gauss = (-x * x / (2.0 * sigma_sq)).exp();
        let value = -x / sigma_sq * gauss;

        // Enforce strict anti-symmetry: f(-x) = -f(x), with center exactly zero
        kernel[center + i] = value;
        kernel[center - i] = -value;
    }

    kernel
}

/// Create a Sobel kernel for edge detection.
///
/// Returns separable 1D kernels for the derivative and smoothing directions.
///
/// # Kernel Types
/// - **Size 3**: Classic Sobel 3×3 kernel (exact match with standard definition)
/// - **Size 5**: Classic Sobel 5×5 kernel (exact match with standard definition)  
/// - **Size ≥ 7**: **Gaussian derivative approximation** (not classic Sobel)
///
/// # Arguments
///
/// * `kernel_size` - The size of the kernel (must be odd and ≥ 3).
///
/// # Returns
///
/// A tuple of two vectors: `(derivative_kernel, smoothing_kernel)`.
///
/// # Returns Errors
///
/// Returns `ImageError::InvalidKernelLength` if `kernel_size` is even or less than 3.
pub fn sobel_kernel_1d(kernel_size: usize) -> Result<(Vec<f32>, Vec<f32>), ImageError> {
    let (kernel_x, kernel_y) = match kernel_size {
        // Classic Sobel kernels (exact definitions)
        3 => (vec![-1.0, 0.0, 1.0], vec![1.0, 2.0, 1.0]),
        5 => (
            vec![-1.0, -2.0, 0.0, 2.0, 1.0],
            vec![1.0, 4.0, 6.0, 4.0, 1.0],
        ),

        // Gaussian derivative approximation for larger kernels
        _ if kernel_size % 2 == 1 && kernel_size >= 7 => {
            // Auto compute sigma using the 3-sigma rule
            let sigma = (kernel_size as f32 - 1.0) / 6.0;
            let smooth = gaussian_kernel_1d(kernel_size, sigma);
            let deriv = gaussian_derivative_1d(kernel_size, sigma);

            // Match the binomial scale pattern of classic Sobel kernels
            // Size 3: sum=2^2=4, Size 5: sum=2^4=16, Size 7: sum=2^6=64
            let scale = 2_f32.powi((kernel_size - 1) as i32);
            let scaled_smooth: Vec<f32> = smooth.into_iter().map(|v| v * scale).collect();

            (deriv, scaled_smooth)
        }

        _ => return Err(ImageError::InvalidKernelLength(kernel_size, kernel_size)),
    };
    Ok((kernel_x, kernel_y))
}

/// Create a normalized 2d sobel kernel.
///
/// # Arguments
///
/// * `kernel_size` - The size of the kernel.
///
/// # Returns
///
/// A tuple of two array of the kernel. (dx_kernel, dy_kernel)
pub fn normalized_sobel_kernel3() -> ([[f32; 3]; 3], [[f32; 3]; 3]) {
    (
        [
            [-0.125, 0.0, 0.125],
            [-0.25, 0.0, 0.25],
            [-0.125, 0.0, 0.125],
        ],
        [
            [-0.125, -0.25, -0.125],
            [0.0, 0.0, 0.0],
            [0.125, 0.25, 0.125],
        ],
    )
}

/// Create list of optimized box blur kernels based on gaussian sigma
///
/// <https://www.peterkovesi.com/papers/FastGaussianSmoothing.pdf>
/// # Arguments
///
/// * `sigma` = The sigma of the gaussian kernel
/// * `kernels` = The number of times the box blur kernels would be applied, ideally from 3-5
///
/// # Returns
///
/// A kernels-sized vector of the kernels.
pub fn box_blur_fast_kernels_1d(sigma: f32, kernels: u8) -> Vec<usize> {
    let n = kernels as f32;
    let ideal_size = (12.0 * sigma * sigma / n + 1.0).sqrt();
    let mut size_l = ideal_size.floor();
    size_l -= if size_l % 2.0 == 0.0 { 1.0 } else { 0.0 };
    let size_u = size_l + 2.0;

    let ideal_m = (12.0 * sigma * sigma - n * size_l * size_l - 4.0 * n * size_l - 3.0 * n)
        / (-4.0 * size_l - 4.0);
    let mut boxes = Vec::new();
    for i in 0..kernels {
        if i < ideal_m.round() as u8 {
            boxes.push(size_l as usize);
        } else {
            boxes.push(size_u as usize);
        }
    }
    boxes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sobel_kernel_1d() -> Result<(), ImageError> {
        let kernel = sobel_kernel_1d(3)?;
        assert_eq!(kernel.0, vec![-1.0, 0.0, 1.0]);
        assert_eq!(kernel.1, vec![1.0, 2.0, 1.0]);

        let kernel = sobel_kernel_1d(5)?;
        assert_eq!(kernel.0, vec![-1.0, -2.0, 0.0, 2.0, 1.0]);
        assert_eq!(kernel.1, vec![1.0, 4.0, 6.0, 4.0, 1.0]);

        // Test Gaussian derivative kernels (size 7)
        let (deriv7, smooth7) = sobel_kernel_1d(7)?;
        assert_eq!(deriv7.len(), 7);
        assert_eq!(smooth7.len(), 7);

        // Derivative should be anti-symmetric
        assert!((deriv7[0] + deriv7[6]).abs() < 1e-6);
        assert!((deriv7[1] + deriv7[5]).abs() < 1e-6);
        assert!((deriv7[2] + deriv7[4]).abs() < 1e-6);
        assert!((deriv7[3]).abs() < 1e-6); // Center should be ~0
        let deriv_sum: f32 = deriv7.iter().sum();
        assert!(
            deriv_sum.abs() < 1e-7,
            "Derivative kernel sum was {}, expected < 1e-7",
            deriv_sum
        );

        // Smoothing should be symmetric
        assert!((smooth7[0] - smooth7[6]).abs() < 1e-6);
        assert!((smooth7[1] - smooth7[5]).abs() < 1e-6);
        assert!((smooth7[2] - smooth7[4]).abs() < 1e-6);

        // Smoothing kernel should follow binomial (2^(size-1)) scaling pattern
        let smooth_sum: f32 = smooth7.iter().sum();
        let expected_sum = 2_f32.powi(7 - 1); // 2^6 = 64.0
        assert!(
            (smooth_sum - expected_sum).abs() < 1e-5,
            "Smoothing kernel sum was {}, expected {}",
            smooth_sum,
            expected_sum
        );

        Ok(())
    }

    #[test]
    fn test_sobel_kernel_1d_even_size() {
        // Even sizes should return error
        let result = sobel_kernel_1d(4);
        assert!(matches!(result, Err(ImageError::InvalidKernelLength(4, 4))));
    }

    #[test]
    fn test_sobel_kernel_1d_too_small() {
        // Size < 3 should return error
        let result = sobel_kernel_1d(1);
        assert!(matches!(result, Err(ImageError::InvalidKernelLength(1, 1))));
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

    #[test]
    fn test_box_blur_fast_kernels_1d() {
        assert_eq!(box_blur_fast_kernels_1d(0.5, 3), vec![1, 1, 1]);
        assert_eq!(box_blur_fast_kernels_1d(0.5, 4), vec![1, 1, 1, 1]);
        assert_eq!(box_blur_fast_kernels_1d(0.5, 5), vec![1, 1, 1, 1, 1]);

        assert_eq!(box_blur_fast_kernels_1d(1.0, 3), vec![1, 1, 3]);
        assert_eq!(box_blur_fast_kernels_1d(1.0, 5), vec![1, 1, 1, 1, 3]);
    }
}
