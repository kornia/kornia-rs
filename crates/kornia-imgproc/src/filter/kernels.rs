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

    #[test]
    fn test_box_blur_fast_kernels_1d() {
        assert_eq!(box_blur_fast_kernels_1d(0.5, 3), vec![1, 1, 1]);
        assert_eq!(box_blur_fast_kernels_1d(0.5, 4), vec![1, 1, 1, 1]);
        assert_eq!(box_blur_fast_kernels_1d(0.5, 5), vec![1, 1, 1, 1, 1]);

        assert_eq!(box_blur_fast_kernels_1d(1.0, 3), vec![1, 1, 3]);
        assert_eq!(box_blur_fast_kernels_1d(1.0, 5), vec![1, 1, 1, 1, 3]);
    }
}
