use kornia::filters::gaussian_blur;
use kornia::geometry::transform::{resize, scale};
use kornia::tensor::Tensor;
use tch::{Device, Kind};

/// Builds a Gaussian pyramid using Kornia operators
pub fn build_pyramid(image: &Tensor, levels: usize) -> Vec<Tensor> {
    let mut pyramid = vec![image.clone()];

    for _ in 1..levels {
        // Apply Gaussian blur
        let blurred = gaussian_blur(&pyramid.last().unwrap(), &[3, 3], &[1.0, 1.0]);

        // Downsample using Kornia's resize
        let (height, width) = blurred.size2().unwrap();
        let downsampled = resize(&blurred, &[height / 2, width / 2], None, None);
        
        pyramid.push(downsampled);
    }

    pyramid
}
