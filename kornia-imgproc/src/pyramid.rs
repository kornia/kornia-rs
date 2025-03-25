use kornia::filters::gaussian_blur;
use kornia::geometry::transform::{resize};
use kornia::tensor::Tensor;
use tch::{Device, Kind};

/// Builds a Gaussian pyramid using Kornia operators
pub fn build_pyramid(image: &Tensor, levels: usize) -> Vec<Tensor> {
    let mut pyramid = vec![image.shallow_clone()];

    for _ in 1..levels {
        let blurred = gaussian_blur(&pyramid.last().unwrap(), &[5, 5], &[1.5, 1.5]);
        let scaled = resize(&blurred, &[blurred.size()[2] / 2, blurred.size()[3] / 2]);
        pyramid.push(scaled);
    }

    pyramid
}
