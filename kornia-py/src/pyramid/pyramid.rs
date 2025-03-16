use image::{GrayImage, Luma};
use imageproc::filter::gaussian_blur_f32;

/// Build a Gaussian pyramid with multiple image scales.
pub fn build_gaussian_pyramid(image: &GrayImage, levels: usize, sigma: f32) -> Vec<GrayImage> {
    let mut pyramid = Vec::new();
    let mut current_img = image.clone();

    for _ in 0..levels {
        // Apply Gaussian blur
        let blurred = gaussian_blur_f32(&current_img, sigma);

        // Downsample (reduce resolution)
        let width = blurred.width() / 2;
        let height = blurred.height() / 2;
        let downsampled = image::imageops::resize(
            &blurred, width, height, image::imageops::FilterType::Triangle,
        );

        pyramid.push(downsampled.clone());
        current_img = downsampled;
    }

    pyramid
}
