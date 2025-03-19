use crate::interpolation::InterpolationMode;
use crate::resize::resize_native;
use kornia_image::{Image, ImageError, ImageSize};

/// Adjusts the brightness of a grayscale (1-channel) image so that its mean brightness
/// matches the given target value.
///
/// This function downsizes the image to quickly estimate its average brightness,
/// computes the difference from the target mean, and then applies that offset to every pixel.
/// All operations are performed on Kornia‑rs’s native `Image<f32, 1>` type.
///
/// # Arguments
///
/// * `src` - The source image (grayscale, with f32 pixel values).
/// * `target_mean` - The desired mean brightness (e.g., 128.0).
///
/// # Returns
///
/// A new image with adjusted brightness, or an `ImageError` if an error occurs.
pub fn adjust_brightness(
    src: &Image<f32, 1>,
    target_mean: f32,
) -> Result<Image<f32, 1>, ImageError> {
    let src_size = src.size();

    // downscaling the image for faster brightness calculation.
    let new_width = 100;
    let new_height = src_size.height * new_width / src_size.width;
    let dst_size = ImageSize {
        width: new_width,
        height: new_height,
    };

    // creating an empty image and passing to `resize_native`.
    let mut small = Image::<f32, 1>::from_size_val(dst_size, 0.0)?;
    resize_native(src, &mut small, InterpolationMode::Nearest)?;

    // computing the average brightness.
    let sum: f32 = small.as_slice().iter().sum();
    let avg_brightness = sum / (new_width * new_height) as f32;

    // offset the brightness to reach target mean.
    let offset = target_mean - avg_brightness;
    let adjusted_data: Vec<f32> = src
        .as_slice()
        .iter()
        .map(|&p| (p + offset).clamp(0.0, 255.0))
        .collect();

    // create adjusted image.
    let adjusted = Image::<f32, 1>::new(src_size, adjusted_data)?;

    Ok(adjusted)
}

#[cfg(test)]

mod tests {
    use super::*;
    use kornia_image::{Image, ImageSize};

    /// testing module with on an image with all pixels constant.
    #[test]
    fn test_adjust_brightness_constant() -> Result<(), ImageError> {
        // 10x10 grayscale image with all pixels set to 80.
        let width = 10;
        let height = 10;
        let size = ImageSize { width, height };
        let constant_val: f32 = 80.0;
        let data = vec![constant_val; width * height];
        let image = Image::<f32, 1>::new(size, data)?;

        // target brightness and adjust image
        let target_val: f32 = 120.0;
        let adjusted = adjust_brightness(&image, target_val)?;

        // checking if function working properly
        let sum: f32 = adjusted.as_slice().iter().sum();

        let avg = sum / (width * height) as f32;

        assert!(
            (avg - target_val as f32).abs() < 1.0,
            "Average brightness {avg} is not close to target {target_val}."
        );

        Ok(())
    }

    /// testing module on image with non-uniform pixel values.
    #[test]
    fn test_adjust_brightness_non_uniform() -> Result<(), ImageError> {
        // 10 x 10 grayscale image with non-uniform pixel values.
        let width = 10;
        let height = 10;
        let size = ImageSize { width, height };
        let data: Vec<f32> = (0..(width * height)).map(|x| (x % 256) as f32).collect();
        let image = Image::<f32, 1>::new(size, data)?;

        // computing current average brightness.
        let current_avg: f32 = image.as_slice().iter().sum::<f32>() / (width * height) as f32;

        // target brightness and adjust image
        let target = (current_avg + 30.0).min(255.0);
        let adjusted = adjust_brightness(&image, target)?;

        // compute average brightness of adjusted image
        let adjusted_avg: f32 = adjusted.as_slice().iter().sum::<f32>() / (width * height) as f32;

        assert!(
            (adjusted_avg - target as f32).abs() < 1.0,
            "New average brightness {adjusted_avg} is not clost to target {target}",
        );

        Ok(())
    }
}
