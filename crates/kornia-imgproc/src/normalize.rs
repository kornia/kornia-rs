//! Image normalization operations for preprocessing and standardization.
//!
//! This module provides utilities for normalizing image pixel values, a critical
//! preprocessing step for machine learning models and computer vision algorithms.
//!
//! # Normalization Methods
//!
//! * **Mean-Std Normalization** ([`normalize_mean_std`]) - Z-score normalization
//! * **Min-Max Normalization** ([`normalize_min_max`]) - Scale to [0, 1] range
//!
//! # Use Cases
//!
//! * **Deep Learning** - Standardize inputs to neural networks (e.g., ImageNet normalization)
//! * **Image Comparison** - Reduce lighting variations before similarity metrics
//! * **Histogram Equalization** - Improve contrast by normalizing intensity distribution
//!
//! # Example: ImageNet-style Normalization
//!
//! ```
//! use kornia_image::{Image, ImageSize};
//! use kornia_imgproc::normalize::normalize_mean_std;
//!
//! let image = Image::<f32, 3>::from_size_val(
//!     ImageSize { width: 224, height: 224 },
//!     0.5,
//! ).unwrap();
//!
//! let mut normalized = Image::<f32, 3>::from_size_val(image.size(), 0.0).unwrap();
//!
//! // ImageNet mean and std per channel (RGB)
//! let mean = [0.485, 0.456, 0.406];
//! let std = [0.229, 0.224, 0.225];
//!
//! normalize_mean_std(&image, &mut normalized, &mean, &std).unwrap();
//! ```
//!
//! # Mathematical Background
//!
//! **Z-score normalization** transforms each pixel to have zero mean and unit variance:
//!
//! ```text
//! normalized = (pixel - μ) / σ
//! ```
//!
//! where μ is the mean and σ is the standard deviation for each channel.
//!
//! # See also
//!
//! * Common ImageNet normalization values widely used in pretrained models
//! * [`crate::enhance`] for other image adjustment operations

use num_traits::Float;

use kornia_image::{allocator::ImageAllocator, Image, ImageError};

use crate::parallel;

/// Normalize an image using per-channel mean and standard deviation (Z-score normalization).
///
/// Applies the transformation `(pixel - μ) / σ` independently to each color channel,
/// producing a normalized image with zero mean and unit standard deviation. This is
/// the standard preprocessing step for many deep learning models.
///
/// # Arguments
///
/// * `src` - The input image with shape (H, W, C).
/// * `dst` - The output normalized image with shape (H, W, C).
/// * `mean` - Array of mean values, one per channel (μ₁, μ₂, ..., μ_C).
/// * `std` - Array of standard deviation values, one per channel (σ₁, σ₂, ..., σ_C).
///
/// # Example: ImageNet Normalization
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_imgproc::normalize::normalize_mean_std;
///
/// let image = Image::<f32, 3>::from_size_val(
///     ImageSize { width: 224, height: 224 },
///     0.5,
/// ).unwrap();
///
/// let mut normalized = Image::<f32, 3>::from_size_val(image.size(), 0.0).unwrap();
///
/// // Standard ImageNet RGB normalization
/// let mean = [0.485, 0.456, 0.406];
/// let std = [0.229, 0.224, 0.225];
///
/// normalize_mean_std(&image, &mut normalized, &mean, &std).unwrap();
/// ```
///
/// # Example: Grayscale Normalization
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_imgproc::normalize::normalize_mean_std;
///
/// let image = Image::<f32, 1>::from_size_val(
///     ImageSize { width: 100, height: 100 },
///     128.0,
/// ).unwrap();
///
/// let mut normalized = Image::<f32, 1>::from_size_val(image.size(), 0.0).unwrap();
///
/// normalize_mean_std(&image, &mut normalized, &[127.5], &[50.0]).unwrap();
/// ```
///
/// # Common Precomputed Values
///
/// * **ImageNet (RGB)**: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
/// * **CIFAR-10 (RGB)**: mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
/// * **MNIST (Grayscale)**: mean=[0.1307], std=[0.3081]
///
/// # Performance
///
/// This function is parallelized using Rayon for efficient processing of large images.
///
/// # Errors
///
/// Returns [`ImageError::InvalidImageSize`] if `src` and `dst` have different dimensions.
///
/// # See also
///
/// * [`normalize_min_max`] for scaling to a specific range instead
/// * Use this before feeding images to pretrained neural networks
pub fn normalize_mean_std<T, const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<T, C, A1>,
    dst: &mut Image<T, C, A2>,
    mean: &[T; C],
    std: &[T; C],
) -> Result<(), ImageError>
where
    T: Send + Sync + Float,
{
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }

    parallel::par_iter_rows(src, dst, |src_pixel, dst_pixel| {
        src_pixel
            .iter()
            .zip(dst_pixel.iter_mut())
            .zip(mean.iter())
            .zip(std.iter())
            .for_each(|(((&src_val, dst_val), &mean_val), &std_val)| {
                *dst_val = (src_val - mean_val) / std_val;
            });
    });

    Ok(())
}

/// Find the minimum and maximum values in an image.
///
/// # Arguments
///
/// * `src` - The input image of shape (height, width, channels).
/// * `dst` - The output image of shape (height, width, channels).
///
/// # Returns
///
/// A tuple containing the minimum and maximum values in the image.
///
/// # Errors
///
/// If the image data is not initialized, an error is returned.
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_image::allocator::CpuAllocator;
/// use kornia_imgproc::normalize::find_min_max;
///
/// let image_data = vec![0u8, 1, 0, 1, 2, 3, 0, 1, 0, 1, 2, 3];
/// let image = Image::<u8, 3, _>::new(
///   ImageSize {
///     width: 2,
///     height: 2,
///   },
///   image_data,
///   CpuAllocator
/// )
/// .unwrap();
///
/// let (min, max) = find_min_max(&image).unwrap();
/// assert_eq!(min, 0);
/// assert_eq!(max, 3);
/// ```
pub fn find_min_max<T, const C: usize, A: ImageAllocator>(
    image: &Image<T, C, A>,
) -> Result<(T, T), ImageError>
where
    T: Clone + Copy + PartialOrd,
{
    // get the first element in the image
    let first_element = match image.as_slice().iter().next() {
        Some(x) => x,
        None => return Err(ImageError::ImageDataNotInitialized),
    };

    let mut min = first_element;
    let mut max = first_element;

    for x in image.as_slice().iter() {
        if x < min {
            min = x;
        }
        if x > max {
            max = x;
        }
    }

    Ok((*min, *max))
}

/// Normalize an image using the minimum and maximum values.
///
/// The formula for normalizing an image is:
///
/// (image - min) * (max - min) / (max_val - min_val) + min
///
/// Each channel is normalized independently.
///
/// # Arguments
///
/// * `src` - The input image of shape (height, width, channels).
/// * `dst` - The output image of shape (height, width, channels).
/// * `min` - The minimum value for each channel.
/// * `max` - The maximum value for each channel.
///
/// # Returns
///
/// The normalized image of shape (height, width, C).
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_image::allocator::CpuAllocator;
/// use kornia_imgproc::normalize::normalize_min_max;
///
/// let image_data = vec![0.0f32, 1.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 0.0, 1.0, 2.0, 3.0];
/// let image = Image::<f32, 3, _>::new(
///   ImageSize {
///     width: 2,
///     height: 2,
///   },
///   image_data,
///   CpuAllocator
/// )
/// .unwrap();
///
/// let mut image_normalized = Image::<f32, 3, _>::from_size_val(image.size(), 0.0, CpuAllocator).unwrap();
///
/// normalize_min_max(&image, &mut image_normalized, 0.0, 1.0).unwrap();
///
/// assert_eq!(image_normalized.num_channels(), 3);
/// assert_eq!(image_normalized.size().width, 2);
/// assert_eq!(image_normalized.size().height, 2);
/// ```
pub fn normalize_min_max<T, const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<T, C, A1>,
    dst: &mut Image<T, C, A2>,
    min: T,
    max: T,
) -> Result<(), ImageError>
where
    T: Send + Sync + Float,
{
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }

    let (min_val, max_val) = find_min_max(src)?;

    parallel::par_iter_rows(src, dst, |src_pixel, dst_pixel| {
        src_pixel
            .iter()
            .zip(dst_pixel.iter_mut())
            .for_each(|(&src_val, dst_val)| {
                *dst_val = (src_val - min_val) * (max - min) / (max_val - min_val) + min;
            });
    });

    Ok(())
}

#[cfg(test)]
mod tests {
    use kornia_image::{Image, ImageError, ImageSize};
    use kornia_tensor::CpuAllocator;

    #[test]
    fn normalize_mean_std() -> Result<(), ImageError> {
        let image_data = vec![
            0.0f32, 1.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 0.0, 1.0, 2.0, 3.0,
        ];

        let image_expected = [
            -0.5f32, 0.0, -0.5, 0.5, 1.0, 2.5, -0.5, 0.0, -0.5, 0.5, 1.0, 2.5,
        ];

        let image = Image::<f32, 3, _>::new(
            ImageSize {
                width: 2,
                height: 2,
            },
            image_data,
            CpuAllocator,
        )?;

        let mean = [0.5, 1.0, 0.5];
        let std = [1.0, 1.0, 1.0];

        let mut normalized = Image::<f32, 3, _>::from_size_val(image.size(), 0.0, CpuAllocator)?;

        super::normalize_mean_std(&image, &mut normalized, &mean, &std)?;

        assert_eq!(normalized.num_channels(), 3);
        assert_eq!(normalized.size().width, 2);
        assert_eq!(normalized.size().height, 2);

        normalized
            .as_slice()
            .iter()
            .zip(image_expected.iter())
            .for_each(|(a, b)| {
                assert!((a - b).abs() < 1e-6);
            });

        Ok(())
    }

    #[test]
    fn find_min_max() -> Result<(), ImageError> {
        let image_data = vec![0u8, 1, 0, 1, 2, 3, 0, 1, 0, 1, 2, 3];
        let image = Image::<u8, 3, _>::new(
            ImageSize {
                width: 2,
                height: 2,
            },
            image_data,
            CpuAllocator,
        )?;

        let (min, max) = super::find_min_max(&image)?;

        assert_eq!(min, 0);
        assert_eq!(max, 3);

        Ok(())
    }

    #[test]
    fn normalize_min_max() -> Result<(), ImageError> {
        let image_data = vec![
            0.0f32, 1.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 0.0, 1.0, 2.0, 3.0,
        ];

        let image_expected = [
            0.0f32, 0.33333334, 0.0, 0.33333334, 0.6666667, 1.0, 0.0, 0.33333334, 0.0, 0.33333334,
            0.6666667, 1.0,
        ];

        let image = Image::<f32, 3, _>::new(
            ImageSize {
                width: 2,
                height: 2,
            },
            image_data,
            CpuAllocator,
        )?;

        let mut normalized = Image::<f32, 3, _>::from_size_val(image.size(), 0.0, CpuAllocator)?;

        super::normalize_min_max(&image, &mut normalized, 0.0, 1.0)?;

        assert_eq!(normalized.num_channels(), 3);
        assert_eq!(normalized.size().width, 2);
        assert_eq!(normalized.size().height, 2);

        normalized
            .as_slice()
            .iter()
            .zip(image_expected.iter())
            .for_each(|(a, b)| {
                assert!((a - b).abs() < 1e-6);
            });

        Ok(())
    }
}
