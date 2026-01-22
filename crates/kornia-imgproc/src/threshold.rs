use num_traits::Zero;
use std::cmp::PartialOrd;

use kornia_image::{allocator::ImageAllocator, Image, ImageError};

use crate::parallel;
use crate::histogram::compute_histogram;

/// Apply a binary threshold to an image.
///
/// # Arguments
///
/// * `src` - The input image of an arbitrary number of channels and type.
/// * `dst` - The output image of an arbitrary number of channels and type.
/// * `threshold` - The threshold value. Must be the same type as the image.
/// * `max_value` - The maximum value to use when the input value is greater than the threshold.
///
/// # Returns
///
/// The thresholded image with the same number of channels as the input image.
///
/// # Examples
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_image::allocator::CpuAllocator;
/// use kornia_imgproc::threshold::threshold_binary;
///
/// let data = vec![100u8, 200, 50, 150, 200, 250];
/// let image = Image::<_, 1, _>::new(ImageSize { width: 2, height: 3 }, data, CpuAllocator).unwrap();
///
/// let mut thresholded = Image::<_, 1, _>::from_size_val(image.size(), 0, CpuAllocator).unwrap();
///
/// threshold_binary(&image, &mut thresholded, 100, 255).unwrap();
/// assert_eq!(thresholded.num_channels(), 1);
/// assert_eq!(thresholded.size().width, 2);
/// assert_eq!(thresholded.size().height, 3);
/// ```
pub fn threshold_binary<T, const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<T, C, A1>,
    dst: &mut Image<T, C, A2>,
    threshold: T,
    max_value: T,
) -> Result<(), ImageError>
where
    T: Copy + Send + Sync + PartialOrd + Zero,
{
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }

    // run the thresholding operation in parallel
    parallel::par_iter_rows_val(src, dst, |src_pixel, dst_pixel| {
        *dst_pixel = if *src_pixel > threshold {
            max_value
        } else {
            T::zero()
        };
    });

    Ok(())
}

/// Apply an inverse binary threshold to an image.
///
/// # Arguments
///
/// * `src` - The input image of an arbitrary number of channels and type.
/// * `dst` - The output image of an arbitrary number of channels and type.
/// * `threshold` - The threshold value. Must be the same type as the image.
/// * `max_value` - The maximum value to use when the input value is less than the threshold.
///
/// # Returns
///
/// The thresholded image with the same number of channels as the input image.
///
/// # Examples
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_image::allocator::CpuAllocator;
/// use kornia_imgproc::threshold::threshold_binary_inverse;
///
/// let data = vec![100u8, 200, 50, 150, 200, 250];
/// let image = Image::<_, 1, _>::new(ImageSize { width: 2, height: 3 }, data, CpuAllocator).unwrap();
///
/// let mut thresholded = Image::<_, 1, _>::from_size_val(image.size(), 0, CpuAllocator).unwrap();
///
/// threshold_binary_inverse(&image, &mut thresholded, 100, 255).unwrap();
/// assert_eq!(thresholded.num_channels(), 1);
/// assert_eq!(thresholded.size().width, 2);
/// assert_eq!(thresholded.size().height, 3);
/// ```
pub fn threshold_binary_inverse<T, const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<T, C, A1>,
    dst: &mut Image<T, C, A2>,
    threshold: T,
    max_value: T,
) -> Result<(), ImageError>
where
    T: Copy + Send + Sync + PartialOrd + Zero,
{
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }

    // run the thresholding operation in parallel
    parallel::par_iter_rows_val(src, dst, |src_pixel, dst_pixel| {
        *dst_pixel = if *src_pixel > threshold {
            T::zero()
        } else {
            max_value
        };
    });

    Ok(())
}

/// Apply a truncated threshold to an image.
///
/// # Arguments
///
/// * `src` - The input image of an arbitrary number of channels and type.
/// * `threshold` - The threshold value. Must be the same type as the image.
///
/// # Returns
///
/// The thresholded image with the same number of channels as the input image.
///
/// # Examples
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_image::allocator::CpuAllocator;
/// use kornia_imgproc::threshold::threshold_truncate;
///
/// let data = vec![100u8, 200, 50, 150, 200, 250];
/// let image = Image::<_, 1, _>::new(ImageSize { width: 2, height: 3 }, data, CpuAllocator).unwrap();
///
/// let mut thresholded = Image::<_, 1, _>::from_size_val(image.size(), 0, CpuAllocator).unwrap();
///
/// threshold_truncate(&image, &mut thresholded, 150).unwrap();
/// assert_eq!(thresholded.num_channels(), 1);
/// assert_eq!(thresholded.size().width, 2);
/// assert_eq!(thresholded.size().height, 3);
/// ```
pub fn threshold_truncate<T, const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<T, C, A1>,
    dst: &mut Image<T, C, A2>,
    threshold: T,
) -> Result<(), ImageError>
where
    T: Copy + Send + Sync + PartialOrd + Zero,
{
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }

    // run the thresholding operation in parallel
    parallel::par_iter_rows_val(src, dst, |src_pixel, dst_pixel| {
        *dst_pixel = if *src_pixel > threshold {
            threshold
        } else {
            *src_pixel
        };
    });

    Ok(())
}

/// Apply a threshold to an image, setting values below the threshold to zero.
///
/// # Arguments
///
/// * `src` - The input image of an arbitrary number of channels and type.
/// * `threshold` - The threshold value. Must be the same type as the image.
///
/// # Returns
///
/// The thresholded image with the same number of channels as the input image.
///
/// # Examples
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_image::allocator::CpuAllocator;
/// use kornia_imgproc::threshold::threshold_to_zero;
///
/// let data = vec![100u8, 200, 50, 150, 200, 250];
/// let image = Image::<_, 3, _>::new(ImageSize { width: 2, height: 1 }, data, CpuAllocator).unwrap();
///
/// let mut thresholded = Image::<_, 3, _>::from_size_val(image.size(), 0, CpuAllocator).unwrap();
///
/// threshold_to_zero(&image, &mut thresholded, 150).unwrap();
/// assert_eq!(thresholded.num_channels(), 3);
/// assert_eq!(thresholded.size().width, 2);
/// assert_eq!(thresholded.size().height, 1);
/// ```
pub fn threshold_to_zero<T, const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<T, C, A1>,
    dst: &mut Image<T, C, A2>,
    threshold: T,
) -> Result<(), ImageError>
where
    T: Copy + Send + Sync + PartialOrd + Zero,
{
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }

    // run the thresholding operation in parallel
    parallel::par_iter_rows_val(src, dst, |src_pixel, dst_pixel| {
        *dst_pixel = if *src_pixel > threshold {
            *src_pixel
        } else {
            T::zero()
        };
    });

    Ok(())
}

/// Apply a threshold to an image, setting values above the threshold to zero.
///
/// # Arguments
///
/// * `src` - The input image of an arbitrary number of channels and type.
/// * `threshold` - The threshold value. Must be the same type as the image.
///
/// # Returns
///
/// The thresholded image with the same number of channels as the input image.
///
/// # Examples
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_image::allocator::CpuAllocator;
/// use kornia_imgproc::threshold::threshold_to_zero_inverse;
///
/// let data = vec![100u8, 200, 50, 150, 200, 250];
/// let image = Image::<_, 3, _>::new(ImageSize { width: 2, height: 1 }, data, CpuAllocator).unwrap();
///
/// let mut thresholded = Image::<_, 3, _>::from_size_val(image.size(), 0, CpuAllocator).unwrap();
///
/// threshold_to_zero_inverse(&image, &mut thresholded, 150).unwrap();
/// assert_eq!(thresholded.num_channels(), 3);
/// assert_eq!(thresholded.size().width, 2);
/// assert_eq!(thresholded.size().height, 1);
/// ```
pub fn threshold_to_zero_inverse<T, const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<T, C, A1>,
    dst: &mut Image<T, C, A2>,
    threshold: T,
) -> Result<(), ImageError>
where
    T: Copy + Send + Sync + PartialOrd + Zero,
{
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }

    // run the thresholding operation in parallel
    parallel::par_iter_rows_val(src, dst, |src_pixel, dst_pixel| {
        *dst_pixel = if *src_pixel > threshold {
            T::zero()
        } else {
            *src_pixel
        };
    });

    Ok(())
}

/// Apply a range threshold to an image.
///
/// # Arguments
///
/// * `src` - The input image of an arbitrary number of channels and type.
/// * `lower_bound` - The lower bound for each channel.
/// * `upper_bound` - The upper bound for each channel.
///
/// # Returns
///
/// The thresholded image with a single channel as byte values.
///
/// Precondition: the input image must have the same number of channels as the lower and upper bounds.
/// Precondition: the input image range must be 0-255.
///
/// # Examples
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_image::allocator::CpuAllocator;
/// use kornia_imgproc::threshold::in_range;
///
/// let data = vec![100u8, 200, 50, 150, 200, 250];
///
/// let image = Image::<u8, 3, _>::new(
///    ImageSize {
///       width: 2,
///       height: 1,
///    },
///    data,
///    CpuAllocator
/// )
/// .unwrap();
///
/// let mut thresholded = Image::<u8, 1, _>::from_size_val(image.size(), 0, CpuAllocator).unwrap();
///
/// in_range(&image, &mut thresholded, &[100, 150, 0], &[200, 200, 200]).unwrap();
/// assert_eq!(thresholded.num_channels(), 1);
/// assert_eq!(thresholded.size().width, 2);
///
/// assert_eq!(thresholded.get_pixel(0, 0, 0).unwrap(), &255);
/// assert_eq!(thresholded.get_pixel(1, 0, 0).unwrap(), &0);
/// ```
pub fn in_range<T, const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<T, C, A1>,
    dst: &mut Image<u8, 1, A2>,
    lower_bound: &[T; C],
    upper_bound: &[T; C],
) -> Result<(), ImageError>
where
    T: Clone + Send + Sync + PartialOrd + Zero,
{
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }

    // parallelize the operation by rows
    parallel::par_iter_rows(src, dst, |src_pixel, dst_pixel| {
        let mut is_in_range = true;
        src_pixel
            .iter()
            .zip(lower_bound.iter().zip(upper_bound.iter()))
            .for_each(|(src_val, (lower, upper))| {
                is_in_range &= src_val >= lower && src_val <= upper;
            });
        dst_pixel[0] = if is_in_range { 255 } else { 0 };
    });

    Ok(())
}

/// Compute Otsu's threshold for a 1-channel u8 image.
///
/// Otsu's method automatically determines an optimal threshold value that minimizes
/// the within-class variance of the resulting binary image. This is a global thresholding
/// method suitable for images with a bimodal histogram.
///
/// # Arguments
///
/// * `src` - The input 1-channel u8 image.
/// * `dst` - The output 1-channel u8 binary image.
/// * `max_value` - The value to assign to pixels that are >= threshold (typically 255).
///
/// # Returns
///
/// The computed optimal threshold value as a u8.
///
/// # Errors
///
/// Returns `ImageError::InvalidImageSize` if source and destination image sizes don't match.
///
/// # Examples
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_image::allocator::CpuAllocator;
/// use kornia_imgproc::threshold::otsu_threshold;
///
/// let data = vec![50u8, 100, 150, 200, 50, 100, 150, 200];
/// let image = Image::<_, 1, _>::new(
///     ImageSize { width: 4, height: 2 },
///     data,
///     CpuAllocator
/// ).unwrap();
///
/// let mut binary = Image::<_, 1, _>::from_size_val(image.size(), 0, CpuAllocator).unwrap();
///
/// let threshold = otsu_threshold(&image, &mut binary, 255).unwrap();
/// assert_eq!(binary.num_channels(), 1);
/// assert_eq!(binary.size().width, 4);
/// assert_eq!(binary.size().height, 2);
/// println!("Otsu threshold: {}", threshold);
/// ```
pub fn otsu_threshold<A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<u8, 1, A1>,
    dst: &mut Image<u8, 1, A2>,
    max_value: u8,
) -> Result<u8, ImageError> {
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }

    // 1) Compute histogram
    let mut hist = vec![0usize; 256];
    compute_histogram(src, &mut hist, 256)?;

    let total: f64 = hist.iter().sum::<usize>() as f64;

    // 2) Compute probability and cumulative values
    let mut prob = [0f64; 256];
    let mut cum_prob = [0f64; 256];
    let mut cum_mean = [0f64; 256];

    for i in 0..256 {
        prob[i] = hist[i] as f64 / total;
        cum_prob[i] = prob[i] + if i > 0 { cum_prob[i - 1] } else { 0.0 };
        cum_mean[i] = (i as f64) * prob[i] + if i > 0 { cum_mean[i - 1] } else { 0.0 };
    }

    let global_mean = cum_mean[255];

    // 3) Find threshold that maximizes between-class variance
    let mut best_t = 1;  // Start from 1 to avoid edge case at 0
    let mut best_var = -1.0;

    for t in 1..256 {
        let w0 = cum_prob[t - 1];
        let w1 = 1.0 - w0;

        if w0 < 1e-6 || w1 < 1e-6 {
            continue;
        }

        let m0 = cum_mean[t - 1] / w0;
        let m1 = (global_mean - cum_mean[t - 1]) / w1;

        let between_var = w0 * w1 * (m0 - m1).powi(2);

        if between_var > best_var {
            best_var = between_var;
            best_t = t;
        }
    }

    // 4) Apply threshold to create binary image
    let threshold = best_t as u8;
    dst.as_slice_mut()
        .iter_mut()
        .zip(src.as_slice().iter())
        .for_each(|(d, &s)| {
            *d = if s >= threshold { max_value } else { 0 };
        });

    Ok(threshold)
}

// TODO: triangle

#[cfg(test)]
mod tests {
    use kornia_image::{Image, ImageError, ImageSize};
    use kornia_tensor::CpuAllocator;

    #[test]
    fn threshold_binary() -> Result<(), ImageError> {
        let data = vec![100u8, 200, 50, 150, 200, 250];
        let data_expected = [0u8, 255, 0, 255, 255, 255];
        let image = Image::<_, 1, _>::new(
            ImageSize {
                width: 2,
                height: 3,
            },
            data,
            CpuAllocator,
        )?;

        let mut thresholded = Image::<_, 1, _>::from_size_val(image.size(), 0, CpuAllocator)?;

        super::threshold_binary(&image, &mut thresholded, 100, 255)?;

        assert_eq!(thresholded.num_channels(), 1);
        assert_eq!(thresholded.size().width, 2);
        assert_eq!(thresholded.size().height, 3);

        assert_eq!(thresholded.as_slice(), data_expected);

        Ok(())
    }

    #[test]
    fn threshold_binary_inverse() -> Result<(), ImageError> {
        let data = vec![100u8, 200, 50, 150, 200, 250];
        let data_expected = [255u8, 0, 255, 0, 0, 0];
        let image = Image::<_, 1, _>::new(
            ImageSize {
                width: 2,
                height: 3,
            },
            data,
            CpuAllocator,
        )?;

        let mut thresholded = Image::<_, 1, _>::from_size_val(image.size(), 0, CpuAllocator)?;

        super::threshold_binary_inverse(&image, &mut thresholded, 100, 255)?;

        assert_eq!(thresholded.num_channels(), 1);
        assert_eq!(thresholded.size().width, 2);
        assert_eq!(thresholded.size().height, 3);

        assert_eq!(thresholded.as_slice(), data_expected);

        Ok(())
    }

    #[test]
    fn threshold_truncate() -> Result<(), ImageError> {
        let data = vec![100u8, 200, 50, 150, 200, 250];
        let data_expected = [100u8, 150, 50, 150, 150, 150];
        let image = Image::<_, 1, _>::new(
            ImageSize {
                width: 2,
                height: 3,
            },
            data,
            CpuAllocator,
        )?;

        let mut thresholded = Image::<_, 1, _>::from_size_val(image.size(), 0, CpuAllocator)?;

        super::threshold_truncate(&image, &mut thresholded, 150)?;

        assert_eq!(thresholded.num_channels(), 1);
        assert_eq!(thresholded.size().width, 2);
        assert_eq!(thresholded.size().height, 3);

        assert_eq!(thresholded.as_slice(), data_expected);

        Ok(())
    }

    #[test]
    fn threshold_to_zero() -> Result<(), ImageError> {
        let data = vec![100u8, 200, 50, 150, 200, 250];
        let data_expected = [0u8, 200, 0, 0, 200, 250];
        let image = Image::<_, 3, _>::new(
            ImageSize {
                width: 2,
                height: 1,
            },
            data,
            CpuAllocator,
        )?;

        let mut thresholded = Image::<_, 3, _>::from_size_val(image.size(), 0, CpuAllocator)?;

        super::threshold_to_zero(&image, &mut thresholded, 150)?;

        assert_eq!(thresholded.num_channels(), 3);
        assert_eq!(thresholded.size().width, 2);
        assert_eq!(thresholded.size().height, 1);

        assert_eq!(thresholded.as_slice(), data_expected);

        Ok(())
    }

    #[test]
    fn threshold_to_zero_inverse() -> Result<(), ImageError> {
        let data = vec![100u8, 200, 50, 150, 200, 250];
        let data_expected = [100u8, 0, 50, 150, 0, 0];
        let image = Image::<_, 3, _>::new(
            ImageSize {
                width: 2,
                height: 1,
            },
            data,
            CpuAllocator,
        )?;

        let mut thresholded = Image::<_, 3, _>::from_size_val(image.size(), 0, CpuAllocator)?;

        super::threshold_to_zero_inverse(&image, &mut thresholded, 150)?;

        assert_eq!(thresholded.num_channels(), 3);
        assert_eq!(thresholded.size().width, 2);
        assert_eq!(thresholded.size().height, 1);

        assert_eq!(thresholded.as_slice(), data_expected);

        Ok(())
    }

    #[test]
    fn test_in_range() -> Result<(), ImageError> {
        let data = vec![100u8, 200, 50, 150, 200, 250];
        let image = Image::<_, 3, _>::new(
            ImageSize {
                width: 2,
                height: 1,
            },
            data,
            CpuAllocator,
        )?;

        let mut thresholded = Image::<u8, 1, _>::from_size_val(image.size(), 0, CpuAllocator)?;

        super::in_range(&image, &mut thresholded, &[100, 150, 0], &[200, 200, 200])?;

        assert_eq!(thresholded.num_channels(), 1);
        assert_eq!(thresholded.size().width, 2);
        assert_eq!(thresholded.size().height, 1);

        assert_eq!(thresholded.get([0, 0, 0]), Some(&255));
        assert_eq!(thresholded.get([0, 1, 0]), Some(&0));

        Ok(())
    }

    #[test]
    fn test_otsu_threshold() -> Result<(), ImageError> {
        // Create a simple bimodal histogram: 50 pixels with value 100, 50 pixels with value 200
        let mut data = vec![100u8; 50];
        data.extend_from_slice(&vec![200u8; 50]);

        let image = Image::<_, 1, _>::new(
            ImageSize {
                width: 10,
                height: 10,
            },
            data,
            CpuAllocator,
        )?;

        let mut binary = Image::<_, 1, _>::from_size_val(image.size(), 0, CpuAllocator)?;

        let threshold = super::otsu_threshold(&image, &mut binary, 255)?;

        assert_eq!(binary.num_channels(), 1);
        assert_eq!(binary.size().width, 10);
        assert_eq!(binary.size().height, 10);

        // Threshold should be found and between the two values
        assert!(threshold > 0);

        // Check that first 50 pixels are 0 (value 100 < threshold) and last 50 are 255 (value 200 >= threshold)
        let binary_slice = binary.as_slice();
        for i in 0..50 {
            assert_eq!(binary_slice[i], 0, "Pixel {} should be 0 (source value 100)", i);
        }
        for i in 50..100 {
            assert_eq!(binary_slice[i], 255, "Pixel {} should be 255 (source value 200)", i);
        }

        Ok(())
    }
}
