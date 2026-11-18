use num_traits::Zero;
use std::cmp::PartialOrd;

use kornia_image::{Image, ImageError};

use crate::parallel;

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
/// use kornia_imgproc::threshold::threshold_binary;
///
/// let data = vec![100u8, 200, 50, 150, 200, 250];
/// let image = Image::<_, 1>::new(ImageSize { width: 2, height: 3 }, data).unwrap();
///
/// let mut thresholded = Image::<_, 1>::from_size_val(image.size(), 0).unwrap();
///
/// threshold_binary(&image, &mut thresholded, 100, 255).unwrap();
/// assert_eq!(thresholded.num_channels(), 1);
/// assert_eq!(thresholded.size().width, 2);
/// assert_eq!(thresholded.size().height, 3);
/// ```
pub fn threshold_binary<T, const C: usize>(
    src: &Image<T, C>,
    dst: &mut Image<T, C>,
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
/// use kornia_imgproc::threshold::threshold_binary_inverse;
///
/// let data = vec![100u8, 200, 50, 150, 200, 250];
/// let image = Image::<_, 1>::new(ImageSize { width: 2, height: 3 }, data).unwrap();
///
/// let mut thresholded = Image::<_, 1>::from_size_val(image.size(), 0).unwrap();
///
/// threshold_binary_inverse(&image, &mut thresholded, 100, 255).unwrap();
/// assert_eq!(thresholded.num_channels(), 1);
/// assert_eq!(thresholded.size().width, 2);
/// assert_eq!(thresholded.size().height, 3);
/// ```
pub fn threshold_binary_inverse<T, const C: usize>(
    src: &Image<T, C>,
    dst: &mut Image<T, C>,
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
/// use kornia_imgproc::threshold::threshold_truncate;
///
/// let data = vec![100u8, 200, 50, 150, 200, 250];
/// let image = Image::<_, 1>::new(ImageSize { width: 2, height: 3 }, data).unwrap();
///
/// let mut thresholded = Image::<_, 1>::from_size_val(image.size(), 0).unwrap();
///
/// threshold_truncate(&image, &mut thresholded, 150).unwrap();
/// assert_eq!(thresholded.num_channels(), 1);
/// assert_eq!(thresholded.size().width, 2);
/// assert_eq!(thresholded.size().height, 3);
/// ```
pub fn threshold_truncate<T, const C: usize>(
    src: &Image<T, C>,
    dst: &mut Image<T, C>,
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
/// use kornia_imgproc::threshold::threshold_to_zero;
///
/// let data = vec![100u8, 200, 50, 150, 200, 250];
/// let image = Image::<_, 3>::new(ImageSize { width: 2, height: 1 }, data).unwrap();
///
/// let mut thresholded = Image::<_, 3>::from_size_val(image.size(), 0).unwrap();
///
/// threshold_to_zero(&image, &mut thresholded, 150).unwrap();
/// assert_eq!(thresholded.num_channels(), 3);
/// assert_eq!(thresholded.size().width, 2);
/// assert_eq!(thresholded.size().height, 1);
/// ```
pub fn threshold_to_zero<T, const C: usize>(
    src: &Image<T, C>,
    dst: &mut Image<T, C>,
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
/// use kornia_imgproc::threshold::threshold_to_zero_inverse;
///
/// let data = vec![100u8, 200, 50, 150, 200, 250];
/// let image = Image::<_, 3>::new(ImageSize { width: 2, height: 1 }, data).unwrap();
///
/// let mut thresholded = Image::<_, 3>::from_size_val(image.size(), 0).unwrap();
///
/// threshold_to_zero_inverse(&image, &mut thresholded, 150).unwrap();
/// assert_eq!(thresholded.num_channels(), 3);
/// assert_eq!(thresholded.size().width, 2);
/// assert_eq!(thresholded.size().height, 1);
/// ```
pub fn threshold_to_zero_inverse<T, const C: usize>(
    src: &Image<T, C>,
    dst: &mut Image<T, C>,
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
/// use kornia_imgproc::threshold::in_range;
///
/// let data = vec![100u8, 200, 50, 150, 200, 250];
///
/// let image = Image::<u8, 3>::new(
///    ImageSize {
///       width: 2,
///       height: 1,
///    },
///    data,
/// )
/// .unwrap();
///
/// let mut thresholded = Image::<u8, 1>::from_size_val(image.size(), 0).unwrap();
///
/// in_range(&image, &mut thresholded, &[100, 150, 0], &[200, 200, 200]).unwrap();
/// assert_eq!(thresholded.num_channels(), 1);
/// assert_eq!(thresholded.size().width, 2);
///
/// assert_eq!(thresholded.get_pixel(0, 0, 0).unwrap(), &255);
/// assert_eq!(thresholded.get_pixel(1, 0, 0).unwrap(), &0);
/// ```
pub fn in_range<T, const C: usize>(
    src: &Image<T, C>,
    dst: &mut Image<u8, 1>,
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

/// Compute the optimal Otsu threshold value from an intensity histogram.
///
/// Implements Otsu's method by finding the threshold that maximizes the between-class
/// variance of pixel intensities.
///
/// # Arguments
///
/// * `hist` - The 256-bin histogram of pixel intensity counts.
/// * `total_pixels` - The total number of pixels in the image.
///
/// # Returns
///
/// The optimal threshold value in the range `[0, 255]`.
///
/// # Errors
///
/// Returns [`ImageError::ImageDataNotInitialized`] if `total_pixels` is zero.
///
/// # Example
///
/// ```
/// use kornia_imgproc::threshold::otsu_threshold_from_histogram;
///
/// let mut hist = [0usize; 256];
/// hist[50] = 100;
/// hist[200] = 100;
///
/// let threshold = otsu_threshold_from_histogram(&hist, 200).unwrap();
/// assert_eq!(threshold, 50);
/// ```
pub fn otsu_threshold_from_histogram(
    hist: &[usize; 256],
    total_pixels: usize,
) -> Result<u8, ImageError> {
    if total_pixels == 0 {
        return Err(ImageError::ImageDataNotInitialized);
    }

    // Total intensity sum across all bins
    let sum_total: f64 = hist
        .iter()
        .enumerate()
        .map(|(i, &count)| (i as f64) * (count as f64))
        .sum();

    let total = total_pixels as f64;
    let mut weight_background: f64 = 0.0;
    let mut sum_background: f64 = 0.0;
    let mut max_variance: f64 = -1.0;
    let mut best_threshold: u8 = 0;

    for (t, &count) in hist.iter().enumerate() {
        if count == 0 {
            continue;
        }

        weight_background += count as f64;
        let weight_foreground = total - weight_background;

        if weight_foreground <= 0.0 {
            break;
        }

        sum_background += (t as f64) * (count as f64);
        let sum_foreground = sum_total - sum_background;

        let mean_background = sum_background / weight_background;
        let mean_foreground = sum_foreground / weight_foreground;

        let mean_diff = mean_background - mean_foreground;
        let variance_between = weight_background * weight_foreground * mean_diff * mean_diff;

        if variance_between > max_variance {
            max_variance = variance_between;
            best_threshold = t as u8;
        }
    }

    if max_variance < 0.0 {
        // If all pixels share the same intensity, return that intensity.
        return Ok(hist
            .iter()
            .position(|&c| c > 0)
            .map(|idx| idx as u8)
            .unwrap_or(0));
    }

    Ok(best_threshold)
}

/// Compute the optimal Otsu threshold value for an 8-bit, 1-channel image.
///
/// Otsu's method automatically calculates the optimum threshold separating two classes
/// of pixels (foreground and background) so that their intra-class variance is minimal.
///
/// # Arguments
///
/// * `src` - The input 8-bit, 1-channel grayscale image.
///
/// # Returns
///
/// The computed threshold value in the range `[0, 255]`.
///
/// # Errors
///
/// Returns [`ImageError::ImageDataNotInitialized`] if the image data is empty.
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_imgproc::threshold::otsu_threshold;
///
/// let data = vec![
///     20u8, 20, 20, 20,
///     200, 200, 200, 200,
/// ];
/// let image = Image::<u8, 1>::new(ImageSize { width: 4, height: 2 }, data).unwrap();
/// let threshold = otsu_threshold(&image).unwrap();
/// assert_eq!(threshold, 20);
/// ```
pub fn otsu_threshold(src: &Image<u8, 1>) -> Result<u8, ImageError> {
    if src.numel() == 0 {
        return Err(ImageError::ImageDataNotInitialized);
    }

    let mut hist = [0usize; 256];
    crate::histogram::compute_histogram(src, &mut hist, 256)?;

    otsu_threshold_from_histogram(&hist, src.numel())
}

/// Apply Otsu's automatic thresholding to an 8-bit, 1-channel image.
///
/// Computes the optimal threshold via [`otsu_threshold`] and applies binary thresholding
/// to `dst`.
///
/// # Arguments
///
/// * `src` - The input 8-bit, 1-channel grayscale image.
/// * `dst` - The pre-allocated output 8-bit, 1-channel image. Must match `src` dimensions.
/// * `max_value` - The maximum value to assign to pixels exceeding the threshold.
///
/// # Returns
///
/// The optimal threshold value computed by Otsu's method.
///
/// # Errors
///
/// Returns [`ImageError::InvalidImageSize`] if `src` and `dst` dimensions differ.
/// Returns [`ImageError::ImageDataNotInitialized`] if the image data is empty.
/// Returns [`ImageError::UnsupportedDevice`] if `src` or `dst` is resident on device (GPU) memory.
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_imgproc::threshold::threshold_otsu;
///
/// let data = vec![
///     10u8, 20, 10, 20,
///     200, 220, 210, 200,
/// ];
/// let image = Image::<u8, 1>::new(ImageSize { width: 4, height: 2 }, data).unwrap();
/// let mut dst = Image::<u8, 1>::from_size_val(image.size(), 0).unwrap();
///
/// let thresh = threshold_otsu(&image, &mut dst, 255).unwrap();
/// assert_eq!(thresh, 20);
/// assert_eq!(dst.as_slice(), &[0, 0, 0, 0, 255, 255, 255, 255]);
/// ```
pub fn threshold_otsu(
    src: &Image<u8, 1>,
    dst: &mut Image<u8, 1>,
    max_value: u8,
) -> Result<u8, ImageError> {
    #[cfg(feature = "cuda")]
    if crate::cuda::dispatch::is_device(src) || crate::cuda::dispatch::is_device(dst) {
        return Err(ImageError::UnsupportedDevice);
    }

    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }

    let threshold = otsu_threshold(src)?;
    threshold_binary(src, dst, threshold, max_value)?;

    Ok(threshold)
}

// TODO: triangle

#[cfg(test)]
mod tests {
    use kornia_image::{Image, ImageError, ImageSize};

    #[test]
    fn threshold_binary() -> Result<(), ImageError> {
        let data = vec![100u8, 200, 50, 150, 200, 250];
        let data_expected = [0u8, 255, 0, 255, 255, 255];
        let image = Image::<_, 1>::new(
            ImageSize {
                width: 2,
                height: 3,
            },
            data,
        )?;

        let mut thresholded = Image::<_, 1>::from_size_val(image.size(), 0)?;

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
        let image = Image::<_, 1>::new(
            ImageSize {
                width: 2,
                height: 3,
            },
            data,
        )?;

        let mut thresholded = Image::<_, 1>::from_size_val(image.size(), 0)?;

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
        let image = Image::<_, 1>::new(
            ImageSize {
                width: 2,
                height: 3,
            },
            data,
        )?;

        let mut thresholded = Image::<_, 1>::from_size_val(image.size(), 0)?;

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
        let image = Image::<_, 3>::new(
            ImageSize {
                width: 2,
                height: 1,
            },
            data,
        )?;

        let mut thresholded = Image::<_, 3>::from_size_val(image.size(), 0)?;

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
        let image = Image::<_, 3>::new(
            ImageSize {
                width: 2,
                height: 1,
            },
            data,
        )?;

        let mut thresholded = Image::<_, 3>::from_size_val(image.size(), 0)?;

        super::threshold_to_zero_inverse(&image, &mut thresholded, 150)?;

        assert_eq!(thresholded.num_channels(), 3);
        assert_eq!(thresholded.size().width, 2);
        assert_eq!(thresholded.size().height, 1);

        assert_eq!(thresholded.as_slice(), data_expected);

        Ok(())
    }

    #[test]
    fn test_threshold_binary_equal_to_threshold() -> Result<(), ImageError> {
        // The implementation uses strict `>` (not `>=`), so a pixel whose value
        // exactly equals the threshold is treated as NOT exceeding it and maps to
        // zero rather than max_val.  Pin that boundary behaviour here.
        let data = vec![100u8, 101, 99];
        let image = Image::<_, 1>::new(
            ImageSize {
                width: 3,
                height: 1,
            },
            data,
        )?;

        let mut thresholded = Image::<_, 1>::from_size_val(image.size(), 0)?;

        super::threshold_binary(&image, &mut thresholded, 100, 255)?;

        // 100 == threshold → 0 (not strictly greater)
        // 101 >  threshold → 255
        // 99  <  threshold → 0
        assert_eq!(thresholded.as_slice(), &[0u8, 255, 0]);

        Ok(())
    }

    #[test]
    fn test_in_range() -> Result<(), ImageError> {
        let data = vec![100u8, 200, 50, 150, 200, 250];
        let image = Image::<_, 3>::new(
            ImageSize {
                width: 2,
                height: 1,
            },
            data,
        )?;

        let mut thresholded = Image::<u8, 1>::from_size_val(image.size(), 0)?;

        super::in_range(&image, &mut thresholded, &[100, 150, 0], &[200, 200, 200])?;

        assert_eq!(thresholded.num_channels(), 1);
        assert_eq!(thresholded.size().width, 2);
        assert_eq!(thresholded.size().height, 1);

        assert_eq!(thresholded.get([0, 0, 0]), Some(&255));
        assert_eq!(thresholded.get([0, 1, 0]), Some(&0));

        Ok(())
    }

    #[test]
    fn test_otsu_threshold_bimodal() -> Result<(), ImageError> {
        let data = vec![10u8, 20, 10, 20, 180, 200, 180, 200];
        let image = Image::<u8, 1>::new(
            ImageSize {
                width: 4,
                height: 2,
            },
            data,
        )?;

        let threshold = super::otsu_threshold(&image)?;
        assert_eq!(threshold, 20);

        let mut dst = Image::<u8, 1>::from_size_val(image.size(), 0)?;
        let computed_thresh = super::threshold_otsu(&image, &mut dst, 255)?;
        assert_eq!(computed_thresh, 20);
        assert_eq!(dst.as_slice(), &[0, 0, 0, 0, 255, 255, 255, 255]);

        Ok(())
    }

    #[test]
    fn test_otsu_threshold_uniform() -> Result<(), ImageError> {
        let data = vec![128u8; 16];
        let image = Image::<u8, 1>::new(
            ImageSize {
                width: 4,
                height: 4,
            },
            data,
        )?;

        let threshold = super::otsu_threshold(&image)?;
        assert_eq!(threshold, 128);

        let mut dst = Image::<u8, 1>::from_size_val(image.size(), 0)?;
        super::threshold_otsu(&image, &mut dst, 255)?;
        assert_eq!(dst.as_slice(), &[0u8; 16]);

        Ok(())
    }

    #[test]
    fn test_otsu_threshold_size_mismatch() -> Result<(), ImageError> {
        let image = Image::<u8, 1>::new(
            ImageSize {
                width: 2,
                height: 2,
            },
            vec![10, 20, 180, 200],
        )?;
        let mut dst = Image::<u8, 1>::from_size_val(
            ImageSize {
                width: 3,
                height: 3,
            },
            0,
        )?;

        let result = super::threshold_otsu(&image, &mut dst, 255);
        assert!(result.is_err());
        match result {
            Err(ImageError::InvalidImageSize(2, 2, 3, 3)) => (),
            _ => panic!("Expected InvalidImageSize error"),
        }

        Ok(())
    }

    #[test]
    fn test_otsu_threshold_from_histogram() -> Result<(), ImageError> {
        let mut hist = [0usize; 256];
        hist[40] = 50;
        hist[50] = 50;
        hist[180] = 50;
        hist[190] = 50;

        let threshold = super::otsu_threshold_from_histogram(&hist, 200)?;
        assert_eq!(threshold, 50);

        let empty_result = super::otsu_threshold_from_histogram(&hist, 0);
        assert!(empty_result.is_err());

        Ok(())
    }

    #[test]
    fn test_otsu_threshold_large_histogram_no_overflow() -> Result<(), ImageError> {
        let mut hist = [0usize; 256];
        // 250 * 20_000_000 = 5_000_000_000, which exceeds 32-bit u32::MAX (4_294_967_295)
        hist[100] = 10_000_000;
        hist[200] = 20_000_000;
        hist[250] = 20_000_000;

        let threshold = super::otsu_threshold_from_histogram(&hist, 50_000_000)?;
        assert_eq!(threshold, 100);

        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_threshold_otsu_device_rejected() -> Result<(), ImageError> {
        let stream = crate::cuda::color::test_utils::default_stream();
        let size = ImageSize {
            width: 4,
            height: 4,
        };
        let host_data = vec![
            10u8, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
        ];

        let host_src = Image::<u8, 1>::new(size, host_data)?;
        let dev_src = host_src.to_cuda(&stream)?;

        // 1. Device src with host dst: rejected with UnsupportedDevice, dst unchanged
        let mut host_dst = Image::<u8, 1>::from_size_val(size, 42)?;
        let result = super::threshold_otsu(&dev_src, &mut host_dst, 255);
        assert!(matches!(result, Err(ImageError::UnsupportedDevice)));
        assert_eq!(host_dst.as_slice(), &[42u8; 16]);

        // 2. Host src with device dst: rejected with UnsupportedDevice, dst unchanged
        let mut dev_dst = Image::<u8, 1>::from_size_val(size, 42)?.to_cuda(&stream)?;
        let result = super::threshold_otsu(&host_src, &mut dev_dst, 255);
        assert!(matches!(result, Err(ImageError::UnsupportedDevice)));
        let back = dev_dst.to_host_owned()?;
        assert_eq!(back.as_slice(), &[42u8; 16]);

        // 3. Device src with device dst: rejected with UnsupportedDevice, dst unchanged
        let mut dev_dst2 = Image::<u8, 1>::from_size_val(size, 42)?.to_cuda(&stream)?;
        let result = super::threshold_otsu(&dev_src, &mut dev_dst2, 255);
        assert!(matches!(result, Err(ImageError::UnsupportedDevice)));
        let back2 = dev_dst2.to_host_owned()?;
        assert_eq!(back2.as_slice(), &[42u8; 16]);

        Ok(())
    }
}
