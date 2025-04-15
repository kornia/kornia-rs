use num_traits::Zero;
use std::cmp::PartialOrd;

use kornia_image::{Image, ImageError};

use crate::parallel;

/// The type of thresholding to apply.
pub enum ThresholdType {
    /// Binary thresholding
    Binary,
    /// Inverse binary thresholding
    BinaryInv,
    /// Truncated thresholding
    Trunc,
    /// To zero thresholding
    ToZero,
    /// Inverse to zero thresholding
    ToZeroInv,
}
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

/// Apply Otsu's thresholding to an image.
///
/// # Arguments
///
/// * `src` - The input image of an arbitrary number of channels and type.
/// * `dst` - The output image of an arbitrary number of channels and type.
/// * `thres_type` - The type of thresholding to apply.
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
/// use kornia_imgproc::threshold::otsu_threshold;
///
/// use kornia_imgproc::threshold::ThresholdType;
///
/// let data = vec![100u8, 200, 50, 150, 200, 250];
/// let image = Image::<_, 1>::new(
///    ImageSize {
///       width: 2,
///     height: 3,
///   },
///   data,
/// ).unwrap();
///     
///
/// let mut thresholded = Image::<_, 1>::from_size_val(image.size(), 0).unwrap();
///
/// otsu_threshold(&image, &mut thresholded, ThresholdType::Binary, 255).unwrap();
///
/// assert_eq!(thresholded.num_channels(), 1);
///
/// assert_eq!(thresholded.size().width, 2);
/// assert_eq!(thresholded.size().height, 3);
/// assert_eq!(thresholded.as_slice(), [0, 255, 0, 255, 255, 255]);
/// ```
///
pub fn otsu_threshold<const C: usize>(
    src: &Image<u8, C>,
    dst: &mut Image<u8, C>,
    thres_type: ThresholdType,
    max_value: u8,
) -> Result<(), ImageError> {
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }

    let height = src.height();
    let width = src.width();
    const BINS: usize = 256;
    let mut histogram = [0u32; BINS];
    let image = src.as_slice();

    // Compute histogram
    for &pixel in image {
        // Assuming pixel is in the range [0, 255]
        histogram[pixel as usize] += 1;
    }

    let total_pixels = (width * height) as f64;
    let mut sum_total = 0.0;

    // Calculate total sum for mean computation
    for (i, &count) in histogram.iter().enumerate() {
        sum_total += i as f64 * count as f64;
    }

    let mut best_variance = 0.0;
    let mut best_threshold = 0;

    // Initialize accumulators
    let mut weight_back = 0.0;
    let mut sum_back = 0.0;

    // Iterate through all possible thresholds
    for (current_threshold, &hist_count) in histogram.iter().enumerate() {
        let current_threshold = current_threshold as u8;

        // Update background class accumulators
        weight_back += hist_count as f64;
        sum_back += current_threshold as f64 * hist_count as f64;

        // Skip empty classes
        if weight_back == 0.0 || weight_back == total_pixels {
            continue;
        }

        // Calculate means for both classes
        let mean_back = sum_back / weight_back;
        let weight_fore = total_pixels - weight_back;
        let sum_fore = sum_total - sum_back;
        let mean_fore = sum_fore / weight_fore;

        // Calculate between-class variance
        let variance = weight_back * weight_fore * (mean_back - mean_fore).powi(2);

        // Update best threshold if variance is higher
        if variance > best_variance {
            best_variance = variance;
            best_threshold = current_threshold;
        }
    }
    // Apply the threshold to the image
    match thres_type {
        ThresholdType::Binary => {
            let _ = threshold_binary(src, dst, best_threshold, max_value);
        }
        ThresholdType::BinaryInv => {
            let _ = threshold_binary_inverse(src, dst, best_threshold, max_value);
        }
        ThresholdType::Trunc => {
            let _ = threshold_truncate(src, dst, best_threshold);
        }
        ThresholdType::ToZero => {
            let _ = threshold_to_zero(src, dst, best_threshold);
        }
        ThresholdType::ToZeroInv => {
            let _ = threshold_to_zero_inverse(src, dst, best_threshold);
        }
    }
    Ok(())
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
    fn test_otsu_threshold() -> Result<(), ImageError> {
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

        super::otsu_threshold(&image, &mut thresholded, super::ThresholdType::Binary, 255)?;

        assert_eq!(thresholded.num_channels(), 1);
        assert_eq!(thresholded.size().width, 2);
        assert_eq!(thresholded.size().height, 3);

        assert_eq!(thresholded.as_slice(), data_expected);
        Ok(())
    }
}
