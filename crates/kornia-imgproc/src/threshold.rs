use num_traits::Zero;
use std::cmp::PartialOrd;

use kornia_image::{Image, ImageError};

use crate::color::kernel_common::check_size;
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
/// When all pixels share the same intensity (a uniform/flat image), no between-class
/// variance can be established. In this case, following OpenCV (`cv2.THRESH_OTSU`),
/// the function returns threshold `0`.
///
/// # Arguments
///
/// * `hist` - The 256-bin histogram of pixel intensity counts.
///
/// # Note on Exact Ties
///
/// When multiple threshold candidates yield the exact same maximal between-class variance
/// (for example on a minimal synthetic input `[0, 100, 200]` where thresholds 0 and 100 produce
/// equal variance), this implementation selects the first maximal threshold (`0`), whereas
/// OpenCV (`cv2.threshold` with `THRESH_OTSU`) selects `100`. In real-world natural images,
/// exact floating-point ties between distinct histogram splits do not occur.
///
/// # Returns
///
/// The optimal threshold value in the range `[0, 255]`.
///
/// # Errors
///
/// Returns [`ImageError::ImageDataNotInitialized`] if the total pixel count in `hist` is zero.
///
/// # Reference
///
/// * Nobuyuki Otsu (1979). "A threshold selection method from gray-level histograms".
///   IEEE Transactions on Systems, Man, and Cybernetics, 9(1), pp. 62–66.
///   DOI: [10.1109/TSMC.1979.4310076](https://doi.org/10.1109/TSMC.1979.4310076)
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
/// let threshold = otsu_threshold_from_histogram(&hist).unwrap();
/// assert_eq!(threshold, 50);
/// ```
pub fn otsu_threshold_from_histogram(hist: &[usize; 256]) -> Result<u8, ImageError> {
    let total_pixels: usize = hist.iter().sum();
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
    let mut max_variance: f64 = 0.0;
    let mut best_threshold: u8 = 0;
    let mut found_split = false;

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
            found_split = true;
        }
    }

    if !found_split {
        // When all pixels share the same intensity (no split found), return 0 to match OpenCV THRESH_OTSU.
        return Ok(0);
    }

    Ok(best_threshold)
}

/// Compute the optimal Otsu threshold value for an 8-bit, 1-channel image.
///
/// Otsu's method automatically calculates the optimum threshold separating two classes
/// of pixels (foreground and background) so that their intra-class variance is minimal.
///
/// When all pixels share the same intensity (a uniform/flat image), no between-class
/// variance can be established. In this case, following OpenCV (`cv2.THRESH_OTSU`),
/// the function returns threshold `0`.
///
/// Note: This function only computes and returns the threshold value. To both compute
/// and apply binary thresholding to a destination image, use [`threshold_otsu`].
///
/// When `feature = "cuda"` is enabled, this function accepts both host-resident and
/// CUDA device-resident images (the histogram is computed on the GPU and transferred back).
///
/// # Note on Exact Ties
///
/// When multiple threshold candidates yield the exact same maximal between-class variance
/// (for example on a minimal synthetic input `[0, 100, 200]` where thresholds 0 and 100 produce
/// equal variance), this implementation selects the first maximal threshold (`0`), whereas
/// OpenCV (`cv2.threshold` with `THRESH_OTSU`) selects `100`. In real-world natural images,
/// exact floating-point ties between distinct histogram splits do not occur.
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
/// # Reference
///
/// * Nobuyuki Otsu (1979). "A threshold selection method from gray-level histograms".
///   IEEE Transactions on Systems, Man, and Cybernetics, 9(1), pp. 62–66.
///   DOI: [10.1109/TSMC.1979.4310076](https://doi.org/10.1109/TSMC.1979.4310076)
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_imgproc::threshold::compute_otsu_threshold;
///
/// let data = vec![
///     20u8, 20, 20, 20,
///     200, 200, 200, 200,
/// ];
/// let image = Image::<u8, 1>::new(ImageSize { width: 4, height: 2 }, data).unwrap();
/// let threshold = compute_otsu_threshold(&image).unwrap();
/// assert_eq!(threshold, 20);
/// ```
pub fn compute_otsu_threshold(src: &Image<u8, 1>) -> Result<u8, ImageError> {
    // Early exit before allocating histogram buffer or launching GPU kernel
    if src.numel() == 0 {
        return Err(ImageError::ImageDataNotInitialized);
    }

    let mut hist = [0usize; 256];
    crate::histogram::compute_histogram(src, &mut hist, 256)?;

    otsu_threshold_from_histogram(&hist)
}

/// Apply Otsu's automatic thresholding to an 8-bit, 1-channel image.
///
/// Computes the optimal threshold via [`compute_otsu_threshold`] and applies binary thresholding
/// to `dst`.
///
/// On uniform/flat images (where all pixels share the same intensity), following OpenCV
/// (`cv2.THRESH_OTSU`), the threshold evaluates to `0`. Consequently, for any non-zero
/// uniform image, all pixels satisfy `src > 0` and are set to `max_value`, while an
/// all-zero image produces all zeros.
///
/// Note: This function computes the threshold and writes the binarized image to `dst`.
/// To compute only the threshold value without writing to a destination image, use [`compute_otsu_threshold`].
///
/// Note: Unlike [`compute_otsu_threshold`], `threshold_otsu` requires host-resident images because the
/// subsequent binary thresholding step operates on CPU memory.
///
/// # Note on Exact Ties
///
/// When multiple threshold candidates yield the exact same maximal between-class variance
/// (for example on a minimal synthetic input `[0, 100, 200]` where thresholds 0 and 100 produce
/// equal variance), this implementation selects the first maximal threshold (`0`), whereas
/// OpenCV (`cv2.threshold` with `THRESH_OTSU`) selects `100`. In real-world natural images,
/// exact floating-point ties between distinct histogram splits do not occur.
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
/// # Reference
///
/// * Nobuyuki Otsu (1979). "A threshold selection method from gray-level histograms".
///   IEEE Transactions on Systems, Man, and Cybernetics, 9(1), pp. 62–66.
///   DOI: [10.1109/TSMC.1979.4310076](https://doi.org/10.1109/TSMC.1979.4310076)
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

    check_size(src, dst)?;

    let threshold = compute_otsu_threshold(src)?;
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

        let threshold = super::compute_otsu_threshold(&image)?;
        assert_eq!(threshold, 20);

        let mut dst = Image::<u8, 1>::from_size_val(image.size(), 0)?;
        let computed_thresh = super::threshold_otsu(&image, &mut dst, 255)?;
        assert_eq!(computed_thresh, 20);
        assert_eq!(dst.as_slice(), &[0, 0, 0, 0, 255, 255, 255, 255]);

        Ok(())
    }

    #[test]
    fn test_otsu_threshold_uniform() -> Result<(), ImageError> {
        // Flat non-zero image (128): threshold is 0 (matching OpenCV THRESH_OTSU),
        // and all pixels > 0 become max_value (255).
        let data = vec![128u8; 16];
        let image = Image::<u8, 1>::new(
            ImageSize {
                width: 4,
                height: 4,
            },
            data,
        )?;

        let threshold = super::compute_otsu_threshold(&image)?;
        assert_eq!(threshold, 0);

        let mut dst = Image::<u8, 1>::from_size_val(image.size(), 0)?;
        let thresh_otsu = super::threshold_otsu(&image, &mut dst, 255)?;
        assert_eq!(thresh_otsu, 0);
        assert_eq!(dst.as_slice(), &[255u8; 16]);

        // Flat zero image: threshold is 0, 0 > 0 is false, so all pixels remain 0.
        let image_zero = Image::<u8, 1>::from_size_val(image.size(), 0)?;
        let threshold_zero = super::compute_otsu_threshold(&image_zero)?;
        assert_eq!(threshold_zero, 0);

        let mut dst_zero = Image::<u8, 1>::from_size_val(image.size(), 255)?;
        let thresh_otsu_zero = super::threshold_otsu(&image_zero, &mut dst_zero, 255)?;
        assert_eq!(thresh_otsu_zero, 0);
        assert_eq!(dst_zero.as_slice(), &[0u8; 16]);

        // Flat max image (255): threshold is 0, 255 > 0 is true, so all pixels become 255.
        let image_max = Image::<u8, 1>::from_size_val(image.size(), 255)?;
        let threshold_max = super::compute_otsu_threshold(&image_max)?;
        assert_eq!(threshold_max, 0);

        let mut dst_max = Image::<u8, 1>::from_size_val(image.size(), 0)?;
        let thresh_otsu_max = super::threshold_otsu(&image_max, &mut dst_max, 255)?;
        assert_eq!(thresh_otsu_max, 0);
        assert_eq!(dst_max.as_slice(), &[255u8; 16]);

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

        let threshold = super::otsu_threshold_from_histogram(&hist)?;
        assert_eq!(threshold, 50);

        // Reviewer example: hist[50]=100, hist[200]=100
        // Total is derived from hist (200), yielding threshold 50 without mismatch bugs.
        let mut hist2 = [0usize; 256];
        hist2[50] = 100;
        hist2[200] = 100;
        assert_eq!(super::otsu_threshold_from_histogram(&hist2)?, 50);

        // Uniform histogram: single non-zero bin returns 0 (matching OpenCV THRESH_OTSU)
        let mut hist_uniform = [0usize; 256];
        hist_uniform[128] = 200;
        assert_eq!(super::otsu_threshold_from_histogram(&hist_uniform)?, 0);

        // Empty histogram: returns ImageDataNotInitialized error
        let empty_hist = [0usize; 256];
        let empty_result = super::otsu_threshold_from_histogram(&empty_hist);
        assert!(matches!(
            empty_result,
            Err(ImageError::ImageDataNotInitialized)
        ));

        Ok(())
    }

    #[test]
    fn test_otsu_threshold_reference_pinned() -> Result<(), ImageError> {
        // 1. 8x8 bimodal image pinned against OpenCV 4.x
        // cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        // Reference output: threshold = 45
        let data = vec![
            40u8, 45, 38, 42, 180, 190, 185, 195, 42, 39, 44, 41, 182, 188, 192, 186, 38, 43, 40,
            45, 185, 191, 189, 193, 41, 37, 42, 39, 181, 187, 190, 184, 39, 41, 38, 43, 183, 189,
            186, 192, 43, 38, 41, 40, 184, 190, 188, 194, 40, 42, 39, 44, 186, 192, 191, 185, 44,
            40, 43, 38, 187, 185, 193, 189,
        ];
        let image = Image::<u8, 1>::new(
            ImageSize {
                width: 8,
                height: 8,
            },
            data,
        )?;
        let mut dst = Image::<u8, 1>::from_size_val(image.size(), 0)?;

        let threshold = super::threshold_otsu(&image, &mut dst, 255)?;
        assert_eq!(threshold, 45);

        // Columns 0..4 have values <= 45 (background -> 0)
        // Columns 4..8 have values >= 180 (foreground -> 255)
        for row in 0..8 {
            let row_slice = &dst.as_slice()[row * 8..(row + 1) * 8];
            assert_eq!(&row_slice[..4], &[0, 0, 0, 0]);
            assert_eq!(&row_slice[4..], &[255, 255, 255, 255]);
        }

        // 2. Flat images pinned against OpenCV 4.x
        // cv2.threshold(np.full((4,4), 128, dtype=np.uint8), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        // Reference output: threshold = 0.0, dst = all 255
        let img_128 = Image::<u8, 1>::from_size_val(
            ImageSize {
                width: 4,
                height: 4,
            },
            128,
        )?;
        let mut dst_128 = Image::<u8, 1>::from_size_val(img_128.size(), 0)?;
        let thresh_128 = super::threshold_otsu(&img_128, &mut dst_128, 255)?;
        assert_eq!(thresh_128, 0);
        assert_eq!(dst_128.as_slice(), &[255u8; 16]);

        // cv2.threshold(np.full((4,4), 0, dtype=np.uint8), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        // Reference output: threshold = 0.0, dst = all 0
        let img_0 = Image::<u8, 1>::from_size_val(
            ImageSize {
                width: 4,
                height: 4,
            },
            0,
        )?;
        let mut dst_0 = Image::<u8, 1>::from_size_val(img_0.size(), 255)?;
        let thresh_0 = super::threshold_otsu(&img_0, &mut dst_0, 255)?;
        assert_eq!(thresh_0, 0);
        assert_eq!(dst_0.as_slice(), &[0u8; 16]);

        Ok(())
    }

    #[test]
    fn test_otsu_threshold_large_histogram_no_overflow() -> Result<(), ImageError> {
        let mut hist = [0usize; 256];
        // 250 * 20_000_000 = 5_000_000_000, which exceeds 32-bit u32::MAX (4_294_967_295)
        hist[100] = 10_000_000;
        hist[200] = 20_000_000;
        hist[250] = 20_000_000;

        let threshold = super::otsu_threshold_from_histogram(&hist)?;
        assert_eq!(threshold, 100);

        Ok(())
    }

    #[test]
    fn test_otsu_threshold_empty_image() -> Result<(), ImageError> {
        let empty_src = Image::<u8, 1>::new(
            ImageSize {
                width: 0,
                height: 0,
            },
            vec![],
        )?;
        let res_compute = super::compute_otsu_threshold(&empty_src);
        assert!(matches!(
            res_compute,
            Err(ImageError::ImageDataNotInitialized)
        ));

        let mut empty_dst = Image::<u8, 1>::new(
            ImageSize {
                width: 0,
                height: 0,
            },
            vec![],
        )?;
        let res_thresh = super::threshold_otsu(&empty_src, &mut empty_dst, 255);
        assert!(matches!(
            res_thresh,
            Err(ImageError::ImageDataNotInitialized)
        ));

        Ok(())
    }

    #[test]
    fn test_otsu_threshold_single_pixel() -> Result<(), ImageError> {
        let single_pixel = Image::<u8, 1>::new(
            ImageSize {
                width: 1,
                height: 1,
            },
            vec![42u8],
        )?;

        let thresh = super::compute_otsu_threshold(&single_pixel)?;
        assert_eq!(thresh, 0);

        let mut dst = Image::<u8, 1>::from_size_val(single_pixel.size(), 0)?;
        let thresh_otsu = super::threshold_otsu(&single_pixel, &mut dst, 255)?;
        assert_eq!(thresh_otsu, 0);
        assert_eq!(dst.as_slice(), &[255u8]);

        Ok(())
    }

    #[test]
    fn test_otsu_threshold_ramp_matches_opencv() -> Result<(), ImageError> {
        // Uniform ramp 0..=255 (16x16)
        // cv2.threshold(np.arange(256, dtype=np.uint8).reshape(16, 16), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        // Reference threshold = 127
        let data: Vec<u8> = (0u8..=255).collect();
        let image = Image::<u8, 1>::new(
            ImageSize {
                width: 16,
                height: 16,
            },
            data,
        )?;

        let thresh = super::compute_otsu_threshold(&image)?;
        assert_eq!(thresh, 127);

        let mut dst = Image::<u8, 1>::from_size_val(image.size(), 0)?;
        let thresh_otsu = super::threshold_otsu(&image, &mut dst, 255)?;
        assert_eq!(thresh_otsu, 127);

        // Values <= 127 are 0; values > 127 are 255
        for (i, &pixel) in dst.as_slice().iter().enumerate() {
            if i <= 127 {
                assert_eq!(pixel, 0);
            } else {
                assert_eq!(pixel, 255);
            }
        }

        Ok(())
    }

    #[test]
    fn test_otsu_exact_tie_breaks_first() -> Result<(), ImageError> {
        // Minimal synthetic input [0, 100, 200] with exact variance ties across splits.
        // kornia selects the first maximal threshold (0), whereas OpenCV selects 100.
        let image = Image::<u8, 1>::new(
            ImageSize {
                width: 3,
                height: 1,
            },
            vec![0, 100, 200],
        )?;
        let thresh = super::compute_otsu_threshold(&image)?;
        assert_eq!(thresh, 0);

        let mut dst = Image::<u8, 1>::from_size_val(image.size(), 0)?;
        let thresh_otsu = super::threshold_otsu(&image, &mut dst, 255)?;
        assert_eq!(thresh_otsu, 0);
        // Pixels <= 0 are 0; pixels > 0 are 255
        assert_eq!(dst.as_slice(), &[0, 255, 255]);

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
