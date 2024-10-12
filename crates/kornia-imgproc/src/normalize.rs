use num_traits::Float;

use kornia_image::{Image, ImageError};

use crate::parallel;

/// Normalize an image using the mean and standard deviation.
///
/// The formula for normalizing an image is:
///
/// (image - mean) / std
///
/// Each channel is normalized independently.
///
/// # Arguments
///
/// * `src` - The input image of shape (height, width, channels).
/// * `dst` - The output image of shape (height, width, channels).
/// * `mean` - The mean value for each channel.
/// * `std` - The standard deviation for each channel.
///
/// # Returns
///
/// The normalized image of shape (height, width, channels).
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_imgproc::normalize::normalize_mean_std;
///
/// let image_data = vec![0f32, 1.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 0.0, 1.0, 2.0, 3.0];
/// let image = Image::<f32, 3>::new(
///   ImageSize {
///     width: 2,
///     height: 2,
///   },
///   image_data,
/// )
/// .unwrap();
///
/// let mut image_normalized = Image::<f32, 3>::from_size_val(image.size(), 0.0).unwrap();
///
/// normalize_mean_std(
///     &image,
///     &mut image_normalized,
///     &[0.5, 1.0, 0.5],
///     &[1.0, 1.0, 1.0],
/// )
/// .unwrap();
///
/// assert_eq!(image_normalized.num_channels(), 3);
/// assert_eq!(image_normalized.size().width, 2);
/// assert_eq!(image_normalized.size().height, 2);
/// ```
pub fn normalize_mean_std<T, const C: usize>(
    src: &Image<T, C>,
    dst: &mut Image<T, C>,
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
/// use kornia_imgproc::normalize::find_min_max;
///
/// let image_data = vec![0u8, 1, 0, 1, 2, 3, 0, 1, 0, 1, 2, 3];
/// let image = Image::<u8, 3>::new(
///   ImageSize {
///     width: 2,
///     height: 2,
///   },
///   image_data,
/// )
/// .unwrap();
///
/// let (min, max) = find_min_max(&image).unwrap();
/// assert_eq!(min, 0);
/// assert_eq!(max, 3);
/// ```
pub fn find_min_max<T, const C: usize>(image: &Image<T, C>) -> Result<(T, T), ImageError>
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
/// use kornia_imgproc::normalize::normalize_min_max;
///
/// let image_data = vec![0.0f32, 1.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 0.0, 1.0, 2.0, 3.0];
/// let image = Image::<f32, 3>::new(
///   ImageSize {
///     width: 2,
///     height: 2,
///   },
///   image_data,
/// )
/// .unwrap();
///
/// let mut image_normalized = Image::<f32, 3>::from_size_val(image.size(), 0.0).unwrap();
///
/// normalize_min_max(&image, &mut image_normalized, 0.0, 1.0).unwrap();
///
/// assert_eq!(image_normalized.num_channels(), 3);
/// assert_eq!(image_normalized.size().width, 2);
/// assert_eq!(image_normalized.size().height, 2);
/// ```
pub fn normalize_min_max<T, const C: usize>(
    src: &Image<T, C>,
    dst: &mut Image<T, C>,
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

    #[test]
    fn normalize_mean_std() -> Result<(), ImageError> {
        let image_data = vec![
            0.0f32, 1.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 0.0, 1.0, 2.0, 3.0,
        ];

        let image_expected = [
            -0.5f32, 0.0, -0.5, 0.5, 1.0, 2.5, -0.5, 0.0, -0.5, 0.5, 1.0, 2.5,
        ];

        let image = Image::<f32, 3>::new(
            ImageSize {
                width: 2,
                height: 2,
            },
            image_data,
        )?;

        let mean = [0.5, 1.0, 0.5];
        let std = [1.0, 1.0, 1.0];

        let mut normalized = Image::<f32, 3>::from_size_val(image.size(), 0.0)?;

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
        let image = Image::<u8, 3>::new(
            ImageSize {
                width: 2,
                height: 2,
            },
            image_data,
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

        let image = Image::<f32, 3>::new(
            ImageSize {
                width: 2,
                height: 2,
            },
            image_data,
        )?;

        let mut normalized = Image::<f32, 3>::from_size_val(image.size(), 0.0)?;

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
