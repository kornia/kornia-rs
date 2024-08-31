use kornia_core::SafeTensorType;
use kornia_image::{Image, ImageError};

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
/// use kornia::image::{Image, ImageSize};
/// use kornia::imgproc::threshold::threshold_binary;
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
pub fn threshold_binary<T, const CHANNELS: usize>(
    src: &Image<T, CHANNELS>,
    dst: &mut Image<T, CHANNELS>,
    threshold: T,
    max_value: T,
) -> Result<(), ImageError>
where
    T: SafeTensorType,
{
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.size().width,
            src.size().height,
            dst.size().width,
            dst.size().height,
        ));
    }

    // TODO: parallelize again
    dst.as_slice_mut()
        .iter_mut()
        .zip(src.as_slice())
        .for_each(|(out, &inp)| {
            *out = if inp > threshold {
                max_value
            } else {
                T::default()
            };
        });

    Ok(())
}

/// Apply an inverse binary threshold to an image.
///
/// # Arguments
///
/// * `src` - The input image of an arbitrary number of channels and type.
/// * `dst` - The output image of an arbitrary number of channels and type.///
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
/// use kornia::image::{Image, ImageSize};
/// use kornia::imgproc::threshold::threshold_binary_inverse;
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
pub fn threshold_binary_inverse<T, const CHANNELS: usize>(
    src: &Image<T, CHANNELS>,
    dst: &mut Image<T, CHANNELS>,
    threshold: T,
    max_value: T,
) -> Result<(), ImageError>
where
    T: SafeTensorType,
{
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.size().width,
            src.size().height,
            dst.size().width,
            dst.size().height,
        ));
    }

    dst.as_slice_mut()
        .iter_mut()
        .zip(src.as_slice())
        .for_each(|(out, &inp)| {
            *out = if inp > threshold {
                T::default()
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
/// use kornia::image::{Image, ImageSize};
/// use kornia::imgproc::threshold::threshold_truncate;
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
pub fn threshold_truncate<T, const CHANNELS: usize>(
    src: &Image<T, CHANNELS>,
    dst: &mut Image<T, CHANNELS>,
    threshold: T,
) -> Result<(), ImageError>
where
    T: SafeTensorType,
{
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.size().width,
            src.size().height,
            dst.size().width,
            dst.size().height,
        ));
    }

    // TODO: parallelize again
    dst.as_slice_mut()
        .iter_mut()
        .zip(src.as_slice())
        .for_each(|(out, inp)| {
            *out = if *inp > threshold { threshold } else { *inp };
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
/// use kornia::image::{Image, ImageSize};
/// use kornia::imgproc::threshold::threshold_to_zero;
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
pub fn threshold_to_zero<T, const CHANNELS: usize>(
    src: &Image<T, CHANNELS>,
    dst: &mut Image<T, CHANNELS>,
    threshold: T,
) -> Result<(), ImageError>
where
    T: SafeTensorType,
{
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.size().width,
            src.size().height,
            dst.size().width,
            dst.size().height,
        ));
    }

    // TODO: parallelize again
    dst.as_slice_mut()
        .iter_mut()
        .zip(src.as_slice())
        .for_each(|(out, &inp)| {
            *out = if inp > threshold { inp } else { T::default() };
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
/// use kornia::image::{Image, ImageSize};
/// use kornia::imgproc::threshold::threshold_to_zero_inverse;
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
pub fn threshold_to_zero_inverse<T, const CHANNELS: usize>(
    src: &Image<T, CHANNELS>,
    dst: &mut Image<T, CHANNELS>,
    threshold: T,
) -> Result<(), ImageError>
where
    T: SafeTensorType,
{
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.size().width,
            src.size().height,
            dst.size().width,
            dst.size().height,
        ));
    }

    // TODO: parallelize again
    dst.as_slice_mut()
        .iter_mut()
        .zip(src.as_slice())
        .for_each(|(out, &inp)| {
            *out = if inp > threshold { T::default() } else { inp };
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
/// Precondition: the input image must have the same number of channels as the bounds.
/// Precondition: the input image range must be 0-255.
///
/// # Examples
///
/// ```
/// use kornia::image::{Image, ImageSize};
/// use kornia::imgproc::threshold::in_range;
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
/// assert_eq!(thresholded.get_pixel(0, 0, 0).unwrap(), 255);
/// assert_eq!(thresholded.get_pixel(1, 0, 0).unwrap(), 0);
/// ```
pub fn in_range<T, const CHANNELS: usize>(
    src: &Image<T, CHANNELS>,
    dst: &mut Image<u8, 1>,
    lower_bound: &[T; CHANNELS],
    upper_bound: &[T; CHANNELS],
) -> Result<(), ImageError>
where
    T: SafeTensorType,
{
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.size().width,
            src.size().height,
            dst.size().width,
            dst.size().height,
        ));
    }

    src.as_slice()
        .chunks_exact(CHANNELS)
        .zip(dst.storage.as_mut_slice())
        .for_each(|(inp, out)| {
            let mut is_in_range = true;
            let mut i = 0;
            while is_in_range && i < CHANNELS {
                is_in_range &= inp[i] >= lower_bound[i] && inp[i] <= upper_bound[i];
                i += 1;
            }
            *out = if is_in_range { 255 } else { 0 };
        });

    Ok(())
}

// TODO: outsu, triangle

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

        thresholded
            .as_slice()
            .iter()
            .zip(data_expected.iter())
            .for_each(|(x, y)| {
                assert_eq!(x, y);
            });

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

        thresholded
            .as_slice()
            .iter()
            .zip(data_expected.iter())
            .for_each(|(x, y)| {
                assert_eq!(x, y);
            });

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

        thresholded
            .as_slice()
            .iter()
            .zip(data_expected.iter())
            .for_each(|(x, y)| {
                assert_eq!(x, y);
            });

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

        thresholded
            .as_slice()
            .iter()
            .zip(data_expected.iter())
            .for_each(|(x, y)| {
                assert_eq!(x, y);
            });

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

        thresholded
            .as_slice()
            .iter()
            .zip(data_expected.iter())
            .for_each(|(x, y)| {
                assert_eq!(x, y);
            });

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
}
