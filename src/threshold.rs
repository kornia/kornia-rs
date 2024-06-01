use crate::image::Image;
use anyhow::Result;

/// Apply a binary threshold to an image.
///
/// # Arguments
///
/// * `src` - The input image of an arbitrary number of channels and type.
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
/// use kornia_rs::image::{Image, ImageSize};
/// use kornia_rs::threshold::threshold_binary;
///
/// let data = vec![100u8, 200, 50, 150, 200, 250];
/// let image = Image::<_, 1>::new(ImageSize { width: 2, height: 3 }, data).unwrap();
///
/// let thresholded = threshold_binary(&image, 100, 255).unwrap();
/// assert_eq!(thresholded.num_channels(), 1);
/// assert_eq!(thresholded.size().width, 2);
/// assert_eq!(thresholded.size().height, 3);
/// ```
pub fn threshold_binary<T, const CHANNELS: usize>(
    src: &Image<T, CHANNELS>,
    threshold: T,
    max_value: T,
) -> Result<Image<T, CHANNELS>>
where
    T: Copy + Clone + Default + Send + Sync + std::cmp::PartialOrd,
{
    let mut dst = Image::<T, CHANNELS>::from_size_val(src.size(), T::default())?;

    ndarray::Zip::from(&mut dst.data)
        .and(&src.data)
        .par_for_each(|out, &inp| {
            *out = if inp > threshold {
                max_value
            } else {
                T::default()
            };
        });

    Ok(dst)
}

/// Apply an inverse binary threshold to an image.
///
/// # Arguments
///
/// * `src` - The input image of an arbitrary number of channels and type.
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
/// use kornia_rs::image::{Image, ImageSize};
/// use kornia_rs::threshold::threshold_binary_inverse;
///
/// let data = vec![100u8, 200, 50, 150, 200, 250];
/// let image = Image::<_, 1>::new(ImageSize { width: 2, height: 3 }, data).unwrap();
///
/// let thresholded = threshold_binary_inverse(&image, 100, 255).unwrap();
/// assert_eq!(thresholded.num_channels(), 1);
/// assert_eq!(thresholded.size().width, 2);
/// assert_eq!(thresholded.size().height, 3);
/// ```
pub fn threshold_binary_inverse<T, const CHANNELS: usize>(
    src: &Image<T, CHANNELS>,
    threshold: T,
    max_value: T,
) -> Result<Image<T, CHANNELS>>
where
    T: Copy + Clone + Default + Send + Sync + std::cmp::PartialOrd,
{
    let mut dst = Image::<T, CHANNELS>::from_size_val(src.size(), T::default())?;

    ndarray::Zip::from(&mut dst.data)
        .and(&src.data)
        .par_for_each(|out, &inp| {
            *out = if inp > threshold {
                T::default()
            } else {
                max_value
            };
        });

    Ok(dst)
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
/// use kornia_rs::image::{Image, ImageSize};
/// use kornia_rs::threshold::threshold_truncate;
///
/// let data = vec![100u8, 200, 50, 150, 200, 250];
/// let image = Image::<_, 1>::new(ImageSize { width: 2, height: 3 }, data).unwrap();
///
/// let thresholded = threshold_truncate(&image, 150).unwrap();
/// assert_eq!(thresholded.num_channels(), 1);
/// assert_eq!(thresholded.size().width, 2);
/// assert_eq!(thresholded.size().height, 3);
/// ```
pub fn threshold_truncate<T, const CHANNELS: usize>(
    src: &Image<T, CHANNELS>,
    threshold: T,
) -> Result<Image<T, CHANNELS>>
where
    T: Copy + Clone + Default + Send + Sync + std::cmp::PartialOrd,
{
    let mut dst = Image::<T, CHANNELS>::from_size_val(src.size(), T::default())?;

    ndarray::Zip::from(&mut dst.data)
        .and(&src.data)
        .par_for_each(|out, &inp| {
            *out = if inp > threshold { threshold } else { inp };
        });

    Ok(dst)
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
/// use kornia_rs::image::{Image, ImageSize};
/// use kornia_rs::threshold::threshold_to_zero;
///
/// let data = vec![100u8, 200, 50, 150, 200, 250];
/// let image = Image::<_, 3>::new(ImageSize { width: 2, height: 1 }, data).unwrap();
///
/// let thresholded = threshold_to_zero(&image, 150).unwrap();
/// assert_eq!(thresholded.num_channels(), 3);
/// assert_eq!(thresholded.size().width, 2);
/// assert_eq!(thresholded.size().height, 1);
/// ```
pub fn threshold_to_zero<T, const CHANNELS: usize>(
    src: &Image<T, CHANNELS>,
    threshold: T,
) -> Result<Image<T, CHANNELS>>
where
    T: Copy + Clone + Default + Send + Sync + std::cmp::PartialOrd,
{
    let mut dst = Image::<T, CHANNELS>::from_size_val(src.size(), T::default())?;

    ndarray::Zip::from(&mut dst.data)
        .and(&src.data)
        .par_for_each(|out, &inp| {
            *out = if inp > threshold { inp } else { T::default() };
        });

    Ok(dst)
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
/// use kornia_rs::image::{Image, ImageSize};
/// use kornia_rs::threshold::threshold_to_zero_inverse;
///
/// let data = vec![100u8, 200, 50, 150, 200, 250];
/// let image = Image::<_, 3>::new(ImageSize { width: 2, height: 1 }, data).unwrap();
///
/// let thresholded = threshold_to_zero_inverse(&image, 150).unwrap();
/// assert_eq!(thresholded.num_channels(), 3);
/// assert_eq!(thresholded.size().width, 2);
/// assert_eq!(thresholded.size().height, 1);
/// ```
pub fn threshold_to_zero_inverse<T, const CHANNELS: usize>(
    src: &Image<T, CHANNELS>,
    threshold: T,
) -> Result<Image<T, CHANNELS>>
where
    T: Copy + Clone + Default + Send + Sync + std::cmp::PartialOrd,
{
    let mut dst = Image::<T, CHANNELS>::from_size_val(src.size(), T::default())?;

    ndarray::Zip::from(&mut dst.data)
        .and(&src.data)
        .par_for_each(|out, &inp| {
            *out = if inp > threshold { T::default() } else { inp };
        });

    Ok(dst)
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
/// use kornia_rs::image::{Image, ImageSize};
/// use kornia_rs::threshold::in_range;
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
/// let thresholded = in_range(&image, &[100, 150, 0], &[200, 200, 200]).unwrap();
/// assert_eq!(thresholded.num_channels(), 1);
/// assert_eq!(thresholded.size().width, 2);
///
/// assert_eq!(thresholded.get_pixel(0, 0, 0).unwrap(), 255);
/// assert_eq!(thresholded.get_pixel(1, 0, 0).unwrap(), 0);
/// ```
pub fn in_range<T, const CHANNELS: usize>(
    src: &Image<T, CHANNELS>,
    lower_bound: &[T; CHANNELS],
    upper_bound: &[T; CHANNELS],
) -> Result<Image<u8, 1>>
where
    T: Sync + std::cmp::PartialOrd,
{
    let mut dst = Image::from_size_val(src.size(), 0)?;

    ndarray::Zip::from(dst.data.rows_mut())
        .and(src.data.rows())
        .par_for_each(|mut out, inp| {
            let mut is_in_range = true;
            let mut i = 0;
            while is_in_range && i < CHANNELS {
                is_in_range &= inp[i] >= lower_bound[i] && inp[i] <= upper_bound[i];
                i += 1;
            }
            out[0] = if is_in_range { 255 } else { 0 };
        });

    Ok(dst)
}

// TODO: outsu, triangle

#[cfg(test)]
mod tests {
    use crate::image::{Image, ImageSize};
    use anyhow::Result;

    #[test]
    fn threshold_binary() -> Result<()> {
        let data = vec![100u8, 200, 50, 150, 200, 250];
        let data_expected = [0u8, 255, 0, 255, 255, 255];
        let image = Image::<_, 1>::new(
            ImageSize {
                width: 2,
                height: 3,
            },
            data,
        )?;

        let thresholded = super::threshold_binary(&image, 100, 255)?;
        assert_eq!(thresholded.num_channels(), 1);
        assert_eq!(thresholded.size().width, 2);
        assert_eq!(thresholded.size().height, 3);

        thresholded
            .data
            .iter()
            .zip(data_expected.iter())
            .for_each(|(x, y)| {
                assert_eq!(x, y);
            });

        Ok(())
    }

    #[test]
    fn threshold_binary_inverse() -> Result<()> {
        let data = vec![100u8, 200, 50, 150, 200, 250];
        let data_expected = [255u8, 0, 255, 0, 0, 0];
        let image = Image::<_, 1>::new(
            ImageSize {
                width: 2,
                height: 3,
            },
            data,
        )?;

        let thresholded = super::threshold_binary_inverse(&image, 100, 255)?;
        assert_eq!(thresholded.num_channels(), 1);
        assert_eq!(thresholded.size().width, 2);
        assert_eq!(thresholded.size().height, 3);

        thresholded
            .data
            .iter()
            .zip(data_expected.iter())
            .for_each(|(x, y)| {
                assert_eq!(x, y);
            });

        Ok(())
    }

    #[test]
    fn threshold_truncate() -> Result<()> {
        let data = vec![100u8, 200, 50, 150, 200, 250];
        let data_expected = [100u8, 150, 50, 150, 150, 150];
        let image = Image::<_, 1>::new(
            ImageSize {
                width: 2,
                height: 3,
            },
            data,
        )?;

        let thresholded = super::threshold_truncate(&image, 150)?;
        assert_eq!(thresholded.num_channels(), 1);
        assert_eq!(thresholded.size().width, 2);
        assert_eq!(thresholded.size().height, 3);

        thresholded
            .data
            .iter()
            .zip(data_expected.iter())
            .for_each(|(x, y)| {
                assert_eq!(x, y);
            });

        Ok(())
    }

    #[test]
    fn threshold_to_zero() -> Result<()> {
        let data = vec![100u8, 200, 50, 150, 200, 250];
        let data_expected = [0u8, 200, 0, 0, 200, 250];
        let image = Image::<_, 3>::new(
            ImageSize {
                width: 2,
                height: 1,
            },
            data,
        )?;

        let thresholded = super::threshold_to_zero(&image, 150)?;
        assert_eq!(thresholded.num_channels(), 3);
        assert_eq!(thresholded.size().width, 2);
        assert_eq!(thresholded.size().height, 1);

        thresholded
            .data
            .iter()
            .zip(data_expected.iter())
            .for_each(|(x, y)| {
                assert_eq!(x, y);
            });

        Ok(())
    }

    #[test]
    fn threshold_to_zero_inverse() -> Result<()> {
        let data = vec![100u8, 200, 50, 150, 200, 250];
        let data_expected = [100u8, 0, 50, 150, 0, 0];
        let image = Image::<_, 3>::new(
            ImageSize {
                width: 2,
                height: 1,
            },
            data,
        )?;

        let thresholded = super::threshold_to_zero_inverse(&image, 150)?;
        assert_eq!(thresholded.num_channels(), 3);
        assert_eq!(thresholded.size().width, 2);
        assert_eq!(thresholded.size().height, 1);

        thresholded
            .data
            .iter()
            .zip(data_expected.iter())
            .for_each(|(x, y)| {
                assert_eq!(x, y);
            });

        Ok(())
    }

    #[test]
    fn test_in_range() -> Result<()> {
        let data = vec![100u8, 200, 50, 150, 200, 250];
        let image = Image::<_, 3>::new(
            ImageSize {
                width: 2,
                height: 1,
            },
            data,
        )?;

        let thresholded = super::in_range(&image, &[100, 150, 0], &[200, 200, 200])?;
        assert_eq!(thresholded.num_channels(), 1);
        assert_eq!(thresholded.size().width, 2);
        assert_eq!(thresholded.size().height, 1);

        assert_eq!(thresholded.get_pixel(0, 0, 0)?, 255);
        assert_eq!(thresholded.get_pixel(1, 0, 0)?, 0);

        Ok(())
    }
}
