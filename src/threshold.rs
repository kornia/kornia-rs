use crate::image::Image;
use anyhow::Result;

/// Apply a binary threshold to an image.
///
/// # Arguments
///
/// * `image` - The input image of an arbitrary number of channels and type.
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
    image: &Image<T, CHANNELS>,
    threshold: T,
    max_value: T,
) -> Result<Image<T, CHANNELS>>
where
    T: Copy + Clone + Default + Send + Sync + std::cmp::PartialOrd,
{
    let mut output = Image::<T, CHANNELS>::from_size_val(image.size(), T::default())?;

    ndarray::Zip::from(&mut output.data)
        .and(&image.data)
        .par_for_each(|out, &inp| {
            *out = if inp > threshold {
                max_value
            } else {
                T::default()
            };
        });

    Ok(output)
}

/// Apply an inverse binary threshold to an image.
///
/// # Arguments
///
/// * `image` - The input image of an arbitrary number of channels and type.
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
    image: &Image<T, CHANNELS>,
    threshold: T,
    max_value: T,
) -> Result<Image<T, CHANNELS>>
where
    T: Copy + Clone + Default + Send + Sync + std::cmp::PartialOrd,
{
    let mut output = Image::<T, CHANNELS>::from_size_val(image.size(), T::default())?;

    ndarray::Zip::from(&mut output.data)
        .and(&image.data)
        .par_for_each(|out, &inp| {
            *out = if inp > threshold {
                T::default()
            } else {
                max_value
            };
        });

    Ok(output)
}

/// Apply a truncated threshold to an image.
///
/// # Arguments
///
/// * `image` - The input image of an arbitrary number of channels and type.
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
    image: &Image<T, CHANNELS>,
    threshold: T,
) -> Result<Image<T, CHANNELS>>
where
    T: Copy + Clone + Default + Send + Sync + std::cmp::PartialOrd,
{
    let mut output = Image::<T, CHANNELS>::from_size_val(image.size(), T::default()).unwrap();

    ndarray::Zip::from(&mut output.data)
        .and(&image.data)
        .par_for_each(|out, &inp| {
            *out = if inp > threshold { threshold } else { inp };
        });

    Ok(output)
}

/// Apply a threshold to an image, setting values below the threshold to zero.
///
/// # Arguments
///
/// * `image` - The input image of an arbitrary number of channels and type.
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
    image: &Image<T, CHANNELS>,
    threshold: T,
) -> Result<Image<T, CHANNELS>>
where
    T: Copy + Clone + Default + Send + Sync + std::cmp::PartialOrd,
{
    let mut output = Image::<T, CHANNELS>::from_size_val(image.size(), T::default()).unwrap();

    ndarray::Zip::from(&mut output.data)
        .and(&image.data)
        .par_for_each(|out, &inp| {
            *out = if inp > threshold { inp } else { T::default() };
        });

    Ok(output)
}

/// Apply a threshold to an image, setting values above the threshold to zero.
///
/// # Arguments
///
/// * `image` - The input image of an arbitrary number of channels and type.
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
    image: &Image<T, CHANNELS>,
    threshold: T,
) -> Result<Image<T, CHANNELS>>
where
    T: Copy + Clone + Default + Send + Sync + std::cmp::PartialOrd,
{
    let mut output = Image::<T, CHANNELS>::from_size_val(image.size(), T::default()).unwrap();

    ndarray::Zip::from(&mut output.data)
        .and(&image.data)
        .par_for_each(|out, &inp| {
            *out = if inp > threshold { T::default() } else { inp };
        });

    Ok(output)
}

// TODO: outsu, triangle

#[cfg(test)]
mod tests {
    use crate::image::{Image, ImageSize};

    #[test]
    fn threshold_binary() {
        let data = vec![100u8, 200, 50, 150, 200, 250];
        let data_expected = [0u8, 255, 0, 255, 255, 255];
        let image = Image::<_, 1>::new(
            ImageSize {
                width: 2,
                height: 3,
            },
            data,
        )
        .unwrap();

        let thresholded = super::threshold_binary(&image, 100, 255).unwrap();
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
    }

    #[test]
    fn threshold_binary_inverse() {
        let data = vec![100u8, 200, 50, 150, 200, 250];
        let data_expected = [255u8, 0, 255, 0, 0, 0];
        let image = Image::<_, 1>::new(
            ImageSize {
                width: 2,
                height: 3,
            },
            data,
        )
        .unwrap();

        let thresholded = super::threshold_binary_inverse(&image, 100, 255).unwrap();
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
    }

    #[test]
    fn threshold_truncate() {
        let data = vec![100u8, 200, 50, 150, 200, 250];
        let data_expected = [100u8, 150, 50, 150, 150, 150];
        let image = Image::<_, 1>::new(
            ImageSize {
                width: 2,
                height: 3,
            },
            data,
        )
        .unwrap();

        let thresholded = super::threshold_truncate(&image, 150).unwrap();
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
    }

    #[test]
    fn threshold_to_zero() {
        let data = vec![100u8, 200, 50, 150, 200, 250];
        let data_expected = [0u8, 200, 0, 0, 200, 250];
        let image = Image::<_, 3>::new(
            ImageSize {
                width: 2,
                height: 1,
            },
            data,
        )
        .unwrap();

        let thresholded = super::threshold_to_zero(&image, 150).unwrap();
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
    }

    #[test]
    fn threshold_to_zero_inverse() {
        let data = vec![100u8, 200, 50, 150, 200, 250];
        let data_expected = [100u8, 0, 50, 150, 0, 0];
        let image = Image::<_, 3>::new(
            ImageSize {
                width: 2,
                height: 1,
            },
            data,
        )
        .unwrap();

        let thresholded = super::threshold_to_zero_inverse(&image, 150).unwrap();
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
    }
}
