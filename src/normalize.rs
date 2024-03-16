use std::usize;

use crate::image::Image;
use anyhow::Result;

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
/// * `image` - The input image of shape (height, width, channels).
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
/// use kornia_rs::image::{Image, ImageSize};
///
/// let image_data = vec![0f32, 1.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 0.0, 1.0, 2.0, 3.0];
/// let image = Image::<f32, 3>::new(
///     ImageSize {
///         width: 2,
///         height: 2,
///     },
///     image_data,
/// )
/// .unwrap();
/// let image_normalized = kornia_rs::normalize::normalize_mean_std(
///     &image,
///     &[0.5, 1.0, 0.5],
///     &[1.0, 1.0, 1.0],
/// )
/// .unwrap();
/// assert_eq!(image_normalized.num_channels(), 3);
/// assert_eq!(image_normalized.size().width, 2);
/// assert_eq!(image_normalized.size().height, 2);
/// ```
pub fn normalize_mean_std<T, const CHANNELS: usize>(
    image: &Image<T, CHANNELS>,
    mean: &[T; CHANNELS],
    std: &[T; CHANNELS],
) -> Result<Image<T, CHANNELS>, std::io::Error>
where
    T: num_traits::Float + num_traits::FromPrimitive + std::fmt::Debug + Send + Sync + Copy,
{
    let mut output = ndarray::Array3::<T>::zeros(image.data.dim());

    ndarray::Zip::from(output.rows_mut())
        .and(image.data.rows())
        .par_for_each(|mut out, inp| {
            for i in 0..CHANNELS {
                out[i] = (inp[i] - mean[i]) / std[i];
            }
        });

    Ok(Image { data: output })
}

/// Find the minimum and maximum values in an image.
///
/// # Arguments
///
/// * `image` - The input image of shape (height, width, channels).
///
/// # Returns
///
/// A tuple containing the minimum and maximum values in the image.
///
/// # Errors
///
/// If the image is empty, an error is returned.
///
/// # Example
///
/// ```
/// use kornia_rs::image::{Image, ImageSize};
///
/// let image_data = vec![0u8, 1, 0, 1, 2, 3, 0, 1, 0, 1, 2, 3];
/// let image = Image::<u8, 3>::new(
///     ImageSize {
///         width: 2,
///         height: 2,
///     },
///     image_data,
/// )
/// .unwrap();
///
/// let (min, max) = kornia_rs::normalize::find_min_max(&image).unwrap();
/// assert_eq!(min, 0);
/// assert_eq!(max, 3);
/// ```
pub fn find_min_max<T: PartialOrd, const CHANNELS: usize>(
    image: &Image<T, CHANNELS>,
) -> Result<(T, T)>
where
    T: Copy,
{
    // get the first element in the image
    let first_element = match image.data.iter().next() {
        Some(x) => x,
        None => return Err(anyhow::anyhow!("Empty image")),
    };

    let mut min = first_element;
    let mut max = first_element;

    for x in image.data.iter() {
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
/// * `image` - The input image of shape (height, width, channels).
/// * `min` - The minimum value for each channel.
/// * `max` - The maximum value for each channel.
///
/// # Returns
///
/// The normalized image of shape (height, width, channels).
///
/// # Example
///
/// ```
/// use kornia_rs::image::{Image, ImageSize};
///
/// let image_data = vec![0.0f32, 1.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 0.0, 1.0, 2.0, 3.0];
/// let image = Image::<f32, 3>::new(
///     ImageSize {
///         width: 2,
///         height: 2,
///     },
///     image_data,
/// )
/// .unwrap();
/// let image_normalized = kornia_rs::normalize::normalize_min_max(&image, 0.0, 1.0).unwrap();
/// assert_eq!(image_normalized.num_channels(), 3);
/// assert_eq!(image_normalized.size().width, 2);
/// assert_eq!(image_normalized.size().height, 2);
/// ```
pub fn normalize_min_max<T, const CHANNELS: usize>(
    image: &Image<T, CHANNELS>,
    min: T,
    max: T,
) -> Result<Image<T, CHANNELS>>
where
    T: num_traits::Float
        + num_traits::FromPrimitive
        + std::fmt::Debug
        + Send
        + Sync
        + Copy
        + Default,
{
    let mut output = Image::<T, CHANNELS>::from_size_val(image.size(), T::default())?;

    let (min_val, max_val) = find_min_max(image)?;

    ndarray::Zip::from(output.data.rows_mut())
        .and(image.data.rows())
        .par_for_each(|mut out, inp| {
            for i in 0..image.num_channels() {
                out[i] = (inp[i] - min_val) * (max - min) / (max_val - min_val) + min;
            }
        });

    Ok(output)
}

#[cfg(test)]
mod tests {
    use crate::image::{Image, ImageSize};

    #[test]
    fn normalize_mean_std() {
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
        )
        .unwrap();
        let mean = [0.5, 1.0, 0.5];
        let std = [1.0, 1.0, 1.0];

        let normalized = super::normalize_mean_std(&image, &mean, &std).unwrap();

        assert_eq!(normalized.num_channels(), 3);
        assert_eq!(normalized.size().width, 2);
        assert_eq!(normalized.size().height, 2);

        normalized
            .data
            .iter()
            .zip(image_expected.iter())
            .for_each(|(a, b)| {
                assert!((a - b).abs() < 1e-6);
            });
    }

    #[test]
    fn find_min_max() {
        let image_data = vec![0u8, 1, 0, 1, 2, 3, 0, 1, 0, 1, 2, 3];
        let image = Image::<u8, 3>::new(
            ImageSize {
                width: 2,
                height: 2,
            },
            image_data,
        )
        .unwrap();

        let (min, max) = super::find_min_max(&image).unwrap();

        assert_eq!(min, 0);
        assert_eq!(max, 3);
    }

    #[test]
    fn normalize_min_max() {
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
        )
        .unwrap();

        let normalized = super::normalize_min_max(&image, 0.0, 1.0).unwrap();

        assert_eq!(normalized.num_channels(), 3);
        assert_eq!(normalized.size().width, 2);
        assert_eq!(normalized.size().height, 2);

        normalized
            .data
            .iter()
            .zip(image_expected.iter())
            .for_each(|(a, b)| {
                assert!((a - b).abs() < 1e-6);
            });
    }
}
