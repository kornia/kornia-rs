// reference: https://www.strchr.com/standard_deviation_in_one_pass
use crate::image::Image;

/// Compute the mean and standard deviation of an image.
///
/// The mean and standard deviation are computed for each channel
/// of the image in one pass.
///
/// # Arguments
///
/// * `image` - The input image to compute the mean and standard deviation.
///
/// # Returns
///
/// A tuple containing the mean and standard deviation of the image.
/// The first element of the tuple is the standard deviation and the
/// second element is the mean.
///
/// # Example
///
/// ```
/// use kornia_rs::image::{Image, ImageSize};
///
/// let image = Image::<u8, 3>::new(
///    ImageSize {
///      width: 2,
///      height: 2,
///  },
/// vec![0, 1, 2, 253, 254, 255, 128, 129, 130, 64, 65, 66],
/// ).unwrap();
///
/// let (std, mean) = kornia_rs::core::std_mean(&image);
/// assert_eq!(std, [93.5183805462862, 93.5183805462862, 93.5183805462862]);
/// assert_eq!(mean, [111.25, 112.25, 113.25]);
/// ```
pub fn std_mean(image: &Image<u8, 3>) -> (Vec<f64>, Vec<f64>) {
    let (sum, sq_sum) = image.data.indexed_iter().fold(
        ([0f64; 3], [0f64; 3]),
        |(mut sum, mut sq_sum), ((_, _, c), val)| {
            sum[c] += *val as f64;
            sq_sum[c] += (*val as f64).powi(2);
            (sum, sq_sum)
        },
    );

    let n = (image.width() * image.height()) as f64;
    let mean = sum.iter().map(|&s| s / n).collect::<Vec<_>>();

    let variance = sq_sum
        .iter()
        .zip(mean.iter())
        .map(|(&sq_s, &m)| (sq_s / n - m.powi(2)).sqrt())
        .collect::<Vec<_>>();

    (variance, mean)
}

#[cfg(test)]
mod tests {
    use crate::image::{Image, ImageSize};
    use anyhow::Result;

    #[test]
    fn test_std_mean() -> Result<()> {
        let image = Image::<u8, 3>::new(
            ImageSize {
                width: 2,
                height: 2,
            },
            vec![0, 1, 2, 253, 254, 255, 128, 129, 130, 64, 65, 66],
        )?;

        let (std, mean) = super::std_mean(&image);
        assert_eq!(std, [93.5183805462862, 93.5183805462862, 93.5183805462862]);
        assert_eq!(mean, [111.25, 112.25, 113.25]);
        Ok(())
    }
}
