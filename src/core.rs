// reference: https://www.strchr.com/standard_deviation_in_one_pass
use crate::image::Image;
use anyhow::Result;

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

/// Perform a bitwise AND operation between two images using a mask.
///
/// The mask is a binary image where the value 0 is considered as False
/// and any other value is considered as True.
///
/// # Arguments
///
/// * `src1` - The first input image.
/// * `src2` - The second input image.
/// * `mask` - The binary mask to apply to the image.
///
/// # Returns
///
/// The output image after applying the mask.
///
/// # Example
///
/// ```
/// use kornia_rs::image::{Image, ImageSize};
///
/// let image = Image::<u8, 3>::new(
///    ImageSize {
///        width: 2,
///        height: 2,
///    },
///    vec![0, 1, 2, 253, 254, 255, 128, 129, 130, 64, 65, 66],
/// ).unwrap();
///
/// let mask = Image::<u8, 1>::new(
///    ImageSize {
///        width: 2,
///        height: 2,
///    },
///    vec![255, 0, 255, 0],
/// ).unwrap();
///
/// let output = kornia_rs::core::bitwise_and(&image, &image, &mask).unwrap();
///
/// assert_eq!(output.size().width, 2);
/// assert_eq!(output.size().height, 2);
///
/// assert_eq!(output.data.as_slice().unwrap(), &vec![0, 1, 2, 0, 0, 0, 128, 129, 130, 0, 0, 0]);
/// ```
pub fn bitwise_and<const CHANNELS: usize>(
    src1: &Image<u8, CHANNELS>,
    src2: &Image<u8, CHANNELS>,
    mask: &Image<u8, 1>,
) -> Result<Image<u8, CHANNELS>> {
    assert!(
        src1.size() == src2.size(),
        "The input images must have the same size",
    );
    assert!(
        src1.size() == mask.size(),
        "The input images and the mask must have the same size"
    );

    // prepare the output image

    let mut dst = Image::from_size_val(src1.size(), 0)?;

    // apply the mask to the image

    ndarray::Zip::from(dst.data.rows_mut())
        .and(src1.data.rows())
        .and(src2.data.rows())
        .and(mask.data.rows())
        .par_for_each(|mut out, inp1, inp2, msk| {
            for c in 0..CHANNELS {
                out[c] = if msk[0] != 0 { inp1[c] & inp2[c] } else { 0 };
            }
        });

    Ok(dst)
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

    #[test]
    fn test_bitwise_and() -> Result<()> {
        let image = Image::<u8, 3>::new(
            ImageSize {
                width: 2,
                height: 2,
            },
            vec![0, 1, 2, 253, 254, 255, 128, 129, 130, 64, 65, 66],
        )?;

        let mask = Image::<u8, 1>::new(
            ImageSize {
                width: 2,
                height: 2,
            },
            vec![255, 0, 255, 0],
        )?;

        let output = super::bitwise_and(&image, &image, &mask)?;

        assert_eq!(output.size().width, 2);
        assert_eq!(output.size().height, 2);
        assert_eq!(output.num_channels(), 3);

        assert_eq!(
            output.data.as_slice().unwrap(),
            &vec![0, 1, 2, 0, 0, 0, 128, 129, 130, 0, 0, 0]
        );
        Ok(())
    }
}
