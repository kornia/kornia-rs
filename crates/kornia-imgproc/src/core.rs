// reference: https://www.strchr.com/standard_deviation_in_one_pass
use kornia_image::{Image, ImageError};

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
/// use kornia::image::{Image, ImageSize};
/// use kornia::imgproc::core::std_mean;
///
/// let image = Image::<u8, 3>::new(
///    ImageSize {
///      width: 2,
///      height: 2,
///  },
/// vec![0, 1, 2, 253, 254, 255, 128, 129, 130, 64, 65, 66],
/// ).unwrap();
///
/// let (std, mean) = std_mean(&image);
///
/// assert_eq!(std, [93.5183805462862, 93.5183805462862, 93.5183805462862]);
/// assert_eq!(mean, [111.25, 112.25, 113.25]);
/// ```
pub fn std_mean(image: &Image<u8, 3>) -> (Vec<f64>, Vec<f64>) {
    let image_data = unsafe {
        ndarray::ArrayView3::from_shape_ptr(
            (
                image.height() as usize,
                image.width() as usize,
                image.num_channels(),
            ),
            image.as_ptr() as *const u8,
        )
    };

    let (sum, sq_sum) = image_data.indexed_iter().fold(
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
/// * `dst` - The output image.
/// * `mask` - The binary mask to apply to the image.
///
/// # Returns
///
/// The output image after applying the mask.
///
/// # Example
///
/// ```
/// use kornia::image::{Image, ImageSize};
/// use kornia::imgproc::core::bitwise_and;
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
/// let mut output = Image::<u8, 3>::from_size_val(image.size(), 0).unwrap();
///
/// bitwise_and(&image, &image, &mut output, &mask).unwrap();
///
/// assert_eq!(output.size().width, 2);
/// assert_eq!(output.size().height, 2);
///
/// assert_eq!(output.data.as_slice().unwrap(), &vec![0, 1, 2, 0, 0, 0, 128, 129, 130, 0, 0, 0]);
/// ```
pub fn bitwise_and<const CHANNELS: usize>(
    src1: &Image<u8, CHANNELS>,
    src2: &Image<u8, CHANNELS>,
    dst: &mut Image<u8, CHANNELS>,
    mask: &Image<u8, 1>,
) -> Result<(), ImageError> {
    if src1.size() != src2.size() {
        return Err(ImageError::InvalidImageSize(
            src1.width(),
            src1.height(),
            src2.width(),
            src2.height(),
        ));
    }

    if src1.size() != mask.size() {
        return Err(ImageError::InvalidImageSize(
            src1.width(),
            src1.height(),
            mask.width(),
            mask.height(),
        ));
    }

    if src1.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src1.width(),
            src1.height(),
            dst.width(),
            dst.height(),
        ));
    }

    // apply the mask to the image

    let src1_data = unsafe {
        ndarray::ArrayView3::from_shape_ptr(
            (
                src1.height() as usize,
                src1.width() as usize,
                src1.num_channels(),
            ),
            src1.as_ptr() as *const u8,
        )
    };

    let src2_data = unsafe {
        ndarray::ArrayView3::from_shape_ptr(
            (
                src2.height() as usize,
                src2.width() as usize,
                src2.num_channels(),
            ),
            src2.as_ptr() as *const u8,
        )
    };

    let dst_data = unsafe {
        ndarray::ArrayView3::from_shape_ptr(
            (
                dst.height() as usize,
                dst.width() as usize,
                dst.num_channels(),
            ),
            dst.as_ptr() as *mut u8,
        )
    };
    let mut dst_data = dst_data.to_owned();

    let mask_data = unsafe {
        ndarray::ArrayView3::from_shape_ptr(
            (
                mask.height() as usize,
                mask.width() as usize,
                mask.num_channels(),
            ),
            mask.as_ptr() as *const u8,
        )
    };

    ndarray::Zip::from(dst_data.rows_mut())
        .and(src1_data.rows())
        .and(src2_data.rows())
        .and(mask_data.rows())
        .par_for_each(|mut out, inp1, inp2, msk| {
            for c in 0..CHANNELS {
                out[c] = if msk[0] != 0 { inp1[c] & inp2[c] } else { 0 };
            }
        });

    Ok(())
}

#[cfg(test)]
mod tests {
    use kornia_image::{Image, ImageError, ImageSize};

    #[test]
    fn test_std_mean() -> Result<(), ImageError> {
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
    fn test_bitwise_and() -> Result<(), ImageError> {
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

        let mut output = Image::<u8, 3>::from_size_val(image.size(), 0)?;

        super::bitwise_and(&image, &image, &mut output, &mask)?;

        assert_eq!(output.size().width, 2);
        assert_eq!(output.size().height, 2);
        assert_eq!(output.num_channels(), 3);

        assert_eq!(
            output.as_slice(),
            vec![0, 1, 2, 0, 0, 0, 128, 129, 130, 0, 0, 0]
        );
        Ok(())
    }
}
