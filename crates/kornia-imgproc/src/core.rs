// reference: https://www.strchr.com/standard_deviation_in_one_pass
use kornia_image::{Image, ImageError};
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::{ParallelSlice, ParallelSliceMut},
};

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
/// use kornia_image::{Image, ImageSize};
/// use kornia_imgproc::core::std_mean;
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
pub fn std_mean(image: &Image<u8, 3>) -> ([f64; 3], [f64; 3]) {
    let (sum, sq_sum) = image.as_slice().chunks_exact(3).fold(
        ([0f64; 3], [0f64; 3]),
        |(mut sum, mut sq_sum), pixel| {
            sum.iter_mut()
                .zip(pixel.iter())
                .for_each(|(s, &p)| *s += p as f64);
            sq_sum
                .iter_mut()
                .zip(pixel.iter())
                .for_each(|(s, &p)| *s += (p as f64).powi(2));
            (sum, sq_sum)
        },
    );

    let n = (image.width() * image.height()) as f64;
    let mean = [sum[0] / n, sum[1] / n, sum[2] / n];

    let variance = [
        (sq_sum[0] / n - mean[0].powi(2)).sqrt(),
        (sq_sum[1] / n - mean[1].powi(2)).sqrt(),
        (sq_sum[2] / n - mean[2].powi(2)).sqrt(),
    ];

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
/// use kornia_image::{Image, ImageSize};
/// use kornia_imgproc::core::bitwise_and;
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
/// assert_eq!(output.as_slice(), &vec![0, 1, 2, 0, 0, 0, 128, 129, 130, 0, 0, 0]);
/// ```
pub fn bitwise_and<const C: usize>(
    src1: &Image<u8, C>,
    src2: &Image<u8, C>,
    dst: &mut Image<u8, C>,
    mask: &Image<u8, 1>,
) -> Result<(), ImageError> {
    if src1.size() != src2.size() {
        return Err(ImageError::InvalidImageSize(
            src1.cols(),
            src1.rows(),
            src2.cols(),
            src2.rows(),
        ));
    }

    if src1.size() != mask.size() {
        return Err(ImageError::InvalidImageSize(
            src1.cols(),
            src1.rows(),
            mask.cols(),
            mask.rows(),
        ));
    }

    if src1.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src1.cols(),
            src1.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }

    // apply the mask to the image

    dst.as_slice_mut()
        .chunks_exact_mut(C)
        .zip(src1.as_slice().chunks_exact(C))
        .zip(src2.as_slice().chunks_exact(C))
        .zip(mask.as_slice().iter())
        .for_each(|(((dst_chunk, src1_chunk), src2_chunk), &mask_chunk)| {
            dst_chunk
                .iter_mut()
                .zip(src1_chunk.iter().zip(src2_chunk.iter()))
                .for_each(|(dst_pixel, (src1_pixel, src2_pixel))| {
                    *dst_pixel = if mask_chunk != 0 {
                        src1_pixel & src2_pixel
                    } else {
                        0
                    };
                });
        });

    Ok(())
}

/// Concatenate images horizontally from a vector of images into a destination image
///
/// # Arguments
///
/// * `src` - The vector of images to concatenate.
/// * `dst` - The destination image.
///
/// Precondition: all images must have the same height
/// Precondition: the output image must have enough space to store the concatenated images
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_imgproc::core::hconcat;
///
/// let image1 = Image::<u8, 3>::new(
///     ImageSize {
///         width: 2,
///         height: 2,
///     },
///     vec![0, 1, 2, 253, 254, 255, 128, 129, 130, 64, 65, 66],
/// ).unwrap();
///
/// let image2 = Image::<u8, 3>::new(
///     ImageSize {
///         width: 2,
///         height: 2,
///     },
///     vec![128, 129, 130, 64, 65, 66, 253, 254, 255, 0, 1, 2],
/// ).unwrap();
///
/// let mut output = Image::<u8, 3>::from_size_val(
///     ImageSize {
///         width: 4,
///         height: 2,
///     },
///     0,
/// ).unwrap();
///
/// hconcat(vec![&image1, &image2], &mut output).unwrap();
/// ```
pub fn hconcat<const C: usize>(
    src: Vec<&Image<u8, C>>,
    dst: &mut Image<u8, C>,
) -> Result<(), ImageError> {
    // check that all images have the same height
    let mut count_cols = 0;
    for img in src.iter() {
        if img.rows() != dst.rows() {
            return Err(ImageError::InvalidImageSize(
                img.cols(),
                img.rows(),
                dst.cols(),
                dst.rows(),
            ));
        }
        count_cols += img.cols();
    }

    if count_cols > dst.cols() {
        return Err(ImageError::InvalidImageSize(
            count_cols,
            dst.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }

    let dst_cols = dst.cols() * C;

    // Copy each source image into the destination image
    for (i, img) in src.iter().enumerate() {
        let img_cols = img.cols() * C;
        dst.as_slice_mut()
            .par_chunks_exact_mut(dst_cols)
            .zip_eq(img.as_slice().par_chunks_exact(img_cols))
            .for_each(|(dst_row, src_row)| {
                dst_row[i * img_cols..(i + 1) * img_cols].copy_from_slice(src_row);
            });
    }

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

        let std_expected = [93.5183805462862, 93.5183805462862, 93.5183805462862];
        let mean_expected = [111.25, 112.25, 113.25];

        let (std, mean) = super::std_mean(&image);
        assert_eq!(std, std_expected);
        assert_eq!(mean, mean_expected);
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

    #[test]
    fn test_hconcat() -> Result<(), ImageError> {
        #[rustfmt::skip]
        let image1 = Image::<u8, 3>::new(
            ImageSize {
                width: 2,
                height: 2,
            },
            vec![
                0, 1, 2,
                253, 254, 255,
                128, 129, 130,
                64, 65, 66,
            ],
        )?;

        #[rustfmt::skip]
        let image2 = Image::<u8, 3>::new(
            ImageSize {
                width: 2,
                height: 2,
            },
            vec![
                128, 129, 130,
                64, 65, 66,
                253, 254, 255,
                0, 1, 2,
            ],
        )?;

        #[rustfmt::skip]
        let expected = vec![
            0, 1, 2, 253, 254, 255,
            128, 129, 130, 64, 65, 66,
            128, 129, 130, 64, 65, 66,
            253, 254, 255, 0, 1, 2,
        ];

        let mut output = Image::<u8, 3>::from_size_val(
            ImageSize {
                width: 4,
                height: 2,
            },
            0,
        )?;

        super::hconcat(vec![&image1, &image2], &mut output)?;

        assert_eq!(output.size().width, 4);
        assert_eq!(output.size().height, 2);
        assert_eq!(output.num_channels(), 3);

        assert_eq!(output.as_slice(), expected);
        Ok(())
    }
}
