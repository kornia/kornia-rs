use kornia_image::{Image, ImageError};

use crate::parallel;

/// Performs weighted addition of two images `src1` and `src2` with weights `alpha`
/// and `beta`, and an optional scalar `gamma`. The formula used is:
///
/// dst(x,y,c) = (src1(x,y,c) * alpha + src2(x,y,c) * beta + gamma)
///
/// # Arguments
///
/// * `src1` - The first input image.
/// * `alpha` - Weight of the first image elements to be multiplied.
/// * `src2` - The second input image.
/// * `beta` - Weight of the second image elements to be multiplied.
/// * `gamma` - Scalar added to each sum.
///
/// # Returns
///
/// Returns a new `Image` where each element is computed as described above.
///
/// # Errors
///
/// Returns an error if the sizes of `src1` and `src2` do not match.
/// Returns an error if the size of `dst` does not match the size of `src1` or `src2`.
pub fn add_weighted<T, const C: usize>(
    src1: &Image<T, C>,
    alpha: T,
    src2: &Image<T, C>,
    beta: T,
    gamma: T,
    dst: &mut Image<T, C>,
) -> Result<(), ImageError>
where
    T: num_traits::Float + num_traits::FromPrimitive + std::fmt::Debug + Send + Sync + Copy,
{
    if src1.size() != src2.size() {
        return Err(ImageError::InvalidImageSize(
            src1.cols(),
            src1.rows(),
            src2.cols(),
            src2.rows(),
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

    // compute the weighted sum
    parallel::par_iter_rows_val_two(src1, src2, dst, |&src1_pixel, &src2_pixel, dst_pixel| {
        *dst_pixel = (src1_pixel * alpha) + (src2_pixel * beta) + gamma;
    });

    Ok(())
}

#[cfg(test)]
mod tests {
    use kornia_image::{Image, ImageError, ImageSize};

    #[test]
    fn test_add_weighted() -> Result<(), ImageError> {
        let src1_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let src1 = Image::<f32, 1>::new(
            ImageSize {
                width: 2,
                height: 2,
            },
            src1_data,
        )?;
        let src2_data = vec![4.0f32, 5.0, 6.0, 7.0];
        let src2 = Image::<f32, 1>::new(
            ImageSize {
                width: 2,
                height: 2,
            },
            src2_data,
        )?;
        let alpha = 2.0f32;
        let beta = 2.0f32;
        let gamma = 1.0f32;
        let expected = [11.0, 15.0, 19.0, 23.0];

        let mut weighted = Image::<f32, 1>::from_size_val(src1.size(), 0.0)?;

        super::add_weighted(&src1, alpha, &src2, beta, gamma, &mut weighted)?;

        weighted
            .as_slice()
            .iter()
            .zip(expected.iter())
            .for_each(|(a, b)| {
                assert!((a - b).abs() < 1e-6);
            });

        Ok(())
    }
}
