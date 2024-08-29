use kornia_core::SafeTensorType;
use kornia_image::{Image, ImageError};

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
pub fn add_weighted<T, const CHANNELS: usize>(
    src1: &Image<T, CHANNELS>,
    alpha: T,
    src2: &Image<T, CHANNELS>,
    dst: &mut Image<T, CHANNELS>,
    beta: T,
    gamma: T,
) -> Result<(), ImageError>
where
    T: num_traits::Float
        + num_traits::FromPrimitive
        + std::fmt::Debug
        + Send
        + Sync
        + Copy
        + SafeTensorType,
{
    if src1.size() != src2.size() {
        return Err(ImageError::InvalidImageSize(
            src1.width(),
            src1.height(),
            src2.width(),
            src2.height(),
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

    let src1_data = unsafe {
        ndarray::ArrayView3::from_shape_ptr(
            (src1.height(), src1.width(), src1.num_channels()),
            src1.as_ptr(),
        )
    };

    let src2_data = unsafe {
        ndarray::ArrayView3::from_shape_ptr(
            (src2.height(), src2.width(), src2.num_channels()),
            src2.as_ptr(),
        )
    };

    let dst_data = unsafe {
        ndarray::ArrayView3::from_shape_ptr(
            (dst.height(), dst.width(), dst.num_channels()),
            dst.as_ptr(),
        )
    };
    let mut dst_data = dst_data.to_owned();

    ndarray::Zip::from(dst_data.rows_mut())
        .and(src1_data.rows())
        .and(src2_data.rows())
        .for_each(|mut dst_pixel, src1_pixels, src2_pixels| {
            for i in 0..CHANNELS {
                dst_pixel[i] = (src1_pixels[i] * alpha) + (src2_pixels[i] * beta) + gamma;
            }
        });

    dst.as_slice_mut()
        .copy_from_slice(dst_data.as_slice().unwrap());

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

        super::add_weighted(&src1, alpha, &src2, &mut weighted, beta, gamma)?;

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
