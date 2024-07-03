use crate::image::Image;
use anyhow::Result;

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
///
/// # Panics
///
/// This function will panic if the sizes of `src1` and `src2` do not match.
pub fn add_weighted<T, const CHANNELS: usize>(
    src1: &Image<T, CHANNELS>,
    alpha: T,
    src2: &Image<T, CHANNELS>,
    beta: T,
    gamma: T,
) -> Result<Image<T, CHANNELS>>
where
    T: num_traits::Float + num_traits::FromPrimitive + std::fmt::Debug + Send + Sync + Copy,
{
    if src1.size() != src2.size() {
        return Err(anyhow::anyhow!(
            "The shape of `src1` and `src2` should be identical"
        ));
    }

    let mut dst = ndarray::Array3::<T>::zeros(src1.data.dim());

    ndarray::Zip::from(dst.rows_mut())
        .and(src1.data.rows())
        .and(src2.data.rows())
        .for_each(|mut dst_pixel, src1_pixels, src2_pixels| {
            for i in 0..CHANNELS {
                dst_pixel[i] = (src1_pixels[i] * alpha) + (src2_pixels[i] * beta) + gamma;
            }
        });

    Ok(Image { data: dst })
}

#[cfg(test)]
mod tests {
    use crate::image::{Image, ImageSize};
    use anyhow::Result;

    #[test]
    fn test_add_weighted() -> Result<()> {
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

        let weighted = super::add_weighted(&src1, alpha, &src2, beta, gamma)?;

        weighted
            .data
            .iter()
            .zip(expected.iter())
            .for_each(|(a, b)| {
                assert!((a - b).abs() < 1e-6);
            });

        Ok(())
    }
}
