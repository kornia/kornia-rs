use crate::parallel;

use super::interpolate::interpolate_pixel;
use super::InterpolationMode;
use kornia_image::{Image, ImageError};
use kornia_tensor::{CpuAllocator, Tensor2};

/// Apply generic geometric transformation to an image.
///
/// # Arguments
///
/// * `src` - The input image container with shape (height, width, C).
/// * `dst` - The output image container with shape (height, width, C).
/// * `map_x` - The x coordinates of the pixels to interpolate.
/// * `map_y` - The y coordinates of the pixels to interpolate.
/// * `interpolation` - The interpolation mode to use.
///
/// # Errors
///
/// * The mapx and mapy must have the same size.
/// * The output image must have the same size as the mapx and mapy.
pub fn remap<const C: usize>(
    src: &Image<f32, C>,
    dst: &mut Image<f32, C>,
    map_x: &Tensor2<f32, CpuAllocator>,
    map_y: &Tensor2<f32, CpuAllocator>,
    interpolation: InterpolationMode,
) -> Result<(), ImageError> {
    if map_x.shape != map_y.shape {
        return Err(ImageError::InvalidImageSize(
            map_x.shape[0],
            map_x.shape[1],
            map_y.shape[0],
            map_y.shape[1],
        ));
    }

    if dst.shape[0..2] != map_x.shape {
        return Err(ImageError::InvalidImageSize(
            src.shape[0],
            src.shape[1],
            dst.shape[0],
            dst.shape[1],
        ));
    }

    // parallelize the remap operation by rows
    parallel::par_iter_rows_resample(dst, map_x, map_y, |&x, &y, dst_pixel| {
        // interpolate the pixel value
        dst_pixel.iter_mut().enumerate().for_each(|(c, pixel)| {
            *pixel = interpolate_pixel(src, x, y, c, interpolation);
        });
    });

    Ok(())
}

#[cfg(test)]
mod tests {
    use kornia_image::{Image, ImageError, ImageSize};
    use kornia_tensor::{CpuAllocator, Tensor2};

    #[test]
    fn remap_smoke() -> Result<(), ImageError> {
        let image = Image::<_, 1>::new(
            ImageSize {
                width: 3,
                height: 3,
            },
            vec![0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )?;

        let new_size = [2, 2];

        let map_x = Tensor2::from_shape_vec(new_size, vec![0.0, 2.0, 0.0, 2.0], CpuAllocator)?;
        let map_y = Tensor2::from_shape_vec(new_size, vec![0.0, 0.0, 2.0, 2.0], CpuAllocator)?;

        let expected = Image::<_, 1>::new(
            ImageSize {
                width: 2,
                height: 2,
            },
            vec![0.0, 2.0, 6.0, 8.0],
        )?;

        let mut image_transformed = Image::<_, 1>::from_size_val(new_size.into(), 0.0)?;

        super::remap(
            &image,
            &mut image_transformed,
            &map_x,
            &map_y,
            super::InterpolationMode::Bilinear,
        )?;

        assert_eq!(image_transformed.num_channels(), 1);
        assert_eq!(image_transformed.size().width, 2);
        assert_eq!(image_transformed.size().height, 2);

        for (a, b) in image_transformed
            .as_slice()
            .iter()
            .zip(expected.as_slice().iter())
        {
            assert!((a - b).abs() < 1e-6);
        }

        Ok(())
    }
}
