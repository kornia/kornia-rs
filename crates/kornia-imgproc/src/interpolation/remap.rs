use super::interpolate::interpolate_pixel;
use super::InterpolationMode;
use kornia_image::{Image, ImageError};

/// Apply generic geometric transformation to an image.
///
/// # Arguments
///
/// * `src` - The input image container with shape (height, width, channels).
/// * `dst` - The output image container with shape (height, width, channels).
/// * `map_x` - The x coordinates of the pixels to interpolate.
/// * `map_y` - The y coordinates of the pixels to interpolate.
/// * `interpolation` - The interpolation mode to use.
///
/// # Errors
///
/// * The mapx and mapy must have the same size.
/// * The output image must have the same size as the mapx and mapy.
pub fn remap<const CHANNELS: usize>(
    src: &Image<f32, CHANNELS>,
    dst: &mut Image<f32, CHANNELS>,
    map_x: &Image<f32, 1>,
    map_y: &Image<f32, 1>,
    interpolation: InterpolationMode,
) -> Result<(), ImageError> {
    if map_x.size() != map_y.size() {
        return Err(ImageError::InvalidImageSize(
            map_x.height(),
            map_y.height(),
            map_x.width(),
            map_y.width(),
        ));
    }

    if dst.size() != map_x.size() {
        return Err(ImageError::InvalidImageSize(
            src.size().width,
            src.size().height,
            dst.size().width,
            dst.size().height,
        ));
    }

    let mapx_data = unsafe {
        ndarray::ArrayView3::from_shape_ptr(
            (
                map_x.height() as usize,
                map_x.width() as usize,
                map_x.num_channels(),
            ),
            map_x.as_ptr() as *const f32,
        )
    };

    let mapy_data = unsafe {
        ndarray::ArrayView3::from_shape_ptr(
            (
                map_y.height() as usize,
                map_y.width() as usize,
                map_y.num_channels(),
            ),
            map_y.as_ptr() as *const f32,
        )
    };

    let src_data = unsafe {
        ndarray::ArrayView3::from_shape_ptr(
            (
                src.height() as usize,
                src.width() as usize,
                src.num_channels(),
            ),
            src.as_ptr() as *const f32,
        )
    };

    let dst_data = unsafe {
        ndarray::ArrayView3::from_shape_ptr(
            (
                dst.height() as usize,
                dst.width() as usize,
                dst.num_channels(),
            ),
            dst.as_ptr() as *const f32,
        )
    };
    // NOTE: might copy
    let mut dst_data = dst_data.to_owned();

    ndarray::Zip::from(dst_data.rows_mut())
        .and(mapx_data.rows())
        .and(mapy_data.rows())
        .par_for_each(|mut out, u, v| {
            let (u, v) = (u[0], v[0]);
            for c in 0..CHANNELS {
                out[c] = interpolate_pixel(&src_data, u, v, c, interpolation);
            }
        });

    Ok(())
}

#[cfg(test)]
mod tests {
    use kornia_image::{Image, ImageError, ImageSize};

    #[test]
    fn remap_smoke() -> Result<(), ImageError> {
        let image = Image::<_, 1>::new(
            ImageSize {
                width: 3,
                height: 3,
            },
            vec![0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )?;
        let map_x = Image::<_, 1>::new(
            ImageSize {
                width: 2,
                height: 2,
            },
            vec![0f32, 2.0, 0.0, 2.0],
        )?;
        let map_y = Image::<_, 1>::new(
            ImageSize {
                width: 2,
                height: 2,
            },
            vec![0f32, 0.0, 2.0, 2.0],
        )?;

        let expected = Image::<_, 1>::new(
            ImageSize {
                width: 2,
                height: 2,
            },
            vec![0.0, 2.0, 6.0, 8.0],
        )?;

        let mut image_transformed = Image::<_, 1>::from_size_val(map_y.size(), 0.0)?;

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
