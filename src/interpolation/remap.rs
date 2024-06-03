use super::interpolate::interpolate_pixel;
use super::InterpolationMode;
use crate::image::Image;
use anyhow::Result;

/// Applyu generic geometric transformation to an image.
///
/// # Arguments
///
/// * `src` - The input image container with shape (height, width, channels).
/// * `map_x` - The x coordinates of the pixels to interpolate.
/// * `map_y` - The y coordinates of the pixels to interpolate.
/// * `interpolation` - The interpolation mode to use.
///
/// # Returns
///
/// The transformed image with shape (height, width, channels) and shape of the mapx and mapy.
///
/// # Errors
///
/// * The mapx and mapy must have the same size.
pub fn remap<const CHANNELS: usize>(
    src: &Image<f32, CHANNELS>,
    map_x: &Image<f32, 1>,
    map_y: &Image<f32, 1>,
    interpolation: InterpolationMode,
) -> Result<Image<f32, CHANNELS>> {
    if map_x.size() != map_y.size() {
        return Err(anyhow::anyhow!("map_x and map_y must have the same size"));
    }

    let mut dst = Image::<_, CHANNELS>::from_size_val(map_x.size(), 0.0)?;

    ndarray::Zip::from(dst.data.rows_mut())
        .and(map_x.data.rows())
        .and(map_y.data.rows())
        .par_for_each(|mut out, u, v| {
            let (u, v) = (u[0], v[0]);
            for c in 0..CHANNELS {
                out[c] = interpolate_pixel(&src.data, u, v, c, interpolation);
            }
        });

    Ok(dst)
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    #[test]
    fn remap_smoke() -> Result<()> {
        use crate::image::{Image, ImageSize};
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

        let image_transformed =
            super::remap(&image, &map_x, &map_y, super::InterpolationMode::Bilinear)?;
        assert_eq!(image_transformed.num_channels(), 1);
        assert_eq!(image_transformed.size().width, 2);
        assert_eq!(image_transformed.size().height, 2);

        for (a, b) in image_transformed.data.iter().zip(expected.data.iter()) {
            assert!((a - b).abs() < 1e-6);
        }

        Ok(())
    }
}
