use kornia_image::{Image, ImageError};

/// Convert an RGB image to an HSV image.
///
/// The input image is assumed to have 3 channels in the order R, G, B.
///
/// # Arguments
///
/// * `src` - The input RGB image assumed to have 3 channels.
/// * `dst` - The output HSV image.
///
/// # Returns
///
/// The HSV image with the following channels:
///
/// * H: The hue channel in the range [0, 255] (0-360 degrees).
/// * S: The saturation channel in the range [0, 255].
/// * V: The value channel in the range [0, 255].
///
/// Precondition: the input image must have 3 channels.
/// Precondition: the output image must have 3 channels.
/// Precondition: the input and output images must have the same size.
///
/// # Example
///
/// ```
/// use kornia::image::{Image, ImageSize};
/// use kornia::imgproc::color::hsv_from_rgb;
///
/// let image = Image::<f32, 3>::new(
///     ImageSize {
///        width: 4,
///        height: 5,
///     },
///     vec![0f32; 4 * 5 * 3],
/// )
/// .unwrap();
///
/// let mut hsv = Image::<f32, 3>::from_size_val(image.size(), 0.0).unwrap();
///
/// hsv_from_rgb(&image, &mut hsv).unwrap();
///
/// assert_eq!(hsv.num_channels(), 3);
/// assert_eq!(hsv.size().width, 4);
/// assert_eq!(hsv.size().height, 5);
/// ```
pub fn hsv_from_rgb(src: &Image<f32, 3>, dst: &mut Image<f32, 3>) -> Result<(), ImageError> {
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.size().width,
            src.size().height,
            dst.size().width,
            dst.size().height,
        ));
    }

    if src.num_channels() != 3 {
        return Err(ImageError::ChannelIndexOutOfBounds(3, src.num_channels()));
    }

    if dst.num_channels() != 3 {
        return Err(ImageError::ChannelIndexOutOfBounds(3, dst.num_channels()));
    }

    ndarray::Zip::from(dst.data.rows_mut())
        .and(src.data.rows())
        .par_for_each(|mut out, inp| {
            assert_eq!(inp.len(), 3);
            // Normalize the input to the range [0, 1]
            let r = inp[0] / 255.;
            let g = inp[1] / 255.;
            let b = inp[2] / 255.;

            let max = r.max(g).max(b);
            let min = r.min(g).min(b);
            let delta = max - min;

            let h = if delta == 0.0 {
                0.0
            } else if max == r {
                60.0 * (((g - b) / delta) % 6.0)
            } else if max == g {
                60.0 * (((b - r) / delta) + 2.0)
            } else {
                60.0 * (((r - g) / delta) + 4.0)
            };

            // Ensure h is in the range [0, 360)

            let h = if h < 0.0 { h + 360.0 } else { h };

            // scale h to [0, 255]

            let h = (h / 360.0) * 255.0;

            let s = if max == 0.0 {
                0.0
            } else {
                (delta / max) * 255.0
            };

            let v = max * 255.0;

            out[0] = h;
            out[1] = s;
            out[2] = v;
        });

    Ok(())
}

#[cfg(test)]
mod tests {
    use kornia_image::{Image, ImageError, ImageSize};

    #[test]
    fn hsv_from_rgb() -> Result<(), ImageError> {
        let image = Image::<f32, 3>::new(
            ImageSize {
                width: 2,
                height: 3,
            },
            vec![
                0.0, 128.0, 255.0, 255.0, 128.0, 0.0, 128.0, 255.0, 0.0, 255.0, 0.0, 128.0, 0.0,
                128.0, 255.0, 255.0, 128.0, 0.0,
            ],
        )?;

        let mut hsv = Image::<f32, 3>::from_size_val(image.size(), 0.0)?;

        super::hsv_from_rgb(&image, &mut hsv)?;

        assert_eq!(hsv.num_channels(), 3);
        assert_eq!(hsv.size().width, 2);
        assert_eq!(hsv.size().height, 3);

        Ok(())
    }
}
