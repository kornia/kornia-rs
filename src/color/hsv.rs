use crate::image::Image;
use anyhow::Result;

/// Convert an RGB image to an HSV image.
///
/// The input image is assumed to have 3 channels in the order R, G, B.
///
/// # Arguments
///
/// * `image` - The input RGB image assumed to have 3 channels.
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
///
/// # Example
///
/// ```
/// use kornia_rs::image::{Image, ImageSize};
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
/// let hsv: Image<f32, 3> = kornia_rs::color::hsv_from_rgb(&image).unwrap();
/// assert_eq!(hsv.num_channels(), 3);
/// assert_eq!(hsv.size().width, 4);
/// assert_eq!(hsv.size().height, 5);
/// ```
pub fn hsv_from_rgb(image: &Image<f32, 3>) -> Result<Image<f32, 3>> {
    let mut output = Image::<f32, 3>::from_size_val(image.size(), 0.0)?;

    ndarray::Zip::from(output.data.rows_mut())
        .and(image.data.rows())
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

    Ok(output)
}

#[cfg(test)]
mod tests {

    use crate::image::{Image, ImageSize};
    use anyhow::Result;

    #[test]
    fn hsv_from_rgb() -> Result<()> {
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

        let hsv = super::hsv_from_rgb(&image)?;
        assert_eq!(hsv.num_channels(), 3);
        assert_eq!(hsv.size().width, 2);
        assert_eq!(hsv.size().height, 3);

        Ok(())
    }
}
