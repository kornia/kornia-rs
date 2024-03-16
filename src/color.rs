use crate::image::Image;
use anyhow::Result;

/// Define the RGB weights for the grayscale conversion.
const RW: f64 = 0.299;
const GW: f64 = 0.587;
const BW: f64 = 0.114;

/// Convert an RGB image to grayscale using the formula:
///
/// Y = 0.299 * R + 0.587 * G + 0.114 * B
///
/// # Arguments
///
/// * `image` - The input RGB image assumed to have 3 channels.
///
/// # Returns
///
/// The grayscale image.
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
///         width: 4,
///         height: 5,
///     },
///     vec![0f32; 4 * 5 * 3],
/// )
/// .unwrap();
/// let gray: Image<f32, 1> = kornia_rs::color::gray_from_rgb(&image).unwrap();
/// assert_eq!(gray.num_channels(), 1);
/// assert_eq!(gray.size().width, 4);
/// assert_eq!(gray.size().height, 5);
/// ```
pub fn gray_from_rgb<T>(image: &Image<T, 3>) -> Result<Image<T, 1>>
where
    T: Default + Copy + Clone + Send + Sync + num_traits::Float,
{
    assert_eq!(image.num_channels(), 3);

    let rw = T::from(RW).ok_or(anyhow::anyhow!("Failed to convert RW"))?;
    let gw = T::from(GW).ok_or(anyhow::anyhow!("Failed to convert GW"))?;
    let bw = T::from(BW).ok_or(anyhow::anyhow!("Failed to convert BW"))?;

    let mut output = Image::<T, 1>::from_size_val(image.size(), T::default())?;

    ndarray::Zip::from(output.data.rows_mut())
        .and(image.data.rows())
        .par_for_each(|mut out, inp| {
            assert_eq!(inp.len(), 3);
            let r = inp[0];
            let g = inp[1];
            let b = inp[2];
            out[0] = rw * r + gw * g + bw * b;
        });

    Ok(output)
}

#[cfg(test)]
mod tests {
    use crate::io::functional as F;

    #[test]
    fn gray_from_rgb() {
        let image_path = std::path::Path::new("tests/data/dog.jpeg");
        let image = F::read_image_jpeg(image_path).unwrap();
        let image_norm = image.cast_and_scale::<f32>(1. / 255.0).unwrap();
        let gray = super::gray_from_rgb(&image_norm.cast::<f64>().unwrap()).unwrap();
        assert_eq!(gray.num_channels(), 1);
        assert_eq!(gray.size().width, 258);
        assert_eq!(gray.size().height, 195);
    }
}
