use crate::image::Image;
use anyhow::Result;

// TODO: ideally we want something like this:
// let rgb: Image<u8, RGB> = load_image("image.jpg");
// let gray: Image<u8, GRAY> = image.map(|x| (76. * x[0] + 150. * x[1] + 29. * x[2]) / 255.);
// or automatically:
// let gray = Image<u8, 1>::try_from(rgb);

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
pub fn gray_from_rgb<T>(image: &Image<T, 3>) -> Result<Image<T, 1>>
where
    T: Default
        + Copy
        + Clone
        + Send
        + Sync
        + num_traits::NumCast
        + num_traits::Float
        + std::fmt::Debug,
{
    assert_eq!(image.num_channels(), 3);

    // TODO: implement this using a map or cast
    // let image_f32 = image.cast::<f32>();
    //let mut output = Array3::<u8>::zeros(image.data.dim());
    //let mut output = Array3::<u8>::zeros((image.image_size().height, image.image_size().width, 1));
    let rw = T::from(0.299).unwrap();
    let gw = T::from(0.587).unwrap();
    let bw = T::from(0.114).unwrap();

    let mut output = Image::<T, 1>::from_shape(image.image_size())?;

    ndarray::Zip::from(output.data.rows_mut())
        .and(image.data.rows())
        .par_for_each(|mut out, inp| {
            assert_eq!(inp.len(), 3);
            let r = num_traits::NumCast::from(inp[0]).unwrap();
            let g = num_traits::NumCast::from(inp[1]).unwrap();
            let b = num_traits::NumCast::from(inp[2]).unwrap();
            let gray = rw * r + gw * g + bw * b;

            out[0] = num_traits::NumCast::from(gray).unwrap();
        });

    Ok(output)
}

#[cfg(test)]
mod tests {
    use crate::io::functions as F;

    #[test]
    fn gray_from_rgb() {
        let image_path = std::path::Path::new("tests/data/dog.jpeg");
        let image = F::read_image_jpeg(image_path).unwrap();
        let image_norm = image.cast_and_scale::<f32>(1. / 255.0).unwrap();
        let gray = super::gray_from_rgb(&image_norm.cast::<f64>().unwrap()).unwrap();
        assert_eq!(gray.num_channels(), 1);
        assert_eq!(gray.image_size().width, 258);
        assert_eq!(gray.image_size().height, 195);
    }
}
