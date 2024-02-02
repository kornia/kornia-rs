use crate::image::Image;
use ndarray::{Array3, Zip};

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
pub fn gray_from_rgb(image: &Image) -> Image {
    assert_eq!(image.num_channels(), 3);

    // TODO: implement this using a map or cast
    // let image_f32 = image.cast::<f32>();
    //let mut output = Array3::<u8>::zeros(image.data.dim());
    let mut output = Array3::<u8>::zeros((image.image_size().height, image.image_size().width, 1));

    Zip::from(output.rows_mut())
        .and(image.data.rows())
        .par_for_each(|mut out, inp| {
            assert_eq!(inp.len(), 3);
            let r = inp[0] as f32;
            let g = inp[1] as f32;
            let b = inp[2] as f32;
            let gray = (76. * r + 150. * g + 29. * b) / 255.;

            out[0] = gray as u8;
        });

    Image { data: output }
}

#[cfg(test)]
mod tests {
    use crate::io::functions as F;

    #[test]
    fn gray_from_rgb() {
        let image_path = std::path::Path::new("tests/data/dog.jpeg");
        let image = F::read_image_jpeg(image_path);
        let gray = super::gray_from_rgb(&image);
        assert_eq!(gray.num_channels(), 1);
        assert_eq!(gray.image_size().width, 258);
        assert_eq!(gray.image_size().height, 195);
    }
}
