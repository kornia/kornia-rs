use kornia_image::Image;
use pyo3::prelude::*;

use crate::image::{FromPyImage, PyImage, ToPyImage};
use kornia_io::png as P;

#[pyfunction]
/// Decodes the PNG Image from raw bytes.
///
/// The following modes are supported:
/// 1. "rgb8" -> 8-bit RGB
/// 2. "rgba8" -> 8-bit RGBA
/// 3. "mono8" -> 8-bit Monochrome
/// 4. "rgb16" -> 16-bit RGB
/// 5. "rgba16" -> 16-bit RGBA
/// 6. "mono16" -> 16-bit Monochrome
///
/// ```py
/// import kornia_rs as K
///
/// img = K.decode_image_png(bytes(img_data), "rgb8")
/// ```
pub fn decode_image_png(image: &mut PyImage, png_data: &[u8], mode: &str) -> PyResult<()> {
    match mode {
        "rgb8" => {
            let mut image: Image<u8, 3> = Image::from_pyimage(image);
            P::decode_image_png_rgb8(&mut image, png_data)
        }
        "rgba8" => {
            let mut image: Image<u8, 4> = Image::from_pyimage(image);
            P::decode_image_png_rgba8(&mut image, png_data)
        }
        "mono8" => {
            let mut image: Image<u8, 1> = Image::from_pyimage(image);
            P::decode_image_png_mono8(&mut image, png_data)
        }
        "rgb16" => {
            let mut image: Image<u16, 3> = Image::from_pyimage(image);
            P::decode_image_png_rgb16(&mut image, png_data)
        }
        "rgba16" => {
            let mut image: Image<u16, 4> = Image::from_pyimage(image);
            P::decode_image_png_rgba16(&mut image, png_data)
        }
        "mono16" => {
            let mut image: Image<u16, 1> = Image::from_pyimage(image);
            P::decode_image_png_mono16(&mut image, png_data)
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            String::from(
                r#"\
        The following are the supported values of mode:
        1) "rgb8" -> 8-bit RGB
        2) "rgba8" -> 8-bit RGBA
        3) "mono8" -> 8-bit Monochrome
        4) "rgb16" -> 16-bit RGB
        5) "rgba16" -> 16-bit RGBA
        6) "mono16" -> 16-bit Monochrome
        "#,
            ),
        )),
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))
}
