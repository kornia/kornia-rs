use crate::image::{FromPyImage, PyImage, ToPyImage};
use kornia_image::Image;
use kornia_io::jpeg as J;
use pyo3::prelude::*;

#[pyfunction]
/// Decodes the JPEG Image from raw bytes.
///
/// The following modes are supported:
/// 1. "rgb" -> 8-bit RGB
/// 2. "mono" -> 8-bit Monochrome
///
/// ```py
/// import kornia_rs as K
///
/// img = K.decode_image_raw_jpeg(bytes(img_data), "rgb")
/// ```
pub fn decode_image_raw_jpeg(src: &[u8], dst: &mut PyImage, mode: &str) -> PyResult<()> {
    match mode {
        "rgb" => {
            let mut image = Image::from_pyimage(image);
            J::decode_image_jpeg_rgb8(src, dst)
        }
        "mono" => {
            let mut image = Image::from_pyimage(image);
            J::decode_image_jpeg_mono8(src, dst)
        }
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                String::from(
                    r#"\
        The following are the supported values of mode:
        1) "rgb" -> 8-bit RGB
        2) "mono" -> 8-bit Monochrome
        "#,
                ),
            ))
        }
    }
}
