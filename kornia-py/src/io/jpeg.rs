use crate::image::{FromPyImage, PyImage, ToPyImage};
use kornia_image::Image;
use kornia_io::jpeg as J;
use pyo3::prelude::*;

#[pyfunction]
pub fn decode_image_raw_jpeg(image: &mut PyImage, jpeg_data: &[u8], mode: &str) -> PyResult<()> {
    match mode {
        "rgb" => {
            let mut image = Image::from_pyimage(image);
            J::decode_image_jpeg_rgb8(image, bytes)
        }
        "mono" => {
            let mut image = Image::from_pyimage(image);
            J::decode_image_jpeg_mono8(image, bytes)
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
