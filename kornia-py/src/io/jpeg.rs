use crate::image::{PyImage, ToPyImage};
use kornia_image::{Image, ImageSize};
use kornia_io::jpeg as J;
use pyo3::prelude::*;

#[pyfunction]
/// Decodes the JPEG Image from raw bytes.
///
/// The following modes are supported:
/// 1. "rgb" -> 8-bit RGB
/// 2. "mono" -> 8-bit Monochrome
pub fn decode_image_raw_jpeg(src: &[u8], image_shape: Vec<usize>, mode: &str) -> PyResult<PyImage> {
    if image_shape.len() != 2 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            String::from("Missing width and height of image, pass [width, height]"),
        ));
    }
    let image_shape = ImageSize {
        width: image_shape[1],
        height: image_shape[0],
    };

    let result = match mode {
        "rgb" => {
            let mut output_image = Image::<u8, 3>::from_size_val(image_shape, 0)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            J::decode_image_jpeg_rgb8(src, &mut output_image)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            output_image.to_pyimage()
        }
        "mono" => {
            let mut output_image = Image::<u8, 1>::from_size_val(image_shape, 0)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            J::decode_image_jpeg_mono8(src, &mut output_image)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            output_image.to_pyimage()
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
    };

    Ok(result)
}
