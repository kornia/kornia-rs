use crate::image::{FromPyImage, PyImage, ToPyImage};
use kornia_image::Image;
use kornia_io::functional as F;
use kornia_io::jpegturbo::JpegTurboDecoder;
use pyo3::prelude::*;

#[pyfunction]
pub fn read_image_jpeg(file_path: &str) -> PyResult<PyImage> {
    let image = F::read_image_jpegturbo_rgb8(file_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyFileExistsError, _>(format!("{}", e)))?;
    Ok(image.to_pyimage())
}

#[pyfunction]
pub fn write_image_jpeg(file_path: &str, image: PyImage) -> PyResult<()> {
    let image = Image::from_pyimage(image)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;
    F::write_image_jpegturbo_rgb8(file_path, &image)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;
    Ok(())
}

#[pyfunction]
pub fn read_image_any(file_path: &str) -> PyResult<PyImage> {
    let image = F::read_image_any_rgb8(file_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyFileExistsError, _>(format!("{}", e)))?;
    Ok(image.to_pyimage())
}

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
/// img = K.decode_image_jpeg(bytes(img_data), "rgb")
/// ```
pub fn decode_image_jpeg(jpeg_data: &[u8], mode: &str) -> PyResult<PyImage> {
    let image = match mode {
        "rgb" => JpegTurboDecoder::new()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?
            .decode_rgb8(jpeg_data)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?
            .to_pyimage(),
        "mono" => JpegTurboDecoder::new()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?
            .decode_gray8(jpeg_data)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?
            .to_pyimage(),
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

    Ok(image)
}
