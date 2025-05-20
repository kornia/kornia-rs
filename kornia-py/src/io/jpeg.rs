use crate::image::{FromPyImage, PyImage, ToPyImage};
use kornia_image::{allocator::CpuAllocator, Image, ImageSize};
use kornia_io::jpeg as J;
use pyo3::prelude::*;

#[pyfunction]
pub fn read_image_jpeg(file_path: &str, mode: &str) -> PyResult<PyImage> {
    let result = match mode {
        "rgb" => {
            let img = J::read_image_jpeg_rgb8(file_path)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            img.to_pyimage()
        }
        "mono" => {
            let img = J::read_image_jpeg_mono8(file_path)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            img.to_pyimage()
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

#[pyfunction]
pub fn write_image_jpeg(file_path: &str, image: PyImage, mode: &str, quality: u8) -> PyResult<()> {
    match mode {
        "rgb" => {
            let image = Image::<u8, 3>::from_pyimage(image)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            J::write_image_jpeg_rgb8(file_path, &image, quality)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        }
        "mono" => {
            let image = Image::<u8, 1>::from_pyimage(image)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            J::write_image_jpeg_gray8(file_path, &image, quality)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
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
    Ok(())
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
/// img = K.decode_image_jpeg(bytes(img_data), (32, 32), "rgb")
/// ```
///
pub fn decode_image_jpeg(src: &[u8], image_shape: (usize, usize), mode: &str) -> PyResult<PyImage> {
    let image_shape = ImageSize {
        width: image_shape.1,
        height: image_shape.0,
    };

    let result = match mode {
        "rgb" => {
            let mut output_image = Image::<u8, 3>::from_size_val(image_shape, 0, CpuAllocator)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            J::decode_image_jpeg_rgb8(src, &mut output_image)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            output_image.to_pyimage()
        }
        "mono" => {
            let mut output_image = Image::<u8, 1>::from_size_val(image_shape, 0, CpuAllocator)
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
