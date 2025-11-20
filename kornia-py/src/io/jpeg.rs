use crate::image::{FromPyImage, PyImage, PyImageLayout, ToPyImage};
use kornia_image::{allocator::CpuAllocator, Image};
use kornia_io::jpeg as J;
use pyo3::prelude::*;

#[pyfunction]
pub fn read_image_jpeg(file_path: &str, mode: &str) -> PyResult<PyImage> {
    let result = match mode {
        "rgb" => {
            let img = J::read_image_jpeg_rgb8(file_path)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            let pyimg = img.to_pyimage().map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyException, _>(format!(
                    "failed to convert image: {}",
                    e
                ))
            })?;
            pyimg
        }
        "mono" => {
            let img = J::read_image_jpeg_mono8(file_path)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            let pyimg = img.to_pyimage().map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyException, _>(format!(
                    "failed to convert image: {}",
                    e
                ))
            })?;
            pyimg
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
            let image = Image::<u8, 3, _>::from_pyimage(image)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            J::write_image_jpeg_rgb8(file_path, &image, quality)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        }
        "mono" => {
            let image = Image::<u8, 1, _>::from_pyimage(image)
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
/// ```py
/// import kornia_rs as K
///
/// img = K.decode_image_jpeg(bytes(img_data))
/// ```
pub fn decode_image_jpeg(src: &[u8]) -> PyResult<PyImage> {
    let layout = J::decode_image_jpeg_info(src)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    let result = match layout.channels {
        3 => {
            let mut output_image = kornia_image::color_spaces::Rgb8::from_size_val(layout.image_size, 0, CpuAllocator)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            J::decode_image_jpeg_rgb8(src, &mut output_image)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            let output_pyimage = output_image.to_pyimage().map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyException, _>(format!(
                    "failed to convert image: {}",
                    e
                ))
            })?;
            output_pyimage
        }
        1 => {
            let mut output_image = kornia_image::color_spaces::Gray8::from_size_val(layout.image_size, 0, CpuAllocator)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            J::decode_image_jpeg_mono8(src, &mut output_image)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            let output_pyimage = output_image.to_pyimage().map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyException, _>(format!(
                    "failed to convert image: {}",
                    e
                ))
            })?;
            output_pyimage
        }
        ch => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unsupported number of channels: {}",
                ch
            )))
        }
    };

    Ok(result)
}

#[pyfunction]
/// Extracts layout metadata from JPEG image raw bytes.
pub fn decode_image_jpeg_info(src: &[u8]) -> PyResult<PyImageLayout> {
    let layout = J::decode_image_jpeg_info(src)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    Ok(layout.into())
}
