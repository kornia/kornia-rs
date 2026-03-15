use crate::image::{FromPyImage, PyImage, ToPyImage};
use kornia_image::{allocator::CpuAllocator, Image};
use kornia_io::jpeg as J;
use pyo3::prelude::*;

/// Reads a JPEG image from a file path.
///
/// # Arguments
/// * `file_path` (str): The path to the JPEG file to read.
/// * `mode` (str): The color mode to decode the image into.
///   Supported values are strictly lowercase `"rgb"` (8-bit RGB) or `"mono"` (8-bit Grayscale).
///
/// # Returns
/// * `numpy.ndarray`: The decoded image as a NumPy array (dtype `uint8`, shape `(H, W, C)` for `"rgb"` or `(H, W)` for `"mono"`).
///
/// # Exceptions
/// * `ValueError`: If the mode is unsupported (case-sensitive) or if the image fails to decode.
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

/// Writes an image tensor to a JPEG file.
///
/// # Arguments
/// * `file_path` (str): The path where the JPEG file will be saved.
/// * `image` (numpy.ndarray): Image data as a contiguous NumPy array (dtype `uint8`). For `"rgb"` mode, expected shape is (H, W, 3). For `"mono"` mode, expected shape is (H, W, 1).
/// * `mode` (str): The color mode of the image.
///   Supported values are strictly lowercase `"rgb"` (8-bit RGB) or `"mono"` (8-bit Grayscale).
/// * `quality` (int): The JPEG encoding quality. Must be in the range 0 to 100 inclusive
///   (where 0 is the lowest quality and 100 is the highest quality).
///
/// # Exceptions
/// * `ValueError`: If the mode is unsupported, the image shape mismatches the mode,
///   or the file fails to write.
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

/// Decodes a JPEG Image from raw bytes.
///
/// # Arguments
/// * `src` (bytes): Raw bytes containing the JPEG encoded data.
///
/// # Returns
/// * numpy.ndarray: The decoded image tensor with dtype uint8.
///
/// # Exceptions
/// * `ValueError`: If the byte data is invalid, decoding fails, or the
///   number of channels is unsupported.
#[pyfunction]
pub fn decode_image_jpeg(src: &[u8]) -> PyResult<PyImage> {
    let layout = J::decode_image_jpeg_layout(src)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    let result = match layout.channels {
        3 => {
            let mut output_image =
                kornia_image::color_spaces::Rgb8::from_size_val(layout.image_size, 0, CpuAllocator)
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
            let mut output_image = kornia_image::color_spaces::Gray8::from_size_val(
                layout.image_size,
                0,
                CpuAllocator,
            )
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

/// Encodes an RGB u8 image to JPEG bytes.
///
/// # Arguments
/// * `image` (numpy.ndarray): RGB image array with shape (H, W, 3) and dtype `uint8`.
/// * `quality` (int): JPEG encoding quality (0-100, where 100 is highest quality).
///
/// # Returns
/// * `bytes`: A byte array containing the JPEG-encoded image data.
///
/// # Exceptions
/// * `ValueError`: If the image format is incompatible or encoding fails.
#[pyfunction]
pub fn encode_image_jpeg(image: PyImage, quality: u8) -> PyResult<Vec<u8>> {
    let image = Image::<u8, 3, _>::from_pyimage(image)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    let mut buffer = Vec::new();
    J::encode_image_jpeg_rgb8(&image, quality, &mut buffer)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    Ok(buffer)
}
