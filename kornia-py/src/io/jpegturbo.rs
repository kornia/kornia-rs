use crate::image::{numpy_as_image, to_pyerr, PyImage, PyImageSize, ToPyImage};
use kornia_io::jpegturbo::{
    read_image_jpegturbo_rgb8, write_image_jpegturbo_rgb8, JpegTurboDecoder, JpegTurboEncoder,
};
use pyo3::prelude::*;

/// Hardware-accelerated JPEG decoder using libjpeg-turbo.
#[pyclass(name = "ImageDecoder", frozen)]
pub struct PyImageDecoder(JpegTurboDecoder);

#[pymethods]
impl PyImageDecoder {
    /// Creates a new instance of the ImageDecoder.
    ///
    /// # Returns
    /// * `ImageDecoder`: A new decoder instance.
    #[new]
    pub fn new() -> PyResult<PyImageDecoder> {
        let decoder = JpegTurboDecoder::new()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;
        Ok(PyImageDecoder(decoder))
    }

    /// Reads the header of a JPEG image to extract its dimensions.
    ///
    /// # Arguments
    /// * `jpeg_data` (bytes): Raw bytes containing the JPEG encoded data.
    ///
    /// # Returns
    /// * `ImageSize`: The dimensions of the image.
    ///
    /// # Exceptions
    /// * `Exception`: If reading the header fails.
    pub fn read_header(&self, jpeg_data: &[u8]) -> PyResult<PyImageSize> {
        let image_size = self
            .0
            .read_header(jpeg_data)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;
        Ok(image_size.into())
    }

    /// Decodes a JPEG image into an 8-bit RGB tensor.
    ///
    /// # Arguments
    /// * `jpeg_data` (bytes): Raw bytes containing the JPEG encoded data.
    ///
    /// # Returns
    /// * `numpy.ndarray`: The decoded RGB image tensor with shape `(H, W, 3)` and dtype `uint8`.
    ///
    /// # Exceptions
    /// * `Exception`: If decoding fails.
    pub fn decode(&self, jpeg_data: &[u8]) -> PyResult<PyImage> {
        let image = self
            .0
            .decode_rgb8(jpeg_data)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;
        let pyimage = image.to_pyimage().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyException, _>(format!(
                "failed to convert image: {}",
                e
            ))
        })?;
        Ok(pyimage)
    }

    /// Decodes a JPEG image into an 8-bit grayscale tensor.
    ///
    /// # Arguments
    /// * `jpeg_data` (bytes): Raw bytes containing the JPEG encoded data.
    ///
    /// # Returns
    /// * `numpy.ndarray`: The decoded grayscale image tensor with dtype `uint8`.
    ///
    /// # Exceptions
    /// * `Exception`: If decoding fails.
    pub fn decode_gray8(&self, jpeg_data: &[u8]) -> PyResult<PyImage> {
        let image = self
            .0
            .decode_gray8(jpeg_data)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;
        let pyimage = image.to_pyimage().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyException, _>(format!(
                "failed to convert image: {}",
                e
            ))
        })?;
        Ok(pyimage)
    }
}

/// Hardware-accelerated JPEG encoder using libjpeg-turbo.
#[pyclass(name = "ImageEncoder", frozen)]
pub struct PyImageEncoder(JpegTurboEncoder);

#[pymethods]
impl PyImageEncoder {
    /// Creates a new instance of the ImageEncoder.
    ///
    /// # Returns
    /// * `ImageEncoder`: A new encoder instance.
    #[new]
    pub fn new() -> PyResult<PyImageEncoder> {
        let encoder = JpegTurboEncoder::new()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;
        Ok(PyImageEncoder(encoder))
    }

    /// Encodes an image tensor into JPEG bytes.
    ///
    /// # Arguments
    /// * `image` (numpy.ndarray): RGB image data with dtype `uint8` and shape `(H, W, 3)`.
    ///
    /// # Returns
    /// * `bytes`: A byte array containing the JPEG-encoded image data.
    ///
    /// # Exceptions
    /// * `Exception`: If encoding fails or the image format is incompatible.
    pub fn encode(&self, py: Python<'_>, image: PyImage) -> PyResult<Vec<u8>> {
        let image = unsafe { numpy_as_image::<3>(py, &image)? };
        let jpeg_data = self.0.encode_rgb8(&image).map_err(to_pyerr)?;
        Ok(jpeg_data)
    }

    /// Sets the encoding quality for the encoder.
    ///
    /// # Arguments
    /// * `quality` (int): The JPEG encoding quality.
    ///
    /// # Exceptions
    /// * `Exception`: If setting the quality fails.
    pub fn set_quality(&self, quality: i32) -> PyResult<()> {
        self.0
            .set_quality(quality)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;
        Ok(())
    }
}

/// Reads an 8-bit RGB JPEG image from a file path using libjpeg-turbo.
///
/// # Arguments
/// * `file_path` (str): The path to the JPEG file to read.
///
/// # Returns
/// * `numpy.ndarray`: The decoded 8-bit RGB image tensor with dtype `uint8`.
///
/// # Exceptions
/// * `FileExistsError`: If the read operation fails (Note: mapped from upstream error).
/// * `Exception`: If the image fails to convert.
///
/// *Python-only helper; not part of kornia-io's Rust API.*
#[pyfunction]
pub fn read_image_jpegturbo(file_path: &str) -> PyResult<PyImage> {
    let image = read_image_jpegturbo_rgb8(file_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyFileExistsError, _>(format!("{}", e)))?;
    let pyimage = image.to_pyimage().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyException, _>(format!("failed to convert image: {}", e))
    })?;
    Ok(pyimage)
}

/// Writes an image tensor to a JPEG file using libjpeg-turbo.
///
/// # Arguments
/// * `file_path` (str): The path where the JPEG file will be saved.
/// * `image` (numpy.ndarray): The image tensor to write (dtype `uint8`, shape `(H, W, 3)`).
/// * `quality` (int): The JPEG encoding quality (0-100).
///
/// # Exceptions
/// * `Exception`: If writing or encoding fails.
///
/// *Python-only helper; not part of kornia-io's Rust API.*
#[pyfunction]
pub fn write_image_jpegturbo(
    py: Python<'_>,
    file_path: &str,
    image: PyImage,
    quality: u8,
) -> PyResult<()> {
    let image = unsafe { numpy_as_image::<3>(py, &image)? };
    write_image_jpegturbo_rgb8(file_path, &image, quality).map_err(to_pyerr)?;
    Ok(())
}

/// Decodes a JPEG image from raw bytes using libjpeg-turbo.
///
/// # Arguments
/// * `jpeg_data` (bytes): Raw bytes containing the JPEG encoded data.
/// * `mode` (str): The color mode to decode the image into.
///   Supported values are strictly lowercase `"rgb"` (8-bit RGB) or `"mono"` (8-bit Grayscale).
///
/// # Returns
/// * `numpy.ndarray`: The decoded image as a NumPy `ndarray` of `uint8`.
///
/// # Exceptions
/// * `ValueError`: If the mode is unsupported (case-sensitive).
/// * `Exception`: If decoding or image conversion fails.
///
/// *Python-only helper; not part of kornia-io's Rust API.*
#[pyfunction]
pub fn decode_image_jpegturbo(jpeg_data: &[u8], mode: &str) -> PyResult<PyImage> {
    let image = match mode {
        "rgb" => JpegTurboDecoder::new()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?
            .decode_rgb8(jpeg_data)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?
            .to_pyimage()
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyException, _>(format!(
                    "failed to convert image: {}",
                    e
                ))
            })?,
        "mono" => JpegTurboDecoder::new()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?
            .decode_gray8(jpeg_data)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?
            .to_pyimage()
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyException, _>(format!(
                    "failed to convert image: {}",
                    e
                ))
            })?,
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
