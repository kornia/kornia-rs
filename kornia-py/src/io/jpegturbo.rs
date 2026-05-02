use crate::image::{alloc_output_pyarray, numpy_as_image, to_pyerr, PyImage, PyImageSize};
use kornia_image::color_spaces::{Gray8, Rgb8};
use kornia_io::jpegturbo::{write_image_jpegturbo_rgb8, JpegTurboDecoder, JpegTurboEncoder};
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
    pub fn decode(&self, py: Python<'_>, jpeg_data: &[u8]) -> PyResult<PyImage> {
        let size = self.0.read_header(jpeg_data).map_err(to_pyerr)?;
        let (dst, out) = unsafe { alloc_output_pyarray::<3>(py, size)? };
        let mut wrapped = Rgb8(dst);
        self.0
            .decode_rgb8_into(jpeg_data, &mut wrapped)
            .map_err(to_pyerr)?;
        Ok(out)
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
    pub fn decode_gray8(&self, py: Python<'_>, jpeg_data: &[u8]) -> PyResult<PyImage> {
        let size = self.0.read_header(jpeg_data).map_err(to_pyerr)?;
        let (dst, out) = unsafe { alloc_output_pyarray::<1>(py, size)? };
        let mut wrapped = Gray8(dst);
        self.0
            .decode_gray8_into(jpeg_data, &mut wrapped)
            .map_err(to_pyerr)?;
        Ok(out)
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

/// Encodes an RGB u8 image to JPEG bytes using libjpeg-turbo.
///
/// Drop-in faster replacement for ``encode_image_jpeg`` (~3-4× faster on
/// 1080p RGB on aarch64). Returns ``Vec<u8>`` with the JPEG-encoded bytes.
#[pyfunction]
pub fn encode_image_jpegturbo(py: Python<'_>, image: PyImage, quality: i32) -> PyResult<Vec<u8>> {
    let image = unsafe { numpy_as_image::<3>(py, &image)? };
    let encoder = JpegTurboEncoder::new().map_err(to_pyerr)?;
    encoder.set_quality(quality).map_err(to_pyerr)?;
    encoder.encode_rgb8(&image).map_err(to_pyerr)
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
pub fn read_image_jpegturbo(py: Python<'_>, file_path: &str) -> PyResult<PyImage> {
    let bytes = std::fs::read(file_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
    let decoder = JpegTurboDecoder::new().map_err(to_pyerr)?;
    let size = decoder.read_header(&bytes).map_err(to_pyerr)?;
    let (dst, out) = unsafe { alloc_output_pyarray::<3>(py, size)? };
    let mut wrapped = Rgb8(dst);
    decoder
        .decode_rgb8_into(&bytes, &mut wrapped)
        .map_err(to_pyerr)?;
    Ok(out)
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
pub fn decode_image_jpegturbo(py: Python<'_>, jpeg_data: &[u8], mode: &str) -> PyResult<PyImage> {
    let decoder = JpegTurboDecoder::new().map_err(to_pyerr)?;
    let size = decoder.read_header(jpeg_data).map_err(to_pyerr)?;
    match mode {
        "rgb" => {
            let (dst, out) = unsafe { alloc_output_pyarray::<3>(py, size)? };
            let mut wrapped = Rgb8(dst);
            decoder
                .decode_rgb8_into(jpeg_data, &mut wrapped)
                .map_err(to_pyerr)?;
            Ok(out)
        }
        "mono" => {
            let (dst, out) = unsafe { alloc_output_pyarray::<1>(py, size)? };
            let mut wrapped = Gray8(dst);
            decoder
                .decode_gray8_into(jpeg_data, &mut wrapped)
                .map_err(to_pyerr)?;
            Ok(out)
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "The following are the supported values of mode:\n  1) \"rgb\"  -> 8-bit RGB\n  2) \"mono\" -> 8-bit Monochrome",
        )),
    }
}
