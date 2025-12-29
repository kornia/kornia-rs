use crate::image::{FromPyImage, PyImage, PyImageSize, ToPyImage};
use kornia_image::Image;
use kornia_io::jpegturbo::{
    read_image_jpegturbo_rgb8, write_image_jpegturbo_rgb8, JpegTurboDecoder, JpegTurboEncoder,
};
use pyo3::prelude::*;

#[pyclass(name = "ImageDecoder", frozen)]
pub struct PyImageDecoder(pub JpegTurboDecoder);

#[pymethods]
impl PyImageDecoder {
    #[new]
    pub fn new() -> PyResult<Self> {
        let decoder = JpegTurboDecoder::new()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(e.to_string()))?;
        Ok(Self(decoder))
    }

    pub fn read_header(&self, jpeg_data: &[u8]) -> PyResult<PyImageSize> {
        let size = self
            .0
            .read_header(jpeg_data)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(e.to_string()))?;
        Ok(size.into())
    }

    pub fn decode(&self, jpeg_data: &[u8]) -> PyResult<PyImage> {
        let image = self
            .0
            .decode_rgb8(jpeg_data)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(e.to_string()))?;
        image
            .to_pyimage()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(e.to_string()))
    }

    pub fn decode_gray8(&self, jpeg_data: &[u8]) -> PyResult<PyImage> {
        let image = self
            .0
            .decode_gray8(jpeg_data)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(e.to_string()))?;
        image
            .to_pyimage()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(e.to_string()))
    }
}

#[pyclass(name = "ImageEncoder", frozen)]
pub struct PyImageEncoder(pub JpegTurboEncoder);

#[pymethods]
impl PyImageEncoder {
    #[new]
    pub fn new() -> PyResult<Self> {
        let encoder = JpegTurboEncoder::new()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(e.to_string()))?;
        Ok(Self(encoder))
    }

    pub fn encode(&self, image: PyImage) -> PyResult<Vec<u8>> {
        let img = Image::from_pyimage(image)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(e.to_string()))?;
        self.0
            .encode_rgb8(&img)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(e.to_string()))
    }

    pub fn set_quality(&self, quality: i32) -> PyResult<()> {
        self.0
            .set_quality(quality)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(e.to_string()))?;
        Ok(())
    }
}

#[pyfunction]
pub fn read_image_jpegturbo(file_path: &str) -> PyResult<PyImage> {
    let image = read_image_jpegturbo_rgb8(file_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyFileExistsError, _>(format!("{}", e)))?;
    let pyimage = image.to_pyimage().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyException, _>(format!("failed to convert image: {}", e))
    })?;
    Ok(pyimage)
}

#[pyfunction]
pub fn write_image_jpegturbo(file_path: &str, image: PyImage, quality: u8) -> PyResult<()> {
    let image = Image::from_pyimage(image)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;
    write_image_jpegturbo_rgb8(file_path, &image, quality)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;
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
/// img = K.decode_image_jpegturbo(bytes(img_data), "rgb")
/// ```
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
