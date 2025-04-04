use crate::image::{FromPyImage, PyImage, PyImageSize, ToPyImage};
use kornia_image::Image;
use kornia_io::jpegturbo::{JpegTurboDecoder, JpegTurboEncoder};
use pyo3::prelude::*;

#[pyclass(name = "ImageDecoder", frozen)]
pub struct PyImageDecoder(JpegTurboDecoder);

#[pymethods]
impl PyImageDecoder {
    #[new]
    pub fn new() -> PyResult<PyImageDecoder> {
        let decoder = JpegTurboDecoder::new()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;
        Ok(PyImageDecoder(decoder))
    }

    pub fn read_header(&self, jpeg_data: &[u8]) -> PyResult<PyImageSize> {
        let image_size = self
            .0
            .read_header(jpeg_data)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;
        Ok(image_size.into())
    }

    pub fn decode(&self, jpeg_data: &[u8]) -> PyResult<PyImage> {
        let image = self
            .0
            .decode_rgb8(jpeg_data)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;
        Ok(image.to_pyimage())
    }
}

#[pyclass(name = "ImageEncoder", frozen)]
pub struct PyImageEncoder(JpegTurboEncoder);

#[pymethods]
impl PyImageEncoder {
    #[new]
    pub fn new() -> PyResult<PyImageEncoder> {
        let encoder = JpegTurboEncoder::new()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;
        Ok(PyImageEncoder(encoder))
    }

    pub fn encode(&self, image: PyImage) -> PyResult<Vec<u8>> {
        let image = Image::from_pyimage(image)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;
        let jpeg_data = self
            .0
            .encode_rgb8(&image)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;
        Ok(jpeg_data)
    }

    pub fn set_quality(&self, quality: i32) -> PyResult<()> {
        self.0
            .set_quality(quality)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;
        Ok(())
    }
}
