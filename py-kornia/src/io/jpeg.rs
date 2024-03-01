use kornia_rs::io::jpeg::{ImageDecoder, ImageEncoder};
use pyo3::prelude::*;

use crate::image::{PyImage, PyImageSize};

#[pyclass(name = "ImageDecoder")]
pub struct PyImageDecoder {
    pub inner: ImageDecoder,
}

#[pymethods]
impl PyImageDecoder {
    #[new]
    pub fn new() -> PyResult<PyImageDecoder> {
        let inner = ImageDecoder::new();
        Ok(PyImageDecoder { inner })
    }

    pub fn read_header(&mut self, jpeg_data: &[u8]) -> PyResult<PyImageSize> {
        let image_size = self.inner.read_header(jpeg_data);
        Ok(image_size.into())
    }

    pub fn decode(&mut self, jpeg_data: &[u8]) -> PyResult<PyImage> {
        let image = self.inner.decode(jpeg_data);
        Ok(image.into())
    }
}

#[pyclass(name = "ImageEncoder")]
pub struct PyImageEncoder {
    pub inner: ImageEncoder,
}

#[pymethods]
impl PyImageEncoder {
    #[new]
    pub fn new() -> PyResult<PyImageEncoder> {
        let inner = ImageEncoder::new();
        Ok(PyImageEncoder { inner })
    }

    pub fn encode(&mut self, image: PyImage) -> PyResult<Vec<u8>> {
        let jpeg_data = self.inner.encode(image.into());
        Ok(jpeg_data)
    }

    pub fn set_quality(&mut self, quality: i32) {
        self.inner.set_quality(quality)
    }
}
