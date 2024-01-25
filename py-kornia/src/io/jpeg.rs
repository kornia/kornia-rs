use kornia_rs::io::jpeg::{ImageDecoder, ImageEncoder};
use pyo3::prelude::*;

use crate::image::PyImageSize;
use crate::tensor::PyTensor;

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

    pub fn decode(&mut self, jpeg_data: &[u8]) -> PyResult<PyTensor> {
        let tensor = self.inner.decode(jpeg_data);
        Ok(tensor.into())
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

    pub fn encode(&mut self, data: &[u8], shape: [usize; 3]) -> PyResult<Vec<u8>> {
        let jpeg_data = self.inner.encode(data, shape);
        Ok(jpeg_data)
    }

    pub fn set_quality(&mut self, quality: i32) {
        self.inner.set_quality(quality)
    }
}
