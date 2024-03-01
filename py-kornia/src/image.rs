use numpy::{PyArray3, ToPyArray};

use kornia_rs::image::{Image, ImageSize};
use pyo3::prelude::*;

#[pyclass(name = "ImageSize")]
#[derive(Clone)]
pub struct PyImageSize {
    inner: ImageSize,
}

#[pymethods]
impl PyImageSize {
    #[new]
    pub fn new(width: usize, height: usize) -> PyResult<PyImageSize> {
        let inner = ImageSize { width, height };
        Ok(PyImageSize { inner })
    }

    #[getter]
    pub fn width(&self) -> usize {
        self.inner.width
    }

    #[getter]
    pub fn height(&self) -> usize {
        self.inner.height
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!(
            "ImageSize(width: {}, height: {})",
            self.inner.width, self.inner.height
        ))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "ImageSize(width: {}, height: {})",
            self.inner.width, self.inner.height
        ))
    }
}

#[pyclass(name = "Image")]
#[derive(Clone)]
pub struct PyImage {
    pub inner: Image<u8, 3>,
}

#[pymethods]
impl PyImage {
    #[getter]
    pub fn shape(&self) -> PyResult<(usize, usize, usize)> {
        Ok((
            self.inner.image_size().height,
            self.inner.image_size().width,
            self.inner.num_channels(),
        ))
    }

    pub fn size(&self) -> PyImageSize {
        self.inner.image_size().into()
    }

    pub fn wdith(&self) -> usize {
        self.inner.image_size().width
    }

    pub fn height(&self) -> usize {
        self.inner.image_size().height
    }

    pub fn num_channels(&self) -> usize {
        self.inner.num_channels()
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!(
            "Image(height: {}, width: {}, num_channels: {}, dtype: u8)",
            self.inner.image_size().height,
            self.inner.image_size().width,
            self.inner.num_channels()
        ))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "Image(height: {}, width: {}, num_channels: {}, dtype: u8)",
            self.inner.image_size().height,
            self.inner.image_size().width,
            self.inner.num_channels()
        ))
    }

    fn numpy(&self, py: Python) -> PyResult<Py<PyArray3<u8>>> {
        Ok(self.inner.data.to_pyarray(py).to_owned())
    }
}

impl From<ImageSize> for PyImageSize {
    fn from(image_size: ImageSize) -> Self {
        PyImageSize { inner: image_size }
    }
}

impl From<Image<u8, 3>> for PyImage {
    fn from(image: Image<u8, 3>) -> Self {
        PyImage { inner: image }
    }
}

impl From<PyImageSize> for ImageSize {
    fn from(image_size: PyImageSize) -> Self {
        image_size.inner
    }
}

impl From<PyImage> for Image<u8, 3> {
    fn from(image: PyImage) -> Self {
        image.inner
    }
}
