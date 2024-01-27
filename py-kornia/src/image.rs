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
    pub inner: Image,
}

#[pymethods]
impl PyImage {
    #[new]
    pub fn new(image_size: PyImageSize, data: Vec<u8>) -> PyResult<PyImage> {
        let image = Image::new(image_size.inner, data);
        Ok(image.into())
    }

    #[getter]
    pub fn image_size(&self) -> PyImageSize {
        self.inner.image_size().into()
    }

    #[getter]
    pub fn num_channels(&self) -> usize {
        self.inner.num_channels()
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!(
            "Image(height: {}, width: {}, num_channels: {})",
            self.inner.image_size().height,
            self.inner.image_size().width,
            self.inner.num_channels()
        ))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "Image(height: {}, width: {}, num_channels: {})",
            self.inner.image_size().height,
            self.inner.image_size().width,
            self.inner.num_channels()
        ))
    }
}

impl From<ImageSize> for PyImageSize {
    fn from(image_size: ImageSize) -> Self {
        PyImageSize { inner: image_size }
    }
}

impl From<Image> for PyImage {
    fn from(image: Image) -> Self {
        PyImage { inner: image }
    }
}

impl From<PyImageSize> for ImageSize {
    fn from(image_size: PyImageSize) -> Self {
        image_size.inner
    }
}

impl From<PyImage> for Image {
    fn from(image: PyImage) -> Self {
        image.inner
    }
}
