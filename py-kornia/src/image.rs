use anyhow::Result;
use numpy::{PyArray3, ToPyArray};

use kornia_rs::image::{Image, ImageSize};
use pyo3::prelude::*;

// type alias for a 3D numpy array of u8
pub type PyImage = Py<PyArray3<u8>>;

/// Trait to convert an image to a PyImage (3D numpy array of u8)
pub trait ToPyImage {
    fn to_pyimage(&self) -> PyImage;
}

impl<const CHANNELS: usize> ToPyImage for kornia_rs::image::Image<u8, CHANNELS> {
    fn to_pyimage(&self) -> PyImage {
        Python::with_gil(|py| self.data.to_pyarray(py).to_owned())
    }
}

/// Trait to convert a PyImage (3D numpy array of u8) to an image
pub trait FromPyImage<const CHANNELS: usize> {
    fn from_pyimage(image: PyImage) -> Result<Image<u8, CHANNELS>>;
}

impl<const CHANNELS: usize> FromPyImage<CHANNELS> for kornia_rs::image::Image<u8, CHANNELS> {
    fn from_pyimage(image: PyImage) -> Result<Image<u8, CHANNELS>> {
        Python::with_gil(|py| {
            let array = image.as_ref(py).to_owned_array();
            let data = match array.as_slice() {
                Some(d) => d.to_vec(),
                None => return Err(anyhow::anyhow!("Image data is not contiguous")),
            };
            let size = ImageSize {
                width: array.shape()[1],
                height: array.shape()[0],
            };
            Ok(Image::new(size, data)?)
        })
    }
}

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

impl From<ImageSize> for PyImageSize {
    fn from(image_size: ImageSize) -> Self {
        PyImageSize { inner: image_size }
    }
}

impl From<PyImageSize> for ImageSize {
    fn from(image_size: PyImageSize) -> Self {
        image_size.inner
    }
}
