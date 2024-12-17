use numpy::{PyArray, PyArray3, PyArrayMethods, PyUntypedArrayMethods};

use kornia_image::{Image, ImageError, ImageSize};
use pyo3::prelude::*;

// type alias for a 3D numpy array of u8
pub type PyImage = Py<PyArray3<u8>>;

/// Trait to convert an image to a PyImage (3D numpy array of u8)
pub trait ToPyImage {
    fn to_pyimage(self) -> PyImage;
}

impl<const C: usize> ToPyImage for Image<u8, C> {
    fn to_pyimage(self) -> PyImage {
        Python::with_gil(|py| unsafe {
            let array = PyArray::<u8, _>::new(py, [self.height(), self.width(), C], false);
            // TODO: verify that the data is contiguous, otherwise iterate over the image and copy
            std::ptr::copy_nonoverlapping(self.as_ptr(), array.data(), self.numel());
            array.unbind()
        })
    }
}

/// Trait to convert a PyImage (3D numpy array of u8) to an image
pub trait FromPyImage<const C: usize> {
    fn from_pyimage(image: PyImage) -> Result<Image<u8, C>, ImageError>;
}

impl<const C: usize> FromPyImage<C> for Image<u8, C> {
    fn from_pyimage(image: PyImage) -> Result<Image<u8, C>, ImageError> {
        Python::with_gil(|py| {
            let pyarray = image.bind(py);

            // TODO: we should find a way to avoid copying the data
            // Possible solutions:
            // - Use a custom ndarray wrapper that does not copy the data
            // - Return directly pyarray and use it in the Rust code
            let data = match pyarray.to_vec() {
                Ok(d) => d,
                Err(_) => return Err(ImageError::ImageDataNotContiguous),
            };

            let size = ImageSize {
                width: pyarray.shape()[1],
                height: pyarray.shape()[0],
            };

            Image::new(size, data)
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
