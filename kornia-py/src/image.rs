use numpy::{PyArray, PyArray3, PyArrayMethods, PyUntypedArrayMethods};

use kornia_image::{allocator::CpuAllocator, Image, ImageError, ImageSize, color_spaces::*};
use pyo3::prelude::*;

// type alias for a 3D numpy array of u8
pub type PyImage = Py<PyArray3<u8>>;

// type alias for a 3D numpy array of u16
pub type PyImageU16 = Py<PyArray3<u16>>;

// type alias for a 3D numpy array of f32
pub type PyImageF32 = Py<PyArray3<f32>>;

/// Trait to convert an image to a PyImage (3D numpy array of u8)
pub trait ToPyImage {
    fn to_pyimage(self) -> Result<PyImage, ImageError>;
}

pub trait ToPyImageU16 {
    fn to_pyimage_u16(self) -> Result<PyImageU16, ImageError>;
}

pub trait ToPyImageF32 {
    fn to_pyimage_f32(self) -> Result<PyImageF32, ImageError>;
}

impl<const C: usize> ToPyImage for Image<u8, C, CpuAllocator> {
    fn to_pyimage(self) -> Result<PyImage, ImageError> {
        Python::attach(|py| unsafe {
            let array = PyArray::<u8, _>::new(py, [self.height(), self.width(), C], false);
            let contiguous = match self.0.to_standard_layout(CpuAllocator) {
                Ok(c) => c,
                Err(_) => {
                    let expected = self.height() * self.width() * C;
                    let actual = self.0.numel();
                    return Err(ImageError::InvalidChannelShape(actual, expected));
                }
            };
            std::ptr::copy_nonoverlapping(
                contiguous.storage.as_ptr(),
                array.data(),
                contiguous.numel(),
            );
            Ok(array.unbind())
        })
    }
}

// Implement ToPyImage for typed color spaces
impl ToPyImage for Rgb8<CpuAllocator> {
    fn to_pyimage(self) -> Result<PyImage, ImageError> {
        self.0.to_pyimage()
    }
}

impl ToPyImage for Rgba8<CpuAllocator> {
    fn to_pyimage(self) -> Result<PyImage, ImageError> {
        self.0.to_pyimage()
    }
}

impl ToPyImage for Bgr8<CpuAllocator> {
    fn to_pyimage(self) -> Result<PyImage, ImageError> {
        self.0.to_pyimage()
    }
}

impl ToPyImage for Bgra8<CpuAllocator> {
    fn to_pyimage(self) -> Result<PyImage, ImageError> {
        self.0.to_pyimage()
    }
}

impl ToPyImage for Gray8<CpuAllocator> {
    fn to_pyimage(self) -> Result<PyImage, ImageError> {
        self.0.to_pyimage()
    }
}

impl<const C: usize> ToPyImageU16 for Image<u16, C, CpuAllocator> {
    fn to_pyimage_u16(self) -> Result<PyImageU16, ImageError> {
        Python::attach(|py| unsafe {
            let array = PyArray::<u16, _>::new(py, [self.height(), self.width(), C], false);
            let contiguous = match self.0.to_standard_layout(CpuAllocator) {
                Ok(c) => c,
                Err(_) => {
                    let expected = self.height() * self.width() * C;
                    let actual = self.0.numel();
                    return Err(ImageError::InvalidChannelShape(actual, expected));
                }
            };
            std::ptr::copy_nonoverlapping(
                contiguous.storage.as_ptr(),
                array.data(),
                contiguous.numel(),
            );
            Ok(array.unbind())
        })
    }
}

// Implement ToPyImageU16 for typed color spaces
impl ToPyImageU16 for Rgb16<CpuAllocator> {
    fn to_pyimage_u16(self) -> Result<PyImageU16, ImageError> {
        self.0.to_pyimage_u16()
    }
}

impl ToPyImageU16 for Rgba16<CpuAllocator> {
    fn to_pyimage_u16(self) -> Result<PyImageU16, ImageError> {
        self.0.to_pyimage_u16()
    }
}

impl ToPyImageU16 for Bgr16<CpuAllocator> {
    fn to_pyimage_u16(self) -> Result<PyImageU16, ImageError> {
        self.0.to_pyimage_u16()
    }
}

impl ToPyImageU16 for Bgra16<CpuAllocator> {
    fn to_pyimage_u16(self) -> Result<PyImageU16, ImageError> {
        self.0.to_pyimage_u16()
    }
}

impl ToPyImageU16 for Gray16<CpuAllocator> {
    fn to_pyimage_u16(self) -> Result<PyImageU16, ImageError> {
        self.0.to_pyimage_u16()
    }
}

impl<const C: usize> ToPyImageF32 for Image<f32, C, CpuAllocator> {
    fn to_pyimage_f32(self) -> Result<PyImageF32, ImageError> {
        Python::attach(|py| unsafe {
            let array = PyArray::<f32, _>::new(py, [self.height(), self.width(), C], false);
            let contiguous = match self.0.to_standard_layout(CpuAllocator) {
                Ok(c) => c,
                Err(_) => {
                    let expected = self.height() * self.width() * C;
                    let actual = self.0.numel();
                    return Err(ImageError::InvalidChannelShape(actual, expected));
                }
            };
            std::ptr::copy_nonoverlapping(
                contiguous.storage.as_ptr(),
                array.data(),
                contiguous.numel(),
            );
            Ok(array.unbind())
        })
    }
}

// Implement ToPyImageF32 for typed color spaces
impl ToPyImageF32 for Rgbf32<CpuAllocator> {
    fn to_pyimage_f32(self) -> Result<PyImageF32, ImageError> {
        self.0.to_pyimage_f32()
    }
}

impl ToPyImageF32 for Rgbaf32<CpuAllocator> {
    fn to_pyimage_f32(self) -> Result<PyImageF32, ImageError> {
        self.0.to_pyimage_f32()
    }
}

impl ToPyImageF32 for Bgrf32<CpuAllocator> {
    fn to_pyimage_f32(self) -> Result<PyImageF32, ImageError> {
        self.0.to_pyimage_f32()
    }
}

impl ToPyImageF32 for Bgraf32<CpuAllocator> {
    fn to_pyimage_f32(self) -> Result<PyImageF32, ImageError> {
        self.0.to_pyimage_f32()
    }
}

impl ToPyImageF32 for Grayf32<CpuAllocator> {
    fn to_pyimage_f32(self) -> Result<PyImageF32, ImageError> {
        self.0.to_pyimage_f32()
    }
}

impl ToPyImageF32 for Hsvf32<CpuAllocator> {
    fn to_pyimage_f32(self) -> Result<PyImageF32, ImageError> {
        self.0.to_pyimage_f32()
    }
}
/// Trait to convert a PyImage (3D numpy array of u8) to an image
pub trait FromPyImage<const C: usize> {
    fn from_pyimage(image: PyImage) -> Result<Image<u8, C, CpuAllocator>, ImageError>;
}

pub trait FromPyImageU16<const C: usize> {
    fn from_pyimage_u16(image: PyImageU16) -> Result<Image<u16, C, CpuAllocator>, ImageError>;
}

pub trait FromPyImageF32<const C: usize> {
    fn from_pyimage_f32(image: PyImageF32) -> Result<Image<f32, C, CpuAllocator>, ImageError>;
}

impl<const C: usize> FromPyImage<C> for Image<u8, C, CpuAllocator> {
    fn from_pyimage(image: PyImage) -> Result<Image<u8, C, CpuAllocator>, ImageError> {
        Python::attach(|py| {
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

            Image::new(size, data, CpuAllocator)
        })
    }
}

impl<const C: usize> FromPyImageU16<C> for Image<u16, C, CpuAllocator> {
    fn from_pyimage_u16(image: PyImageU16) -> Result<Image<u16, C, CpuAllocator>, ImageError> {
        Python::attach(|py| {
            let pyarray = image.bind(py);
            let data = match pyarray.to_vec() {
                Ok(d) => d,
                Err(_) => return Err(ImageError::ImageDataNotContiguous),
            };

            let size = ImageSize {
                width: pyarray.shape()[1],
                height: pyarray.shape()[0],
            };

            Image::new(size, data, CpuAllocator)
        })
    }
}

impl<const C: usize> FromPyImageF32<C> for Image<f32, C, CpuAllocator> {
    fn from_pyimage_f32(image: PyImageF32) -> Result<Image<f32, C, CpuAllocator>, ImageError> {
        Python::attach(|py| {
            let pyarray = image.bind(py);
            let data = match pyarray.to_vec() {
                Ok(d) => d,
                Err(_) => return Err(ImageError::ImageDataNotContiguous),
            };

            let size = ImageSize {
                width: pyarray.shape()[1],
                height: pyarray.shape()[0],
            };

            Image::new(size, data, CpuAllocator)
        })
    }
}

#[pyclass(name = "ImageSize", frozen)]
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
