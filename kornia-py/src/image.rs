use numpy::{PyArray, PyArray3, PyArrayMethods, PyUntypedArrayMethods};

use kornia_image::{
    allocator::CpuAllocator, color_spaces::*, ChannelsOrder, Image, ImageError, ImageLayout,
    ImagePixelFormat, ImageSize,
};
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

// Macro to implement image to numpy array conversion
macro_rules! impl_image_to_pyarray {
    ($dtype:ty, $trait:ident, $method:ident, $array_type:ty) => {
        impl<const C: usize> $trait for Image<$dtype, C, CpuAllocator> {
            fn $method(self) -> Result<$array_type, ImageError> {
                Python::attach(|py| unsafe {
                    let array = PyArray::<$dtype, _>::new(py, [self.height(), self.width(), C], false);
                    let contiguous = match self.to_standard_layout(CpuAllocator) {
                        Ok(c) => c,
                        Err(_) => {
                            let expected = self.height() * self.width() * C;
                            let actual = self.numel();
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
    };
}

impl_image_to_pyarray!(u8, ToPyImage, to_pyimage, PyImage);
impl_image_to_pyarray!(u16, ToPyImageU16, to_pyimage_u16, PyImageU16);
impl_image_to_pyarray!(f32, ToPyImageF32, to_pyimage_f32, PyImageF32);

// Macro to implement trait for typed color spaces (delegates to inner Image)
macro_rules! impl_colorspace_to_pyarray {
    ($trait:ident, $method:ident, $return_type:ty, $($type:ty),+ $(,)?) => {
        $(
            impl $trait for $type {
                fn $method(self) -> Result<$return_type, ImageError> {
                    self.0.$method()
                }
            }
        )+
    };
}

// u8 color spaces
impl_colorspace_to_pyarray!(
    ToPyImage, to_pyimage, PyImage,
    Rgb8<CpuAllocator>,
    Rgba8<CpuAllocator>,
    Bgr8<CpuAllocator>,
    Bgra8<CpuAllocator>,
    Gray8<CpuAllocator>,
);

// u16 color spaces
impl_colorspace_to_pyarray!(
    ToPyImageU16, to_pyimage_u16, PyImageU16,
    Rgb16<CpuAllocator>,
    Rgba16<CpuAllocator>,
    Bgr16<CpuAllocator>,
    Bgra16<CpuAllocator>,
    Gray16<CpuAllocator>,
);

// f32 color spaces
impl_colorspace_to_pyarray!(
    ToPyImageF32, to_pyimage_f32, PyImageF32,
    Rgbf32<CpuAllocator>,
    Rgbaf32<CpuAllocator>,
    Bgrf32<CpuAllocator>,
    Bgraf32<CpuAllocator>,
    Grayf32<CpuAllocator>,
    Hsvf32<CpuAllocator>,
);
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

// Macro to implement numpy array to image conversion
macro_rules! impl_pyarray_to_image {
    ($dtype:ty, $trait:ident, $method:ident, $array_type:ty) => {
        impl<const C: usize> $trait<C> for Image<$dtype, C, CpuAllocator> {
            fn $method(image: $array_type) -> Result<Image<$dtype, C, CpuAllocator>, ImageError> {
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
    };
}

impl_pyarray_to_image!(u8, FromPyImage, from_pyimage, PyImage);
impl_pyarray_to_image!(u16, FromPyImageU16, from_pyimage_u16, PyImageU16);
impl_pyarray_to_image!(f32, FromPyImageF32, from_pyimage_f32, PyImageF32);

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

#[pyclass(name = "ImageLayout", frozen)]
#[derive(Clone)]
pub struct PyImageLayout {
    inner: ImageLayout,
}

fn parse_channels_order(order: &str) -> Option<ChannelsOrder> {
    match order.to_lowercase().as_str() {
        "channels_last" => Some(ChannelsOrder::ChannelsLast),
        "channels_first" => Some(ChannelsOrder::ChannelsFirst),
        _ => None,
    }
}

fn parse_pixel_format(pixel_format: &str) -> Option<ImagePixelFormat> {
    match pixel_format.to_lowercase().as_str() {
        "u8" | "uint8" => Some(ImagePixelFormat::U8),
        "u16" | "uint16" => Some(ImagePixelFormat::U16),
        "f32" | "float32" => Some(ImagePixelFormat::F32),
        _ => None,
    }
}

fn channels_order_as_str(order: ChannelsOrder) -> &'static str {
    match order {
        ChannelsOrder::ChannelsLast => "channels_last",
        ChannelsOrder::ChannelsFirst => "channels_first",
    }
}

fn pixel_format_as_str(pixel_format: ImagePixelFormat) -> &'static str {
    match pixel_format {
        ImagePixelFormat::U8 => "u8",
        ImagePixelFormat::U16 => "u16",
        ImagePixelFormat::F32 => "f32",
    }
}

#[pymethods]
impl PyImageLayout {
    #[new]
    pub fn new(
        image_size: PyImageSize,
        channels: u8,
        channels_order: &str,
        pixel_format: &str,
    ) -> PyResult<Self> {
        let order = parse_channels_order(channels_order).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "unsupported channels_order: {}",
                channels_order
            ))
        })?;
        let format = parse_pixel_format(pixel_format).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "unsupported pixel_format: {}",
                pixel_format
            ))
        })?;
        Ok(Self {
            inner: ImageLayout {
                image_size: image_size.clone().into(),
                channels,
                channels_order: order,
                pixel_format: format,
            },
        })
    }

    #[getter]
    pub fn image_size(&self) -> PyImageSize {
        self.inner.image_size.into()
    }

    #[getter]
    pub fn channels(&self) -> u8 {
        self.inner.channels
    }

    #[getter]
    pub fn channels_order(&self) -> &'static str {
        channels_order_as_str(self.inner.channels_order)
    }

    #[getter]
    pub fn pixel_format(&self) -> &'static str {
        pixel_format_as_str(self.inner.pixel_format)
    }

    fn __repr__(&self) -> PyResult<String> {
        let size = self.inner.image_size;
        Ok(format!(
            "ImageLayout(image_size=ImageSize(width={}, height={}), channels={}, channels_order='{}', pixel_format='{}')",
            size.width,
            size.height,
            self.channels(),
            self.channels_order(),
            self.pixel_format()
        ))
    }

    fn __str__(&self) -> PyResult<String> {
        self.__repr__()
    }
}

impl From<ImageLayout> for PyImageLayout {
    fn from(layout: ImageLayout) -> Self {
        PyImageLayout { inner: layout }
    }
}

impl From<PyImageLayout> for ImageLayout {
    fn from(layout: PyImageLayout) -> Self {
        layout.inner
    }
}
