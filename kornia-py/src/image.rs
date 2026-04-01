use numpy::{PyArray, PyArray3, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::PyTypeInfo;

use kornia_image::{
    allocator::{CpuAllocator, ForeignAllocator},
    color_spaces::*,
    Image, ImageError, ImageLayout, ImageSize, PixelFormat,
};
use pyo3::prelude::*;

// type alias for a 3D numpy array of u8
pub type PyImage = Py<PyArray3<u8>>;

// type alias for a 3D numpy array of u16
pub type PyImageU16 = Py<PyArray3<u16>>;

// type alias for a 3D numpy array of f32
pub type PyImageF32 = Py<PyArray3<f32>>;

// TODO: Replace FromPyImage/ToPyImage with zero-copy ForeignAllocator helpers in IO code.
// - Write side (FromPyImage): replace with numpy_as_image() for zero-copy reads.
//   Requires kornia-io write functions to accept generic allocators (not just CpuAllocator).
// - Read side (ToPyImage): replace with alloc_output_pyarray() + direct decode into PyArray buffer.
//   Requires kornia-io decoders to accept an output pointer / write into pre-allocated memory.
// - Also applies to apriltag.rs (one from_pyimage call).
// These traits are only used by io/ and apriltag.rs — all imgproc bindings already use zero-copy.

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
                    let array =
                        PyArray::<$dtype, _>::new(py, [self.height(), self.width(), C], false);
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
    ToPyImage,
    to_pyimage,
    PyImage,
    Rgb8<CpuAllocator>,
    Rgba8<CpuAllocator>,
    Bgr8<CpuAllocator>,
    Bgra8<CpuAllocator>,
    Gray8<CpuAllocator>,
);

// u16 color spaces
impl_colorspace_to_pyarray!(
    ToPyImageU16,
    to_pyimage_u16,
    PyImageU16,
    Rgb16<CpuAllocator>,
    Rgba16<CpuAllocator>,
    Bgr16<CpuAllocator>,
    Bgra16<CpuAllocator>,
    Gray16<CpuAllocator>,
);

// f32 color spaces
impl_colorspace_to_pyarray!(
    ToPyImageF32,
    to_pyimage_f32,
    PyImageF32,
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

/// Represents the dimensions of an image.
///
/// # Fields
///
/// * `width` - The width of the image.
/// * `height` - The height of the image.
#[pyclass(name = "ImageSize", frozen)]
#[derive(Clone)]
pub struct PyImageSize {
    inner: ImageSize,
}

#[pymethods]
impl PyImageSize {
    /// Creates a new ImageSize instance.
    ///
    /// # Arguments
    ///
    /// * `width` - The width of the image.
    /// * `height` - The height of the image.
    #[new]
    pub fn new(width: usize, height: usize) -> PyResult<PyImageSize> {
        let inner = ImageSize { width, height };
        Ok(PyImageSize { inner })
    }

    /// Returns the width of the image.
    #[getter]
    pub fn width(&self) -> usize {
        self.inner.width
    }

    /// Returns the height of the image.
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

/// Represents the data type of the image pixels.
///
/// Supports standard unsigned 8-bit (U8), unsigned 16-bit (U16),
/// and 32-bit floating point (F32) formats.
#[pyclass(name = "PixelFormat")]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyPixelFormat {
    U8,
    U16,
    F32,
}

impl From<PixelFormat> for PyPixelFormat {
    fn from(value: PixelFormat) -> Self {
        match value {
            PixelFormat::U8 => PyPixelFormat::U8,
            PixelFormat::U16 => PyPixelFormat::U16,
            PixelFormat::F32 => PyPixelFormat::F32,
        }
    }
}

impl From<PyPixelFormat> for PixelFormat {
    fn from(value: PyPixelFormat) -> Self {
        match value {
            PyPixelFormat::U8 => PixelFormat::U8,
            PyPixelFormat::U16 => PixelFormat::U16,
            PyPixelFormat::F32 => PixelFormat::F32,
        }
    }
}

/// Represents the memory layout and format of an image.
///
/// Defines the physical dimensions, number of channels, and pixel
/// data type required to correctly interpret an image buffer.
///
/// # Fields
///
/// * `image_size` - The dimensions of the image.
/// * `channels` - The number of channels (e.g., 3 for RGB).
/// * `pixel_format` - The data type of the pixels.
#[pyclass(name = "ImageLayout", frozen)]
#[derive(Clone)]
pub struct PyImageLayout {
    inner: ImageLayout,
}

#[pymethods]
impl PyImageLayout {
    /// Creates a new ImageLayout instance.
    ///
    /// # Arguments
    ///
    /// * `image_size` - The dimensions of the image.
    /// * `channels` - The number of channels in the image.
    /// * `pixel_format` - The data type of the pixels.
    #[new]
    pub fn new(
        image_size: PyImageSize,
        channels: u8,
        pixel_format: PyPixelFormat,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: ImageLayout {
                image_size: image_size.clone().into(),
                channels,
                pixel_format: pixel_format.into(),
            },
        })
    }

    /// Returns the dimensions of the image.
    #[getter]
    pub fn image_size(&self) -> PyImageSize {
        self.inner.image_size.into()
    }

    /// Returns the number of channels in the image.
    #[getter]
    pub fn channels(&self) -> u8 {
        self.inner.channels
    }

    /// Returns the pixel format of the image.
    #[getter]
    pub fn pixel_format(&self) -> PyPixelFormat {
        self.inner.pixel_format.into()
    }

    fn __repr__(&self) -> PyResult<String> {
        let size = self.inner.image_size;
        let pf_str = match self.inner.pixel_format {
            PixelFormat::U8 => "U8",
            PixelFormat::U16 => "U16",
            PixelFormat::F32 => "F32",
        };
        Ok(format!(
            "ImageLayout(image_size=ImageSize(width={}, height={}), channels={}, pixel_format=PixelFormat.{})",
            size.width,
            size.height,
            self.channels(),
            pf_str
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

pub(crate) const LUMINANCE_WEIGHTS: [f64; 3] = [0.299, 0.587, 0.114];

/// Convert any Display error into a PyException.
pub(crate) fn to_pyerr(e: impl std::fmt::Display) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e))
}

/// Zero-copy wrap a numpy u8 array as a Rust Image for reading.
///
/// The caller MUST ensure the Py<PyArray3<u8>> stays alive for the lifetime of the Image.
pub(crate) unsafe fn numpy_as_image<const C: usize>(
    py: Python<'_>,
    image: &Py<PyArray3<u8>>,
) -> PyResult<Image<u8, C, ForeignAllocator>> {
    let arr = image.bind(py);
    if !arr.is_c_contiguous() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "numpy array is not C-contiguous",
        ));
    }
    let shape = arr.shape();
    if shape[2] != C {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "expected {} channels, got {}",
            C, shape[2]
        )));
    }
    let (h, w) = (shape[0], shape[1]);
    let size = ImageSize { width: w, height: h };
    Image::from_raw_parts(size, arr.data() as *const u8, h * w * C, ForeignAllocator)
        .map_err(to_pyerr)
}

/// Allocate a PyArray3<u8> and wrap its buffer as a mutable Rust Image for writing.
pub(crate) unsafe fn alloc_output_pyarray<const C: usize>(
    py: Python<'_>,
    size: ImageSize,
) -> PyResult<(Image<u8, C, ForeignAllocator>, Py<PyArray3<u8>>)> {
    let arr = PyArray::<u8, _>::new(py, [size.height, size.width, C], false);
    let len = size.height * size.width * C;
    let img = Image::from_raw_parts(size, arr.data() as *const u8, len, ForeignAllocator)
        .map_err(to_pyerr)?;
    Ok((img, arr.unbind()))
}

/// Allocate a PyArray3<f32> and wrap its buffer as a mutable Rust Image for writing.
pub(crate) unsafe fn alloc_output_pyarray_f32<const C: usize>(
    py: Python<'_>,
    size: ImageSize,
) -> PyResult<(Image<f32, C, ForeignAllocator>, Py<PyArray3<f32>>)> {
    let arr = PyArray::<f32, _>::new(py, [size.height, size.width, C], false);
    let len = size.height * size.width * C * std::mem::size_of::<f32>();
    let img = Image::from_raw_parts(size, arr.data() as *const f32, len, ForeignAllocator)
        .map_err(to_pyerr)?;
    Ok((img, arr.unbind()))
}

/// Copy numpy u8 data into a CpuAllocator f32 Image (for Category B ops needing f32).
///
/// Uses zero-copy read via numpy_as_image, then a single u8→f32 collect.
pub(crate) fn numpy_to_f32_image<const C: usize>(
    py: Python<'_>,
    image: &Py<PyArray3<u8>>,
) -> PyResult<Image<f32, C, CpuAllocator>> {
    let src = unsafe { numpy_as_image::<C>(py, image)? };
    let f32_data: Vec<f32> = src.as_slice().iter().map(|&v| v as f32).collect();
    Image::new(src.size(), f32_data, CpuAllocator).map_err(to_pyerr)
}

/// Get raw u8 data and dimensions from a PyArray3.
pub(crate) fn pyarray_data<'py>(arr: &Bound<'py, PyArray3<u8>>) -> (&'py [u8], usize, usize, usize) {
    let s = arr.shape();
    let (h, w, c) = (s[0], s[1], s[2]);
    (unsafe { std::slice::from_raw_parts(arr.data(), h * w * c) }, h, w, c)
}

/// Create a PyArray3<u8> from a Vec with given dimensions.
pub(crate) fn vec_to_pyarray(
    py: Python<'_>,
    data: Vec<u8>,
    h: usize,
    w: usize,
    c: usize,
) -> Py<PyArray3<u8>> {
    unsafe {
        let arr = PyArray::<u8, _>::new(py, [h, w, c], false);
        std::ptr::copy_nonoverlapping(data.as_ptr(), arr.data(), data.len());
        arr.unbind()
    }
}

/// Create a PyArray3<f32> from a Vec with given dimensions.
fn vec_to_pyarray_f32(
    py: Python<'_>,
    data: Vec<f32>,
    h: usize,
    w: usize,
    c: usize,
) -> Py<PyArray3<f32>> {
    unsafe {
        let arr = PyArray::<f32, _>::new(py, [h, w, c], false);
        std::ptr::copy_nonoverlapping(data.as_ptr(), arr.data(), data.len());
        arr.unbind()
    }
}

/// Nearest-neighbor resize for any channel count.
fn resize_nearest(
    src: &[u8],
    src_h: usize,
    src_w: usize,
    dst_h: usize,
    dst_w: usize,
    c: usize,
) -> Vec<u8> {
    let mut out = vec![0u8; dst_h * dst_w * c];
    for y in 0..dst_h {
        let sy = (y * src_h / dst_h).min(src_h - 1);
        for x in 0..dst_w {
            let sx = (x * src_w / dst_w).min(src_w - 1);
            let si = (sy * src_w + sx) * c;
            let di = (y * dst_w + x) * c;
            out[di..di + c].copy_from_slice(&src[si..si + c]);
        }
    }
    out
}

/// Horizontal flip for any channel count.
fn flip_h_generic(src: &[u8], h: usize, w: usize, c: usize) -> Vec<u8> {
    let mut out = vec![0u8; h * w * c];
    for y in 0..h {
        for x in 0..w {
            let si = (y * w + x) * c;
            let di = (y * w + (w - 1 - x)) * c;
            out[di..di + c].copy_from_slice(&src[si..si + c]);
        }
    }
    out
}

/// Vertical flip for any channel count.
fn flip_v_generic(src: &[u8], h: usize, w: usize, c: usize) -> Vec<u8> {
    let mut out = vec![0u8; h * w * c];
    for y in 0..h {
        let sr = y * w * c;
        let dr = (h - 1 - y) * w * c;
        out[dr..dr + w * c].copy_from_slice(&src[sr..sr + w * c]);
    }
    out
}

/// Crop for any channel count.
fn crop_generic(
    src: &[u8],
    src_w: usize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    c: usize,
) -> Vec<u8> {
    let mut out = vec![0u8; h * w * c];
    for row in 0..h {
        let si = ((y + row) * src_w + x) * c;
        let di = row * w * c;
        out[di..di + w * c].copy_from_slice(&src[si..si + w * c]);
    }
    out
}

/// Rotate 90 degrees CCW, k times. Returns (data, new_h, new_w).
fn rot90_generic(src: &[u8], h: usize, w: usize, c: usize, k: i32) -> (Vec<u8>, usize, usize) {
    match k {
        1 => {
            let mut out = vec![0u8; h * w * c];
            for y in 0..h {
                for x in 0..w {
                    let si = (y * w + x) * c;
                    let di = ((w - 1 - x) * h + y) * c;
                    out[di..di + c].copy_from_slice(&src[si..si + c]);
                }
            }
            (out, w, h)
        }
        2 => {
            let mut out = vec![0u8; h * w * c];
            for y in 0..h {
                for x in 0..w {
                    let si = (y * w + x) * c;
                    let di = ((h - 1 - y) * w + (w - 1 - x)) * c;
                    out[di..di + c].copy_from_slice(&src[si..si + c]);
                }
            }
            (out, h, w)
        }
        3 => {
            let mut out = vec![0u8; h * w * c];
            for y in 0..h {
                for x in 0..w {
                    let si = (y * w + x) * c;
                    let di = (x * h + (h - 1 - y)) * c;
                    out[di..di + c].copy_from_slice(&src[si..si + c]);
                }
            }
            (out, w, h)
        }
        _ => (src.to_vec(), h, w),
    }
}

/// Adjust contrast: (pixel - mean) * factor + mean, clamped to [0, 255].
pub(crate) fn adjust_contrast_generic(src: &[u8], factor: f64) -> Vec<u8> {
    let mean: f64 = src.iter().map(|&v| v as f64).sum::<f64>() / src.len() as f64;
    src.iter()
        .map(|&v| ((v as f64 - mean) * factor + mean).clamp(0.0, 255.0) as u8)
        .collect()
}

/// Adjust saturation for 3-channel RGB. factor=1.0 is identity.
pub(crate) fn adjust_saturation_generic(src: &[u8], npixels: usize, factor: f64) -> Vec<u8> {
    let mut out = vec![0u8; npixels * 3];
    for i in 0..npixels {
        let r = src[i * 3] as f64;
        let g = src[i * 3 + 1] as f64;
        let b = src[i * 3 + 2] as f64;
        let gray = r * LUMINANCE_WEIGHTS[0] + g * LUMINANCE_WEIGHTS[1] + b * LUMINANCE_WEIGHTS[2];
        out[i * 3] = ((gray + (r - gray) * factor).clamp(0.0, 255.0)) as u8;
        out[i * 3 + 1] = ((gray + (g - gray) * factor).clamp(0.0, 255.0)) as u8;
        out[i * 3 + 2] = ((gray + (b - gray) * factor).clamp(0.0, 255.0)) as u8;
    }
    out
}

/// Adjust hue for 3-channel RGB. factor in [-0.5, 0.5].
pub(crate) fn adjust_hue_generic(src: &[u8], npixels: usize, factor: f64) -> Vec<u8> {
    let mut out = vec![0u8; npixels * 3];
    for i in 0..npixels {
        let r = src[i * 3] as f64 / 255.0;
        let g = src[i * 3 + 1] as f64 / 255.0;
        let b = src[i * 3 + 2] as f64 / 255.0;

        let max = r.max(g).max(b);
        let min = r.min(g).min(b);
        let diff = max - min;

        let mut hue = if diff < 1e-10 {
            0.0
        } else if (max - r).abs() < 1e-10 {
            ((g - b) / diff).rem_euclid(6.0)
        } else if (max - g).abs() < 1e-10 {
            (b - r) / diff + 2.0
        } else {
            (r - g) / diff + 4.0
        };
        hue /= 6.0;

        let sat = if max < 1e-10 { 0.0 } else { diff / max };
        let val = max;

        hue = (hue + factor).rem_euclid(1.0);

        let h6 = hue * 6.0;
        let hi = h6.floor() as i32 % 6;
        let f = h6 - h6.floor();
        let p = val * (1.0 - sat);
        let q = val * (1.0 - sat * f);
        let t = val * (1.0 - sat * (1.0 - f));

        let (ro, go, bo) = match hi {
            0 => (val, t, p),
            1 => (q, val, p),
            2 => (p, val, t),
            3 => (p, q, val),
            4 => (t, p, val),
            _ => (val, p, q),
        };

        out[i * 3] = (ro * 255.0).clamp(0.0, 255.0) as u8;
        out[i * 3 + 1] = (go * 255.0).clamp(0.0, 255.0) as u8;
        out[i * 3 + 2] = (bo * 255.0).clamp(0.0, 255.0) as u8;
    }
    out
}

fn mode_from_channels(channels: usize) -> String {
    match channels {
        1 => "L".to_string(),
        3 => "RGB".to_string(),
        4 => "RGBA".to_string(),
        c => format!("{}ch", c),
    }
}

/// A high-level image object backed by numpy arrays and Rust operations.
///
/// Always stores data as HWC (height, width, channels) numpy arrays.
/// RGB color order by default.
///
/// Thread-safe and serialization-friendly for use with Ray Data,
/// multiprocessing, and other parallel execution frameworks.
#[pyclass(name = "Image", weakref)]
pub struct PyImageApi {
    data: Py<PyArray3<u8>>,
    mode: String,
}

impl PyImageApi {
    /// Internal: wrap a Py<PyArray3<u8>> with auto-detected mode
    pub fn wrap(py: Python<'_>, data: Py<PyArray3<u8>>, mode: Option<String>) -> Self {
        let channels = data.bind(py).shape()[2];
        let mode = mode.unwrap_or_else(|| mode_from_channels(channels));
        Self { data, mode }
    }

    /// Internal: wrap a PyImage from other bindings
    pub fn from_pyimage(py: Python<'_>, image: PyImage, mode: Option<String>) -> Self {
        Self::wrap(py, image, mode)
    }
}

#[pymethods]
impl PyImageApi {
    /// Create an Image from a numpy array.
    #[new]
    #[pyo3(signature = (data, mode=None))]
    fn new(py: Python<'_>, data: &Bound<'_, PyAny>, mode: Option<String>) -> PyResult<Self> {
        let shape: Vec<usize> = data.getattr("shape")?.extract()?;
        let arr = if shape.len() == 2 {
            data.call_method1("reshape", ((shape[0], shape[1], 1usize),))?
                .extract::<Py<PyArray3<u8>>>()?
        } else if shape.len() == 3 {
            data.extract::<Py<PyArray3<u8>>>()?
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Expected 2D or 3D array, got {}D",
                shape.len()
            )));
        };
        Ok(Self::wrap(py, arr, mode))
    }

    // --- Static constructors ---

    /// Create an Image from raw pixel data or numpy array. Zero-copy for numpy.
    #[staticmethod]
    #[pyo3(signature = (data, width=None, height=None, channels=3, mode=None))]
    fn frombytes(
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        width: Option<usize>,
        height: Option<usize>,
        channels: Option<usize>,
        mode: Option<String>,
    ) -> PyResult<Self> {
        // Try as 3D numpy array (zero-copy)
        if let Ok(arr) = data.extract::<Py<PyArray3<u8>>>() {
            return Ok(Self::wrap(py, arr, mode));
        }

        // Try shape attribute for 2D arrays or array-like objects
        if let Ok(shape_attr) = data.getattr("shape") {
            if let Ok(shape) = shape_attr.extract::<Vec<usize>>() {
                if shape.len() == 2 {
                    // 2D: reshape to (H, W, 1) — zero-copy view for numpy arrays
                    if let Ok(arr) = data
                        .call_method1("reshape", ((shape[0], shape[1], 1usize),))
                        .and_then(|r| Ok(r.extract::<Py<PyArray3<u8>>>()?))

                    {
                        return Ok(Self::wrap(py, arr, mode));
                    }
                } else if shape.len() != 3 {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Expected 2D or 3D array, got {}D",
                        shape.len()
                    )));
                }
            }
        }

        // Raw bytes/bytearray/memoryview — need explicit dimensions
        let w = width.ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "width and height are required for raw bytes input",
            )
        })?;
        let h = height.ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "width and height are required for raw bytes input",
            )
        })?;
        let c = channels.unwrap_or(3);
        let expected = h * w * c;

        // Extract bytes from various sources
        let bytes: Vec<u8> = if let Ok(v) = data.extract::<Vec<u8>>() {
            v
        } else if let Ok(tobytes) = data.call_method0("tobytes") {
            tobytes.extract::<Vec<u8>>()?
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Cannot extract bytes from input",
            ));
        };

        if bytes.len() != expected {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Expected {} bytes ({}x{}x{}), got {}",
                expected, h, w, c, bytes.len()
            )));
        }

        let arr = vec_to_pyarray(py, bytes, h, w, c);
        Ok(Self::wrap(py, arr, mode))
    }

    /// Load an image from file (JPEG, PNG, TIFF). Returns RGB.
    #[staticmethod]
    fn load(py: Python<'_>, path: &str) -> PyResult<Self> {
        let path_str = pyo3::types::PyString::new(py, path);
        let arr_any = crate::io::functional::read_image(path_str.into_any())?;
        let arr: Py<PyArray3<u8>> = arr_any.extract(py)?;
        Ok(Self::wrap(py, arr, None))
    }

    /// Decode encoded image bytes (JPEG, PNG) into an Image.
    #[staticmethod]
    #[pyo3(signature = (data, mode="RGB"))]
    fn decode(py: Python<'_>, data: &[u8], mode: &str) -> PyResult<Self> {
        let native_mode = match mode {
            "RGB" => "rgb",
            "RGBA" => "rgba",
            "L" => "mono",
            _ => "rgb",
        };

        let arr = if data.len() >= 2 && data[0] == 0xff && data[1] == 0xd8 {
            // JPEG
            match crate::io::jpegturbo::decode_image_jpegturbo(data, native_mode) {
                Ok(a) => a,
                Err(_) => crate::io::jpeg::decode_image_jpeg(data)?,
            }
        } else if data.len() >= 4 && &data[0..4] == b"\x89PNG" {
            // PNG: parse dimensions from IHDR
            if data.len() < 24 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Data too short to be a valid PNG",
                ));
            }
            let width = u32::from_be_bytes([data[16], data[17], data[18], data[19]]) as usize;
            let height = u32::from_be_bytes([data[20], data[21], data[22], data[23]]) as usize;
            crate::io::png::decode_image_png_u8(data, (height, width), native_mode)?
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unsupported image format: not JPEG or PNG",
            ));
        };

        Ok(Self::wrap(py, arr, Some(mode.to_string())))
    }

    // --- Properties ---

    #[getter]
    pub fn width(&self, py: Python<'_>) -> usize {
        self.data.bind(py).shape()[1]
    }

    #[getter]
    pub fn height(&self, py: Python<'_>) -> usize {
        self.data.bind(py).shape()[0]
    }

    #[getter]
    fn channels(&self, py: Python<'_>) -> usize {
        self.data.bind(py).shape()[2]
    }

    #[getter]
    pub fn mode(&self) -> &str {
        &self.mode
    }

    #[getter]
    fn size(&self, py: Python<'_>) -> (usize, usize) {
        let s = self.data.bind(py).shape();
        (s[1], s[0])
    }

    #[getter]
    fn shape(&self, py: Python<'_>) -> (usize, usize, usize) {
        let s = self.data.bind(py).shape();
        (s[0], s[1], s[2])
    }

    #[getter]
    fn dtype(&self, py: Python<'_>) -> Py<PyAny> {
        self.data.bind(py).dtype().clone().unbind().into_any()
    }

    #[getter]
    pub fn data(&self, py: Python<'_>) -> Py<PyArray3<u8>> {
        self.data.clone_ref(py)
    }

    #[getter]
    fn nbytes(&self, py: Python<'_>) -> usize {
        let s = self.data.bind(py).shape();
        s[0] * s[1] * s[2]
    }

    // --- IO ---

    /// Save image to file. Format detected from extension.
    #[pyo3(signature = (path, quality=95))]
    fn save(&self, py: Python<'_>, path: &str, quality: u8) -> PyResult<()> {
        let c = self.data.bind(py).shape()[2];
        if c != 3 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "save requires 3-channel RGB image, got {} channels",
                c
            )));
        }

        let ext = path.rsplit('.').next().unwrap_or("").to_lowercase();
        match ext.as_str() {
            "jpg" | "jpeg" => {
                crate::io::jpeg::write_image_jpeg(path, self.data.clone_ref(py), "rgb", quality)
            }
            "png" => crate::io::png::write_image_png_u8(path, self.data.clone_ref(py), "rgb"),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unsupported format: .{}. Supported: .jpg, .jpeg, .png",
                ext
            ))),
        }
    }

    /// Return a copy of the underlying numpy array.
    fn to_numpy(&self, py: Python<'_>) -> PyResult<Py<PyArray3<u8>>> {
        let copy = self.data.bind(py).call_method0("copy")?;
        Ok(copy.extract()?)
    }

    /// Return a deep copy of this image.
    pub fn copy(&self, py: Python<'_>) -> PyResult<Self> {
        let copy: Py<PyArray3<u8>> = self.data.bind(py).call_method0("copy")?.extract()?;
        Ok(Self {
            data: copy,
            mode: self.mode.clone(),
        })
    }

    // --- Chainable transforms ---

    /// Resize image to (width, height).
    #[pyo3(signature = (width, height, interpolation="bilinear"))]
    fn resize(
        &self,
        py: Python<'_>,
        width: usize,
        height: usize,
        interpolation: &str,
    ) -> PyResult<Self> {
        let arr = self.data.bind(py);
        let s = arr.shape();
        let c = s[2];
        if c == 3 {
            let result =
                crate::resize::resize(py, self.data.clone_ref(py), (height, width), interpolation)?;
            Ok(Self::wrap(py, result, Some(self.mode.clone())))
        } else {
            let (src, src_h, src_w, _) = pyarray_data(&arr);
            let out = resize_nearest(src, src_h, src_w, height, width, c);
            Ok(Self::wrap(py, vec_to_pyarray(py, out, height, width, c), Some(self.mode.clone())))
        }
    }

    /// Flip image horizontally.
    pub fn flip_horizontal(&self, py: Python<'_>) -> PyResult<Self> {
        let arr = self.data.bind(py);
        let c = arr.shape()[2];
        if c == 3 {
            let result = crate::flip::horizontal_flip(py, self.data.clone_ref(py))?;
            Ok(Self::wrap(py, result, Some(self.mode.clone())))
        } else {
            let (src, h, w, _) = pyarray_data(&arr);
            let out = flip_h_generic(src, h, w, c);
            Ok(Self::wrap(py, vec_to_pyarray(py, out, h, w, c), Some(self.mode.clone())))
        }
    }

    /// Flip image vertically.
    pub fn flip_vertical(&self, py: Python<'_>) -> PyResult<Self> {
        let arr = self.data.bind(py);
        let c = arr.shape()[2];
        if c == 3 {
            let result = crate::flip::vertical_flip(py, self.data.clone_ref(py))?;
            Ok(Self::wrap(py, result, Some(self.mode.clone())))
        } else {
            let (src, h, w, _) = pyarray_data(&arr);
            let out = flip_v_generic(src, h, w, c);
            Ok(Self::wrap(py, vec_to_pyarray(py, out, h, w, c), Some(self.mode.clone())))
        }
    }

    /// Crop image at (x, y) with given width and height.
    pub fn crop(
        &self,
        py: Python<'_>,
        x: usize,
        y: usize,
        width: usize,
        height: usize,
    ) -> PyResult<Self> {
        let arr = self.data.bind(py);
        let c = arr.shape()[2];
        if c == 3 {
            let result = crate::crop::crop(py, self.data.clone_ref(py), x, y, width, height)?;
            Ok(Self::wrap(py, result, Some(self.mode.clone())))
        } else {
            let (src, _, src_w, _) = pyarray_data(&arr);
            let out = crop_generic(src, src_w, x, y, width, height, c);
            Ok(Self::wrap(
                py,
                vec_to_pyarray(py, out, height, width, c),
                Some(self.mode.clone()),
            ))
        }
    }

    /// Apply Gaussian blur.
    #[pyo3(signature = (kernel_size=3, sigma=1.0))]
    fn gaussian_blur(&self, py: Python<'_>, kernel_size: usize, sigma: f32) -> PyResult<Self> {
        let c = self.data.bind(py).shape()[2];
        if c == 3 {
            let result = crate::blur::gaussian_blur(
                py,
                self.data.clone_ref(py),
                (kernel_size, kernel_size),
                (sigma, sigma),
            )?;
            Ok(Self::wrap(py, result, Some(self.mode.clone())))
        } else {
            self.copy(py)
        }
    }

    /// Apply box blur.
    #[pyo3(signature = (kernel_size=3))]
    fn box_blur(&self, py: Python<'_>, kernel_size: usize) -> PyResult<Self> {
        let c = self.data.bind(py).shape()[2];
        if c == 3 {
            let result =
                crate::blur::box_blur(py, self.data.clone_ref(py), (kernel_size, kernel_size))?;
            Ok(Self::wrap(py, result, Some(self.mode.clone())))
        } else {
            self.copy(py)
        }
    }

    /// Adjust brightness. Factor is additive in [0,1] range.
    pub fn adjust_brightness(&self, py: Python<'_>, factor: f32) -> PyResult<Self> {
        let arr = self.data.bind(py);
        let c = arr.shape()[2];
        if c == 3 {
            let result = crate::brightness::adjust_brightness_py(py, self.data.clone_ref(py), factor)?;
            Ok(Self::wrap(py, result, Some(self.mode.clone())))
        } else {
            let (src, h, w, _) = pyarray_data(&arr);
            let offset = factor * 255.0;
            let out: Vec<u8> = src
                .iter()
                .map(|&v| (v as f32 + offset).clamp(0.0, 255.0) as u8)
                .collect();
            Ok(Self::wrap(py, vec_to_pyarray(py, out, h, w, c), Some(self.mode.clone())))
        }
    }

    /// Adjust contrast. factor=1.0 is identity, >1 increases contrast.
    pub fn adjust_contrast(&self, py: Python<'_>, factor: f64) -> PyResult<Self> {
        let arr = self.data.bind(py);
        let (src, h, w, c) = pyarray_data(&arr);
        let out = adjust_contrast_generic(src, factor);
        Ok(Self::wrap(py, vec_to_pyarray(py, out, h, w, c), Some(self.mode.clone())))
    }

    /// Adjust saturation. factor=1.0 is identity, 0.0 is grayscale.
    pub fn adjust_saturation(&self, py: Python<'_>, factor: f64) -> PyResult<Self> {
        let arr = self.data.bind(py);
        let s = arr.shape();
        let c = s[2];
        if c != 3 {
            return self.copy(py);
        }
        let (src, h, w, _) = pyarray_data(&arr);
        let out = adjust_saturation_generic(src, h * w, factor);
        Ok(Self::wrap(py, vec_to_pyarray(py, out, h, w, 3), Some(self.mode.clone())))
    }

    /// Adjust hue. factor is in [-0.5, 0.5], fraction of hue wheel.
    pub fn adjust_hue(&self, py: Python<'_>, factor: f64) -> PyResult<Self> {
        let arr = self.data.bind(py);
        let s = arr.shape();
        let c = s[2];
        if c != 3 || factor == 0.0 {
            return self.copy(py);
        }
        let (src, h, w, _) = pyarray_data(&arr);
        let out = adjust_hue_generic(src, h * w, factor);
        Ok(Self::wrap(py, vec_to_pyarray(py, out, h, w, 3), Some(self.mode.clone())))
    }

    /// Normalize image to float32 using mean and std per channel.
    fn normalize(
        &self,
        py: Python<'_>,
        mean: (f32, f32, f32),
        std: (f32, f32, f32),
    ) -> PyResult<Py<PyArray3<f32>>> {
        let arr = self.data.bind(py);
        let s = arr.shape();
        let c = s[2];
        if c == 3 {
            crate::normalize::normalize_mean_std(
                py,
                self.data.clone_ref(py),
                [mean.0, mean.1, mean.2],
                [std.0, std.1, std.2],
            )
        } else {
            let (src, h, w, _) = pyarray_data(&arr);
            let mean_arr = [mean.0, mean.1, mean.2];
            let std_arr = [std.0, std.1, std.2];
            let out: Vec<f32> = src
                .iter()
                .enumerate()
                .map(|(idx, &v)| {
                    let ch = idx % c;
                    let m = if ch < 3 { mean_arr[ch] } else { 0.0 };
                    let s = if ch < 3 { std_arr[ch] } else { 1.0 };
                    (v as f32 / 255.0 - m) / s
                })
                .collect();
            Ok(vec_to_pyarray_f32(py, out, h, w, c))
        }
    }

    /// Convert RGB image to grayscale (1 channel).
    fn to_grayscale(&self, py: Python<'_>) -> PyResult<Self> {
        let c = self.data.bind(py).shape()[2];
        if c == 1 {
            return self.copy(py);
        }
        if c == 3 {
            let result = crate::color::gray_from_rgb(py, self.data.clone_ref(py))?;
            Ok(Self::wrap(py, result, Some("L".to_string())))
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Cannot convert {}-channel image to grayscale",
                c
            )))
        }
    }

    /// Convert grayscale to RGB (3 channels).
    fn to_rgb(&self, py: Python<'_>) -> PyResult<Self> {
        let c = self.data.bind(py).shape()[2];
        if c == 3 {
            return self.copy(py);
        }
        if c == 1 {
            let result = crate::color::rgb_from_gray(py, self.data.clone_ref(py))?;
            Ok(Self::wrap(py, result, Some("RGB".to_string())))
        } else if c == 4 {
            let result = crate::color::rgb_from_rgba(py, self.data.clone_ref(py), None)?;
            Ok(Self::wrap(py, result, Some("RGB".to_string())))
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Cannot convert {}-channel image to RGB",
                c
            )))
        }
    }

    /// Rotate image by angle degrees (counter-clockwise).
    pub fn rotate(&self, py: Python<'_>, angle: f64) -> PyResult<Self> {
        let s = self.data.bind(py).shape();
        let c = s[2];
        if c == 3 {
            let (h, w) = (s[0] as f64, s[1] as f64);
            let (cx, cy) = (w / 2.0, h / 2.0);
            let rad = angle.to_radians();
            let cos_a = rad.cos() as f32;
            let sin_a = rad.sin() as f32;
            let tx = (cx - cos_a as f64 * cx + sin_a as f64 * cy) as f32;
            let ty = (cy - sin_a as f64 * cx - cos_a as f64 * cy) as f32;
            let m = [cos_a, -sin_a, tx, sin_a, cos_a, ty];
            let result =
                crate::warp::warp_affine(py, self.data.clone_ref(py), m, (s[0], s[1]), "bilinear")?;
            Ok(Self::wrap(py, result, Some(self.mode.clone())))
        } else {
            // Pure Rust fallback: 90-degree multiples
            let k = ((angle / 90.0).round() as i32).rem_euclid(4);
            if k == 0 {
                return self.copy(py);
            }
            let arr = self.data.bind(py);
            let (src, h, w, _) = pyarray_data(&arr);
            let (out, new_h, new_w) = rot90_generic(src, h, w, c, k);
            Ok(Self::wrap(
                py,
                vec_to_pyarray(py, out, new_h, new_w, c),
                Some(self.mode.clone()),
            ))
        }
    }

    // --- Serialization for multiprocess (Ray Data, etc.) ---

    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let cls = Self::type_object(py).unbind().into_any();
        let args = pyo3::types::PyTuple::new(
            py,
            &[
                self.data.bind(py).as_any(),
                pyo3::types::PyString::new(py, &self.mode).as_any(),
            ],
        )?
        .unbind();
        Ok((cls, args.into_any()))
    }

    fn __getstate__(&self, py: Python<'_>) -> (Py<PyArray3<u8>>, String) {
        (self.data.clone_ref(py), self.mode.clone())
    }

    fn __setstate__(&mut self, _py: Python<'_>, state: (Py<PyArray3<u8>>, String)) {
        self.data = state.0;
        self.mode = state.1;
    }

    // --- Dunder methods ---

    fn __repr__(&self, py: Python<'_>) -> String {
        let s = self.data.bind(py).shape();
        format!(
            "Image(mode={}, size={}x{}, dtype=uint8)",
            self.mode, s[1], s[0]
        )
    }

    fn __eq__(&self, py: Python<'_>, other: &Self) -> bool {
        if self.mode != other.mode {
            return false;
        }
        let a = self.data.bind(py);
        let b = other.data.bind(py);
        if a.shape() != b.shape() {
            return false;
        }
        let len: usize = a.shape().iter().product();
        let sa = unsafe { std::slice::from_raw_parts(a.data(), len) };
        let sb = unsafe { std::slice::from_raw_parts(b.data(), len) };
        sa == sb
    }

    #[pyo3(signature = (dtype=None, copy=None))]
    fn __array__(
        &self,
        py: Python<'_>,
        dtype: Option<&str>,
        copy: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let arr = self.data.bind(py);
        if let Some(dt) = dtype {
            Ok(arr
                .call_method1("astype", (dt,))?
                .call_method0("copy")?
                .unbind())
        } else if copy.unwrap_or(false) {
            Ok(arr.call_method0("copy")?.unbind())
        } else {
            Ok(self.data.clone_ref(py).into_any())
        }
    }

    fn __len__(&self, py: Python<'_>) -> usize {
        self.data.bind(py).shape()[0]
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __exit__(
        &self,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc_val: Option<&Bound<'_, PyAny>>,
        _exc_tb: Option<&Bound<'_, PyAny>>,
    ) -> bool {
        false
    }
}
