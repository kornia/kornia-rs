use numpy::{PyArray, PyArray3, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::PyTypeInfo;

use kornia_image::{
    allocator::{CpuAllocator, ForeignAllocator},
    color_spaces::*,
    Image, ImageError, ImageLayout, ImageSize, PixelFormat,
};
use pyo3::prelude::*;

pub type PyImage = Py<PyArray3<u8>>;
pub type PyImageU16 = Py<PyArray3<u16>>;
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
#[pyclass(name = "ImageSize", frozen, from_py_object)]
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

    fn __repr__(&self) -> String {
        format!(
            "ImageSize(width: {}, height: {})",
            self.inner.width, self.inner.height
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
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
#[pyclass(name = "PixelFormat", from_py_object)]
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
#[pyclass(name = "ImageLayout", frozen, from_py_object)]
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
                image_size: image_size.into(),
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

pub(crate) fn parse_interpolation(
    s: &str,
) -> PyResult<kornia_imgproc::interpolation::InterpolationMode> {
    use kornia_imgproc::interpolation::InterpolationMode;
    match s.to_lowercase().as_str() {
        "nearest" => Ok(InterpolationMode::Nearest),
        "bilinear" => Ok(InterpolationMode::Bilinear),
        "bicubic" => Ok(InterpolationMode::Bicubic),
        "lanczos" => Ok(InterpolationMode::Lanczos),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Invalid interpolation mode",
        )),
    }
}

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
    let size = ImageSize {
        width: w,
        height: h,
    };
    Image::from_raw_parts(size, arr.data() as *const u8, h * w * C, ForeignAllocator)
        .map_err(to_pyerr)
}

pub(crate) type AllocOutput<T, const C: usize, P> = (Image<T, C, ForeignAllocator>, Py<P>);

pub(crate) unsafe fn alloc_output_pyarray<const C: usize>(
    py: Python<'_>,
    size: ImageSize,
) -> PyResult<AllocOutput<u8, C, PyArray3<u8>>> {
    let arr = PyArray::<u8, _>::new(py, [size.height, size.width, C], false);
    let len = size.height * size.width * C;
    let img = Image::from_raw_parts(size, arr.data() as *const u8, len, ForeignAllocator)
        .map_err(to_pyerr)?;
    Ok((img, arr.unbind()))
}

pub(crate) unsafe fn alloc_output_pyarray_f32<const C: usize>(
    py: Python<'_>,
    size: ImageSize,
) -> PyResult<AllocOutput<f32, C, PyArray3<f32>>> {
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
pub(crate) fn pyarray_data<'py>(
    arr: &Bound<'py, PyArray3<u8>>,
) -> (&'py [u8], usize, usize, usize) {
    let s = arr.shape();
    let (h, w, c) = (s[0], s[1], s[2]);
    (
        unsafe { std::slice::from_raw_parts(arr.data(), h * w * c) },
        h,
        w,
        c,
    )
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

/// Create a `PyArray3<u16>` from a Vec with given dimensions. Sister of
/// `vec_to_pyarray`; lets the u16 IO paths (PNG-16 decode, `frombytes` u16)
/// land into a typed numpy view without a numpy round-trip.
pub(crate) fn vec_to_pyarray_u16(
    py: Python<'_>,
    data: Vec<u16>,
    h: usize,
    w: usize,
    c: usize,
) -> Py<PyArray3<u16>> {
    unsafe {
        let arr = PyArray::<u16, _>::new(py, [h, w, c], false);
        std::ptr::copy_nonoverlapping(data.as_ptr(), arr.data(), data.len());
        arr.unbind()
    }
}

/// Apply brightness using saturating integer add/sub.
/// Compiles to NEON uqadd/uqsub — processes 16 bytes per instruction.
/// Works for both src→dst and in-place (pass same slice as both args).
#[inline]
pub(crate) fn apply_brightness_sat(src: &[u8], dst: &mut [u8], offset: f32) {
    let off_i16 = offset.round() as i16;
    if off_i16 >= 0 {
        let off = (off_i16 as u16).min(255) as u8;
        for (d, &s) in dst.iter_mut().zip(src.iter()) {
            *d = s.saturating_add(off);
        }
    } else {
        let off = ((-off_i16) as u16).min(255) as u8;
        for (d, &s) in dst.iter_mut().zip(src.iter()) {
            *d = s.saturating_sub(off);
        }
    }
}

/// Writes brightness-adjusted pixels directly into a new PyArray — zero intermediate allocations.
pub(crate) fn adjust_brightness_into_pyarray(
    py: Python<'_>,
    src: &[u8],
    offset: f32,
    h: usize,
    w: usize,
    c: usize,
) -> Py<PyArray3<u8>> {
    unsafe {
        let arr = PyArray::<u8, _>::new(py, [h, w, c], false);
        let dst = std::slice::from_raw_parts_mut(arr.data(), src.len());
        apply_brightness_sat(src, dst, offset);
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
/// Returns `Some(k)` where k ∈ {0,1,2,3} if `angle` is *exactly* a
/// multiple of 90° (no epsilon — `90.0001` falls through to the general
/// warp path). Enables snap-to-transpose for the common `rotate(90)` case
/// without silently changing results for near-90° angles.
fn exact_k90(angle: f64) -> Option<u8> {
    let k = angle / 90.0;
    if k == k.trunc() && k.is_finite() {
        Some(((k as i64).rem_euclid(4)) as u8)
    } else {
        None
    }
}

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

/// Adjust contrast via LUT: lut[v] = clamp(mean + (v - mean) * factor).
fn adjust_contrast_into_pyarray(
    py: Python<'_>,
    src: &[u8],
    factor: f64,
    h: usize,
    w: usize,
    c: usize,
) -> Py<PyArray3<u8>> {
    let mean = src.iter().map(|&v| v as u64).sum::<u64>() as f64 / src.len() as f64;
    let mut lut = [0u8; 256];
    for (v, slot) in lut.iter_mut().enumerate() {
        *slot = (mean + (v as f64 - mean) * factor).clamp(0.0, 255.0) as u8;
    }
    unsafe {
        let arr = PyArray::<u8, _>::new(py, [h, w, c], false);
        let dst = std::slice::from_raw_parts_mut(arr.data(), src.len());
        for (d, &s) in dst.iter_mut().zip(src.iter()) {
            *d = lut[s as usize];
        }
        arr.unbind()
    }
}

/// Adjust saturation: two-pass (grayscale + blend) for vectorization.
fn adjust_saturation_into_pyarray(
    py: Python<'_>,
    src: &[u8],
    npixels: usize,
    factor: f32,
    h: usize,
    w: usize,
) -> Py<PyArray3<u8>> {
    let inv = 1.0 - factor;
    let lw = [
        LUMINANCE_WEIGHTS[0] as f32,
        LUMINANCE_WEIGHTS[1] as f32,
        LUMINANCE_WEIGHTS[2] as f32,
    ];
    unsafe {
        let arr = PyArray::<u8, _>::new(py, [h, w, 3], false);
        let dst = std::slice::from_raw_parts_mut(arr.data(), npixels * 3);
        // Pass 1: grayscale
        let mut gray = vec![0u8; npixels];
        for i in 0..npixels {
            let base = i * 3;
            let r = *src.get_unchecked(base) as f32;
            let g = *src.get_unchecked(base + 1) as f32;
            let b = *src.get_unchecked(base + 2) as f32;
            *gray.get_unchecked_mut(i) = (r * lw[0] + g * lw[1] + b * lw[2]) as u8;
        }
        // Pass 2: blend
        for i in 0..npixels {
            let base = i * 3;
            let gw = *gray.get_unchecked(i) as f32 * inv;
            *dst.get_unchecked_mut(base) =
                (gw + *src.get_unchecked(base) as f32 * factor).clamp(0.0, 255.0) as u8;
            *dst.get_unchecked_mut(base + 1) =
                (gw + *src.get_unchecked(base + 1) as f32 * factor).clamp(0.0, 255.0) as u8;
            *dst.get_unchecked_mut(base + 2) =
                (gw + *src.get_unchecked(base + 2) as f32 * factor).clamp(0.0, 255.0) as u8;
        }
        arr.unbind()
    }
}

/// Adjust hue via Rodrigues rotation (branchless, vectorizable).
fn adjust_hue_into_pyarray(
    py: Python<'_>,
    src: &[u8],
    npixels: usize,
    factor: f32,
    h: usize,
    w: usize,
) -> Py<PyArray3<u8>> {
    let angle = factor * std::f32::consts::TAU;
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    let ot = 1.0_f32 / 3.0;
    let st = ot.sqrt();
    let a = cos_a + ot * (1.0 - cos_a);
    let b = ot * (1.0 - cos_a) - st * sin_a;
    let c = ot * (1.0 - cos_a) + st * sin_a;

    unsafe {
        let arr = PyArray::<u8, _>::new(py, [h, w, 3], false);
        let dst = std::slice::from_raw_parts_mut(arr.data(), npixels * 3);
        for i in 0..npixels {
            let base = i * 3;
            let r = *src.get_unchecked(base) as f32;
            let g = *src.get_unchecked(base + 1) as f32;
            let bv = *src.get_unchecked(base + 2) as f32;
            *dst.get_unchecked_mut(base) = (a * r + b * g + c * bv).clamp(0.0, 255.0) as u8;
            *dst.get_unchecked_mut(base + 1) = (c * r + a * g + b * bv).clamp(0.0, 255.0) as u8;
            *dst.get_unchecked_mut(base + 2) = (b * r + c * g + a * bv).clamp(0.0, 255.0) as u8;
        }
        arr.unbind()
    }
}

fn mode_from_channels(channels: usize) -> String {
    match channels {
        1 => "L".to_string(),
        3 => "RGB".to_string(),
        4 => "RGBA".to_string(),
        c => format!("{}ch", c),
    }
}

/// PIL-style mode for a 16-bit grayscale image. Mirrors PIL's `"I;16"`.
fn mode_from_channels_u16(channels: usize) -> String {
    match channels {
        1 => "I;16".to_string(),
        3 => "RGB;16".to_string(),
        4 => "RGBA;16".to_string(),
        c => format!("{}ch;16", c),
    }
}

/// Internal storage variant. The two variants are mutually exclusive — the
/// backing dtype is part of the Image's identity (a uint16 image cannot be
/// silently downcast on access without a copy).
#[derive(Debug)]
pub enum ImageData {
    U8(Py<PyArray3<u8>>),
    U16(Py<PyArray3<u16>>),
}

impl ImageData {
    /// `(height, width, channels)` for either variant, without copying.
    fn shape3(&self, py: Python<'_>) -> [usize; 3] {
        match self {
            ImageData::U8(a) => {
                let s = a.bind(py).shape();
                [s[0], s[1], s[2]]
            }
            ImageData::U16(a) => {
                let s = a.bind(py).shape();
                [s[0], s[1], s[2]]
            }
        }
    }

    fn channels(&self, py: Python<'_>) -> usize {
        self.shape3(py)[2]
    }

    /// dtype name as exposed by numpy. Used for `Image.dtype.name` parity.
    fn dtype_name(&self) -> &'static str {
        match self {
            ImageData::U8(_) => "uint8",
            ImageData::U16(_) => "uint16",
        }
    }

    /// Element size in bytes — 1 for u8, 2 for u16. Used by `nbytes` and
    /// the buffer protocol.
    fn itemsize(&self) -> usize {
        match self {
            ImageData::U8(_) => 1,
            ImageData::U16(_) => 2,
        }
    }

    /// True for 16-bit storage. Used by 8-bit-only methods to fail fast with
    /// a clear `NotImplementedError`.
    fn is_u16(&self) -> bool {
        matches!(self, ImageData::U16(_))
    }
}

/// A high-level image object backed by numpy arrays and Rust operations.
///
/// Always stores data as HWC (height, width, channels) numpy arrays. Supports
/// both 8-bit (`uint8`) and 16-bit (`uint16`) bit depths — the latter is
/// required for depth maps and scientific imagery where lossy transit codecs
/// (JPEG) would corrupt object-edge discontinuities.
///
/// Imgproc methods (resize, flip, blur, color conversions, …) currently only
/// support 8-bit images. Calling them on a 16-bit Image raises
/// `NotImplementedError` with a clear remediation message.
///
/// Thread-safe and serialization-friendly for use with Ray Data,
/// multiprocessing, and other parallel execution frameworks.
#[pyclass(name = "Image", weakref, module = "kornia_rs.image")]
pub struct PyImageApi {
    data: ImageData,
    mode: String,
}

/// Shorthand for constructing a Python `ValueError`.
fn value_err<M: Into<String>>(msg: M) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyValueError, _>(msg.into())
}

/// Shared error for 8-bit-only methods called on a 16-bit Image. We surface
/// a clear remediation path so users hit a known gap, not a mystery type
/// mismatch deep inside an imgproc kernel.
fn u16_imgproc_unsupported(method: &str) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(format!(
        "Image.{}() is not yet implemented for uint16 images. \
         Convert to uint8 first via `np.array(img).astype('uint8')` and \
         re-wrap with `Image.fromarray()`.",
        method
    ))
}

impl PyImageApi {
    /// Wrap an 8-bit numpy array. Mode defaults to `"L"`/`"RGB"`/`"RGBA"`
    /// derived from channel count.
    pub fn wrap(py: Python<'_>, data: Py<PyArray3<u8>>, mode: Option<String>) -> Self {
        let channels = data.bind(py).shape()[2];
        let mode = mode.unwrap_or_else(|| mode_from_channels(channels));
        Self {
            data: ImageData::U8(data),
            mode,
        }
    }

    /// Wrap a 16-bit numpy array. Mode defaults to `"I;16"`/`"RGB;16"`/etc.
    pub fn wrap_u16(py: Python<'_>, data: Py<PyArray3<u16>>, mode: Option<String>) -> Self {
        let channels = data.bind(py).shape()[2];
        let mode = mode.unwrap_or_else(|| mode_from_channels_u16(channels));
        Self {
            data: ImageData::U16(data),
            mode,
        }
    }

    /// Wrap a Vec<u8> result as a new u8 `PyImageApi` with the current mode.
    /// Imgproc methods produce these — the result must be u8 because all
    /// imgproc kernels currently operate on u8.
    fn wrap_vec(&self, py: Python<'_>, out: Vec<u8>, h: usize, w: usize, c: usize) -> Self {
        Self::wrap(
            py,
            vec_to_pyarray(py, out, h, w, c),
            Some(self.mode.clone()),
        )
    }

    /// Returns the u8 backing array, or a `NotImplementedError` if the Image
    /// is u16. Used by 8-bit-only imgproc methods after their early gate.
    fn require_u8<'a>(&'a self, method: &str) -> PyResult<&'a Py<PyArray3<u8>>> {
        match &self.data {
            ImageData::U8(a) => Ok(a),
            ImageData::U16(_) => Err(u16_imgproc_unsupported(method)),
        }
    }

    /// Single canonical encode path used by both `encode()` and `save()`.
    /// Dispatches on (format, dtype, channel count) and returns PNG/JPEG
    /// bytes. JPEG is u8-only by spec; PNG handles u8 (1/3/4 ch) and u16
    /// (1/3/4 ch).
    fn encode_to_bytes(&self, py: Python<'_>, format: &str, quality: u8) -> PyResult<Vec<u8>> {
        let c = self.data.channels(py);
        let is_u16 = self.data.is_u16();

        match format {
            "jpg" | "jpeg" => {
                if is_u16 {
                    return Err(value_err(
                        "JPEG cannot encode 16-bit images (lossy DCT corrupts \
                         object-edge discontinuities). Use \"png\" instead.",
                    ));
                }
                if c != 3 {
                    return Err(value_err(format!(
                        "JPEG requires 3-channel RGB image, got {} channels",
                        c
                    )));
                }
                let arr = match &self.data {
                    ImageData::U8(a) => a.clone_ref(py),
                    ImageData::U16(_) => unreachable!("checked is_u16 above"),
                };
                crate::io::jpeg::encode_image_jpeg(arr, quality)
            }
            "png" => {
                let mut buffer = Vec::new();
                match (&self.data, c) {
                    (ImageData::U8(a), 3) => {
                        let img = Image::<u8, 3, _>::from_pyimage(a.clone_ref(py))
                            .map_err(|e| value_err(e.to_string()))?;
                        kornia_io::png::encode_image_png_rgb8(&img, &mut buffer)
                            .map_err(|e| value_err(e.to_string()))?;
                    }
                    (ImageData::U8(a), 4) => {
                        let img = Image::<u8, 4, _>::from_pyimage(a.clone_ref(py))
                            .map_err(|e| value_err(e.to_string()))?;
                        kornia_io::png::encode_image_png_rgba8(&img, &mut buffer)
                            .map_err(|e| value_err(e.to_string()))?;
                    }
                    (ImageData::U8(a), 1) => {
                        let img = Image::<u8, 1, _>::from_pyimage(a.clone_ref(py))
                            .map_err(|e| value_err(e.to_string()))?;
                        kornia_io::png::encode_image_png_gray8(&img, &mut buffer)
                            .map_err(|e| value_err(e.to_string()))?;
                    }
                    (ImageData::U16(a), 3) => {
                        let img = Image::<u16, 3, _>::from_pyimage_u16(a.clone_ref(py))
                            .map_err(|e| value_err(e.to_string()))?;
                        kornia_io::png::encode_image_png_rgb16(&img, &mut buffer)
                            .map_err(|e| value_err(e.to_string()))?;
                    }
                    (ImageData::U16(a), 4) => {
                        let img = Image::<u16, 4, _>::from_pyimage_u16(a.clone_ref(py))
                            .map_err(|e| value_err(e.to_string()))?;
                        kornia_io::png::encode_image_png_rgba16(&img, &mut buffer)
                            .map_err(|e| value_err(e.to_string()))?;
                    }
                    (ImageData::U16(a), 1) => {
                        let img = Image::<u16, 1, _>::from_pyimage_u16(a.clone_ref(py))
                            .map_err(|e| value_err(e.to_string()))?;
                        kornia_io::png::encode_image_png_gray16(&img, &mut buffer)
                            .map_err(|e| value_err(e.to_string()))?;
                    }
                    _ => {
                        return Err(value_err(format!(
                            "PNG requires 1/3/4-channel image, got {} channels (dtype={})",
                            c,
                            self.data.dtype_name()
                        )))
                    }
                };
                Ok(buffer)
            }
            other => Err(value_err(format!(
                "Unsupported format {:?}. Supported: \"jpeg\"/\"jpg\", \"png\"",
                other
            ))),
        }
    }
}

#[pymethods]
impl PyImageApi {
    /// Create an Image from a numpy array.
    ///
    /// Auto-detects the bit depth: ``uint8`` arrays produce an 8-bit Image,
    /// ``uint16`` arrays produce a 16-bit Image (depth maps, scientific data).
    /// 2D shapes are treated as single-channel and reshaped to ``(H, W, 1)``.
    #[new]
    #[pyo3(signature = (data, mode=None))]
    fn new(py: Python<'_>, data: &Bound<'_, PyAny>, mode: Option<String>) -> PyResult<Self> {
        Self::frombuffer(py, data, mode)
    }

    // --- Static constructors ---

    /// Create a zero-copy Image from a numpy array or array-like object.
    ///
    /// The Image shares memory with the input array — mutations to either
    /// are visible in both. Accepts 2D (H, W) or 3D (H, W, C) arrays. Bit
    /// depth is inferred from the array's dtype: ``uint8`` is the default
    /// path, ``uint16`` is the depth-map path.
    ///
    /// Args:
    ///     data: numpy array (``uint8`` or ``uint16``), 2D or 3D in HWC layout.
    ///     mode: color mode string. Auto-inferred from channels + bit depth
    ///         when omitted (`"RGB"` / `"L"` for u8, `"RGB;16"` / `"I;16"` for u16).
    ///
    /// Returns:
    ///     Image wrapping the array data without copying (when contiguous).
    #[staticmethod]
    #[pyo3(signature = (data, mode=None))]
    fn frombuffer(py: Python<'_>, data: &Bound<'_, PyAny>, mode: Option<String>) -> PyResult<Self> {
        // Fast paths: 3D arrays of the supported dtypes — no shape gymnastics.
        if let Ok(arr) = data.extract::<Py<PyArray3<u8>>>() {
            return Ok(Self::wrap(py, arr, mode));
        }
        if let Ok(arr) = data.extract::<Py<PyArray3<u16>>>() {
            return Ok(Self::wrap_u16(py, arr, mode));
        }

        // 2D fallback: reshape to (H, W, 1) and retry per-dtype. We probe
        // shape first so we don't reshape a non-array object.
        if let Ok(shape_attr) = data.getattr("shape") {
            if let Ok(shape) = shape_attr.extract::<Vec<usize>>() {
                if shape.len() == 2 {
                    let reshaped = data
                        .call_method1("reshape", ((shape[0], shape[1], 1usize),))
                        .map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                                "frombuffer reshape failed: {}",
                                e
                            ))
                        })?;
                    if let Ok(arr) = reshaped.extract::<Py<PyArray3<u8>>>() {
                        return Ok(Self::wrap(py, arr, mode));
                    }
                    if let Ok(arr) = reshaped.extract::<Py<PyArray3<u16>>>() {
                        return Ok(Self::wrap_u16(py, arr, mode));
                    }
                    return Err(value_err(
                        "frombuffer: 2D array dtype must be uint8 or uint16",
                    ));
                } else if shape.len() != 3 {
                    return Err(value_err(format!(
                        "Expected 2D or 3D array, got {}D",
                        shape.len()
                    )));
                }
                // 3D but dtype mismatched — give a precise error.
                return Err(value_err(
                    "frombuffer: 3D array dtype must be uint8 or uint16",
                ));
            }
        }

        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "frombuffer requires a numpy array or array-like object with .shape",
        ))
    }

    /// PIL-compatible alias for :meth:`frombuffer`.
    ///
    /// PIL's :func:`Image.fromarray` is the standard way to wrap a numpy
    /// array — exposing the same name here means existing PIL-style code
    /// works against ``kornia_rs.image.Image`` without renames.
    #[staticmethod]
    #[pyo3(signature = (data, mode=None))]
    fn fromarray(py: Python<'_>, data: &Bound<'_, PyAny>, mode: Option<String>) -> PyResult<Self> {
        Self::frombuffer(py, data, mode)
    }

    /// Create an Image by copying raw pixel data into a new buffer.
    ///
    /// Accepts bytes, bytearray, memoryview, or any object with a
    /// `.tobytes()` method. Width, height, and channels must be specified
    /// so the flat data can be reshaped to (H, W, C).
    ///
    /// Args:
    ///     data: raw pixel data (bytes, bytearray, memoryview, or
    ///         `.tobytes()`-capable).
    ///     width: image width in pixels.
    ///     height: image height in pixels.
    ///     channels: number of channels (default 3).
    ///     mode: color mode string. Auto-inferred from channels + dtype.
    ///     dtype: ``"uint8"`` (default) or ``"uint16"``. ``uint16`` requires
    ///         exactly ``2 * height * width * channels`` bytes (little-endian
    ///         on the host).
    ///
    /// Returns:
    ///     Image owning a copy of the data.
    #[staticmethod]
    #[pyo3(signature = (data, width, height, channels=3, mode=None, dtype=None))]
    fn frombytes(
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        width: usize,
        height: usize,
        channels: Option<usize>,
        mode: Option<String>,
        dtype: Option<&str>,
    ) -> PyResult<Self> {
        let c = channels.unwrap_or(3);
        let dtype = dtype.unwrap_or("uint8");
        let itemsize = match dtype {
            "uint8" | "u8" => 1usize,
            "uint16" | "u16" => 2usize,
            other => {
                return Err(value_err(format!(
                    "Unsupported dtype {:?}: must be \"uint8\" or \"uint16\"",
                    other
                )))
            }
        };
        let expected = height * width * c * itemsize;

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
            return Err(value_err(format!(
                "Expected {} bytes ({}x{}x{}, dtype={}), got {}",
                expected,
                height,
                width,
                c,
                dtype,
                bytes.len()
            )));
        }

        if itemsize == 1 {
            let arr = vec_to_pyarray(py, bytes, height, width, c);
            Ok(Self::wrap(py, arr, mode))
        } else {
            // Re-interpret little-endian byte pairs as uint16. Allocates a
            // typed buffer; numpy will hold the canonical view.
            let mut u16_buf = Vec::with_capacity(bytes.len() / 2);
            for pair in bytes.chunks_exact(2) {
                u16_buf.push(u16::from_le_bytes([pair[0], pair[1]]));
            }
            let arr = vec_to_pyarray_u16(py, u16_buf, height, width, c);
            Ok(Self::wrap_u16(py, arr, mode))
        }
    }

    /// Load an image from a file (JPEG, PNG, TIFF). Auto-detects bit depth
    /// from the file: PNG-16 / TIFF-16 produce a 16-bit Image; everything
    /// else produces an 8-bit Image.
    ///
    /// PIL parity: equivalent to ``PIL.Image.open(path)``.
    #[staticmethod]
    fn load(py: Python<'_>, path: &str) -> PyResult<Self> {
        let path_str = pyo3::types::PyString::new(py, path);
        let arr_any = crate::io::functional::read_image(path_str.into_any())?;
        // The functional dispatcher returns either Py<PyArray3<u8>> or
        // Py<PyArray3<u16>> depending on the file's pixel format. Try the u8
        // path first (the dominant case) and fall through to u16.
        if let Ok(arr) = arr_any.extract::<Py<PyArray3<u8>>>(py) {
            return Ok(Self::wrap(py, arr, None));
        }
        let arr: Py<PyArray3<u16>> = arr_any.extract(py).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "load: file decoded to unsupported dtype (expected uint8 or uint16)",
            )
        })?;
        Ok(Self::wrap_u16(py, arr, None))
    }

    /// Decode encoded image bytes (JPEG, PNG) into an Image.
    ///
    /// Auto-detects the format from magic bytes and the bit depth from the
    /// PNG IHDR chunk. PNG-16 (16-bit) bytes produce a 16-bit Image; PNG-8
    /// and JPEG produce 8-bit Images. ``mode`` selects the channel layout
    /// (``"RGB"`` / ``"RGBA"`` / ``"L"``); for PNG-16 it maps to the 16-bit
    /// equivalent.
    #[staticmethod]
    #[pyo3(signature = (data, mode="RGB"))]
    fn decode(py: Python<'_>, data: &[u8], mode: &str) -> PyResult<Self> {
        let native_mode = match mode {
            "RGB" => "rgb",
            "RGBA" => "rgba",
            "L" => "mono",
            _ => "rgb",
        };

        // JPEG: always 8-bit per channel.
        if data.len() >= 2 && data[0] == 0xff && data[1] == 0xd8 {
            let arr = match crate::io::jpegturbo::decode_image_jpegturbo(data, native_mode) {
                Ok(a) => a,
                Err(_) => crate::io::jpeg::decode_image_jpeg(data)?,
            };
            return Ok(Self::wrap(py, arr, Some(mode.to_string())));
        }

        // PNG: parse dimensions + bit depth from IHDR (length 13 starting at byte 16
        // after the 8-byte signature + 4 length + 4 type bytes). IHDR layout:
        //   width(4) height(4) bit_depth(1) color_type(1) compression(1) filter(1) interlace(1)
        if data.len() >= 4 && &data[0..4] == b"\x89PNG" {
            if data.len() < 26 {
                return Err(value_err("Data too short to be a valid PNG"));
            }
            let width = u32::from_be_bytes([data[16], data[17], data[18], data[19]]) as usize;
            let height = u32::from_be_bytes([data[20], data[21], data[22], data[23]]) as usize;
            let bit_depth = data[24];

            return if bit_depth == 16 {
                let arr =
                    crate::io::png::decode_image_png_u16(data, (height, width), native_mode)?;
                let mode_u16 = match mode {
                    "RGB" => "RGB;16".to_string(),
                    "RGBA" => "RGBA;16".to_string(),
                    "L" => "I;16".to_string(),
                    other => other.to_string(),
                };
                Ok(Self::wrap_u16(py, arr, Some(mode_u16)))
            } else {
                let arr =
                    crate::io::png::decode_image_png_u8(data, (height, width), native_mode)?;
                Ok(Self::wrap(py, arr, Some(mode.to_string())))
            };
        }

        Err(value_err("Unsupported image format: not JPEG or PNG"))
    }

    // --- Properties ---

    #[getter]
    pub fn width(&self, py: Python<'_>) -> usize {
        self.data.shape3(py)[1]
    }

    #[getter]
    pub fn height(&self, py: Python<'_>) -> usize {
        self.data.shape3(py)[0]
    }

    #[getter]
    fn channels(&self, py: Python<'_>) -> usize {
        self.data.channels(py)
    }

    #[getter]
    pub fn mode(&self) -> &str {
        &self.mode
    }

    #[getter]
    fn size(&self, py: Python<'_>) -> (usize, usize) {
        let s = self.data.shape3(py);
        (s[1], s[0])
    }

    #[getter]
    fn shape(&self, py: Python<'_>) -> (usize, usize, usize) {
        let s = self.data.shape3(py);
        (s[0], s[1], s[2])
    }

    #[getter]
    fn dtype(&self, py: Python<'_>) -> Py<PyAny> {
        match &self.data {
            ImageData::U8(a) => a.bind(py).dtype().clone().unbind().into_any(),
            ImageData::U16(a) => a.bind(py).dtype().clone().unbind().into_any(),
        }
    }

    /// The underlying numpy array. Returns ``Py<PyAny>`` because the dtype
    /// depends on the bit depth (uint8 or uint16); callers can pass to
    /// numpy directly or test ``img.dtype`` to branch.
    #[getter]
    pub fn data(&self, py: Python<'_>) -> Py<PyAny> {
        match &self.data {
            ImageData::U8(a) => a.clone_ref(py).into_any(),
            ImageData::U16(a) => a.clone_ref(py).into_any(),
        }
    }

    #[getter]
    fn nbytes(&self, py: Python<'_>) -> usize {
        let s = self.data.shape3(py);
        s[0] * s[1] * s[2] * self.data.itemsize()
    }

    // --- IO ---

    /// Save the image to a file path or file-like object.
    ///
    /// PIL-compatible: ``fp`` accepts either a ``str``/``Path`` (writes a
    /// file directly) or a writable object with ``.write(bytes)`` (writes
    /// encoded bytes — works with ``io.BytesIO()`` for in-memory capture).
    /// Format is resolved in this order:
    ///
    ///   1. explicit ``format=`` argument (case-insensitive: ``"PNG"`` /
    ///      ``"JPEG"`` / ``"JPG"``)
    ///   2. extension on the path (``.png``, ``.jpg``, ``.jpeg``)
    ///   3. error
    ///
    /// 16-bit Images can only save to PNG (JPEG would lose precision and
    /// smear depth discontinuities); the call returns a ``ValueError``
    /// otherwise.
    #[pyo3(signature = (fp, format=None, quality=95))]
    fn save(
        &self,
        py: Python<'_>,
        fp: &Bound<'_, PyAny>,
        format: Option<&str>,
        quality: u8,
    ) -> PyResult<()> {
        // Resolve target type: path-like vs file-like.
        let path_str: Option<String> = fp.extract().ok();

        // Resolve format.
        let resolved_format = match format {
            Some(f) => f.to_lowercase(),
            None => {
                let path = path_str.as_deref().ok_or_else(|| {
                    value_err(
                        "save: format= is required when target is a file-like object",
                    )
                })?;
                path.rsplit('.').next().unwrap_or("").to_lowercase()
            }
        };

        // Encode to bytes (handles u8/u16 dispatch internally).
        let bytes = self.encode_to_bytes(py, &resolved_format, quality)?;

        if let Some(path) = path_str {
            std::fs::write(&path, &bytes).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                    "save: failed to write {}: {}",
                    path, e
                ))
            })
        } else {
            // File-like: requires .write(bytes). PIL accepts BytesIO, file
            // handles, S3 file objects, etc. — anything quacking like a writer.
            let pybytes = pyo3::types::PyBytes::new(py, &bytes);
            fp.call_method1("write", (pybytes,)).map_err(|e| {
                value_err(format!(
                    "save: target object must have .write(bytes) — {}",
                    e
                ))
            })?;
            Ok(())
        }
    }

    /// Encode the image to in-memory bytes — the in-memory complement of `save`.
    ///
    /// PIL parity: ``Image.save(BytesIO(), format=...)`` works because the
    /// underlying encoder writes to any file-like; here we expose the same
    /// thing with a direct ``bytes`` return so callers don't need to set up a
    /// ``BytesIO`` round-trip. Use this when you need to ship an encoded image
    /// over a wire (Zenoh, MQTT, gRPC) or embed it in a container (MCAP).
    ///
    /// Args:
    ///     format: ``"jpeg"`` (alias ``"jpg"``) or ``"png"``. Case-insensitive.
    ///     quality: JPEG quality 1-100 (ignored for PNG). Default 95.
    ///
    /// Returns:
    ///     bytes: The encoded image data.
    ///
    /// Example::
    ///
    ///     img = Image.frombuffer(rgb_array)
    ///     jpeg_bytes = img.encode("jpeg", quality=80)
    ///     png_bytes = img.encode("png")
    #[pyo3(signature = (format, quality=95))]
    fn encode(&self, py: Python<'_>, format: &str, quality: u8) -> PyResult<Vec<u8>> {
        self.encode_to_bytes(py, &format.to_lowercase(), quality)
    }

    /// Return a copy of the underlying numpy array. Dtype matches the
    /// Image's bit depth (``uint8`` or ``uint16``); returns ``Py<PyAny>``
    /// because the static type isn't known at compile time.
    fn to_numpy(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.data {
            ImageData::U8(a) => {
                let copy = a.bind(py).call_method0("copy")?;
                Ok(copy.extract::<Py<PyArray3<u8>>>()?.into_any())
            }
            ImageData::U16(a) => {
                let copy = a.bind(py).call_method0("copy")?;
                Ok(copy.extract::<Py<PyArray3<u16>>>()?.into_any())
            }
        }
    }

    /// Return a deep copy of this image.
    pub fn copy(&self, py: Python<'_>) -> PyResult<Self> {
        let new_data = match &self.data {
            ImageData::U8(a) => {
                let copy: Py<PyArray3<u8>> = a.bind(py).call_method0("copy")?.extract()?;
                ImageData::U8(copy)
            }
            ImageData::U16(a) => {
                let copy: Py<PyArray3<u16>> = a.bind(py).call_method0("copy")?.extract()?;
                ImageData::U16(copy)
            }
        };
        Ok(Self {
            data: new_data,
            mode: self.mode.clone(),
        })
    }

    // --- Chainable transforms ---

    /// Resize image to (width, height). 8-bit only.
    #[pyo3(signature = (width, height, interpolation="bilinear"))]
    fn resize(
        &self,
        py: Python<'_>,
        width: usize,
        height: usize,
        interpolation: &str,
    ) -> PyResult<Self> {
        let data = self.require_u8("resize")?;
        let arr = data.bind(py);
        let c = arr.shape()[2];
        if c == 3 {
            let result = crate::resize::resize(
                py,
                data.clone_ref(py),
                (height, width),
                interpolation,
                true,
            )?;
            Ok(Self::wrap(py, result, Some(self.mode.clone())))
        } else {
            let (src, src_h, src_w, _) = pyarray_data(arr);
            let out = resize_nearest(src, src_h, src_w, height, width, c);
            Ok(self.wrap_vec(py, out, height, width, c))
        }
    }

    /// Flip image horizontally. 8-bit only.
    pub fn flip_horizontal(&self, py: Python<'_>) -> PyResult<Self> {
        let data = self.require_u8("flip_horizontal")?;
        let arr = data.bind(py);
        let c = arr.shape()[2];
        if c == 3 {
            let result = crate::flip::horizontal_flip(py, data.clone_ref(py))?;
            Ok(Self::wrap(py, result, Some(self.mode.clone())))
        } else {
            let (src, h, w, _) = pyarray_data(arr);
            let out = flip_h_generic(src, h, w, c);
            Ok(self.wrap_vec(py, out, h, w, c))
        }
    }

    /// Flip image vertically. 8-bit only.
    pub fn flip_vertical(&self, py: Python<'_>) -> PyResult<Self> {
        let data = self.require_u8("flip_vertical")?;
        let arr = data.bind(py);
        let c = arr.shape()[2];
        if c == 3 {
            let result = crate::flip::vertical_flip(py, data.clone_ref(py))?;
            Ok(Self::wrap(py, result, Some(self.mode.clone())))
        } else {
            let (src, h, w, _) = pyarray_data(arr);
            let out = flip_v_generic(src, h, w, c);
            Ok(self.wrap_vec(py, out, h, w, c))
        }
    }

    /// Crop image at (x, y) with given width and height. 8-bit only.
    pub fn crop(
        &self,
        py: Python<'_>,
        x: usize,
        y: usize,
        width: usize,
        height: usize,
    ) -> PyResult<Self> {
        let data = self.require_u8("crop")?;
        let arr = data.bind(py);
        let c = arr.shape()[2];
        if c == 3 {
            let result = crate::crop::crop(py, data.clone_ref(py), x, y, width, height)?;
            Ok(Self::wrap(py, result, Some(self.mode.clone())))
        } else {
            let (src, _, src_w, _) = pyarray_data(arr);
            let out = crop_generic(src, src_w, x, y, width, height, c);
            Ok(self.wrap_vec(py, out, height, width, c))
        }
    }

    /// Apply Gaussian blur. 8-bit only.
    #[pyo3(signature = (kernel_size=3, sigma=1.0))]
    fn gaussian_blur(&self, py: Python<'_>, kernel_size: usize, sigma: f32) -> PyResult<Self> {
        let data = self.require_u8("gaussian_blur")?;
        let c = data.bind(py).shape()[2];
        if c == 3 {
            let result = crate::blur::gaussian_blur(
                py,
                data.clone_ref(py),
                (kernel_size, kernel_size),
                (sigma, sigma),
            )?;
            Ok(Self::wrap(py, result, Some(self.mode.clone())))
        } else {
            self.copy(py)
        }
    }

    /// Apply box blur. 8-bit only.
    #[pyo3(signature = (kernel_size=3))]
    fn box_blur(&self, py: Python<'_>, kernel_size: usize) -> PyResult<Self> {
        let data = self.require_u8("box_blur")?;
        let c = data.bind(py).shape()[2];
        if c == 3 {
            let result =
                crate::blur::box_blur(py, data.clone_ref(py), (kernel_size, kernel_size))?;
            Ok(Self::wrap(py, result, Some(self.mode.clone())))
        } else {
            self.copy(py)
        }
    }

    /// Adjust brightness. Factor is additive in [0,1] range. 8-bit only.
    pub fn adjust_brightness(&self, py: Python<'_>, factor: f32) -> PyResult<Self> {
        let data = self.require_u8("adjust_brightness")?;
        let arr = data.bind(py);
        let (src, h, w, c) = pyarray_data(arr);
        Ok(Self::wrap(
            py,
            adjust_brightness_into_pyarray(py, src, factor * 255.0, h, w, c),
            Some(self.mode.clone()),
        ))
    }

    /// Adjust contrast. factor=1.0 is identity, >1 increases contrast. 8-bit only.
    pub fn adjust_contrast(&self, py: Python<'_>, factor: f64) -> PyResult<Self> {
        let data = self.require_u8("adjust_contrast")?;
        let arr = data.bind(py);
        let (src, h, w, c) = pyarray_data(arr);
        Ok(Self::wrap(
            py,
            adjust_contrast_into_pyarray(py, src, factor, h, w, c),
            Some(self.mode.clone()),
        ))
    }

    /// Adjust saturation. factor=1.0 is identity, 0.0 is grayscale. 8-bit only.
    pub fn adjust_saturation(&self, py: Python<'_>, factor: f64) -> PyResult<Self> {
        let data = self.require_u8("adjust_saturation")?;
        let arr = data.bind(py);
        if arr.shape()[2] != 3 {
            return self.copy(py);
        }
        let (src, h, w, _) = pyarray_data(arr);
        Ok(Self::wrap(
            py,
            adjust_saturation_into_pyarray(py, src, h * w, factor as f32, h, w),
            Some(self.mode.clone()),
        ))
    }

    /// Adjust hue. factor is in [-0.5, 0.5], fraction of hue wheel. 8-bit only.
    pub fn adjust_hue(&self, py: Python<'_>, factor: f64) -> PyResult<Self> {
        let data = self.require_u8("adjust_hue")?;
        let arr = data.bind(py);
        if arr.shape()[2] != 3 || factor == 0.0 {
            return self.copy(py);
        }
        let (src, h, w, _) = pyarray_data(arr);
        Ok(Self::wrap(
            py,
            adjust_hue_into_pyarray(py, src, h * w, factor as f32, h, w),
            Some(self.mode.clone()),
        ))
    }

    /// Normalize image to float32 using mean and std per channel. 8-bit only.
    fn normalize(
        &self,
        py: Python<'_>,
        mean: (f32, f32, f32),
        std: (f32, f32, f32),
    ) -> PyResult<Py<PyArray3<f32>>> {
        let data = self.require_u8("normalize")?;
        let arr = data.bind(py);
        let (src, h, w, c) = pyarray_data(arr);
        let npixels = h * w;
        let out = unsafe { PyArray::<f32, _>::new(py, [h, w, c], false) };
        let dst = unsafe { std::slice::from_raw_parts_mut(out.data(), npixels * c) };

        // Precompute per-channel: out[i] = src[i] * scale[ch] + offset[ch]
        // where scale = 1/(255*std), offset = -mean/std.
        const INV_255: f32 = 1.0 / 255.0;
        let mean_arr = [mean.0, mean.1, mean.2];
        let std_arr = [std.0, std.1, std.2];
        let mut scale = [0.0f32; 3];
        let mut offset = [0.0f32; 3];
        for ch in 0..c.min(3) {
            let inv_std = 1.0 / std_arr[ch];
            scale[ch] = INV_255 * inv_std;
            offset[ch] = -mean_arr[ch] * inv_std;
        }
        if c == 3 {
            kornia_imgproc::normalize::normalize_rgb_u8(src, dst, npixels, &scale, &offset);
        } else {
            for (idx, (&s, d)) in src.iter().zip(dst.iter_mut()).enumerate() {
                let ch = idx % c;
                *d = if ch < 3 {
                    s as f32 * scale[ch] + offset[ch]
                } else {
                    s as f32 * INV_255
                };
            }
        }
        Ok(out.unbind())
    }

    /// Convert RGB image to grayscale (1 channel). 8-bit only.
    fn to_grayscale(&self, py: Python<'_>) -> PyResult<Self> {
        let data = self.require_u8("to_grayscale")?;
        let arr = data.bind(py);
        let s = arr.shape();
        let (h, w, c) = (s[0], s[1], s[2]);
        if c == 1 {
            return self.copy(py);
        }
        if c != 3 {
            return Err(value_err(format!(
                "Cannot convert {}-channel image to grayscale",
                c
            )));
        }
        let src = unsafe { std::slice::from_raw_parts(arr.data(), h * w * c) };
        let out = unsafe { PyArray::<u8, _>::new(py, [h, w, 1], false) };
        let dst = unsafe { std::slice::from_raw_parts_mut(out.data(), h * w) };
        let npixels = h * w;

        kornia_imgproc::color::rgb_to_gray_u8(src, dst, npixels);

        Ok(Self::wrap(py, out.unbind(), Some("L".to_string())))
    }

    /// Convert grayscale to RGB (3 channels). 8-bit only.
    fn to_rgb(&self, py: Python<'_>) -> PyResult<Self> {
        let data = self.require_u8("to_rgb")?;
        let c = data.bind(py).shape()[2];
        if c == 3 {
            return self.copy(py);
        }
        if c == 1 {
            let result = crate::color::rgb_from_gray(py, data.clone_ref(py))?;
            Ok(Self::wrap(py, result, Some("RGB".to_string())))
        } else if c == 4 {
            let result = crate::color::rgb_from_rgba(py, data.clone_ref(py), None)?;
            Ok(Self::wrap(py, result, Some("RGB".to_string())))
        } else {
            Err(value_err(format!(
                "Cannot convert {}-channel image to RGB",
                c
            )))
        }
    }

    /// Rotate image by angle degrees (counter-clockwise). 8-bit only.
    ///
    /// Fast paths (exact k·90° only, no epsilon):
    ///  - 0°: copy
    ///  - 180°: buffer-reversal (always; any H, W, C)
    ///  - ±90°/±270° with H == W: transpose+flip (shape preserved)
    /// Non-exact or non-square 90°/270°: general bilinear warp.
    pub fn rotate(&self, py: Python<'_>, angle: f64) -> PyResult<Self> {
        let data = self.require_u8("rotate")?;
        let s = data.bind(py).shape();
        let (h, w, c) = (s[0], s[1], s[2]);

        if let Some(k) = exact_k90(angle) {
            match k {
                0 => return self.copy(py),
                2 => {
                    let arr = data.bind(py);
                    let (src, _, _, _) = pyarray_data(arr);
                    let (out, _, _) = rot90_generic(src, h, w, c, 2);
                    return Ok(self.wrap_vec(py, out, h, w, c));
                }
                1 | 3 if h == w => {
                    let arr = data.bind(py);
                    let (src, _, _, _) = pyarray_data(arr);
                    let (out, nh, nw) = rot90_generic(src, h, w, c, k as i32);
                    return Ok(self.wrap_vec(py, out, nh, nw, c));
                }
                _ => {} // fall through to warp for non-square 90/270
            }
        }

        let (cx, cy) = (w as f64 / 2.0, h as f64 / 2.0);
        let rad = angle.to_radians();
        let cos_a = rad.cos() as f32;
        let sin_a = rad.sin() as f32;
        let tx = (cx - cos_a as f64 * cx + sin_a as f64 * cy) as f32;
        let ty = (cy - sin_a as f64 * cx - cos_a as f64 * cy) as f32;
        let m = [cos_a, -sin_a, tx, sin_a, cos_a, ty];
        let result =
            crate::warp::warp_affine(py, data.clone_ref(py), m, (h, w), "bilinear", None)?;
        Ok(Self::wrap(py, result, Some(self.mode.clone())))
    }

    // --- Serialization for multiprocess (Ray Data, etc.) ---

    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let cls = Self::type_object(py).unbind().into_any();
        // Pickle as (numpy_array, mode) — `frombuffer` will auto-detect the
        // dtype on reconstruction, so we don't need to thread an extra tag.
        let arr_any: Py<PyAny> = match &self.data {
            ImageData::U8(a) => a.clone_ref(py).into_any(),
            ImageData::U16(a) => a.clone_ref(py).into_any(),
        };
        let args = pyo3::types::PyTuple::new(
            py,
            [
                arr_any.bind(py).clone(),
                pyo3::types::PyString::new(py, &self.mode).into_any(),
            ],
        )?
        .unbind();
        Ok((cls, args.into_any()))
    }

    fn __getstate__(&self, py: Python<'_>) -> (Py<PyAny>, String) {
        let arr_any: Py<PyAny> = match &self.data {
            ImageData::U8(a) => a.clone_ref(py).into_any(),
            ImageData::U16(a) => a.clone_ref(py).into_any(),
        };
        (arr_any, self.mode.clone())
    }

    fn __setstate__(&mut self, py: Python<'_>, state: (Py<PyAny>, String)) -> PyResult<()> {
        // Reconstruct the storage variant from the array's runtime dtype.
        let bound = state.0.bind(py);
        if let Ok(arr) = bound.extract::<Py<PyArray3<u8>>>() {
            self.data = ImageData::U8(arr);
        } else if let Ok(arr) = bound.extract::<Py<PyArray3<u16>>>() {
            self.data = ImageData::U16(arr);
        } else {
            return Err(value_err(
                "__setstate__: array dtype must be uint8 or uint16",
            ));
        }
        self.mode = state.1;
        Ok(())
    }

    // --- Dunder methods ---

    fn __repr__(&self, py: Python<'_>) -> String {
        let s = self.data.shape3(py);
        format!(
            "Image(mode={}, size={}x{}, dtype={})",
            self.mode,
            s[1],
            s[0],
            self.data.dtype_name()
        )
    }

    fn __eq__(&self, py: Python<'_>, other: &Self) -> bool {
        if self.mode != other.mode {
            return false;
        }
        // Different bit depths can't be equal — short-circuit before any
        // raw-byte comparison that would crash on stride mismatches.
        match (&self.data, &other.data) {
            (ImageData::U8(a), ImageData::U8(b)) => {
                let a = a.bind(py);
                let b = b.bind(py);
                if a.shape() != b.shape() {
                    return false;
                }
                let len: usize = a.shape().iter().product();
                let sa = unsafe { std::slice::from_raw_parts(a.data(), len) };
                let sb = unsafe { std::slice::from_raw_parts(b.data(), len) };
                sa == sb
            }
            (ImageData::U16(a), ImageData::U16(b)) => {
                let a = a.bind(py);
                let b = b.bind(py);
                if a.shape() != b.shape() {
                    return false;
                }
                let len: usize = a.shape().iter().product();
                let sa = unsafe { std::slice::from_raw_parts(a.data(), len) };
                let sb = unsafe { std::slice::from_raw_parts(b.data(), len) };
                sa == sb
            }
            _ => false,
        }
    }

    #[pyo3(signature = (dtype=None, copy=None))]
    fn __array__(
        &self,
        py: Python<'_>,
        dtype: Option<&str>,
        copy: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let arr_any: Py<PyAny> = match &self.data {
            ImageData::U8(a) => a.clone_ref(py).into_any(),
            ImageData::U16(a) => a.clone_ref(py).into_any(),
        };
        let bound = arr_any.bind(py);
        if let Some(dt) = dtype {
            Ok(bound
                .call_method1("astype", (dt,))?
                .call_method0("copy")?
                .unbind())
        } else if copy.unwrap_or(false) {
            Ok(bound.call_method0("copy")?.unbind())
        } else {
            Ok(arr_any)
        }
    }

    fn __len__(&self, py: Python<'_>) -> usize {
        self.data.shape3(py)[0]
    }

    /// PEP 3118 buffer protocol — delegates to the backing numpy array so
    /// `memoryview(img)`, `torch.asarray(img)`, and `torch.frombuffer(img)`
    /// get zero-copy access without going through `np.asarray`. Works for
    /// both u8 and u16 — numpy reports the correct itemsize via the buffer
    /// protocol's `format` field.
    unsafe fn __getbuffer__(
        slf: pyo3::PyRefMut<'_, Self>,
        view: *mut pyo3::ffi::Py_buffer,
        flags: std::os::raw::c_int,
    ) -> PyResult<()> {
        let py = slf.py();
        let arr_ptr = match &slf.data {
            ImageData::U8(a) => a.bind(py).as_ptr(),
            ImageData::U16(a) => a.bind(py).as_ptr(),
        };
        let ret = unsafe { pyo3::ffi::PyObject_GetBuffer(arr_ptr, view, flags) };
        if ret != 0 {
            Err(PyErr::fetch(py))
        } else {
            Ok(())
        }
    }

    unsafe fn __releasebuffer__(&self, view: *mut pyo3::ffi::Py_buffer) {
        unsafe { pyo3::ffi::PyBuffer_Release(view) };
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
