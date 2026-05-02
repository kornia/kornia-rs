use numpy::{PyArray, PyArray3, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::PyTypeInfo;

use kornia_image::{
    allocator::{CpuAllocator, ForeignAllocator},
    Image, ImageLayout, ImageSize, PixelFormat,
};
use pyo3::prelude::*;

pub type PyImage = Py<PyArray3<u8>>;
pub type PyImageU16 = Py<PyArray3<u16>>;
pub type PyImageF32 = Py<PyArray3<f32>>;

/// Represents the dimensions of an image.
///
/// # Fields
///
/// * `width` - The width of the image.
/// * `height` - The height of the image.
#[pyclass(name = "ImageSize", frozen, from_py_object, module = "kornia_rs.image")]
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
#[pyclass(name = "PixelFormat", from_py_object, module = "kornia_rs.image")]
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
#[pyclass(
    name = "ImageLayout",
    frozen,
    from_py_object,
    module = "kornia_rs.image"
)]
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

/// Zero-copy wrap a numpy u16 array as a Rust Image for reading.
///
/// The caller MUST ensure the Py<PyArray3<u16>> stays alive for the lifetime of the Image.
pub(crate) unsafe fn numpy_as_image_u16<const C: usize>(
    py: Python<'_>,
    image: &Py<PyArray3<u16>>,
) -> PyResult<Image<u16, C, ForeignAllocator>> {
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
    let len_bytes = h * w * C * std::mem::size_of::<u16>();
    Image::from_raw_parts(size, arr.data() as *const u16, len_bytes, ForeignAllocator)
        .map_err(to_pyerr)
}

/// Zero-copy wrap a numpy f32 array as a Rust Image for reading.
///
/// The caller MUST ensure the Py<PyArray3<f32>> stays alive for the lifetime of the Image.
pub(crate) unsafe fn numpy_as_image_f32<const C: usize>(
    py: Python<'_>,
    image: &Py<PyArray3<f32>>,
) -> PyResult<Image<f32, C, ForeignAllocator>> {
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
    let len_bytes = h * w * C * std::mem::size_of::<f32>();
    Image::from_raw_parts(size, arr.data() as *const f32, len_bytes, ForeignAllocator)
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

pub(crate) unsafe fn alloc_output_pyarray_u16<const C: usize>(
    py: Python<'_>,
    size: ImageSize,
) -> PyResult<AllocOutput<u16, C, PyArray3<u16>>> {
    let arr = PyArray::<u16, _>::new(py, [size.height, size.width, C], false);
    let len = size.height * size.width * C * std::mem::size_of::<u16>();
    let img = Image::from_raw_parts(size, arr.data() as *const u16, len, ForeignAllocator)
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

/// Canonical format tag for a path's extension. Returns the format kornia
/// understands ("PNG"/"JPEG"/"TIFF"/"WEBP") or None if the extension is
/// unknown — matching how PIL exposes ``Image.format`` after a load.
///
/// The single source of truth for extension → format mapping. Lowercase the
/// result for the encoder key used by `Image.encode` / `encode_to_bytes`.
pub(crate) fn format_from_path(path: &str) -> Option<&'static str> {
    let ext = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())?
        .to_lowercase();
    match ext.as_str() {
        "png" => Some("PNG"),
        "jpg" | "jpeg" => Some("JPEG"),
        "tif" | "tiff" => Some("TIFF"),
        "webp" => Some("WEBP"),
        _ => None,
    }
}

/// Scale a u16 buffer down to u8 by ``v >> 8`` — fast and equivalent to
/// the PIL/ImageMagick convention of ``v / 257`` for 16-bit color.
fn convert_u16_to_u8(
    py: Python<'_>,
    data: &ImageData,
    channels: usize,
    mode: String,
) -> PyResult<PyImageApi> {
    let src_arr = match data {
        ImageData::U16(a) => a.bind(py),
        _ => return Err(value_err("convert_u16_to_u8 called on non-u16 image")),
    };
    let s = src_arr.shape();
    let (h, w) = (s[0], s[1]);
    if s[2] != channels {
        return Err(value_err(format!(
            "convert: source has {} channels, target requires {}",
            s[2], channels
        )));
    }
    let src = unsafe { std::slice::from_raw_parts(src_arr.data(), h * w * channels) };
    let out = unsafe { PyArray::<u8, _>::new(py, [h, w, channels], false) };
    let dst = unsafe { std::slice::from_raw_parts_mut(out.data(), h * w * channels) };
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = (s >> 8) as u8;
    }
    Ok(PyImageApi::wrap(py, out.unbind(), Some(mode)))
}

/// Scale a u8 buffer up to u16 via ``v * 257`` so 0xFF maps to 0xFFFF —
/// matches PIL's "I;16" upcast convention.
fn convert_u8_to_u16(
    py: Python<'_>,
    data: &ImageData,
    channels: usize,
    mode: String,
) -> PyResult<PyImageApi> {
    let src_arr = match data {
        ImageData::U8(a) => a.bind(py),
        _ => return Err(value_err("convert_u8_to_u16 called on non-u8 image")),
    };
    let s = src_arr.shape();
    let (h, w) = (s[0], s[1]);
    if s[2] != channels {
        return Err(value_err(format!(
            "convert: source has {} channels, target requires {}",
            s[2], channels
        )));
    }
    let src = unsafe { std::slice::from_raw_parts(src_arr.data(), h * w * channels) };
    let out = unsafe { PyArray::<u16, _>::new(py, [h, w, channels], false) };
    let dst = unsafe { std::slice::from_raw_parts_mut(out.data(), h * w * channels) };
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = (*s as u16) * 257;
    }
    Ok(PyImageApi::wrap_u16(py, out.unbind(), Some(mode)))
}

/// Materialize the per-channel u8 fill values from a Python ``color`` arg.
/// Accepts ``None`` (zero), a scalar broadcast across channels, or a
/// per-channel tuple/list of the right length.
/// Fill a freshly-allocated PyArray slice with a per-channel color.
///
/// `color` accepts:
///   * `None`               → zero-fill
///   * a Python scalar      → broadcast to every channel
///   * a Python tuple/list  → per-channel value (length must equal `channels`)
///
/// The `dtype_hint` is used in the error message when the user passes an
/// unparseable color (e.g. a string). Generic over `T: Default + Copy + numpy::Element`
/// where `T` extracts from a Python int / float.
fn fill_color<T>(
    slice: &mut [T],
    color: Option<&Bound<'_, pyo3::PyAny>>,
    channels: usize,
    dtype_hint: &str,
) -> PyResult<()>
where
    T: Copy + Default + numpy::Element + for<'a, 'py> pyo3::FromPyObject<'a, 'py>,
{
    let fill: Vec<T> = match color {
        None => vec![T::default(); channels],
        Some(c) => {
            if let Ok(v) = c.extract::<T>() {
                vec![v; channels]
            } else if let Ok(t) = c.extract::<Vec<T>>() {
                if t.len() != channels {
                    return Err(value_err(format!(
                        "Image.new: color tuple has {} entries, mode requires {}",
                        t.len(),
                        channels
                    )));
                }
                t
            } else {
                return Err(value_err(format!(
                    "Image.new: color must be a {} scalar or tuple/list of {}",
                    dtype_hint, dtype_hint
                )));
            }
        }
    };
    for chunk in slice.chunks_exact_mut(channels) {
        chunk.copy_from_slice(&fill);
    }
    Ok(())
}

/// PIL-style mode string. u16 paths get the `;16` suffix (with `I` instead
/// of `L` for single-channel, mirroring PIL's `"I;16"`).
fn mode_from_channels(channels: usize, is_u16: bool) -> String {
    let (gray, suffix) = if is_u16 { ("I", ";16") } else { ("L", "") };
    match channels {
        1 => format!("{}{}", gray, suffix),
        3 => format!("RGB{}", suffix),
        4 => format!("RGBA{}", suffix),
        c => format!("{}ch{}", c, suffix),
    }
}

/// PIL-style mode string for f32 storage. Single-channel uses PIL's
/// canonical ``"F"`` (32-bit float); multi-channel uses an "f" suffix.
fn mode_from_channels_f32(channels: usize) -> String {
    match channels {
        1 => "F".to_string(),
        3 => "RGBf".to_string(),
        4 => "RGBAf".to_string(),
        c => format!("{}chf", c),
    }
}

/// Internal storage variant. The two variants are mutually exclusive — the
/// backing dtype is part of the Image's identity (a uint16 image cannot be
/// silently downcast on access without a copy).
#[derive(Debug)]
pub enum ImageData {
    U8(Py<PyArray3<u8>>),
    U16(Py<PyArray3<u16>>),
    F32(Py<PyArray3<f32>>),
}

impl ImageData {
    /// `(height, width, channels)` for any variant, without copying.
    fn shape3(&self, py: Python<'_>) -> [usize; 3] {
        let s = match self {
            ImageData::U8(a) => a.bind(py).shape().to_vec(),
            ImageData::U16(a) => a.bind(py).shape().to_vec(),
            ImageData::F32(a) => a.bind(py).shape().to_vec(),
        };
        [s[0], s[1], s[2]]
    }

    fn channels(&self, py: Python<'_>) -> usize {
        self.shape3(py)[2]
    }

    /// dtype name as exposed by numpy. Used for `Image.dtype.name` parity.
    fn dtype_name(&self) -> &'static str {
        match self {
            ImageData::U8(_) => "uint8",
            ImageData::U16(_) => "uint16",
            ImageData::F32(_) => "float32",
        }
    }

    /// Element size in bytes. Drives `nbytes` and the buffer protocol.
    fn itemsize(&self) -> usize {
        match self {
            ImageData::U8(_) => 1,
            ImageData::U16(_) => 2,
            ImageData::F32(_) => 4,
        }
    }

    /// True for 16-bit storage. Used by 8-bit-only methods to fail fast with
    /// a clear `NotImplementedError`.
    fn is_u16(&self) -> bool {
        matches!(self, ImageData::U16(_))
    }

    /// True for f32 storage.
    fn is_f32(&self) -> bool {
        matches!(self, ImageData::F32(_))
    }

    /// Bumps the underlying numpy array's refcount and returns it as `Py<PyAny>`.
    /// Drives `data` getter, `__array__`, `__getstate__`, `__reduce__`.
    fn as_pyany(&self, py: Python<'_>) -> Py<PyAny> {
        match self {
            ImageData::U8(a) => a.clone_ref(py).into_any(),
            ImageData::U16(a) => a.clone_ref(py).into_any(),
            ImageData::F32(a) => a.clone_ref(py).into_any(),
        }
    }

    /// numpy dtype descriptor for the variant — drives the `dtype` getter.
    fn dtype_obj(&self, py: Python<'_>) -> Py<PyAny> {
        match self {
            ImageData::U8(a) => a.bind(py).dtype().clone().unbind().into_any(),
            ImageData::U16(a) => a.bind(py).dtype().clone().unbind().into_any(),
            ImageData::F32(a) => a.bind(py).dtype().clone().unbind().into_any(),
        }
    }

    /// Raw `*mut PyObject` for the buffer protocol.
    fn as_ptr(&self, py: Python<'_>) -> *mut pyo3::ffi::PyObject {
        match self {
            ImageData::U8(a) => a.bind(py).as_ptr(),
            ImageData::U16(a) => a.bind(py).as_ptr(),
            ImageData::F32(a) => a.bind(py).as_ptr(),
        }
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
    /// Canonical format the Image was decoded from (e.g. ``"PNG"``, ``"JPEG"``,
    /// ``"TIFF"``). ``None`` for in-memory-constructed Images. Set by
    /// ``Image.load`` / ``Image.decode`` / ``Image.open``.
    format: Option<&'static str>,
}

/// Shorthand for constructing a Python `ValueError`.
fn value_err<M: Into<String>>(msg: M) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyValueError, _>(msg.into())
}

enum FlipDir {
    Horizontal,
    Vertical,
}

/// Mirror of `u16_imgproc_unsupported` for f32-only Images. Imgproc kernels
/// are u8-only today; users with float storage should ``.convert("L"/"RGB")``
/// or operate in numpy directly.
fn f32_imgproc_unsupported(method: &str) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(format!(
        "Image.{}() is not yet implemented for float32 images. \
         Use img.convert(\"L\"/\"RGB\"/\"RGBA\") to scale to uint8, or operate \
         on img.data with numpy. flip_horizontal/flip_vertical/crop work on \
         float32 already.",
        method
    ))
}

/// Shared error for 8-bit-only methods called on a 16-bit Image. We surface
/// a clear remediation path so users hit a known gap, not a mystery type
/// mismatch deep inside an imgproc kernel.
fn u16_imgproc_unsupported(method: &str) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(format!(
        "Image.{}() is not yet implemented for uint16 images. \
         Convert to uint8 first via `img.convert(\"L\")` (or \"RGB\" / \
         \"RGBA\" for 3/4 channels). flip_horizontal/flip_vertical/crop \
         already work on uint16.",
        method
    ))
}

impl PyImageApi {
    /// Wrap an 8-bit numpy array. Mode defaults to `"L"`/`"RGB"`/`"RGBA"`
    /// derived from channel count.
    pub fn wrap(py: Python<'_>, data: Py<PyArray3<u8>>, mode: Option<String>) -> Self {
        let channels = data.bind(py).shape()[2];
        let mode = mode.unwrap_or_else(|| mode_from_channels(channels, false));
        Self {
            data: ImageData::U8(data),
            mode,
            format: None,
        }
    }

    /// Wrap a 16-bit numpy array. Mode defaults to `"I;16"`/`"RGB;16"`/etc.
    pub fn wrap_u16(py: Python<'_>, data: Py<PyArray3<u16>>, mode: Option<String>) -> Self {
        let channels = data.bind(py).shape()[2];
        let mode = mode.unwrap_or_else(|| mode_from_channels(channels, true));
        Self {
            data: ImageData::U16(data),
            mode,
            format: None,
        }
    }

    /// Wrap a 32-bit float numpy array. Mode defaults to PIL's "F" for
    /// single-channel; multi-channel uses ``"RGBf"`` / ``"RGBAf"`` /
    /// ``"<n>chf"``.
    pub fn wrap_f32(py: Python<'_>, data: Py<PyArray3<f32>>, mode: Option<String>) -> Self {
        let channels = data.bind(py).shape()[2];
        let mode = mode.unwrap_or_else(|| mode_from_channels_f32(channels));
        Self {
            data: ImageData::F32(data),
            mode,
            format: None,
        }
    }

    fn with_format(mut self, format: &'static str) -> Self {
        self.format = Some(format);
        self
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

    /// Internal Rust-only entry for the kornia-style crop signature.
    /// Used both by the Python ``crop`` method (after parsing the PIL/kornia
    /// shape) and by augmentations that already have explicit (x,y,w,h).
    pub(crate) fn crop_xywh(
        &self,
        py: Python<'_>,
        x: usize,
        y: usize,
        width: usize,
        height: usize,
    ) -> PyResult<Self> {
        match &self.data {
            ImageData::U8(_) => {
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
            ImageData::U16(a) => {
                let arr = a.bind(py);
                let s = arr.shape();
                let (src_h, src_w, c) = (s[0], s[1], s[2]);
                if y + height > src_h || x + width > src_w {
                    return Err(value_err(format!(
                        "crop: box ({}, {}, {}x{}) out of bounds for ({}, {}, {})",
                        x, y, width, height, src_h, src_w, c
                    )));
                }
                let src = unsafe { std::slice::from_raw_parts(arr.data(), src_h * src_w * c) };
                let out_arr = unsafe { PyArray::<u16, _>::new(py, [height, width, c], false) };
                let dst =
                    unsafe { std::slice::from_raw_parts_mut(out_arr.data(), height * width * c) };
                for row in 0..height {
                    let s_off = ((y + row) * src_w + x) * c;
                    let d_off = row * width * c;
                    dst[d_off..d_off + width * c].copy_from_slice(&src[s_off..s_off + width * c]);
                }
                Ok(Self::wrap_u16(
                    py,
                    out_arr.unbind(),
                    Some(self.mode.clone()),
                ))
            }
            ImageData::F32(a) => {
                let arr = a.bind(py);
                let s = arr.shape();
                let (src_h, src_w, c) = (s[0], s[1], s[2]);
                if y + height > src_h || x + width > src_w {
                    return Err(value_err(format!(
                        "crop: box ({}, {}, {}x{}) out of bounds for ({}, {}, {})",
                        x, y, width, height, src_h, src_w, c
                    )));
                }
                let src = unsafe { std::slice::from_raw_parts(arr.data(), src_h * src_w * c) };
                let out_arr = unsafe { PyArray::<f32, _>::new(py, [height, width, c], false) };
                let dst =
                    unsafe { std::slice::from_raw_parts_mut(out_arr.data(), height * width * c) };
                for row in 0..height {
                    let s_off = ((y + row) * src_w + x) * c;
                    let d_off = row * width * c;
                    dst[d_off..d_off + width * c].copy_from_slice(&src[s_off..s_off + width * c]);
                }
                Ok(Self::wrap_f32(
                    py,
                    out_arr.unbind(),
                    Some(self.mode.clone()),
                ))
            }
        }
    }

    /// dtype-trivial flip kernel — flips a buffer by row (vertical) or by
    /// column-of-channel-tuples (horizontal). Caller is responsible for
    /// wrapping the resulting PyArray into the right ImageData variant.
    fn flip_pod_into<T: Copy + numpy::Element>(
        &self,
        py: Python<'_>,
        a: &Py<PyArray3<T>>,
        dir: FlipDir,
    ) -> PyResult<Py<PyArray3<T>>> {
        let arr = a.bind(py);
        let s = arr.shape();
        let (h, w, c) = (s[0], s[1], s[2]);
        let src = unsafe { std::slice::from_raw_parts(arr.data(), h * w * c) };
        let out = unsafe { PyArray::<T, _>::new(py, [h, w, c], false) };
        let dst = unsafe { std::slice::from_raw_parts_mut(out.data(), h * w * c) };
        match dir {
            FlipDir::Horizontal => {
                for row in 0..h {
                    let row_off = row * w * c;
                    let src_row = &src[row_off..row_off + w * c];
                    let dst_row = &mut dst[row_off..row_off + w * c];
                    for (d, s) in dst_row
                        .chunks_exact_mut(c)
                        .zip(src_row.chunks_exact(c).rev())
                    {
                        d.copy_from_slice(s);
                    }
                }
            }
            FlipDir::Vertical => {
                for row in 0..h {
                    let s_off = row * w * c;
                    let d_off = (h - 1 - row) * w * c;
                    dst[d_off..d_off + w * c].copy_from_slice(&src[s_off..s_off + w * c]);
                }
            }
        }
        Ok(out.unbind())
    }

    fn flip_u16(&self, py: Python<'_>, dir: FlipDir) -> PyResult<Self> {
        let a = match &self.data {
            ImageData::U16(a) => a,
            _ => return self.copy(py),
        };
        let out = self.flip_pod_into(py, a, dir)?;
        Ok(Self::wrap_u16(py, out, Some(self.mode.clone())))
    }

    fn flip_f32(&self, py: Python<'_>, dir: FlipDir) -> PyResult<Self> {
        let a = match &self.data {
            ImageData::F32(a) => a,
            _ => return self.copy(py),
        };
        let out = self.flip_pod_into(py, a, dir)?;
        Ok(Self::wrap_f32(py, out, Some(self.mode.clone())))
    }

    /// Returns the u8 backing array, or a `NotImplementedError` if the Image
    /// is u16. Used by 8-bit-only imgproc methods after their early gate.
    pub(crate) fn require_u8<'a>(&'a self, method: &str) -> PyResult<&'a Py<PyArray3<u8>>> {
        match &self.data {
            ImageData::U8(a) => Ok(a),
            ImageData::U16(_) => Err(u16_imgproc_unsupported(method)),
            ImageData::F32(_) => Err(f32_imgproc_unsupported(method)),
        }
    }

    /// Single canonical encode path used by both `encode()` and `save()`.
    /// Dispatches on (format, dtype, channel count) and returns PNG/JPEG
    /// bytes. JPEG is u8-only by spec; PNG handles u8 (1/3/4 ch) and u16
    /// (1/3/4 ch).
    fn encode_to_bytes(
        &self,
        py: Python<'_>,
        format: &str,
        quality: u8,
        compress_level: Option<u8>,
        subsampling: Option<&str>,
    ) -> PyResult<Vec<u8>> {
        let c = self.data.channels(py);
        let is_u16 = self.data.is_u16();
        let is_f32 = self.data.is_f32();

        match format {
            "jpg" | "jpeg" => {
                if is_u16 || is_f32 {
                    return Err(value_err(format!(
                        "JPEG cannot encode {} images. Use \"png\" or \"tiff\" instead.",
                        self.data.dtype_name()
                    )));
                }
                if c != 3 {
                    return Err(value_err(format!(
                        "JPEG requires 3-channel RGB image, got {} channels",
                        c
                    )));
                }
                let arr = match &self.data {
                    ImageData::U8(a) => a.clone_ref(py),
                    _ => unreachable!("u16/f32 rejected above"),
                };
                // libjpeg-turbo first (~3-4× faster than zune-jpeg on aarch64);
                // fall back to zune-jpeg only if the libjpeg-turbo init fails
                // — belt-and-suspenders for builds without the turbojpeg feature.
                match crate::io::jpegturbo::encode_image_jpegturbo(
                    py,
                    arr.clone_ref(py),
                    quality as i32,
                    subsampling,
                ) {
                    Ok(b) => Ok(b),
                    Err(_) => crate::io::jpeg::encode_image_jpeg(py, arr, quality),
                }
            }
            "png" => {
                let mut buffer = Vec::new();
                match (&self.data, c) {
                    (ImageData::U8(a), 3) => {
                        let img = unsafe { numpy_as_image::<3>(py, a)? };
                        kornia_io::png::encode_image_png_rgb8(&img, &mut buffer, compress_level)
                            .map_err(to_pyerr)?;
                    }
                    (ImageData::U8(a), 4) => {
                        let img = unsafe { numpy_as_image::<4>(py, a)? };
                        kornia_io::png::encode_image_png_rgba8(&img, &mut buffer, compress_level)
                            .map_err(to_pyerr)?;
                    }
                    (ImageData::U8(a), 1) => {
                        let img = unsafe { numpy_as_image::<1>(py, a)? };
                        kornia_io::png::encode_image_png_gray8(&img, &mut buffer, compress_level)
                            .map_err(to_pyerr)?;
                    }
                    (ImageData::U16(a), 3) => {
                        let img = unsafe { numpy_as_image_u16::<3>(py, a)? };
                        kornia_io::png::encode_image_png_rgb16(&img, &mut buffer, compress_level)
                            .map_err(to_pyerr)?;
                    }
                    (ImageData::U16(a), 4) => {
                        let img = unsafe { numpy_as_image_u16::<4>(py, a)? };
                        kornia_io::png::encode_image_png_rgba16(&img, &mut buffer, compress_level)
                            .map_err(to_pyerr)?;
                    }
                    (ImageData::U16(a), 1) => {
                        let img = unsafe { numpy_as_image_u16::<1>(py, a)? };
                        kornia_io::png::encode_image_png_gray16(&img, &mut buffer, compress_level)
                            .map_err(to_pyerr)?;
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
            "webp" => {
                let mut buffer = Vec::new();
                match (&self.data, c) {
                    (ImageData::U8(a), 3) => {
                        let img = unsafe { numpy_as_image::<3>(py, a)? };
                        kornia_io::webp::encode_image_webp_rgb8(&img, &mut buffer)
                            .map_err(to_pyerr)?;
                    }
                    (ImageData::U8(a), 4) => {
                        let img = unsafe { numpy_as_image::<4>(py, a)? };
                        kornia_io::webp::encode_image_webp_rgba8(&img, &mut buffer)
                            .map_err(to_pyerr)?;
                    }
                    (ImageData::U8(a), 1) => {
                        let img = unsafe { numpy_as_image::<1>(py, a)? };
                        kornia_io::webp::encode_image_webp_gray8(&img, &mut buffer)
                            .map_err(to_pyerr)?;
                    }
                    (ImageData::U16(_), _) => {
                        return Err(value_err(
                            "WebP encode requires uint8; convert via img.convert('RGB') first",
                        ))
                    }
                    _ => {
                        return Err(value_err(format!(
                            "WebP requires 1/3/4-channel u8 image, got {} channels (dtype={})",
                            c,
                            self.data.dtype_name()
                        )))
                    }
                };
                Ok(buffer)
            }
            "tiff" | "tif" => {
                let mut buffer = Vec::new();
                match (&self.data, c) {
                    (ImageData::U8(a), 3) => {
                        let img = unsafe { numpy_as_image::<3>(py, a)? };
                        kornia_io::tiff::encode_image_tiff_rgb8(&img, &mut buffer)
                            .map_err(to_pyerr)?;
                    }
                    (ImageData::U8(a), 1) => {
                        let img = unsafe { numpy_as_image::<1>(py, a)? };
                        kornia_io::tiff::encode_image_tiff_mono8(&img, &mut buffer)
                            .map_err(to_pyerr)?;
                    }
                    (ImageData::U16(a), 3) => {
                        let img = unsafe { numpy_as_image_u16::<3>(py, a)? };
                        kornia_io::tiff::encode_image_tiff_rgb16(&img, &mut buffer)
                            .map_err(to_pyerr)?;
                    }
                    (ImageData::U16(a), 1) => {
                        let img = unsafe { numpy_as_image_u16::<1>(py, a)? };
                        kornia_io::tiff::encode_image_tiff_mono16(&img, &mut buffer)
                            .map_err(to_pyerr)?;
                    }
                    (ImageData::F32(a), 3) => {
                        let img = unsafe { numpy_as_image_f32::<3>(py, a)? };
                        kornia_io::tiff::encode_image_tiff_rgb32f(&img, &mut buffer)
                            .map_err(to_pyerr)?;
                    }
                    (ImageData::F32(a), 1) => {
                        let img = unsafe { numpy_as_image_f32::<1>(py, a)? };
                        kornia_io::tiff::encode_image_tiff_mono32f(&img, &mut buffer)
                            .map_err(to_pyerr)?;
                    }
                    _ => {
                        return Err(value_err(format!(
                            "TIFF requires 1 or 3-channel image, got {} channels (dtype={})",
                            c,
                            self.data.dtype_name()
                        )))
                    }
                };
                Ok(buffer)
            }
            other => Err(value_err(format!(
                "Unsupported format {:?}. Supported: \"jpeg\"/\"jpg\", \"png\", \"webp\", \"tiff\"/\"tif\"",
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
        if let Ok(arr) = data.extract::<Py<PyArray3<f32>>>() {
            return Ok(Self::wrap_f32(py, arr, mode));
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
                    if let Ok(arr) = reshaped.extract::<Py<PyArray3<f32>>>() {
                        return Ok(Self::wrap_f32(py, arr, mode));
                    }
                    return Err(value_err(
                        "frombuffer: 2D array dtype must be uint8, uint16, or float32",
                    ));
                } else if shape.len() != 3 {
                    return Err(value_err(format!(
                        "Expected 2D or 3D array, got {}D",
                        shape.len()
                    )));
                }
                // 3D but dtype mismatched — give a precise error.
                return Err(value_err(
                    "frombuffer: 3D array dtype must be uint8, uint16, or float32",
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
        let format = format_from_path(path);
        let path_str = pyo3::types::PyString::new(py, path);
        let arr_any = crate::io::functional::read_image(py, path_str.into_any())?;
        // The functional dispatcher returns either Py<PyArray3<u8>> or
        // Py<PyArray3<u16>> depending on the file's pixel format. Try the u8
        // path first (the dominant case) and fall through to u16.
        let img = if let Ok(arr) = arr_any.extract::<Py<PyArray3<u8>>>(py) {
            Self::wrap(py, arr, None)
        } else {
            let arr: Py<PyArray3<u16>> = arr_any.extract(py).map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "load: file decoded to unsupported dtype (expected uint8 or uint16)",
                )
            })?;
            Self::wrap_u16(py, arr, None)
        };
        Ok(match format {
            Some(f) => img.with_format(f),
            None => img,
        })
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
            other => {
                return Err(value_err(format!(
                    "decode: unsupported mode {:?}; expected \"RGB\", \"RGBA\", or \"L\"",
                    other
                )))
            }
        };

        // JPEG: always 8-bit per channel.
        if data.len() >= 2 && data[0] == 0xff && data[1] == 0xd8 {
            let arr = match crate::io::jpegturbo::decode_image_jpegturbo(py, data, native_mode) {
                Ok(a) => a,
                Err(_) => crate::io::jpeg::decode_image_jpeg(py, data)?,
            };
            return Ok(Self::wrap(py, arr, Some(mode.to_string())).with_format("JPEG"));
        }

        if data.len() >= 4 && &data[0..4] == b"\x89PNG" {
            let layout = kornia_io::png::decode_image_png_layout(data)
                .map_err(|e| value_err(format!("decode: invalid PNG: {}", e)))?;
            let (height, width) = (layout.image_size.height, layout.image_size.width);
            return match layout.pixel_format {
                PixelFormat::U16 => {
                    let arr = crate::io::png::decode_image_png_u16(
                        py,
                        data,
                        (height, width),
                        native_mode,
                    )?;
                    let channels = match mode {
                        "RGB" => 3,
                        "RGBA" => 4,
                        "L" => 1,
                        _ => unreachable!("native_mode validated above"),
                    };
                    Ok(
                        Self::wrap_u16(py, arr, Some(mode_from_channels(channels, true)))
                            .with_format("PNG"),
                    )
                }
                _ => {
                    let arr = crate::io::png::decode_image_png_u8(
                        py,
                        data,
                        (height, width),
                        native_mode,
                    )?;
                    Ok(Self::wrap(py, arr, Some(mode.to_string())).with_format("PNG"))
                }
            };
        }

        if data.len() >= 12 && &data[0..4] == b"RIFF" && &data[8..12] == b"WEBP" {
            let layout = kornia_io::webp::decode_image_webp_layout(data)
                .map_err(|e| value_err(format!("decode: invalid WebP: {}", e)))?;
            let size = layout.image_size;
            return match mode {
                "RGB" => {
                    let (dst, out) = unsafe { alloc_output_pyarray::<3>(py, size)? };
                    let mut wrapped = kornia_image::color_spaces::Rgb8(dst);
                    kornia_io::webp::decode_image_webp_rgb8(data, &mut wrapped)
                        .map_err(to_pyerr)?;
                    Ok(Self::wrap(py, out, Some(mode.to_string())).with_format("WEBP"))
                }
                "RGBA" => {
                    let (dst, out) = unsafe { alloc_output_pyarray::<4>(py, size)? };
                    let mut wrapped = kornia_image::color_spaces::Rgba8(dst);
                    kornia_io::webp::decode_image_webp_rgba8(data, &mut wrapped)
                        .map_err(to_pyerr)?;
                    Ok(Self::wrap(py, out, Some(mode.to_string())).with_format("WEBP"))
                }
                "L" => {
                    let (dst, out) = unsafe { alloc_output_pyarray::<1>(py, size)? };
                    let mut wrapped = kornia_image::color_spaces::Gray8(dst);
                    kornia_io::webp::decode_image_webp_gray8(data, &mut wrapped)
                        .map_err(to_pyerr)?;
                    Ok(Self::wrap(py, out, Some(mode.to_string())).with_format("WEBP"))
                }
                _ => unreachable!("native_mode validated above"),
            };
        }

        let is_tiff = data.len() >= 4 && (data.starts_with(b"II*\0") || data.starts_with(b"MM\0*"));
        if is_tiff {
            let layout = kornia_io::tiff::decode_image_tiff_layout(data)
                .map_err(|e| value_err(format!("decode: invalid TIFF: {}", e)))?;
            let size = layout.image_size;
            // Choose the right typed decoder based on the file's actual
            // pixel format + the user-requested mode (which fixes channels).
            let want_channels = match mode {
                "RGB" => 3,
                "RGBA" => 4,
                "L" => 1,
                _ => unreachable!("native_mode validated above"),
            };
            return match layout.pixel_format {
                PixelFormat::U8 => {
                    if want_channels == 3 {
                        let (dst, out) = unsafe { alloc_output_pyarray::<3>(py, size)? };
                        let mut wrapped = kornia_image::color_spaces::Rgb8(dst);
                        kornia_io::tiff::decode_image_tiff_rgb8(data, &mut wrapped)
                            .map_err(to_pyerr)?;
                        Ok(Self::wrap(py, out, Some(mode.to_string())).with_format("TIFF"))
                    } else if want_channels == 1 {
                        let (dst, out) = unsafe { alloc_output_pyarray::<1>(py, size)? };
                        let mut wrapped = kornia_image::color_spaces::Gray8(dst);
                        kornia_io::tiff::decode_image_tiff_mono8(data, &mut wrapped)
                            .map_err(to_pyerr)?;
                        Ok(Self::wrap(py, out, Some(mode.to_string())).with_format("TIFF"))
                    } else {
                        Err(value_err(format!(
                            "decode: TIFF u8 with mode={:?} not supported (channels={})",
                            mode, want_channels
                        )))
                    }
                }
                PixelFormat::U16 => {
                    let mode_u16 = mode_from_channels(want_channels, true);
                    if want_channels == 3 {
                        let (dst, out) = unsafe { alloc_output_pyarray_u16::<3>(py, size)? };
                        let mut wrapped = kornia_image::color_spaces::Rgb16(dst);
                        kornia_io::tiff::decode_image_tiff_rgb16(data, &mut wrapped)
                            .map_err(to_pyerr)?;
                        Ok(Self::wrap_u16(py, out, Some(mode_u16)).with_format("TIFF"))
                    } else if want_channels == 1 {
                        let (dst, out) = unsafe { alloc_output_pyarray_u16::<1>(py, size)? };
                        let mut wrapped = kornia_image::color_spaces::Gray16(dst);
                        kornia_io::tiff::decode_image_tiff_mono16(data, &mut wrapped)
                            .map_err(to_pyerr)?;
                        Ok(Self::wrap_u16(py, out, Some(mode_u16)).with_format("TIFF"))
                    } else {
                        Err(value_err(format!(
                            "decode: TIFF u16 with mode={:?} not supported (channels={})",
                            mode, want_channels
                        )))
                    }
                }
                PixelFormat::F32 => Err(value_err(
                    "decode: float32 TIFF must use kornia_rs.io.read_image_tiff_f32",
                )),
            };
        }

        Err(value_err(
            "Unsupported image format: not JPEG, PNG, WebP, or TIFF",
        ))
    }

    /// Unified entry point — accepts a path, a bytes/bytearray buffer, or any
    /// file-like object with ``.read()``. PIL parity: ``PIL.Image.open(fp)``.
    #[staticmethod]
    #[pyo3(signature = (fp, mode="RGB"))]
    fn open(py: Python<'_>, fp: &Bound<'_, PyAny>, mode: &str) -> PyResult<Self> {
        if let Ok(path) = fp.extract::<String>() {
            return Self::load(py, &path);
        }
        let bytes_obj: Vec<u8> = if let Ok(b) = fp.extract::<Vec<u8>>() {
            b
        } else if fp.hasattr("read")? {
            let read_result = fp.call_method0("read")?;
            read_result.extract().map_err(|_| {
                value_err("Image.open: file-like .read() must return bytes/bytearray")
            })?
        } else {
            return Err(value_err(
                "Image.open: fp must be a path string, bytes/bytearray, or a readable file-like",
            ));
        };
        Self::decode(py, &bytes_obj, mode)
    }

    /// Create a new blank Image with the given mode and size, optionally
    /// filled with ``color``. PIL parity: ``PIL.Image.new(mode, size, color=0)``.
    ///
    /// - ``size`` is ``(width, height)`` per PIL convention.
    /// - ``color`` is a scalar (broadcast to all channels) or a per-channel
    ///   tuple matching the mode's channel count.
    #[staticmethod]
    #[pyo3(name = "new", signature = (mode, size, color=None))]
    fn new_blank(
        py: Python<'_>,
        mode: &str,
        size: (usize, usize),
        color: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let (width, height) = size;
        enum Dtype {
            U8,
            U16,
            F32,
        }
        let (channels, dtype) = match mode {
            "L" => (1, Dtype::U8),
            "RGB" => (3, Dtype::U8),
            "RGBA" => (4, Dtype::U8),
            "I;16" => (1, Dtype::U16),
            "RGB;16" => (3, Dtype::U16),
            "RGBA;16" => (4, Dtype::U16),
            "F" => (1, Dtype::F32),
            "RGBf" => (3, Dtype::F32),
            "RGBAf" => (4, Dtype::F32),
            other => {
                return Err(value_err(format!(
                    "Image.new: unsupported mode {:?}; expected L/RGB/RGBA, I;16/RGB;16/RGBA;16, or F/RGBf/RGBAf",
                    other
                )))
            }
        };

        match dtype {
            Dtype::U8 => {
                let arr = unsafe { PyArray::<u8, _>::new(py, [height, width, channels], false) };
                let len = height * width * channels;
                let slice = unsafe { std::slice::from_raw_parts_mut(arr.data(), len) };
                fill_color::<u8>(slice, color, channels, "uint8 (0-255)")?;
                Ok(Self::wrap(py, arr.unbind(), Some(mode.to_string())))
            }
            Dtype::U16 => {
                let arr = unsafe { PyArray::<u16, _>::new(py, [height, width, channels], false) };
                let len = height * width * channels;
                let slice = unsafe { std::slice::from_raw_parts_mut(arr.data(), len) };
                fill_color::<u16>(slice, color, channels, "uint16 (0-65535)")?;
                Ok(Self::wrap_u16(py, arr.unbind(), Some(mode.to_string())))
            }
            Dtype::F32 => {
                let arr = unsafe { PyArray::<f32, _>::new(py, [height, width, channels], false) };
                let len = height * width * channels;
                let slice = unsafe { std::slice::from_raw_parts_mut(arr.data(), len) };
                fill_color::<f32>(slice, color, channels, "float32")?;
                Ok(Self::wrap_f32(py, arr.unbind(), Some(mode.to_string())))
            }
        }
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

    /// Canonical format the Image was decoded from (e.g. ``"PNG"``, ``"JPEG"``,
    /// ``"TIFF"``, ``"WEBP"``). ``None`` for in-memory-constructed Images.
    #[getter]
    fn format(&self) -> Option<&str> {
        self.format.as_deref()
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
        self.data.dtype_obj(py)
    }

    /// The underlying numpy array. Returns ``Py<PyAny>`` because the dtype
    /// depends on the bit depth (uint8 or uint16); callers can pass to
    /// numpy directly or test ``img.dtype`` to branch.
    #[getter]
    pub fn data(&self, py: Python<'_>) -> Py<PyAny> {
        self.data.as_pyany(py)
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
    #[pyo3(signature = (fp, format=None, quality=95, compress_level=None, subsampling=None))]
    fn save(
        &self,
        py: Python<'_>,
        fp: &Bound<'_, PyAny>,
        format: Option<&str>,
        quality: u8,
        compress_level: Option<u8>,
        subsampling: Option<&str>,
    ) -> PyResult<()> {
        // Resolve target type: path-like vs file-like.
        let path_str: Option<String> = fp.extract().ok();

        // Resolve format.
        let resolved_format = match format {
            Some(f) => f.to_lowercase(),
            None => {
                let path = path_str.as_deref().ok_or_else(|| {
                    value_err("save: format= is required when target is a file-like object")
                })?;
                format_from_path(path)
                    .ok_or_else(|| {
                        value_err(format!(
                            "save: could not determine format from path {:?}; pass format=",
                            path
                        ))
                    })?
                    .to_lowercase()
            }
        };

        // Encode to bytes (handles u8/u16 dispatch internally).
        let bytes =
            self.encode_to_bytes(py, &resolved_format, quality, compress_level, subsampling)?;

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

    /// Encode the image to in-memory bytes — the in-memory complement of ``save``.
    ///
    /// Args:
    ///     format: ``"jpeg"``/``"jpg"``, ``"png"``, ``"webp"``, ``"tiff"``/``"tif"``.
    ///         Case-insensitive.
    ///     quality: JPEG / WebP quality 1-100. Ignored for PNG / TIFF. Default 95.
    ///     compress_level: PNG zlib level 0..=9 (ignored for non-PNG).
    ///         ``0`` skips deflate (largest, fastest); ``1`` uses fdeflate
    ///         (NEON / AVX2 fast path, smaller than ``0`` and far faster than
    ///         the default); ``2..=9`` are standard zlib levels. ``None`` keeps
    ///         the `png` crate default ("balanced").
    ///     subsampling: JPEG chroma subsampling — ``"4:2:0"`` (default; matches
    ///         cv2 / PIL), ``"4:2:2"``, or ``"4:4:4"`` (no subsampling, largest
    ///         files, needed for synthetic / text content). Ignored for non-JPEG.
    ///
    /// Returns:
    ///     bytes: The encoded image data.
    ///
    /// Example::
    ///
    ///     img = Image.frombuffer(rgb_array)
    ///     jpeg_bytes = img.encode("jpeg", quality=80)
    ///     png_bytes  = img.encode("png", compress_level=1)
    #[pyo3(signature = (format, quality=95, compress_level=None, subsampling=None))]
    fn encode(
        &self,
        py: Python<'_>,
        format: &str,
        quality: u8,
        compress_level: Option<u8>,
        subsampling: Option<&str>,
    ) -> PyResult<Vec<u8>> {
        self.encode_to_bytes(
            py,
            &format.to_lowercase(),
            quality,
            compress_level,
            subsampling,
        )
    }

    /// Return the raw pixel buffer as Python ``bytes`` (no encoding, no header).
    ///
    /// PIL parity: equivalent to ``PIL.Image.tobytes()``. Layout is HWC,
    /// row-major; element width matches ``dtype`` (``uint8`` -> 1 byte/elem,
    /// ``uint16`` -> 2 bytes little-endian native).
    fn tobytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let arr = self.data.as_pyany(py);
        let bound = arr.bind(py);
        // numpy's .tobytes() returns a fresh bytes object; cheap memcpy and
        // matches PIL semantics (caller-owned, GIL-safe).
        let bytes_obj = bound.call_method0("tobytes")?;
        bytes_obj
            .cast_into::<pyo3::types::PyBytes>()
            .map_err(|e| value_err(format!("tobytes: numpy did not return bytes: {}", e)))
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
            ImageData::F32(a) => {
                let copy = a.bind(py).call_method0("copy")?;
                Ok(copy.extract::<Py<PyArray3<f32>>>()?.into_any())
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
            ImageData::F32(a) => {
                let copy: Py<PyArray3<f32>> = a.bind(py).call_method0("copy")?.extract()?;
                ImageData::F32(copy)
            }
        };
        Ok(Self {
            data: new_data,
            mode: self.mode.clone(),
            format: self.format.clone(),
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

    /// Flip image horizontally. Supports 8-bit, 16-bit, and float32 Images.
    pub fn flip_horizontal(&self, py: Python<'_>) -> PyResult<Self> {
        match &self.data {
            ImageData::U8(_) => {
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
            ImageData::U16(_) => self.flip_u16(py, FlipDir::Horizontal),
            ImageData::F32(_) => self.flip_f32(py, FlipDir::Horizontal),
        }
    }

    /// Flip image vertically. Supports 8-bit, 16-bit, and float32 Images.
    pub fn flip_vertical(&self, py: Python<'_>) -> PyResult<Self> {
        match &self.data {
            ImageData::U8(_) => {
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
            ImageData::U16(_) => self.flip_u16(py, FlipDir::Vertical),
            ImageData::F32(_) => self.flip_f32(py, FlipDir::Vertical),
        }
    }

    /// Crop image. 8-bit only.
    ///
    /// Two call conventions are supported:
    ///
    /// - kornia: ``img.crop(x, y, width, height)``
    /// - PIL: ``img.crop((left, upper, right, lower))`` — equivalent to
    ///   ``img.crop(left, upper, right - left, lower - upper)``
    #[pyo3(signature = (x, y=None, width=None, height=None))]
    pub fn crop(
        &self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        y: Option<usize>,
        width: Option<usize>,
        height: Option<usize>,
    ) -> PyResult<Self> {
        let (cx, cy, cw, ch) = if let Ok(box_) = x.extract::<(usize, usize, usize, usize)>() {
            if y.is_some() || width.is_some() || height.is_some() {
                return Err(value_err(
                    "crop: pass either a 4-tuple box OR (x, y, width, height) — not both",
                ));
            }
            let (left, upper, right, lower) = box_;
            if right < left || lower < upper {
                return Err(value_err(
                    "crop: PIL-style box requires right >= left and lower >= upper",
                ));
            }
            (left, upper, right - left, lower - upper)
        } else {
            let cx: usize = x.extract().map_err(|_| {
                value_err("crop: first arg must be int x or 4-tuple (left,upper,right,lower)")
            })?;
            let cy = y.ok_or_else(|| value_err("crop: missing y"))?;
            let cw = width.ok_or_else(|| value_err("crop: missing width"))?;
            let ch = height.ok_or_else(|| value_err("crop: missing height"))?;
            (cx, cy, cw, ch)
        };
        self.crop_xywh(py, cx, cy, cw, ch)
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
            let result = crate::blur::box_blur(py, data.clone_ref(py), (kernel_size, kernel_size))?;
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

    /// PIL-style ``img.convert(mode)`` — return a new Image in the requested
    /// mode. Supported targets:
    ///
    /// - ``"L"`` (1ch u8 grayscale): from RGB / RGBA / I;16 / RGB;16 / RGBA;16
    /// - ``"RGB"`` (3ch u8): from L / RGBA / I;16 / RGB;16 / RGBA;16
    /// - ``"RGBA"`` (4ch u8): from L / RGB (alpha=255) or directly
    /// - ``"I;16"`` (1ch u16 grayscale): from L (×257) or RGB;16
    ///
    /// Returns a new Image; the original is unchanged. Conversions between
    /// 8-bit and 16-bit dtypes scale the integer range proportionally
    /// (×257 / ÷257) so that pure white / pure black map correctly.
    fn convert(&self, py: Python<'_>, mode: &str) -> PyResult<Self> {
        if mode == self.mode {
            return self.copy(py);
        }
        match (self.mode.as_str(), mode) {
            ("RGB", "L") | ("RGBA", "L") => {
                if self.mode == "RGBA" {
                    let rgb = self.convert(py, "RGB")?;
                    return rgb.convert(py, "L");
                }
                self.to_grayscale(py)
            }
            ("L", "RGB") | ("RGBA", "RGB") | ("L", "RGBA") => self.to_rgb(py),
            ("RGB", "RGBA") => {
                let data = self.require_u8("convert")?;
                let arr = data.bind(py);
                let s = arr.shape();
                let (h, w) = (s[0], s[1]);
                let src = unsafe { std::slice::from_raw_parts(arr.data(), h * w * 3) };
                let out = unsafe { PyArray::<u8, _>::new(py, [h, w, 4], false) };
                let dst = unsafe { std::slice::from_raw_parts_mut(out.data(), h * w * 4) };
                for (i, px) in src.chunks_exact(3).enumerate() {
                    let o = i * 4;
                    dst[o] = px[0];
                    dst[o + 1] = px[1];
                    dst[o + 2] = px[2];
                    dst[o + 3] = 255;
                }
                Ok(Self::wrap(py, out.unbind(), Some("RGBA".to_string())))
            }
            ("I;16", "L") => convert_u16_to_u8(py, &self.data, 1, "L".to_string()),
            ("RGB;16", "RGB") => convert_u16_to_u8(py, &self.data, 3, "RGB".to_string()),
            ("RGBA;16", "RGBA") => convert_u16_to_u8(py, &self.data, 4, "RGBA".to_string()),
            ("L", "I;16") => convert_u8_to_u16(py, &self.data, 1, "I;16".to_string()),
            ("RGB", "RGB;16") => convert_u8_to_u16(py, &self.data, 3, "RGB;16".to_string()),
            ("RGBA", "RGBA;16") => convert_u8_to_u16(py, &self.data, 4, "RGBA;16".to_string()),
            _ => Err(value_err(format!(
                "convert: {:?} -> {:?} is not supported",
                self.mode, mode
            ))),
        }
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
        let result = crate::warp::warp_affine(py, data.clone_ref(py), m, (h, w), "bilinear", None)?;
        Ok(Self::wrap(py, result, Some(self.mode.clone())))
    }

    // --- Serialization for multiprocess (Ray Data, etc.) ---

    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let cls = Self::type_object(py).unbind().into_any();
        // Pickle as (numpy_array, mode) — `frombuffer` will auto-detect the
        // dtype on reconstruction, so we don't need to thread an extra tag.
        let arr_any = self.data.as_pyany(py);
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
        (self.data.as_pyany(py), self.mode.clone())
    }

    fn __setstate__(&mut self, py: Python<'_>, state: (Py<PyAny>, String)) -> PyResult<()> {
        let bound = state.0.bind(py);
        if let Ok(arr) = bound.extract::<Py<PyArray3<u8>>>() {
            self.data = ImageData::U8(arr);
        } else if let Ok(arr) = bound.extract::<Py<PyArray3<u16>>>() {
            self.data = ImageData::U16(arr);
        } else if let Ok(arr) = bound.extract::<Py<PyArray3<f32>>>() {
            self.data = ImageData::F32(arr);
        } else {
            return Err(value_err(
                "__setstate__: array dtype must be uint8, uint16, or float32",
            ));
        }
        self.mode = state.1;
        self.format = None;
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
            (ImageData::F32(a), ImageData::F32(b)) => {
                let a = a.bind(py);
                let b = b.bind(py);
                if a.shape() != b.shape() {
                    return false;
                }
                let len: usize = a.shape().iter().product();
                let sa = unsafe { std::slice::from_raw_parts(a.data(), len) };
                let sb = unsafe { std::slice::from_raw_parts(b.data(), len) };
                // Bit-equality on f32 (NaN!=NaN, but PIL's Image.__eq__ doesn't
                // treat NaN images as equal either).
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
        let arr_any = self.data.as_pyany(py);
        let bound = arr_any.bind(py);
        if let Some(dt) = dtype {
            Ok(bound.call_method1("astype", (dt,))?.unbind())
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
        let arr_ptr = slf.data.as_ptr(py);
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
