use numpy::{PyArray, PyArray3, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::PyTypeInfo;

use crate::backing;
use kornia_image::{ColorSpace, Image, ImageError, ImageLayout, ImageSize, PixelFormat};
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
) -> PyResult<Image<u8, C>> {
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
    Image::from_raw_parts(
        size,
        arr.data() as *const u8,
        h * w * C,
        kornia_image::allocator::host_alloc(),
    )
    .map_err(to_pyerr)
}

/// Zero-copy wrap a numpy u16 array as a Rust Image for reading.
///
/// The caller MUST ensure the Py<PyArray3<u16>> stays alive for the lifetime of the Image.
pub(crate) unsafe fn numpy_as_image_u16<const C: usize>(
    py: Python<'_>,
    image: &Py<PyArray3<u16>>,
) -> PyResult<Image<u16, C>> {
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
    Image::from_raw_parts(
        size,
        arr.data() as *const u16,
        len_bytes,
        kornia_image::allocator::host_alloc(),
    )
    .map_err(to_pyerr)
}

/// Zero-copy wrap a numpy f32 array as a Rust Image for reading.
///
/// The caller MUST ensure the Py<PyArray3<f32>> stays alive for the lifetime of the Image.
pub(crate) unsafe fn numpy_as_image_f32<const C: usize>(
    py: Python<'_>,
    image: &Py<PyArray3<f32>>,
) -> PyResult<Image<f32, C>> {
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
    Image::from_raw_parts(
        size,
        arr.data() as *const f32,
        len_bytes,
        kornia_image::allocator::host_alloc(),
    )
    .map_err(to_pyerr)
}

pub(crate) type AllocOutput<T, const C: usize, P> = (Image<T, C>, Py<P>);

pub(crate) unsafe fn alloc_output_pyarray<const C: usize>(
    py: Python<'_>,
    size: ImageSize,
) -> PyResult<AllocOutput<u8, C, PyArray3<u8>>> {
    let arr = PyArray::<u8, _>::new(py, [size.height, size.width, C], false);
    let len = size.height * size.width * C;
    let img = Image::from_raw_parts(
        size,
        arr.data() as *const u8,
        len,
        kornia_image::allocator::host_alloc(),
    )
    .map_err(to_pyerr)?;
    Ok((img, arr.unbind()))
}

pub(crate) unsafe fn alloc_output_pyarray_u16<const C: usize>(
    py: Python<'_>,
    size: ImageSize,
) -> PyResult<AllocOutput<u16, C, PyArray3<u16>>> {
    let arr = PyArray::<u16, _>::new(py, [size.height, size.width, C], false);
    let len = size.height * size.width * C * std::mem::size_of::<u16>();
    let img = Image::from_raw_parts(
        size,
        arr.data() as *const u16,
        len,
        kornia_image::allocator::host_alloc(),
    )
    .map_err(to_pyerr)?;
    Ok((img, arr.unbind()))
}

pub(crate) unsafe fn alloc_output_pyarray_f32<const C: usize>(
    py: Python<'_>,
    size: ImageSize,
) -> PyResult<AllocOutput<f32, C, PyArray3<f32>>> {
    let arr = PyArray::<f32, _>::new(py, [size.height, size.width, C], false);
    let len = size.height * size.width * C * std::mem::size_of::<f32>();
    let img = Image::from_raw_parts(
        size,
        arr.data() as *const f32,
        len,
        kornia_image::allocator::host_alloc(),
    )
    .map_err(to_pyerr)?;
    Ok((img, arr.unbind()))
}

/// Copy numpy u8 data into a kornia_image::allocator::host_alloc() f32 Image (for Category B ops needing f32).
///
/// Uses zero-copy read via numpy_as_image, then a single u8→f32 collect.
pub(crate) fn numpy_to_f32_image<const C: usize>(
    py: Python<'_>,
    image: &Py<PyArray3<u8>>,
) -> PyResult<Image<f32, C>> {
    let src = unsafe { numpy_as_image::<C>(py, image)? };
    let f32_data: Vec<f32> = src.as_slice().iter().map(|&v| v as f32).collect();
    Image::new(src.size(), f32_data).map_err(to_pyerr)
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
fn convert_u16_to_u8(img: &PyImageApi, channels: usize, mode: String) -> PyResult<PyImageApi> {
    if img.dtype != backing::Dtype::U16 {
        return Err(value_err("convert_u16_to_u8 called on non-u16 image"));
    }
    let [h, w, c] = img.shape;
    if c != channels {
        return Err(value_err(format!(
            "convert: source has {} channels, target requires {}",
            c, channels
        )));
    }
    let n = h * w * channels;
    let src = unsafe { std::slice::from_raw_parts(img.backing.data_ptr() as *const u16, n) };
    let mut out = backing::AlignedBytes::zeroed(n);
    let dst = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr(), n) };
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = (s >> 8) as u8;
    }
    Ok(PyImageApi::from_owned_bytes(
        out,
        backing::Dtype::U8,
        [h, w, channels],
        img.color_space,
        mode,
    ))
}

/// Scale a u8 buffer up to u16 via ``v * 257`` so 0xFF maps to 0xFFFF —
/// matches PIL's "I;16" upcast convention.
fn convert_u8_to_u16(img: &PyImageApi, channels: usize, mode: String) -> PyResult<PyImageApi> {
    if img.dtype != backing::Dtype::U8 {
        return Err(value_err("convert_u8_to_u16 called on non-u8 image"));
    }
    let [h, w, c] = img.shape;
    if c != channels {
        return Err(value_err(format!(
            "convert: source has {} channels, target requires {}",
            c, channels
        )));
    }
    let n = h * w * channels;
    let src = unsafe { std::slice::from_raw_parts(img.backing.data_ptr(), n) };
    let mut out = backing::AlignedBytes::zeroed(n * 2);
    let dst = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u16, n) };
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = (*s as u16) * 257;
    }
    Ok(PyImageApi::from_owned_bytes(
        out,
        backing::Dtype::U16,
        [h, w, channels],
        img.color_space,
        mode,
    ))
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
pub(crate) fn mode_from_channels(channels: usize, is_u16: bool) -> String {
    let (gray, suffix) = if is_u16 { ("I", ";16") } else { ("L", "") };
    match channels {
        1 => format!("{}{}", gray, suffix),
        3 => format!("RGB{}", suffix),
        4 => format!("RGBA{}", suffix),
        c => format!("{}ch{}", c, suffix),
    }
}

/// Default color space derived from channel count: 1->Gray, 4->Rgba, else Rgb.
pub(crate) fn default_color_space(channels: usize) -> kornia_image::ColorSpace {
    match channels {
        1 => kornia_image::ColorSpace::Gray,
        4 => kornia_image::ColorSpace::Rgba,
        _ => kornia_image::ColorSpace::Rgb,
    }
}

/// PIL-style mode string for f32 storage. Single-channel uses PIL's
/// canonical ``"F"`` (32-bit float); multi-channel uses an "f" suffix.
pub(crate) fn mode_from_channels_f32(channels: usize) -> String {
    match channels {
        1 => "F".to_string(),
        3 => "RGBf".to_string(),
        4 => "RGBAf".to_string(),
        c => format!("{}chf", c),
    }
}

/// Default PIL-style mode string for a `(dtype, channels)` pair — the single
/// mapping every "infer the mode from the buffer's dtype" call site shares.
pub(crate) fn mode_for_dtype(dtype: backing::Dtype, channels: usize) -> String {
    match dtype {
        backing::Dtype::U8 => mode_from_channels(channels, false),
        backing::Dtype::U16 => mode_from_channels(channels, true),
        backing::Dtype::F32 => mode_from_channels_f32(channels),
    }
}

/// A high-level image object backed by a numpy-agnostic [`backing::Backing`]
/// buffer plus shape/dtype/color metadata.
///
/// Storage is HWC (height, width, channels). A freshly-ingested numpy array is
/// *borrowed* zero-copy (the original ndarray is kept alive); all imgproc /
/// color / convert ops produce an *owned* 64-byte-aligned buffer. Supports 8-bit
/// (`uint8`), 16-bit (`uint16`), and 32-bit float (`float32`) depths.
///
/// Imgproc methods (resize, flip, blur, color conversions, …) currently only
/// support 8-bit images. Calling them on a 16-bit / f32 Image raises
/// `NotImplementedError` with a clear remediation message.
///
/// Thread-safe and serialization-friendly for use with Ray Data,
/// multiprocessing, and other parallel execution frameworks.
#[pyclass(name = "Image", weakref, module = "kornia_rs.image")]
pub struct PyImageApi {
    pub(crate) backing: backing::Backing,
    pub(crate) dtype: backing::Dtype,
    /// `(height, width, channels)`.
    pub(crate) shape: [usize; 3],
    /// The per-pixel color space this Image is interpreted as. Defaults by
    /// channel count on construction (1->Gray, 3->Rgb, 4->Rgba) and is updated
    /// by `cvt_color`.
    pub(crate) color_space: kornia_image::ColorSpace,
    pub(crate) mode: String,
    /// Canonical format the Image was decoded from (e.g. ``"PNG"``, ``"JPEG"``,
    /// ``"TIFF"``). ``None`` for in-memory-constructed Images. Set by
    /// ``Image.load`` / ``Image.decode`` / ``Image.open``.
    pub(crate) format: Option<&'static str>,
}

impl PyImageApi {
    #[inline]
    fn nchannels(&self) -> usize {
        self.shape[2]
    }
    /// `(height, width, channels)` accessor for other crate modules.
    #[inline]
    pub(crate) fn shape_hwc(&self) -> (usize, usize, usize) {
        (self.shape[0], self.shape[1], self.shape[2])
    }
    /// Public wrapper so augmentations can build owned results.
    pub(crate) fn wrap_u8_result_pub(&self, py: Python<'_>, arr: Py<PyArray3<u8>>) -> Self {
        self.wrap_u8_result(py, arr)
    }
    #[inline]
    fn nbytes_total(&self) -> usize {
        self.shape[0] * self.shape[1] * self.shape[2] * self.dtype.itemsize()
    }
    #[inline]
    fn nelems(&self) -> usize {
        self.shape[0] * self.shape[1] * self.shape[2]
    }
    #[inline]
    fn is_u16(&self) -> bool {
        self.dtype == backing::Dtype::U16
    }
    #[inline]
    fn is_f32(&self) -> bool {
        self.dtype == backing::Dtype::F32
    }
    #[inline]
    fn dtype_name(&self) -> &'static str {
        self.dtype.name()
    }

    /// numpy dtype descriptor for the `dtype` getter.
    ///
    /// Delegates to the free function [`dtype_to_numpy_obj`].
    fn dtype_obj(&self, py: Python<'_>) -> Py<PyAny> {
        dtype_to_numpy_obj(self.dtype, py)
    }

    /// Build an owned (copied) image from a typed numpy array. Used to convert
    /// the numpy outputs of the imgproc submodule helpers into `Backing::Owned`.
    fn owned_from_numpy_u8(
        &self,
        py: Python<'_>,
        arr: Py<PyArray3<u8>>,
        mode: Option<String>,
    ) -> Self {
        let c = arr.bind(py).shape()[2];
        let mode = mode.unwrap_or_else(|| mode_from_channels(c, false));
        Self::copy_numpy_into_owned::<u8>(py, &arr, backing::Dtype::U8, self.color_space, mode)
    }

    /// Allocate a numpy view (zero-copy) over this image's backing, tied to a
    /// keep-alive that owns the bytes. For borrowed-numpy images this returns a
    /// clone of the original ndarray (true memory identity); for owned buffers
    /// it builds a fresh view whose base is `keep` (must keep the bytes alive).
    unsafe fn make_typed_view<T: numpy::Element>(
        &self,
        py: Python<'_>,
        keep: Py<PyAny>,
    ) -> PyResult<Py<PyArray3<T>>> {
        let [h, w, c] = self.shape;
        unsafe {
            crate::numpy_view::view3::<T>(
                py,
                self.backing.data_ptr(),
                h,
                w,
                c,
                keep,
                self.backing.readonly(),
            )
        }
    }

    /// Build a device-resident `Image` from a [`crate::device::DeviceImage`].
    /// `dtype`/`shape` are derived from the device buffer.
    #[cfg(feature = "cuda")]
    pub(crate) fn from_device(
        img: crate::device::DeviceImage,
        color_space: kornia_image::ColorSpace,
        mode: String,
    ) -> Self {
        let dtype = img.dtype_enum();
        let shape = img.shape_hwc();
        Self {
            backing: backing::Backing::Device {
                img: std::sync::Arc::new(img),
                readonly: false,
            },
            dtype,
            shape,
            color_space,
            mode,
            format: None,
        }
    }

    /// Build a device-resident `Image` from a shared [`crate::device::DeviceImage`]
    /// handle (e.g. a zero-copy DLPack import). `dtype`/`shape` are derived.
    #[cfg(feature = "cuda")]
    pub(crate) fn from_device_arc(
        img: std::sync::Arc<crate::device::DeviceImage>,
        readonly: bool,
        color_space: kornia_image::ColorSpace,
        mode: String,
    ) -> Self {
        let dtype = img.dtype_enum();
        let shape = img.shape_hwc();
        Self {
            backing: backing::Backing::Device { img, readonly },
            dtype,
            shape,
            color_space,
            mode,
            format: None,
        }
    }

    /// Borrow the underlying device image, if this Image is device-resident.
    #[cfg(feature = "cuda")]
    pub(crate) fn as_device(&self) -> Option<&crate::device::DeviceImage> {
        match &self.backing {
            backing::Backing::Device { img, .. } => Some(img.as_ref()),
            _ => None,
        }
    }

    /// True when the image is device-resident (CUDA). Cheap; no feature gate.
    pub(crate) fn is_device(&self) -> bool {
        self.backing.device().0 == dlpack_rs::ffi::DLDeviceType::kDLCUDA as i32
    }

    /// Clone the shared handle to the underlying device image (cheap `Arc`
    /// clone) — used as a DLPack keep-alive and by `.cuda()` on a device image.
    #[cfg(feature = "cuda")]
    pub(crate) fn device_arc(&self) -> Option<std::sync::Arc<crate::device::DeviceImage>> {
        match &self.backing {
            backing::Backing::Device { img, .. } => Some(img.clone()),
            _ => None,
        }
    }
}

/// Map a [`backing::Dtype`] to the corresponding numpy dtype descriptor object.
///
/// This is the single canonical place for the Dtype → numpy dtype mapping.
/// All call sites in this module should go through this function rather than
/// duplicating the match table.
fn dtype_to_numpy_obj(dtype: backing::Dtype, py: Python<'_>) -> Py<PyAny> {
    let d = match dtype {
        backing::Dtype::U8 => numpy::dtype::<u8>(py),
        backing::Dtype::U16 => numpy::dtype::<u16>(py),
        backing::Dtype::F32 => numpy::dtype::<f32>(py),
    };
    d.into_any().unbind()
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

/// Dispatch a color-space conversion over a `PyImageApi`, producing a new
/// owned `PyImageApi`. Validates nothing — callers must pre-check dtype / legality.
fn dispatch_cvt(
    py: Python<'_>,
    img: &PyImageApi,
    from: kornia_image::ColorSpace,
    to: kornia_image::ColorSpace,
) -> PyResult<PyImageApi> {
    use kornia_image::ColorSpace as CS;
    use kornia_imgproc::color as kc;

    let is_f32 = img.is_f32();

    // u8 N->M conversion into a fresh owned buffer.
    macro_rules! u8_conv {
        ($cin:literal, $cout:literal, $func:expr) => {{
            let src = unsafe { img.borrow_self::<u8, $cin>().map_err(to_pyerr)? };
            img.run_into_owned_u8::<$cout, _>(py, src.size(), |dst| ($func)(&src, dst))
        }};
    }
    // f32 N->M conversion into a fresh owned buffer.
    macro_rules! f32_conv {
        ($cin:literal, $cout:literal, $func:expr) => {{
            let src = unsafe { img.borrow_self::<f32, $cin>().map_err(to_pyerr)? };
            img.run_into_owned_f32::<$cout, _>(py, src.size(), |dst| ($func)(&src, dst))
        }};
    }
    macro_rules! f32_3to3 {
        ($func:path) => {{
            f32_conv!(3, 3, $func)
        }};
    }
    macro_rules! u8_3to3 {
        ($func:path) => {{
            u8_conv!(3, 3, $func)
        }};
    }

    match (from, to) {
        // channel-swap BGR<->RGB: works on both u8 and f32
        (CS::Rgb, CS::Bgr) | (CS::Bgr, CS::Rgb) if is_f32 => {
            f32_3to3!(kc::bgr_from_rgb)
        }
        (CS::Rgb, CS::Bgr) | (CS::Bgr, CS::Rgb) => {
            u8_3to3!(kc::bgr_from_rgb)
        }
        (CS::Rgb, CS::Gray) if !is_f32 => u8_conv!(3, 1, kc::gray_from_rgb_u8),
        (CS::Rgb, CS::Gray) if is_f32 => f32_conv!(3, 1, kc::gray_from_rgb_f32),
        (CS::Gray, CS::Rgb) if !is_f32 => u8_conv!(1, 3, kc::rgb_from_gray),
        (CS::Gray, CS::Rgb) if is_f32 => f32_conv!(1, 3, kc::rgb_from_gray),
        // f32-only perceptual/cylindrical conversions (3->3); cvt_color already
        // rejects non-f32 storage for these via the `requires_f32` guard above.
        (CS::Rgb, CS::Hsv) => f32_3to3!(kc::hsv_from_rgb),
        (CS::Hsv, CS::Rgb) => f32_3to3!(kc::rgb_from_hsv),
        (CS::Rgb, CS::Hls) => f32_3to3!(kc::hls_from_rgb),
        (CS::Hls, CS::Rgb) => f32_3to3!(kc::rgb_from_hls),
        (CS::Rgb, CS::Lab) => f32_3to3!(kc::lab_from_rgb),
        (CS::Lab, CS::Rgb) => f32_3to3!(kc::rgb_from_lab),
        (CS::Rgb, CS::Luv) => f32_3to3!(kc::luv_from_rgb),
        (CS::Luv, CS::Rgb) => f32_3to3!(kc::rgb_from_luv),
        (CS::Rgb, CS::Xyz) => f32_3to3!(kc::xyz_from_rgb),
        (CS::Xyz, CS::Rgb) => f32_3to3!(kc::rgb_from_xyz),
        (CS::Rgb, CS::LinearRgb) => f32_3to3!(kc::linear_rgb_from_rgb),
        (CS::LinearRgb, CS::Rgb) => f32_3to3!(kc::rgb_from_linear_rgb),
        // YCbCr / Yuv support both u8 and f32 (YuvFamily). Route through f32
        // when storage is f32, fall through to u8 path otherwise.
        (CS::Rgb, CS::YCbCr) if is_f32 => f32_3to3!(kc::ycbcr_from_rgb),
        (CS::YCbCr, CS::Rgb) if is_f32 => f32_3to3!(kc::rgb_from_ycbcr),
        (CS::Rgb, CS::Yuv) if is_f32 => f32_3to3!(kc::yuv_from_rgb),
        (CS::Yuv, CS::Rgb) if is_f32 => f32_3to3!(kc::rgb_from_yuv),
        (CS::Rgb, CS::YCbCr) => u8_3to3!(kc::ycbcr_from_rgb),
        (CS::YCbCr, CS::Rgb) => u8_3to3!(kc::rgb_from_ycbcr),
        (CS::Rgb, CS::Yuv) => u8_3to3!(kc::yuv_from_rgb),
        (CS::Yuv, CS::Rgb) => u8_3to3!(kc::rgb_from_yuv),
        // Alpha-family conversions (u8-only; alpha kernels have no f32 path)
        (CS::Rgb, CS::Rgba) if !is_f32 => u8_conv!(3, 4, kc::rgba_from_rgb),
        (CS::Rgb, CS::Bgra) if !is_f32 => u8_conv!(3, 4, kc::bgra_from_rgb),
        (CS::Rgba, CS::Rgb) if !is_f32 => u8_conv!(4, 3, |s, d| kc::rgb_from_rgba(s, d, None)),
        (CS::Bgra, CS::Rgb) if !is_f32 => u8_conv!(4, 3, |s, d| kc::rgb_from_bgra(s, d, None)),
        // f32 with alpha pairs: clear error (alpha kernels are u8-only)
        (CS::Rgb, CS::Rgba) | (CS::Rgb, CS::Bgra) | (CS::Rgba, CS::Rgb) | (CS::Bgra, CS::Rgb) => {
            Err(value_err(format!(
                "alpha-family conversions ({from:?}->{to:?}) require uint8 storage; call to_uint8() first"
            )))
        }
        _ => Err(value_err(ImageError::UnsupportedColorConversion { from, to }.to_string())),
    }
}

impl PyImageApi {
    /// Construct an owned image from a fresh aligned byte buffer.
    pub(crate) fn from_owned_bytes(
        b: backing::AlignedBytes,
        dtype: backing::Dtype,
        shape: [usize; 3],
        cs: ColorSpace,
        mode: String,
    ) -> Self {
        Self {
            backing: backing::Backing::Owned(b),
            dtype,
            shape,
            color_space: cs,
            mode,
            format: None,
        }
    }

    /// Generic helper: copy a typed 3-D numpy array into a fresh owned aligned buffer.
    /// This is the single implementation body shared by `wrap`, `wrap_u16`, `wrap_f32`,
    /// and `owned_from_numpy_u8`.
    fn copy_numpy_into_owned<T: numpy::Element>(
        py: Python<'_>,
        arr: &Py<PyArray3<T>>,
        dtype: backing::Dtype,
        cs: ColorSpace,
        mode: String,
    ) -> Self {
        let b = arr.bind(py);
        let s = b.shape();
        let (h, w, c) = (s[0], s[1], s[2]);
        // Checked multiplication: the numpy array is already allocated so this
        // should never overflow in practice, but we guard the unsafe slice
        // creation defensively. If somehow the shape is adversarial, this is
        // unreachable — numpy itself would have failed to allocate first.
        let n_bytes = backing::byte_len(h, w, c, dtype)
            .expect("image dimensions overflow usize — should have been caught at ingest");
        let src = unsafe { std::slice::from_raw_parts(b.data() as *const u8, n_bytes) };
        let bytes = backing::AlignedBytes::from_slice(src);
        Self::from_owned_bytes(bytes, dtype, [h, w, c], cs, mode)
    }

    /// Wrap a freshly-allocated 8-bit numpy array by copying it into an owned
    /// aligned buffer. Mode defaults to `"L"`/`"RGB"`/`"RGBA"` by channel count.
    pub fn wrap(py: Python<'_>, data: Py<PyArray3<u8>>, mode: Option<String>) -> Self {
        let c = data.bind(py).shape()[2];
        let mode = mode.unwrap_or_else(|| mode_from_channels(c, false));
        let cs = default_color_space(c);
        Self::copy_numpy_into_owned::<u8>(py, &data, backing::Dtype::U8, cs, mode)
    }

    /// Wrap a freshly-allocated 16-bit numpy array into an owned buffer.
    pub fn wrap_u16(py: Python<'_>, data: Py<PyArray3<u16>>, mode: Option<String>) -> Self {
        let c = data.bind(py).shape()[2];
        let mode = mode.unwrap_or_else(|| mode_from_channels(c, true));
        let cs = default_color_space(c);
        Self::copy_numpy_into_owned::<u16>(py, &data, backing::Dtype::U16, cs, mode)
    }

    /// Wrap a freshly-allocated 32-bit float numpy array into an owned buffer.
    pub fn wrap_f32(py: Python<'_>, data: Py<PyArray3<f32>>, mode: Option<String>) -> Self {
        let c = data.bind(py).shape()[2];
        let mode = mode.unwrap_or_else(|| mode_from_channels_f32(c));
        let cs = default_color_space(c);
        Self::copy_numpy_into_owned::<f32>(py, &data, backing::Dtype::F32, cs, mode)
    }

    fn with_format(mut self, format: &'static str) -> Self {
        self.format = Some(format);
        self
    }

    /// Borrow a numpy ndarray zero-copy as the default ingest path. Keeps the
    /// original ndarray alive (memory identity preserved). Optionally forces a
    /// deep copy into an owned aligned buffer.
    pub(crate) fn from_numpy_borrow(
        py: Python<'_>,
        arr: &Bound<'_, PyAny>,
        mode: Option<String>,
        cs: Option<ColorSpace>,
        copy: bool,
    ) -> PyResult<Self> {
        // Resolve a typed, C-contiguous 3D numpy array (reshape 2D -> (H,W,1)).
        let (obj, dtype): (Bound<'_, PyAny>, backing::Dtype) = {
            // Probe ndim and dtype via attributes/extraction.
            let resolved =
                if let Ok(shape) = arr.getattr("shape").and_then(|s| s.extract::<Vec<usize>>()) {
                    if shape.len() == 2 {
                        arr.call_method1("reshape", ((shape[0], shape[1], 1usize),))?
                    } else if shape.len() == 3 {
                        arr.clone()
                    } else {
                        return Err(value_err(format!(
                            "Expected 2D or 3D array, got {}D",
                            shape.len()
                        )));
                    }
                } else {
                    return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                        "Image requires a numpy array or array-like object with .shape",
                    ));
                };
            // Determine dtype from the authoritative numpy dtype descriptor,
            // not from speculative typed extraction (which can hide implicit casts).
            let dtype_name: String = resolved
                .getattr("dtype")
                .and_then(|d| d.getattr("name"))
                .and_then(|n| n.extract::<String>())
                .map_err(|_| value_err("could not read numpy array dtype"))?;
            let dtype = backing::Dtype::from_numpy_str(&dtype_name)?;
            (resolved, dtype)
        };

        // Pull shape, contiguity, writeability and base pointer via numpy.
        let (h, w, c, ptr, c_contig, writeable) = {
            let writeable = obj
                .getattr("flags")
                .ok()
                .and_then(|f| f.getattr("writeable").ok())
                .and_then(|w| w.extract::<bool>().ok())
                .unwrap_or(true);
            macro_rules! probe {
                ($t:ty) => {{
                    let a = obj.extract::<Py<PyArray3<$t>>>()?;
                    let b = a.bind(py);
                    let s = b.shape();
                    (
                        s[0],
                        s[1],
                        s[2],
                        b.data() as *mut u8,
                        b.is_c_contiguous(),
                        writeable,
                    )
                }};
            }
            match dtype {
                backing::Dtype::U8 => probe!(u8),
                backing::Dtype::U16 => probe!(u16),
                backing::Dtype::F32 => probe!(f32),
            }
        };

        let mode = mode.unwrap_or_else(|| mode_for_dtype(dtype, c));
        let cs = cs.unwrap_or_else(|| default_color_space(c));

        if copy || !c_contig {
            // Force an owned, contiguous buffer (ascontiguousarray semantics).
            let n = backing::byte_len(h, w, c, dtype)?;
            let contig = if c_contig {
                obj
            } else {
                let np = py.import("numpy")?;
                np.call_method1("ascontiguousarray", (&obj,))?
            };
            let bptr = match dtype {
                backing::Dtype::U8 => {
                    contig.extract::<Py<PyArray3<u8>>>()?.bind(py).data() as *const u8
                }
                backing::Dtype::U16 => {
                    contig.extract::<Py<PyArray3<u16>>>()?.bind(py).data() as *const u8
                }
                backing::Dtype::F32 => {
                    contig.extract::<Py<PyArray3<f32>>>()?.bind(py).data() as *const u8
                }
            };
            let src = unsafe { std::slice::from_raw_parts(bptr, n) };
            let bytes = backing::AlignedBytes::from_slice(src);
            return Ok(Self::from_owned_bytes(bytes, dtype, [h, w, c], cs, mode));
        }

        // Zero-copy borrow: keep the original ndarray alive.
        let nn = std::ptr::NonNull::new(ptr)
            .ok_or_else(|| value_err("numpy array has null data pointer"))?;
        let backing = backing::Backing::Borrowed {
            ptr: nn,
            keep: backing::BorrowGuard::PyObject {
                obj: obj.clone().unbind(),
                buffer: None,
            },
            readonly: !writeable,
            // numpy borrows are always host memory.
            device: (dlpack_rs::ffi::K_DL_CPU as i32, 0),
        };
        Ok(Self {
            backing,
            dtype,
            shape: [h, w, c],
            color_space: cs,
            mode,
            format: None,
        })
    }

    /// Compute borrow of self's data as a typed read Image (no copy).
    ///
    /// # Safety
    /// The returned Image borrows self's buffer; self must outlive it.
    pub(crate) unsafe fn borrow_self<T: Clone, const C: usize>(
        &self,
    ) -> Result<Image<T, C>, ImageError> {
        unsafe { backing::borrow_image::<T, C>(&self.backing, self.shape) }
    }

    /// Allocate an owned u8 output of channel count `CO`, run `f` over a mutable
    /// borrow of it, and wrap as an owned image preserving mode + color_space.
    fn run_into_owned_u8<const CO: usize, F>(
        &self,
        py: Python<'_>,
        out_size: ImageSize,
        f: F,
    ) -> PyResult<Self>
    where
        F: FnOnce(&mut Image<u8, CO>) -> Result<(), ImageError> + Send,
    {
        let (mut bytes, size) = backing::alloc_output_owned::<CO>(backing::Dtype::U8, out_size)?;
        let mut dst = unsafe {
            Image::<u8, CO>::from_raw_parts(
                size,
                bytes.as_mut_ptr(),
                size.width * size.height * CO * std::mem::size_of::<u8>(),
                kornia_image::allocator::host_alloc(),
            )
            .map_err(to_pyerr)?
        };
        py.detach(|| f(&mut dst)).map_err(to_pyerr)?;
        Ok(Self::from_owned_bytes(
            bytes,
            backing::Dtype::U8,
            [size.height, size.width, CO],
            self.color_space,
            self.mode.clone(),
        ))
    }

    /// f32 sibling of [`Self::run_into_owned_u8`].
    fn run_into_owned_f32<const CO: usize, F>(
        &self,
        py: Python<'_>,
        out_size: ImageSize,
        f: F,
    ) -> PyResult<Self>
    where
        F: FnOnce(&mut Image<f32, CO>) -> Result<(), ImageError> + Send,
    {
        let (mut bytes, size) = backing::alloc_output_owned::<CO>(backing::Dtype::F32, out_size)?;
        let mut dst = unsafe {
            Image::<f32, CO>::from_raw_parts(
                size,
                bytes.as_mut_ptr() as *const f32,
                size.width * size.height * CO * std::mem::size_of::<f32>(),
                kornia_image::allocator::host_alloc(),
            )
            .map_err(to_pyerr)?
        };
        py.detach(|| f(&mut dst)).map_err(to_pyerr)?;
        Ok(Self::from_owned_bytes(
            bytes,
            backing::Dtype::F32,
            [size.height, size.width, CO],
            self.color_space,
            self.mode.clone(),
        ))
    }

    /// Build a numpy array (typed by dtype) that *views* self's u8 backing,
    /// for handing to an imgproc submodule helper. Tied to a transient keep so
    /// the data stays valid; safe because `self` outlives the call.
    fn as_numpy_u8(&self, py: Python<'_>) -> PyResult<Py<PyArray3<u8>>> {
        let keep = self.borrow_keepalive(py);
        unsafe { self.make_typed_view::<u8>(py, keep) }
    }

    /// Keep-alive object for a transient numpy view over self's backing. For a
    /// borrowed-numpy image this is the original ndarray (real owner); for an
    /// owned buffer it's `None` (the caller guarantees `self` outlives the view).
    ///
    /// # INVARIANT
    ///
    /// When `Backing::Owned`, this returns `py.None()`. The numpy view is only
    /// used transiently inside `&self` methods called from Python (`#[pymethods]`);
    /// the GIL and pyo3's calling convention guarantee `self` stays alive for the
    /// duration of the call, so the buffer is never actually freed while the view
    /// is live.
    ///
    /// INVARIANT: No concurrent write to the backing buffer while a writable numpy
    /// view is live. Enforced by GIL — all mutations go through `&mut self` methods
    /// called from Python.
    fn borrow_keepalive(&self, py: Python<'_>) -> Py<PyAny> {
        match &self.backing {
            backing::Backing::Borrowed {
                keep: backing::BorrowGuard::PyObject { obj, .. },
                ..
            } => obj.clone_ref(py),
            // INVARIANT: self outlives this view (called from #[pymethods] &self
            // method; GIL holds). The None base means numpy has no keep-alive, but
            // self is guaranteed alive for the duration of the enclosing call.
            _ => py.None(),
        }
    }

    /// Return a deep-copied owned clone (independent of self's storage).
    pub(crate) fn clone_handle(&self, _py: Python<'_>) -> Self {
        let n = self.nbytes_total();
        let src = unsafe { std::slice::from_raw_parts(self.backing.data_ptr(), n) };
        let bytes = backing::AlignedBytes::from_slice(src);
        Self {
            backing: backing::Backing::Owned(bytes),
            dtype: self.dtype,
            shape: self.shape,
            color_space: self.color_space,
            mode: self.mode.clone(),
            format: self.format,
        }
    }

    /// Wrap a Vec<u8> result as a new owned u8 image preserving mode + cs.
    fn wrap_vec(&self, _py: Python<'_>, out: Vec<u8>, h: usize, w: usize, c: usize) -> Self {
        let bytes = backing::AlignedBytes::from_slice(&out);
        Self::from_owned_bytes(
            bytes,
            backing::Dtype::U8,
            [h, w, c],
            self.color_space,
            self.mode.clone(),
        )
    }

    /// Wrap a PyArray3<u8> produced by an imgproc submodule as a new owned
    /// image, preserving mode + color_space from `self`.
    fn wrap_u8_result(&self, py: Python<'_>, arr: Py<PyArray3<u8>>) -> Self {
        self.owned_from_numpy_u8(py, arr, Some(self.mode.clone()))
    }

    /// Borrow the backing as a `&[u8]` slice over all element bytes.
    fn raw_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.backing.data_ptr(), self.nbytes_total()) }
    }

    /// Borrow the backing as a `&[u8]` element slice (u8 dtype only).
    pub(crate) fn u8_elems(&self) -> &[u8] {
        debug_assert!(self.dtype == backing::Dtype::U8);
        unsafe { std::slice::from_raw_parts(self.backing.data_ptr(), self.nelems()) }
    }

    /// Zero-copy numpy view of the backing. Borrowed-numpy images return the
    /// original ndarray (memory identity); owned images return a fresh view
    /// whose base is the Image itself (keeping the aligned buffer alive).
    fn numpy_view_of(slf: Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let me = slf.borrow();
        // Device-resident: auto-copy back to host (D2H), then view the owned host
        // buffer. Unlike the host path below, this always copies — the array does
        // NOT share the device buffer, so it is returned read-only. Writing it
        // would land on the throwaway host copy and be silently lost; callers who
        // want a writable host image must go through `.cpu()`.
        #[cfg(feature = "cuda")]
        if me.as_device().is_some() {
            let host = me.to_host_internal(py)?;
            drop(me);
            let arr = Self::numpy_view_of(Bound::new(py, host)?)?;
            let kwargs = pyo3::types::PyDict::new(py);
            kwargs.set_item("write", false)?;
            arr.bind(py).call_method("setflags", (), Some(&kwargs))?;
            return Ok(arr);
        }
        // Host guard: numpy views dereference the buffer on the host.
        me.backing.ensure_host()?;
        // Borrowed numpy: return the original ndarray for true identity — but ONLY
        // when the kept object actually IS a numpy ndarray (the `from_numpy` borrow).
        // For other producers kept by `from_dlpack` (e.g. a torch tensor), do NOT
        // return the producer verbatim; fall through to build a real numpy view over
        // the backing pointer (with the Image as base, which keeps the producer alive).
        if let backing::Backing::Borrowed {
            keep: backing::BorrowGuard::PyObject { obj, buffer: None },
            ..
        } = &me.backing
        {
            if obj.bind(py).cast::<numpy::PyUntypedArray>().is_ok() {
                return Ok(obj.clone_ref(py));
            }
        }
        // Owned, dlpack-imported, or buffer-backed borrow: build a view (Image as base).
        let dtype = me.dtype;
        let base: Py<PyAny> = slf.clone().into_any().unbind();
        let arr = unsafe {
            match dtype {
                backing::Dtype::U8 => me.make_typed_view::<u8>(py, base)?.into_any(),
                backing::Dtype::U16 => me.make_typed_view::<u16>(py, base)?.into_any(),
                backing::Dtype::F32 => me.make_typed_view::<f32>(py, base)?.into_any(),
            }
        };
        Ok(arr)
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
        // Host guard: crop copies pixel bytes on the host.
        self.backing.ensure_host()?;
        let [src_h, src_w, c] = self.shape;
        if y + height > src_h || x + width > src_w {
            return Err(value_err(format!(
                "crop: box ({}, {}, {}x{}) out of bounds for ({}, {}, {})",
                x, y, width, height, src_h, src_w, c
            )));
        }
        if self.dtype == backing::Dtype::U8 && c == 3 {
            // Crop directly into an UNINITIALIZED owned output — ONE copy, no
            // pre-zeroing (the OpenCV/numpy trick: never zero a buffer you fully
            // overwrite). The old path did two copies (source -> numpy array ->
            // owned backing) plus a zeroing.
            let src = unsafe { self.borrow_self::<u8, 3>().map_err(to_pyerr)? };
            let out_size = ImageSize { width, height };
            let n = width * height * 3;
            // SAFETY: `crop_image` writes all `n` bytes (height*width*3) before the
            // buffer is wrapped/read, so the uninitialized alloc is fully covered.
            let mut bytes = backing::AlignedBytes::uninit(n);
            let mut dst = unsafe {
                Image::<u8, 3>::from_raw_parts(
                    out_size,
                    bytes.as_mut_ptr(),
                    n,
                    kornia_image::allocator::host_alloc(),
                )
                .map_err(to_pyerr)?
            };
            py.detach(|| kornia_imgproc::crop::crop_image(&src, &mut dst, x, y))
                .map_err(to_pyerr)?;
            return Ok(Self::from_owned_bytes(
                bytes,
                backing::Dtype::U8,
                [height, width, 3],
                self.color_space,
                self.mode.clone(),
            ));
        }
        // Generic byte-level crop for any dtype / channel count.
        // Validate output dimensions before allocation to catch overflow.
        let _total = backing::byte_len(height, width, c, self.dtype)?;
        let isz = self.dtype.itemsize();
        let row_stride = src_w * c * isz;
        let out_row = width * c * isz;
        let src =
            unsafe { std::slice::from_raw_parts(self.backing.data_ptr(), src_h * row_stride) };
        let mut bytes = backing::AlignedBytes::zeroed(height * out_row);
        let dst = unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr(), height * out_row) };
        for row in 0..height {
            let s_off = (y + row) * row_stride + x * c * isz;
            let d_off = row * out_row;
            dst[d_off..d_off + out_row].copy_from_slice(&src[s_off..s_off + out_row]);
        }
        Ok(Self::from_owned_bytes(
            bytes,
            self.dtype,
            [height, width, c],
            self.color_space,
            self.mode.clone(),
        ))
    }

    /// dtype-trivial flip producing an owned buffer (byte-level, any dtype).
    fn flip_pod(&self, dir: FlipDir) -> Self {
        let [h, w, c] = self.shape;
        let isz = self.dtype.itemsize();
        let elem = c * isz; // bytes per pixel
        let row = w * elem;
        let n = h * row;
        let src = unsafe { std::slice::from_raw_parts(self.backing.data_ptr(), n) };
        let mut bytes = backing::AlignedBytes::zeroed(n);
        let dst = unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr(), n) };
        match dir {
            FlipDir::Horizontal => {
                for r in 0..h {
                    let off = r * row;
                    let src_row = &src[off..off + row];
                    let dst_row = &mut dst[off..off + row];
                    for (d, s) in dst_row
                        .chunks_exact_mut(elem)
                        .zip(src_row.chunks_exact(elem).rev())
                    {
                        d.copy_from_slice(s);
                    }
                }
            }
            FlipDir::Vertical => {
                for r in 0..h {
                    let s_off = r * row;
                    let d_off = (h - 1 - r) * row;
                    dst[d_off..d_off + row].copy_from_slice(&src[s_off..s_off + row]);
                }
            }
        }
        Self::from_owned_bytes(
            bytes,
            self.dtype,
            [h, w, c],
            self.color_space,
            self.mode.clone(),
        )
    }

    /// Gate for 8-bit-only imgproc methods: error on u16 / f32 storage.
    pub(crate) fn require_u8(&self, method: &str) -> PyResult<()> {
        match self.dtype {
            backing::Dtype::U8 => Ok(()),
            backing::Dtype::U16 => Err(u16_imgproc_unsupported(method)),
            backing::Dtype::F32 => Err(f32_imgproc_unsupported(method)),
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
        #[cfg_attr(not(feature = "turbojpeg"), allow(unused_variables))] subsampling: Option<&str>,
    ) -> PyResult<Vec<u8>> {
        // Host guard: encoding reads pixel bytes on the host.
        self.backing.ensure_host()?;
        let c = self.nchannels();
        let is_u16 = self.is_u16();
        let is_f32 = self.is_f32();

        match format {
            "jpg" | "jpeg" => {
                if is_u16 || is_f32 {
                    return Err(value_err(format!(
                        "JPEG cannot encode {} images. Use \"png\" or \"tiff\" instead.",
                        self.dtype_name()
                    )));
                }
                if c != 3 {
                    return Err(value_err(format!(
                        "JPEG requires 3-channel RGB image, got {} channels",
                        c
                    )));
                }
                let arr = self.as_numpy_u8(py)?;
                // libjpeg-turbo first (~3-4× faster than zune-jpeg on aarch64);
                // fall back to pure-Rust jpeg if the turbojpeg feature is absent.
                #[cfg(feature = "turbojpeg")]
                if let Ok(b) = crate::io::jpegturbo::encode_image_jpegturbo(
                    py,
                    arr.clone_ref(py),
                    quality as i32,
                    subsampling,
                ) {
                    return Ok(b);
                }
                crate::io::jpeg::encode_image_jpeg(py, arr, quality)
            }
            "png" => {
                let mut buffer = Vec::new();
                match (self.dtype, c) {
                    (backing::Dtype::U8, 3) => {
                        let img = unsafe { self.borrow_self::<u8, 3>().map_err(to_pyerr)? };
                        kornia_io::png::encode_image_png_rgb8(&img, &mut buffer, compress_level)
                            .map_err(to_pyerr)?;
                    }
                    (backing::Dtype::U8, 4) => {
                        let img = unsafe { self.borrow_self::<u8, 4>().map_err(to_pyerr)? };
                        kornia_io::png::encode_image_png_rgba8(&img, &mut buffer, compress_level)
                            .map_err(to_pyerr)?;
                    }
                    (backing::Dtype::U8, 1) => {
                        let img = unsafe { self.borrow_self::<u8, 1>().map_err(to_pyerr)? };
                        kornia_io::png::encode_image_png_gray8(&img, &mut buffer, compress_level)
                            .map_err(to_pyerr)?;
                    }
                    (backing::Dtype::U16, 3) => {
                        let img = unsafe { self.borrow_self::<u16, 3>().map_err(to_pyerr)? };
                        kornia_io::png::encode_image_png_rgb16(&img, &mut buffer, compress_level)
                            .map_err(to_pyerr)?;
                    }
                    (backing::Dtype::U16, 4) => {
                        let img = unsafe { self.borrow_self::<u16, 4>().map_err(to_pyerr)? };
                        kornia_io::png::encode_image_png_rgba16(&img, &mut buffer, compress_level)
                            .map_err(to_pyerr)?;
                    }
                    (backing::Dtype::U16, 1) => {
                        let img = unsafe { self.borrow_self::<u16, 1>().map_err(to_pyerr)? };
                        kornia_io::png::encode_image_png_gray16(&img, &mut buffer, compress_level)
                            .map_err(to_pyerr)?;
                    }
                    _ => {
                        return Err(value_err(format!(
                            "PNG requires 1/3/4-channel image, got {} channels (dtype={})",
                            c,
                            self.dtype_name()
                        )))
                    }
                };
                Ok(buffer)
            }
            "webp" => {
                let mut buffer = Vec::new();
                match (self.dtype, c) {
                    (backing::Dtype::U8, 3) => {
                        let img = unsafe { self.borrow_self::<u8, 3>().map_err(to_pyerr)? };
                        kornia_io::webp::encode_image_webp_rgb8(&img, &mut buffer)
                            .map_err(to_pyerr)?;
                    }
                    (backing::Dtype::U8, 4) => {
                        let img = unsafe { self.borrow_self::<u8, 4>().map_err(to_pyerr)? };
                        kornia_io::webp::encode_image_webp_rgba8(&img, &mut buffer)
                            .map_err(to_pyerr)?;
                    }
                    (backing::Dtype::U8, 1) => {
                        let img = unsafe { self.borrow_self::<u8, 1>().map_err(to_pyerr)? };
                        kornia_io::webp::encode_image_webp_gray8(&img, &mut buffer)
                            .map_err(to_pyerr)?;
                    }
                    (backing::Dtype::U16, _) => {
                        return Err(value_err(
                            "WebP encode requires uint8; convert via img.convert('RGB') first",
                        ))
                    }
                    _ => {
                        return Err(value_err(format!(
                            "WebP requires 1/3/4-channel u8 image, got {} channels (dtype={})",
                            c,
                            self.dtype_name()
                        )))
                    }
                };
                Ok(buffer)
            }
            "tiff" | "tif" => {
                let mut buffer = Vec::new();
                match (self.dtype, c) {
                    (backing::Dtype::U8, 3) => {
                        let img = unsafe { self.borrow_self::<u8, 3>().map_err(to_pyerr)? };
                        kornia_io::tiff::encode_image_tiff_rgb8(&img, &mut buffer)
                            .map_err(to_pyerr)?;
                    }
                    (backing::Dtype::U8, 1) => {
                        let img = unsafe { self.borrow_self::<u8, 1>().map_err(to_pyerr)? };
                        kornia_io::tiff::encode_image_tiff_mono8(&img, &mut buffer)
                            .map_err(to_pyerr)?;
                    }
                    (backing::Dtype::U16, 3) => {
                        let img = unsafe { self.borrow_self::<u16, 3>().map_err(to_pyerr)? };
                        kornia_io::tiff::encode_image_tiff_rgb16(&img, &mut buffer)
                            .map_err(to_pyerr)?;
                    }
                    (backing::Dtype::U16, 1) => {
                        let img = unsafe { self.borrow_self::<u16, 1>().map_err(to_pyerr)? };
                        kornia_io::tiff::encode_image_tiff_mono16(&img, &mut buffer)
                            .map_err(to_pyerr)?;
                    }
                    (backing::Dtype::F32, 3) => {
                        let img = unsafe { self.borrow_self::<f32, 3>().map_err(to_pyerr)? };
                        kornia_io::tiff::encode_image_tiff_rgb32f(&img, &mut buffer)
                            .map_err(to_pyerr)?;
                    }
                    (backing::Dtype::F32, 1) => {
                        let img = unsafe { self.borrow_self::<f32, 1>().map_err(to_pyerr)? };
                        kornia_io::tiff::encode_image_tiff_mono32f(&img, &mut buffer)
                            .map_err(to_pyerr)?;
                    }
                    _ => {
                        return Err(value_err(format!(
                            "TIFF requires 1 or 3-channel image, got {} channels (dtype={})",
                            c,
                            self.dtype_name()
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
    #[pyo3(signature = (data, mode=None, color_space=None))]
    fn new(
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        mode: Option<String>,
        color_space: Option<crate::color_space::PyColorSpace>,
    ) -> PyResult<Self> {
        let cs = color_space.map(Into::into);
        Self::from_numpy_borrow(py, data, mode, cs, false)
    }

    /// Create an Image from a numpy array. Zero-copy by default (``copy=False``):
    /// the Image borrows the array's memory and keeps it alive. Pass
    /// ``copy=True`` to own an independent aligned copy.
    #[staticmethod]
    #[pyo3(signature = (data, mode=None, color_space=None, copy=false))]
    fn from_numpy(
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        mode: Option<String>,
        color_space: Option<crate::color_space::PyColorSpace>,
        copy: bool,
    ) -> PyResult<Self> {
        let cs = color_space.map(Into::into);
        Self::from_numpy_borrow(py, data, mode, cs, copy)
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
        Self::from_numpy_borrow(py, data, mode, None, false)
    }

    /// PIL-compatible alias for :meth:`frombuffer`.
    ///
    /// PIL's :func:`Image.fromarray` is the standard way to wrap a numpy
    /// array — exposing the same name here means existing PIL-style code
    /// works against ``kornia_rs.image.Image`` without renames.
    #[staticmethod]
    #[pyo3(signature = (data, mode=None))]
    fn fromarray(py: Python<'_>, data: &Bound<'_, PyAny>, mode: Option<String>) -> PyResult<Self> {
        Self::from_numpy_borrow(py, data, mode, None, false)
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
            let arr = {
                #[cfg(feature = "turbojpeg")]
                {
                    match crate::io::jpegturbo::decode_image_jpegturbo(py, data, native_mode) {
                        Ok(a) => a,
                        Err(_) => crate::io::jpeg::decode_image_jpeg(py, data)?,
                    }
                }
                #[cfg(not(feature = "turbojpeg"))]
                crate::io::jpeg::decode_image_jpeg(py, data)?
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
    pub fn width(&self) -> usize {
        self.shape[1]
    }

    #[getter]
    pub fn height(&self) -> usize {
        self.shape[0]
    }

    #[getter]
    fn channels(&self) -> usize {
        self.shape[2]
    }

    #[getter]
    pub fn mode(&self) -> &str {
        &self.mode
    }

    /// Canonical format the Image was decoded from (e.g. ``"PNG"``, ``"JPEG"``,
    /// ``"TIFF"``, ``"WEBP"``). ``None`` for in-memory-constructed Images.
    #[getter]
    fn format(&self) -> Option<&str> {
        self.format
    }

    /// The color space this Image is interpreted as.
    #[getter]
    fn color_space(&self) -> crate::color_space::PyColorSpace {
        self.color_space.into()
    }

    #[getter]
    fn size(&self) -> (usize, usize) {
        (self.shape[1], self.shape[0])
    }

    #[getter]
    fn shape(&self) -> (usize, usize, usize) {
        (self.shape[0], self.shape[1], self.shape[2])
    }

    #[getter]
    fn dtype(&self, py: Python<'_>) -> Py<PyAny> {
        self.dtype_obj(py)
    }

    /// The underlying numpy array. For a **host** image this is a zero-copy view
    /// sharing memory with the backing (the original ndarray for a borrowed-numpy
    /// Image; otherwise a fresh view whose base is this Image, kept alive). For a
    /// **device** image it is a read-only host copy (D2H) — it does not share the
    /// device buffer, so writes would be lost; use `.cpu()` for a writable host
    /// image.
    #[getter]
    pub fn data(slf: Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        Self::numpy_view_of(slf)
    }

    #[getter]
    fn nbytes(&self) -> usize {
        self.nbytes_total()
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
        self.backing.ensure_host()?;
        Ok(pyo3::types::PyBytes::new(py, self.raw_bytes()))
    }

    /// Return a copy of the underlying buffer as a fresh numpy array (owns its
    /// own memory, independent of this Image).
    #[pyo3(signature = (copy=true))]
    fn to_numpy(slf: Bound<'_, Self>, copy: bool) -> PyResult<Py<PyAny>> {
        let view = Self::numpy_view_of(slf.clone())?;
        if copy {
            let py = slf.py();
            Ok(view.bind(py).call_method0("copy")?.unbind())
        } else {
            Ok(view)
        }
    }

    /// Return a deep copy of this image (owned, independent storage).
    pub fn copy(&self, py: Python<'_>) -> PyResult<Self> {
        self.backing.ensure_host()?;
        Ok(self.clone_handle(py))
    }

    // --- Chainable transforms ---

    /// Resize image to (width, height). 8-bit only.
    ///
    /// `antialias=True` (default) matches PIL / torchvision semantics — the
    /// cubic/lanczos kernel is widened by the downscale factor to pre-filter
    /// aliasing. `antialias=False` matches OpenCV `INTER_CUBIC` /
    /// `INTER_LANCZOS4` (fixed kernel, faster at strong downscale, no AA).
    /// `Nearest` and `Bilinear` are unaffected by the flag.
    #[pyo3(signature = (width, height, interpolation="bilinear", antialias=true))]
    fn resize(
        &self,
        py: Python<'_>,
        width: usize,
        height: usize,
        interpolation: &str,
        antialias: bool,
    ) -> PyResult<Self> {
        self.backing.ensure_host()?;
        self.require_u8("resize")?;
        let [src_h, src_w, c] = self.shape;
        if c == 3 {
            // One-copy path: borrow src zero-copy, allocate dst once, write
            // directly.  The old path went src → PyArray → owned, wasting a copy.
            let src = unsafe { self.borrow_self::<u8, 3>().map_err(to_pyerr)? };
            let interp = parse_interpolation(interpolation)?;
            let out_size = ImageSize { width, height };
            self.run_into_owned_u8::<3, _>(py, out_size, |dst| {
                kornia_imgproc::resize::resize_fast_rgb_aa(&src, dst, interp, antialias)
            })
        } else {
            let out = resize_nearest(self.u8_elems(), src_h, src_w, height, width, c);
            Ok(self.wrap_vec(py, out, height, width, c))
        }
    }

    /// Fused **resize + per-channel normalize + HWC→CHW** into an owned float32
    /// tensor `Image`, for **any** target size. General bilinear (NEON/AVX2
    /// vectorized, single pass); exact 2× downscale takes a faster fused box path.
    ///
    /// The whole path is copy-free: the input numpy buffer is borrowed in place,
    /// the `(3, height, width)` output is allocated once and written directly, and
    /// the returned `Image` wraps that same buffer. Use `.numpy()` for a zero-copy
    /// view and `.data_ptr` for the host address to hand to an external library.
    ///
    /// `mean`/`std` are per-channel in `[0, 1]` (PyTorch convention); the output is
    /// `(x/255 − mean) / std`. Input must be `(H, W, 3)` uint8, C-contiguous.
    #[pyo3(signature = (width, height, mean, std))]
    fn resize_normalize_to_tensor(
        &self,
        py: Python<'_>,
        width: usize,
        height: usize,
        mean: [f32; 3],
        std: [f32; 3],
    ) -> PyResult<Self> {
        self.backing.ensure_host()?;
        self.require_u8("resize_normalize_to_tensor")?;
        let [src_h, src_w, c] = self.shape;
        if c != 3 {
            return Err(value_err(format!("expected 3 channels (RGB), got {c}")));
        }
        if width == 0 || height == 0 {
            return Err(value_err("target width and height must be > 0"));
        }
        // Zero-copy borrow of the HWC u8 input.
        let src = self.u8_elems();
        let params = kornia_imgproc::resize::NormalizeParams::<3>::from_mean_std(mean, std);

        // Single owned output allocation (CHW), written in place.
        let n = 3 * height * width;
        let mut bytes = backing::AlignedBytes::zeroed(n * 4);
        let out_slice =
            unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut f32, n) };
        let result = py.detach(|| {
            kornia_imgproc::resize::resize_normalize_to_tensor_u8_to_f32_bilinear(
                src, src_w, src_h, out_slice, width, height, &params,
            )
        });
        result.map_err(to_pyerr)?;
        // CHW tensor stored as shape [3, H, W].
        Ok(Self::from_owned_bytes(
            bytes,
            backing::Dtype::F32,
            [3, height, width],
            self.color_space,
            "RGBf".to_string(),
        ))
    }

    /// Address (as an int) of the underlying contiguous data buffer.
    ///
    /// For a host image this is a host pointer; for a device (e.g. DLPack-imported CUDA)
    /// image it is the **device** pointer in that device's address space — check
    /// `__dlpack_device__()` to learn which. Stable for the lifetime of this `Image`; hand
    /// it (with `.shape` / `.nbytes`) to an external library without going through numpy.
    #[getter]
    fn data_ptr(&self) -> usize {
        self.backing.data_ptr() as usize
    }

    /// Numpy array of the underlying buffer. A host image shares memory
    /// (zero-copy view). A device (CUDA) image is copied back to host (D2H) and
    /// returned **read-only** — writes would land on the throwaway copy, not the
    /// device buffer; use `.cpu()` for a writable host image.
    /// Distinct from `to_numpy()`, which returns an owned deep copy.
    fn numpy(slf: Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        Self::numpy_view_of(slf)
    }

    /// Flip image horizontally. Supports 8-bit, 16-bit, and float32 Images.
    pub fn flip_horizontal(&self, py: Python<'_>) -> PyResult<Self> {
        self.backing.ensure_host()?;
        let [h, w, c] = self.shape;
        if self.dtype == backing::Dtype::U8 && c == 3 {
            let src = unsafe { self.borrow_self::<u8, 3>().map_err(to_pyerr)? };
            return self.run_into_owned_u8::<3, _>(py, src.size(), |dst| {
                kornia_imgproc::flip::horizontal_flip(&src, dst)
            });
        }
        if self.dtype == backing::Dtype::U8 {
            let out = flip_h_generic(self.u8_elems(), h, w, c);
            return Ok(self.wrap_vec(py, out, h, w, c));
        }
        Ok(self.flip_pod(FlipDir::Horizontal))
    }

    /// Flip image vertically. Supports 8-bit, 16-bit, and float32 Images.
    pub fn flip_vertical(&self, py: Python<'_>) -> PyResult<Self> {
        self.backing.ensure_host()?;
        let [h, w, c] = self.shape;
        if self.dtype == backing::Dtype::U8 && c == 3 {
            let src = unsafe { self.borrow_self::<u8, 3>().map_err(to_pyerr)? };
            return self.run_into_owned_u8::<3, _>(py, src.size(), |dst| {
                kornia_imgproc::flip::vertical_flip(&src, dst)
            });
        }
        if self.dtype == backing::Dtype::U8 {
            let out = flip_v_generic(self.u8_elems(), h, w, c);
            return Ok(self.wrap_vec(py, out, h, w, c));
        }
        Ok(self.flip_pod(FlipDir::Vertical))
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
        self.backing.ensure_host()?;
        self.require_u8("gaussian_blur")?;
        if self.nchannels() == 3 {
            let src = unsafe { self.borrow_self::<u8, 3>().map_err(to_pyerr)? };
            self.run_into_owned_u8::<3, _>(py, src.size(), |dst| {
                kornia_imgproc::filter::gaussian_blur_u8(
                    &src,
                    dst,
                    (kernel_size, kernel_size),
                    (sigma, sigma),
                )
            })
        } else {
            self.copy(py)
        }
    }

    /// Apply box blur. 8-bit only.
    #[pyo3(signature = (kernel_size=3))]
    fn box_blur(&self, py: Python<'_>, kernel_size: usize) -> PyResult<Self> {
        self.backing.ensure_host()?;
        self.require_u8("box_blur")?;
        if self.nchannels() == 3 {
            let src = unsafe { self.borrow_self::<u8, 3>().map_err(to_pyerr)? };
            self.run_into_owned_u8::<3, _>(py, src.size(), |dst| {
                kornia_imgproc::filter::box_blur_u8(&src, dst, (kernel_size, kernel_size))
            })
        } else {
            self.copy(py)
        }
    }

    /// Adjust brightness. Factor is additive in [0,1] range. 8-bit only.
    pub fn adjust_brightness(&self, py: Python<'_>, factor: f32) -> PyResult<Self> {
        self.backing.ensure_host()?;
        self.require_u8("adjust_brightness")?;
        let [h, w, c] = self.shape;
        Ok(self.wrap_u8_result(
            py,
            adjust_brightness_into_pyarray(py, self.u8_elems(), factor * 255.0, h, w, c),
        ))
    }

    /// Adjust contrast. factor=1.0 is identity, >1 increases contrast. 8-bit only.
    pub fn adjust_contrast(&self, py: Python<'_>, factor: f64) -> PyResult<Self> {
        self.backing.ensure_host()?;
        self.require_u8("adjust_contrast")?;
        let [h, w, c] = self.shape;
        Ok(self.wrap_u8_result(
            py,
            adjust_contrast_into_pyarray(py, self.u8_elems(), factor, h, w, c),
        ))
    }

    /// Adjust saturation. factor=1.0 is identity, 0.0 is grayscale. 8-bit only.
    pub fn adjust_saturation(&self, py: Python<'_>, factor: f64) -> PyResult<Self> {
        self.backing.ensure_host()?;
        self.require_u8("adjust_saturation")?;
        let [h, w, c] = self.shape;
        if c != 3 {
            return self.copy(py);
        }
        Ok(self.wrap_u8_result(
            py,
            adjust_saturation_into_pyarray(py, self.u8_elems(), h * w, factor as f32, h, w),
        ))
    }

    /// Adjust hue. factor is in [-0.5, 0.5], fraction of hue wheel. 8-bit only.
    pub fn adjust_hue(&self, py: Python<'_>, factor: f64) -> PyResult<Self> {
        self.backing.ensure_host()?;
        self.require_u8("adjust_hue")?;
        let [h, w, c] = self.shape;
        if c != 3 || factor == 0.0 {
            return self.copy(py);
        }
        Ok(self.wrap_u8_result(
            py,
            adjust_hue_into_pyarray(py, self.u8_elems(), h * w, factor as f32, h, w),
        ))
    }

    /// Normalize image to float32 using mean and std per channel. 8-bit only.
    fn normalize(
        slf: Bound<'_, Self>,
        py: Python<'_>,
        mean: (f32, f32, f32),
        std: (f32, f32, f32),
    ) -> PyResult<Py<PyAny>> {
        let me = slf.borrow();
        me.backing.ensure_host()?;
        me.require_u8("normalize")?;
        let [h, w, c] = me.shape;
        let src = me.u8_elems();
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
        Ok(out.unbind().into_any())
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
        self.backing.ensure_host()?;
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
                self.require_u8("convert")?;
                let [h, w, _] = self.shape;
                let src = self.u8_elems();
                let mut bytes = backing::AlignedBytes::zeroed(h * w * 4);
                let dst = unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr(), h * w * 4) };
                for (i, px) in src.chunks_exact(3).enumerate() {
                    let o = i * 4;
                    dst[o] = px[0];
                    dst[o + 1] = px[1];
                    dst[o + 2] = px[2];
                    dst[o + 3] = 255;
                }
                Ok(Self::from_owned_bytes(
                    bytes,
                    backing::Dtype::U8,
                    [h, w, 4],
                    default_color_space(4),
                    "RGBA".to_string(),
                ))
            }
            ("I;16", "L") => convert_u16_to_u8(self, 1, "L".to_string()),
            ("RGB;16", "RGB") => convert_u16_to_u8(self, 3, "RGB".to_string()),
            ("RGBA;16", "RGBA") => convert_u16_to_u8(self, 4, "RGBA".to_string()),
            ("L", "I;16") => convert_u8_to_u16(self, 1, "I;16".to_string()),
            ("RGB", "RGB;16") => convert_u8_to_u16(self, 3, "RGB;16".to_string()),
            ("RGBA", "RGBA;16") => convert_u8_to_u16(self, 4, "RGBA;16".to_string()),
            _ => Err(value_err(format!(
                "convert: {:?} -> {:?} is not supported",
                self.mode, mode
            ))),
        }
    }

    /// Convert RGB image to grayscale (1 channel). 8-bit only.
    fn to_grayscale(&self, py: Python<'_>) -> PyResult<Self> {
        self.backing.ensure_host()?;
        self.require_u8("to_grayscale")?;
        let [h, w, c] = self.shape;
        if c == 1 {
            return self.copy(py);
        }
        if c != 3 {
            return Err(value_err(format!(
                "Cannot convert {}-channel image to grayscale",
                c
            )));
        }
        let size = ImageSize {
            width: w,
            height: h,
        };
        let src = unsafe { self.borrow_self::<u8, 3>().map_err(to_pyerr)? };
        let mut out = self.run_into_owned_u8::<1, _>(py, size, |dst| {
            kornia_imgproc::color::gray_from_rgb_u8(&src, dst)
        })?;
        out.mode = "L".to_string();
        out.color_space = default_color_space(1);
        Ok(out)
    }

    /// Convert grayscale to RGB (3 channels). 8-bit only.
    fn to_rgb(&self, py: Python<'_>) -> PyResult<Self> {
        self.backing.ensure_host()?;
        self.require_u8("to_rgb")?;
        let c = self.nchannels();
        if c == 3 {
            return self.copy(py);
        }
        let arr = self.as_numpy_u8(py)?;
        if c == 1 || c == 4 {
            let result = if c == 1 {
                crate::color::rgb_from_gray(py, arr.bind(py).as_any())?
            } else {
                crate::color::rgb_from_rgba(py, arr.bind(py).as_any(), None)?
            };
            let result: PyImage = result.bind(py).extract().map_err(to_pyerr)?;
            let mut out = self.owned_from_numpy_u8(py, result, Some("RGB".to_string()));
            out.color_space = default_color_space(3);
            Ok(out)
        } else {
            Err(value_err(format!(
                "Cannot convert {}-channel image to RGB",
                c
            )))
        }
    }

    // --- Color space conversion ---

    /// Cast u8 -> f32 by dividing by 255 (range [0,1]). f32 input is returned
    /// unchanged (cloned handle). Required before converting to f32-only spaces
    /// (HSV, Lab, Luv, XYZ, LinearRgb, YCbCr, Yuv).
    fn to_float(&self, py: Python<'_>) -> PyResult<Self> {
        self.backing.ensure_host()?;
        match self.dtype {
            backing::Dtype::F32 => Ok(self.clone_handle(py)),
            backing::Dtype::U8 => {
                let [h, w, c] = self.shape;
                let src = self.u8_elems();
                let n = h * w * c;
                let mut bytes = backing::AlignedBytes::zeroed(n * 4);
                let dst =
                    unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut f32, n) };
                for (d, &s) in dst.iter_mut().zip(src.iter()) {
                    *d = s as f32 / 255.0;
                }
                Ok(Self::from_owned_bytes(
                    bytes,
                    backing::Dtype::F32,
                    [h, w, c],
                    self.color_space,
                    mode_from_channels_f32(c),
                ))
            }
            backing::Dtype::U16 => Err(u16_imgproc_unsupported("to_float")),
        }
    }

    /// Cast f32 [0,1] -> u8 by multiplying by 255 (saturating round). u8 input
    /// is returned unchanged (cloned handle).
    fn to_uint8(&self, py: Python<'_>) -> PyResult<Self> {
        self.backing.ensure_host()?;
        match self.dtype {
            backing::Dtype::U8 => Ok(self.clone_handle(py)),
            backing::Dtype::F32 => {
                let [h, w, c] = self.shape;
                let n = h * w * c;
                let src =
                    unsafe { std::slice::from_raw_parts(self.backing.data_ptr() as *const f32, n) };
                let mut bytes = backing::AlignedBytes::zeroed(n);
                let dst = unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr(), n) };
                for (d, &s) in dst.iter_mut().zip(src.iter()) {
                    *d = (s * 255.0).round().clamp(0.0, 255.0) as u8;
                }
                Ok(Self::from_owned_bytes(
                    bytes,
                    backing::Dtype::U8,
                    [h, w, c],
                    self.color_space,
                    mode_from_channels(c, false),
                ))
            }
            backing::Dtype::U16 => Err(u16_imgproc_unsupported("to_uint8")),
        }
    }

    /// Convert to another color space, returning a new tagged Image. Strict
    /// dtype: f32-only spaces require a float image (call `to_float()` first).
    fn cvt_color(&self, py: Python<'_>, to: crate::color_space::PyColorSpace) -> PyResult<Self> {
        self.backing.ensure_host()?;
        use kornia_image::ColorSpace as CS;
        let from = self.color_space;
        let to: CS = to.into();
        if from == to {
            return Ok(self.clone_handle(py));
        }
        if !CS::supports(from, to) {
            return Err(value_err(
                ImageError::UnsupportedColorConversion { from, to }.to_string(),
            ));
        }
        // strict dtype: f32-only target (or source) needs f32 storage
        let needs_f32 = to.requires_f32() || from.requires_f32();
        if needs_f32 && !self.is_f32() {
            return Err(value_err(format!(
                "{to:?} requires float32; call img.to_float() first"
            )));
        }
        if self.is_u16() {
            return Err(u16_imgproc_unsupported("cvt_color"));
        }
        let mut img = dispatch_cvt(py, self, from, to)?;
        img.color_space = to;
        Ok(img)
    }

    /// Convert to grayscale. 8-bit: u8 output. Alias for
    /// `cvt_color(ColorSpace.Gray)`.
    fn to_gray(&self, py: Python<'_>) -> PyResult<Self> {
        self.cvt_color(py, crate::color_space::PyColorSpace::Gray)
    }

    /// Convert to HSV (requires float input — call `to_float()` first).
    fn to_hsv(&self, py: Python<'_>) -> PyResult<Self> {
        self.cvt_color(py, crate::color_space::PyColorSpace::Hsv)
    }

    /// Convert to CIE L*a*b* (requires float input — call `to_float()` first).
    fn to_lab(&self, py: Python<'_>) -> PyResult<Self> {
        self.cvt_color(py, crate::color_space::PyColorSpace::Lab)
    }

    /// Convert to BGR (channel-swap of RGB). Works on u8 or f32 storage.
    fn to_bgr(&self, py: Python<'_>) -> PyResult<Self> {
        self.cvt_color(py, crate::color_space::PyColorSpace::Bgr)
    }

    /// Apply a colormap to a single-channel (L) image, producing an RGB image.
    ///
    /// Accepts any of the 21 OpenCV colormap names (case-insensitive):
    /// ``autumn``, ``bone``, ``jet``, ``winter``, ``rainbow``, ``ocean``,
    /// ``summer``, ``spring``, ``cool``, ``hsv``, ``pink``, ``hot``,
    /// ``parula``, ``magma``, ``inferno``, ``plasma``, ``viridis``,
    /// ``cividis``, ``twilight``, ``turbo``, ``deepgreen``.
    ///
    /// On aarch64 the LUT lookup is NEON-accelerated (16 px/iter).
    fn colormap(&self, py: Python<'_>, colormap: &str) -> PyResult<Self> {
        self.backing.ensure_host()?;
        self.require_u8("colormap")?;
        let c = self.nchannels();
        if c != 1 {
            return Err(value_err(format!(
                "colormap() requires a single-channel image, got {} channels",
                c
            )));
        }
        let arr = self.as_numpy_u8(py)?;
        let result = crate::color::apply_colormap(py, arr.bind(py).as_any(), colormap)?;
        let result: PyImage = result.bind(py).extract().map_err(to_pyerr)?;
        let mut out = self.owned_from_numpy_u8(py, result, Some("RGB".to_string()));
        out.color_space = default_color_space(3);
        Ok(out)
    }

    /// Rotate image by angle degrees (counter-clockwise). 8-bit only.
    ///
    /// Fast paths (exact k·90° only, no epsilon):
    ///  - 0°: copy
    ///  - 180°: buffer-reversal (always; any H, W, C)
    ///  - ±90°/±270° with H == W: transpose+flip (shape preserved)
    /// Non-exact or non-square 90°/270°: general bilinear warp.
    pub fn rotate(&self, py: Python<'_>, angle: f64) -> PyResult<Self> {
        self.backing.ensure_host()?;
        self.require_u8("rotate")?;
        let [h, w, c] = self.shape;

        if let Some(k) = exact_k90(angle) {
            match k {
                0 => return self.copy(py),
                2 => {
                    let (out, _, _) = rot90_generic(self.u8_elems(), h, w, c, 2);
                    return Ok(self.wrap_vec(py, out, h, w, c));
                }
                1 | 3 if h == w => {
                    let (out, nh, nw) = rot90_generic(self.u8_elems(), h, w, c, k as i32);
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
        let arr = self.as_numpy_u8(py)?;
        let result = crate::warp::warp_affine(py, arr, m, (h, w), "bilinear", None)?;
        Ok(self.wrap_u8_result(py, result))
    }

    // --- Serialization for multiprocess (Ray Data, etc.) ---

    /// Return `(Image, (arr, mode, color_space))` so that `constructor is Image`
    /// and color_space survives pickle / multiprocessing / Ray Data.
    #[allow(clippy::type_complexity)]
    fn __reduce__(
        slf: Bound<'_, Self>,
    ) -> PyResult<(
        Py<PyAny>,
        (Py<PyAny>, Option<String>, crate::color_space::PyColorSpace),
    )> {
        let py = slf.py();
        let cls: Py<PyAny> = Self::type_object(py).into_any().unbind();
        // Pass an owned-copy numpy array so reconstruction is independent of any
        // borrow keep-alive (pickle serializes by value regardless).
        let arr = Self::numpy_view_of(slf.clone())?;
        let arr = arr.bind(py).call_method0("copy")?.unbind();
        let me = slf.borrow();
        let mode = Some(me.mode.clone());
        let cs: crate::color_space::PyColorSpace = me.color_space.into();
        Ok((cls, (arr, mode, cs)))
    }

    // --- Dunder methods ---

    fn __repr__(&self) -> String {
        format!(
            "Image(mode={}, size={}x{}, dtype={})",
            self.mode,
            self.shape[1],
            self.shape[0],
            self.dtype_name()
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        if self.mode != other.mode
            || self.dtype != other.dtype
            || self.shape != other.shape
            || self.color_space != other.color_space
        {
            return false;
        }
        // Device images cannot be byte-compared on the host (dereferencing device
        // memory is UB). Equal only if they alias the same device buffer.
        if !self.backing.is_host() || !other.backing.is_host() {
            return self.backing.device() == other.backing.device()
                && self.backing.data_ptr() == other.backing.data_ptr();
        }
        self.raw_bytes() == other.raw_bytes()
    }

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        // Device images cannot be byte-hashed on the host; hash the device +
        // buffer address instead of dereferencing device memory.
        if self.backing.is_host() {
            self.raw_bytes().hash(&mut h);
        } else {
            self.backing.device().hash(&mut h);
            (self.backing.data_ptr() as usize).hash(&mut h);
        }
        self.mode.hash(&mut h);
        (self.dtype as u8).hash(&mut h);
        self.shape.hash(&mut h);
        format!("{:?}", self.color_space).hash(&mut h);
        h.finish()
    }

    #[pyo3(signature = (dtype=None, copy=None))]
    fn __array__(
        slf: Bound<'_, Self>,
        py: Python<'_>,
        dtype: Option<&str>,
        copy: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let view = Self::numpy_view_of(slf)?;
        let bound = view.bind(py);
        if let Some(dt) = dtype {
            Ok(bound.call_method1("astype", (dt,))?.unbind())
        } else if copy.unwrap_or(false) {
            Ok(bound.call_method0("copy")?.unbind())
        } else {
            Ok(view)
        }
    }

    fn __len__(&self) -> usize {
        self.shape[0]
    }

    /// PEP 3118 buffer protocol — fills the `Py_buffer` directly from the
    /// backing buffer (no numpy round-trip). `obj` holds a strong ref to this
    /// Image so the buffer outlives the memoryview; shape/strides are heap
    /// boxed and freed in `__releasebuffer__`.
    unsafe fn __getbuffer__(
        slf: pyo3::PyRefMut<'_, Self>,
        view: *mut pyo3::ffi::Py_buffer,
        flags: std::os::raw::c_int,
    ) -> PyResult<()> {
        if view.is_null() {
            return Err(pyo3::exceptions::PyBufferError::new_err("null view"));
        }
        // Host guard: the PEP-3118 buffer exposes the backing pointer for host
        // access; refuse to hand out a device pointer as a host buffer.
        slf.backing.ensure_host()?;
        let readonly = slf.backing.readonly();
        if (flags & pyo3::ffi::PyBUF_WRITABLE) == pyo3::ffi::PyBUF_WRITABLE && readonly {
            return Err(pyo3::exceptions::PyBufferError::new_err(
                "Image buffer is read-only",
            ));
        }
        let [h, w, c] = slf.shape;
        let itemsize = slf.dtype.itemsize() as pyo3::ffi::Py_ssize_t;
        let fmt: &'static [u8] = match slf.dtype {
            backing::Dtype::U8 => b"B\0",
            backing::Dtype::U16 => b"H\0",
            backing::Dtype::F32 => b"f\0",
        };
        // Heap-boxed shape/strides arrays (3 dims).
        let shape_box: Box<[pyo3::ffi::Py_ssize_t; 3]> = Box::new([h as _, w as _, c as _]);
        let strides_box: Box<[pyo3::ffi::Py_ssize_t; 3]> = Box::new([
            (w * c) as pyo3::ffi::Py_ssize_t * itemsize,
            c as pyo3::ffi::Py_ssize_t * itemsize,
            itemsize,
        ]);

        let total_bytes = backing::byte_len(h, w, c, slf.dtype)
            .map_err(|e| pyo3::exceptions::PyBufferError::new_err(e.to_string()))?;
        let v = unsafe { &mut *view };
        v.buf = slf.backing.data_ptr() as *mut std::ffi::c_void;
        v.len = total_bytes as pyo3::ffi::Py_ssize_t;
        v.readonly = readonly as std::os::raw::c_int;
        v.itemsize = itemsize;
        v.format = if (flags & pyo3::ffi::PyBUF_FORMAT) == pyo3::ffi::PyBUF_FORMAT {
            fmt.as_ptr() as *mut std::os::raw::c_char
        } else {
            std::ptr::null_mut()
        };
        v.ndim = 3;
        v.shape = Box::into_raw(shape_box) as *mut pyo3::ffi::Py_ssize_t;
        v.strides = Box::into_raw(strides_box) as *mut pyo3::ffi::Py_ssize_t;
        v.suboffsets = std::ptr::null_mut();
        v.internal = std::ptr::null_mut();
        // obj holds a strong ref to the Image so the buffer outlives the view.
        let obj_ptr = slf.as_ptr();
        unsafe { pyo3::ffi::Py_INCREF(obj_ptr) };
        v.obj = obj_ptr;
        Ok(())
    }

    unsafe fn __releasebuffer__(&self, view: *mut pyo3::ffi::Py_buffer) {
        if view.is_null() {
            return;
        }
        let v = unsafe { &mut *view };
        if !v.shape.is_null() {
            let _ = unsafe { Box::from_raw(v.shape as *mut [pyo3::ffi::Py_ssize_t; 3]) };
            v.shape = std::ptr::null_mut();
        }
        if !v.strides.is_null() {
            let _ = unsafe { Box::from_raw(v.strides as *mut [pyo3::ffi::Py_ssize_t; 3]) };
            v.strides = std::ptr::null_mut();
        }
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

    // ─── DLPack protocol ───────────────────────────────────────────────────────

    /// Export this image as a DLPack capsule (zero-copy).
    ///
    /// Implements the DLPack protocol used by NumPy 2.x, PyTorch, etc.
    ///
    /// - If `max_version` is `(1, 0)` or higher (NumPy 2.x, modern consumers),
    ///   returns a **versioned** capsule (`"dltensor_versioned"`) with `flags = 0`
    ///   so consumers treat the tensor as mutable (not read-only).
    /// - Otherwise falls back to the **legacy** capsule (`"dltensor"`).  Note
    ///   that some consumers (NumPy 2.x) will mark legacy tensors read-only.
    ///
    /// The returned capsule points directly into the Image's backing buffer.
    /// A `Py<PyAny>` handle to the Image is stored in the `ImageExport`
    /// keep-alive, keeping the buffer alive until the consumer's deleter runs.
    ///
    /// Zero-copy pass-through export: the tensor carries the image's own device
    /// (CPU or, for a DLPack-imported device image, e.g. CUDA).
    ///
    /// Keyword arguments:
    /// - `stream`: the consumer's CUDA stream (DLPack convention). For a device
    ///   image the producing stream is ordered *before* it without blocking the
    ///   host — a record-event + `cuStreamWaitEvent` fence. `-1` means "no
    ///   synchronization"; `None` (or an image with no carried stream) falls back
    ///   to a full host sync so the consumer always sees fully-written pixels.
    /// - `dl_device`: a cross-device request is rejected; only an explicit
    ///   request matching the image's own device is honoured.
    /// - `copy`: `copy=True` is not supported.
    #[pyo3(signature = (*, stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__(
        slf: Bound<'_, Self>,
        py: Python<'_>,
        stream: Option<Py<PyAny>>,
        max_version: Option<(u32, u32)>,
        dl_device: Option<Py<PyAny>>,
        copy: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        // Device export: order the producing stream before the consumer's so it
        // sees fully-written pixels. A concrete consumer stream is fenced into
        // without blocking the host (record-event + wait); `-1` skips sync at the
        // consumer's request; `None` (or no carried stream) does a full host sync.
        #[cfg(feature = "cuda")]
        if let Some(dev) = slf.borrow().as_device() {
            let consumer: Option<isize> = match &stream {
                Some(obj) => Some(obj.bind(py).extract::<isize>()?),
                None => None,
            };
            // Same consumer-stream policy as CudaTensor::__dlpack__ (validates the
            // handle, fences without a host block, -1 skips, None host-syncs).
            match dev.cuda_stream() {
                Some(s) => crate::cuda_ext::dlpack_fence_consumer(s, consumer)?,
                // No carried stream: nothing to fence; honor an explicit -1 opt-out.
                None if consumer != Some(-1) => dev.synchronize()?,
                None => {}
            }
        }
        #[cfg(not(feature = "cuda"))]
        let _ = stream;
        let own_device = slf.borrow().backing.device();
        // Validate dl_device: only a request matching the image's own device is
        // honoured. Cross-device export would require a host/device copy, which
        // this zero-copy path does not perform.
        if let Some(ref dev) = dl_device {
            let (dev_type, dev_id): (i32, i32) = dev.extract(py)?;
            if (dev_type, dev_id) != own_device {
                return Err(pyo3::exceptions::PyBufferError::new_err(format!(
                    "__dlpack__: cross-device export not supported; image is on \
                     device ({}, {}) but ({dev_type}, {dev_id}) was requested",
                    own_device.0, own_device.1
                )));
            }
        }
        // copy=True is not yet supported; explicitly reject it.
        if copy.unwrap_or(false) {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "__dlpack__: copy=True is not yet supported; call img.copy() first",
            ));
        }
        let (data, shape, dt, readonly) = {
            let s = slf.borrow();
            let dt = crate::dlpack::dtype_to_dl(s.dtype);
            let data = s.backing.data_ptr() as *mut std::ffi::c_void;
            let shape: Vec<i64> = s.shape.iter().map(|&d| d as i64).collect();
            let readonly = s.backing.readonly();
            (data, shape, dt, readonly)
        };
        // Build the export with `slf` as the keep-alive.
        // The `ImageExport` owns a `Py<PyAny>` handle that keeps the Image alive
        // so the backing buffer outlives the exported DLPack tensor.
        let keepalive: Py<PyAny> = slf.into_any().unbind();
        let export = crate::dlpack::ImageExport {
            keepalive: std::mem::ManuallyDrop::new(keepalive),
            data,
            shape,
            dtype: dt,
            device: own_device,
        };
        use dlpack_rs::pyo3_glue::IntoDLPack;
        // If the consumer supports DLPack v1.0+, return a versioned capsule.
        // Set DLPACK_FLAG_BITMASK_READ_ONLY when the backing is read-only so that
        // compliant consumers (NumPy 2.x, PyTorch ≥2) honour the immutability.
        let flags: u64 = if readonly {
            dlpack_rs::ffi::DLPACK_FLAG_BITMASK_READ_ONLY
        } else {
            0
        };
        let capsule = if max_version.is_some_and(|(maj, _)| maj >= 1) {
            export.into_capsule_versioned(py, flags)?
        } else {
            export.into_capsule(py)?
        };
        Ok(capsule.into_any().unbind())
    }

    /// Return the DLPack device tuple `(device_type, device_id)` for this image.
    ///
    /// Host images report `(kDLCPU=1, 0)`; a DLPack-imported device image reports
    /// its source device (e.g. `(kDLCUDA=2, id)`).
    fn __dlpack_device__(&self) -> (i32, i32) {
        self.backing.device()
    }

    /// Zero-copy ingest of a PEP-3118 buffer (ROS2 bytearray / memoryview).
    ///
    /// Borrows the underlying buffer without copying: the `Image` keeps `data`
    /// alive (via a `BorrowGuard::PyObject` that also releases the `Py_buffer`
    /// view on drop).  Mutations to the source are immediately visible via the
    /// `Image`, and vice-versa.
    ///
    /// Args:
    ///     data: any object that exposes a PEP-3118 buffer (``bytearray``,
    ///         ``memoryview``, ``bytes``, NumPy array contiguous bytes, etc.).
    ///     width: image width in pixels.
    ///     height: image height in pixels.
    ///     channels: number of channels (default 3).
    ///     dtype: ``"uint8"`` (default), ``"uint16"``, or ``"float32"``.
    ///     mode: PIL-style mode string; inferred when omitted.
    ///     color_space: :class:`ColorSpace` value; inferred from channels when omitted.
    ///
    /// Returns:
    ///     Image that borrows the caller's buffer (zero-copy).
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (data, width, height, channels=3, dtype="uint8", mode=None, color_space=None))]
    fn from_buffer(
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        width: usize,
        height: usize,
        channels: usize,
        dtype: &str,
        mode: Option<String>,
        color_space: Option<crate::color_space::PyColorSpace>,
    ) -> PyResult<Self> {
        let dt = backing::Dtype::from_numpy_str(dtype)?;
        // SAFETY: `view` is zero-initialised; `PyObject_GetBuffer` fills it on success.
        let mut view = Box::new(unsafe { std::mem::zeroed::<pyo3::ffi::Py_buffer>() });
        // Request a simple (flat, contiguous) buffer.  On success the caller owns
        // the view and must release it with `PyBuffer_Release` exactly once.
        let rc = unsafe {
            pyo3::ffi::PyObject_GetBuffer(data.as_ptr(), view.as_mut(), pyo3::ffi::PyBUF_SIMPLE)
        };
        if rc != 0 {
            return Err(PyErr::fetch(py));
        }
        let need = backing::byte_len(height, width, channels, dt)?;
        if (view.len as usize) < need {
            // Release the view before returning the error.
            unsafe { pyo3::ffi::PyBuffer_Release(view.as_mut()) };
            return Err(value_err(format!(
                "buffer too small: {} < {} ({}x{}x{} {})",
                view.len, need, height, width, channels, dtype
            )));
        }
        let ptr = std::ptr::NonNull::new(view.buf as *mut u8)
            .ok_or_else(|| value_err("from_buffer: null buffer pointer"))?;
        let readonly = view.readonly != 0;
        // BorrowGuard::PyObject Drop calls PyBuffer_Release exactly once (see backing.rs).
        let keep = backing::BorrowGuard::PyObject {
            obj: data.clone().unbind(),
            buffer: Some(view),
        };
        Ok(Self {
            backing: backing::Backing::Borrowed {
                ptr,
                keep,
                readonly,
                // PEP-3118 buffers are host memory.
                device: (dlpack_rs::ffi::K_DL_CPU as i32, 0),
            },
            dtype: dt,
            shape: [height, width, channels],
            color_space: color_space
                .map(Into::into)
                .unwrap_or_else(|| default_color_space(channels)),
            mode: mode.unwrap_or_else(|| mode_from_channels(channels, dt == backing::Dtype::U16)),
            format: None,
        })
    }

    /// Import a DLPack tensor as a zero-copy Image (static method).
    ///
    /// Accepts any Python object that implements `__dlpack__()`: numpy arrays
    /// (>= 1.22), PyTorch tensors (CPU or CUDA), CuPy arrays, etc.
    ///
    /// Device pass-through: CPU tensors import as host images; non-CPU tensors
    /// (e.g. CUDA) import as device images that carry their source device. The
    /// device buffer is never dereferenced on the host — host-only operations
    /// (numpy export, pixel compute, save) raise `ValueError` on a device image.
    /// A device image can be re-exported via `__dlpack__`, preserving its device.
    ///
    /// Validation:
    /// - Tensor must be C-contiguous; raises `ValueError` otherwise.
    /// - `ndim` must be 3 (`(H, W, C)`) or 2 (`(H, W)` → treated as `(H, W, 1)`).
    /// - dtype must be `uint8`, `uint16`, or `float32`; raises `ValueError` otherwise.
    /// - Channel count `C` must be in `{1, 3, 4}`; raises `ValueError` otherwise.
    #[staticmethod]
    #[pyo3(signature = (obj, copy = true))]
    fn from_dlpack(py: Python<'_>, obj: &Bound<'_, PyAny>, copy: bool) -> PyResult<Self> {
        use dlpack_rs::ffi::{DLManagedTensor, DLManagedTensorVersioned};
        use pyo3::types::{PyCapsule, PyCapsuleMethods};
        use std::ffi::CStr;
        // `copy` only affects the CUDA branch (host imports are always the
        // zero-copy producer-keepalive borrow below).
        let _ = copy;

        // Device inference: a CUDA source imports as a proper device-resident
        // `Image` (carries a stream + typed resource, so `.numpy()`/`.cpu()` and
        // GPU ops work), rather than a raw host-unusable pointer. Host tensors
        // fall through to the zero-copy host borrow below.
        #[cfg(feature = "cuda")]
        if let Ok(dev) = obj.call_method0("__dlpack_device__") {
            if let Ok((ty, _id)) = dev.extract::<(i32, i32)>() {
                if ty == dlpack_rs::ffi::DLDeviceType::kDLCUDA as i32 {
                    let arc = crate::cuda_ext::dlpack_to_device_arc(py, obj, copy)?;
                    let cs = default_color_space(arc.channels());
                    let mode = mode_for_dtype(arc.dtype_enum(), arc.channels());
                    // A zero-copy alias (copy=false) may reference a read-only producer.
                    return Ok(Self::from_device_arc(arc, !copy, cs, mode));
                }
            }
        }

        // DLPack import design: zero-copy via producer keep-alive + capsule consumption.
        //
        // We call `obj.__dlpack__()` to get a `PyCapsule` (named "dltensor" or
        // "dltensor_versioned"). We read the pointer and extract metadata, then
        // CONSUME (rename) the capsule to "used_dltensor" / "used_dltensor_versioned"
        // so the capsule's C destructor will NOT call the producer's deleter when
        // the capsule is later GC'd.
        //
        // Instead, `obj` (the original producer: numpy array, torch tensor, etc.)
        // is stored as our BorrowGuard::PyObject keep-alive.  While `obj` is alive,
        // the buffer is valid.  When our Image drops, `obj` is dropped (its Python
        // refcount decrements), and the GC eventually frees the producer and its buffer.
        //
        // Consuming the capsule prevents double-free: if the capsule were not renamed,
        // its C destructor would also call the producer's internal deleter (e.g.
        // decrement `at::Storage` refcount), which could free the buffer while `obj`
        // still holds it — undefined behaviour.  Renaming disables the destructor path
        // and transfers lifetime management to `obj` exclusively.

        // 1. Call `obj.__dlpack__(max_version=(1,0))` to get the capsule.
        //    Passing max_version lets compliant producers (NumPy ≥1.24, PyTorch ≥2.0) return
        //    the versioned "dltensor_versioned" capsule, which carries the read-only flag.
        //    Fall back to the no-arg call for pre-spec producers that reject the keyword.
        let capsule_obj = {
            let kwargs = pyo3::types::PyDict::new(py);
            kwargs.set_item("max_version", (1u32, 0u32))?;
            obj.call_method("__dlpack__", (), Some(&kwargs))
                .or_else(|e| {
                    if e.is_instance_of::<pyo3::exceptions::PyTypeError>(py) {
                        obj.call_method0("__dlpack__")
                    } else {
                        Err(e)
                    }
                })
                .map_err(|_| {
                    pyo3::exceptions::PyTypeError::new_err(
                        "from_dlpack: object does not implement __dlpack__()",
                    )
                })?
        };

        // 2. Downcast to PyCapsule.
        let capsule: pyo3::Bound<'_, PyCapsule> = capsule_obj.cast_into()?;

        // 3. Determine variant and read the DLTensor.
        const NAME_DL: &CStr = c"dltensor";
        const NAME_DLV: &CStr = c"dltensor_versioned";

        let cap_name = capsule.name()?;
        let name_cstr: &CStr = match &cap_name {
            Some(n) => unsafe { n.as_cstr() },
            None => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "from_dlpack: DLPack capsule has no name",
                ));
            }
        };

        // Extract the DLTensor reference (borrowed; valid while capsule is alive).
        let (device, ndim, raw_shape, raw_strides, dtype_raw, data_raw, byte_offset, readonly) =
            if name_cstr == NAME_DL {
                let nn = capsule.pointer_checked(Some(NAME_DL))?;
                let mt = unsafe { &*(nn.as_ptr() as *const DLManagedTensor) };
                let t = &mt.dl_tensor;
                // SECURITY: `ndim` is producer-controlled. Validate it (and the shape
                // pointer) BEFORE using it as a slice length — a negative `ndim` casts
                // to `usize::MAX` (instant UB) and an oversized one reads out of bounds.
                crate::dlpack::validate_dlpack_rank(t.ndim, t.shape)?;
                let sh = unsafe { std::slice::from_raw_parts(t.shape, t.ndim as usize) };
                let st = if t.strides.is_null() {
                    None
                } else {
                    Some(unsafe { std::slice::from_raw_parts(t.strides, t.ndim as usize) })
                };
                (
                    t.device,
                    t.ndim as usize,
                    sh,
                    st,
                    t.dtype,
                    t.data,
                    t.byte_offset,
                    false,
                )
            } else if name_cstr == NAME_DLV {
                let nn = capsule.pointer_checked(Some(NAME_DLV))?;
                let mt = unsafe { &*(nn.as_ptr() as *const DLManagedTensorVersioned) };
                let t = &mt.dl_tensor;
                // SECURITY: see note in the `NAME_DL` branch — validate before slicing.
                crate::dlpack::validate_dlpack_rank(t.ndim, t.shape)?;
                let sh = unsafe { std::slice::from_raw_parts(t.shape, t.ndim as usize) };
                let st = if t.strides.is_null() {
                    None
                } else {
                    Some(unsafe { std::slice::from_raw_parts(t.strides, t.ndim as usize) })
                };
                let ro = (mt.flags & dlpack_rs::ffi::DLPACK_FLAG_BITMASK_READ_ONLY) != 0;
                (
                    t.device,
                    t.ndim as usize,
                    sh,
                    st,
                    t.dtype,
                    t.data,
                    t.byte_offset,
                    ro,
                )
            } else {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "from_dlpack: unexpected capsule name {:?}; expected 'dltensor' or \
                     'dltensor_versioned'",
                    name_cstr
                )));
            };

        // 4. Capture the source device for zero-copy pass-through interop.
        //    CPU tensors stay host-accessible; non-CPU (e.g. CUDA) tensors are
        //    imported as device images: their buffer is never dereferenced on the
        //    host (host ops are gated by `Backing::ensure_host`), and `__dlpack__`
        //    re-exports them carrying this same device.
        let src_device = (device.device_type as i32, device.device_id);

        // 5. Validate contiguity (strides == None, or compact row-major).
        if let Some(strides) = raw_strides {
            let mut expected = 1i64;
            let ok = (0..ndim).rev().all(|i| {
                let ok = strides[i] == expected;
                expected *= raw_shape[i];
                ok
            });
            if !ok {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "from_dlpack: tensor is not C-contiguous; call .contiguous() first",
                ));
            }
        }

        // 6. Resolve shape: ndim 2 → (H, W, 1), ndim 3 → (H, W, C).
        //    Reject non-positive dimensions before casting to usize — a negative i64
        //    wraps to usize::MAX and bypasses all subsequent size checks.
        for (i, &d) in raw_shape.iter().enumerate() {
            if d <= 0 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "from_dlpack: shape dimension {i} must be positive, got {d}"
                )));
            }
        }
        let shape_hwc = match ndim {
            2 => [raw_shape[0] as usize, raw_shape[1] as usize, 1],
            3 => [
                raw_shape[0] as usize,
                raw_shape[1] as usize,
                raw_shape[2] as usize,
            ],
            n => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "from_dlpack: expected 2D or 3D tensor, got {}D",
                    n
                )));
            }
        };
        let [h, w, c] = shape_hwc;

        // 7. Validate dtype.
        let dtype = crate::dlpack::dl_to_dtype(dtype_raw)?;

        // 8. Validate channel count.
        if !matches!(c, 1 | 3 | 4) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "from_dlpack: channel count must be 1, 3, or 4; got {}",
                c
            )));
        }

        // 8b. SECURITY: reject shapes whose host byte-extent would overflow `usize`.
        //     `Backing::Borrowed` stores no length; `raw_bytes`/`u8_elems` recompute it
        //     later with unchecked multiplication, so an overflowing product would wrap
        //     to a small length over a larger (or smaller) real buffer. Check it once here.
        h.checked_mul(w)
            .and_then(|hw| hw.checked_mul(c))
            .and_then(|hwc| hwc.checked_mul(dtype.itemsize()))
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "from_dlpack: image dimensions overflow usize",
                )
            })?;

        // 9. Compute effective data pointer (with byte_offset).
        //    Null check must happen BEFORE adding byte_offset: null+nonzero produces a
        //    non-null address that would pass the NonNull guard (same guard as kornia-tensor).
        if data_raw.is_null() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "from_dlpack: tensor has null data pointer",
            ));
        }
        let byte_offset_usize = usize::try_from(byte_offset).map_err(|_| {
            pyo3::exceptions::PyValueError::new_err(
                "from_dlpack: byte_offset does not fit in usize on this platform",
            )
        })?;
        let base = (data_raw as usize)
            .checked_add(byte_offset_usize)
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "from_dlpack: byte_offset overflow on data_ptr",
                )
            })? as *mut u8;

        let ptr = std::ptr::NonNull::new(base)
            .expect("from_dlpack: base pointer is null after adding byte_offset (unreachable)");

        // 10. Build the Image with the PRODUCER as keep-alive.
        //
        // We store `obj` (the original producer — a numpy array, torch tensor,
        // or other DLPack-capable object) as our keep-alive.  This keeps the
        // producer alive while our Image borrows its buffer.
        //
        // The capsule (which we obtained above from `obj.__dlpack__()`) is
        // consumed here by renaming it to "used_dltensor" so the capsule's
        // C destructor will NOT call the DLManagedTensor's deleter when the
        // capsule is GC'd.  The buffer lifetime is managed by `obj` directly:
        // as long as `obj` is alive, the buffer is valid.
        //
        // Safety rationale:
        // - For numpy arrays: numpy keeps the buffer alive as long as the array
        //   object exists. `obj` holds the array, so the buffer outlives us.
        // - For torch tensors: torch's `at::Storage` is refcounted. The tensor
        //   `obj` holds a reference to `at::Storage`; while `obj` is alive,
        //   `at::Storage` is alive, and the buffer is valid.
        // - For our own Image (bidirectional): `obj` = Image → `obj`'s backing
        //   keeps the buffer alive. The Image's `keepalive` (in `ImageExport`)
        //   is NOT involved here; the chain is `img2.keep → obj(=t) → t.storage
        //   → ImageExport → img` (implicit; mediated by torch's lifecycle).
        //
        // We consume the capsule (rename) to prevent double-free: the capsule's
        // C destructor would otherwise also call the deleter, which would
        // decrement the producer's internal refcount, possibly freeing the buffer
        // while `obj` still claims to own it.
        {
            use pyo3::ffi::PyCapsule_SetName;
            // Consumed name for the capsule (DLPack spec: "used_dltensor" / "used_dltensor_versioned")
            let consumed_name: &'static CStr = if name_cstr == c"dltensor_versioned" {
                c"used_dltensor_versioned"
            } else {
                c"used_dltensor"
            };
            // SAFETY: capsule is a valid PyCapsule (validated by cast_into) and the
            // new name is a valid C string with static lifetime.
            let ok = unsafe { PyCapsule_SetName(capsule.as_ptr(), consumed_name.as_ptr()) };
            if ok != 0 {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "from_dlpack: failed to consume DLPack capsule (PyCapsule_SetName failed)",
                ));
            }
        }

        let mode = mode_for_dtype(dtype, c);

        Ok(Self {
            backing: backing::Backing::Borrowed {
                ptr,
                keep: backing::BorrowGuard::PyObject {
                    obj: obj.clone().unbind(),
                    buffer: None,
                },
                readonly,
                device: src_device,
            },
            dtype,
            shape: [h, w, c],
            color_space: default_color_space(c),
            mode,
            format: None,
        })
    }
}

// ── Device (CUDA) surface ─────────────────────────────────────────────────────

/// Error raised when a CUDA operation is attempted on a build without the
/// `cuda` feature.
#[allow(dead_code)] // used only in `#[cfg(not(feature = "cuda"))]` arms
fn cuda_not_compiled() -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(
        "CUDA support is not compiled in (build kornia-rs with the 'cuda' feature)",
    )
}

/// A device-resident image (CUDA). Backs `Image.cuda.from_numpy(...)` etc.
/// A device `Image` shares the same `Image` class — only its `.device` differs.
#[cfg(feature = "cuda")]
impl PyImageApi {
    /// Upload a host image to CUDA device memory (H2D), or share the handle if it
    /// is already on device. Only `uint8` (1/3/4 ch) and `float32` (1/3 ch) are
    /// supported on device.
    pub(crate) fn to_device_internal(
        &self,
        py: Python<'_>,
        stream: std::sync::Arc<cudarc::driver::CudaStream>,
    ) -> PyResult<Self> {
        // Already on device: share the same buffer (cheap Arc clone).
        if let Some(arc) = self.device_arc() {
            return Ok(Self::from_device_arc(
                arc,
                self.backing.readonly(),
                self.color_space,
                self.mode.clone(),
            ));
        }
        self.backing.ensure_host()?;
        let backing = &self.backing;
        let shape = self.shape;
        let dtype = self.dtype;
        let channels = self.shape[2];
        let dev =
            py.detach(move || upload_device(backing, shape, dtype, channels, &stream))?;
        Ok(Self::from_device(dev, self.color_space, self.mode.clone()))
    }

    /// Copy a device image back to host (D2H), or return an owned host copy if it
    /// is already on host.
    pub(crate) fn to_host_internal(&self, py: Python<'_>) -> PyResult<Self> {
        match self.as_device() {
            Some(dev) => {
                let (bytes, shape, dtype) = dev.download_to_owned(py)?;
                Ok(Self::from_owned_bytes(
                    bytes,
                    dtype,
                    shape,
                    self.color_space,
                    self.mode.clone(),
                ))
            }
            None => {
                self.backing.ensure_host()?;
                Ok(self.clone_handle(py))
            }
        }
    }
}

/// Borrow a host backing as a typed `Image<T, C>` and upload it (H2D).
#[cfg(feature = "cuda")]
fn upload_device(
    backing: &backing::Backing,
    shape: [usize; 3],
    dtype: backing::Dtype,
    channels: usize,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
) -> PyResult<crate::device::DeviceImage> {
    use crate::device::DeviceImage;
    macro_rules! up {
        ($t:ty, $c:literal, $variant:ident) => {{
            // SAFETY: caller validated a host backing (ensure_host); borrow bytes.
            let host =
                unsafe { backing::borrow_image::<$t, $c>(backing, shape) }.map_err(to_pyerr)?;
            let dev = host.to_cuda_image(stream).map_err(to_pyerr)?;
            DeviceImage::$variant(dev)
        }};
    }
    Ok(match (dtype, channels) {
        (backing::Dtype::U8, 1) => up!(u8, 1, U8C1),
        (backing::Dtype::U8, 3) => up!(u8, 3, U8C3),
        (backing::Dtype::U8, 4) => up!(u8, 4, U8C4),
        (backing::Dtype::F32, 1) => up!(f32, 1, F32C1),
        (backing::Dtype::F32, 3) => up!(f32, 3, F32C3),
        _ => {
            return Err(value_err(format!(
                "device images support uint8 (1/3/4 channels) or float32 (1/3 channels); \
                 got {} with {} channels",
                dtype.name(),
                channels
            )))
        }
    })
}

/// Allocate a zero-initialized device-resident `Image`.
#[cfg(feature = "cuda")]
fn zeros_device(
    py: Python<'_>,
    width: usize,
    height: usize,
    channels: usize,
    dtype: &str,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
) -> PyResult<PyImageApi> {
    use crate::device::DeviceImage;
    let dt = backing::Dtype::from_numpy_str(dtype)?;
    let size = ImageSize { width, height };
    let dev = py.detach(|| -> PyResult<DeviceImage> {
        macro_rules! z {
            ($t:ty, $c:literal, $variant:ident) => {
                DeviceImage::$variant(Image::<$t, $c>::zeros_cuda(size, stream).map_err(to_pyerr)?)
            };
        }
        Ok(match (dt, channels) {
            (backing::Dtype::U8, 1) => z!(u8, 1, U8C1),
            (backing::Dtype::U8, 3) => z!(u8, 3, U8C3),
            (backing::Dtype::U8, 4) => z!(u8, 4, U8C4),
            (backing::Dtype::F32, 1) => z!(f32, 1, F32C1),
            (backing::Dtype::F32, 3) => z!(f32, 3, F32C3),
            _ => {
                return Err(value_err(format!(
                    "device images support uint8 (1/3/4 channels) or float32 (1/3 channels); \
                     got {dtype} with {channels} channels"
                )))
            }
        })
    })?;
    let cs = default_color_space(channels);
    let mode = match dt {
        backing::Dtype::U8 => mode_from_channels(channels, false),
        backing::Dtype::F32 => mode_from_channels_f32(channels),
        backing::Dtype::U16 => unreachable!("u16 rejected above"),
    };
    Ok(PyImageApi::from_device(dev, cs, mode))
}

/// The stream kornia device work actually runs on, plus an optional foreign
/// `CUstream` handle to fence completion into afterwards (see [`resolve_stream`]).
#[cfg(feature = "cuda")]
pub(crate) struct ResolvedStream {
    /// Stream kornia submits its copy / kernel onto.
    pub(crate) launch: std::sync::Arc<cudarc::driver::CudaStream>,
    /// A borrowed foreign stream to order *after* `launch` completes, or `None`.
    pub(crate) foreign: Option<usize>,
}

/// Resolve an optional Python `Stream` for a device op.
///
/// An owned kornia stream is submitted onto directly. A foreign (adopted)
/// stream can't be submitted onto through cudarc 0.19, so work runs on kornia's
/// default stream and the foreign handle is returned for [`fence_into_foreign`]
/// to order the caller's stream after ours. `None` uses the default stream.
#[cfg(feature = "cuda")]
fn resolve_stream(stream: Option<PyRef<'_, PyStream>>) -> PyResult<ResolvedStream> {
    match stream.map(|s| s.inner.clone()) {
        Some(StreamInner::Owned(s)) => Ok(ResolvedStream {
            launch: s,
            foreign: None,
        }),
        Some(StreamInner::Foreign(h)) => Ok(ResolvedStream {
            launch: crate::cuda_ext::default_stream()?,
            foreign: Some(h),
        }),
        None => Ok(ResolvedStream {
            launch: crate::cuda_ext::default_stream()?,
            foreign: None,
        }),
    }
}

/// After device work is enqueued on `launch`, make a borrowed foreign stream
/// wait on its completion, so the caller's subsequent work on that stream is
/// ordered after kornia's — without blocking the host. No-op when there is no
/// foreign stream to fence into.
#[cfg(feature = "cuda")]
fn fence_into_foreign(resolved: &ResolvedStream) -> PyResult<()> {
    // Same record-event + `cuStreamWaitEvent` primitive the preprocessor uses;
    // a `ResolvedStream` just carries the (launch, foreign-handle) pair it needs.
    crate::cuda_ext::fence_stream_into(&resolved.launch, resolved.foreign)
}

/// Map a producer's raw `CUstream` handle to the value the CUDA Array Interface
/// (v3) `stream` key expects: `None` when there is no stream, `1` for the legacy
/// default stream (CAI disallows the ambiguous `0`), otherwise the handle. Real
/// CAI consumers (CuPy, Numba) reject a literal `0`.
#[cfg(feature = "cuda")]
pub(crate) fn cai_stream_value(py: Python<'_>, raw: Option<usize>) -> Py<PyAny> {
    use pyo3::IntoPyObject;
    match raw {
        None => py.None(),
        Some(0) => 1i64.into_pyobject(py).unwrap().into_any().unbind(),
        Some(h) => (h as i64).into_pyobject(py).unwrap().into_any().unbind(),
    }
}

/// Backing of a [`PyStream`]: either a cudarc-owned stream (kornia's own
/// default stream) or a *borrowed* foreign `CUstream` handle adopted from
/// NVIDIA's stack (`cuda-python` / `cuda.core` / CuPy / a raw handle).
///
/// A foreign handle is never owned: we do not create it and must never
/// `cuStreamDestroy` it. cudarc 0.19 exposes no non-owning `CudaStream`
/// wrapper, so kornia can't submit kernels directly onto a foreign stream;
/// instead device work runs on kornia's default stream and is **fenced into**
/// the foreign stream with a CUDA event (see [`resolve_stream`] /
/// [`fence_into_foreign`]), which is equivalent for cross-stream ordering.
#[cfg(feature = "cuda")]
#[derive(Clone)]
pub(crate) enum StreamInner {
    Owned(std::sync::Arc<cudarc::driver::CudaStream>),
    Foreign(usize),
}

#[cfg(feature = "cuda")]
impl PyStream {
    /// Raw `CUstream` handle (address) of this stream, regardless of variant.
    /// Used by other modules (e.g. the preprocessor) to fence work into it.
    pub(crate) fn raw_handle(&self) -> usize {
        match &self.inner {
            StreamInner::Owned(s) => s.cu_stream() as usize,
            StreamInner::Foreign(h) => *h,
        }
    }
}

/// A CUDA stream handle, shareable with kornia device transfers and the DLPack
/// / `cuda.core` `stream=` protocols. Only meaningful on a `cuda`-enabled build.
///
/// Build one with [`Stream::default`], or adopt an existing NVIDIA stream with
/// [`Stream::from_handle`] / [`Stream::from_cuda_stream`] so kornia device ops
/// order their work into your stream.
#[pyclass(name = "Stream", module = "kornia_rs.cuda", frozen)]
pub struct PyStream {
    #[cfg(feature = "cuda")]
    pub(crate) inner: StreamInner,
}

#[pymethods]
impl PyStream {
    /// The process-wide default CUDA stream for `device` (default 0). The
    /// stream's device is the selector for where `Image.to_cuda(stream)` /
    /// `Image.zeros(..., stream=stream)` place data — mirrors Rust's
    /// `CudaContext::new(ordinal).default_stream()`.
    #[staticmethod]
    #[pyo3(signature = (device = 0))]
    fn default(py: Python<'_>, device: i32) -> PyResult<Self> {
        let _ = py;
        #[cfg(feature = "cuda")]
        {
            Ok(Self {
                inner: StreamInner::Owned(crate::cuda_ext::default_stream_for(device)?),
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = device;
            Err(cuda_not_compiled())
        }
    }

    /// Adopt an existing raw `CUstream` handle (as an integer) from NVIDIA's
    /// stack — e.g. `cuda.core.Stream.handle`, a cuda-python `CUstream`, or a
    /// CuPy `stream.ptr`. kornia does **not** take ownership of the stream (it
    /// is never destroyed here); device ops fence their work into it via a CUDA
    /// event so your subsequent work on the same stream is correctly ordered
    /// after kornia's.
    #[staticmethod]
    fn from_handle(handle: usize) -> PyResult<Self> {
        #[cfg(feature = "cuda")]
        {
            Ok(Self {
                inner: StreamInner::Foreign(handle),
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = handle;
            Err(cuda_not_compiled())
        }
    }

    /// Adopt a stream from any object implementing the `cuda-python` /
    /// `cuda.core` stream protocol (`__cuda_stream__() -> (version, handle)`),
    /// or exposing an integer `.ptr` / `.handle`, or that is itself an int.
    /// See [`from_handle`](Self::from_handle) for the ownership/fencing model.
    #[staticmethod]
    fn from_cuda_stream(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        #[cfg(feature = "cuda")]
        {
            let handle = extract_stream_handle(obj)?;
            Ok(Self {
                inner: StreamInner::Foreign(handle),
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = obj;
            Err(cuda_not_compiled())
        }
    }

    /// Block the host until all work on this stream completes.
    fn synchronize(&self) -> PyResult<()> {
        #[cfg(feature = "cuda")]
        {
            match &self.inner {
                StreamInner::Owned(s) => s.synchronize().map_err(to_pyerr),
                StreamInner::Foreign(h) => {
                    // SAFETY: `h` is a live `CUstream` the caller vouched for.
                    unsafe {
                        cudarc::driver::sys::cuStreamSynchronize(
                            *h as cudarc::driver::sys::CUstream,
                        )
                    }
                    .result()
                    .map_err(to_pyerr)
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(cuda_not_compiled())
        }
    }

    /// Raw `CUstream` handle as an integer, for the DLPack / `cuda.core`
    /// `stream=` protocols.
    #[getter]
    fn cuda_stream_ptr(&self) -> usize {
        #[cfg(feature = "cuda")]
        {
            match &self.inner {
                StreamInner::Owned(s) => s.cu_stream() as usize,
                StreamInner::Foreign(h) => *h,
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            0
        }
    }

    /// NVIDIA `cuda-python` / `cuda.core` stream protocol: returns
    /// `(protocol_version, cuStream_handle)` so a kornia `Stream` can be passed
    /// wherever `cuda.core` accepts a stream-like object.
    fn __cuda_stream__(&self) -> (i32, usize) {
        (0, self.cuda_stream_ptr())
    }
}

/// Pull a raw `CUstream` handle out of a Python object: the `cuda-python` /
/// `cuda.core` `__cuda_stream__()` protocol first, then a plain integer, then a
/// `.ptr` / `.handle` attribute (CuPy / cuda.core).
#[cfg(feature = "cuda")]
fn extract_stream_handle(obj: &Bound<'_, PyAny>) -> PyResult<usize> {
    // 1) __cuda_stream__() -> (version, handle)
    if let Ok(res) = obj.call_method0("__cuda_stream__") {
        let (_ver, handle): (i32, usize) = res.extract().map_err(|_| {
            value_err("__cuda_stream__() did not return a (version, handle) tuple of ints")
        })?;
        return Ok(handle);
    }
    // 2) a bare integer handle
    if let Ok(handle) = obj.extract::<usize>() {
        return Ok(handle);
    }
    // 3) an object carrying an integer `.ptr` (CuPy) or `.handle` (cuda.core)
    for attr in ["ptr", "handle"] {
        if let Ok(v) = obj.getattr(attr).and_then(|a| a.extract::<usize>()) {
            return Ok(v);
        }
    }
    Err(value_err(
        "expected a CUDA stream: an object with __cuda_stream__(), an int handle, \
         or a .ptr / .handle integer attribute",
    ))
}

#[pymethods]
impl PyImageApi {
    /// Allocate a zero-initialized `Image` of `(height, width, channels)`.
    /// `dtype` is `"uint8"` or `"float32"`. With `stream=None` the image is on
    /// the host; passing a `Stream` allocates directly on that stream's CUDA
    /// device (mirrors Rust `zeros_cuda(size, &stream)`), avoiding a host→device
    /// bounce. To place existing data on a device, use `.to_cuda(stream)`.
    #[staticmethod]
    #[pyo3(signature = (width, height, channels, dtype = "uint8", stream = None))]
    fn zeros(
        py: Python<'_>,
        width: usize,
        height: usize,
        channels: usize,
        dtype: &str,
        stream: Option<PyRef<'_, PyStream>>,
    ) -> PyResult<Self> {
        #[cfg(feature = "cuda")]
        if stream.is_some() {
            let resolved = resolve_stream(stream)?;
            let dev = zeros_device(py, width, height, channels, dtype, &resolved.launch)?;
            fence_into_foreign(&resolved)?;
            return Ok(dev);
        }
        #[cfg(not(feature = "cuda"))]
        let _ = &stream;

        let dt = backing::Dtype::from_numpy_str(dtype)?;
        let nbytes = backing::byte_len(height, width, channels, dt)?;
        let bytes = backing::AlignedBytes::zeroed(nbytes);
        let cs = default_color_space(channels);
        let mode = mode_for_dtype(dt, channels);
        Ok(Self::from_owned_bytes(
            bytes,
            dt,
            [height, width, channels],
            cs,
            mode,
        ))
    }

    /// Device this image lives on: `"cpu"` or `"cuda:{id}"` (mirrors Rust's
    /// `MemoryDomain`). Placement is via `.to_cuda(stream)` / `.cpu()`.
    #[getter]
    fn device(&self) -> String {
        let (ty, id) = self.backing.device();
        if ty == dlpack_rs::ffi::DLDeviceType::kDLCUDA as i32 {
            format!("cuda:{id}")
        } else {
            "cpu".to_string()
        }
    }

    /// The [CUDA Array Interface] (v3) for zero-copy sharing with cupy / numba /
    /// nvidia `cuda-python`. Present only on device images; raises
    /// `AttributeError` on host images (use `__array_interface__` / `.numpy()`),
    /// so consumers correctly treat a host `Image` as a non-CUDA array.
    ///
    /// [CUDA Array Interface]: https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html
    #[getter]
    fn __cuda_array_interface__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        #[cfg(feature = "cuda")]
        {
            let dev = self.as_device().ok_or_else(|| {
                pyo3::exceptions::PyAttributeError::new_err(
                    "__cuda_array_interface__ is only available on CUDA device images; \
                     this image is on host (use .numpy() / __array_interface__)",
                )
            })?;
            let [h, w, c] = self.shape;
            let typestr = match self.dtype {
                backing::Dtype::U8 => "|u1",
                backing::Dtype::U16 => "<u2",
                backing::Dtype::F32 => "<f4",
            };
            let d = pyo3::types::PyDict::new(py);
            d.set_item("shape", (h, w, c))?;
            d.set_item("typestr", typestr)?;
            d.set_item("data", (dev.as_ptr() as usize, self.backing.readonly()))?;
            // C-contiguous HWC — `strides = None` per the interface.
            d.set_item("strides", py.None())?;
            d.set_item("version", 3)?;
            // Producer stream so a consumer can order its work after ours.
            d.set_item("stream", cai_stream_value(py, dev.stream_ptr()))?;
            Ok(d.into_any().unbind())
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = py;
            Err(pyo3::exceptions::PyAttributeError::new_err(
                "__cuda_array_interface__ is only available on CUDA device images",
            ))
        }
    }

    /// Copy this image to host (CPU) memory. A device image is copied back (D2H);
    /// a host image returns an owned copy. `.numpy()` calls this implicitly.
    #[pyo3(signature = (stream = None))]
    fn cpu(&self, py: Python<'_>, stream: Option<PyRef<'_, PyStream>>) -> PyResult<Self> {
        let _ = stream; // D2H uses the image's own carried stream.
        #[cfg(feature = "cuda")]
        {
            self.to_host_internal(py)
        }
        #[cfg(not(feature = "cuda"))]
        {
            if self.backing.is_host() {
                Ok(self.clone_handle(py))
            } else {
                Err(cuda_not_compiled())
            }
        }
    }

    /// Copy this image to CUDA device memory (H2D), returning a device `Image`.
    /// A no-op handle share if it is already on device. Requires the `cuda`
    /// feature. `stream`: optional `Stream`.
    #[pyo3(signature = (stream = None))]
    fn to_cuda(&self, py: Python<'_>, stream: Option<PyRef<'_, PyStream>>) -> PyResult<Self> {
        #[cfg(feature = "cuda")]
        {
            let resolved = resolve_stream(stream)?;
            let dev = self.to_device_internal(py, resolved.launch.clone())?;
            fence_into_foreign(&resolved)?;
            Ok(dev)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (py, stream);
            Err(cuda_not_compiled())
        }
    }
}
