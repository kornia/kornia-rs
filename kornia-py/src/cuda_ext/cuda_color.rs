//! Device-resident GPU color conversions (`kornia_rs.imgproc.*` device path).
//!
//! Genuinely device-only, so the whole module is `#[cfg(feature = "cuda")]`-gated
//! by its `mod color;` declaration in the parent. Each op takes and returns a
//! device-resident unified [`PyImageApi`] (`Backing::Device`); `crate::color`'s
//! residency dispatch routes a device `Image` here and a host/numpy one to the
//! CPU path. Re-exported from the parent as `crate::cuda_ext::<op>`.

use super::*;

/// View a plain `Image` as its `#[repr(transparent)]` color-space newtype.
///
/// SAFETY: every `define_color_space!` newtype is `#[repr(transparent)]` over
/// `Image<T, C>`, so the pointer casts are layout-sound.
macro_rules! as_newtype {
    ($img:expr, $nt:ty) => {
        unsafe { &*($img as *const _ as *const $nt) }
    };
    (mut $img:expr, $nt:ty) => {
        unsafe { &mut *($img as *mut _ as *mut $nt) }
    };
}

/// Run `src.convert(&mut dst)` through the repr(transparent) newtypes.
macro_rules! convert_pair {
    ($src:expr, $snt:ty, $dst:expr, $dnt:ty) => {{
        let s = as_newtype!($src, $snt);
        let d = as_newtype!(mut $dst, $dnt);
        s.convert(d).map_err(err)
    }};
}

/// Run `src.convert_with_bg(&mut dst, bg)` through the repr(transparent)
/// newtypes — the alpha-compositing sibling of [`convert_pair!`].
macro_rules! convert_pair_bg {
    ($src:expr, $snt:ty, $dst:expr, $dnt:ty, $bg:expr) => {{
        let s = as_newtype!($src, $snt);
        let d = as_newtype!(mut $dst, $dnt);
        s.convert_with_bg(d, $bg).map_err(err)
    }};
}

/// PIL-style mode string for a device output of channel count `channels`.
fn device_mode<T: 'static>(channels: usize) -> String {
    let dt = if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        crate::backing::Dtype::F32
    } else {
        crate::backing::Dtype::U8
    };
    crate::image::mode_for_dtype(dt, channels)
}

/// Borrow a `&PyImageApi` as its device source variant, or raise a clear error.
macro_rules! device_src {
    ($img:expr, $pyname:expr, $srcvar:ident) => {{
        let dev = $img.as_device().ok_or_else(|| {
            PyValueError::new_err(concat!(
                $pyname,
                ": expected a device Image (on a CUDA device); for a host image pass \
                 its numpy array, or move it to a device with .to_cuda(stream)"
            ))
        })?;
        let Inner::$srcvar(src) = dev else {
            return Err(PyValueError::new_err(concat!(
                $pyname,
                ": wrong input dtype/channels for this conversion"
            )));
        };
        src
    }};
}

/// The stream a device source image lives on, for allocating a same-device
/// destination. Falls back to device 0's default stream if the backing carries
/// no stream (shouldn't normally happen for a typed device image).
fn source_stream<T, const C: usize>(src: &Image<T, C>) -> PyResult<Arc<CudaStream>>
where
    T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits + 'static,
{
    match src.cuda_stream() {
        Some(s) => Ok(s.clone()),
        None => default_stream(),
    }
}

/// Allocate a device destination and run one `ConvertColor` pair, returning a
/// device-resident unified `Image` tagged with the output color space.
macro_rules! conv_fn {
    ($(#[$meta:meta])* $pyname:ident, $srcvar:ident, $snt:ty, $t:ty, $dc:literal, $dvar:ident, $dnt:ty, $dcs:expr) => {
        $(#[$meta])*
        #[pyfunction]
        pub fn $pyname(img: &PyImageApi) -> PyResult<PyImageApi> {
            let src = device_src!(img, stringify!($pyname), $srcvar);
            // Allocate the destination on the SOURCE's own stream/device, not a
            // hardcoded device 0: a source resident on a non-default GPU would
            // otherwise get a device-0 dst and the downstream residency check
            // (`pair_residency`) would reject the mismatched pair with
            // `DeviceMismatch` on multi-GPU systems.
            let stream = source_stream(src)?;
            // SAFETY: the ConvertColor kernel writes every output pixel (one
            // thread per pixel, bounds-guarded), so the uninitialized destination
            // is fully overwritten before any read.
            let mut dst = unsafe { Image::<$t, $dc>::uninit_cuda(src.size(), &stream) }.map_err(err)?;
            convert_pair!(src, $snt, &mut dst, $dnt)?;
            Ok(PyImageApi::from_device(
                Inner::$dvar(dst),
                $dcs,
                device_mode::<$t>($dc),
            ))
        }
    };
}

conv_fn!(
    /// RGB8 → Gray8 (BT.601, bit-exact vs the CPU path).
    gray_from_rgb, U8C3, Rgb8, u8, 1, U8C1, Gray8, ColorSpace::Gray
);
conv_fn!(
    /// Gray8 → RGB8 broadcast.
    rgb_from_gray, U8C1, Gray8, u8, 3, U8C3, Rgb8, ColorSpace::Rgb
);
conv_fn!(
    /// RGB8 → BGR8 channel swap (symmetric).
    bgr_from_rgb, U8C3, Rgb8, u8, 3, U8C3, Bgr8, ColorSpace::Bgr
);
conv_fn!(
    /// RGB8 → RGBA8 (opaque alpha).
    rgba_from_rgb, U8C3, Rgb8, u8, 4, U8C4, Rgba8, ColorSpace::Rgba
);
// RGBA8/BGRA8 → RGB8 are provided by `rgb_from_rgba_bg`/`rgb_from_bgra_bg`
// (below) instead of `conv_fn!`, so the device path can honor the `background`
// alpha-composite argument like the CPU path rather than only dropping alpha.
conv_fn!(
    /// RGB8 → YCbCr8 (full-range Q14, bit-exact vs the CPU path).
    ycbcr_from_rgb, U8C3, Rgb8, u8, 3, U8C3, YCbCr8, ColorSpace::YCbCr
);
conv_fn!(
    /// YCbCr8 → RGB8.
    rgb_from_ycbcr, U8C3, YCbCr8, u8, 3, U8C3, Rgb8, ColorSpace::Rgb
);
conv_fn!(
    /// RGB f32 → HSV f32 (kornia conventions, [0,255] scale).
    hsv_from_rgb, F32C3, Rgbf32, f32, 3, F32C3, Hsvf32, ColorSpace::Hsv
);
conv_fn!(
    /// HSV f32 → RGB f32.
    rgb_from_hsv, F32C3, Hsvf32, f32, 3, F32C3, Rgbf32, ColorSpace::Rgb
);
conv_fn!(
    /// RGB f32 → CIE Lab f32 (RGB in [0,1], L in [0,100]).
    lab_from_rgb, F32C3, Rgbf32, f32, 3, F32C3, Labf32, ColorSpace::Lab
);
conv_fn!(
    /// Lab f32 → RGB f32.
    rgb_from_lab, F32C3, Labf32, f32, 3, F32C3, Rgbf32, ColorSpace::Rgb
);

/// Sepia tone on RGB8 (Q8 fixed point, bit-exact vs the CPU path).
#[pyfunction]
pub fn sepia_from_rgb(img: &PyImageApi) -> PyResult<PyImageApi> {
    let src = device_src!(img, "sepia_from_rgb", U8C3);
    // Allocate the destination on the source's own stream/device (see `conv_fn!`).
    let stream = source_stream(src)?;
    // SAFETY: the sepia kernel writes every output pixel, so the uninitialized
    // destination is fully overwritten before any read.
    let mut dst = unsafe { Image::<u8, 3>::uninit_cuda(src.size(), &stream) }.map_err(err)?;
    color::sepia_from_rgb_u8(src, &mut dst).map_err(err)?;
    Ok(PyImageApi::from_device(
        Inner::U8C3(dst),
        ColorSpace::Rgb,
        device_mode::<u8>(3),
    ))
}

/// Apply one of the 21 OpenCV colormaps to a Gray8 image (name as in
/// `kornia_rs.imgproc.apply_colormap`).
#[pyfunction]
pub fn apply_colormap(img: &PyImageApi, colormap: &str) -> PyResult<PyImageApi> {
    let src = device_src!(img, "apply_colormap", U8C1);
    let cmap = color::ColormapType::from_name(colormap)
        .ok_or_else(|| PyValueError::new_err(format!("unknown colormap '{colormap}'")))?;
    // Allocate the destination on the source's own stream/device (see `conv_fn!`).
    let stream = source_stream(src)?;
    // SAFETY: the colormap kernel writes every output pixel, so the uninitialized
    // destination is fully overwritten before any read.
    let mut dst = unsafe { Image::<u8, 3>::uninit_cuda(src.size(), &stream) }.map_err(err)?;
    color::apply_colormap(src, &mut dst, cmap).map_err(err)?;
    Ok(PyImageApi::from_device(
        Inner::U8C3(dst),
        ColorSpace::Rgb,
        device_mode::<u8>(3),
    ))
}

/// Bayer demosaic (pattern: "rggb" | "bggr" | "grbg" | "gbrg").
#[pyfunction]
pub fn rgb_from_bayer(img: &PyImageApi, pattern: &str) -> PyResult<PyImageApi> {
    use kornia_image::color_spaces::BayerPattern;
    let src = device_src!(img, "rgb_from_bayer", U8C1);
    // Allocate the destination on the source's own stream/device (see `conv_fn!`).
    let stream = source_stream(src)?;
    let pat = match pattern.to_ascii_lowercase().as_str() {
        "rggb" => BayerPattern::Rggb,
        "bggr" => BayerPattern::Bggr,
        "grbg" => BayerPattern::Grbg,
        "gbrg" => BayerPattern::Gbrg,
        p => {
            return Err(PyValueError::new_err(format!(
                "unknown bayer pattern '{p}'"
            )))
        }
    };
    // SAFETY: the bayer demosaic kernel writes every output pixel (bounds-guarded,
    // replicate-border reads), so the uninitialized destination is fully
    // overwritten before any read.
    let mut dst = unsafe { Image::<u8, 3>::uninit_cuda(src.size(), &stream) }.map_err(err)?;
    color::rgb_from_bayer(src, pat, &mut dst).map_err(err)?;
    Ok(PyImageApi::from_device(
        Inner::U8C3(dst),
        ColorSpace::Rgb,
        device_mode::<u8>(3),
    ))
}

/// `{Rgba8,Bgra8}8 → RGB8` on device with optional `background` alpha
/// compositing (mirrors the CPU path's `background=` — otherwise a device image
/// would silently drop alpha while the host path composited; `None` = opaque
/// alpha drop). The two only differ by source newtype, so share this body.
macro_rules! conv_bg_fn {
    ($(#[$meta:meta])* $pyname:ident, $snt:ty, $errname:literal) => {
        $(#[$meta])*
        pub fn $pyname(img: &PyImageApi, background: Option<[u8; 3]>) -> PyResult<PyImageApi> {
            let src = device_src!(img, $errname, U8C4);
            let stream = source_stream(src)?;
            // SAFETY: the blend/drop kernel writes every output pixel (one
            // thread per pixel, bounds-guarded), so the uninitialized
            // destination is fully overwritten before any read.
            let mut dst = unsafe { Image::<u8, 3>::uninit_cuda(src.size(), &stream) }.map_err(err)?;
            convert_pair_bg!(src, $snt, &mut dst, Rgb8, background)?;
            Ok(PyImageApi::from_device(
                Inner::U8C3(dst),
                ColorSpace::Rgb,
                device_mode::<u8>(3),
            ))
        }
    };
}
conv_bg_fn!(
    /// RGBA8 → RGB8 on device, compositing over `background` when given.
    rgb_from_rgba_bg, Rgba8, "rgb_from_rgba"
);
conv_bg_fn!(
    /// BGRA8 → RGB8 on device, compositing over `background` when given.
    rgb_from_bgra_bg, Bgra8, "rgb_from_bgra"
);
