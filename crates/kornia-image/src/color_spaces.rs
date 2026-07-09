//! Runtime color-space vocabulary and typed color-space wrappers.
//!
//! This module contains both the runtime [`ColorSpace`] enum + [`DynImage`] type
//! (shared by Rust and Python) and the typed zero-cost newtype wrappers produced
//! by [`define_color_space!`].

use crate::{
    error::ImageError,
    image::{Image, ImageSize},
};
use kornia_tensor::AllocHandle;
use std::ops::{Deref, DerefMut};

// ===== Runtime color-space vocabulary ===========================================

/// A per-pixel color space. The shared vocabulary for the high-level
/// conversion API (`.cvt()` typed path and `cvt_color` runtime path).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ColorSpace {
    /// Red-Green-Blue (3 channels, u8 or f32).
    Rgb,
    /// Blue-Green-Red (3 channels, OpenCV native order).
    Bgr,
    /// Grayscale luminance (1 channel).
    Gray,
    /// Red-Green-Blue-Alpha (4 channels).
    Rgba,
    /// Blue-Green-Red-Alpha (4 channels).
    Bgra,
    /// Hue-Saturation-Value (3 channels, f32 only).
    Hsv,
    /// Hue-Lightness-Saturation (3 channels, f32 only).
    Hls,
    /// CIE L*a*b* (3 channels, f32 only).
    Lab,
    /// CIE L*u*v* (3 channels, f32 only).
    Luv,
    /// CIE XYZ (3 channels, f32 only).
    Xyz,
    /// Linear (gamma-decoded) RGB (3 channels, f32 only).
    LinearRgb,
    /// Y′CbCr (3 channels).
    YCbCr,
    /// YUV (3 channels).
    Yuv,
}

impl ColorSpace {
    /// Number of channels an image in this space has.
    ///
    /// # Returns
    ///
    /// The channel count for this color space (1, 3, or 4).
    pub const fn channels(self) -> usize {
        match self {
            ColorSpace::Gray => 1,
            ColorSpace::Rgba | ColorSpace::Bgra => 4,
            _ => 3,
        }
    }

    /// True for spaces whose kernels only operate on f32 data.
    ///
    /// # Returns
    ///
    /// `true` if this space requires f32 pixel data.
    pub const fn requires_f32(self) -> bool {
        matches!(
            self,
            ColorSpace::Hsv
                | ColorSpace::Hls
                | ColorSpace::Lab
                | ColorSpace::Luv
                | ColorSpace::Xyz
                | ColorSpace::LinearRgb
        )
    }

    /// Whether a direct kernel exists for `from -> to`. Mirrors the
    /// `ConvertColor` impls in kornia-imgproc (RGB-hub graph).
    ///
    /// This is the **single source of truth** for which color conversions are
    /// legal at the Rust API level. The kornia-imgproc dispatch macro must
    /// cover every pair that returns `true` here.
    pub const fn supports(from: ColorSpace, to: ColorSpace) -> bool {
        use ColorSpace::*;
        matches!(
            (from, to),
            (Rgb, Gray)
                | (Gray, Rgb)
                | (Rgb, Bgr)
                | (Bgr, Rgb)
                | (Rgb, Rgba)
                | (Rgba, Rgb)
                | (Rgb, Bgra)
                | (Bgra, Rgb)
                | (Rgb, Hsv)
                | (Hsv, Rgb)
                | (Rgb, Hls)
                | (Hls, Rgb)
                | (Rgb, Lab)
                | (Lab, Rgb)
                | (Rgb, Luv)
                | (Luv, Rgb)
                | (Rgb, Xyz)
                | (Xyz, Rgb)
                | (Rgb, LinearRgb)
                | (LinearRgb, Rgb)
                | (Rgb, YCbCr)
                | (YCbCr, Rgb)
                | (Rgb, Yuv)
                | (Yuv, Rgb)
        )
    }

    /// Whether a `from -> to` conversion is valid for the given element dtype.
    ///
    /// Combines the legality check of [`Self::supports`] with the dtype
    /// constraint encoded in [`Self::requires_f32`]:
    ///
    /// - f32-only spaces (HSV, HLS, Lab, Luv, XYZ, LinearRgb) require
    ///   `is_f32 == true`.
    /// - The u8-compatible alpha spaces (Rgba, Bgra paired with Rgb/Bgr)
    ///   are valid for both dtypes.
    ///
    /// Returns `false` if `supports(from, to)` is `false` regardless of dtype.
    pub const fn supports_dtype(from: ColorSpace, to: ColorSpace, is_f32: bool) -> bool {
        if !Self::supports(from, to) {
            return false;
        }
        // If either endpoint is f32-only, the image must be f32.
        if from.requires_f32() || to.requires_f32() {
            return is_f32;
        }
        true
    }
}

/// Owned image whose channel count is known only at runtime. Mirrors Python's
/// dynamic numpy-backed Image; produced by the runtime `cvt_color` path.
pub enum DynImage<T> {
    /// 1-channel (e.g. Gray).
    C1(ColorSpace, Image<T, 1>),
    /// 3-channel (Rgb/Bgr/Hsv/...).
    C3(ColorSpace, Image<T, 3>),
    /// 4-channel (Rgba/Bgra).
    C4(ColorSpace, Image<T, 4>),
}

impl<T> DynImage<T> {
    /// The color space tag carried by this image.
    ///
    /// # Returns
    ///
    /// The [`ColorSpace`] tag of this image.
    pub fn color_space(&self) -> ColorSpace {
        match self {
            DynImage::C1(s, _) | DynImage::C3(s, _) | DynImage::C4(s, _) => *s,
        }
    }

    /// Image dimensions.
    ///
    /// # Returns
    ///
    /// The [`ImageSize`] (width × height) of this image.
    pub fn size(&self) -> ImageSize {
        match self {
            DynImage::C1(_, i) => i.size(),
            DynImage::C3(_, i) => i.size(),
            DynImage::C4(_, i) => i.size(),
        }
    }

    /// Channel count (1, 3, or 4).
    ///
    /// # Returns
    ///
    /// The channel count: 1, 3, or 4.
    pub fn channels(&self) -> usize {
        match self {
            DynImage::C1(..) => 1,
            DynImage::C3(..) => 3,
            DynImage::C4(..) => 4,
        }
    }

    /// Contiguous (H, W, C) row-major data.
    ///
    /// # Returns
    ///
    /// A contiguous byte slice in H×W×C row-major order.
    pub fn as_slice(&self) -> &[T] {
        match self {
            DynImage::C1(_, i) => i.as_slice(),
            DynImage::C3(_, i) => i.as_slice(),
            DynImage::C4(_, i) => i.as_slice(),
        }
    }
}

macro_rules! impl_try_from_dyn {
    ($newtype:ident, $t:ty, C1, $space:expr_2021) => {
        impl std::convert::TryFrom<DynImage<$t>> for crate::color_spaces::$newtype {
            type Error = ImageError;
            fn try_from(d: DynImage<$t>) -> Result<Self, ImageError> {
                match d {
                    DynImage::C1(s, img) if s == $space => Ok(Self(img)),
                    other => Err(ImageError::ColorSpaceMismatch {
                        expected: $space,
                        got: other.color_space(),
                    }),
                }
            }
        }
    };
    ($newtype:ident, $t:ty, C3, $space:expr_2021) => {
        impl std::convert::TryFrom<DynImage<$t>> for crate::color_spaces::$newtype {
            type Error = ImageError;
            fn try_from(d: DynImage<$t>) -> Result<Self, ImageError> {
                match d {
                    DynImage::C3(s, img) if s == $space => Ok(Self(img)),
                    other => Err(ImageError::ColorSpaceMismatch {
                        expected: $space,
                        got: other.color_space(),
                    }),
                }
            }
        }
    };
    ($newtype:ident, $t:ty, C4, $space:expr_2021) => {
        impl std::convert::TryFrom<DynImage<$t>> for crate::color_spaces::$newtype {
            type Error = ImageError;
            fn try_from(d: DynImage<$t>) -> Result<Self, ImageError> {
                match d {
                    DynImage::C4(s, img) if s == $space => Ok(Self(img)),
                    other => Err(ImageError::ColorSpaceMismatch {
                        expected: $space,
                        got: other.color_space(),
                    }),
                }
            }
        }
    };
}

impl_try_from_dyn!(Rgbf32, f32, C3, ColorSpace::Rgb);
impl_try_from_dyn!(Bgrf32, f32, C3, ColorSpace::Bgr);
impl_try_from_dyn!(Grayf32, f32, C1, ColorSpace::Gray);
impl_try_from_dyn!(Hsvf32, f32, C3, ColorSpace::Hsv);
impl_try_from_dyn!(Hlsf32, f32, C3, ColorSpace::Hls);
impl_try_from_dyn!(Labf32, f32, C3, ColorSpace::Lab);
impl_try_from_dyn!(Luvf32, f32, C3, ColorSpace::Luv);
impl_try_from_dyn!(Xyzf32, f32, C3, ColorSpace::Xyz);
impl_try_from_dyn!(LinearRgbf32, f32, C3, ColorSpace::LinearRgb);
impl_try_from_dyn!(YCbCrf32, f32, C3, ColorSpace::YCbCr);
impl_try_from_dyn!(Yuvf32, f32, C3, ColorSpace::Yuv);
impl_try_from_dyn!(Rgb8, u8, C3, ColorSpace::Rgb);
impl_try_from_dyn!(Bgr8, u8, C3, ColorSpace::Bgr);
impl_try_from_dyn!(Gray8, u8, C1, ColorSpace::Gray);
impl_try_from_dyn!(Rgba8, u8, C4, ColorSpace::Rgba);
impl_try_from_dyn!(Bgra8, u8, C4, ColorSpace::Bgra);
impl_try_from_dyn!(YCbCr8, u8, C3, ColorSpace::YCbCr);
impl_try_from_dyn!(Yuv8, u8, C3, ColorSpace::Yuv);

// ===== Typed color-space newtype wrappers =======================================

/// Macro to define a color space wrapper type with explicit bit depth
macro_rules! define_color_space {
    ($name:ident, $type:ty, $channels:expr_2021, $doc:expr_2021) => {
        #[doc = $doc]
        ///
        /// This is a zero-cost wrapper that provides compile-time type safety.
        #[repr(transparent)]
        pub struct $name(pub Image<$type, $channels>);

        impl $name {
            #[doc = concat!("Create ", stringify!($name), " image from size and data")]
            pub fn from_size_vec(size: ImageSize, data: Vec<$type>) -> Result<Self, ImageError> {
                Self::from_size_vec_in(size, data, kornia_tensor::host_alloc())
            }

            #[doc = concat!("Like `from_size_vec` but with an explicit allocator handle")]
            pub fn from_size_vec_in(
                size: ImageSize,
                data: Vec<$type>,
                alloc: AllocHandle,
            ) -> Result<Self, ImageError> {
                Ok(Self(Image::new_in(size, data, alloc)?))
            }

            #[doc = concat!("Create ", stringify!($name), " image from size with default value")]
            pub fn from_size_val(size: ImageSize, val: $type) -> Result<Self, ImageError> {
                Self::from_size_val_in(size, val, kornia_tensor::host_alloc())
            }

            #[doc = concat!("Like `from_size_val` but with an explicit allocator handle")]
            pub fn from_size_val_in(
                size: ImageSize,
                val: $type,
                alloc: AllocHandle,
            ) -> Result<Self, ImageError> {
                Ok(Self(Image::from_size_val_in(size, val, alloc)?))
            }

            /// Unwrap into the underlying Image
            pub fn into_inner(self) -> Image<$type, $channels> {
                self.0
            }

            /// Get a reference to the underlying Image
            pub fn as_image(&self) -> &Image<$type, $channels> {
                &self.0
            }

            /// Get a mutable reference to the underlying Image
            pub fn as_image_mut(&mut self) -> &mut Image<$type, $channels> {
                &mut self.0
            }

            #[cfg(feature = "cuda")]
            #[doc = concat!("Upload to a device-resident ", stringify!($name), " (H2D copy).")]
            ///
            /// The result stays typed, so it flows through the same `ConvertColor`
            /// APIs as host images — device operands dispatch to CUDA kernels.
            pub fn to_cuda(
                &self,
                stream: &std::sync::Arc<cudarc::driver::CudaStream>,
            ) -> Result<Self, ImageError> {
                Ok(Self(self.0.to_cuda(stream)?))
            }

            #[cfg(feature = "cuda")]
            #[doc = concat!("Allocate a zero-initialised device-resident ", stringify!($name), ".")]
            pub fn zeros_cuda(
                size: ImageSize,
                stream: &std::sync::Arc<cudarc::driver::CudaStream>,
            ) -> Result<Self, ImageError> {
                Ok(Self(Image::zeros_cuda(size, stream)?))
            }

            #[cfg(feature = "cuda")]
            #[doc = concat!("Allocate a zero-initialised pinned-memory host ", stringify!($name), ".")]
            pub fn zeros_pinned(
                size: ImageSize,
                ctx: &std::sync::Arc<cudarc::driver::CudaContext>,
            ) -> Result<Self, ImageError> {
                Ok(Self(Image::zeros_pinned(size, ctx)?))
            }

            #[cfg(feature = "cuda")]
            #[doc = concat!("Copy a device-resident ", stringify!($name), " to a new, owned host ", stringify!($name), " on its own carried stream.")]
            pub fn to_host_owned(&self) -> Result<Self, ImageError> {
                Ok(Self(self.0.to_host_owned()?))
            }

            #[cfg(feature = "cuda")]
            #[doc = concat!("Copy a device-resident ", stringify!($name), " back to host (D2H).")]
            pub fn to_host(
                &self,
                stream: &std::sync::Arc<cudarc::driver::CudaStream>,
            ) -> Result<Self, ImageError> {
                Ok(Self(self.0.to_host_image(stream)?))
            }
        }

        impl Deref for $name {
            type Target = Image<$type, $channels>;
            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }

        impl DerefMut for $name {
            fn deref_mut(&mut self) -> &mut Self::Target {
                &mut self.0
            }
        }

        impl AsRef<Image<$type, $channels>> for $name {
            fn as_ref(&self) -> &Image<$type, $channels> {
                &self.0
            }
        }
    };
}

// Define explicit color space types with bit depths
define_color_space!(
    Rgb8,
    u8,
    3,
    "RGB color space with 8-bit unsigned integer channels"
);
define_color_space!(
    Rgb16,
    u16,
    3,
    "RGB color space with 16-bit unsigned integer channels"
);
define_color_space!(
    Rgbf32,
    f32,
    3,
    "RGB color space with 32-bit floating point channels"
);
define_color_space!(
    Rgbf64,
    f64,
    3,
    "RGB color space with 64-bit floating point channels"
);

define_color_space!(
    Bgr8,
    u8,
    3,
    "BGR color space with 8-bit unsigned integer channels"
);
define_color_space!(
    Bgr16,
    u16,
    3,
    "BGR color space with 16-bit unsigned integer channels"
);
define_color_space!(
    Bgrf32,
    f32,
    3,
    "BGR color space with 32-bit floating point channels"
);
define_color_space!(
    Bgrf64,
    f64,
    3,
    "BGR color space with 64-bit floating point channels"
);

define_color_space!(
    Gray8,
    u8,
    1,
    "Grayscale with 8-bit unsigned integer channels"
);
define_color_space!(
    Gray16,
    u16,
    1,
    "Grayscale with 16-bit unsigned integer channels"
);
define_color_space!(
    Grayf32,
    f32,
    1,
    "Grayscale with 32-bit floating point channels"
);
define_color_space!(
    Grayf64,
    f64,
    1,
    "Grayscale with 64-bit floating point channels"
);

define_color_space!(
    Rgba8,
    u8,
    4,
    "RGBA color space with 8-bit unsigned integer channels"
);
define_color_space!(
    Rgba16,
    u16,
    4,
    "RGBA color space with 16-bit unsigned integer channels"
);
define_color_space!(
    Rgbaf32,
    f32,
    4,
    "RGBA color space with 32-bit floating point channels"
);
define_color_space!(
    Rgbaf64,
    f64,
    4,
    "RGBA color space with 64-bit floating point channels"
);

define_color_space!(
    Bgra8,
    u8,
    4,
    "BGRA color space with 8-bit unsigned integer channels"
);
define_color_space!(
    Bgra16,
    u16,
    4,
    "BGRA color space with 16-bit unsigned integer channels"
);
define_color_space!(
    Bgraf32,
    f32,
    4,
    "BGRA color space with 32-bit floating point channels"
);
define_color_space!(
    Bgraf64,
    f64,
    4,
    "BGRA color space with 64-bit floating point channels"
);

define_color_space!(
    Hsvf32,
    f32,
    3,
    "HSV color space with 32-bit floating point channels"
);
define_color_space!(
    Hsvf64,
    f64,
    3,
    "HSV color space with 64-bit floating point channels"
);

define_color_space!(
    Hlsf32,
    f32,
    3,
    "HLS (hue, lightness, saturation) color space with 32-bit floating point channels"
);
define_color_space!(
    Hlsf64,
    f64,
    3,
    "HLS (hue, lightness, saturation) color space with 64-bit floating point channels"
);

define_color_space!(
    Xyzf32,
    f32,
    3,
    "CIE XYZ color space with 32-bit floating point channels"
);
define_color_space!(
    Xyzf64,
    f64,
    3,
    "CIE XYZ color space with 64-bit floating point channels"
);

define_color_space!(
    Labf32,
    f32,
    3,
    "CIE L*a*b* color space with 32-bit floating point channels"
);
define_color_space!(
    Labf64,
    f64,
    3,
    "CIE L*a*b* color space with 64-bit floating point channels"
);

define_color_space!(
    Luvf32,
    f32,
    3,
    "CIE L*u*v* color space with 32-bit floating point channels"
);
define_color_space!(
    Luvf64,
    f64,
    3,
    "CIE L*u*v* color space with 64-bit floating point channels"
);

define_color_space!(
    YCbCr8,
    u8,
    3,
    "YCbCr (a.k.a. YCrCb in OpenCV) color space with 8-bit unsigned integer channels. Channel order [Y, Cr, Cb], full range."
);
define_color_space!(
    YCbCrf32,
    f32,
    3,
    "YCbCr color space with 32-bit floating point channels"
);
define_color_space!(
    YCbCrf64,
    f64,
    3,
    "YCbCr color space with 64-bit floating point channels"
);

define_color_space!(
    Yuv8,
    u8,
    3,
    "YUV (planar, full-range) color space with 8-bit unsigned integer channels. Channel order [Y, U=Cb, V=Cr]."
);
define_color_space!(
    Yuvf32,
    f32,
    3,
    "YUV color space (planar, full-range) with 32-bit floating point channels"
);
define_color_space!(
    Yuvf64,
    f64,
    3,
    "YUV color space (planar, full-range) with 64-bit floating point channels"
);

// ===== Packed / planar video formats (hand-written; don't fit define_color_space!) ====
//
// These store luma and (subsampled) chroma in a single byte buffer with non-trivial
// layouts, so they can't be a thin `Image<T, C>` wrapper. Each carries an `ImageSize`
// and a `Vec<u8>`, and validates its length on construction.
//
// HOST/CPU-ONLY: unlike `Image<T, C>` these own a plain `Vec<u8>` and carry no
// `AllocHandle`, so they cannot live in device or custom-allocator memory. They are
// interchange formats decoded to `Image`/`DynImage` before any allocator-aware op.

/// Packed 4:2:2 YUYV (a.k.a. YUY2): byte order `Y0 U Y1 V` per 2-pixel group.
///
/// Data length is `width * height * 2`. Width must be even.
pub struct Yuyv8 {
    size: ImageSize,
    data: Vec<u8>,
}

/// Packed 4:2:2 UYVY: byte order `U Y0 V Y1` per 2-pixel group.
///
/// Data length is `width * height * 2`. Width must be even.
pub struct Uyvy8 {
    size: ImageSize,
    data: Vec<u8>,
}

/// Packed 4:2:2 YVYU: byte order `Y0 V Y1 U` per 2-pixel group.
///
/// Data length is `width * height * 2`. Width must be even.
pub struct Yvyu8 {
    size: ImageSize,
    data: Vec<u8>,
}

macro_rules! define_packed_422 {
    ($name:ident, $doc:expr_2021) => {
        impl $name {
            #[doc = concat!("Create a ", stringify!($name), " buffer from a size and a packed 4:2:2 byte vector (len = width*height*2).")]
            pub fn from_size_vec(size: ImageSize, data: Vec<u8>) -> Result<Self, ImageError> {
                let expected = size.width * size.height * 2;
                if data.len() != expected || size.width % 2 != 0 {
                    return Err(ImageError::InvalidImageSize(
                        data.len(),
                        size.width,
                        size.height,
                        expected,
                    ));
                }
                Ok(Self { size, data })
            }

            /// Image size (width, height in luma pixels).
            pub fn size(&self) -> ImageSize {
                self.size
            }

            /// Image width in luma pixels.
            pub fn width(&self) -> usize {
                self.size.width
            }

            /// Image height in luma pixels.
            pub fn height(&self) -> usize {
                self.size.height
            }

            /// The packed byte buffer.
            pub fn as_slice(&self) -> &[u8] {
                &self.data
            }

            /// Consume and return the underlying byte buffer.
            pub fn into_vec(self) -> Vec<u8> {
                self.data
            }
        }
    };
}

define_packed_422!(Yuyv8, "YUYV");
define_packed_422!(Uyvy8, "UYVY");
define_packed_422!(Yvyu8, "YVYU");

/// Planar 4:2:0 NV12: full-res Y plane followed by interleaved `UV` chroma (half res).
///
/// Data length is `width * height * 3 / 2`. Width and height must be even.
pub struct Nv12 {
    size: ImageSize,
    data: Vec<u8>,
}

/// Planar 4:2:0 NV21: full-res Y plane followed by interleaved `VU` chroma (half res).
///
/// Data length is `width * height * 3 / 2`. Width and height must be even.
pub struct Nv21 {
    size: ImageSize,
    data: Vec<u8>,
}

/// Planar 4:2:0 I420 (a.k.a. IYUV): Y plane, then U plane, then V plane.
///
/// Data length is `width * height * 3 / 2`. Width and height must be even.
pub struct I420 {
    size: ImageSize,
    data: Vec<u8>,
}

/// Planar 4:2:0 YV12: Y plane, then V plane, then U plane.
///
/// Data length is `width * height * 3 / 2`. Width and height must be even.
pub struct Yv12 {
    size: ImageSize,
    data: Vec<u8>,
}

macro_rules! define_planar_420 {
    ($name:ident, $doc:expr_2021) => {
        impl $name {
            #[doc = concat!("Create a ", stringify!($name), " buffer from a size and a planar 4:2:0 byte vector (len = width*height*3/2).")]
            pub fn from_size_vec(size: ImageSize, data: Vec<u8>) -> Result<Self, ImageError> {
                let expected = size.width * size.height * 3 / 2;
                if data.len() != expected || size.width % 2 != 0 || size.height % 2 != 0 {
                    return Err(ImageError::InvalidImageSize(
                        data.len(),
                        size.width,
                        size.height,
                        expected,
                    ));
                }
                Ok(Self { size, data })
            }

            /// Image size (width, height in luma pixels).
            pub fn size(&self) -> ImageSize {
                self.size
            }

            /// Image width in luma pixels.
            pub fn width(&self) -> usize {
                self.size.width
            }

            /// Image height in luma pixels.
            pub fn height(&self) -> usize {
                self.size.height
            }

            /// The full byte buffer (Y plane followed by chroma).
            pub fn as_slice(&self) -> &[u8] {
                &self.data
            }

            /// The luma (Y) plane: the first `width * height` bytes.
            pub fn y_plane(&self) -> &[u8] {
                &self.data[..self.size.width * self.size.height]
            }

            /// Consume and return the underlying byte buffer.
            pub fn into_vec(self) -> Vec<u8> {
                self.data
            }
        }
    };
}

define_planar_420!(Nv12, "NV12");
define_planar_420!(Nv21, "NV21");
define_planar_420!(I420, "I420");
define_planar_420!(Yv12, "YV12");

impl Nv12 {
    /// Interleaved `UV` chroma plane (half-res), the second part of the buffer.
    pub fn uv_plane(&self) -> &[u8] {
        &self.data[self.size.width * self.size.height..]
    }
}

impl Nv21 {
    /// Interleaved `VU` chroma plane (half-res), the second part of the buffer.
    pub fn uv_plane(&self) -> &[u8] {
        &self.data[self.size.width * self.size.height..]
    }
}

impl I420 {
    /// The U plane: bytes `[W*H, W*H + W*H/4)`.
    pub fn u_plane(&self) -> &[u8] {
        let n = self.size.width * self.size.height;
        &self.data[n..n + n / 4]
    }

    /// The V plane: bytes `[W*H + W*H/4, end)`.
    pub fn v_plane(&self) -> &[u8] {
        let n = self.size.width * self.size.height;
        &self.data[n + n / 4..]
    }
}

impl Yv12 {
    /// The V plane: bytes `[W*H, W*H + W*H/4)`.
    pub fn v_plane(&self) -> &[u8] {
        let n = self.size.width * self.size.height;
        &self.data[n..n + n / 4]
    }

    /// The U plane: bytes `[W*H + W*H/4, end)`.
    pub fn u_plane(&self) -> &[u8] {
        let n = self.size.width * self.size.height;
        &self.data[n + n / 4..]
    }
}

define_color_space!(
    LinearRgbf32,
    f32,
    3,
    "Linear (gamma-expanded) RGB color space with 32-bit floating point channels"
);
define_color_space!(
    LinearRgbf64,
    f64,
    3,
    "Linear (gamma-expanded) RGB color space with 64-bit floating point channels"
);

// ===== Bayer mosaic (hand-written; carries a runtime pattern field) ====================

/// Bayer mosaic pattern, named by the color at the 2×2 sensor cell's positions
/// `(row%2, col%2)` for `(0,0), (0,1), (1,0), (1,1)`.
///
/// e.g. [`BayerPattern::Rggb`] = R at (0,0), G at (0,1), G at (1,0), B at (1,1).
///
/// Note the OpenCV naming offset: kornia [`BayerPattern::Rggb`] corresponds to
/// `cv2.COLOR_BayerBG2RGB` (OpenCV names its Bayer codes by the opposite corner
/// of the 2×2 cell).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BayerPattern {
    /// R, G / G, B
    Rggb,
    /// B, G / G, R
    Bggr,
    /// G, R / B, G
    Grbg,
    /// G, B / R, G
    Gbrg,
}

/// Single-channel Bayer-mosaic image (8-bit) tagged with its [`BayerPattern`].
///
/// Hand-written rather than via `define_color_space!` because it carries the
/// `pattern` field alongside the underlying single-channel image.
pub struct Bayer8 {
    image: Image<u8, 1>,
    /// The mosaic pattern of `image`.
    pub pattern: BayerPattern,
}

impl Bayer8 {
    /// Create a Bayer8 image from a size, raw mosaic data, and a pattern.
    pub fn from_size_vec(
        size: ImageSize,
        data: Vec<u8>,
        pattern: BayerPattern,
        alloc: AllocHandle,
    ) -> Result<Self, ImageError> {
        Ok(Self {
            image: Image::new_in(size, data, alloc)?,
            pattern,
        })
    }

    /// Get a reference to the underlying single-channel mosaic image.
    pub fn as_image(&self) -> &Image<u8, 1> {
        &self.image
    }

    /// Unwrap into the underlying single-channel mosaic image.
    pub fn into_inner(self) -> Image<u8, 1> {
        self.image
    }
}

impl Deref for Bayer8 {
    type Target = Image<u8, 1>;
    fn deref(&self) -> &Self::Target {
        &self.image
    }
}

#[cfg(test)]
mod tests {
    use super::ColorSpace;

    #[test]
    fn channels_and_dtype_metadata() {
        assert_eq!(ColorSpace::Gray.channels(), 1);
        assert_eq!(ColorSpace::Rgb.channels(), 3);
        assert_eq!(ColorSpace::Rgba.channels(), 4);
        assert!(ColorSpace::Hsv.requires_f32());
        assert!(!ColorSpace::Gray.requires_f32());
        assert!(!ColorSpace::Bgr.requires_f32());
    }

    #[test]
    fn legality_table_matches_rgb_hub() {
        assert!(ColorSpace::supports(ColorSpace::Rgb, ColorSpace::Hsv));
        assert!(ColorSpace::supports(ColorSpace::Hsv, ColorSpace::Rgb));
        assert!(ColorSpace::supports(ColorSpace::Rgb, ColorSpace::Gray));
        // non-adjacent pair is rejected
        assert!(!ColorSpace::supports(ColorSpace::Hsv, ColorSpace::Lab));
        assert!(!ColorSpace::supports(ColorSpace::Gray, ColorSpace::Hsv));
    }

    use crate::color_spaces::{Grayf32, Rgbf32};
    use crate::{ColorSpace as CS, DynImage, ImageSize};
    use std::convert::TryFrom;

    #[test]
    fn dyn_image_tag_size_and_recovery() {
        let size = ImageSize {
            width: 2,
            height: 2,
        };
        let rgb = Rgbf32::from_size_val(size, 0.25).unwrap();
        let dynimg = DynImage::C3(CS::Rgb, rgb.into_inner());
        assert_eq!(dynimg.color_space(), CS::Rgb);
        assert_eq!(dynimg.channels(), 3);
        assert_eq!(dynimg.size(), size);
        // typed recovery succeeds for matching space+channels
        let back: Rgbf32 = Rgbf32::try_from(dynimg).unwrap();
        assert_eq!(back.as_slice()[0], 0.25);
    }

    #[test]
    fn dyn_image_recovery_rejects_wrong_space() {
        let size = ImageSize {
            width: 2,
            height: 2,
        };
        let gray = Grayf32::from_size_val(size, 0.0).unwrap();
        let dynimg = DynImage::C1(CS::Gray, gray.into_inner());
        // recovering as Rgbf32 must fail (channel mismatch C1 vs C3)
        assert!(Rgbf32::try_from(dynimg).is_err());
    }

    /// Self-consistency: `requires_f32`, `channels`, and `supports_dtype` must
    /// agree for the full RGB-hub set of spaces.
    #[test]
    fn legality_table_self_consistency() {
        use ColorSpace::*;

        // --- requires_f32 membership ---
        let f32_only = [Hsv, Hls, Lab, Luv, Xyz, LinearRgb];
        let non_f32_only = [Rgb, Bgr, Gray, Rgba, Bgra, YCbCr, Yuv];

        for &s in &f32_only {
            assert!(s.requires_f32(), "{s:?} should require f32");
        }
        for &s in &non_f32_only {
            assert!(!s.requires_f32(), "{s:?} should not require f32");
        }

        // --- channel counts ---
        assert_eq!(Gray.channels(), 1);
        for &s in &[Rgb, Bgr, Hsv, Hls, Lab, Luv, Xyz, LinearRgb, YCbCr, Yuv] {
            assert_eq!(s.channels(), 3, "{s:?} should be 3-channel");
        }
        for &s in &[Rgba, Bgra] {
            assert_eq!(s.channels(), 4, "{s:?} should be 4-channel");
        }

        // --- supports_dtype agrees with supports + requires_f32 ---
        // f32-only conversions are valid for f32, invalid for u8
        assert!(ColorSpace::supports_dtype(Rgb, Hsv, true));
        assert!(!ColorSpace::supports_dtype(Rgb, Hsv, false));
        assert!(ColorSpace::supports_dtype(Lab, Rgb, true));
        assert!(!ColorSpace::supports_dtype(Lab, Rgb, false));

        // dtype-agnostic conversions are valid for both
        assert!(ColorSpace::supports_dtype(Rgb, Gray, true));
        assert!(ColorSpace::supports_dtype(Rgb, Gray, false));
        assert!(ColorSpace::supports_dtype(Rgb, Bgr, true));
        assert!(ColorSpace::supports_dtype(Rgb, Bgr, false));

        // unsupported pairs rejected regardless of dtype
        assert!(!ColorSpace::supports_dtype(Hsv, Lab, true));
        assert!(!ColorSpace::supports_dtype(Hsv, Lab, false));
        assert!(!ColorSpace::supports_dtype(Gray, Hsv, true));
    }
}
