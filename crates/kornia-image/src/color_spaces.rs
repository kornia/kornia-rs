use crate::{
    allocator::ImageAllocator,
    error::ImageError,
    image::{Image, ImageSize},
};
use std::ops::{Deref, DerefMut};

/// Macro to define a color space wrapper type with explicit bit depth
macro_rules! define_color_space {
    ($name:ident, $type:ty, $channels:expr, $doc:expr) => {
        #[doc = $doc]
        ///
        /// This is a zero-cost wrapper that provides compile-time type safety.
        #[repr(transparent)]
        pub struct $name<A: ImageAllocator>(pub Image<$type, $channels, A>);

        impl<A: ImageAllocator> $name<A> {
            #[doc = concat!("Create ", stringify!($name), " image from size and data")]
            pub fn from_size_vec(
                size: ImageSize,
                data: Vec<$type>,
                alloc: A,
            ) -> Result<Self, ImageError> {
                Ok(Self(Image::new(size, data, alloc)?))
            }

            #[doc = concat!("Create ", stringify!($name), " image from size with default value")]
            pub fn from_size_val(
                size: ImageSize,
                val: $type,
                alloc: A,
            ) -> Result<Self, ImageError> {
                Ok(Self(Image::from_size_val(size, val, alloc)?))
            }

            /// Unwrap into the underlying Image
            pub fn into_inner(self) -> Image<$type, $channels, A> {
                self.0
            }

            /// Get a reference to the underlying Image
            pub fn as_image(&self) -> &Image<$type, $channels, A> {
                &self.0
            }

            /// Get a mutable reference to the underlying Image
            pub fn as_image_mut(&mut self) -> &mut Image<$type, $channels, A> {
                &mut self.0
            }
        }

        impl<A: ImageAllocator> Deref for $name<A> {
            type Target = Image<$type, $channels, A>;
            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }

        impl<A: ImageAllocator> DerefMut for $name<A> {
            fn deref_mut(&mut self) -> &mut Self::Target {
                &mut self.0
            }
        }

        impl<A: ImageAllocator> AsRef<Image<$type, $channels, A>> for $name<A> {
            fn as_ref(&self) -> &Image<$type, $channels, A> {
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
    ($name:ident, $doc:expr) => {
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
    ($name:ident, $doc:expr) => {
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
pub struct Bayer8<A: ImageAllocator> {
    image: Image<u8, 1, A>,
    /// The mosaic pattern of `image`.
    pub pattern: BayerPattern,
}

impl<A: ImageAllocator> Bayer8<A> {
    /// Create a Bayer8 image from a size, raw mosaic data, and a pattern.
    pub fn from_size_vec(
        size: ImageSize,
        data: Vec<u8>,
        pattern: BayerPattern,
        alloc: A,
    ) -> Result<Self, ImageError> {
        Ok(Self {
            image: Image::new(size, data, alloc)?,
            pattern,
        })
    }

    /// Get a reference to the underlying single-channel mosaic image.
    pub fn as_image(&self) -> &Image<u8, 1, A> {
        &self.image
    }

    /// Unwrap into the underlying single-channel mosaic image.
    pub fn into_inner(self) -> Image<u8, 1, A> {
        self.image
    }
}

impl<A: ImageAllocator> Deref for Bayer8<A> {
    type Target = Image<u8, 1, A>;
    fn deref(&self) -> &Self::Target {
        &self.image
    }
}
