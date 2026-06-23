//! Runtime color-space vocabulary shared by Rust and Python.

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
    pub const fn channels(self) -> usize {
        match self {
            ColorSpace::Gray => 1,
            ColorSpace::Rgba | ColorSpace::Bgra => 4,
            _ => 3,
        }
    }

    /// True for spaces whose kernels only operate on f32 data.
    pub const fn requires_f32(self) -> bool {
        matches!(
            self,
            ColorSpace::Hsv | ColorSpace::Hls | ColorSpace::Lab
                | ColorSpace::Luv | ColorSpace::Xyz | ColorSpace::LinearRgb
        )
    }

    /// Whether a direct kernel exists for `from -> to`. Mirrors the
    /// `ConvertColor` impls in kornia-imgproc (RGB-hub graph).
    pub const fn supports(from: ColorSpace, to: ColorSpace) -> bool {
        use ColorSpace::*;
        matches!(
            (from, to),
            (Rgb, Gray) | (Gray, Rgb)
                | (Rgb, Bgr) | (Bgr, Rgb)
                | (Rgb, Rgba) | (Rgba, Rgb)
                | (Rgb, Bgra) | (Bgra, Rgb)
                | (Rgb, Hsv) | (Hsv, Rgb)
                | (Rgb, Hls) | (Hls, Rgb)
                | (Rgb, Lab) | (Lab, Rgb)
                | (Rgb, Luv) | (Luv, Rgb)
                | (Rgb, Xyz) | (Xyz, Rgb)
                | (Rgb, LinearRgb) | (LinearRgb, Rgb)
                | (Rgb, YCbCr) | (YCbCr, Rgb)
                | (Rgb, Yuv) | (Yuv, Rgb)
        )
    }
}

use crate::{allocator::ImageAllocator, error::ImageError, image::Image, image::ImageSize};

/// Owned image whose channel count is known only at runtime. Mirrors Python's
/// dynamic numpy-backed Image; produced by the runtime `cvt_color` path.
pub enum DynImage<T, A: ImageAllocator> {
    /// 1-channel (e.g. Gray).
    C1(ColorSpace, Image<T, 1, A>),
    /// 3-channel (Rgb/Bgr/Hsv/...).
    C3(ColorSpace, Image<T, 3, A>),
    /// 4-channel (Rgba/Bgra).
    C4(ColorSpace, Image<T, 4, A>),
}

impl<T, A: ImageAllocator> DynImage<T, A> {
    /// The color space tag carried by this image.
    pub fn color_space(&self) -> ColorSpace {
        match self {
            DynImage::C1(s, _) | DynImage::C3(s, _) | DynImage::C4(s, _) => *s,
        }
    }

    /// Image dimensions.
    pub fn size(&self) -> ImageSize {
        match self {
            DynImage::C1(_, i) => i.size(),
            DynImage::C3(_, i) => i.size(),
            DynImage::C4(_, i) => i.size(),
        }
    }

    /// Channel count (1, 3, or 4).
    pub fn channels(&self) -> usize {
        match self {
            DynImage::C1(..) => 1,
            DynImage::C3(..) => 3,
            DynImage::C4(..) => 4,
        }
    }

    /// Contiguous (H, W, C) row-major data.
    pub fn as_slice(&self) -> &[T] {
        match self {
            DynImage::C1(_, i) => i.as_slice(),
            DynImage::C3(_, i) => i.as_slice(),
            DynImage::C4(_, i) => i.as_slice(),
        }
    }
}

macro_rules! impl_try_from_dyn {
    ($newtype:ident, $t:ty, C1, $space:expr) => {
        impl<A: ImageAllocator> std::convert::TryFrom<DynImage<$t, A>>
            for crate::color_spaces::$newtype<A>
        {
            type Error = ImageError;
            fn try_from(d: DynImage<$t, A>) -> Result<Self, ImageError> {
                match d {
                    DynImage::C1(s, img) if s == $space => Ok(Self(img)),
                    other => Err(ImageError::UnsupportedColorConversion {
                        from: other.color_space(), to: $space,
                    }),
                }
            }
        }
    };
    ($newtype:ident, $t:ty, C3, $space:expr) => {
        impl<A: ImageAllocator> std::convert::TryFrom<DynImage<$t, A>>
            for crate::color_spaces::$newtype<A>
        {
            type Error = ImageError;
            fn try_from(d: DynImage<$t, A>) -> Result<Self, ImageError> {
                match d {
                    DynImage::C3(s, img) if s == $space => Ok(Self(img)),
                    other => Err(ImageError::UnsupportedColorConversion {
                        from: other.color_space(), to: $space,
                    }),
                }
            }
        }
    };
    ($newtype:ident, $t:ty, C4, $space:expr) => {
        impl<A: ImageAllocator> std::convert::TryFrom<DynImage<$t, A>>
            for crate::color_spaces::$newtype<A>
        {
            type Error = ImageError;
            fn try_from(d: DynImage<$t, A>) -> Result<Self, ImageError> {
                match d {
                    DynImage::C4(s, img) if s == $space => Ok(Self(img)),
                    other => Err(ImageError::UnsupportedColorConversion {
                        from: other.color_space(), to: $space,
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

    use crate::allocator::CpuAllocator;
    use crate::color_spaces::{Grayf32, Rgbf32};
    use crate::{ColorSpace as CS, DynImage, ImageSize};
    use std::convert::TryFrom;

    #[test]
    fn dyn_image_tag_size_and_recovery() {
        let size = ImageSize { width: 2, height: 2 };
        let rgb = Rgbf32::from_size_val(size, 0.25, CpuAllocator).unwrap();
        let dynimg = DynImage::C3(CS::Rgb, rgb.into_inner());
        assert_eq!(dynimg.color_space(), CS::Rgb);
        assert_eq!(dynimg.channels(), 3);
        assert_eq!(dynimg.size(), size);
        // typed recovery succeeds for matching space+channels
        let back: Rgbf32<_> = Rgbf32::try_from(dynimg).unwrap();
        assert_eq!(back.as_slice()[0], 0.25);
    }

    #[test]
    fn dyn_image_recovery_rejects_wrong_space() {
        let size = ImageSize { width: 2, height: 2 };
        let gray = Grayf32::from_size_val(size, 0.0, CpuAllocator).unwrap();
        let dynimg = DynImage::C1(CS::Gray, gray.into_inner());
        // recovering as Rgbf32 must fail (channel mismatch C1 vs C3)
        assert!(Rgbf32::try_from(dynimg).is_err());
    }
}
