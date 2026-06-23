use kornia_image::{
    allocator::ImageAllocator,
    color_spaces::{
        Bgr8, Bgra8, Bgraf32, Bgrf32, Gray8, Grayf32, Grayf64, Hlsf32, Hlsf64, Hsvf32, Hsvf64,
        Labf32, Labf64, LinearRgbf32, LinearRgbf64, Luvf32, Luvf64, Nv12, Nv21, Rgb8, Rgba8,
        Rgbaf32, Rgbf32, Rgbf64, Uyvy8, Xyzf32, Xyzf64, YCbCr8, YCbCrf32, YCbCrf64, Yuv8, Yuvf32,
        Yuvf64, Yuyv8, Yv12, Yvyu8, I420,
    },
    ImageError, ImageSize,
};

/// Trait for type-safe color space conversion
///
/// This trait provides a clean, ergonomic API for converting between different
/// color spaces with compile-time type safety.
///
/// # Example
///
/// ```
/// use kornia_image::ImageSize;
/// use kornia_image::allocator::CpuAllocator;
/// use kornia_imgproc::color::{Rgbf32, Grayf32, ConvertColor};
///
/// let rgb = Rgbf32::from_size_vec(
///     ImageSize { width: 4, height: 5 },
///     vec![0.5f32; 4 * 5 * 3],
///     CpuAllocator
/// ).unwrap();
///
/// let mut gray = Grayf32::from_size_val(rgb.size(), 0.0, CpuAllocator).unwrap();
///
/// rgb.convert(&mut gray).unwrap();
/// ```
pub trait ConvertColor<Dst> {
    /// Convert this image to another color space
    fn convert(&self, dst: &mut Dst) -> Result<(), ImageError>;
}

/// Macro to implement color conversions
macro_rules! impl_convert {
    ($src:ty => $dst:ty, $func:path) => {
        impl<A1, A2> ConvertColor<$dst> for $src
        where
            A1: ImageAllocator,
            A2: ImageAllocator,
        {
            fn convert(&self, dst: &mut $dst) -> Result<(), ImageError> {
                $func(&self.0, &mut dst.0)
            }
        }
    };

    // For conversions with optional parameters (like background)
    ($src:ty => $dst:ty, $func:path, bg: $bg:expr) => {
        impl<A1, A2> ConvertColor<$dst> for $src
        where
            A1: ImageAllocator,
            A2: ImageAllocator,
        {
            fn convert(&self, dst: &mut $dst) -> Result<(), ImageError> {
                $func(&self.0, &mut dst.0, $bg)
            }
        }
    };
}

/// Macro to implement color conversions with background support
macro_rules! impl_convert_with_bg {
    ($src:ty => $dst:ty, $func:path) => {
        impl<A1, A2> ConvertColorWithBackground<$dst> for $src
        where
            A1: ImageAllocator,
            A2: ImageAllocator,
        {
            fn convert_with_bg(
                &self,
                dst: &mut $dst,
                bg: Option<[u8; 3]>,
            ) -> Result<(), ImageError> {
                $func(&self.0, &mut dst.0, bg)
            }
        }
    };
}

// ===== Bayer mosaic -> RGB (hand-written: reads the runtime pattern field) =====
impl<A1, A2> ConvertColor<Rgb8<A2>> for kornia_image::color_spaces::Bayer8<A1>
where
    A1: ImageAllocator,
    A2: ImageAllocator,
{
    fn convert(&self, dst: &mut Rgb8<A2>) -> Result<(), ImageError> {
        crate::color::rgb_from_bayer(self.as_image(), self.pattern, &mut dst.0)
    }
}

// ===== RGB -> Gray Conversions =====
impl_convert!(Rgbf32<A1> => Grayf32<A2>, crate::color::gray_from_rgb_f32);
impl_convert!(Rgbf64<A1> => Grayf64<A2>, crate::color::gray_from_rgb);
impl_convert!(Rgb8<A1> => Gray8<A2>, crate::color::gray_from_rgb_u8);

// ===== Gray -> RGB Conversions =====
impl_convert!(Gray8<A1> => Rgb8<A2>, crate::color::rgb_from_gray);
impl_convert!(Grayf32<A1> => Rgbf32<A2>, crate::color::rgb_from_gray);
impl_convert!(Grayf64<A1> => Rgbf64<A2>, crate::color::rgb_from_gray);

// ===== RGB <-> BGR Conversions =====
impl_convert!(Rgb8<A1> => Bgr8<A2>, crate::color::bgr_from_rgb);
impl_convert!(Bgr8<A1> => Rgb8<A2>, crate::color::bgr_from_rgb);
impl_convert!(Rgbf32<A1> => Bgrf32<A2>, crate::color::bgr_from_rgb);
impl_convert!(Bgrf32<A1> => Rgbf32<A2>, crate::color::bgr_from_rgb);

// ===== RGB <-> HSV Conversions =====
impl_convert!(Rgbf32<A1> => Hsvf32<A2>, crate::color::hsv_from_rgb);
impl_convert!(Hsvf32<A1> => Rgbf32<A2>, crate::color::rgb_from_hsv);
impl_convert!(Rgbf64<A1> => Hsvf64<A2>, crate::color::hsv_from_rgb);
impl_convert!(Hsvf64<A1> => Rgbf64<A2>, crate::color::rgb_from_hsv);

// ===== RGB <-> HLS Conversions =====
impl_convert!(Rgbf32<A1> => Hlsf32<A2>, crate::color::hls_from_rgb);
impl_convert!(Hlsf32<A1> => Rgbf32<A2>, crate::color::rgb_from_hls);
impl_convert!(Rgbf64<A1> => Hlsf64<A2>, crate::color::hls_from_rgb);
impl_convert!(Hlsf64<A1> => Rgbf64<A2>, crate::color::rgb_from_hls);

// ===== RGB <-> linear-RGB (sRGB transfer) =====
impl_convert!(Rgbf32<A1> => LinearRgbf32<A2>, crate::color::linear_rgb_from_rgb);
impl_convert!(LinearRgbf32<A1> => Rgbf32<A2>, crate::color::rgb_from_linear_rgb);
impl_convert!(Rgbf64<A1> => LinearRgbf64<A2>, crate::color::linear_rgb_from_rgb);
impl_convert!(LinearRgbf64<A1> => Rgbf64<A2>, crate::color::rgb_from_linear_rgb);

// ===== RGB <-> XYZ Conversions =====
impl_convert!(Rgbf32<A1> => Xyzf32<A2>, crate::color::xyz_from_rgb);
impl_convert!(Xyzf32<A1> => Rgbf32<A2>, crate::color::rgb_from_xyz);
impl_convert!(Rgbf64<A1> => Xyzf64<A2>, crate::color::xyz_from_rgb);
impl_convert!(Xyzf64<A1> => Rgbf64<A2>, crate::color::rgb_from_xyz);

// ===== RGB <-> Lab Conversions =====
impl_convert!(Rgbf32<A1> => Labf32<A2>, crate::color::lab_from_rgb);
impl_convert!(Labf32<A1> => Rgbf32<A2>, crate::color::rgb_from_lab);
impl_convert!(Rgbf64<A1> => Labf64<A2>, crate::color::lab_from_rgb);
impl_convert!(Labf64<A1> => Rgbf64<A2>, crate::color::rgb_from_lab);

// ===== RGB <-> Luv Conversions =====
impl_convert!(Rgbf32<A1> => Luvf32<A2>, crate::color::luv_from_rgb);
impl_convert!(Luvf32<A1> => Rgbf32<A2>, crate::color::rgb_from_luv);
impl_convert!(Rgbf64<A1> => Luvf64<A2>, crate::color::luv_from_rgb);
impl_convert!(Luvf64<A1> => Rgbf64<A2>, crate::color::rgb_from_luv);

// ===== RGB <-> YCbCr Conversions =====
impl_convert!(Rgb8<A1> => YCbCr8<A2>, crate::color::ycbcr_from_rgb);
impl_convert!(YCbCr8<A1> => Rgb8<A2>, crate::color::rgb_from_ycbcr);
impl_convert!(Rgbf32<A1> => YCbCrf32<A2>, crate::color::ycbcr_from_rgb);
impl_convert!(YCbCrf32<A1> => Rgbf32<A2>, crate::color::rgb_from_ycbcr);
impl_convert!(Rgbf64<A1> => YCbCrf64<A2>, crate::color::ycbcr_from_rgb);
impl_convert!(YCbCrf64<A1> => Rgbf64<A2>, crate::color::rgb_from_ycbcr);

// ===== RGB <-> YUV (planar 3-channel) Conversions =====
impl_convert!(Rgb8<A1> => Yuv8<A2>, crate::color::yuv_from_rgb);
impl_convert!(Yuv8<A1> => Rgb8<A2>, crate::color::rgb_from_yuv);
impl_convert!(Rgbf32<A1> => Yuvf32<A2>, crate::color::yuv_from_rgb);
impl_convert!(Yuvf32<A1> => Rgbf32<A2>, crate::color::rgb_from_yuv);
impl_convert!(Rgbf64<A1> => Yuvf64<A2>, crate::color::yuv_from_rgb);
impl_convert!(Yuvf64<A1> => Rgbf64<A2>, crate::color::rgb_from_yuv);

// ===== RGBA -> RGB Conversions =====
impl_convert!(Rgba8<A1> => Rgb8<A2>, crate::color::rgb_from_rgba, bg: None);

// ===== BGRA -> RGB Conversions =====
impl_convert!(Bgra8<A1> => Rgb8<A2>, crate::color::rgb_from_bgra, bg: None);

// ===== RGB -> RGBA Conversions (add opaque alpha) =====
impl_convert!(Rgb8<A1> => Rgba8<A2>, crate::color::rgba_from_rgb);
impl_convert!(Rgbf32<A1> => Rgbaf32<A2>, crate::color::rgba_from_rgb);

// ===== RGB -> BGRA Conversions (swap R/B + add opaque alpha) =====
impl_convert!(Rgb8<A1> => Bgra8<A2>, crate::color::bgra_from_rgb);
impl_convert!(Rgbf32<A1> => Bgraf32<A2>, crate::color::bgra_from_rgb);

// ===== Video format decode -> RGB (BT.601 limited range) =====
//
// The packed/planar video wrappers carry a raw byte buffer rather than a typed `Image`,
// so they get hand-written `ConvertColor` impls (the `impl_convert!` macro assumes `.0`).
macro_rules! impl_video_decode {
    ($src:ty, $func:path) => {
        impl<A2> ConvertColor<Rgb8<A2>> for $src
        where
            A2: ImageAllocator,
        {
            fn convert(&self, dst: &mut Rgb8<A2>) -> Result<(), ImageError> {
                $func(self.as_slice(), &mut dst.0)
            }
        }
    };
}

impl_video_decode!(Yuyv8, crate::color::rgb_from_yuyv);
impl_video_decode!(Uyvy8, crate::color::rgb_from_uyvy);
impl_video_decode!(Yvyu8, crate::color::rgb_from_yvyu);
impl_video_decode!(Nv12, crate::color::rgb_from_nv12);
impl_video_decode!(Nv21, crate::color::rgb_from_nv21);
impl_video_decode!(I420, crate::color::rgb_from_i420);
impl_video_decode!(Yv12, crate::color::rgb_from_yv12);

// ===== Conversion with background support =====

/// Trait for color conversion with optional background color
pub trait ConvertColorWithBackground<Dst> {
    /// Convert with optional background color for alpha blending
    fn convert_with_bg(&self, dst: &mut Dst, bg: Option<[u8; 3]>) -> Result<(), ImageError>;
}

impl_convert_with_bg!(Rgba8<A1> => Rgb8<A2>, crate::color::rgb_from_rgba);
impl_convert_with_bg!(Bgra8<A1> => Rgb8<A2>, crate::color::rgb_from_bgra);

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::ImageSize;
    use kornia_tensor::CpuAllocator;

    #[test]
    fn test_gray_from_rgb_f32() -> Result<(), ImageError> {
        let rgb = Rgbf32::from_size_vec(
            ImageSize {
                width: 2,
                height: 2,
            },
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5],
            CpuAllocator,
        )?;

        let mut gray = Grayf32::from_size_val(rgb.size(), 0.0, CpuAllocator)?;

        rgb.convert(&mut gray)?;

        assert_eq!(gray.num_channels(), 1);
        assert_eq!(gray.size().width, 2);
        assert_eq!(gray.size().height, 2);

        Ok(())
    }

    #[test]
    fn test_gray_from_rgb_u8() -> Result<(), ImageError> {
        let rgb = Rgb8::from_size_vec(
            ImageSize {
                width: 2,
                height: 1,
            },
            vec![255, 0, 0, 0, 255, 0],
            CpuAllocator,
        )?;

        let mut gray = Gray8::from_size_val(rgb.size(), 0, CpuAllocator)?;

        rgb.convert(&mut gray)?;

        assert_eq!(gray.num_channels(), 1);

        Ok(())
    }

    #[test]
    fn test_rgb_from_gray() -> Result<(), ImageError> {
        let gray = Grayf32::from_size_vec(
            ImageSize {
                width: 2,
                height: 2,
            },
            vec![0.0, 0.5, 1.0, 0.25],
            CpuAllocator,
        )?;

        let mut rgb = Rgbf32::from_size_val(gray.size(), 0.0, CpuAllocator)?;

        gray.convert(&mut rgb)?;

        assert_eq!(rgb.num_channels(), 3);
        assert_eq!(rgb.size().width, 2);
        assert_eq!(rgb.size().height, 2);

        Ok(())
    }

    #[test]
    fn test_bgr_from_rgb() -> Result<(), ImageError> {
        let rgb = Rgb8::from_size_vec(
            ImageSize {
                width: 1,
                height: 1,
            },
            vec![255, 128, 64],
            CpuAllocator,
        )?;

        let mut bgr = Bgr8::from_size_val(rgb.size(), 0, CpuAllocator)?;

        rgb.convert(&mut bgr)?;

        // Check that RGB was swapped to BGR
        let bgr_data = bgr.as_slice();
        assert_eq!(bgr_data[0], 64); // B
        assert_eq!(bgr_data[1], 128); // G
        assert_eq!(bgr_data[2], 255); // R

        Ok(())
    }

    #[test]
    fn test_rgb_from_bgr() -> Result<(), ImageError> {
        let bgr = Bgr8::from_size_vec(
            ImageSize {
                width: 1,
                height: 1,
            },
            vec![64, 128, 255],
            CpuAllocator,
        )?;

        let mut rgb = Rgb8::from_size_val(bgr.size(), 0, CpuAllocator)?;

        bgr.convert(&mut rgb)?;

        // Check that BGR was swapped to RGB
        let rgb_data = rgb.as_slice();
        assert_eq!(rgb_data[0], 255); // R
        assert_eq!(rgb_data[1], 128); // G
        assert_eq!(rgb_data[2], 64); // B

        Ok(())
    }

    #[test]
    fn test_ycbcr_and_yuv_round_trip_u8() -> Result<(), ImageError> {
        use kornia_image::color_spaces::{YCbCr8, Yuv8};
        let size = ImageSize {
            width: 4,
            height: 2,
        };
        let data: Vec<u8> = (0..4 * 2 * 3).map(|v| (v * 7 + 11) as u8).collect();
        let rgb = Rgb8::from_size_vec(size, data, CpuAllocator)?;

        // YCbCr round-trip
        let mut ycc = YCbCr8::from_size_val(size, 0, CpuAllocator)?;
        let mut back = Rgb8::from_size_val(size, 0, CpuAllocator)?;
        rgb.convert(&mut ycc)?;
        ycc.convert(&mut back)?;
        for (a, b) in rgb.as_slice().iter().zip(back.as_slice().iter()) {
            assert!((*a as i32 - *b as i32).abs() <= 3);
        }

        // YUV (swapped chroma) must differ from YCbCr in channels 1/2 but round-trip too.
        let mut yuv = Yuv8::from_size_val(size, 0, CpuAllocator)?;
        rgb.convert(&mut yuv)?;
        assert_eq!(ycc.as_slice()[1], yuv.as_slice()[2]);
        assert_eq!(ycc.as_slice()[2], yuv.as_slice()[1]);
        Ok(())
    }

    #[test]
    fn test_yuyv_decode_to_rgb() -> Result<(), ImageError> {
        use kornia_image::color_spaces::Yuyv8;
        let size = ImageSize {
            width: 2,
            height: 1,
        };
        // Y=16, U=V=128 -> black (limited range).
        let yuyv = Yuyv8::from_size_vec(size, vec![16, 128, 16, 128])?;
        let mut rgb = Rgb8::from_size_val(size, 0, CpuAllocator)?;
        yuyv.convert(&mut rgb)?;
        assert_eq!(rgb.as_slice(), &[0, 0, 0, 0, 0, 0]);
        Ok(())
    }

    #[test]
    fn test_hsv_from_rgb() -> Result<(), ImageError> {
        let rgb = Rgbf32::from_size_vec(
            ImageSize {
                width: 1,
                height: 1,
            },
            vec![255.0, 0.0, 0.0],
            CpuAllocator,
        )?;

        let mut hsv = Hsvf32::from_size_val(rgb.size(), 0.0, CpuAllocator)?;

        rgb.convert(&mut hsv)?;

        assert_eq!(hsv.num_channels(), 3);

        Ok(())
    }

    #[test]
    fn test_deref_works_with_existing_api() -> Result<(), ImageError> {
        let rgb = Rgb8::from_size_vec(
            ImageSize {
                width: 4,
                height: 4,
            },
            vec![0u8; 4 * 4 * 3],
            CpuAllocator,
        )?;

        // Test that Deref allows us to use Image methods
        assert_eq!(rgb.width(), 4);
        assert_eq!(rgb.height(), 4);
        assert_eq!(rgb.num_channels(), 3);
        assert_eq!(rgb.size().width, 4);

        Ok(())
    }

    #[test]
    fn test_rgb_from_rgba() -> Result<(), ImageError> {
        let rgba = Rgba8::from_size_vec(
            ImageSize {
                width: 2,
                height: 1,
            },
            vec![255, 128, 64, 255, 100, 50, 25, 128],
            CpuAllocator,
        )?;

        let mut rgb = Rgb8::from_size_val(rgba.size(), 0, CpuAllocator)?;

        rgba.convert(&mut rgb)?;

        assert_eq!(rgb.num_channels(), 3);
        let data = rgb.as_slice();
        assert_eq!(data[0], 255);
        assert_eq!(data[1], 128);
        assert_eq!(data[2], 64);

        Ok(())
    }

    #[test]
    fn test_rgb_from_rgba_with_background() -> Result<(), ImageError> {
        use super::ConvertColorWithBackground;

        let rgba = Rgba8::from_size_vec(
            ImageSize {
                width: 1,
                height: 1,
            },
            vec![255, 0, 0, 128], // Red with 50% alpha
            CpuAllocator,
        )?;

        let mut rgb = Rgb8::from_size_val(rgba.size(), 0, CpuAllocator)?;

        // Convert with white background
        rgba.convert_with_bg(&mut rgb, Some([100, 100, 100]))?;

        let data = rgb.as_slice();
        // Should blend red with gray background
        assert_eq!(data[0], 178); // (255 * 0.5 + 100 * 0.5)
        assert_eq!(data[1], 50); // (0 * 0.5 + 100 * 0.5)
        assert_eq!(data[2], 50); // (0 * 0.5 + 100 * 0.5)

        Ok(())
    }

    #[test]
    fn test_rgb_from_bgra() -> Result<(), ImageError> {
        let bgra = Bgra8::from_size_vec(
            ImageSize {
                width: 1,
                height: 1,
            },
            vec![64, 128, 255, 255], // BGR + A
            CpuAllocator,
        )?;

        let mut rgb = Rgb8::from_size_val(bgra.size(), 0, CpuAllocator)?;

        bgra.convert(&mut rgb)?;

        let data = rgb.as_slice();
        assert_eq!(data[0], 255); // R
        assert_eq!(data[1], 128); // G
        assert_eq!(data[2], 64); // B

        Ok(())
    }

    #[test]
    fn test_rgba_from_rgb() -> Result<(), ImageError> {
        let rgb = Rgb8::from_size_vec(
            ImageSize {
                width: 2,
                height: 1,
            },
            vec![1, 2, 3, 4, 5, 6],
            CpuAllocator,
        )?;
        let mut rgba = Rgba8::from_size_val(rgb.size(), 0, CpuAllocator)?;
        rgb.convert(&mut rgba)?;
        assert_eq!(rgba.as_slice(), &[1, 2, 3, 255, 4, 5, 6, 255]);
        Ok(())
    }

    #[test]
    fn test_bgra_from_rgb() -> Result<(), ImageError> {
        let rgb = Rgb8::from_size_vec(
            ImageSize {
                width: 2,
                height: 1,
            },
            vec![1, 2, 3, 4, 5, 6],
            CpuAllocator,
        )?;
        let mut bgra = Bgra8::from_size_val(rgb.size(), 0, CpuAllocator)?;
        rgb.convert(&mut bgra)?;
        assert_eq!(bgra.as_slice(), &[3, 2, 1, 255, 6, 5, 4, 255]);
        Ok(())
    }

    #[test]
    fn test_explicit_type_aliases() -> Result<(), ImageError> {
        use crate::color::{Bgr8, Gray8, Rgb8, Rgba8};

        // Test Rgb8
        let rgb = Rgb8::from_size_vec(
            ImageSize {
                width: 2,
                height: 1,
            },
            vec![255, 0, 0, 0, 255, 0],
            CpuAllocator,
        )?;

        let mut gray = Gray8::from_size_val(rgb.size(), 0, CpuAllocator)?;

        rgb.convert(&mut gray)?;
        assert_eq!(gray.num_channels(), 1);

        // Test Bgr8
        let mut bgr = Bgr8::from_size_val(rgb.size(), 0, CpuAllocator)?;
        rgb.convert(&mut bgr)?;
        assert_eq!(bgr.num_channels(), 3);

        // Test Rgba8
        let rgba = Rgba8::from_size_vec(
            ImageSize {
                width: 1,
                height: 1,
            },
            vec![255, 128, 64, 255],
            CpuAllocator,
        )?;
        let mut rgb_out = Rgb8::from_size_val(rgba.size(), 0, CpuAllocator)?;
        rgba.convert(&mut rgb_out)?;

        assert_eq!(rgb_out.as_slice()[0], 255);

        Ok(())
    }
}

// ===== Zero-cost typed allocating conversion layer =====

/// Allocate a zeroed owned image of a color-space newtype at a given size.
///
/// Backed by [`kornia_image::allocator::CpuAllocator`] so that `.cvt()` returns
/// owned, heap-allocated data without any allocator generic in the call site.
pub trait NewColorImage: Sized {
    /// Allocate a zero-filled image of `size`.
    fn new_zeroed(size: ImageSize) -> Result<Self, ImageError>;
    /// Size of an already-constructed instance (used to size the output).
    fn size_of(&self) -> ImageSize;
}

macro_rules! impl_new_color_image {
    ($newtype:ident, $t:ty) => {
        impl NewColorImage
            for kornia_image::color_spaces::$newtype<kornia_image::allocator::CpuAllocator>
        {
            fn new_zeroed(size: ImageSize) -> Result<Self, ImageError> {
                kornia_image::color_spaces::$newtype::from_size_val(
                    size,
                    <$t>::default(),
                    kornia_image::allocator::CpuAllocator,
                )
            }
            fn size_of(&self) -> ImageSize {
                self.size()
            }
        }
    };
}

impl_new_color_image!(Rgbf32, f32);
impl_new_color_image!(Bgrf32, f32);
impl_new_color_image!(Grayf32, f32);
impl_new_color_image!(Hsvf32, f32);
impl_new_color_image!(Hlsf32, f32);
impl_new_color_image!(Labf32, f32);
impl_new_color_image!(Luvf32, f32);
impl_new_color_image!(Xyzf32, f32);
impl_new_color_image!(LinearRgbf32, f32);
impl_new_color_image!(YCbCrf32, f32);
impl_new_color_image!(Yuvf32, f32);
impl_new_color_image!(Rgb8, u8);
impl_new_color_image!(Bgr8, u8);
impl_new_color_image!(Gray8, u8);
impl_new_color_image!(Rgba8, u8);
impl_new_color_image!(Bgra8, u8);
impl_new_color_image!(YCbCr8, u8);
impl_new_color_image!(Yuv8, u8);
impl_new_color_image!(Rgbaf32, f32);
impl_new_color_image!(Bgraf32, f32);

/// Helper trait so `.cvt()` can read the source image size without coupling to
/// a concrete allocator.
///
/// All color-space newtypes implement this via their `Deref<Target = Image<…>>`.
pub trait SrcSize {
    /// Source image size.
    fn src_size(&self) -> ImageSize;
}

macro_rules! impl_src_size {
    ($newtype:ident) => {
        impl<A: ImageAllocator> SrcSize
            for kornia_image::color_spaces::$newtype<A>
        {
            fn src_size(&self) -> ImageSize {
                self.size()
            }
        }
    };
}

impl_src_size!(Rgbf32);
impl_src_size!(Bgrf32);
impl_src_size!(Grayf32);
impl_src_size!(Hsvf32);
impl_src_size!(Hlsf32);
impl_src_size!(Labf32);
impl_src_size!(Luvf32);
impl_src_size!(Xyzf32);
impl_src_size!(LinearRgbf32);
impl_src_size!(YCbCrf32);
impl_src_size!(Yuvf32);
impl_src_size!(Rgb8);
impl_src_size!(Bgr8);
impl_src_size!(Gray8);
impl_src_size!(Rgba8);
impl_src_size!(Bgra8);
impl_src_size!(YCbCr8);
impl_src_size!(Yuv8);
impl_src_size!(Rgbaf32);
impl_src_size!(Bgraf32);

/// Ergonomic allocating conversion built on [`ConvertColor`].
///
/// Zero-cost sugar: allocates the correctly-sized owned destination and delegates
/// to the existing kernel. The source must expose its size via [`SrcSize`] (all
/// newtypes do, through `Deref` to `Image`).
///
/// # Example
///
/// ```
/// use kornia_image::ImageSize;
/// use kornia_image::allocator::CpuAllocator;
/// use kornia_imgproc::color::{Rgbf32, Hsvf32, ConvertColorExt};
///
/// let rgb = Rgbf32::from_size_vec(
///     ImageSize { width: 4, height: 3 },
///     vec![0.5f32; 4 * 3 * 3],
///     CpuAllocator,
/// ).unwrap();
///
/// let hsv: Hsvf32<_> = rgb.cvt().unwrap();
/// assert_eq!(hsv.size(), rgb.size());
/// ```
pub trait ConvertColorExt {
    /// Allocate and convert to `Dst`. `Dst` is chosen by inference or turbofish.
    fn cvt<Dst>(&self) -> Result<Dst, ImageError>
    where
        Self: ConvertColor<Dst> + SrcSize,
        Dst: NewColorImage;
}

impl<Src> ConvertColorExt for Src {
    fn cvt<Dst>(&self) -> Result<Dst, ImageError>
    where
        Self: ConvertColor<Dst> + SrcSize,
        Dst: NewColorImage,
    {
        let mut dst = Dst::new_zeroed(self.src_size())?;
        self.convert(&mut dst)?;
        Ok(dst)
    }
}

#[cfg(test)]
mod cvt_color_tests {
    use crate::color::Tagged;
    use kornia_image::allocator::CpuAllocator;
    use kornia_image::color_spaces::Rgbf32;
    use kornia_image::{ColorSpace, DynImage, ImageSize};

    #[test]
    fn runtime_cvt_color_returns_tagged_dynimage() {
        let size = ImageSize { width: 4, height: 4 };
        let rgb = Rgbf32::from_size_vec(size, vec![0.5f32; 4 * 4 * 3], CpuAllocator).unwrap();
        let hsv = rgb.cvt_color(ColorSpace::Hsv).unwrap();
        assert_eq!(hsv.color_space(), ColorSpace::Hsv);
        assert_eq!(hsv.channels(), 3);
        let gray = rgb.cvt_color(ColorSpace::Gray).unwrap();
        assert_eq!(gray.color_space(), ColorSpace::Gray);
        assert_eq!(gray.channels(), 1);
        assert!(matches!(gray, DynImage::C1(..)));
    }

    #[test]
    fn runtime_cvt_color_rejects_unsupported_pair() {
        let size = ImageSize { width: 2, height: 2 };
        let rgb = Rgbf32::from_size_vec(size, vec![0.0f32; 2 * 2 * 3], CpuAllocator).unwrap();
        // Rgb has no direct path to YCbCr? It does — pick a truly illegal target by
        // constructing from a non-Rgb source instead:
        let hsv = rgb.cvt_color(ColorSpace::Hsv).unwrap();
        let hsv_typed: kornia_image::color_spaces::Hsvf32<_> = hsv.try_into().unwrap();
        let err = hsv_typed.cvt_color(ColorSpace::Lab);
        assert!(err.is_err());
    }
}

#[cfg(test)]
mod cvt_ext_tests {
    use crate::color::ConvertColorExt;
    use kornia_image::allocator::CpuAllocator;
    use kornia_image::color_spaces::{Grayf32, Hsvf32, Rgbf32};
    use kornia_image::ImageSize;

    #[test]
    fn cvt_allocates_and_converts_typed() {
        let size = ImageSize {
            width: 4,
            height: 3,
        };
        let rgb = Rgbf32::from_size_vec(size, vec![0.5f32; 4 * 3 * 3], CpuAllocator).unwrap();
        // typed, allocating: no manual dst construction
        let hsv: Hsvf32<_> = rgb.cvt().unwrap();
        assert_eq!(hsv.size(), size);
        // channel-changing conversion is natural — Dst encodes C
        let gray: Grayf32<_> = rgb.cvt().unwrap();
        assert_eq!(gray.num_channels(), 1);
    }

    #[test]
    fn cvt_round_trip_rgb_hsv() {
        let size = ImageSize {
            width: 8,
            height: 8,
        };
        let data: Vec<f32> = (0..8 * 8 * 3).map(|i| (i % 255) as f32 / 255.0).collect();
        let rgb = Rgbf32::from_size_vec(size, data.clone(), CpuAllocator).unwrap();
        let hsv: Hsvf32<_> = rgb.cvt().unwrap();
        let back: Rgbf32<_> = hsv.cvt().unwrap();
        for (a, b) in data.iter().zip(back.as_slice().iter()) {
            assert!((a - b).abs() < 1e-3, "round-trip drift {a} vs {b}");
        }
    }
}

// ===== Runtime Tagged dispatch layer =====

use kornia_image::{ColorSpace, DynImage};
use kornia_image::allocator::CpuAllocator;

/// Runtime, color-space-tagged conversion. The source newtype encodes its
/// space and dtype, so only the target `to` is supplied; the result is a
/// `DynImage` tagged with `to`. Same legal set as the typed `.cvt()` path.
pub trait Tagged<T> {
    /// This image's color space.
    fn space(&self) -> ColorSpace;
    /// Convert to `to`, returning an owned tagged `DynImage`.
    fn cvt_color(&self, to: ColorSpace) -> Result<DynImage<T, CpuAllocator>, ImageError>;
}

/// Generates a `Tagged` impl for one source newtype. Each `to => Dst, Cn`
/// arm names the destination newtype and the DynImage channel constructor.
macro_rules! impl_tagged {
    ($src:ty, $t:ty, $space:expr, { $( $to:ident => $dst:ty , $ctor:ident );* $(;)? }) => {
        impl<A: ImageAllocator> Tagged<$t> for $src
        where Self: SrcSize {
            fn space(&self) -> ColorSpace { $space }
            fn cvt_color(&self, to: ColorSpace) -> Result<DynImage<$t, CpuAllocator>, ImageError> {
                match to {
                    $( ColorSpace::$to => {
                        let out: $dst = self.cvt()?;
                        Ok(DynImage::$ctor(to, out.into_inner()))
                    } )*
                    _ => Err(ImageError::UnsupportedColorConversion { from: $space, to }),
                }
            }
        }
    };
}

// ---- f32 RGB source: all f32 targets ----
impl_tagged!(kornia_image::color_spaces::Rgbf32<A>, f32, ColorSpace::Rgb, {
    Gray      => kornia_image::color_spaces::Grayf32<CpuAllocator>, C1;
    Bgr       => kornia_image::color_spaces::Bgrf32<CpuAllocator>, C3;
    Hsv       => kornia_image::color_spaces::Hsvf32<CpuAllocator>, C3;
    Hls       => kornia_image::color_spaces::Hlsf32<CpuAllocator>, C3;
    Lab       => kornia_image::color_spaces::Labf32<CpuAllocator>, C3;
    Luv       => kornia_image::color_spaces::Luvf32<CpuAllocator>, C3;
    Xyz       => kornia_image::color_spaces::Xyzf32<CpuAllocator>, C3;
    LinearRgb => kornia_image::color_spaces::LinearRgbf32<CpuAllocator>, C3;
    YCbCr     => kornia_image::color_spaces::YCbCrf32<CpuAllocator>, C3;
    Yuv       => kornia_image::color_spaces::Yuvf32<CpuAllocator>, C3;
});

// ---- f32 inverse sources back to RGB ----
impl_tagged!(kornia_image::color_spaces::Hsvf32<A>, f32, ColorSpace::Hsv, {
    Rgb => kornia_image::color_spaces::Rgbf32<CpuAllocator>, C3;
});
impl_tagged!(kornia_image::color_spaces::Hlsf32<A>, f32, ColorSpace::Hls, {
    Rgb => kornia_image::color_spaces::Rgbf32<CpuAllocator>, C3;
});
impl_tagged!(kornia_image::color_spaces::Labf32<A>, f32, ColorSpace::Lab, {
    Rgb => kornia_image::color_spaces::Rgbf32<CpuAllocator>, C3;
});
impl_tagged!(kornia_image::color_spaces::Luvf32<A>, f32, ColorSpace::Luv, {
    Rgb => kornia_image::color_spaces::Rgbf32<CpuAllocator>, C3;
});
impl_tagged!(kornia_image::color_spaces::Xyzf32<A>, f32, ColorSpace::Xyz, {
    Rgb => kornia_image::color_spaces::Rgbf32<CpuAllocator>, C3;
});
impl_tagged!(kornia_image::color_spaces::LinearRgbf32<A>, f32, ColorSpace::LinearRgb, {
    Rgb => kornia_image::color_spaces::Rgbf32<CpuAllocator>, C3;
});
impl_tagged!(kornia_image::color_spaces::YCbCrf32<A>, f32, ColorSpace::YCbCr, {
    Rgb => kornia_image::color_spaces::Rgbf32<CpuAllocator>, C3;
});
impl_tagged!(kornia_image::color_spaces::Yuvf32<A>, f32, ColorSpace::Yuv, {
    Rgb => kornia_image::color_spaces::Rgbf32<CpuAllocator>, C3;
});
impl_tagged!(kornia_image::color_spaces::Bgrf32<A>, f32, ColorSpace::Bgr, {
    Rgb => kornia_image::color_spaces::Rgbf32<CpuAllocator>, C3;
});
impl_tagged!(kornia_image::color_spaces::Grayf32<A>, f32, ColorSpace::Gray, {
    Rgb => kornia_image::color_spaces::Rgbf32<CpuAllocator>, C3;
});

// ---- u8 RGB source: u8-valid targets ----
impl_tagged!(kornia_image::color_spaces::Rgb8<A>, u8, ColorSpace::Rgb, {
    Gray  => kornia_image::color_spaces::Gray8<CpuAllocator>, C1;
    Bgr   => kornia_image::color_spaces::Bgr8<CpuAllocator>, C3;
    Rgba  => kornia_image::color_spaces::Rgba8<CpuAllocator>, C4;
    YCbCr => kornia_image::color_spaces::YCbCr8<CpuAllocator>, C3;
    Yuv   => kornia_image::color_spaces::Yuv8<CpuAllocator>, C3;
});
impl_tagged!(kornia_image::color_spaces::Gray8<A>, u8, ColorSpace::Gray, {
    Rgb => kornia_image::color_spaces::Rgb8<CpuAllocator>, C3;
});
impl_tagged!(kornia_image::color_spaces::Bgr8<A>, u8, ColorSpace::Bgr, {
    Rgb => kornia_image::color_spaces::Rgb8<CpuAllocator>, C3;
});
