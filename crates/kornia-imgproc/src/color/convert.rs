use kornia_image::{
    allocator::ImageAllocator,
    color_spaces::{
        Bgr8, Bgra8, Bgrf32, Gray8, Grayf32, Grayf64, Hsvf32, Rgb8, Rgba8, Rgbf32, Rgbf64,
    },
    ImageError,
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

// ===== RGB -> Gray Conversions =====
impl_convert!(Rgbf32<A1> => Grayf32<A2>, crate::color::gray_from_rgb);
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

// ===== RGB -> HSV Conversions =====
impl_convert!(Rgbf32<A1> => Hsvf32<A2>, crate::color::hsv_from_rgb);

// ===== RGBA -> RGB Conversions =====
impl_convert!(Rgba8<A1> => Rgb8<A2>, crate::color::rgb_from_rgba, bg: None);

// ===== BGRA -> RGB Conversions =====
impl_convert!(Bgra8<A1> => Rgb8<A2>, crate::color::rgb_from_bgra, bg: None);

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
    fn test_rgb_to_gray_f32() -> Result<(), ImageError> {
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
    fn test_rgb_to_gray_u8() -> Result<(), ImageError> {
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
    fn test_gray_to_rgb() -> Result<(), ImageError> {
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
    fn test_rgb_to_bgr() -> Result<(), ImageError> {
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
    fn test_bgr_to_rgb() -> Result<(), ImageError> {
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
    fn test_rgb_to_hsv() -> Result<(), ImageError> {
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
    fn test_rgba_to_rgb() -> Result<(), ImageError> {
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
    fn test_rgba_to_rgb_with_background() -> Result<(), ImageError> {
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
    fn test_bgra_to_rgb() -> Result<(), ImageError> {
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
