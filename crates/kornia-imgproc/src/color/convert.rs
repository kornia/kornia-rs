use kornia_image::{
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
/// use kornia_imgproc::color::{Rgbf32, Grayf32, ConvertColor};
///
/// let rgb = Rgbf32::from_size_vec(
///     ImageSize { width: 4, height: 5 },
///     vec![0.5f32; 4 * 5 * 3],
/// ).unwrap();
///
/// let mut gray = Grayf32::from_size_val(rgb.size(), 0.0).unwrap();
///
/// rgb.convert(&mut gray).unwrap();
/// ```
pub trait ConvertColor<Dst> {
    /// Convert this image to another color space
    fn convert(&self, dst: &mut Dst) -> Result<(), ImageError>;
}

/// Macro to implement color conversions.
///
/// Every arm is **residency-aware**: host pairs run the CPU path unchanged,
/// device pairs dispatch to the named CUDA adapter in `super::cuda_dispatch`
/// (with cross-stream event fencing handled by `DeviceExec::run`), and mixed
/// pairs error with [`ImageError::MixedResidency`].
macro_rules! impl_convert {
    // Residency-aware: device operands dispatch to the CUDA adapter.
    ($src:ty => $dst:ty, $func:path, cuda: $cuda:path) => {
        impl ConvertColor<$dst> for $src {
            fn convert(&self, dst: &mut $dst) -> Result<(), ImageError> {
                #[cfg(feature = "cuda")]
                {
                    use crate::color::cuda_dispatch::{pair_residency, Residency};
                    if let Residency::Device(exec) = pair_residency(&self.0, &dst.0)? {
                        return exec.run(|stream| $cuda(&self.0, &mut dst.0, stream));
                    }
                }
                $func(&self.0, &mut dst.0)
            }
        }
    };

    // Residency-aware with an optional-parameter CPU path (like background).
    ($src:ty => $dst:ty, $func:path, bg: $bg:expr_2021, cuda: $cuda:path) => {
        impl ConvertColor<$dst> for $src {
            fn convert(&self, dst: &mut $dst) -> Result<(), ImageError> {
                #[cfg(feature = "cuda")]
                {
                    use crate::color::cuda_dispatch::{pair_residency, Residency};
                    if let Residency::Device(exec) = pair_residency(&self.0, &dst.0)? {
                        return exec.run(|stream| $cuda(&self.0, &mut dst.0, stream));
                    }
                }
                $func(&self.0, &mut dst.0, $bg)
            }
        }
    };
}

/// Macro to implement color conversions with background support
macro_rules! impl_convert_with_bg {
    // Residency-aware: the CUDA adapter also receives the background.
    ($src:ty => $dst:ty, $func:path, cuda: $cuda:path) => {
        impl ConvertColorWithBackground<$dst> for $src {
            fn convert_with_bg(
                &self,
                dst: &mut $dst,
                bg: Option<[u8; 3]>,
            ) -> Result<(), ImageError> {
                #[cfg(feature = "cuda")]
                {
                    use crate::color::cuda_dispatch::{pair_residency, Residency};
                    if let Residency::Device(exec) = pair_residency(&self.0, &dst.0)? {
                        return exec.run(|stream| $cuda(&self.0, &mut dst.0, bg, stream));
                    }
                }
                $func(&self.0, &mut dst.0, bg)
            }
        }
    };
}

// ===== Bayer mosaic -> RGB (hand-written: reads the runtime pattern field) =====
// `rgb_from_bayer` is itself residency-aware, so no host guard is needed here.
impl ConvertColor<Rgb8> for kornia_image::color_spaces::Bayer8 {
    fn convert(&self, dst: &mut Rgb8) -> Result<(), ImageError> {
        crate::color::rgb_from_bayer(self.as_image(), self.pattern, &mut dst.0)
    }
}

// ===== RGB -> Gray Conversions =====
impl_convert!(Rgbf32 => Grayf32, crate::color::gray_from_rgb_f32,
    cuda: crate::color::cuda_dispatch::gray_from_rgb_f32_cuda);
impl_convert!(Rgbf64 => Grayf64, crate::color::gray_from_rgb,
    cuda: crate::color::cuda_dispatch::gray_from_rgb_f64_cuda);
impl_convert!(Rgb8 => Gray8, crate::color::gray_from_rgb_u8,
    cuda: crate::color::cuda_dispatch::gray_from_rgb_u8_cuda);

// ===== Gray -> RGB Conversions =====
impl_convert!(Gray8 => Rgb8, crate::color::rgb_from_gray,
    cuda: crate::color::cuda_dispatch::rgb_from_gray_u8_cuda);
impl_convert!(Grayf32 => Rgbf32, crate::color::rgb_from_gray,
    cuda: crate::color::cuda_dispatch::rgb_from_gray_f32_cuda);
impl_convert!(Grayf64 => Rgbf64, crate::color::rgb_from_gray,
    cuda: crate::color::cuda_dispatch::rgb_from_gray_f64_cuda);

// ===== RGB <-> BGR Conversions =====
impl_convert!(Rgb8 => Bgr8, crate::color::bgr_from_rgb,
    cuda: crate::color::cuda_dispatch::bgr_from_rgb_u8_cuda);
impl_convert!(Bgr8 => Rgb8, crate::color::bgr_from_rgb,
    cuda: crate::color::cuda_dispatch::bgr_from_rgb_u8_cuda);
impl_convert!(Rgbf32 => Bgrf32, crate::color::bgr_from_rgb,
    cuda: crate::color::cuda_dispatch::bgr_from_rgb_f32_cuda);
impl_convert!(Bgrf32 => Rgbf32, crate::color::bgr_from_rgb,
    cuda: crate::color::cuda_dispatch::bgr_from_rgb_f32_cuda);

// ===== RGB <-> HSV Conversions =====
impl_convert!(Rgbf32 => Hsvf32, crate::color::hsv_from_rgb,
    cuda: crate::color::cuda_dispatch::hsv_from_rgb_f32_cuda);
impl_convert!(Hsvf32 => Rgbf32, crate::color::rgb_from_hsv,
    cuda: crate::color::cuda_dispatch::rgb_from_hsv_f32_cuda);
impl_convert!(Rgbf64 => Hsvf64, crate::color::hsv_from_rgb,
    cuda: crate::color::cuda_dispatch::hsv_from_rgb_f64_cuda);
impl_convert!(Hsvf64 => Rgbf64, crate::color::rgb_from_hsv,
    cuda: crate::color::cuda_dispatch::rgb_from_hsv_f64_cuda);

// ===== RGB <-> HLS Conversions =====
impl_convert!(Rgbf32 => Hlsf32, crate::color::hls_from_rgb,
    cuda: crate::color::cuda_dispatch::hls_from_rgb_f32_cuda);
impl_convert!(Hlsf32 => Rgbf32, crate::color::rgb_from_hls,
    cuda: crate::color::cuda_dispatch::rgb_from_hls_f32_cuda);
impl_convert!(Rgbf64 => Hlsf64, crate::color::hls_from_rgb,
    cuda: crate::color::cuda_dispatch::hls_from_rgb_f64_cuda);
impl_convert!(Hlsf64 => Rgbf64, crate::color::rgb_from_hls,
    cuda: crate::color::cuda_dispatch::rgb_from_hls_f64_cuda);

// ===== RGB <-> linear-RGB (sRGB transfer) =====
impl_convert!(Rgbf32 => LinearRgbf32, crate::color::linear_rgb_from_rgb,
    cuda: crate::color::cuda_dispatch::linear_rgb_from_rgb_f32_cuda);
impl_convert!(LinearRgbf32 => Rgbf32, crate::color::rgb_from_linear_rgb,
    cuda: crate::color::cuda_dispatch::rgb_from_linear_rgb_f32_cuda);
impl_convert!(Rgbf64 => LinearRgbf64, crate::color::linear_rgb_from_rgb,
    cuda: crate::color::cuda_dispatch::linear_rgb_from_rgb_f64_cuda);
impl_convert!(LinearRgbf64 => Rgbf64, crate::color::rgb_from_linear_rgb,
    cuda: crate::color::cuda_dispatch::rgb_from_linear_rgb_f64_cuda);

// ===== RGB <-> XYZ Conversions =====
impl_convert!(Rgbf32 => Xyzf32, crate::color::xyz_from_rgb,
    cuda: crate::color::cuda_dispatch::xyz_from_rgb_f32_cuda);
impl_convert!(Xyzf32 => Rgbf32, crate::color::rgb_from_xyz,
    cuda: crate::color::cuda_dispatch::rgb_from_xyz_f32_cuda);
impl_convert!(Rgbf64 => Xyzf64, crate::color::xyz_from_rgb,
    cuda: crate::color::cuda_dispatch::xyz_from_rgb_f64_cuda);
impl_convert!(Xyzf64 => Rgbf64, crate::color::rgb_from_xyz,
    cuda: crate::color::cuda_dispatch::rgb_from_xyz_f64_cuda);

// ===== RGB <-> Lab Conversions =====
impl_convert!(Rgbf32 => Labf32, crate::color::lab_from_rgb,
    cuda: crate::color::cuda_dispatch::lab_from_rgb_f32_cuda);
impl_convert!(Labf32 => Rgbf32, crate::color::rgb_from_lab,
    cuda: crate::color::cuda_dispatch::rgb_from_lab_f32_cuda);
impl_convert!(Rgbf64 => Labf64, crate::color::lab_from_rgb,
    cuda: crate::color::cuda_dispatch::lab_from_rgb_f64_cuda);
impl_convert!(Labf64 => Rgbf64, crate::color::rgb_from_lab,
    cuda: crate::color::cuda_dispatch::rgb_from_lab_f64_cuda);

// ===== RGB <-> Luv Conversions =====
impl_convert!(Rgbf32 => Luvf32, crate::color::luv_from_rgb,
    cuda: crate::color::cuda_dispatch::luv_from_rgb_f32_cuda);
impl_convert!(Luvf32 => Rgbf32, crate::color::rgb_from_luv,
    cuda: crate::color::cuda_dispatch::rgb_from_luv_f32_cuda);
impl_convert!(Rgbf64 => Luvf64, crate::color::luv_from_rgb,
    cuda: crate::color::cuda_dispatch::luv_from_rgb_f64_cuda);
impl_convert!(Luvf64 => Rgbf64, crate::color::rgb_from_luv,
    cuda: crate::color::cuda_dispatch::rgb_from_luv_f64_cuda);

// ===== RGB <-> YCbCr Conversions =====
impl_convert!(Rgb8 => YCbCr8, crate::color::ycbcr_from_rgb,
    cuda: crate::color::cuda_dispatch::ycbcr_from_rgb_u8_cuda);
impl_convert!(YCbCr8 => Rgb8, crate::color::rgb_from_ycbcr,
    cuda: crate::color::cuda_dispatch::rgb_from_ycbcr_u8_cuda);
impl_convert!(Rgbf32 => YCbCrf32, crate::color::ycbcr_from_rgb,
    cuda: crate::color::cuda_dispatch::ycbcr_from_rgb_f32_cuda);
impl_convert!(YCbCrf32 => Rgbf32, crate::color::rgb_from_ycbcr,
    cuda: crate::color::cuda_dispatch::rgb_from_ycbcr_f32_cuda);
impl_convert!(Rgbf64 => YCbCrf64, crate::color::ycbcr_from_rgb,
    cuda: crate::color::cuda_dispatch::ycbcr_from_rgb_f64_cuda);
impl_convert!(YCbCrf64 => Rgbf64, crate::color::rgb_from_ycbcr,
    cuda: crate::color::cuda_dispatch::rgb_from_ycbcr_f64_cuda);

// ===== RGB <-> YUV (planar 3-channel) Conversions =====
impl_convert!(Rgb8 => Yuv8, crate::color::yuv_from_rgb,
    cuda: crate::color::cuda_dispatch::yuv_from_rgb_u8_cuda);
impl_convert!(Yuv8 => Rgb8, crate::color::rgb_from_yuv,
    cuda: crate::color::cuda_dispatch::rgb_from_yuv_u8_cuda);
impl_convert!(Rgbf32 => Yuvf32, crate::color::yuv_from_rgb,
    cuda: crate::color::cuda_dispatch::yuv_from_rgb_f32_cuda);
impl_convert!(Yuvf32 => Rgbf32, crate::color::rgb_from_yuv,
    cuda: crate::color::cuda_dispatch::rgb_from_yuv_f32_cuda);
impl_convert!(Rgbf64 => Yuvf64, crate::color::yuv_from_rgb,
    cuda: crate::color::cuda_dispatch::yuv_from_rgb_f64_cuda);
impl_convert!(Yuvf64 => Rgbf64, crate::color::rgb_from_yuv,
    cuda: crate::color::cuda_dispatch::rgb_from_yuv_f64_cuda);

// ===== RGBA -> RGB Conversions =====
impl_convert!(Rgba8 => Rgb8, crate::color::rgb_from_rgba, bg: None,
    cuda: crate::color::cuda_dispatch::rgb_from_rgba_u8_cuda);

// ===== BGRA -> RGB Conversions =====
impl_convert!(Bgra8 => Rgb8, crate::color::rgb_from_bgra, bg: None,
    cuda: crate::color::cuda_dispatch::rgb_from_bgra_u8_cuda);

// ===== RGB -> RGBA Conversions (add opaque alpha) =====
impl_convert!(Rgb8 => Rgba8, crate::color::rgba_from_rgb,
    cuda: crate::color::cuda_dispatch::rgba_from_rgb_u8_cuda);
impl_convert!(Rgbf32 => Rgbaf32, crate::color::rgba_from_rgb,
    cuda: crate::color::cuda_dispatch::rgba_from_rgb_f32_cuda);

// ===== RGB -> BGRA Conversions (swap R/B + add opaque alpha) =====
impl_convert!(Rgb8 => Bgra8, crate::color::bgra_from_rgb,
    cuda: crate::color::cuda_dispatch::bgra_from_rgb_u8_cuda);
impl_convert!(Rgbf32 => Bgraf32, crate::color::bgra_from_rgb,
    cuda: crate::color::cuda_dispatch::bgra_from_rgb_f32_cuda);

// ===== Video format decode -> RGB (BT.601 limited range) =====
//
// The packed/planar video wrappers carry a raw byte buffer rather than a typed `Image`,
// so they get hand-written `ConvertColor` impls (the `impl_convert!` macro assumes `.0`).
macro_rules! impl_video_decode {
    ($src:ty, $func:path) => {
        impl ConvertColor<Rgb8> for $src {
            fn convert(&self, dst: &mut Rgb8) -> Result<(), ImageError> {
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

impl_convert_with_bg!(Rgba8 => Rgb8, crate::color::rgb_from_rgba,
    cuda: crate::color::cuda_dispatch::rgb_from_rgba_bg_u8_cuda);
impl_convert_with_bg!(Bgra8 => Rgb8, crate::color::rgb_from_bgra,
    cuda: crate::color::cuda_dispatch::rgb_from_bgra_bg_u8_cuda);

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::ImageSize;

    #[test]
    fn test_gray_from_rgb_f32() -> Result<(), ImageError> {
        let rgb = Rgbf32::from_size_vec(
            ImageSize {
                width: 2,
                height: 2,
            },
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5],
        )?;

        let mut gray = Grayf32::from_size_val(rgb.size(), 0.0)?;

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
        )?;

        let mut gray = Gray8::from_size_val(rgb.size(), 0)?;

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
        )?;

        let mut rgb = Rgbf32::from_size_val(gray.size(), 0.0)?;

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
        )?;

        let mut bgr = Bgr8::from_size_val(rgb.size(), 0)?;

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
        )?;

        let mut rgb = Rgb8::from_size_val(bgr.size(), 0)?;

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
        let rgb = Rgb8::from_size_vec(size, data)?;

        // YCbCr round-trip
        let mut ycc = YCbCr8::from_size_val(size, 0)?;
        let mut back = Rgb8::from_size_val(size, 0)?;
        rgb.convert(&mut ycc)?;
        ycc.convert(&mut back)?;
        for (a, b) in rgb.as_slice().iter().zip(back.as_slice().iter()) {
            assert!((*a as i32 - *b as i32).abs() <= 3);
        }

        // YUV (swapped chroma) must differ from YCbCr in channels 1/2 but round-trip too.
        let mut yuv = Yuv8::from_size_val(size, 0)?;
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
        let mut rgb = Rgb8::from_size_val(size, 0)?;
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
        )?;

        let mut hsv = Hsvf32::from_size_val(rgb.size(), 0.0)?;

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
        )?;

        let mut rgb = Rgb8::from_size_val(rgba.size(), 0)?;

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
        )?;

        let mut rgb = Rgb8::from_size_val(rgba.size(), 0)?;

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
        )?;

        let mut rgb = Rgb8::from_size_val(bgra.size(), 0)?;

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
        )?;
        let mut rgba = Rgba8::from_size_val(rgb.size(), 0)?;
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
        )?;
        let mut bgra = Bgra8::from_size_val(rgb.size(), 0)?;
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
        )?;

        let mut gray = Gray8::from_size_val(rgb.size(), 0)?;

        rgb.convert(&mut gray)?;
        assert_eq!(gray.num_channels(), 1);

        // Test Bgr8
        let mut bgr = Bgr8::from_size_val(rgb.size(), 0)?;
        rgb.convert(&mut bgr)?;
        assert_eq!(bgr.num_channels(), 3);

        // Test Rgba8
        let rgba = Rgba8::from_size_vec(
            ImageSize {
                width: 1,
                height: 1,
            },
            vec![255, 128, 64, 255],
        )?;
        let mut rgb_out = Rgb8::from_size_val(rgba.size(), 0)?;
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
}

/// Helper trait so `.cvt()` can read the source image size without coupling to
/// a concrete allocator.
///
/// All color-space newtypes implement this via their `Deref<Target = Image<…>>`.
pub trait SrcSize {
    /// Source image size.
    fn src_size(&self) -> ImageSize;
}

/// Emit both `NewColorImage` and `SrcSize` impls for a color-space newtype.
macro_rules! impl_color_newtype {
    ($newtype:ident, $t:ty) => {
        impl NewColorImage for kornia_image::color_spaces::$newtype {
            fn new_zeroed(size: ImageSize) -> Result<Self, ImageError> {
                kornia_image::color_spaces::$newtype::from_size_val(size, <$t>::default())
            }
        }

        impl SrcSize for kornia_image::color_spaces::$newtype {
            fn src_size(&self) -> ImageSize {
                self.size()
            }
        }
    };
}

impl_color_newtype!(Rgbf32, f32);
impl_color_newtype!(Bgrf32, f32);
impl_color_newtype!(Grayf32, f32);
impl_color_newtype!(Hsvf32, f32);
impl_color_newtype!(Hlsf32, f32);
impl_color_newtype!(Labf32, f32);
impl_color_newtype!(Luvf32, f32);
impl_color_newtype!(Xyzf32, f32);
impl_color_newtype!(LinearRgbf32, f32);
impl_color_newtype!(YCbCrf32, f32);
impl_color_newtype!(Yuvf32, f32);
impl_color_newtype!(Rgb8, u8);
impl_color_newtype!(Bgr8, u8);
impl_color_newtype!(Gray8, u8);
impl_color_newtype!(Rgba8, u8);
impl_color_newtype!(Bgra8, u8);
impl_color_newtype!(YCbCr8, u8);
impl_color_newtype!(Yuv8, u8);
impl_color_newtype!(Rgbaf32, f32);
impl_color_newtype!(Bgraf32, f32);

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
/// use kornia_imgproc::color::{Rgbf32, Hsvf32, ConvertColorExt};
///
/// let rgb = Rgbf32::from_size_vec(
///     ImageSize { width: 4, height: 3 },
///     vec![0.5f32; 4 * 3 * 3],
/// ).unwrap();
///
/// let hsv: Hsvf32 = rgb.cvt().unwrap();
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
    use kornia_image::color_spaces::Rgbf32;
    use kornia_image::{ColorSpace, DynImage, ImageSize};

    #[test]
    fn runtime_cvt_color_returns_tagged_dynimage() {
        let size = ImageSize {
            width: 4,
            height: 4,
        };
        let rgb = Rgbf32::from_size_vec(size, vec![0.5f32; 4 * 4 * 3]).unwrap();
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
        let size = ImageSize {
            width: 2,
            height: 2,
        };
        let rgb = Rgbf32::from_size_vec(size, vec![0.0f32; 2 * 2 * 3]).unwrap();
        // Rgb has no direct path to YCbCr? It does — pick a truly illegal target by
        // constructing from a non-Rgb source instead:
        let hsv = rgb.cvt_color(ColorSpace::Hsv).unwrap();
        let hsv_typed: kornia_image::color_spaces::Hsvf32 = hsv.try_into().unwrap();
        let err = hsv_typed.cvt_color(ColorSpace::Lab);
        assert!(err.is_err());
    }
}

#[cfg(test)]
mod cvt_ext_tests {
    use crate::color::ConvertColorExt;
    use kornia_image::color_spaces::{Grayf32, Hsvf32, Rgbf32};
    use kornia_image::ImageSize;

    #[test]
    fn cvt_allocates_and_converts_typed() {
        let size = ImageSize {
            width: 4,
            height: 3,
        };
        let rgb = Rgbf32::from_size_vec(size, vec![0.5f32; 4 * 3 * 3]).unwrap();
        // typed, allocating: no manual dst construction
        let hsv: Hsvf32 = rgb.cvt().unwrap();
        assert_eq!(hsv.size(), size);
        // channel-changing conversion is natural — Dst encodes C
        let gray: Grayf32 = rgb.cvt().unwrap();
        assert_eq!(gray.num_channels(), 1);
    }

    #[test]
    fn cvt_round_trip_rgb_hsv() {
        let size = ImageSize {
            width: 8,
            height: 8,
        };
        let data: Vec<f32> = (0..8 * 8 * 3).map(|i| (i % 255) as f32 / 255.0).collect();
        let rgb = Rgbf32::from_size_vec(size, data.clone()).unwrap();
        let hsv: Hsvf32 = rgb.cvt().unwrap();
        let back: Rgbf32 = hsv.cvt().unwrap();
        for (a, b) in data.iter().zip(back.as_slice().iter()) {
            assert!((a - b).abs() < 1e-3, "round-trip drift {a} vs {b}");
        }
    }
}

// ===== Runtime Tagged dispatch layer =====

use kornia_image::{ColorSpace, DynImage};

/// Runtime, color-space-tagged conversion. The source newtype encodes its
/// space and dtype, so only the target `to` is supplied; the result is a
/// `DynImage` tagged with `to`. Same legal set as the typed `.cvt()` path.
pub trait Tagged<T> {
    /// This image's color space.
    fn space(&self) -> ColorSpace;
    /// Convert to `to`, returning an owned tagged `DynImage`.
    fn cvt_color(&self, to: ColorSpace) -> Result<DynImage<T>, ImageError>;
}

/// Generates a `Tagged` impl for one source newtype. Each `to => Dst, Cn`
/// arm names the destination newtype and the DynImage channel constructor.
macro_rules! impl_tagged {
    ($src:ty, $t:ty, $space:expr_2021, { $( $to:ident => $dst:ty , $ctor:ident );* $(;)? }) => {
        impl Tagged<$t> for $src
        where Self: SrcSize {
            fn space(&self) -> ColorSpace { $space }
            fn cvt_color(&self, to: ColorSpace) -> Result<DynImage<$t>, ImageError> {
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
impl_tagged!(kornia_image::color_spaces::Rgbf32, f32, ColorSpace::Rgb, {
    Gray      => kornia_image::color_spaces::Grayf32, C1;
    Bgr       => kornia_image::color_spaces::Bgrf32, C3;
    Rgba      => kornia_image::color_spaces::Rgbaf32, C4;
    Bgra      => kornia_image::color_spaces::Bgraf32, C4;
    Hsv       => kornia_image::color_spaces::Hsvf32, C3;
    Hls       => kornia_image::color_spaces::Hlsf32, C3;
    Lab       => kornia_image::color_spaces::Labf32, C3;
    Luv       => kornia_image::color_spaces::Luvf32, C3;
    Xyz       => kornia_image::color_spaces::Xyzf32, C3;
    LinearRgb => kornia_image::color_spaces::LinearRgbf32, C3;
    YCbCr     => kornia_image::color_spaces::YCbCrf32, C3;
    Yuv       => kornia_image::color_spaces::Yuvf32, C3;
});

// ---- f32 inverse sources back to RGB ----
impl_tagged!(kornia_image::color_spaces::Hsvf32, f32, ColorSpace::Hsv, {
    Rgb => kornia_image::color_spaces::Rgbf32, C3;
});
impl_tagged!(kornia_image::color_spaces::Hlsf32, f32, ColorSpace::Hls, {
    Rgb => kornia_image::color_spaces::Rgbf32, C3;
});
impl_tagged!(kornia_image::color_spaces::Labf32, f32, ColorSpace::Lab, {
    Rgb => kornia_image::color_spaces::Rgbf32, C3;
});
impl_tagged!(kornia_image::color_spaces::Luvf32, f32, ColorSpace::Luv, {
    Rgb => kornia_image::color_spaces::Rgbf32, C3;
});
impl_tagged!(kornia_image::color_spaces::Xyzf32, f32, ColorSpace::Xyz, {
    Rgb => kornia_image::color_spaces::Rgbf32, C3;
});
impl_tagged!(kornia_image::color_spaces::LinearRgbf32, f32, ColorSpace::LinearRgb, {
    Rgb => kornia_image::color_spaces::Rgbf32, C3;
});
impl_tagged!(kornia_image::color_spaces::YCbCrf32, f32, ColorSpace::YCbCr, {
    Rgb => kornia_image::color_spaces::Rgbf32, C3;
});
impl_tagged!(kornia_image::color_spaces::Yuvf32, f32, ColorSpace::Yuv, {
    Rgb => kornia_image::color_spaces::Rgbf32, C3;
});
impl_tagged!(kornia_image::color_spaces::Bgrf32, f32, ColorSpace::Bgr, {
    Rgb => kornia_image::color_spaces::Rgbf32, C3;
});
impl_tagged!(kornia_image::color_spaces::Grayf32, f32, ColorSpace::Gray, {
    Rgb => kornia_image::color_spaces::Rgbf32, C3;
});

// ---- u8 RGB source: u8-valid targets ----
impl_tagged!(kornia_image::color_spaces::Rgb8, u8, ColorSpace::Rgb, {
    Gray  => kornia_image::color_spaces::Gray8, C1;
    Bgr   => kornia_image::color_spaces::Bgr8, C3;
    Rgba  => kornia_image::color_spaces::Rgba8, C4;
    Bgra  => kornia_image::color_spaces::Bgra8, C4;
    YCbCr => kornia_image::color_spaces::YCbCr8, C3;
    Yuv   => kornia_image::color_spaces::Yuv8, C3;
});
impl_tagged!(kornia_image::color_spaces::Gray8, u8, ColorSpace::Gray, {
    Rgb => kornia_image::color_spaces::Rgb8, C3;
});
impl_tagged!(kornia_image::color_spaces::Bgr8, u8, ColorSpace::Bgr, {
    Rgb => kornia_image::color_spaces::Rgb8, C3;
});
impl_tagged!(kornia_image::color_spaces::Rgba8, u8, ColorSpace::Rgba, {
    Rgb => kornia_image::color_spaces::Rgb8, C3;
});
impl_tagged!(kornia_image::color_spaces::Bgra8, u8, ColorSpace::Bgra, {
    Rgb => kornia_image::color_spaces::Rgb8, C3;
});

#[cfg(test)]
mod legality_drift_tests {
    //! Drift-lock: Tagged dispatch arms must agree with ColorSpace::supports_dtype().
    //!
    //! For every (from, to) pair: cvt_color(to).is_ok() must equal
    //! ColorSpace::supports_dtype(from, to, is_f32).  If an impl_tagged! arm is
    //! added or removed without updating the supports table (or vice-versa), this
    //! test will catch the regression.
    //!
    //! `supports_dtype` is used (rather than `supports`) because the Tagged
    //! dispatch is also gated by element type: f32-only spaces (Hsv, Hls, Lab,
    //! Luv, Xyz, LinearRgb) cannot be produced from u8 source images.

    use super::Tagged;
    use kornia_image::color_spaces::{
        Bgr8, Bgra8, Bgrf32, Gray8, Grayf32, Hlsf32, Hsvf32, Labf32, LinearRgbf32, Luvf32, Rgb8,
        Rgba8, Rgbf32, Xyzf32, YCbCrf32, Yuvf32,
    };
    use kornia_image::{ColorSpace, ImageSize};

    /// All ColorSpace variants enumerated manually (no strum/EnumIter in scope).
    /// 13 variants × 16 source types = 208 (from, to) pairs checked.
    const ALL_SPACES: &[ColorSpace] = &[
        ColorSpace::Rgb,
        ColorSpace::Bgr,
        ColorSpace::Gray,
        ColorSpace::Rgba,
        ColorSpace::Bgra,
        ColorSpace::Hsv,
        ColorSpace::Hls,
        ColorSpace::Lab,
        ColorSpace::Luv,
        ColorSpace::Xyz,
        ColorSpace::LinearRgb,
        ColorSpace::YCbCr,
        ColorSpace::Yuv,
    ];

    fn size() -> ImageSize {
        ImageSize {
            width: 2,
            height: 2,
        }
    }

    /// Check one source image against every possible target.
    /// `is_f32` controls whether f32-only spaces are considered supported.
    macro_rules! check_all_targets {
        ($src:expr_2021, $from:expr_2021, $is_f32:expr_2021) => {{
            let src = $src;
            let from: ColorSpace = $from;
            let is_f32: bool = $is_f32;
            for &to in ALL_SPACES {
                let result = src.cvt_color(to);
                let expected_ok = ColorSpace::supports_dtype(from, to, is_f32);
                assert_eq!(
                    result.is_ok(),
                    expected_ok,
                    "{:?} -> {:?} (is_f32={}): supports_dtype={} but cvt_color returned {}",
                    from,
                    to,
                    is_f32,
                    expected_ok,
                    if result.is_ok() { "Ok" } else { "Err" },
                );
            }
        }};
    }

    #[test]
    fn tagged_dispatch_matches_supports_table() {
        let sz = size();

        // ---- f32 sources (is_f32 = true) ----

        check_all_targets!(
            Rgbf32::from_size_vec(sz, vec![0.5f32; 2 * 2 * 3]).unwrap(),
            ColorSpace::Rgb,
            true
        );
        check_all_targets!(
            Grayf32::from_size_vec(sz, vec![0.5f32; 2 * 2]).unwrap(),
            ColorSpace::Gray,
            true
        );
        check_all_targets!(
            Hsvf32::from_size_vec(sz, vec![0.5f32; 2 * 2 * 3]).unwrap(),
            ColorSpace::Hsv,
            true
        );
        check_all_targets!(
            Hlsf32::from_size_vec(sz, vec![0.5f32; 2 * 2 * 3]).unwrap(),
            ColorSpace::Hls,
            true
        );
        check_all_targets!(
            Labf32::from_size_vec(sz, vec![0.5f32; 2 * 2 * 3]).unwrap(),
            ColorSpace::Lab,
            true
        );
        check_all_targets!(
            Luvf32::from_size_vec(sz, vec![0.5f32; 2 * 2 * 3]).unwrap(),
            ColorSpace::Luv,
            true
        );
        check_all_targets!(
            Xyzf32::from_size_vec(sz, vec![0.5f32; 2 * 2 * 3]).unwrap(),
            ColorSpace::Xyz,
            true
        );
        check_all_targets!(
            LinearRgbf32::from_size_vec(sz, vec![0.5f32; 2 * 2 * 3]).unwrap(),
            ColorSpace::LinearRgb,
            true
        );
        check_all_targets!(
            YCbCrf32::from_size_vec(sz, vec![0.5f32; 2 * 2 * 3]).unwrap(),
            ColorSpace::YCbCr,
            true
        );
        check_all_targets!(
            Yuvf32::from_size_vec(sz, vec![0.5f32; 2 * 2 * 3]).unwrap(),
            ColorSpace::Yuv,
            true
        );
        check_all_targets!(
            Bgrf32::from_size_vec(sz, vec![0.5f32; 2 * 2 * 3]).unwrap(),
            ColorSpace::Bgr,
            true
        );

        // ---- u8 sources (is_f32 = false) ----

        check_all_targets!(
            Rgb8::from_size_vec(sz, vec![128u8; 2 * 2 * 3]).unwrap(),
            ColorSpace::Rgb,
            false
        );
        check_all_targets!(
            Gray8::from_size_vec(sz, vec![128u8; 2 * 2]).unwrap(),
            ColorSpace::Gray,
            false
        );
        check_all_targets!(
            Bgr8::from_size_vec(sz, vec![128u8; 2 * 2 * 3]).unwrap(),
            ColorSpace::Bgr,
            false
        );
        check_all_targets!(
            Rgba8::from_size_vec(sz, vec![128u8; 2 * 2 * 4]).unwrap(),
            ColorSpace::Rgba,
            false
        );
        check_all_targets!(
            Bgra8::from_size_vec(sz, vec![128u8; 2 * 2 * 4]).unwrap(),
            ColorSpace::Bgra,
            false
        );
    }
}
