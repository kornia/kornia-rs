use crate::parallel;
use kornia_image::{Image, ImageError};

pub(crate) mod kernels;

// ===== Sealed-trait dispatch =========================================================

use crate::color::kernel_common::{check_size, sealed};

/// Compile-time dispatch to the right RGB→gray kernel for each pixel type.
///
/// Implemented for `u8`, `f32`, and `f64`. Sealed: no external implementations.
///
/// | Type  | Kernel                                        |
/// |-------|-----------------------------------------------|
/// | `u8`  | `(77·R + 150·G + 29·B) >> 8` — NEON / AVX2  |
/// | `f32` | `0.299·R + 0.587·G + 0.114·B` — NEON / AVX2+FMA |
/// | `f64` | Same weights, portable scalar                 |
pub trait GrayFromRgb: sealed::Sealed + Sized {
    #[doc(hidden)]
    fn gray_from_rgb_impl(src: &Image<Self, 3>, dst: &mut Image<Self, 1>)
        -> Result<(), ImageError>;
}

impl GrayFromRgb for u8 {
    fn gray_from_rgb_impl(src: &Image<u8, 3>, dst: &mut Image<u8, 1>) -> Result<(), ImageError> {
        check_size(src, dst)?;
        kernels::gray_from_rgb_u8(src.as_slice(), dst.as_slice_mut(), src.rows() * src.cols());
        Ok(())
    }
}

impl GrayFromRgb for f32 {
    fn gray_from_rgb_impl(src: &Image<f32, 3>, dst: &mut Image<f32, 1>) -> Result<(), ImageError> {
        check_size(src, dst)?;
        kernels::gray_from_rgb_f32(src.as_slice(), dst.as_slice_mut(), src.rows() * src.cols());
        Ok(())
    }
}

impl GrayFromRgb for f64 {
    fn gray_from_rgb_impl(src: &Image<f64, 3>, dst: &mut Image<f64, 1>) -> Result<(), ImageError> {
        check_size(src, dst)?;
        parallel::par_iter_rows(src, dst, |src_pixel, dst_pixel| {
            dst_pixel[0] = 0.299 * src_pixel[0] + 0.587 * src_pixel[1] + 0.114 * src_pixel[2];
        });
        Ok(())
    }
}

// ===== Public API ==================================================================

/// Convert an RGB image to grayscale.
///
/// Y = 0.299·R + 0.587·G + 0.114·B
///
/// Dispatches at compile time based on pixel type `T`:
///
/// | `T`   | Path                                              |
/// |-------|---------------------------------------------------|
/// | `u8`  | `(77·R + 150·G + 29·B) >> 8` via NEON or AVX2   |
/// | `f32` | FMA-fused BT.601 via NEON `vld3q_f32` or AVX2   |
/// | `f64` | Portable scalar BT.601                           |
///
/// Large images (> 1 M px) are split across Rayon threads regardless of type.
///
/// # Arguments
///
/// * `src` - The source RGB image with shape (H, W, 3).
/// * `dst` - The destination grayscale image with shape (H, W, 1). Must have
///   the same spatial dimensions as `src`.
///
/// # Returns
///
/// Returns `Ok(())` on success.
///
/// # Errors
///
/// Returns [`ImageError::InvalidImageSize`] if `src` and `dst` have different
/// spatial dimensions.
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_imgproc::color::gray_from_rgb;
///
/// // u8 — NEON / AVX2
/// let rgb = Image::<u8, 3>::new(
///     ImageSize { width: 4, height: 5 },
///     vec![0u8; 4 * 5 * 3],
/// ).unwrap();
/// let mut gray = Image::<u8, 1>::from_size_val(rgb.size(), 0).unwrap();
/// gray_from_rgb(&rgb, &mut gray).unwrap();
///
/// // f32 — NEON / AVX2+FMA
/// let rgb_f32 = Image::<f32, 3>::new(
///     ImageSize { width: 4, height: 5 },
///     vec![0f32; 4 * 5 * 3],
/// ).unwrap();
/// let mut gray_f32 = Image::<f32, 1>::from_size_val(rgb_f32.size(), 0.0).unwrap();
/// gray_from_rgb(&rgb_f32, &mut gray_f32).unwrap();
/// ```
pub fn gray_from_rgb<T>(src: &Image<T, 3>, dst: &mut Image<T, 1>) -> Result<(), ImageError>
where
    T: GrayFromRgb,
{
    T::gray_from_rgb_impl(src, dst)
}

/// Convert an RGB8 image to grayscale (`u8`).
///
/// Thin wrapper around [`gray_from_rgb`] for backward compatibility.
///
/// # Arguments
///
/// * `src` - The source RGB u8 image with shape (H, W, 3).
/// * `dst` - The destination grayscale u8 image with shape (H, W, 1).
///
/// # Returns
///
/// Returns `Ok(())` on success.
///
/// # Errors
///
/// Returns [`ImageError::InvalidImageSize`] if `src` and `dst` differ in size.
pub fn gray_from_rgb_u8(src: &Image<u8, 3>, dst: &mut Image<u8, 1>) -> Result<(), ImageError> {
    gray_from_rgb(src, dst)
}

/// Convert an RGB f32 image to grayscale (`f32`).
///
/// Thin wrapper around [`gray_from_rgb`] for backward compatibility.
///
/// # Arguments
///
/// * `src` - The source RGB f32 image with shape (H, W, 3).
/// * `dst` - The destination grayscale f32 image with shape (H, W, 1).
///
/// # Returns
///
/// Returns `Ok(())` on success.
///
/// # Errors
///
/// Returns [`ImageError::InvalidImageSize`] if `src` and `dst` differ in size.
pub fn gray_from_rgb_f32(src: &Image<f32, 3>, dst: &mut Image<f32, 1>) -> Result<(), ImageError> {
    gray_from_rgb(src, dst)
}

// ===== Other conversions ===========================================================

/// Compile-time dispatch to the right gray→RGB broadcast kernel for each pixel type.
///
/// `u8` and `f32` get NEON kernels (`vld1q` + `vst3q`); any other `T` (e.g. `f64`,
/// `u16`) uses the portable scalar broadcast. Sealed: no external implementations.
pub trait RgbFromGray: sealed::Sealed + Copy + Send + Sync {
    #[doc(hidden)]
    fn rgb_from_gray_impl(src: &Image<Self, 1>, dst: &mut Image<Self, 3>)
        -> Result<(), ImageError>;
}

// Generic scalar broadcast — the oracle and the path for every non-NEON `T`.
#[inline]
fn rgb_from_gray_scalar<T>(src: &Image<T, 1>, dst: &mut Image<T, 3>) -> Result<(), ImageError>
where
    T: Copy + Send + Sync,
{
    check_size(src, dst)?;
    parallel::par_iter_rows(src, dst, |src_pixel, dst_pixel| {
        dst_pixel[0] = src_pixel[0];
        dst_pixel[1] = src_pixel[0];
        dst_pixel[2] = src_pixel[0];
    });
    Ok(())
}

macro_rules! impl_rgb_from_gray_scalar {
    ($($t:ty),*) => {$(
        impl RgbFromGray for $t {
            fn rgb_from_gray_impl(
                src: &Image<$t, 1>,
                dst: &mut Image<$t, 3>,
            ) -> Result<(), ImageError> {
                rgb_from_gray_scalar(src, dst)
            }
        }
    )*};
}
impl_rgb_from_gray_scalar!(u16, i32, f64);

impl RgbFromGray for u8 {
    fn rgb_from_gray_impl(src: &Image<u8, 1>, dst: &mut Image<u8, 3>) -> Result<(), ImageError> {
        check_size(src, dst)?;
        kernels::rgb_from_gray_u8(src.as_slice(), dst.as_slice_mut(), src.rows() * src.cols());
        Ok(())
    }
}

impl RgbFromGray for f32 {
    fn rgb_from_gray_impl(src: &Image<f32, 1>, dst: &mut Image<f32, 3>) -> Result<(), ImageError> {
        check_size(src, dst)?;
        kernels::rgb_from_gray_f32(src.as_slice(), dst.as_slice_mut(), src.rows() * src.cols());
        Ok(())
    }
}

/// Convert a grayscale image to RGB by replicating the value across all three channels.
///
/// Dispatches at compile time on pixel type `T`: `u8`/`f32` use NEON
/// (`vld1q` + `vst3q`); any other `T` uses a portable scalar broadcast.
///
/// # Arguments
///
/// * `src` - The source grayscale image with shape (H, W, 1).
/// * `dst` - The destination RGB image with shape (H, W, 3). Must have the
///   same spatial dimensions as `src`.
///
/// # Returns
///
/// Returns `Ok(())` on success.
///
/// # Errors
///
/// Returns [`ImageError::InvalidImageSize`] if `src` and `dst` have different
/// spatial dimensions.
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_imgproc::color::rgb_from_gray;
///
/// let image = Image::<f32, 1>::new(
///     ImageSize { width: 4, height: 5 },
///     vec![0f32; 4 * 5 * 1],
/// ).unwrap();
/// let mut rgb = Image::<f32, 3>::from_size_val(image.size(), 0.0).unwrap();
/// rgb_from_gray(&image, &mut rgb).unwrap();
/// ```
pub fn rgb_from_gray<T>(src: &Image<T, 1>, dst: &mut Image<T, 3>) -> Result<(), ImageError>
where
    T: RgbFromGray,
{
    T::rgb_from_gray_impl(src, dst)
}

#[cfg(test)]
mod tests {
    use kornia_image::{ops, Image, ImageSize};
    use kornia_io::jpeg::read_image_jpeg_rgb8;

    #[test]
    fn test_gray_from_rgb() -> Result<(), Box<dyn std::error::Error>> {
        let image = read_image_jpeg_rgb8("../../tests/data/dog.jpeg")?;

        let mut image_norm = Image::from_size_val(image.size(), 0.0)?;
        ops::cast_and_scale(&image, &mut image_norm, 1. / 255.0)?;

        let mut gray = Image::<f32, 1>::from_size_val(image_norm.size(), 0.0)?;
        super::gray_from_rgb(&image_norm, &mut gray)?;

        assert_eq!(gray.num_channels(), 1);
        assert_eq!(gray.cols(), 258);
        assert_eq!(gray.rows(), 195);

        Ok(())
    }

    #[test]
    fn test_gray_from_rgb_regression() -> Result<(), Box<dyn std::error::Error>> {
        #[rustfmt::skip]
        let image = Image::new(
            ImageSize { width: 2, height: 3 },
            vec![
                1.0_f32, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
            ],
        )?;

        let mut gray = Image::<f32, 1>::from_size_val(image.size(), 0.0)?;
        super::gray_from_rgb(&image, &mut gray)?;

        let expected: Image<f32, 1> = Image::new(
            ImageSize {
                width: 2,
                height: 3,
            },
            vec![0.299, 0.587, 0.114, 0.0, 0.0, 0.0],
        )?;

        for (a, b) in gray.as_slice().iter().zip(expected.as_slice().iter()) {
            assert!((a - b).abs() < 1e-6);
        }

        Ok(())
    }

    #[test]
    fn test_rgb_from_grayscale() -> Result<(), Box<dyn std::error::Error>> {
        let image = Image::new(
            ImageSize {
                width: 2,
                height: 3,
            },
            vec![0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0],
        )?;

        let mut rgb = Image::<f32, 3>::from_size_val(image.size(), 0.0)?;
        super::rgb_from_gray(&image, &mut rgb)?;

        #[rustfmt::skip]
        let expected: Image<f32, 3> = Image::new(
            ImageSize { width: 2, height: 3 },
            vec![
                0.0, 0.0, 0.0,
                1.0, 1.0, 1.0,
                2.0, 2.0, 2.0,
                3.0, 3.0, 3.0,
                4.0, 4.0, 4.0,
                5.0, 5.0, 5.0,
            ],
        )?;

        assert_eq!(rgb.as_slice(), expected.as_slice());

        Ok(())
    }

    // ----- rgb_from_gray broadcast: NEON vs scalar oracle -----

    #[test]
    fn rgb_from_gray_u8_matches_scalar() -> Result<(), Box<dyn std::error::Error>> {
        // 7x3 = 21 px exercises the 16-px NEON body + 5-px scalar tail.
        let size = ImageSize {
            width: 7,
            height: 3,
        };
        let gray: Vec<u8> = (0..21).map(|v| (v * 11 % 256) as u8).collect();
        let src = Image::<u8, 1>::new(size, gray.clone())?;
        let mut rgb = Image::<u8, 3>::from_size_val(size, 0)?;
        super::rgb_from_gray(&src, &mut rgb)?;
        for (i, &g) in gray.iter().enumerate() {
            assert_eq!(rgb.as_slice()[i * 3], g);
            assert_eq!(rgb.as_slice()[i * 3 + 1], g);
            assert_eq!(rgb.as_slice()[i * 3 + 2], g);
        }
        Ok(())
    }

    #[test]
    fn rgb_from_gray_u8_large_strip() -> Result<(), Box<dyn std::error::Error>> {
        // > PAR_THRESHOLD: exercises the 1->3 strip split.
        let size = ImageSize {
            width: 1024,
            height: 1025,
        };
        let npix = 1024 * 1025;
        let gray: Vec<u8> = (0..npix).map(|v| (v % 256) as u8).collect();
        let src = Image::<u8, 1>::new(size, gray.clone())?;
        let mut rgb = Image::<u8, 3>::from_size_val(size, 0)?;
        super::rgb_from_gray(&src, &mut rgb)?;
        for (i, &g) in gray.iter().enumerate() {
            assert_eq!(rgb.as_slice()[i * 3], g, "px {i}");
            assert_eq!(rgb.as_slice()[i * 3 + 1], g, "px {i}");
            assert_eq!(rgb.as_slice()[i * 3 + 2], g, "px {i}");
        }
        Ok(())
    }

    #[test]
    fn rgb_from_gray_f32_large_strip() -> Result<(), Box<dyn std::error::Error>> {
        let size = ImageSize {
            width: 1024,
            height: 1025,
        };
        let npix = 1024 * 1025;
        let gray: Vec<f32> = (0..npix).map(|v| (v % 256) as f32 * 0.25).collect();
        let src = Image::<f32, 1>::new(size, gray.clone())?;
        let mut rgb = Image::<f32, 3>::from_size_val(size, 0.0)?;
        super::rgb_from_gray(&src, &mut rgb)?;
        for (i, &g) in gray.iter().enumerate() {
            assert_eq!(rgb.as_slice()[i * 3], g, "px {i}");
            assert_eq!(rgb.as_slice()[i * 3 + 1], g, "px {i}");
            assert_eq!(rgb.as_slice()[i * 3 + 2], g, "px {i}");
        }
        Ok(())
    }

    #[test]
    fn test_gray_from_rgb_u8() -> Result<(), Box<dyn std::error::Error>> {
        let image = Image::new(
            ImageSize {
                width: 1,
                height: 2,
            },
            vec![0u8, 128, 255, 128, 0, 128],
        )?;

        let mut gray = Image::<u8, 1>::from_size_val(image.size(), 0)?;
        // unified entry point dispatches to the u8 NEON/AVX2 kernel
        super::gray_from_rgb(&image, &mut gray)?;

        // OpenCV Q14 values — cv2.cvtColor(RGB2GRAY) of these pixels gives exactly this.
        assert_eq!(gray.as_slice(), &[104, 53]);

        Ok(())
    }

    // ----- f32 grayscale tests -----

    #[test]
    fn test_gray_from_rgb_f32_regression() -> Result<(), Box<dyn std::error::Error>> {
        #[rustfmt::skip]
        let src = Image::new(
            ImageSize { width: 6, height: 1 },
            vec![
                1.0_f32, 0.0, 0.0,
                0.0,     1.0, 0.0,
                0.0,     0.0, 1.0,
                0.0,     0.0, 0.0,
                1.0,     1.0, 1.0,
                0.5,     0.5, 0.5,
            ],
        )?;

        let mut dst = Image::<f32, 1>::from_size_val(src.size(), 0.0)?;
        super::gray_from_rgb(&src, &mut dst)?;

        let expected = [0.299_f32, 0.587, 0.114, 0.0, 1.0, 0.5];
        for (got, exp) in dst.as_slice().iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-5, "got {got}, expected {exp}");
        }

        Ok(())
    }

    #[test]
    fn test_gray_from_rgb_f32_odd_width() -> Result<(), Box<dyn std::error::Error>> {
        // 7×3 = 21 pixels — exercises the 4-pixel step and 1-pixel scalar tail of the
        // NEON kernel (21 = 2×8 + 4 + 1), and the 8-pixel AVX2 tail.
        let src = Image::new(
            ImageSize {
                width: 7,
                height: 3,
            },
            (0..63).map(|v| v as f32 / 62.0).collect::<Vec<_>>(),
        )?;

        let mut dst_simd = Image::<f32, 1>::from_size_val(src.size(), 0.0)?;
        let mut dst_scalar = Image::<f64, 1>::from_size_val(src.size(), 0.0)?;
        let src_f64 = Image::new(
            src.size(),
            src.as_slice().iter().map(|&v| v as f64).collect::<Vec<_>>(),
        )?;

        super::gray_from_rgb(&src, &mut dst_simd)?;
        super::gray_from_rgb(&src_f64, &mut dst_scalar)?;

        for (i, (a, b)) in dst_simd
            .as_slice()
            .iter()
            .zip(dst_scalar.as_slice().iter())
            .enumerate()
        {
            assert!(
                (*a as f64 - b).abs() < 1e-5,
                "pixel {i}: f32-SIMD {a} != f64-scalar {b}"
            );
        }

        Ok(())
    }

    #[test]
    fn test_gray_from_rgb_f32_large() -> Result<(), Box<dyn std::error::Error>> {
        // 1024×1025 = 1 049 600 px — just above PAR_THRESHOLD (1 048 576), triggering the
        // rayon strip-split path. Verifies strip-boundary correctness and thread consistency.
        let npix = 1024 * 1025;
        let src = Image::new(
            ImageSize {
                width: 1024,
                height: 1025,
            },
            (0..npix * 3)
                .map(|v| (v % 256) as f32 / 255.0)
                .collect::<Vec<_>>(),
        )?;

        let mut dst_simd = Image::<f32, 1>::from_size_val(src.size(), 0.0)?;
        let mut dst_scalar = Image::<f64, 1>::from_size_val(src.size(), 0.0)?;
        let src_f64 = Image::new(
            src.size(),
            src.as_slice().iter().map(|&v| v as f64).collect::<Vec<_>>(),
        )?;

        super::gray_from_rgb(&src, &mut dst_simd)?;
        super::gray_from_rgb(&src_f64, &mut dst_scalar)?;

        for (i, (a, b)) in dst_simd
            .as_slice()
            .iter()
            .zip(dst_scalar.as_slice().iter())
            .enumerate()
        {
            assert!(
                (*a as f64 - b).abs() < 1e-5,
                "pixel {i}: f32-SIMD {a} != f64-scalar {b}"
            );
        }

        Ok(())
    }
}
