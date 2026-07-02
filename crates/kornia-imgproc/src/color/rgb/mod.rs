use crate::parallel;
use kornia_image::{Image, ImageError};

pub(crate) mod kernels;

// ===== Sealed-trait dispatch =========================================================

use crate::color::kernel_common::{check_size, sealed};

/// Compile-time dispatch to the right channel/alpha kernels for each pixel type.
///
/// `u8` and `f32` get NEON kernels (AVX2 scaffold → scalar); `f64` (and any other
/// type via the generic public fns) uses the portable scalar path. Sealed: no
/// external implementations.
pub trait ChannelOps: sealed::Sealed + Sized + Copy + Send + Sync {
    #[doc(hidden)]
    fn bgr_from_rgb_impl(src: &Image<Self, 3>, dst: &mut Image<Self, 3>) -> Result<(), ImageError>;

    #[doc(hidden)]
    fn rgba_from_rgb_impl(src: &Image<Self, 3>, dst: &mut Image<Self, 4>)
        -> Result<(), ImageError>;

    #[doc(hidden)]
    fn bgra_from_rgb_impl(src: &Image<Self, 3>, dst: &mut Image<Self, 4>)
        -> Result<(), ImageError>;
}

impl ChannelOps for u8 {
    fn bgr_from_rgb_impl(src: &Image<u8, 3>, dst: &mut Image<u8, 3>) -> Result<(), ImageError> {
        check_size(src, dst)?;
        kernels::bgr_from_rgb_u8(src.as_slice(), dst.as_slice_mut(), src.rows() * src.cols());
        Ok(())
    }
    fn rgba_from_rgb_impl(src: &Image<u8, 3>, dst: &mut Image<u8, 4>) -> Result<(), ImageError> {
        check_size(src, dst)?;
        kernels::rgba_from_rgb_u8(src.as_slice(), dst.as_slice_mut(), src.rows() * src.cols());
        Ok(())
    }
    fn bgra_from_rgb_impl(src: &Image<u8, 3>, dst: &mut Image<u8, 4>) -> Result<(), ImageError> {
        check_size(src, dst)?;
        kernels::bgra_from_rgb_u8(src.as_slice(), dst.as_slice_mut(), src.rows() * src.cols());
        Ok(())
    }
}

impl ChannelOps for f32 {
    fn bgr_from_rgb_impl(src: &Image<f32, 3>, dst: &mut Image<f32, 3>) -> Result<(), ImageError> {
        check_size(src, dst)?;
        kernels::bgr_from_rgb_f32(src.as_slice(), dst.as_slice_mut(), src.rows() * src.cols());
        Ok(())
    }
    fn rgba_from_rgb_impl(src: &Image<f32, 3>, dst: &mut Image<f32, 4>) -> Result<(), ImageError> {
        check_size(src, dst)?;
        kernels::rgba_from_rgb_f32(src.as_slice(), dst.as_slice_mut(), src.rows() * src.cols());
        Ok(())
    }
    fn bgra_from_rgb_impl(src: &Image<f32, 3>, dst: &mut Image<f32, 4>) -> Result<(), ImageError> {
        check_size(src, dst)?;
        kernels::bgra_from_rgb_f32(src.as_slice(), dst.as_slice_mut(), src.rows() * src.cols());
        Ok(())
    }
}

impl ChannelOps for f64 {
    fn bgr_from_rgb_impl(src: &Image<f64, 3>, dst: &mut Image<f64, 3>) -> Result<(), ImageError> {
        check_size(src, dst)?;
        parallel::par_iter_rows(src, dst, |s, d| {
            d[0] = s[2];
            d[1] = s[1];
            d[2] = s[0];
        });
        Ok(())
    }
    fn rgba_from_rgb_impl(src: &Image<f64, 3>, dst: &mut Image<f64, 4>) -> Result<(), ImageError> {
        check_size(src, dst)?;
        parallel::par_iter_rows(src, dst, |s, d| {
            d[0] = s[0];
            d[1] = s[1];
            d[2] = s[2];
            d[3] = 1.0;
        });
        Ok(())
    }
    fn bgra_from_rgb_impl(src: &Image<f64, 3>, dst: &mut Image<f64, 4>) -> Result<(), ImageError> {
        check_size(src, dst)?;
        parallel::par_iter_rows(src, dst, |s, d| {
            d[0] = s[2];
            d[1] = s[1];
            d[2] = s[0];
            d[3] = 1.0;
        });
        Ok(())
    }
}

// ===== RGBA / BGRA -> RGB (drop alpha) =============================================

/// Convert an RGBA image to RGB image.
///
/// # Arguments
///
/// * `src` - The input RGBA image.
/// * `dst` - The output RGB image.
///
/// Precondition: the input image must have 4 channels.
/// Precondition: the output image must have 3 channels.
/// Precondition: the input and output images must have the same size.
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_imgproc::color::rgb_from_rgba;
///
/// let src = Image::<u8, 4>::new(ImageSize { width: 3, height: 2 }, vec![
///     0, 1, 2, 255, // (0, 0)
///     3, 4, 5, 255, // (0, 1)
///     6, 7, 8, 255, // (0, 2)
///     9, 10, 11, 255, // (1, 0)
///     12, 13, 14, 255, // (1, 1)
///     15, 16, 17, 255, // (1, 2)
/// ]).unwrap();
///
/// let mut dst = Image::<u8, 3>::new(ImageSize { width: 3, height: 2 }, vec![0; 18]).unwrap();
///
/// rgb_from_rgba(&src, &mut dst, None).unwrap();
/// ```
pub fn rgb_from_rgba(
    src: &Image<u8, 4>,
    dst: &mut Image<u8, 3>,
    background: Option<[u8; 3]>,
) -> Result<(), ImageError> {
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }

    if let Some(bg) = background {
        // alpha blend the background with the source image
        parallel::par_iter_rows(src, dst, |src_pixel, dst_pixel| {
            let (r, g, b, a) = (src_pixel[0], src_pixel[1], src_pixel[2], src_pixel[3]);
            alpha_blend(r, g, b, a, &bg, dst_pixel);
        });
    } else {
        // just drop the alpha channel in the last index
        parallel::par_iter_rows(src, dst, |src_pixel, dst_pixel| {
            dst_pixel[..3].copy_from_slice(&src_pixel[..3]);
        });
    }

    Ok(())
}

/// Convert a BGRA image to RGB.
///
/// # Arguments
///
/// * `src` - The input BGRA image.
/// * `dst` - The output RGB image.
///
/// Precondition: the input image must have 4 channels.
/// Precondition: the output image must have 3 channels.
/// Precondition: the input and output images must have the same size.
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_imgproc::color::rgb_from_bgra;
///
/// let src = Image::<u8, 4>::new(ImageSize { width: 3, height: 2 }, vec![
///     0, 1, 2, 255, // (0, 0)
///     3, 4, 5, 255, // (0, 1)
///     6, 7, 8, 255, // (0, 2)
///     9, 10, 11, 255, // (1, 0)
///     12, 13, 14, 255, // (1, 1)
///     15, 16, 17, 255, // (1, 2)
/// ]).unwrap();
///
/// let mut dst = Image::<u8, 3>::from_size_val(src.size(), 0).unwrap();
///
/// rgb_from_bgra(&src, &mut dst, None).unwrap();
/// ```
pub fn rgb_from_bgra(
    src: &Image<u8, 4>,
    dst: &mut Image<u8, 3>,
    background: Option<[u8; 3]>,
) -> Result<(), ImageError> {
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }

    if let Some(bg) = background {
        // alpha blend the background with the source image
        parallel::par_iter_rows(src, dst, |src_pixel, dst_pixel| {
            let (b, g, r, a) = (src_pixel[0], src_pixel[1], src_pixel[2], src_pixel[3]);
            alpha_blend(r, g, b, a, &bg, dst_pixel);
        });
    } else {
        // flip only the red and blue channels, keep the green channel as is
        parallel::par_iter_rows(src, dst, |src_pixel, dst_pixel| {
            let (b, g, r) = (src_pixel[0], src_pixel[1], src_pixel[2]);
            dst_pixel[0] = r;
            dst_pixel[1] = g;
            dst_pixel[2] = b;
        });
    }

    Ok(())
}

// ===== RGB <-> BGR (channel reverse) ==============================================

/// Convert an RGB image to BGR by swapping the red and blue channels.
///
/// This is symmetric and is also used for BGR → RGB. Dispatches at compile time on
/// pixel type `T`: `u8`/`f32` use NEON (`vld3q`/`vst3q`); `f64` uses the portable
/// scalar path.
///
/// # Arguments
///
/// * `src` - The input RGB image.
/// * `dst` - The output BGR image.
///
/// Precondition: the input and output images must have the same size.
pub fn bgr_from_rgb<T>(src: &Image<T, 3>, dst: &mut Image<T, 3>) -> Result<(), ImageError>
where
    T: ChannelOps,
{
    T::bgr_from_rgb_impl(src, dst)
}

// ===== RGB -> RGBA / BGRA (add opaque alpha) ======================================

/// Convert an RGB image to RGBA by appending an opaque alpha channel.
///
/// The alpha value is the maximally opaque value for the type: `255` for `u8`,
/// `1.0` for floating-point. Dispatches at compile time on pixel type `T`:
/// `u8`/`f32` use NEON (`vld3q` + `vst4q`); `f64` uses the portable scalar path.
///
/// # Arguments
///
/// * `src` - The input RGB image.
/// * `dst` - The output RGBA image.
///
/// Precondition: the input and output images must have the same size.
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_imgproc::color::rgba_from_rgb;
///
/// let src = Image::<u8, 3>::new(
///     ImageSize { width: 2, height: 1 },
///     vec![1, 2, 3, 4, 5, 6],
/// ).unwrap();
/// let mut dst = Image::<u8, 4>::from_size_val(src.size(), 0).unwrap();
/// rgba_from_rgb(&src, &mut dst).unwrap();
/// ```
pub fn rgba_from_rgb<T>(src: &Image<T, 3>, dst: &mut Image<T, 4>) -> Result<(), ImageError>
where
    T: ChannelOps,
{
    T::rgba_from_rgb_impl(src, dst)
}

/// Convert an RGB image to BGRA by swapping R/B and appending an opaque alpha channel.
///
/// The alpha value is the maximally opaque value for the type: `255` for `u8`,
/// `1.0` for floating-point. Dispatches at compile time on pixel type `T`:
/// `u8`/`f32` use NEON (`vld3q` + `vst4q`); `f64` uses the portable scalar path.
///
/// # Arguments
///
/// * `src` - The input RGB image.
/// * `dst` - The output BGRA image.
///
/// Precondition: the input and output images must have the same size.
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_imgproc::color::bgra_from_rgb;
///
/// let src = Image::<u8, 3>::new(
///     ImageSize { width: 2, height: 1 },
///     vec![1, 2, 3, 4, 5, 6],
/// ).unwrap();
/// let mut dst = Image::<u8, 4>::from_size_val(src.size(), 0).unwrap();
/// bgra_from_rgb(&src, &mut dst).unwrap();
/// ```
pub fn bgra_from_rgb<T>(src: &Image<T, 3>, dst: &mut Image<T, 4>) -> Result<(), ImageError>
where
    T: ChannelOps,
{
    T::bgra_from_rgb_impl(src, dst)
}

#[inline]
fn alpha_blend(r: u8, g: u8, b: u8, a: u8, bg: &[u8; 3], rgb: &mut [u8]) {
    let alpha = a as f32 / 255.0;
    rgb[0] = (r as f32 * alpha + bg[0] as f32 * (1.0 - alpha)).round() as u8;
    rgb[1] = (g as f32 * alpha + bg[1] as f32 * (1.0 - alpha)).round() as u8;
    rgb[2] = (b as f32 * alpha + bg[2] as f32 * (1.0 - alpha)).round() as u8;
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::ImageSize;

    #[test]
    fn test_rgb_from_rgba() -> Result<(), ImageError> {
        // NOTE: verified with opencv
        let src = Image::<u8, 4>::new(
            ImageSize {
                width: 3,
                height: 2,
            },
            vec![
                0, 1, 2, 255, // (0, 0)
                3, 4, 5, 255, // (0, 1)
                6, 7, 8, 255, // (0, 2)
                9, 10, 11, 255, // (1, 0)
                12, 13, 14, 255, // (1, 1)
                15, 16, 17, 255, // (1, 2)
            ],
        )?;

        let mut dst = Image::<u8, 3>::from_size_val(src.size(), 0)?;

        let expected = Image::<u8, 3>::new(
            ImageSize {
                width: 3,
                height: 2,
            },
            vec![
                0, 1, 2, // (0, 0)
                3, 4, 5, // (0, 1)
                6, 7, 8, // (0, 2)
                9, 10, 11, // (1, 0)
                12, 13, 14, // (1, 1)
                15, 16, 17, // (1, 2)
            ],
        )?;

        rgb_from_rgba(&src, &mut dst, None)?;

        assert_eq!(dst.as_slice(), expected.as_slice());

        Ok(())
    }

    #[test]
    fn test_rgb_from_rgba_with_background() -> Result<(), ImageError> {
        // NOTE: verified with PIL
        let src = Image::<u8, 4>::new(
            ImageSize {
                width: 1,
                height: 1,
            },
            vec![255, 0, 0, 128],
        )?;

        let mut dst = Image::<u8, 3>::from_size_val(src.size(), 0)?;

        let expected = Image::<u8, 3>::new(
            ImageSize {
                width: 1,
                height: 1,
            },
            vec![178, 50, 50],
        )?;

        rgb_from_rgba(&src, &mut dst, Some([100, 100, 100]))?;

        assert_eq!(dst.as_slice(), expected.as_slice());

        Ok(())
    }

    #[test]
    fn test_rgb_from_bgra_without_background() -> Result<(), ImageError> {
        // NOTE: verified with opencv
        let src = Image::<u8, 4>::new(
            ImageSize {
                width: 1,
                height: 1,
            },
            vec![0, 0, 255, 128],
        )?;

        let mut dst = Image::<u8, 3>::from_size_val(src.size(), 0)?;

        let expected = Image::<u8, 3>::new(
            ImageSize {
                width: 1,
                height: 1,
            },
            vec![255, 0, 0],
        )?;

        rgb_from_bgra(&src, &mut dst, None)?;

        assert_eq!(dst.as_slice(), expected.as_slice());

        Ok(())
    }

    #[test]
    fn test_rgb_from_bgra_with_background() -> Result<(), ImageError> {
        // NOTE: verified with opencv
        let src = Image::<u8, 4>::new(
            ImageSize {
                width: 1,
                height: 1,
            },
            vec![0, 0, 255, 128],
        )?;

        let mut dst = Image::<u8, 3>::from_size_val(src.size(), 0)?;

        let expected = Image::<u8, 3>::new(
            ImageSize {
                width: 1,
                height: 1,
            },
            vec![178, 50, 50],
        )?;

        rgb_from_bgra(&src, &mut dst, Some([100, 100, 100]))?;

        assert_eq!(dst.as_slice(), expected.as_slice());

        Ok(())
    }

    #[test]
    fn test_bgr_from_rgb() -> Result<(), ImageError> {
        #[rustfmt::skip]
        let image = Image::new(
            ImageSize {
                width: 1,
                height: 3,
            },
            vec![
                0.0_f32, 1.0, 2.0,
                3.0, 4.0, 5.0,
                6.0, 7.0, 8.0,
            ],
        )?;

        let mut bgr = Image::<f32, 3>::from_size_val(image.size(), 0.0)?;

        super::bgr_from_rgb(&image, &mut bgr)?;

        #[rustfmt::skip]
        let expected: Image<f32, 3> = Image::new(
            ImageSize {
                width: 1,
                height: 3,
            },
            vec![
                2.0, 1.0, 0.0,
                5.0, 4.0, 3.0,
                8.0, 7.0, 6.0,
            ],
        )?;

        assert_eq!(bgr.as_slice(), expected.as_slice());

        Ok(())
    }

    // ----- SIMD-vs-scalar parity (odd 7x3 = 21 px: exercises 16-px NEON body + tail) -----

    fn ramp_u8(n: usize) -> Vec<u8> {
        (0..n).map(|v| (v * 7 % 256) as u8).collect()
    }
    fn ramp_f32(n: usize) -> Vec<f32> {
        (0..n).map(|v| (v as f32) * 0.5).collect()
    }

    #[test]
    fn bgr_from_rgb_u8_large_strip() -> Result<(), ImageError> {
        // 1024x1025 = 1,049,600 px > PAR_THRESHOLD: exercises the rayon strip split
        // and the per-strip source-slice offset (3->3 path).
        let size = ImageSize {
            width: 1024,
            height: 1025,
        };
        let npix = 1024 * 1025;
        let src = Image::<u8, 3>::new(size, ramp_u8(npix * 3))?;
        let mut dst = Image::<u8, 3>::from_size_val(size, 0)?;
        super::bgr_from_rgb(&src, &mut dst)?;
        let s = src.as_slice();
        let d = dst.as_slice();
        for i in 0..npix {
            assert_eq!(d[i * 3], s[i * 3 + 2], "B at px {i}");
            assert_eq!(d[i * 3 + 1], s[i * 3 + 1], "G at px {i}");
            assert_eq!(d[i * 3 + 2], s[i * 3], "R at px {i}");
        }
        Ok(())
    }

    #[test]
    fn rgba_from_rgb_u8_large_strip() -> Result<(), ImageError> {
        // > PAR_THRESHOLD: exercises the 3->4 strip split (dst sized by dst pixels,
        // src offset by 3 elements/pixel).
        let size = ImageSize {
            width: 1024,
            height: 1025,
        };
        let npix = 1024 * 1025;
        let src = Image::<u8, 3>::new(size, ramp_u8(npix * 3))?;
        let mut dst = Image::<u8, 4>::from_size_val(size, 0)?;
        super::rgba_from_rgb(&src, &mut dst)?;
        let s = src.as_slice();
        let d = dst.as_slice();
        for i in 0..npix {
            assert_eq!(d[i * 4], s[i * 3], "R at px {i}");
            assert_eq!(d[i * 4 + 1], s[i * 3 + 1], "G at px {i}");
            assert_eq!(d[i * 4 + 2], s[i * 3 + 2], "B at px {i}");
            assert_eq!(d[i * 4 + 3], 255, "A at px {i}");
        }
        Ok(())
    }

    #[test]
    fn bgr_from_rgb_u8_matches_scalar() -> Result<(), ImageError> {
        let size = ImageSize {
            width: 7,
            height: 3,
        };
        let src = Image::<u8, 3>::new(size, ramp_u8(7 * 3 * 3))?;
        let mut dst = Image::<u8, 3>::from_size_val(size, 0)?;
        super::bgr_from_rgb(&src, &mut dst)?;

        let s = src.as_slice();
        for i in 0..7 * 3 {
            assert_eq!(dst.as_slice()[i * 3], s[i * 3 + 2], "B at px {i}");
            assert_eq!(dst.as_slice()[i * 3 + 1], s[i * 3 + 1], "G at px {i}");
            assert_eq!(dst.as_slice()[i * 3 + 2], s[i * 3], "R at px {i}");
        }
        Ok(())
    }

    #[test]
    fn bgr_from_rgb_u8_round_trip() -> Result<(), ImageError> {
        let size = ImageSize {
            width: 7,
            height: 3,
        };
        let src = Image::<u8, 3>::new(size, ramp_u8(7 * 3 * 3))?;
        let mut bgr = Image::<u8, 3>::from_size_val(size, 0)?;
        let mut back = Image::<u8, 3>::from_size_val(size, 0)?;
        super::bgr_from_rgb(&src, &mut bgr)?;
        super::bgr_from_rgb(&bgr, &mut back)?;
        assert_eq!(src.as_slice(), back.as_slice());
        Ok(())
    }

    #[test]
    fn bgr_from_rgb_f32_round_trip() -> Result<(), ImageError> {
        let size = ImageSize {
            width: 7,
            height: 3,
        };
        let src = Image::<f32, 3>::new(size, ramp_f32(7 * 3 * 3))?;
        let mut bgr = Image::<f32, 3>::from_size_val(size, 0.0)?;
        let mut back = Image::<f32, 3>::from_size_val(size, 0.0)?;
        super::bgr_from_rgb(&src, &mut bgr)?;
        super::bgr_from_rgb(&bgr, &mut back)?;
        assert_eq!(src.as_slice(), back.as_slice());
        Ok(())
    }

    #[test]
    fn rgba_from_rgb_u8_matches_scalar() -> Result<(), ImageError> {
        let size = ImageSize {
            width: 7,
            height: 3,
        };
        let src = Image::<u8, 3>::new(size, ramp_u8(7 * 3 * 3))?;
        let mut dst = Image::<u8, 4>::from_size_val(size, 0)?;
        super::rgba_from_rgb(&src, &mut dst)?;

        let s = src.as_slice();
        for i in 0..7 * 3 {
            assert_eq!(dst.as_slice()[i * 4], s[i * 3], "R at px {i}");
            assert_eq!(dst.as_slice()[i * 4 + 1], s[i * 3 + 1], "G at px {i}");
            assert_eq!(dst.as_slice()[i * 4 + 2], s[i * 3 + 2], "B at px {i}");
            assert_eq!(dst.as_slice()[i * 4 + 3], 255, "A at px {i}");
        }
        Ok(())
    }

    #[test]
    fn rgba_from_rgb_u8_known_value() -> Result<(), ImageError> {
        let src = Image::<u8, 3>::new(
            ImageSize {
                width: 2,
                height: 1,
            },
            vec![1, 2, 3, 4, 5, 6],
        )?;
        let mut dst = Image::<u8, 4>::from_size_val(src.size(), 0)?;
        super::rgba_from_rgb(&src, &mut dst)?;
        assert_eq!(dst.as_slice(), &[1, 2, 3, 255, 4, 5, 6, 255]);
        Ok(())
    }

    #[test]
    fn rgba_from_rgb_f32_matches_scalar() -> Result<(), ImageError> {
        let size = ImageSize {
            width: 7,
            height: 3,
        };
        let src = Image::<f32, 3>::new(size, ramp_f32(7 * 3 * 3))?;
        let mut dst = Image::<f32, 4>::from_size_val(size, 0.0)?;
        super::rgba_from_rgb(&src, &mut dst)?;

        let s = src.as_slice();
        for i in 0..7 * 3 {
            assert_eq!(dst.as_slice()[i * 4], s[i * 3]);
            assert_eq!(dst.as_slice()[i * 4 + 1], s[i * 3 + 1]);
            assert_eq!(dst.as_slice()[i * 4 + 2], s[i * 3 + 2]);
            assert_eq!(dst.as_slice()[i * 4 + 3], 1.0);
        }
        Ok(())
    }

    #[test]
    fn bgra_from_rgb_u8_matches_scalar() -> Result<(), ImageError> {
        let size = ImageSize {
            width: 7,
            height: 3,
        };
        let src = Image::<u8, 3>::new(size, ramp_u8(7 * 3 * 3))?;
        let mut dst = Image::<u8, 4>::from_size_val(size, 0)?;
        super::bgra_from_rgb(&src, &mut dst)?;

        let s = src.as_slice();
        for i in 0..7 * 3 {
            assert_eq!(dst.as_slice()[i * 4], s[i * 3 + 2], "B at px {i}");
            assert_eq!(dst.as_slice()[i * 4 + 1], s[i * 3 + 1], "G at px {i}");
            assert_eq!(dst.as_slice()[i * 4 + 2], s[i * 3], "R at px {i}");
            assert_eq!(dst.as_slice()[i * 4 + 3], 255, "A at px {i}");
        }
        Ok(())
    }

    #[test]
    fn bgra_from_rgb_u8_known_value() -> Result<(), ImageError> {
        let src = Image::<u8, 3>::new(
            ImageSize {
                width: 2,
                height: 1,
            },
            vec![1, 2, 3, 4, 5, 6],
        )?;
        let mut dst = Image::<u8, 4>::from_size_val(src.size(), 0)?;
        super::bgra_from_rgb(&src, &mut dst)?;
        assert_eq!(dst.as_slice(), &[3, 2, 1, 255, 6, 5, 4, 255]);
        Ok(())
    }

    #[test]
    fn bgra_from_rgb_f32_matches_scalar() -> Result<(), ImageError> {
        let size = ImageSize {
            width: 7,
            height: 3,
        };
        let src = Image::<f32, 3>::new(size, ramp_f32(7 * 3 * 3))?;
        let mut dst = Image::<f32, 4>::from_size_val(size, 0.0)?;
        super::bgra_from_rgb(&src, &mut dst)?;

        let s = src.as_slice();
        for i in 0..7 * 3 {
            assert_eq!(dst.as_slice()[i * 4], s[i * 3 + 2]);
            assert_eq!(dst.as_slice()[i * 4 + 1], s[i * 3 + 1]);
            assert_eq!(dst.as_slice()[i * 4 + 2], s[i * 3]);
            assert_eq!(dst.as_slice()[i * 4 + 3], 1.0);
        }
        Ok(())
    }
}
