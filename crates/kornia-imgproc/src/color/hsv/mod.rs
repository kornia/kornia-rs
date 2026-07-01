use crate::parallel;
use kornia_image::{Image, ImageError};

mod kernels;

// ===== Sealed-trait dispatch =========================================================

use crate::color::kernel_common::{check_size, sealed};

/// Compile-time dispatch to the right RGB→HSV kernel for each pixel type.
///
/// Implemented for `f32` (NEON / AVX2) and `f64` (portable scalar). Sealed.
pub trait HsvFromRgb: sealed::Sealed + Sized {
    #[doc(hidden)]
    fn hsv_from_rgb_impl(src: &Image<Self, 3>, dst: &mut Image<Self, 3>) -> Result<(), ImageError>;
}

/// Compile-time dispatch to the right HSV→RGB kernel for each pixel type.
pub trait RgbFromHsv: sealed::Sealed + Sized {
    #[doc(hidden)]
    fn rgb_from_hsv_impl(src: &Image<Self, 3>, dst: &mut Image<Self, 3>) -> Result<(), ImageError>;
}

impl HsvFromRgb for f32 {
    fn hsv_from_rgb_impl(src: &Image<f32, 3>, dst: &mut Image<f32, 3>) -> Result<(), ImageError> {
        check_size(src, dst)?;
        kernels::hsv_from_rgb_f32(src.as_slice(), dst.as_slice_mut(), src.rows() * src.cols());
        Ok(())
    }
}
impl RgbFromHsv for f32 {
    fn rgb_from_hsv_impl(src: &Image<f32, 3>, dst: &mut Image<f32, 3>) -> Result<(), ImageError> {
        check_size(src, dst)?;
        kernels::rgb_from_hsv_f32(src.as_slice(), dst.as_slice_mut(), src.rows() * src.cols());
        Ok(())
    }
}

impl HsvFromRgb for f64 {
    fn hsv_from_rgb_impl(src: &Image<f64, 3>, dst: &mut Image<f64, 3>) -> Result<(), ImageError> {
        check_size(src, dst)?;
        parallel::par_iter_rows(src, dst, |s, d| {
            let (h, sa, v) = hsv_from_rgb_scalar_f64(s[0], s[1], s[2]);
            d[0] = h;
            d[1] = sa;
            d[2] = v;
        });
        Ok(())
    }
}
impl RgbFromHsv for f64 {
    fn rgb_from_hsv_impl(src: &Image<f64, 3>, dst: &mut Image<f64, 3>) -> Result<(), ImageError> {
        check_size(src, dst)?;
        parallel::par_iter_rows(src, dst, |s, d| {
            let (r, g, b) = rgb_from_hsv_scalar_f64(s[0], s[1], s[2]);
            d[0] = r;
            d[1] = g;
            d[2] = b;
        });
        Ok(())
    }
}

// f64 scalar oracle (values in [0,255]); mirrors the f32 kernel math exactly.
#[inline]
fn hsv_from_rgb_scalar_f64(r8: f64, g8: f64, b8: f64) -> (f64, f64, f64) {
    let (r, g, b) = (r8 / 255.0, g8 / 255.0, b8 / 255.0);
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;
    let h = if delta == 0.0 {
        0.0
    } else if max == r {
        60.0 * (((g - b) / delta) % 6.0)
    } else if max == g {
        60.0 * (((b - r) / delta) + 2.0)
    } else {
        60.0 * (((r - g) / delta) + 4.0)
    };
    let h = if h < 0.0 { h + 360.0 } else { h };
    let s = if max == 0.0 {
        0.0
    } else {
        (delta / max) * 255.0
    };
    ((h / 360.0) * 255.0, s, max * 255.0)
}

#[inline]
fn rgb_from_hsv_scalar_f64(h8: f64, s8: f64, v8: f64) -> (f64, f64, f64) {
    let s = s8 / 255.0;
    let v = v8 / 255.0;
    let hh = (h8 / 255.0) * 6.0; // [0,6)
    let c = v * s;
    let hmod2 = hh - 2.0 * (hh * 0.5).floor();
    let x = c * (1.0 - (hmod2 - 1.0).abs());
    let m = v - c;
    let (r1, g1, b1) = match hh.floor() as i32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    ((r1 + m) * 255.0, (g1 + m) * 255.0, (b1 + m) * 255.0)
}

// ===== Public API ==================================================================

/// Convert an RGB image to HSV.
///
/// Channel values are in `[0, 255]`. Output H encodes `[0, 360)` degrees scaled to
/// `[0, 255]`; S and V are `[0, 255]`. Dispatches at compile time on pixel type `T`
/// (`f32` → NEON/AVX2, `f64` → portable scalar).
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_imgproc::color::hsv_from_rgb;
///
/// let rgb = Image::<f32, 3>::from_size_val(
///     ImageSize { width: 4, height: 5 }, 0.0).unwrap();
/// let mut hsv = Image::<f32, 3>::from_size_val(rgb.size(), 0.0).unwrap();
/// hsv_from_rgb(&rgb, &mut hsv).unwrap();
/// ```
pub fn hsv_from_rgb<T>(src: &Image<T, 3>, dst: &mut Image<T, 3>) -> Result<(), ImageError>
where
    T: HsvFromRgb,
{
    T::hsv_from_rgb_impl(src, dst)
}

/// Convert an HSV image back to RGB. Inverse of [`hsv_from_rgb`].
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_imgproc::color::rgb_from_hsv;
///
/// let hsv = Image::<f32, 3>::from_size_val(
///     ImageSize { width: 4, height: 5 }, 0.0).unwrap();
/// let mut rgb = Image::<f32, 3>::from_size_val(hsv.size(), 0.0).unwrap();
/// rgb_from_hsv(&hsv, &mut rgb).unwrap();
/// ```
pub fn rgb_from_hsv<T>(src: &Image<T, 3>, dst: &mut Image<T, 3>) -> Result<(), ImageError>
where
    T: RgbFromHsv,
{
    T::rgb_from_hsv_impl(src, dst)
}

/// Convert an RGB f32 image to HSV. Thin wrapper around [`hsv_from_rgb`].
pub fn hsv_from_rgb_f32(src: &Image<f32, 3>, dst: &mut Image<f32, 3>) -> Result<(), ImageError> {
    hsv_from_rgb(src, dst)
}

/// Convert an HSV f32 image to RGB. Thin wrapper around [`rgb_from_hsv`].
pub fn rgb_from_hsv_f32(src: &Image<f32, 3>, dst: &mut Image<f32, 3>) -> Result<(), ImageError> {
    rgb_from_hsv(src, dst)
}

#[cfg(test)]
mod tests {
    use kornia_image::{Image, ImageError, ImageSize};

    #[test]
    fn hsv_from_rgb_regression() -> Result<(), ImageError> {
        let image = Image::<f32, 3>::new(
            ImageSize {
                width: 2,
                height: 3,
            },
            vec![
                0.0, 128.0, 255.0, 255.0, 128.0, 0.0, 128.0, 255.0, 0.0, 255.0, 0.0, 128.0, 0.0,
                128.0, 255.0, 255.0, 128.0, 0.0,
            ],
        )?;
        let expected = [
            148.66667, 255.0, 255.0, 21.333334, 255.0, 255.0, 63.666668, 255.0, 255.0, 233.66667,
            255.0, 255.0, 148.66667, 255.0, 255.0, 21.333334, 255.0, 255.0,
        ];
        let mut hsv = Image::<f32, 3>::from_size_val(image.size(), 0.0)?;
        super::hsv_from_rgb(&image, &mut hsv)?;
        for (a, b) in hsv.as_slice().iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-3, "{a} != {b}");
        }
        Ok(())
    }

    #[test]
    fn rgb_hsv_round_trip() -> Result<(), ImageError> {
        // dense-ish sweep; exercises 4-px NEON body + scalar tail (7 px wide)
        let npix = 7 * 5;
        let data: Vec<f32> = (0..npix * 3).map(|v| (v % 256) as f32).collect();
        let src = Image::<f32, 3>::new(
            ImageSize {
                width: 7,
                height: 5,
            },
            data,
        )?;
        let mut hsv = Image::<f32, 3>::from_size_val(src.size(), 0.0)?;
        let mut back = Image::<f32, 3>::from_size_val(src.size(), 0.0)?;
        super::hsv_from_rgb(&src, &mut hsv)?;
        super::rgb_from_hsv(&hsv, &mut back)?;
        for (a, b) in src.as_slice().iter().zip(back.as_slice().iter()) {
            assert!((a - b).abs() < 0.2, "round-trip {a} != {b}");
        }
        Ok(())
    }

    #[test]
    fn f32_simd_matches_f64_scalar() -> Result<(), ImageError> {
        let npix = 7 * 3;
        let data: Vec<f32> = (0..npix * 3).map(|v| (v * 7 % 256) as f32).collect();
        let src = Image::<f32, 3>::new(
            ImageSize {
                width: 7,
                height: 3,
            },
            data.clone(),
        )?;
        let src64 = Image::<f64, 3>::new(src.size(), data.iter().map(|&v| v as f64).collect())?;
        let mut simd = Image::<f32, 3>::from_size_val(src.size(), 0.0)?;
        let mut scalar = Image::<f64, 3>::from_size_val(src.size(), 0.0)?;
        super::hsv_from_rgb(&src, &mut simd)?;
        super::hsv_from_rgb(&src64, &mut scalar)?;
        for (a, b) in simd.as_slice().iter().zip(scalar.as_slice().iter()) {
            assert!((*a as f64 - b).abs() < 1e-3, "{a} != {b}");
        }
        Ok(())
    }

    #[test]
    fn large_image_strip_path() -> Result<(), ImageError> {
        // > PAR_THRESHOLD (1,048,576) to exercise the rayon strip split.
        let (w, h) = (1024, 1025);
        let data: Vec<f32> = (0..w * h * 3).map(|v| (v % 256) as f32).collect();
        let src = Image::<f32, 3>::new(
            ImageSize {
                width: w,
                height: h,
            },
            data,
        )?;
        let mut hsv = Image::<f32, 3>::from_size_val(src.size(), 0.0)?;
        let mut back = Image::<f32, 3>::from_size_val(src.size(), 0.0)?;
        super::hsv_from_rgb(&src, &mut hsv)?;
        super::rgb_from_hsv(&hsv, &mut back)?;
        for (a, b) in src.as_slice().iter().zip(back.as_slice().iter()) {
            assert!((a - b).abs() < 0.2);
        }
        Ok(())
    }
}
