use crate::parallel;
use kornia_image::{Image, ImageError};

pub(crate) mod kernels;

// ===== Sealed-trait dispatch =========================================================

use crate::color::kernel_common::{check_size, sealed};

/// Compile-time dispatch to the right RGB→HLS kernel for each pixel type.
///
/// Implemented for `f32` (NEON / AVX2) and `f64` (portable scalar). Sealed.
pub trait HlsFromRgb: sealed::Sealed + Sized {
    #[doc(hidden)]
    fn hls_from_rgb_impl(src: &Image<Self, 3>, dst: &mut Image<Self, 3>) -> Result<(), ImageError>;
}

/// Compile-time dispatch to the right HLS→RGB kernel for each pixel type.
pub trait RgbFromHls: sealed::Sealed + Sized {
    #[doc(hidden)]
    fn rgb_from_hls_impl(src: &Image<Self, 3>, dst: &mut Image<Self, 3>) -> Result<(), ImageError>;
}

impl HlsFromRgb for f32 {
    fn hls_from_rgb_impl(src: &Image<f32, 3>, dst: &mut Image<f32, 3>) -> Result<(), ImageError> {
        check_size(src, dst)?;
        kernels::hls_from_rgb_f32(src.as_slice(), dst.as_slice_mut(), src.rows() * src.cols());
        Ok(())
    }
}
impl RgbFromHls for f32 {
    fn rgb_from_hls_impl(src: &Image<f32, 3>, dst: &mut Image<f32, 3>) -> Result<(), ImageError> {
        check_size(src, dst)?;
        kernels::rgb_from_hls_f32(src.as_slice(), dst.as_slice_mut(), src.rows() * src.cols());
        Ok(())
    }
}

impl HlsFromRgb for f64 {
    fn hls_from_rgb_impl(src: &Image<f64, 3>, dst: &mut Image<f64, 3>) -> Result<(), ImageError> {
        check_size(src, dst)?;
        parallel::par_iter_rows(src, dst, |s, d| {
            let (h, l, sa) = hls_from_rgb_scalar_f64(s[0], s[1], s[2]);
            d[0] = h;
            d[1] = l;
            d[2] = sa;
        });
        Ok(())
    }
}
impl RgbFromHls for f64 {
    fn rgb_from_hls_impl(src: &Image<f64, 3>, dst: &mut Image<f64, 3>) -> Result<(), ImageError> {
        check_size(src, dst)?;
        parallel::par_iter_rows(src, dst, |s, d| {
            let (r, g, b) = rgb_from_hls_scalar_f64(s[0], s[1], s[2]);
            d[0] = r;
            d[1] = g;
            d[2] = b;
        });
        Ok(())
    }
}

// f64 scalar oracle (values in [0,255]); mirrors the f32 kernel math exactly.
// Image channel order is [H, L, S].
#[inline]
fn hls_from_rgb_scalar_f64(r8: f64, g8: f64, b8: f64) -> (f64, f64, f64) {
    let (r, g, b) = (r8 / 255.0, g8 / 255.0, b8 / 255.0);
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let diff = max - min;
    let sum = max + min;
    let l = sum * 0.5;
    let (h, s) = if diff == 0.0 {
        (0.0, 0.0)
    } else {
        let s = if l <= 0.5 {
            diff / sum
        } else {
            diff / (2.0 - sum)
        };
        let h = if max == r {
            60.0 * (((g - b) / diff) % 6.0)
        } else if max == g {
            60.0 * (((b - r) / diff) + 2.0)
        } else {
            60.0 * (((r - g) / diff) + 4.0)
        };
        let h = if h < 0.0 { h + 360.0 } else { h };
        (h, s)
    };
    ((h / 360.0) * 255.0, l * 255.0, s * 255.0)
}

#[inline]
fn rgb_from_hls_scalar_f64(h8: f64, l8: f64, s8: f64) -> (f64, f64, f64) {
    let l = l8 / 255.0;
    let s = s8 / 255.0;
    if s == 0.0 {
        return (l * 255.0, l * 255.0, l * 255.0);
    }
    let h_deg = (h8 / 255.0) * 360.0;
    let q = if l < 0.5 {
        l * (1.0 + s)
    } else {
        l + s - l * s
    };
    let p = 2.0 * l - q;
    let hk = h_deg / 360.0;
    let r = hue2rgb_f64(p, q, hk + 1.0 / 3.0);
    let g = hue2rgb_f64(p, q, hk);
    let b = hue2rgb_f64(p, q, hk - 1.0 / 3.0);
    (r * 255.0, g * 255.0, b * 255.0)
}

#[inline]
fn hue2rgb_f64(p: f64, q: f64, t: f64) -> f64 {
    let t = if t < 0.0 { t + 1.0 } else { t };
    let t = if t > 1.0 { t - 1.0 } else { t };
    if t < 1.0 / 6.0 {
        p + (q - p) * 6.0 * t
    } else if t < 0.5 {
        q
    } else if t < 2.0 / 3.0 {
        p + (q - p) * (2.0 / 3.0 - t) * 6.0
    } else {
        p
    }
}

// ===== Public API ==================================================================

/// Convert an RGB image to HLS.
///
/// Channel values are in `[0, 255]`. Output H encodes `[0, 360)` degrees scaled to
/// `[0, 255]`; L and S are `[0, 255]`. Image channel order is `[H, L, S]`. Dispatches
/// at compile time on pixel type `T` (`f32` → NEON/AVX2, `f64` → portable scalar).
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_imgproc::color::hls_from_rgb;
///
/// let rgb = Image::<f32, 3>::from_size_val(
///     ImageSize { width: 4, height: 5 }, 0.0).unwrap();
/// let mut hls = Image::<f32, 3>::from_size_val(rgb.size(), 0.0).unwrap();
/// hls_from_rgb(&rgb, &mut hls).unwrap();
/// ```
pub fn hls_from_rgb<T>(src: &Image<T, 3>, dst: &mut Image<T, 3>) -> Result<(), ImageError>
where
    T: HlsFromRgb,
{
    T::hls_from_rgb_impl(src, dst)
}

/// Convert an HLS image back to RGB. Inverse of [`hls_from_rgb`].
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_imgproc::color::rgb_from_hls;
///
/// let hls = Image::<f32, 3>::from_size_val(
///     ImageSize { width: 4, height: 5 }, 0.0).unwrap();
/// let mut rgb = Image::<f32, 3>::from_size_val(hls.size(), 0.0).unwrap();
/// rgb_from_hls(&hls, &mut rgb).unwrap();
/// ```
pub fn rgb_from_hls<T>(src: &Image<T, 3>, dst: &mut Image<T, 3>) -> Result<(), ImageError>
where
    T: RgbFromHls,
{
    T::rgb_from_hls_impl(src, dst)
}

/// Convert an RGB f32 image to HLS. Thin wrapper around [`hls_from_rgb`].
pub fn hls_from_rgb_f32(src: &Image<f32, 3>, dst: &mut Image<f32, 3>) -> Result<(), ImageError> {
    hls_from_rgb(src, dst)
}

/// Convert an HLS f32 image to RGB. Thin wrapper around [`rgb_from_hls`].
pub fn rgb_from_hls_f32(src: &Image<f32, 3>, dst: &mut Image<f32, 3>) -> Result<(), ImageError> {
    rgb_from_hls(src, dst)
}

#[cfg(test)]
mod tests {
    use kornia_image::{Image, ImageError, ImageSize};

    // Hand-computed via the documented HLS math (channel order [H, L, S]).
    #[test]
    fn hls_from_rgb_regression() -> Result<(), ImageError> {
        // Pure red (255,0,0): max=1,min=0,diff=1,L=0.5,S=1 (L<=0.5 branch),H=0.
        // Pure green (0,255,0): H=120°,L=0.5,S=1.
        // Pure blue (0,0,255): H=240°,L=0.5,S=1.
        let image = Image::<f32, 3>::new(
            ImageSize {
                width: 3,
                height: 1,
            },
            vec![255.0, 0.0, 0.0, 0.0, 255.0, 0.0, 0.0, 0.0, 255.0],
        )?;
        let expected = [
            // H_out=(H/360)*255, L_out=L*255, S_out=S*255
            0.0,
            127.5,
            255.0, // red:   H=0
            (120.0 / 360.0) * 255.0,
            127.5,
            255.0, // green: H=120
            (240.0 / 360.0) * 255.0,
            127.5,
            255.0, // blue:  H=240
        ];
        let mut hls = Image::<f32, 3>::from_size_val(image.size(), 0.0)?;
        super::hls_from_rgb(&image, &mut hls)?;
        for (a, b) in hls.as_slice().iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-3, "{a} != {b}");
        }
        Ok(())
    }

    #[test]
    fn rgb_hls_round_trip() -> Result<(), ImageError> {
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
        let mut hls = Image::<f32, 3>::from_size_val(src.size(), 0.0)?;
        let mut back = Image::<f32, 3>::from_size_val(src.size(), 0.0)?;
        super::hls_from_rgb(&src, &mut hls)?;
        super::rgb_from_hls(&hls, &mut back)?;
        for (a, b) in src.as_slice().iter().zip(back.as_slice().iter()) {
            assert!((a - b).abs() < 0.3, "round-trip {a} != {b}");
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
        super::hls_from_rgb(&src, &mut simd)?;
        super::hls_from_rgb(&src64, &mut scalar)?;
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
        let mut hls = Image::<f32, 3>::from_size_val(src.size(), 0.0)?;
        let mut back = Image::<f32, 3>::from_size_val(src.size(), 0.0)?;
        super::hls_from_rgb(&src, &mut hls)?;
        super::rgb_from_hls(&hls, &mut back)?;
        for (a, b) in src.as_slice().iter().zip(back.as_slice().iter()) {
            assert!((a - b).abs() < 0.3);
        }
        Ok(())
    }
}
