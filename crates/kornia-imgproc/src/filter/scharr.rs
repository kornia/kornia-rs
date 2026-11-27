use kornia_image::{Image, ImageError};

/// Scharr smoothing taps `[3, 10, 3]`, applied across the derivative axis.
const SMOOTH_OUTER: f32 = 3.0;
const SMOOTH_CENTER: f32 = 10.0;
/// `1 / (2 * (3 + 10 + 3))`: the scale of [`super::kernels::normalized_scharr_kernel3`].
const SCALE: f32 = 1.0 / 32.0;

/// Compute the Scharr x and y derivatives of a single-channel f32 image, separably.
///
/// Same kernels, scale (1/32) and replicated border as
/// [`super::scharr_spatial_gradient_float`], evaluated as two passes per row:
/// a vertical `[3, 10, 3]` smoothing and `[-1, 0, 1]` difference, then the
/// horizontal pass. That is 10 operations per pixel instead of 18
/// multiply-adds, and results differ from the 3x3 version by float rounding
/// only. For integer-valued pixels (a u8 image cast to f32) both functions
/// return the same values, and those equal OpenCV's
/// `cv::Scharr(src, CV_32F, 1, 0, 1/32, 0, BORDER_REPLICATE)` (and the `dy` one).
///
/// # Arguments
///
/// * `src` - The source image.
/// * `dx` - The x-derivative, same size as `src`.
/// * `dy` - The y-derivative, same size as `src`.
///
/// # Returns
///
/// `Ok(())` on success.
///
/// # Errors
///
/// Returns [`ImageError::InvalidImageSize`] if `dx` or `dy` differ in size from `src`.
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_imgproc::filter::scharr_spatial_gradient_separable_f32;
///
/// let size = ImageSize { width: 4, height: 3 };
/// let src = Image::<f32, 1>::new(size, (0..12).map(|v| (v % 4) as f32).collect()).unwrap();
/// let mut dx = Image::<f32, 1>::from_size_val(size, 0.0).unwrap();
/// let mut dy = Image::<f32, 1>::from_size_val(size, 0.0).unwrap();
/// scharr_spatial_gradient_separable_f32(&src, &mut dx, &mut dy).unwrap();
/// assert_eq!(dx.as_slice()[1], 1.0);
/// assert_eq!(dy.as_slice()[1], 0.0);
/// ```
pub fn scharr_spatial_gradient_separable_f32(
    src: &Image<f32, 1>,
    dx: &mut Image<f32, 1>,
    dy: &mut Image<f32, 1>,
) -> Result<(), ImageError> {
    for d in [&*dx, &*dy] {
        if src.size() != d.size() {
            return Err(ImageError::InvalidImageSize(
                src.cols(),
                src.rows(),
                d.cols(),
                d.rows(),
            ));
        }
    }
    let (w, h) = (src.cols(), src.rows());
    if w == 0 || h == 0 {
        return Ok(());
    }
    let (src, dx, dy) = (src.as_slice(), dx.as_slice_mut(), dy.as_slice_mut());
    // The row buffers are taken here, outside the AVX2 function, so the row
    // loops inline into it without crossing a closure.
    crate::scratch::with_scratch::<f32, { 2 * (ROWS_STACK_WIDTH + 2) }, _>(2 * (w + 2), |buf| {
        let (smooth, diff) = buf.split_at_mut(w + 2);
        #[cfg(target_arch = "x86_64")]
        if crate::simd::cpu_features().has_avx2 {
            // SAFETY: AVX2 is present.
            unsafe { scharr_rows_avx2(src, w, h, dx, dy, smooth, diff) };
            return;
        }
        scharr_rows(src, w, h, dx, dy, smooth, diff);
    });
    Ok(())
}

/// Row widths up to this keep the two row buffers on the stack (16 kB).
const ROWS_STACK_WIDTH: usize = 2048;

/// The portable row loops compiled for AVX2 (no FMA, so the same rounding).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scharr_rows_avx2(
    src: &[f32],
    w: usize,
    h: usize,
    dx: &mut [f32],
    dy: &mut [f32],
    smooth: &mut [f32],
    diff: &mut [f32],
) {
    scharr_rows(src, w, h, dx, dy, smooth, diff);
}

/// `smooth` and `diff` hold the vertical results of one row with one
/// replicated column on each side (`w + 2` each).
#[inline(always)]
fn scharr_rows(
    src: &[f32],
    w: usize,
    h: usize,
    dx: &mut [f32],
    dy: &mut [f32],
    smooth: &mut [f32],
    diff: &mut [f32],
) {
    let rows = dx.chunks_exact_mut(w).zip(dy.chunks_exact_mut(w));
    for (y, (dx_row, dy_row)) in rows.enumerate() {
        let above = &src[y.saturating_sub(1) * w..][..w];
        let center = &src[y * w..][..w];
        let below = &src[(y + 1).min(h - 1) * w..][..w];
        vertical_pass(above, center, below, &mut smooth[1..=w], &mut diff[1..=w]);
        for buf in [&mut *smooth, &mut *diff] {
            buf[0] = buf[1];
            buf[w + 1] = buf[w];
        }
        horizontal_pass(smooth, diff, dx_row, dy_row);
    }
}

#[inline(always)]
fn vertical_pass(
    above: &[f32],
    center: &[f32],
    below: &[f32],
    smooth: &mut [f32],
    diff: &mut [f32],
) {
    let it = smooth
        .iter_mut()
        .zip(diff.iter_mut())
        .zip(above)
        .zip(center)
        .zip(below);
    for ((((s, d), &a), &c), &b) in it {
        *s = (a + b) * SMOOTH_OUTER + c * SMOOTH_CENTER;
        *d = b - a;
    }
}

/// `smooth` and `diff` carry one padding column on each side of the row.
#[inline(always)]
fn horizontal_pass(smooth: &[f32], diff: &[f32], dx: &mut [f32], dy: &mut [f32]) {
    let it = dx
        .iter_mut()
        .zip(dy.iter_mut())
        .zip(smooth.windows(3))
        .zip(diff.windows(3));
    for (((gx, gy), s), d) in it {
        *gx = (s[2] - s[0]) * SCALE;
        *gy = ((d[0] + d[2]) * SMOOTH_OUTER + d[1] * SMOOTH_CENTER) * SCALE;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::scharr_spatial_gradient_float;
    use kornia_image::ImageSize;

    fn lcg_bytes(n: usize, seed: u64) -> Vec<u8> {
        let mut x = seed;
        (0..n)
            .map(|_| {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (x >> 56) as u8
            })
            .collect()
    }

    fn both(src: &Image<f32, 1>) -> Result<[Vec<f32>; 4], ImageError> {
        let mut dx = Image::<f32, 1>::from_size_val(src.size(), 0.0)?;
        let mut dy = Image::<f32, 1>::from_size_val(src.size(), 0.0)?;
        let mut dx3 = Image::<f32, 1>::from_size_val(src.size(), 0.0)?;
        let mut dy3 = Image::<f32, 1>::from_size_val(src.size(), 0.0)?;
        scharr_spatial_gradient_separable_f32(src, &mut dx, &mut dy)?;
        scharr_spatial_gradient_float(src, &mut dx3, &mut dy3)?;
        Ok([dx, dy, dx3, dy3].map(|i| i.as_slice().to_vec()))
    }

    #[test]
    fn separable_equals_3x3_on_integer_pixels() -> Result<(), ImageError> {
        for (i, (w, h)) in [
            (1, 1),
            (2, 1),
            (1, 3),
            (3, 2),
            (17, 9),
            (752, 480),
            (2 * ROWS_STACK_WIDTH + 5, 3),
        ]
        .into_iter()
        .enumerate()
        {
            let data = lcg_bytes(w * h, i as u64)
                .into_iter()
                .map(f32::from)
                .collect();
            let src = Image::<f32, 1>::new(
                ImageSize {
                    width: w,
                    height: h,
                },
                data,
            )?;
            let [dx, dy, dx3, dy3] = both(&src)?;
            assert_eq!(dx, dx3, "dx {w}x{h}");
            assert_eq!(dy, dy3, "dy {w}x{h}");
        }
        Ok(())
    }

    #[test]
    fn separable_close_to_3x3_on_float_pixels() -> Result<(), ImageError> {
        let (w, h) = (41, 23);
        let data = lcg_bytes(w * h, 9)
            .into_iter()
            .map(|v| f32::from(v) / 255.0 + 0.37)
            .collect();
        let src = Image::<f32, 1>::new(
            ImageSize {
                width: w,
                height: h,
            },
            data,
        )?;
        let [dx, dy, dx3, dy3] = both(&src)?;
        for (a, b) in dx.iter().chain(&dy).zip(dx3.iter().chain(&dy3)) {
            assert!((a - b).abs() <= 1e-6, "{a} vs {b}");
        }
        Ok(())
    }

    /// The AVX2 build of the row loops (where present) equals the default one.
    #[test]
    fn isa_paths_agree() -> Result<(), ImageError> {
        let (w, h) = (53, 21);
        let data: Vec<f32> = lcg_bytes(w * h, 5)
            .into_iter()
            .map(|v| f32::from(v) * 0.7 + 0.3)
            .collect();
        let src = Image::<f32, 1>::new(
            ImageSize {
                width: w,
                height: h,
            },
            data.clone(),
        )?;
        let mut dx = Image::<f32, 1>::from_size_val(src.size(), 0.0)?;
        let mut dy = Image::<f32, 1>::from_size_val(src.size(), 0.0)?;
        scharr_spatial_gradient_separable_f32(&src, &mut dx, &mut dy)?;
        let (mut ex, mut ey) = (vec![0.0; w * h], vec![0.0; w * h]);
        let (mut smooth, mut diff) = (vec![0.0; w + 2], vec![0.0; w + 2]);
        scharr_rows(&data, w, h, &mut ex, &mut ey, &mut smooth, &mut diff);
        assert_eq!((dx.as_slice(), dy.as_slice()), (&ex[..], &ey[..]));
        Ok(())
    }

    #[test]
    fn separable_rejects_wrong_size() -> Result<(), ImageError> {
        let src = Image::<f32, 1>::from_size_val(
            ImageSize {
                width: 4,
                height: 4,
            },
            0.0,
        )?;
        let mut dx = Image::<f32, 1>::from_size_val(
            ImageSize {
                width: 4,
                height: 3,
            },
            0.0,
        )?;
        let mut dy = Image::<f32, 1>::from_size_val(src.size(), 0.0)?;
        assert!(scharr_spatial_gradient_separable_f32(&src, &mut dx, &mut dy).is_err());
        Ok(())
    }
}
