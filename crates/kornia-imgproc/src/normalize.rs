use num_traits::Float;

use kornia_image::{allocator::ImageAllocator, Image, ImageError};

use crate::parallel;

/// Normalize an image using the mean and standard deviation.
///
/// The formula for normalizing an image is:
///
/// (image - mean) / std
///
/// Each channel is normalized independently.
///
/// # Arguments
///
/// * `src` - The input image of shape (height, width, channels).
/// * `dst` - The output image of shape (height, width, channels).
/// * `mean` - The mean value for each channel.
/// * `std` - The standard deviation for each channel.
///
/// # Returns
///
/// The normalized image of shape (height, width, channels).
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_image::allocator::CpuAllocator;
/// use kornia_imgproc::normalize::normalize_mean_std;
///
/// let image_data = vec![0f32, 1.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 0.0, 1.0, 2.0, 3.0];
/// let image = Image::<f32, 3, _>::new(
///   ImageSize {
///     width: 2,
///     height: 2,
///   },
///   image_data,
///   CpuAllocator
/// )
/// .unwrap();
///
/// let mut image_normalized = Image::<f32, 3, _>::from_size_val(image.size(), 0.0, CpuAllocator).unwrap();
///
/// normalize_mean_std(
///     &image,
///     &mut image_normalized,
///     &[0.5, 1.0, 0.5],
///     &[1.0, 1.0, 1.0],
/// )
/// .unwrap();
///
/// assert_eq!(image_normalized.num_channels(), 3);
/// assert_eq!(image_normalized.size().width, 2);
/// assert_eq!(image_normalized.size().height, 2);
/// ```
pub fn normalize_mean_std<T, const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<T, C, A1>,
    dst: &mut Image<T, C, A2>,
    mean: &[T; C],
    std: &[T; C],
) -> Result<(), ImageError>
where
    T: Send + Sync + Float,
{
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }

    parallel::par_iter_rows(src, dst, |src_pixel, dst_pixel| {
        src_pixel
            .iter()
            .zip(dst_pixel.iter_mut())
            .zip(mean.iter())
            .zip(std.iter())
            .for_each(|(((&src_val, dst_val), &mean_val), &std_val)| {
                *dst_val = (src_val - mean_val) / std_val;
            });
    });

    Ok(())
}

/// Find the minimum and maximum values in an image.
///
/// # Arguments
///
/// * `src` - The input image of shape (height, width, channels).
/// * `dst` - The output image of shape (height, width, channels).
///
/// # Returns
///
/// A tuple containing the minimum and maximum values in the image.
///
/// # Errors
///
/// If the image data is not initialized, an error is returned.
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_image::allocator::CpuAllocator;
/// use kornia_imgproc::normalize::find_min_max;
///
/// let image_data = vec![0u8, 1, 0, 1, 2, 3, 0, 1, 0, 1, 2, 3];
/// let image = Image::<u8, 3, _>::new(
///   ImageSize {
///     width: 2,
///     height: 2,
///   },
///   image_data,
///   CpuAllocator
/// )
/// .unwrap();
///
/// let (min, max) = find_min_max(&image).unwrap();
/// assert_eq!(min, 0);
/// assert_eq!(max, 3);
/// ```
pub fn find_min_max<T, const C: usize, A: ImageAllocator>(
    image: &Image<T, C, A>,
) -> Result<(T, T), ImageError>
where
    T: Clone + Copy + PartialOrd,
{
    // get the first element in the image
    let first_element = match image.as_slice().iter().next() {
        Some(x) => x,
        None => return Err(ImageError::ImageDataNotInitialized),
    };

    let mut min = first_element;
    let mut max = first_element;

    for x in image.as_slice().iter() {
        if x < min {
            min = x;
        }
        if x > max {
            max = x;
        }
    }

    Ok((*min, *max))
}

/// Normalize an image using the minimum and maximum values.
///
/// The formula for normalizing an image is:
///
/// (image - min) * (max - min) / (max_val - min_val) + min
///
/// Each channel is normalized independently.
///
/// # Arguments
///
/// * `src` - The input image of shape (height, width, channels).
/// * `dst` - The output image of shape (height, width, channels).
/// * `min` - The minimum value for each channel.
/// * `max` - The maximum value for each channel.
///
/// # Returns
///
/// The normalized image of shape (height, width, C).
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_image::allocator::CpuAllocator;
/// use kornia_imgproc::normalize::normalize_min_max;
///
/// let image_data = vec![0.0f32, 1.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 0.0, 1.0, 2.0, 3.0];
/// let image = Image::<f32, 3, _>::new(
///   ImageSize {
///     width: 2,
///     height: 2,
///   },
///   image_data,
///   CpuAllocator
/// )
/// .unwrap();
///
/// let mut image_normalized = Image::<f32, 3, _>::from_size_val(image.size(), 0.0, CpuAllocator).unwrap();
///
/// normalize_min_max(&image, &mut image_normalized, 0.0, 1.0).unwrap();
///
/// assert_eq!(image_normalized.num_channels(), 3);
/// assert_eq!(image_normalized.size().width, 2);
/// assert_eq!(image_normalized.size().height, 2);
/// ```
pub fn normalize_min_max<T, const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<T, C, A1>,
    dst: &mut Image<T, C, A2>,
    min: T,
    max: T,
) -> Result<(), ImageError>
where
    T: Send + Sync + Float,
{
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }

    let (min_val, max_val) = find_min_max(src)?;

    parallel::par_iter_rows(src, dst, |src_pixel, dst_pixel| {
        src_pixel
            .iter()
            .zip(dst_pixel.iter_mut())
            .for_each(|(&src_val, dst_val)| {
                *dst_val = (src_val - min_val) * (max - min) / (max_val - min_val) + min;
            });
    });

    Ok(())
}

/// Normalize a u8 RGB image directly to f32: `out[i] = src[i] * scale[ch] + offset[ch]`.
///
/// Fuses the u8→f32 cast with per-channel scale+offset in a single pass.
/// Runtime-dispatches to the best available SIMD kernel for the host CPU:
///
/// - aarch64 + NEON: `normalize_rgb_u8_neon` (8 pixels/iter via vld3/vst3)
/// - x86_64 + AVX2+FMA: `normalize_rgb_u8_avx2` (8 pixels/iter via 3×8-lane FMA)
/// - otherwise: portable scalar
///
/// The feature probe is cached process-wide in [`crate::simd::cpu_features`],
/// so per-call dispatch is a single branch on a pre-computed bool.
///
/// # Arguments
///
/// * `src` - Raw RGB u8 pixel data (length = npixels * 3).
/// * `dst` - Output f32 buffer (length = npixels * 3).
/// * `npixels` - Number of pixels (height * width).
/// * `scale` - Per-channel scale factors (typically `1.0 / (std * 255.0)`).
/// * `offset` - Per-channel offsets (typically `-mean / std`).
pub fn normalize_rgb_u8(
    src: &[u8],
    dst: &mut [f32],
    npixels: usize,
    scale: &[f32; 3],
    offset: &[f32; 3],
) {
    #[cfg(target_arch = "aarch64")]
    {
        // NEON is architectural on aarch64 — no runtime check needed.
        // SAFETY: caller guarantees src.len() >= npixels*3, dst.len() >= npixels*3.
        unsafe { normalize_rgb_u8_neon(src, dst, npixels, scale, offset) };
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        let cpu = crate::simd::cpu_features();
        if cpu.has_avx2 && cpu.has_fma {
            // SAFETY: AVX2+FMA feature flags confirmed by runtime probe;
            // length preconditions same as scalar path.
            unsafe { normalize_rgb_u8_avx2(src, dst, npixels, scale, offset) };
            return;
        }
    }

    #[allow(unreachable_code)]
    normalize_rgb_u8_scalar(src, dst, npixels, scale, offset);
}

/// aarch64 NEON implementation: 8 pixels per iteration via `vld3_u8` / `vst3q_f32`.
///
/// # Safety
/// - `src.len() >= npixels * 3` and `dst.len() >= npixels * 3`.
/// - NEON is baseline on aarch64-unknown-linux-gnu (no runtime check needed).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn normalize_rgb_u8_neon(
    src: &[u8],
    dst: &mut [f32],
    npixels: usize,
    scale: &[f32; 3],
    offset: &[f32; 3],
) {
    use std::arch::aarch64::*;
    let bulk = npixels & !7; // 8 pixels per iteration
    let s0 = vdupq_n_f32(scale[0]);
    let s1 = vdupq_n_f32(scale[1]);
    let s2 = vdupq_n_f32(scale[2]);
    let o0 = vdupq_n_f32(offset[0]);
    let o1 = vdupq_n_f32(offset[1]);
    let o2 = vdupq_n_f32(offset[2]);
    let mut i = 0usize;
    while i < bulk {
        // Load 8 RGB pixels (24 bytes), deinterleave into R/G/B vectors
        let rgb = vld3_u8(src.as_ptr().add(i * 3));
        // Widen u8→u16→u32→f32, then fma: val * scale + offset
        let r16 = vmovl_u8(rgb.0);
        let r_lo = vfmaq_f32(o0, vcvtq_f32_u32(vmovl_u16(vget_low_u16(r16))), s0);
        let r_hi = vfmaq_f32(o0, vcvtq_f32_u32(vmovl_u16(vget_high_u16(r16))), s0);
        let g16 = vmovl_u8(rgb.1);
        let g_lo = vfmaq_f32(o1, vcvtq_f32_u32(vmovl_u16(vget_low_u16(g16))), s1);
        let g_hi = vfmaq_f32(o1, vcvtq_f32_u32(vmovl_u16(vget_high_u16(g16))), s1);
        let b16 = vmovl_u8(rgb.2);
        let b_lo = vfmaq_f32(o2, vcvtq_f32_u32(vmovl_u16(vget_low_u16(b16))), s2);
        let b_hi = vfmaq_f32(o2, vcvtq_f32_u32(vmovl_u16(vget_high_u16(b16))), s2);
        // Interleave back to [R,G,B, R,G,B, ...] and store
        vst3q_f32(dst.as_mut_ptr().add(i * 3), float32x4x3_t(r_lo, g_lo, b_lo));
        vst3q_f32(
            dst.as_mut_ptr().add((i + 4) * 3),
            float32x4x3_t(r_hi, g_hi, b_hi),
        );
        i += 8;
    }
    // Scalar tail
    for i in bulk..npixels {
        let base = i * 3;
        *dst.get_unchecked_mut(base) = *src.get_unchecked(base) as f32 * scale[0] + offset[0];
        *dst.get_unchecked_mut(base + 1) =
            *src.get_unchecked(base + 1) as f32 * scale[1] + offset[1];
        *dst.get_unchecked_mut(base + 2) =
            *src.get_unchecked(base + 2) as f32 * scale[2] + offset[2];
    }
}

/// x86_64 AVX2+FMA implementation. Processes 8 pixels (24 f32 outputs) per
/// outer iteration as three 8-lane FMAs.
///
/// # Why not deinterleave like NEON's `vld3`?
///
/// AVX2 has no direct 3-way byte deinterleave. Instead we exploit the fact
/// that RGB bytes cycle with period 3, and AVX2 lanes are 8-wide, so
/// `lcm(8,3) = 24` — every 24 bytes the channel-pattern for an 8-lane FMA
/// repeats. We precompute three scale/offset vectors with the RGB-cycled
/// coefficients once, then the hot loop does a straight u8→u32→f32→FMA
/// chain per 8 bytes. 8 bytes × 3 iterations = 24 bytes = 8 pixels per outer.
///
/// # Safety
/// - `src.len() >= npixels * 3` and `dst.len() >= npixels * 3`.
/// - Caller must ensure AVX2 and FMA are available (check
///   [`crate::simd::cpu_features`] before calling).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn normalize_rgb_u8_avx2(
    src: &[u8],
    dst: &mut [f32],
    npixels: usize,
    scale: &[f32; 3],
    offset: &[f32; 3],
) {
    use std::arch::x86_64::*;

    // Channel-pattern for the three 8-lane FMAs that cover a 24-byte stride:
    //   bytes  0.. 7 -> channels [R G B R G B R G]
    //   bytes  8..15 -> channels [B R G B R G B R]
    //   bytes 16..23 -> channels [G B R G B R G B]
    let s = [
        _mm256_setr_ps(
            scale[0], scale[1], scale[2], scale[0], scale[1], scale[2], scale[0], scale[1],
        ),
        _mm256_setr_ps(
            scale[2], scale[0], scale[1], scale[2], scale[0], scale[1], scale[2], scale[0],
        ),
        _mm256_setr_ps(
            scale[1], scale[2], scale[0], scale[1], scale[2], scale[0], scale[1], scale[2],
        ),
    ];
    let o = [
        _mm256_setr_ps(
            offset[0], offset[1], offset[2], offset[0], offset[1], offset[2], offset[0],
            offset[1],
        ),
        _mm256_setr_ps(
            offset[2], offset[0], offset[1], offset[2], offset[0], offset[1], offset[2],
            offset[0],
        ),
        _mm256_setr_ps(
            offset[1], offset[2], offset[0], offset[1], offset[2], offset[0], offset[1],
            offset[2],
        ),
    ];

    let bulk = npixels & !7; // 8 pixels per outer iter
    let mut i = 0usize;
    while i < bulk {
        let base = i * 3;
        // Three 8-byte loads cover the 24-byte stride for 8 pixels.
        // Each load widens u8→u32 (8 lanes) → f32, then FMA with the
        // channel-rotated scale/offset vector for that 8-byte window.
        let b0 = _mm_loadl_epi64(src.as_ptr().add(base) as *const __m128i);
        let f0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(b0));
        _mm256_storeu_ps(dst.as_mut_ptr().add(base), _mm256_fmadd_ps(f0, s[0], o[0]));

        let b1 = _mm_loadl_epi64(src.as_ptr().add(base + 8) as *const __m128i);
        let f1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(b1));
        _mm256_storeu_ps(
            dst.as_mut_ptr().add(base + 8),
            _mm256_fmadd_ps(f1, s[1], o[1]),
        );

        let b2 = _mm_loadl_epi64(src.as_ptr().add(base + 16) as *const __m128i);
        let f2 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(b2));
        _mm256_storeu_ps(
            dst.as_mut_ptr().add(base + 16),
            _mm256_fmadd_ps(f2, s[2], o[2]),
        );

        i += 8;
    }

    // Scalar tail (1-7 pixels).
    for i in bulk..npixels {
        let base = i * 3;
        *dst.get_unchecked_mut(base) = *src.get_unchecked(base) as f32 * scale[0] + offset[0];
        *dst.get_unchecked_mut(base + 1) =
            *src.get_unchecked(base + 1) as f32 * scale[1] + offset[1];
        *dst.get_unchecked_mut(base + 2) =
            *src.get_unchecked(base + 2) as f32 * scale[2] + offset[2];
    }
}

/// Portable scalar reference — kept available on all targets as the fallback
/// dispatch target and as the numerical baseline for cross-backend tests.
fn normalize_rgb_u8_scalar(
    src: &[u8],
    dst: &mut [f32],
    npixels: usize,
    scale: &[f32; 3],
    offset: &[f32; 3],
) {
    for i in 0..npixels {
        let base = i * 3;
        dst[base] = src[base] as f32 * scale[0] + offset[0];
        dst[base + 1] = src[base + 1] as f32 * scale[1] + offset[1];
        dst[base + 2] = src[base + 2] as f32 * scale[2] + offset[2];
    }
}

#[cfg(test)]
mod tests {
    use kornia_image::{Image, ImageError, ImageSize};
    use kornia_tensor::CpuAllocator;

    #[test]
    fn normalize_mean_std() -> Result<(), ImageError> {
        let image_data = vec![
            0.0f32, 1.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 0.0, 1.0, 2.0, 3.0,
        ];

        let image_expected = [
            -0.5f32, 0.0, -0.5, 0.5, 1.0, 2.5, -0.5, 0.0, -0.5, 0.5, 1.0, 2.5,
        ];

        let image = Image::<f32, 3, _>::new(
            ImageSize {
                width: 2,
                height: 2,
            },
            image_data,
            CpuAllocator,
        )?;

        let mean = [0.5, 1.0, 0.5];
        let std = [1.0, 1.0, 1.0];

        let mut normalized = Image::<f32, 3, _>::from_size_val(image.size(), 0.0, CpuAllocator)?;

        super::normalize_mean_std(&image, &mut normalized, &mean, &std)?;

        assert_eq!(normalized.num_channels(), 3);
        assert_eq!(normalized.size().width, 2);
        assert_eq!(normalized.size().height, 2);

        normalized
            .as_slice()
            .iter()
            .zip(image_expected.iter())
            .for_each(|(a, b)| {
                assert!((a - b).abs() < 1e-6);
            });

        Ok(())
    }

    #[test]
    fn find_min_max() -> Result<(), ImageError> {
        let image_data = vec![0u8, 1, 0, 1, 2, 3, 0, 1, 0, 1, 2, 3];
        let image = Image::<u8, 3, _>::new(
            ImageSize {
                width: 2,
                height: 2,
            },
            image_data,
            CpuAllocator,
        )?;

        let (min, max) = super::find_min_max(&image)?;

        assert_eq!(min, 0);
        assert_eq!(max, 3);

        Ok(())
    }

    #[test]
    fn normalize_min_max() -> Result<(), ImageError> {
        let image_data = vec![
            0.0f32, 1.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 0.0, 1.0, 2.0, 3.0,
        ];

        let image_expected = [
            0.0f32, 0.33333334, 0.0, 0.33333334, 0.6666667, 1.0, 0.0, 0.33333334, 0.0, 0.33333334,
            0.6666667, 1.0,
        ];

        let image = Image::<f32, 3, _>::new(
            ImageSize {
                width: 2,
                height: 2,
            },
            image_data,
            CpuAllocator,
        )?;

        let mut normalized = Image::<f32, 3, _>::from_size_val(image.size(), 0.0, CpuAllocator)?;

        super::normalize_min_max(&image, &mut normalized, 0.0, 1.0)?;

        assert_eq!(normalized.num_channels(), 3);
        assert_eq!(normalized.size().width, 2);
        assert_eq!(normalized.size().height, 2);

        normalized
            .as_slice()
            .iter()
            .zip(image_expected.iter())
            .for_each(|(a, b)| {
                assert!((a - b).abs() < 1e-6);
            });

        Ok(())
    }

    #[test]
    fn normalize_rgb_u8_basic() {
        // 2 pixels: [0,128,255] [100,200,50]
        let src: Vec<u8> = vec![0, 128, 255, 100, 200, 50];
        let mut dst = vec![0.0f32; 6];
        let scale = [1.0 / 255.0; 3];
        let offset = [0.0; 3];

        super::normalize_rgb_u8(&src, &mut dst, 2, &scale, &offset);

        let expected = [
            0.0,
            128.0 / 255.0,
            1.0,
            100.0 / 255.0,
            200.0 / 255.0,
            50.0 / 255.0,
        ];
        for (a, b) in dst.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5, "got {} expected {}", a, b);
        }
    }

    /// On x86_64 with AVX2+FMA, the vectorized kernel must be bit-equivalent
    /// (to FMA rounding) against the scalar reference across a pseudo-random
    /// input. Fixed seed for reproducibility; only runs where the feature
    /// is actually available so CI can flag AVX2-only regressions.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn normalize_rgb_u8_avx2_matches_scalar() {
        if !crate::simd::cpu_features().has_avx2 || !crate::simd::cpu_features().has_fma {
            eprintln!("skipping: host lacks AVX2+FMA");
            return;
        }
        let npix = 1000;
        let mut src = vec![0u8; npix * 3];
        // Deterministic LCG — no external rand dep needed.
        let mut s: u32 = 0xdeadbeef;
        for b in src.iter_mut() {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            *b = (s >> 24) as u8;
        }
        let scale = [1.0 / (0.229 * 255.0), 1.0 / (0.224 * 255.0), 1.0 / (0.225 * 255.0)];
        let offset = [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225];

        let mut dst_avx = vec![0.0f32; npix * 3];
        let mut dst_scalar = vec![0.0f32; npix * 3];
        unsafe {
            super::normalize_rgb_u8_avx2(&src, &mut dst_avx, npix, &scale, &offset);
        }
        super::normalize_rgb_u8_scalar(&src, &mut dst_scalar, npix, &scale, &offset);

        for (i, (a, b)) in dst_avx.iter().zip(dst_scalar.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "mismatch at {}: avx={} scalar={}", i, a, b);
        }
    }

    #[test]
    fn normalize_rgb_u8_with_mean_std() {
        // Verify (x/255 - mean) / std via scale=1/(std*255), offset=-mean/std
        let src: Vec<u8> = vec![255, 0, 128];
        let mean = [0.485, 0.456, 0.406];
        let std_dev = [0.229, 0.224, 0.225];
        let scale = [
            1.0 / (std_dev[0] * 255.0),
            1.0 / (std_dev[1] * 255.0),
            1.0 / (std_dev[2] * 255.0),
        ];
        let offset = [
            -mean[0] / std_dev[0],
            -mean[1] / std_dev[1],
            -mean[2] / std_dev[2],
        ];
        let mut dst = vec![0.0f32; 3];
        super::normalize_rgb_u8(&src, &mut dst, 1, &scale, &offset);

        // Expected: (1.0 - 0.485) / 0.229 = 2.2489
        //           (0.0 - 0.456) / 0.224 = -2.0357
        //           (128/255 - 0.406) / 0.225 = 0.4275
        let expected = [
            (1.0 - mean[0]) / std_dev[0],
            (0.0 - mean[1]) / std_dev[1],
            (128.0 / 255.0 - mean[2]) / std_dev[2],
        ];
        for (a, b) in dst.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-3, "got {} expected {}", a, b);
        }
    }
}
