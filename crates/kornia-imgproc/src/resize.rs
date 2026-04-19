use crate::{
    interpolation::{
        grid::meshgrid_from_fn, interpolate_pixel_fast, validate_interpolation, InterpolationMode,
    },
    parallel,
};
use kornia_image::{allocator::ImageAllocator, Image, ImageError};

/// Resize an image to a new size.
///
/// The function resizes an image to a new size using the specified interpolation mode.
/// It supports any number of channels and data types.
///
/// # Arguments
///
/// * `src` - The input image container.
/// * `dst` - The output image container.
/// * `optional_args` - Optional arguments for the resize operation.
///
/// # Returns
///
/// The resized image with the new size.
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_image::allocator::CpuAllocator;
/// use kornia_imgproc::resize::resize_native;
/// use kornia_imgproc::interpolation::InterpolationMode;
///
/// let image = Image::<_, 3, _>::new(
///     ImageSize {
///         width: 4,
///         height: 5,
///     },
///     vec![0f32; 4 * 5 * 3],
///     CpuAllocator
/// )
/// .unwrap();
///
/// let new_size = ImageSize {
///     width: 2,
///     height: 3,
/// };
///
/// let mut image_resized = Image::<_, 3, _>::from_size_val(new_size, 0.0, CpuAllocator).unwrap();
///
/// resize_native(
///     &image,
///     &mut image_resized,
///     InterpolationMode::Nearest,
/// )
/// .unwrap();
///
/// assert_eq!(image_resized.num_channels(), 3);
/// assert_eq!(image_resized.size().width, 2);
/// assert_eq!(image_resized.size().height, 3);
/// ```
pub fn resize_native<const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<f32, C, A1>,
    dst: &mut Image<f32, C, A2>,
    interpolation: InterpolationMode,
) -> Result<(), ImageError>
where
{
    validate_interpolation(interpolation)?;

    // check if the input and output images have the same size
    // and copy the input image to the output image if they have the same size
    if src.size() == dst.size() {
        dst.as_slice_mut().copy_from_slice(src.as_slice());
        return Ok(());
    }

    // create a grid of x and y coordinates for the output image
    // and interpolate the values from the input image.
    let (dst_rows, dst_cols) = (dst.rows(), dst.cols());
    let step_x = if dst.cols() > 1 {
        (src.cols() - 1) as f32 / (dst.cols() - 1) as f32
    } else {
        0.0
    };
    let step_y = if dst.rows() > 1 {
        (src.rows() - 1) as f32 / (dst.rows() - 1) as f32
    } else {
        0.0
    };
    let (map_x, map_y) = meshgrid_from_fn(dst_cols, dst_rows, |x, y| {
        Ok((x as f32 * step_x, y as f32 * step_y))
    })?;

    // iterate over the output image and interpolate the pixel values
    parallel::par_iter_rows_resample(dst, &map_x, &map_y, |&x, &y, dst_pixel| {
        // interpolate the pixel values for each channel
        for (k, pixel) in dst_pixel.iter_mut().enumerate() {
            *pixel = interpolate_pixel_fast(src, x, y, k, interpolation);
        }
    });

    Ok(())
}

/// Resize an image to a new size using the [fast_image_resize](https://crates.io/crates/fast_image_resize) crate.
///
/// The function resizes an image to a new size using the specified interpolation mode.
/// It supports only 3-channel images and u8 data type.
///
/// # Arguments
///
/// * `image` - The input image container with 3 channels.
/// * `new_size` - The new size of the image.
/// * `interpolation` - The interpolation mode to use.
///
/// # Returns
///
/// The resized image with the new size.
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_image::allocator::CpuAllocator;
/// use kornia_imgproc::resize::resize_fast_rgb;
/// use kornia_imgproc::interpolation::InterpolationMode;
///
/// let image = Image::<_, 3, _>::new(
///     ImageSize {
///         width: 4,
///         height: 5,
///     },
///     vec![0u8; 4 * 5 * 3],
///     CpuAllocator
/// )
/// .unwrap();
///
/// let new_size = ImageSize {
///   width: 2,
///   height: 3,
/// };
///
/// let mut image_resized = Image::<_, 3, _>::from_size_val(new_size, 0, CpuAllocator).unwrap();
///
/// resize_fast_rgb(
///   &image,
///   &mut image_resized,
///   InterpolationMode::Nearest,
/// )
/// .unwrap();
///
/// assert_eq!(image_resized.num_channels(), 3);
/// assert_eq!(image_resized.size().width, 2);
/// assert_eq!(image_resized.size().height, 3);
/// ```
///
/// # Errors
///
/// The function returns an error if the image cannot be resized.
pub fn resize_fast_rgb<A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<u8, 3, A1>,
    dst: &mut Image<u8, 3, A2>,
    interpolation: InterpolationMode,
) -> Result<(), ImageError> {
    resize_fast_u8::<3, _, _>(src, dst, interpolation)
}

/// Generic fast u8 resize for 1/3/4-channel images (PIL-quality antialiased downscale).
pub fn resize_fast_u8<const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<u8, C, A1>,
    dst: &mut Image<u8, C, A2>,
    interpolation: InterpolationMode,
) -> Result<(), ImageError> {
    resize_fast_u8_aa::<C, _, _>(src, dst, interpolation, true)
}

/// Generic fast u8 resize with explicit antialias control.
///
/// `antialias=true` (default via [`resize_fast_u8`]) matches PIL / torchvision
/// `antialias=True` — the cubic/lanczos kernel is widened by the downscale
/// factor to pre-filter aliasing.
///
/// `antialias=false` matches OpenCV `INTER_CUBIC` / `INTER_LANCZOS4` semantics
/// — fixed 4-tap / 8-tap kernel regardless of scale. Much faster at strong
/// downscale but does not anti-alias.
///
/// `Nearest` and `Bilinear` are unaffected by this flag.
pub fn resize_fast_u8_aa<const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<u8, C, A1>,
    dst: &mut Image<u8, C, A2>,
    interpolation: InterpolationMode,
    antialias: bool,
) -> Result<(), ImageError> {
    let (src_w, src_h) = (src.cols(), src.rows());
    let (dst_w, dst_h) = (dst.cols(), dst.rows());

    // 2x exact downscale → box-average (equivalent to bilinear at 2x).
    if matches!(interpolation, InterpolationMode::Bilinear)
        && C == 3
        && src_w == dst_w * 2
        && src_h == dst_h * 2
        && src_w >= 2
        && src_h >= 2
    {
        pyrdown_2x_rgb_u8(src.as_slice(), dst.as_slice_mut(), src_w, src_h);
        return Ok(());
    }

    // Nearest-neighbor fast-path: works for any C, any ratio.
    if matches!(interpolation, InterpolationMode::Nearest) && src_w >= 1 && src_h >= 1 {
        resize_nearest_u8::<C>(
            src.as_slice(),
            src_w,
            src_h,
            dst.as_slice_mut(),
            dst_w,
            dst_h,
        );
        return Ok(());
    }

    // General bilinear path: our own u8 implementation with precomputed
    // x-coefficient tables. Supports C = 1, 3, 4.
    if matches!(interpolation, InterpolationMode::Bilinear)
        && src_w >= 2
        && src_h >= 2
        && (C == 1 || C == 3 || C == 4)
    {
        resize_bilinear_u8_nch::<C>(
            src.as_slice(),
            src_w,
            src_h,
            dst.as_slice_mut(),
            dst_w,
            dst_h,
        );
        return Ok(());
    }

    resize_fast_impl(src, dst, interpolation, antialias)
}

fn resize_nearest_u8<const C: usize>(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    dst: &mut [u8],
    dst_w: usize,
    dst_h: usize,
) {
    use rayon::prelude::*;
    let src_stride = src_w * C;
    let dst_stride = dst_w * C;
    let sx = src_w as f64 / dst_w as f64;
    let sy = src_h as f64 / dst_h as f64;

    let xmap: Vec<usize> = (0..dst_w)
        .map(|x| {
            let v = ((x as f64 + 0.5) * sx).floor() as i64;
            v.clamp(0, src_w as i64 - 1) as usize
        })
        .collect();

    dst.par_chunks_exact_mut(dst_stride)
        .enumerate()
        .for_each(|(y, dst_row)| {
            let yi = (((y as f64 + 0.5) * sy).floor() as i64).clamp(0, src_h as i64 - 1) as usize;
            let src_row = &src[yi * src_stride..(yi + 1) * src_stride];
            for (x, xi) in xmap.iter().enumerate() {
                let so = xi * C;
                let d_o = x * C;
                dst_row[d_o..d_o + C].copy_from_slice(&src_row[so..so + C]);
            }
        });
}

/// Generic N-channel bilinear u8 resize with precomputed x tables.
fn resize_bilinear_u8_nch<const C: usize>(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    dst: &mut [u8],
    dst_w: usize,
    dst_h: usize,
) {
    use rayon::prelude::*;
    const Q: u32 = 14;
    const SCALE: u32 = 1 << Q;
    let scale_x = src_w as f64 / dst_w as f64;
    let scale_y = src_h as f64 / dst_h as f64;

    let mut xofs = Vec::<u32>::with_capacity(dst_w);
    let mut xfx = Vec::<u32>::with_capacity(dst_w);
    let mut xfx1 = Vec::<u32>::with_capacity(dst_w);
    for x in 0..dst_w {
        let sx = (x as f64 + 0.5) * scale_x - 0.5;
        let xi = sx.floor() as i64;
        let f = sx - xi as f64;
        let (xi, f) = if xi < 0 {
            (0i64, 0.0)
        } else if xi >= src_w as i64 - 1 {
            (src_w as i64 - 2, 1.0)
        } else {
            (xi, f)
        };
        let fq = ((f * SCALE as f64).round() as u32).min(SCALE);
        xofs.push(xi as u32);
        xfx.push(fq);
        xfx1.push(SCALE - fq);
    }

    let src_stride = src_w * C;
    let dst_stride = dst_w * C;

    dst.par_chunks_exact_mut(dst_stride)
        .enumerate()
        .for_each(|(y, dst_row)| {
            let sy = (y as f64 + 0.5) * scale_y - 0.5;
            let yi = sy.floor() as i64;
            let f = sy - yi as f64;
            let (yi, f) = if yi < 0 {
                (0i64, 0.0)
            } else if yi >= src_h as i64 - 1 {
                (src_h as i64 - 2, 1.0)
            } else {
                (yi, f)
            };
            let fy = ((f * SCALE as f64).round() as u32).min(SCALE);
            let fy1 = SCALE - fy;

            let row0 = &src[(yi as usize) * src_stride..(yi as usize + 1) * src_stride];
            let row1 = &src[(yi as usize + 1) * src_stride..(yi as usize + 2) * src_stride];

            let round = 1u64 << 27;
            for x in 0..dst_w {
                let xi = xofs[x] as usize;
                let fx = xfx[x] as u64;
                let fx1 = xfx1[x] as u64;
                let off = xi * C;
                for ch in 0..C {
                    let p00 = row0[off + ch] as u64;
                    let p01 = row0[off + C + ch] as u64;
                    let p10 = row1[off + ch] as u64;
                    let p11 = row1[off + C + ch] as u64;
                    let top = p00 * fx1 + p01 * fx;
                    let bot = p10 * fx1 + p11 * fx;
                    let v = ((top * fy1 as u64 + bot * fy as u64 + round) >> 28) as u8;
                    dst_row[x * C + ch] = v;
                }
            }
        });
}

/// 2x box-averaging downsample for RGB u8. Equivalent to bilinear at 2x and
/// ~4x faster on aarch64 thanks to dedicated NEON.
fn pyrdown_2x_rgb_u8(src: &[u8], dst: &mut [u8], src_w: usize, src_h: usize) {
    let _ = src_h;
    let dst_w = src_w / 2;
    let src_stride = src_w * 3;
    let dst_stride = dst_w * 3;

    // Group 8 output rows per rayon task so small strides (e.g. 540p = 2.8 KB/row)
    // amortize spawn overhead. At 1080p this yields ~68 tasks of 22 KB each,
    // well above the ~10 KB threshold where rayon dispatch dominates.
    const ROWS_PER_TASK: usize = 8;
    let chunk_bytes = dst_stride * ROWS_PER_TASK;

    use rayon::prelude::*;
    dst.par_chunks_mut(chunk_bytes)
        .enumerate()
        .for_each(|(ti, dst_chunk)| {
            let y_base = ti * ROWS_PER_TASK;
            let nrows = dst_chunk.len() / dst_stride;
            for dy in 0..nrows {
                let y = y_base + dy;
                let r0 = &src[(2 * y) * src_stride..(2 * y + 1) * src_stride];
                let r1 = &src[(2 * y + 1) * src_stride..(2 * y + 2) * src_stride];
                let dst_row = &mut dst_chunk[dy * dst_stride..(dy + 1) * dst_stride];
                pyrdown_2x_rgb_u8_row(r0, r1, dst_row, dst_w);
            }
        });
}

#[cfg(target_arch = "aarch64")]
fn pyrdown_2x_rgb_u8_row(r0: &[u8], r1: &[u8], dst: &mut [u8], dst_w: usize) {
    use std::arch::aarch64::*;
    unsafe {
        // Process 16 output pixels (= 48 bytes dst, 96 bytes per src row) per iter.
        let bulk = dst_w & !15;
        let mut x = 0usize;
        while x < bulk {
            let s0 = vld3q_u8(r0.as_ptr().add(x * 2 * 3));
            let s1 = vld3q_u8(r0.as_ptr().add(x * 2 * 3 + 48));
            let s2 = vld3q_u8(r1.as_ptr().add(x * 2 * 3));
            let s3 = vld3q_u8(r1.as_ptr().add(x * 2 * 3 + 48));
            // Each of s0.N is 16 u8 (two adjacent src pixels per output).
            // Sum adjacent pairs within row via vpaddlq_u8 → 8 u16 per half.
            let mut out = uint8x16x3_t(vdupq_n_u8(0), vdupq_n_u8(0), vdupq_n_u8(0));
            let mut k = 0;
            while k < 3 {
                let a = match k {
                    0 => s0.0,
                    1 => s0.1,
                    _ => s0.2,
                };
                let b = match k {
                    0 => s1.0,
                    1 => s1.1,
                    _ => s1.2,
                };
                let c = match k {
                    0 => s2.0,
                    1 => s2.1,
                    _ => s2.2,
                };
                let d = match k {
                    0 => s3.0,
                    1 => s3.1,
                    _ => s3.2,
                };
                // horiz pair-add u8 → u16
                let ab_lo = vpaddlq_u8(a); // 8 u16
                let ab_hi = vpaddlq_u8(b);
                let cd_lo = vpaddlq_u8(c);
                let cd_hi = vpaddlq_u8(d);
                // add vertical pair
                let sum_lo = vaddq_u16(ab_lo, cd_lo);
                let sum_hi = vaddq_u16(ab_hi, cd_hi);
                // divide by 4 with rounding, narrow to u8
                let o_lo = vrshrn_n_u16(sum_lo, 2);
                let o_hi = vrshrn_n_u16(sum_hi, 2);
                let o = vcombine_u8(o_lo, o_hi);
                match k {
                    0 => out.0 = o,
                    1 => out.1 = o,
                    _ => out.2 = o,
                }
                k += 1;
            }
            vst3q_u8(dst.as_mut_ptr().add(x * 3), out);
            x += 16;
        }
        // Scalar tail.
        while x < dst_w {
            for ch in 0..3 {
                let sum = r0[(2 * x) * 3 + ch] as u16
                    + r0[(2 * x + 1) * 3 + ch] as u16
                    + r1[(2 * x) * 3 + ch] as u16
                    + r1[(2 * x + 1) * 3 + ch] as u16;
                dst[x * 3 + ch] = ((sum + 2) >> 2) as u8;
            }
            x += 1;
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn pyrdown_2x_rgb_u8_row(r0: &[u8], r1: &[u8], dst: &mut [u8], dst_w: usize) {
    for x in 0..dst_w {
        for ch in 0..3 {
            let sum = r0[(2 * x) * 3 + ch] as u16
                + r0[(2 * x + 1) * 3 + ch] as u16
                + r1[(2 * x) * 3 + ch] as u16
                + r1[(2 * x + 1) * 3 + ch] as u16;
            dst[x * 3 + ch] = ((sum + 2) >> 2) as u8;
        }
    }
}

/// Resize a grayscale (single-channel) image to a new size using the [fast_image_resize](https://crates.io/crates/fast_image_resize) crate.
///
/// The function resizes a grayscale image to a new size using the specified interpolation mode.
/// It supports only 1-channel images and u8 data type.
///
/// # Arguments
///
/// * `src` - The input grayscale image container with 1 channel.
/// * `dst` - The output grayscale image container with 1 channel.
/// * `interpolation` - The interpolation mode to use.
///
/// # Returns
///
/// The resized image with the new size.
///
/// # Errors
///
/// The function returns an error if the image cannot be resized.
pub fn resize_fast_mono<A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<u8, 1, A1>,
    dst: &mut Image<u8, 1, A2>,
    interpolation: InterpolationMode,
) -> Result<(), ImageError> {
    resize_fast_u8::<1, _, _>(src, dst, interpolation)
}

/// Grayscale resize with explicit antialias control. See [`resize_fast_u8_aa`].
pub fn resize_fast_mono_aa<A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<u8, 1, A1>,
    dst: &mut Image<u8, 1, A2>,
    interpolation: InterpolationMode,
    antialias: bool,
) -> Result<(), ImageError> {
    resize_fast_u8_aa::<1, _, _>(src, dst, interpolation, antialias)
}

/// RGB resize with explicit antialias control. See [`resize_fast_u8_aa`].
pub fn resize_fast_rgb_aa<A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<u8, 3, A1>,
    dst: &mut Image<u8, 3, A2>,
    interpolation: InterpolationMode,
    antialias: bool,
) -> Result<(), ImageError> {
    resize_fast_u8_aa::<3, _, _>(src, dst, interpolation, antialias)
}

fn resize_fast_impl<const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<u8, C, A1>,
    dst: &mut Image<u8, C, A2>,
    interpolation: InterpolationMode,
    antialias: bool,
) -> Result<(), ImageError> {
    if !(C == 1 || C == 3 || C == 4) {
        return Err(ImageError::UnsupportedChannelCount(C));
    }
    let (src_w, src_h) = (src.cols(), src.rows());
    let (dst_w, dst_h) = (dst.cols(), dst.rows());
    match interpolation {
        InterpolationMode::Nearest => {
            resize_nearest_u8::<C>(
                src.as_slice(),
                src_w,
                src_h,
                dst.as_slice_mut(),
                dst_w,
                dst_h,
            );
        }
        InterpolationMode::Bilinear => {
            resize_bilinear_u8_nch::<C>(
                src.as_slice(),
                src_w,
                src_h,
                dst.as_slice_mut(),
                dst_w,
                dst_h,
            );
        }
        InterpolationMode::Bicubic => {
            resize_separable_u8::<C>(
                src.as_slice(),
                src_w,
                src_h,
                dst.as_slice_mut(),
                dst_w,
                dst_h,
                FilterKind::Cubic,
                antialias,
            );
        }
        InterpolationMode::Lanczos => {
            resize_separable_u8::<C>(
                src.as_slice(),
                src_w,
                src_h,
                dst.as_slice_mut(),
                dst_w,
                dst_h,
                FilterKind::Lanczos3,
                antialias,
            );
        }
    }
    Ok(())
}

#[derive(Copy, Clone)]
enum FilterKind {
    Cubic,
    Lanczos3,
}

impl FilterKind {
    fn support(self) -> f64 {
        match self {
            FilterKind::Cubic => 2.0,
            FilterKind::Lanczos3 => 3.0,
        }
    }
    fn weight(self, x: f64) -> f64 {
        let ax = x.abs();
        match self {
            FilterKind::Cubic => {
                let a = -0.5;
                if ax < 1.0 {
                    (a + 2.0) * ax * ax * ax - (a + 3.0) * ax * ax + 1.0
                } else if ax < 2.0 {
                    a * ax * ax * ax - 5.0 * a * ax * ax + 8.0 * a * ax - 4.0 * a
                } else {
                    0.0
                }
            }
            FilterKind::Lanczos3 => {
                if ax < 1e-12 {
                    1.0
                } else if ax < 3.0 {
                    let px = std::f64::consts::PI * x;
                    let s = px.sin();
                    let s3 = (px / 3.0).sin();
                    3.0 * s * s3 / (px * px)
                } else {
                    0.0
                }
            }
        }
    }
}

/// Precompute per-output-pixel (offset, weights, ksize).
///
/// When `antialias` is true (PIL / torchvision `antialias=True` semantics),
/// the filter is widened by the downscale factor so the kernel pre-filters
/// aliasing. This matches PIL's output within Q14 rounding noise but grows
/// `ksize` linearly with scale — e.g. bicubic 1080→224 uses ksize≈20.
///
/// When `antialias` is false (OpenCV INTER_CUBIC / INTER_LANCZOS4 semantics),
/// `ksize` is fixed at twice the filter support (4 for bicubic, 6 for lanczos)
/// regardless of scale. Much faster on strong downscale but aliases.
#[allow(clippy::needless_range_loop)]
fn precompute_contribs(
    src_size: usize,
    dst_size: usize,
    filt: FilterKind,
    antialias: bool,
) -> (Vec<i32>, Vec<i32>, usize) {
    const Q: i32 = 14;
    const SCALE: i32 = 1 << Q;
    let scale = src_size as f64 / dst_size as f64;
    let filt_scale = if antialias { scale.max(1.0) } else { 1.0 };
    let support = filt.support() * filt_scale;
    let ksize = ((support.ceil() as usize) * 2).max(2);

    let mut offsets = vec![0i32; dst_size];
    let mut weights = vec![0i32; dst_size * ksize];

    for i in 0..dst_size {
        let center = (i as f64 + 0.5) * scale - 0.5;
        let left = (center - support).ceil() as i64;
        offsets[i] = left as i32;

        let inv_filt_scale = 1.0 / filt_scale;
        let mut raw = vec![0f64; ksize];
        let mut sum = 0f64;
        for k in 0..ksize {
            let x = (left + k as i64) as f64 - center;
            let w = filt.weight(x * inv_filt_scale) * inv_filt_scale;
            raw[k] = w;
            sum += w;
        }
        let mut qw = vec![0i32; ksize];
        let mut qsum = 0i32;
        let norm = if sum.abs() > 1e-12 {
            SCALE as f64 / sum
        } else {
            0.0
        };
        for k in 0..ksize {
            let v = (raw[k] * norm).round() as i32;
            qw[k] = v;
            qsum += v;
        }
        if qsum != SCALE {
            let mut max_k = 0usize;
            let mut max_abs = 0i32;
            for k in 0..ksize {
                if qw[k].abs() > max_abs {
                    max_abs = qw[k].abs();
                    max_k = k;
                }
            }
            qw[max_k] += SCALE - qsum;
        }
        weights[i * ksize..(i + 1) * ksize].copy_from_slice(&qw);
    }
    (offsets, weights, ksize)
}

/// Compact per-tap LUT: xsrc as u16, weight as i16. At kx=30 and dst_w=224
/// the combined table fits L1 (27 KB) where u32/i32 would spill (54 KB).
#[allow(clippy::needless_range_loop)]
fn build_xsrc_lut(xofs: &[i32], dst_w: usize, kx: usize, src_w: usize) -> Vec<u16> {
    debug_assert!(src_w <= u16::MAX as usize);
    let mut xsrc = Vec::<u16>::with_capacity(dst_w * kx);
    for x in 0..dst_w {
        let x0 = xofs[x];
        for t in 0..kx {
            let sx = (x0 + t as i32).clamp(0, src_w as i32 - 1) as u16;
            xsrc.push(sx);
        }
    }
    xsrc
}

fn pack_xw_i16(xw: &[i32]) -> Vec<i16> {
    xw.iter().map(|&w| w as i16).collect()
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn load_px(p: *const u8) -> std::arch::aarch64::int16x4_t {
    use std::arch::aarch64::*;
    vreinterpret_s16_u16(vget_low_u16(vmovl_u8(vld1_u8(p))))
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn load_px_edge(p: *const u8) -> std::arch::aarch64::int16x4_t {
    use std::arch::aarch64::*;
    let t: [i16; 4] = [*p as i16, *p.add(1) as i16, *p.add(2) as i16, 0];
    vld1_s16(t.as_ptr())
}

/// Horizontal pass over 4 source rows simultaneously (C=3 only).
/// Shares one `xsrc`/`xw` load across all 4 rows so the coefficient LUT is
/// fetched once per tap rather than four times. 4 independent accumulators
/// also hide the ~4-cycle `vmlal_n_s16` latency on A-class cores.
///
/// Output storage uses `vst1_s16` (8-byte vector store) for interior pixels
/// and scalar lane extraction for the final pixel. `vset_lane_s16::<3>(0, .)`
/// zeroes the 4th lane before storing so the 2-byte overflow into the next
/// pixel's slot is a benign zero that gets overwritten by its R on the next
/// iteration. This replaces 12 SMOV+STRH pairs per iter with 4 vector stores
/// + 4 lane-zeros — ~10 cycles faster per dst pixel on A78.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn horizontal_row_c3_x4_neon(
    src_rows: [&[u8]; 4],
    outs: [&mut [i16]; 4],
    dst_w: usize,
    kx: usize,
    xsrc: &[u16],
    xw: &[i16],
    last_sx_safe: usize,
    round1: i32,
) {
    use std::arch::aarch64::*;
    let round_v = vdupq_n_s32(round1);
    let [s0p, s1p, s2p, s3p] = [
        src_rows[0].as_ptr(),
        src_rows[1].as_ptr(),
        src_rows[2].as_ptr(),
        src_rows[3].as_ptr(),
    ];
    let [mut o0, mut o1, mut o2, mut o3] = {
        let [a, b, c, d] = outs;
        [
            a.as_mut_ptr(),
            b.as_mut_ptr(),
            c.as_mut_ptr(),
            d.as_mut_ptr(),
        ]
    };
    let last_x = dst_w.saturating_sub(1);
    for x in 0..dst_w {
        let ibase = x * kx;
        let mut a0 = round_v;
        let mut a1 = round_v;
        let mut a2 = round_v;
        let mut a3 = round_v;
        for t in 0..kx {
            let sx = *xsrc.get_unchecked(ibase + t) as usize;
            let w = *xw.get_unchecked(ibase + t);
            let safe = sx < last_sx_safe;
            let off = sx * 3;
            let p0 = if safe {
                load_px(s0p.add(off))
            } else {
                load_px_edge(s0p.add(off))
            };
            let p1 = if safe {
                load_px(s1p.add(off))
            } else {
                load_px_edge(s1p.add(off))
            };
            let p2 = if safe {
                load_px(s2p.add(off))
            } else {
                load_px_edge(s2p.add(off))
            };
            let p3 = if safe {
                load_px(s3p.add(off))
            } else {
                load_px_edge(s3p.add(off))
            };
            a0 = vmlal_n_s16(a0, p0, w);
            a1 = vmlal_n_s16(a1, p1, w);
            a2 = vmlal_n_s16(a2, p2, w);
            a3 = vmlal_n_s16(a3, p3, w);
        }
        let s0 = vqmovn_s32(vshrq_n_s32::<14>(a0));
        let s1 = vqmovn_s32(vshrq_n_s32::<14>(a1));
        let s2 = vqmovn_s32(vshrq_n_s32::<14>(a2));
        let s3 = vqmovn_s32(vshrq_n_s32::<14>(a3));
        if x < last_x {
            // Interior: store 8 bytes (4 × i16). Lane 3's garbage lands in
            // the next pixel's R slot and is overwritten by its lane 0 on
            // the next iteration, so it's benign. Only the last pixel needs
            // scalar stores to avoid overflow past row end.
            vst1_s16(o0, s0);
            vst1_s16(o1, s1);
            vst1_s16(o2, s2);
            vst1_s16(o3, s3);
        } else {
            // Last pixel: scalar stores (no room for overflow).
            *o0 = vget_lane_s16::<0>(s0);
            *o0.add(1) = vget_lane_s16::<1>(s0);
            *o0.add(2) = vget_lane_s16::<2>(s0);
            *o1 = vget_lane_s16::<0>(s1);
            *o1.add(1) = vget_lane_s16::<1>(s1);
            *o1.add(2) = vget_lane_s16::<2>(s1);
            *o2 = vget_lane_s16::<0>(s2);
            *o2.add(1) = vget_lane_s16::<1>(s2);
            *o2.add(2) = vget_lane_s16::<2>(s2);
            *o3 = vget_lane_s16::<0>(s3);
            *o3.add(1) = vget_lane_s16::<1>(s3);
            *o3.add(2) = vget_lane_s16::<2>(s3);
        }
        o0 = o0.add(3);
        o1 = o1.add(3);
        o2 = o2.add(3);
        o3 = o3.add(3);
    }
}

/// Run the horizontal pass over a contiguous run of source rows, writing
/// i16 intermediate rows into `out`. Uses the 4-row NEON kernel for C=3 to
/// amortize coefficient loads across four src rows.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn horizontal_batch<const C: usize>(
    src: &[u8],
    src_stride: usize,
    out: &mut [i16],
    hbuf_row_len: usize,
    sy_start: usize,
    dst_w: usize,
    kx: usize,
    xsrc: &[u16],
    xw: &[i16],
    last_sx_safe: usize,
    round1: i32,
) {
    let n_rows = out.len() / hbuf_row_len;
    let mut i = 0usize;
    #[cfg(target_arch = "aarch64")]
    {
        if C == 3 {
            while i + 4 <= n_rows {
                let sy0 = sy_start + i;
                let s0 = &src[sy0 * src_stride..(sy0 + 1) * src_stride];
                let s1 = &src[(sy0 + 1) * src_stride..(sy0 + 2) * src_stride];
                let s2 = &src[(sy0 + 2) * src_stride..(sy0 + 3) * src_stride];
                let s3 = &src[(sy0 + 3) * src_stride..(sy0 + 4) * src_stride];
                let (out_head, out_rest) = out[i * hbuf_row_len..].split_at_mut(hbuf_row_len);
                let (out_1, out_rest) = out_rest.split_at_mut(hbuf_row_len);
                let (out_2, out_rest) = out_rest.split_at_mut(hbuf_row_len);
                let (out_3, _) = out_rest.split_at_mut(hbuf_row_len);
                unsafe {
                    horizontal_row_c3_x4_neon(
                        [s0, s1, s2, s3],
                        [out_head, out_1, out_2, out_3],
                        dst_w,
                        kx,
                        xsrc,
                        xw,
                        last_sx_safe,
                        round1,
                    );
                }
                i += 4;
            }
        }
    }
    while i < n_rows {
        let sy = sy_start + i;
        let src_row = &src[sy * src_stride..(sy + 1) * src_stride];
        let out_row = &mut out[i * hbuf_row_len..(i + 1) * hbuf_row_len];
        horizontal_row::<C>(src_row, out_row, dst_w, kx, xsrc, xw, last_sx_safe, round1);
        i += 1;
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn horizontal_row<const C: usize>(
    src_row: &[u8],
    out: &mut [i16],
    dst_w: usize,
    kx: usize,
    xsrc: &[u16],
    xw: &[i16],
    last_sx_safe: usize,
    round1: i32,
) {
    const Q: i32 = 14;
    #[cfg(target_arch = "aarch64")]
    {
        if C == 3 {
            unsafe {
                use std::arch::aarch64::*;
                let round_v = vdupq_n_s32(round1);
                // 4-wide unroll: four independent accumulators hide the ~4-cycle
                // vmlal latency on A-class cores, nearly quadrupling MAC throughput
                // when kx is large (lanczos, bicubic at strong downscales).
                let mut x = 0;
                while x + 4 <= dst_w {
                    let b0 = x * kx;
                    let b1 = (x + 1) * kx;
                    let b2 = (x + 2) * kx;
                    let b3 = (x + 3) * kx;
                    let mut a0 = round_v;
                    let mut a1 = round_v;
                    let mut a2 = round_v;
                    let mut a3 = round_v;
                    for t in 0..kx {
                        let sx0 = *xsrc.get_unchecked(b0 + t) as usize;
                        let sx1 = *xsrc.get_unchecked(b1 + t) as usize;
                        let sx2 = *xsrc.get_unchecked(b2 + t) as usize;
                        let sx3 = *xsrc.get_unchecked(b3 + t) as usize;
                        let w0 = *xw.get_unchecked(b0 + t);
                        let w1 = *xw.get_unchecked(b1 + t);
                        let w2 = *xw.get_unchecked(b2 + t);
                        let w3 = *xw.get_unchecked(b3 + t);
                        let base = src_row.as_ptr();
                        let p0 = if sx0 < last_sx_safe {
                            load_px(base.add(sx0 * 3))
                        } else {
                            load_px_edge(base.add(sx0 * 3))
                        };
                        let p1 = if sx1 < last_sx_safe {
                            load_px(base.add(sx1 * 3))
                        } else {
                            load_px_edge(base.add(sx1 * 3))
                        };
                        let p2 = if sx2 < last_sx_safe {
                            load_px(base.add(sx2 * 3))
                        } else {
                            load_px_edge(base.add(sx2 * 3))
                        };
                        let p3 = if sx3 < last_sx_safe {
                            load_px(base.add(sx3 * 3))
                        } else {
                            load_px_edge(base.add(sx3 * 3))
                        };
                        a0 = vmlal_n_s16(a0, p0, w0);
                        a1 = vmlal_n_s16(a1, p1, w1);
                        a2 = vmlal_n_s16(a2, p2, w2);
                        a3 = vmlal_n_s16(a3, p3, w3);
                    }
                    let s0 = vqmovn_s32(vshrq_n_s32::<14>(a0));
                    let s1 = vqmovn_s32(vshrq_n_s32::<14>(a1));
                    let s2 = vqmovn_s32(vshrq_n_s32::<14>(a2));
                    let s3 = vqmovn_s32(vshrq_n_s32::<14>(a3));
                    let o = out.as_mut_ptr().add(x * 3);
                    *o = vget_lane_s16::<0>(s0);
                    *o.add(1) = vget_lane_s16::<1>(s0);
                    *o.add(2) = vget_lane_s16::<2>(s0);
                    *o.add(3) = vget_lane_s16::<0>(s1);
                    *o.add(4) = vget_lane_s16::<1>(s1);
                    *o.add(5) = vget_lane_s16::<2>(s1);
                    *o.add(6) = vget_lane_s16::<0>(s2);
                    *o.add(7) = vget_lane_s16::<1>(s2);
                    *o.add(8) = vget_lane_s16::<2>(s2);
                    *o.add(9) = vget_lane_s16::<0>(s3);
                    *o.add(10) = vget_lane_s16::<1>(s3);
                    *o.add(11) = vget_lane_s16::<2>(s3);
                    x += 4;
                }
                while x < dst_w {
                    let ibase = x * kx;
                    let mut acc = round_v;
                    for t in 0..kx {
                        let sx = *xsrc.get_unchecked(ibase + t) as usize;
                        let w = *xw.get_unchecked(ibase + t);
                        let p = src_row.as_ptr().add(sx * 3);
                        let px = if sx < last_sx_safe {
                            load_px(p)
                        } else {
                            load_px_edge(p)
                        };
                        acc = vmlal_n_s16(acc, px, w);
                    }
                    let sat = vqmovn_s32(vshrq_n_s32::<14>(acc));
                    let o = out.as_mut_ptr().add(x * 3);
                    *o = vget_lane_s16::<0>(sat);
                    *o.add(1) = vget_lane_s16::<1>(sat);
                    *o.add(2) = vget_lane_s16::<2>(sat);
                    x += 1;
                }
            }
            return;
        }
    }
    for x in 0..dst_w {
        let ibase = x * kx;
        for ch in 0..C {
            let mut acc: i32 = 0;
            for t in 0..kx {
                let sx = xsrc[ibase + t] as usize;
                acc += src_row[sx * C + ch] as i32 * xw[ibase + t] as i32;
            }
            let v = (acc + round1) >> Q;
            out[x * C + ch] = v.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn resize_global_two_pass<const C: usize>(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    dst: &mut [u8],
    dst_w: usize,
    dst_h: usize,
    xsrc: &[u16],
    xw: &[i16],
    kx: usize,
    yofs: &[i32],
    yw: &[i32],
    ky: usize,
    last_sx_safe: usize,
    round1: i32,
    round2: i32,
) {
    use rayon::prelude::*;
    let _ = dst_h;
    let src_stride = src_w * C;
    let dst_stride = dst_w * C;
    let hbuf_row_len = dst_w * C;
    let mut hbuf = vec![0i16; src_h * hbuf_row_len];
    const H_BATCH: usize = 16;
    hbuf.par_chunks_mut(hbuf_row_len * H_BATCH)
        .enumerate()
        .for_each(|(bi, batch_out)| {
            let sy0 = bi * H_BATCH;
            horizontal_batch::<C>(
                src,
                src_stride,
                batch_out,
                hbuf_row_len,
                sy0,
                dst_w,
                kx,
                xsrc,
                xw,
                last_sx_safe,
                round1,
            );
        });

    const V_BATCH: usize = 8;
    dst.par_chunks_mut(dst_stride * V_BATCH)
        .enumerate()
        .for_each(|(bi, batch_dst)| {
            let yo0 = bi * V_BATCH;
            let batch_rows = batch_dst.len() / dst_stride;
            let n = dst_w * C;
            let mut rows: Vec<&[i16]> = Vec::with_capacity(ky);
            let mut w: Vec<i16> = Vec::with_capacity(ky);
            for r in 0..batch_rows {
                let yo = yo0 + r;
                let y0 = yofs[yo];
                rows.clear();
                w.clear();
                for k in 0..ky {
                    let sy = (y0 + k as i32).clamp(0, src_h as i32 - 1) as usize;
                    rows.push(&hbuf[sy * hbuf_row_len..(sy + 1) * hbuf_row_len]);
                    w.push(yw[yo * ky + k] as i16);
                }
                let dst_row = &mut batch_dst[r * dst_stride..(r + 1) * dst_stride];
                vertical_single_row(&rows, &w, dst_row, n, round2);
            }
        });
}

#[allow(clippy::too_many_arguments)]
fn resize_separable_u8<const C: usize>(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    dst: &mut [u8],
    dst_w: usize,
    dst_h: usize,
    filt: FilterKind,
    antialias: bool,
) {
    use rayon::prelude::*;
    const Q: i32 = 14;

    let (xofs, xw, kx) = precompute_contribs(src_w, dst_w, filt, antialias);
    let (yofs, yw, ky) = precompute_contribs(src_h, dst_h, filt, antialias);
    let src_stride = src_w * C;
    let dst_stride = dst_w * C;

    let xsrc = build_xsrc_lut(&xofs, dst_w, kx, src_w);
    // Pack weights i32→i16 once. Q14 coeffs have ≤ 16384 peak magnitude so they
    // fit i16 with headroom; halves LUT bandwidth and lets the horizontal inner
    // loop issue `vmlal_n_s16` directly without per-iter `as i16` casts.
    let xw = pack_xw_i16(&xw);
    let last_sx_safe = src_w.saturating_sub(1);

    let hbuf_row_len = dst_w * C;
    let round1: i32 = 1 << (Q - 1);
    let round2: i32 = 1 << (Q - 1);
    let nthreads = rayon::current_num_threads().max(1);
    let per_row = hbuf_row_len * 2;

    // Per-thread hbuf slice (global plan) = (src_h / nthreads) * per_row.
    // If that fits private L2 (~256KB), the global two-pass plan wins because
    // strip fusion adds (dst_h/strip_h * ky) rows of overlap overhead.
    //
    // If the per-thread slice overflows L2, switch to strip-fused execution
    // with tile sizes chosen so the local hbuf stays in L2.
    let per_thread_global = src_h.div_ceil(nthreads) * per_row;
    let l2_target = 192 * 1024;
    let use_strips = per_thread_global > l2_target;

    if !use_strips {
        // Global two-pass: cheap for small dst_w where per-thread hbuf fits L2.
        resize_global_two_pass::<C>(
            src,
            src_w,
            src_h,
            dst,
            dst_w,
            dst_h,
            &xsrc,
            &xw,
            kx,
            &yofs,
            &yw,
            ky,
            last_sx_safe,
            round1,
            round2,
        );
        return;
    }

    // Strip-fused H→V: bound strip_h so (strip_h*scale_y + ky) * per_row ≤ L2.
    let scale_y_q8 = ((src_h as u64) << 8) / (dst_h.max(1) as u64);
    let band_cap = (l2_target / per_row.max(1)).saturating_sub(ky);
    let strip_h_mem = ((band_cap as u64) << 8) / scale_y_q8.max(1);
    let strip_h_par = dst_h.div_ceil(nthreads);
    let strip_h = (strip_h_mem as usize)
        .min(strip_h_par.max(8))
        .min(dst_h)
        .max(1);

    let mut strip_slices: Vec<&mut [u8]> = dst.chunks_mut(strip_h * dst_stride).collect();
    let strips: Vec<(usize, usize)> = (0..dst_h)
        .step_by(strip_h)
        .map(|s| (s, (s + strip_h).min(dst_h)))
        .collect();

    strip_slices.par_iter_mut().zip(strips.par_iter()).for_each(
        |(strip_dst, &(yo_start, yo_end))| {
            let n = dst_w * C;

            // Source-row span needed for this strip: [sy_min, sy_max).
            let mut sy_min = i32::MAX;
            let mut sy_max = i32::MIN;
            for &y0 in &yofs[yo_start..yo_end] {
                sy_min = sy_min.min(y0);
                sy_max = sy_max.max(y0 + ky as i32);
            }
            // Clamp to image bounds for allocation; border replication is
            // handled inside the per-row clamp below.
            let sy_span_start = sy_min.max(0) as usize;
            let sy_span_end = sy_max.min(src_h as i32) as usize;
            let band_rows = sy_span_end.saturating_sub(sy_span_start);
            if band_rows == 0 {
                return;
            }

            // Local hbuf: exactly the rows this strip consumes. Unlike the
            // global version, this stays in L1/L2 across H and V.
            let mut temp: Vec<i16> = Vec::with_capacity(band_rows * hbuf_row_len);
            // SAFETY: horizontal pass writes every element before any read.
            #[allow(clippy::uninit_vec)]
            unsafe {
                temp.set_len(band_rows * hbuf_row_len)
            };

            // Horizontal pass over the needed src rows only. Processes rows
            // in groups of 4 to share coefficient (xsrc/xw) loads and fill
            // the MAC pipelines.
            horizontal_batch::<C>(
                src,
                src_stride,
                &mut temp,
                hbuf_row_len,
                sy_span_start,
                dst_w,
                kx,
                &xsrc,
                &xw,
                last_sx_safe,
                round1,
            );

            // Vertical pass: consume local hbuf (L1/L2-hot) straight to output.
            let mut rows: Vec<&[i16]> = Vec::with_capacity(ky);
            let mut w: Vec<i16> = Vec::with_capacity(ky);
            for (oi, yo) in (yo_start..yo_end).enumerate() {
                let y0 = yofs[yo];
                rows.clear();
                w.clear();
                for k in 0..ky {
                    // Map global src index to band-local. Rows outside the
                    // band (edge replication) clamp to the nearest kept row.
                    let sy_global = (y0 + k as i32).clamp(0, src_h as i32 - 1) as usize;
                    let sy_local = sy_global.min(sy_span_end - 1).saturating_sub(sy_span_start);
                    rows.push(&temp[sy_local * hbuf_row_len..(sy_local + 1) * hbuf_row_len]);
                    w.push(yw[yo * ky + k] as i16);
                }
                let dst_row = &mut strip_dst[oi * dst_stride..(oi + 1) * dst_stride];
                vertical_single_row(&rows, &w, dst_row, n, round2);
            }
        },
    );
}

/// Vertical pass for one destination row.
///
/// Uses 4 parallel accumulator chains per 8-lane block (8 chains total across
/// lo/hi halves). The chain length of `vmlal_s16` is 4c latency on A78; with
/// a single chain per half, `ky` taps serialize into `ky * 4c` — for ky=6
/// lanczos that's 24c per block, limiting V-pass throughput. Splitting taps
/// across 4 rolling chains keeps all 4 vmlal issue slots busy per cycle, so
/// the critical path drops to roughly `ky/4 * 4c + reduce` ≈ 10-15c per block.
#[inline(always)]
fn vertical_single_row(rows: &[&[i16]], w: &[i16], dst_row: &mut [u8], n: usize, round2: i32) {
    let ky = rows.len();
    let mut i = 0usize;
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let round_v = vdupq_n_s32(round2);
        let zero = vdupq_n_s32(0);
        while i + 8 <= n {
            // Four rolling accumulator chains per half-lane, so 8 total.
            // Seeding acc0 with round means we skip adding round at the end.
            let mut a0l = round_v;
            let mut a1l = zero;
            let mut a2l = zero;
            let mut a3l = zero;
            let mut a0h = round_v;
            let mut a1h = zero;
            let mut a2h = zero;
            let mut a3h = zero;

            // Main loop: consume 4 taps at a time, one per chain.
            let mut k = 0;
            while k + 4 <= ky {
                let v0 = vld1q_s16(rows.get_unchecked(k).as_ptr().add(i));
                let v1 = vld1q_s16(rows.get_unchecked(k + 1).as_ptr().add(i));
                let v2 = vld1q_s16(rows.get_unchecked(k + 2).as_ptr().add(i));
                let v3 = vld1q_s16(rows.get_unchecked(k + 3).as_ptr().add(i));
                let w0 = vdup_n_s16(*w.get_unchecked(k));
                let w1 = vdup_n_s16(*w.get_unchecked(k + 1));
                let w2 = vdup_n_s16(*w.get_unchecked(k + 2));
                let w3 = vdup_n_s16(*w.get_unchecked(k + 3));
                a0l = vmlal_s16(a0l, vget_low_s16(v0), w0);
                a1l = vmlal_s16(a1l, vget_low_s16(v1), w1);
                a2l = vmlal_s16(a2l, vget_low_s16(v2), w2);
                a3l = vmlal_s16(a3l, vget_low_s16(v3), w3);
                a0h = vmlal_s16(a0h, vget_high_s16(v0), w0);
                a1h = vmlal_s16(a1h, vget_high_s16(v1), w1);
                a2h = vmlal_s16(a2h, vget_high_s16(v2), w2);
                a3h = vmlal_s16(a3h, vget_high_s16(v3), w3);
                k += 4;
            }
            // Tail 0..3: round-robin into the same chains.
            while k < ky {
                let v = vld1q_s16(rows.get_unchecked(k).as_ptr().add(i));
                let wk = vdup_n_s16(*w.get_unchecked(k));
                let vl = vget_low_s16(v);
                let vh = vget_high_s16(v);
                match k & 3 {
                    0 => {
                        a0l = vmlal_s16(a0l, vl, wk);
                        a0h = vmlal_s16(a0h, vh, wk);
                    }
                    1 => {
                        a1l = vmlal_s16(a1l, vl, wk);
                        a1h = vmlal_s16(a1h, vh, wk);
                    }
                    _ => {
                        a2l = vmlal_s16(a2l, vl, wk);
                        a2h = vmlal_s16(a2h, vh, wk);
                    }
                }
                k += 1;
            }

            // Tree-reduce 4 chains → 1 per half, then narrow, saturate, pack.
            let acc_lo = vaddq_s32(vaddq_s32(a0l, a1l), vaddq_s32(a2l, a3l));
            let acc_hi = vaddq_s32(vaddq_s32(a0h, a1h), vaddq_s32(a2h, a3h));
            let packed = vcombine_s16(
                vqmovn_s32(vshrq_n_s32::<14>(acc_lo)),
                vqmovn_s32(vshrq_n_s32::<14>(acc_hi)),
            );
            vst1_u8(dst_row.as_mut_ptr().add(i), vqmovun_s16(packed));
            i += 8;
        }
    }
    while i < n {
        let mut acc: i32 = 0;
        for k in 0..ky {
            acc += rows[k][i] as i32 * w[k] as i32;
        }
        dst_row[i] = (((acc + round2) >> 14).clamp(0, 255)) as u8;
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use kornia_image::{Image, ImageError, ImageSize};
    use kornia_tensor::{CpuAllocator, TensorError};

    #[test]
    fn resize_smoke_ch3() -> Result<(), ImageError> {
        let image = Image::<_, 3, _>::new(
            ImageSize {
                width: 3,
                height: 4,
            },
            (0..3 * 4 * 3).map(|x| x as f32).collect::<Vec<f32>>(),
            CpuAllocator,
        )?;

        let new_size = ImageSize {
            width: 2,
            height: 3,
        };

        let mut image_resized = Image::<_, 3, _>::from_size_val(new_size, 0.0, CpuAllocator)?;

        super::resize_native(
            &image,
            &mut image_resized,
            super::InterpolationMode::Bilinear,
        )?;

        assert_eq!(image_resized.num_channels(), 3);
        assert_eq!(image_resized.size().width, 2);
        assert_eq!(image_resized.size().height, 3);

        assert_eq!(
            image_resized.as_slice(),
            [
                0.0, 1.0, 2.0, 6.0, 7.0, 8.0, 13.5, 14.5, 15.5, 19.5, 20.5, 21.5, 27.0, 28.0, 29.0,
                33.0, 34.0, 35.0
            ]
        );

        Ok(())
    }

    #[test]
    fn resize_smoke_ch1() -> Result<(), ImageError> {
        use kornia_image::{Image, ImageSize};
        let image = Image::<_, 1, _>::new(
            ImageSize {
                width: 2,
                height: 3,
            },
            vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0],
            CpuAllocator,
        )?;

        let new_size = ImageSize {
            width: 2,
            height: 3,
        };

        let mut image_resized = Image::<_, 1, _>::from_size_val(new_size, 0.0f32, CpuAllocator)?;

        super::resize_native(
            &image,
            &mut image_resized,
            super::InterpolationMode::Nearest,
        )?;

        assert_eq!(image_resized.num_channels(), 1);
        assert_eq!(image_resized.size().width, 2);
        assert_eq!(image_resized.size().height, 3);

        assert_eq!(image_resized.as_slice(), image_resized.as_slice());

        Ok(())
    }

    #[test]
    fn meshgrid() -> Result<(), TensorError> {
        let (map_x, map_y) =
            crate::interpolation::grid::meshgrid_from_fn(2, 3, |x, y| Ok((x as f32, y as f32)))?;

        assert_eq!(map_x.shape, [3, 2]);
        assert_eq!(map_y.shape, [3, 2]);
        assert_eq!(map_x.get([0, 0]), Some(&0.0));
        assert_eq!(map_x.get([0, 1]), Some(&1.0));
        assert_eq!(map_y.get([0, 0]), Some(&0.0));
        assert_eq!(map_y.get([2, 0]), Some(&2.0));

        Ok(())
    }

    #[test]
    fn resize_native_unsupported_interpolation() -> Result<(), ImageError> {
        let image = Image::<_, 1, _>::new(
            ImageSize {
                width: 2,
                height: 2,
            },
            vec![0.0f32; 4],
            CpuAllocator,
        )?;

        let mut dst = Image::<_, 1, _>::from_size_val(
            ImageSize {
                width: 1,
                height: 1,
            },
            0.0f32,
            CpuAllocator,
        )?;

        let err = super::resize_native(&image, &mut dst, super::InterpolationMode::Bicubic);
        assert!(err.is_err());

        let err = super::resize_native(&image, &mut dst, super::InterpolationMode::Lanczos);
        assert!(err.is_err());
        Ok(())
    }

    #[test]
    fn resize_fast() -> Result<(), ImageError> {
        use kornia_image::{Image, ImageSize};
        let image = Image::<_, 3, _>::new(
            ImageSize {
                width: 4,
                height: 5,
            },
            vec![0u8; 4 * 5 * 3],
            CpuAllocator,
        )?;

        let new_size = ImageSize {
            width: 2,
            height: 3,
        };

        let mut image_resized = Image::<_, 3, _>::from_size_val(new_size, 0, CpuAllocator)?;

        super::resize_fast_rgb(
            &image,
            &mut image_resized,
            super::InterpolationMode::Nearest,
        )?;

        assert_eq!(image_resized.num_channels(), 3);
        assert_eq!(image_resized.size().width, 2);
        assert_eq!(image_resized.size().height, 3);
        Ok(())
    }
}
