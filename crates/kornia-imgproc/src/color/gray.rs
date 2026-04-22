use crate::parallel;
use kornia_image::{allocator::ImageAllocator, Image, ImageError};

/// Define the RGB weights for the grayscale conversion.
const RW: f64 = 0.299;
const GW: f64 = 0.587;
const BW: f64 = 0.114;

/// Convert an RGB image to grayscale using the formula:
///
/// Y = 0.299 * R + 0.587 * G + 0.114 * B
///
/// # Arguments
///
/// * `src` - The input RGB image.
/// * `dst` - The output grayscale image.
///
/// Precondition: the input image must have 3 channels.
/// Precondition: the output image must have 1 channel.
/// Precondition: the input and output images must have the same size.
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_image::allocator::CpuAllocator;
/// use kornia_imgproc::color::gray_from_rgb;
///
/// let image = Image::<f32, 3, _>::new(
///     ImageSize {
///         width: 4,
///         height: 5,
///     },
///     vec![0f32; 4 * 5 * 3],
///     CpuAllocator
/// )
/// .unwrap();
///
/// let mut gray = Image::<f32, 1, _>::from_size_val(image.size(), 0.0, CpuAllocator).unwrap();
///
/// gray_from_rgb(&image, &mut gray).unwrap();
/// assert_eq!(gray.num_channels(), 1);
/// assert_eq!(gray.size().width, 4);
/// assert_eq!(gray.size().height, 5);
/// ```
pub fn gray_from_rgb<T, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<T, 3, A1>,
    dst: &mut Image<T, 1, A2>,
) -> Result<(), ImageError>
where
    T: Send + Sync + num_traits::Float,
{
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }

    let rw = T::from(RW).ok_or(ImageError::CastError)?;
    let gw = T::from(GW).ok_or(ImageError::CastError)?;
    let bw = T::from(BW).ok_or(ImageError::CastError)?;

    parallel::par_iter_rows(src, dst, |src_pixel, dst_pixel| {
        let r = src_pixel[0];
        let g = src_pixel[1];
        let b = src_pixel[2];
        dst_pixel[0] = rw * r + gw * g + bw * b;
    });

    Ok(())
}

/// Convert an RGB8 image to grayscale using the formula:
///
/// Y = 77 * R + 150 * G + 29 * B
///
/// # Arguments
///
/// * `src` - The input RGB8 image.
/// * `dst` - The output grayscale image.
///
/// Precondition: the input image must have 3 channels.
/// Precondition: the output image must have 1 channel.
/// Precondition: the input and output images must have the same size.
pub fn gray_from_rgb_u8<A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<u8, 3, A1>,
    dst: &mut Image<u8, 1, A2>,
) -> Result<(), ImageError> {
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }

    let npixels = src.rows() * src.cols();
    rgb_to_gray_u8(src.as_slice(), dst.as_slice_mut(), npixels);

    Ok(())
}

/// RGB u8 to grayscale u8: gray = (77*R + 150*G + 29*B) >> 8.
///
/// Parallelized over row-strips for large images; single-threaded SIMD below
/// the threshold to avoid rayon dispatch overhead.
pub fn rgb_to_gray_u8(src: &[u8], dst: &mut [u8], npixels: usize) {
    // Thread-dispatch only above ~1M px. At 640×480 (307k px) rayon's spawn
    // cost exceeds the ~0.05ms compute budget, but at 1080p (~2M px) the
    // work is big enough that splitting across the two A78AE perf cores
    // recovers ~40% over single-thread NEON.
    const PAR_THRESHOLD: usize = 1024 * 1024;
    if npixels < PAR_THRESHOLD {
        #[cfg(target_arch = "aarch64")]
        rgb_to_gray_u8_neon(src, dst, npixels);
        #[cfg(not(target_arch = "aarch64"))]
        rgb_to_gray_u8_scalar(src, dst, npixels);
        return;
    }

    use rayon::prelude::*;
    // Pick strip size so strips are cache-friendly but large enough to amortize
    // rayon overhead. 32-pixel alignment keeps the SIMD fast-path intact.
    let nthreads = rayon::current_num_threads().max(1);
    let strip = npixels.div_ceil(nthreads).next_multiple_of(32);

    dst.par_chunks_mut(strip)
        .enumerate()
        .for_each(|(i, dchunk)| {
            let start = i * strip;
            let n = dchunk.len();
            let sstart = start * 3;
            let send = sstart + n * 3;
            let schunk = &src[sstart..send];
            #[cfg(target_arch = "aarch64")]
            rgb_to_gray_u8_neon(schunk, dchunk, n);
            #[cfg(not(target_arch = "aarch64"))]
            rgb_to_gray_u8_scalar(schunk, dchunk, n);
        });
}

#[cfg(target_arch = "aarch64")]
fn rgb_to_gray_u8_neon(src: &[u8], dst: &mut [u8], npixels: usize) {
    use std::arch::aarch64::*;
    unsafe {
        let w_r = vdup_n_u8(77);
        let w_g = vdup_n_u8(150);
        let w_b = vdup_n_u8(29);
        let sp = src.as_ptr();
        let dp = dst.as_mut_ptr();

        // 32 pixels per iteration (2x vld3q_u8)
        let bulk32 = npixels & !31;
        let mut i = 0usize;
        while i < bulk32 {
            let rgb0 = vld3q_u8(sp.add(i * 3));
            let rgb1 = vld3q_u8(sp.add((i + 16) * 3));

            let a0_lo = vmlal_u8(
                vmlal_u8(vmull_u8(vget_low_u8(rgb0.0), w_r), vget_low_u8(rgb0.1), w_g),
                vget_low_u8(rgb0.2),
                w_b,
            );
            let a1_lo = vmlal_u8(
                vmlal_u8(vmull_u8(vget_low_u8(rgb1.0), w_r), vget_low_u8(rgb1.1), w_g),
                vget_low_u8(rgb1.2),
                w_b,
            );
            let a0_hi = vmlal_u8(
                vmlal_u8(
                    vmull_u8(vget_high_u8(rgb0.0), w_r),
                    vget_high_u8(rgb0.1),
                    w_g,
                ),
                vget_high_u8(rgb0.2),
                w_b,
            );
            let a1_hi = vmlal_u8(
                vmlal_u8(
                    vmull_u8(vget_high_u8(rgb1.0), w_r),
                    vget_high_u8(rgb1.1),
                    w_g,
                ),
                vget_high_u8(rgb1.2),
                w_b,
            );

            vst1q_u8(
                dp.add(i),
                vcombine_u8(vshrn_n_u16(a0_lo, 8), vshrn_n_u16(a0_hi, 8)),
            );
            vst1q_u8(
                dp.add(i + 16),
                vcombine_u8(vshrn_n_u16(a1_lo, 8), vshrn_n_u16(a1_hi, 8)),
            );
            i += 32;
        }
        // 16-pixel remainder
        if i + 16 <= npixels {
            let rgb = vld3q_u8(sp.add(i * 3));
            let a_lo = vmlal_u8(
                vmlal_u8(vmull_u8(vget_low_u8(rgb.0), w_r), vget_low_u8(rgb.1), w_g),
                vget_low_u8(rgb.2),
                w_b,
            );
            let a_hi = vmlal_u8(
                vmlal_u8(vmull_u8(vget_high_u8(rgb.0), w_r), vget_high_u8(rgb.1), w_g),
                vget_high_u8(rgb.2),
                w_b,
            );
            vst1q_u8(
                dp.add(i),
                vcombine_u8(vshrn_n_u16(a_lo, 8), vshrn_n_u16(a_hi, 8)),
            );
            i += 16;
        }
        // Scalar tail
        while i < npixels {
            let si = i * 3;
            *dp.add(i) = ((77 * *sp.add(si) as u32
                + 150 * *sp.add(si + 1) as u32
                + 29 * *sp.add(si + 2) as u32)
                >> 8) as u8;
            i += 1;
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn rgb_to_gray_u8_scalar(src: &[u8], dst: &mut [u8], npixels: usize) {
    for (i, out) in dst.iter_mut().take(npixels).enumerate() {
        let si = i * 3;
        *out =
            ((77 * src[si] as u32 + 150 * src[si + 1] as u32 + 29 * src[si + 2] as u32) >> 8) as u8;
    }
}

/// Convert a grayscale image to an RGB image by replicating the grayscale value across all three channels.
///
/// # Arguments
///
/// * `src` - The input grayscale image.
/// * `dst` - The output RGB image.
///
/// Precondition: the input image must have 1 channel.
/// Precondition: the output image must have 3 channels.
/// Precondition: the input and output images must have the same size.
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_image::allocator::CpuAllocator;
/// use kornia_imgproc::color::rgb_from_gray;
///
/// let image = Image::<f32, 1, _>::new(
///     ImageSize {
///         width: 4,
///         height: 5,
///     },
///     vec![0f32; 4 * 5 * 1],
///     CpuAllocator
/// )
/// .unwrap();
///
/// let mut rgb = Image::<f32, 3, _>::from_size_val(image.size(), 0.0, CpuAllocator).unwrap();
///
/// rgb_from_gray(&image, &mut rgb).unwrap();
/// ```
pub fn rgb_from_gray<T, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<T, 1, A1>,
    dst: &mut Image<T, 3, A2>,
) -> Result<(), ImageError>
where
    T: Copy + Send + Sync,
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
        dst_pixel[0] = src_pixel[0];
        dst_pixel[1] = src_pixel[0];
        dst_pixel[2] = src_pixel[0];
    });

    Ok(())
}

/// Convert an RGB image to BGR by swapping the red and blue channels.
///
/// # Arguments
///
/// * `src` - The input RGB image.
/// * `dst` - The output BGR image.
///
/// Precondition: the input and output images must have the same size.
pub fn bgr_from_rgb<T, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<T, 3, A1>,
    dst: &mut Image<T, 3, A2>,
) -> Result<(), ImageError>
where
    T: Copy + Send + Sync,
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
        dst_pixel
            .iter_mut()
            .zip(src_pixel.iter().rev())
            .for_each(|(d, s)| {
                *d = *s;
            });
    });

    Ok(())
}

#[cfg(test)]
mod tests {
    use kornia_image::{ops, Image, ImageSize};
    use kornia_io::jpeg::read_image_jpeg_rgb8;
    use kornia_tensor::CpuAllocator;

    #[test]
    fn test_gray_from_rgb() -> Result<(), Box<dyn std::error::Error>> {
        let image = read_image_jpeg_rgb8("../../tests/data/dog.jpeg")?;

        let mut image_norm = Image::from_size_val(image.size(), 0.0, CpuAllocator)?;
        ops::cast_and_scale(&image, &mut image_norm, 1. / 255.0)?;

        let mut gray = Image::<f32, 1, _>::from_size_val(image_norm.size(), 0.0, CpuAllocator)?;
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
            ImageSize {
                width: 2,
                height: 3,
            },
            vec![
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
            ],
            CpuAllocator
        )?;

        let mut gray = Image::<f32, 1, _>::from_size_val(image.size(), 0.0, CpuAllocator)?;

        super::gray_from_rgb(&image, &mut gray)?;

        let expected: Image<f32, 1, _> = Image::new(
            ImageSize {
                width: 2,
                height: 3,
            },
            vec![0.299, 0.587, 0.114, 0.0, 0.0, 0.0],
            CpuAllocator,
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
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            CpuAllocator,
        )?;

        let mut rgb = Image::<f32, 3, _>::from_size_val(image.size(), 0.0, CpuAllocator)?;

        super::rgb_from_gray(&image, &mut rgb)?;

        #[rustfmt::skip]
        let expected: Image<f32, 3, _> = Image::new(
            ImageSize {
                width: 2,
                height: 3,
            },
            vec![
                0.0, 0.0, 0.0,
                1.0, 1.0, 1.0,
                2.0, 2.0, 2.0,
                3.0, 3.0, 3.0,
                4.0, 4.0, 4.0,
                5.0, 5.0, 5.0,
            ],
            CpuAllocator
        )?;

        assert_eq!(rgb.as_slice(), expected.as_slice());

        Ok(())
    }

    #[test]
    fn test_bgr_from_rgb() -> Result<(), Box<dyn std::error::Error>> {
        #[rustfmt::skip]
        let image = Image::new(
            ImageSize {
                width: 1,
                height: 3,
            },
            vec![
                0.0, 1.0, 2.0,
                3.0, 4.0, 5.0,
                6.0, 7.0, 8.0,
            ],
            CpuAllocator
        )?;

        let mut bgr = Image::<f32, 3, _>::from_size_val(image.size(), 0.0, CpuAllocator)?;

        super::bgr_from_rgb(&image, &mut bgr)?;

        #[rustfmt::skip]
        let expected: Image<f32, 3, _> = Image::new(
            ImageSize {
                width: 1,
                height: 3,
            },
            vec![
                2.0, 1.0, 0.0,
                5.0, 4.0, 3.0,
                8.0, 7.0, 6.0,
            ],
            CpuAllocator
        )?;

        assert_eq!(bgr.as_slice(), expected.as_slice());

        Ok(())
    }

    #[test]
    fn test_gray_from_rgb_u8() -> Result<(), Box<dyn std::error::Error>> {
        let image = Image::new(
            ImageSize {
                width: 1,
                height: 2,
            },
            vec![0, 128, 255, 128, 0, 128],
            CpuAllocator,
        )?;

        let mut gray = Image::<u8, 1, _>::from_size_val(image.size(), 0, CpuAllocator)?;

        super::gray_from_rgb_u8(&image, &mut gray)?;

        assert_eq!(gray.as_slice(), &[103, 53]);

        Ok(())
    }
}
