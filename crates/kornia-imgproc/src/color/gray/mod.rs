use crate::parallel;
use kornia_image::{allocator::ImageAllocator, Image, ImageError};

mod kernels;
pub use kernels::{rgb_to_gray_f32, rgb_to_gray_u8};

/// BT.601 f64 luma weights, used by the generic `gray_from_rgb`.
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

// ===== RGB8 → Gray8 ================================================================

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

// ===== RGB f32 → Gray f32 ==========================================================

/// Convert an RGB f32 image to grayscale.
///
/// Y = 0.299 * R + 0.587 * G + 0.114 * B
///
/// Dispatches to:
/// - **NEON** (aarch64): `vld3q_f32` structured deinterleave + `vfmaq_f32`, 8 px/iter.
/// - **AVX2+FMA** (x86_64): sequential 256-bit loads with shuffle-deinterleave, 8 px/iter.
/// - **Scalar** fallback on all other targets.
///
/// Large images (> 1 M px) are split across Rayon threads; small images run
/// on a single thread to avoid spawn overhead.
///
/// # Arguments
///
/// * `src` - The input RGB f32 image.
/// * `dst` - The output grayscale f32 image.
///
/// # Errors
///
/// Returns [`ImageError::InvalidImageSize`] if `src` and `dst` sizes differ.
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize, allocator::CpuAllocator};
/// use kornia_imgproc::color::gray_from_rgb_f32;
///
/// let src = Image::<f32, 3, _>::new(
///     ImageSize { width: 2, height: 1 },
///     vec![1.0, 0.0, 0.0,  0.0, 1.0, 0.0],
///     CpuAllocator,
/// ).unwrap();
/// let mut dst = Image::<f32, 1, _>::from_size_val(src.size(), 0.0, CpuAllocator).unwrap();
/// gray_from_rgb_f32(&src, &mut dst).unwrap();
/// ```
pub fn gray_from_rgb_f32<A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<f32, 3, A1>,
    dst: &mut Image<f32, 1, A2>,
) -> Result<(), ImageError> {
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }
    let npix = src.rows() * src.cols();
    rgb_to_gray_f32(src.as_slice(), dst.as_slice_mut(), npix);
    Ok(())
}

// ===== Other conversions ===========================================================

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

    // ----- f32 grayscale tests -----

    #[test]
    fn test_gray_from_rgb_f32_regression() -> Result<(), Box<dyn std::error::Error>> {
        // 6 pixels: pure R, G, B, black, white, and a mixed pixel.
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
            CpuAllocator,
        )?;

        let mut dst = Image::<f32, 1, _>::from_size_val(src.size(), 0.0, CpuAllocator)?;
        super::gray_from_rgb_f32(&src, &mut dst)?;

        let expected = [0.299_f32, 0.587, 0.114, 0.0, 1.0, 0.5];
        for (got, exp) in dst.as_slice().iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-5, "got {got}, expected {exp}");
        }

        Ok(())
    }

    #[test]
    fn test_gray_from_rgb_f32_matches_generic() -> Result<(), Box<dyn std::error::Error>> {
        // Verify the f32-specific path agrees with the generic gray_from_rgb.
        let src = Image::new(
            ImageSize {
                width: 4,
                height: 4,
            },
            (0..48).map(|v| v as f32 / 47.0).collect::<Vec<_>>(),
            CpuAllocator,
        )?;

        let mut dst_f32 = Image::<f32, 1, _>::from_size_val(src.size(), 0.0, CpuAllocator)?;
        let mut dst_generic = Image::<f32, 1, _>::from_size_val(src.size(), 0.0, CpuAllocator)?;

        super::gray_from_rgb_f32(&src, &mut dst_f32)?;
        super::gray_from_rgb(&src, &mut dst_generic)?;

        for (a, b) in dst_f32.as_slice().iter().zip(dst_generic.as_slice().iter()) {
            assert!((a - b).abs() < 1e-5, "f32 path {a} != generic path {b}");
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
            CpuAllocator,
        )?;

        let mut dst_simd = Image::<f32, 1, _>::from_size_val(src.size(), 0.0, CpuAllocator)?;
        let mut dst_generic = Image::<f32, 1, _>::from_size_val(src.size(), 0.0, CpuAllocator)?;

        super::gray_from_rgb_f32(&src, &mut dst_simd)?;
        super::gray_from_rgb(&src, &mut dst_generic)?;

        for (i, (a, b)) in dst_simd
            .as_slice()
            .iter()
            .zip(dst_generic.as_slice().iter())
            .enumerate()
        {
            assert!((a - b).abs() < 1e-5, "pixel {i}: SIMD {a} != generic {b}");
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
            CpuAllocator,
        )?;

        let mut dst_simd = Image::<f32, 1, _>::from_size_val(src.size(), 0.0, CpuAllocator)?;
        let mut dst_generic = Image::<f32, 1, _>::from_size_val(src.size(), 0.0, CpuAllocator)?;

        super::gray_from_rgb_f32(&src, &mut dst_simd)?;
        super::gray_from_rgb(&src, &mut dst_generic)?;

        for (i, (a, b)) in dst_simd
            .as_slice()
            .iter()
            .zip(dst_generic.as_slice().iter())
            .enumerate()
        {
            assert!((a - b).abs() < 1e-5, "pixel {i}: SIMD {a} != generic {b}");
        }

        Ok(())
    }
}
