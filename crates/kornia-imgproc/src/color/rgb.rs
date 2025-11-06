use crate::parallel;
use kornia_image::{allocator::ImageAllocator, Image, ImageError};

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
/// use kornia_image::allocator::CpuAllocator;
/// use kornia_imgproc::color::rgb_from_rgba;
///
/// let src = Image::<u8, 4, CpuAllocator>::new(ImageSize { width: 3, height: 2 }, vec![
///     0, 1, 2, 255, // (0, 0)
///     3, 4, 5, 255, // (0, 1)
///     6, 7, 8, 255, // (0, 2)
///     9, 10, 11, 255, // (1, 0)
///     12, 13, 14, 255, // (1, 1)
///     15, 16, 17, 255, // (1, 2)
/// ], CpuAllocator).unwrap();
///
/// let mut dst = Image::<u8, 3, CpuAllocator>::new(ImageSize { width: 3, height: 2 }, vec![0; 18], CpuAllocator).unwrap();
///
/// rgb_from_rgba(&src, &mut dst, None).unwrap();
/// ```
pub fn rgb_from_rgba<A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<u8, 4, A1>,
    dst: &mut Image<u8, 3, A2>,
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
/// use kornia_image::allocator::CpuAllocator;
/// use kornia_imgproc::color::rgb_from_bgra;
///
/// let src = Image::<u8, 4, CpuAllocator>::new(ImageSize { width: 3, height: 2 }, vec![
///     0, 1, 2, 255, // (0, 0)
///     3, 4, 5, 255, // (0, 1)
///     6, 7, 8, 255, // (0, 2)
///     9, 10, 11, 255, // (1, 0)
///     12, 13, 14, 255, // (1, 1)
///     15, 16, 17, 255, // (1, 2)
/// ], CpuAllocator).unwrap();
///
/// let mut dst = Image::<u8, 3, CpuAllocator>::from_size_val(src.size(), 0, CpuAllocator).unwrap();
///
/// rgb_from_bgra(&src, &mut dst, None).unwrap();
/// ```
pub fn rgb_from_bgra<A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<u8, 4, A1>,
    dst: &mut Image<u8, 3, A2>,
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
    use kornia_image::{allocator::CpuAllocator, ImageSize};

    #[test]
    fn test_rgb_from_rgba() -> Result<(), ImageError> {
        // NOTE: verified with opencv
        let src = Image::<u8, 4, CpuAllocator>::new(
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
            CpuAllocator,
        )?;

        let mut dst = Image::<u8, 3, CpuAllocator>::from_size_val(src.size(), 0, CpuAllocator)?;

        let expected = Image::<u8, 3, CpuAllocator>::new(
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
            CpuAllocator,
        )?;

        rgb_from_rgba(&src, &mut dst, None)?;

        assert_eq!(dst.as_slice(), expected.as_slice());

        Ok(())
    }

    #[test]
    fn test_rgb_from_rgba_with_background() -> Result<(), ImageError> {
        // NOTE: verified with PIL
        let src = Image::<u8, 4, CpuAllocator>::new(
            ImageSize {
                width: 1,
                height: 1,
            },
            vec![255, 0, 0, 128],
            CpuAllocator,
        )?;

        let mut dst = Image::<u8, 3, CpuAllocator>::from_size_val(src.size(), 0, CpuAllocator)?;

        let expected = Image::<u8, 3, CpuAllocator>::new(
            ImageSize {
                width: 1,
                height: 1,
            },
            vec![178, 50, 50],
            CpuAllocator,
        )?;

        rgb_from_rgba(&src, &mut dst, Some([100, 100, 100]))?;

        assert_eq!(dst.as_slice(), expected.as_slice());

        Ok(())
    }

    #[test]
    fn test_rgb_from_bgra_without_background() -> Result<(), ImageError> {
        // NOTE: verified with opencv
        let src = Image::<u8, 4, CpuAllocator>::new(
            ImageSize {
                width: 1,
                height: 1,
            },
            vec![0, 0, 255, 128],
            CpuAllocator,
        )?;

        let mut dst = Image::<u8, 3, CpuAllocator>::from_size_val(src.size(), 0, CpuAllocator)?;

        let expected = Image::<u8, 3, CpuAllocator>::new(
            ImageSize {
                width: 1,
                height: 1,
            },
            vec![255, 0, 0],
            CpuAllocator,
        )?;

        rgb_from_bgra(&src, &mut dst, None)?;

        assert_eq!(dst.as_slice(), expected.as_slice());

        Ok(())
    }

    #[test]
    fn test_rgb_from_bgra_with_background() -> Result<(), ImageError> {
        // NOTE: verified with opencv
        let src = Image::<u8, 4, CpuAllocator>::new(
            ImageSize {
                width: 1,
                height: 1,
            },
            vec![0, 0, 255, 128],
            CpuAllocator,
        )?;

        let mut dst = Image::<u8, 3, CpuAllocator>::from_size_val(src.size(), 0, CpuAllocator)?;

        let expected = Image::<u8, 3, CpuAllocator>::new(
            ImageSize {
                width: 1,
                height: 1,
            },
            vec![178, 50, 50],
            CpuAllocator,
        )?;

        rgb_from_bgra(&src, &mut dst, Some([100, 100, 100]))?;

        assert_eq!(dst.as_slice(), expected.as_slice());

        Ok(())
    }
}
