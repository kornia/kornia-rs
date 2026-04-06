use crate::parallel;
use kornia_image::{allocator::ImageAllocator, Image, ImageError};

/// Computes the Excess Green Index (ExG) for each pixel in an RGB image.
///
/// The ExG index is defined as: `ExG = 2*G - R - B`, clamped to [0.0, 1.0].
/// It is widely used in precision agriculture to isolate vegetation from
/// soil and other background elements.
///
/// # Arguments
///
/// * `src` - Input RGB image with pixel values normalized to [0.0, 1.0].
/// * `dst` - Output single-channel image where each pixel contains the ExG value.
///
/// # Errors
///
/// Returns [`ImageError`] if the source and destination images have different sizes.
///
/// # Example
///
/// ```rust
/// use kornia_image::{Image, ImageSize};
/// use kornia_image::allocator::CpuAllocator;
/// use kornia_imgproc::vegetation::excess_green;
///
/// let src = Image::<f32, 3, _>::new(
///     ImageSize { width: 2, height: 2 },
///     vec![0.2, 0.6, 0.1,  0.5, 0.3, 0.4,
///          0.1, 0.8, 0.2,  0.9, 0.4, 0.3],
///     CpuAllocator,
/// ).unwrap();
///
/// let mut dst = Image::<f32, 1, _>::from_size_val(src.size(), 0.0, CpuAllocator).unwrap();
/// excess_green(&src, &mut dst).unwrap();
/// ```
pub fn excess_green<T, A1: ImageAllocator, A2: ImageAllocator>(
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

    let two = T::from(2.0).ok_or(ImageError::CastError)?;
    let zero = T::zero();
    let one = T::one();

    // parallelize the ExG computation by rows
    parallel::par_iter_rows(src, dst, |src_pixel, dst_pixel| {
        let r = src_pixel[0];
        let g = src_pixel[1];
        let b = src_pixel[2];
        dst_pixel[0] = (two * g - r - b).clamp(zero, one);
    });

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::{ImageSize, allocator::CpuAllocator};

    #[test]
    fn test_excess_green_basic() {
        let src = Image::<f32, 3, CpuAllocator>::new(
            ImageSize { width: 1, height: 1 },
            vec![0.2, 0.6, 0.1],
            CpuAllocator,
        ).unwrap();

        let mut dst = Image::<f32, 1, _>::from_size_val(src.size(), 0.0, CpuAllocator).unwrap();
        excess_green(&src, &mut dst).unwrap();

        // ExG = 2*0.6 - 0.2 - 0.1 = 0.9
        assert!((dst.as_slice()[0] - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_excess_green_clamp_min() {
        let src = Image::<f32, 3, CpuAllocator>::new(
            ImageSize { width: 1, height: 1 },
            vec![0.8, 0.1, 0.6],
            CpuAllocator,
        ).unwrap();

        let mut dst = Image::<f32, 1, _>::from_size_val(src.size(), 0.0, CpuAllocator).unwrap();
        excess_green(&src, &mut dst).unwrap();

        // ExG = 2*0.1 - 0.8 - 0.6 = -1.2 → clamped to 0.0
        assert_eq!(dst.as_slice()[0], 0.0);
    }

    #[test]
    fn test_excess_green_clamp_max() {
        let src = Image::<f32, 3, CpuAllocator>::new(
            ImageSize { width: 1, height: 1 },
            vec![0.0, 1.0, 0.0],
            CpuAllocator,
        ).unwrap();

        let mut dst = Image::<f32, 1, _>::from_size_val(src.size(), 0.0, CpuAllocator).unwrap();
        excess_green(&src, &mut dst).unwrap();

        // ExG = 2*1.0 - 0.0 - 0.0 = 2.0 → clamped to 1.0
        assert_eq!(dst.as_slice()[0], 1.0);
    }

    #[test]
    fn test_excess_green_size_mismatch() {
        let src = Image::<f32, 3, CpuAllocator>::new(
            ImageSize { width: 2, height: 1 },
            vec![0.2, 0.6, 0.1, 0.2, 0.6, 0.1],
            CpuAllocator,
        ).unwrap();

        let mut dst = Image::<f32, 1, _>::from_size_val(
            ImageSize { width: 1, height: 1 }, 0.0, CpuAllocator
        ).unwrap();

        assert!(excess_green(&src, &mut dst).is_err());
    }

    #[test]
    fn test_excess_green_f64() {
        // test that the function works with f64 as well
        let src = Image::<f64, 3, CpuAllocator>::new(
            ImageSize { width: 1, height: 1 },
            vec![0.2_f64, 0.6_f64, 0.1_f64],
            CpuAllocator,
        ).unwrap();

        let mut dst = Image::<f64, 1, _>::from_size_val(src.size(), 0.0, CpuAllocator).unwrap();
        excess_green(&src, &mut dst).unwrap();

        assert!((dst.as_slice()[0] - 0.9).abs() < 1e-10);
    }
}