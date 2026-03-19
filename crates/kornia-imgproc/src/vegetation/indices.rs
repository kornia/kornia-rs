use kornia_image::{allocator::ImageAllocator, Image, ImageError};

/// Computes the Excess Green Index (ExG) for each pixel in an RGB image.
///
/// The ExG index is defined as: `ExG = 2*G - R - B`, clamped to [0.0, 1.0].
/// It is widely used in precision agriculture to isolate vegetation from
/// soil and other background elements.
///
/// # Arguments
///
/// * `src` - Input RGB image with f32 pixel values normalized to [0.0, 1.0].
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
/// use kornia_imgproc::vegetation::excess_green;
/// use kornia_image::allocator::CpuAllocator;
///
/// let src = Image::<f32, 3, CpuAllocator>::new(
///     ImageSize { width: 1, height: 1 },
///     vec![0.2, 0.6, 0.1],
///     CpuAllocator,
/// ).unwrap();
///
/// let mut dst = Image::<f32, 1, CpuAllocator>::new(
///     ImageSize { width: 1, height: 1 },
///     vec![0.0],
///     CpuAllocator,
/// ).unwrap();
///
/// excess_green(&src, &mut dst).unwrap();
/// ```
pub fn excess_green<A: ImageAllocator>(
    src: &Image<f32, 3, A>,
    dst: &mut Image<f32, 1, A>,
) -> Result<(), ImageError> {
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }

    let src_data = src.as_slice();
    let dst_data = dst.as_slice_mut();

    for (i, pixel) in dst_data.iter_mut().enumerate() {
        let r = src_data[i * 3];
        let g = src_data[i * 3 + 1];
        let b = src_data[i * 3 + 2];
        *pixel = (2.0 * g - r - b).clamp(0.0, 1.0);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::{ImageSize, allocator::CpuAllocator};

    #[test]
    fn test_excess_green_basic() {
        // healthy vegetation pixel: high green, low red and blue
        let src = Image::<f32, 3, CpuAllocator>::new(
            ImageSize { width: 1, height: 1 },
            vec![0.2, 0.6, 0.1],
            CpuAllocator,
        ).unwrap();

        let mut dst = Image::<f32, 1, CpuAllocator>::new(
            ImageSize { width: 1, height: 1 },
            vec![0.0],
            CpuAllocator,
        ).unwrap();

        excess_green(&src, &mut dst).unwrap();

        // ExG = 2*0.6 - 0.2 - 0.1 = 0.9
        let result = dst.as_slice()[0];
        assert!((result - 0.9).abs() < 1e-6, "Expected 0.9, got {}", result);
    }

    #[test]
    fn test_excess_green_clamp_min() {
        // non-vegetation pixel: result should be clamped to 0.0
        let src = Image::<f32, 3, CpuAllocator>::new(
            ImageSize { width: 1, height: 1 },
            vec![0.8, 0.1, 0.6],
            CpuAllocator,
        ).unwrap();

        let mut dst = Image::<f32, 1, CpuAllocator>::new(
            ImageSize { width: 1, height: 1 },
            vec![0.0],
            CpuAllocator,
        ).unwrap();

        excess_green(&src, &mut dst).unwrap();

        // ExG = 2*0.1 - 0.8 - 0.6 = -1.2 → clamped to 0.0
        assert_eq!(dst.as_slice()[0], 0.0);
    }

    #[test]
    fn test_excess_green_clamp_max() {
        // pure green pixel: result should be clamped to 1.0
        let src = Image::<f32, 3, CpuAllocator>::new(
            ImageSize { width: 1, height: 1 },
            vec![0.0, 1.0, 0.0],
            CpuAllocator,
        ).unwrap();

        let mut dst = Image::<f32, 1, CpuAllocator>::new(
            ImageSize { width: 1, height: 1 },
            vec![0.0],
            CpuAllocator,
        ).unwrap();

        excess_green(&src, &mut dst).unwrap();

        // ExG = 2*1.0 - 0.0 - 0.0 = 2.0 → clamped to 1.0
        assert_eq!(dst.as_slice()[0], 1.0);
    }

    #[test]
    fn test_excess_green_size_mismatch() {
        // src and dst have different sizes — should return error
        let src = Image::<f32, 3, CpuAllocator>::new(
            ImageSize { width: 2, height: 1 },
            vec![0.2, 0.6, 0.1, 0.2, 0.6, 0.1],
            CpuAllocator,
        ).unwrap();

        let mut dst = Image::<f32, 1, CpuAllocator>::new(
            ImageSize { width: 1, height: 1 },
            vec![0.0],
            CpuAllocator,
        ).unwrap();

        assert!(excess_green(&src, &mut dst).is_err());
    }
}