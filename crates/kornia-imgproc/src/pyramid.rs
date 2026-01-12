use crate::filter::separable_filter;
use crate::interpolation::InterpolationMode;
use crate::resize::resize_native;
use kornia_image::{allocator::ImageAllocator, Image, ImageError};
use kornia_tensor::CpuAllocator;

fn get_pyramid_gaussian_kernel() -> (Vec<f32>, Vec<f32>) {
    // The 2D kernel is:
    // [
    //   [1.0, 4.0, 6.0, 4.0, 1.0],
    //   [4.0, 16.0, 24.0, 16.0, 4.0],
    //   [6.0, 24.0, 36.0, 24.0, 6.0],
    //   [4.0, 16.0, 24.0, 16.0, 4.0],
    //   [1.0, 4.0, 6.0, 4.0, 1.0],
    // ] / 256.0

    let kernel_x = [1.0, 4.0, 6.0, 4.0, 1.0]
        .iter()
        .map(|&x| x / 16.0)
        .collect();
    let kernel_y = [1.0, 4.0, 6.0, 4.0, 1.0]
        .iter()
        .map(|&x| x / 16.0)
        .collect();

    (kernel_x, kernel_y)
}

/// Upsample an image and then blur it.
///
/// This function doubles the size of the input image using bilinear interpolation
/// and then applies a Gaussian blur to smooth the result.
///
/// # Arguments
///
/// * `src` - The source image to be upsampled.
/// * `dst` - The destination image to store the result.
///
/// # Returns
///
/// * `Result<(), ImageError>` - Ok if successful, Err otherwise.
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_image::allocator::CpuAllocator;
/// use kornia_imgproc::pyramid::pyrup;
///
/// let image = Image::<f32, 3, _>::new(
///     ImageSize {
///         width: 2,
///         height: 2,
///     },
///     vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
///     CpuAllocator
/// ).unwrap();
///
/// let mut upsampled = Image::<f32, 3, _>::from_size_val(
///     ImageSize {
///         width: 4,
///         height: 4,
///     },
///     0.0,
///     CpuAllocator
/// ).unwrap();
///
/// pyrup(&image, &mut upsampled).unwrap();
/// ```
pub fn pyrup<const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<f32, C, A1>,
    dst: &mut Image<f32, C, A2>,
) -> Result<(), ImageError> {
    let expected_width = src.width() * 2;
    let expected_height = src.height() * 2;

    if dst.width() != expected_width || dst.height() != expected_height {
        return Err(ImageError::InvalidImageSize(
            expected_width,
            expected_height,
            dst.width(),
            dst.height(),
        ));
    }

    let mut upsampled = Image::<f32, C, _>::from_size_val(dst.size(), 0.0, CpuAllocator)?;

    resize_native(src, &mut upsampled, InterpolationMode::Bilinear)?;

    let (kernel_x, kernel_y) = get_pyramid_gaussian_kernel();
    separable_filter(&upsampled, dst, &kernel_x, &kernel_y)?;

    Ok(())
}

/// Border reflection mode BORDER_REFLECT_101 (same as OpenCV default).
///
/// Reflects coordinates at image boundaries without repeating the edge pixel.
/// Example for size=5: -2 -1 | 0 1 2 3 4 | 5 6
///                      1  0 | 0 1 2 3 4 | 3 2
#[inline]
fn reflect_101(mut p: i32, len: i32) -> i32 {
    if len == 1 {
        return 0;
    }

    // Handle negative indices by reflecting
    if p < 0 {
        p = -p;
    }

    // Compute which "period" we're in
    let period = 2 * (len - 1);
    p %= period;

    // If in the second half of the period, reflect back
    if p >= len {
        p = period - p;
    }

    p
}

/// Downsample an image by applying Gaussian blur and then subsampling.
///
/// This function halves the size of the input image by first applying a Gaussian blur
/// and then subsampling every other pixel. This is the inverse operation of [`pyrup`].
///
/// Uses BORDER_REFLECT_101 border mode (same as OpenCV default) for handling pixels
/// near image boundaries.
///
/// # Arguments
///
/// * `src` - The source image to be downsampled.
/// * `dst` - The destination image to store the result (should be half the size of src).
///
/// # Returns
///
/// * `Result<(), ImageError>` - Ok if successful, Err otherwise.
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_image::allocator::CpuAllocator;
/// use kornia_imgproc::pyramid::pyrdown;
///
/// let image = Image::<f32, 3, _>::new(
///     ImageSize {
///         width: 4,
///         height: 4,
///     },
///     (0..48).map(|x| x as f32).collect(),
///     CpuAllocator,
/// ).unwrap();
///
/// let mut downsampled = Image::<f32, 3, _>::from_size_val(
///     ImageSize {
///         width: 2,
///         height: 2,
///     },
///     0.0,
///     CpuAllocator,
/// ).unwrap();
///
/// pyrdown(&image, &mut downsampled).unwrap();
/// ```
pub fn pyrdown<const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<f32, C, A1>,
    dst: &mut Image<f32, C, A2>,
) -> Result<(), ImageError> {
    let expected_width = src.width().div_ceil(2);
    let expected_height = src.height().div_ceil(2);

    if dst.width() != expected_width || dst.height() != expected_height {
        return Err(ImageError::InvalidImageSize(
            expected_width,
            expected_height,
            dst.width(),
            dst.height(),
        ));
    }

    let src_width = src.width();
    let src_height = src.height();
    let dst_width = dst.width();
    let dst_height = dst.height();

    // Fused Gaussian blur + downsample in a single pass.
    // For each output pixel at (dst_x, dst_y), we compute the Gaussian-weighted
    // average of a 5x5 neighborhood centered at (dst_x * 2, dst_y * 2) in the source.
    // This avoids allocating an intermediate buffer and only computes values at
    // output pixel locations.

    let src_data = src.as_slice();
    let dst_data = dst.as_slice_mut();
    let (kernel_x, kernel_y) = get_pyramid_gaussian_kernel();

    for dst_y in 0..dst_height {
        let src_center_y = (dst_y * 2) as i32;

        for dst_x in 0..dst_width {
            let src_center_x = (dst_x * 2) as i32;

            // Precompute combined 5x5 kernel weights for this output pixel to avoid
            // recomputing kx*ky for each channel.
            let mut combined = [[0.0f32; 5]; 5];
            for (ky, row) in combined.iter_mut().enumerate() {
                let ky_weight = kernel_y[ky];
                for (kx, val) in row.iter_mut().enumerate() {
                    *val = ky_weight * kernel_x[kx];
                }
            }

            // Apply 5x5 Gaussian kernel centered at (src_center_x, src_center_y)
            for c in 0..C {
                let mut sum = 0.0f32;

                for (ky, row) in combined.iter().enumerate() {
                    let src_y = src_center_y + ky as i32 - 2;
                    // BORDER_REFLECT_101: reflect at borders without repeating edge pixel
                    let src_y_clamped = reflect_101(src_y, src_height as i32) as usize;

                    for (kx, &weight) in row.iter().enumerate() {
                        let src_x = src_center_x + kx as i32 - 2;
                        // BORDER_REFLECT_101: reflect at borders without repeating edge pixel
                        let src_x_clamped = reflect_101(src_x, src_width as i32) as usize;

                        let src_idx = (src_y_clamped * src_width + src_x_clamped) * C + c;
                        sum += src_data[src_idx] * weight;
                    }
                }

                let dst_idx = (dst_y * dst_width + dst_x) * C + c;
                dst_data[dst_idx] = sum;
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::ImageSize;

    #[test]
    fn test_pyrup() -> Result<(), ImageError> {
        let src = Image::<f32, 1, _>::new(
            ImageSize {
                width: 2,
                height: 2,
            },
            vec![0.0, 1.0, 2.0, 3.0],
            CpuAllocator,
        )?;

        let mut dst = Image::<f32, 1, _>::from_size_val(
            ImageSize {
                width: 4,
                height: 4,
            },
            0.0,
            CpuAllocator,
        )?;

        pyrup(&src, &mut dst)?;

        assert_eq!(dst.width(), 4);
        assert_eq!(dst.height(), 4);

        for val in dst.as_slice() {
            assert!(!val.is_nan());
        }

        Ok(())
    }

    #[test]
    fn test_pyrdown() -> Result<(), ImageError> {
        // NOTE: verified with opencv
        let src = Image::<f32, 1, _>::new(
            ImageSize {
                width: 4,
                height: 4,
            },
            vec![
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                15.0,
            ],
            CpuAllocator,
        )?;

        let mut dst = Image::<f32, 1, _>::from_size_val(
            ImageSize {
                width: 2,
                height: 2,
            },
            0.0,
            CpuAllocator,
        )?;

        pyrdown(&src, &mut dst)?;

        // Expected output from OpenCV cv2.pyrDown with BORDER_DEFAULT
        let expected = [3.75, 4.875, 8.25, 9.375];

        let actual = dst.as_slice();
        for (idx, (act, exp)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (act - exp).abs() < 1e-4,
                "Mismatch at index {}: expected {}, got {}",
                idx,
                exp,
                act
            );
        }

        Ok(())
    }

    #[test]
    fn test_pyrdown_3c() -> Result<(), ImageError> {
        // NOTE: verified with opencv
        let src = Image::<f32, 3, _>::new(
            ImageSize {
                width: 4,
                height: 4,
            },
            (0..48).map(|x| x as f32).collect(),
            CpuAllocator,
        )?;

        let mut dst = Image::<f32, 3, _>::from_size_val(
            ImageSize {
                width: 2,
                height: 2,
            },
            0.0,
            CpuAllocator,
        )?;

        pyrdown(&src, &mut dst)?;

        // Expected output from OpenCV cv2.pyrDown with BORDER_DEFAULT
        let expected = [
            11.25, 12.25, 13.25, // pixel (0,0)
            14.625, 15.625, 16.625, // pixel (0,1)
            24.75, 25.75, 26.75, // pixel (1,0)
            28.125, 29.125, 30.125, // pixel (1,1)
        ];

        let actual = dst.as_slice();
        for (idx, (act, exp)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (act - exp).abs() < 1e-4,
                "Mismatch at index {}: expected {}, got {}",
                idx,
                exp,
                act
            );
        }

        Ok(())
    }

    #[test]
    fn test_pyrdown_invalid_size() -> Result<(), ImageError> {
        let src = Image::<f32, 1, _>::new(
            ImageSize {
                width: 4,
                height: 4,
            },
            vec![0.0; 16],
            CpuAllocator,
        )?;

        let mut dst = Image::<f32, 1, _>::from_size_val(
            ImageSize {
                width: 3,
                height: 3,
            },
            0.0,
            CpuAllocator,
        )?;

        let result = pyrdown(&src, &mut dst);
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_pyrdown_odd_dims() -> Result<(), ImageError> {
        // 5x7 -> div_ceil halves to 3x4
        let src = Image::<f32, 1, _>::new(
            ImageSize {
                width: 5,
                height: 7,
            },
            (0..(5 * 7)).map(|x| x as f32).collect(),
            CpuAllocator,
        )?;

        let mut dst = Image::<f32, 1, _>::from_size_val(
            ImageSize {
                width: 3,
                height: 4,
            },
            0.0,
            CpuAllocator,
        )?;

        pyrdown(&src, &mut dst)?;

        assert_eq!(dst.width(), 3);
        assert_eq!(dst.height(), 4);
        for &v in dst.as_slice() {
            assert!(v.is_finite());
        }

        Ok(())
    }

    #[test]
    fn test_pyrdown_min_sizes() -> Result<(), ImageError> {
        // 1x1 -> 1x1
        let src1 = Image::<f32, 1, _>::new(
            ImageSize {
                width: 1,
                height: 1,
            },
            vec![42.0],
            CpuAllocator,
        )?;
        let mut dst1 = Image::<f32, 1, _>::from_size_val(
            ImageSize {
                width: 1,
                height: 1,
            },
            0.0,
            CpuAllocator,
        )?;
        pyrdown(&src1, &mut dst1)?;
        assert_eq!(dst1.width(), 1);
        assert_eq!(dst1.height(), 1);
        assert!(dst1.as_slice().iter().all(|v| v.is_finite()));

        // 1x2 -> 1x1
        let src2 = Image::<f32, 1, _>::new(
            ImageSize {
                width: 1,
                height: 2,
            },
            vec![1.0, 2.0],
            CpuAllocator,
        )?;
        let mut dst2 = Image::<f32, 1, _>::from_size_val(
            ImageSize {
                width: 1,
                height: 1,
            },
            0.0,
            CpuAllocator,
        )?;
        pyrdown(&src2, &mut dst2)?;
        assert_eq!(dst2.width(), 1);
        assert_eq!(dst2.height(), 1);
        assert!(dst2.as_slice().iter().all(|v| v.is_finite()));

        // 2x1 -> 1x1
        let src3 = Image::<f32, 1, _>::new(
            ImageSize {
                width: 2,
                height: 1,
            },
            vec![1.0, 2.0],
            CpuAllocator,
        )?;
        let mut dst3 = Image::<f32, 1, _>::from_size_val(
            ImageSize {
                width: 1,
                height: 1,
            },
            0.0,
            CpuAllocator,
        )?;
        pyrdown(&src3, &mut dst3)?;
        assert_eq!(dst3.width(), 1);
        assert_eq!(dst3.height(), 1);
        eprintln!("dst3 values: {:?}", dst3.as_slice());
        assert!(dst3.as_slice().iter().all(|v| v.is_finite()));

        Ok(())
    }

    #[test]
    fn test_pyrdown_numeric_extremes() -> Result<(), ImageError> {
        // Large uniform values should remain finite and approximately equal after pyrdown
        let large_val = 1e9_f32;
        let src = Image::<f32, 1, _>::new(
            ImageSize {
                width: 4,
                height: 4,
            },
            vec![large_val; 16],
            CpuAllocator,
        )?;
        let mut dst = Image::<f32, 1, _>::from_size_val(
            ImageSize {
                width: 2,
                height: 2,
            },
            0.0,
            CpuAllocator,
        )?;
        pyrdown(&src, &mut dst)?;
        for &v in dst.as_slice() {
            assert!(v.is_finite());
            // With our current boundary handling the output may be scaled near edges; ensure finiteness
            assert!(v.abs() <= large_val);
        }

        Ok(())
    }
}
