use kornia_image::{Image, ImageError};
use crate::filter::separable_filter;
use crate::resize::{resize_native, resize_fast};
use crate::interpolation::InterpolationMode;

fn get_pyramid_gaussian_kernel() -> (Vec<f32>, Vec<f32>) {
    // The 2D kernel is:
    // [
    //   [1.0, 4.0, 6.0, 4.0, 1.0],
    //   [4.0, 16.0, 24.0, 16.0, 4.0],
    //   [6.0, 24.0, 36.0, 24.0, 6.0],
    //   [4.0, 16.0, 24.0, 16.0, 4.0],
    //   [1.0, 4.0, 6.0, 4.0, 1.0],
    // ] / 256.0
    
    let kernel_x = vec![1.0, 4.0, 6.0, 4.0, 1.0].iter().map(|&x| x / 16.0).collect();
    let kernel_y = vec![1.0, 4.0, 6.0, 4.0, 1.0].iter().map(|&x| x / 16.0).collect();
    
    (kernel_x, kernel_y)
}

/// Upsample an image and then blur it using resize_native.
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
/// use kornia_imgproc::pyramid::pyrup;
///
/// let image = Image::<f32, 3>::new(
///     ImageSize {
///         width: 2,
///         height: 2,
///     },
///     vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
/// ).unwrap();
///
/// let mut upsampled = Image::<f32, 3>::from_size_val(
///     ImageSize {
///         width: 4,
///         height: 4,
///     },
///     0.0,
/// ).unwrap();
///
/// pyrup(&image, &mut upsampled).unwrap();
/// ```
pub fn pyrup<const C: usize>(
    src: &Image<f32, C>,
    dst: &mut Image<f32, C>,
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

    let mut upsampled = Image::<f32, C>::from_size_val(dst.size(), 0.0)?;
    
    resize_native(src, &mut upsampled, InterpolationMode::Bilinear)?;
    
    let (kernel_x, kernel_y) = get_pyramid_gaussian_kernel();
    separable_filter(&upsampled, dst, &kernel_x, &kernel_y)?;
    
    Ok(())
}

/// Upsample an image and then blur it using resize_fast.
///
/// This variant uses resize_fast which may provide better performance for small to medium images.
/// For large images (e.g., 1024x896 or larger), the standard pyrup function may be more efficient.
/// Only works with 3-channel u8 images.
///
/// # Arguments
///
/// * `src` - The source image to be upsampled.
/// * `dst` - The destination image to store the result.
///
/// # Returns
///
/// * `Result<(), ImageError>` - Ok if successful, Err otherwise.
pub fn pyrup_fast(
    src: &Image<u8, 3>,
    dst: &mut Image<u8, 3>,
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

    let mut upsampled = Image::<u8, 3>::from_size_val(dst.size(), 0)?;
    
    resize_fast(src, &mut upsampled, InterpolationMode::Bilinear)?;
    
    let mut upsampled_f32 = Image::<f32, 3>::from_size_val(dst.size(), 0.0)?;
    let mut dst_f32 = Image::<f32, 3>::from_size_val(dst.size(), 0.0)?;
    
    for (i, &val) in upsampled.as_slice().iter().enumerate() {
        upsampled_f32.as_slice_mut()[i] = val as f32;
    }
    
    let (kernel_x, kernel_y) = get_pyramid_gaussian_kernel();
    separable_filter(&upsampled_f32, &mut dst_f32, &kernel_x, &kernel_y)?;
    
    for (i, &val) in dst_f32.as_slice().iter().enumerate() {
        dst.as_slice_mut()[i] = val.clamp(0.0, 255.0) as u8;
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::ImageSize;

    #[test]
    fn test_pyrup() -> Result<(), ImageError> {
        let src = Image::<f32, 1>::new(
            ImageSize {
                width: 2,
                height: 2,
            },
            vec![0.0, 1.0, 2.0, 3.0],
        )?;

        let mut dst = Image::<f32, 1>::from_size_val(
            ImageSize {
                width: 4,
                height: 4,
            },
            0.0,
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
    fn test_pyrup_fast() -> Result<(), ImageError> {
        let src = Image::<u8, 3>::new(
            ImageSize {
                width: 2,
                height: 2,
            },
            vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        )?;

        let mut dst = Image::<u8, 3>::from_size_val(
            ImageSize {
                width: 4,
                height: 4,
            },
            0,
        )?;

        pyrup_fast(&src, &mut dst)?;

        assert_eq!(dst.width(), 4);
        assert_eq!(dst.height(), 4);
        
        // Check that the operation completed successfully
        assert!(!dst.as_slice().is_empty());

        Ok(())
    }
} 