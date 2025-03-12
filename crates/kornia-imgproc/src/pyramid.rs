use kornia_image::{Image, ImageError};
use crate::filter::separable_filter;
use crate::resize::resize_native;
use crate::interpolation::InterpolationMode;

/// Returns a pre-computed Gaussian kernel for pyramid operations.
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

/// Blur an image and then downsample it.
///
/// This function applies a Gaussian blur to the input image and then
/// downsamples it by a factor of 2 (or a custom factor) using bilinear interpolation.
///
/// # Arguments
///
/// * `src` - The source image to be downsampled.
/// * `dst` - The destination image to store the result.
/// * `factor` - The downsampling factor (default: 2.0).
///
/// # Returns
///
/// * `Result<(), ImageError>` - Ok if successful, Err otherwise.
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_imgproc::pyramid::pyrdown;
///
/// let image = Image::<f32, 3>::new(
///     ImageSize {
///         width: 4,
///         height: 4,
///     },
///     vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
///          16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0,
///          32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0],
/// ).unwrap();
///
/// let mut downsampled = Image::<f32, 3>::from_size_val(
///     ImageSize {
///         width: 2,
///         height: 2,
///     },
///     0.0,
/// ).unwrap();
///
/// pyrdown(&image, &mut downsampled, 2.0).unwrap();
/// ```
pub fn pyrdown<const C: usize>(
    src: &Image<f32, C>,
    dst: &mut Image<f32, C>,
    factor: f32,
) -> Result<(), ImageError> {
    // Check that the destination image is the correct size based on the factor
    let expected_width = (src.width() as f32 / factor) as usize;
    let expected_height = (src.height() as f32 / factor) as usize;
    
    if dst.width() != expected_width || dst.height() != expected_height {
        return Err(ImageError::InvalidImageSize(
            expected_width,
            expected_height,
            dst.width(),
            dst.height(),
        ));
    }

    let mut blurred = Image::<f32, C>::from_size_val(src.size(), 0.0)?;
    
    let (kernel_x, kernel_y) = get_pyramid_gaussian_kernel();
    separable_filter(src, &mut blurred, &kernel_x, &kernel_y)?;
    
    resize_native(&blurred, dst, InterpolationMode::Bilinear)?;
    
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
    fn test_pyrdown() -> Result<(), ImageError> {
        let src = Image::<f32, 1>::new(
            ImageSize {
                width: 4,
                height: 4,
            },
            (0..16).map(|x| x as f32).collect(),
        )?;

        let mut dst = Image::<f32, 1>::from_size_val(
            ImageSize {
                width: 2,
                height: 2,
            },
            0.0,
        )?;

        pyrdown(&src, &mut dst, 2.0)?;

        assert_eq!(dst.width(), 2);
        assert_eq!(dst.height(), 2);
        
        for val in dst.as_slice() {
            assert!(!val.is_nan());
        }

        Ok(())
    }
} 