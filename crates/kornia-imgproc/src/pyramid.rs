use crate::filter::separable_filter;
use kornia_image::{Image, ImageError};

fn get_pyramid_gaussian_kernel() -> (Vec<f32>, Vec<f32>) {
    // The 2D kernel is:
    // [
    //   [1.0, 4.0, 6.0, 4.0, 1.0],
    //   [4.0, 16.0, 24.0, 16.0, 4.0],
    //   [6.0, 24.0, 36.0, 24.0, 6.0],
    //   [4.0, 16.0, 24.0, 16.0, 4.0],
    //   [1.0, 4.0, 6.0, 4.0, 1.0],
    // ] / 256.0

    let kernel_x = vec![1.0, 4.0, 6.0, 4.0, 1.0]
        .iter()
        .map(|&x| x / 16.0)
        .collect();
    let kernel_y = vec![1.0, 4.0, 6.0, 4.0, 1.0]
        .iter()
        .map(|&x| x / 16.0)
        .collect();

    (kernel_x, kernel_y)
}

/// Upsamples an image and then blurs it.
///
/// This function performs the upsampling step of the Gaussian pyramid construction.
/// First, it upsamples the source image by injecting even zero rows and columns,
/// and then convolves the result with the kernel defined multiplied by 4.
///
/// By default, the size of the output image is computed as (src.width*2, src.height*2).
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

    for y in 0..src.height() {
        for x in 0..src.width() {
            for c in 0..C {
                let src_val = *src.get_pixel(x, y, c)?;
                upsampled.set_pixel(x * 2, y * 2, c, src_val)?;
            }
        }
    }

    let (kernel_x, kernel_y) = get_pyramid_gaussian_kernel();
    let kernel_x_scaled: Vec<f32> = kernel_x.iter().map(|&x| x * 4.0).collect();
    let kernel_y_scaled: Vec<f32> = kernel_y.iter().map(|&y| y * 4.0).collect();

    separable_filter(&upsampled, dst, &kernel_x_scaled, &kernel_y_scaled)?;

    Ok(())
}

/// Blur an image and then downsample it.
///
/// This function performs the downsampling step of the Gaussian pyramid construction.
/// It first convolves the source image with a 5x5 Gaussian kernel:
/// [[1, 4, 6, 4, 1],
///  [4, 16, 24, 16, 4],
///  [6, 24, 36, 24, 6],
///  [4, 16, 24, 16, 4],
///  [1, 4, 6, 4, 1]] / 256
///
/// Then, it downsamples the image by rejecting even rows and columns.
///
/// By default, the size of the output image is computed as ((src.width() + 1) / 2, (src.height() + 1) / 2).
///
/// # Arguments
///
/// * `src` - The source image to be downsampled.
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
/// pyrdown(&image, &mut downsampled).unwrap();
/// ```
pub fn pyrdown<const C: usize>(
    src: &Image<f32, C>,
    dst: &mut Image<f32, C>,
) -> Result<(), ImageError> {
    let expected_width = ((src.width() + 1) / 2) as usize;
    let expected_height = ((src.height() + 1) / 2) as usize;

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

    for y in 0..dst.height() {
        for x in 0..dst.width() {
            for c in 0..C {
                let src_x = 2 * x;
                let src_y = 2 * y;

                if src_x < blurred.width() && src_y < blurred.height() {
                    let src_val = *blurred.get_pixel(src_x, src_y, c)?;
                    dst.set_pixel(x, y, c, src_val)?;
                }
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

        // Check that corner pixels are influenced by the original corner values
        let top_left = *dst.get_pixel(0, 0, 0)?;
        let top_right = *dst.get_pixel(3, 0, 0)?;
        let bottom_left = *dst.get_pixel(0, 3, 0)?;
        let bottom_right = *dst.get_pixel(3, 3, 0)?;

        // Check that original source pixels have the strongest influence at their positions
        assert!(top_left > 0.0); // Influenced by src[0,0] which is 0.0
        assert!(top_right > top_left); // Influenced by src[1,0] which is 1.0
        assert!(bottom_left > top_left); // Influenced by src[0,1] which is 2.0
        assert!(bottom_right > top_right); // Influenced by src[1,1] which is 3.0

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

        pyrdown(&src, &mut dst)?;

        assert_eq!(dst.width(), 2);
        assert_eq!(dst.height(), 2);

        for val in dst.as_slice() {
            assert!(!val.is_nan());
        }

        Ok(())
    }
}
