use kornia_image::{Image, ImageError};

/// Compute the mean squared error (MSE) between two images.
///
/// The MSE is defined as:
///
/// $ MSE = \frac{1}{n} \sum_{i=1}^{n} (I_1 - I_2)^2 $
///
/// where `I_1` and `I_2` are the two images and `n` is the number of pixels.
///
/// # Arguments
///
/// * `image1` - The first input image with shape (H, W, C).
/// * `image2` - The second input image with shape (H, W, C).
///
/// # Returns
///
/// The mean squared error between the two images.
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_imgproc::metrics::mse;
///
/// let image1 = Image::<f32, 1>::new(
///    ImageSize {
///      width: 2,
///      height: 3,
///    },
///    vec![0f32, 1f32, 2f32, 3f32, 4f32, 5f32],
/// )
/// .unwrap();
///
/// let image2 = Image::<f32, 1>::new(
///    ImageSize {
///      width: 2,
///      height: 3,
///    },
///    vec![0f32, 1f32, 2f32, 3f32, 4f32, 5f32],
/// )
/// .unwrap();
///
/// let mse = mse(&image1, &image2).unwrap();
/// assert_eq!(mse, 0f32);
/// ```
///
/// # Panics
///
/// Panics if the two images have different shapes.
pub fn mse<const C: usize>(
    image1: &Image<f32, C>,
    image2: &Image<f32, C>,
) -> Result<f32, ImageError> {
    if image1.size() != image2.size() {
        return Err(ImageError::InvalidImageSize(
            image1.rows(),
            image1.cols(),
            image2.rows(),
            image2.cols(),
        ));
    }

    let mse = image1
        .as_slice()
        .iter()
        .zip(image2.as_slice().iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>();

    Ok(mse / (image1.numel() as f32))
}

/// Compute the peak signal-to-noise ratio (PSNR) between two images.
///
/// The PSNR is defined as:
///
/// $ PSNR = 20 \log_{10} \left( \frac{MAX}{\sqrt{MSE}} \right) $
///
/// where `MAX` is the maximum possible pixel value and `MSE` is the mean squared error.
///
/// # Arguments
///
/// * `image1` - The first input image with shape (H, W, C).
/// * `image2` - The second input image with shape (H, W, C).
/// * `max_value` - The maximum possible pixel value.
///
/// # Returns
///
/// The peak signal-to-noise ratio between the two images, in dB, or
/// `f32::INFINITY` when the images are identical (MSE = 0).
///
/// # Errors
///
/// Returns [`ImageError::InvalidImageSize`] if the two images have different
/// sizes.
///
/// # Example
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_imgproc::metrics::psnr;
///
/// let image1 = Image::<f32, 3>::new(
///   ImageSize {
///     width: 1,
///     height: 2,
///   },
///   vec![0.0f32, 0.1, 0.2, 0.3, 0.4, 0.5],
/// )
/// .unwrap();
///
/// let image2 = Image::<f32, 3>::new(
///   ImageSize {
///     width: 1,
///     height: 2,
///   },
///   vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6],
/// )
/// .unwrap();
///
/// let psnr = psnr(&image1, &image2, 1.0).unwrap();
///
/// // Pixel values in [0, 1], every pixel off by 0.1: MSE = 0.01, so
/// // PSNR = 20·log10(1 / 0.1) = 20 dB.
/// assert!((psnr - 20.0).abs() < 1e-4);
/// ```
///
/// # Note
///
/// The PSNR is expressed in decibels (dB).
///
/// The PSNR is used to measure the quality of a reconstructed image. The higher the PSNR, the better the quality of the reconstructed image.
/// The PSNR is widely used in image and video compression.
/// Underneath, the PSNR is based on the mean squared error [mse].
pub fn psnr<const C: usize>(
    image1: &Image<f32, C>,
    image2: &Image<f32, C>,
    max_value: f32,
) -> Result<f32, ImageError> {
    if image1.size() != image2.size() {
        return Err(ImageError::InvalidImageSize(
            image1.height(),
            image1.width(),
            image2.height(),
            image2.width(),
        ));
    }

    let mse = mse(image1, image2)?;

    if mse == 0f32 {
        return Ok(f32::INFINITY);
    }

    Ok(20f32 * (max_value / mse.sqrt()).log10())
}

#[cfg(test)]
mod tests {
    use kornia_image::{Image, ImageError, ImageSize};

    #[test]
    fn test_equal() -> Result<(), ImageError> {
        let image1 = Image::<_, 1>::new(
            ImageSize {
                width: 2,
                height: 3,
            },
            vec![0f32, 1f32, 2f32, 3f32, 4f32, 5f32],
        )?;
        let image2 = Image::<_, 1>::new(
            ImageSize {
                width: 2,
                height: 3,
            },
            vec![0f32, 1f32, 2f32, 3f32, 4f32, 5f32],
        )?;
        let mse = crate::metrics::mse(&image1, &image2)?;
        assert_eq!(mse, 0f32);

        Ok(())
    }

    #[test]
    fn test_not_equal() -> Result<(), ImageError> {
        let image1 = Image::<_, 1>::new(
            ImageSize {
                width: 2,
                height: 2,
            },
            vec![0f32, 1f32, 2f32, 3f32],
        )?;
        let image2 = Image::<_, 1>::new(
            ImageSize {
                width: 2,
                height: 2,
            },
            vec![0f32, 3f32, 2f32, 3f32],
        )?;
        let mse = crate::metrics::mse(&image1, &image2)?;
        assert_eq!(mse, 1.0);

        Ok(())
    }

    #[test]
    fn test_psnr() -> Result<(), ImageError> {
        let image1 = Image::<_, 3>::new(
            ImageSize {
                width: 1,
                height: 2,
            },
            vec![0f32, 1f32, 2f32, 3f32, 4f32, 5f32],
        )?;
        let image2 = Image::<_, 3>::new(
            ImageSize {
                width: 1,
                height: 2,
            },
            vec![1f32, 3f32, 2f32, 4f32, 5f32, 6f32],
        )?;
        // MSE = 4/3, so PSNR = 20·log10(1 / √(4/3)) = -10·log10(4/3).
        let psnr = crate::metrics::psnr(&image1, &image2, 1.0)?;
        assert!((psnr - (-1.249_387_4)).abs() < 1e-5, "psnr = {psnr}");

        Ok(())
    }

    #[test]
    fn test_psnr_known_values() -> Result<(), ImageError> {
        let size = ImageSize {
            width: 2,
            height: 2,
        };
        // Every pixel off by 1 gives MSE = 1, so PSNR = 20·log10(max_value).
        let image1 = Image::<_, 1>::new(size, vec![10f32, 20.0, 30.0, 40.0])?;
        let image2 = Image::<_, 1>::new(size, vec![11f32, 19.0, 31.0, 39.0])?;
        let psnr = crate::metrics::psnr(&image1, &image2, 255.0)?;
        assert!((psnr - 48.130_804).abs() < 1e-4, "psnr = {psnr}");

        // MSE = 1 with max_value = 1 is 0 dB.
        let psnr = crate::metrics::psnr(&image1, &image2, 1.0)?;
        assert!(psnr.abs() < 1e-6, "psnr = {psnr}");

        // A smaller error must give a higher PSNR.
        let image3 = Image::<_, 1>::new(size, vec![10.5f32, 20.0, 30.0, 40.0])?;
        let closer = crate::metrics::psnr(&image1, &image3, 255.0)?;
        assert!(closer > 48.130_804, "psnr = {closer}");

        Ok(())
    }
}
