use kornia_image::{Image, ImageError};

/// CUDA kernels for integral image
#[cfg(feature = "cuda")]
pub(crate) mod cuda;
/// CPU kernels for integral image
pub(crate) mod kernels;

/// Computes the integral image (summed area table) for a `f32` image.
///
/// The value at `(x, y)` in the output image is the sum of all pixels above and
/// to the left of `(x, y)`, inclusive.
///
/// # Arguments
///
/// * `src` - Input image.
/// * `dst` - Pre-allocated output image of the same dimensions. Must be `f32`.
///
/// # Returns
///
/// `Ok(())` on success, or a [`ImageError`] if dimensions mismatch.
///
/// # Errors
///
/// Returns [`ImageError::InvalidImageSize`] if `src` and `dst` dimensions differ.
///
/// # Example
///
/// ```rust
/// use kornia_image::{Image, ImageSize};
/// use kornia_imgproc::filter::integral_image_f32;
///
/// let src = Image::<f32, 1>::new(ImageSize { width: 2, height: 2 }, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
/// let mut dst = Image::<f32, 1>::from_size_val(ImageSize { width: 2, height: 2 }, 0.0).unwrap();
/// integral_image_f32(&src, &mut dst).unwrap();
///
/// let out = dst.as_slice();
/// assert_eq!(out[0], 1.0);
/// assert_eq!(out[1], 3.0);
/// assert_eq!(out[2], 4.0);
/// assert_eq!(out[3], 10.0);
/// ```
pub fn integral_image_f32<const C: usize>(
    src: &Image<f32, C>,
    dst: &mut Image<f32, C>,
) -> Result<(), ImageError> {
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }

    crate::try_device!(src, dst, |stream| {
        #[cfg(feature = "cuda")]
        cuda::integral_image_cuda(src, dst, stream)?;
        #[cfg(not(feature = "cuda"))]
        unreachable!();
        Ok(())
    });

    kernels::integral_image_f32_to_f32(src, dst);
    Ok(())
}

/// Computes the integral image (summed area table) for a `u8` image.
///
/// See [`integral_image_f32`] for details.
pub fn integral_image_u8<const C: usize>(
    src: &Image<u8, C>,
    dst: &mut Image<f32, C>,
) -> Result<(), ImageError> {
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }

    crate::try_device!(src, dst, |stream| {
        #[cfg(feature = "cuda")]
        cuda::integral_image_cuda(src, dst, stream)?;
        #[cfg(not(feature = "cuda"))]
        unreachable!();
        Ok(())
    });

    kernels::integral_image_u8_to_f32(src, dst);
    Ok(())
}

#[cfg(test)]
mod tests;
