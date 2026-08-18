use kornia_image::{Image, ImageError};

/// CUDA kernels for laplacian filter
#[cfg(feature = "cuda")]
pub(crate) mod cuda;
/// CPU kernels for laplacian filter
pub(crate) mod kernels;

/// Applies the Laplacian operator to an image.
///
/// The Laplacian is a 2nd-order derivative filter used for edge detection.
/// This function uses a 3x3 kernel: `[0, 1, 0; 1, -4, 1; 0, 1, 0]`.
///
/// # Arguments
///
/// * `src` - Input image.
/// * `dst` - Pre-allocated output image of the same dimensions. Must be `i16`.
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
/// use kornia_imgproc::filter::laplacian_u8;
///
/// let src = Image::<u8, 1>::new(ImageSize { width: 3, height: 3 }, vec![0; 9]).unwrap();
/// let mut dst = Image::<i16, 1>::from_size_val(ImageSize { width: 3, height: 3 }, 0).unwrap();
/// laplacian_u8(&src, &mut dst).unwrap();
/// ```
pub fn laplacian_u8<const C: usize>(
    src: &Image<u8, C>,
    dst: &mut Image<i16, C>,
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
        cuda::laplacian_u8_cuda(src, dst, stream)?;
        #[cfg(not(feature = "cuda"))]
        unreachable!();
        Ok(())
    });

    kernels::laplacian_u8_to_i16(src, dst);
    Ok(())
}

#[cfg(test)]
mod tests;
