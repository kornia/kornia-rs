//! Resource limits applied when decoding untrusted image data.
//!
//! Image headers declare their own dimensions, so a few bytes of input can request
//! gigabytes of output memory ("decompression bomb"). Every decoder in this crate
//! validates the declared dimensions against [`MAX_IMAGE_PIXELS`] before allocating
//! or decoding pixel data, and allocates fallibly wherever this crate owns the output
//! buffer, so that an out-of-memory condition is reported as an error instead of
//! aborting the process.

use crate::error::IoError;
use kornia_image::{Image, ImageSize};

/// Maximum number of pixels (`width * height`) accepted by the decoders.
///
/// Set to 2^30 (e.g. 32768 x 32768), which comfortably covers camera and panorama
/// images while rejecting headers that would request tens of gigabytes.
pub const MAX_IMAGE_PIXELS: usize = 1 << 30;

/// Validates declared image dimensions against [`MAX_IMAGE_PIXELS`].
///
/// # Arguments
///
/// * `width` - The declared image width in pixels.
/// * `height` - The declared image height in pixels.
///
/// # Returns
///
/// `Ok(())` if the image is within the limit.
///
/// # Errors
///
/// Returns [`IoError::ImageTooLarge`] if `width * height` overflows or exceeds
/// [`MAX_IMAGE_PIXELS`].
pub fn check_image_dimensions(width: usize, height: usize) -> Result<(), IoError> {
    match width.checked_mul(height) {
        Some(pixels) if pixels <= MAX_IMAGE_PIXELS => Ok(()),
        _ => Err(IoError::ImageTooLarge {
            width,
            height,
            max_pixels: MAX_IMAGE_PIXELS,
        }),
    }
}

/// Allocates `len` default-initialized elements, reporting allocation failure as an
/// error instead of aborting.
pub(crate) fn try_alloc_zeroed<T: Clone + Default>(len: usize) -> Result<Vec<T>, IoError> {
    let mut buf = Vec::new();
    buf.try_reserve_exact(len)
        .map_err(|_| IoError::AllocationFailed(len.saturating_mul(std::mem::size_of::<T>())))?;
    buf.resize(len, T::default());
    Ok(buf)
}

/// Allocates a zero-initialized output image for a decoder.
///
/// This is the single allocation point of the decoders: it validates the declared
/// dimensions against [`MAX_IMAGE_PIXELS`], computes the element count with checked
/// arithmetic and allocates fallibly.
pub(crate) fn alloc_image<T: Clone + Default, const C: usize>(
    size: ImageSize,
) -> Result<Image<T, C>, IoError> {
    check_image_dimensions(size.width, size.height)?;
    let data = try_alloc_zeroed(size.checked_len(C)?)?;
    Ok(Image::new(size, data)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_oversized_and_overflowing_dimensions() {
        assert!(check_image_dimensions(32768, 32768).is_ok());
        assert!(check_image_dimensions(65535, 65535).is_err());
        assert!(check_image_dimensions(usize::MAX, 2).is_err());
    }

    #[test]
    fn alloc_image_checks_limits() -> Result<(), IoError> {
        let img = alloc_image::<u16, 3>(ImageSize {
            width: 4,
            height: 2,
        })?;
        assert_eq!(img.as_slice(), &[0u16; 24]);
        assert!(matches!(
            alloc_image::<u8, 1>(ImageSize {
                width: 65535,
                height: 65535,
            }),
            Err(IoError::ImageTooLarge { .. })
        ));
        Ok(())
    }
}
