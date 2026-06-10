//! Turn an interleaved RGB u8 image into the f32 gray buffer XFeat expects:
//! channel-mean gray in `[0, 1]`, bilinearly resampled so both dimensions are
//! multiples of 32 (XFeat's stride alignment).
//!
//! Keypoints come back in this *aligned* resolution's pixel frame, so the whole
//! pipeline (matching, homography, corner projection, drawing) stays in aligned
//! coordinates consistently. The caller scales drawing back to display
//! resolution if needed.

use kornia_xfeat::preproc::{align_to_32, bilinear_resample_gray, rgb_u8_to_gray_f32};

/// An f32 gray image whose dimensions are multiples of 32, ready for XFeat.
pub struct AlignedGray {
    /// Row-major gray pixels in `[0, 1]`, length `height * width`.
    pub data: Vec<f32>,
    /// Aligned height (multiple of 32).
    pub height: usize,
    /// Aligned width (multiple of 32).
    pub width: usize,
}

/// Convert an interleaved RGB u8 buffer (`h * w * 3`) to an [`AlignedGray`].
///
/// Returns `None` if the image is too small to align to a non-zero multiple of
/// 32 in either dimension.
pub fn rgb8_to_aligned_gray(rgb: &[u8], h: usize, w: usize) -> Option<AlignedGray> {
    debug_assert_eq!(rgb.len(), h * w * 3);

    let mut gray = vec![0.0f32; h * w];
    rgb_u8_to_gray_f32(rgb, &mut gray, h, w);

    let (h_out, w_out) = align_to_32(h, w);
    if h_out == 0 || w_out == 0 {
        return None;
    }

    let mut resized = vec![0.0f32; h_out * w_out];
    bilinear_resample_gray(&gray, &mut resized, h, w, h_out, w_out);

    Some(AlignedGray {
        data: resized,
        height: h_out,
        width: w_out,
    })
}
