use kornia_image::{allocator::ImageAllocator, Image, ImageError};
use rayon::prelude::*;

/// The mode to convert YUV to RGB.
///
/// These modes correspond to ITU-R Broadcasting Television standards that define
/// the coefficients and ranges for YUV color space conversion:
///
/// ## Official ITU-R Documentation:
/// - **BT.601**: [ITU-R BT.601-7](https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.601-7-201103-I!!PDF-E.pdf) - SDTV standard
/// - **BT.709**: [ITU-R BT.709-6](https://www.itu.int/rec/R-REC-BT.709-6-201506-I/en) - HDTV standard
/// - **BT.2020**: [ITU-R BT.2020-2](https://www.itu.int/rec/R-REC-BT.2020-2-201510-I/en) - Ultra HD standard
///
/// ## Additional Resources:
/// - [ITU-R BT.2407-0](https://www.itu.int/dms_pub/itu-r/opb/rep/R-REP-BT.2407-2017-PDF-E.pdf) - Color gamut conversion guide
/// - [Ultra HD Forum Guidelines](https://ultrahdforum.org/wp-content/uploads/UHD-Guidelines-V2.5-Fall2021.pdf) - Industry best practices
pub enum YuvToRgbMode {
    /// BT.601 full range (0-255 for Y, U, V).
    /// Used for SDTV, older cameras, and JPEG images.
    Bt601Full,
    /// BT.709 full range (0-255 for Y, U, V).
    /// Used for HDTV, modern displays, and sRGB content.
    Bt709Full,
    /// BT.601 limited range (16-235 for Y, 16-240 for U, V).
    /// Used for broadcast television and professional video equipment.
    Bt601Limited,
}

/// Convert a YUYV image to an RGB image.
///
/// # Arguments
///
/// * `src` - The YUYV image data.
/// * `dst` - The RGB image to store the result.
/// * `mode` - The mode to convert YUV to RGB.
///
/// # Returns
///
/// The RGB image in HxWx3 format.
pub fn convert_yuyv_to_rgb_u8<A: ImageAllocator>(
    src: &[u8],
    dst: &mut Image<u8, 3, A>,
    mode: YuvToRgbMode,
) -> Result<(), ImageError> {
    // the yuyv image is 2 bytes per pixel, so we need to divide by 2
    let (width, height) = (dst.width(), dst.height());
    if src.len() != width * height * 2 {
        return Err(ImageError::InvalidImageSize(
            src.len(),
            width,
            height,
            width * height * 2,
        ));
    }

    let rgb_data = dst.as_slice_mut();

    rgb_data
        .par_chunks_exact_mut(width * 3)
        .enumerate()
        .for_each(|(row, rgb_row)| {
            let yuyv_row_start = row * width * 2; // 2 bytes per pixel in YUYV
            let yuyv_row = &src[yuyv_row_start..yuyv_row_start + width * 2];

            rgb_row
                .chunks_exact_mut(6)
                .enumerate()
                .for_each(|(col, rgb_chunk)| {
                    let yuyv_idx = col * 4;
                    if yuyv_idx + 3 < yuyv_row.len() {
                        let y0 = yuyv_row[yuyv_idx];
                        let u = yuyv_row[yuyv_idx + 1];
                        let y1 = yuyv_row[yuyv_idx + 2];
                        let v = yuyv_row[yuyv_idx + 3];

                        // Convert YUV to RGB for first pixel
                        let (r0, g0, b0) = match mode {
                            YuvToRgbMode::Bt601Full => yuv_to_rgb_u8_bt601_full(y0, u, v),
                            YuvToRgbMode::Bt709Full => yuv_to_rgb_u8_bt709_full(y0, u, v),
                            YuvToRgbMode::Bt601Limited => yuv_to_rgb_u8_bt601_limited(y0, u, v),
                        };

                        // Convert YUV to RGB for second pixel
                        let (r1, g1, b1) = match mode {
                            YuvToRgbMode::Bt601Full => yuv_to_rgb_u8_bt601_full(y1, u, v),
                            YuvToRgbMode::Bt709Full => yuv_to_rgb_u8_bt709_full(y1, u, v),
                            YuvToRgbMode::Bt601Limited => yuv_to_rgb_u8_bt601_limited(y1, u, v),
                        };

                        // Write both RGB pixels
                        rgb_chunk[0] = r0;
                        rgb_chunk[1] = g0;
                        rgb_chunk[2] = b0;
                        rgb_chunk[3] = r1;
                        rgb_chunk[4] = g1;
                        rgb_chunk[5] = b1;
                    }
                });
        });

    Ok(())
}

/// Convert a UYVY image to an RGB image.
///
/// UYVY format stores pixels as: U Y0 V Y1 (4 bytes for 2 pixels)
/// where U and V are shared between two adjacent Y samples.
///
/// # Arguments
///
/// * `src` - The UYVY image data.
/// * `dst` - The RGB image to store the result.
/// * `mode` - The mode to convert YUV to RGB.
///
/// # Returns
///
/// The RGB image in HxWx3 format.
pub fn convert_uyvy_to_rgb_u8<A: ImageAllocator>(
    src: &[u8],
    dst: &mut Image<u8, 3, A>,
    mode: YuvToRgbMode,
) -> Result<(), ImageError> {
    // the uyvy image is 2 bytes per pixel, so we need to divide by 2
    let (width, height) = (dst.width(), dst.height());
    if src.len() != width * height * 2 {
        return Err(ImageError::InvalidImageSize(
            src.len(),
            width,
            height,
            width * height * 2,
        ));
    }

    let rgb_data = dst.as_slice_mut();

    rgb_data
        .par_chunks_exact_mut(width * 3)
        .enumerate()
        .for_each(|(row, rgb_row)| {
            let uyvy_row_start = row * width * 2; // 2 bytes per pixel in UYVY
            let uyvy_row = &src[uyvy_row_start..uyvy_row_start + width * 2];

            rgb_row
                .chunks_exact_mut(6)
                .enumerate()
                .for_each(|(col, rgb_chunk)| {
                    let uyvy_idx = col * 4;
                    if uyvy_idx + 3 < uyvy_row.len() {
                        // UYVY byte order: U Y0 V Y1
                        let u = uyvy_row[uyvy_idx];
                        let y0 = uyvy_row[uyvy_idx + 1];
                        let v = uyvy_row[uyvy_idx + 2];
                        let y1 = uyvy_row[uyvy_idx + 3];

                        // Convert YUV to RGB for first pixel
                        let (r0, g0, b0) = match mode {
                            YuvToRgbMode::Bt601Full => yuv_to_rgb_u8_bt601_full(y0, u, v),
                            YuvToRgbMode::Bt709Full => yuv_to_rgb_u8_bt709_full(y0, u, v),
                            YuvToRgbMode::Bt601Limited => yuv_to_rgb_u8_bt601_limited(y0, u, v),
                        };

                        // Convert YUV to RGB for second pixel
                        let (r1, g1, b1) = match mode {
                            YuvToRgbMode::Bt601Full => yuv_to_rgb_u8_bt601_full(y1, u, v),
                            YuvToRgbMode::Bt709Full => yuv_to_rgb_u8_bt709_full(y1, u, v),
                            YuvToRgbMode::Bt601Limited => yuv_to_rgb_u8_bt601_limited(y1, u, v),
                        };

                        // Write both RGB pixels
                        rgb_chunk[0] = r0;
                        rgb_chunk[1] = g0;
                        rgb_chunk[2] = b0;
                        rgb_chunk[3] = r1;
                        rgb_chunk[4] = g1;
                        rgb_chunk[5] = b1;
                    }
                });
        });

    Ok(())
}

#[inline]
fn yuv_to_rgb_u8_bt601_full(y: u8, u: u8, v: u8) -> (u8, u8, u8) {
    // Convert to signed integers and apply offsets
    let y_val = y as i32;
    let u_val = (u as i32) - 128;
    let v_val = (v as i32) - 128;

    // Fixed-point coefficients (multiplied by 1024)
    // 1.402 * 1024 = 1436
    // 0.344136 * 1024 = 352
    // 0.714136 * 1024 = 731
    // 1.772 * 1024 = 1815

    let r = y_val + ((1436 * v_val + 512) >> 10);
    let g = y_val - ((352 * u_val + 731 * v_val + 512) >> 10);
    let b = y_val + ((1815 * u_val + 512) >> 10);

    (
        r.clamp(0, 255) as u8,
        g.clamp(0, 255) as u8,
        b.clamp(0, 255) as u8,
    )
}

#[inline]
fn yuv_to_rgb_u8_bt709_full(y: u8, u: u8, v: u8) -> (u8, u8, u8) {
    let y_val = y as i32;
    let u_val = (u as i32) - 128;
    let v_val = (v as i32) - 128;

    // BT.709 coefficients * 1024
    // 1.5748 * 1024 = 1612
    // 0.187324 * 1024 = 192
    // 0.468124 * 1024 = 479
    // 1.8556 * 1024 = 1900

    let r = y_val + ((1612 * v_val + 512) >> 10);
    let g = y_val - ((192 * u_val + 479 * v_val + 512) >> 10);
    let b = y_val + ((1900 * u_val + 512) >> 10);

    (
        r.clamp(0, 255) as u8,
        g.clamp(0, 255) as u8,
        b.clamp(0, 255) as u8,
    )
}

#[inline]
fn yuv_to_rgb_u8_bt601_limited(y: u8, u: u8, v: u8) -> (u8, u8, u8) {
    // Apply range scaling first
    let y_val = ((y as i32 - 16) * 1192 + 512) >> 10; // 1192 = 1.164 * 1024
    let u_val = (u as i32) - 128;
    let v_val = (v as i32) - 128;

    // BT.601 limited coefficients * 1024
    let r = y_val + ((1634 * v_val + 512) >> 10);
    let g = y_val - ((401 * u_val + 832 * v_val + 512) >> 10);
    let b = y_val + ((2066 * u_val + 512) >> 10);

    (
        r.clamp(0, 255) as u8,
        g.clamp(0, 255) as u8,
        b.clamp(0, 255) as u8,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::ImageSize;
    use kornia_tensor::CpuAllocator;

    fn create_image(width: usize, height: usize) -> Image<u8, 3, CpuAllocator> {
        Image::<_, 3, _>::from_size_val(ImageSize { width, height }, 0_u8, CpuAllocator).unwrap()
    }

    #[test]
    fn test_invalid_buffer_size_too_small() {
        let src = vec![0_u8; 10]; // too small. should be 16 bytes
        let mut dst = create_image(4, 2);

        let result = convert_uyvy_to_rgb_u8(&src, &mut dst, YuvToRgbMode::Bt601Full);

        assert!(result.is_err());
        if let Err(ImageError::InvalidImageSize(actual, w, h, expected)) = result {
            assert_eq!(actual, 10);
            assert_eq!(w, 4);
            assert_eq!(h, 2);
            assert_eq!(expected, 16);
        } else {
            panic!("Expected InvalidImageSize error");
        }
    }

    #[test]
    fn test_invalid_buffer_size_too_large() {
        let src = vec![0_u8; 20]; // too large. should be 16 bytes
        let mut dst = create_image(4, 2);

        let result = convert_uyvy_to_rgb_u8(&src, &mut dst, YuvToRgbMode::Bt601Full);

        assert!(result.is_err());
        if let Err(ImageError::InvalidImageSize(actual, w, h, expected)) = result {
            assert_eq!(actual, 20);
            assert_eq!(w, 4);
            assert_eq!(h, 2);
            assert_eq!(expected, 16);
        } else {
            panic!("Expected InvalidImageSize error");
        }
    }

    #[test]
    fn test_correct_buffer_size() {
        let mut dst = create_image(4, 2);
        let src = vec![128u8; 16]; // correct size. 4*2*2 = 16 bytes

        let result = convert_uyvy_to_rgb_u8(&src, &mut dst, YuvToRgbMode::Bt601Full);

        assert!(result.is_ok());
    }
}
