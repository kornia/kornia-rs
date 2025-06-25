use kornia_image::{allocator::ImageAllocator, Image, ImageError};
use rayon::prelude::*;

/// The mode to convert YUV to RGB.
pub enum YuvToRgbMode {
    /// BT.601 full range.
    Bt601Full,
    /// BT.709 full range.
    Bt709Full,
    /// BT.601 limited range.
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
