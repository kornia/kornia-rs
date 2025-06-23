use kornia_image::{allocator::ImageAllocator, Image};

/// Convert a YUYV image to an RGB image.
///
/// # Arguments
///
/// * `yuyv_data` - The YUYV image data.
/// * `rgb_image` - The RGB image to store the result.
///
/// # Returns
///
/// The RGB image in HxWx3 format.
pub fn convert_yuyv_to_rgb_u8<A: ImageAllocator>(
    yuyv_data: &[u8],
    rgb_image: &mut Image<u8, 3, A>,
) {
    let width = rgb_image.width();
    let rgb_data = rgb_image.as_slice_mut();

    rgb_data
        .chunks_exact_mut(width * 3)
        .enumerate()
        .for_each(|(row, rgb_row)| {
            let yuyv_row_start = row * width * 2; // 2 bytes per pixel in YUYV
            let yuyv_row = &yuyv_data[yuyv_row_start..yuyv_row_start + width * 2];

            rgb_row
                .chunks_exact_mut(6)
                .enumerate()
                .for_each(|(col, rgb_chunk)| {
                    let yuyv_idx = col * 4;
                    if yuyv_idx + 3 < yuyv_row.len() {
                        let y0 = yuyv_row[yuyv_idx] as f32;
                        let u = yuyv_row[yuyv_idx + 1] as f32 - 128.0;
                        let y1 = yuyv_row[yuyv_idx + 2] as f32;
                        let v = yuyv_row[yuyv_idx + 3] as f32 - 128.0;

                        // Convert YUV to RGB for first pixel
                        let (r0, g0, b0) = yuv_to_rgb(y0, u, v);

                        // Convert YUV to RGB for second pixel
                        let (r1, g1, b1) = yuv_to_rgb(y1, u, v);

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
}

#[inline]
fn clamp_u8(value: f32) -> u8 {
    value.clamp(0.0, 255.0) as u8
}

#[inline]
fn yuv_to_rgb(y: f32, u: f32, v: f32) -> (u8, u8, u8) {
    let r = clamp_u8(y + 1.402 * v);
    let g = clamp_u8(y - 0.344136 * u - 0.714136 * v);
    let b = clamp_u8(y + 1.772 * u);
    (r, g, b)
}
