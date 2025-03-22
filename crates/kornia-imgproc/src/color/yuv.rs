use kornia_image::Image;

/// Convert an NV12 buffer to RGB
///
/// # Arguments
///
/// * `buffer` - Input buffer in NV12 format
/// * `cols` - Width of the image
/// * `rows` - Height of the image
///
/// # Returns
///
/// A Result containing the RGB Image or an error
///
/// # Example
///
/// ```
/// use kornia_image::Image;
/// use kornia_imgproc::color::nv12_to_rgb;
///
/// // Create a simple NV12 buffer (Y plane followed by interleaved UV)
/// let width = 4;
/// let height = 4;
/// let buffer = vec![0u8; width * height * 3 / 2]; // NV12 is 1.5 bytes per pixel
///
/// let rgb = nv12_to_rgb(&buffer, width, height).unwrap();
/// assert_eq!(rgb.num_channels(), 3);
/// assert_eq!(rgb.size().width, width);
/// assert_eq!(rgb.size().height, height);
/// ```
pub fn nv12_to_rgb<E>(
    buffer: &[u8],
    cols: usize,
    rows: usize,
) -> Result<Image<u8, 3>, E>
where
    E: From<kornia_image::ImageError>,
{
    // Allocate memory for RGB output
    let mut rgb_data = vec![0u8; cols * rows * 3];
    
    // Implement NV12 to RGB conversion
    // NV12 format: Y plane followed by interleaved U and V planes
    let y_plane_size = cols * rows;
    let uv_plane_offset = y_plane_size;
    
    // Check if the buffer size is sufficient
    if buffer.len() < y_plane_size + (cols * rows / 2) {
        return Err(E::from(kornia_image::ImageError::InvalidImageSize(
            cols, rows, 0, 0
        )));
    }
    
    for y in 0..rows {
        for x in 0..cols {
            let y_index = y * cols + x;
            let y_value = buffer[y_index] as f32;
            
            // UV values are subsampled (one U,V pair per 2x2 Y values)
            let uv_index = uv_plane_offset + (y / 2) * cols + (x / 2) * 2;
            let u_value = buffer[uv_index] as f32 - 128.0;
            let v_value = buffer[uv_index + 1] as f32 - 128.0;
            
            // Standard YUV to RGB conversion
            let r = y_value + 1.402 * v_value;
            let g = y_value - 0.344136 * u_value - 0.714136 * v_value;
            let b = y_value + 1.772 * u_value;
            
            // Clamp values to 0-255 range
            let r = r.max(0.0).min(255.0) as u8;
            let g = g.max(0.0).min(255.0) as u8;
            let b = b.max(0.0).min(255.0) as u8;
            
            // Write to output buffer
            let rgb_index = y_index * 3;
            rgb_data[rgb_index] = r;
            rgb_data[rgb_index + 1] = g;
            rgb_data[rgb_index + 2] = b;
        }
    }
    
    Image::<u8, 3>::new(
        [cols, rows].into(),
        rgb_data,
    ).map_err(E::from)
}

#[cfg(test)]
mod tests {
    use kornia_image::Image;
    
    #[test]
    fn nv12_to_rgb_basic() -> Result<(), Box<dyn std::error::Error>> {
        // Create a simple NV12 test pattern
        // 2x2 image with first pixel white, others black
        // Y plane: [235, 16, 16, 16]  (235 is white in Y, 16 is black)
        // UV plane: [128, 128]  (128,128 represents no color in UV)
        let buffer = vec![235, 16, 16, 16, 128, 128]; // 2x2 image requires 6 bytes in NV12
        
        let result = super::nv12_to_rgb(&buffer, 2, 2)?;
        
        // Check dimensions
        assert_eq!(result.cols(), 2);
        assert_eq!(result.rows(), 2);
        assert_eq!(result.num_channels(), 3);
        
        // First pixel should be white (or very close)
        let first_pixel = [result.as_slice()[0], result.as_slice()[1], result.as_slice()[2]];
        assert!(first_pixel.iter().all(|&v| v >= 240), "First pixel should be white");
        
        // Other pixels should be black (or very close)
        for i in 1..4 {
            let pixel_start = i * 3;
            let pixel = [
                result.as_slice()[pixel_start],
                result.as_slice()[pixel_start + 1],
                result.as_slice()[pixel_start + 2],
            ];
            assert!(pixel.iter().all(|&v| v <= 20), "Non-first pixels should be black");
        }
        
        Ok(())
    }
    
    #[test]
    fn nv12_to_rgb_invalid_buffer() {
        // Test with buffer that's too small
        let buffer = vec![0u8; 5]; // Too small for 2x2 image which requires 6 bytes
        let result = super::nv12_to_rgb(&buffer, 2, 2);
        assert!(result.is_err());
    }
}