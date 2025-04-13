use kornia_image::Image;
use std::cmp::{max, min};

/// Helper function to set a pixel's color, handling bounds checking.
fn set_pixel<const C: usize>(img: &mut Image<u8, C>, x: i64, y: i64, color: [u8; C]) {
    if x >= 0 && x < img.cols() as i64 && y >= 0 && y < img.rows() as i64 {
        let cols = img.cols() as i64;
        // Calculate the linear index for the start of the pixel data
        let pixel_linear_index = (y * cols + x) * C as i64;
        let start = pixel_linear_index as usize;
        // Bounds check above makes this indexing safe.
        img.as_slice_mut()[start..start + C].copy_from_slice(&color);
    }
}

/// Helper function to draw a filled horizontal line segment efficiently.
/// Clips coordinates to image bounds.
fn draw_horizontal_line_segment<const C: usize>(
    img: &mut Image<u8, C>,
    y: i64,
    x_start: i64,
    x_end: i64,
    color: [u8; C],
) {
    let img_cols = img.cols() as i64;
    let img_rows = img.rows() as i64;

    // Ensure y is within bounds
    if y < 0 || y >= img_rows {
        return;
    }

    // Clip x coordinates
    let x_min = max(0, min(x_start, x_end));
    let x_max = min(img_cols, max(x_start, x_end)); // Use exclusive end for range

    // Nothing to draw if clipped range is invalid
    if x_min >= x_max {
        return;
    }

    let y_usize = y as usize;
    let x_min_usize = x_min as usize;
    let x_max_usize = x_max as usize;
    let cols_usize = img.cols(); // Use usize for indexing

    // Calculate start index and length for the slice
    let start_index = (y_usize * cols_usize + x_min_usize) * C;
    let segment_len_pixels = x_max_usize - x_min_usize;
    let end_index = start_index + segment_len_pixels * C;

    // Get the mutable slice for the relevant row segment using safe indexing
    let data = img.as_slice_mut();
    if end_index <= data.len() {
        // Ensure slice bounds are valid
        let segment_slice = &mut data[start_index..end_index];

        // Fill the segment efficiently
        segment_slice.chunks_exact_mut(C).for_each(|pixel_chunk| {
            pixel_chunk.copy_from_slice(&color);
        });
    }
    // If end_index > data.len(), it implies an issue with coordinate clipping or calculation,
    // but the check prevents a panic.
}

/// Draws a line on an image inplace using a standard Bresenham's line algorithm.
///
/// # Arguments
///
/// * `img` - The image to draw on.
/// * `p0` - The start point of the line as a tuple of (x, y).
/// * `p1` - The end point of the line as a tuple of (x, y).
/// * `color` - The color of the line as an array of `C` elements.
/// * `thickness` - The thickness of the line.
///   **(Note: Thickness > 1 uses a simple square approximation by drawing a filled square
///   at each point along the line's path. This can appear blocky, especially for
///   non-horizontal/vertical lines, and differs from geometrically precise thick line
///   rendering algorithms.)**
pub fn draw_line<const C: usize>(
    img: &mut Image<u8, C>,
    p0: (i64, i64),
    p1: (i64, i64),
    color: [u8; C],
    thickness: usize,
) {
    let (mut x0, mut y0) = p0;
    let (x1, y1) = p1;

    let dx = (x1 - x0).abs();
    let dy = (y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };

    let mut err = dx - dy;

    // Precompute thickness offsets for the square drawing loop
    let half_thickness = (thickness / 2) as i64;
    let thickness_offset = if thickness <= 1 { 0 } else { half_thickness };
    // Adjust range for even thickness to avoid bias (centers the square better)
    let i_range = if thickness % 2 == 0 && thickness > 0 {
        -(half_thickness - 1)..=half_thickness
    } else {
        -thickness_offset..=thickness_offset
    };
    let j_range = i_range.clone();

    loop {
        if thickness <= 1 {
            set_pixel(img, x0, y0, color);
        } else {
            // Approximate thickness by drawing a small filled square centered at (x0, y0)
            for i in i_range.clone() {
                for j in j_range.clone() {
                    set_pixel(img, x0 + i, y0 + j, color);
                }
            }
        }

        // Check end condition
        if x0 == x1 && y0 == y1 {
            break;
        }

        // Bresenham algorithm steps
        let e2 = 2 * err;

        if e2 > -dy {
            err -= dy;
            x0 += sx;
        }
        if e2 < dx {
            err += dx;
            y0 += sy;
        }
    }
}

/// Draws a rectangle outline on an image inplace.
///
/// # Arguments
///
/// * `img` - The image to draw on.
/// * `top_left` - The top-left corner coordinates (x, y).
/// * `bottom_right` - The bottom-right corner coordinates (x, y).
/// * `color` - The color of the rectangle outline.
/// * `thickness` - The thickness of the lines. (See `draw_line` note on thickness > 1).
pub fn draw_rect<const C: usize>(
    img: &mut Image<u8, C>,
    top_left: (i64, i64),
    bottom_right: (i64, i64),
    color: [u8; C],
    thickness: usize,
) {
    let (x0, y0) = top_left;
    let (x1, y1) = bottom_right;

    // Determine the actual min/max coordinates
    let lx0 = min(x0, x1);
    let ly0 = min(y0, y1);
    let lx1 = max(x0, x1);
    let ly1 = max(y0, y1);

    if thickness == 0 {
        return;
    }

    // Calculate thickness offsets for horizontal lines
    // Floor/Ceil handles both odd and even thickness correctly for iteration range
    let half_thickness_floor = ((thickness - 1) / 2) as i64;
    let half_thickness_ceil = (thickness / 2) as i64;

    // Draw horizontal lines efficiently using the helper function
    // Iterate from -floor to +ceil around the target y coordinate
    for y_offset in -half_thickness_floor..=half_thickness_ceil {
        // Top border lines - use exclusive end for segment drawing (+1)
        draw_horizontal_line_segment(img, ly0 + y_offset, lx0, lx1 + 1, color);
        // Bottom border lines - use exclusive end for segment drawing (+1)
        draw_horizontal_line_segment(img, ly1 + y_offset, lx0, lx1 + 1, color);
    }

    // Draw vertical lines using draw_line
    // Adjust vertical range to avoid overdrawing corners excessively with thick lines,
    // but ensure full coverage for thickness 1.
    // Draw from the pixel row just below the top horizontal band
    // to the pixel row just above the bottom horizontal band.
    let vert_y0 = ly0 + half_thickness_ceil + 1;
    let vert_y1 = ly1 - half_thickness_floor - 1;

    // Only draw vertical segments if the adjusted range is valid (i.e., height allows space between horizontal bands)
    if vert_y0 <= vert_y1 {
        // Use the original draw_line for vertical segments, passing the specified thickness
        draw_line(img, (lx0, vert_y0), (lx0, vert_y1), color, thickness); // Left
        draw_line(img, (lx1, vert_y0), (lx1, vert_y1), color, thickness); // Right
    }
    // No explicit 'else' needed:
    // - If thickness is 1 and height is 0 or 1, horizontal drawing covers it.
    // - If thickness > 1 and vert_y0 > vert_y1, the horizontal bands overlap or meet,
    //   covering the vertical space, so no separate vertical draw is needed.
}

/// Draws a filled rectangle on an image inplace, optimized for row iteration.
///
/// # Arguments
///
/// * `img` - The image to draw on.
/// * `top_left` - The top-left corner coordinates (x, y).
/// * `bottom_right` - The bottom-right corner coordinates (x, y).
/// * `color` - The fill color of the rectangle.
pub fn draw_filled_rect<const C: usize>(
    img: &mut Image<u8, C>,
    top_left: (i64, i64),
    bottom_right: (i64, i64),
    color: [u8; C],
) {
    let (x_start, y_start) = top_left;
    let (x_end, y_end) = bottom_right;

    // Determine the actual min/max coordinates
    let x_min_coord = min(x_start, x_end);
    let y_min_coord = min(y_start, y_end);
    let x_max_coord = max(x_start, x_end);
    let y_max_coord = max(y_start, y_end);

    // Clip y coordinates to image bounds to reduce loop iterations
    let img_rows = img.rows() as i64;
    let y_min = max(0, y_min_coord);
    // Iterate up to, but not including, y_max_coord + 1 (exclusive end)
    let y_max = min(img_rows, y_max_coord + 1);

    // Iterate through rows and draw horizontal line segments
    // The helper function handles x-clipping internally.
    for y in y_min..y_max {
        // Pass coordinates ensuring x_max_coord is included in the fill (exclusive end)
        draw_horizontal_line_segment(img, y, x_min_coord, x_max_coord + 1, color);
    }
}

#[cfg(test)]
mod tests {
    use super::*; // Import draw_line, draw_rect, draw_filled_rect
    use kornia_image::{Image, ImageError, ImageSize};

    #[rustfmt::skip]
    #[test]
    fn test_draw_line() -> Result<(), ImageError> {
        let mut img = Image::new(
            ImageSize { width: 5, height: 5 }, vec![0u8; 25],
        )?;
        // Standard Bresenham line from (0,0) to (4,4)
        draw_line(&mut img, (0, 0), (4, 4), [255], 1);
        assert_eq!(
            img.as_slice(),
            &[
                255,   0,   0,   0,   0, // (0,0)
                  0, 255,   0,   0,   0, // (1,1)
                  0,   0, 255,   0,   0, // (2,2)
                  0,   0,   0, 255,   0, // (3,3)
                  0,   0,   0,   0, 255, // (4,4)
            ]
        );
        Ok(())
    }

    #[rustfmt::skip]
    #[test]
    fn test_draw_rect() -> Result<(), ImageError> {
        let mut img = Image::new(
            ImageSize { width: 5, height: 5 }, vec![0u8; 25],
        )?;
        // Draw rect from (1,1) to (3,3) with thickness 1
        draw_rect(&mut img, (1, 1), (3, 3), [128], 1);
        assert_eq!(
            img.as_slice(),
            &[
                  0,   0,   0,   0,   0,
                  0, 128, 128, 128,   0, // Row 1: (1,1) to (3,1)
                  0, 128,   0, 128,   0, // Row 2: (1,2) and (3,2)
                  0, 128, 128, 128,   0, // Row 3: (1,3) to (3,3)
                  0,   0,   0,   0,   0,
            ]
        );
        Ok(())
    }

    #[rustfmt::skip]
    #[test]
    fn test_draw_rect_thick() -> Result<(), ImageError> {
        let mut img = Image::new(
            ImageSize { width: 7, height: 7 }, vec![0u8; 49],
        )?;
        // Draw rect from (1,1) to (5,5) with thickness 3
        draw_rect(&mut img, (1, 1), (5, 5), [128], 3);

        // Expected output based on horizontal segments and vertical lines with square thickness
        #[rustfmt::skip]
        let expected = vec![
            128, 128, 128, 128, 128, 128, 128, // Row 0 (Top border)
            128, 128, 128, 128, 128, 128, 128, // Row 1 (Top border)
            128, 128, 128, 128, 128, 128, 128, // Row 2 (Top border)
            128, 128, 128,   0, 128, 128, 128, // Row 3 (Vertical borders)
            128, 128, 128, 128, 128, 128, 128, // Row 4 (Bottom border)
            128, 128, 128, 128, 128, 128, 128, // Row 5 (Bottom border)
            128, 128, 128, 128, 128, 128, 128, // Row 6 (Bottom border)
        ];
        assert_eq!(img.as_slice(), expected.as_slice());
        Ok(())
    }

    #[rustfmt::skip]
    #[test]
    fn test_draw_rect_rgb() -> Result<(), ImageError> {
        let mut img = Image::<u8, 3>::from_size_val(
            ImageSize { width: 5, height: 5 }, 0u8,
        )?;
        draw_rect(&mut img, (1, 1), (3, 3), [0, 255, 0], 1); // Green rectangle
        // Check corners and mid-points of sides (channel 1 = Green)
        assert_eq!(img.get_pixel(1, 1, 1)?, &255); // Top-left G
        assert_eq!(img.get_pixel(1, 3, 1)?, &255); // Top-right G
        assert_eq!(img.get_pixel(3, 1, 1)?, &255); // Bottom-left G
        assert_eq!(img.get_pixel(3, 3, 1)?, &255); // Bottom-right G
        assert_eq!(img.get_pixel(1, 2, 1)?, &255); // Mid-top G
        assert_eq!(img.get_pixel(2, 1, 1)?, &255); // Mid-left G
        assert_eq!(img.get_pixel(2, 3, 1)?, &255); // Mid-right G
        assert_eq!(img.get_pixel(3, 2, 1)?, &255); // Mid-bottom G
        // Check center (should be unchanged)
        assert_eq!(img.get_pixel(2, 2, 0)?, &0); // Center R
        assert_eq!(img.get_pixel(2, 2, 1)?, &0); // Center G
        assert_eq!(img.get_pixel(2, 2, 2)?, &0); // Center B
        Ok(())
    }

    #[rustfmt::skip]
    #[test]
    fn test_draw_filled_rect() -> Result<(), ImageError> {
        let mut img = Image::new(
            ImageSize { width: 5, height: 5 }, vec![0u8; 25],
        )?;
        // Fill rect from (1,1) to (3,2) inclusive
        draw_filled_rect(&mut img, (1, 1), (3, 2), [200]);
        assert_eq!(
            img.as_slice(),
            &[
                  0,   0,   0,   0,   0,
                  0, 200, 200, 200,   0, // Row 1: (1,1) to (3,1)
                  0, 200, 200, 200,   0, // Row 2: (1,2) to (3,2)
                  0,   0,   0,   0,   0,
                  0,   0,   0,   0,   0,
            ]
        );
        Ok(())
    }

    #[rustfmt::skip]
    #[test]
    fn test_draw_filled_rect_rgb() -> Result<(), ImageError> {
        let mut img = Image::<u8, 3>::from_size_val(
            ImageSize { width: 4, height: 3 }, 0u8,
        )?;
        // Fill rect from (1,0) to (2,1) inclusive with Blue
        draw_filled_rect(&mut img, (1, 0), (2, 1), [0, 0, 255]);
        assert_eq!(
            img.as_slice(),
            &[ // R G B   R G B   R G B   R G B
                0, 0, 0,  0, 0, 255,  0, 0, 255,  0, 0, 0, // Row 0: (1,0) and (2,0) are Blue
                0, 0, 0,  0, 0, 255,  0, 0, 255,  0, 0, 0, // Row 1: (1,1) and (2,1) are Blue
                0, 0, 0,  0, 0,   0,  0, 0,   0,  0, 0, 0, // Row 2: Unchanged
            ]
        );
        Ok(())
    }

    #[test]
    fn test_draw_filled_rect_out_of_bounds() -> Result<(), ImageError> {
        let mut img = Image::<u8, 1>::from_size_val(
            ImageSize {
                width: 3,
                height: 3,
            },
            0,
        )?;
        // Rectangle completely outside
        draw_filled_rect(&mut img, (10, 10), (12, 12), [255]);
        assert_eq!(img.as_slice(), &[0, 0, 0, 0, 0, 0, 0, 0, 0]);

        // Rectangle partially outside (top-left)
        draw_filled_rect(&mut img, (-1, -1), (1, 1), [100]);
        assert_eq!(
            img.as_slice(),
            &[
                100, 100, 0, // Row 0: (0,0), (1,0)
                100, 100, 0, // Row 1: (0,1), (1,1)
                0, 0, 0, // Row 2
            ]
        );

        // Reset image
        img.fill(0)?;
        // Rectangle partially outside (bottom-right)
        draw_filled_rect(&mut img, (1, 1), (3, 3), [150]);
        assert_eq!(
            img.as_slice(),
            &[
                0, 0, 0, 0, 150, 150, // Row 1: (1,1), (2,1)
                0, 150, 150, // Row 2: (1,2), (2,2)
            ]
        );
        Ok(())
    }

    #[test]
    fn test_draw_rect_out_of_bounds() -> Result<(), ImageError> {
        let mut img = Image::<u8, 1>::from_size_val(
            ImageSize {
                width: 4,
                height: 4,
            },
            0,
        )?;
        // Rectangle partially outside (top-left)
        draw_rect(&mut img, (-1, -1), (2, 2), [100], 1);
        // Expected: Top line at y=0 (clipped from -1), Left line at x=0 (clipped from -1)
        // Bottom line at y=2, Right line at x=2
        assert_eq!(
            img.as_slice(),
            &[
                100, 100, 100, 0, // Row 0: (0,0) to (2,0)
                100, 0, 100, 0, // Row 1: (0,1) and (2,1)
                100, 100, 100, 0, // Row 2: (0,2) to (2,2)
                0, 0, 0, 0, // Row 3
            ]
        );

        // Reset image
        img.fill(0)?;
        // Rectangle partially outside (bottom-right)
        draw_rect(&mut img, (1, 1), (4, 4), [120], 1);
        // Expected: Top y=1, Left x=1, Bottom y=3 (clipped from 4), Right x=3 (clipped from 4)
        assert_eq!(
            img.as_slice(),
            &[
                0, 0, 0, 0, 0, 120, 120, 120, // Row 1: (1,1) to (3,1)
                0, 120, 0, 120, // Row 2: (1,2) and (3,2)
                0, 120, 120, 120, // Row 3: (1,3) to (3,3)
            ]
        );
        Ok(())
    }

    #[test]
    fn test_draw_rect_zero_thickness() -> Result<(), ImageError> {
        let mut img = Image::<u8, 1>::from_size_val(
            ImageSize {
                width: 3,
                height: 3,
            },
            1,
        )?;
        draw_rect(&mut img, (0, 0), (2, 2), [0], 0);
        // Expect no change
        assert_eq!(img.as_slice(), &[1, 1, 1, 1, 1, 1, 1, 1, 1]);
        Ok(())
    }

    #[test]
    fn test_draw_rect_single_pixel() -> Result<(), ImageError> {
        let mut img = Image::<u8, 1>::from_size_val(
            ImageSize {
                width: 3,
                height: 3,
            },
            0,
        )?;
        // A rect where corners are the same is effectively a point
        draw_rect(&mut img, (1, 1), (1, 1), [255], 1);
        // Expect only the single pixel (1,1) to be set
        assert_eq!(img.as_slice(), &[0, 0, 0, 0, 255, 0, 0, 0, 0,]);
        Ok(())
    }

    #[test]
    fn test_draw_rect_horizontal_line() -> Result<(), ImageError> {
        let mut img = Image::<u8, 1>::from_size_val(
            ImageSize {
                width: 5,
                height: 3,
            },
            0,
        )?;
        // A rect with same y0, y1 is a horizontal line
        draw_rect(&mut img, (1, 1), (3, 1), [255], 1);
        assert_eq!(
            img.as_slice(),
            &[
                0, 0, 0, 0, 0, 0, 255, 255, 255, 0, // Should draw the line at y=1
                0, 0, 0, 0, 0,
            ]
        );
        Ok(())
    }

    #[test]
    fn test_draw_rect_vertical_line() -> Result<(), ImageError> {
        let mut img = Image::<u8, 1>::from_size_val(
            ImageSize {
                width: 3,
                height: 5,
            },
            0,
        )?;
        // A rect with same x0, x1 is a vertical line
        draw_rect(&mut img, (1, 1), (1, 3), [255], 1);
        assert_eq!(
            img.as_slice(),
            &[
                0, 0, 0, 0, 255, 0, // y=1
                0, 255, 0, // y=2
                0, 255, 0, // y=3
                0, 0, 0,
            ]
        );
        Ok(())
    }
}
