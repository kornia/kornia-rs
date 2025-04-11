use kornia_image::Image;
use std::cmp::{max, min};

/// Helper function to set a pixel's color, handling bounds checking.
#[inline]
fn set_pixel<const C: usize>(img: &mut Image<u8, C>, x: i64, y: i64, color: [u8; C]) {
    if x >= 0 && x < img.cols() as i64 && y >= 0 && y < img.rows() as i64 {
        let pixel_linear_index = (y * img.cols() as i64 + x) * C as i64;
        // Bounds check above makes this indexing safe.
        let start = pixel_linear_index as usize;
        // Check if the slice operation is within bounds (extra safety)
        if start + C <= img.as_slice().len() {
             img.as_slice_mut()[start..start + C].copy_from_slice(&color);
        } else {
             eprintln!("Warning: Attempted to write out of bounds at ({}, {})", x, y);
        }
    }
}

/// Draws a line on an image inplace using a standard Bresenham's line algorithm.
///
/// # Arguments
///
/// * `img` - The image to draw on.
/// * `p0` - The start point of the line as a tuple of (x, y).
/// * `p1` - The end point of the line as a tuple of (x, y).
/// * `color` - The color of the line as an array of `C` elements.
/// * `thickness` - The thickness of the line. (Note: thickness > 1 is approximate).
pub fn draw_line<const C: usize>(
    img: &mut Image<u8, C>,
    p0: (i64, i64),
    p1: (i64, i64),
    color: [u8; C],
    thickness: usize,
) { 
    // Create local variables for moving start point
    let (mut x0, mut y0) = p0;
    let (x1, y1) = p1;

    let dx = (x1 - x0).abs();
    let dy = (y1 - y0).abs();
    // Get slopes
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };

    let mut err = dx - dy; // Use standard error initialization

    let half_thickness = thickness as i64 / 2;
    // Offset for centering thickness, slightly different approach
    let offset_x = if thickness > 1 { half_thickness } else { 0 };
    let offset_y = if thickness > 1 { half_thickness } else { 0 };


    loop {
        // Draw the pixel(s) for thickness
        if thickness == 1 {
            set_pixel(img, x0, y0, color);
        } else {
            // Approximate thickness by drawing a small filled rectangle centered at the point
            // This is not perfect for diagonal lines but simple to implement.
            for i in -offset_x..=offset_x {
                 for j in -offset_y..=offset_y {
                      set_pixel(img, x0 + i, y0 + j, color);
                 }
            }
        }


        if x0 == x1 && y0 == y1 {
            break;
        }

        let e2 = 2 * err;

        if e2 > -dy {
            err -= dy;
            x0 += sx;
        }
        if e2 < dx { // Use standard comparison with dx
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
/// * `thickness` - The thickness of the lines.
pub fn draw_rect<const C: usize>(
    img: &mut Image<u8, C>,
    top_left: (i64, i64),
    bottom_right: (i64, i64),
    color: [u8; C],
    thickness: usize,
) {
    let (x0, y0) = top_left;
    let (x1, y1) = bottom_right;

    // Ensure coordinates are ordered correctly for line drawing
    let (lx0, lx1) = (min(x0, x1), max(x0, x1));
    let (ly0, ly1) = (min(y0, y1), max(y0, y1));

    // Draw the four lines of the rectangle
    draw_line(img, (lx0, ly0), (lx1, ly0), color, thickness); // Top
    draw_line(img, (lx0, ly1), (lx1, ly1), color, thickness); // Bottom
    // Draw vertical lines fully to connect corners properly
    draw_line(img, (lx0, ly0), (lx0, ly1), color, thickness); // Left
    draw_line(img, (lx1, ly0), (lx1, ly1), color, thickness); // Right
}

/// Draws a filled rectangle on an image inplace.
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

    // Ensure coordinates are ordered correctly
    let x_min_coord = min(x_start, x_end);
    let y_min_coord = min(y_start, y_end);
    let x_max_coord = max(x_start, x_end);
    let y_max_coord = max(y_start, y_end);


    // Clamp coordinates to image bounds
    let x_min = max(0, x_min_coord);
    let y_min = max(0, y_min_coord);
    let x_max = min(img.cols() as i64, x_max_coord); // Use exclusive end for iteration
    let y_max = min(img.rows() as i64, y_max_coord); // Use exclusive end for iteration

    for y in y_min..y_max {
        for x in x_min..x_max {
            set_pixel(img, x, y, color);
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*; // Import functions from the parent module
    use kornia_image::{Image, ImageError, ImageSize};

    #[rustfmt::skip]
    #[test]
    fn test_draw_line() -> Result<(), ImageError> {
        let mut img = Image::new(
            ImageSize { width: 5, height: 5 }, vec![0u8; 25],
        )?;
        draw_line(&mut img, (0, 0), (4, 4), [255], 1);
        // This is the expected output for a standard Bresenham diagonal
        assert_eq!(
            img.as_slice(),
            &[
                255,   0,   0,   0,   0,
                  0, 255,   0,   0,   0,
                  0,   0, 255,   0,   0,
                  0,   0,   0, 255,   0,
                  0,   0,   0,   0, 255,
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
        draw_rect(&mut img, (1, 1), (3, 3), [128], 1);
        assert_eq!(
            img.as_slice(),
            &[
                  0,   0,   0,   0,   0,
                  0, 128, 128, 128,   0,
                  0, 128,   0, 128,   0, // Center pixel is not drawn for outline
                  0, 128, 128, 128,   0,
                  0,   0,   0,   0,   0,
            ]
        );
        Ok(())
    }

     #[rustfmt::skip]
    #[test]
    fn test_draw_rect_rgb() -> Result<(), ImageError> {
        let mut img = Image::<u8, 3>::from_size_val(
            ImageSize { width: 5, height: 5 }, 0u8,
        )?;
        draw_rect(&mut img, (1, 1), (3, 3), [0, 255, 0], 1); // Green rectangle
        // Check a few corner and edge pixels
        assert_eq!(img.get_pixel(1, 1, 1)?, &255); // Top-left green
        assert_eq!(img.get_pixel(3, 1, 1)?, &255); // Top-right green
        assert_eq!(img.get_pixel(1, 3, 1)?, &255); // Bottom-left green
        assert_eq!(img.get_pixel(3, 3, 1)?, &255); // Bottom-right green
        assert_eq!(img.get_pixel(2, 1, 1)?, &255); // Top edge green
        assert_eq!(img.get_pixel(1, 2, 1)?, &255); // Left edge green
        assert_eq!(img.get_pixel(2, 2, 1)?, &0);   // Center should be unchanged (black)
        Ok(())
    }

    #[rustfmt::skip]
    #[test]
    fn test_draw_filled_rect() -> Result<(), ImageError> {
        let mut img = Image::new(
            ImageSize { width: 5, height: 5 }, vec![0u8; 25],
        )?;
        // Draw rectangle from (1,1) inclusive to (4,3) exclusive
        draw_filled_rect(&mut img, (1, 1), (4, 3), [200]);
        assert_eq!(
            img.as_slice(),
            &[
                  0,   0,   0,   0,   0,
                  0, 200, 200, 200,   0, // Row 1 (y=1), x=1,2,3
                  0, 200, 200, 200,   0, // Row 2 (y=2), x=1,2,3
                  0,   0,   0,   0,   0, // Row 3 (y=3) is outside
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
        // Draw rectangle from (1,0) inclusive to (3,2) exclusive
        draw_filled_rect(&mut img, (1, 0), (3, 2), [0, 0, 255]); // Blue rectangle
        assert_eq!(
            img.as_slice(),
            &[
                0, 0, 0,   0, 0, 255,   0, 0, 255,   0, 0, 0, // Row 0 (y=0), x=1,2
                0, 0, 0,   0, 0, 255,   0, 0, 255,   0, 0, 0, // Row 1 (y=1), x=1,2
                0, 0, 0,   0, 0,   0,   0, 0,   0,   0, 0, 0, // Row 2 (y=2) is outside
            ]
        );
        Ok(())
    }
}
