use kornia_image::Image;
use std::cmp::{max, min};

/// Helper function to set a pixel's color, handling bounds checking.
fn set_pixel<const C: usize>(img: &mut Image<u8, C>, x: i64, y: i64, color: [u8; C]) {
    if x >= 0 && x < img.cols() as i64 && y >= 0 && y < img.rows() as i64 {
        let cols = img.cols() as i64;
        let pixel_linear_index = (y * cols + x) * C as i64;
        let start = pixel_linear_index as usize;
        // Bounds check above makes this indexing safe.
        img.as_slice_mut()[start..start + C].copy_from_slice(&color);
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
/// * `thickness` - The thickness of the line.
///   **(Note: Thickness > 1 uses a simple square approximation and may appear blocky,
///   especially for diagonal lines.)**
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

    let half_thickness = (thickness / 2) as i64;
    let thickness_offset = if thickness <= 1 { 0 } else { half_thickness };
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
            // Approximate thickness by drawing a small filled square
            for i in i_range.clone() {
                 for j in j_range.clone() {
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

    let lx0 = min(x0, x1);
    let ly0 = min(y0, y1);
    let lx1 = max(x0, x1);
    let ly1 = max(y0, y1);

    let half_thickness = (thickness / 2) as i64;

    // Draw horizontal lines
    draw_line(img, (lx0, ly0), (lx1, ly0), color, thickness); // Top
    draw_line(img, (lx0, ly1), (lx1, ly1), color, thickness); // Bottom
    // Draw vertical lines (adjust slightly for thickness to reduce corner overlap)
    draw_line(img, (lx0, ly0 + half_thickness), (lx0, ly1 - half_thickness), color, thickness); // Left
    draw_line(img, (lx1, ly0 + half_thickness), (lx1, ly1 - half_thickness), color, thickness); // Right
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

    let x_min_coord = min(x_start, x_end);
    let y_min_coord = min(y_start, y_end);
    let x_max_coord = max(x_start, x_end);
    let y_max_coord = max(y_start, y_end);

    let img_cols = img.cols() as i64;
    let img_rows = img.rows() as i64;
    let x_min = max(0, x_min_coord);
    let y_min = max(0, y_min_coord);
    let x_max = min(img_cols, x_max_coord);
    let y_max = min(img_rows, y_max_coord);

    let data = img.as_slice_mut();
    let data_len = data.len();

    for y in y_min..y_max {
        let row_start_index = (y * img_cols) * C as i64;
        for x in x_min..x_max {
            let pixel_start_index = (row_start_index + x * C as i64) as usize;
             if pixel_start_index + C <= data_len {
                 data[pixel_start_index..pixel_start_index + C].copy_from_slice(&color);
             }
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::{Image, ImageError, ImageSize};

    #[rustfmt::skip]
    #[test]
    fn test_draw_line() -> Result<(), ImageError> {
        let mut img = Image::new(
            ImageSize { width: 5, height: 5 }, vec![0u8; 25],
        )?;
        draw_line(&mut img, (0, 0), (4, 4), [255], 1);
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
                  0, 128,   0, 128,   0,
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
        assert_eq!(img.get_pixel(1, 1, 1)?, &255);
        assert_eq!(img.get_pixel(3, 1, 1)?, &255);
        assert_eq!(img.get_pixel(1, 3, 1)?, &255);
        assert_eq!(img.get_pixel(3, 3, 1)?, &255);
        assert_eq!(img.get_pixel(2, 1, 1)?, &255);
        assert_eq!(img.get_pixel(1, 2, 1)?, &255);
        assert_eq!(img.get_pixel(2, 2, 1)?, &0);
        Ok(())
    }

    #[rustfmt::skip]
    #[test]
    fn test_draw_filled_rect() -> Result<(), ImageError> {
        let mut img = Image::new(
            ImageSize { width: 5, height: 5 }, vec![0u8; 25],
        )?;
        draw_filled_rect(&mut img, (1, 1), (4, 3), [200]);
        assert_eq!(
            img.as_slice(),
            &[
                  0,   0,   0,   0,   0,
                  0, 200, 200, 200,   0,
                  0, 200, 200, 200,   0,
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
        draw_filled_rect(&mut img, (1, 0), (3, 2), [0, 0, 255]); // Blue rectangle
        assert_eq!(
            img.as_slice(),
            &[
                0, 0, 0,   0, 0, 255,   0, 0, 255,   0, 0, 0,
                0, 0, 0,   0, 0, 255,   0, 0, 255,   0, 0, 0,
                0, 0, 0,   0, 0,   0,   0, 0,   0,   0, 0, 0,
            ]
        );
        Ok(())
    }
}
