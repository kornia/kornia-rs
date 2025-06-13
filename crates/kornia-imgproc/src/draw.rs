use kornia_image::{allocator::ImageAllocator, Image};

/// Draws a line on an image inplace.
///
/// # Arguments
///
/// * `img` - The image to draw on.
/// * `p0` - The start point of the line as a tuple of (x, y).
/// * `p1` - The end point of the line as a tuple of (x, y).
/// * `color` - The color of the line as an array of `C` elements.
/// * `thickness` - The thickness of the line.
pub fn draw_line<const C: usize, A: ImageAllocator>(
    img: &mut Image<u8, C, A>,
    p0: (i64, i64),
    p1: (i64, i64),
    color: [u8; C],
    thickness: usize,
) {
    // Create local variables for moving start point
    let (mut x0, mut y0) = p0;
    let (x1, y1) = p1;

    // Get absolute x/y offset
    let dx = if x0 > x1 { x0 - x1 } else { x1 - x0 };
    let dy = if y0 > y1 { y0 - y1 } else { y1 - y0 };

    // Get slopes
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };

    // Initialize error
    let mut err = if dx > dy { dx } else { -dy } / 2;
    let mut err2;

    loop {
        // Set pixels for thickness
        for i in 0..thickness as i64 {
            for j in 0..thickness as i64 {
                let x = x0 + i - (thickness as i64 / 2);
                let y = y0 + j - (thickness as i64 / 2);

                // check if the pixel is within the image bounds otherwise skip
                if x >= 0 && x < img.cols() as i64 && y >= 0 && y < img.rows() as i64 {
                    let pixel_linear_index = (y * img.cols() as i64 + x) * C as i64;
                    for (c, &color_channel) in color.iter().enumerate() {
                        // TODO: implement safe pixel access
                        img.as_slice_mut()[pixel_linear_index as usize + c] = color_channel;
                    }
                }
            }
        }

        // Check end condition
        if x0 == x1 && y0 == y1 {
            break;
        }

        // Store old error
        err2 = 2 * err;

        // Adjust error and start position
        if err2 > -dx {
            err -= dy;
            x0 += sx;
        }
        if err2 < dy {
            err += dx;
            y0 += sy;
        }
    }
}

/// Draws a polygon on an image inplace.
///
/// # Arguments
///
/// * `img` - The image to draw on.
/// * `points` - A slice of points representing the vertices of the polygon in order.
/// * `color` - The color of the polygon lines as an array of `C` elements.
/// * `thickness` - The thickness of the polygon lines.
///
pub fn draw_polygon<const C: usize, A: ImageAllocator>(
    img: &mut Image<u8, C, A>,
    points: &[(i64, i64)],
    color: [u8; C],
    thickness: usize,
) {
    if points.len() < 3 {
        return;
    }

    let num_points = points.len();

    for i in 0..num_points {
        let start = points[i];
        // mod to close the polygon
        let end = points[(i + 1) % num_points];
        draw_line(img, start, end, color, thickness);
    }
}

/// Draws a colored polygon on an image inplace.
///
/// Uses the scanline-fill algorithm.
///
/// # Arguments
///
/// * `img` - The image to draw on.
/// * `points` - A slice of points representing the vertices of the polygon inn order.
/// * `color` - The color filled in the polygon drawn.
/// * `line_color` - The color of the pen used to draw the polygon.
/// * `thickness` - The thickness of the polygon lines.
///
pub fn draw_filled_polygon<const C: usize, A: ImageAllocator>(
    img: &mut Image<u8, C, A>,
    points: &[(i64, i64)],
    fill_color: [u8; C],
    line_color: [u8; C],
    thickness: usize,
) {
    let n = points.len();
    if n < 3 {
        return;
    }

    // adding edges to edge list.
    let mut edges = Vec::with_capacity(n);
    let mut max_y = i64::MIN;
    let mut min_y = i64::MAX;
    for i in 0..n {
        let (x0, y0) = points[i];
        let (x1, y1) = points[(i + 1) % n];
        edges.push((x0, y0, x1, y1));

        // bounds for y to find range of scanlines
        if y0 < min_y {
            min_y = y0;
        }
        if y0 > max_y {
            max_y = y0;
        }
    }

    let mut x_intersections = Vec::with_capacity(n);

    // for each scanline
    for y in min_y..=max_y {
        x_intersections.clear();
        for &(x0, y0, x1, y1) in &edges {
            // checking if the scanline intersects the edges.
            if (y0 <= y && y1 > y) || (y1 <= y && y0 > y) {
                // finding the intersection with linear interpolation
                let x = x0 as f64
                    + (y as f64 - y0 as f64) * (x1 as f64 - x0 as f64) / (y1 as f64 - y0 as f64);
                x_intersections.push(x);
            }
        }

        // sort the intersections by x coordinate
        x_intersections.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // filling the region between intersections with color
        for pair in x_intersections.chunks(2) {
            if let [x_start, x_end] = pair {
                let xs = x_start.ceil() as i64;
                let xe = x_end.floor() as i64;
                for x in xs..=xe {
                    if x >= 0 && x < img.cols() as i64 && y >= 0 && y < img.rows() as i64 {
                        let base = (y * img.cols() as i64 + x) * C as i64;
                        let pixel_slice = &mut img.as_slice_mut()[base as usize..base as usize + C];
                        pixel_slice.copy_from_slice(&fill_color);
                    }
                }
            }
        }
    }

    // drawing the polygon outline
    draw_polygon(img, points, line_color, thickness);
}

#[cfg(test)]
mod tests {
    use super::{draw_filled_polygon, draw_line, draw_polygon};
    use kornia_image::{Image, ImageError, ImageSize};
    use kornia_tensor::CpuAllocator;

    #[test]
    fn test_draw_line() -> Result<(), ImageError> {
        let mut img = Image::new(
            ImageSize {
                width: 5,
                height: 5,
            },
            vec![0; 25],
            CpuAllocator,
        )?;

        draw_line(&mut img, (0, 0), (4, 4), [255], 1);

        #[rustfmt::skip]
        assert_eq!(
            img.as_slice(),
            vec![
                255, 0, 0, 0, 0,
                255, 255, 0, 0, 0,
                0, 255, 255, 0, 0,
                0, 0, 255, 255, 0,
                0, 0, 0, 255, 255
            ]
        );
        Ok(())
    }

    #[test]
    fn test_draw_polygon() -> Result<(), ImageError> {
        let mut img = Image::new(
            ImageSize {
                width: 5,
                height: 5,
            },
            vec![0; 25],
            CpuAllocator,
        )?;

        let points: [(i64, i64); 3] = [(0, 0), (0, 3), (4, 0)];
        draw_polygon(&mut img, &points, [255], 1);

        #[rustfmt::skip]
        assert_eq!(
            img.as_slice(),
            vec![
                255, 255, 255, 255, 255,
                255, 0, 0, 255, 0,
                255, 0, 255, 0, 0,
                255, 255, 0, 0, 0,
                0, 0, 0, 0, 0
            ]
        );
        Ok(())
    }

    #[test]
    fn test_draw_filled_polygon() -> Result<(), ImageError> {
        let mut img = Image::new(
            ImageSize {
                width: 5,
                height: 5,
            },
            vec![0; 25],
            CpuAllocator,
        )?;

        let points: [(i64, i64); 3] = [(0, 0), (0, 3), (4, 0)];
        draw_filled_polygon(&mut img, &points, [150], [255], 1);

        #[rustfmt::skip]
        assert_eq!(
            img.as_slice(),
            vec![
                255, 255, 255, 255, 255,
                255, 150, 150, 255, 0,
                255, 150, 255, 0, 0,
                255, 255, 0, 0, 0,
                0, 0, 0, 0, 0
            ]
        );
        Ok(())
    }

    #[test]
    fn test_draw_filled_single_pixel_width() -> Result<(), ImageError> {
        let mut img = Image::new(
            ImageSize {
                width: 5,
                height: 5,
            },
            vec![0; 25],
            CpuAllocator,
        )?;

        let points: [(i64, i64); 3] = [(0, 0), (3, 0), (4, 0)];
        draw_filled_polygon(&mut img, &points, [150], [255], 1);

        #[rustfmt::skip]
        assert_eq!(
            img.as_slice(),
            vec![
                255, 255, 255, 255, 255,
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0
            ]
        );
        Ok(())
    }
}
