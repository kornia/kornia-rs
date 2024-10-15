use kornia_image::Image;

/// Draws a line on an image inplace.
///
/// # Arguments
///
/// * `img` - The image to draw on.
/// * `p0` - The start point of the line as a tuple of (x, y).
/// * `p1` - The end point of the line as a tuple of (x, y).
/// * `color` - The color of the line as an array of `C` elements.
/// * `thickness` - The thickness of the line.
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

#[cfg(test)]
mod tests {
    use super::draw_line;
    use kornia_image::{Image, ImageError, ImageSize};

    #[rustfmt::skip]
    #[test]
    fn test_draw_line() -> Result<(), ImageError> {
        let mut img = Image::new(
            ImageSize {
                width: 5,
                height: 5,
            },
            vec![0; 25],
        )?;
        draw_line(&mut img, (0, 0), (4, 4), [255], 1);
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
}
