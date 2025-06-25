use kornia_image::ImageSize;

/// Calculates the total number of tiles needed to cover an image of the given size,
/// including partial tiles at the edges.
///
/// # Arguments
///
/// * `size` - The size of the image.
/// * `tile_size` - The size of each tile.
///
/// # Returns
///
/// A `Point2d` representing the number of tiles along the x and y axes.
pub(crate) fn find_total_tiles(size: ImageSize, tile_size: usize) -> Point2d {
    Point2d {
        x: size.width.div_ceil(tile_size),
        y: size.height.div_ceil(tile_size),
    }
}

/// Calculates the number of full tiles that fit within an image of the given size.
///
/// # Arguments
///
/// * `size` - The size of the image.
/// * `tile_size` - The size of each tile.
///
/// # Returns
///
/// A `Point2d` representing the number of full tiles along the x and y axes.
pub(crate) fn find_full_tiles(size: ImageSize, tile_size: usize) -> Point2d {
    Point2d {
        x: size.width / tile_size,
        y: size.height / tile_size,
    }
}

/// Represents a point in 2D space.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct Point2d {
    /// The x-coordinate of the point.
    pub x: usize,
    /// The y-coordinate of the point.
    pub y: usize,
}

#[derive(Default, Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
/// Represents a pixel that can be white, black, or skipped.
pub enum Pixel {
    /// A white pixel.
    White = 255,
    /// A black pixel.
    Black = 0,
    /// A pixel to be skipped.
    #[default]
    Skip = 127,
}

impl PartialEq<u8> for Pixel {
    fn eq(&self, other: &u8) -> bool {
        *self as u8 == *other
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pixel_value() {
        let white = Pixel::White;
        let black = Pixel::Black;
        let skip = Pixel::Skip;

        assert_eq!(white, 255);
        assert_eq!(black, 0);
        assert_eq!(skip, 127);
    }

    #[test]
    fn test_find_total_tiles() {
        let size = ImageSize {
            width: 101,
            height: 257,
        };

        let tiles = find_total_tiles(size, 4);
        let expected = Point2d { x: 26, y: 65 };

        assert_eq!(tiles, expected)
    }

    #[test]
    fn test_find_full_tiles() {
        let size = ImageSize {
            width: 101,
            height: 257,
        };

        let tiles = find_full_tiles(size, 4);
        let expected = Point2d { x: 25, y: 64 };

        assert_eq!(tiles, expected)
    }
}
