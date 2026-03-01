use crate::{
    errors::AprilTagError,
    utils::{find_total_tiles, Point2d},
};
use kornia_image::{allocator::ImageAllocator, Image, ImageSize};

/// A lightweight, zero-allocation view into a rectangular sub-region of image data.
///
/// `TileView` borrows the underlying flat image slice and computes row slices
/// on-the-fly, eliminating the need for any intermediate buffer or unsafe code.
pub struct TileView<'a, T> {
    img_data: &'a [T],
    img_width: usize,
    tile_x_start: usize,
    tile_y_start: usize,
    tile_width: usize,
    tile_height: usize,
}

// Manual Copy/Clone to avoid requiring T: Copy/Clone (all fields are Copy regardless of T).
impl<'a, T> Copy for TileView<'a, T> {}
impl<'a, T> Clone for TileView<'a, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T> TileView<'a, T> {
    /// Returns the number of rows (height) of this tile.
    pub fn len(&self) -> usize {
        self.tile_height
    }

    /// Returns `true` if the tile has zero rows.
    pub fn is_empty(&self) -> bool {
        self.tile_height == 0
    }

    /// Returns the row at the given y-offset within the tile.
    pub fn row(&self, y: usize) -> &'a [T] {
        let start = (self.tile_y_start + y) * self.img_width + self.tile_x_start;
        &self.img_data[start..start + self.tile_width]
    }

    /// Returns an iterator over the rows of this tile.
    pub fn iter(&self) -> TileViewRows<'a, T> {
        TileViewRows {
            view: *self,
            current_row: 0,
        }
    }
}

impl<'a, T> std::ops::Index<usize> for TileView<'a, T> {
    type Output = [T];

    fn index(&self, y: usize) -> &Self::Output {
        self.row(y)
    }
}

impl<'a, T: PartialEq> PartialEq for TileView<'a, T> {
    fn eq(&self, other: &Self) -> bool {
        if self.tile_height != other.tile_height || self.tile_width != other.tile_width {
            return false;
        }
        (0..self.tile_height).all(|y| self.row(y) == other.row(y))
    }
}

impl<'a, T: PartialEq> PartialEq<&[&[T]]> for TileView<'a, T> {
    fn eq(&self, other: &&[&[T]]) -> bool {
        if self.tile_height != other.len() {
            return false;
        }
        (0..self.tile_height).all(|y| self.row(y) == other[y])
    }
}

impl<'a, T: std::fmt::Debug> std::fmt::Debug for TileView<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let rows: Vec<&[T]> = (0..self.tile_height).map(|y| self.row(y)).collect();
        f.debug_list().entries(rows.iter()).finish()
    }
}

/// An iterator over the rows of a [`TileView`].
pub struct TileViewRows<'a, T> {
    view: TileView<'a, T>,
    current_row: usize,
}

impl<'a, T> Iterator for TileViewRows<'a, T> {
    type Item = &'a [T];

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_row >= self.view.tile_height {
            return None;
        }
        let row = self.view.row(self.current_row);
        self.current_row += 1;
        Some(row)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.view.tile_height - self.current_row;
        (remaining, Some(remaining))
    }
}

impl<'a, T> ExactSizeIterator for TileViewRows<'a, T> {}

impl<'a, T> IntoIterator for TileView<'a, T> {
    type Item = &'a [T];
    type IntoIter = TileViewRows<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        TileViewRows {
            view: self,
            current_row: 0,
        }
    }
}

impl<'a, T> IntoIterator for &TileView<'a, T> {
    type Item = &'a [T];
    type IntoIter = TileViewRows<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Contains metadata and data for a single tile of an image.
///
/// `TileInfo` holds the position, indices, and a [`TileView`] that provides
/// zero-allocation access to the tile's pixel data.
#[derive(Debug, Clone, PartialEq)]
pub struct TileInfo<'a, T> {
    /// The 2D position (x, y) of the tile in tile coordinates.
    pub pos: Point2d,
    /// The sequential index of the tile (including partial tiles).
    pub index: usize,
    /// The index among full (non-partial) tiles.
    pub full_index: usize,
    /// A zero-allocation view into the tile's pixel data.
    pub data: TileView<'a, T>,
}

/// Represents a tile of an image, which can be either a full-sized tile or a partial tile at the image edge.
#[derive(Debug, Clone, PartialEq)]
pub enum ImageTile<'a, T> {
    /// A full-sized tile with dimensions equal to the specified tile size.
    FullTile(TileInfo<'a, T>),
    /// A partial tile, typically at the image edge, with dimensions smaller than the specified tile size.
    PartialTile(TileInfo<'a, T>),
}

/// An iterator over tiles of an image, yielding non-overlapping rectangular regions.
///
/// Each item yielded by the iterator contains a [`TileView`] that computes row slices
/// on-the-fly from the underlying image data, with zero intermediate allocation.
/// The tile size is specified at construction, and the iterator will yield tiles of the given size,
/// except for tiles at the image edges, which may be smaller if the image dimensions are not
/// multiples of the tile size.
pub struct TileIterator<'a, T> {
    img_data: &'a [T],
    img_size: ImageSize,
    tile_size: usize,
    tiles_dim: Point2d,
    last_tile_px: Point2d,
    next_tile_index: Point2d,
    /// The index of the next tile to be yielded by the iterator (counts all tiles, including partial ones).
    next_index: usize,
    /// The index of the next full (non-partial) tile to be yielded by the iterator.
    next_full_index: usize,
}

impl<'a, T> TileIterator<'a, T> {
    /// Creates a new `TileIterator` from a reference to an image and the desired tile size.
    ///
    /// # Arguments
    ///
    /// * `img` - A reference to the image to be tiled.
    /// * `tile_size` - The width and height of each tile in pixels.
    ///
    /// # Returns
    ///
    /// Returns a `TileIterator` that yields non-overlapping tiles of the specified size from the image.
    /// Tiles at the image edges may be smaller if the image dimensions are not multiples of the tile size.
    pub fn from_image<const C: usize, A: ImageAllocator>(
        img: &'a Image<T, C, A>,
        tile_size: usize,
    ) -> Result<Self, AprilTagError> {
        let img_size = img.size();
        if img.width() < tile_size || img.height() < tile_size {
            return Err(AprilTagError::InvalidImageSize);
        }

        let tiles_len = find_total_tiles(img_size, tile_size);
        let last_tile_px = Point2d {
            x: if img.width() % tile_size == 0 {
                tile_size
            } else {
                img_size.width % tile_size
            },
            y: if img.height() % tile_size == 0 {
                tile_size
            } else {
                img_size.height % tile_size
            },
        };

        Ok(Self {
            img_data: img.as_slice(),
            img_size,
            tile_size,
            tiles_dim: tiles_len,
            last_tile_px,
            next_tile_index: Point2d::default(),
            next_index: 0,
            next_full_index: 0,
        })
    }
}

impl<'a, T> Iterator for TileIterator<'a, T> {
    type Item = ImageTile<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        // Stop iteration if we've processed all tiles
        if self.next_tile_index.y >= self.tiles_dim.y {
            return None;
        }

        // number of horizontal pixels in the current tile
        let tile_x_px = if self.next_tile_index.x == self.tiles_dim.x - 1 {
            self.last_tile_px.x
        } else {
            self.tile_size
        };

        // number of vertical pixels in the current tile
        let tile_y_px = if self.next_tile_index.y == self.tiles_dim.y - 1 {
            self.last_tile_px.y
        } else {
            self.tile_size
        };

        let data = TileView {
            img_data: self.img_data,
            img_width: self.img_size.width,
            tile_x_start: self.next_tile_index.x * self.tile_size,
            tile_y_start: self.next_tile_index.y * self.tile_size,
            tile_width: tile_x_px,
            tile_height: tile_y_px,
        };

        let next_tile_index = self.next_tile_index;
        let index = self.next_index;

        // Update indices
        self.next_tile_index.x += 1;
        if self.next_tile_index.x >= self.tiles_dim.x {
            self.next_tile_index.x = 0;
            self.next_tile_index.y += 1;
        }

        self.next_index += 1;

        let tile = if data.len() == self.tile_size && data[0].len() == self.tile_size {
            self.next_full_index += 1;
            ImageTile::FullTile(TileInfo {
                data,
                pos: next_tile_index,
                index,
                full_index: self.next_full_index - 1,
            })
        } else {
            ImageTile::PartialTile(TileInfo {
                data,
                pos: next_tile_index,
                index,
                full_index: self.next_full_index - 1,
            })
        };

        Some(tile)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::{allocator::CpuAllocator, Image, ImageSize};

    #[test]
    fn test_tile_iterator_basic() -> Result<(), Box<dyn std::error::Error>> {
        let data = vec![127u8; 100];
        let image: Image<_, 1, _> = Image::new(
            ImageSize {
                width: 25,
                height: 4,
            },
            data,
            CpuAllocator,
        )?;

        let tile_iter = TileIterator::from_image(&image, 4)?;
        let mut counter = 0;

        for tile in tile_iter {
            let tile = match tile {
                ImageTile::FullTile(tile) => tile,
                ImageTile::PartialTile(tile) => tile,
            };

            for tile_row in tile.data {
                for px in tile_row {
                    assert_eq!(*px, 127);
                    counter += 1;
                }
            }
        }

        assert_eq!(counter, 100);

        Ok(())
    }

    #[test]
    fn test_tile_iterator() -> Result<(), Box<dyn std::error::Error>> {
        #[rustfmt::skip]
        let data = vec![
            1,  2,  3,  4,  5,
            6,  7,  8,  9,  10,
            11, 12, 13, 14, 15,
            16, 17, 18, 19, 20,
            21, 22, 23, 24, 25,
            26, 27, 28, 29, 30,
            31, 32, 33, 34, 35,
        ];
        let img: Image<_, 1, _> = Image::new(
            ImageSize {
                width: 5,
                height: 7,
            },
            data,
            CpuAllocator,
        )?;

        let mut iter = TileIterator::from_image(&img, 2)?;

        macro_rules! test_iter_next {
            ($variant:ident, $x:expr, $y:expr, $index:expr, $full_index:expr, $rows:expr) => {
                let next = iter
                    .next()
                    .ok_or("Failed to get the next value from iterator")?;
                match next {
                    ImageTile::$variant(ref info) => {
                        assert_eq!(info.pos, Point2d { x: $x, y: $y });
                        assert_eq!(info.index, $index);
                        assert_eq!(info.full_index, $full_index);
                        let expected: &[&[u8]] = $rows;
                        assert_eq!(info.data, expected);
                    }
                    _ => panic!(
                        "Expected ImageTile::{}, got {:?}",
                        stringify!($variant),
                        next
                    ),
                }
            };
        }

        #[rustfmt::skip]
        {
            //              Tile Type    x  y  i  fi  expected value
            test_iter_next!(FullTile,    0, 0, 0,  0, &[&[1, 2], &[6, 7]]);
            test_iter_next!(FullTile,    1, 0, 1,  1, &[&[3, 4], &[8, 9]]);
            test_iter_next!(PartialTile, 2, 0, 2,  1, &[&[5], &[10]]);
            test_iter_next!(FullTile,    0, 1, 3,  2, &[&[11, 12], &[16, 17]]);
            test_iter_next!(FullTile,    1, 1, 4,  3, &[&[13, 14], &[18, 19]]);
            test_iter_next!(PartialTile, 2, 1, 5,  3, &[&[15], &[20]]);
            test_iter_next!(FullTile,    0, 2, 6,  4, &[&[21, 22], &[26, 27]]);
            test_iter_next!(FullTile,    1, 2, 7,  5, &[&[23, 24], &[28, 29]]);
            test_iter_next!(PartialTile, 2, 2, 8,  5, &[&[25], &[30]]);
            test_iter_next!(PartialTile, 0, 3, 9,  5, &[&[31, 32]]);
            test_iter_next!(PartialTile, 1, 3, 10, 5, &[&[33, 34]]);
            test_iter_next!(PartialTile, 2, 3, 11, 5, &[&[35]]);
        };
        assert_eq!(iter.next(), None);

        Ok(())
    }

    #[test]
    fn test_invalid_image() -> Result<(), Box<dyn std::error::Error>> {
        let img: Image<_, 1, _> = Image::from_size_val(
            ImageSize {
                width: 3,
                height: 4,
            },
            0,
            CpuAllocator,
        )?;

        let tile_iterator = TileIterator::from_image(&img, 4);
        matches!(tile_iterator, Err(AprilTagError::InvalidImageSize));

        Ok(())
    }
}
