use crate::{
    errors::AprilTagError,
    utils::{find_full_tiles, find_total_tiles, Point2d},
};
use kornia_image::{allocator::ImageAllocator, Image, ImageSize};
use rayon::{
    iter::plumbing::{bridge, Producer, ProducerCallback},
    prelude::*,
};

/// Contains metadata and data for a single tile of an image.
///
/// `TileInfo` holds the position, indices, and a borrowed slice of pixel data for a tile.
/// The `data` field is a borrowed slice of rows, where each row is a slice of pixel data.
#[derive(Debug, Clone, PartialEq)]
pub struct TileInfo<'a, T> {
    /// The 2D position (x, y) of the tile in tile coordinates.
    pub pos: Point2d,
    /// The sequential index of the tile (including partial tiles).
    pub index: usize,
    /// The index among full (non-partial) tiles.
    pub full_index: usize,
    /// A borrowed slice of rows, where each row is a slice of pixel data.
    pub data: &'a [&'a [T]],
}

#[derive(Debug, Clone, PartialEq)]
/// Represents a tile of an image, which can be either a full-sized tile or a partial tile at the image edge.
///
/// Each variant contains a slice of rows, where each row is a slice of pixel data.
pub enum ImageTile<'a, T> {
    /// A full-sized tile with dimensions equal to the specified tile size.
    FullTile(TileInfo<'a, T>),
    /// A partial tile, typically at the image edge, with dimensions smaller than the specified tile size.
    PartialTile(TileInfo<'a, T>),
}

/// An iterator over tiles of an image, yielding non-overlapping rectangular regions as slices.
///
/// Each item yielded by the iterator is a slice of rows, where each row is a slice of pixel data.
/// The tile size is specified at construction, and the iterator will yield tiles of the given size,
/// except for tiles at the image edges, which may be smaller if the image dimensions are not
/// multiples of the tile size.
///
/// # SAFETY
///
/// The slice of row references (`&[&[T]]`) returned by each call to [`next()`](TileIterator::next)
/// is only valid until the next call to `next()` or until the iterator is dropped.
///
/// After calling `next()` again, or dropping the iterator, any previously returned slice
/// may point to invalid memory and must not be accessed. Do not store or use these slices
/// beyond the lifetime of the iterator or across calls to `next()`.
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
    buffer: Vec<&'a [T]>,
}

impl<'a, T> Clone for TileIterator<'a, T> {
    fn clone(&self) -> Self {
        Self {
            img_data: self.img_data,
            img_size: self.img_size,
            tile_size: self.tile_size,
            tiles_dim: self.tiles_dim,
            last_tile_px: self.last_tile_px,
            next_tile_index: self.next_tile_index,
            next_index: self.next_index,
            next_full_index: self.next_full_index,
            buffer: self.buffer.clone(),
        }
    }
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
            buffer: Vec::with_capacity(tile_size),
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

        self.buffer.clear();
        for y_px in 0..tile_y_px {
            let row = ((self.next_tile_index.y * self.tile_size) + y_px) * self.img_size.width;
            let start_index = row + (self.next_tile_index.x * self.tile_size);
            let end_index = start_index + tile_x_px;

            let row_pxs = &self.img_data[start_index..end_index];
            self.buffer.push(row_pxs);
        }

        let next_tile_index = self.next_tile_index;
        let index = self.next_index;

        // Update indices
        self.next_tile_index.x += 1;
        if self.next_tile_index.x >= self.tiles_dim.x {
            self.next_tile_index.x = 0;
            self.next_tile_index.y += 1;
        }

        self.next_index += 1;
        // TODO: Check the safety of this usage of from_raw_parts more deeply.
        let data = unsafe { std::slice::from_raw_parts(self.buffer.as_ptr(), tile_y_px) };

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

/// NOTE: The Image for TileIterator must have atleast 2 full sized tiles
pub struct ParTileIterator<'a, T> {
    base: TileIterator<'a, T>,
}

impl<'a, T: Sync> IntoParallelIterator for TileIterator<'a, T> {
    type Iter = ParTileIterator<'a, T>;

    type Item = ImageTile<'a, T>;

    fn into_par_iter(self) -> Self::Iter {
        ParTileIterator { base: self }
    }
}

impl<'a, T: Sync> ParallelIterator for ParTileIterator<'a, T> {
    type Item = ImageTile<'a, T>;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::UnindexedConsumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(self.len())
    }
}

impl<'a, T: Sync> IndexedParallelIterator for ParTileIterator<'a, T> {
    fn len(&self) -> usize {
        self.base.tiles_dim.x * self.base.tiles_dim.y
    }

    fn drive<C: rayon::iter::plumbing::Consumer<Self::Item>>(self, consumer: C) -> C::Result {
        bridge(self, consumer)
    }

    fn with_producer<CB: ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
        callback.callback(TileIteratorProducer {
            end_index: self.len(),
            full_tiles_dim: find_full_tiles(self.base.img_size, self.base.tile_size),
            base: self.base,
        })
    }
}

pub struct TileIteratorProducer<'a, T> {
    base: TileIterator<'a, T>,
    full_tiles_dim: Point2d,
    end_index: usize,
}

impl<'a, T> Clone for TileIteratorProducer<'a, T> {
    fn clone(&self) -> Self {
        Self {
            base: self.base.clone(),
            end_index: self.end_index,
            full_tiles_dim: self.full_tiles_dim,
        }
    }
}

impl<'a, T: Sync> Producer for TileIteratorProducer<'a, T> {
    type Item = ImageTile<'a, T>;

    type IntoIter = Self;

    fn into_iter(self) -> Self::IntoIter {
        self
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let mut left = self.clone();
        left.end_index = index;

        let right_tile_index = Point2d {
            x: index % self.base.tiles_dim.x,
            y: index / self.base.tiles_dim.x,
        };
        let mut right_tile_full_index = right_tile_index.y * self.full_tiles_dim.x;
        right_tile_full_index += right_tile_index.x.min(self.full_tiles_dim.x - 1);

        let mut right = self;
        right.base.next_index = index;
        right.base.next_tile_index = right_tile_index;
        right.base.next_full_index = right_tile_full_index;

        (left, right)
    }
}

impl<'a, T> ExactSizeIterator for TileIteratorProducer<'a, T> {}

impl<'a, T> DoubleEndedIterator for TileIteratorProducer<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        unimplemented!()
    }
}

impl<'a, T> Iterator for TileIteratorProducer<'a, T> {
    type Item = ImageTile<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.base.next_index >= self.end_index {
            return None;
        }
        self.base.next()
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
                let tile_row = tile_row as &[u8];
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
                assert_eq!(
                    iter.next()
                        .ok_or("Failed to get the next value from iterator")?,
                    ImageTile::$variant(TileInfo {
                        data: $rows,
                        pos: Point2d { x: $x, y: $y },
                        index: $index,
                        full_index: $full_index,
                    })
                );
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
