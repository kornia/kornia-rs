use kornia_image::{allocator::ImageAllocator, Image, ImageSize};

use crate::utils::Point2d;

/// TODO
#[derive(Debug, Default, Clone, Copy)]
pub struct TileIndex {
    /// TODO
    pub pos: Point2d,
    /// TODO
    pub index: usize,
    /// TODO
    pub full_index: usize,
}

/// TODO
pub type ImageSlice<'a, T> = &'a [&'a [T]];

/// TODO
#[derive(Debug, Clone)]
pub enum ImageTile<'a, T> {
    /// TODO
    FullTile(ImageSlice<'a, T>),
    /// TODO
    PartialTile(ImageSlice<'a, T>),
}

impl<'a, T> ImageTile<'a, T> {
    /// TODO
    pub fn inner(self) -> ImageSlice<'a, T> {
        match self {
            ImageTile::FullTile(im) => im,
            ImageTile::PartialTile(im) => im,
        }
    }
}

/// An enumerator over tiles of an image, yielding the `(y, x)` tile indices and the tile data as slices.
/// Each item is a tuple of `((tile_y, tile_x), tile)`, where tile is a slice of rows, and each row is a slice of pixel data.
pub struct TileEnumerator<'a, T>(TileIterator<'a, T>);

impl<'a, T> Iterator for TileEnumerator<'a, T> {
    type Item = (TileIndex, ImageTile<'a, T>);

    fn next(&mut self) -> Option<Self::Item> {
        let pos = Point2d {
            x: self.0.next_tile_x_index,
            y: self.0.next_tile_y_index,
        };

        let index = self.0.next_index;

        if let Some(item) = self.0.next() {
            let index = TileIndex {
                pos,
                index,
                full_index: self.0.next_full_index - 1,
            };
            return Some((index, item));
        }

        None
    }
}

/// An iterator over tiles of an image, yielding non-overlapping rectangular regions as slices.
///
/// Each item yielded by the iterator is a slice of rows, where each row is a slice of pixel data.
/// The tile size is specified at construction, and the iterator will yield tiles of the given size,
/// except for tiles at the image edges, which may be smaller if the image dimensions are not
/// multiples of the tile size.
pub struct TileIterator<'a, T> {
    /// Reference to the image pixel data as a flat slice.
    img_data: &'a [T],
    /// The width and height of the image.
    img_size: ImageSize,
    /// The size (width and height) of each tile in pixels.
    tile_size: usize,
    /// Number of horizontal tiles.
    tiles_x_len: usize,
    /// Number of vertical tiles.
    tiles_y_len: usize,
    /// Number of horizontal pixels in the last tile column.
    last_tile_x_px: usize,
    /// Number of vertical pixels in the last tile row.
    last_tile_y_px: usize,
    /// Index of the next tile to yield horizontally.
    next_tile_x_index: usize,
    /// Index of the next tile to yield vertically.
    next_tile_y_index: usize,
    /// TODO
    next_index: usize,
    /// TODO
    next_full_index: usize,
    /// Buffer holding references to the rows of the current tile.
    buffer: Vec<&'a [T]>,
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
    ) -> Self {
        let img_size = img.size();

        let tiles_x_len = (img_size.width as f32 / tile_size as f32).ceil() as usize;
        let tiles_y_len = (img_size.height as f32 / tile_size as f32).ceil() as usize;

        let last_tile_x_px = if img.width() % tile_size == 0 {
            tile_size
        } else {
            img_size.width % tile_size
        };

        let last_tile_y_px = if img.height() % tile_size == 0 {
            tile_size
        } else {
            img_size.height % tile_size
        };

        Self {
            img_data: img.as_slice(),
            img_size,
            tile_size,
            tiles_x_len,
            tiles_y_len,
            last_tile_x_px,
            last_tile_y_px,
            next_tile_x_index: 0,
            next_tile_y_index: 0,
            next_index: 0,
            next_full_index: 0,
            buffer: Vec::with_capacity(tile_size),
        }
    }

    /// Returns the number of tiles along the horizontal x-axis.
    ///
    /// This value represents how many tiles fit across the width of the image,
    /// given the specified tile size.
    #[inline]
    pub fn tiles_x_len(&self) -> usize {
        self.tiles_x_len
    }

    /// Returns the number of tiles along the vertical y-axis.
    ///
    /// This value represents how many tiles fit across the height of the image,
    /// given the specified tile size.
    #[inline]
    pub fn tiles_y_len(&self) -> usize {
        self.tiles_y_len
    }

    /// Returns an enumerator over the tiles, yielding both the tile indices and the tile data.
    ///
    /// The enumerator yields items of the form `((tile_y, tile_x), tile)`, where `tile` is a slice of rows,
    /// and each row is a slice of pixel data.
    #[inline]
    pub fn tile_enumerator(self) -> TileEnumerator<'a, T> {
        TileEnumerator(self)
    }
}

impl<'a, T> Iterator for TileIterator<'a, T> {
    type Item = ImageTile<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        // Stop iteration if we've processed all tiles
        if self.next_tile_y_index >= self.tiles_y_len {
            return None;
        }

        // number of horizontal pixels in the current tile
        let tile_x_px = if self.next_tile_x_index == self.tiles_x_len - 1 {
            self.last_tile_x_px
        } else {
            self.tile_size
        };

        // number of vertical pixels in the current tile
        let tile_y_px = if self.next_tile_y_index == self.tiles_y_len - 1 {
            self.last_tile_y_px
        } else {
            self.tile_size
        };

        self.buffer.clear();
        for y_px in 0..tile_y_px {
            let row = ((self.next_tile_y_index * self.tile_size) + y_px) * self.img_size.width;
            let start_index = row + (self.next_tile_x_index * self.tile_size);
            let end_index = start_index + tile_x_px;

            let row_pxs = &self.img_data[start_index..end_index];
            self.buffer.push(row_pxs);
        }

        // Update indices
        self.next_tile_x_index += 1;
        if self.next_tile_x_index >= self.tiles_x_len {
            self.next_tile_x_index = 0;
            self.next_tile_y_index += 1;
        }

        self.next_index += 1;

        let data = unsafe { std::slice::from_raw_parts(self.buffer.as_mut_ptr(), tile_y_px) };

        let tile = if data.len() == self.tile_size && data[0].len() == self.tile_size {
            self.next_full_index += 1;
            ImageTile::FullTile(data)
        } else {
            ImageTile::PartialTile(data)
        };

        Some(tile)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::{allocator::CpuAllocator, Image, ImageSize};

    #[test]
    fn test_tile_iterator() {
        let data = vec![127u8; 100];
        let image: Image<_, 1, _> = Image::new(
            ImageSize {
                width: 25,
                height: 4,
            },
            data,
            CpuAllocator,
        )
        .unwrap();

        let tile_iter = TileIterator::from_image(&image, 4);
        let mut counter = 0;

        for tile in tile_iter {
            for tile_row in tile.inner() {
                let tile_row = tile_row as &[u8];
                for px in tile_row {
                    assert_eq!(*px, 127);
                    counter += 1;
                }
            }
        }

        assert_eq!(counter, 100);
    }
}
