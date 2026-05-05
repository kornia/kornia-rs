use crate::iter::ImageTile;
use crate::utils::{find_full_tiles, Pixel, Point2d};
use crate::{errors::AprilTagError, iter::TileIterator};
use kornia_image::{allocator::ImageAllocator, Image, ImageError, ImageSize};

/// Stores the minimum and maximum pixel values for each tile for [adaptive_threshold]
///
/// The tiles are indexed in row-major order, i.e., tile IDs increase first along the x-axis (columns),
/// then along the y-axis (rows) only for the full tiles.
pub struct TileMinMax {
    min: Vec<u8>,
    max: Vec<u8>,
    tile_size: usize,
}

impl TileMinMax {
    /// Creates a new `TileBuffers` with capacity based on the image size and tile size.
    ///
    /// # Parameters
    ///
    /// - `img_size`: The size of the image.
    /// - `tile_size`: The size of each tile.
    ///
    /// # Returns
    ///
    /// A `TileBuffers` instance with preallocated capacity for tile minima and maxima.
    pub fn new(img_size: ImageSize, tile_size: usize) -> Self {
        let tiles_full = find_full_tiles(img_size, tile_size);
        let num_tiles = tiles_full.x * tiles_full.y;

        Self {
            min: vec![0; num_tiles],
            max: vec![0; num_tiles],
            tile_size,
        }
    }

    fn neighbor_blur(
        &self,
        current_tile: Point2d,
        current_index: usize,
        tiles_len: Point2d,
    ) -> (u8, u8) {
        let mut neighbor_min = self.min[current_index];
        let mut neighbor_max = self.max[current_index];

        if current_tile.y > 0 {
            // Uppermost tile
            self.neighbor_min_max(
                &mut neighbor_min,
                &mut neighbor_max,
                current_index - tiles_len.x,
            );

            if current_tile.x < tiles_len.x - 1 {
                // Upper right tile
                self.neighbor_min_max(
                    &mut neighbor_min,
                    &mut neighbor_max,
                    current_index - tiles_len.x + 1,
                );
            }

            if current_tile.x > 0 {
                // Upper left tile
                self.neighbor_min_max(
                    &mut neighbor_min,
                    &mut neighbor_max,
                    current_index - tiles_len.x - 1,
                );
            }
        }

        if current_tile.y < tiles_len.y - 1 {
            // Bottom tile
            self.neighbor_min_max(
                &mut neighbor_min,
                &mut neighbor_max,
                current_index + tiles_len.x,
            );

            if current_tile.x < tiles_len.x - 1 {
                // Bottom right tile
                self.neighbor_min_max(
                    &mut neighbor_min,
                    &mut neighbor_max,
                    current_index + tiles_len.x + 1,
                );
            }

            if current_tile.x > 0 {
                // Bottom left tile
                self.neighbor_min_max(
                    &mut neighbor_min,
                    &mut neighbor_max,
                    current_index + tiles_len.x - 1,
                );
            }
        }

        if current_tile.x < tiles_len.x - 1 {
            // Right tile
            self.neighbor_min_max(&mut neighbor_min, &mut neighbor_max, current_index + 1);
        }

        if current_tile.x > 0 {
            // Left tile
            self.neighbor_min_max(&mut neighbor_min, &mut neighbor_max, current_index - 1);
        }

        (neighbor_min, neighbor_max)
    }

    fn neighbor_min_max(
        &self,
        neighbor_min: &mut u8,
        neighbor_max: &mut u8,
        neighbor_index: usize,
    ) {
        if self.min[neighbor_index] < *neighbor_min {
            *neighbor_min = self.min[neighbor_index]
        }

        if self.max[neighbor_index] > *neighbor_max {
            *neighbor_max = self.max[neighbor_index];
        }
    }
}

/// Applies an adaptive thresholding algorithm to binarize an image.
///
/// Adaptive thresholding divides the image into smaller tiles and computes a local threshold
/// for each tile based on the minimum and maximum pixel values within the tile. This allows
/// the algorithm to handle varying lighting conditions across the image.
///
/// # Parameters
///
/// - `src`: The source grayscale image.
/// - `dst`: The destination image where the binarized result will be stored. Must have the same size as `src`.
/// - `tile_min_max`: A mutable reference to a [`TileMinMax`] struct used to store the minimum and maximum pixel
///   values for each tile. This buffer is filled during processing and reused across calls to avoid repeated
///   allocations.
/// - `min_white_black_diff`: The minimum difference between the maximum and minimum pixel values in a tile
///   for it to be considered for thresholding. Tiles with lower contrast are skipped.
///
/// # Returns
///
/// - `Ok(())` if the operation is successful.
/// - `Err(AprilTagError)` if the source and destination images have different sizes.
///
/// # Behavior
///
/// 1. The image is divided into tiles of size `tile_size x tile_size`.
/// 2. For each tile:
///    - The minimum (`local_min`) and maximum (`local_max`) pixel values are computed.
///    - If the difference between `local_max` and `local_min` is less than `min_white_black_diff`,
///      the tile is skipped, and its pixels are marked as [Pixel::Skip].
///    - Otherwise, the threshold for the tile is computed as:
///      `threshold = local_min + (local_max - local_min) / 2`.
///    - Pixels in the tile are binarized based on whether they are above or below the threshold.
/// 3. Neighboring tiles are considered to refine the threshold for each tile, ensuring smooth transitions.
///
/// # Examples
///
/// ```
/// use kornia_image::{allocator::CpuAllocator, Image, ImageSize};
/// use kornia_apriltag::threshold::{adaptive_threshold, TileMinMax};
/// use kornia_apriltag::utils::Pixel;
///
/// let src = Image::new(
///     ImageSize {
///         width: 2,
///         height: 3,
///     },
///     vec![0, 50, 100, 150, 200, 250],
///     CpuAllocator,
/// )
/// .unwrap();
/// let mut dst = Image::from_size_val(src.size(), Pixel::Skip, CpuAllocator).unwrap();
///
/// let mut tile_buffers = TileMinMax::new(src.size(), 2);
/// adaptive_threshold(&src, &mut dst, &mut tile_buffers, 20).unwrap();
/// assert_eq!(dst.as_slice(), &[0, 0, 255, 255, 255, 255]);
/// ```
// TODO: Add support for parallelism
pub fn adaptive_threshold<A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<u8, 1, A1>,
    dst: &mut Image<Pixel, 1, A2>,
    tile_min_max: &mut TileMinMax,
    min_white_black_diff: u8,
) -> Result<(), AprilTagError> {
    if src.size() != dst.size() {
        return Err(
            ImageError::InvalidImageSize(src.cols(), src.rows(), dst.cols(), dst.rows()).into(),
        );
    }

    if src.width() < tile_min_max.tile_size || src.height() < tile_min_max.tile_size {
        return Err(AprilTagError::InvalidImageSize);
    }

    let tile_iterator = TileIterator::from_image(src, tile_min_max.tile_size)?;
    let tiles_full_len = find_full_tiles(src.size(), tile_min_max.tile_size);

    if tile_min_max.min.len() != tiles_full_len.x * tiles_full_len.y {
        // It is guaranteed for tile_min and tile_max to have same length by design
        // so, avoiding additional check for tile_max
        return Err(AprilTagError::ImageTileSizeMismatch);
    }

    let dst_data = dst.as_slice_mut();

    // Calculate extrema (i.e. min & max grayscale value) of each tile
    tile_iterator.for_each(|tile| {
        let ImageTile::FullTile(tile) = tile else {
            // Skip non-full tiles
            return;
        };

        let mut local_min = 255;
        let mut local_max = 0;

        for row in tile.data {
            for px in row {
                if px < &local_min {
                    local_min = *px;
                }

                if px > &local_max {
                    local_max = *px;
                }
            }
        }

        tile_min_max.min[tile.full_index] = local_min;
        tile_min_max.max[tile.full_index] = local_max;
    });

    // Binarize the image
    TileIterator::from_image(src, tile_min_max.tile_size)?.for_each(|tile| {
        let (neighbor_min, neighbor_max, tile) = match tile {
            ImageTile::FullTile(tile) => {
                let (min, max) =
                    tile_min_max.neighbor_blur(tile.pos, tile.full_index, tiles_full_len);

                if max - min < min_white_black_diff {
                    for y_px in 0..tile.data.len() {
                        let row = ((tile.pos.y * tile_min_max.tile_size) + y_px) * src.width();
                        let start_index = row + (tile.pos.x * tile_min_max.tile_size);
                        let end_index = start_index + tile.data[0].len();

                        for px in dst_data.iter_mut().take(end_index).skip(start_index) {
                            *px = Pixel::Skip;
                        }
                    }

                    return;
                }

                (min, max, tile)
            }
            ImageTile::PartialTile(tile) => {
                let is_partial_y = tile.data.len() < tile_min_max.tile_size;
                let is_partial_x = tile.data[0].len() < tile_min_max.tile_size;

                let (pos, full_index) = if is_partial_y && is_partial_x {
                    (
                        Point2d {
                            x: tile.pos.x - 1,
                            y: tile.pos.y - 1,
                        },
                        tile.full_index,
                    )
                } else if is_partial_x {
                    (
                        Point2d {
                            x: tile.pos.x - 1,
                            y: tile.pos.y,
                        },
                        tile.full_index,
                    )
                } else {
                    (
                        Point2d {
                            x: tile.pos.x,
                            y: tile.pos.y - 1,
                        },
                        tile.full_index + tile.pos.x + 1 - tiles_full_len.x,
                    )
                };

                let (min, max) = tile_min_max.neighbor_blur(pos, full_index, tiles_full_len);
                (min, max, tile)
            }
        };

        let thresh = neighbor_min + (neighbor_max - neighbor_min) / 2;

        for (y_px, row) in tile.data.iter().enumerate() {
            let row_index = ((tile.pos.y * tile_min_max.tile_size) + y_px) * src.width()
                + tile.pos.x * tile_min_max.tile_size;

            for (x_px, px) in row.iter().enumerate() {
                dst_data[row_index + x_px] = if px > &thresh {
                    Pixel::White
                } else {
                    Pixel::Black
                };
            }
        }
    });

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::{allocator::CpuAllocator, ImageSize};
    use kornia_io::png::read_image_png_mono8;

    #[test]
    fn test_neighbor_min_max() {
        let tile_buffers = TileMinMax {
            min: vec![10, 20, 30, 40],
            max: vec![50, 60, 70, 80],
            tile_size: 2,
        };

        let mut neighbor_min = 25;
        let mut neighbor_max = 65;

        tile_buffers.neighbor_min_max(&mut neighbor_min, &mut neighbor_max, 1);
        assert_eq!(neighbor_min, 20);
        assert_eq!(neighbor_max, 65);

        tile_buffers.neighbor_min_max(&mut neighbor_min, &mut neighbor_max, 2);
        assert_eq!(neighbor_min, 20);
        assert_eq!(neighbor_max, 70);
    }

    #[test]
    fn test_adaptive_threshold_basic() -> Result<(), Box<dyn std::error::Error>> {
        #[rustfmt::skip]
        let src = Image::new(
            ImageSize {
                width: 5,
                height: 6,
            },
            vec![
                0,   50,  100, 150, 200,
                250, 0,   50,  100, 150,
                200, 250, 0,   50,  100,
                150, 200, 250, 0,   50,
                100, 150, 200, 250, 0,
                80,  127, 221, 20,  100,
            ],
            CpuAllocator,
        )?;
        let mut dst = Image::from_size_val(src.size(), Pixel::Skip, CpuAllocator)?;

        let mut tile_buffers = TileMinMax::new(src.size(), 2);
        adaptive_threshold(&src, &mut dst, &mut tile_buffers, 20)?;

        #[rustfmt::skip]
        let expected = vec![
            0,   0,   0,   255, 255,
            255, 0,   0,   0,   255,
            255, 255, 0,   0,   0,
            255, 255, 255, 0,   0,
            0,   255, 255, 255, 0,
            0,   255, 255, 0,   0
        ];

        assert_eq!(dst.as_slice(), expected.as_slice());
        Ok(())
    }

    #[test]
    fn test_adaptive_threshold_uniform_image() -> Result<(), Box<dyn std::error::Error>> {
        let src = Image::new(
            ImageSize {
                width: 4,
                height: 4,
            },
            vec![100; 16],
            CpuAllocator,
        )?;
        let mut dst = Image::from_size_val(src.size(), Pixel::Skip, CpuAllocator)?;

        let mut tile_buffers = TileMinMax::new(src.size(), 2);
        adaptive_threshold(&src, &mut dst, &mut tile_buffers, 20)?;
        assert_eq!(dst.as_slice(), &[Pixel::Skip; 16]);
        Ok(())
    }

    #[test]
    fn test_adaptive_threshold_synthetic_image() -> Result<(), Box<dyn std::error::Error>> {
        let src = read_image_png_mono8("../../tests/data/apriltag.png")?;
        let mut bin = Image::from_size_val(src.size(), Pixel::Skip, CpuAllocator)?;

        let mut tile_buffers = TileMinMax::new(src.size(), 4);
        adaptive_threshold(&src, &mut bin, &mut tile_buffers, 20)?;

        assert_eq!(bin.as_slice(), src.as_slice());
        Ok(())
    }

    #[test]
    fn invalid_buffer_size() -> Result<(), Box<dyn std::error::Error>> {
        let img_size = ImageSize {
            width: 4,
            height: 4,
        };

        let src = Image::new(img_size, vec![100u8; 16], CpuAllocator)?;

        let mut dst = Image::from_size_val(img_size, Pixel::default(), CpuAllocator)?;

        let mut tile_buffers = TileMinMax::new(
            ImageSize {
                width: 3,
                height: 2,
            },
            2,
        );
        let result = adaptive_threshold(&src, &mut dst, &mut tile_buffers, 20);
        assert!(matches!(result, Err(AprilTagError::ImageTileSizeMismatch)));

        let mut tile_buffers = TileMinMax::new(src.size(), 5);
        let result = adaptive_threshold(&src, &mut dst, &mut tile_buffers, 20);
        assert!(matches!(result, Err(AprilTagError::InvalidImageSize)));

        Ok(())
    }
}
