use crate::errors::AprilTagError;
use crate::iter::TileIterator;
use crate::utils::Pixel;
use kornia_image::{allocator::ImageAllocator, Image, ImageError, ImageSize};

/// Stores the minimum and maximum pixel values for each tile for [adaptive_threshold]
pub struct TileBuffers<T> {
    tile_min: Vec<T>,
    tile_max: Vec<T>,
    tile_size: usize,
}

impl<T: Default + PartialOrd + Copy> TileBuffers<T> {
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
        let tiles_x_len = (img_size.width as f32 / tile_size as f32).ceil() as usize;
        let tiles_y_len = (img_size.height as f32 / tile_size as f32).ceil() as usize;
        let num_tiles = tiles_x_len * tiles_y_len;

        Self {
            tile_min: vec![T::default(); num_tiles],
            tile_max: vec![T::default(); num_tiles],
            tile_size,
        }
    }

    /// Updates the provided `neighbor_min` and `neighbor_max` values with the minimum and maximum
    /// values from the specified neighboring tile.
    ///
    /// # Parameters
    ///
    /// - `neighbor_min`: Mutable reference to the current minimum value among neighbors.
    /// - `neighbor_max`: Mutable reference to the current maximum value among neighbors.
    /// - `neighbor_tile`: Tuple `(y, x)` specifying the coordinates of the neighboring tile.
    /// - `tiles_x_len`: The number of tiles along the x-axis (width) of the tile grid.
    ///
    /// This function compares the min and max values of the specified neighbor tile with the current
    /// `neighbor_min` and `neighbor_max`, updating them if the neighbor's values are more extreme.
    ///
    /// # Safety
    ///
    /// This function assumes that `neighbor_tile` is a valid tile coordinate and that the computed
    /// `neighbor_index` is within bounds of the `tile_min` and `tile_max` vectors. The caller must
    /// ensure that the provided indices are valid, otherwise this may panic or cause undefined behavior.
    fn neighbor_mix_max(
        &self,
        neighbor_min: &mut T,
        neighbor_max: &mut T,
        neighbor_tile: (usize, usize), // (y, x)
        tiles_x_len: usize,            // (y, x)
    ) {
        let neighbor_index = neighbor_tile.0 * tiles_x_len + neighbor_tile.1;

        if self.tile_min[neighbor_index] < *neighbor_min {
            *neighbor_min = self.tile_min[neighbor_index]
        }

        if self.tile_max[neighbor_index] > *neighbor_max {
            *neighbor_max = self.tile_max[neighbor_index];
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
/// - `tile_buffers`: A mutable reference to a [`TileBuffers`] struct used to store the minimum and maximum pixel values for
///   each tile. This buffer is filled during processing and reused across calls to avoid repeated allocations.
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
/// use kornia_apriltag::threshold::{adaptive_threshold, TileBuffers};
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
/// let mut tile_buffers = TileBuffers::new(src.size(), 2);
/// adaptive_threshold(&src, &mut dst, &mut tile_buffers, 20).unwrap();
/// assert_eq!(dst.as_slice(), &[0, 0, 0, 255, 255, 255]);
/// ```
// TODO: Add support for parallelism
pub fn adaptive_threshold<A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<u8, 1, A1>,
    dst: &mut Image<Pixel, 1, A2>,
    tile_buffers: &mut TileBuffers<u8>,
    min_white_black_diff: u8,
) -> Result<(), AprilTagError> {
    if src.size() != dst.size() {
        return Err(
            ImageError::InvalidImageSize(src.cols(), src.rows(), dst.cols(), dst.rows()).into(),
        );
    }

    let tile_iterator = TileIterator::from_image(src, tile_buffers.tile_size);
    let tiles_x_len = tile_iterator.tiles_x_len();
    let tiles_y_len = tile_iterator.tiles_y_len();

    let expected_tile_count = tiles_x_len * tiles_y_len;
    if tile_buffers.tile_min.len() != expected_tile_count {
        // It is guaranteed for tile_min and tile_max to have same length by design
        // so, avoiding additional check for tile_max
        return Err(AprilTagError::InvalidTileBufferSize(
            tile_buffers.tile_min.len(),
            expected_tile_count,
        ));
    }

    let dst_data = dst.as_slice_mut();

    // Calculate extrema (i.e. min & max grayscale value) of each tile
    for (i, tile) in tile_iterator.enumerate() {
        let mut local_min = 255;
        let mut local_max = 0;

        for row in tile {
            for px in row as &[u8] {
                if px < &local_min {
                    local_min = *px;
                }

                if px > &local_max {
                    local_max = *px;
                }
            }
        }

        tile_buffers.tile_min[i] = local_min;
        tile_buffers.tile_max[i] = local_max;
    }

    // Binarize the image
    for ((y, x), tile) in TileIterator::from_image(src, tile_buffers.tile_size).tile_enumerator() {
        let tile_index = (y * tiles_x_len) + x;

        let mut neighbor_min = tile_buffers.tile_min[tile_index];
        let mut neighbor_max = tile_buffers.tile_max[tile_index];

        // Low constrast tile, Skip processing
        if neighbor_max - neighbor_min < min_white_black_diff {
            for y_px in 0..tile.len() {
                let row = ((y * tile_buffers.tile_size) + y_px) * src.width();
                let start_index = row + (x * tile_buffers.tile_size);
                let end_index = start_index + tile[0].len();

                for px in dst_data.iter_mut().take(end_index).skip(start_index) {
                    *px = Pixel::Skip;
                }
            }

            continue;
        }

        if x + 1 != tiles_x_len {
            // Rightmost neighbor
            tile_buffers.neighbor_mix_max(
                &mut neighbor_min,
                &mut neighbor_max,
                (y, x + 1),
                tiles_x_len,
            );

            if y != 0 {
                // Top-right neighbor
                tile_buffers.neighbor_mix_max(
                    &mut neighbor_min,
                    &mut neighbor_max,
                    (y - 1, x + 1),
                    tiles_x_len,
                );
            }

            if y + 1 != tiles_y_len {
                // Bottom-right corner neighbor
                tile_buffers.neighbor_mix_max(
                    &mut neighbor_min,
                    &mut neighbor_max,
                    (y + 1, x + 1),
                    tiles_x_len,
                );
            }
        }

        if x != 0 {
            // Leftmost neighbor
            tile_buffers.neighbor_mix_max(
                &mut neighbor_min,
                &mut neighbor_max,
                (y, x - 1),
                tiles_x_len,
            );

            if y != 0 {
                // Top-left neighbor
                tile_buffers.neighbor_mix_max(
                    &mut neighbor_min,
                    &mut neighbor_max,
                    (y - 1, x - 1),
                    tiles_x_len,
                );
            }

            if x != 0 && y + 1 != tiles_y_len {
                // Bottom-left corner neighbor
                tile_buffers.neighbor_mix_max(
                    &mut neighbor_min,
                    &mut neighbor_max,
                    (y + 1, x - 1),
                    tiles_x_len,
                );
            }
        }

        if y + 1 != tiles_y_len {
            // Bottom-most neighbor
            tile_buffers.neighbor_mix_max(
                &mut neighbor_min,
                &mut neighbor_max,
                (y + 1, x),
                tiles_x_len,
            );
        }

        if y != 0 {
            // Uppermost neighbor exists
            tile_buffers.neighbor_mix_max(
                &mut neighbor_min,
                &mut neighbor_max,
                (y - 1, x),
                tiles_x_len,
            );
        }

        let thresh = neighbor_min + (neighbor_max - neighbor_min) / 2;

        for (y_px, row) in tile.iter().enumerate() {
            let row_index =
                ((y * tile_buffers.tile_size) + y_px) * src.width() + x * tile_buffers.tile_size;

            for (x_px, px) in (row as &[u8]).iter().enumerate() {
                dst_data[row_index + x_px] = if px > &thresh {
                    Pixel::White
                } else {
                    Pixel::Black
                };
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::{allocator::CpuAllocator, ImageSize};

    #[test]
    fn test_neighbor_min_max() {
        let tile_buffers = TileBuffers {
            tile_min: vec![10, 20, 30, 40],
            tile_max: vec![50, 60, 70, 80],
            tile_size: 2,
        };

        let mut neighbor_min = 25;
        let mut neighbor_max = 65;

        tile_buffers.neighbor_mix_max(&mut neighbor_min, &mut neighbor_max, (0, 1), 2);
        assert_eq!(neighbor_min, 20);
        assert_eq!(neighbor_max, 65);

        tile_buffers.neighbor_mix_max(&mut neighbor_min, &mut neighbor_max, (1, 0), 2);
        assert_eq!(neighbor_min, 20);
        assert_eq!(neighbor_max, 70);
    }

    #[test]
    fn test_adaptive_threshold_basic() {
        #[rustfmt::skip]
        let src = Image::new(
            ImageSize {
                width: 5,
                height: 5,
            },
            vec![
                0,   50,  100, 150, 200,
                250, 0,   50,  100, 150,
                200, 250, 0,   50,  100,
                150, 200, 250, 0,   50,
                100, 150, 200, 250, 0,
            ],
            CpuAllocator,
        )
        .unwrap();
        let mut dst = Image::from_size_val(src.size(), Pixel::Skip, CpuAllocator).unwrap();

        let mut tile_buffers = TileBuffers::new(src.size(), 2);
        adaptive_threshold(&src, &mut dst, &mut tile_buffers, 20).unwrap();

        #[rustfmt::skip]
        let expected = vec![
            0,   0,   0,   255, 255,
            255, 0,   0,   0,   255,
            255, 255, 0,   0,   0,
            255, 255, 255, 0,   0,
            0,   255, 255, 255, 127
        ];

        assert_eq!(dst.as_slice(), expected.as_slice());
    }

    #[test]
    fn test_adaptive_threshold_uniform_image() {
        let src = Image::new(
            ImageSize {
                width: 4,
                height: 4,
            },
            vec![100; 16],
            CpuAllocator,
        )
        .unwrap();
        let mut dst = Image::from_size_val(src.size(), Pixel::Skip, CpuAllocator).unwrap();

        let mut tile_buffers = TileBuffers::new(src.size(), 2);
        adaptive_threshold(&src, &mut dst, &mut tile_buffers, 20).unwrap();
        assert_eq!(dst.as_slice(), &[Pixel::Skip; 16]);
    }

    #[test]
    fn invalid_buffer_size() {
        let src = Image::new(
            ImageSize {
                width: 4,
                height: 4,
            },
            vec![100u8; 16],
            CpuAllocator,
        )
        .unwrap();

        let mut dst = Image::from_size_val(
            ImageSize {
                width: 4,
                height: 4,
            },
            Pixel::default(),
            CpuAllocator,
        )
        .unwrap();

        let mut tile_buffers = TileBuffers::new(
            ImageSize {
                width: 3,
                height: 2,
            },
            2,
        );
        let result = adaptive_threshold(&src, &mut dst, &mut tile_buffers, 20);

        assert!(matches!(
            result,
            Err(AprilTagError::InvalidTileBufferSize(2, 4))
        ));
    }
}
