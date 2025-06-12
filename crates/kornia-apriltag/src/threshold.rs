use crate::utils::Pixel;
use crate::{errors::AprilTagError, iter::TileIterator};
use kornia_image::{allocator::ImageAllocator, Image, ImageError, ImageSize};

/// Stores the minimum and maximum pixel values for each tile for [adaptive_threshold]
pub struct TileBuffers {
    tile_min: Vec<u8>,
    tile_max: Vec<u8>,
    tile_size: usize,
}

impl TileBuffers {
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
        let tiles_x_len = img_size.width / tile_size;
        let tiles_y_len = img_size.height / tile_size;
        let num_tiles = tiles_x_len * tiles_y_len;

        Self {
            tile_min: vec![0; num_tiles],
            tile_max: vec![0; num_tiles],
            tile_size,
        }
    }

    fn neighbor_blur(&self, current_tile: (usize, usize), tiles_len: (usize, usize)) -> (u8, u8) {
        let (tiles_y_len, tiles_x_len) = tiles_len;
        let (y, x) = current_tile;
        let index = y * tiles_x_len + x;

        let mut neighbor_min = self.tile_min[index];
        let mut neighbor_max = self.tile_max[index];

        if y != 0 {
            // Uppermost tile
            self.neighbor_min_max(
                &mut neighbor_min,
                &mut neighbor_max,
                (y - 1, x),
                tiles_x_len,
            );

            if x + 1 != tiles_x_len {
                // Upper right tile
                self.neighbor_min_max(
                    &mut neighbor_min,
                    &mut neighbor_max,
                    (y - 1, x + 1),
                    tiles_x_len,
                );
            }

            if x != 0 {
                // Upper left tile
                self.neighbor_min_max(
                    &mut neighbor_min,
                    &mut neighbor_max,
                    (y - 1, x - 1),
                    tiles_x_len,
                );
            }
        }

        if y + 1 != tiles_y_len {
            // Bottom tile
            self.neighbor_min_max(
                &mut neighbor_min,
                &mut neighbor_max,
                (y + 1, x),
                tiles_x_len,
            );

            if x + 1 != tiles_x_len {
                // Bottom right tile
                self.neighbor_min_max(
                    &mut neighbor_min,
                    &mut neighbor_max,
                    (y + 1, x + 1),
                    tiles_x_len,
                );
            }

            if x != 0 {
                // Bottom left tile
                self.neighbor_min_max(
                    &mut neighbor_min,
                    &mut neighbor_max,
                    (y + 1, x - 1),
                    tiles_x_len,
                );
            }
        }

        if x + 1 != tiles_x_len {
            // Right tile
            self.neighbor_min_max(
                &mut neighbor_min,
                &mut neighbor_max,
                (y, x + 1),
                tiles_x_len,
            );
        }

        if x != 0 {
            // Left tile
            self.neighbor_min_max(
                &mut neighbor_min,
                &mut neighbor_max,
                (y, x - 1),
                tiles_x_len,
            );
        }

        (neighbor_min, neighbor_max)
    }

    fn neighbor_min_max(
        &self,
        neighbor_min: &mut u8,
        neighbor_max: &mut u8,
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
/// assert_eq!(dst.as_slice(), &[0, 0, 255, 255, 255, 255]);
/// ```
// TODO: Add support for parallelism
pub fn adaptive_threshold<A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<u8, 1, A1>,
    dst: &mut Image<Pixel, 1, A2>,
    tile_buffers: &mut TileBuffers,
    min_white_black_diff: u8,
) -> Result<(), AprilTagError> {
    if src.size() != dst.size() {
        return Err(
            ImageError::InvalidImageSize(src.cols(), src.rows(), dst.cols(), dst.rows()).into(),
        );
    }

    if src.width() < tile_buffers.tile_size || src.height() < tile_buffers.tile_size {
        return Err(AprilTagError::InvalidImageSize);
    }

    let tile_iterator = TileIterator::from_image(src, tile_buffers.tile_size);

    let tiles_full_x_len = src.width() / tile_buffers.tile_size;
    let tiles_full_y_len = src.height() / tile_buffers.tile_size;

    let expected_tile_count = tiles_full_x_len * tiles_full_y_len;
    if tile_buffers.tile_min.len() != expected_tile_count {
        // It is guaranteed for tile_min and tile_max to have same length by design
        // so, avoiding additional check for tile_max
        return Err(AprilTagError::InvalidTileBufferSize(
            tile_buffers.tile_min.len(),
            expected_tile_count,
        ));
    }

    // let src_data = src.as_slice();
    let dst_data = dst.as_slice_mut();

    // Calculate extrema (i.e. min & max grayscale value) of each tile
    tile_iterator.tile_enumerator().for_each(|((y, x), tile)| {
        if tile.len() != tile_buffers.tile_size || tile[0].len() != tile_buffers.tile_size {
            // Skip non-full tiles
            return;
        }

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

        let index = y * tiles_full_x_len + x;
        tile_buffers.tile_min[index] = local_min;
        tile_buffers.tile_max[index] = local_max;
    });

    let mut im_min = vec![255; expected_tile_count];
    let mut im_max = vec![0; expected_tile_count];

    // Binarize the image
    TileIterator::from_image(src, tile_buffers.tile_size)
        .tile_enumerator()
        .for_each(|((y, x), tile)| {
            let not_full_tile =
                tile.len() != tile_buffers.tile_size || tile[0].len() != tile_buffers.tile_size;

            let tile_index = y * tiles_full_x_len + x;

            let (neighbor_min, neighbor_max) = if not_full_tile {
                if tile.len() != tile_buffers.tile_size && tile[0].len() != tile_buffers.tile_size {
                    // Bottom-Right tile of image
                    tile_buffers.neighbor_blur((y - 1, x - 1), (tiles_full_y_len, tiles_full_x_len))
                } else if tile[0].len() != tile_buffers.tile_size {
                    // Right tile of image
                    tile_buffers.neighbor_blur((y, x - 1), (tiles_full_y_len, tiles_full_x_len))
                } else {
                    // Bottom tile of image
                    tile_buffers.neighbor_blur((y - 1, x), (tiles_full_y_len, tiles_full_x_len))
                }
            } else {
                let (neighbor_min, neighbor_max) =
                    tile_buffers.neighbor_blur((y, x), (tiles_full_y_len, tiles_full_x_len));

                im_min[tile_index] = neighbor_min;
                im_max[tile_index] = neighbor_max;

                if neighbor_max - neighbor_min < min_white_black_diff {
                    for y_px in 0..tile.len() {
                        let row = ((y * tile_buffers.tile_size) + y_px) * src.width();
                        let start_index = row + (x * tile_buffers.tile_size);
                        let end_index = start_index + tile[0].len();

                        for px in dst_data.iter_mut().take(end_index).skip(start_index) {
                            *px = Pixel::Skip;
                        }
                    }

                    return;
                }

                (neighbor_min, neighbor_max)
            };

            let thresh = neighbor_min + (neighbor_max - neighbor_min) / 2;

            for (y_px, row) in tile.iter().enumerate() {
                let row_index = ((y * tile_buffers.tile_size) + y_px) * src.width()
                    + x * tile_buffers.tile_size;

                for (x_px, px) in (row as &[u8]).iter().enumerate() {
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
        let tile_buffers = TileBuffers {
            tile_min: vec![10, 20, 30, 40],
            tile_max: vec![50, 60, 70, 80],
            tile_size: 2,
        };

        let mut neighbor_min = 25;
        let mut neighbor_max = 65;

        tile_buffers.neighbor_min_max(&mut neighbor_min, &mut neighbor_max, (0, 1), 2);
        assert_eq!(neighbor_min, 20);
        assert_eq!(neighbor_max, 65);

        tile_buffers.neighbor_min_max(&mut neighbor_min, &mut neighbor_max, (1, 0), 2);
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
            0,   255, 255, 255, 0
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
    fn test_adaptive_threshold_synthetic_image() {
        let src = read_image_png_mono8("../../tests/data/apriltag.png").unwrap();
        let mut bin = Image::from_size_val(src.size(), Pixel::Skip, CpuAllocator).unwrap();

        let mut tile_buffers = TileBuffers::new(src.size(), 4);
        adaptive_threshold(&src, &mut bin, &mut tile_buffers, 20).unwrap();

        assert_eq!(bin.as_slice(), src.as_slice())
    }

    #[test]
    fn invalid_buffer_size() {
        let img_size = ImageSize {
            width: 4,
            height: 4,
        };

        let src = Image::new(img_size, vec![100u8; 16], CpuAllocator).unwrap();

        let mut dst = Image::from_size_val(img_size, Pixel::default(), CpuAllocator).unwrap();

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
            Err(AprilTagError::InvalidTileBufferSize(1, 4))
        ));

        let mut tile_buffers = TileBuffers::new(src.size(), 5);
        let result = adaptive_threshold(&src, &mut dst, &mut tile_buffers, 20);
        assert!(matches!(result, Err(AprilTagError::InvalidImageSize)));
    }
}
