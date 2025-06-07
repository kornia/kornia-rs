use crate::{errors::AprilTagError, utils::PixelTrait};
use kornia_image::{allocator::ImageAllocator, Image, ImageError, ImageSize};
use kornia_imgproc::iter::TileIterator;
use std::ops::{Add, Div, Sub};

/// Stores the minimum and maximum pixel values for each tile for [adaptive_threshold]
pub struct TileBuffers<T> {
    tile_min: Vec<T>,
    tile_max: Vec<T>,
}

impl<T: PixelTrait + Copy> TileBuffers<T> {
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
            tile_min: vec![T::BLACK; num_tiles],
            tile_max: vec![T::BLACK; num_tiles],
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
/// - `tile_size`: The size of the tiles used for local thresholding. Each tile is a square of `tile_size x tile_size` pixels.
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
///      the tile is skipped, and its pixels are marked as [PixelTrait::SKIP_PROCESSING].
///    - Otherwise, the threshold for the tile is computed as:
///      `threshold = local_min + (local_max - local_min) / 2`.
///    - Pixels in the tile are binarized based on whether they are above or below the threshold.
/// 3. Neighboring tiles are considered to refine the threshold for each tile, ensuring smooth transitions.
///
/// # Recommended Values for `min_white_black_diff`
///
/// | Image Type | `min_white_black_diff` |
/// |------------|------------------------|
/// | 8-bit      | 10-20                  |
/// | 16-bit     | 500-1000               |
/// | 32-bit     | 100,000-1,000,000      |
/// | float      | 0.05-0.1               |
///
/// # Examples
///
/// ```
/// use kornia_image::{allocator::CpuAllocator, Image, ImageSize};
/// use kornia_apriltag::threshold::{adaptive_threshold, TileBuffers};
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
/// let mut dst = Image::from_size_val(src.size(), 0u8, CpuAllocator).unwrap();
///
/// let mut tile_buffers = TileBuffers::new(src.size(), 2);
/// adaptive_threshold(&src, &mut dst, &mut tile_buffers, 2, 20).unwrap();
/// assert_eq!(dst.as_slice(), &[0, 0, 0, 255, 255, 255]);
/// ```
// TODO: Add support for parallelism
pub fn adaptive_threshold<
    T: PixelTrait
        + PartialOrd
        + Copy
        + Add<T, Output = T>
        + Sub<T, Output = T>
        + Div<T, Output = T>
        + From<u8>,
    A1: ImageAllocator,
    A2: ImageAllocator,
>(
    src: &Image<T, 1, A1>,
    dst: &mut Image<T, 1, A2>,
    tile_buffers: &mut TileBuffers<T>,
    tile_size: usize,
    min_white_black_diff: T,
) -> Result<(), AprilTagError> {
    if src.size() != dst.size() {
        return Err(
            ImageError::InvalidImageSize(src.cols(), src.rows(), dst.cols(), dst.rows()).into(),
        );
    }

    let tile_iterator = TileIterator::from_image(src, tile_size);
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
        let mut local_min = T::WHITE;
        let mut local_max = T::BLACK;

        for row in tile {
            for px in row as &[T] {
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
    for ((y, x), tile) in TileIterator::from_image(src, tile_size).tile_enumerator() {
        let tile_index = (y * tiles_x_len) + x;

        let mut neighbor_min = tile_buffers.tile_min[tile_index];
        let mut neighbor_max = tile_buffers.tile_max[tile_index];

        // Low constrast tile, Skip processing
        if neighbor_max - neighbor_min < min_white_black_diff {
            for y_px in 0..tile.len() {
                let row = ((y * tile_size) + y_px) * src.width();
                let start_index = row + (x * tile_size);
                let end_index = start_index + tile[0].len();

                for px in dst_data.iter_mut().take(end_index).skip(start_index) {
                    *px = T::SKIP_PROCESSING;
                }
            }

            continue;
        }

        macro_rules! neighbor_min_max {
            ($condition:expr, $index:expr) => {
                if $condition {
                    let tile_index = $index;

                    if tile_buffers.tile_min[tile_index] < neighbor_min {
                        neighbor_min = tile_buffers.tile_min[tile_index]
                    }

                    if tile_buffers.tile_max[tile_index] > neighbor_max {
                        neighbor_max = tile_buffers.tile_max[tile_index];
                    }
                }
            };
        }

        // Rightmost neighbor
        neighbor_min_max!(x + 1 != tiles_x_len, tile_index + 1);

        // Leftmost neighbor
        neighbor_min_max!(x != 0, tile_index - 1);

        // Bottom-most neighbor
        neighbor_min_max!(y + 1 != tiles_y_len, tile_index + tiles_x_len);

        // Uppermost neighbor exists
        neighbor_min_max!(y != 0, tile_index - tiles_x_len);

        // Top-right neighbor
        neighbor_min_max!(x + 1 != tiles_x_len && y != 0, tile_index - tiles_x_len + 1);

        // Top-left neighbor
        neighbor_min_max!(x != 0 && y != 0, tile_index - tiles_x_len - 1);

        // Bottom-right corner neighbor
        neighbor_min_max!(
            x + 1 != tiles_x_len && y + 1 != tiles_y_len,
            tile_index + tiles_x_len + 1
        );

        // Bottom-left corner neighbor
        neighbor_min_max!(x != 0 && y + 1 != tiles_y_len, tile_index + tiles_x_len - 1);

        let thresh = neighbor_min + (neighbor_max - neighbor_min) / T::from(2u8);

        for (y_px, row) in tile.iter().enumerate() {
            let row_index = ((y * tile_size) + y_px) * src.width() + x * tile_size;

            for (x_px, px) in (row as &[T]).iter().enumerate() {
                dst_data[row_index + x_px] = if px > &thresh { T::WHITE } else { T::BLACK };
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
        let mut dst = Image::from_size_val(src.size(), 0u8, CpuAllocator).unwrap();

        let mut tile_buffers = TileBuffers::new(src.size(), 2);
        adaptive_threshold(&src, &mut dst, &mut tile_buffers, 2, 20).unwrap();

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
        let mut dst = Image::from_size_val(src.size(), 0u8, CpuAllocator).unwrap();

        let mut tile_buffers = TileBuffers::new(src.size(), 2);
        adaptive_threshold(&src, &mut dst, &mut tile_buffers, 2, 20).unwrap();
        assert_eq!(dst.as_slice(), &[u8::SKIP_PROCESSING; 16]);
    }

    #[test]
    #[should_panic(
        expected = "called `Result::unwrap()` on an `Err` value: InvalidTileBufferSize(2, 4)"
    )]
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
            0u8,
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
        adaptive_threshold(&src, &mut dst, &mut tile_buffers, 2, 20).unwrap();
    }
}
