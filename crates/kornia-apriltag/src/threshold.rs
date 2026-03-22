use crate::utils::{find_full_tiles, Pixel, Point2d};
use crate::errors::AprilTagError;
use kornia_image::{allocator::ImageAllocator, Image, ImageError, ImageSize};
use rayon::prelude::*;

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

    let tiles_full_len = find_full_tiles(src.size(), tile_min_max.tile_size);

    if tile_min_max.min.len() != tiles_full_len.x * tiles_full_len.y {
        // It is guaranteed for tile_min and tile_max to have same length by design
        // so, avoiding additional check for tile_max
        return Err(AprilTagError::ImageTileSizeMismatch);
    }

    let src_data = src.as_slice();
    let width = src.width();
    let height = src.height();
    let tile_size = tile_min_max.tile_size;

    // Phase 1: compute min/max for each full tile in parallel.
    //
    // Each tile index i maps to row (i / tiles_full_len.x) and col (i % tiles_full_len.x).
    // Each thread writes to a unique (min[i], max[i]) pair -- no data races.
    // src_data is a shared read-only slice and is Sync.
    tile_min_max
        .min
        .par_iter_mut()
        .zip(tile_min_max.max.par_iter_mut())
        .enumerate()
        .for_each(|(tile_idx, (tile_min, tile_max))| {
            let tile_y = tile_idx / tiles_full_len.x;
            let tile_x = tile_idx % tiles_full_len.x;
            let py_start = tile_y * tile_size;
            let px_start = tile_x * tile_size;

            let mut local_min = u8::MAX;
            let mut local_max = 0u8;

            for py in py_start..py_start + tile_size {
                let row_start = py * width + px_start;
                for &val in &src_data[row_start..row_start + tile_size] {
                    if val < local_min {
                        local_min = val;
                    }
                    if val > local_max {
                        local_max = val;
                    }
                }
            }

            *tile_min = local_min;
            *tile_max = local_max;
        });

    // Phase 2: binarize the image.
    //
    // Full tile rows are processed in parallel via par_chunks_mut. Each strip is an
    // independent &mut [Pixel] slice of length tile_size * width -- no overlaps.
    // tile_min_max is accessed read-only (&TileMinMax is Sync) and src_data is &[u8] (Sync).
    // Partial rows at the bottom edge (height % tile_size != 0) are handled sequentially after
    // the parallel block, matching the behavior of the sequential PartialTile path.
    let tile_min_max_ref: &TileMinMax = tile_min_max;
    let dst_data = dst.as_slice_mut();

    let full_strip_len = tile_size * width;
    let full_rows_data_len = tiles_full_len.y * full_strip_len;

    dst_data[..full_rows_data_len]
        .par_chunks_mut(full_strip_len)
        .enumerate()
        .for_each(|(tile_row, strip)| {
            // Full tile columns within this tile row.
            for tile_col in 0..tiles_full_len.x {
                let tile_idx = tile_row * tiles_full_len.x + tile_col;
                let pos = Point2d {
                    x: tile_col,
                    y: tile_row,
                };
                let (neighbor_min, neighbor_max) =
                    tile_min_max_ref.neighbor_blur(pos, tile_idx, tiles_full_len);

                let px_start = tile_col * tile_size;

                if neighbor_max - neighbor_min < min_white_black_diff {
                    for py in 0..tile_size {
                        let strip_row_start = py * width + px_start;
                        for px in &mut strip[strip_row_start..strip_row_start + tile_size] {
                            *px = Pixel::Skip;
                        }
                    }
                } else {
                    let thresh = neighbor_min + (neighbor_max - neighbor_min) / 2;
                    for py in 0..tile_size {
                        let strip_row_start = py * width + px_start;
                        let src_row_start = (tile_row * tile_size + py) * width + px_start;
                        for (off, pixel) in strip
                            [strip_row_start..strip_row_start + tile_size]
                            .iter_mut()
                            .enumerate()
                        {
                            *pixel = if src_data[src_row_start + off] > thresh {
                                Pixel::White
                            } else {
                                Pixel::Black
                            };
                        }
                    }
                }
            }

            // Right-edge partial column (width % tile_size != 0).
            // Mirrors the PartialTile(partial_x) path: use the rightmost full tile column's
            // neighbor blur and always threshold (no min_white_black_diff skip).
            let partial_x_start = tiles_full_len.x * tile_size;
            if partial_x_start < width {
                let partial_width = width - partial_x_start;
                let last_col = tiles_full_len.x.saturating_sub(1);
                let tile_idx = tile_row * tiles_full_len.x + last_col;
                let pos = Point2d {
                    x: last_col,
                    y: tile_row,
                };
                let (neighbor_min, neighbor_max) =
                    tile_min_max_ref.neighbor_blur(pos, tile_idx, tiles_full_len);
                let thresh = neighbor_min + (neighbor_max - neighbor_min) / 2;

                for py in 0..tile_size {
                    let strip_row_start = py * width + partial_x_start;
                    let src_row_start = (tile_row * tile_size + py) * width + partial_x_start;
                    for (off, pixel) in strip[strip_row_start..strip_row_start + partial_width]
                        .iter_mut()
                        .enumerate()
                    {
                        *pixel = if src_data[src_row_start + off] > thresh {
                            Pixel::White
                        } else {
                            Pixel::Black
                        };
                    }
                }
            }
        });

    // Partial rows at the bottom edge (height % tile_size != 0).
    // Mirrors the PartialTile(partial_y / partial_xy) path: always threshold, no Skip check.
    let partial_y_start = tiles_full_len.y * tile_size;
    if partial_y_start < height {
        let partial_height = height - partial_y_start;
        let last_row = tiles_full_len.y.saturating_sub(1);

        // Full columns in the bottom partial row.
        for tile_col in 0..tiles_full_len.x {
            // PartialTile(partial_y) path: pos = (tile_col, pos.y - 1), same full_index logic.
            let full_idx = last_row * tiles_full_len.x + tile_col;
            let pos = Point2d {
                x: tile_col,
                y: last_row,
            };
            let (neighbor_min, neighbor_max) =
                tile_min_max_ref.neighbor_blur(pos, full_idx, tiles_full_len);
            let thresh = neighbor_min + (neighbor_max - neighbor_min) / 2;

            let px_start = tile_col * tile_size;
            for py in 0..partial_height {
                let abs_row = partial_y_start + py;
                let dst_start = abs_row * width + px_start;
                for (off, pixel) in
                    dst_data[dst_start..dst_start + tile_size].iter_mut().enumerate()
                {
                    *pixel = if src_data[dst_start + off] > thresh {
                        Pixel::White
                    } else {
                        Pixel::Black
                    };
                }
            }
        }

        // Bottom-right partial corner (partial_xy): pos = (last_col, last_row).
        let partial_x_start = tiles_full_len.x * tile_size;
        if partial_x_start < width {
            let partial_width = width - partial_x_start;
            let last_col = tiles_full_len.x.saturating_sub(1);
            let full_idx = last_row * tiles_full_len.x + last_col;
            let pos = Point2d {
                x: last_col,
                y: last_row,
            };
            let (neighbor_min, neighbor_max) =
                tile_min_max_ref.neighbor_blur(pos, full_idx, tiles_full_len);
            let thresh = neighbor_min + (neighbor_max - neighbor_min) / 2;

            for py in 0..partial_height {
                let abs_row = partial_y_start + py;
                let dst_start = abs_row * width + partial_x_start;
                for (off, pixel) in
                    dst_data[dst_start..dst_start + partial_width].iter_mut().enumerate()
                {
                    *pixel = if src_data[dst_start + off] > thresh {
                        Pixel::White
                    } else {
                        Pixel::Black
                    };
                }
            }
        }
    }

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
