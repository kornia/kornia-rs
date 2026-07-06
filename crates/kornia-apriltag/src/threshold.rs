use crate::errors::AprilTagError;
use crate::utils::{find_full_tiles, Pixel, Point2d};
use kornia_image::{Image, ImageError, ImageSize};
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

    /// Fills `self.min` and `self.max` by scanning every full tile in `src`
    /// (NEON / AVX2 / scalar via [`crate::ops::fill_tile_stats`]).
    pub fn compute(&mut self, src: &Image<u8, 1>) {
        let img_data = src.as_slice();
        let img_width = src.width();
        let tile_size = self.tile_size;
        let tiles_x = img_width / tile_size;
        let tiles_y = src.height() / tile_size;
        crate::ops::fill_tile_stats(
            img_data,
            img_width,
            tile_size,
            tiles_x,
            tiles_y,
            &mut self.min,
            &mut self.max,
        );
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
/// use kornia_image::{Image, ImageSize};
/// use kornia_apriltag::threshold::{adaptive_threshold, TileMinMax};
/// use kornia_apriltag::utils::Pixel;
///
/// let src = Image::new(
///     ImageSize {
///         width: 2,
///         height: 3,
///     },
///     vec![0, 50, 100, 150, 200, 250],
/// )
/// .unwrap();
/// let mut dst = Image::from_size_val(src.size(), Pixel::Skip).unwrap();
///
/// let mut tile_buffers = TileMinMax::new(src.size(), 2);
/// adaptive_threshold(&src, &mut dst, &mut tile_buffers, 20).unwrap();
/// assert_eq!(dst.as_slice(), &[0, 0, 255, 255, 255, 255]);
/// ```
// TODO: Add support for parallelism
pub fn adaptive_threshold(
    src: &Image<u8, 1>,
    dst: &mut Image<Pixel, 1>,
    tile_min_max: &mut TileMinMax,
    min_white_black_diff: u8,
) -> Result<(), AprilTagError> {
    // Default split of 0.5 reproduces the classic AprilTag midpoint threshold
    // `min + (max - min) / 2`.
    adaptive_threshold_with_split(src, dst, tile_min_max, min_white_black_diff, 0.5)
}

/// Like [`adaptive_threshold`], but with a configurable `split` controlling where between
/// each tile's local minimum and maximum the black/white cut is placed.
///
/// The per-tile threshold is `min + (max - min) * split`. A value of `0.5` matches the
/// classic AprilTag midpoint behaviour. Lowering it (e.g. `0.33`) biases the binarization
/// toward white, which preserves thin bright quiet-zone margins around small or low-contrast
/// tags and prevents a tag's black border from merging with neighbouring dark regions.
///
/// `split` is clamped to `[0.0, 1.0]`.
pub fn adaptive_threshold_with_split(
    src: &Image<u8, 1>,
    dst: &mut Image<Pixel, 1>,
    tile_min_max: &mut TileMinMax,
    min_white_black_diff: u8,
    split: f32,
) -> Result<(), AprilTagError> {
    let split = split.clamp(0.0, 1.0);
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

    // Phase 1: tile statistics (sequential, NEON-accelerated, ~19 µs on Orin).
    tile_min_max.compute(src);

    let width = src.width();
    let height = src.height();
    let ts = tile_min_max.tile_size;
    let total_tile_cols = width.div_ceil(ts);

    let src_slice = src.as_slice();
    // Coerce &mut to & — tile_min_max is READ-ONLY after compute().
    let tm: &TileMinMax = tile_min_max;

    // Phase 2: classify pixels — parallel over tile rows.
    //
    // `par_chunks_mut(ts * width)` gives each Rayon thread one "tile row" of pixel rows
    // at a time (ts consecutive pixel-rows, disjoint from all other threads).  The tile
    // columns are handled sequentially within each thread, including the partial right
    // column when width is not a multiple of ts.
    //
    // After the parallel phase, any partial bottom row (height % ts != 0) is processed
    // sequentially — it is only 1–(ts-1) pixels tall and negligible in cost.
    let full_row_pixels = tiles_full_len.y * ts * width;
    let (full_dst, partial_dst) = dst.as_slice_mut().split_at_mut(full_row_pixels);

    full_dst
        .par_chunks_mut(ts * width)
        .enumerate()
        .for_each(|(ty, strip)| {
            for tx in 0..total_tile_cols {
                let px_start = tx * ts;
                let px_end = (px_start + ts).min(width);

                // For partial right column: clamp to the last full x tile.
                let (nb_min, nb_max) = if tx < tiles_full_len.x {
                    tm.neighbor_blur(
                        Point2d { x: tx, y: ty },
                        ty * tiles_full_len.x + tx,
                        tiles_full_len,
                    )
                } else {
                    let cx = tiles_full_len.x - 1;
                    tm.neighbor_blur(
                        Point2d { x: cx, y: ty },
                        ty * tiles_full_len.x + cx,
                        tiles_full_len,
                    )
                };

                if nb_max - nb_min < min_white_black_diff {
                    for row in 0..ts {
                        let row_off = row * width;
                        strip[row_off + px_start..row_off + px_end]
                            .iter_mut()
                            .for_each(|p| *p = Pixel::Skip);
                    }
                } else {
                    let thresh = nb_min + ((nb_max - nb_min) as f32 * split) as u8;
                    for row in 0..ts {
                        let row_off = row * width;
                        let src_off = (ty * ts + row) * width;
                        crate::ops::classify_row(
                            &src_slice[src_off + px_start..src_off + px_end],
                            &mut strip[row_off + px_start..row_off + px_end],
                            thresh,
                        );
                    }
                }
            }
        });

    // Sequential: partial bottom row (at most ts-1 pixel rows, often 0 or 1).
    if !partial_dst.is_empty() {
        let py_start = tiles_full_len.y * ts;
        let actual_rows = height - py_start;
        let cy = tiles_full_len.y.saturating_sub(1);

        for tx in 0..total_tile_cols {
            let px_start = tx * ts;
            let px_end = (px_start + ts).min(width);
            let cx = tx.min(tiles_full_len.x.saturating_sub(1));
            let (nb_min, nb_max) = tm.neighbor_blur(
                Point2d { x: cx, y: cy },
                cy * tiles_full_len.x + cx,
                tiles_full_len,
            );

            if nb_max - nb_min < min_white_black_diff {
                for row in 0..actual_rows {
                    let row_off = row * width;
                    partial_dst[row_off + px_start..row_off + px_end]
                        .iter_mut()
                        .for_each(|p| *p = Pixel::Skip);
                }
            } else {
                let thresh = nb_min + ((nb_max - nb_min) as f32 * split) as u8;
                for row in 0..actual_rows {
                    let row_off = row * width;
                    let src_off = (py_start + row) * width;
                    crate::ops::classify_row(
                        &src_slice[src_off + px_start..src_off + px_end],
                        &mut partial_dst[row_off + px_start..row_off + px_end],
                        thresh,
                    );
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::ImageSize;
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
        )?;
        let mut dst = Image::from_size_val(src.size(), Pixel::Skip)?;

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
        )?;
        let mut dst = Image::from_size_val(src.size(), Pixel::Skip)?;

        let mut tile_buffers = TileMinMax::new(src.size(), 2);
        adaptive_threshold(&src, &mut dst, &mut tile_buffers, 20)?;
        assert_eq!(dst.as_slice(), &[Pixel::Skip; 16]);
        Ok(())
    }

    #[test]
    fn test_adaptive_threshold_synthetic_image() -> Result<(), Box<dyn std::error::Error>> {
        let src = read_image_png_mono8("../../tests/data/apriltag.png")?;
        let mut bin = Image::from_size_val(src.size(), Pixel::Skip)?;

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

        let src = Image::new(img_size, vec![100u8; 16])?;

        let mut dst = Image::from_size_val(img_size, Pixel::default())?;

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

    // Deterministic LCG pseudo-random bytes for parity inputs.
    #[cfg(target_arch = "x86_64")]
    fn lcg_bytes(n: usize, seed: u32) -> Vec<u8> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (s >> 24) as u8
            })
            .collect()
    }

    /// AVX2 `classify_row` must be bit-identical to scalar across all lengths/thresholds.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_classify_row_avx2_parity() {
        if !crate::ops::has_avx2() {
            eprintln!("AVX2 not present; skipping");
            return;
        }
        for len in [0usize, 1, 7, 15, 16, 31, 32, 33, 100, 257] {
            let src = lcg_bytes(len, 0xC0FFEE ^ len as u32);
            for &thresh in &[0u8, 1, 64, 127, 128, 200, 254, 255] {
                let mut a = vec![Pixel::Black; len];
                let mut b = vec![Pixel::Black; len];
                // SAFETY: guarded by has_avx2; equal lengths.
                unsafe { crate::ops::avx2::classify_row(&src, &mut a, thresh) };
                for (s, d) in src.iter().zip(b.iter_mut()) {
                    *d = if *s > thresh {
                        Pixel::White
                    } else {
                        Pixel::Black
                    };
                }
                assert_eq!(a, b, "mismatch len={len} thresh={thresh}");
            }
        }
    }

    /// AVX2 `fill_tile_stats` must match the scalar tile min/max exactly.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_fill_tile_stats_avx2_parity() {
        if !crate::ops::has_avx2() {
            eprintln!("AVX2 not present; skipping");
            return;
        }
        let tile_size = 4usize;
        // Include widths whose tile count is and isn't a multiple of 8.
        for &(tiles_x, tiles_y) in &[(8usize, 3usize), (10, 4), (37, 5), (3, 2)] {
            let img_width = tiles_x * tile_size;
            let img_height = tiles_y * tile_size;
            let img = lcg_bytes(
                img_width * img_height,
                0xBEEF ^ (tiles_x * 131 + tiles_y) as u32,
            );

            let n = tiles_x * tiles_y;
            let (mut amin, mut amax) = (vec![0u8; n], vec![0u8; n]);
            // SAFETY: dimensions are exact multiples, so all loads are in bounds.
            unsafe {
                crate::ops::avx2::fill_tile_stats(
                    &img, img_width, tile_size, tiles_x, tiles_y, &mut amin, &mut amax,
                )
            };

            let (mut smin, mut smax) = (vec![0u8; n], vec![0u8; n]);
            for ty in 0..tiles_y {
                for tx in 0..tiles_x {
                    let (mut mn, mut mx) = (255u8, 0u8);
                    for row in 0..tile_size {
                        let rs = (ty * tile_size + row) * img_width + tx * tile_size;
                        for &px in &img[rs..rs + tile_size] {
                            mn = mn.min(px);
                            mx = mx.max(px);
                        }
                    }
                    smin[ty * tiles_x + tx] = mn;
                    smax[ty * tiles_x + tx] = mx;
                }
            }
            assert_eq!(amin, smin, "min mismatch {tiles_x}x{tiles_y}");
            assert_eq!(amax, smax, "max mismatch {tiles_x}x{tiles_y}");
        }
    }
}
