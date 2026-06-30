use crate::errors::AprilTagError;
use crate::utils::{find_full_tiles, Pixel, Point2d};
use kornia_image::{allocator::ImageAllocator, Image, ImageError, ImageSize};
use rayon::prelude::*;

/// Scalar min/max over one `tile_size`×`tile_size` tile at column `tile_x`, row `tile_y`.
///
/// Shared by every dispatch path's scalar tail/fallback so the per-tile reduction
/// is written exactly once.
#[inline]
fn tile_min_max(
    img_data: &[u8],
    img_width: usize,
    tile_size: usize,
    tile_x: usize,
    tile_y: usize,
) -> (u8, u8) {
    let mut lo = 255u8;
    let mut hi = 0u8;
    for row in 0..tile_size {
        let row_start = (tile_y * tile_size + row) * img_width + tile_x * tile_size;
        for &px in &img_data[row_start..row_start + tile_size] {
            lo = lo.min(px);
            hi = hi.max(px);
        }
    }
    (lo, hi)
}

/// Classifies pixels in a contiguous row: pixel > thresh → 255 (White), else 0 (Black).
///
/// On aarch64 uses NEON `vcgtq_u8` which maps exactly to Pixel::White/Black since those
/// are repr(u8) values 255 and 0 respectively.
///
/// # Safety
/// `src` and `dst` must have the same length. `Pixel` must be `#[repr(u8)]`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn classify_row_neon(src: &[u8], dst: &mut [Pixel], thresh: u8) {
    use std::arch::aarch64::*;
    let thresh_v = vdupq_n_u8(thresh);
    let len = src.len();
    // Safety: Pixel is #[repr(u8)] so *mut Pixel == *mut u8 for layout purposes.
    let dst_u8 = core::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut u8, len);
    let mut i = 0;
    while i + 16 <= len {
        let px = vld1q_u8(src.as_ptr().add(i));
        vst1q_u8(dst_u8.as_mut_ptr().add(i), vcgtq_u8(px, thresh_v));
        i += 16;
    }
    while i < len {
        dst_u8[i] = if src[i] > thresh { 255 } else { 0 };
        i += 1;
    }
}

/// AVX2 variant of [`classify_row_neon`]: 32 pixels per iteration.
///
/// AVX2 only has signed byte compares, so we bias both operands by `0x80` to turn
/// the unsigned `px > thresh` into a signed `cmpgt`. The result is `0xFF`/`0x00`,
/// which equals `Pixel::White` (255) / `Pixel::Black` (0) directly.
///
/// # Safety
/// AVX2 must be available; `src.len() == dst.len()`; `Pixel` is `#[repr(u8)]`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn classify_row_avx2(src: &[u8], dst: &mut [Pixel], thresh: u8) {
    use std::arch::x86_64::*;
    let len = src.len();
    // SAFETY: Pixel is #[repr(u8)] so *mut Pixel == *mut u8 for layout purposes.
    let dst_u8 = core::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut u8, len);
    let bias = _mm256_set1_epi8(0x80u8 as i8);
    let tv = _mm256_xor_si256(_mm256_set1_epi8(thresh as i8), bias);
    let mut i = 0;
    while i + 32 <= len {
        let px = _mm256_loadu_si256(src.as_ptr().add(i) as *const __m256i);
        let gt = _mm256_cmpgt_epi8(_mm256_xor_si256(px, bias), tv);
        _mm256_storeu_si256(dst_u8.as_mut_ptr().add(i) as *mut __m256i, gt);
        i += 32;
    }
    while i < len {
        dst_u8[i] = if src[i] > thresh { 255 } else { 0 };
        i += 1;
    }
}

#[inline(always)]
fn classify_row(src: &[u8], dst: &mut [Pixel], thresh: u8) {
    // AArch64: NEON is baseline.
    #[cfg(target_arch = "aarch64")]
    // SAFETY: aarch64 always has NEON; Pixel is #[repr(u8)].
    return unsafe { classify_row_neon(src, dst, thresh) };

    // x86_64: AVX2 when the runtime probe confirms it.
    #[cfg(target_arch = "x86_64")]
    if crate::simd::has_avx2() {
        // SAFETY: AVX2 confirmed by runtime probe; Pixel is #[repr(u8)].
        return unsafe { classify_row_avx2(src, dst, thresh) };
    }

    // Portable fallback.
    #[cfg(not(target_arch = "aarch64"))]
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = if *s > thresh { Pixel::White } else { Pixel::Black };
    }
}

/// Fills `tile_min` and `tile_max` for all full tiles in row-major order.
///
/// **Batch-of-4-tiles strategy for tile_size=4 (the common case):**
/// 16 consecutive image pixels (= 4 side-by-side tiles) fit in one NEON q-register,
/// so each `vld1q_u8` processes a whole row of 4 tiles with no gather overhead.
/// After accumulating `tile_size` rows we apply two rounds of `vpminq_u8`/`vpmaxq_u8`
/// to reduce the 16-lane register into 4 per-tile scalars (lanes 0-3).
///
/// # Safety
/// Caller must ensure `tiles_y * tile_size * img_width + tiles_x * tile_size ≤ img_data.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn fill_tile_stats_neon(
    img_data: &[u8],
    img_width: usize,
    tile_size: usize,
    tiles_x: usize,
    tiles_y: usize,
    tile_min: &mut [u8],
    tile_max: &mut [u8],
) {
    use std::arch::aarch64::*;

    for tile_y in 0..tiles_y {
        let mut tile_x = 0usize;

        // Fast path: consume 4 tile-columns at a time using 16-byte contiguous loads.
        // One q-register covers exactly 4 consecutive tiles (4 columns × tile_size=4 pixels).
        while tile_size == 4 && tile_x + 4 <= tiles_x {
            // Accumulate row-min/max across tile_size rows for all 4 tiles simultaneously.
            let mut vmin = vdupq_n_u8(255);
            let mut vmax = vdupq_n_u8(0);

            for row in 0..tile_size {
                let offset = (tile_y * tile_size + row) * img_width + tile_x * tile_size;
                let v = vld1q_u8(img_data.as_ptr().add(offset));
                vmin = vminq_u8(vmin, v);
                vmax = vmaxq_u8(vmax, v);
            }

            // Two rounds of pairwise-min collapse 16 lanes into 4 per-tile minima.
            // Round 1: [min(px0,px1), min(px2,px3), min(px4,px5), ..., (repeat 8..15)]
            // Round 2: [min(px0..3), min(px4..7), min(px8..11), min(px12..15), ...]
            vmin = vpminq_u8(vmin, vmin);
            vmin = vpminq_u8(vmin, vmin);
            vmax = vpmaxq_u8(vmax, vmax);
            vmax = vpmaxq_u8(vmax, vmax);

            let base = tile_y * tiles_x + tile_x;
            tile_min[base] = vgetq_lane_u8::<0>(vmin);
            tile_min[base + 1] = vgetq_lane_u8::<1>(vmin);
            tile_min[base + 2] = vgetq_lane_u8::<2>(vmin);
            tile_min[base + 3] = vgetq_lane_u8::<3>(vmin);
            tile_max[base] = vgetq_lane_u8::<0>(vmax);
            tile_max[base + 1] = vgetq_lane_u8::<1>(vmax);
            tile_max[base + 2] = vgetq_lane_u8::<2>(vmax);
            tile_max[base + 3] = vgetq_lane_u8::<3>(vmax);

            tile_x += 4;
        }

        // Scalar tail: remaining tile columns (< 4 when tiles_x % 4 != 0, or tile_size != 4).
        while tile_x < tiles_x {
            let idx = tile_y * tiles_x + tile_x;
            let (lo, hi) = tile_min_max(img_data, img_width, tile_size, tile_x, tile_y);
            tile_min[idx] = lo;
            tile_max[idx] = hi;
            tile_x += 1;
        }
    }
}

/// AVX2 variant of [`fill_tile_stats_neon`]: batches **8** tiles per iteration.
///
/// For `tile_size == 4`, 32 contiguous bytes are exactly 8 side-by-side tiles, so
/// one `loadu` covers a row of 8 tiles. After accumulating row min/max, each tile's
/// 4 bytes lie within one 32-bit lane, so `srli_epi32` (which shifts *within* each
/// lane, never across tile boundaries) reduces all 4 bytes into the lane's byte 0.
///
/// # Safety
/// Caller must ensure `tiles_y * tile_size * img_width + tiles_x * tile_size ≤ img_data.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn fill_tile_stats_avx2(
    img_data: &[u8],
    img_width: usize,
    tile_size: usize,
    tiles_x: usize,
    tiles_y: usize,
    tile_min: &mut [u8],
    tile_max: &mut [u8],
) {
    use std::arch::x86_64::*;

    for tile_y in 0..tiles_y {
        let mut tile_x = 0usize;

        // Fast path: 8 tile-columns at a time via 32-byte contiguous loads.
        while tile_size == 4 && tile_x + 8 <= tiles_x {
            let mut vmin = _mm256_set1_epi8(0xFFu8 as i8);
            let mut vmax = _mm256_setzero_si256();

            for row in 0..tile_size {
                let offset = (tile_y * tile_size + row) * img_width + tile_x * tile_size;
                let v = _mm256_loadu_si256(img_data.as_ptr().add(offset) as *const __m256i);
                vmin = _mm256_min_epu8(vmin, v);
                vmax = _mm256_max_epu8(vmax, v);
            }

            // Reduce each 4-byte tile (one 32-bit lane) to its min/max in byte 0.
            let mn = _mm256_min_epu8(vmin, _mm256_srli_epi32::<16>(vmin));
            let mn = _mm256_min_epu8(mn, _mm256_srli_epi32::<8>(mn));
            let mx = _mm256_max_epu8(vmax, _mm256_srli_epi32::<16>(vmax));
            let mx = _mm256_max_epu8(mx, _mm256_srli_epi32::<8>(mx));

            // Extract the low byte of each of the 8 u32 lanes.
            let mut bmin = [0u8; 32];
            let mut bmax = [0u8; 32];
            _mm256_storeu_si256(bmin.as_mut_ptr() as *mut __m256i, mn);
            _mm256_storeu_si256(bmax.as_mut_ptr() as *mut __m256i, mx);

            let base = tile_y * tiles_x + tile_x;
            for g in 0..8 {
                tile_min[base + g] = bmin[g * 4];
                tile_max[base + g] = bmax[g * 4];
            }
            tile_x += 8;
        }

        // Scalar tail: remaining tile columns (< 8, or tile_size != 4).
        while tile_x < tiles_x {
            let idx = tile_y * tiles_x + tile_x;
            let (lo, hi) = tile_min_max(img_data, img_width, tile_size, tile_x, tile_y);
            tile_min[idx] = lo;
            tile_max[idx] = hi;
            tile_x += 1;
        }
    }
}

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

    /// Fills `self.min` and `self.max` by scanning every full tile in `src`.
    ///
    /// On AArch64 uses NEON with a batch-of-4-tiles fast path (tile_size=4); falls back to
    /// scalar on other targets.
    pub fn compute<A: ImageAllocator>(&mut self, src: &Image<u8, 1, A>) {
        let img_data = src.as_slice();
        let img_width = src.width();
        let tile_size = self.tile_size;
        let tiles_x = img_width / tile_size;
        let tiles_y = src.height() / tile_size;

        // AArch64: NEON batch-of-4-tiles path.
        #[cfg(target_arch = "aarch64")]
        // SAFETY: tiles_x/tiles_y come from floor division, so all tile accesses
        // stay within img_data bounds.
        return unsafe {
            fill_tile_stats_neon(
                img_data, img_width, tile_size, tiles_x, tiles_y, &mut self.min, &mut self.max,
            )
        };

        // x86_64: AVX2 batch-of-8-tiles path when available.
        #[cfg(target_arch = "x86_64")]
        if crate::simd::has_avx2() {
            // SAFETY: AVX2 confirmed by runtime probe; bounds as above.
            return unsafe {
                fill_tile_stats_avx2(
                    img_data, img_width, tile_size, tiles_x, tiles_y, &mut self.min, &mut self.max,
                )
            };
        }

        // Portable scalar fallback (non-aarch64 without AVX2).
        #[cfg(not(target_arch = "aarch64"))]
        for tile_y in 0..tiles_y {
            for tile_x in 0..tiles_x {
                let idx = tile_y * tiles_x + tile_x;
                let (lo, hi) = tile_min_max(img_data, img_width, tile_size, tile_x, tile_y);
                self.min[idx] = lo;
                self.max[idx] = hi;
            }
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
    let total_tile_cols = (width + ts - 1) / ts;

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
                let tile_w = px_end - px_start;

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
                    let thresh = nb_min + (nb_max - nb_min) / 2;
                    for row in 0..ts {
                        let row_off = row * width;
                        let src_off = (ty * ts + row) * width;
                        classify_row(
                            &src_slice[src_off + px_start..src_off + px_end],
                            &mut strip[row_off + px_start..row_off + px_end],
                            thresh,
                        );
                    }
                }
                let _ = tile_w; // suppress unused-variable lint
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
            let (nb_min, nb_max) =
                tm.neighbor_blur(Point2d { x: cx, y: cy }, cy * tiles_full_len.x + cx, tiles_full_len);

            if nb_max - nb_min < min_white_black_diff {
                for row in 0..actual_rows {
                    let row_off = row * width;
                    partial_dst[row_off + px_start..row_off + px_end]
                        .iter_mut()
                        .for_each(|p| *p = Pixel::Skip);
                }
            } else {
                let thresh = nb_min + (nb_max - nb_min) / 2;
                for row in 0..actual_rows {
                    let row_off = row * width;
                    let src_off = (py_start + row) * width;
                    classify_row(
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
        if !crate::simd::has_avx2() {
            eprintln!("AVX2 not present; skipping");
            return;
        }
        for len in [0usize, 1, 7, 15, 16, 31, 32, 33, 100, 257] {
            let src = lcg_bytes(len, 0xC0FFEE ^ len as u32);
            for &thresh in &[0u8, 1, 64, 127, 128, 200, 254, 255] {
                let mut a = vec![Pixel::Black; len];
                let mut b = vec![Pixel::Black; len];
                // SAFETY: guarded by has_avx2; equal lengths.
                unsafe { classify_row_avx2(&src, &mut a, thresh) };
                for (s, d) in src.iter().zip(b.iter_mut()) {
                    *d = if *s > thresh { Pixel::White } else { Pixel::Black };
                }
                assert_eq!(a, b, "mismatch len={len} thresh={thresh}");
            }
        }
    }

    /// AVX2 `fill_tile_stats` must match the scalar tile min/max exactly.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_fill_tile_stats_avx2_parity() {
        if !crate::simd::has_avx2() {
            eprintln!("AVX2 not present; skipping");
            return;
        }
        let tile_size = 4usize;
        // Include widths whose tile count is and isn't a multiple of 8.
        for &(tiles_x, tiles_y) in &[(8usize, 3usize), (10, 4), (37, 5), (3, 2)] {
            let img_width = tiles_x * tile_size;
            let img_height = tiles_y * tile_size;
            let img = lcg_bytes(img_width * img_height, 0xBEEF ^ (tiles_x * 131 + tiles_y) as u32);

            let n = tiles_x * tiles_y;
            let (mut amin, mut amax) = (vec![0u8; n], vec![0u8; n]);
            // SAFETY: dimensions are exact multiples, so all loads are in bounds.
            unsafe {
                fill_tile_stats_avx2(&img, img_width, tile_size, tiles_x, tiles_y, &mut amin, &mut amax)
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
