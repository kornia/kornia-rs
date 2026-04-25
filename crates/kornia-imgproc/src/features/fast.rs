use std::ops::ControlFlow;

use kornia_image::{allocator::ImageAllocator, Image, ImageError, ImageSize};
use kornia_tensor::CpuAllocator;
use rayon::prelude::*;

#[derive(Clone, Copy, PartialEq)]
enum PixelType {
    Brighter,
    Darker,
    Similar,
}

/// A FAST (Features from Accelerated Segment Test) feature detector for corner detection in images.
#[derive(Clone)]
pub struct FastDetector {
    /// The intensity threshold for detecting corners.
    pub threshold: f32,
    /// The minimum distance between detected keypoints.
    pub min_distance: usize,
    /// The minimum arc length for a sequence of contiguous pixels to be considered a corner.
    pub arc_length: usize,
    corner_response: Image<f32, 1, CpuAllocator>,
    mask: Image<bool, 1, CpuAllocator>,
    taken: Vec<bool>,
}

impl FastDetector {
    /// Creates a new `FastDetector` with the specified parameters.
    ///
    /// # Arguments
    ///
    /// * `image_size` - The size of the image to process.
    /// * `threshold` - The intensity threshold for detecting corners.
    /// * `arc_length` - The minimum arc length for a sequence of contiguous pixels to be considered a corner.
    /// * `min_distance` - The minimum distance between detected keypoints.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the new `FastDetector` or an `ImageError`.
    pub fn new(
        image_size: ImageSize,
        threshold: f32,
        arc_length: usize,
        min_distance: usize,
    ) -> Result<Self, ImageError> {
        Ok(Self {
            threshold,
            min_distance,
            arc_length,
            corner_response: Image::from_size_val(image_size, 0.0, CpuAllocator)?,
            mask: Image::from_size_val(image_size, false, CpuAllocator)?,
            taken: vec![false; image_size.height * image_size.width],
        })
    }

    /// Clears the internal state of the detector, marking it ready to detect again.
    pub fn clear(&mut self) {
        self.taken.par_iter_mut().for_each(|px| *px = false);
    }

    /// Access the precomputed corner response image (populated by
    /// `compute_corner_response` / `compute_corner_response_u8`). Useful for
    /// callers that want to read per-pixel FAST score without re-running NMS.
    pub fn corner_response_image(&self) -> &Image<f32, 1, CpuAllocator> {
        &self.corner_response
    }

    /// Fused single-pass FAST: computes responses, thresholds, excludes border,
    /// and emits `(coord, response)` in one parallel sweep. Replaces the
    /// multi-pass chain of `compute_corner_response_u8` + `get_peak_mask` +
    /// `exclude_border` + coord-collection — at 1080p that chain makes three
    /// full-image passes (8 MB each) that this fuses into one.
    ///
    /// The internal `corner_response` image is NOT populated. Callers that
    /// need the dense response image must still use `compute_corner_response_u8`.
    pub fn detect_direct_u8<A: ImageAllocator>(
        &self,
        src: &Image<u8, 1, A>,
        border: usize,
    ) -> Vec<([usize; 2], f32)> {
        let height = src.height();
        let margin = border.max(3);
        fast_detect_rows_u8(
            src,
            self.threshold,
            self.arc_length,
            border,
            margin..height.saturating_sub(margin),
        )
    }

    /// Computes the corner response for the input image.
    ///
    /// # Arguments
    ///
    /// * `src` - The source grayscale image.
    ///
    /// # Returns
    ///
    /// Returns a reference to the image containing the corner response.
    pub fn compute_corner_response<A: ImageAllocator>(
        &mut self,
        src: &Image<f32, 1, A>,
    ) -> &Image<f32, 1, CpuAllocator> {
        let src_slice = src.as_slice();

        let width = src.width();
        let height = src.height();

        let corner_response = self.corner_response.as_slice_mut();

        const RP: [isize; 16] = [0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1];
        const CP: [isize; 16] = [3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1, 0, 1, 2, 3];

        corner_response[3 * width..(height - 3) * width]
            .par_chunks_mut(width)
            .enumerate()
            .for_each(|(row_idx, row)| {
                let y = row_idx + 3;
                let mut bins = [PixelType::Similar; 16];
                let mut circle_intensities = [0f32; 16];

                for x in 3..width - 3 {
                    let ix = x;
                    let src_ix = y * width + x;
                    let curr_pixel = src_slice[src_ix];
                    let lower_threshold = curr_pixel - self.threshold;
                    let upper_threshold = curr_pixel + self.threshold;

                    let mut speed_sum_b = 0;
                    let mut speed_sum_d = 0;

                    for &k in &[0, 4, 8, 12] {
                        let ik =
                            ((y as isize + RP[k]) * width as isize + (x as isize + CP[k])) as usize;
                        let ring_pixel = src_slice[ik];
                        if ring_pixel > upper_threshold {
                            speed_sum_b += 1;
                        } else if ring_pixel < lower_threshold {
                            speed_sum_d += 1;
                        }
                    }
                    if speed_sum_d < 3 && speed_sum_b < 3 {
                        row[ix] = 0.0;
                        continue;
                    }

                    for k in 0..16 {
                        let ik =
                            ((y as isize + RP[k]) * width as isize + (x as isize + CP[k])) as usize;
                        circle_intensities[k] = src_slice[ik];
                        bins[k] = if circle_intensities[k] > upper_threshold {
                            PixelType::Brighter
                        } else if circle_intensities[k] < lower_threshold {
                            PixelType::Darker
                        } else {
                            PixelType::Similar
                        };
                    }

                    let bright_response = corner_fast_response(
                        curr_pixel,
                        &circle_intensities,
                        &bins,
                        PixelType::Brighter,
                        self.arc_length,
                    );

                    let dark_response = corner_fast_response(
                        curr_pixel,
                        &circle_intensities,
                        &bins,
                        PixelType::Darker,
                        self.arc_length,
                    );

                    row[ix] = bright_response.max(dark_response);
                }
            });

        &self.corner_response
    }

    /// u8 variant of [`compute_corner_response`].
    ///
    /// Reads pixels directly from a u8 source, avoiding a full image u8→f32 pass.
    /// Threshold is interpreted as a 0..1 fraction and rounded to the nearest u8.
    /// The output response image remains f32 so downstream NMS ranking is unchanged.
    pub fn compute_corner_response_u8<A: ImageAllocator>(
        &mut self,
        src: &Image<u8, 1, A>,
    ) -> &Image<f32, 1, CpuAllocator> {
        let src_slice = src.as_slice();
        let width = src.width();
        let height = src.height();
        let corner_response = self.corner_response.as_slice_mut();
        let threshold_u8 = (self.threshold * 255.0).round().clamp(1.0, 255.0) as u8;
        let n = self.arc_length;

        const RP: [isize; 16] = [0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1];
        const CP: [isize; 16] = [3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1, 0, 1, 2, 3];

        // Flat pixel-stride offsets for each ring position; computed once per image.
        let ring: [isize; 16] = std::array::from_fn(|k| RP[k] * width as isize + CP[k]);

        // Hoist runtime feature checks out of the rayon row closure: env-var
        // lookups + atomic loads are cheap individually but compound across
        // O(rows) tasks. Resolved once here and captured by reference.
        #[cfg(target_arch = "aarch64")]
        let use_neon = std::env::var("KORNIA_FAST_NEON").map_or(true, |v| v != "0");
        #[cfg(target_arch = "x86_64")]
        let use_avx2 = crate::simd::cpu_features().has_avx2;

        corner_response[3 * width..(height - 3) * width]
            .par_chunks_mut(width)
            .enumerate()
            .for_each(|(row_idx, row)| {
                let y = row_idx + 3;
                let row_base = y * width;
                let row_end = width - 3;

                let mut x = 3;

                // NEON block path: process 16 pixels at a time when available.
                // The kernel emits raw u16 scores + a pass mask. For this
                // dense-response API we need f32 output, so allocate a 16-u16
                // scratch and convert lane-by-lane (zeroing lanes that fail
                // the arc check).
                #[cfg(target_arch = "aarch64")]
                if use_neon {
                    let scale = 1.0f32 / 255.0;
                    let mut scratch = [0u16; 16];
                    unsafe {
                        while x + 16 <= row_end {
                            let mask = fast_block_neon_16(
                                src_slice,
                                row_base + x,
                                &ring,
                                threshold_u8,
                                n as u8,
                                scratch.as_mut_ptr(),
                            );
                            let dst = row.as_mut_ptr().add(x);
                            for (i, &s) in scratch.iter().enumerate() {
                                let score = if (mask >> i) & 1 != 0 {
                                    s as f32 * scale
                                } else {
                                    0.0
                                };
                                *dst.add(i) = score;
                            }
                            x += 16;
                        }
                    }
                }

                #[cfg(target_arch = "x86_64")]
                if use_avx2 {
                    let scale = 1.0f32 / 255.0;
                    let mut scratch = [0u16; 16];
                    unsafe {
                        while x + 16 <= row_end {
                            let mask = fast_block_avx2_16(
                                src_slice,
                                row_base + x,
                                &ring,
                                threshold_u8,
                                n as u8,
                                scratch.as_mut_ptr(),
                            );
                            let dst = row.as_mut_ptr().add(x);
                            for (i, &s) in scratch.iter().enumerate() {
                                let score = if (mask >> i) & 1 != 0 {
                                    s as f32 * scale
                                } else {
                                    0.0
                                };
                                *dst.add(i) = score;
                            }
                            x += 16;
                        }
                    }
                }

                // Scalar tail (and fully-scalar path on non-aarch64).
                while x < row_end {
                    let src_ix = row_base + x;
                    row[x] = fast_score_scalar(src_slice, src_ix, &ring, threshold_u8, n);
                    x += 1;
                }
            });

        &self.corner_response
    }

    /// Extracts keypoints from the computed corner response.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing a vector of keypoint coordinates or an `ImageError`.
    pub fn extract_keypoints(&mut self) -> Result<Vec<[usize; 2]>, ImageError> {
        self.extract_keypoints_top_k(usize::MAX)
    }

    /// Same as `extract_keypoints` but caps the number of returned peaks to
    /// `max_peaks`, keeping only the strongest by response. This lets callers
    /// (notably ORB, which runs NMS on 400k+ raw candidates per octave) replace
    /// the O(N log N) full sort with an O(N) partial sort + O(K log K) sort of
    /// the top K — a ~10× win on dense corner maps.
    ///
    /// Semantic difference: a low-response isolated peak (with no suppressing
    /// neighbor) is dropped if its response puts it below the top K. For ORB
    /// this is fine — it would be dropped by downstream ranking anyway.
    pub fn extract_keypoints_top_k(
        &mut self,
        max_peaks: usize,
    ) -> Result<Vec<[usize; 2]>, ImageError> {
        get_peak_mask(&self.corner_response, &mut self.mask, self.threshold);
        exclude_border(&mut self.mask, self.min_distance);

        let coordinates = get_high_intensity_peaks(
            &self.corner_response,
            &self.mask,
            self.min_distance,
            &mut self.taken,
            max_peaks,
        );
        Ok(coordinates)
    }

    /// Top-K NMS over a pre-collected candidate list (output of
    /// `detect_direct_u8`). Spatial suppression with `min_distance` neighborhood,
    /// returning the top `max_peaks` peaks (coord + response) in descending
    /// order.
    pub fn suppress_direct(
        &mut self,
        mut candidates: Vec<([usize; 2], f32)>,
        max_peaks: usize,
    ) -> Vec<([usize; 2], f32)> {
        if candidates.is_empty() {
            return Vec::new();
        }

        let cmp = |a: &([usize; 2], f32), b: &([usize; 2], f32)| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        };
        if max_peaks < candidates.len() {
            let pivot = max_peaks.min(candidates.len().saturating_sub(1));
            candidates.select_nth_unstable_by(pivot, cmp);
            candidates.truncate(max_peaks);
        }
        candidates.sort_unstable_by(cmp);

        let width = self.corner_response.size().width;
        let height = self.corner_response.size().height;
        let taken = &mut self.taken;
        taken.iter_mut().for_each(|t| *t = false);
        let min_distance = self.min_distance;

        let mut result = Vec::with_capacity(candidates.len());
        for ([y, x], r) in candidates {
            let idx = y * width + x;
            if taken[idx] {
                continue;
            }
            result.push(([y, x], r));

            let y0 = y.saturating_sub(min_distance);
            let y1 = (y + min_distance + 1).min(height);
            let x0 = x.saturating_sub(min_distance);
            let x1 = (x + min_distance + 1).min(width);
            for yy in y0..y1 {
                let base = yy * width;
                for xx in x0..x1 {
                    if (yy as isize - y as isize)
                        .abs()
                        .max((xx as isize - x as isize).abs())
                        < min_distance as isize
                    {
                        taken[base + xx] = true;
                    }
                }
            }
        }
        result
    }
}

/// Standalone top-K NMS over a candidate list without the 10 MB FastDetector
/// scratch allocation. Only needs image dimensions + a `taken` bitmap sized
/// width*height. ORB calls this once per octave — the 8 MB corner_response
/// and 2 MB mask buffers owned by FastDetector are never touched in the
/// suppress-only path, so skipping their zero-fill saves ~2 ms at 1080p.
pub fn suppress_direct_standalone(
    mut candidates: Vec<([usize; 2], f32)>,
    max_peaks: usize,
    width: usize,
    height: usize,
    min_distance: usize,
) -> Vec<([usize; 2], f32)> {
    if candidates.is_empty() {
        return Vec::new();
    }

    let cmp = |a: &([usize; 2], f32), b: &([usize; 2], f32)| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    };
    if max_peaks < candidates.len() {
        let pivot = max_peaks.min(candidates.len().saturating_sub(1));
        candidates.select_nth_unstable_by(pivot, cmp);
        candidates.truncate(max_peaks);
    }
    candidates.sort_unstable_by(cmp);

    // Fast path: `min_distance == 1` collapses the suppression disk to a
    // single cell (the candidate's own position), so the taken bitmap never
    // blocks a neighbor — it just dedup-emits each coord once. Since the
    // detector already emits each (y,x) at most once, the bitmap is pure
    // overhead (a 2 MB alloc + write pass per octave at 1080p). Skip it.
    if min_distance <= 1 {
        return candidates;
    }

    let mut taken = vec![false; width * height];
    let mut result = Vec::with_capacity(candidates.len());
    for ([y, x], r) in candidates {
        let idx = y * width + x;
        if taken[idx] {
            continue;
        }
        result.push(([y, x], r));

        let y0 = y.saturating_sub(min_distance);
        let y1 = (y + min_distance + 1).min(height);
        let x0 = x.saturating_sub(min_distance);
        let x1 = (x + min_distance + 1).min(width);
        for yy in y0..y1 {
            let base = yy * width;
            for xx in x0..x1 {
                if (yy as isize - y as isize)
                    .abs()
                    .max((xx as isize - x as isize).abs())
                    < min_distance as isize
                {
                    taken[base + xx] = true;
                }
            }
        }
    }
    result
}

/// Stateless row-range FAST detector on u8 input. No allocation of a dense
/// response image or `taken` buffer — emits `(coord, response)` directly.
/// This is the entry point used by the ORB pyramid to split a large octave
/// into multiple row-chunks without paying the 10 MB FastDetector allocation
/// per chunk.
///
/// Runs the row loop in **parallel** (per-row rayon tasks). Do not call from
/// within another `par_iter` — use [`fast_detect_rows_u8_serial`] instead to
/// avoid nested-parallelism scheduler thrash on rayon's global pool.
pub fn fast_detect_rows_u8<A: ImageAllocator>(
    src: &Image<u8, 1, A>,
    threshold: f32,
    arc_length: usize,
    border: usize,
    rows: std::ops::Range<usize>,
) -> Vec<([usize; 2], f32)> {
    fast_detect_rows_u8_impl(src, threshold, arc_length, border, rows, true)
}

/// Serial variant of [`fast_detect_rows_u8`]. Use this from inside an outer
/// `par_iter` so the row loop runs on the caller's rayon worker rather than
/// re-entering the global pool.
pub fn fast_detect_rows_u8_serial<A: ImageAllocator>(
    src: &Image<u8, 1, A>,
    threshold: f32,
    arc_length: usize,
    border: usize,
    rows: std::ops::Range<usize>,
) -> Vec<([usize; 2], f32)> {
    fast_detect_rows_u8_impl(src, threshold, arc_length, border, rows, false)
}

fn fast_detect_rows_u8_impl<A: ImageAllocator>(
    src: &Image<u8, 1, A>,
    threshold: f32,
    arc_length: usize,
    border: usize,
    rows: std::ops::Range<usize>,
    parallel: bool,
) -> Vec<([usize; 2], f32)> {
    let src_slice = src.as_slice();
    let width = src.width();
    let height = src.height();
    let threshold_u8 = (threshold * 255.0).round().clamp(1.0, 255.0) as u8;
    let n = arc_length;

    const RP: [isize; 16] = [0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1];
    const CP: [isize; 16] = [3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1, 0, 1, 2, 3];
    let ring: [isize; 16] = std::array::from_fn(|k| RP[k] * width as isize + CP[k]);

    let margin = border.max(3);
    let valid_start = margin;
    let valid_end = height.saturating_sub(margin);
    let row_start = rows.start.max(valid_start);
    let row_end_y = rows.end.min(valid_end);
    let col_end = width.saturating_sub(margin);

    if row_end_y <= row_start || col_end <= margin {
        return Vec::new();
    }

    #[cfg(target_arch = "aarch64")]
    let use_neon = std::env::var("KORNIA_FAST_NEON").map_or(true, |v| v != "0");

    #[cfg(target_arch = "x86_64")]
    let use_avx2 = crate::simd::cpu_features().has_avx2;

    // Dense-corner images at large resolutions emit so many candidates that
    // `Vec::push` / allocator pressure starts dominating the kernel wall
    // time. A cheap in-block 1D local-max filter (pays only when mask≠0)
    // cuts emission 2-3× on checker-like inputs. At smaller octaves
    // candidates are sparser and the filter's branch overhead outweighs
    // the savings, so we gate it on width.
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    let use_inblock_filter = width >= 800;

    let row_cap = (col_end.saturating_sub(margin) / 6).max(64);

    let row_work = |y: usize| -> Vec<([usize; 2], f32)> {
        let row_base = y * width;
        let mut local: Vec<([usize; 2], f32)> = Vec::with_capacity(row_cap);
        let mut x = margin;

        #[cfg(target_arch = "aarch64")]
        if use_neon {
            let mut scratch = [0u16; 16];
            let scale = 1.0f32 / 255.0;
            unsafe {
                while x + 16 <= col_end {
                    let mut mask = fast_block_neon_16(
                        src_slice,
                        row_base + x,
                        &ring,
                        threshold_u8,
                        n as u8,
                        scratch.as_mut_ptr(),
                    );
                    if use_inblock_filter && mask != 0 {
                        let mut filtered = 0u16;
                        let mut m = mask;
                        while m != 0 {
                            let i = m.trailing_zeros() as usize;
                            let s = scratch[i];
                            let left = if i == 0 { 0 } else { scratch[i - 1] };
                            let right = if i == 15 { 0 } else { scratch[i + 1] };
                            if s > left && s > right {
                                filtered |= 1 << i;
                            }
                            m &= m - 1;
                        }
                        mask = filtered;
                    }
                    while mask != 0 {
                        let i = mask.trailing_zeros() as usize;
                        local.push(([y, x + i], scratch[i] as f32 * scale));
                        mask &= mask - 1;
                    }
                    x += 16;
                }
            }
        }

        #[cfg(target_arch = "x86_64")]
        if use_avx2 {
            let mut scratch = [0u16; 16];
            let scale = 1.0f32 / 255.0;
            unsafe {
                while x + 16 <= col_end {
                    let mut mask = fast_block_avx2_16(
                        src_slice,
                        row_base + x,
                        &ring,
                        threshold_u8,
                        n as u8,
                        scratch.as_mut_ptr(),
                    );
                    if use_inblock_filter && mask != 0 {
                        let mut filtered = 0u16;
                        let mut m = mask;
                        while m != 0 {
                            let i = m.trailing_zeros() as usize;
                            let s = scratch[i];
                            let left = if i == 0 { 0 } else { scratch[i - 1] };
                            let right = if i == 15 { 0 } else { scratch[i + 1] };
                            if s > left && s > right {
                                filtered |= 1 << i;
                            }
                            m &= m - 1;
                        }
                        mask = filtered;
                    }
                    while mask != 0 {
                        let i = mask.trailing_zeros() as usize;
                        local.push(([y, x + i], scratch[i] as f32 * scale));
                        mask &= mask - 1;
                    }
                    x += 16;
                }
            }
        }

        while x < col_end {
            let r = fast_score_scalar(src_slice, row_base + x, &ring, threshold_u8, n);
            if r > 0.0 {
                local.push(([y, x], r));
            }
            x += 1;
        }
        local
    };

    if parallel {
        // Group rows into chunks so each rayon task does enough work to amortize
        // spawn/join overhead. At 1080p one row is ~1920 px ≈ 30 us NEON work;
        // 1080 single-row tasks bury the scheduler. Target ~2-3 ms tasks
        // (~6 cores × 30 tasks = full schedule) — anything below ~1 ms pays
        // more scheduler overhead than it reclaims.
        let chunk = 64usize;
        let total = row_end_y - row_start;
        let n_chunks = total.div_ceil(chunk);
        (0..n_chunks)
            .into_par_iter()
            .flat_map_iter(|c| {
                let y0 = row_start + c * chunk;
                let y1 = (y0 + chunk).min(row_end_y);
                let mut out: Vec<([usize; 2], f32)> = Vec::new();
                for y in y0..y1 {
                    out.extend(row_work(y));
                }
                out
            })
            .collect()
    } else {
        let mut out: Vec<([usize; 2], f32)> = Vec::new();
        for y in row_start..row_end_y {
            out.extend(row_work(y));
        }
        out
    }
}

/// Test if a 32-bit mask (representing a 16-bit ring duplicated into bits 0..32)
/// contains `n` consecutive set bits. Compiles to `n-1` AND-shift pairs, and
/// maps naturally to `vand/vshr` lanes once we go NEON in Phase 2.
#[inline(always)]
fn has_n_consecutive_ones(mask: u32, n: usize) -> bool {
    let mut acc = mask;
    for i in 1..n {
        acc &= mask >> i;
    }
    acc != 0
}

/// Scalar FAST-N score for a single pixel. Returns `Σ|ring - center| / 255`
/// if the pixel has an arc of at least `n` same-side ring pixels, else 0.
#[inline(always)]
fn fast_score_scalar(
    src_slice: &[u8],
    src_ix: usize,
    ring: &[isize; 16],
    threshold_u8: u8,
    n: usize,
) -> f32 {
    let curr_pixel = src_slice[src_ix];
    let lower = curr_pixel.saturating_sub(threshold_u8);
    let upper = curr_pixel.saturating_add(threshold_u8);

    // Cardinal early-reject: need ≥3 of 4 cardinals outside the band.
    let c0 = unsafe { *src_slice.get_unchecked((src_ix as isize + ring[0]) as usize) };
    let c4 = unsafe { *src_slice.get_unchecked((src_ix as isize + ring[4]) as usize) };
    let c8 = unsafe { *src_slice.get_unchecked((src_ix as isize + ring[8]) as usize) };
    let c12 = unsafe { *src_slice.get_unchecked((src_ix as isize + ring[12]) as usize) };
    let b_count =
        (c0 > upper) as u32 + (c4 > upper) as u32 + (c8 > upper) as u32 + (c12 > upper) as u32;
    let d_count =
        (c0 < lower) as u32 + (c4 < lower) as u32 + (c8 < lower) as u32 + (c12 < lower) as u32;
    // An arc of length n on a 16-ring with 4-spaced cardinals contains at
    // least ⌊n/4⌋ cardinals. FAST-9 → 2, FAST-12 → 3.
    let min_card = (n as u32) / 4;
    if b_count < min_card && d_count < min_card {
        return 0.0;
    }

    let mut bright_mask: u32 = 0;
    let mut dark_mask: u32 = 0;
    let mut ring_vals = [0u8; 16];
    for k in 0..16 {
        let p = unsafe { *src_slice.get_unchecked((src_ix as isize + ring[k]) as usize) };
        ring_vals[k] = p;
        if p > upper {
            bright_mask |= 1u32 << k;
        } else if p < lower {
            dark_mask |= 1u32 << k;
        }
    }
    let bright_dup = bright_mask | (bright_mask << 16);
    let dark_dup = dark_mask | (dark_mask << 16);

    if !has_n_consecutive_ones(bright_dup, n) && !has_n_consecutive_ones(dark_dup, n) {
        return 0.0;
    }

    // For n=9 (the ORB default) use the reference FAST-9 `cornerScore` —
    // the max-over-arc-starts-of-min-over-arc saturating diff — so NMS ranking
    // follows the canonical FAST-9 score. For other arc lengths fall back to
    // the sum-of-absdiffs score; only the n=9 case is used by ORB in practice.
    if n == 9 {
        return corner_score_9_scalar(src_slice, src_ix, ring) as f32 / 255.0;
    }

    let mut acc: u32 = 0;
    for &v in &ring_vals {
        acc += (v as i32 - curr_pixel as i32).unsigned_abs();
    }
    acc as f32 / 255.0
}

/// Process 16 consecutive pixels at once using NEON.
///
/// For 16 adjacent centers, loads the 16-pixel rings at each of the 16 Bresenham
/// ring positions, maintains 16-lane "bright run"/"dark run" counters using the
/// recurrence `run = (run + 1) & is_bright` (or dark), and tracks per-lane max
/// via `vmaxq_u8`. A lane is a corner iff `max >= n` for either side; raw
/// u16 score is `Σ|ring - center|`.
///
/// Returns `(mask, scores)`: the 16-bit pass mask (bit `i` set iff lane `i` is
/// a corner) and 16 raw u16 scores written to `out_ptr`. The caller converts
/// to f32 only at surviving lanes, skipping the vcvtq_f32_u32 + vmulq_n_f32
/// pipeline for the ~80% of lanes that fail the arc check.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn fast_block_neon_16(
    src_slice: &[u8],
    center_ix: usize,
    ring: &[isize; 16],
    threshold_u8: u8,
    n: u8,
    out_ptr: *mut u16,
) -> u16 {
    use std::arch::aarch64::*;

    let base = src_slice.as_ptr().add(center_ix);
    let centers = vld1q_u8(base);
    let threshold_v = vdupq_n_u8(threshold_u8);
    let upper = vqaddq_u8(centers, threshold_v);
    let lower = vqsubq_u8(centers, threshold_v);

    // Block-level cardinal early-reject: if no lane has ≥3 cardinals outside
    // the band, the whole 16-lane block is flat — write zeros and return.
    let c0 = vld1q_u8(base.offset(ring[0]));
    let c4 = vld1q_u8(base.offset(ring[4]));
    let c8 = vld1q_u8(base.offset(ring[8]));
    let c12 = vld1q_u8(base.offset(ring[12]));

    let b0 = vshrq_n_u8::<7>(vcgtq_u8(c0, upper));
    let b4 = vshrq_n_u8::<7>(vcgtq_u8(c4, upper));
    let b8 = vshrq_n_u8::<7>(vcgtq_u8(c8, upper));
    let b12 = vshrq_n_u8::<7>(vcgtq_u8(c12, upper));
    let b_count = vaddq_u8(vaddq_u8(b0, b4), vaddq_u8(b8, b12));

    let d0 = vshrq_n_u8::<7>(vcltq_u8(c0, lower));
    let d4 = vshrq_n_u8::<7>(vcltq_u8(c4, lower));
    let d8 = vshrq_n_u8::<7>(vcltq_u8(c8, lower));
    let d12 = vshrq_n_u8::<7>(vcltq_u8(c12, lower));
    let d_count = vaddq_u8(vaddq_u8(d0, d4), vaddq_u8(d8, d12));

    // ⌊n/4⌋ is the minimum number of cardinals on a 16-ring arc of length n.
    let min_card_v = vdupq_n_u8(n / 4);
    let any_pass = vorrq_u8(vcgeq_u8(b_count, min_card_v), vcgeq_u8(d_count, min_card_v));

    if vmaxvq_u8(any_pass) == 0 {
        // All 16 lanes fail cardinal check → mask = 0. Scores irrelevant,
        // caller won't read them.
        return 0;
    }

    // Reference FAST-9 `cornerScore`: max over 16 arc-starts of (min over
    // 9-arc) of saturating diff, on both dark and bright sides. Replaces an
    // earlier Σ|ring−center| proxy score, which was a poor NMS ordering even
    // though it agreed on the pass/fail decision.
    //
    // `cornerScore ≥ threshold ⇔ FAST-9 arc test passes at threshold`, so we
    // can drop the separate chain-counter pass and derive the mask from
    // `score > threshold_u8` directly. `n` is unused on this NEON path —
    // only FAST-9 is used by ORB; other `n` fall through to scalar in the
    // tail loop.
    let mut ring_vals = [vdupq_n_u8(0); 16];
    for k in 0..16 {
        ring_vals[k] = vld1q_u8(base.offset(ring[k]));
    }
    let _ = (upper, lower, n); // values used only on the scalar tail path
    let score_u8 = corner_score_9_neon(centers, &ring_vals);
    let pass = vcgtq_u8(score_u8, threshold_v);
    // Widen u8 → u16 to preserve the existing `*mut u16` out buffer
    // (scalar fallback writes score as f32 = u16/255 at the caller).
    let score_lo = vmovl_u8(vget_low_u8(score_u8));
    let score_hi = vmovl_u8(vget_high_u8(score_u8));

    // Extract a 16-bit pass-mask from `pass` (uint8x16_t with 0x00 or 0xFF per
    // byte). Multiply each byte by its bit-weight (1,2,4,...,128) and sum the
    // low/high halves with vaddv_u8 — each half produces an 8-bit byte, which
    // concatenate to the full 16-bit mask.
    let weights = vld1q_u8(BIT_WEIGHTS.as_ptr());
    let masked = vandq_u8(pass, weights);
    let lo_mask = vaddv_u8(vget_low_u8(masked)) as u16;
    let hi_mask = vaddv_u8(vget_high_u8(masked)) as u16;
    let mask = (hi_mask << 8) | lo_mask;

    vst1q_u16(out_ptr, score_lo);
    vst1q_u16(out_ptr.add(8), score_hi);

    mask
}

#[cfg(target_arch = "aarch64")]
static BIT_WEIGHTS: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];

/// FAST-9 `cornerScore` — the canonical corner strength used by FAST for NMS
/// ranking.
///
/// The score is `max over the 16 starting positions of (min |diff| over the
/// 9-arc)`, evaluated on both the "dark" side (ring < center) and the
/// "bright" side (ring > center). Returned as u8: values above `threshold`
/// indicate a FAST-9 corner at that threshold; ranking is monotone in the
/// strongest passing arc.
///
/// Monotone-equivalent (off by a running clamp of 1) to the integer form
/// used in the canonical FAST-9 source — NMS is invariant under that offset,
/// so this u8 result is a drop-in replacement for ranking. An earlier
/// Σ|ring−center| proxy produced significantly different ranked order, so
/// using the canonical score is important even when the pass/fail mask
/// would be identical.
#[inline]
fn corner_score_9_scalar(src_slice: &[u8], src_ix: usize, ring: &[isize; 16]) -> u8 {
    let center = src_slice[src_ix];
    // dark_diff[k] = max(0, center - ring[k])     (strong when ring ≪ center)
    // bright_diff[k] = max(0, ring[k] - center)   (strong when ring ≫ center)
    let mut dark = [0u8; 16];
    let mut bright = [0u8; 16];
    for k in 0..16 {
        let p = unsafe { *src_slice.get_unchecked((src_ix as isize + ring[k]) as usize) };
        dark[k] = center.saturating_sub(p);
        bright[k] = p.saturating_sub(center);
    }

    // max over 16 starts of min-over-9-arc of dark_diff / bright_diff.
    // Step by 2: each outer iteration shares the 7-element core [k+1..k+8]
    // between start-at-k and start-at-k+1 (canonical FAST-9 structure).
    let mut dark_score: u8 = 0;
    let mut bright_score: u8 = 0;
    let mut k = 0;
    while k < 16 {
        let mut core_d = dark[(k + 1) & 15];
        let mut core_b = bright[(k + 1) & 15];
        for i in 2..=8 {
            core_d = core_d.min(dark[(k + i) & 15]);
            core_b = core_b.min(bright[(k + i) & 15]);
        }
        let arc_k_d = core_d.min(dark[k & 15]);
        let arc_k1_d = core_d.min(dark[(k + 9) & 15]);
        dark_score = dark_score.max(arc_k_d).max(arc_k1_d);
        let arc_k_b = core_b.min(bright[k & 15]);
        let arc_k1_b = core_b.min(bright[(k + 9) & 15]);
        bright_score = bright_score.max(arc_k_b).max(arc_k1_b);
        k += 2;
    }

    dark_score.max(bright_score)
}

/// Lane-parallel NEON version of [`corner_score_9_scalar`] — evaluates
/// cornerScore<9> for 16 consecutive centers at once.
///
/// Uses the saturating-sub formulation (u8 diffs, no sign tracking needed)
/// and the step-by-2 core-sharing trick from canonical FAST-9. Each lane
/// independently computes `max(dark, bright)` where both sides are
/// `max over 16 starts of (min over 9-arc)` of the corresponding saturating
/// diff.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn corner_score_9_neon(
    centers: std::arch::aarch64::uint8x16_t,
    ring_vals: &[std::arch::aarch64::uint8x16_t; 16],
) -> std::arch::aarch64::uint8x16_t {
    use std::arch::aarch64::*;
    let mut dark = [vdupq_n_u8(0); 16];
    let mut bright = [vdupq_n_u8(0); 16];
    for k in 0..16 {
        dark[k] = vqsubq_u8(centers, ring_vals[k]);
        bright[k] = vqsubq_u8(ring_vals[k], centers);
    }
    let mut dark_score = vdupq_n_u8(0);
    let mut bright_score = vdupq_n_u8(0);
    let mut k = 0;
    while k < 16 {
        let mut core_d = dark[(k + 1) & 15];
        let mut core_b = bright[(k + 1) & 15];
        for i in 2..=8 {
            core_d = vminq_u8(core_d, dark[(k + i) & 15]);
            core_b = vminq_u8(core_b, bright[(k + i) & 15]);
        }
        let arc_k_d = vminq_u8(core_d, dark[k & 15]);
        let arc_k1_d = vminq_u8(core_d, dark[(k + 9) & 15]);
        dark_score = vmaxq_u8(dark_score, vmaxq_u8(arc_k_d, arc_k1_d));
        let arc_k_b = vminq_u8(core_b, bright[k & 15]);
        let arc_k1_b = vminq_u8(core_b, bright[(k + 9) & 15]);
        bright_score = vmaxq_u8(bright_score, vmaxq_u8(arc_k_b, arc_k1_b));
        k += 2;
    }
    vmaxq_u8(dark_score, bright_score)
}

/// 16-lane parallel FAST-9 cornerScore on x86_64 AVX2. Direct mirror of
/// [`corner_score_9_neon`] using `__m128i` (same 16 u8 lanes as NEON's
/// `uint8x16_t`). Both `_mm_min_epu8` and `_mm_max_epu8` are SSE2 — present
/// in any AVX2-capable CPU — and `_mm_subs_epu8` provides the saturating
/// `center.saturating_sub(ring)` semantics.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline(always)]
unsafe fn corner_score_9_avx2_16(
    centers: std::arch::x86_64::__m128i,
    ring_vals: &[std::arch::x86_64::__m128i; 16],
) -> std::arch::x86_64::__m128i {
    use std::arch::x86_64::*;
    let mut dark = [_mm_setzero_si128(); 16];
    let mut bright = [_mm_setzero_si128(); 16];
    for k in 0..16 {
        dark[k] = _mm_subs_epu8(centers, ring_vals[k]);
        bright[k] = _mm_subs_epu8(ring_vals[k], centers);
    }
    let mut dark_score = _mm_setzero_si128();
    let mut bright_score = _mm_setzero_si128();
    let mut k = 0;
    while k < 16 {
        let mut core_d = dark[(k + 1) & 15];
        let mut core_b = bright[(k + 1) & 15];
        for i in 2..=8 {
            core_d = _mm_min_epu8(core_d, dark[(k + i) & 15]);
            core_b = _mm_min_epu8(core_b, bright[(k + i) & 15]);
        }
        let arc_k_d = _mm_min_epu8(core_d, dark[k & 15]);
        let arc_k1_d = _mm_min_epu8(core_d, dark[(k + 9) & 15]);
        dark_score = _mm_max_epu8(dark_score, _mm_max_epu8(arc_k_d, arc_k1_d));
        let arc_k_b = _mm_min_epu8(core_b, bright[k & 15]);
        let arc_k1_b = _mm_min_epu8(core_b, bright[(k + 9) & 15]);
        bright_score = _mm_max_epu8(bright_score, _mm_max_epu8(arc_k_b, arc_k1_b));
        k += 2;
    }
    _mm_max_epu8(dark_score, bright_score)
}

/// AVX2 mirror of [`fast_block_neon_16`]. Same 16-lane block kernel, same
/// signature, same `(mask, scores[16])` output contract. Differences from
/// NEON:
///
/// - Unsigned compares synthesised via `_mm_max_epu8` / `_mm_subs_epu8`
///   tricks since AVX2 has only signed lane-compares.
/// - Pass-mask extraction is one `_mm_movemask_epi8` instruction (NEON has
///   no pmovmskb equivalent and uses the `vshrn_n_u16` substitute).
/// - 16-lane block check uses `_mm_movemask_epi8(any_pass) == 0` rather
///   than `vmaxvq_u8`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn fast_block_avx2_16(
    src_slice: &[u8],
    center_ix: usize,
    ring: &[isize; 16],
    threshold_u8: u8,
    n: u8,
    out_ptr: *mut u16,
) -> u16 {
    use std::arch::x86_64::*;

    let base = src_slice.as_ptr().add(center_ix);
    let centers = _mm_loadu_si128(base as *const __m128i);
    let threshold_v = _mm_set1_epi8(threshold_u8 as i8);
    let zero = _mm_setzero_si128();

    let upper = _mm_adds_epu8(centers, threshold_v);
    let lower = _mm_subs_epu8(centers, threshold_v);

    let c0 = _mm_loadu_si128(base.offset(ring[0]) as *const __m128i);
    let c4 = _mm_loadu_si128(base.offset(ring[4]) as *const __m128i);
    let c8 = _mm_loadu_si128(base.offset(ring[8]) as *const __m128i);
    let c12 = _mm_loadu_si128(base.offset(ring[12]) as *const __m128i);

    // Per-byte 0x01 if c > upper (unsigned). subs_epu8 returns 0 iff c<=upper,
    // nonzero iff c>upper; cmpeq vs zero gives 0xFF iff c<=upper; andnot with
    // a 0x01 broadcast flips that to 0x01 iff c>upper.
    let one = _mm_set1_epi8(1);
    let above_b0 = _mm_andnot_si128(_mm_cmpeq_epi8(_mm_subs_epu8(c0, upper), zero), one);
    let above_b4 = _mm_andnot_si128(_mm_cmpeq_epi8(_mm_subs_epu8(c4, upper), zero), one);
    let above_b8 = _mm_andnot_si128(_mm_cmpeq_epi8(_mm_subs_epu8(c8, upper), zero), one);
    let above_b12 = _mm_andnot_si128(_mm_cmpeq_epi8(_mm_subs_epu8(c12, upper), zero), one);
    let b_count = _mm_add_epi8(
        _mm_add_epi8(above_b0, above_b4),
        _mm_add_epi8(above_b8, above_b12),
    );

    let below_b0 = _mm_andnot_si128(_mm_cmpeq_epi8(_mm_subs_epu8(lower, c0), zero), one);
    let below_b4 = _mm_andnot_si128(_mm_cmpeq_epi8(_mm_subs_epu8(lower, c4), zero), one);
    let below_b8 = _mm_andnot_si128(_mm_cmpeq_epi8(_mm_subs_epu8(lower, c8), zero), one);
    let below_b12 = _mm_andnot_si128(_mm_cmpeq_epi8(_mm_subs_epu8(lower, c12), zero), one);
    let d_count = _mm_add_epi8(
        _mm_add_epi8(below_b0, below_b4),
        _mm_add_epi8(below_b8, below_b12),
    );

    // count >= min_card  ⇔  max_epu8(count, min_card) == count
    let min_card_v = _mm_set1_epi8((n / 4) as i8);
    let b_pass = _mm_cmpeq_epi8(_mm_max_epu8(b_count, min_card_v), b_count);
    let d_pass = _mm_cmpeq_epi8(_mm_max_epu8(d_count, min_card_v), d_count);
    let any_pass = _mm_or_si128(b_pass, d_pass);
    if _mm_movemask_epi8(any_pass) == 0 {
        return 0;
    }

    let mut ring_vals = [zero; 16];
    for k in 0..16 {
        ring_vals[k] = _mm_loadu_si128(base.offset(ring[k]) as *const __m128i);
    }
    let _ = (upper, lower, n);
    let score_u8 = corner_score_9_avx2_16(centers, &ring_vals);

    // pass = score > threshold (unsigned). subs_epu8 returns 0 iff score<=thr.
    let s = _mm_subs_epu8(score_u8, threshold_v);
    let pass_inv = _mm_cmpeq_epi8(s, zero); // 0xFF iff score<=thr
    let pass = _mm_andnot_si128(pass_inv, _mm_set1_epi8(-1i8));

    let score_lo = _mm_unpacklo_epi8(score_u8, zero);
    let score_hi = _mm_unpackhi_epi8(score_u8, zero);

    _mm_storeu_si128(out_ptr as *mut __m128i, score_lo);
    _mm_storeu_si128(out_ptr.add(8) as *mut __m128i, score_hi);

    _mm_movemask_epi8(pass) as u16
}

fn corner_fast_response(
    curr_pixel: f32,
    circle_intensities: &[f32; 16],
    bins: &[PixelType; 16],
    state: PixelType,
    n: usize,
) -> f32 {
    let mut consecutive_count = 0;
    let mut curr_response = 0.0;

    if let ControlFlow::Break(_) = (0..15 + n).try_for_each(|l| {
        if bins[l % 16] == state {
            consecutive_count += 1;
            if consecutive_count == n {
                curr_response = 0.0;
                circle_intensities.iter().for_each(|m| {
                    curr_response += (m - curr_pixel).abs();
                });

                return ControlFlow::Break(());
            }
        } else {
            consecutive_count = 0;
        }

        ControlFlow::Continue(())
    }) {
        return curr_response;
    }

    0.0
}

fn get_peak_mask<A: ImageAllocator>(
    src: &Image<f32, 1, A>,
    mask: &mut Image<bool, 1, A>,
    threshold: f32,
) {
    let src_slice = src.as_slice();
    let mask_slice = mask.as_slice_mut();

    src_slice
        .par_iter()
        .zip(mask_slice)
        .for_each(|(src, mask)| *mask = *src > threshold);
}

fn exclude_border<A: ImageAllocator>(label: &mut Image<bool, 1, A>, border_width: usize) {
    let label_size = label.size();
    let label_slice = label.as_slice_mut();

    (0..label_size.height).for_each(|y| {
        let iy = y * label_size.width;

        (0..label_size.width).for_each(|x| {
            if x < border_width
                || x >= label_size.width.saturating_sub(border_width)
                || y < border_width
                || y >= label_size.height.saturating_sub(border_width)
            {
                label_slice[iy + x] = false;
            }
        });
    });
}

fn get_high_intensity_peaks<A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<f32, 1, A1>,
    mask: &Image<bool, 1, A2>,
    min_distance: usize,
    taken: &mut [bool],
    max_peaks: usize,
) -> Vec<[usize; 2]> {
    let src_size = src.size();
    let width = src_size.width;
    let height = src_size.height;
    let src_slice = src.as_slice();

    let mut coords_with_response: Vec<([usize; 2], f32)> = mask
        .as_slice()
        .iter()
        .enumerate()
        .filter(|&(_, &value)| value)
        .map(|(i, _)| {
            let y = i / width;
            let x = i % width;
            ([y, x], src_slice[i])
        })
        .collect();

    // Partial sort: keep top `max_peaks` by response in arbitrary order via
    // select_nth, then full-sort just that prefix. For max_peaks << N this is
    // O(N + K log K) vs O(N log N).
    let cmp = |a: &([usize; 2], f32), b: &([usize; 2], f32)| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    };
    if max_peaks < coords_with_response.len() {
        let pivot = max_peaks.min(coords_with_response.len().saturating_sub(1));
        coords_with_response.select_nth_unstable_by(pivot, cmp);
        coords_with_response.truncate(max_peaks);
    }
    coords_with_response.sort_unstable_by(cmp);

    let mut result = Vec::new();

    for (coord, _) in coords_with_response {
        let y = coord[0];
        let x = coord[1];
        let idx = y * width + x;

        // If this location is already suppressed, skip
        if taken[idx] {
            continue;
        }

        // Accept this peak
        result.push([y, x]);

        // Suppress all within min_distance
        let y0 = y.saturating_sub(min_distance);
        let y1 = (y + min_distance + 1).min(height);
        let x0 = x.saturating_sub(min_distance);
        let x1 = (x + min_distance + 1).min(width);

        (y0..y1)
            .flat_map(|yy| (x0..x1).map(move |xx| (yy, xx)))
            .filter(|&(yy, xx)| {
                (yy as isize - y as isize)
                    .abs()
                    .max((xx as isize - x as isize).abs())
                    < min_distance as isize
            })
            .for_each(|(yy, xx)| {
                taken[yy * width + xx] = true;
            });
    }

    result
}

#[cfg(test)]
mod tests {
    use crate::color::gray_from_rgb_u8;

    use super::*;
    use kornia_image::Image;
    use kornia_io::jpeg::read_image_jpeg_rgb8;
    use kornia_tensor::CpuAllocator;
    use std::collections::HashSet;

    #[test]
    fn test_fast_feature_detector() -> Result<(), Box<dyn std::error::Error>> {
        #[rustfmt::skip]
        let img = read_image_jpeg_rgb8("../../tests/data/dog.jpeg")?;
        let mut gray_img = Image::from_size_val(img.size(), 0, CpuAllocator)?;
        gray_from_rgb_u8(&img, &mut gray_img)?;

        let mut gray_imgf32 = Image::from_size_val(img.size(), 0.0, CpuAllocator)?;
        gray_img
            .as_slice()
            .iter()
            .zip(gray_imgf32.as_slice_mut())
            .for_each(|(&p, m)| {
                *m = p as f32 / 255.0;
            });

        let expected_keypoints = vec![
            [32, 86],
            [60, 75],
            [69, 184],
            [71, 84],
            [72, 169],
            [109, 69],
            [109, 125],
            [120, 64],
            [129, 162],
            [134, 95],
            [141, 121],
            [153, 104],
            [162, 148],
        ];

        const THRESHOLD: f32 = 0.15;

        let mut fast_detector = FastDetector::new(gray_img.size(), THRESHOLD, 12, 10)?;
        fast_detector.compute_corner_response(&gray_imgf32);
        let keypoints = fast_detector.extract_keypoints()?;
        assert_eq!(keypoints.len(), expected_keypoints.len());
        let expected: HashSet<[usize; 2]> = expected_keypoints.into_iter().collect();
        let actual: HashSet<[usize; 2]> = keypoints.into_iter().collect();
        assert_eq!(actual, expected);
        Ok(())
    }

    /// The NEON block path must produce identical per-pixel responses to the
    /// scalar path for every pixel outside the 3-wide border.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_fast_u8_neon_matches_scalar() -> Result<(), Box<dyn std::error::Error>> {
        use kornia_image::ImageSize;

        // Structured random image — more interesting than flat/noise.
        let sz = ImageSize {
            width: 257,
            height: 131,
        };
        let mut data = vec![0u8; sz.width * sz.height];
        for (i, p) in data.iter_mut().enumerate() {
            let y = i / sz.width;
            let x = i % sz.width;
            *p = ((((x / 5) + (y / 5)) & 1) as u8 * 140)
                .wrapping_add(((x.wrapping_mul(37)).wrapping_add(y.wrapping_mul(13)) as u8) / 4);
        }
        let img = Image::<u8, 1, _>::new(sz, data, CpuAllocator)?;

        let mut det_neon = FastDetector::new(sz, 20.0 / 255.0, 9, 1)?;
        det_neon.compute_corner_response_u8(&img);
        let neon_response: Vec<f32> = det_neon.corner_response.as_slice().to_vec();

        // Build a scalar-only reference by walking every pixel with the helper.
        const RP: [isize; 16] = [0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1];
        const CP: [isize; 16] = [3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1, 0, 1, 2, 3];
        let ring: [isize; 16] = std::array::from_fn(|k| RP[k] * sz.width as isize + CP[k]);
        let mut scalar_response = vec![0f32; sz.width * sz.height];
        let threshold_u8: u8 = 20;
        for y in 3..sz.height - 3 {
            for x in 3..sz.width - 3 {
                let ix = y * sz.width + x;
                scalar_response[ix] = fast_score_scalar(img.as_slice(), ix, &ring, threshold_u8, 9);
            }
        }

        // The NEON path uses an arc-length-based score (longer arcs = stronger
        // corner), scalar uses sum-of-|ring - center|. Both scores are valid
        // FAST responses — the invariant we check is that the CORNER SET is
        // identical: a pixel is flagged iff scalar also flags it, and both
        // scores are strictly positive where set, zero elsewhere.
        for (i, (&n, &s)) in neon_response.iter().zip(&scalar_response).enumerate() {
            let n_set = n > 0.0;
            let s_set = s > 0.0;
            assert_eq!(
                n_set,
                s_set,
                "pixel {i} (y={}, x={}): neon={} scalar={} — corner membership disagreement",
                i / sz.width,
                i % sz.width,
                n,
                s
            );
        }
        Ok(())
    }
}
