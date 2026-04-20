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
        fast_detect_rows_u8(src, self.threshold, self.arc_length, border, margin..height.saturating_sub(margin))
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

        corner_response[3 * width..(height - 3) * width]
            .par_chunks_mut(width)
            .enumerate()
            .for_each(|(row_idx, row)| {
                let y = row_idx + 3;
                let row_base = y * width;
                let row_end = width - 3;

                let mut x = 3;

                // NEON block path: process 16 pixels at a time when available.
                #[cfg(target_arch = "aarch64")]
                if std::env::var("KORNIA_FAST_NEON").map_or(true, |v| v != "0") {
                    unsafe {
                        while x + 16 <= row_end {
                            fast_block_neon_16(
                                src_slice,
                                row_base + x,
                                &ring,
                                threshold_u8,
                                n as u8,
                                row.as_mut_ptr().add(x),
                            );
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

    let row_cap = (col_end.saturating_sub(margin) / 6).max(64);

    let row_work = |y: usize| -> Vec<([usize; 2], f32)> {
        let row_base = y * width;
        let mut local: Vec<([usize; 2], f32)> = Vec::with_capacity(row_cap);
        let mut x = margin;

        #[cfg(target_arch = "aarch64")]
        if use_neon {
            let mut scratch = [0f32; 16];
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
                    // Iterate only set bits via ctz — O(popcount) instead of
                    // O(16) per block. Big win in dense-corner regions where
                    // ~20% of pixels are FAST candidates.
                    while mask != 0 {
                        let i = mask.trailing_zeros() as usize;
                        local.push(([y, x + i], scratch[i]));
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
        (row_start..row_end_y)
            .into_par_iter()
            .flat_map_iter(row_work)
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
    let b_count = (c0 > upper) as u32
        + (c4 > upper) as u32
        + (c8 > upper) as u32
        + (c12 > upper) as u32;
    let d_count = (c0 < lower) as u32
        + (c4 < lower) as u32
        + (c8 < lower) as u32
        + (c12 < lower) as u32;
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
/// via `vmaxq_u8`. A lane is a corner iff `max >= n` for either side; response
/// is `Σ|ring - center| / 255`, masked to 0 where the arc check fails.
///
/// Returns a 16-bit mask with bit `i` set iff lane `i` is a corner — lets the
/// caller iterate only surviving lanes via `trailing_zeros` instead of a fixed
/// 16-iteration branch chain.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn fast_block_neon_16(
    src_slice: &[u8],
    center_ix: usize,
    ring: &[isize; 16],
    threshold_u8: u8,
    n: u8,
    out_ptr: *mut f32,
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
        // All 16 lanes fail cardinal check → mask = 0. Scores irrelevant.
        return 0;
    }

    let one_v = vdupq_n_u8(1);
    let mut bright_run = vdupq_n_u8(0);
    let mut bright_max = vdupq_n_u8(0);
    let mut dark_run = vdupq_n_u8(0);
    let mut dark_max = vdupq_n_u8(0);
    let mut acc_lo = vdupq_n_u16(0);
    let mut acc_hi = vdupq_n_u16(0);

    // 24 iterations = 16 + (9 - 1), enough to wrap any arc of length n ≤ 9.
    // For each iter: load ring pixel, advance run counters, track max, also
    // accumulate |v - center| into u16 halves for the first 16 iters.
    for k in 0..24usize {
        let offset = ring[k % 16];
        let vals = vld1q_u8(base.offset(offset));
        let is_bright = vcgtq_u8(vals, upper);
        let is_dark = vcltq_u8(vals, lower);
        bright_run = vandq_u8(vaddq_u8(bright_run, one_v), is_bright);
        dark_run = vandq_u8(vaddq_u8(dark_run, one_v), is_dark);
        bright_max = vmaxq_u8(bright_max, bright_run);
        dark_max = vmaxq_u8(dark_max, dark_run);

        if k < 16 {
            let diff = vabdq_u8(vals, centers);
            acc_lo = vaddq_u16(acc_lo, vmovl_u8(vget_low_u8(diff)));
            acc_hi = vaddq_u16(acc_hi, vmovl_u8(vget_high_u8(diff)));
        }
    }

    let n_v = vdupq_n_u8(n);
    let pass = vorrq_u8(vcgeq_u8(bright_max, n_v), vcgeq_u8(dark_max, n_v));

    // Extract a 16-bit pass-mask from `pass` (uint8x16_t with 0x00 or 0xFF per
    // byte). Multiply each byte by its bit-weight (1,2,4,...,128) and sum the
    // low/high halves with vaddv_u8 — each half produces an 8-bit byte, which
    // concatenate to the full 16-bit mask.
    let weights = vld1q_u8(BIT_WEIGHTS.as_ptr());
    let masked = vandq_u8(pass, weights);
    let lo_mask = vaddv_u8(vget_low_u8(masked)) as u16;
    let hi_mask = vaddv_u8(vget_high_u8(masked)) as u16;
    let mask = (hi_mask << 8) | lo_mask;

    // Expand pass (u8, 0/0xFF per lane) to u16 (0/0xFFFF) via signed extension.
    let pass_lo_u16 =
        vreinterpretq_u16_s16(vmovl_s8(vreinterpret_s8_u8(vget_low_u8(pass))));
    let pass_hi_u16 =
        vreinterpretq_u16_s16(vmovl_s8(vreinterpret_s8_u8(vget_high_u8(pass))));

    let acc_lo_m = vandq_u16(acc_lo, pass_lo_u16);
    let acc_hi_m = vandq_u16(acc_hi, pass_hi_u16);

    let scale = 1.0f32 / 255.0;
    let f0 = vmulq_n_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(acc_lo_m))), scale);
    let f1 = vmulq_n_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(acc_lo_m))), scale);
    let f2 = vmulq_n_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(acc_hi_m))), scale);
    let f3 = vmulq_n_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(acc_hi_m))), scale);

    vst1q_f32(out_ptr, f0);
    vst1q_f32(out_ptr.add(4), f1);
    vst1q_f32(out_ptr.add(8), f2);
    vst1q_f32(out_ptr.add(12), f3);

    mask
}

#[cfg(target_arch = "aarch64")]
static BIT_WEIGHTS: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];

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
        let sz = ImageSize { width: 257, height: 131 };
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
        let neon_response: Vec<f32> =
            det_neon.corner_response.as_slice().to_vec();

        // Build a scalar-only reference by walking every pixel with the helper.
        const RP: [isize; 16] = [0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1];
        const CP: [isize; 16] = [3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1, 0, 1, 2, 3];
        let ring: [isize; 16] =
            std::array::from_fn(|k| RP[k] * sz.width as isize + CP[k]);
        let mut scalar_response = vec![0f32; sz.width * sz.height];
        let threshold_u8: u8 = 20;
        for y in 3..sz.height - 3 {
            for x in 3..sz.width - 3 {
                let ix = y * sz.width + x;
                scalar_response[ix] =
                    fast_score_scalar(img.as_slice(), ix, &ring, threshold_u8, 9);
            }
        }

        for (i, (&n, &s)) in neon_response.iter().zip(&scalar_response).enumerate() {
            assert!(
                (n - s).abs() < 1e-6,
                "pixel {i} (y={}, x={}): neon={} scalar={}",
                i / sz.width,
                i % sz.width,
                n,
                s
            );
        }
        Ok(())
    }
}
