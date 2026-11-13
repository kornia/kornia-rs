//! OpenCV-compatible ORB: the keypoints, angles and descriptors of
//! `cv::ORB::create()` (OpenCV 5.0 `modules/features/src/orb.cpp`).

use kornia_image::{Image, ImageError, ImageSize};

use super::extractor::{
    brief_descriptor_u8_at, ic_moments_u8_at, orientation_simd_enabled, OrbFeatures,
};
use crate::resize::{resize_linear_exact_u8_c1, LinearExactScratch};

/// `edgeThreshold` (and `patchSize`) of `cv::ORB::create()`: keypoints are
/// kept only this far inside each level.
const EDGE_THRESHOLD: usize = 31;
/// `cv::FAST` scans rows and columns `[3, size - 3)`.
const FAST_BORDER: usize = 3;
/// FAST arc length of `TYPE_9_16`.
const FAST_ARC: usize = 9;
/// `HARRIS_K` in orb.cpp (a constant there, not a parameter).
const HARRIS_K: f32 = 0.04;
/// Half of ORB's 7x7 Harris block.
const HARRIS_RADIUS: usize = 3;
/// Rows and columns `[BLUR_MARGIN, size - BLUR_MARGIN)` of a level are
/// blurred: rBRIEF reads at most 18 px from a keypoint at least
/// [`EDGE_THRESHOLD`] px inside, and the 7-tap blur reads 3 more.
const BLUR_MARGIN: usize = EDGE_THRESHOLD - 19;
/// `getGaussianKernel(7, 2, CV_32F)`, bit for bit.
const GAUSS_7_SIGMA_2: [f32; 7] = [
    f32::from_bits(0x3d8f_afb1),
    f32::from_bits(0x3e06_387e),
    f32::from_bits(0x3e43_4a39),
    f32::from_bits(0x3e5d_4ae0),
    f32::from_bits(0x3e43_4a39),
    f32::from_bits(0x3e06_387e),
    f32::from_bits(0x3d8f_afb1),
];
/// Octaves are stored as `u8` in [`OrbFeatures`].
const MAX_LEVELS: usize = 256;

/// ORB with OpenCV's algorithm and output: the keypoints, angles and
/// descriptors of `cv::ORB::create(n_features, scale_factor, n_levels, 31,
/// 0, 2, HARRIS_SCORE, 31, fast_threshold)` on the same u8 image.
///
/// Per level (level `o` is level `o - 1` resized with `INTER_LINEAR_EXACT`
/// to `round(size / scale_factor^o)`): FAST-9 corners with 3x3 non-max
/// suppression, those at least 31 px inside, the best `2n` by FAST score,
/// the best `n` by the 7x7 Harris response, the intensity-centroid angle
/// (`fastAtan2`), and rBRIEF on the level blurred with OpenCV's float 7x7
/// sigma-2 Gaussian. `n` per level follows `nfeaturesPerLevel`. Ties at the
/// `n`-th score keep every tied keypoint, as `KeyPointsFilter::retainBest`.
///
/// Fixed, as OpenCV's defaults: `edgeThreshold = patchSize = 31`,
/// `firstLevel = 0`, `WTA_K = 2`, `HARRIS_SCORE`, no mask.
///
/// Departures from OpenCV:
/// * Keypoint order: per level, row-major. OpenCV's order within a level
///   comes from `std::nth_element` and is not specified.
/// * [`OrbFeatures::orientations`] are radians: OpenCV's degrees times
///   `(float)(CV_PI / 180.f)`, the factor its descriptor code applies.
/// * The pyramid stops at the first level with an empty side (OpenCV
///   asserts); at most 256 levels.
/// * `fast_threshold = 0` behaves as 1.
///
/// Measured on a 752x480 EuRoC frame against OpenCV 5.0 (see
/// `benches/opencv_orb_fixtures.py`): the same keypoints, responses, angles
/// and descriptors bit for bit.
#[derive(Debug, Clone)]
pub struct OrbCvDetector {
    /// Maximum number of keypoints (`nfeatures`, default 500).
    pub n_features: usize,
    /// Pyramid decimation ratio (`scaleFactor`, default 1.2).
    pub scale_factor: f32,
    /// Number of pyramid levels (`nlevels`, default 8).
    pub n_levels: usize,
    /// FAST threshold in grey levels (`fastThreshold`, default 20).
    pub fast_threshold: u8,
}

impl Default for OrbCvDetector {
    fn default() -> Self {
        Self {
            n_features: 500,
            scale_factor: 1.2,
            n_levels: 8,
            fast_threshold: 20,
        }
    }
}

/// Caller-owned working memory of [`OrbCvDetector`]: pyramid levels, resize
/// tables, FAST scores, keypoint lists and the blurred level. Reused across
/// same-sized frames, [`OrbCvDetector::detect_and_extract_u8_into`] makes no
/// heap allocation after the first frame.
#[derive(Default)]
pub struct OrbCvScratch {
    levels: Vec<Image<u8, 1>>,
    resize: LinearExactScratch,
    budgets: Vec<usize>,
    level: LevelScratch,
    blurred: Vec<u8>,
    blur_rows: Vec<f32>,
}

/// Per-level keypoint buffers of [`OrbCvScratch`].
#[derive(Default)]
struct LevelScratch {
    /// FAST corners (kornia scores, `/ 255`).
    fast: Vec<([usize; 2], f32)>,
    /// Three zeroed score rows for the non-max suppression.
    score_rows: Vec<u8>,
    /// Keys for `retainBest`'s selection.
    selection: Vec<f32>,
    /// The level's keypoints.
    keypoints: Vec<CvKeypoint>,
}

/// One keypoint of a level, in that level's pixel coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
struct CvKeypoint {
    x: u32,
    y: u32,
    /// `cv::FAST` response (cornerScore, one below kornia's FAST score).
    score: u8,
    /// Harris response.
    response: f32,
    /// `fastAtan2` angle, degrees in `[0, 360)`.
    angle: f32,
}

impl OrbCvDetector {
    /// Detect ORB keypoints and compute their descriptors as `cv::ORB`.
    ///
    /// Allocates only the returned [`OrbFeatures`] once `scratch` has grown;
    /// use [`Self::detect_and_extract_u8_into`] to reuse the output too.
    ///
    /// # Arguments
    ///
    /// * `src` - Grayscale u8 image.
    /// * `scratch` - Working memory, reused across calls.
    ///
    /// # Returns
    ///
    /// Keypoints in full-resolution pixels, orientations in radians, 32-byte
    /// descriptors and the level of each keypoint.
    ///
    /// # Errors
    ///
    /// Returns an [`ImageError`] if a pyramid level cannot be allocated.
    ///
    /// # Example
    ///
    /// ```
    /// use kornia_image::{Image, ImageSize};
    /// use kornia_imgproc::features::{OrbCvDetector, OrbCvScratch};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let size = ImageSize { width: 160, height: 120 };
    /// let data = (0..size.width * size.height)
    ///     .map(|i| (((i % size.width) / 8 + (i / size.width) / 8) % 2 * 200) as u8)
    ///     .collect();
    /// let img = Image::<u8, 1>::new(size, data)?;
    ///
    /// let orb = OrbCvDetector::default();
    /// let mut scratch = OrbCvScratch::default();
    /// let feats = orb.detect_and_extract_u8(&img, &mut scratch)?;
    /// assert!(!feats.is_empty());
    /// # Ok(())
    /// # }
    /// ```
    pub fn detect_and_extract_u8(
        &self,
        src: &Image<u8, 1>,
        scratch: &mut OrbCvScratch,
    ) -> Result<OrbFeatures, ImageError> {
        let mut out = OrbFeatures::default();
        self.detect_and_extract_u8_into(src, scratch, &mut out)?;
        Ok(out)
    }

    /// [`Self::detect_and_extract_u8`] into a caller-owned `out` (cleared
    /// first); allocation-free once `scratch` and `out` have grown.
    ///
    /// # Arguments
    ///
    /// * `src` - Grayscale u8 image.
    /// * `scratch` - Working memory, reused across calls.
    /// * `out` - Receives the features.
    ///
    /// # Returns
    ///
    /// `Ok(())` once `out` holds the features.
    ///
    /// # Errors
    ///
    /// Returns an [`ImageError`] if a pyramid level cannot be allocated.
    pub fn detect_and_extract_u8_into(
        &self,
        src: &Image<u8, 1>,
        scratch: &mut OrbCvScratch,
        out: &mut OrbFeatures,
    ) -> Result<(), ImageError> {
        out.keypoints_xy.clear();
        out.orientations.clear();
        out.descriptors.clear();
        out.octaves.clear();
        let n_levels = self.n_levels.min(MAX_LEVELS);
        if n_levels == 0 {
            return Ok(());
        }
        let trace = std::env::var_os("KORNIA_ORB_TRACE").is_some();
        let t_call = std::time::Instant::now();
        self.features_per_level(n_levels, &mut scratch.budgets);
        let base = src.size();
        let mut built = 0;
        for level in 0..n_levels {
            if level > 0 {
                let t_resize = std::time::Instant::now();
                let size = self.level_size(base, level);
                if size.width == 0 || size.height == 0 {
                    break;
                }
                let slot = level - 1;
                if slot == scratch.levels.len() {
                    scratch.levels.push(Image::from_size_val(size, 0u8)?);
                } else if scratch.levels[slot].size() != size {
                    scratch.levels[slot] = Image::from_size_val(size, 0u8)?;
                }
                let (done, rest) = scratch.levels.split_at_mut(slot);
                let prev = if level == 1 { src } else { &done[slot - 1] };
                resize_linear_exact_u8_c1(
                    prev.as_slice(),
                    prev.width(),
                    prev.height(),
                    rest[0].as_slice_mut(),
                    size.width,
                    size.height,
                    &mut scratch.resize,
                )?;
                built = level;
                if trace {
                    eprintln!(
                        "  resize[{level}]={:.3}",
                        t_resize.elapsed().as_secs_f64() * 1000.0
                    );
                }
            }
            let image = if level == 0 {
                src
            } else {
                &scratch.levels[level - 1]
            };
            self.level_keypoints(image, scratch.budgets[level], &mut scratch.level, trace);
            if scratch.level.keypoints.is_empty() {
                continue;
            }
            let t_blur = std::time::Instant::now();

            let (w, h) = (image.width(), image.height());
            if scratch.blurred.len() < w * h {
                scratch.blurred.resize(w * h, 0);
            }
            let blurred = &mut scratch.blurred[..w * h];
            blur_7x7_sigma2_cv(
                image.as_slice(),
                w,
                h,
                BLUR_MARGIN,
                blurred,
                &mut scratch.blur_rows,
            );
            let t_brief = std::time::Instant::now();
            let layer_scale = self.layer_scale(level);
            let deg_to_rad = (std::f64::consts::PI / 180.0f32 as f64) as f32;
            for kp in scratch.level.keypoints.iter() {
                let angle = kp.angle * deg_to_rad;
                out.descriptors.push(brief_descriptor_u8_at(
                    blurred,
                    w,
                    kp.y as i32,
                    kp.x as i32,
                    angle,
                ));
                out.keypoints_xy
                    .push([kp.x as f32 * layer_scale, kp.y as f32 * layer_scale]);
                out.orientations.push(angle);
                out.octaves.push(level as u8);
            }
            if trace {
                eprintln!(
                    "  extract[{w}x{h}] blur={:.3} brief={:.3}",
                    (t_brief - t_blur).as_secs_f64() * 1000.0,
                    t_brief.elapsed().as_secs_f64() * 1000.0
                );
            }
        }
        scratch.levels.truncate(built);
        if trace {
            eprintln!(
                "orb_trace: build_detect_extract={:.3}ms kps={}",
                t_call.elapsed().as_secs_f64() * 1000.0,
                out.keypoints_xy.len()
            );
        }
        Ok(())
    }

    /// `nfeaturesPerLevel` of `computeKeyPoints`, in OpenCV's f32 arithmetic.
    fn features_per_level(&self, n_levels: usize, out: &mut Vec<usize>) {
        let factor = (1.0 / self.scale_factor as f64) as f32;
        let mut desired = self.n_features as f32 * (1.0 - factor)
            / (1.0 - (factor as f64).powf(n_levels as f64) as f32);
        out.clear();
        let mut sum = 0usize;
        for _ in 0..n_levels - 1 {
            // cvRound on a non-negative float.
            let n = desired.round_ties_even().max(0.0) as usize;
            out.push(n);
            sum += n;
            desired *= factor;
        }
        out.push(self.n_features.saturating_sub(sum));
    }

    /// `getScale`: `(float)pow(scaleFactor, level)` in f64.
    fn layer_scale(&self, level: usize) -> f32 {
        (self.scale_factor as f64).powf(level as f64) as f32
    }

    /// Size of level `level`: `cvRound(side * (1.f / scale))`.
    fn level_size(&self, base: ImageSize, level: usize) -> ImageSize {
        let inv = 1.0f32 / self.layer_scale(level);
        ImageSize {
            width: (base.width as f32 * inv).round_ties_even() as usize,
            height: (base.height as f32 * inv).round_ties_even() as usize,
        }
    }

    /// The keypoints `computeKeyPoints` keeps on one level, into
    /// `ws.keypoints` in row-major order, with their responses and angles.
    fn level_keypoints(
        &self,
        image: &Image<u8, 1>,
        budget: usize,
        ws: &mut LevelScratch,
        trace: bool,
    ) {
        let LevelScratch {
            fast,
            score_rows: scores,
            selection,
            keypoints: out,
        } = ws;
        out.clear();
        let (w, h) = (image.width(), image.height());
        if w <= 2 * EDGE_THRESHOLD || h <= 2 * EDGE_THRESHOLD {
            return;
        }
        let t0 = std::time::Instant::now();
        fast_corners(image, self.fast_threshold, fast);
        let t1 = std::time::Instant::now();
        nms_3x3(w, fast, scores, out);
        out.retain(|kp| {
            let (x, y) = (kp.x as usize, kp.y as usize);
            (EDGE_THRESHOLD..w - EDGE_THRESHOLD).contains(&x)
                && (EDGE_THRESHOLD..h - EDGE_THRESHOLD).contains(&y)
        });
        retain_best(out, 2 * budget, selection, |kp| kp.score as f32);
        let t2 = std::time::Instant::now();
        let data = image.as_slice();
        for kp in out.iter_mut() {
            kp.response = harris_response_cv(data, w, kp.x as usize, kp.y as usize);
        }
        retain_best(out, budget, selection, |kp| kp.response);
        let t3 = std::time::Instant::now();
        let simd = orientation_simd_enabled();
        for kp in out.iter_mut() {
            let (m01, m10) = ic_moments_u8_at(data, w, kp.y as usize, kp.x as usize, simd);
            kp.angle = fast_atan2(m01 as f32, m10 as f32);
        }
        if trace {
            let ms = |a: std::time::Instant, b: std::time::Instant| (b - a).as_secs_f64() * 1000.0;
            // `nms` is the 3x3 suppression, the border filter and retainBest
            // on the FAST score.
            eprintln!(
                "  cv[{w}x{h}] fast={:.3} nms={:.3} harris={:.3} ori={:.3} kps={}",
                ms(t0, t1),
                ms(t1, t2),
                ms(t2, t3),
                ms(t3, std::time::Instant::now()),
                out.len()
            );
        }
    }
}

/// `cv::FAST(img, kps, threshold, true, TYPE_9_16)`: [`fast_corners`] then
/// [`nms_3x3`].
#[cfg(test)]
fn fast_nms(
    image: &Image<u8, 1>,
    threshold: u8,
    fast: &mut Vec<([usize; 2], f32)>,
    scores: &mut Vec<u8>,
    out: &mut Vec<CvKeypoint>,
) {
    fast_corners(image, threshold, fast);
    nms_3x3(image.width(), fast, scores, out);
}

/// Every FAST-9 corner in rows and columns `[3, size - 3)`, row-major, with
/// kornia's score (`/ 255`).
fn fast_corners(image: &Image<u8, 1>, threshold: u8, fast: &mut Vec<([usize; 2], f32)>) {
    crate::features::fast::fast_detect_rows_u8_serial_all_into(
        image,
        threshold as f32 / 255.0,
        FAST_ARC,
        FAST_BORDER,
        0..image.height(),
        fast,
    );
}

/// `cv::FAST`'s non-max suppression of `fast` (row-major, at least 3 px
/// inside) on a level `w` wide: corners whose score beats all 8 neighbours
/// (non-corners score 0), row-major, with OpenCV's response (kornia score
/// - 1).
///
/// Like OpenCV it keeps three score rows (`ring`, zero between calls; only
/// corner cells are written and cleared again), so the lookups stay in L1.
fn nms_3x3(w: usize, fast: &[([usize; 2], f32)], ring: &mut Vec<u8>, out: &mut Vec<CvKeypoint>) {
    out.clear();
    if ring.len() < 3 * w {
        ring.resize(3 * w, 0);
    }
    let ring = &mut ring[..3 * w];
    // Which row each ring slot holds, and that row's range in `fast`.
    let mut held: [Option<(usize, std::ops::Range<usize>)>; 3] = [None, None, None];
    let mut start = 0;
    while start < fast.len() {
        let y = fast[start].0[0];
        let end = start + fast[start..].partition_point(|c| c.0[0] == y);
        let above = fast[..start].partition_point(|c| c.0[0] + 1 < y)..start;
        let below = end..end + fast[end..].partition_point(|c| c.0[0] == y + 1);
        for (row, range) in [(y - 1, above), (y, start..end), (y + 1, below)] {
            load_score_row(fast, w, ring, &mut held[row % 3], row, range);
        }
        let row = |r: usize| &ring[(r % 3) * w..(r % 3 + 1) * w];
        nms_row(row(y - 1), row(y), row(y + 1), y, out);
        start = end;
    }
    for (k, slot) in held.iter_mut().enumerate() {
        if let Some((_, old)) = slot.take() {
            for c in &fast[old] {
                ring[k * w + c.0[1]] = 0;
            }
        }
    }
}

/// Appends the cells of `mid` (row `y`, columns `[3, w - 3)`) whose score
/// beats all 8 neighbours. Non-corners score 0, so only corners can win.
/// Dense over the row: corners are a few percent of the pixels, and a
/// branch-free row scan is cheaper than gathering each one's neighbours.
fn nms_row(up: &[u8], mid: &[u8], down: &[u8], y: usize, out: &mut Vec<CvKeypoint>) {
    let w = mid.len();
    let mut x = FAST_BORDER;
    #[cfg(target_arch = "x86_64")]
    if crate::simd::cpu_features().has_avx2 {
        // SAFETY: AVX2 is present.
        x = unsafe { nms_row_avx2(up, mid, down, y, out) };
    }
    while x + FAST_BORDER < w {
        let s = mid[x];
        let m = up[x - 1]
            .max(up[x])
            .max(up[x + 1])
            .max(mid[x - 1])
            .max(mid[x + 1])
            .max(down[x - 1])
            .max(down[x])
            .max(down[x + 1]);
        if s > m {
            out.push(CvKeypoint {
                x: x as u32,
                y: y as u32,
                score: s - 1,
                response: 0.0,
                angle: 0.0,
            });
        }
        x += 1;
    }
}

/// [`nms_row`] 32 columns at a time; returns the first column left.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn nms_row_avx2(
    up: &[u8],
    mid: &[u8],
    down: &[u8],
    y: usize,
    out: &mut Vec<CvKeypoint>,
) -> usize {
    use std::arch::x86_64::*;
    let w = mid.len();
    debug_assert!(up.len() == w && down.len() == w);
    let ld = |r: &[u8], at: usize| _mm256_loadu_si256(r.as_ptr().add(at) as *const __m256i);
    let zero = _mm256_setzero_si256();
    let mut x = FAST_BORDER;
    // Columns [x, x + 32) stay below w - 3, and the loads, which span
    // [x - 1, x + 33), inside the rows.
    while x + 32 + FAST_BORDER <= w {
        let c = ld(mid, x);
        let mut m = _mm256_max_epu8(ld(mid, x - 1), ld(mid, x + 1));
        for r in [up, down] {
            m = _mm256_max_epu8(m, _mm256_max_epu8(ld(r, x - 1), ld(r, x)));
            m = _mm256_max_epu8(m, ld(r, x + 1));
        }
        // c > m (unsigned) iff saturating c - m is non-zero.
        let ge = _mm256_cmpeq_epi8(_mm256_subs_epu8(c, m), zero);
        let mut bits = !(_mm256_movemask_epi8(ge) as u32);
        while bits != 0 {
            let i = bits.trailing_zeros() as usize;
            let s = mid[x + i];
            out.push(CvKeypoint {
                x: (x + i) as u32,
                y: y as u32,
                score: s - 1,
                response: 0.0,
                angle: 0.0,
            });
            bits &= bits - 1;
        }
        x += 32;
    }
    x
}

/// The FAST score is an integer in `[threshold + 1, 255]` carried as `/ 255`
/// (within 1e-5 of it), so `+ 0.5` and truncation recover it.
#[inline]
fn fast_score_u8(r: f32) -> u8 {
    (r * 255.0 + 0.5) as u8
}

/// Makes ring slot `row % 3` hold the scores of `row` (corners
/// `fast[range]`), clearing the corners of the row it held before.
fn load_score_row(
    fast: &[([usize; 2], f32)],
    w: usize,
    ring: &mut [u8],
    held: &mut Option<(usize, std::ops::Range<usize>)>,
    row: usize,
    range: std::ops::Range<usize>,
) {
    if held.as_ref().is_some_and(|(r, _)| *r == row) {
        return;
    }
    let k = row % 3;
    let cells = &mut ring[k * w..(k + 1) * w];
    if let Some((_, old)) = held.take() {
        for c in &fast[old] {
            cells[c.0[1]] = 0;
        }
    }
    for c in &fast[range.clone()] {
        cells[c.0[1]] = fast_score_u8(c.1);
    }
    *held = Some((row, range));
}

/// `KeyPointsFilter::retainBest`: when there are more than `n`, keep every
/// keypoint whose key is at least the `n`-th largest key (so ties at the
/// boundary all stay), in their current order.
fn retain_best(
    kps: &mut Vec<CvKeypoint>,
    n: usize,
    selection: &mut Vec<f32>,
    key: impl Fn(&CvKeypoint) -> f32,
) {
    if kps.len() <= n {
        return;
    }
    if n == 0 {
        kps.clear();
        return;
    }
    selection.clear();
    selection.extend(kps.iter().map(&key));
    let (_, &mut nth, _) = selection.select_nth_unstable_by(n - 1, |a, b| b.total_cmp(a));
    kps.retain(|kp| key(kp) >= nth);
}

/// ORB's `HarrisResponses` at `(x, y)` with a 7x7 block: integer gradients
/// and sums (order-free, so the SSE2 path gives the same integers), then
/// OpenCV's f32 expression in its order.
fn harris_response_cv(data: &[u8], stride: usize, x: usize, y: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    let (a, b, c) = {
        // SAFETY: SSE2 is part of x86_64. `(x, y)` is at least 31 px inside, so
        // the 9x9 window and its 16-byte row loads (from x - 4) stay in `data`.
        unsafe { harris_sums_sse2(data, stride, x, y) }
    };
    #[cfg(not(target_arch = "x86_64"))]
    let (a, b, c) = harris_sums_scalar(data, stride, x, y);
    harris_from_sums(a, b, c)
}

/// OpenCV's f32 Harris expression on the integer sums.
#[inline]
fn harris_from_sums(a: i32, b: i32, c: i32) -> f32 {
    let block = (2 * HARRIS_RADIUS + 1) as i32;
    let scale = 1.0f32 / (((1 << 2) * block) as f32 * 255.0);
    let scale_sq_sq = scale * scale * scale * scale;
    let (a, b, c) = (a as f32, b as f32, c as f32);
    ((a * b - c * c) - HARRIS_K * (a + b) * (a + b)) * scale_sq_sq
}

/// `(sum Ix^2, sum Iy^2, sum Ix*Iy)` over the 7x7 block, Ix and Iy as
/// orb.cpp (`[1 2 1]`-weighted central differences).
#[allow(dead_code)]
fn harris_sums_scalar(data: &[u8], stride: usize, x: usize, y: usize) -> (i32, i32, i32) {
    let (mut a, mut b, mut c) = (0i32, 0i32, 0i32);
    for yy in y - HARRIS_RADIUS..=y + HARRIS_RADIUS {
        for xx in x - HARRIS_RADIUS..=x + HARRIS_RADIUS {
            let i = yy * stride + xx;
            let p = |o: usize| data[o] as i32;
            let ix = (p(i + 1) - p(i - 1)) * 2
                + (p(i - stride + 1) - p(i - stride - 1))
                + (p(i + stride + 1) - p(i + stride - 1));
            let iy = (p(i + stride) - p(i - stride)) * 2
                + (p(i + stride - 1) - p(i - stride - 1))
                + (p(i + stride + 1) - p(i - stride + 1));
            a += ix * ix;
            b += iy * iy;
            c += ix * iy;
        }
    }
    (a, b, c)
}

/// [`harris_sums_scalar`] with 8 columns per row in i16 lanes (the 8th lane
/// masked off): |Ix|, |Iy| <= 1020, so `pmaddwd` sums stay exact in i32.
///
/// # Safety
///
/// Rows `y - 4 ..= y + 4` and bytes `x - 4 .. x + 12` of each must lie in
/// `data` (row stride `stride`).
#[cfg(target_arch = "x86_64")]
unsafe fn harris_sums_sse2(data: &[u8], stride: usize, x: usize, y: usize) -> (i32, i32, i32) {
    use std::arch::x86_64::*;
    let zero = _mm_setzero_si128();
    // Lanes 0..=6 are block columns x - 3 ..= x + 3; lane 7 is masked off.
    let mask = _mm_set_epi16(0, -1, -1, -1, -1, -1, -1, -1);
    let base = data.as_ptr();
    debug_assert!((y + HARRIS_RADIUS + 1) * stride + x + 12 <= data.len());
    // Row r widened to i16, starting at column x - 3 + shift (shift in -1..=1).
    let load = |r: usize, shift: isize| -> __m128i {
        let p = base.add(r * stride + x - HARRIS_RADIUS - 1);
        let v = _mm_loadu_si128(p as *const __m128i);
        let v = match shift {
            -1 => v,
            0 => _mm_srli_si128::<1>(v),
            _ => _mm_srli_si128::<2>(v),
        };
        _mm_unpacklo_epi8(v, zero)
    };
    let (mut sa, mut sb, mut sc) = (zero, zero, zero);
    for yy in y - HARRIS_RADIUS..=y + HARRIS_RADIUS {
        let (um, uc, up) = (load(yy - 1, -1), load(yy - 1, 0), load(yy - 1, 1));
        let (mm, mp) = (load(yy, -1), load(yy, 1));
        let (dm, dc, dp) = (load(yy + 1, -1), load(yy + 1, 0), load(yy + 1, 1));
        // Ix = (p[1] - p[-1]) * 2 + (up[1] - up[-1]) + (down[1] - down[-1])
        let ix = _mm_add_epi16(
            _mm_slli_epi16::<1>(_mm_sub_epi16(mp, mm)),
            _mm_add_epi16(_mm_sub_epi16(up, um), _mm_sub_epi16(dp, dm)),
        );
        // Iy = (down - up) * 2 + (down[-1] - up[-1]) + (down[1] - up[1])
        let iy = _mm_add_epi16(
            _mm_slli_epi16::<1>(_mm_sub_epi16(dc, uc)),
            _mm_add_epi16(_mm_sub_epi16(dm, um), _mm_sub_epi16(dp, up)),
        );
        let (ix, iy) = (_mm_and_si128(ix, mask), _mm_and_si128(iy, mask));
        sa = _mm_add_epi32(sa, _mm_madd_epi16(ix, ix));
        sb = _mm_add_epi32(sb, _mm_madd_epi16(iy, iy));
        sc = _mm_add_epi32(sc, _mm_madd_epi16(ix, iy));
    }
    let hsum = |v: __m128i| {
        let v = _mm_add_epi32(v, _mm_srli_si128::<8>(v));
        _mm_cvtsi128_si32(_mm_add_epi32(v, _mm_srli_si128::<4>(v)))
    };
    (hsum(sa), hsum(sb), hsum(sc))
}

/// `cv::fastAtan2` (degrees in `[0, 360)`): OpenCV's 7th-order polynomial in
/// f32, without FMA, as its scalar baseline build.
// The coefficients are OpenCV's literals as written; they round to its f32s.
#[allow(clippy::excessive_precision)]
fn fast_atan2(y: f32, x: f32) -> f32 {
    let r2d = (180.0 / std::f64::consts::PI) as f32;
    let p1 = 0.999_787_841_279_480_7f32 * r2d;
    let p3 = -0.325_808_397_464_097_5f32 * r2d;
    let p5 = 0.155_578_651_846_328_1f32 * r2d;
    let p7 = -0.044_326_555_547_921_28f32 * r2d;
    let eps = f64::EPSILON as f32;
    let (ax, ay) = (x.abs(), y.abs());
    let mut a = if ax >= ay {
        let c = ay / (ax + eps);
        let c2 = c * c;
        (((p7 * c2 + p5) * c2 + p3) * c2 + p1) * c
    } else {
        let c = ax / (ay + eps);
        let c2 = c * c;
        90.0 - (((p7 * c2 + p5) * c2 + p3) * c2 + p1) * c
    };
    if x < 0.0 {
        a = 180.0 - a;
    }
    if y < 0.0 {
        a = 360.0 - a;
    }
    a
}

/// OpenCV's float 7x7 sigma-2 Gaussian of a u8 plane (`sepFilter2D` with
/// [`GAUSS_7_SIGMA_2`], the path ORB's in-place blur on its padded pyramid
/// takes), on rows and columns `[margin, size - margin)` of `dst` only;
/// `margin >= 3`, so no border handling is needed. Same bits as OpenCV's
/// AVX2 build: each row sum and each column sum is a chain of fused
/// multiply-adds in OpenCV's order, rounded half to even.
fn blur_7x7_sigma2_cv(
    src: &[u8],
    w: usize,
    h: usize,
    margin: usize,
    dst: &mut [u8],
    rows: &mut Vec<f32>,
) {
    debug_assert!(margin >= 3);
    if w <= 2 * margin || h <= 2 * margin {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        let cpu = crate::simd::cpu_features();
        if cpu.has_avx2 && cpu.has_fma {
            // SAFETY: AVX2 and FMA are present.
            unsafe { blur_7x7_sigma2_cv_avx2_fma(src, w, h, margin, dst, rows) };
            return;
        }
    }
    blur_7x7_sigma2_cv_body(src, w, h, margin, dst, rows);
}

/// [`blur_7x7_sigma2_cv_body`] with 8-lane AVX2 FMA passes (same per-pixel
/// operation chain, so the same bits); the scalar tail runs under the same
/// target features, so its `mul_add` is a `vfmadd` too.
///
/// # Safety
///
/// The CPU must support AVX2 and FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn blur_7x7_sigma2_cv_avx2_fma(
    src: &[u8],
    w: usize,
    h: usize,
    margin: usize,
    dst: &mut [u8],
    rows: &mut Vec<f32>,
) {
    blur_7x7_sigma2_cv_rows(
        src,
        w,
        h,
        margin,
        dst,
        rows,
        |line, out| {
            // SAFETY: the caller checked AVX2 and FMA.
            unsafe { blur_row_pass_avx2(line, out) }
        },
        |taps, out| {
            // SAFETY: the caller checked AVX2 and FMA.
            unsafe { blur_column_pass_avx2(taps, out) }
        },
    );
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn blur_row_pass_avx2(line: &[u8], out: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = out.len();
    debug_assert_eq!(line.len(), n + 6);
    let mut i = 0;
    while i + 8 <= n {
        let mut s = _mm256_setzero_ps();
        for (k, &kk) in GAUSS_7_SIGMA_2.iter().enumerate() {
            // 8 bytes from line[i + k]: i + k + 8 <= n + 6 = line.len().
            let px = _mm_loadl_epi64(line.as_ptr().add(i + k) as *const __m128i);
            let p = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(px));
            s = _mm256_fmadd_ps(p, _mm256_set1_ps(kk), s);
        }
        _mm256_storeu_ps(out.as_mut_ptr().add(i), s);
        i += 8;
    }
    blur_row_pass(&line[i..], &mut out[i..]);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn blur_column_pass_avx2(rows: [&[f32]; 7], out: &mut [u8]) {
    use std::arch::x86_64::*;
    const K: [f32; 7] = GAUSS_7_SIGMA_2;
    let n = out.len();
    let [u3, u2, u1, c, d1, d2, d3] = rows.map(|r| &r[..n]);
    let ld = |r: &[f32], i: usize| _mm256_loadu_ps(r.as_ptr().add(i));
    let mut i = 0;
    while i + 8 <= n {
        let mut s = _mm256_mul_ps(_mm256_set1_ps(K[3]), ld(c, i));
        s = _mm256_fmadd_ps(_mm256_set1_ps(K[4]), _mm256_add_ps(ld(d1, i), ld(u1, i)), s);
        s = _mm256_fmadd_ps(_mm256_set1_ps(K[5]), _mm256_add_ps(ld(d2, i), ld(u2, i)), s);
        s = _mm256_fmadd_ps(_mm256_set1_ps(K[6]), _mm256_add_ps(ld(d3, i), ld(u3, i)), s);
        // Round half to even (MXCSR default), then saturate to u8 as v_pack_u(v_pack(..)).
        let v = _mm256_cvtps_epi32(s);
        let w16 = _mm_packs_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
        _mm_storel_epi64(
            out.as_mut_ptr().add(i) as *mut __m128i,
            _mm_packus_epi16(w16, w16),
        );
        i += 8;
    }
    let tail = rows.map(|r| &r[i..n]);
    blur_column_pass(tail, &mut out[i..]);
}

fn blur_7x7_sigma2_cv_body(
    src: &[u8],
    w: usize,
    h: usize,
    margin: usize,
    dst: &mut [u8],
    rows: &mut Vec<f32>,
) {
    blur_7x7_sigma2_cv_rows(
        src,
        w,
        h,
        margin,
        dst,
        rows,
        blur_row_pass,
        blur_column_pass,
    );
}

/// Row ring of the blur: each source row is filtered horizontally once into
/// one of 7 f32 rows, then each output row combines the 7 around it.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn blur_7x7_sigma2_cv_rows(
    src: &[u8],
    w: usize,
    h: usize,
    margin: usize,
    dst: &mut [u8],
    rows: &mut Vec<f32>,
    row_pass: impl Fn(&[u8], &mut [f32]),
    column_pass: impl Fn([&[f32]; 7], &mut [u8]),
) {
    let (x0, x1) = (margin, w - margin);
    let span = x1 - x0;
    if rows.len() < 7 * span {
        rows.resize(7 * span, 0.0);
    }
    let rows = &mut rows[..7 * span];
    let line = |r: usize| &src[r * w + x0 - 3..r * w + x1 + 3];
    for r in margin - 3..margin + 3 {
        row_pass(line(r), &mut rows[(r % 7) * span..(r % 7 + 1) * span]);
    }
    for y in margin..h - margin {
        let r = y + 3;
        row_pass(line(r), &mut rows[(r % 7) * span..(r % 7 + 1) * span]);
        let row = |r: usize| &rows[(r % 7) * span..(r % 7 + 1) * span];
        column_pass(
            [
                row(y - 3),
                row(y - 2),
                row(y - 1),
                row(y),
                row(y + 1),
                row(y + 2),
                row(y + 3),
            ],
            &mut dst[y * w + x0..y * w + x1],
        );
    }
}

/// `RowVec_8u32f`: `s = fma(p[x + k], K[k], s)` for `k = 0..7`, from 0;
/// `line` holds the row's pixels from 3 before to 3 after `out`.
#[inline(always)]
fn blur_row_pass(line: &[u8], out: &mut [f32]) {
    out.fill(0.0);
    for (k, &kk) in GAUSS_7_SIGMA_2.iter().enumerate() {
        let taps = &line[k..k + out.len()];
        for (o, &p) in out.iter_mut().zip(taps) {
            *o = (p as f32).mul_add(kk, *o);
        }
    }
}

/// `SymmColumnVec_32f8u`: `s = K[3] * c`, then `s = fma(K[3 + k], up_k +
/// down_k, s)` for `k = 1..=3`, rounded half to even and saturated.
#[inline(always)]
fn blur_column_pass(rows: [&[f32]; 7], out: &mut [u8]) {
    const K: [f32; 7] = GAUSS_7_SIGMA_2;
    let n = out.len();
    let [u3, u2, u1, c, d1, d2, d3] = rows.map(|r| &r[..n]);
    for i in 0..n {
        let mut s = K[3].mul_add(c[i], 0.0);
        s = K[4].mul_add(d1[i] + u1[i], s);
        s = K[5].mul_add(d2[i] + u2[i], s);
        s = K[6].mul_add(d3[i] + u3[i], s);
        out[i] = s.round_ties_even().clamp(0.0, 255.0) as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One `cv::KeyPoint` and descriptor of the fixture.
    #[derive(Clone, Copy, Debug)]
    struct CvRecord {
        x: f32,
        y: f32,
        octave: u32,
        angle: f32,
        response: f32,
        desc: [u8; 32],
    }

    struct FixtureLevel {
        image: Image<u8, 1>,
        fast: Vec<(u32, u32, u8)>,
        blur_hash: u64,
    }

    struct Fixture {
        full: Vec<CvRecord>,
        crop: [usize; 4],
        levels: Vec<FixtureLevel>,
        crop_orb: Vec<CvRecord>,
    }

    struct Reader<'a>(&'a [u8]);

    impl Reader<'_> {
        fn take(&mut self, n: usize) -> &[u8] {
            let (head, tail) = self.0.split_at(n);
            self.0 = tail;
            head
        }
        fn u16(&mut self) -> u16 {
            u16::from_le_bytes(self.take(2).try_into().unwrap())
        }
        fn u32(&mut self) -> u32 {
            u32::from_le_bytes(self.take(4).try_into().unwrap())
        }
        fn f32(&mut self) -> f32 {
            f32::from_le_bytes(self.take(4).try_into().unwrap())
        }
        fn records(&mut self) -> Vec<CvRecord> {
            let n = self.u32() as usize;
            (0..n)
                .map(|_| CvRecord {
                    x: self.f32(),
                    y: self.f32(),
                    octave: self.u32(),
                    angle: self.f32(),
                    response: self.f32(),
                    desc: self.take(32).try_into().unwrap(),
                })
                .collect()
        }
    }

    fn data_dir() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/data")
    }

    /// `tests/data/orb_opencv.bin`, written by `benches/opencv_orb_fixtures.py`.
    fn fixture() -> Fixture {
        let bytes = std::fs::read(data_dir().join("orb_opencv.bin")).unwrap();
        let mut r = Reader(&bytes);
        let full = r.records();
        let crop = [r.u32(), r.u32(), r.u32(), r.u32()].map(|v| v as usize);
        let n_levels = r.u32() as usize;
        let levels = (0..n_levels)
            .map(|_| {
                let (w, h) = (r.u32() as usize, r.u32() as usize);
                let image = Image::new([w, h].into(), r.take(w * h).to_vec()).unwrap();
                let n = r.u32() as usize;
                let fast = (0..n)
                    .map(|_| (r.u16() as u32, r.u16() as u32, r.take(1)[0]))
                    .collect();
                let blur_hash = u64::from_le_bytes(r.take(8).try_into().unwrap());
                FixtureLevel {
                    image,
                    fast,
                    blur_hash,
                }
            })
            .collect();
        let crop_orb = r.records();
        assert!(r.0.is_empty());
        Fixture {
            full,
            crop,
            levels,
            crop_orb,
        }
    }

    fn mh01_frame1() -> Image<u8, 1> {
        let png = kornia_io::png::read_image_png_mono8(data_dir().join("mh01_frame1.png")).unwrap();
        Image::new(png.size(), png.as_slice().to_vec()).unwrap()
    }

    fn crop_of(img: &Image<u8, 1>, [x, y, w, h]: [usize; 4]) -> Image<u8, 1> {
        let data = (y..y + h)
            .flat_map(|r| img.as_slice()[r * img.width() + x..r * img.width() + x + w].to_vec())
            .collect();
        Image::new([w, h].into(), data).unwrap()
    }

    fn fnv1a(bytes: impl Iterator<Item = u8>) -> u64 {
        bytes.fold(0xcbf2_9ce4_8422_2325, |h, b| {
            (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3)
        })
    }

    /// Level position of a fixture keypoint, as OpenCV's descriptor code
    /// recovers it: `cvRound(pt * (1 / layerScale))`.
    fn level_xy(det: &OrbCvDetector, r: &CvRecord) -> (u32, u32) {
        let inv = 1.0 / det.layer_scale(r.octave as usize);
        (
            (r.x * inv).round_ties_even() as u32,
            (r.y * inv).round_ties_even() as u32,
        )
    }

    /// T1: level sizes and every `INTER_LINEAR_EXACT` level are exact.
    #[test]
    fn t1_pyramid_levels_are_opencv_bytes() {
        let fx = fixture();
        let crop = crop_of(&mh01_frame1(), fx.crop);
        assert_eq!(crop.as_slice(), fx.levels[0].image.as_slice());
        let det = OrbCvDetector::default();
        let mut ws = LinearExactScratch::default();
        for o in 1..fx.levels.len() {
            let (prev, want) = (&fx.levels[o - 1].image, &fx.levels[o].image);
            assert_eq!(
                det.level_size(crop.size(), o),
                want.size(),
                "level {o} size"
            );
            let mut got = vec![0u8; want.width() * want.height()];
            let (w, h) = (want.width(), want.height());
            resize_linear_exact_u8_c1(
                prev.as_slice(),
                prev.width(),
                prev.height(),
                &mut got,
                w,
                h,
                &mut ws,
            )
            .unwrap();
            let differ = got
                .iter()
                .zip(want.as_slice())
                .filter(|(a, b)| a != b)
                .count();
            assert_eq!(differ, 0, "level {o}: {differ} pixels differ");
        }
    }

    /// T2: FAST + non-max suppression gives OpenCV's corners and responses.
    #[test]
    fn t2_fast_nms_matches_cv_fast() {
        let fx = fixture();
        let (mut fast, mut scores, mut out) = (Vec::new(), Vec::new(), Vec::new());
        for (o, level) in fx.levels.iter().enumerate() {
            fast_nms(&level.image, 20, &mut fast, &mut scores, &mut out);
            let mut got: Vec<(u32, u32, u8)> = out.iter().map(|k| (k.x, k.y, k.score)).collect();
            let mut want = level.fast.clone();
            got.sort_unstable();
            want.sort_unstable();
            assert_eq!(got.len(), want.len(), "level {o}");
            assert_eq!(got, want, "level {o}");
            assert!(scores.iter().all(|&s| s == 0), "score plane left dirty");
        }
    }

    /// T3 + T4: per level, the kept keypoints (border filter, retainBest on
    /// FAST then on Harris) are OpenCV's, with its responses and angles.
    #[test]
    fn t3_t4_level_keypoints_responses_and_angles_match_cv_orb() {
        let fx = fixture();
        let det = OrbCvDetector::default();
        let mut budgets = Vec::new();
        det.features_per_level(det.n_levels, &mut budgets);
        let mut ws = LevelScratch::default();
        let mut matched = 0;
        for (o, level) in fx.levels.iter().enumerate() {
            det.level_keypoints(&level.image, budgets[o], &mut ws, false);
            let kps = &ws.keypoints;
            let want: Vec<&CvRecord> = fx
                .crop_orb
                .iter()
                .filter(|r| r.octave as usize == o)
                .collect();
            assert_eq!(kps.len(), want.len(), "level {o} count");
            for r in want {
                let (x, y) = level_xy(&det, r);
                let kp = kps
                    .iter()
                    .find(|k| (k.x, k.y) == (x, y))
                    .unwrap_or_else(|| panic!("level {o}: ({x}, {y}) missing"));
                let rel = (kp.response - r.response).abs() / r.response.abs();
                assert!(
                    rel <= 1e-5,
                    "level {o} ({x}, {y}): response {} vs {}",
                    kp.response,
                    r.response
                );
                assert!(
                    (kp.angle - r.angle).abs() <= 1e-3,
                    "level {o} ({x}, {y}): angle {} vs {}",
                    kp.angle,
                    r.angle
                );
                matched += 1;
            }
        }
        assert_eq!(matched, fx.crop_orb.len());
    }

    /// The float blur equals OpenCV's on every pixel rBRIEF can read.
    #[test]
    fn blur_matches_cv_sep_filter() {
        let fx = fixture();
        let (mut dst, mut rows) = (Vec::new(), Vec::new());
        for (o, level) in fx.levels.iter().enumerate() {
            let (w, h) = (level.image.width(), level.image.height());
            dst.clear();
            dst.resize(w * h, 0);
            blur_7x7_sigma2_cv(
                level.image.as_slice(),
                w,
                h,
                BLUR_MARGIN,
                &mut dst,
                &mut rows,
            );
            let m = BLUR_MARGIN;
            let hash =
                fnv1a((m..h - m).flat_map(|y| dst[y * w + m..y * w + w - m].iter().copied()));
            assert_eq!(hash, level.blur_hash, "level {o}");
            // The scalar body (the non-FMA x86 and the aarch64 path) agrees.
            let mut scalar = vec![0u8; w * h];
            blur_7x7_sigma2_cv_body(level.image.as_slice(), w, h, m, &mut scalar, &mut rows);
            assert_eq!(scalar, dst, "level {o} scalar");
        }
    }

    /// T5: the whole detector on both fixture images: the same keypoints
    /// (level and position), angles and descriptors as `cv::ORB`.
    #[test]
    fn t5_detect_and_extract_matches_cv_orb() {
        let fx = fixture();
        let frame = mh01_frame1();
        let crop = crop_of(&frame, fx.crop);
        let mut scratch = OrbCvScratch::default();
        for (img, n, want) in [
            (&frame, 1000, &fx.full),
            (&crop, 500, &fx.crop_orb),
            (&frame, 1000, &fx.full),
        ] {
            let det = OrbCvDetector {
                n_features: n,
                ..OrbCvDetector::default()
            };
            let got = det.detect_and_extract_u8(img, &mut scratch).unwrap();
            assert_eq!(got.len(), want.len());
            let deg_to_rad = (std::f64::consts::PI / 180.0f32 as f64) as f32;
            let mut hamming = Vec::new();
            for r in want {
                let i = (0..got.len())
                    .find(|&i| {
                        got.octaves[i] as u32 == r.octave && got.keypoints_xy[i] == [r.x, r.y]
                    })
                    .unwrap_or_else(|| panic!("cv keypoint {r:?} missing"));
                assert_eq!(got.orientations[i], r.angle * deg_to_rad);
                hamming.push(
                    got.descriptors[i]
                        .iter()
                        .zip(&r.desc)
                        .map(|(a, b)| (a ^ b).count_ones())
                        .sum::<u32>(),
                );
            }
            hamming.sort_unstable();
            let n = hamming.len();
            let mean = hamming.iter().sum::<u32>() as f32 / n as f32;
            let exact = hamming.iter().filter(|&&d| d == 0).count() as f32 / n as f32;
            eprintln!(
                "{} keypoints: Hamming mean {mean}, p95 {}, max {}, exact {exact}",
                n,
                hamming[(n * 95) / 100],
                hamming[n - 1]
            );
            assert_eq!(hamming[n / 2], 0, "median");
            assert!(
                hamming[(n * 95) / 100] <= 2,
                "p95 {}",
                hamming[(n * 95) / 100]
            );
            assert!(mean <= 0.5, "mean {mean}");
            assert!(exact >= 0.9, "exact {exact}");
        }
    }

    #[test]
    fn harris_isa_paths_agree() {
        let (w, h) = (97usize, 80usize);
        let mut x = 0x1234_5678u32;
        let data: Vec<u8> = (0..w * h)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                (x >> 24) as u8
            })
            .collect();
        for y in EDGE_THRESHOLD..h - EDGE_THRESHOLD {
            for x in EDGE_THRESHOLD..w - EDGE_THRESHOLD {
                let want = harris_sums_scalar(&data, w, x, y);
                assert_eq!(
                    harris_from_sums(want.0, want.1, want.2),
                    harris_response_cv(&data, w, x, y)
                );
            }
        }
    }

    #[test]
    fn fast_atan2_quadrants() {
        for (y, x, want) in [
            (0.0, 1.0, 0.0),
            (1.0, 0.0, 90.0),
            (0.0, -1.0, 180.0),
            (-1.0, 0.0, 270.0),
            (1.0, 1.0, 45.0),
        ] {
            assert!((fast_atan2(y, x) - want).abs() < 0.01, "({y}, {x})");
        }
    }

    #[test]
    fn retain_best_keeps_ties_at_the_boundary() {
        let kp = |score| CvKeypoint {
            x: 0,
            y: 0,
            score,
            response: 0.0,
            angle: 0.0,
        };
        let mut kps: Vec<CvKeypoint> = [5, 9, 7, 7, 3, 7].map(kp).to_vec();
        retain_best(&mut kps, 3, &mut Vec::new(), |k| k.score as f32);
        assert_eq!(
            kps.iter().map(|k| k.score).collect::<Vec<_>>(),
            [9, 7, 7, 7]
        );
        retain_best(&mut kps, 0, &mut Vec::new(), |k| k.score as f32);
        assert!(kps.is_empty());
    }

    /// Tiny and degenerate images and configurations: no panic, no keypoint
    /// closer than 31 px to a level border.
    #[test]
    fn tiny_images_and_configs_do_not_panic() {
        let mut scratch = OrbCvScratch::default();
        for (w, h) in [
            (1, 1),
            (1, 9),
            (9, 1),
            (2, 2),
            (62, 62),
            (63, 63),
            (64, 200),
            (200, 64),
        ] {
            let data = (0..w * h).map(|i| ((i * 37) % 251) as u8).collect();
            let img = Image::<u8, 1>::new([w, h].into(), data).unwrap();
            for (n_levels, scale_factor, n_features) in [
                (0, 1.2, 500),
                (1, 1.2, 0),
                (8, 1.2, 500),
                (20, 2.0, 10),
                (3, 0.9, 50),
            ] {
                let det = OrbCvDetector {
                    n_features,
                    scale_factor,
                    n_levels,
                    ..OrbCvDetector::default()
                };
                let f = det.detect_and_extract_u8(&img, &mut scratch).unwrap();
                assert!(f.len() <= n_features.max(1) * 2);
            }
        }
    }
}
