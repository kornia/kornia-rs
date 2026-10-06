//! Sparse stereo correspondence: per-keypoint disparity and metric depth from a
//! rectified, row-aligned pair (e.g. [`StereoRectifier`](super::StereoRectifier) output).
//!
//! The algorithm is ORB-SLAM3's `Frame::ComputeStereoMatches`, generalised over the
//! descriptor: for every left keypoint, search the right keypoints on the same row band
//! and inside the disparity range for the best descriptor match, then refine the
//! disparity to sub-pixel with a centred sliding-window SAD and a line or parabola fit, and
//! finally drop matches whose SAD is a gross outlier relative to the median.
//!
//! ```text
//!   left kp (uL, vL)            right view, rows vL ± band
//!        |                  .---------------------------.
//!        v                  |   *      * <- candidates  |
//!   uR in [uL-maxD, uL-minD]'---------------------------'
//!        |                          |
//!        +--- best descriptor ---> uR0 --- SAD + parabola ---> uR
//!                                       disparity = uL - uR,  depth = bf / disparity
//! ```
//!
//! Descriptors are either binary (Hamming distance, e.g. ORB) or `f32` rows compared by
//! dot product, which is cosine similarity for L2-normalised descriptors (e.g. XFeat).
//! Keypoints may carry pyramid octaves; single-scale detectors leave them out.
//!
//! Departures from ORB-SLAM3:
//! - The default sub-pixel fit is equiangular rather than a parabola; see
//!   [`SubPixelFit`]. `SubPixelFit::Parabola` restores ORB-SLAM3's.
//! - The right search window is bounds-checked on BOTH sides. ORB-SLAM3 checks
//!   `uR0 + L - w >= 0`, which leaves `uR0 - L - w` unchecked; detectors with a wide
//!   border margin (ORB) hide this, detectors that fire at the border (XFeat) do not.
//! - Ties go to the lowest right index, a rule the CUDA twin reproduces exactly.
//!
//! Every per-keypoint decision lives in a small scalar helper below whose CUDA mirror in
//! `matcher_cuda.rs` has the same expression tree; the GPU output is identical to this.

use kornia_image::Image;
use rayon::prelude::*;

/// Pyramid levels supported by both backends (ORB-SLAM3 uses 8).
pub const MAX_LEVELS: usize = 8;
/// Upper bound on `half_window + search_range` of [`SadRefine`]. The CUDA twin holds one
/// right-window column per warp lane, so the window `2 * (w + L) + 1` must be `<= 32`.
pub const MAX_SAD_RADIUS: usize = 15;

/// Errors from [`StereoMatcher`].
#[derive(Debug, thiserror::Error)]
pub enum StereoMatchError {
    /// The configuration cannot be used.
    #[error("invalid stereo match config: {0}")]
    InvalidConfig(&'static str),
    /// Left and right descriptors differ in kind or width, or disagree with the keypoints.
    #[error("descriptor mismatch: {0}")]
    Descriptors(&'static str),
    /// A keypoint side's arrays have inconsistent lengths.
    #[error("{side} keypoints: {what} has {got} entries, expected {expected}")]
    Length {
        /// `"left"` or `"right"`.
        side: &'static str,
        /// Which array.
        what: &'static str,
        /// Elements found (keypoints, bytes or floats, per `what`).
        got: usize,
        /// Elements expected.
        expected: usize,
    },
    /// SAD refinement is on but the pyramids are missing, mismatched, or too deep.
    #[error("pyramid: {0}")]
    Pyramid(&'static str),
    /// A device operand was allocated on a stream other than the matcher's.
    #[error("{0} was allocated on another stream")]
    Stream(&'static str),
    /// An image or CUDA operation failed.
    #[error(transparent)]
    Image(#[from] kornia_image::ImageError),
}

/// How the sub-pixel offset is fitted to the three SADs around the integer optimum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SubPixelFit {
    /// Two lines of equal and opposite slope (Shimizu & Okutomi), suited to L1 costs
    /// like SAD whose minimum is V-shaped. On a synthetic rectified pair at disparities
    /// 7.0–7.9 it cut the mean disparity error from 0.065 to 0.042 px (p90 0.127 to
    /// 0.087) against [`Parabola`](Self::Parabola), which pixel-locks towards integers.
    #[default]
    Equiangular,
    /// ORB-SLAM3's parabola, for parity with it.
    Parabola,
}

/// Sub-pixel refinement by centred SAD block matching (ORB-SLAM3's `w` and `L`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SadRefine {
    /// Patch half-size; the patch is `(2w+1)²`. ORB-SLAM3: 5.
    pub half_window: usize,
    /// Search half-range in pixels around the coarse match. ORB-SLAM3: 5.
    pub search_range: usize,
    /// Matches with `sad >= median_factor * median_sad` are dropped; `None` keeps all.
    /// ORB-SLAM3: `1.5 * 1.4`.
    pub median_factor: Option<f32>,
    /// Sub-pixel fit.
    pub fit: SubPixelFit,
}

impl Default for SadRefine {
    fn default() -> Self {
        Self {
            half_window: 5,
            search_range: 5,
            median_factor: Some(1.5 * 1.4),
            fit: SubPixelFit::default(),
        }
    }
}

/// Configuration for [`StereoMatcher`].
#[derive(Debug, Clone, PartialEq)]
pub struct StereoMatchConfig {
    /// Focal length times metric baseline, so `depth = bf / disparity`.
    pub bf: f32,
    /// Smallest accepted disparity in pixels.
    pub min_disparity: f32,
    /// Largest accepted disparity in pixels (exclusive). ORB-SLAM3: `bf / baseline`, i.e.
    /// nothing closer than one baseline.
    pub max_disparity: f32,
    /// Row band half-height at octave 0; scaled by the right keypoint's octave scale.
    /// ORB-SLAM3: 2.
    pub row_band: f32,
    /// Pyramid scale between octaves (1.0 for single-scale detectors).
    pub scale_factor: f32,
    /// Pyramid levels the keypoint octaves index into, `1..=MAX_LEVELS`.
    pub n_levels: usize,
    /// Binary descriptors: accept the best match only if its Hamming distance is below
    /// this. ORB-SLAM3: `(TH_HIGH + TH_LOW) / 2 = 75`.
    pub max_hamming: u32,
    /// Float descriptors: accept the best match only if its dot product exceeds this.
    pub min_similarity: f32,
    /// Sub-pixel refinement; `None` reports the coarse match's `u` as is.
    pub sad: Option<SadRefine>,
}

impl StereoMatchConfig {
    /// ORB-SLAM3's defaults for a rig with focal length `fx` (pixels) and metric
    /// `baseline`, single-scale. Set [`scale_factor`](Self::scale_factor) and
    /// [`n_levels`](Self::n_levels) for pyramid features.
    pub fn new(fx: f32, baseline: f32) -> Self {
        let bf = fx * baseline;
        Self {
            bf,
            min_disparity: 0.0,
            max_disparity: bf / baseline,
            row_band: 2.0,
            scale_factor: 1.0,
            n_levels: 1,
            max_hamming: 75,
            min_similarity: 0.8,
            sad: Some(SadRefine::default()),
        }
    }

    fn validate(&self) -> Result<(), StereoMatchError> {
        let bad = StereoMatchError::InvalidConfig;
        let positive = |v: f32| v.is_finite() && v > 0.0;
        if !positive(self.bf) {
            return Err(bad("bf must be positive"));
        }
        // Written as negated positive conditions so NaN fails every check.
        if !(self.min_disparity >= 0.0 && self.max_disparity > self.min_disparity) {
            return Err(bad("need 0 <= min_disparity < max_disparity"));
        }
        if !(self.row_band >= 0.0 && self.scale_factor >= 1.0) {
            return Err(bad("need row_band >= 0 and scale_factor >= 1"));
        }
        if self.n_levels == 0 || self.n_levels > MAX_LEVELS {
            return Err(bad("n_levels must be in 1..=MAX_LEVELS"));
        }
        if let Some(s) = self.sad {
            if s.search_range == 0 || s.half_window + s.search_range > MAX_SAD_RADIUS {
                return Err(bad(
                    "SAD needs search_range >= 1 and half_window + search_range <= 15",
                ));
            }
            if s.median_factor.is_some_and(|f| !positive(f)) {
                return Err(bad("median_factor must be positive"));
            }
        }
        Ok(())
    }
}

/// Descriptor rows of one view.
#[derive(Debug, Clone, Copy)]
pub enum StereoDescriptors<'a> {
    /// Binary descriptors, `bytes` per keypoint, compared by Hamming distance.
    Binary {
        /// Row-major `n × bytes`.
        data: &'a [u8],
        /// Bytes per descriptor.
        bytes: usize,
    },
    /// Float descriptors, `dim` per keypoint, compared by dot product.
    Float {
        /// Row-major `n × dim`.
        data: &'a [f32],
        /// Floats per descriptor.
        dim: usize,
    },
}

/// Keypoints of one view.
#[derive(Debug, Clone, Copy)]
pub struct StereoKeypoints<'a> {
    /// `[x, y]` in level-0 pixels of the rectified image.
    pub xy: &'a [[f32; 2]],
    /// Pyramid octave per keypoint; `None` means all at octave 0.
    pub octaves: Option<&'a [u8]>,
    /// One descriptor per keypoint.
    pub descriptors: StereoDescriptors<'a>,
}

/// Per-left-keypoint result; `-1` marks "no stereo match" (ORB-SLAM3's sentinel).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StereoMatches {
    /// Matched x in the right image (level-0 pixels).
    pub u_right: Vec<f32>,
    /// Metric depth in the left camera.
    pub depth: Vec<f32>,
    /// Index of the matched right keypoint.
    pub right_idx: Vec<i32>,
}

impl StereoMatches {
    /// Number of left keypoints with a match.
    pub fn num_matched(&self) -> usize {
        self.right_idx.iter().filter(|&&i| i >= 0).count()
    }

    fn reset(&mut self, n: usize) {
        for v in [&mut self.u_right, &mut self.depth] {
            v.clear();
            v.resize(n, -1.0);
        }
        self.right_idx.clear();
        self.right_idx.resize(n, -1);
    }
}

/// Sparse stereo matcher; see the [module docs](self).
#[derive(Debug, Clone)]
pub struct StereoMatcher {
    pub(crate) cfg: StereoMatchConfig,
    /// `scale_factor^o` and its inverse per octave, padded to [`MAX_LEVELS`] with 1.0.
    pub(crate) scale: [f32; MAX_LEVELS],
    pub(crate) inv_scale: [f32; MAX_LEVELS],
}

impl StereoMatcher {
    /// Validates `cfg` and precomputes the octave scale tables.
    ///
    /// # Errors
    /// [`StereoMatchError::InvalidConfig`] on an unusable configuration.
    pub fn new(cfg: StereoMatchConfig) -> Result<Self, StereoMatchError> {
        cfg.validate()?;
        let mut scale = [1.0f32; MAX_LEVELS];
        let mut inv_scale = [1.0f32; MAX_LEVELS];
        for o in 0..cfg.n_levels {
            scale[o] = cfg.scale_factor.powi(o as i32);
            inv_scale[o] = 1.0 / scale[o];
        }
        Ok(Self {
            cfg,
            scale,
            inv_scale,
        })
    }

    /// The configuration in use.
    pub fn config(&self) -> &StereoMatchConfig {
        &self.cfg
    }

    /// Matches `left` against `right` into `out` (resized to `left.xy.len()`).
    ///
    /// `left_pyramid` / `right_pyramid` hold the rectified images, one per octave
    /// (index = octave); only needed when SAD refinement is on, and then must cover
    /// every octave the keypoints use.
    ///
    /// # Errors
    /// [`StereoMatchError`] on inconsistent inputs; matching itself never fails.
    pub fn match_into(
        &self,
        left_pyramid: &[Image<u8, 1>],
        right_pyramid: &[Image<u8, 1>],
        left: &StereoKeypoints,
        right: &StereoKeypoints,
        out: &mut StereoMatches,
    ) -> Result<(), StereoMatchError> {
        check_keypoints(left, "left")?;
        check_keypoints(right, "right")?;
        check_descriptor_pair(&left.descriptors, &right.descriptors)?;
        let rows = self.check_pyramids(left_pyramid, right_pyramid)?;
        out.reset(left.xy.len());
        if left.xy.is_empty() || right.xy.is_empty() || rows <= 0 {
            return Ok(());
        }
        let cfg = &self.cfg;

        // Row buckets in ascending right index. Without an image `rows` is unbounded, so,
        // like the CUDA twin, every right keypoint is scanned against its span instead.
        let spans: Vec<Option<(i32, i32)>> = (0..right.xy.len())
            .map(|ir| {
                let o = octave_of(right.octaves, ir, cfg.n_levels)?;
                Some(row_span(
                    right.xy[ir][1],
                    cfg.row_band * self.scale[o],
                    rows,
                ))
            })
            .collect();
        let bucketed = rows != i32::MAX;
        let row_bucket: Vec<Vec<u32>> = if bucketed {
            // Spans are clipped to [0, rows - 1], so the table never outgrows the image.
            let mut b = vec![Vec::new(); rows as usize];
            for (ir, span) in spans.iter().enumerate() {
                if let Some((lo, hi)) = *span {
                    for r in lo..=hi {
                        b[r as usize].push(ir as u32);
                    }
                }
            }
            b
        } else {
            vec![(0..right.xy.len() as u32).collect()]
        };

        // Left keypoints are independent until the median reject, so they match in
        // parallel; results land by index, keeping the output deterministic.
        let match_one = |il: usize| -> Option<(f32, f32, usize, i32)> {
            let [u_l, v_l] = left.xy[il];
            let o_l = octave_of(left.octaves, il, cfg.n_levels)?;
            let (min_u, max_u) = disparity_window(u_l, cfg.min_disparity, cfg.max_disparity)?;
            let row = left_row(v_l, rows);
            let cands = row_bucket.get(if bucketed { row as usize } else { 0 })?;
            let mut best = Best::new(&left.descriptors);
            for &ir in cands {
                let ir = ir as usize;
                // `row` must lie in the right keypoint's span (redundant when bucketed).
                let Some((lo, hi)) = spans[ir] else { continue };
                if row < lo || row > hi {
                    continue;
                }
                let o_r = octave_of(right.octaves, ir, cfg.n_levels).unwrap_or(0);
                if !octave_gate(left.octaves.is_some(), o_l, o_r) {
                    continue;
                }
                let u_r = right.xy[ir][0];
                if u_r >= min_u && u_r <= max_u {
                    best.offer(&left.descriptors, il, &right.descriptors, ir);
                }
            }
            let ir0 = best.accept(cfg)?;
            let u_r0 = right.xy[ir0][0];
            let (u_r, sad) = match cfg.sad {
                None => (u_r0, 0),
                Some(s) => {
                    let li = left_pyramid.get(o_l)?;
                    let ri = right_pyramid.get(o_l)?;
                    refine_sad(
                        li,
                        ri,
                        u_l,
                        v_l,
                        u_r0,
                        self.inv_scale[o_l],
                        self.scale[o_l],
                        s,
                    )?
                }
            };
            let (u_r, depth) = finish(u_l, u_r, cfg)?;
            Some((u_r, depth, ir0, sad))
        };
        let results: Vec<Option<(f32, f32, usize, i32)>> = (0..left.xy.len())
            .into_par_iter()
            .with_min_len(64)
            .map(match_one)
            .collect();
        let mut accepted: Vec<(i32, usize)> = Vec::new();
        for (il, r) in results.into_iter().enumerate() {
            if let Some((u_r, depth, ir0, sad)) = r {
                out.u_right[il] = u_r;
                out.depth[il] = depth;
                out.right_idx[il] = ir0 as i32;
                accepted.push((sad, il));
            }
        }

        if let Some(factor) = cfg.sad.and_then(|s| s.median_factor) {
            if !accepted.is_empty() {
                let mut sads: Vec<i32> = accepted.iter().map(|a| a.0).collect();
                let k = sads.len() / 2;
                let (_, median, _) = sads.select_nth_unstable(k);
                let th = median_threshold(factor, *median);
                for &(sad, il) in &accepted {
                    if sad as f32 >= th {
                        out.u_right[il] = -1.0;
                        out.depth[il] = -1.0;
                        out.right_idx[il] = -1;
                    }
                }
            }
        }
        Ok(())
    }

    /// Validates the pyramids and returns the level-0 row count (`i32::MAX` when SAD is
    /// off and no image is given).
    pub(crate) fn check_pyramids(
        &self,
        left: &[Image<u8, 1>],
        right: &[Image<u8, 1>],
    ) -> Result<i32, StereoMatchError> {
        let err = StereoMatchError::Pyramid;
        if self.cfg.sad.is_none() {
            // Without SAD the images are not read; rows come from level 0 if given,
            // else are unbounded (the row clamps never bind).
            return Ok(left.first().map_or(i32::MAX, |i| i.height() as i32));
        }
        if left.is_empty() || left.len() != right.len() {
            return Err(err("SAD needs equal-length, non-empty left/right pyramids"));
        }
        if left.len() > MAX_LEVELS || left.len() < self.cfg.n_levels {
            return Err(err("pyramid depth must be n_levels..=MAX_LEVELS"));
        }
        if left.iter().zip(right).any(|(l, r)| l.size() != r.size()) {
            return Err(err("left/right pyramid levels differ in size"));
        }
        Ok(left[0].height() as i32)
    }
}

fn check_keypoints(k: &StereoKeypoints, side: &'static str) -> Result<(), StereoMatchError> {
    let n = k.xy.len();
    if let Some(o) = k.octaves {
        if o.len() != n {
            return Err(StereoMatchError::Length {
                side,
                what: "octaves",
                got: o.len(),
                expected: n,
            });
        }
    }
    let (what, width, len) = match k.descriptors {
        StereoDescriptors::Binary { data, bytes } => ("descriptor bytes", bytes, data.len()),
        StereoDescriptors::Float { data, dim } => ("descriptor floats", dim, data.len()),
    };
    if width == 0 {
        return Err(StereoMatchError::Descriptors("zero-width descriptors"));
    }
    if len != n * width {
        return Err(StereoMatchError::Length {
            side,
            what,
            got: len,
            expected: n * width,
        });
    }
    Ok(())
}

pub(crate) fn check_descriptor_pair(
    l: &StereoDescriptors,
    r: &StereoDescriptors,
) -> Result<(), StereoMatchError> {
    match (l, r) {
        (
            StereoDescriptors::Binary { bytes: a, .. },
            StereoDescriptors::Binary { bytes: b, .. },
        )
        | (StereoDescriptors::Float { dim: a, .. }, StereoDescriptors::Float { dim: b, .. }) => {
            if a != b {
                return Err(StereoMatchError::Descriptors(
                    "left/right descriptor widths differ",
                ));
            }
            Ok(())
        }
        _ => Err(StereoMatchError::Descriptors(
            "left/right descriptor kinds differ",
        )),
    }
}

// ── Scalar helpers, each mirrored by a CUDA function of the same name ────────────────

/// Octave of keypoint `i`, or `None` if it indexes past the configured levels.
fn octave_of(octaves: Option<&[u8]>, i: usize, n_levels: usize) -> Option<usize> {
    let o = octaves.map_or(0, |o| o[i] as usize);
    (o < n_levels).then_some(o)
}

/// Inclusive row span `[floor(y - r), ceil(y + r)]` clipped to the image.
fn row_span(y: f32, r: f32, rows: i32) -> (i32, i32) {
    let lo = ((y - r).floor() as i32).max(0);
    let hi = ((y + r).ceil() as i32).min(rows - 1);
    (lo, hi)
}

/// The row a left keypoint searches.
fn left_row(v: f32, rows: i32) -> i32 {
    (v as i32).clamp(0, rows - 1)
}

/// Accepted right `u` range for a left keypoint, or `None` if it is empty of valid `u`.
fn disparity_window(u_l: f32, min_d: f32, max_d: f32) -> Option<(f32, f32)> {
    let min_u = u_l - max_d;
    let max_u = u_l - min_d;
    (max_u >= 0.0).then_some((min_u, max_u))
}

/// ORB-SLAM3's ±1 octave gate; always open for single-scale keypoints.
fn octave_gate(has_octaves: bool, o_l: usize, o_r: usize) -> bool {
    !has_octaves || o_l.abs_diff(o_r) <= 1
}

fn hamming(a: &[u8], b: &[u8]) -> u32 {
    a.iter().zip(b).map(|(x, y)| (x ^ y).count_ones()).sum()
}

/// f32 dot product in the CUDA twin's warp order: 32 strided partial sums (lane `l`
/// takes `l, l + 32, …`), then the pairwise tree a `shfl_down` reduction performs. Fixing
/// the order is what makes the GPU result bit-identical; it is also more accurate than
/// one long running sum.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut p = [0.0f32; 32];
    for (l, pl) in p.iter_mut().enumerate() {
        let mut k = l;
        while k < a.len() {
            *pl += a[k] * b[k];
            k += 32;
        }
    }
    for off in [16, 8, 4, 2, 1] {
        for i in 0..off {
            p[i] += p[i + off];
        }
    }
    p[0]
}

/// Running best candidate. Strict improvement over an ascending scan keeps the lowest
/// right index among ties.
enum Best {
    Binary { dist: u32, ir: Option<usize> },
    Float { sim: f32, ir: Option<usize> },
}

impl Best {
    fn new(d: &StereoDescriptors) -> Self {
        match d {
            StereoDescriptors::Binary { .. } => Best::Binary {
                dist: u32::MAX,
                ir: None,
            },
            StereoDescriptors::Float { .. } => Best::Float {
                sim: f32::NEG_INFINITY,
                ir: None,
            },
        }
    }

    fn offer(&mut self, l: &StereoDescriptors, il: usize, r: &StereoDescriptors, ir: usize) {
        match (self, l, r) {
            (
                Best::Binary { dist, ir: best },
                StereoDescriptors::Binary { data: ld, bytes },
                StereoDescriptors::Binary { data: rd, .. },
            ) => {
                let d = hamming(
                    &ld[il * bytes..(il + 1) * bytes],
                    &rd[ir * bytes..(ir + 1) * bytes],
                );
                if d < *dist {
                    *dist = d;
                    *best = Some(ir);
                }
            }
            (
                Best::Float { sim, ir: best },
                StereoDescriptors::Float { data: ld, dim },
                StereoDescriptors::Float { data: rd, .. },
            ) => {
                let s = dot(&ld[il * dim..(il + 1) * dim], &rd[ir * dim..(ir + 1) * dim]);
                if s > *sim {
                    *sim = s;
                    *best = Some(ir);
                }
            }
            _ => unreachable!("descriptor kinds checked by check_descriptor_pair"),
        }
    }

    fn accept(&self, cfg: &StereoMatchConfig) -> Option<usize> {
        match *self {
            Best::Binary { dist, ir } => ir.filter(|_| dist < cfg.max_hamming),
            Best::Float { sim, ir } => ir.filter(|_| sim > cfg.min_similarity),
        }
    }
}

/// Centred SAD between the left patch at `(su_l, sv)` and the right patch at
/// `(su_r, sv)`: `Σ |(L - L_centre) - (R - R_centre)|` (ORB-SLAM3 subtracts each patch's
/// centre pixel, so a brightness offset between the cameras cancels).
fn centred_sad(l: &[u8], r: &[u8], stride: usize, su_l: i32, su_r: i32, sv: i32, w: i32) -> i32 {
    let at = |img: &[u8], x: i32, y: i32| img[y as usize * stride + x as usize] as i32;
    let lc = at(l, su_l, sv);
    let rc = at(r, su_r, sv);
    let mut sad = 0i32;
    for dy in -w..=w {
        for dx in -w..=w {
            let a = at(l, su_l + dx, sv + dy) - lc;
            let b = at(r, su_r + dx, sv + dy) - rc;
            sad += (a - b).abs();
        }
    }
    sad
}

/// Sub-pixel right `u` (level 0) and the best SAD, or `None` if the patch does not fit,
/// the optimum sits on the search edge, or the parabola is degenerate.
#[allow(clippy::too_many_arguments)]
fn refine_sad(
    li: &Image<u8, 1>,
    ri: &Image<u8, 1>,
    u_l: f32,
    v_l: f32,
    u_r0: f32,
    inv_scale: f32,
    scale: f32,
    s: SadRefine,
) -> Option<(f32, i32)> {
    let (w, l) = (s.half_window as i32, s.search_range as i32);
    let su_l = (u_l * inv_scale).round() as i32;
    let sv = (v_l * inv_scale).round() as i32;
    let su_r0 = (u_r0 * inv_scale).round() as i32;
    let (iw, ih) = (li.width() as i32, li.height() as i32);
    if !sad_fits(su_l, sv, su_r0, w, l, iw, ih) {
        return None;
    }
    let (ls, rs) = (li.as_slice(), ri.as_slice());
    let mut best_sad = i32::MAX;
    let mut best_inc = 0i32;
    let mut sads = [0i32; 2 * MAX_SAD_RADIUS + 1];
    for inc in -l..=l {
        let sad = centred_sad(ls, rs, li.width(), su_l, su_r0 + inc, sv, w);
        sads[(inc + l) as usize] = sad;
        if sad < best_sad {
            best_sad = sad;
            best_inc = inc;
        }
    }
    if best_inc == -l || best_inc == l {
        return None;
    }
    let i = (best_inc + l) as usize;
    let delta = sub_pixel_offset(s.fit, sads[i - 1], sads[i], sads[i + 1])?;
    Some((sub_pixel_u(scale, su_r0, best_inc, delta), best_sad))
}

/// Both patches and the whole right search window lie inside the image.
fn sad_fits(su_l: i32, sv: i32, su_r0: i32, w: i32, l: i32, iw: i32, ih: i32) -> bool {
    su_l - w >= 0
        && su_l + w < iw
        && sv - w >= 0
        && sv + w < ih
        && su_r0 - l - w >= 0
        && su_r0 + l + w < iw
}

/// Sub-pixel offset of the optimum from three SADs centred on the integer minimum,
/// in `[-1, 1]`, or `None` if the fit is degenerate.
fn sub_pixel_offset(fit: SubPixelFit, s1: i32, s2: i32, s3: i32) -> Option<f32> {
    let (d1, d2, d3) = (s1 as f32, s2 as f32, s3 as f32);
    let denom = match fit {
        SubPixelFit::Parabola => 2.0 * (d1 + d3 - 2.0 * d2),
        SubPixelFit::Equiangular => 2.0 * (d1 - d2).max(d3 - d2),
    };
    if denom == 0.0 {
        return None;
    }
    let delta = (d1 - d3) / denom;
    (-1.0..=1.0).contains(&delta).then_some(delta)
}

fn sub_pixel_u(scale: f32, su_r0: i32, inc: i32, delta: f32) -> f32 {
    scale * (su_r0 as f32 + inc as f32 + delta)
}

/// Disparity gate and depth; ORB-SLAM3 nudges a non-positive disparity to 0.01.
fn finish(u_l: f32, u_r: f32, cfg: &StereoMatchConfig) -> Option<(f32, f32)> {
    let mut u_r = u_r;
    let mut disparity = u_l - u_r;
    if !(disparity >= cfg.min_disparity && disparity < cfg.max_disparity) {
        return None;
    }
    if disparity <= 0.0 {
        disparity = 0.01;
        u_r = u_l - 0.01;
    }
    Some((u_r, cfg.bf / disparity))
}

fn median_threshold(factor: f32, median: i32) -> f32 {
    factor * median as f32
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use kornia_image::ImageSize;
    use rand::{rngs::StdRng, RngExt, SeedableRng};

    pub(crate) const W: usize = 320;
    pub(crate) const H: usize = 240;

    /// Smooth texture with enough gradient for SAD; sampled at fractional `x`.
    pub(crate) fn texture(x: f32, y: f32) -> u8 {
        let v = 128.0
            + 60.0 * (0.21 * x + 0.05 * y).sin()
            + 40.0 * (0.13 * y - 0.07 * x).cos()
            + 20.0 * (0.37 * x * 0.5 + 0.29 * y).sin();
        v.clamp(0.0, 255.0) as u8
    }

    /// [`pair`] with rows 100..140 of the right view corrupted by ±60 of hashed noise:
    /// matches there keep an interior SAD minimum but a high SAD, which is what the
    /// median reject removes.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) fn pair_noisy(d: f32) -> (Image<u8, 1>, Image<u8, 1>) {
        let (l, r) = pair(d);
        let mut rv = r.as_slice().to_vec();
        for y in 100..140 {
            for x in 0..W {
                let h = (x as u32).wrapping_mul(2_654_435_761) ^ (y as u32).wrapping_mul(40_503);
                let n = (h >> 13) % 121;
                rv[y * W + x] = (rv[y * W + x] as i32 + n as i32 - 60).clamp(0, 255) as u8;
            }
        }
        (l, Image::new(r.size(), rv).unwrap())
    }

    /// A rectified pair where right(x, y) = left(x + d, y), i.e. disparity `d` everywhere.
    pub(crate) fn pair(d: f32) -> (Image<u8, 1>, Image<u8, 1>) {
        let size = ImageSize {
            width: W,
            height: H,
        };
        let mut l = vec![0u8; W * H];
        let mut r = vec![0u8; W * H];
        for y in 0..H {
            for x in 0..W {
                l[y * W + x] = texture(x as f32, y as f32);
                r[y * W + x] = texture(x as f32 + d, y as f32);
            }
        }
        (Image::new(size, l).unwrap(), Image::new(size, r).unwrap())
    }

    /// Left keypoints on a grid; right keypoints at the integer-rounded true match, with
    /// shuffled order and distractors, sharing descriptors with their left partner.
    pub(crate) struct Scene {
        pub lxy: Vec<[f32; 2]>,
        pub rxy: Vec<[f32; 2]>,
        pub lbin: Vec<u8>,
        pub rbin: Vec<u8>,
        pub lf: Vec<f32>,
        pub rf: Vec<f32>,
        /// right index of each left keypoint's true partner
        pub partner: Vec<usize>,
    }

    pub(crate) fn scene(d: f32, seed: u64) -> Scene {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut lxy = Vec::new();
        for y in (20..H - 20).step_by(13) {
            for x in (40..W - 20).step_by(17) {
                lxy.push([x as f32, y as f32]);
            }
        }
        let n = lxy.len();
        let desc_f = |rng: &mut StdRng| -> Vec<f32> {
            let v: Vec<f32> = (0..64).map(|_| rng.random::<f32>() - 0.5).collect();
            let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.iter().map(|x| x / n).collect()
        };
        let mut lbin = Vec::new();
        let mut lf = Vec::new();
        for _ in 0..n {
            lbin.extend((0..32).map(|_| rng.random::<u8>()));
            lf.extend(desc_f(&mut rng));
        }
        // Right: partners (perturbed descriptors) + 30% distractors, then shuffled.
        // (xy, binary descriptor, float descriptor, partner left index)
        type Entry = ([f32; 2], Vec<u8>, Vec<f32>, Option<usize>);
        let mut entries: Vec<Entry> = Vec::new();
        for (i, xy) in lxy.iter().enumerate() {
            let mut b = lbin[i * 32..(i + 1) * 32].to_vec();
            b[0] ^= 0b101; // Hamming 2
            let f: Vec<f32> = lf[i * 64..(i + 1) * 64].to_vec();
            entries.push(([(xy[0] - d).round(), xy[1]], b, f, Some(i)));
        }
        for _ in 0..n * 3 / 10 {
            let xy = [rng.random_range(0..W) as f32, rng.random_range(0..H) as f32];
            let b: Vec<u8> = (0..32).map(|_| rng.random::<u8>()).collect();
            entries.push((xy, b, desc_f(&mut rng), None));
        }
        for i in (1..entries.len()).rev() {
            let j = rng.random_range(0..=i);
            entries.swap(i, j);
        }
        let mut partner = vec![usize::MAX; n];
        let (mut rxy, mut rbin, mut rf) = (Vec::new(), Vec::new(), Vec::new());
        for (j, (xy, b, f, p)) in entries.into_iter().enumerate() {
            rxy.push(xy);
            rbin.extend(b);
            rf.extend(f);
            if let Some(i) = p {
                partner[i] = j;
            }
        }
        Scene {
            lxy,
            rxy,
            lbin,
            rbin,
            lf,
            rf,
            partner,
        }
    }

    pub(crate) fn kps<'a>(
        xy: &'a [[f32; 2]],
        bin: &'a [u8],
        f: &'a [f32],
        binary: bool,
    ) -> StereoKeypoints<'a> {
        StereoKeypoints {
            xy,
            octaves: None,
            descriptors: if binary {
                StereoDescriptors::Binary {
                    data: bin,
                    bytes: 32,
                }
            } else {
                StereoDescriptors::Float { data: f, dim: 64 }
            },
        }
    }

    fn cfg() -> StereoMatchConfig {
        // fx 300 px, 10 cm baseline -> bf 30, disparities up to 300 px.
        StereoMatchConfig::new(300.0, 0.1)
    }

    #[test]
    fn recovers_subpixel_disparity_both_descriptor_kinds() -> Result<(), StereoMatchError> {
        let d = 7.3f32;
        let (li, ri) = pair(d);
        let s = scene(d, 7);
        let m = StereoMatcher::new(cfg())?;
        for binary in [true, false] {
            let (l, r) = (
                kps(&s.lxy, &s.lbin, &s.lf, binary),
                kps(&s.rxy, &s.rbin, &s.rf, binary),
            );
            let mut out = StereoMatches::default();
            m.match_into(
                std::slice::from_ref(&li),
                std::slice::from_ref(&ri),
                &l,
                &r,
                &mut out,
            )?;
            let mut n = 0;
            for il in 0..s.lxy.len() {
                if out.right_idx[il] < 0 {
                    continue;
                }
                n += 1;
                assert_eq!(out.right_idx[il] as usize, s.partner[il], "wrong partner");
                let disp = s.lxy[il][0] - out.u_right[il];
                // Equiangular fit: max error 0.19 px over disparities 7.0-7.9.
                assert!((disp - d).abs() < 0.25, "disparity {disp} vs {d}");
                assert!((out.depth[il] - 30.0 / disp).abs() < 1e-4);
            }
            // The median reject drops some by design; most must survive.
            assert!(
                n * 10 >= s.lxy.len() * 7,
                "binary={binary}: only {n}/{} matched",
                s.lxy.len()
            );
        }
        Ok(())
    }

    #[test]
    fn coarse_only_reports_the_matched_keypoint_u() -> Result<(), StereoMatchError> {
        let s = scene(7.0, 3);
        let mut c = cfg();
        c.sad = None;
        let m = StereoMatcher::new(c)?;
        let mut out = StereoMatches::default();
        let (l, r) = (
            kps(&s.lxy, &s.lbin, &s.lf, true),
            kps(&s.rxy, &s.rbin, &s.rf, true),
        );
        m.match_into(&[], &[], &l, &r, &mut out)?;
        for il in 0..s.lxy.len() {
            assert_eq!(out.right_idx[il] as usize, s.partner[il]);
            assert_eq!(out.u_right[il], s.rxy[s.partner[il]][0]);
        }
        Ok(())
    }

    /// A match whose search window would cross the image border is skipped, not read
    /// out of bounds (ORB-SLAM3's guard checks only the far side).
    #[test]
    fn border_window_is_skipped_not_read_out_of_bounds() -> Result<(), StereoMatchError> {
        let (li, ri) = pair(10.0);
        let lxy = [[14.0f32, 100.0]];
        let rxy = [[4.0f32, 100.0]]; // u_r0 - L - w = -6
        let lbin = [7u8; 32];
        let (lf, rf) = ([0.125f32; 64], [0.125f32; 64]);
        let m = StereoMatcher::new(cfg())?;
        let mut out = StereoMatches::default();
        m.match_into(
            &[li],
            &[ri],
            &kps(&lxy, &lbin, &lf, true),
            &kps(&rxy, &lbin, &rf, true),
            &mut out,
        )?;
        assert_eq!(out.right_idx, vec![-1]);
        Ok(())
    }

    #[test]
    fn ties_go_to_the_lowest_right_index() -> Result<(), StereoMatchError> {
        let lxy = [[100.0f32, 50.0]];
        let rxy = [[90.0f32, 50.0], [95.0, 50.0], [92.0, 50.0]];
        let lbin = [0u8; 32];
        let rbin = [[1u8; 32], [0u8; 32], [0u8; 32]].concat();
        let lf = [0.125f32; 64];
        let rf = [[0.0f32; 64], [0.125f32; 64], [0.125f32; 64]].concat();
        let mut c = cfg();
        c.sad = None;
        c.min_similarity = 0.5;
        let m = StereoMatcher::new(c)?;
        for binary in [true, false] {
            let mut out = StereoMatches::default();
            m.match_into(
                &[],
                &[],
                &kps(&lxy, &lbin, &lf, binary),
                &kps(&rxy, &rbin, &rf, binary),
                &mut out,
            )?;
            assert_eq!(out.right_idx, vec![1], "binary={binary}");
        }
        Ok(())
    }

    /// Without an image the rows are unbounded: a keypoint at y = inf must neither size
    /// a row table nor stop the normal pair from matching.
    #[test]
    fn stray_keypoint_at_infinite_y_is_harmless() -> Result<(), StereoMatchError> {
        let lxy = [[100.0f32, f32::INFINITY], [100.0, 50.0]];
        let rxy = [[95.0f32, f32::INFINITY], [95.0, 50.0]];
        let b = [0u8; 64];
        let mut c = cfg();
        c.sad = None;
        let m = StereoMatcher::new(c)?;
        let mut out = StereoMatches::default();
        m.match_into(
            &[],
            &[],
            &kps(&lxy, &b, &[], true),
            &kps(&rxy, &b, &[], true),
            &mut out,
        )?;
        assert_eq!(out.right_idx, vec![-1, 1]);
        Ok(())
    }

    #[test]
    fn octave_gate_and_row_band_scale_with_octave() -> Result<(), StereoMatchError> {
        let lxy = [[100.0f32, 50.0]];
        // Right kp 3 rows off: outside the octave-0 band (2), inside octave 1's (2 * 2).
        let rxy = [[95.0f32, 53.0]];
        let b = [0u8; 32];
        let mut c = cfg();
        c.sad = None;
        c.scale_factor = 2.0;
        c.n_levels = 3;
        let m = StereoMatcher::new(c)?;
        let run = |ol: u8, or: u8| -> Result<i32, StereoMatchError> {
            let (lo, ro) = ([ol], [or]);
            let mut l = kps(&lxy, &b, &[], true);
            let mut r = kps(&rxy, &b, &[], true);
            l.octaves = Some(&lo);
            r.octaves = Some(&ro);
            let mut out = StereoMatches::default();
            m.match_into(&[], &[], &l, &r, &mut out)?;
            Ok(out.right_idx[0])
        };
        assert_eq!(
            run(0, 0)?,
            -1,
            "row band 2 at octave 0 misses a 3-row offset"
        );
        assert_eq!(run(1, 1)?, 0, "band 4 at octave 1 covers it");
        assert_eq!(run(0, 2)?, -1, "octaves 2 apart are gated");
        assert_eq!(run(0, 3)?, -1, "octave past n_levels never matches");
        Ok(())
    }

    #[test]
    fn rejects_inconsistent_inputs() {
        let m = StereoMatcher::new(cfg()).unwrap();
        let xy = [[1.0f32, 1.0]];
        let b = [0u8; 32];
        let f = [0.0f32; 64];
        let mut out = StereoMatches::default();
        let l = kps(&xy, &b, &f, true);
        let r = kps(&xy, &b, &f, false);
        assert!(matches!(
            m.match_into(&[], &[], &l, &r, &mut out),
            Err(StereoMatchError::Descriptors(_))
        ));
        let short = StereoKeypoints {
            xy: &xy,
            octaves: None,
            descriptors: StereoDescriptors::Binary {
                data: &b[..16],
                bytes: 32,
            },
        };
        assert!(matches!(
            m.match_into(&[], &[], &short, &l, &mut out),
            Err(StereoMatchError::Length { .. })
        ));
        let zero = StereoKeypoints {
            xy: &xy,
            octaves: None,
            descriptors: StereoDescriptors::Binary {
                data: &[],
                bytes: 0,
            },
        };
        assert!(matches!(
            m.match_into(&[], &[], &zero, &zero, &mut out),
            Err(StereoMatchError::Descriptors(_))
        ));
        assert!(matches!(
            m.match_into(&[], &[], &l, &l, &mut out),
            Err(StereoMatchError::Pyramid(_))
        ));
        let mut c = cfg();
        c.n_levels = 9;
        assert!(StereoMatcher::new(c).is_err());
    }
}
