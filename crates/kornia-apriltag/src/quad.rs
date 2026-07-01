use crate::{
    segmentation::GradientInfo,
    utils::{homography_compute, Pixel},
    DecodeTagsConfig,
};
use kornia_algebra::{Mat3F32, Vec3F32};
use kornia_image::{allocator::ImageAllocator, Image};
use kornia_imgproc::filter::kernels::gaussian_kernel_1d;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::{
    cell::RefCell,
    f32::{self, consts::PI},
    ops::ControlFlow,
    sync::OnceLock,
};

// Precomputed once: gaussian_kernel_1d(filter_size, 1.0) where filter_size is derived from
// SIGMA=1.0 CUTOFF=0.05 constants in quad_segment_maxima. Always 7 elements.
static GAUSSIAN_KERNEL: OnceLock<Vec<f32>> = OnceLock::new();

#[inline(always)]
fn gaussian_kernel() -> &'static [f32] {
    GAUSSIAN_KERNEL.get_or_init(|| {
        const SIGMA: f32 = 1.0;
        const CUTOFF: f32 = 0.05;
        let filter_size =
            (2.0 * ((-(CUTOFF.ln()) * 2.0 * SIGMA * SIGMA).sqrt() + 1.0) + 1.0) as usize;
        gaussian_kernel_1d(filter_size, SIGMA)
    })
}

/// Per-thread scratch buffers for compute_line_fit_prefix_sums (weights pass + lfps output).
#[derive(Default)]
struct LfpWorkspace {
    weights: Vec<f32>,
    lfps: Vec<LineFit>,
}

/// Per-thread scratch buffers for quad_segment_maxima (all internal allocations).
#[derive(Default)]
struct SegWorkspace {
    soa: Vec<f32>,
    errors: Vec<f32>,
    smoothed_errors: Vec<f32>,
    maxima: Vec<usize>,
    maxima_errs: Vec<f32>,
}

/// Combined per-thread workspace for fit_quads hot path (eliminates all per-cluster allocations).
#[derive(Default)]
struct FitWorkspace {
    cluster: Vec<GradientInfo>,
    lfw: LfpWorkspace,
    seg: SegWorkspace,
    /// Compact (slope_bits<<32|orig_idx) u64 pairs for sort-by-key.
    sort_pairs: Vec<u64>,
    /// Scratch permutation buffer so we scatter-then-copy instead of sorting 16-byte structs.
    scratch: Vec<GradientInfo>,
}

thread_local! {
    static FIT_WS: RefCell<FitWorkspace> = RefCell::new(FitWorkspace::default());
}

const SLOPE_OFFSET_BASE: i32 = 2 << 15; // Base value for slope offset calculations.
const SLOPE_OFFSET_DOUBLE: i32 = 2 * SLOPE_OFFSET_BASE; // Double the base value for extended range.

/// Constants defining quadrant boundaries for slope offset logic.
const QUADRANTS: [[i32; 2]; 2] = [
    [-SLOPE_OFFSET_BASE, 0],
    [SLOPE_OFFSET_DOUBLE, SLOPE_OFFSET_BASE],
];

#[derive(Debug, Clone, Copy, PartialEq)]
/// Options for fitting quadrilaterals (quads) to clusters of gradient information.
pub struct FitQuadConfig {
    /// Cosine of the critical angle in radians.
    pub cos_critical_rad: f32,
    /// Maximum mean squared error allowed for line fitting.
    pub max_line_fit_mse: f32,
    /// Maximum number of maxima to consider.
    pub max_nmaxima: usize,
    /// Minimum number of pixels required in a cluster to be considered.
    pub min_cluster_pixels: usize,
}

impl Default for FitQuadConfig {
    fn default() -> Self {
        Self {
            cos_critical_rad: (10.0 * PI / 180.0).cos(),
            max_line_fit_mse: 10.0,
            max_nmaxima: 10,
            min_cluster_pixels: 5,
        }
    }
}

/// Represents a detected quadrilateral in the image, corresponding to a tag candidate.
#[derive(Debug, Clone, PartialEq)]
pub struct Quad {
    /// The four corners of the quadrilateral, in image coordinates.
    ///
    /// Order: [Bottom-left, Bottom-right, Top-right, Top-left]
    pub corners: [kornia_algebra::Vec2F32; 4],
    /// Indicates whether the border is reversed (black border inside white border).
    pub reversed_border: bool,
    /// The 3x3 homography matrix mapping tag coordinates to image coordinates.
    pub homography: Mat3F32,
}

impl Default for Quad {
    fn default() -> Self {
        Self {
            corners: [kornia_algebra::Vec2F32::ZERO; 4],
            reversed_border: false,
            homography: Mat3F32::ZERO,
        }
    }
}

impl Quad {
    /// Projects a point (x, y) from tag coordinates to image coordinates using the quad's homography.
    ///
    /// # Arguments
    ///
    /// * `x` - The x coordinate in tag space.
    /// * `y` - The y coordinate in tag space.
    ///
    /// # Returns
    ///
    /// A `Vec2F32` representing the projected point in image coordinates.
    pub fn homography_project(&self, x: f32, y: f32) -> kornia_algebra::Vec2F32 {
        let p = self.homography * Vec3F32::new(x, y, 1.0);
        kornia_algebra::Vec2F32::new(p.x / p.z, p.y / p.z)
    }

    /// Updates the homography matrix for the quad based on its current corner positions.
    ///
    /// # Returns
    ///
    /// `true` if the homography was successfully computed and updated, `false` otherwise.
    pub fn update_homographies(&mut self) -> bool {
        let corr_arr = [
            [-1.0, -1.0, self.corners[0].x, self.corners[0].y],
            [1.0, -1.0, self.corners[1].x, self.corners[1].y],
            [1.0, 1.0, self.corners[2].x, self.corners[2].y],
            [-1.0, 1.0, self.corners[3].x, self.corners[3].y],
        ];

        if let Some(h) = homography_compute(corr_arr) {
            self.homography = h;
            return true;
        }

        false
    }
}

/// Fits quadrilaterals (quads) to clusters of gradient information in the image.
///
/// # Arguments
///
/// * `src` - The source image. Must be `Sync` so it can be shared across Rayon threads.
/// * `clusters` - A mutable reference to a HashMap containing clusters of `GradientInfo`.
/// * `config` - Configuration for decoding tags.
///
/// # Returns
///
/// A vector of detected `Quad` structures.
// TODO: Support multiple tag families
pub fn fit_quads<A: ImageAllocator + Sync>(
    src: &Image<Pixel, 1, A>,
    strip_maps: &[FxHashMap<(usize, usize), Vec<GradientInfo>>],
    config: &DecodeTagsConfig,
) -> Vec<Quad> {
    let max_cluster_len = 4 * (src.width() + src.height());
    let min_tag_width = config.min_tag_width;
    let normal_border = config.normal_border;
    let reversed_border = config.reversed_border;
    let downscale_factor = config.downscale_factor;
    let fit_quad_config = config.fit_quad_config;

    // Collect unique keys across all strip maps.
    let mut seen: FxHashMap<(usize, usize), ()> = FxHashMap::default();
    for map in strip_maps {
        for k in map.keys() {
            seen.entry(*k).or_insert(());
        }
    }
    let keys: Vec<(usize, usize)> = seen.into_keys().collect();

    keys.into_par_iter()
        .filter_map(|key| {
            let total_len: usize = strip_maps
                .iter()
                .filter_map(|m| m.get(&key))
                .map(|v| v.len())
                .sum();
            if total_len < fit_quad_config.min_cluster_pixels || total_len > max_cluster_len {
                return None;
            }
            FIT_WS.with(|cell| {
                let ws = &mut *cell.borrow_mut();
                ws.cluster.clear();
                for map in strip_maps {
                    if let Some(entries) = map.get(&key) {
                        ws.cluster.extend_from_slice(entries);
                    }
                }
                let mut quad = fit_single_quad(
                    src,
                    &mut ws.cluster,
                    min_tag_width,
                    normal_border,
                    reversed_border,
                    &fit_quad_config,
                    &mut ws.lfw,
                    &mut ws.seg,
                    &mut ws.sort_pairs,
                    &mut ws.scratch,
                )?;
                if downscale_factor > 1 {
                    let df = downscale_factor as f32;
                    quad.corners.iter_mut().for_each(|c| {
                        c.x = (c.x - 0.5) * df + 0.5;
                        c.y = (c.y - 0.5) * df + 0.5;
                    });
                }
                Some(quad)
            })
        })
        .collect()
}

/// Fits a single quadrilateral (quad) to a cluster of gradient information in the image.
///
/// # Arguments
///
/// * `src` - The source image.
/// * `cluster` - A mutable slice of `GradientInfo` representing the cluster.
/// * `min_tag_width` - Minimum width of the tag to be considered.
/// * `normal_border` - Indicates if the normal border is expected.
/// * `reversed_border` - Indicates if the reversed border is expected.
/// * `config` - Configuration for quad fitting process
///
/// # Returns
///
/// An `Option<Quad>` containing the detected quadrilateral if successful, or `None` otherwise.
fn fit_single_quad<A: ImageAllocator>(
    src: &Image<Pixel, 1, A>,
    cluster: &mut [GradientInfo],
    min_tag_width: usize,
    normal_border: bool,
    reversed_border: bool,
    config: &FitQuadConfig,
    lfw: &mut LfpWorkspace,
    seg: &mut SegWorkspace,
    sort_pairs: &mut Vec<u64>,
    scratch: &mut Vec<GradientInfo>,
) -> Option<Quad> {
    if cluster.len() < 24 {
        return None;
    }

    let mut x_max = cluster[0].pos.x;
    let mut x_min = x_max;
    let mut y_max = cluster[0].pos.y;
    let mut y_min = y_max;

    cluster.iter().for_each(|GradientInfo { pos, .. }| {
        if pos.x > x_max {
            x_max = pos.x;
        } else if pos.x < x_min {
            x_min = pos.x;
        }

        if pos.y > y_max {
            y_max = pos.y;
        } else if pos.y < y_min {
            y_min = pos.y;
        }
    });

    if (x_max - x_min) * (y_max - y_min) < min_tag_width as u32 {
        return None;
    }

    let cx = (x_min + x_max) as f32 * 0.5 + 0.05118;
    let cy = (y_min + y_max) as f32 * 0.5 - 0.028581;

    let mut dot = 0.0f32;
    for g in cluster.iter() {
        let dx = g.pos.x as f32 - cx;
        let dy = g.pos.y as f32 - cy;
        dot += dx * (g.gx as i32) as f32 + dy * (g.gy as i32) as f32;
    }

    let mut quad = Quad {
        reversed_border: dot < 0.0,
        ..Default::default()
    };

    if !reversed_border && quad.reversed_border {
        return None;
    }
    if !normal_border && !quad.reversed_border {
        return None;
    }

    for g in cluster.iter_mut() {
        let mut dx = g.pos.x as f32 - cx;
        let mut dy = g.pos.y as f32 - cy;
        let quadrant = QUADRANTS[(dy > 0.0) as usize][(dx > 0.0) as usize];
        if dy < 0.0 { dy = -dy; dx = -dx; }
        if dx < 0.0 { let tmp = dx; dx = dy; dy = -tmp; }
        g.slope = quadrant as f32 + dy / dx;
    }

    // Sort compact (sort_key<<32|orig_idx) u64 pairs — 8 bytes vs 16 for GradientInfo.
    // Slopes include negative values (quadrant offset can be -65536), so we need the IEEE 754
    // total-order bit transform: flip sign bit for positives, flip all bits for negatives.
    #[inline(always)]
    fn slope_sort_key(v: f32) -> u32 {
        let b = v.to_bits();
        if b >> 31 == 0 { b ^ 0x8000_0000 } else { !b }
    }
    sort_pairs.resize(cluster.len(), 0u64);
    for (i, g) in cluster.iter().enumerate() {
        sort_pairs[i] = ((slope_sort_key(g.slope) as u64) << 32) | i as u64;
    }
    sort_pairs.sort_unstable();
    // Scatter into scratch in sorted order, copy back — avoids moving 16-byte structs during sort.
    // Cluster is L1-hot (~3.5KB) from the slope pass, so random reads are fast.
    scratch.clear();
    for &pair in sort_pairs.iter() {
        scratch.push(cluster[(pair & 0xFFFF_FFFF) as usize]);
    }
    cluster.copy_from_slice(scratch.as_slice());

    compute_line_fit_prefix_sums_into(src, cluster, lfw);

    let mut indices = [0usize; 4];

    if !quad_segment_maxima_ws(cluster, &lfw.lfps, &mut indices, config, seg) {
        return None;
    }

    let mut lines = [[0.0f32; 4]; 4];

    if let ControlFlow::Break(_) = (0..4).try_for_each(|i| {
        let i0 = indices[i];
        let i1 = indices[(i + 1) & 3];

        let mut mse = 0.0f32;
        fit_line(&lfw.lfps, i0, i1, Some(&mut lines[i]), None, Some(&mut mse));

        if mse > config.max_line_fit_mse {
            return ControlFlow::Break(());
        }

        ControlFlow::Continue(())
    }) {
        return None;
    };

    if let ControlFlow::Break(_) = (0..4).try_for_each(|i| {
        let a00 = lines[i][3];
        let a01 = -lines[(i + 1) & 3][3];
        let a10 = -lines[i][2];
        let a11 = lines[(i + 1) & 3][2];
        let b0 = -lines[i][0] + lines[(i + 1) & 3][0];
        let b1 = -lines[i][1] + lines[(i + 1) & 3][1];

        let det = a00 * a11 - a10 * a01;

        if det.abs() < 0.001 {
            return ControlFlow::Break(());
        }

        let w00 = a11 / det;
        let w01 = -a01 / det;

        let l0 = w00 * b0 + w01 * b1;

        quad.corners[i].x = lines[i][0] + l0 * a00;
        quad.corners[i].y = lines[i][1] + l0 * a10;

        ControlFlow::Continue(())
    }) {
        return None;
    };

    // Slope sort produces CW-in-image corners [TR,BR,BL,TL], matching C's internal ordering.

    let mut area = 0.0f32;

    let mut length = [0f32; 3];
    let mut p: f32;

    // Calculate area of triangle formed by points 0, 1, 2
    (0..3).for_each(|i| {
        let idxa = i;
        let idxb = (i + 1) % 3;
        length[i] = ((quad.corners[idxb].x - quad.corners[idxa].x).powi(2)
            + (quad.corners[idxb].y - quad.corners[idxa].y).powi(2))
        .sqrt();
    });

    p = (length[0] + length[1] + length[2]) / 2.0;
    area += (p * (p - length[0]) * (p - length[1]) * (p - length[2])).sqrt();

    // Calculate area of triangle formed by points 2, 3, 0
    (0..3).for_each(|i| {
        let idxs = [2, 3, 0, 2];
        let idxa = idxs[i];
        let idxb = idxs[i + 1];

        length[i] = ((quad.corners[idxb].x - quad.corners[idxa].x).powi(2)
            + (quad.corners[idxb].y - quad.corners[idxa].y).powi(2))
        .sqrt();
    });

    p = (length[0] + length[1] + length[2]) / 2.0;
    area += (p * (p - length[0]) * (p - length[1]) * (p - length[2])).sqrt();

    // Reject quads that are too small
    if area < 0.95 * min_tag_width as f32 * min_tag_width as f32 {
        return None;
    }

    // Reject quads whose cumulative angle change isn't equal to 2PI
    if let ControlFlow::Break(_) = (0..4).try_for_each(|i| {
        let i0 = i;
        let i1 = (i + 1) & 3;
        let i2 = (i + 2) & 3;

        let dx1 = quad.corners[i1].x - quad.corners[i0].x;
        let dy1 = quad.corners[i1].y - quad.corners[i0].y;
        let dx2 = quad.corners[i2].x - quad.corners[i1].x;
        let dy2 = quad.corners[i2].y - quad.corners[i1].y;

        let cos_dtheta =
            (dx1 * dx2 + dy1 * dy2) / ((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2)).sqrt();

        // Reject screen-CCW quads (negative cross product in image coords = wrong winding).
        // Slope sort gives screen-CW [TR,BR,BL,TL]; cross product is positive for screen-CW.
        let cross_cond = dx1 * dy2 < dy1 * dx2;
        let cos_cond = !(-config.cos_critical_rad..=config.cos_critical_rad).contains(&cos_dtheta);
        if cos_cond || cross_cond {
            return ControlFlow::Break(());
        }

        ControlFlow::Continue(())
    }) {
        return None;
    };

    Some(quad)
}

/// Stores prefix sums for weighted line fitting over a set of points.
#[derive(Default, Debug, Clone)]
struct LineFit {
    /// Weighted sum of x coordinates ($\sum_i w_i x_i$)
    mx: f32,
    /// Weighted sum of y coordinates ($\sum_i w_i y_i$)
    my: f32,
    /// Weighted sum of squared x coordinates ($\sum_i w_i x_i^2$)
    mxx: f32,
    /// Weighted sum of x·y products ($\sum_i w_i x_i y_i$)
    mxy: f32,
    /// Weighted sum of squared y coordinates ($\sum_i w_i y_i^2$)
    myy: f32,
    /// Total weight ($\sum_i w_i$)
    w: f32,
}

/// Computes prefix sums for weighted line fitting over a set of gradient information points.
///
/// # Arguments
///
/// * `src` - The source image.
/// * `gradient_infos` - A slice of `GradientInfo` representing the cluster.
///
/// # Returns
///
/// A vector of `LineFit` structures containing prefix sums for each point.
// Hot path: fills lfw.weights and lfw.lfps without allocating.
fn compute_line_fit_prefix_sums_into<A: ImageAllocator>(
    src: &Image<Pixel, 1, A>,
    gradient_infos: &[GradientInfo],
    lfw: &mut LfpWorkspace,
) {
    let src_slice = src.as_slice();
    let width = src.width();
    let height = src.height();
    let n = gradient_infos.len();
    lfw.weights.resize(n, 0.0);
    lfw.lfps.resize_with(n, LineFit::default);
    // `weights` is only consumed by the NEON path; the scalar path computes weights inline.
    #[cfg(target_arch = "aarch64")]
    let weights = &mut lfw.weights;
    let lfps = &mut lfw.lfps;

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;

        let mut k = 0;
        while k + 4 <= n {
            let mut g2 = [0.0f32; 4];
            for j in 0..4 {
                let g = &gradient_infos[k + j];
                let x = g.pos.x as f32 * 0.5 + 0.5;
                let y = g.pos.y as f32 * 0.5 + 0.5;
                let ix = x as usize;
                let iy = y as usize;
                if ix > 0 && ix + 1 < width && iy > 0 && iy + 1 < height {
                    let gx = src_slice[iy * width + ix + 1] as i32
                        - src_slice[iy * width + ix - 1] as i32;
                    let gy = src_slice[(iy + 1) * width + ix] as i32
                        - src_slice[(iy - 1) * width + ix] as i32;
                    g2[j] = (gx * gx + gy * gy) as f32;
                }
            }
            let wv = unsafe {
                vaddq_f32(vsqrtq_f32(vld1q_f32(g2.as_ptr())), vdupq_n_f32(1.0))
            };
            unsafe { vst1q_f32(weights.as_mut_ptr().add(k), wv) };
            k += 4;
        }
        while k < n {
            let g = &gradient_infos[k];
            let x = g.pos.x as f32 * 0.5 + 0.5;
            let y = g.pos.y as f32 * 0.5 + 0.5;
            let ix = x as usize;
            let iy = y as usize;
            if ix > 0 && ix + 1 < width && iy > 0 && iy + 1 < height {
                let gx = src_slice[iy * width + ix + 1] as i32
                    - src_slice[iy * width + ix - 1] as i32;
                let gy = src_slice[(iy + 1) * width + ix] as i32
                    - src_slice[(iy - 1) * width + ix] as i32;
                weights[k] = ((gx * gx + gy * gy) as f32).sqrt() + 1.0;
            } else {
                weights[k] = 1.0;
            }
            k += 1;
        }

        let (mut smx, mut smy, mut smxx, mut smxy, mut smyy, mut sw) =
            (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
        for i in 0..n {
            let w = weights[i];
            let x = gradient_infos[i].pos.x as f32 * 0.5 + 0.5;
            let y = gradient_infos[i].pos.y as f32 * 0.5 + 0.5;
            smx += w * x;
            smy += w * y;
            smxx += w * x * x;
            smxy += w * x * y;
            smyy += w * y * y;
            sw += w;
            lfps[i] = LineFit { mx: smx, my: smy, mxx: smxx, mxy: smxy, myy: smyy, w: sw };
        }
        return;
    }

    #[allow(unreachable_code)]
    {
        let (mut smx, mut smy, mut smxx, mut smxy, mut smyy, mut sw) =
            (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
        for (i, gi) in gradient_infos.iter().enumerate() {
            let x = gi.pos.x as f32 * 0.5 + 0.5;
            let y = gi.pos.y as f32 * 0.5 + 0.5;
            let ix = x as usize;
            let iy = y as usize;
            let w = if ix > 0 && ix + 1 < width && iy > 0 && iy + 1 < height {
                let gx = src_slice[iy * width + ix + 1] as i32
                    - src_slice[iy * width + ix - 1] as i32;
                let gy = src_slice[(iy + 1) * width + ix] as i32
                    - src_slice[(iy - 1) * width + ix] as i32;
                ((gx * gx + gy * gy) as f32).sqrt() + 1.0
            } else {
                1.0
            };
            smx += w * x;
            smy += w * y;
            smxx += w * x * x;
            smxy += w * x * y;
            smyy += w * y * y;
            sw += w;
            lfps[i] = LineFit { mx: smx, my: smy, mxx: smxx, mxy: smxy, myy: smyy, w: sw };
        }
    }
}

// Test/compatibility wrapper: allocates fresh vecs.
#[cfg(test)]
fn compute_line_fit_prefix_sums<A: ImageAllocator>(
    src: &Image<Pixel, 1, A>,
    gradient_infos: &[GradientInfo],
) -> Vec<LineFit> {
    let mut lfw = LfpWorkspace::default();
    compute_line_fit_prefix_sums_into(src, gradient_infos, &mut lfw);
    lfw.lfps
}

/// Segments the gradient information into four maxima corresponding to the corners of a quadrilateral.
///
/// This function analyzes the error profile of line fits over the gradient information and finds
/// four maxima (peaks) in the error signal, which are interpreted as the corners of a quad. It
/// returns `true` if a valid segmentation is found and writes the indices of the maxima into `indices`.
///
/// # Arguments
///
/// * `gradient_infos` - Slice of `GradientInfo` representing the cluster.
/// * `lfps` - Slice of `LineFit` prefix sums for the cluster.
/// * `indices` - Mutable reference to an array where the four corner indices will be written.
/// * `config` - Configuration for quad fitting process
///
/// # Returns
///
/// `true` if four valid maxima are found and written to `indices`, `false` otherwise.
// Hot path: uses pre-allocated SegWorkspace buffers, no per-call malloc.
fn quad_segment_maxima_ws(
    gradient_infos: &[GradientInfo],
    lfps: &[LineFit],
    indices: &mut [usize; 4],
    config: &FitQuadConfig,
    seg: &mut SegWorkspace,
) -> bool {
    debug_assert_eq!(gradient_infos.len(), lfps.len());
    let len = gradient_infos.len();
    let window_size = 20.min(len / 12);
    if window_size < 2 {
        return false;
    }

    let pad = len + 1;
    // Resize scratch buffers — no malloc if capacity already sufficient.
    seg.soa.resize(6 * pad, 0.0);
    // Zero the 6 prefix-sum sentinels at index 0 of each sub-array.
    for k in 0..6usize { seg.soa[k * pad] = 0.0; }
    seg.errors.resize(len, 0.0);
    seg.smoothed_errors.resize(len, 0.0);
    seg.maxima.resize(len, 0);
    seg.maxima_errs.resize(len, 0.0);

    let ws = window_size;
    let inner_end = len - ws;

    for i in (0..ws).chain(inner_end..len) {
        fit_line(lfps, (i + len - ws) % len, (i + ws) % len, None, seg.errors.get_mut(i), None);
    }

    for k in 0..len {
        let lk = &lfps[k];
        seg.soa[k + 1]         = lk.mx;
        seg.soa[pad + k + 1]   = lk.my;
        seg.soa[2*pad + k + 1] = lk.mxx;
        seg.soa[3*pad + k + 1] = lk.mxy;
        seg.soa[4*pad + k + 1] = lk.myy;
        seg.soa[5*pad + k + 1] = lk.w;
    }

    let n_f32 = (2 * ws + 1) as f32;

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        let p = seg.soa.as_ptr();
        let ep = seg.errors.as_mut_ptr();
        let mx_p  = p;
        let my_p  = unsafe { p.add(pad) };
        let mxx_p = unsafe { p.add(2 * pad) };
        let mxy_p = unsafe { p.add(3 * pad) };
        let myy_p = unsafe { p.add(4 * pad) };
        let w_p   = unsafe { p.add(5 * pad) };

        let n_v   = unsafe { vdupq_n_f32(n_f32) };
        let four  = unsafe { vdupq_n_f32(4.0) };
        let half  = unsafe { vdupq_n_f32(0.5) };

        let mut i = ws;
        while i + 4 <= inner_end {
            unsafe {
                let hi = i + ws + 1;
                let lo = i - ws;
                let sw_mx  = vsubq_f32(vld1q_f32(mx_p.add(hi)),  vld1q_f32(mx_p.add(lo)));
                let sw_my  = vsubq_f32(vld1q_f32(my_p.add(hi)),  vld1q_f32(my_p.add(lo)));
                let sw_mxx = vsubq_f32(vld1q_f32(mxx_p.add(hi)), vld1q_f32(mxx_p.add(lo)));
                let sw_mxy = vsubq_f32(vld1q_f32(mxy_p.add(hi)), vld1q_f32(mxy_p.add(lo)));
                let sw_myy = vsubq_f32(vld1q_f32(myy_p.add(hi)), vld1q_f32(myy_p.add(lo)));
                let sw_w   = vsubq_f32(vld1q_f32(w_p.add(hi)),   vld1q_f32(w_p.add(lo)));
                // 2-step Newton-Raphson reciprocal (~32 cycles) vs 5 vdivq_f32 (~70 cycles).
                let inv_w0 = vrecpeq_f32(sw_w);
                let inv_w1 = vmulq_f32(vrecpsq_f32(sw_w, inv_w0), inv_w0);
                let inv_w  = vmulq_f32(vrecpsq_f32(sw_w, inv_w1), inv_w1);
                let ex    = vmulq_f32(sw_mx,  inv_w);
                let ey    = vmulq_f32(sw_my,  inv_w);
                let cxx   = vsubq_f32(vmulq_f32(sw_mxx, inv_w), vmulq_f32(ex, ex));
                let cxy   = vsubq_f32(vmulq_f32(sw_mxy, inv_w), vmulq_f32(ex, ey));
                let cyy   = vsubq_f32(vmulq_f32(sw_myy, inv_w), vmulq_f32(ey, ey));
                let diff  = vsubq_f32(cxx, cyy);
                let disc  = vaddq_f32(vmulq_f32(diff, diff), vmulq_f32(four, vmulq_f32(cxy, cxy)));
                let eig   = vmulq_f32(half, vsubq_f32(vaddq_f32(cxx, cyy), vsqrtq_f32(disc)));
                vst1q_f32(ep.add(i), vmulq_f32(eig, n_v));
            }
            i += 4;
        }
        while i < inner_end {
            fit_line(lfps, i - ws, i + ws, None, seg.errors.get_mut(i), None);
            i += 1;
        }
    }

    // x86_64: AVX2 line-fit error over 8 windows at once. Uses an IEEE-exact
    // reciprocal (`_mm256_div_ps(1, w)`) — more precise than NEON's Newton-Raphson —
    // but `mx * (1/w)` still differs from the scalar `fit_line`'s `mx / w` by ≤1 ULP,
    // so the segment-maxima it feeds can pick a different index on borderline ties.
    // Acceptable and consistent with the NEON path; covered by the C-parity suite.
    #[cfg(target_arch = "x86_64")]
    {
        if crate::ops::has_avx2() {
            use std::arch::x86_64::*;
            let p = seg.soa.as_ptr();
            let ep = seg.errors.as_mut_ptr();
            // SAFETY: AVX2 confirmed by runtime probe; all loads stay within
            // seg.soa (pad = len+1) and stores within seg.errors (len) because the
            // SIMD loop is bounded by `i + 8 <= inner_end` and `hi <= len`.
            unsafe {
                let mx_p = p;
                let my_p = p.add(pad);
                let mxx_p = p.add(2 * pad);
                let mxy_p = p.add(3 * pad);
                let myy_p = p.add(4 * pad);
                let w_p = p.add(5 * pad);

                let n_v = _mm256_set1_ps(n_f32);
                let four = _mm256_set1_ps(4.0);
                let halfv = _mm256_set1_ps(0.5);
                let one = _mm256_set1_ps(1.0);

                let mut i = ws;
                while i + 8 <= inner_end {
                    let hi = i + ws + 1;
                    let lo = i - ws;
                    let sw = |base: *const f32| {
                        _mm256_sub_ps(
                            _mm256_loadu_ps(base.add(hi)),
                            _mm256_loadu_ps(base.add(lo)),
                        )
                    };
                    let sw_mx = sw(mx_p);
                    let sw_my = sw(my_p);
                    let sw_mxx = sw(mxx_p);
                    let sw_mxy = sw(mxy_p);
                    let sw_myy = sw(myy_p);
                    let sw_w = sw(w_p);

                    let inv_w = _mm256_div_ps(one, sw_w);
                    let ex = _mm256_mul_ps(sw_mx, inv_w);
                    let ey = _mm256_mul_ps(sw_my, inv_w);
                    let cxx = _mm256_sub_ps(_mm256_mul_ps(sw_mxx, inv_w), _mm256_mul_ps(ex, ex));
                    let cxy = _mm256_sub_ps(_mm256_mul_ps(sw_mxy, inv_w), _mm256_mul_ps(ex, ey));
                    let cyy = _mm256_sub_ps(_mm256_mul_ps(sw_myy, inv_w), _mm256_mul_ps(ey, ey));
                    let diff = _mm256_sub_ps(cxx, cyy);
                    let disc = _mm256_add_ps(
                        _mm256_mul_ps(diff, diff),
                        _mm256_mul_ps(four, _mm256_mul_ps(cxy, cxy)),
                    );
                    let eig = _mm256_mul_ps(
                        halfv,
                        _mm256_sub_ps(_mm256_add_ps(cxx, cyy), _mm256_sqrt_ps(disc)),
                    );
                    _mm256_storeu_ps(ep.add(i), _mm256_mul_ps(eig, n_v));
                    i += 8;
                }
                while i < inner_end {
                    fit_line(lfps, i - ws, i + ws, None, seg.errors.get_mut(i), None);
                    i += 1;
                }
            }
        } else {
            for i in ws..inner_end {
                fit_line(lfps, i - ws, i + ws, None, seg.errors.get_mut(i), None);
            }
        }
    }

    // Portable scalar fallback (non-aarch64, non-x86_64).
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    for i in ws..inner_end {
        fit_line(lfps, i - ws, i + ws, None, seg.errors.get_mut(i), None);
    }

    let kernel = gaussian_kernel();
    let filter_size = kernel.len();
    let half = filter_size / 2;

    for iy in (0..half).chain(len - half..len) {
        let mut acc = 0.0f32;
        for i in 0..filter_size {
            let idx = (iy + i + len - half) % len;
            acc += seg.errors[idx] * kernel[i];
        }
        seg.smoothed_errors[iy] = acc;
    }
    crate::ops::smooth_interior(&seg.errors, kernel, &mut seg.smoothed_errors, half, len);

    // Swap so seg.errors now holds the smoothed values (seg.smoothed_errors gets the raw, unused).
    std::mem::swap(&mut seg.errors, &mut seg.smoothed_errors);

    let mut nmaxima = 0usize;
    for i in 0..len {
        if seg.errors[i] > seg.errors[(i + 1) % len] && seg.errors[i] > seg.errors[(i + len - 1) % len] {
            seg.maxima[nmaxima] = i;
            seg.maxima_errs[nmaxima] = seg.errors[i];
            nmaxima += 1;
        }
    }

    if nmaxima < 4 {
        return false;
    }

    if nmaxima > config.max_nmaxima {
        let mut maxima_errs_copy = seg.maxima_errs[..nmaxima].to_vec();
        maxima_errs_copy.sort_by(|a, b| b.total_cmp(a));
        let maxima_thresh = maxima_errs_copy[config.max_nmaxima];
        let mut out = 0usize;
        for i in 0..nmaxima {
            if seg.maxima_errs[i] <= maxima_thresh {
                continue;
            }
            let v = seg.maxima[i];
            seg.maxima[out] = v;
            out += 1;
        }
        if out < 4 {
            return false;
        }
        nmaxima = out;
    }

    let mut best_indices = [0usize, 0, 0, 0];
    let mut best_error = f32::INFINITY;

    let mut err01 = 0.0f32;
    let mut err12 = 0.0f32;
    let mut err23 = 0.0f32;
    let mut err30 = 0.0f32;

    let mut mse01 = 0.0f32;
    let mut mse12 = 0.0f32;
    let mut mse23 = 0.0f32;
    let mut mse30 = 0.0f32;

    let mut params01 = [0.0f32; 4];
    let mut params12 = [0.0f32; 4];

    (0..nmaxima - 3).for_each(|m0| {
        let i0 = seg.maxima[m0];
        ((m0 + 1)..(nmaxima - 2)).for_each(|m1| {
            let i1 = seg.maxima[m1];
            fit_line(lfps, i0, i1, Some(&mut params01), Some(&mut err01), Some(&mut mse01));
            if mse01 > config.max_line_fit_mse { return; }
            ((m1 + 1)..nmaxima - 1).for_each(|m2| {
                let i2 = seg.maxima[m2];
                fit_line(lfps, i1, i2, Some(&mut params12), Some(&mut err12), Some(&mut mse12));
                if mse12 > config.max_line_fit_mse { return; }
                let dot = params01[2] * params12[2] + params01[3] * params12[3];
                if dot.abs() > config.cos_critical_rad { return; }
                ((m2 + 1)..nmaxima).for_each(|m3| {
                    let i3 = seg.maxima[m3];
                    fit_line(lfps, i2, i3, None, Some(&mut err23), Some(&mut mse23));
                    if mse23 > config.max_line_fit_mse { return; }
                    fit_line(lfps, i3, i0, None, Some(&mut err30), Some(&mut mse30));
                    if mse30 > config.max_line_fit_mse { return; }
                    let err = err01 + err12 + err23 + err30;
                    if err < best_error {
                        best_error = err;
                        best_indices[0] = i0;
                        best_indices[1] = i1;
                        best_indices[2] = i2;
                        best_indices[3] = i3;
                    }
                });
            });
        });
    });

    if best_error.is_infinite() {
        return false;
    }

    best_indices.iter().enumerate().for_each(|(i, b)| { indices[i] = *b; });
    (best_error / gradient_infos.len() as f32) < config.max_line_fit_mse
}

// Test/compatibility wrapper: allocates fresh SegWorkspace.
#[cfg(test)]
fn quad_segment_maxima(
    gradient_infos: &[GradientInfo],
    lfps: &[LineFit],
    indices: &mut [usize; 4],
    config: &FitQuadConfig,
) -> bool {
    let mut seg = SegWorkspace::default();
    quad_segment_maxima_ws(gradient_infos, lfps, indices, config, &mut seg)
}

/// Fits a line to a segment of points using prefix sums for weighted least squares.
///
/// This function computes the best-fit line parameters for the segment of points between indices `i0` and `i1`
/// (inclusive, possibly wrapping around the end of the array), using the provided prefix sums (`lfps`).
/// Optionally outputs the line parameters, error, and mean squared error.
///
/// # Arguments
///
/// * `lfps` - Slice of `LineFit` prefix sums for the points.
/// * `i0` - Start index of the segment (inclusive).
/// * `i1` - End index of the segment (inclusive).
/// * `lineparm` - Optional mutable reference to an array where the line parameters will be written.
///   The array is [ex, ey, nx, ny], where (ex, ey) is the centroid and (nx, ny) is the direction.
/// * `err` - Optional mutable reference to a float where the error will be written.
/// * `mse` - Optional mutable reference to a float where the mean squared error will be written.
fn fit_line(
    lfps: &[LineFit],
    i0: usize,
    i1: usize,
    lineparm: Option<&mut [f32; 4]>,
    err: Option<&mut f32>,
    mse: Option<&mut f32>,
) {
    if i0 == i1 || !(i0 < lfps.len() && i1 < lfps.len()) {
        return;
    }

    let mut mx: f32;
    let mut my: f32;
    let mut mxx: f32;
    let mut myy: f32;
    let mut mxy: f32;
    let mut w: f32;
    let n: usize;

    if i0 < i1 {
        n = i1 - i0 + 1;

        mx = lfps[i1].mx;
        my = lfps[i1].my;
        mxx = lfps[i1].mxx;
        mxy = lfps[i1].mxy;
        myy = lfps[i1].myy;
        w = lfps[i1].w;

        if i0 > 0 {
            mx -= lfps[i0 - 1].mx;
            my -= lfps[i0 - 1].my;
            mxx -= lfps[i0 - 1].mxx;
            mxy -= lfps[i0 - 1].mxy;
            myy -= lfps[i0 - 1].myy;
            w -= lfps[i0 - 1].w;
        }
    } else if i0 > 0 {
        mx = lfps[lfps.len() - 1].mx - lfps[i0 - 1].mx;
        my = lfps[lfps.len() - 1].my - lfps[i0 - 1].my;
        mxx = lfps[lfps.len() - 1].mxx - lfps[i0 - 1].mxx;
        mxy = lfps[lfps.len() - 1].mxy - lfps[i0 - 1].mxy;
        myy = lfps[lfps.len() - 1].myy - lfps[i0 - 1].myy;
        w = lfps[lfps.len() - 1].w - lfps[i0 - 1].w;

        mx += lfps[i1].mx;
        my += lfps[i1].my;
        mxx += lfps[i1].mxx;
        mxy += lfps[i1].mxy;
        myy += lfps[i1].myy;
        w += lfps[i1].w;

        n = lfps.len() - i0 + i1 + 1;
    } else {
        return;
    }

    if n < 2 {
        return;
    }

    let ex = mx / w;
    let ey = my / w;
    let cxx = mxx / w - ex * ex;
    let cxy = mxy / w - ex * ey;
    let cyy = myy / w - ey * ey;

    let eig_small = 0.5 * (cxx + cyy - ((cxx - cyy) * (cxx - cyy) + 4.0 * cxy * cxy).sqrt());

    if let Some(lineparm) = lineparm {
        lineparm[0] = ex;
        lineparm[1] = ey;

        let eig = 0.5 * (cxx + cyy + ((cxx - cyy) * (cxx - cyy) + 4.0 * cxy * cxy).sqrt());
        let nx1 = cxx - eig;
        let ny1 = cxy;
        let m1 = nx1 * nx1 + ny1 * ny1;
        let nx2 = cxy;
        let ny2 = cyy - eig;
        let m2 = nx2 * nx2 + ny2 * ny2;

        let nx: f32;
        let ny: f32;
        let m: f32;

        if m1 > m2 {
            nx = nx1;
            ny = ny1;
            m = m1;
        } else {
            nx = nx2;
            ny = ny2;
            m = m2;
        }

        let length = m.sqrt();

        if length.abs() < 1e-12 {
            lineparm[2] = 0.0;
            lineparm[3] = 0.0;
        } else {
            lineparm[2] = nx / length;
            lineparm[3] = ny / length;
        }
    }

    if let Some(err) = err {
        *err = n as f32 * eig_small;
    }

    if let Some(mse) = mse {
        *mse = eig_small;
    }
}

#[cfg(test)]
mod tests {
    use kornia_image::allocator::CpuAllocator;
    use kornia_io::png::read_image_png_mono8;

    use crate::{
        family::TagFamilyKind,
        segmentation::{
            find_connected_components, find_gradient_clusters, GradientDirection, GradientInfo,
        },
        threshold::{adaptive_threshold, TileMinMax},
        union_find::UnionFind,
        utils::{Pixel, Point2d},
    };

    use super::*;

    #[test]
    fn test_fit_quads() -> Result<(), Box<dyn std::error::Error>> {
        let src = read_image_png_mono8("../../tests/data/apriltag.png")?;

        let mut bin = Image::from_size_val(src.size(), Pixel::Skip, CpuAllocator)?;
        let mut tile_min_max = TileMinMax::new(src.size(), 4);
        let mut uf = UnionFind::new(src.as_slice().len());
        adaptive_threshold(&src, &mut bin, &mut tile_min_max, 20)?;
        find_connected_components(&bin, &mut uf)?;
        let clusters = find_gradient_clusters(&bin, &mut uf);

        let mut decode_tag_config = DecodeTagsConfig::new(vec![TagFamilyKind::Tag36H11])?;
        decode_tag_config.downscale_factor = 1;

        let quads = fit_quads(&bin, &[clusters], &decode_tag_config);

        let expected_quad = [[[27, 3], [27, 27], [3, 27], [3, 3]]];

        assert_eq!(quads.len(), expected_quad.len());

        for (Quad { corners: point, .. }, expected_quad) in quads.iter().zip(expected_quad) {
            // We allow a ±1 error here to avoid CI failures due to small precision differences.
            // The expected outputs were generated with C code using 64-bit precision, while our code uses f32.
            assert!((point[0].x as usize).abs_diff(expected_quad[0][0]) <= 1);
            assert!((point[0].y as usize).abs_diff(expected_quad[0][1]) <= 1);

            assert!((point[1].x as usize).abs_diff(expected_quad[1][0]) <= 1);
            assert!((point[1].y as usize).abs_diff(expected_quad[1][1]) <= 1);

            assert!((point[2].x as usize).abs_diff(expected_quad[2][0]) <= 1);
            assert!((point[2].y as usize).abs_diff(expected_quad[2][1]) <= 1);

            assert!((point[3].x as usize).abs_diff(expected_quad[3][0]) <= 1);
            assert!((point[3].y as usize).abs_diff(expected_quad[3][1]) <= 1);
        }

        Ok(())
    }

    #[test]
    fn test_quad_segment_maxima_edge_cases() {
        // Test 1: Edge cases - too few points, empty input, constant slope
        let gradient_infos = vec![
            GradientInfo {
                pos: Point2d { x: 0u32, y: 0u32 },
                gx: GradientDirection::TowardsWhite,
                gy: GradientDirection::TowardsBlack,
                slope: 0.0,
            };
            20
        ];
        let lfps = vec![LineFit::default(); 20];
        let mut indices = [0; 4];

        // Should return false because window_size < 2 (20/12 = 1.66... < 2)
        assert!(!quad_segment_maxima(
            &gradient_infos,
            &lfps,
            &mut indices,
            &FitQuadConfig::default()
        ));

        // Test 2: Empty input
        let empty_gradient_infos = Vec::<GradientInfo>::new();
        let empty_lfps = Vec::<LineFit>::new();
        assert!(!quad_segment_maxima(
            &empty_gradient_infos,
            &empty_lfps,
            &mut indices,
            &FitQuadConfig::default()
        ));

        // Test 3: Constant slope (no maxima)
        let mut constant_slope_infos = Vec::new();
        for i in 0..100 {
            constant_slope_infos.push(GradientInfo {
                pos: Point2d { x: i as u32, y: i as u32 },
                gx: GradientDirection::TowardsWhite,
                gy: GradientDirection::TowardsBlack,
                slope: 0.0,
            });
        }
        let constant_lfps = vec![LineFit::default(); 100];
        assert!(!quad_segment_maxima(
            &constant_slope_infos,
            &constant_lfps,
            &mut indices,
            &FitQuadConfig::default()
        ));
    }

    #[test]
    fn test_quad_segment_maxima() -> Result<(), Box<dyn std::error::Error>> {
        let src = read_image_png_mono8("../../tests/data/apriltag.png")?;
        let mut bin = Image::from_size_val(src.size(), Pixel::Skip, CpuAllocator)?;
        let mut tile_min_max = TileMinMax::new(src.size(), 4);
        let mut uf = UnionFind::new(src.as_slice().len());
        adaptive_threshold(&src, &mut bin, &mut tile_min_max, 20)?;
        find_connected_components(&bin, &mut uf)?;
        let clusters = find_gradient_clusters(&bin, &mut uf);

        // Find the largest cluster to test with
        let largest_cluster = clusters
            .values()
            .max_by_key(|cluster| cluster.len())
            .unwrap();

        if largest_cluster.len() >= 24 {
            let lfps = compute_line_fit_prefix_sums(&bin, largest_cluster);
            let mut indices = [0; 4];

            let result = quad_segment_maxima(
                largest_cluster,
                &lfps,
                &mut indices,
                &FitQuadConfig::default(),
            );

            assert!(!result);
        }

        Ok(())
    }

    #[test]
    fn test_fit_line() {
        // Create some test LineFit data
        let mut lfps: Vec<LineFit> = Vec::new();

        // Create a simple line of points: (0,0), (1,1), (2,2), (3,3)
        for i in 0..4 {
            let mut lfp = LineFit::default();
            if i > 0 {
                lfp = lfps[i - 1].clone();
            }

            let x = i as f32;
            let y = i as f32;
            let w = 1.0f32;

            lfp.mx += w * x;
            lfp.my += w * y;
            lfp.mxx += w * x * x;
            lfp.mxy += w * x * y;
            lfp.myy += w * y * y;
            lfp.w += w;

            lfps.push(lfp);
        }

        // Test 1: Normal case with all parameters
        let mut lineparm = [0.0f32; 4];
        let mut err = 0.0f32;
        let mut mse = 0.0f32;

        fit_line(
            &lfps,
            0,
            3,
            Some(&mut lineparm),
            Some(&mut err),
            Some(&mut mse),
        );

        // Check that parameters were computed
        assert!(lineparm[0] != 0.0 || lineparm[1] != 0.0); // ex, ey should be computed
        assert!(err >= 0.0); // Error can be zero for perfect fit
        assert!(mse >= 0.0); // MSE can be zero for perfect fit

        // Test 2: Edge case - same indices
        let mut lineparm2 = [0.0f32; 4];
        let mut err2 = 0.0f32;
        let mut mse2 = 0.0f32;

        fit_line(
            &lfps,
            1,
            1,
            Some(&mut lineparm2),
            Some(&mut err2),
            Some(&mut mse2),
        );

        // Should not modify parameters when i0 == i1
        assert_eq!(lineparm2[0], 0.0);
        assert_eq!(lineparm2[1], 0.0);
        assert_eq!(err2, 0.0);
        assert_eq!(mse2, 0.0);

        // Test 3: Edge case - out of bounds indices
        let mut lineparm3 = [0.0f32; 4];
        let mut err3 = 0.0f32;
        let mut mse3 = 0.0f32;

        fit_line(
            &lfps,
            0,
            10,
            Some(&mut lineparm3),
            Some(&mut err3),
            Some(&mut mse3),
        );

        // Should not modify parameters when indices are out of bounds
        assert_eq!(lineparm3[0], 0.0);
        assert_eq!(lineparm3[1], 0.0);
        assert_eq!(err3, 0.0);
        assert_eq!(mse3, 0.0);

        // Test 4: Edge case - i0 > i1 (wrapped case)
        let mut lineparm4 = [0.0f32; 4];
        let mut err4 = 0.0f32;
        let mut mse4 = 0.0f32;

        fit_line(
            &lfps,
            3,
            1,
            Some(&mut lineparm4),
            Some(&mut err4),
            Some(&mut mse4),
        );

        // Should compute parameters for wrapped case
        assert!(lineparm4[0] != 0.0 || lineparm4[1] != 0.0);
        assert!(err4 >= 0.0);
        assert!(mse4 >= 0.0);

        // Test 5: No output parameters (should not crash)
        fit_line(&lfps, 0, 3, None, None, None);
    }

    #[test]
    fn test_compute_lfps() {
        use crate::segmentation::GradientDirection;
        use crate::utils::Pixel;
        use kornia_image::{Image, ImageSize};

        // Create a 4x4 image with all white pixels
        let size = ImageSize {
            width: 4,
            height: 4,
        };
        let img = Image::<Pixel, 1, _>::from_size_val(size, Pixel::White, CpuAllocator).unwrap();

        // Test 1: Single point
        let gradient_infos = vec![GradientInfo {
            pos: Point2d { x: 1u32, y: 2u32 },
            gx: GradientDirection::TowardsWhite,
            gy: GradientDirection::TowardsBlack,
            slope: 0.0,
        }];
        let lfps = compute_line_fit_prefix_sums(&img, &gradient_infos);
        assert_eq!(lfps.len(), 1);
        // The weighted mean should be close to the point's position (scaled by 0.5 + delta)
        let expected_x = 1.0 * 0.5 + 0.5;
        let expected_y = 2.0 * 0.5 + 0.5;
        assert!((lfps[0].mx / lfps[0].w - expected_x).abs() < 1e-4);
        assert!((lfps[0].my / lfps[0].w - expected_y).abs() < 1e-4);

        // Test 2: Multiple points in a line
        let gradient_infos = vec![
            GradientInfo {
                pos: Point2d { x: 0u32, y: 0u32 },
                gx: GradientDirection::TowardsWhite,
                gy: GradientDirection::TowardsBlack,
                slope: 0.0,
            },
            GradientInfo {
                pos: Point2d { x: 1u32, y: 1u32 },
                gx: GradientDirection::TowardsWhite,
                gy: GradientDirection::TowardsBlack,
                slope: 0.0,
            },
            GradientInfo {
                pos: Point2d { x: 2u32, y: 2u32 },
                gx: GradientDirection::TowardsWhite,
                gy: GradientDirection::TowardsBlack,
                slope: 0.0,
            },
        ];
        let lfps = compute_line_fit_prefix_sums(&img, &gradient_infos);
        assert_eq!(lfps.len(), 3);
        // The last element should be the sum of all previous
        let last = &lfps[2];
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_w = 0.0;
        for GradientInfo { pos, .. } in gradient_infos.iter().take(3) {
            let x = pos.x as f32 * 0.5 + 0.5;
            let y = pos.y as f32 * 0.5 + 0.5;
            sum_x += x;
            sum_y += y;
            sum_w += 1.0;
        }
        assert!((last.mx - sum_x).abs() < 1e-4);
        assert!((last.my - sum_y).abs() < 1e-4);
        assert!((last.w - sum_w).abs() < 1e-4);

        // Test 3: Empty input
        let gradient_infos = vec![];
        let lfps = compute_line_fit_prefix_sums(&img, &gradient_infos);
        assert_eq!(lfps.len(), 0);
    }
}
