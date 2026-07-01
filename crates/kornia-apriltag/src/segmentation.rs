use std::ops::Mul;

use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::{
    errors::AprilTagError,
    union_find::{ParStripUF, UnionFind},
    utils::{Pixel, Point2d},
};
use kornia_image::{Image};

/// Returns `Some(color)` if every pixel in [row_off+1 .. row_off+width-1] and the
/// corresponding top row [top_off+1 .. top_off+width-1] are all the same non-Skip
/// color (Black or White). Returns `None` if the rows are mixed or contain Skip pixels.
///
/// On aarch64 the check is done 16 pixels at a time with NEON.
#[inline]
fn solid_row_color(
    src_data: &[Pixel],
    row_off: usize,
    top_off: usize,
    width: usize,
) -> Option<Pixel> {
    let n = width - 2; // inner columns
    if n == 0 {
        return None;
    }
    let first = src_data[row_off + 1];
    if first == Pixel::Skip {
        return None;
    }

    let mut i = 0usize;

    // AArch64: NEON 16-wide equality scan.
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        let color_v = unsafe { vdupq_n_u8(first as u8) };
        let curr_ptr = src_data.as_ptr() as *const u8;
        unsafe {
            while i + 16 <= n {
                let c = vld1q_u8(curr_ptr.add(row_off + 1 + i));
                let t = vld1q_u8(curr_ptr.add(top_off + 1 + i));
                let c_ok = vceqq_u8(c, color_v);
                let t_ok = vceqq_u8(t, color_v);
                if vminvq_u8(vandq_u8(c_ok, t_ok)) != 0xFF {
                    return None;
                }
                i += 16;
            }
        }
    }

    // x86_64: AVX2 32-wide equality scan.
    #[cfg(target_arch = "x86_64")]
    if crate::ops::has_avx2() {
        use std::arch::x86_64::*;
        // SAFETY: AVX2 confirmed by runtime probe; Pixel is #[repr(u8)].
        unsafe {
            let color_v = _mm256_set1_epi8(first as u8 as i8);
            let curr_ptr = src_data.as_ptr() as *const u8;
            while i + 32 <= n {
                let c = _mm256_loadu_si256(curr_ptr.add(row_off + 1 + i) as *const __m256i);
                let t = _mm256_loadu_si256(curr_ptr.add(top_off + 1 + i) as *const __m256i);
                let c_ok = _mm256_cmpeq_epi8(c, color_v);
                let t_ok = _mm256_cmpeq_epi8(t, color_v);
                let both = _mm256_movemask_epi8(_mm256_and_si256(c_ok, t_ok));
                if both != -1 {
                    return None;
                }
                i += 32;
            }
        }
    }

    // Shared scalar tail (also the full scan on targets without SIMD).
    while i < n {
        if src_data[row_off + 1 + i] != first || src_data[top_off + 1 + i] != first {
            return None;
        }
        i += 1;
    }

    Some(first)
}

/// Extend a horizontal run starting at `row_off + x`.
///
/// Bulk-writes `run_root` to `par_uf.parent` for all pixels in the run, and
/// adds the run length to `par_uf.size[run_root]` in a single operation instead
/// of per-pixel increments.  On aarch64 the inner loop uses NEON u32×4 stores
/// for the parent fill.
///
/// Returns the number of pixels consumed (0 if the run ends immediately).
#[inline]
fn extend_run_bulk(
    src_data: &[Pixel],
    par_uf: &mut ParStripUF<'_>,
    row_off: usize,
    offset: usize,
    x_start: usize,
    x_limit: usize,
    pixel: Pixel,
    run_root: usize,
) -> usize {
    // Count consecutive same-color pixels.
    let avail = x_limit.saturating_sub(x_start);
    let mut len = 0usize;

    // AArch64: NEON 16-wide scan.
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        let pv = unsafe { vdupq_n_u8(pixel as u8) };
        let src_ptr = src_data.as_ptr() as *const u8;
        while len + 16 <= avail {
            let v = unsafe { vld1q_u8(src_ptr.add(row_off + x_start + len)) };
            let eq = unsafe { vceqq_u8(v, pv) };
            if unsafe { vminvq_u8(eq) } != 0xFF {
                break;
            }
            len += 16;
        }
    }

    // x86_64: AVX2 32-wide scan.
    #[cfg(target_arch = "x86_64")]
    if crate::ops::has_avx2() {
        use std::arch::x86_64::*;
        // SAFETY: AVX2 confirmed by runtime probe; Pixel is #[repr(u8)].
        unsafe {
            let pv = _mm256_set1_epi8(pixel as u8 as i8);
            let src_ptr = src_data.as_ptr() as *const u8;
            while len + 32 <= avail {
                let v = _mm256_loadu_si256(src_ptr.add(row_off + x_start + len) as *const __m256i);
                if _mm256_movemask_epi8(_mm256_cmpeq_epi8(v, pv)) != -1 {
                    break;
                }
                len += 32;
            }
        }
    }

    // Scalar tail (or entire scan on targets without SIMD).
    while len < avail && src_data[row_off + x_start + len] == pixel {
        len += 1;
    }

    if len == 0 {
        return 0;
    }

    // Bulk-write run_root to parent array.
    let dst = &mut par_uf.parent[row_off + x_start - offset..row_off + x_start + len - offset];
    let rv = run_root as u32;

    // AArch64: NEON u32×4 stores.
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        let rv4 = unsafe { vdupq_n_u32(rv) };
        let ptr = dst.as_mut_ptr();
        let mut k = 0usize;
        while k + 4 <= len {
            unsafe { vst1q_u32(ptr.add(k), rv4) };
            k += 4;
        }
        while k < len {
            unsafe { *ptr.add(k) = rv };
            k += 1;
        }
    }

    // Other targets: `fill` already lowers to an optimized memset, so a hand-rolled
    // AVX2 store wins nothing here.
    #[cfg(not(target_arch = "aarch64"))]
    dst.fill(rv);

    // Single size increment.
    par_uf.size[run_root - offset] += len as u32;
    len
}

/// Phase 1 inner loop for CC.
///
/// Two-pass per row:
/// 1. Horizontal run-based pass: scan left-to-right, doing ONE UF connect for the first
///    pixel of each same-color run (to connect it to its left neighbor, including x=0),
///    then extend the run via direct parent writes (O(1) per pixel instead of O(α) UF).
/// 2. Top/diagonal-connection pass: UF connect per top-crossing.
fn cc_strip_phase1(src_data: &[Pixel], width: usize, y_start: usize, y_end: usize, par_uf: &mut ParStripUF<'_>) {
    let offset = par_uf.offset;
    for y in y_start..y_end {
        let row_off = y * width;

        // Pass 1: horizontal run-based parent assignment.
        let mut x = 1usize;
        while x < width - 1 {
            let i = row_off + x;
            let pixel = src_data[i];
            if pixel == Pixel::Skip {
                x += 1;
                continue;
            }
            let left_i = i - 1;
            if pixel == src_data[left_i] && src_data[left_i] != Pixel::Skip {
                // Continuing a same-color run. Initialize left_i lazily if needed,
                // then look up its root (O(1) since left was already processed or fresh).
                if par_uf.parent[left_i - offset] == u32::MAX {
                    par_uf.parent[left_i - offset] = left_i as u32;
                }
                let run_root = par_uf.get_representative(left_i);
                // Direct parent write — skip UF union for subsequent run pixels.
                par_uf.parent[i - offset] = run_root as u32;
                par_uf.size[run_root - offset] += 1;
                x += 1;
                // Extend run: bulk-write parent and count length for single size add.
                let run_len = extend_run_bulk(src_data, par_uf, row_off, offset, x, width - 1, pixel, run_root);
                x += run_len;
            } else {
                // New run — initialize as self-root if not already set.
                if par_uf.parent[i - offset] == u32::MAX {
                    par_uf.parent[i - offset] = i as u32;
                }
                x += 1;
            }
        }

        // Pass 2: top and diagonal UF connects (only when the row above is in this strip).
        if y > y_start && y > 0 {
            let top_off = (y - 1) * width;

            // Solid-row fast path: if the entire inner width of this row and the row
            // above are the same single color, all UF connects for x≥2 are no-ops
            // (they share one run root from Pass 1 and one root from the top row).
            // Black solid rows: one connect(x=1) suffices.
            // White solid rows: process x=1 normally (top-left border diagonal), skip x≥2.
            let solid = solid_row_color(src_data, row_off, top_off, width);
            if let Some(solid_pixel) = solid {
                // x=1: process normally (handles border diagonals for White).
                let x = 1usize;
                let i = row_off + x;
                let top_i = top_off + x;
                if src_data[i] != Pixel::Skip {
                    if src_data[i] == src_data[top_i] { par_uf.connect(i, top_i); }
                    if solid_pixel == Pixel::White {
                        let top_left_i = top_i - 1;
                        if src_data[i] == src_data[top_left_i] { par_uf.connect(i, top_left_i); }
                        let top_right_i = top_i + 1;
                        if src_data[top_i] != src_data[top_right_i] && src_data[i] == src_data[top_right_i] {
                            par_uf.connect(i, top_right_i);
                        }
                    }
                }
                // x≥2: all no-ops (both row's pixels share a single component root
                // after the x=1 connect above), so skip the rest.
            } else {
                // Run-interior skip: track (last_pixel, last_top_pixel) in registers.
                // When current pixel is same color as previous (same horizontal run) AND
                // top pixel is same as previous top pixel (same top run) AND we already
                // called connect() for that run pair: the straight top-connect is a no-op.
                let mut last_connected = false;
                let mut last_pixel = Pixel::Skip;
                let mut last_top_pixel = Pixel::Skip;
                for x in 1..width - 1 {
                    let i = row_off + x;
                    let pixel = src_data[i];
                    if pixel == Pixel::Skip {
                        last_connected = false;
                        last_pixel = Pixel::Skip;
                        last_top_pixel = Pixel::Skip;
                        continue;
                    }
                    let left_i = i - 1;
                    let top_i = top_off + x;
                    let top_pixel = src_data[top_i];
                    if pixel == top_pixel {
                        // Skip if this pixel is interior to a matching run pair already connected.
                        let is_interior = last_connected && pixel == last_pixel && top_pixel == last_top_pixel;
                        if !is_interior {
                            par_uf.connect(i, top_i);
                            last_connected = true;
                        }
                    } else {
                        last_connected = false;
                    }
                    last_pixel = pixel;
                    last_top_pixel = top_pixel;
                    if pixel == Pixel::White {
                        let top_left_i = top_i - 1;
                        if (x == 1 || !(src_data[top_left_i] == src_data[left_i] || src_data[top_left_i] == top_pixel))
                            && pixel == src_data[top_left_i] { par_uf.connect(i, top_left_i); }
                        let top_right_i = top_i + 1;
                        if top_pixel != src_data[top_right_i] && pixel == src_data[top_right_i] { par_uf.connect(i, top_right_i); }
                    }
                }
            }
        }
    }
}

/// Finds connected components in a binary image using union-find.
///
/// Uses a parallel two-phase algorithm:
/// - Phase 1: process each horizontal strip on its own Rayon thread; skip
///   connections whose top-neighbor is in the adjacent (upper) strip.
/// - Phase 2: sequential border-merge — re-run the top-connection logic
///   only for the first row of each strip against the last row of the strip above.
///
/// # Arguments
///
/// * `src` - Reference to the source image containing `Pixel` values.
/// * `uf` - Mutable reference to a [`UnionFind`] structure for tracking connected components.
///   Call [`UnionFind::reset`] before reuse.
///
/// # Returns
///
/// * `Result<(), AprilTagError>` - `Ok(())` on success.
pub fn find_connected_components(
    src: &Image<Pixel, 1>,
    uf: &mut UnionFind,
) -> Result<(), AprilTagError> {
    let width = src.width();
    let height = src.height();
    let src_data = src.as_slice();
    let n_pixels = width * height;

    if n_pixels != uf.len() {
        return Err(AprilTagError::InvalidUnionFindSize(n_pixels, uf.len()));
    }

    let n_threads = rayon::current_num_threads().max(1);
    let strip_h = (height + n_threads - 1) / n_threads;
    let strip_pixels = strip_h * width;

    // Phase 1: parallel — process each strip's interior connections.
    uf.process_strips_parallel(n_threads, strip_pixels, |t, mut par_uf| {
        let y_start = t * strip_h;
        let y_end = ((t + 1) * strip_h).min(height);
        cc_strip_phase1(src_data, width, y_start, y_end, &mut par_uf);
    });

    // Phase 2: sequential — merge cross-strip connections.
    for t in 1..n_threads {
        let y = t * strip_h;
        if y >= height {
            continue;
        }
        let row_off = y * width;
        let top_off = row_off - width;
        let mut last_connected = false;
        let mut last_pixel = Pixel::Skip;
        let mut last_top_pixel = Pixel::Skip;
        for x in 1..width - 1 {
            let i = row_off + x;
            let pixel = src_data[i];
            if pixel == Pixel::Skip {
                last_connected = false;
                last_pixel = Pixel::Skip;
                last_top_pixel = Pixel::Skip;
                continue;
            }
            let left_i = i - 1;
            let top_i = top_off + x;
            let top_pixel = src_data[top_i];
            if pixel == top_pixel {
                let is_interior = last_connected && pixel == last_pixel && top_pixel == last_top_pixel;
                if !is_interior {
                    uf.connect(i, top_i);
                    last_connected = true;
                }
            } else {
                last_connected = false;
            }
            last_pixel = pixel;
            last_top_pixel = top_pixel;
            if pixel == Pixel::White {
                let top_left_i = top_i - 1;
                if (x == 1
                    || !(src_data[top_left_i] == src_data[left_i]
                        || src_data[top_left_i] == top_pixel))
                    && pixel == src_data[top_left_i]
                {
                    uf.connect(i, top_left_i);
                }
                let top_right_i = top_i + 1;
                if top_pixel != src_data[top_right_i] && pixel == src_data[top_right_i] {
                    uf.connect(i, top_right_i);
                }
            }
        }
    }
    Ok(())
}

/// Information about the gradient at a specific pixel location.
#[derive(Debug, Clone, Copy)]
pub struct GradientInfo {
    /// The coordinates of the pixel, represented as the mid-point assuming twice the size of the image.
    pub pos: Point2d<u32>,
    /// The gradient direction in the x-axis.
    pub gx: GradientDirection,
    /// The gradient direction in the y-axis.
    pub gy: GradientDirection,
    /// The slope of the gradient at this pixel.
    pub slope: f32,
}

/// Represents the direction of a gradient between two pixels.
///
/// Used to indicate whether the gradient is towards a white pixel, towards a black pixel, or if there is no gradient.
#[derive(Debug, Clone, Copy)]
#[repr(i16)]
pub enum GradientDirection {
    /// Gradient is towards a white pixel (value 255).
    TowardsWhite = 255,
    /// Gradient is towards a black pixel (value -255).
    TowardsBlack = -255,
    /// No gradient (value 0).
    None = 0,
}

impl Mul<isize> for GradientDirection {
    type Output = GradientDirection;

    fn mul(self, rhs: isize) -> Self::Output {
        match rhs.cmp(&0) {
            std::cmp::Ordering::Equal => GradientDirection::None,
            std::cmp::Ordering::Greater => self,
            std::cmp::Ordering::Less => match self {
                GradientDirection::TowardsWhite => GradientDirection::TowardsBlack,
                GradientDirection::TowardsBlack => GradientDirection::TowardsWhite,
                _ => GradientDirection::None,
            },
        }
    }
}

impl Pixel {
    /// Computes the gradient direction between two pixels.
    ///
    /// # Arguments
    /// * `other` - The pixel to compare against.
    ///
    /// # Returns
    /// A `GradientDirection` indicating the direction of the gradient.
    pub fn gradient_to(&self, other: Pixel) -> GradientDirection {
        match (self, other) {
            (Pixel::Black, Pixel::White) => GradientDirection::TowardsBlack,
            (Pixel::White, Pixel::Black) => GradientDirection::TowardsWhite,
            _ => GradientDirection::None,
        }
    }
}

/// Finds and groups gradient transitions between connected components in a binary image.
///
/// For each pixel, this function checks its neighbors and, if the neighbor belongs to a different
/// connected component (with sufficient size), records the gradient information between the two components.
/// The results are grouped by the pair of component representatives.
///
/// Rows are processed in parallel strips using rayon. The union-find is shared read-only via
/// `get_representative_ref` / `get_set_size_ref` which traverse the parent chain without mutation.
///
/// # Arguments
///
/// * `src` - Reference to the source image containing `Pixel` values.
/// * `uf` - Reference to a [`UnionFind`] structure for tracking connected components.
///
/// # Returns
///
/// A `FxHashMap` keyed by `(representative_a, representative_b)` pairs mapping to gradient info vectors.
pub fn find_gradient_clusters(
    src: &Image<Pixel, 1>,
    uf: &UnionFind,
) -> FxHashMap<(usize, usize), Vec<GradientInfo>> {
    let height = src.height();
    let width = src.width();
    let src_slice = src.as_slice();

    let n_threads = rayon::current_num_threads().max(1);
    let inner_rows = height.saturating_sub(2);
    let strip_h = (inner_rows + n_threads - 1) / n_threads;

    // Each thread processes rows [y_start, y_end) and reads the union-find without mutation
    // via get_representative_ref / get_set_size_ref (Sync, no locking needed).
    let local_maps: Vec<FxHashMap<(usize, usize), Vec<GradientInfo>>> = (0..n_threads)
        .into_par_iter()
        .map(|t| {
            let y_start = 1 + t * strip_h;
            if y_start >= height - 1 {
                return FxHashMap::default();
            }
            let y_end = (y_start + strip_h).min(height - 1);
            let mut local: FxHashMap<(usize, usize), Vec<GradientInfo>> = FxHashMap::with_capacity_and_hasher(128, Default::default());

            for y in y_start..y_end {
                let mut connected_last = false;

                for x in 1..width - 1 {
                    let i = y * width + x;
                    let current_pixel = src_slice[i];

                    if current_pixel == Pixel::Skip {
                        connected_last = false;
                        continue;
                    }

                    let current_pixel_representative = uf.get_representative_ref(i);

                    if uf.get_set_size_ref(current_pixel_representative) < 25 {
                        connected_last = false;
                        continue;
                    }

                    let mut any_connected = false;
                    let mut do_conn =
                        |dx: isize, dy: isize, neighbor_i: usize, any_connected: &mut bool| {
                            let neighbor_pixel = src_slice[neighbor_i];
                            if neighbor_pixel == Pixel::Skip {
                                return;
                            }

                            if current_pixel != neighbor_pixel {
                                let neighbor_pixel_representative =
                                    uf.get_representative_ref(neighbor_i);

                                if uf.get_set_size_ref(neighbor_pixel_representative) < 25 {
                                    return;
                                }

                                let key = if current_pixel_representative
                                    < neighbor_pixel_representative
                                {
                                    (current_pixel_representative, neighbor_pixel_representative)
                                } else {
                                    (neighbor_pixel_representative, current_pixel_representative)
                                };

                                let entry = local.entry(key).or_default();

                                let delta = neighbor_pixel.gradient_to(current_pixel);
                                let gradient_info = GradientInfo {
                                    pos: Point2d {
                                        x: (2 * x as isize + dx) as u32,
                                        y: (2 * y as isize + dy) as u32,
                                    },
                                    gx: delta * dx,
                                    gy: delta * dy,
                                    slope: 0.0,
                                };

                                entry.push(gradient_info);
                                *any_connected = true;
                            }
                        };

                    do_conn(1, 0, i + 1, &mut any_connected);
                    do_conn(0, 1, i + width, &mut any_connected);

                    if !connected_last {
                        do_conn(-1, 1, i + width - 1, &mut any_connected);
                    }

                    any_connected = false;

                    do_conn(1, 1, i + width + 1, &mut any_connected);

                    connected_last = any_connected;
                }
            }

            local
        })
        .collect();

    // Merge thread-local maps into one, preserving row-major (y, x) insertion order
    // so that the slope-sort in fit_single_quad has a stable, deterministic tiebreak.
    let mut clusters: FxHashMap<(usize, usize), Vec<GradientInfo>> = FxHashMap::with_capacity_and_hasher(128, Default::default());
    for map in local_maps {
        for (key, mut entries) in map {
            clusters.entry(key).or_default().append(&mut entries);
        }
    }
    for entries in clusters.values_mut() {
        entries.sort_unstable_by(|a, b| a.pos.y.cmp(&b.pos.y).then(a.pos.x.cmp(&b.pos.x)));
    }
    clusters
}

// ── NEON-accelerated gradient-cluster scan using pre-built rep_cache ──────────

/// Emit one gradient entry when the pixel at `$ni` is in a different large component
/// with a different color from `$cur_rep` / `$current_pixel`.
///
/// rep_cache encodes u32::MAX for skip/small-component pixels, else root as u32.
macro_rules! maybe_add_gradient {
    (
        $rep_cache:expr, $src_slice:expr, $local:expr,
        $cur_rep:expr, $current_pixel:expr,
        $x:expr, $y:expr, $ni:expr, $dx:expr, $dy:expr, $any_conn:expr
    ) => {{
        let nr: u32 = $rep_cache[$ni];
        if nr != u32::MAX {
            let nrep: usize = nr as usize;
            if $cur_rep != nrep {
                let npix: Pixel = $src_slice[$ni];
                if $current_pixel != npix {
                    let key = if $cur_rep < nrep {
                        ($cur_rep, nrep)
                    } else {
                        (nrep, $cur_rep)
                    };
                    let delta = npix.gradient_to($current_pixel);
                    $local.entry(key).or_insert_with(|| Vec::with_capacity(256)).push(GradientInfo {
                        pos: Point2d {
                            x: (2 * $x as isize + ($dx as isize)) as u32,
                            y: (2 * $y as isize + ($dy as isize)) as u32,
                        },
                        gx: delta * ($dx as isize),
                        gy: delta * ($dy as isize),
                        slope: 0.0,
                    });
                    $any_conn = true;
                }
            }
        }
    }};
}

/// Process one pixel of the gradient-cluster scan: emit gradients to its right,
/// below, below-left (unless the previous pixel already connected diagonally), and
/// below-right neighbors, and update `$connected_last` for the next pixel.
///
/// This is the per-pixel body shared by the scalar, NEON, and AVX2 inner loops; the
/// SIMD variants only differ in how they *skip* spans that provably have no gradient.
macro_rules! process_grad_pixel {
    (
        $rep_cache:expr, $src_slice:expr, $local:expr, $width:expr,
        $row_off:expr, $x:expr, $y:expr, $connected_last:expr
    ) =>
    // `any_connected` is written by the right/below/below-left taps but only the
    // below-right result feeds `connected_last`; the earlier writes are intentional
    // side-effects of the shared `maybe_add_gradient!` macro, hence allow(unused_assignments).
    {{
        #[allow(unused_assignments)]
        {
            let x = $x;
            let i = $row_off + x;
            let cur_rep = $rep_cache[i];
            if cur_rep == u32::MAX {
                $connected_last = false;
            } else {
                let cur_rep_usize = cur_rep as usize;
                let current_pixel = $src_slice[i];
                let mut any_connected = false;
                maybe_add_gradient!($rep_cache, $src_slice, $local, cur_rep_usize, current_pixel, x, $y, i + 1,             1i32,  0i32, any_connected);
                maybe_add_gradient!($rep_cache, $src_slice, $local, cur_rep_usize, current_pixel, x, $y, i + $width,         0i32,  1i32, any_connected);
                if !$connected_last {
                    maybe_add_gradient!($rep_cache, $src_slice, $local, cur_rep_usize, current_pixel, x, $y, i + $width - 1, -1i32, 1i32, any_connected);
                }
                any_connected = false;
                maybe_add_gradient!($rep_cache, $src_slice, $local, cur_rep_usize, current_pixel, x, $y, i + $width + 1,     1i32,  1i32, any_connected);
                $connected_last = any_connected;
            }
        }
    }};
}

/// Scalar inner loop for gradient cluster scan (non-aarch64 fallback).
#[cfg(not(target_arch = "aarch64"))]
fn gradient_clusters_inner_scalar(
    src_slice: &[Pixel],
    rep_cache: &[u32],
    width: usize,
    y_start: usize,
    y_end: usize,
) -> FxHashMap<(usize, usize), Vec<GradientInfo>> {
    let mut local: FxHashMap<(usize, usize), Vec<GradientInfo>> = FxHashMap::with_capacity_and_hasher(128, Default::default());
    for y in y_start..y_end {
        let row_off = y * width;
        let mut connected_last = false;
        for x in 1..width - 1 {
            process_grad_pixel!(rep_cache, src_slice, local, width, row_off, x, y, connected_last);
        }
    }
    local
}

/// NEON-accelerated inner loop (aarch64 only).
///
/// Uses a 16-pixel COLOR-based fast-path: loads 16 src bytes per vector and checks
/// same-color / Skip for all 4 neighbor directions simultaneously.  Same-color pairs
/// never produce a gradient regardless of component membership.
///
/// Fast-path rate on a typical AprilTag image: ~85–90% of pixels skip 16 at a time.
/// Falls back to one inlined scalar pixel (with rep_cache O(1) check) for the ~10–15%
/// of pixels near genuine Black↔White component boundaries.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn gradient_clusters_inner_neon(
    src_slice: &[Pixel],
    rep_cache: &[u32],
    width: usize,
    y_start: usize,
    y_end: usize,
) -> FxHashMap<(usize, usize), Vec<GradientInfo>> {
    use std::arch::aarch64::*;
    let mut local: FxHashMap<(usize, usize), Vec<GradientInfo>> = FxHashMap::with_capacity_and_hasher(128, Default::default());
    let skip_v = vdupq_n_u8(127u8); // Pixel::Skip = 127

    let rcp = rep_cache.as_ptr() as *const i8;
    for y in y_start..y_end {
        let row_off = y * width;

        // Prefetch the next row's rep_cache ahead of the slow-path accesses.
        // NOTE: intentionally uses rcp (byte pointer) with (y+1)*width offset —
        // this places the prefetch at rep_cache byte[(y+1)*width], which corresponds to
        // u32 element (y+1)*width/4. The prefix is wrong per se but empirically
        // outperforms both the correct-offset version and no prefetch on Jetson Orin;
        // likely warms data that the hardware prefetcher would otherwise miss.
        if y + 1 < y_end {
            let next = rcp.add((y + 1) * width);
            let mut px = 0isize;
            while px < width as isize {
                core::arch::asm!(
                    "prfm pldl1keep, [{p}]",
                    p = in(reg) next.offset(px * 4),
                    options(nostack, readonly, preserves_flags)
                );
                px += 16; // 16 u32 = 64 bytes = 1 cache line
            }
        }

        let mut connected_last = false;
        let mut x = 1usize;

        // 16-pixel color-based fast-path.
        // no_grad(a, b) = (a == b) | (a == Skip) | (b == Skip)
        // When ALL 16 lanes of all 4 directions are 0xFF → no pixel in [x, x+15]
        // can possibly have a gradient → advance 16 pixels in ~8 NEON ops.
        while x + 16 <= width - 1 {
            let src_ptr = src_slice.as_ptr().add(row_off + x) as *const u8;

            let curr_c = vld1q_u8(src_ptr);
            let right_c = vld1q_u8(src_ptr.add(1));
            let below_c = vld1q_u8(src_ptr.add(width));
            let bl_c    = vld1q_u8(src_ptr.add(width - 1));
            let br_c    = vld1q_u8(src_ptr.add(width + 1));

            let no_r  = vorrq_u8(vceqq_u8(curr_c, right_c),
                        vorrq_u8(vceqq_u8(curr_c, skip_v), vceqq_u8(right_c, skip_v)));
            let no_d  = vorrq_u8(vceqq_u8(curr_c, below_c),
                        vorrq_u8(vceqq_u8(curr_c, skip_v), vceqq_u8(below_c, skip_v)));
            let mut no_bl = vorrq_u8(vceqq_u8(curr_c, bl_c),
                             vorrq_u8(vceqq_u8(curr_c, skip_v), vceqq_u8(bl_c, skip_v)));
            // Pixel x+0 skips its below-left check when connected_last is true;
            // mark lane 0 as "no gradient" so it doesn't block the fast-path.
            if connected_last {
                no_bl = vsetq_lane_u8::<0>(0xFFu8, no_bl);
            }
            let no_br = vorrq_u8(vceqq_u8(curr_c, br_c),
                        vorrq_u8(vceqq_u8(curr_c, skip_v), vceqq_u8(br_c, skip_v)));

            let all_ok = vandq_u8(vandq_u8(no_r, no_d), vandq_u8(no_bl, no_br));

            if vminvq_u8(all_ok) == 0xFFu8 {
                x += 16;
                connected_last = false;
                continue;
            }

            // Some pixel in [x, x+15] is near a genuine Black↔White boundary.
            // Process all 16 pixels scalarly.
            {
                let chunk_end = (x + 16).min(width - 1);
                let mut xi = x;
                // NEON inner loop: 4 rep_cache u32 at a time → skip whole group if all MAX.
                let max_u32v = vdupq_n_u32(u32::MAX);
                while xi + 4 <= chunk_end {
                    let rcv = vld1q_u32(rep_cache.as_ptr().add(row_off + xi) as *const u32);
                    if vminvq_u32(vceqq_u32(rcv, max_u32v)) != 0 {
                        xi += 4;
                        connected_last = false;
                        continue;
                    }
                    for offset in 0..4usize {
                        if xi + offset >= chunk_end { break; }
                        process_grad_pixel!(rep_cache, src_slice, local, width, row_off, xi + offset, y, connected_last);
                    }
                    xi += 4;
                }
                // Tail: up to 3 pixels not covered by the 4-wide inner loop.
                while xi < chunk_end {
                    process_grad_pixel!(rep_cache, src_slice, local, width, row_off, xi, y, connected_last);
                    xi += 1;
                }
                x = chunk_end;
            }
        }

        // Tail: up to 15 pixels at the right edge not covered by the NEON outer loop.
        while x < width - 1 {
            process_grad_pixel!(rep_cache, src_slice, local, width, row_off, x, y, connected_last);
            x += 1;
        }
    }

    local
}

/// AVX2-accelerated inner loop (x86_64), mirroring [`gradient_clusters_inner_neon`].
///
/// 32-pixel COLOR-based fast-path: one `loadu` per neighbor direction (curr/right/
/// below/below-left/below-right); a pixel can only produce a gradient where the two
/// colors differ and neither is `Skip`. When all 32 lanes across all 4 directions
/// say "no gradient", the whole span is skipped. The rare boundary spans fall back
/// to the identical per-pixel `maybe_add_gradient!` logic, so output equals scalar.
///
/// # Safety
/// AVX2 must be available; row layout must allow reading `width+1` ahead (guaranteed
/// because `y_end < height` for inner strips and the fast-path is bounded by `width-1`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn gradient_clusters_inner_avx2(
    src_slice: &[Pixel],
    rep_cache: &[u32],
    width: usize,
    y_start: usize,
    y_end: usize,
) -> FxHashMap<(usize, usize), Vec<GradientInfo>> {
    use std::arch::x86_64::*;
    let mut local: FxHashMap<(usize, usize), Vec<GradientInfo>> =
        FxHashMap::with_capacity_and_hasher(128, Default::default());
    let skip_v = _mm256_set1_epi8(127i8); // Pixel::Skip = 127
    // Mask with only byte 0 set, used to neutralize the below-left lane of pixel x
    // when `connected_last` (that pixel skips its below-left check).
    let lane0 = _mm256_setr_epi8(
        -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0,
    );

    for y in y_start..y_end {
        let row_off = y * width;
        let mut connected_last = false;
        let mut x = 1usize;

        // 32-pixel color-based fast-path.
        while x + 32 <= width - 1 {
            let src_ptr = src_slice.as_ptr().add(row_off + x) as *const u8;
            let curr = _mm256_loadu_si256(src_ptr as *const __m256i);
            let right = _mm256_loadu_si256(src_ptr.add(1) as *const __m256i);
            let below = _mm256_loadu_si256(src_ptr.add(width) as *const __m256i);
            let bl = _mm256_loadu_si256(src_ptr.add(width - 1) as *const __m256i);
            let br = _mm256_loadu_si256(src_ptr.add(width + 1) as *const __m256i);

            let eqskip_curr = _mm256_cmpeq_epi8(curr, skip_v);
            let no_r = _mm256_or_si256(
                _mm256_cmpeq_epi8(curr, right),
                _mm256_or_si256(eqskip_curr, _mm256_cmpeq_epi8(right, skip_v)),
            );
            let no_d = _mm256_or_si256(
                _mm256_cmpeq_epi8(curr, below),
                _mm256_or_si256(eqskip_curr, _mm256_cmpeq_epi8(below, skip_v)),
            );
            let mut no_bl = _mm256_or_si256(
                _mm256_cmpeq_epi8(curr, bl),
                _mm256_or_si256(eqskip_curr, _mm256_cmpeq_epi8(bl, skip_v)),
            );
            if connected_last {
                no_bl = _mm256_or_si256(no_bl, lane0);
            }
            let no_br = _mm256_or_si256(
                _mm256_cmpeq_epi8(curr, br),
                _mm256_or_si256(eqskip_curr, _mm256_cmpeq_epi8(br, skip_v)),
            );
            let all_ok = _mm256_and_si256(_mm256_and_si256(no_r, no_d), _mm256_and_si256(no_bl, no_br));

            if _mm256_movemask_epi8(all_ok) == -1 {
                x += 32;
                connected_last = false;
                continue;
            }

            // Boundary span: process [x, chunk_end) with the exact scalar logic.
            let chunk_end = (x + 32).min(width - 1);
            let mut xi = x;
            while xi < chunk_end {
                process_grad_pixel!(rep_cache, src_slice, local, width, row_off, xi, y, connected_last);
                xi += 1;
            }
            x = chunk_end;
        }

        // Tail: up to 31 pixels at the right edge.
        while x < width - 1 {
            process_grad_pixel!(rep_cache, src_slice, local, width, row_off, x, y, connected_last);
            x += 1;
        }
    }

    local
}

/// Dispatch to NEON (aarch64), AVX2 (x86_64), or scalar inner loop.
fn gradient_clusters_inner(
    src_slice: &[Pixel],
    rep_cache: &[u32],
    width: usize,
    y_start: usize,
    y_end: usize,
) -> FxHashMap<(usize, usize), Vec<GradientInfo>> {
    // AArch64: NEON is baseline.
    #[cfg(target_arch = "aarch64")]
    // SAFETY: AArch64 always has NEON (mandatory in ARMv8-A).
    return unsafe { gradient_clusters_inner_neon(src_slice, rep_cache, width, y_start, y_end) };

    // x86_64: AVX2 when the runtime probe confirms it.
    #[cfg(target_arch = "x86_64")]
    if crate::ops::has_avx2() {
        // SAFETY: AVX2 confirmed by runtime probe.
        return unsafe { gradient_clusters_inner_avx2(src_slice, rep_cache, width, y_start, y_end) };
    }

    // Portable scalar fallback.
    #[cfg(not(target_arch = "aarch64"))]
    gradient_clusters_inner_scalar(src_slice, rep_cache, width, y_start, y_end)
}


/// Gradient-cluster scan using a pre-built rep_cache (no union-find traversal in the hot path).
///
/// rep_cache encodes u32::MAX for pixels that should be skipped (isolated or small components),
/// else the root pixel index as u32. Built by [`UnionFind::compress_and_fill_rep_cache`].
pub(crate) fn find_gradient_clusters_with_cache(
    src: &Image<Pixel, 1>,
    rep_cache: &[u32],
) -> Vec<FxHashMap<(usize, usize), Vec<GradientInfo>>> {
    let height = src.height();
    let width = src.width();
    let src_slice = src.as_slice();

    let n_threads = rayon::current_num_threads().max(1);
    let inner_rows = height.saturating_sub(2);
    let strip_h = (inner_rows + n_threads - 1) / n_threads;

    (0..n_threads)
        .into_par_iter()
        .map(|t| {
            let y_start = 1 + t * strip_h;
            if y_start >= height - 1 {
                return FxHashMap::default();
            }
            let y_end = (y_start + strip_h).min(height - 1);
            gradient_clusters_inner(src_slice, rep_cache, width, y_start, y_end)
        })
        .collect()
}

/// Finds and groups gradient transitions between connected components in a binary image.
/// Builds a temporary rep_cache from `uf` internally.
///
/// Callers that already have a rep_cache should prefer [`find_gradient_clusters_with_cache`].
pub fn find_gradient_clusters_cached(
    src: &Image<Pixel, 1>,
    uf: &mut UnionFind,
) -> Vec<FxHashMap<(usize, usize), Vec<GradientInfo>>> {
    let mut rep_cache = vec![u32::MAX; uf.len()];
    uf.compress_and_fill_rep_cache(&mut rep_cache, 25);
    find_gradient_clusters_with_cache(src, &rep_cache)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::threshold::{adaptive_threshold, TileMinMax};
    use kornia_image::{ImageSize};
    use kornia_io::png::read_image_png_mono8;

    #[test]
    fn test_basic_segmentation() -> Result<(), Box<dyn std::error::Error>> {
        use Pixel::*;
        #[rustfmt::skip]
        let bin_data = vec![
            Black, Black, White, White, White,
            Black, Black, White, White, White,
            White, White, Black, Black, Black,
            White, White, Black, Black, Black
        ];

        let bin = Image::new(
            ImageSize {
                width: 5,
                height: 4,
            },
            bin_data,
        )?;

        let mut uf = UnionFind::new(20);
        find_connected_components(&bin, &mut uf)?;

        assert_eq!(uf.get_representative(0), 0);
        assert_eq!(uf.get_representative(1), 0);
        assert_eq!(uf.get_representative(2), 2);
        assert_eq!(uf.get_representative(3), 2);
        assert_eq!(uf.get_representative(4), 4);
        assert_eq!(uf.get_representative(5), 0);
        assert_eq!(uf.get_representative(6), 0);
        assert_eq!(uf.get_representative(7), 2);
        assert_eq!(uf.get_representative(8), 2);
        assert_eq!(uf.get_representative(9), 9);
        assert_eq!(uf.get_representative(10), 2);
        assert_eq!(uf.get_representative(11), 2);
        assert_eq!(uf.get_representative(12), 12);
        assert_eq!(uf.get_representative(13), 12);
        assert_eq!(uf.get_representative(14), 14);
        assert_eq!(uf.get_representative(15), 2);
        assert_eq!(uf.get_representative(16), 2);
        assert_eq!(uf.get_representative(17), 12);
        assert_eq!(uf.get_representative(18), 12);
        assert_eq!(uf.get_representative(19), 19);

        Ok(())
    }

    #[test]
    fn test_segmentation() -> Result<(), Box<dyn std::error::Error>> {
        let src = read_image_png_mono8("../../tests/data/apriltag.png")?;
        let mut bin = Image::from_size_val(src.size(), Pixel::Skip)?;

        let mut tile_min_max = TileMinMax::new(src.size(), 4);
        adaptive_threshold(&src, &mut bin, &mut tile_min_max, 20)?;

        let mut uf = UnionFind::new(bin.as_slice().len());
        find_connected_components(&bin, &mut uf)?;

        let mut union_representatives = String::new();

        for i in 0..bin.as_slice().len() {
            let representative = uf.get_representative(i).to_string();

            union_representatives.push_str(&representative);
            union_representatives.push(' ');
        }

        if std::env::var("REGEN_GOLDEN").is_ok() {
            std::fs::write(
                "../../tests/data/apriltag_pixel_representatives.txt",
                &union_representatives,
            )?;
            return Ok(());
        }

        let expected =
            std::fs::read_to_string("../../tests/data/apriltag_pixel_representatives.txt")?;

        // The stored representatives are union-find root indices, whose absolute
        // values depend on scan/merge order (and thus differ across the NEON/AVX2/
        // scalar CC paths) even when the resulting partition is identical. Canonicalize
        // both sequences to first-occurrence order so the assertion checks the component
        // *structure* — the meaningful invariant — not the arbitrary root ids.
        fn canonical(labels: &str) -> Vec<u32> {
            let mut map = std::collections::HashMap::new();
            labels
                .split_whitespace()
                .map(|l| {
                    let next = map.len() as u32;
                    *map.entry(l.to_string()).or_insert(next)
                })
                .collect()
        }

        assert_eq!(
            canonical(&union_representatives),
            canonical(&expected),
            "connected-component partition differs from the golden fixture"
        );

        Ok(())
    }

    #[test]
    fn test_gradient_clusters() -> Result<(), Box<dyn std::error::Error>> {
        let src = read_image_png_mono8("../../tests/data/apriltag.png")?;
        let mut bin = Image::from_size_val(src.size(), Pixel::Skip)?;

        let mut tile_min_max = TileMinMax::new(src.size(), 4);
        adaptive_threshold(&src, &mut bin, &mut tile_min_max, 20)?;

        let mut uf = UnionFind::new(bin.as_slice().len());
        find_connected_components(&bin, &mut uf)?;

        let gradient_clusters = find_gradient_clusters(&bin, &uf);

        // Since the order of HashMap iteration is random, we cannot rely on the order of clusters.
        // However, we know from the expected data file that there are exactly 3 unique clusters,
        // each with a distinct length: 48, 188, and 192. We match clusters by their length and
        // compare their string representations to the expected output for each size.
        let expected = std::fs::read_to_string("../../tests/data/apriltag_gradient_clusters.txt")?;
        let mut expected_len_48 = String::new();
        let mut expected_len_188 = String::new();
        let mut expected_len_192 = String::new();

        for line in expected.lines() {
            if line.starts_with("size 48:") {
                expected_len_48 = line.to_string();
            } else if line.starts_with("size 188:") {
                expected_len_188 = line.to_string();
            } else if line.starts_with("size 192:") {
                expected_len_192 = line.to_string();
            }
        }

        for (_, infos) in gradient_clusters.iter() {
            let mut clusters = format!("size {}:\t", infos.len());

            for info in infos {
                let g_str = |g: GradientDirection| match g {
                    GradientDirection::None => 0,
                    GradientDirection::TowardsBlack => -255,
                    GradientDirection::TowardsWhite => 255,
                };
                clusters.push_str(
                    format!(
                        " (x={} y={} gx={} gy={})",
                        info.pos.x,
                        info.pos.y,
                        g_str(info.gx),
                        g_str(info.gy)
                    )
                    .as_str(),
                );
            }

            match infos.len() {
                48 => assert_eq!(expected_len_48, clusters),
                188 => assert_eq!(expected_len_188, clusters),
                192 => assert_eq!(expected_len_192, clusters),
                _ => panic!(
                    "Unexpected length of clusters, expected either 48, 188, or 192 but found {}",
                    infos.len()
                ),
            }
        }

        Ok(())
    }
}
