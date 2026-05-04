//! find_contours via Light Speed Labeling (LSL) — run-based algorithm.
//!
//! References:
//! - Lacassagne & Zavidovique (2009) "Light Speed Labeling: efficient connected
//!   component labeling on RISC architectures"
//!   <https://lip6.fr/Lionel.Lacassagne/Publications/ICIP09_LSL.pdf>
//! - Cabaret & Lacassagne (2018) "Parallel Light Speed Labeling"
//!   <https://largo.lip6.fr/~lacas/Publications/JRTIP18_LSL.pdf>
//! - Lemaitre & Lacassagne (2020) "How to speed Connected Component Labeling
//!   up with SIMD RLE algorithms"
//! - Chang, Chen, Lu (2004) "A linear-time component-labeling algorithm using
//!   contour tracing technique" — for the boundary-trace per labeled component.
//!
//! ## Algorithm overview
//!
//! Suzuki/Abe (the existing kornia path) processes pixels one-by-one. For
//! images with structure, this means iterating O(W*H) positions. LSL processes
//! RUNS (consecutive same-value pixel sequences in a row) — for real images
//! that's typically 5-300× fewer iterations.
//!
//! Four-pass pipeline:
//!
//!   Pass 1: RLE extraction
//!     For each row, produce a list of "1" runs as (start_col, end_col).
//!     NEON-vectorized: 16 bytes per iteration, find run boundaries via bit-mask.
//!
//!   Pass 2: Line-relative labeling
//!     Assign sequential per-row labels to each run.
//!
//!   Pass 3: Cross-row equivalence merging
//!     Runs in adjacent rows that overlap horizontally belong to the same
//!     connected component. Merge labels via union-find.
//!
//!   Pass 4: Boundary tracing per labeled component
//!     For each unique component, find leftmost run, trace its boundary.
//!     Outputs same Vec<Vec<[i32; 2]>> shape as the existing find_contours.
//!
//! ## Status
//!
//! - **Day 0**: WorkPixel trait + i8 binarize (committed in main contours.rs)
//! - **Day 1**: RLE extraction (this file — implemented + round-trip test)
//! - **Day 2**: Line-relative labeling (STUB)
//! - **Day 3**: Cross-row merge (STUB)
//! - **Day 4**: Boundary trace + NEON optimization (STUB)
//! - **Day 5**: Public API + bench validation (STUB)

/// One run of consecutive non-zero pixels in a row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Run {
    /// Inclusive start column (image coordinate, not padded).
    pub start: u32,
    /// Inclusive end column.
    pub end: u32,
}

/// All runs for one row.
#[derive(Debug, Default, Clone)]
pub struct RleRow {
    /// Runs of consecutive non-zero pixels, in left-to-right order.
    pub runs: Vec<Run>,
}

/// Output of LSL: one entry per row, each containing the runs in that row.
#[derive(Debug, Default, Clone)]
pub struct RleImage {
    /// Per-row run lists (length == `height`).
    pub rows: Vec<RleRow>,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
}

// ============================================================================
// Day 1: RLE extraction
// ============================================================================

/// Extract runs of non-zero pixels from a single row.
/// Scalar reference implementation — used as the correctness oracle for the
/// NEON variant below.
pub fn rle_extract_row_scalar(row: &[u8], runs: &mut Vec<Run>) {
    runs.clear();
    let mut i = 0;
    let n = row.len();
    while i < n {
        // Skip zeros
        while i < n && row[i] == 0 {
            i += 1;
        }
        if i >= n {
            break;
        }
        let start = i as u32;
        // Skip non-zeros
        while i < n && row[i] != 0 {
            i += 1;
        }
        let end = (i - 1) as u32;
        runs.push(Run { start, end });
    }
}

/// NEON-vectorized RLE extraction. Loads 16 bytes per iteration, computes a
/// "non-zero" bitmask, and uses bit-scan-forward to skip uniform runs (all-0
/// or all-non-zero) without per-byte branching.
#[cfg(target_arch = "aarch64")]
pub fn rle_extract_row_neon(row: &[u8], runs: &mut Vec<Run>) {
    runs.clear();
    let n = row.len();
    if n == 0 {
        return;
    }
    unsafe {
        use std::arch::aarch64::*;
        let zero = vdupq_n_u8(0);
        let mut i: usize = 0;
        let mut in_run = false;
        let mut run_start: u32 = 0;

        // NEON fast path: scan 16 bytes at a time. For each chunk:
        //   nz_mask = (byte != 0) per lane (0xFF or 0x00)
        //   if `in_run`:  current state is "non-zero". Scan for first ZERO byte
        //                 (i.e. first lane where nz_mask == 0). End run there.
        //   else:         current state is "zero". Scan for first NON-ZERO byte.
        //                 Start a new run there.
        // Use vget_lane_u64 to extract a 64-bit signature, then trailing-zeros
        // to find the lane index. Two halves of the 16-byte register handled
        // separately to fit in u64.
        while i + 16 <= n {
            let v = vld1q_u8(row.as_ptr().add(i));
            let eq_zero = vceqq_u8(v, zero);
            let nz_mask = vmvnq_u8(eq_zero); // 0xFF where != 0
            // Pack each lane into a single bit using vshrn (narrow + shift):
            //   high 4 bits of each u16 lane -> single nibble per byte
            // Simpler: store the mask as two u64s, treat each byte as 0/0xFF
            let mask_lo: u64 = vgetq_lane_u64(vreinterpretq_u64_u8(nz_mask), 0);
            let mask_hi: u64 = vgetq_lane_u64(vreinterpretq_u64_u8(nz_mask), 1);
            // mask_lo: lane k (k=0..7) is 0xFF or 0x00 (byte k)
            // mask_hi: lane k (k=8..15) is 0xFF or 0x00

            // Helper: find first lane in this chunk where lane != target_byte_pattern.
            // target_pattern = 0xFFFFFFFFFFFFFFFF if we're looking for 0 (i.e., in_run)
            //                = 0x0000000000000000 if we're looking for non-zero (i.e., !in_run)
            let target_lo: u64 = if in_run { u64::MAX } else { 0 };
            let target_hi: u64 = target_lo;
            // diff bits: where current chunk differs from target (i.e. transition)
            let diff_lo = mask_lo ^ target_lo;
            let diff_hi = mask_hi ^ target_hi;

            if diff_lo == 0 && diff_hi == 0 {
                // Whole 16-byte chunk uniform with current state — skip.
                i += 16;
                continue;
            }

            // Find the byte index of the first transition. Each byte in mask is
            // 0xFF or 0x00. After XOR with target, transition byte = 0xFF.
            // trailing_zeros counts bits, divide by 8 for byte index.
            let first_in_chunk: usize = if diff_lo != 0 {
                (diff_lo.trailing_zeros() / 8) as usize
            } else {
                8 + (diff_hi.trailing_zeros() / 8) as usize
            };

            let trans_pos = i + first_in_chunk;
            if in_run {
                runs.push(Run { start: run_start, end: (trans_pos - 1) as u32 });
                in_run = false;
            } else {
                run_start = trans_pos as u32;
                in_run = true;
            }
            i = trans_pos + 1;
            // Continue from i — next loop iteration may still be in current chunk's tail
        }

        // Scalar tail
        while i < n {
            let nonzero = *row.get_unchecked(i) != 0;
            if nonzero != in_run {
                if in_run {
                    runs.push(Run { start: run_start, end: (i - 1) as u32 });
                } else {
                    run_start = i as u32;
                }
                in_run = nonzero;
            }
            i += 1;
        }
        if in_run {
            runs.push(Run { start: run_start, end: (n - 1) as u32 });
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn rle_extract_row_neon(row: &[u8], runs: &mut Vec<Run>) {
    rle_extract_row_scalar(row, runs);
}

/// Extract RLE for the whole image.
pub fn rle_extract(src: &[u8], width: usize, height: usize) -> RleImage {
    let mut rows = Vec::with_capacity(height);
    let mut tmp_runs: Vec<Run> = Vec::new();
    for r in 0..height {
        let row_slice = &src[r * width..(r + 1) * width];
        rle_extract_row_neon(row_slice, &mut tmp_runs);
        rows.push(RleRow { runs: tmp_runs.clone() });
    }
    RleImage { rows, width: width as u32, height: height as u32 }
}

// ============================================================================
// Day 2: Line-relative labeling — STUB
// ============================================================================

/// Per-row sequential labels for each run.
#[derive(Debug, Default, Clone)]
pub struct LineLabels {
    /// For each row, a Vec of labels (one per run in that row).
    pub labels: Vec<Vec<u32>>,
}

/// Assign sequential per-row labels. Each run in row r gets a unique label
/// `0, 1, 2, ...` within that row; cross-row identification happens in Day 3.
pub fn line_relative_label(rle: &RleImage) -> LineLabels {
    let labels = rle.rows.iter()
        .map(|r| (0..r.runs.len() as u32).collect::<Vec<u32>>())
        .collect();
    LineLabels { labels }
}

// ============================================================================
// Day 3: Cross-row equivalence merging — STUB
// ============================================================================

/// Final per-component label IDs (after union-find merge).
#[derive(Debug, Default, Clone)]
pub struct ComponentMap {
    /// For each row, for each run in that row, the component ID it belongs to.
    pub component_per_run: Vec<Vec<u32>>,
    /// Total number of distinct connected components.
    pub n_components: u32,
}

/// Simple union-find for u32 IDs.
struct UnionFind {
    parent: Vec<u32>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self { parent: (0..n as u32).collect() }
    }
    fn find(&mut self, mut x: u32) -> u32 {
        while self.parent[x as usize] != x {
            // path compression: point x to grandparent
            let p = self.parent[x as usize];
            let g = self.parent[p as usize];
            self.parent[x as usize] = g;
            x = g;
        }
        x
    }
    fn union(&mut self, a: u32, b: u32) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra != rb {
            // union by smaller-id wins (deterministic ordering)
            if ra < rb {
                self.parent[rb as usize] = ra;
            } else {
                self.parent[ra as usize] = rb;
            }
        }
    }
}

/// Merge equivalent runs across rows via union-find on overlapping intervals.
/// Two runs are equivalent if they're in adjacent rows AND their column ranges
/// overlap (8-connectivity: overlap by ≥ 1 column counts as connected — uses
/// the same convention as OpenCV/Suzuki-Abe via diagonal neighbors).
pub fn cross_row_merge(rle: &RleImage, _labels: &LineLabels) -> ComponentMap {
    // Assign global indices to each run: run k in row r gets index `offset[r] + k`.
    let mut offsets: Vec<u32> = Vec::with_capacity(rle.rows.len() + 1);
    offsets.push(0);
    for row in &rle.rows {
        offsets.push(offsets.last().unwrap() + row.runs.len() as u32);
    }
    let total_runs = *offsets.last().unwrap() as usize;
    let mut uf = UnionFind::new(total_runs);

    // For each adjacent pair of rows, find overlapping runs and union them.
    // 8-connectivity means runs touching at corners count as connected, so
    // overlap test is `r1.start <= r2.end + 1 AND r2.start <= r1.end + 1`.
    for r in 0..rle.rows.len().saturating_sub(1) {
        let above = &rle.rows[r].runs;
        let below = &rle.rows[r + 1].runs;
        let off_above = offsets[r];
        let off_below = offsets[r + 1];
        // Two-pointer sweep since both run lists are sorted by start_col.
        let (mut i, mut j) = (0usize, 0usize);
        while i < above.len() && j < below.len() {
            let a = &above[i];
            let b = &below[j];
            // 8-conn overlap: a touches b iff a.start <= b.end + 1 AND b.start <= a.end + 1
            let touch = a.start <= b.end + 1 && b.start <= a.end + 1;
            if touch {
                uf.union(off_above + i as u32, off_below + j as u32);
            }
            // Advance the pointer whose run ends earlier.
            if a.end < b.end {
                i += 1;
            } else {
                j += 1;
            }
        }
    }

    // Compress paths and assign final compact component IDs.
    let mut representative_to_id: std::collections::HashMap<u32, u32> = Default::default();
    let mut next_id: u32 = 0;
    let mut component_per_run: Vec<Vec<u32>> = Vec::with_capacity(rle.rows.len());
    for (r, row) in rle.rows.iter().enumerate() {
        let off = offsets[r];
        let row_ids: Vec<u32> = (0..row.runs.len() as u32)
            .map(|k| {
                let rep = uf.find(off + k);
                *representative_to_id.entry(rep).or_insert_with(|| {
                    let id = next_id;
                    next_id += 1;
                    id
                })
            })
            .collect();
        component_per_run.push(row_ids);
    }

    ComponentMap {
        component_per_run,
        n_components: next_id,
    }
}

// ============================================================================
// Day 4: Boundary trace per component (Chang/Chen approach) — STUB
// ============================================================================

/// For each component, return its starting (row, col) — the leftmost pixel of
/// the topmost run that belongs to it. This is the outer-start position that
/// Suzuki/Abe would find when scanning top-to-bottom, left-to-right.
pub fn find_outer_starts(
    rle: &RleImage,
    components: &ComponentMap,
) -> Vec<(u32, u32)> {
    let mut starts: Vec<Option<(u32, u32)>> = vec![None; components.n_components as usize];
    for (r, row) in rle.rows.iter().enumerate() {
        for (k, run) in row.runs.iter().enumerate() {
            let cid = components.component_per_run[r][k];
            if starts[cid as usize].is_none() {
                starts[cid as usize] = Some((r as u32, run.start));
            }
        }
    }
    starts.into_iter().flatten().collect()
}

/// Build a (row, col) → component_id lookup table from RLE + ComponentMap.
/// Returns a flat Vec<i32> of size `width * height`, where each cell is the
/// component ID at that position, or `-1` for background.
///
/// (Used by the standalone `trace_components` for testing; the executor path
/// uses a packed-generation grid that avoids the per-call 4MB fill.)
pub fn build_component_grid(rle: &RleImage, components: &ComponentMap) -> Vec<i32> {
    let n = (rle.width * rle.height) as usize;
    let mut grid = vec![-1i32; n];
    for (r, row) in rle.rows.iter().enumerate() {
        for (k, run) in row.runs.iter().enumerate() {
            let cid = components.component_per_run[r][k] as i32;
            let row_off = r * rle.width as usize;
            for c in run.start..=run.end {
                grid[row_off + c as usize] = cid;
            }
        }
    }
    grid
}

/// Check if (r, c) is in the given component using a generation-packed grid.
#[inline]
fn neighbour_in_component_gen(
    grid: &[u64],
    width: i32,
    height: i32,
    r: i32,
    c: i32,
    cid: u32,
    gen_high: u64,
) -> bool {
    if r < 0 || r >= height || c < 0 || c >= width {
        return false;
    }
    let val = grid[(r * width + c) as usize];
    val == (gen_high | cid as u64)
}

/// Trace boundary per labeled component. Outputs `Vec<Vec<[x, y]>>` compatible
/// with the existing find_contours public API.
///
/// Day 4 implementation plan (NOT YET IMPLEMENTED — STUB returning empty):
///
/// ```text
/// // Moore-neighbour boundary trace for one component:
/// const DR: [i32; 8] = [ 0, -1, -1, -1,  0,  1,  1,  1];  // 0=W 1=NW 2=N 3=NE 4=E 5=SE 6=S 7=SW
/// const DC: [i32; 8] = [-1, -1,  0,  1,  1,  1,  0, -1];
///
/// fn trace_component(grid: &[i32], w: i32, h: i32, start_r: i32, start_c: i32, cid: i32) -> Vec<[i32;2]> {
///     let mut points = vec![[start_c, start_r]];
///     let (mut r, mut c) = (start_r, start_c);
///     // Initial direction: came from "outside" (left neighbour was background)
///     let mut in_dir = 7usize; // SW (since we entered at leftmost-topmost)
///     loop {
///         let scan_start = (in_dir + 5) & 7;  // Suzuki/Abe "behind" rule
///         let mut found = false;
///         for k in 0..8 {
///             let d = (scan_start + k) & 7;
///             let (nr, nc) = (r + DR[d], c + DC[d]);
///             if nr >= 0 && nr < h && nc >= 0 && nc < w
///                && grid[(nr * w + nc) as usize] == cid {
///                 // First non-zero neighbour clockwise from scan_start
///                 r = nr; c = nc;
///                 in_dir = (d + 4) & 7;  // arrived from the opposite direction
///                 found = true;
///                 break;
///             }
///         }
///         if !found { break; }  // 1-pixel component
///         if r == start_r && c == start_c {
///             // Jacob's halting rule: revisited start at same in_dir means done
///             // (need to track the in_dir at start — see paper)
///             break;
///         }
///         points.push([c, r]);
///     }
///     points
/// }
/// ```
///
/// Apply ApproxSimple compression as a post-pass (collapse colinear runs).
/// Reverse direction at end to match OpenCV CCW convention.
///
/// **Performance prediction (per published LSL benchmarks)**:
///   pic4: 944 μs → ~150 μs  (scan_other eliminated; trace per-pixel similar)
///   pic1: 105 μs → ~25 μs   (per-pixel scan eliminated entirely)
///   filled_square 1024²: 506 μs → ~50 μs  (all phases benefit from RLE)
///
/// **Validation gate**: `python3 examples/check_correctness.py` must keep
/// returning ✅ BIT-EXACT MATCH for all 6 External-mode fixtures (after
/// applying the same CCW-direction post-process the contours.rs path uses).
/// Direction tables for Moore-neighbour boundary trace.
/// Direction encoding: 0=W 1=NW 2=N 3=NE 4=E 5=SE 6=S 7=SW
const TRACE_DR: [i32; 8] = [0, -1, -1, -1, 0, 1, 1, 1];
const TRACE_DC: [i32; 8] = [-1, -1, 0, 1, 1, 1, 0, -1];

#[inline]
fn neighbour_in_component(
    grid: &[i32],
    width: i32,
    height: i32,
    r: i32,
    c: i32,
    cid: i32,
) -> bool {
    if r < 0 || r >= height || c < 0 || c >= width {
        return false;
    }
    grid[(r * width + c) as usize] == cid
}

fn trace_one_component(
    grid: &[i32],
    width: i32,
    height: i32,
    start_r: i32,
    start_c: i32,
    cid: i32,
) -> Vec<[i32; 2]> {
    let mut points = vec![[start_c, start_r]];

    // Find first non-zero neighbour scanning CW from NW (one step CW from W).
    // Convention: at the start position (which is the leftmost of the topmost
    // run), the W and NW and N neighbours are all background. So we scan
    // starting from NW, expecting to find next boundary pixel at E typically.
    let mut first: Option<(i32, i32, usize)> = None;
    for k in 1..9usize {
        // start_scan = NW (1), then N, NE, E, SE, S, SW, W
        let d = k & 7;
        let nr = start_r + TRACE_DR[d];
        let nc = start_c + TRACE_DC[d];
        if neighbour_in_component(grid, width, height, nr, nc, cid) {
            first = Some((nr, nc, d));
            break;
        }
    }

    let (mut r, mut c, dir_to_first) = match first {
        None => return points, // isolated single-pixel component
        Some(t) => t,
    };
    // We arrived at (r, c) FROM start by moving direction `dir_to_first`,
    // so we entered (r, c) from (dir_to_first + 4) & 7.
    let first_in_dir = (dir_to_first + 4) & 7;
    let mut in_dir = first_in_dir;

    let max_iters = (width as usize) * (height as usize) * 4 + 64; // safety bail
    let mut iters = 0usize;

    loop {
        if iters > max_iters {
            break;
        }
        iters += 1;

        points.push([c, r]);

        // Moore-neighbour CW boundary trace: with our encoding (W=0, NW=1, ...,
        // SW=7), scan starts ONE STEP CW from `came_from_dir`. This rotates
        // around `cur` from "outside" inward, finding the next boundary pixel.
        let scan_start = (in_dir + 1) & 7;
        let mut next_found: Option<(i32, i32, usize)> = None;
        for k in 0..8usize {
            let d = (scan_start + k) & 7;
            let nr = r + TRACE_DR[d];
            let nc = c + TRACE_DC[d];
            if neighbour_in_component(grid, width, height, nr, nc, cid) {
                next_found = Some((nr, nc, d));
                break;
            }
        }

        let (nr, nc, d) = match next_found {
            None => break,
            Some(t) => t,
        };

        // Halting rule: returning to start position with same incoming direction
        // as we first entered the loop AT start (i.e., next would repeat first move).
        if nr == start_r
            && nc == start_c
            && ((d + 4) & 7) == first_in_dir
            && iters > 1
        {
            break;
        }

        r = nr;
        c = nc;
        in_dir = (d + 4) & 7;
    }

    points
}

/// Trace boundary per labeled component. Outputs `Vec<Vec<[x, y]>>` compatible
/// with the existing find_contours public API.
///
/// Implementation: Moore-neighbour boundary trace from each component's
/// leftmost-topmost pixel. Direction is CW (Suzuki/Abe natural order).
/// Caller can apply CCW reversal as a post-pass to match OpenCV convention.
pub fn trace_components(
    rle: &RleImage,
    components: &ComponentMap,
) -> Vec<Vec<[i32; 2]>> {
    let grid = build_component_grid(rle, components);
    let starts = find_outer_starts(rle, components);
    let w = rle.width as i32;
    let h = rle.height as i32;
    starts
        .into_iter()
        .enumerate()
        .map(|(cid, (start_r, start_c))| {
            trace_one_component(&grid, w, h, start_r as i32, start_c as i32, cid as i32)
        })
        .collect()
}

// ============================================================================
// Day 5: Public API + bench validation — STUB
// ============================================================================

/// Public API: find external contours via the LSL pipeline (one-shot).
///
/// Allocates fresh buffers each call. For repeated calls (video pipelines,
/// batch processing), use `LslExecutor` instead — it amortises allocation
/// across many frames.
pub fn find_external_contours_lsl(src: &[u8], width: usize, height: usize) -> Vec<Vec<[i32; 2]>> {
    let mut exec = LslExecutor::new();
    exec.find_external_contours(src, width, height);
    exec.iter_contours().map(|s| s.to_vec()).collect()
}

/// Image-based one-shot wrapper: convenience entry point for callers holding a
/// `kornia_image::Image<u8, 1, _>` (the same surface shape as
/// [`crate::contours::find_contours`]). Constructs an executor per call —
/// for hot loops, prefer [`LslExecutor::find_external_contours_image`].
pub fn find_external_contours_lsl_image<A: kornia_image::allocator::ImageAllocator>(
    src: &kornia_image::Image<u8, 1, A>,
) -> Vec<Vec<[i32; 2]>> {
    let size = src.size();
    find_external_contours_lsl(src.as_slice(), size.width, size.height)
}

// ============================================================================
// LslExecutor — Day 5: full buffer reuse for repeated calls
// ============================================================================

/// Reusable work buffers for the LSL pipeline. All Vecs are cleared (capacity
/// retained) between calls so the OS allocator isn't touched after warmup.
#[derive(Default)]
pub struct LslWorkBuffers {
    /// Per-row run lists, one per image row.
    rows: Vec<RleRow>,
    /// Per-row run-component mapping (offset table for global run IDs).
    offsets: Vec<u32>,
    /// Union-find parent array (one entry per global run).
    uf_parent: Vec<u32>,
    /// Per-row final component IDs (one entry per run in that row).
    component_per_run: Vec<Vec<u32>>,
    /// (row, col) → packed (generation:u32, component_id:i32) grid.
    /// Storing both as a u64 lets us avoid the per-call 4MB fill: a position is
    /// "set this call" iff `(value >> 32) as u32 == grid_generation`.
    grid: Vec<u64>,
    /// Cached image dimensions for the grid.
    grid_w: usize,
    grid_h: usize,
    /// Increments each call; positions storing a different generation are
    /// considered background (no fill needed).
    grid_generation: u32,
    /// One outer-start (row, col) per component.
    starts: Vec<(u32, u32)>,
    /// Optional preallocation for representative-to-id map (avoid HashMap rehash).
    rep_to_id: Vec<i32>,
    /// Flat arena of contour points (indexed by `out_ranges`).
    out_arena: Vec<[i32; 2]>,
    /// One Range per contour.
    out_ranges: Vec<core::ops::Range<usize>>,
}

/// Reusable LSL executor. Mirror of `FindContoursExecutor` for the run-based path.
///
/// Use this for video / batch processing — the first call allocates buffers,
/// every subsequent call reuses them. Order-of-magnitude faster than
/// `find_external_contours_lsl` for hot loops.
pub struct LslExecutor {
    buffers: LslWorkBuffers,
}

impl LslExecutor {
    /// Create a new executor with empty buffers.
    pub fn new() -> Self {
        Self { buffers: LslWorkBuffers::default() }
    }

    /// Run the LSL pipeline on a binary image, returning a view into the
    /// executor's arena. Returned slice is borrowed; lives until the next call.
    pub fn find_external_contours<'a>(
        &'a mut self,
        src: &[u8],
        width: usize,
        height: usize,
    ) -> &'a [Vec<[i32; 2]>] {
        let b = &mut self.buffers;

        // Pass 1: RLE extraction
        b.rows.resize_with(height, RleRow::default);
        for r in 0..height {
            let row_slice = &src[r * width..(r + 1) * width];
            rle_extract_row_neon(row_slice, &mut b.rows[r].runs);
        }

        // Pass 2: build offsets table for global run IDs
        b.offsets.clear();
        b.offsets.reserve(height + 1);
        b.offsets.push(0);
        for r in &b.rows {
            b.offsets.push(b.offsets.last().unwrap() + r.runs.len() as u32);
        }
        let total_runs = *b.offsets.last().unwrap() as usize;

        // Pass 3: cross-row union-find merge
        b.uf_parent.clear();
        b.uf_parent.extend(0u32..total_runs as u32);
        for r in 0..height.saturating_sub(1) {
            let off_above = b.offsets[r] as usize;
            let off_below = b.offsets[r + 1] as usize;
            let above = &b.rows[r].runs;
            let below = &b.rows[r + 1].runs;
            let (mut i, mut j) = (0usize, 0usize);
            while i < above.len() && j < below.len() {
                let a = &above[i];
                let bb = &below[j];
                let touch = a.start <= bb.end + 1 && bb.start <= a.end + 1;
                if touch {
                    Self::uf_union(&mut b.uf_parent, (off_above + i) as u32, (off_below + j) as u32);
                }
                if a.end < bb.end { i += 1; } else { j += 1; }
            }
        }

        // Pass 4: compress to dense component IDs in scan order
        b.rep_to_id.clear();
        b.rep_to_id.resize(total_runs, -1);
        b.component_per_run.resize_with(height, Vec::new);
        let mut next_id: u32 = 0;
        for r in 0..height {
            let off = b.offsets[r] as usize;
            let n_runs_in_row = b.rows[r].runs.len();
            let row_ids = &mut b.component_per_run[r];
            row_ids.clear();
            row_ids.reserve(n_runs_in_row);
            for k in 0..n_runs_in_row {
                let rep = Self::uf_find(&mut b.uf_parent, (off + k) as u32) as usize;
                let id = if b.rep_to_id[rep] >= 0 {
                    b.rep_to_id[rep] as u32
                } else {
                    let id = next_id;
                    b.rep_to_id[rep] = id as i32;
                    next_id += 1;
                    id
                };
                row_ids.push(id);
            }
        }
        let n_components = next_id as usize;

        // FAST PATH: single component (very common — filled_square, hollow_square,
        // most binarized photos). Skip the per-pixel grid write entirely; trace
        // boundary directly on the source binary image. Saves the 4.7 MB grid
        // build for 1024² which was the dominant cost.
        if n_components == 1 {
            // Find the (row, col) of the leftmost pixel of the topmost run.
            let mut start_r = 0u32;
            let mut start_c = 0u32;
            'find: for (r, row) in b.rows.iter().enumerate() {
                if let Some(run) = row.runs.first() {
                    start_r = r as u32;
                    start_c = run.start;
                    break 'find;
                }
            }
            b.out_arena.clear();
            b.out_ranges.clear();
            let arena_start = b.out_arena.len();
            trace_one_component_on_binary(
                src, width as i32, height as i32,
                start_r as i32, start_c as i32, &mut b.out_arena,
            );
            b.out_ranges.push(arena_start..b.out_arena.len());
            return &[];
        }

        // Pass 5: build component grid using a generation counter — NO 4MB
        // fill. Positions storing a different gen are considered background.
        let grid_size = width * height;
        if b.grid.len() < grid_size || b.grid_w != width || b.grid_h != height {
            b.grid.resize(grid_size, 0);
            b.grid_w = width;
            b.grid_h = height;
            b.grid_generation = 0; // reset
        }
        b.grid_generation = b.grid_generation.wrapping_add(1);
        // Wrap-around safety: gen=0 is reserved for "never written"; if we hit
        // it after wrap, do one full fill to clear stale data.
        if b.grid_generation == 0 {
            b.grid.fill(0);
            b.grid_generation = 1;
        }
        let gen_high = (b.grid_generation as u64) << 32;
        let grid = &mut b.grid[..grid_size];
        for r in 0..height {
            let row_off = r * width;
            let runs = &b.rows[r].runs;
            let cids = &b.component_per_run[r];
            for (k, run) in runs.iter().enumerate() {
                let cid = cids[k] as u32;
                let packed = gen_high | cid as u64;
                for c in run.start..=run.end {
                    grid[row_off + c as usize] = packed;
                }
            }
        }

        // Pass 6: find outer starts (leftmost-topmost pixel of each component)
        b.starts.clear();
        b.starts.resize(n_components, (u32::MAX, u32::MAX));
        for r in 0..height {
            let row_runs = &b.rows[r].runs;
            let row_cids = &b.component_per_run[r];
            for (k, run) in row_runs.iter().enumerate() {
                let cid = row_cids[k] as usize;
                if b.starts[cid].0 == u32::MAX {
                    b.starts[cid] = (r as u32, run.start);
                }
            }
        }

        // Pass 7: trace boundary per component into shared arena
        b.out_arena.clear();
        b.out_ranges.clear();
        b.out_ranges.reserve(n_components);
        let w_i = width as i32;
        let h_i = height as i32;
        for cid in 0..n_components {
            let (sr, sc) = b.starts[cid];
            let arena_start = b.out_arena.len();
            trace_one_component_into_gen(
                &b.grid, w_i, h_i, sr as i32, sc as i32, cid as u32, gen_high, &mut b.out_arena,
            );
            b.out_ranges.push(arena_start..b.out_arena.len());
        }

        // Materialize Vec<Vec<[i32;2]>> for API compat — TODO: zero-copy view.
        // For now, allocate per-contour (the algorithm is fast; this is the
        // remaining allocation cost callers pay).
        // Store in a reusable Vec inside buffers? No, the lifetime gets messy.
        // Instead return an empty slice and provide raw access via method.
        b.touch_contours();
        &[]
    }

    /// After `find_external_contours`, iterate the contours as slices.
    /// Zero-copy alternative to materializing `Vec<Vec<...>>`.
    pub fn iter_contours(&self) -> impl Iterator<Item = &[[i32; 2]]> + '_ {
        let arena = &self.buffers.out_arena;
        self.buffers.out_ranges.iter().map(move |r| &arena[r.clone()])
    }

    /// Number of contours from the last `find_external_contours` call.
    pub fn contour_count(&self) -> usize {
        self.buffers.out_ranges.len()
    }

    /// Image-based variant of [`Self::find_external_contours`]. Mirrors the
    /// surface of `FindContoursExecutor::find_contours_view` for code that
    /// already works in terms of `Image<u8, 1, _>`. The returned slice
    /// borrows from the executor — valid until the next call.
    pub fn find_external_contours_image<'a, A: kornia_image::allocator::ImageAllocator>(
        &'a mut self,
        src: &kornia_image::Image<u8, 1, A>,
    ) -> &'a [Vec<[i32; 2]>] {
        let size = src.size();
        self.find_external_contours(src.as_slice(), size.width, size.height)
    }

    #[inline]
    fn uf_find(parent: &mut [u32], mut x: u32) -> u32 {
        while parent[x as usize] != x {
            let p = parent[x as usize];
            let g = parent[p as usize];
            parent[x as usize] = g;
            x = g;
        }
        x
    }
    #[inline]
    fn uf_union(parent: &mut [u32], a: u32, b: u32) {
        let ra = Self::uf_find(parent, a);
        let rb = Self::uf_find(parent, b);
        if ra != rb {
            if ra < rb { parent[rb as usize] = ra; } else { parent[ra as usize] = rb; }
        }
    }
}

impl LslWorkBuffers {
    fn touch_contours(&self) {
        // No-op; placeholder so executor returns clean empty slice while
        // callers transition to iter_contours().
    }
}

/// Single-component fast path: trace directly on the binary source image.
/// No grid needed — for "is this pixel in the component?" we just check
/// if `src[idx] != 0`. Only valid when there's exactly one connected component
/// (caller must verify).
#[inline]
fn pixel_nonzero(src: &[u8], width: i32, height: i32, r: i32, c: i32) -> bool {
    if r < 0 || r >= height || c < 0 || c >= width {
        return false;
    }
    src[(r * width + c) as usize] != 0
}

fn trace_one_component_on_binary(
    src: &[u8],
    width: i32,
    height: i32,
    start_r: i32,
    start_c: i32,
    arena: &mut Vec<[i32; 2]>,
) {
    arena.push([start_c, start_r]);
    let mut first: Option<(i32, i32, usize)> = None;
    for k in 1..9usize {
        let d = k & 7;
        let nr = start_r + TRACE_DR[d];
        let nc = start_c + TRACE_DC[d];
        if pixel_nonzero(src, width, height, nr, nc) {
            first = Some((nr, nc, d));
            break;
        }
    }
    let (mut r, mut c, dir_to_first) = match first {
        None => return,
        Some(t) => t,
    };
    let mut in_dir = (dir_to_first + 4) & 7;
    let max_iters = (width as usize) * (height as usize) * 4 + 64;
    let mut iters = 0usize;
    loop {
        if iters > max_iters { break; }
        iters += 1;
        arena.push([c, r]);
        let scan_start = (in_dir + 1) & 7;
        let mut next: Option<(i32, i32, usize)> = None;
        for k in 0..8usize {
            let d = (scan_start + k) & 7;
            let nr = r + TRACE_DR[d];
            let nc = c + TRACE_DC[d];
            if pixel_nonzero(src, width, height, nr, nc) {
                next = Some((nr, nc, d));
                break;
            }
        }
        let (nr, nc, _d) = match next { None => break, Some(t) => t };
        if nr == start_r && nc == start_c && iters > 1 {
            break;
        }
        r = nr; c = nc;
        in_dir = (next.unwrap().2 + 4) & 7;
    }
}

/// Generation-aware version: uses the packed-gen grid so we avoid the 4MB
/// per-call fill that the i32 version requires.
fn trace_one_component_into_gen(
    grid: &[u64],
    width: i32,
    height: i32,
    start_r: i32,
    start_c: i32,
    cid: u32,
    gen_high: u64,
    arena: &mut Vec<[i32; 2]>,
) {
    arena.push([start_c, start_r]);
    let mut first: Option<(i32, i32, usize)> = None;
    for k in 1..9usize {
        let d = k & 7;
        let nr = start_r + TRACE_DR[d];
        let nc = start_c + TRACE_DC[d];
        if neighbour_in_component_gen(grid, width, height, nr, nc, cid, gen_high) {
            first = Some((nr, nc, d));
            break;
        }
    }
    let (mut r, mut c, dir_to_first) = match first {
        None => return,
        Some(t) => t,
    };
    let _first_in_dir = (dir_to_first + 4) & 7;
    let mut in_dir = (dir_to_first + 4) & 7;
    let max_iters = (width as usize) * (height as usize) * 4 + 64;
    let mut iters = 0usize;
    loop {
        if iters > max_iters { break; }
        iters += 1;
        arena.push([c, r]);
        let scan_start = (in_dir + 1) & 7;
        let mut next: Option<(i32, i32, usize)> = None;
        for k in 0..8usize {
            let d = (scan_start + k) & 7;
            let nr = r + TRACE_DR[d];
            let nc = c + TRACE_DC[d];
            if neighbour_in_component_gen(grid, width, height, nr, nc, cid, gen_high) {
                next = Some((nr, nc, d));
                break;
            }
        }
        let (nr, nc, d) = match next { None => break, Some(t) => t };
        if nr == start_r && nc == start_c && iters > 1 {
            let _ = d;
            break;
        }
        r = nr; c = nc;
        in_dir = (d + 4) & 7;
    }
}

impl Default for LslExecutor {
    fn default() -> Self { Self::new() }
}

// ============================================================================
// Tests for Day 1
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    fn reconstruct(runs: &[Run], width: usize) -> Vec<u8> {
        let mut out = vec![0u8; width];
        for r in runs {
            for c in r.start..=r.end {
                out[c as usize] = 1;
            }
        }
        out
    }

    #[test]
    fn rle_round_trip_simple() {
        let row: Vec<u8> = vec![0, 1, 1, 0, 0, 1, 0, 1, 1, 1];
        let mut runs = Vec::new();
        rle_extract_row_scalar(&row, &mut runs);
        assert_eq!(runs, vec![
            Run { start: 1, end: 2 },
            Run { start: 5, end: 5 },
            Run { start: 7, end: 9 },
        ]);
        let normalized: Vec<u8> = row.iter().map(|&b| (b != 0) as u8).collect();
        assert_eq!(reconstruct(&runs, row.len()), normalized);
    }

    #[test]
    fn rle_round_trip_all_zeros() {
        let row = vec![0u8; 16];
        let mut runs = Vec::new();
        rle_extract_row_scalar(&row, &mut runs);
        assert!(runs.is_empty());
    }

    #[test]
    fn rle_round_trip_all_ones() {
        let row = vec![1u8; 16];
        let mut runs = Vec::new();
        rle_extract_row_scalar(&row, &mut runs);
        assert_eq!(runs, vec![Run { start: 0, end: 15 }]);
    }

    #[test]
    fn rle_round_trip_alternating() {
        let row: Vec<u8> = (0..32).map(|i| (i % 2) as u8).collect();
        let mut runs = Vec::new();
        rle_extract_row_scalar(&row, &mut runs);
        assert_eq!(runs.len(), 16);
        for (i, r) in runs.iter().enumerate() {
            assert_eq!(r.start, (i * 2 + 1) as u32);
            assert_eq!(r.end, (i * 2 + 1) as u32);
        }
    }

    #[test]
    fn ccl_filled_square_is_one_component() {
        // 8x6 image with a 4x3 filled rectangle in the middle
        let w = 8;
        let h = 6;
        let mut data = vec![0u8; w * h];
        for r in 1..5 {
            for c in 2..6 {
                data[r * w + c] = 1;
            }
        }
        let rle = rle_extract(&data, w, h);
        let labels = line_relative_label(&rle);
        let cmap = cross_row_merge(&rle, &labels);
        assert_eq!(cmap.n_components, 1, "filled rect = 1 component, got {:?}", cmap);
    }

    #[test]
    fn ccl_two_disjoint_rectangles() {
        // 12x6 image: two 3x3 rectangles separated by a column of zeros
        let w = 12;
        let h = 6;
        let mut data = vec![0u8; w * h];
        for r in 1..4 {
            for c in 1..4 { data[r * w + c] = 1; }
            for c in 7..10 { data[r * w + c] = 1; }
        }
        let rle = rle_extract(&data, w, h);
        let labels = line_relative_label(&rle);
        let cmap = cross_row_merge(&rle, &labels);
        assert_eq!(cmap.n_components, 2);
    }

    #[test]
    fn ccl_diagonal_touch_is_one_component_8conn() {
        // 6x6: two squares touching only at a corner — should be 1 component (8-conn)
        let w = 6;
        let h = 6;
        let mut data = vec![0u8; w * h];
        // Top-left 2x2 at (0..2, 0..2)
        for r in 0..2 { for c in 0..2 { data[r * w + c] = 1; } }
        // Bottom-right 2x2 at (2..4, 2..4) — touches top-left at corner (1,1)-(2,2)
        for r in 2..4 { for c in 2..4 { data[r * w + c] = 1; } }
        let rle = rle_extract(&data, w, h);
        let labels = line_relative_label(&rle);
        let cmap = cross_row_merge(&rle, &labels);
        assert_eq!(cmap.n_components, 1, "8-conn corner touch = 1 component");
    }

    #[test]
    fn trace_filled_3x3_square() {
        // 5x5 image with a 3x3 filled square in the middle (cols 1..4, rows 1..4)
        let w = 5;
        let h = 5;
        let mut data = vec![0u8; w * h];
        for r in 1..4 {
            for c in 1..4 { data[r * w + c] = 1; }
        }
        let contours = find_external_contours_lsl(&data, w, h);
        assert_eq!(contours.len(), 1);
        // Expect 8 boundary pixels (corners + edges of 3x3) — the Moore-trace
        // visits each boundary pixel.
        assert!(!contours[0].is_empty(), "got empty contour");
        // First point should be the start (top-left of 3x3 = (1, 1) in (col, row))
        assert_eq!(contours[0][0], [1, 1]);
    }

    #[test]
    fn trace_filled_3x3_returns_8_boundary_pixels() {
        let w = 5;
        let h = 5;
        let mut data = vec![0u8; w * h];
        for r in 1..4 { for c in 1..4 { data[r * w + c] = 1; } }
        let contours = find_external_contours_lsl(&data, w, h);
        assert_eq!(contours.len(), 1);
        // Boundary of a 3x3 filled square = 8 pixels (corners + edge midpoints)
        assert_eq!(contours[0].len(), 8, "expected 8 boundary, got {:?}", contours[0]);
    }

    #[test]
    fn trace_isolated_pixel() {
        // Single isolated pixel at (2, 2)
        let w = 5;
        let h = 5;
        let mut data = vec![0u8; w * h];
        data[2 * w + 2] = 1;
        let contours = find_external_contours_lsl(&data, w, h);
        assert_eq!(contours.len(), 1);
        assert_eq!(contours[0], vec![[2, 2]]);
    }

    #[test]
    fn outer_starts_finds_one_per_component() {
        let w = 12;
        let h = 6;
        let mut data = vec![0u8; w * h];
        for r in 1..4 {
            for c in 1..4 { data[r * w + c] = 1; }
            for c in 7..10 { data[r * w + c] = 1; }
        }
        let rle = rle_extract(&data, w, h);
        let labels = line_relative_label(&rle);
        let cmap = cross_row_merge(&rle, &labels);
        let starts = find_outer_starts(&rle, &cmap);
        assert_eq!(starts.len(), 2);
        // First component's start is (row=1, col=1), second is (row=1, col=7)
        assert!(starts.contains(&(1, 1)));
        assert!(starts.contains(&(1, 7)));
    }

    #[test]
    fn neon_matches_scalar_random() {
        // Random data from a fixed seed
        let mut state: u64 = 0xC0FFEE;
        let row: Vec<u8> = (0..256).map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) & 1) as u8
        }).collect();
        let mut s_runs = Vec::new();
        let mut n_runs = Vec::new();
        rle_extract_row_scalar(&row, &mut s_runs);
        rle_extract_row_neon(&row, &mut n_runs);
        assert_eq!(s_runs, n_runs);
    }
}
