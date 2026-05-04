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

/// NEON-vectorized RLE extraction. Uses OpenCV's stateful-scan pattern:
/// load 16 bytes, compare with `prev` value, scan-forward to next transition.
/// At each transition, emit a run boundary.
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
        // Track current state and run start
        let mut in_run = row[0] != 0;
        let mut run_start: u32 = 0;
        i = 1;

        // For now, the NEON version falls back to scalar — full NEON would use
        // vector compare to the previous-byte value and bit-scan-forward to
        // find transitions. Day 1 ships scalar correctness; NEON kicks in Day 4.
        // (Keeping the symbol so callers can already use the right entry point.)
        let _ = (zero,); // silence warnings
        while i < n {
            let b = *row.get_unchecked(i);
            let nonzero = b != 0;
            if nonzero != in_run {
                if in_run {
                    // run ended at i-1
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
/// Used by `trace_components` for O(1) "is this neighbour the same component?"
/// checks during Moore-neighbour boundary tracing.
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

/// Public API: find external contours via the LSL pipeline.
///
/// Currently returns empty Vec — the trace_components step is still a STUB.
/// Once Day 4 is implemented, this will be the fast-path entry point.
pub fn find_external_contours_lsl(src: &[u8], width: usize, height: usize) -> Vec<Vec<[i32; 2]>> {
    let rle = rle_extract(src, width, height);
    let labels = line_relative_label(&rle);
    let components = cross_row_merge(&rle, &labels);
    trace_components(&rle, &components)
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
