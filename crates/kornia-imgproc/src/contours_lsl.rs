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

/// Assign sequential per-row labels (Day 2). STUB.
pub fn line_relative_label(_rle: &RleImage) -> LineLabels {
    todo!("Day 2: assign sequential per-row labels — see Lacassagne 2009 Section 3.2");
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

/// Merge equivalent runs across rows via union-find on overlapping intervals.
/// Day 3. STUB.
pub fn cross_row_merge(_rle: &RleImage, _labels: &LineLabels) -> ComponentMap {
    todo!("Day 3: union-find merge across overlapping runs in adjacent rows");
}

// ============================================================================
// Day 4: Boundary trace per component (Chang/Chen approach) — STUB
// ============================================================================

/// Trace boundary per labeled component. Outputs Vec<Vec<[i32;2]>> compatible
/// with the existing find_contours public API. Day 4. STUB.
pub fn trace_components(
    _rle: &RleImage,
    _components: &ComponentMap,
) -> Vec<Vec<[i32; 2]>> {
    todo!("Day 4: Chang/Chen contour-tracing per component");
}

// ============================================================================
// Day 5: Public API + bench validation — STUB
// ============================================================================

// pub fn find_contours_lsl(...) -> Result<Vec<Vec<[i32;2]>>, ContoursError>
// (Will live here once Days 2-4 are implemented and validated bit-exact
//  against cv2.findContours via examples/check_correctness.py.)

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
