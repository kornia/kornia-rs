//! Link-runs contour finder — Rust port of OpenCV's `LinkRunner`
//! (`modules/imgproc/src/contours_link.cpp`).
//!
//! ## Why a separate module
//!
//! Our LSL path does RLE → label (union-find) → grid build → Moore-neighbour
//! re-trace. That's three full passes over the data after RLE. Link-runs
//! collapses everything after RLE into a single forward sweep: each run is
//! converted to two endpoint nodes (start, end), and adjacent rows are stitched
//! together via a two-pointer walk that sets `link` pointers directly. The
//! contour structure IS the linked list — to read out a contour, walk `link`
//! from an external-contour start index until you return to it.
//!
//! Strictly **O(total runs)** vs LSL's O(perimeter pixels × components),
//! which is why OpenCV's link-runs path crushes our LSL on `sparse_noise`
//! at 2048² (~3.5k components, each touched 4-8 times by the Moore trace).
//!
//! ## Status
//!
//! External contours only (no holes / hierarchy). NONE-mode emission only —
//! SIMPLE-mode collapse can be added by reading consecutive `pt`s during
//! `convert_links` and skipping points where the (dx, dy) direction is
//! unchanged.

use core::ops::Range;
use kornia_image::{allocator::ImageAllocator, Image};

// LRPs are stored as four parallel Vecs (Structure-of-Arrays) instead of
// `Vec<Lrp>`. Rationale:
//
// * `establish_links` reads `x` (i16) and `next` (i32) most heavily, writes
//   `link` (i32) sometimes. With AoS, every touch loads the whole 12-byte
//   LRP. With SoA the hot streams are `xs` (2 B/entry) and `nexts`
//   (4 B/entry); `links` is mostly write-only.
// * `convert_links` walks chains via `link` (random access), reads
//   `x`, `y` once per node.
//
// On `sparse_noise 1024²` (524k LRPs ≈ 6 MB AoS, exceeds Jetson Orin's 4 MB
// L2), splitting into per-field Vecs keeps the per-phase working set
// inside L2.

#[derive(Clone, Copy, PartialEq, Eq)]
enum ConnectFlag {
    Single,
    ConnectingAbove,
    ConnectingBelow,
}

/// Reusable executor for the link-runs path.
///
/// Output is stored in a shared point arena + per-contour range table to
/// avoid one `Vec` allocation per contour. Use [`Self::iter_contours`] to
/// walk the contours as zero-copy slices, or [`Self::contours_owned`] to
/// materialize the legacy `Vec<Vec<[i32; 2]>>` shape.
#[derive(Default)]
pub struct LinkRunsExecutor {
    /// Per-LRP `link` field: index of next LRP in contour traversal
    /// (-1 sentinel = unset).
    links: Vec<i32>,
    /// Per-LRP `next` field: index of next LRP in this row's run-pair list.
    nexts: Vec<i32>,
    /// Per-LRP x-coordinate.
    xs: Vec<i16>,
    /// Per-LRP y-coordinate.
    ys: Vec<i16>,
    ext_rns: Vec<i32>,
    /// Flat point arena — every contour's points concatenated.
    arena: Vec<[i32; 2]>,
    /// One range per contour, slicing into `arena`.
    ranges: Vec<Range<usize>>,
}

impl LinkRunsExecutor {
    /// Create a new executor.
    pub fn new() -> Self {
        Self::default()
    }

    // ------------------------------------------------------------------
    // SoA accessors — single-line `#[inline]` wrappers so the algorithm
    // code reads close to the AoS original. The compiler should fold these
    // into direct loads/stores.
    // ------------------------------------------------------------------
    #[inline(always)]
    fn push_lrp(&mut self, x: i32, y: i32) -> i32 {
        debug_assert!(x >= 0 && x <= i16::MAX as i32, "x={x}");
        debug_assert!(y >= 0 && y <= i16::MAX as i32, "y={y}");
        let idx = self.links.len() as i32;
        self.links.push(-1);
        self.nexts.push(-1);
        self.xs.push(x as i16);
        self.ys.push(y as i16);
        idx
    }
    #[inline(always)]
    fn push_empty(&mut self) -> i32 {
        let idx = self.links.len() as i32;
        self.links.push(-1);
        self.nexts.push(-1);
        self.xs.push(0);
        self.ys.push(0);
        idx
    }
    #[inline(always)]
    fn x(&self, i: i32) -> i32 { self.xs[i as usize] as i32 }
    #[inline(always)]
    fn y(&self, i: i32) -> i32 { self.ys[i as usize] as i32 }
    #[inline(always)]
    fn nxt(&self, i: i32) -> i32 { self.nexts[i as usize] }
    #[inline(always)]
    fn lnk(&self, i: i32) -> i32 { self.links[i as usize] }
    #[inline(always)]
    fn set_nxt(&mut self, i: i32, v: i32) { self.nexts[i as usize] = v; }
    #[inline(always)]
    fn set_lnk(&mut self, i: i32, v: i32) { self.links[i as usize] = v; }
    #[inline(always)]
    fn lrp_count(&self) -> i32 { self.links.len() as i32 }

    /// Run on an image. Returns the per-contour range table; pair with
    /// [`Self::arena`] for the actual points. Valid until next call.
    pub fn find_external_contours_image<'a, A: ImageAllocator>(
        &'a mut self,
        src: &Image<u8, 1, A>,
    ) -> &'a [Range<usize>] {
        let size = src.size();
        self.find_external_contours(src.as_slice(), size.width, size.height)
    }

    /// Run on a raw binary slice. Returns the per-contour range table.
    pub fn find_external_contours<'a>(
        &'a mut self,
        src: &[u8],
        width: usize,
        height: usize,
    ) -> &'a [Range<usize>] {
        self.process(src, width as i32, height as i32);
        self.convert_links();
        &self.ranges
    }

    /// Borrow the flat point arena. Index via the ranges from
    /// [`Self::find_external_contours`] or the slices from
    /// [`Self::iter_contours`].
    pub fn arena(&self) -> &[[i32; 2]] {
        &self.arena
    }

    /// Iterate contours as zero-copy slices into the arena.
    pub fn iter_contours(&self) -> impl Iterator<Item = &[[i32; 2]]> + '_ {
        let arena = &self.arena;
        self.ranges.iter().map(move |r| &arena[r.clone()])
    }

    /// Materialize the legacy `Vec<Vec<[i32; 2]>>` shape — one `Vec` per
    /// contour, copied out of the arena. Use [`Self::iter_contours`] when
    /// you can avoid the allocation.
    pub fn contours_owned(&self) -> Vec<Vec<[i32; 2]>> {
        self.ranges
            .iter()
            .map(|r| self.arena[r.clone()].to_vec())
            .collect()
    }

    /// Number of contours from the last call.
    pub fn contour_count(&self) -> usize {
        self.ranges.len()
    }

    /// Build the linked-run graph for the entire image.
    ///
    /// **Why no pre-sizing of `rns` / `arena`**: profiling (commit history
    /// retains `LINKRUNS_PROF` flag) showed that on the warm path
    /// (`LinkRunsExecutor::clear()` retains capacity) allocation cost is
    /// already zero — the bottleneck is pointer-chasing through the LRP
    /// graph in `convert_links` + `establish_links`, not allocation. On the
    /// cold path, an A/B test of `width*height/4` pre-reserve made
    /// `sparse_noise 256²` 86% slower (heuristic under-reserves the
    /// noise case AND eats a wasted alloc + page-fault-on-touch). The real
    /// lever for sparse-noise is cache-friendlier chain layout
    /// (OpenCV's BlockStorage pattern), tracked separately.
    #[allow(clippy::needless_range_loop)]
    fn process(&mut self, src: &[u8], width: i32, height: i32) {
        // SoA: clear all four parallel Vecs.
        self.links.clear();
        self.nexts.clear();
        self.xs.clear();
        self.ys.clear();
        self.ext_rns.clear();
        self.arena.clear();
        self.ranges.clear();

        // Sentinel head (matches OpenCV's `rns.push_back(LRP())`).
        self.push_empty();
        let mut upper_line = self.lrp_count() - 1;
        let mut cur = upper_line;

        // First row
        let row0 = &src[..width as usize];
        let mut j = 0;
        while j < width {
            let s = find_start(row0, j);
            if s == width { break; }
            let e_excl = find_end(row0, s + 1);
            let start_idx = self.push_lrp(s, 0);
            self.set_nxt(cur, start_idx);
            cur = start_idx;
            let end_idx = self.push_lrp(e_excl - 1, 0);
            self.set_nxt(cur, end_idx);
            self.set_lnk(cur, end_idx);
            self.ext_rns.push(cur);
            cur = end_idx;
            j = e_excl;
        }
        upper_line = self.nxt(upper_line);
        let mut upper_total = self.lrp_count() - 1;

        let mut last_elem = cur;
        self.set_nxt(cur, -1);
        let mut prev_point = -1i32;

        for i in 1..height {
            let row = &src[(i as usize) * (width as usize)..((i + 1) as usize) * (width as usize)];
            let all_total = self.lrp_count();
            let mut j = 0;
            while j < width {
                let s = find_start(row, j);
                if s == width { break; }
                let e_excl = find_end(row, s + 1);
                let start_idx = self.push_lrp(s, i);
                self.set_nxt(cur, start_idx);
                cur = start_idx;
                let end_idx = self.push_lrp(e_excl - 1, i);
                self.set_nxt(cur, end_idx);
                cur = end_idx;
                j = e_excl;
            }
            let lower_line = self.nxt(last_elem);
            let lower_total = self.lrp_count() - all_total;
            last_elem = cur;
            self.set_nxt(cur, -1);

            self.establish_links(
                &mut prev_point,
                upper_line,
                lower_line,
                upper_total,
                lower_total,
            );

            upper_line = lower_line;
            upper_total = lower_total;
        }

        // Last line: close upper run-pairs.
        let mut upper_run = upper_line;
        for _ in 0..(upper_total / 2) {
            let n = self.nxt(upper_run);
            self.set_lnk(n, upper_run);
            upper_run = self.nxt(n);
        }
    }

    /// Two-pointer walk of upper and lower run-pairs (one pair = start LRP +
    /// end LRP via `next`). Sets `link` pointers between overlapping pairs.
    /// Faithful port of `LinkRunner::establishLinks` (contours_link.cpp:146).
    fn establish_links(
        &mut self,
        prev_point: &mut i32,
        mut upper_run: i32,
        mut lower_run: i32,
        upper_total: i32,
        lower_total: i32,
    ) {
        let mut k: i32 = 0;
        let mut n: i32 = 0;
        let mut flag = ConnectFlag::Single;
        while k < upper_total / 2 && n < lower_total / 2 {
            match flag {
                ConnectFlag::Single => {
                    let upper_next = self.nxt(upper_run);
                    let lower_next = self.nxt(lower_run);
                    let upper_next_x = self.x(upper_next);
                    let lower_next_x = self.x(lower_next);
                    if upper_next_x < lower_next_x {
                        if upper_next_x >= self.x(lower_run) - 1 {
                            self.set_lnk(lower_run, upper_run);
                            flag = ConnectFlag::ConnectingAbove;
                            *prev_point = upper_next;
                        } else {
                            self.set_lnk(upper_next, upper_run);
                        }
                        k += 1;
                        upper_run = self.nxt(upper_next);
                    } else {
                        if self.x(upper_run) <= lower_next_x + 1 {
                            self.set_lnk(lower_run, upper_run);
                            flag = ConnectFlag::ConnectingBelow;
                            *prev_point = lower_next;
                        } else {
                            self.set_lnk(lower_run, lower_next);
                            self.ext_rns.push(lower_run);
                        }
                        n += 1;
                        lower_run = self.nxt(lower_next);
                    }
                }
                ConnectFlag::ConnectingAbove => {
                    let lower_next = self.nxt(lower_run);
                    if self.x(upper_run) > self.x(lower_next) + 1 {
                        self.set_lnk(*prev_point, lower_next);
                        flag = ConnectFlag::Single;
                        n += 1;
                        lower_run = self.nxt(lower_next);
                    } else {
                        self.set_lnk(*prev_point, upper_run);
                        let upper_next = self.nxt(upper_run);
                        if self.x(upper_next) < self.x(lower_next) {
                            k += 1;
                            *prev_point = upper_next;
                            upper_run = self.nxt(upper_next);
                        } else {
                            flag = ConnectFlag::ConnectingBelow;
                            *prev_point = lower_next;
                            n += 1;
                            lower_run = self.nxt(lower_next);
                        }
                    }
                }
                ConnectFlag::ConnectingBelow => {
                    let upper_next = self.nxt(upper_run);
                    if self.x(lower_run) > self.x(upper_next) + 1 {
                        self.set_lnk(upper_next, *prev_point);
                        flag = ConnectFlag::Single;
                        k += 1;
                        upper_run = self.nxt(upper_next);
                    } else {
                        // OpenCV's int_rns push (hole start). Single ext_rns
                        // collection — matches cv2.findContoursLinkRuns shape.
                        self.ext_rns.push(lower_run);
                        self.set_lnk(lower_run, *prev_point);
                        let lower_next = self.nxt(lower_run);
                        if self.x(lower_next) < self.x(upper_next) {
                            n += 1;
                            *prev_point = lower_next;
                            lower_run = self.nxt(lower_next);
                        } else {
                            flag = ConnectFlag::ConnectingAbove;
                            k += 1;
                            *prev_point = upper_next;
                            upper_run = self.nxt(upper_next);
                        }
                    }
                }
            }
        }

        // Drain remaining lower runs as new external contours
        while n < lower_total / 2 {
            if flag != ConnectFlag::Single {
                let lower_next = self.nxt(lower_run);
                self.set_lnk(*prev_point, lower_next);
                flag = ConnectFlag::Single;
                lower_run = self.nxt(lower_next);
                n += 1;
                continue;
            }
            let lower_next = self.nxt(lower_run);
            self.set_lnk(lower_run, lower_next);
            self.ext_rns.push(lower_run);
            lower_run = self.nxt(lower_next);
            n += 1;
        }

        // Drain remaining upper runs (close them to themselves)
        while k < upper_total / 2 {
            if flag != ConnectFlag::Single {
                let upper_next = self.nxt(upper_run);
                self.set_lnk(upper_next, *prev_point);
                flag = ConnectFlag::Single;
                upper_run = self.nxt(upper_next);
                k += 1;
                continue;
            }
            let upper_next = self.nxt(upper_run);
            self.set_lnk(upper_next, upper_run);
            upper_run = self.nxt(upper_next);
            k += 1;
        }
    }

    /// Walk each external-contour chain via `link` and emit points into the
    /// shared arena, recording one Range per contour. Single shared
    /// allocation instead of `ext_rns.len()` per-contour `Vec`s — the
    /// difference is decisive on sparse-noise images with thousands of
    /// tiny contours.
    fn convert_links(&mut self) {
        self.arena.clear();
        self.ranges.clear();
        let n = self.ext_rns.len();
        self.ranges.reserve(n);
        // Locally bind the field-Vecs so the inner loop doesn't reborrow
        // self each iteration (helps the compiler eliminate bounds checks
        // on the parallel slices).
        let xs = &self.xs[..];
        let ys = &self.ys[..];
        let links = &mut self.links[..];
        for i in 0..n {
            let start = self.ext_rns[i];
            let mut cur = start;
            if links[cur as usize] == -1 {
                continue;
            }
            let arena_start = self.arena.len();
            loop {
                self.arena.push([xs[cur as usize] as i32, ys[cur as usize] as i32]);
                let next = links[cur as usize];
                links[cur as usize] = -1;
                cur = next;
                if cur == start || cur == -1 {
                    break;
                }
            }
            self.ranges.push(arena_start..self.arena.len());
        }
    }
}

/// Convenience one-shot wrapper. Materializes the legacy `Vec<Vec<...>>`
/// shape — for hot loops, prefer
/// [`LinkRunsExecutor::find_external_contours`] + iter_contours().
pub fn find_external_contours_linkruns(
    src: &[u8],
    width: usize,
    height: usize,
) -> Vec<Vec<[i32; 2]>> {
    let mut exec = LinkRunsExecutor::new();
    exec.find_external_contours(src, width, height);
    exec.contours_owned()
}

// ----------------------------------------------------------------------------
// Row-scan helpers — equivalent to OpenCV's findStartContourPoint /
// findEndContourPoint. NEON path scans 16 bytes per iter using
// `vceqq_u8` + bit-extract; falls back to scalar for the row tail.
// ----------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
#[inline]
fn find_start(row: &[u8], mut j: i32) -> i32 {
    use core::arch::aarch64::*;
    let w = row.len() as i32;
    // SAFETY: NEON intrinsics, baseline on aarch64 Linux; bounds are checked
    // before each load.
    unsafe {
        while j + 16 <= w {
            let v = vld1q_u8(row.as_ptr().add(j as usize));
            // Compare each byte to zero -> 0xFF where byte == 0
            let zmask = vceqq_u8(v, vdupq_n_u8(0));
            // Reduce: if any lane is non-zero (a non-zero byte exists), there
            // is a run start somewhere in this 16-byte window. Use minv: if
            // min(zmask) == 0xFF, every byte was zero (skip). Otherwise some
            // byte is non-zero.
            if vminvq_u8(zmask) == 0 {
                // At least one non-zero byte present in this window
                // Compute "is non-zero" mask = ~zmask, then find first set bit.
                let nz = vmvnq_u8(zmask);
                // Pack 16 lanes (each 0x00 or 0xFF) into a 64-bit half-vector
                // by narrowing to 4-bit-per-lane via `vshrn_n_u16` on a
                // reinterpreted u16 view. Cheaper trick: store, scan.
                let mut buf = [0u8; 16];
                vst1q_u8(buf.as_mut_ptr(), nz);
                for (k, b) in buf.iter().enumerate() {
                    if *b != 0 {
                        return j + k as i32;
                    }
                }
            }
            j += 16;
        }
    }
    while j < w && row[j as usize] == 0 {
        j += 1;
    }
    j
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn find_end(row: &[u8], mut j: i32) -> i32 {
    use core::arch::aarch64::*;
    let w = row.len() as i32;
    unsafe {
        while j + 16 <= w {
            let v = vld1q_u8(row.as_ptr().add(j as usize));
            let zmask = vceqq_u8(v, vdupq_n_u8(0));
            // If any byte is zero, the run ends within this window
            if vmaxvq_u8(zmask) != 0 {
                let mut buf = [0u8; 16];
                vst1q_u8(buf.as_mut_ptr(), zmask);
                for (k, b) in buf.iter().enumerate() {
                    if *b != 0 {
                        return j + k as i32;
                    }
                }
            }
            j += 16;
        }
    }
    while j < w && row[j as usize] != 0 {
        j += 1;
    }
    j
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn find_start(row: &[u8], mut j: i32) -> i32 {
    let w = row.len() as i32;
    while j < w && row[j as usize] == 0 {
        j += 1;
    }
    j
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn find_end(row: &[u8], mut j: i32) -> i32 {
    let w = row.len() as i32;
    while j < w && row[j as usize] != 0 {
        j += 1;
    }
    j
}

// ----------------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Two-level nesting: white outer, dark middle ring, white shape enclosed inside.
    /// Link-runs reports BOTH the outer-white outline AND the inner-white outline
    /// AND the dark hole between them. Expected: 3 contours (matches
    /// cv2.findContoursLinkRuns).
    #[test]
    fn enclosed_white_inside_dark_inside_white() {
        let w = 16;
        let h = 16;
        // Default white
        let mut data = vec![1u8; w * h];
        // Outer dark ring at rows 4..12, cols 3..13
        for r in 4..12 { for c in 3..13 { data[r * w + c] = 0; } }
        // Inner white block at rows 6..10, cols 5..11 (fully enclosed by dark)
        for r in 6..10 { for c in 5..11 { data[r * w + c] = 1; } }
        let contours = find_external_contours_linkruns(&data, w, h);
        assert_eq!(
            contours.len(), 3,
            "outer white + enclosed white = 2 components, got {} ({:?})",
            contours.len(),
            contours.iter().enumerate().map(|(i, c)| {
                let xs: Vec<i32> = c.iter().map(|p| p[0]).collect();
                let ys: Vec<i32> = c.iter().map(|p| p[1]).collect();
                let xb = (xs.iter().min().copied().unwrap_or(0), xs.iter().max().copied().unwrap_or(0));
                let yb = (ys.iter().min().copied().unwrap_or(0), ys.iter().max().copied().unwrap_or(0));
                format!("c{i}: n={} xb={:?} yb={:?}", c.len(), xb, yb)
            }).collect::<Vec<_>>(),
        );
    }

    /// 6×6 checkerboard. Link-runs counts both the foreground union (1 outer)
    /// and every interior 1×1 hole. Expected count matches
    /// cv2.findContoursLinkRuns on the same fixture.
    #[test]
    fn checkerboard_matches_opencv_count() {
        let w = 6;
        let h = 6;
        let mut data = vec![0u8; w * h];
        for r in 0..h {
            for c in 0..w {
                if (r + c) % 2 == 0 { data[r * w + c] = 1; }
            }
        }
        // cv2.findContoursLinkRuns on this exact fixture returns 9 (verified
        // separately). Hard-coded so the test catches future regressions.
        let contours = find_external_contours_linkruns(&data, w, h);
        assert_eq!(
            contours.len(), 9,
            "checkerboard expected 9 contours (cv2.findContoursLinkRuns), got {} ({:?})",
            contours.len(),
            contours.iter().enumerate().map(|(i, c)| {
                format!("c{i}: n={} pts={:?}", c.len(), c)
            }).collect::<Vec<_>>(),
        );
    }

    /// White background with TWO vertically-stacked holes (same x range,
    /// separated by 5 white rows). cv2.findContoursLinkRuns = 3.
    #[test]
    fn background_with_two_stacked_holes_matches_opencv() {
        let w = 30;
        let h = 30;
        let mut data = vec![1u8; w * h];
        // Hole 1: rows 5..12, x 10..20
        for r in 5..12 { for c in 10..20 { data[r * w + c] = 0; } }
        // Hole 2: rows 18..25, x 10..20
        for r in 18..25 { for c in 10..20 { data[r * w + c] = 0; } }
        let contours = find_external_contours_linkruns(&data, w, h);
        assert_eq!(
            contours.len(), 3,
            "background + 2 stacked holes = 3 contours (cv2 ground truth), got {} ({:?})",
            contours.len(),
            contours.iter().enumerate().map(|(i, c)| {
                let xs: Vec<i32> = c.iter().map(|p| p[0]).collect();
                let ys: Vec<i32> = c.iter().map(|p| p[1]).collect();
                let xb = (xs.iter().min().copied().unwrap_or(0), xs.iter().max().copied().unwrap_or(0));
                let yb = (ys.iter().min().copied().unwrap_or(0), ys.iter().max().copied().unwrap_or(0));
                format!("c{i}: n={} xb={:?} yb={:?}", c.len(), xb, yb)
            }).collect::<Vec<_>>(),
        );
    }

    /// White background with TWO holes side-by-side. cv2 = 3.
    #[test]
    fn background_with_two_side_by_side_holes_matches_opencv() {
        let w = 40;
        let h = 30;
        let mut data = vec![1u8; w * h];
        // Hole 1: (10..23, 5..15)
        for r in 10..23 { for c in 5..15 { data[r * w + c] = 0; } }
        // Hole 2: (10..23, 20..30)
        for r in 10..23 { for c in 20..30 { data[r * w + c] = 0; } }
        let contours = find_external_contours_linkruns(&data, w, h);
        assert_eq!(
            contours.len(), 3,
            "background + 2 side-by-side holes = 3 (cv2 ground truth), got {} ({:?})",
            contours.len(),
            contours.iter().enumerate().map(|(i, c)| {
                let xs: Vec<i32> = c.iter().map(|p| p[0]).collect();
                let ys: Vec<i32> = c.iter().map(|p| p[1]).collect();
                let xb = (xs.iter().min().copied().unwrap_or(0), xs.iter().max().copied().unwrap_or(0));
                let yb = (ys.iter().min().copied().unwrap_or(0), ys.iter().max().copied().unwrap_or(0));
                format!("c{i}: n={} xb={:?} yb={:?}", c.len(), xb, yb)
            }).collect::<Vec<_>>(),
        );
    }

    /// White background with single 13×13 black hole. cv2 = 2 (outer + hole).
    #[test]
    fn background_with_one_central_hole_matches_opencv() {
        let w = 30;
        let h = 30;
        // All-white background
        let mut data = vec![1u8; w * h];
        // Black hole at (10..23, 10..23)
        for r in 10..23 {
            for c in 10..23 {
                data[r * w + c] = 0;
            }
        }
        let contours = find_external_contours_linkruns(&data, w, h);
        assert_eq!(
            contours.len(), 2,
            "background + 1 hole = 2 contours (cv2 ground truth), got {} ({:?})",
            contours.len(),
            contours.iter().enumerate().map(|(i, c)| {
                let xs: Vec<i32> = c.iter().map(|p| p[0]).collect();
                let ys: Vec<i32> = c.iter().map(|p| p[1]).collect();
                let xb = (xs.iter().min().copied().unwrap_or(0), xs.iter().max().copied().unwrap_or(0));
                let yb = (ys.iter().min().copied().unwrap_or(0), ys.iter().max().copied().unwrap_or(0));
                format!("c{i}: n={} xb={:?} yb={:?}", c.len(), xb, yb)
            }).collect::<Vec<_>>(),
        );
    }

    /// One foreground component with a 1-row "hole-shaped" gap that splits
    /// the row into 2 runs. cv2 = 2 (outer + 1 hole).
    #[test]
    fn split_then_merge_matches_opencv() {
        let w = 10;
        let h = 3;
        // Row 0: ##########  (1 run)
        // Row 1: ####..####  (2 runs, gap in middle = hole start)
        // Row 2: ##########  (1 run, hole closed)
        let mut data = vec![0u8; w * h];
        for c in 0..w { data[c] = 1; }
        for c in 0..4 { data[w + c] = 1; }
        for c in 6..w { data[w + c] = 1; }
        for c in 0..w { data[2 * w + c] = 1; }
        let contours = find_external_contours_linkruns(&data, w, h);
        assert_eq!(
            contours.len(), 2,
            "split-then-merge = 2 contours (cv2 ground truth), got {} ({:?})",
            contours.len(), contours
        );
    }

    /// Three top-row runs that merge through a wide bottom row. RETR_EXTERNAL
    /// should be 1 component; if linkruns returns 3, the cross-row link
    /// machinery is failing to merge.
    #[test]
    fn three_top_runs_merge_via_bottom_row() {
        let w = 15;
        let h = 3;
        // Row 0 + 1: ###...###...###  (3 disjoint columns, top + middle row)
        // Row 2:    ###############   (1 long run merging everything)
        let mut data = vec![0u8; w * h];
        for r in 0..2 {
            for c in 0..3 { data[r * w + c] = 1; }
            for c in 6..9 { data[r * w + c] = 1; }
            for c in 12..15 { data[r * w + c] = 1; }
        }
        for c in 0..15 { data[2 * w + c] = 1; }
        let contours = find_external_contours_linkruns(&data, w, h);
        assert_eq!(
            contours.len(), 1,
            "expected 1 merged contour, got {}: {:?}",
            contours.len(), contours
        );
    }

    #[test]
    fn isolated_single_pixel() {
        let mut data = vec![0u8; 5 * 5];
        data[2 * 5 + 2] = 1;
        let contours = find_external_contours_linkruns(&data, 5, 5);
        assert_eq!(contours.len(), 1);
        // Single-pixel run produces 2 LRPs both at (2, 2)
        assert!(!contours[0].is_empty(), "should produce points");
    }

    #[test]
    fn two_disjoint_squares() {
        let w = 12;
        let h = 6;
        let mut data = vec![0u8; w * h];
        for r in 1..4 {
            for c in 1..4 { data[r * w + c] = 1; }
            for c in 7..10 { data[r * w + c] = 1; }
        }
        let contours = find_external_contours_linkruns(&data, w, h);
        assert_eq!(contours.len(), 2, "expected 2 components");
    }

    #[test]
    fn filled_3x3_emits_run_endpoints() {
        let w = 5;
        let h = 5;
        let mut data = vec![0u8; w * h];
        for r in 1..4 {
            for c in 1..4 { data[r * w + c] = 1; }
        }
        let contours = find_external_contours_linkruns(&data, w, h);
        assert_eq!(contours.len(), 1);
        // 3 rows × 2 endpoints (start, end) per run = 6 points
        assert_eq!(contours[0].len(), 6, "got {:?}", contours[0]);
    }
}
