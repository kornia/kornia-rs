//! Contour finding implementation based on Suzuki and Abe (1985).

use kornia_image::allocator::ImageAllocator;
use kornia_image::Image;
use rayon::prelude::*;
use std::ops::Range;

/// Controls which contours are returned by find_contours
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetrievalMode {
    /// Return only the outermost contours.
    External,
    /// Return all contours without hierarchy.
    List,
    /// Return all contours with a two-level hierarchy (outer + holes).
    CComp,
    /// Return all contours with full hierarchy.
    Tree,
}

/// Controls how contour points are stored after tracing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContourApproximationMode {
    /// Store every border pixel — no compression.
    None,
    /// Store only the endpoints of horizontal, vertical, and diagonal segments,
    /// compressing straight runs into two points.
    Simple,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BorderType {
    Outer,
    Hole,
}

/// A single contour: an ordered list of [x, y] pixel coordinates
pub type Contour = Vec<[i32; 2]>;
/// `next, prev, first_child, parent` indices into the contours array, or -1
pub type HierarchyEntry = [i32; 4];

/// Error type returned by contour-finding functions
#[derive(Debug, thiserror::Error)]
pub enum ContoursError {
    /// Returned when the number of distinct borders exceeds [`i16::MAX`]
    #[error("find_contours: too many borders (nbd overflow)")]
    NbdOverflow,
}

/// Output of find_contours and FindContoursExecutor::find_contours.
pub struct ContoursResult {
    /// Detected contours
    pub contours: Vec<Contour>,
    /// Hierarchy entries, parallel to contours
    pub hierarchy: Vec<HierarchyEntry>,
}

// Directions: 0=W 1=NW 2=N 3=NE 4=E 5=SE 6=S 7=SW
const DIR_LUT: [usize; 9] = [1, 2, 3, 0, 0, 4, 7, 6, 5];
const DIR_DR: [i32; 8] = [0, -1, -1, -1, 0, 1, 1, 1];
const DIR_DC: [i32; 8] = [-1, -1, 0, 1, 1, 1, 0, -1];

/// Neighbour offsets pre-computed from the padded-image stride.
/// Kept in a single struct so trace_border stays within clippy's
/// `too_many_arguments` limit (7)
struct TracerOffsets {
    /// 8-element flat-offset table (one per direction, 0=W . .  7=SW)
    o8: [isize; 8],
    /// 16-element duplicate of `o8` for modulo-free linear scans
    o16: [isize; 16],
}

/// Per-border start position and metadata, passed together to trace_border
struct TracerStart {
    idx: usize,
    row: i32,
    col: i32,
    dir: usize,
    nbd: i16,
    method: ContourApproximationMode,
}

// Four i16 values equal to 1 packed into a u64
// NOTE: assumes little-endian byte order
const ALL_ONES_I16: u64 = 0x0001_0001_0001_0001;

/// Reusable executor for running find_contours on successive frames
/// reused across calls, avoiding repeated heap allocation for those buffers
/// after the first warm-up frame, per-call bookkeeping (ranges, hierarchy,
/// border_types) and the output contour vectors are still freshly allocated
/// on each call
/// For multi-stream workloads, use one executor per rayon thread
pub struct FindContoursExecutor {
    img: Vec<i16>,
    arena: Vec<[i32; 2]>,
}

impl FindContoursExecutor {
    /// Create a new executor with empty internal buffers.
    pub fn new() -> Self {
        Self {
            img: Vec::new(),
            arena: Vec::new(),
        }
    }

    /// Equivalent to find_contours but avoids repeated heap allocation
    pub fn find_contours<A: ImageAllocator>(
        &mut self,
        src: &Image<u8, 1, A>,
        mode: RetrievalMode,
        method: ContourApproximationMode,
    ) -> Result<ContoursResult, ContoursError> {
        self.arena.clear();
        find_contours_impl(src, mode, method, &mut self.img, &mut self.arena)
    }
}

impl Default for FindContoursExecutor {
    fn default() -> Self {
        Self::new()
    }
}

// Compile-time assertion: FindContoursExecutor is Send
const _: () = {
    fn assert_send<T: Send>() {}
    let _ = assert_send::<FindContoursExecutor>;
};

/// Convenience API - allocates fresh buffers each call
/// For repeated use on a series of images prefer FindContoursExecutor
pub fn find_contours<A: ImageAllocator>(
    src: &Image<u8, 1, A>,
    mode: RetrievalMode,
    method: ContourApproximationMode,
) -> Result<ContoursResult, ContoursError> {
    let mut img = Vec::new();
    let mut arena = Vec::new();
    find_contours_impl(src, mode, method, &mut img, &mut arena)
}

fn find_contours_impl<A: ImageAllocator>(
    src: &Image<u8, 1, A>,
    mode: RetrievalMode,
    method: ContourApproximationMode,
    img: &mut Vec<i16>,
    arena: &mut Vec<[i32; 2]>,
) -> Result<ContoursResult, ContoursError> {
    let height = src.height();
    let width = src.width();
    let padded_w = width + 2;
    let padded_h = height + 2;
    let padded_n = padded_h * padded_w;

    // Grow or reuse the working image buffer
    if img.len() < padded_n {
        img.resize(padded_n, 0);
    }
    let img_slice = &mut img[..padded_n];
    img_slice.fill(0i16);

    // Binarise: parallel for ≥ 512×512 (amortises rayon overhead), sequential below
    let src_data = src.as_slice();
    let interior = &mut img_slice[padded_w..padded_w + height * padded_w];
    if width * height >= 512 * 512 {
        interior
            .par_chunks_mut(padded_w)
            .enumerate()
            .for_each(|(r, dst_row)| {
                let src_row = &src_data[r * width..(r + 1) * width];
                for (d, &s) in dst_row[1..=width].iter_mut().zip(src_row.iter()) {
                    *d = (s != 0) as i16;
                }
            });
    } else {
        for (r, dst_row) in interior.chunks_mut(padded_w).enumerate() {
            let src_row = &src_data[r * width..(r + 1) * width];
            for (d, &s) in dst_row[1..=width].iter_mut().zip(src_row.iter()) {
                *d = (s != 0) as i16;
            }
        }
    }

    // 8-directional flat offsets in the padded image, plus a 16-element duplicate
    // for modulo-free neighbour scans in trace_border
    let pw = padded_w as isize;
    let o8: [isize; 8] = [-1, -pw - 1, -pw, -pw + 1, 1, pw + 1, pw, pw - 1];
    let mut o16 = [0isize; 16];
    for k in 0..16 {
        o16[k] = o8[k & 7];
    }
    let toff = TracerOffsets { o8, o16 };

    let mut ranges: Vec<Range<usize>> = Vec::new();
    let mut hierarchy: Vec<HierarchyEntry> = Vec::new();
    let mut border_types: Vec<BorderType> = Vec::new();

    let mut nbd: i16 = 1;
    hierarchy.push([-1, -1, -1, -1]);
    border_types.push(BorderType::Outer); // frame sentinel

    let img_ptr = img_slice.as_mut_ptr();

    // Main raster scan - sequential because lnbd carries state across columns
    for r in 1..=height {
        let mut lnbd: i16 = 1;
        let row_base = r * padded_w;
        let mut c = 1usize;

        'col: loop {
            if c > width {
                break 'col;
            }

            // SAFETY: row_base + c is always in the interior of img_slice:
            // c ranges over 1.. = width
            // row_base = r * padded_w with r in 1.. = height
            // padded dimensions are (height+2) * (width+2), so
            // row_base + c ∈ [padded_w, padded_w*height + width] ⊂ [0, padded_n)
            let pixel = unsafe { *img_ptr.add(row_base + c) };

            // batch advance over zero runs
            if pixel == 0 {
                c += 1;
                // SAFETY: c + 4 <= width -> row_base + c + 3 < row_base + padded_w - 1
                // (the padded row has two extra columns), so all 8 bytes of the
                // u64 load fall within the allocation.
                while c + 4 <= width {
                    let word =
                        unsafe { (img_ptr.add(row_base + c) as *const u64).read_unaligned() };
                    if word != 0 {
                        break;
                    }
                    c += 4;
                }
                while c <= width && unsafe { *img_ptr.add(row_base + c) } == 0 {
                    c += 1;
                }
                continue 'col;
            }

            // pixel != 0: check for contour start conditions
            let idx = row_base + c;
            let left = unsafe { *img_ptr.add(idx - 1) };
            let right = unsafe { *img_ptr.add(idx + 1) };

            let is_outer = (pixel == 1) & (left == 0);
            let is_hole = (pixel >= 1) & (right == 0) & !is_outer;

            if is_outer || is_hole {
                if nbd == i16::MAX {
                    return Err(ContoursError::NbdOverflow);
                }
                nbd += 1;

                let border_type = if is_outer {
                    BorderType::Outer
                } else {
                    BorderType::Hole
                };
                let parent = determine_parent(lnbd as i32, border_type, &hierarchy, &border_types);
                let start_dir: usize = if is_outer { 0 } else { 4 };
                let ts = TracerStart {
                    idx,
                    row: r as i32,
                    col: c as i32,
                    dir: start_dir,
                    nbd,
                    method,
                };
                let range = trace_border(img_ptr, ts, &toff, arena);

                let hier_entry = update_hierarchy(&mut hierarchy, nbd as usize, parent);
                hierarchy.push(hier_entry);
                border_types.push(border_type);
                ranges.push(range);
            } else if pixel == 1 {
                // All-1 SWAR skip: interior pixels (both neighbours nonzero) can
                // never be a contour start.  Guard the last chunk element against
                // a potential hole start (right neighbour may be 0)
                // SAFETY: c + 4 <= width -> 8-byte load stays within the padded row
                c += 1;
                while c + 4 <= width {
                    let word =
                        unsafe { (img_ptr.add(row_base + c) as *const u64).read_unaligned() };
                    if word != ALL_ONES_I16 {
                        break;
                    }
                    // Pixels c..c+3 are all 1, left of c = prior pixel = 1 != 0
                    // Check right of c+3 (= pixel[c+4]) for potential is_hole
                    let right_peek = unsafe { *img_ptr.add(row_base + c + 4) };
                    if right_peek == 0 {
                        // c+3 may be is_hole; skip only c..c+2, let loop handle c+3
                        c += 3;
                        break;
                    }
                    c += 4;
                }
                continue 'col;
            } else {
                // Labelled pixel: update lnbd to track the most recently seen border
                lnbd = pixel.unsigned_abs() as i16;
            }

            c += 1;
        }
    }

    // Materialise contours from the flat arena (rayon overhead outweighs benefit here)
    let raw_contours: Vec<Contour> = ranges.iter().map(|r| arena[r.clone()].to_vec()).collect();

    Ok(filter_by_mode(raw_contours, hierarchy, border_types, mode))
}

#[inline(always)]
fn trace_border(
    img: *mut i16,
    ts: TracerStart,
    toff: &TracerOffsets,
    arena: &mut Vec<[i32; 2]>,
) -> Range<usize> {
    let TracerStart {
        idx: start_idx,
        row: start_row,
        col: start_col,
        dir: start_dir,
        nbd,
        method,
    } = ts;
    let arena_start = arena.len();

    // Find first nonzero neighbour.
    let mut first_nb_idx = 0usize;
    let mut first_nb_dir = 0usize;
    let mut found = false;
    for k in 0..8usize {
        let d = start_dir + k;
        // SAFETY: d & 7 is always in 0..8
        let nb = (start_idx as isize + toff.o16[d & 7]) as usize;
        if unsafe { *img.add(nb) } != 0 {
            first_nb_idx = nb;
            first_nb_dir = d & 7;
            found = true;
            break;
        }
    }

    if !found {
        unsafe { *img.add(start_idx) = -nbd };
        arena.push([start_col - 1, start_row - 1]);
        return arena_start..arena.len();
    }

    arena.push([start_col - 1, start_row - 1]);

    let mut i2_idx = first_nb_idx;
    let mut i2_row = start_row + DIR_DR[first_nb_dir];
    let mut i2_col = start_col + DIR_DC[first_nb_dir];
    let mut dir_in = first_nb_dir;
    let mut dir_out_final = dir_in;

    loop {
        let cur = unsafe { *img.add(i2_idx) };
        let left_nb = unsafe { *img.add(i2_idx - 1) };
        let right_nb = unsafe { *img.add(i2_idx + 1) };

        if left_nb == 0 && cur == 1 {
            unsafe { *img.add(i2_idx) = nbd };
        } else if right_nb == 0 && cur > 0 {
            unsafe { *img.add(i2_idx) = -nbd };
        }

        // Scan 8 neighbours starting at (dir_in + 5) & 7 , the Suzuki-Abe behind rule
        let scan_start = (dir_in + 5) & 7;
        let mut i3_idx = 0usize;
        let mut i3_row = i2_row;
        let mut i3_col = i2_col;
        let mut dir_out = scan_start;
        let mut found_next = false;

        for k in 0..8usize {
            let s = scan_start + k;
            // SAFETY: s & 7 is always in 0..8, so offsets.get_unchecked(s & 7)
            // is always in bounds, Using get_unchecked avoids a redundant bounds
            // check that the compiler cannot eliminate inside the hot inner loop
            let nb = unsafe { (i2_idx as isize + *toff.o8.get_unchecked(s & 7)) as usize };
            if unsafe { *img.add(nb) } != 0 {
                i3_idx = nb;
                i3_row = i2_row + DIR_DR[s & 7];
                i3_col = i2_col + DIR_DC[s & 7];
                dir_out = s & 7;
                found_next = true;
                break;
            }
        }

        if !found_next {
            break;
        }
        dir_out_final = dir_out;

        if i2_idx == start_idx && i3_idx == first_nb_idx {
            break;
        }

        match method {
            ContourApproximationMode::None => {
                arena.push([i2_col - 1, i2_row - 1]);
            }
            ContourApproximationMode::Simple => {
                if dir_in != dir_out {
                    arena.push([i2_col - 1, i2_row - 1]);
                }
            }
        }

        i2_idx = i3_idx;
        i2_row = i3_row;
        i2_col = i3_col;
        dir_in = dir_out;
    }

    // For Simple mode: emit the last segment's corner if the direction changed.
    if method == ContourApproximationMode::Simple && i2_idx != start_idx {
        let dr = start_row - i2_row;
        let dc = start_col - i2_col;
        // dr and dc each ∈ {-1, 0, 1} because i2 and start are 8-connected
        // neighbours, so the index (dr+1)*3+(dc+1) ∈ 0..=8.
        // Index 4 (dr=0, dc=0) cannot occur because i2 != start_idx at this
        // point; in debug builds assert that invariant
        let idx4 = ((dr + 1) * 3 + (dc + 1)) as usize;
        debug_assert_ne!(idx4, 4, "i2 has the same (row,col) as start: (dr,dc)==(0,0) is impossible here because i2 != start at this point in the loop");
        let dir_out_to_start = DIR_LUT[idx4];
        let pt = [i2_col - 1, i2_row - 1];
        if dir_out_final != dir_out_to_start && arena[arena_start..].last() != Some(&pt) {
            arena.push(pt);
        }
    }

    if unsafe { *img.add(start_idx) } == 1 {
        unsafe { *img.add(start_idx) = nbd };
    }

    arena_start..arena.len()
}

#[inline(always)]
fn determine_parent(
    lnbd: i32,
    border_type: BorderType,
    hierarchy: &[HierarchyEntry],
    border_types: &[BorderType],
) -> i32 {
    let lnbd_idx = (lnbd - 1) as usize;
    // lnbd is always set from a pixel value we previously wrote (nbd >= 2),
    // so lnbd_idx is always valid
    debug_assert!(
        lnbd_idx < hierarchy.len(),
        "lnbd={lnbd} yields out-of-bounds index; hierarchy has {} entries",
        hierarchy.len()
    );
    if lnbd_idx >= hierarchy.len() {
        return -1;
    }
    let lnbd_is_hole = matches!(border_types[lnbd_idx], BorderType::Hole);
    match border_type {
        BorderType::Outer => hierarchy[lnbd_idx][3],
        BorderType::Hole => {
            if lnbd_is_hole {
                hierarchy[lnbd_idx][3]
            } else {
                lnbd
            }
        }
    }
}

#[inline(always)]
fn update_hierarchy(hierarchy: &mut [HierarchyEntry], nbd: usize, parent: i32) -> HierarchyEntry {
    let mut entry = [-1i32, -1, -1, parent];
    if parent >= 0 {
        let pidx = (parent - 1) as usize;
        // parent is always a previously-traced nbd, so pidx < hierarchy.len()
        debug_assert!(
            pidx < hierarchy.len(),
            "parent={parent} yields out-of-bounds pidx; hierarchy has {} entries",
            hierarchy.len()
        );
        if pidx < hierarchy.len() {
            if hierarchy[pidx][2] == -1 {
                hierarchy[pidx][2] = nbd as i32;
            } else {
                let mut sib = hierarchy[pidx][2] as usize;
                // Walk to the last sibling, Termination is guaranteed because
                // nbd is strictly monotonically increasing: every next pointer
                // (hierarchy[x][0]) points to a border traced *after* x, so it
                // always has a strictly larger index, Cycles are impossible
                while hierarchy[sib - 1][0] != -1 {
                    sib = hierarchy[sib - 1][0] as usize;
                }
                hierarchy[sib - 1][0] = nbd as i32;
                entry[1] = sib as i32;
            }
        }
    }
    entry
}

fn filter_by_mode(
    contours: Vec<Contour>,
    hierarchy: Vec<HierarchyEntry>,
    border_types: Vec<BorderType>,
    mode: RetrievalMode,
) -> ContoursResult {
    match mode {
        RetrievalMode::External => {
            let mut fc = Vec::new();
            let mut fh = Vec::new();
            for (i, h) in hierarchy.iter().enumerate().skip(1) {
                if matches!(border_types[i], BorderType::Outer) && h[3] <= 0 {
                    fc.push(contours[i - 1].clone());
                    fh.push([-1i32, -1, -1, -1]);
                }
            }
            ContoursResult {
                contours: fc,
                hierarchy: fh,
            }
        }
        RetrievalMode::List => {
            let fh = vec![[-1i32, -1, -1, -1]; contours.len()];
            ContoursResult {
                contours,
                hierarchy: fh,
            }
        }
        RetrievalMode::CComp => {
            // Two-level hierarchy: outer contours at level 1, holes at level 2
            // Outer contours inside holes are re-rooted to level 1
            let mut fh: Vec<HierarchyEntry> = hierarchy[1..].to_vec();
            for i in 0..fh.len() {
                let parent = fh[i][3];
                let is_outer_inside_hole = parent > 0
                    && matches!(border_types[i + 1], BorderType::Outer)
                    && matches!(border_types[(parent - 1) as usize], BorderType::Hole);
                if is_outer_inside_hole {
                    fh[i][3] = -1; // re-root: outer-inside-hole -> top level
                }
            }
            ContoursResult {
                contours,
                hierarchy: fh,
            }
        }
        RetrievalMode::Tree => {
            // Full hierarchy as computed, frame sentinel (index 0) stripped
            ContoursResult {
                contours,
                hierarchy: hierarchy[1..].to_vec(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::allocator::CpuAllocator;
    use kornia_tensor::Tensor3;

    fn make_img(w: usize, h: usize, data: Vec<u8>) -> Image<u8, 1, CpuAllocator> {
        Image(Tensor3::from_shape_vec([h, w, 1], data, CpuAllocator).expect("tensor"))
    }

    /// 3×3 filled square: 8 border pixels, None approx keeps all
    #[test]
    fn test_simple_square_no_approx() {
        let img = make_img(
            5,
            5,
            vec![
                0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
            ],
        );
        let r = find_contours(
            &img,
            RetrievalMode::External,
            ContourApproximationMode::None,
        )
        .unwrap();
        assert_eq!(r.contours.len(), 1);
        assert_eq!(r.contours[0].len(), 8);
    }

    /// Simple approx collapses the same square to its 4 corners
    #[test]
    fn test_simple_square_simple_approx() {
        let img = make_img(
            5,
            5,
            vec![
                0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
            ],
        );
        let r = find_contours(
            &img,
            RetrievalMode::External,
            ContourApproximationMode::Simple,
        )
        .unwrap();
        assert_eq!(r.contours.len(), 1);
        assert_eq!(r.contours[0].len(), 4);
    }

    /// The 4 corners of a 3×3 square must be the pixels at (1,1),(3,1),(3,3),(1,3)
    #[test]
    fn test_simple_square_corner_coordinates() {
        let img = make_img(
            5,
            5,
            vec![
                0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
            ],
        );
        let r = find_contours(
            &img,
            RetrievalMode::External,
            ContourApproximationMode::Simple,
        )
        .unwrap();
        let pts = &r.contours[0];
        // All points should be on the boundary of the 3×3 block (cols 1-3, rows 1-3)
        for &[x, y] in pts {
            assert!(
                (1..=3).contains(&x) && (1..=3).contains(&y),
                "point [{x},{y}] outside square"
            );
        }
        // Must include top-left corner (1,1) and bottom-right (3,3)
        assert!(pts.contains(&[1, 1]), "missing top-left corner");
        assert!(pts.contains(&[3, 3]), "missing bottom-right corner");
    }

    /// Isolated single pixel -> 1-point contour at the exact pixel coordinate
    #[test]
    fn test_isolated_pixel_coordinates() {
        let img = make_img(3, 3, vec![0, 0, 0, 0, 1, 0, 0, 0, 0]);
        let r = find_contours(
            &img,
            RetrievalMode::External,
            ContourApproximationMode::None,
        )
        .unwrap();
        assert_eq!(r.contours.len(), 1);
        assert_eq!(r.contours[0], vec![[1, 1]]);
    }

    /// Simple approx on a horizontal strip produces fewer points than None
    #[test]
    fn test_simple_approx_fewer_points_than_none() {
        let img = make_img(
            9,
            3,
            vec![
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
        );
        let none = find_contours(
            &img,
            RetrievalMode::External,
            ContourApproximationMode::None,
        )
        .unwrap();
        let simp = find_contours(
            &img,
            RetrievalMode::External,
            ContourApproximationMode::Simple,
        )
        .unwrap();
        assert!(none.contours[0].len() > simp.contours[0].len());
        assert!(simp.contours[0].len() >= 2);
    }

    /// Hollow square: External sees only the outer ring; List sees outer + hole
    #[test]
    fn test_hollow_square_external_vs_list() {
        let img = make_img(
            6,
            6,
            vec![
                0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1,
                1, 0, 0, 0, 0, 0, 0, 0,
            ],
        );
        let ext = find_contours(
            &img,
            RetrievalMode::External,
            ContourApproximationMode::Simple,
        )
        .unwrap();
        assert_eq!(ext.contours.len(), 1);
        let list =
            find_contours(&img, RetrievalMode::List, ContourApproximationMode::Simple).unwrap();
        assert_eq!(list.contours.len(), 2);
    }

    /// Hollow square with CComp: hole contour must have a valid parent index
    #[test]
    fn test_hollow_square_ccomp_hierarchy() {
        let img = make_img(
            6,
            6,
            vec![
                0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1,
                1, 0, 0, 0, 0, 0, 0, 0,
            ],
        );
        let r =
            find_contours(&img, RetrievalMode::CComp, ContourApproximationMode::Simple).unwrap();
        assert_eq!(r.contours.len(), 2, "CComp should return both contours");
        assert!(
            r.hierarchy.iter().any(|h| h[3] >= 0),
            "hole must have a parent"
        );
    }

    /// Outer ring + hole + inner square: verifies all four retrieval modes simultaneously
    #[test]
    fn test_all_retrieval_modes_nested_image() {
        let img = make_img(
            8,
            8,
            vec![
                0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1,
                1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
            ],
        );
        let ext = find_contours(
            &img,
            RetrievalMode::External,
            ContourApproximationMode::Simple,
        )
        .unwrap();
        let list =
            find_contours(&img, RetrievalMode::List, ContourApproximationMode::Simple).unwrap();
        let ccomp =
            find_contours(&img, RetrievalMode::CComp, ContourApproximationMode::Simple).unwrap();
        let tree =
            find_contours(&img, RetrievalMode::Tree, ContourApproximationMode::Simple).unwrap();

        assert_eq!(ext.contours.len(), 1, "External: outermost only");
        assert_eq!(list.contours.len(), 3, "List: all 3");
        assert_eq!(ccomp.contours.len(), 3, "CComp: all 3");
        assert_eq!(tree.contours.len(), 3, "Tree: all 3");

        // List discards all hierarchy links
        assert!(list.hierarchy.iter().all(|h| *h == [-1, -1, -1, -1]));
        // CComp and Tree preserve links; at least one contour has a parent
        assert_eq!(ccomp.hierarchy.len(), 3);
        assert_eq!(tree.hierarchy.len(), 3);
        assert!(ccomp.hierarchy.iter().any(|h| h[3] >= 0));
    }

    // Edge cases

    /// All-zero image -> no contours and empty hierarchy
    #[test]
    fn test_all_zeros_no_contours() {
        let img = make_img(10, 10, vec![0u8; 100]);
        let r = find_contours(&img, RetrievalMode::List, ContourApproximationMode::None).unwrap();
        assert!(r.contours.is_empty());
        assert!(r.hierarchy.is_empty());
    }

    /// 1×1 foreground image -> 1 contour with the single point at (0,0)
    #[test]
    fn test_single_pixel_image() {
        let img = make_img(1, 1, vec![1]);
        let r = find_contours(
            &img,
            RetrievalMode::External,
            ContourApproximationMode::None,
        )
        .unwrap();
        assert_eq!(r.contours.len(), 1);
        assert_eq!(r.contours[0], vec![[0, 0]]);
    }

    // SWAR code path coverage

    /// 48×48 block inside a 64×64 image: the all-1 SWAR skip must not miss the
    /// outer contour, Perimeter of a 48×48 block = 4 × 47 = 188 pixels
    #[test]
    fn test_large_filled_block_swar_all_ones_path() {
        const S: usize = 64;
        let mut data = vec![0u8; S * S];
        for r in 8..56 {
            for c in 8..56 {
                data[r * S + c] = 1;
            }
        }
        let img = make_img(S, S, data);
        let r = find_contours(
            &img,
            RetrievalMode::External,
            ContourApproximationMode::None,
        )
        .unwrap();
        assert_eq!(r.contours.len(), 1, "exactly one outer contour");
        assert_eq!(r.contours[0].len(), 4 * 47, "perimeter of 48×48 block");
    }

    /// 199-zero prefix then one foreground pixel: exercises the zero-skip SWAR path
    #[test]
    fn test_long_zero_run_swar_zero_skip_path() {
        let mut data = vec![0u8; 400];
        data[200] = 1;
        let img = make_img(400, 1, data);
        let r = find_contours(
            &img,
            RetrievalMode::External,
            ContourApproximationMode::None,
        )
        .unwrap();
        assert_eq!(r.contours.len(), 1);
        assert_eq!(r.contours[0], vec![[200, 0]]);
    }
}
