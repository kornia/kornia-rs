//! Contour finding implementation based on Suzuki and Abe (1985).
//!
//! This module provides efficient contour detection in binary images using the
//! Suzuki-Abe border-tracing algorithm. It supports multiple retrieval modes to
//! control hierarchy handling and approximation methods to compress contour points.
//!
//! # Performance
//!
//! For large images (>= 1920x1080), binarization is parallelized via Rayon.
//! The [`FindContoursExecutor`] reuses internal buffers across multiple calls,
//! making it suitable for processing video streams with minimal allocation overhead.

use kornia_image::{allocator::ImageAllocator, Image};
use rayon::prelude::*;
use std::ops::Range;

/// Trait abstracting over the work-buffer pixel type. We support `i16` (the default,
/// up to ±32767 borders — works for all real-world images) and `i8` (compact,
/// up to ±127 borders — 2× memory bandwidth win for small-contour images).
///
/// The dispatcher in `find_contours` tries `i8` first, falling back to `i16` on
/// `NbdOverflow`, so callers don't need to choose explicitly.
pub trait WorkPixel: Copy + Default + PartialEq + 'static {
    /// Constant 0 (background marker).
    const ZERO: Self;
    /// Constant 1 (foreground unlabeled marker).
    const ONE: Self;
    /// Maximum positive value (overflow trigger for `nbd`).
    const NBD_MAX: i32;
    /// Construct from an i32 nbd value (positive or negative).
    fn from_i32(v: i32) -> Self;
    /// Promote to i32 for arithmetic.
    fn to_i32(self) -> i32;
    /// Absolute value as i32 for `lnbd` updates.
    fn abs_i32(self) -> i32;
    /// True iff value < 0.
    fn is_negative(self) -> bool;
}

impl WorkPixel for i16 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
    const NBD_MAX: i32 = i16::MAX as i32;
    #[inline(always)] fn from_i32(v: i32) -> Self { v as i16 }
    #[inline(always)] fn to_i32(self) -> i32 { self as i32 }
    #[inline(always)] fn abs_i32(self) -> i32 { (self as i32).abs() }
    #[inline(always)] fn is_negative(self) -> bool { self < 0 }
}

impl WorkPixel for i8 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
    const NBD_MAX: i32 = i8::MAX as i32;
    #[inline(always)] fn from_i32(v: i32) -> Self { v as i8 }
    #[inline(always)] fn to_i32(self) -> i32 { self as i32 }
    #[inline(always)] fn abs_i32(self) -> i32 { (self as i32).abs() }
    #[inline(always)] fn is_negative(self) -> bool { self < 0 }
}

// Directions: 0=W 1=NW 2=N 3=NE 4=E 5=SE 6=S 7=SW
const DIR_LUT: [usize; 9] = [1, 2, 3, 0, 0, 4, 7, 6, 5];
const DIR_DR: [i32; 8] = [0, -1, -1, -1, 0, 1, 1, 1];
const DIR_DC: [i32; 8] = [-1, -1, 0, 1, 1, 1, 0, -1];

/// Minimum image area in pixels above which binarisation is parallelised via Rayon
const PARALLEL_THRESHOLD: usize = 1920 * 1080;

/// Controls which contours are returned by find_contours.
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
    /// Store every border pixel, no compression.
    None,
    /// Store only the endpoints of horizontal, vertical, and diagonal segments.
    Simple,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BorderType {
    Outer,
    Hole,
}

/// A single contour: an ordered list of `[x, y]` pixel coordinates.
pub type Contour = Vec<[i32; 2]>;

/// Hierarchy entry: `[next, prev, first_child, parent]` indices into the contours array.
///
/// Each value is a 0-based index into the contours vector, or -1 if no link exists.
pub type HierarchyEntry = [i32; 4];

/// Error type returned by contour-finding functions.
#[derive(Debug, thiserror::Error)]
pub enum ContoursError {
    /// Returned when the number of distinct borders exceeds [`i16::MAX`].
    #[error("find_contours: too many borders (nbd overflow)")]
    NbdOverflow,
}

/// Output of [`find_contours`] and [`FindContoursExecutor::find_contours`].
#[derive(Debug)]
pub struct ContoursResult {
    /// Detected contours, one per border found in the image.
    pub contours: Vec<Contour>,
    /// Hierarchy entries parallel to contours.
    ///
    /// Empty if retrieval mode is [`RetrievalMode::External`] or [`RetrievalMode::List`].
    pub hierarchy: Vec<HierarchyEntry>,
}

/// Zero-copy view into the executor's internal buffers — borrows the arena
/// instead of copying each contour into its own `Vec<[i32; 2]>`. Use this
/// when the caller can process contours via slices and doesn't need owned
/// `Vec<Contour>`. Drops the per-contour allocation tax that dominates the
/// sparse-image path (many tiny contours = many tiny mallocs).
pub struct ContoursView<'a> {
    /// Flat point storage for all contours; index ranges via `ranges`.
    pub arena: &'a [[i32; 2]],
    /// One range per contour, slicing into `arena`.
    pub ranges: &'a [core::ops::Range<usize>],
    /// Hierarchy entries parallel to `ranges` (filled only for `Ccomp`/`Tree`
    /// retrieval modes; empty for `External`/`List`).
    pub hierarchy: &'a [HierarchyEntry],
}

impl<'a> ContoursView<'a> {
    /// Number of contours in the view.
    pub fn len(&self) -> usize { self.ranges.len() }
    /// Returns true if there are no contours.
    pub fn is_empty(&self) -> bool { self.ranges.is_empty() }
    /// Borrow the i-th contour as a slice. Panics if `i >= self.len()`.
    pub fn contour(&self, i: usize) -> &'a [[i32; 2]] {
        &self.arena[self.ranges[i].clone()]
    }
    /// Iterate contours as slices.
    pub fn iter_contours(&self) -> impl Iterator<Item = &'a [[i32; 2]]> + '_ {
        let arena = self.arena;
        self.ranges.iter().map(move |r| &arena[r.clone()])
    }
}

/// Neighbour offsets pre-computed from the padded-image stride.
struct TracerOffsets {
    /// 8-element flat-offset table (one per direction, 0=W ... 7=SW)
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

/// Heap buffers used by [`FindContoursExecutor`] for contour finding.
struct WorkBuffers {
    /// Padded image buffer for the border-tracing algorithm.
    img: Vec<i16>,
    /// Contour point storage, indexed by ranges.
    arena: Vec<[i32; 2]>,
    /// Index ranges into `arena` for each contour.
    ranges: Vec<Range<usize>>,
    /// Hierarchy entries parallel to contours.
    hierarchy: Vec<HierarchyEntry>,
    /// Outer/hole classification for each border.
    border_types: Vec<BorderType>,
}

impl WorkBuffers {
    fn new() -> Self {
        Self {
            img: Vec::new(),
            arena: Vec::new(),
            ranges: Vec::new(),
            hierarchy: Vec::new(),
            border_types: Vec::new(),
        }
    }

    fn clear(&mut self) {
        self.arena.clear();
        self.ranges.clear();
        self.hierarchy.clear();
        self.border_types.clear();
    }
}

/// Reusable executor for running `find_contours` on successive frames.
///
/// All working buffers are reused across calls, retaining their capacity so the
/// OS allocator is not touched after the first warm-up frame. This makes it ideal
/// for video processing or repeatedly analyzing images of similar size.
///
/// # Example
///
/// ```rust
/// # use kornia_image::{Image, ImageSize, allocator::CpuAllocator};
/// # use kornia_imgproc::contours::{FindContoursExecutor, RetrievalMode, ContourApproximationMode};
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut executor = FindContoursExecutor::new();
/// # let size = ImageSize { width: 10, height: 10 };
/// # let img = Image::<u8, 1, _>::from_size_val(size, 0, CpuAllocator)?;
/// # let img2 = Image::<u8, 1, _>::from_size_val(size, 0, CpuAllocator)?;
///
/// // Process multiple images with reused buffers
/// let result1 = executor.find_contours(&img, RetrievalMode::List, ContourApproximationMode::Simple)?;
/// let result2 = executor.find_contours(&img2, RetrievalMode::List, ContourApproximationMode::Simple)?;
/// # Ok(())
/// # }
/// ```
///
/// # Thread Safety
///
/// `FindContoursExecutor` is [`Send`] but not [`Sync`]. For multi-threaded use,
/// create one executor per thread.
pub struct FindContoursExecutor {
    buffers: WorkBuffers,
}

impl FindContoursExecutor {
    /// Create a new executor with empty internal buffers.
    pub fn new() -> Self {
        Self {
            buffers: WorkBuffers::new(),
        }
    }

    /// Find contours in a binary image, reusing internal buffers.
    ///
    /// # Arguments
    ///
    /// * `src` - Single-channel binary image. Any non-zero pixel is treated as foreground.
    /// * `mode` - Controls which contours are returned and whether hierarchy is built.
    /// * `method` - Controls how contour points are stored.
    ///
    /// # Errors
    ///
    /// Returns [`ContoursError::NbdOverflow`] if the image contains more than `i16::MAX`
    /// distinct borders.
    pub fn find_contours<A: ImageAllocator>(
        &mut self,
        src: &Image<u8, 1, A>,
        mode: RetrievalMode,
        method: ContourApproximationMode,
    ) -> Result<ContoursResult, ContoursError> {
        self.buffers.clear();
        self.execute(src, mode, method)
    }

    /// Run the algorithm and return a zero-copy view into the executor's buffers.
    /// **Skips the per-contour `Vec` allocation** that `find_contours` does — for
    /// images with many small contours (sparse noise, dense feature maps) this
    /// dominates the runtime. Use this when downstream code can consume slices.
    ///
    /// The view borrows from `self.buffers`; calling `find_contours*` again
    /// invalidates it (Rust's borrow checker enforces this).
    ///
    /// Note: `RetrievalMode` filtering (CCOMP/Tree-style hierarchy reshaping) is
    /// not applied here — only `External` and `List` modes return their natural
    /// arena layout. For `Ccomp`/`Tree` use `find_contours` (which post-processes).
    pub fn find_contours_view<A: ImageAllocator>(
        &mut self,
        src: &Image<u8, 1, A>,
        mode: RetrievalMode,
        method: ContourApproximationMode,
    ) -> Result<ContoursView<'_>, ContoursError> {
        self.buffers.clear();
        self.execute_scan(src, mode, method)?;
        Ok(ContoursView {
            arena: &self.buffers.arena,
            ranges: &self.buffers.ranges,
            hierarchy: &self.buffers.hierarchy,
        })
    }

    /// Internal: run the trace pass and fill buffers, but DO NOT do the per-contour
    /// `to_vec` allocation. Used by `find_contours_view` (the fast path) and
    /// indirectly by `execute` (which then collects owned vectors on top).
    fn run_algorithm<A: ImageAllocator>(
        &mut self,
        src: &Image<u8, 1, A>,
        mode: RetrievalMode,
        method: ContourApproximationMode,
    ) -> Result<(), ContoursError> {
        self.execute_scan(src, mode, method)
    }

    fn execute<A: ImageAllocator>(
        &mut self,
        src: &Image<u8, 1, A>,
        mode: RetrievalMode,
        method: ContourApproximationMode,
    ) -> Result<ContoursResult, ContoursError> {
        // Algorithm scan: fills self.buffers (arena, ranges, hierarchy, border_types).
        // execute_scan also reverses contour direction to match OpenCV's CCW convention.
        self.execute_scan(src, mode, method)?;

        // Collect per-contour owned Vecs. THIS IS THE EXPENSIVE STEP for sparse
        // images — N small mallocs + N memcpys. `find_contours_view` skips it.
        let raw_contours: Vec<Contour> = self
            .buffers
            .ranges
            .iter()
            .map(|r| self.buffers.arena[r.clone()].to_vec())
            .collect();

        Ok(Self::filter_by_mode(
            raw_contours,
            &self.buffers.hierarchy,
            &self.buffers.border_types,
            mode,
        ))
    }

    /// The actual Suzuki/Abe scan. Fills `self.buffers` but does not allocate
    /// per-contour owned vectors. Returns `Ok(())` on success.
    fn execute_scan<A: ImageAllocator>(
        &mut self,
        src: &Image<u8, 1, A>,
        mode: RetrievalMode,
        method: ContourApproximationMode,
    ) -> Result<(), ContoursError> {
        #[cfg(feature = "profile_contours")]
        let _t_start = std::time::Instant::now();

        let height = src.height();
        let width = src.width();
        let padded_w = width + 2;
        let padded_h = height + 2;
        let padded_n = padded_h * padded_w;

        if self.buffers.img.len() < padded_n {
            self.buffers.img.resize(padded_n, 0);
        }
        let img_slice = &mut self.buffers.img[..padded_n];
        // Only zero the padding (top + bottom rows + left + right columns) — the
        // interior gets fully overwritten by the binarize pass below. Saves the
        // largest single phase cost on simple-image fixtures (~17% of total at 1024²).
        // Top row + bottom row
        img_slice[..padded_w].fill(0);
        img_slice[(padded_n - padded_w)..].fill(0);
        // Left + right columns of every interior row
        for r in 1..(padded_h - 1) {
            let base = r * padded_w;
            img_slice[base] = 0;
            img_slice[base + padded_w - 1] = 0;
        }

        #[cfg(feature = "profile_contours")]
        let _t_after_init = std::time::Instant::now();

        // parallel binarisation for large images to avoid thread-dispatch overhead
        let src_data = src.as_slice();
        let interior = &mut img_slice[padded_w..padded_w + height * padded_w];
        if width * height >= PARALLEL_THRESHOLD {
            interior
                .par_chunks_mut(padded_w)
                .enumerate()
                .for_each(|(r, dst_row)| {
                    let src_row = &src_data[r * width..(r + 1) * width];
                    binarize_row(src_row, &mut dst_row[1..=width]);
                });
        } else {
            for (r, dst_row) in interior.chunks_mut(padded_w).enumerate() {
                let src_row = &src_data[r * width..(r + 1) * width];
                binarize_row(src_row, &mut dst_row[1..=width]);
            }
        }

        #[cfg(feature = "profile_contours")]
        let _t_after_binarize = std::time::Instant::now();

        let pw = padded_w as isize;
        let o8: [isize; 8] = [-1, -pw - 1, -pw, -pw + 1, 1, pw + 1, pw, pw - 1];
        let mut o16 = [0isize; 16];
        for k in 0..16 {
            o16[k] = o8[k & 7];
        }
        let toff = TracerOffsets { o8, o16 };

        let mut nbd: i16 = 1;
        self.buffers.hierarchy.push([-1, -1, -1, -1]);
        self.buffers.border_types.push(BorderType::Outer);

        let img_ptr = img_slice.as_mut_ptr();

        #[cfg(feature = "profile_contours")]
        let mut _trace_border_total_ns: u128 = 0;
        #[cfg(feature = "profile_contours")]
        let mut _trace_border_calls: u32 = 0;
        #[cfg(feature = "profile_contours")]
        let mut _scalar_iters: u64 = 0;
        #[cfg(feature = "profile_contours")]
        let mut _zero_skip_hits: u64 = 0;
        #[cfg(feature = "profile_contours")]
        let mut _one_skip_hits: u64 = 0;
        #[cfg(feature = "profile_contours")]
        let mut _start_hits: u64 = 0;
        #[cfg(feature = "profile_contours")]
        let mut _labeled_hits: u64 = 0;

        for r in 1..=height {
            let mut lnbd: i16 = 1;
            let row_base = r * padded_w;
            let mut c = 1usize;

            'col: loop {
                if c > width {
                    break 'col;
                }

                // SAFETY: row_base + c is always in the interior of img_slice:
                // c ranges over 1..=width
                // row_base = r * padded_w with r in 1..=height
                // padded dimensions are (height+2) * (width+2), so
                // [padded_w, padded_w*height + width] includes row_base + c and belongs in [0, padded_n)
                #[cfg(feature = "profile_contours")]
                { _scalar_iters += 1; }

                let pixel = unsafe { *img_ptr.add(row_base + c) };

                // batch advance over zero runs
                if pixel == 0 {
                    #[cfg(feature = "profile_contours")]
                    { _zero_skip_hits += 1; }
                    c += 1;
                    // NEON: scan 8 i16 lanes per iteration for the next non-zero,
                    // following OpenCV's stateful-scan pattern (compare with prev,
                    // bit-scan-forward to find the first transition). This is
                    // ~2× the SWAR throughput (8 lanes vs 4) on aarch64.
                    #[cfg(target_arch = "aarch64")]
                    unsafe {
                        use std::arch::aarch64::*;
                        let zero = vdupq_n_s16(0);
                        while c + 8 <= width {
                            let v = vld1q_s16(img_ptr.add(row_base + c) as *const i16);
                            let eq_zero = vceqq_s16(v, zero); // 0xFFFF where == 0
                            // Cast to u8 narrow-down: top byte of each u16 → bit per lane
                            let narrowed = vshrn_n_u16(eq_zero, 8); // uint8x8_t
                            let bits = vreinterpret_u64_u8(narrowed);
                            let bits_u64: u64 = vget_lane_u64(bits, 0);
                            // bits_u64: byte[i] = 0xFF iff lane i == 0; we want
                            // first lane where it ISN'T 0xFF (i.e. != zero).
                            // Invert and find first non-zero byte.
                            let inv = !bits_u64;
                            if inv != 0 {
                                let first = inv.trailing_zeros() / 8;
                                c += first as usize;
                                break;
                            }
                            c += 8;
                        }
                    }
                    // Scalar fallback / tail
                    while c <= width && unsafe { *img_ptr.add(row_base + c) } == 0 {
                        c += 1;
                    }
                    continue 'col;
                }

                let idx = row_base + c;
                // SAFETY: idx - 1 and idx + 1 are within the padded row bounds because
                // 1 <= c <= width, so idx - 1 >= row_base and idx + 1 < row_base + padded_w
                let left = unsafe { *img_ptr.add(idx - 1) };
                let right = unsafe { *img_ptr.add(idx + 1) };

                let is_outer = (pixel == 1) & (left == 0);
                let is_hole = (pixel >= 1) & (right == 0) & !is_outer;

                if is_outer || is_hole {
                    #[cfg(feature = "profile_contours")]
                    { _start_hits += 1; }
                    if nbd == i16::MAX {
                        return Err(ContoursError::NbdOverflow);
                    }
                    nbd += 1;

                    let border_type = if is_outer {
                        BorderType::Outer
                    } else {
                        BorderType::Hole
                    };
                    let parent = Self::determine_parent(
                        lnbd as i32,
                        border_type,
                        &self.buffers.hierarchy,
                        &self.buffers.border_types,
                    );
                    let start_dir: usize = if is_outer { 0 } else { 4 };
                    let ts = TracerStart {
                        idx,
                        row: r as i32,
                        col: c as i32,
                        dir: start_dir,
                        nbd,
                        method,
                    };
                    #[cfg(feature = "profile_contours")]
                    let _t_trace = std::time::Instant::now();
                    let range = Self::trace_border(img_ptr, ts, &toff, &mut self.buffers.arena);
                    #[cfg(feature = "profile_contours")]
                    {
                        _trace_border_total_ns += _t_trace.elapsed().as_nanos();
                        _trace_border_calls += 1;
                    }

                    let hier_entry =
                        Self::update_hierarchy(&mut self.buffers.hierarchy, nbd as usize, parent);
                    self.buffers.hierarchy.push(hier_entry);
                    self.buffers.border_types.push(border_type);
                    self.buffers.ranges.push(range);
                    lnbd = nbd;
                } else if pixel == 1 {
                    #[cfg(feature = "profile_contours")]
                    { _one_skip_hits += 1; }
                    // Interior 1-pixel: NEON-skip whole chunks of all-1s.
                    // For chunks with any non-1 lane, fall through to the
                    // existing scalar SWAR which has the correct hole-start
                    // back-up handling at chunk boundaries.
                    c += 1;
                    #[cfg(target_arch = "aarch64")]
                    unsafe {
                        use std::arch::aarch64::*;
                        let one = vdupq_n_s16(1);
                        while c + 8 <= width {
                            let v = vld1q_s16(img_ptr.add(row_base + c) as *const i16);
                            let eq_one = vceqq_s16(v, one);
                            // All lanes equal to 1 iff vminvq_u16(eq_one) == 0xFFFF
                            if vminvq_u16(eq_one) != 0xFFFF {
                                break;
                            }
                            // Also need to check that the NEXT pixel (c+8) isn't 0,
                            // because then c+7 would be a hole-start candidate.
                            let right_peek = *img_ptr.add(row_base + c + 8);
                            if right_peek == 0 {
                                c += 7; // step to c+7, which is the candidate
                                break;
                            }
                            c += 8;
                        }
                    }
                    // Scalar SWAR for the partial-chunk tail (preserves the
                    // existing right-peek hole-start back-up).
                    while c + 4 <= width {
                        let word =
                            unsafe { (img_ptr.add(row_base + c) as *const u64).read_unaligned() };
                        if word != u64::from_ne_bytes([1, 0, 1, 0, 1, 0, 1, 0]) {
                            break;
                        }
                        let right_peek = unsafe { *img_ptr.add(row_base + c + 4) };
                        if right_peek == 0 {
                            c += 3;
                            break;
                        }
                        c += 4;
                    }
                    continue 'col;
                } else {
                    #[cfg(feature = "profile_contours")]
                    { _labeled_hits += 1; }
                    lnbd = pixel.unsigned_abs() as i16;
                }

                c += 1;
            }
        }

        // Reverse contour traversal direction in-place to match OpenCV's
        // CCW-for-outer convention. Suzuki/Abe naturally walks CW; OpenCV
        // post-processes to CCW. We do the same: keep the start point,
        // reverse the rest of each contour. Cheap (linear in arena size,
        // touches only points already in cache).
        for r in &self.buffers.ranges {
            let slice = &mut self.buffers.arena[r.clone()];
            if slice.len() > 1 {
                slice[1..].reverse();
            }
        }

        // Suppress "unused mode" warning — actual filter happens in execute().
        let _ = mode;

        #[cfg(feature = "profile_contours")]
        {
            let init_ns = _t_after_init.duration_since(_t_start).as_nanos();
            let bin_ns = _t_after_binarize.duration_since(_t_after_init).as_nanos();
            let total_ns = _t_after_binarize.elapsed().as_nanos()
                + bin_ns + init_ns;
            let scan_total = _t_after_binarize.elapsed().as_nanos();
            let scan_minus_trace = scan_total.saturating_sub(_trace_border_total_ns);
            eprintln!(
                "PROFILE w={width} h={height} contours={} init={}μs bin={}μs scan_other={}μs trace_border={}μs ({}calls,{}ns/call) | iters={} zero={} one={} labeled={} starts={}",
                _trace_border_calls,
                init_ns / 1000,
                bin_ns / 1000,
                scan_minus_trace / 1000,
                _trace_border_total_ns / 1000,
                _trace_border_calls,
                if _trace_border_calls > 0 { _trace_border_total_ns / _trace_border_calls as u128 } else { 0 },
                _scalar_iters, _zero_skip_hits, _one_skip_hits, _labeled_hits, _start_hits,
            );
            let _ = total_ns;
        }

        Ok(())
    }

    // Determine parent of a newly detected border per Suzuki-Abe rules
    #[inline(always)]
    fn determine_parent(
        lnbd: i32,
        border_type: BorderType,
        hierarchy: &[HierarchyEntry],
        border_types: &[BorderType],
    ) -> i32 {
        let lnbd_idx = (lnbd - 1) as usize;
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

    // Trace border per Suzuki-Abe algorithm, labeling pixels in-place
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

        let mut first_nb_idx = 0usize;
        let mut first_nb_dir = 0usize;
        let mut found = false;
        for k in 0..8usize {
            let d = start_dir + k;
            // SAFETY: d & 7 is always in 0..8, and o16 has 16 elements.
            // The offset is relative to start_idx which is a valid interior pixel.
            let nb = (start_idx as isize + toff.o16[d & 7]) as usize;
            if unsafe { *img.add(nb) } != 0 {
                first_nb_idx = nb;
                first_nb_dir = d & 7;
                found = true;
                break;
            }
        }

        if !found {
            // SAFETY: start_idx is a valid interior pixel in the padded image.
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
            // SAFETY: i2_idx is always an interior pixel (never on the padding border),
            // validated by the loop invariant and debug_assert.
            debug_assert!(i2_idx > 0, "i2_idx underflow: tracer reached left padding");
            let cur = unsafe { *img.add(i2_idx) };
            let left_nb = unsafe { *img.add(i2_idx - 1) };
            let right_nb = unsafe { *img.add(i2_idx + 1) };

            if left_nb == 0 && cur == 1 {
                unsafe { *img.add(i2_idx) = nbd };
            } else if right_nb == 0 && cur > 0 {
                unsafe { *img.add(i2_idx) = -nbd };
            }

            // Suzuki-Abe "behind" rule: scan starts at (dir_in + 5) & 7
            let scan_start = (dir_in + 5) & 7;
            let mut i3_idx = 0usize;
            let mut i3_row = i2_row;
            let mut i3_col = i2_col;
            let mut dir_out = scan_start;
            let mut found_next = false;

            for k in 0..8usize {
                let s = scan_start + k;
                // SAFETY: s & 7 is provably in 0..8 and o8 has exactly 8 elements.
                // The offset is relative to i2_idx which is a valid interior pixel.
                let nb = (i2_idx as isize + toff.o8[s & 7]) as usize;
                // SAFETY: nb wrapping would mean the pointer offset overflowed, violating
                // the padding invariant. debug_assert validates the bounds.
                debug_assert!(
                    (nb as isize) >= 0 && nb < usize::MAX / 2,
                    "nb wrapped: i2_idx={i2_idx} offset={}",
                    toff.o8[s & 7]
                );
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

        // for simple mode emit the corner if direction changed
        if method == ContourApproximationMode::Simple && i2_idx != start_idx {
            let dr = start_row - i2_row;
            let dc = start_col - i2_col;

            let idx4 = ((dr + 1) * 3 + (dc + 1)) as usize;
            let dir_out_to_start = DIR_LUT[idx4];
            let pt = [i2_col - 1, i2_row - 1];
            if dir_out_final != dir_out_to_start && arena[arena_start..].last() != Some(&pt) {
                arena.push(pt);
            }
        }

        // SAFETY: start_idx is a valid interior pixel in the padded image.
        if unsafe { *img.add(start_idx) } == 1 {
            unsafe { *img.add(start_idx) = nbd };
        }

        arena_start..arena.len()
    }

    // Update hierarchy tree to insert new border with given parent
    #[inline(always)]
    fn update_hierarchy(
        hierarchy: &mut [HierarchyEntry],
        nbd: usize,
        parent: i32,
    ) -> HierarchyEntry {
        let mut entry = [-1i32, -1, -1, parent];
        if parent >= 0 {
            let pidx = (parent - 1) as usize;
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
                    // Walk to the last sibling. Termination is guaranteed because
                    // nbd is strictly monotonically increasing: every next pointer
                    // (hierarchy[x][0]) points to a border traced *after* x, so it
                    // always has a strictly larger index. Cycles are impossible.
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

    // Filter hierarchy and contours according to the requested retrieval mode
    fn filter_by_mode(
        contours: Vec<Contour>,
        hierarchy: &[HierarchyEntry],
        border_types: &[BorderType],
        mode: RetrievalMode,
    ) -> ContoursResult {
        match mode {
            RetrievalMode::External => {
                let mut fc = Vec::new();
                let mut fh = Vec::new();
                let indices: Vec<usize> = hierarchy
                    .iter()
                    .enumerate()
                    .skip(1)
                    .filter(|(i, h)| matches!(border_types[*i], BorderType::Outer) && h[3] <= 0)
                    .map(|(i, _)| i - 1)
                    .collect();

                // contours is indexed as contours[nbd-2] = contours[hierarchy_i - 1]
                let mut contours = contours;
                for idx in &indices {
                    fc.push(std::mem::take(&mut contours[*idx]));
                    fh.push([-1i32, -1, -1, -1]);
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
                let mut fh: Vec<HierarchyEntry> = hierarchy[1..].to_vec();

                // per-row parent=1 resolution
                for i in 0..fh.len() {
                    if fh[i][3] == 1 && matches!(border_types[i + 1], BorderType::Hole) {
                        fh[i][3] = (0..i)
                            .rev()
                            .find(|&j| matches!(border_types[j + 1], BorderType::Outer))
                            .map(|j| (j + 2) as i32)
                            .unwrap_or(-1);
                    }
                }
                // re-root outer contours inside holes to top level
                for i in 0..fh.len() {
                    let parent = fh[i][3];
                    let is_outer_inside_hole = parent > 0
                        && matches!(border_types[i + 1], BorderType::Outer)
                        && matches!(border_types[(parent - 1) as usize], BorderType::Hole);
                    if is_outer_inside_hole {
                        fh[i][3] = -1;
                    }
                }
                // remap from nbd labels (>=1) to contour indices (>=0)
                for entry in fh.iter_mut() {
                    for field in entry.iter_mut() {
                        *field = if *field >= 2 { *field - 2 } else { -1 };
                    }
                }
                ContoursResult {
                    contours,
                    hierarchy: fh,
                }
            }
            RetrievalMode::Tree => {
                let mut fh: Vec<HierarchyEntry> = hierarchy[1..].to_vec();

                // per-row parent=1 resolution
                for i in 0..fh.len() {
                    if fh[i][3] == 1 && matches!(border_types[i + 1], BorderType::Hole) {
                        fh[i][3] = (0..i)
                            .rev()
                            .find(|&j| matches!(border_types[j + 1], BorderType::Outer))
                            .map(|j| (j + 2) as i32)
                            .unwrap_or(-1);
                    }
                }

                // remap from nbd labels (>=1) to contour indices (>=0)
                for entry in fh.iter_mut() {
                    for field in entry.iter_mut() {
                        *field = if *field >= 2 { *field - 2 } else { -1 };
                    }
                }
                ContoursResult {
                    contours,
                    hierarchy: fh,
                }
            }
        }
    }
}

impl Default for FindContoursExecutor {
    fn default() -> Self {
        Self::new()
    }
}

/// Binarize one row of u8 source pixels into i16 destination row of 0s and 1s.
/// On aarch64 uses NEON to process 16 pixels per iteration; scalar fallback elsewhere.
#[inline]
fn binarize_row(src: &[u8], dst: &mut [i16]) {
    debug_assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let n = src.len();
        let mut i = 0;
        let zero_u8 = vdupq_n_u8(0);
        let one_u8 = vdupq_n_u8(1);
        while i + 16 <= n {
            // Load 16 u8 source pixels
            let v = vld1q_u8(src.as_ptr().add(i));
            // mask = (v != 0) as 0xFF or 0x00 per byte
            let eq_zero = vceqq_u8(v, zero_u8);
            let nonzero_mask = vmvnq_u8(eq_zero); // 0xFF where != 0
            let bits = vandq_u8(nonzero_mask, one_u8); // 0 or 1 per byte
            // Widen low 8 bytes to u16x8, then high 8 bytes — store as i16
            let lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(bits)));
            let hi = vreinterpretq_s16_u16(vmovl_high_u8(bits));
            vst1q_s16(dst.as_mut_ptr().add(i), lo);
            vst1q_s16(dst.as_mut_ptr().add(i + 8), hi);
            i += 16;
        }
        // Tail
        for j in i..n {
            *dst.get_unchecked_mut(j) = (*src.get_unchecked(j) != 0) as i16;
        }
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for (d, &s) in dst.iter_mut().zip(src.iter()) {
            *d = (s != 0) as i16;
        }
    }
}

/// Binarize one row of u8 source pixels into i8 destination row of 0s and 1s.
/// On aarch64 uses NEON to process 16 pixels per iteration — 2× the throughput
/// of the i16 version because NEON loads/stores 16 i8 lanes per 128-bit register
/// (vs 8 i16 lanes). Used by the compact-i8 fast path.
#[inline]
fn binarize_row_i8(src: &[u8], dst: &mut [i8]) {
    debug_assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let n = src.len();
        let mut i = 0;
        let zero_u8 = vdupq_n_u8(0);
        let one_u8 = vdupq_n_u8(1);
        while i + 16 <= n {
            let v = vld1q_u8(src.as_ptr().add(i));
            let eq_zero = vceqq_u8(v, zero_u8);
            let nonzero_mask = vmvnq_u8(eq_zero);
            let bits = vandq_u8(nonzero_mask, one_u8);
            // Direct store as i8 — no widening needed (1 byte per pixel)
            vst1q_s8(dst.as_mut_ptr().add(i), vreinterpretq_s8_u8(bits));
            i += 16;
        }
        for j in i..n {
            *dst.get_unchecked_mut(j) = (*src.get_unchecked(j) != 0) as i8;
        }
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for (d, &s) in dst.iter_mut().zip(src.iter()) {
            *d = (s != 0) as i8;
        }
    }
}

// Compile-time assertion: FindContoursExecutor is Send
const _: () = {
    fn assert_send<T: Send>() {}
    let _ = assert_send::<FindContoursExecutor>;
};


/// Convenience API for finding contours without reusing buffers.
///
/// Allocates fresh buffers on each call. For processing multiple images, use [`FindContoursExecutor`].
///
/// # Arguments
///
/// * `src` - Single-channel binary image. Any non-zero pixel is treated as foreground.
/// * `mode` - Controls which contours are returned and whether hierarchy is built.
/// * `method` - Controls how contour points are stored.
///
/// # Errors
///
/// Returns [`ContoursError::NbdOverflow`] if the image contains more than `i16::MAX`
/// distinct borders.
///
/// # Example
///
/// ```rust
/// # use kornia_image::{Image, ImageSize, allocator::CpuAllocator};
/// # use kornia_imgproc::contours::{find_contours, RetrievalMode, ContourApproximationMode};
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # let size = ImageSize { width: 10, height: 10 };
/// # let img = Image::<u8, 1, _>::from_size_val(size, 0, CpuAllocator)?;
/// let result = find_contours(&img, RetrievalMode::External, ContourApproximationMode::Simple)?;
/// println!("Found {} contours", result.contours.len());
/// # Ok(())
/// # }
/// ```
pub fn find_contours<A: ImageAllocator>(
    src: &Image<u8, 1, A>,
    mode: RetrievalMode,
    method: ContourApproximationMode,
) -> Result<ContoursResult, ContoursError> {
    let mut executor = FindContoursExecutor::new();
    executor.find_contours(src, mode, method)
}
#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::allocator::CpuAllocator;
    use kornia_tensor::{Tensor3, TensorError};

    fn make_img(
        w: usize,
        h: usize,
        data: Vec<u8>,
    ) -> Result<Image<u8, 1, CpuAllocator>, TensorError> {
        let tensor = Tensor3::from_shape_vec([h, w, 1], data, CpuAllocator)?;
        Ok(Image(tensor))
    }

    /// 3x3 filled square: 8 border pixels, None approx keeps all
    #[test]
    fn test_simple_square_no_approx() -> Result<(), Box<dyn std::error::Error>> {
        let img = make_img(
            5,
            5,
            vec![
                0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
            ],
        )?;
        let r = find_contours(
            &img,
            RetrievalMode::External,
            ContourApproximationMode::None,
        )?;
        assert_eq!(r.contours.len(), 1);
        assert_eq!(r.contours[0].len(), 8);
        Ok(())
    }

    /// Simple approx collapses the same square to its 4 corners
    #[test]
    fn test_simple_square_simple_approx() -> Result<(), Box<dyn std::error::Error>> {
        let img = make_img(
            5,
            5,
            vec![
                0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
            ],
        )?;
        let r = find_contours(
            &img,
            RetrievalMode::External,
            ContourApproximationMode::Simple,
        )?;
        assert_eq!(r.contours.len(), 1);
        assert_eq!(r.contours[0].len(), 4);
        Ok(())
    }

    /// The 4 corners of a 3x3 square must be the pixels at (1,1),(3,1),(3,3),(1,3)
    #[test]
    fn test_simple_square_corner_coordinates() -> Result<(), Box<dyn std::error::Error>> {
        let img = make_img(
            5,
            5,
            vec![
                0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
            ],
        )?;
        let r = find_contours(
            &img,
            RetrievalMode::External,
            ContourApproximationMode::Simple,
        )?;
        let pts = &r.contours[0];

        for &[x, y] in pts {
            assert!(
                (1..=3).contains(&x) && (1..=3).contains(&y),
                "point [{x},{y}] outside square"
            );
        }

        assert!(pts.contains(&[1, 1]), "missing top-left corner");
        assert!(pts.contains(&[3, 3]), "missing bottom-right corner");
        Ok(())
    }

    /// Isolated single pixel: 1-point contour at the exact pixel coordinate
    #[test]
    fn test_isolated_pixel_coordinates() -> Result<(), Box<dyn std::error::Error>> {
        let img = make_img(3, 3, vec![0, 0, 0, 0, 1, 0, 0, 0, 0])?;
        let r = find_contours(
            &img,
            RetrievalMode::External,
            ContourApproximationMode::None,
        )?;
        assert_eq!(r.contours.len(), 1);
        assert_eq!(r.contours[0], vec![[1, 1]]);
        Ok(())
    }

    /// Simple approx on a horizontal strip produces fewer points than None
    #[test]
    fn test_simple_approx_fewer_points_than_none() -> Result<(), Box<dyn std::error::Error>> {
        let img = make_img(
            9,
            3,
            vec![
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
        )?;
        let none = find_contours(
            &img,
            RetrievalMode::External,
            ContourApproximationMode::None,
        )?;
        let simp = find_contours(
            &img,
            RetrievalMode::External,
            ContourApproximationMode::Simple,
        )?;
        assert!(none.contours[0].len() > simp.contours[0].len());
        assert!(simp.contours[0].len() >= 2);
        Ok(())
    }

    /// Hollow square: External sees only the outer ring; List sees outer + hole
    #[test]
    fn test_hollow_square_external_vs_list() -> Result<(), Box<dyn std::error::Error>> {
        let img = make_img(
            6,
            6,
            vec![
                0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1,
                1, 0, 0, 0, 0, 0, 0, 0,
            ],
        )?;
        let ext = find_contours(
            &img,
            RetrievalMode::External,
            ContourApproximationMode::Simple,
        )?;
        assert_eq!(ext.contours.len(), 1);
        let list = find_contours(&img, RetrievalMode::List, ContourApproximationMode::Simple)?;
        assert_eq!(list.contours.len(), 2);
        Ok(())
    }

    /// Hollow square with CComp: hole contour must have a valid parent index
    #[test]
    fn test_hollow_square_ccomp_hierarchy() -> Result<(), Box<dyn std::error::Error>> {
        let img = make_img(
            6,
            6,
            vec![
                0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1,
                1, 0, 0, 0, 0, 0, 0, 0,
            ],
        )?;
        let r = find_contours(&img, RetrievalMode::CComp, ContourApproximationMode::Simple)?;
        assert_eq!(r.contours.len(), 2, "CComp should return both contours");
        assert!(
            r.hierarchy.iter().any(|h| h[3] >= 0),
            "hole must have a parent"
        );
        Ok(())
    }

    /// Outer ring + hole + inner square: verifies all four retrieval modes simultaneously
    #[test]
    fn test_all_retrieval_modes_nested_image() -> Result<(), Box<dyn std::error::Error>> {
        let img = make_img(
            8,
            8,
            vec![
                0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1,
                1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
            ],
        )?;
        let ext = find_contours(
            &img,
            RetrievalMode::External,
            ContourApproximationMode::Simple,
        )?;
        let list = find_contours(&img, RetrievalMode::List, ContourApproximationMode::Simple)?;
        let ccomp = find_contours(&img, RetrievalMode::CComp, ContourApproximationMode::Simple)?;
        let tree = find_contours(&img, RetrievalMode::Tree, ContourApproximationMode::Simple)?;

        assert_eq!(ext.contours.len(), 1, "External: outermost only");
        assert_eq!(list.contours.len(), 3, "List: all 3");
        assert_eq!(ccomp.contours.len(), 3, "CComp: all 3");
        assert_eq!(tree.contours.len(), 3, "Tree: all 3");

        assert!(list.hierarchy.iter().all(|h| *h == [-1, -1, -1, -1]));

        assert_eq!(ccomp.hierarchy.len(), 3);
        assert_eq!(tree.hierarchy.len(), 3);
        assert!(ccomp.hierarchy.iter().any(|h| h[3] >= 0));
        Ok(())
    }

    /// All-zero image: no contours and empty hierarchy
    #[test]
    fn test_all_zeros_no_contours() -> Result<(), Box<dyn std::error::Error>> {
        let img = make_img(10, 10, vec![0u8; 100])?;
        let r = find_contours(&img, RetrievalMode::List, ContourApproximationMode::None)?;
        assert!(r.contours.is_empty());
        assert!(r.hierarchy.is_empty());
        Ok(())
    }

    /// 1x1 foreground image: 1 contour with the single point at (0,0)
    #[test]
    fn test_single_pixel_image() -> Result<(), Box<dyn std::error::Error>> {
        let img = make_img(1, 1, vec![1])?;
        let r = find_contours(
            &img,
            RetrievalMode::External,
            ContourApproximationMode::None,
        )?;
        assert_eq!(r.contours.len(), 1);
        assert_eq!(r.contours[0], vec![[0, 0]]);
        Ok(())
    }

    /// 48x48 block inside a 64x64 image: the all-1 SWAR skip must not miss the
    /// outer contour, Perimeter of a 48x48 block = 4 x 47 = 188 pixels
    #[test]
    fn test_large_filled_block_swar_all_ones_path() -> Result<(), Box<dyn std::error::Error>> {
        const S: usize = 64;
        let mut data = vec![0u8; S * S];
        for r in 8..56 {
            for c in 8..56 {
                data[r * S + c] = 1;
            }
        }
        let img = make_img(S, S, data)?;
        let r = find_contours(
            &img,
            RetrievalMode::External,
            ContourApproximationMode::None,
        )?;
        assert_eq!(r.contours.len(), 1, "exactly one outer contour");
        assert_eq!(r.contours[0].len(), 4 * 47, "perimeter of 48x48 block");
        Ok(())
    }

    /// 199-zero prefix then one foreground pixel: exercises the zero-skip SWAR path
    #[test]
    fn test_long_zero_run_swar_zero_skip_path() -> Result<(), Box<dyn std::error::Error>> {
        let mut data = vec![0u8; 400];
        data[200] = 1;
        let img = make_img(400, 1, data)?;
        let r = find_contours(
            &img,
            RetrievalMode::External,
            ContourApproximationMode::None,
        )?;
        assert_eq!(r.contours.len(), 1);
        assert_eq!(r.contours[0], vec![[200, 0]]);
        Ok(())
    }

    /// Executor run on two *different* images back-to-back must return correct
    /// results for both, catches buffer-reuse bugs where stale data from the
    /// first call pollutes the second.
    #[test]
    fn test_executor_different_images_back_to_back() -> Result<(), Box<dyn std::error::Error>> {
        let img_a = make_img(3, 3, vec![0, 0, 0, 0, 1, 0, 0, 0, 0])?; // single pixel
        let img_b = make_img(
            5,
            5,
            vec![
                0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
            ],
        )?; // 3x3 square
        let mut exec = FindContoursExecutor::new();
        let ra = exec.find_contours(
            &img_a,
            RetrievalMode::External,
            ContourApproximationMode::None,
        )?;
        let rb = exec.find_contours(
            &img_b,
            RetrievalMode::External,
            ContourApproximationMode::None,
        )?;
        assert_eq!(ra.contours.len(), 1);
        assert_eq!(ra.contours[0], vec![[1, 1]], "first call: single pixel");
        assert_eq!(rb.contours.len(), 1);
        assert_eq!(rb.contours[0].len(), 8, "second call: 3x3 square perimeter");
        Ok(())
    }

    /// Simple approx on an L-shaped contour: direction changes at the corner
    /// must be emitted, straight runs must be suppressed.
    /// L-shape (4 wide, 3 tall, bottom-right 2x2 missing):
    ///   . . . . . .
    ///   . 1 . . . .
    ///   . 1 . . . .
    ///   . 1 1 1 . .
    ///   . . . . . .
    #[test]
    fn test_simple_approx_l_shape() -> Result<(), Box<dyn std::error::Error>> {
        let img = make_img(
            5,
            4,
            vec![0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0],
        )?;
        let none = find_contours(
            &img,
            RetrievalMode::External,
            ContourApproximationMode::None,
        )?;
        let simp = find_contours(
            &img,
            RetrievalMode::External,
            ContourApproximationMode::Simple,
        )?;

        assert!(
            simp.contours[0].len() < none.contours[0].len(),
            "Simple should compress straight runs"
        );

        assert!(
            simp.contours[0].contains(&[1, 3]),
            "corner pixel missing from Simple contour"
        );
        Ok(())
    }

    /// Hierarchy link fields must be 0-based indices into the contours vec, not
    /// raw nbd labels. Verifies both Tree and CComp modes.
    #[test]
    fn test_hierarchy_indices_are_zero_based() -> Result<(), Box<dyn std::error::Error>> {
        // Hollow square: outer border is contour 0, hole is contour 1.
        let img = make_img(
            6,
            6,
            vec![
                0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1,
                1, 0, 0, 0, 0, 0, 0, 0,
            ],
        )?;
        let tree = find_contours(&img, RetrievalMode::Tree, ContourApproximationMode::Simple)?;
        assert_eq!(tree.contours.len(), 2);

        let hole = tree
            .hierarchy
            .iter()
            .find(|h| h[3] >= 0)
            .ok_or(ContoursError::NbdOverflow)?;
        assert_eq!(hole[3], 0, "Tree: parent must be 0-based contour index");

        let ccomp = find_contours(&img, RetrievalMode::CComp, ContourApproximationMode::Simple)?;
        let hole = ccomp
            .hierarchy
            .iter()
            .find(|h| h[3] >= 0)
            .ok_or(ContoursError::NbdOverflow)?;
        assert_eq!(hole[3], 0, "CComp: parent must be 0-based contour index");
        Ok(())
    }
}
