use rayon::prelude::*;

use crate::utils::Pixel;
use kornia_image::{Image};


/// A single horizontal run of non-Skip pixels.
#[derive(Clone, Copy)]
// parent at offset 0 so uf_find's hot load is immediate (no displacement).
// size as u16 (max real component = 26600 px under decimate=2; saturating_add is safe for decimate=1).
// #[repr(C)] locks this layout at 12B instead of 16B → 5 runs/cache-line vs 4.
#[repr(C)]
struct Run {
    parent: u32,
    col_start: u16,
    col_end: u16,
    size: u16,
    color: u8,
    _pad: u8,
}

/// Run-Length Encoding Connected Components.
///
/// Instead of a per-pixel UnionFind (854 KB of parent+size arrays, L3-bound),
/// we track RUNS (contiguous same-color horizontal segments). Typical frame has
/// ~4000 runs × 8 B = 32 KB → fits entirely in L1D cache. UF path traversals
/// become 5-cycle L1 hits rather than 50-cycle L3 hits.
///
/// The runs buffer is pre-allocated with worst-case capacity and shared directly
/// across Rayon threads using global run indices — eliminating the scatter phase
/// (one fewer Rayon spawn) and keeping Phase-1 root-resolution L1-warm on the
/// same core that wrote each thread's runs.
pub(crate) struct RleCC {
    /// Flat runs buffer. Layout: [thread 0 region | thread 1 region | ...].
    /// Each region has max_per_thread slots; only [base..base+count] are valid.
    runs: Vec<Run>,
    /// row_start[y] = global index of first run in row y; row_start[height] = sentinel.
    row_start: Vec<u32>,
    /// Per-run resolved root values. Sized to total_capacity; holes stay unread.
    run_vals: Vec<u32>,
    /// Actual run count per thread (set by scan_runs, consumed by fill_rep_cache).
    thread_counts: Vec<u32>,
    /// Slots allocated per thread in the runs buffer (stride).
    max_per_thread: usize,
    height: usize,
    width: usize,
}

impl RleCC {
    pub(crate) fn new(height: usize, width: usize) -> Self {
        Self {
            runs: Vec::new(),
            row_start: vec![0u32; height + 1],
            run_vals: Vec::new(),
            thread_counts: Vec::new(),
            max_per_thread: 0,
            height,
            width,
        }
    }

    /// Resize/reset for a new frame without freeing allocations.
    pub(crate) fn reset(&mut self, height: usize, width: usize) {
        self.height = height;
        self.width = width;
        if self.row_start.len() < height + 1 {
            self.row_start.resize(height + 1, 0);
        }
    }

    /// Run the full RLE-CC pipeline and write output into `rep_cache`.
    ///
    /// After this call, `rep_cache[i]` = canonical run-index for pixel i if its
    /// component has ≥ `min_size` pixels, or `u32::MAX` otherwise.
    pub(crate) fn process(
        &mut self,
        src: &Image<Pixel, 1>,
        rep_cache: &mut Vec<u32>,
        min_size: usize,
    ) {
        let height = src.height();
        let width = src.width();
        self.reset(height, width);

        let n_threads = rayon::current_num_threads().max(1);
        let strip_h = (height + n_threads - 1) / n_threads;

        self.scan_runs(src.as_slice(), height, width, n_threads, strip_h);

        for t in 1..n_threads {
            let y = t * strip_h;
            if y >= height { break; }
            let prev_start = self.row_start[y - 1] as usize;
            let a_end      = (t - 1) * self.max_per_thread + self.thread_counts[t - 1] as usize;
            let cur_start  = self.row_start[y] as usize;
            let next_start = self.row_start[y + 1] as usize;
            let runs_ptr = RunsPtr(self.runs.as_mut_ptr());
            merge_adjacent_rows(runs_ptr, prev_start, a_end, cur_start, next_start);
        }

        fill_rep_cache_parallel(
            &self.runs,
            &self.row_start,
            &mut self.run_vals,
            height,
            width,
            min_size,
            self.max_per_thread,
            &self.thread_counts,
            rep_cache,
        );
    }

    fn scan_runs(&mut self, src: &[Pixel], height: usize, width: usize, n_threads: usize, strip_h: usize) {
        // Worst-case runs per thread: one run every 2 pixels × strip height.
        let max_per_thread = ((width / 2 + 1) * strip_h).max(1);
        self.max_per_thread = max_per_thread;
        let total_capacity = max_per_thread * n_threads;

        // Grow the shared runs buffer to accommodate all threads' worst-case output.
        // Threads write at global offset (t * max_per_thread) with global run indices.
        // No scatter phase needed: runs are written in-place with correct global parents.
        if self.runs.len() < total_capacity {
            let extra = total_capacity - self.runs.len();
            self.runs.reserve(extra);
            // SAFETY: all positions [0, total_capacity) are written before any read:
            //   Phase A writes [base..base+count]; Phase B writes only within that range;
            //   fill_rep_cache reads only global indices stored in row_start.
            unsafe { self.runs.set_len(total_capacity); }
        }

        let runs_ptr = RunsPtr(self.runs.as_mut_ptr());
        let src_ptr = PixelPtr(src.as_ptr() as *const u8);

        // Each thread returns (local_row_start: Vec<u32>, actual_count: u32).
        // local_row_start[iy] is already a GLOBAL index (base + local_offset).
        let per_thread: Vec<(Vec<u32>, u32)> = (0..n_threads)
            .into_par_iter()
            .map(move |t| {
                let y_start = t * strip_h;
                let y_end = (y_start + strip_h).min(height);
                let strip_rows = y_end.saturating_sub(y_start);
                let base = (t * max_per_thread) as u32;

                let mut local_row_start = vec![base; strip_rows + 1];
                let mut run_idx = base;

                // Phase A: scan rows + write runs directly into the shared buffer.
                for iy in 0..strip_rows {
                    local_row_start[iy] = run_idx;
                    let y = y_start + iy;
                    let row_off = y * width;
                    let mut x = 1usize;
                    while x < width - 1 {
                        let pixel = unsafe { src_ptr.get(row_off + x) };
                        if pixel == Pixel::Skip { x += 1; continue; }
                        let col_start = x as u16;
                        x += 1;
                        // NEON: extend run 16 bytes at a time on aarch64.
                        #[cfg(target_arch = "aarch64")]
                        unsafe {
                            use std::arch::aarch64::*;
                            let pv = vdupq_n_u8(pixel as u8);
                            let base_ptr = src_ptr.0.add(row_off);
                            while x + 16 <= width - 1 {
                                let chunk = vld1q_u8(base_ptr.add(x));
                                if vminvq_u8(vceqq_u8(chunk, pv)) == 0xFF {
                                    x += 16;
                                } else {
                                    break;
                                }
                            }
                        }
                        // AVX2: extend run 32 bytes at a time on x86_64.
                        #[cfg(target_arch = "x86_64")]
                        if crate::ops::has_avx2() {
                            // SAFETY: AVX2 confirmed by runtime probe; reads bounded by width-1.
                            unsafe {
                                use std::arch::x86_64::*;
                                let pv = _mm256_set1_epi8(pixel as u8 as i8);
                                let base_ptr = src_ptr.0.add(row_off);
                                while x + 32 <= width - 1 {
                                    let chunk = _mm256_loadu_si256(base_ptr.add(x) as *const __m256i);
                                    if _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, pv)) == -1 {
                                        x += 32;
                                    } else {
                                        break;
                                    }
                                }
                            }
                        }
                        while x < width - 1 && unsafe { src_ptr.get(row_off + x) } == pixel {
                            x += 1;
                        }
                        let col_end = x as u16;
                        // Write directly to global slot; parent is the global run index
                        // (self-pointing root). No rebase required later.
                        unsafe {
                            *runs_ptr.add(run_idx as usize) = Run {
                                parent: run_idx,
                                col_start,
                                col_end,
                                size: col_end - col_start,
                                color: pixel as u8,
                                _pad: 0,
                            };
                        }
                        run_idx += 1;
                    }
                }
                local_row_start[strip_rows] = run_idx;

                // Phase B: intra-strip UF, using global indices into the shared buffer.
                for iy in 1..strip_rows {
                    let a_start = local_row_start[iy - 1] as usize;
                    let a_end   = local_row_start[iy] as usize;
                    let b_start = local_row_start[iy] as usize;
                    let b_end   = local_row_start[iy + 1] as usize;
                    merge_adjacent_rows(runs_ptr, a_start, a_end, b_start, b_end);
                }

                let count = run_idx - base;
                (local_row_start, count)
            })
            .collect();

        // Sequential: update row_start from the per-thread global row indices.
        self.thread_counts.clear();
        for (t, (local_row_start, count)) in per_thread.into_iter().enumerate() {
            let y_start = t * strip_h;
            let strip_rows = (y_start + strip_h).min(height).saturating_sub(y_start);
            for iy in 0..strip_rows {
                self.row_start[y_start + iy] = local_row_start[iy];
            }
            // Last thread sets the sentinel for row_start[height].
            if t == n_threads - 1 {
                self.row_start[height] = local_row_start[strip_rows];
            }
            self.thread_counts.push(count);
        }
    }
}

// Method-based pointer wrappers: closures must use the METHOD (not .0 field access)
// so Rust 2021 precise-capture sees the whole struct (which has Send+Sync), not the
// inner raw pointer (which does not).

/// Mutable runs array pointer, shareable across Rayon threads.
/// Each thread accesses only its disjoint row range.
#[derive(Copy, Clone)]
struct RunsPtr(*mut Run);
unsafe impl Send for RunsPtr {}
unsafe impl Sync for RunsPtr {}
impl RunsPtr {
    #[inline(always)] fn add(self, n: usize) -> *mut Run { unsafe { self.0.add(n) } }
}

/// Read-only runs array pointer, shareable across Rayon threads.
#[derive(Copy, Clone)]
struct RunsConstPtr(*const Run);
unsafe impl Send for RunsConstPtr {}
unsafe impl Sync for RunsConstPtr {}
impl RunsConstPtr {
    #[inline(always)] fn add(self, n: usize) -> *const Run { unsafe { self.0.add(n) } }
    #[inline(always)] fn as_ptr(self) -> *const Run { self.0 }
}

/// Read-only u32 slice pointer, shareable across Rayon threads.
#[derive(Copy, Clone)]
struct U32Ptr(*const u32);
unsafe impl Send for U32Ptr {}
unsafe impl Sync for U32Ptr {}
impl U32Ptr {
    #[inline(always)] fn add(self, n: usize) -> *const u32 { unsafe { self.0.add(n) } }
}

/// Read-only Pixel (u8) slice pointer, shareable across Rayon threads.
#[derive(Copy, Clone)]
struct PixelPtr(*const u8);
unsafe impl Send for PixelPtr {}
unsafe impl Sync for PixelPtr {}
impl PixelPtr {
    #[inline(always)] unsafe fn get(self, n: usize) -> Pixel {
        // SAFETY: caller ensures n is in-bounds.
        core::mem::transmute(*self.0.add(n))
    }
}

/// Mutable u32 slice pointer (cache output), shareable across Rayon threads.
/// Each thread writes to a disjoint strip.
#[derive(Copy, Clone)]
struct U32MutPtr(*mut u32);
unsafe impl Send for U32MutPtr {}
unsafe impl Sync for U32MutPtr {}
impl U32MutPtr {
    #[inline(always)] fn add(self, n: usize) -> *mut u32 { unsafe { self.0.add(n) } }
}

/// Merge two adjacent rows' run lists: connect overlapping same-color run pairs.
/// `a_start..a_end` = runs for the upper row; `b_start..b_end` = lower row.
#[inline]
fn merge_adjacent_rows(
    runs_ptr: RunsPtr,
    a_start: usize,
    a_end: usize,
    b_start: usize,
    b_end: usize,
) {
    if a_start == a_end || b_start == b_end {
        return;
    }
    let ptr = runs_ptr.0;
    let mut a = a_start;
    let mut b = b_start;
    while a < a_end && b < b_end {
        let ra = unsafe { &*ptr.add(a) };
        let rb = unsafe { &*ptr.add(b) };
        if ra.color == rb.color && ra.col_start < rb.col_end && rb.col_start < ra.col_end {
            uf_union(runs_ptr, a, b);
        }
        if ra.col_end <= rb.col_end { a += 1; } else { b += 1; }
    }
}

/// Path-halving find with lazy initialisation.
fn uf_find(runs_ptr: RunsPtr, mut id: usize) -> usize {
    let ptr = runs_ptr.0;
    loop {
        let p = unsafe { (*ptr.add(id)).parent as usize };
        if p == id { return id; }
        let pp = unsafe { (*ptr.add(p)).parent as usize };
        unsafe { (*ptr.add(id)).parent = pp as u32; }
        id = pp;
    }
}

/// Union by size.
fn uf_union(runs_ptr: RunsPtr, a: usize, b: usize) {
    let ra = uf_find(runs_ptr, a);
    let rb = uf_find(runs_ptr, b);
    if ra == rb { return; }
    let ptr = runs_ptr.0;
    let sa = unsafe { (*ptr.add(ra)).size };
    let sb = unsafe { (*ptr.add(rb)).size };
    if sa >= sb {
        unsafe { (*ptr.add(rb)).parent = ra as u32; }
        unsafe { (*ptr.add(ra)).size = sa.saturating_add(sb); }
    } else {
        unsafe { (*ptr.add(ra)).parent = rb as u32; }
        unsafe { (*ptr.add(rb)).size = sb.saturating_add(sa); }
    }
}

/// Two-phase parallel fill:
///
/// Phase 1 (parallel) — uf_find_const once per RUN per thread chunk.
///   Each Rayon worker resolves the same runs it wrote during scan_runs
///   → data is L1-warm on the same core, no L2/L3 miss for the UF traversal.
///
/// Phase 2 (parallel) — fill rep_cache for each thread's row strip.
///   Each worker fills only the rows whose runs it resolved in Phase 1.
fn fill_rep_cache_parallel(
    runs: &[Run],
    row_start: &[u32],
    run_vals: &mut Vec<u32>,
    height: usize,
    width: usize,
    min_size: usize,
    max_per_thread: usize,
    thread_counts: &[u32],
    rep_cache: &mut Vec<u32>,
) {
    if rep_cache.len() < height * width {
        rep_cache.resize(height * width, u32::MAX);
    }

    let n_threads = thread_counts.len().max(1);
    let total_actual: u32 = thread_counts.iter().sum();
    if total_actual == 0 {
        rep_cache[..height * width].fill(u32::MAX);
        return;
    }

    // run_vals is indexed by GLOBAL run index; holes are never read.
    // Reuse the Vec<u32> stored on RleCC to avoid per-frame heap allocation.
    let total_capacity = max_per_thread * n_threads;
    if run_vals.len() < total_capacity {
        run_vals.resize(total_capacity, 0u32);
    }

    // Fused Phase 1+2: each Rayon task resolves roots then fills rep_cache for its strip.
    // Phase 2 reads run_vals[r] only for r in [base_t, base_t+count_t) — same range Phase 1
    // just wrote → no cross-thread run_vals dependency; one fewer par_iter spawn (~50µs).
    let strip_h = (height + n_threads - 1) / n_threads;
    let runs_ptr = RunsConstPtr(runs.as_ptr());
    let run_vals_ptr = U32MutPtr(run_vals.as_mut_ptr());
    let row_start_ptr = U32Ptr(row_start.as_ptr());
    let cache_ptr = U32MutPtr(rep_cache.as_mut_ptr());
    let thread_counts_ptr = U32Ptr(thread_counts.as_ptr());
    (0..n_threads).into_par_iter().for_each(move |t| {
        let base = t * max_per_thread;
        let count = unsafe { *thread_counts_ptr.add(t) } as usize;
        // Phase 1: resolve root for each run this thread owns.
        for r in base..base + count {
            let root = uf_find_const(runs_ptr.as_ptr(), r);
            let root_size = unsafe { (*runs_ptr.add(root)).size } as usize;
            let val = if root_size >= min_size { root as u32 } else { u32::MAX };
            unsafe { *run_vals_ptr.add(r) = val; }
        }
        // Phase 2: fill rep_cache for this thread's rows.
        // Strategy: pre-fill entire row with u32::MAX (one streaming store per row),
        // then overwrite only valid-component run pixels. Eliminates per-gap fills and
        // fill_x tracking; trades 2× total writes for 17× fewer fill() call sites.
        let y_start = t * strip_h;
        let y_end = (y_start + strip_h).min(height);
        let thread_end = base + count;
        for y in y_start..y_end {
            // Pre-fill whole row with sentinel (borders, gaps, and below-size runs).
            let row_off = y * width;
            let row_slice = unsafe {
                std::slice::from_raw_parts_mut(cache_ptr.add(row_off), width)
            };
            row_slice.fill(u32::MAX);
            // Overwrite only valid (above-min_size) run pixels with the component id.
            let r_start = unsafe { *row_start_ptr.add(y) } as usize;
            let r_end = if y + 1 == y_end {
                thread_end
            } else {
                (unsafe { *row_start_ptr.add(y + 1) }) as usize
            };
            for r in r_start..r_end {
                let val = unsafe { *run_vals_ptr.add(r) };
                if val == u32::MAX { continue; }
                let run = unsafe { &*runs_ptr.add(r) };
                let col_s = run.col_start as usize;
                let col_e = run.col_end as usize;
                unsafe {
                    std::slice::from_raw_parts_mut(cache_ptr.add(row_off + col_s), col_e - col_s)
                }.fill(val);
            }
        }
    });
}

/// Read-only path traversal (no path compression).
#[inline]
fn uf_find_const(runs_ptr: *const Run, mut id: usize) -> usize {
    loop {
        let p = unsafe { (*runs_ptr.add(id)).parent as usize };
        if p == id { return id; }
        id = p;
    }
}

