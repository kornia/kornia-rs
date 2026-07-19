//! Median blur — byte-for-byte with `cv2.medianBlur` AND VPI's
//! `MedianFilter` (verified empirically: the two agree bit-for-bit for 3×3
//! and 5×5 u8 with replicate borders), so one implementation matches both.
//!
//! The median is an exact order statistic — no rounding, no float — so
//! parity only requires the same border rule (`BORDER_REPLICATE`, which is
//! what `cv2.medianBlur` hardwires) and an exact median. Sorting networks
//! (min/max only) give that on CPU and GPU alike; the CUDA kernels in
//! `cuda/median.rs` are textual twins of the networks below.

use kornia_image::{Image, ImageError};
use rayon::prelude::*;

/// Compare-exchange: after the call `*a <= *b`.
#[inline(always)]
fn ce(v: &mut [u8], a: usize, b: usize) {
    let (x, y) = (v[a], v[b]);
    v[a] = x.min(y);
    v[b] = x.max(y);
}

/// Paeth's 19-exchange median-of-9 network. The CUDA kernel source is
/// GENERATED from this same list (`cuda/median.rs`), so the sides cannot
/// drift.
pub(crate) const NET9: [(usize, usize); 19] = [
    (1, 2),
    (4, 5),
    (7, 8),
    (0, 1),
    (3, 4),
    (6, 7),
    (1, 2),
    (4, 5),
    (7, 8),
    (0, 3),
    (5, 8),
    (4, 7),
    (3, 6),
    (1, 4),
    (2, 5),
    (4, 7),
    (4, 2),
    (6, 4),
    (4, 2),
];

/// Exact median of 9 via [`NET9`].
#[inline(always)]
pub(crate) fn median9(v: &mut [u8; 9]) -> u8 {
    for &(a, b) in NET9.iter() {
        ce(v, a, b);
    }
    v[4]
}

/// Smith's classic 99-exchange median-of-25 network — also the generator
/// for the CUDA 5×5 kernel.
pub(crate) const NET25: [(usize, usize); 99] = [
    (0, 1),
    (3, 4),
    (2, 4),
    (2, 3),
    (6, 7),
    (5, 7),
    (5, 6),
    (9, 10),
    (8, 10),
    (8, 9),
    (12, 13),
    (11, 13),
    (11, 12),
    (15, 16),
    (14, 16),
    (14, 15),
    (18, 19),
    (17, 19),
    (17, 18),
    (21, 22),
    (20, 22),
    (20, 21),
    (23, 24),
    (2, 5),
    (3, 6),
    (0, 6),
    (0, 3),
    (4, 7),
    (1, 7),
    (1, 4),
    (11, 14),
    (8, 14),
    (8, 11),
    (12, 15),
    (9, 15),
    (9, 12),
    (13, 16),
    (10, 16),
    (10, 13),
    (20, 23),
    (17, 23),
    (17, 20),
    (21, 24),
    (18, 24),
    (18, 21),
    (19, 22),
    (8, 17),
    (9, 18),
    (0, 18),
    (0, 9),
    (10, 19),
    (1, 19),
    (1, 10),
    (11, 20),
    (2, 20),
    (2, 11),
    (12, 21),
    (3, 21),
    (3, 12),
    (13, 22),
    (4, 22),
    (4, 13),
    (14, 23),
    (5, 23),
    (5, 14),
    (15, 24),
    (6, 24),
    (6, 15),
    (7, 16),
    (7, 19),
    (13, 21),
    (15, 23),
    (7, 13),
    (7, 15),
    (1, 9),
    (3, 11),
    (5, 17),
    (11, 17),
    (9, 17),
    (4, 10),
    (6, 12),
    (7, 14),
    (4, 6),
    (4, 7),
    (12, 14),
    (10, 14),
    (6, 7),
    (10, 12),
    (6, 10),
    (6, 17),
    (12, 17),
    (7, 17),
    (7, 10),
    (12, 18),
    (7, 12),
    (10, 18),
    (12, 20),
    (10, 20),
    (10, 12),
];

/// Exact median of 25 via [`NET25`].
#[inline(always)]
pub(crate) fn median25(v: &mut [u8; 25]) -> u8 {
    for &(a, b) in NET25.iter() {
        ce(v, a, b);
    }
    v[12]
}

/// Median blur for 8-bit images — byte-for-byte with
/// `cv2.medianBlur(src, ksize)` and VPI's CUDA `MedianFilter` (the two are
/// themselves bit-identical for these kernel sizes). `ksize` must be 3
/// or 5; borders replicate. Device pairs run the CUDA sorting-network
/// kernels, byte-identical to the CPU path.
pub fn median_blur<const C: usize>(
    src: &Image<u8, C>,
    dst: &mut Image<u8, C>,
    ksize: usize,
) -> Result<(), ImageError> {
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }
    if ksize != 3 && ksize != 5 {
        return Err(ImageError::InvalidKernelLength(ksize, ksize));
    }

    #[cfg(feature = "cuda")]
    {
        use crate::try_device;
        try_device!(src, dst, |stream| cuda_adapters::median_blur_cuda(
            src, dst, ksize, stream
        ));
    }

    let (w, h) = (src.cols(), src.rows());
    let s = src.as_slice();
    let r = (ksize / 2) as i64;

    // C1 on aarch64: the NEON module owns its parallelization and tiling
    // (two-rows-per-pass 3×3, sorted-column 5×5, per-op-size chunking).
    #[cfg(target_arch = "aarch64")]
    if C == 1 && neon::par_c1(s, dst.as_slice_mut(), w, h, ksize) {
        return Ok(());
    }

    let min_rows = (h.div_ceil(rayon::current_num_threads() * 4)).max(16);
    dst.as_slice_mut()
        .par_chunks_mut(w * C)
        .enumerate()
        .with_min_len(min_rows)
        .for_each(|(y, drow)| {
            // Portable path: one gather loop for both window sizes
            // (25-slot buffer, first ksize² slots used); the row clamps
            // depend only on y and are hoisted out of the pixel loop.
            let mut sy = [0usize; 5];
            for (i, dy) in (-r..=r).enumerate() {
                sy[i] = (y as i64 + dy).clamp(0, h as i64 - 1) as usize;
            }
            for x in 0..w {
                let mut sx = [0usize; 5];
                for (i, dx) in (-r..=r).enumerate() {
                    sx[i] = (x as i64 + dx).clamp(0, w as i64 - 1) as usize;
                }
                for c in 0..C {
                    let mut win = [0u8; 25];
                    let mut n = 0;
                    for row in sy.iter().take(ksize) {
                        for col in sx.iter().take(ksize) {
                            win[n] = s[(row * w + col) * C + c];
                            n += 1;
                        }
                    }
                    drow[x * C + c] = if ksize == 3 {
                        let mut w9: [u8; 9] = win[..9].try_into().unwrap();
                        median9(&mut w9)
                    } else {
                        median25(&mut win)
                    };
                }
            }
        });
    Ok(())
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use rayon::prelude::*;
    use std::arch::aarch64::*;

    /// Whole-image C1 entry: owns tiling and rayon chunking (the
    /// parallelization heuristics live next to the kernels they describe).
    /// Returns false for shapes the vector kernels don't cover.
    pub(super) fn par_c1(s: &[u8], dst: &mut [u8], w: usize, h: usize, ksize: usize) -> bool {
        if ksize == 3 {
            if w < 34 {
                return false;
            }
            if h >= 2 {
                // Two output rows per task share their 4 source-row loads
                // (memory-bound); static per-core split — at ~0.3 ms total,
                // rayon per-task overhead dominates smaller chunks.
                let pairs = h.div_ceil(2);
                let min_pairs = pairs.div_ceil(rayon::current_num_threads()).max(1);
                dst.par_chunks_mut(w * 2)
                    .enumerate()
                    .with_min_len(min_pairs)
                    .for_each(|(i, dchunk)| {
                        let y = i * 2;
                        if dchunk.len() == w * 2 && median3_row2(s, dchunk, w, h, y) {
                            return;
                        }
                        for (j, drow) in dchunk.chunks_mut(w).enumerate() {
                            let ok = median3_row(s, drow, w, h, y + j);
                            debug_assert!(ok, "w >= 34 guaranteed");
                        }
                    });
            } else {
                let ok = median3_row(s, dst, w, h, 0);
                debug_assert!(ok);
            }
            true
        } else {
            if w < 48 {
                return false;
            }
            // Quarter-per-core chunks for the ~2 ms 5×5: stealing stays
            // available without per-task overhead dominating.
            let min_rows = h.div_ceil(rayon::current_num_threads() * 4).max(16);
            dst.par_chunks_mut(w)
                .enumerate()
                .with_min_len(min_rows)
                .for_each(|(y, drow)| {
                    let ok = median5_row(s, drow, w, h, y);
                    debug_assert!(ok, "w >= 48 guaranteed");
                });
            true
        }
    }

    /// Sorted-column triple (lo, mid, hi) — the working type of the 3×3
    /// kernels.
    pub(super) type S = (uint8x16_t, uint8x16_t, uint8x16_t);

    /// `med9 = med3(max(col lows), med3(col mids), min(col highs))` on the
    /// left/mid/right sorted-column triples.
    #[inline(always)]
    unsafe fn fold(l: S, m: S, r: S) -> uint8x16_t {
        med3(
            max3(l.0, m.0, r.0),
            med3(l.1, m.1, r.1),
            min3(l.2, m.2, r.2),
        )
    }
    /// Shift a sorted-column triple one column left (prev block's lane 15
    /// enters at lane 0).
    #[inline(always)]
    unsafe fn shl(p: &S, c: &S) -> S {
        (
            vextq_u8(p.0, c.0, 15),
            vextq_u8(p.1, c.1, 15),
            vextq_u8(p.2, c.2, 15),
        )
    }
    /// Shift a sorted-column triple one column right (next block's lane 0
    /// enters at lane 15).
    #[inline(always)]
    unsafe fn shr(c: &S, n: &S) -> S {
        (
            vextq_u8(c.0, n.0, 1),
            vextq_u8(c.1, n.1, 1),
            vextq_u8(c.2, n.2, 1),
        )
    }
    /// Synthetic replicate columns for the image edges.
    #[inline(always)]
    unsafe fn dup0(c: &S) -> S {
        (
            vdupq_laneq_u8(c.0, 0),
            vdupq_laneq_u8(c.1, 0),
            vdupq_laneq_u8(c.2, 0),
        )
    }
    #[inline(always)]
    unsafe fn dup15(c: &S) -> S {
        (
            vdupq_laneq_u8(c.0, 15),
            vdupq_laneq_u8(c.1, 15),
            vdupq_laneq_u8(c.2, 15),
        )
    }

    #[inline(always)]
    unsafe fn min3(a: uint8x16_t, b: uint8x16_t, c: uint8x16_t) -> uint8x16_t {
        vminq_u8(vminq_u8(a, b), c)
    }
    #[inline(always)]
    unsafe fn max3(a: uint8x16_t, b: uint8x16_t, c: uint8x16_t) -> uint8x16_t {
        vmaxq_u8(vmaxq_u8(a, b), c)
    }
    /// Exact median-of-3, lane-wise.
    #[inline(always)]
    unsafe fn med3(a: uint8x16_t, b: uint8x16_t, c: uint8x16_t) -> uint8x16_t {
        vmaxq_u8(vminq_u8(a, b), vminq_u8(vmaxq_u8(a, b), c))
    }

    /// One output row, C1, 3×3: the classic exact identity
    /// `med9 = med3(max(col_lows), med3(col_mids), min(col_highs))` on 16
    /// lanes with rolling `vext`-shared column sorts and synthetic
    /// replicate border columns (see [`median3_row2`], the two-row variant
    /// used on the hot path; this single-row form serves odd final rows).
    /// Exact median — byte-identical to the network walk.
    pub(super) fn median3_row(s: &[u8], drow: &mut [u8], w: usize, h: usize, y: usize) -> bool {
        if w < 34 {
            return false;
        }
        // SAFETY: aligned blocks read [x, x+16) with x+16 <= w; unaligned
        // tail blocks read within [w-17, w); rows are clamp-mapped.
        unsafe {
            let r0 = s.as_ptr().add(y.saturating_sub(1).min(h - 1) * w);
            let r1 = s.as_ptr().add(y * w);
            let r2 = s.as_ptr().add((y + 1).min(h - 1) * w);
            let sort3v = |off: usize| -> S {
                let a = vld1q_u8(r0.add(off));
                let b = vld1q_u8(r1.add(off));
                let c = vld1q_u8(r2.add(off));
                (min3(a, b, c), med3(a, b, c), max3(a, b, c))
            };
            let mut cur = sort3v(0);
            let mut prev = dup0(&cur);
            let mut x = 0usize;
            while x + 32 <= w {
                let next = sort3v(x + 16);
                vst1q_u8(
                    drow.as_mut_ptr().add(x),
                    fold(shl(&prev, &cur), cur, shr(&cur, &next)),
                );
                prev = cur;
                cur = next;
                x += 16;
            }
            let xt = w - 16;
            let c = sort3v(xt);
            let l = sort3v(xt - 1);
            vst1q_u8(drow.as_mut_ptr().add(xt), fold(l, c, shr(&c, &dup15(&c))));
            let mut xg = x;
            while xg < xt {
                let c = sort3v(xg);
                let l = sort3v(xg - 1);
                let n = sort3v(xg + 1);
                vst1q_u8(drow.as_mut_ptr().add(xg), fold(l, c, n));
                xg += 16;
            }
        }
        true
    }

    /// TWO output rows, C1, 3×3: same identity as [`median3_row`] but rows
    /// y and y+1 share their 4 source-row loads (the op is memory-bound —
    /// 4 loads per 2 outputs instead of 6). Byte-identical.
    pub(super) fn median3_row2(s: &[u8], drow2: &mut [u8], w: usize, h: usize, y: usize) -> bool {
        if w < 34 || y + 1 >= h {
            return false;
        }
        // SAFETY: same bounds as median3_row; rows clamp-mapped.
        unsafe {
            let r0 = s.as_ptr().add(y.saturating_sub(1).min(h - 1) * w);
            let r1 = s.as_ptr().add(y * w);
            let r2 = s.as_ptr().add((y + 1).min(h - 1) * w);
            let r3 = s.as_ptr().add((y + 2).min(h - 1) * w);
            let (d0, d1) = drow2.split_at_mut(w);
            let sort2 = |off: usize| -> (S, S) {
                let a = vld1q_u8(r0.add(off));
                let b = vld1q_u8(r1.add(off));
                let c = vld1q_u8(r2.add(off));
                let d = vld1q_u8(r3.add(off));
                (
                    (min3(a, b, c), med3(a, b, c), max3(a, b, c)),
                    (min3(b, c, d), med3(b, c, d), max3(b, c, d)),
                )
            };
            let (mut cur_a, mut cur_b) = sort2(0);
            let mut prev_a = dup0(&cur_a);
            let mut prev_b = dup0(&cur_b);
            let mut x = 0usize;
            // 32 columns per iteration; the two 16-wide results per row go
            // out through one non-temporal pair store (stnp) — the output
            // is write-once, so pulling its cache lines in (RFO) is pure
            // wasted DRAM traffic on this memory-bound op.
            while x + 48 <= w {
                let (mid_a, mid_b) = sort2(x + 16);
                let (next_a, next_b) = sort2(x + 32);
                let m0a = fold(shl(&prev_a, &cur_a), cur_a, shr(&cur_a, &mid_a));
                let m1a = fold(shl(&cur_a, &mid_a), mid_a, shr(&mid_a, &next_a));
                let m0b = fold(shl(&prev_b, &cur_b), cur_b, shr(&cur_b, &mid_b));
                let m1b = fold(shl(&cur_b, &mid_b), mid_b, shr(&mid_b, &next_b));
                std::arch::asm!(
                    "stnp {0:q}, {1:q}, [{2}]",
                    in(vreg) m0a, in(vreg) m1a, in(reg) d0.as_mut_ptr().add(x),
                    options(nostack, preserves_flags)
                );
                std::arch::asm!(
                    "stnp {0:q}, {1:q}, [{2}]",
                    in(vreg) m0b, in(vreg) m1b, in(reg) d1.as_mut_ptr().add(x),
                    options(nostack, preserves_flags)
                );
                prev_a = mid_a;
                prev_b = mid_b;
                cur_a = next_a;
                cur_b = next_b;
                x += 32;
            }
            while x + 32 <= w {
                let (next_a, next_b) = sort2(x + 16);
                vst1q_u8(
                    d0.as_mut_ptr().add(x),
                    fold(shl(&prev_a, &cur_a), cur_a, shr(&cur_a, &next_a)),
                );
                vst1q_u8(
                    d1.as_mut_ptr().add(x),
                    fold(shl(&prev_b, &cur_b), cur_b, shr(&cur_b, &next_b)),
                );
                prev_a = cur_a;
                prev_b = cur_b;
                cur_a = next_a;
                cur_b = next_b;
                x += 16;
            }
            let xt = w - 16;
            let (c_a, c_b) = sort2(xt);
            let (l_a, l_b) = sort2(xt - 1);
            vst1q_u8(
                d0.as_mut_ptr().add(xt),
                fold(l_a, c_a, shr(&c_a, &dup15(&c_a))),
            );
            vst1q_u8(
                d1.as_mut_ptr().add(xt),
                fold(l_b, c_b, shr(&c_b, &dup15(&c_b))),
            );
            let mut xg = x;
            while xg < xt {
                let (c_a, c_b) = sort2(xg);
                let (l_a, l_b) = sort2(xg - 1);
                let (n_a, n_b) = sort2(xg + 1);
                vst1q_u8(d0.as_mut_ptr().add(xg), fold(l_a, c_a, n_a));
                vst1q_u8(d1.as_mut_ptr().add(xg), fold(l_b, c_b, n_b));
                xg += 16;
            }
        }
        true
    }

    /// One output row, C1, 5×5: rolling SORTED-COLUMN scheme. Each aligned
    /// 16-column block sorts its 5-element columns once (9 exchanges); the
    /// ±1/±2-shifted sorted columns come from `vext` against neighbor
    /// blocks. The remaining median-of-25 selection uses a 71-exchange
    /// network derived from Smith's 99 by greedy deletion, PROVEN exact on
    /// the sorted-column input subspace via the zero-one principle (all
    /// 6^5 column-sorted 0-1 patterns verified — see the unit test).
    /// Borders synthesize replicate columns by broadcasting edge lanes.
    /// Exact median — byte-identical to the scalar network walk.
    pub(super) fn median5_row(s: &[u8], drow: &mut [u8], w: usize, h: usize, y: usize) -> bool {
        if w < 48 {
            return false;
        }
        // SAFETY: aligned blocks read [x, x+16) with x+16 <= w; unaligned
        // tail blocks read within [w-18, w); rows are clamp-mapped.
        unsafe {
            let mut rows: [*const u8; 5] = [std::ptr::null(); 5];
            for (dy, row) in rows.iter_mut().enumerate() {
                let sy = (y + dy).saturating_sub(2).min(h - 1);
                *row = s.as_ptr().add(sy * w);
            }
            let sort_cols = |off: usize| -> [uint8x16_t; 5] {
                let mut c = [
                    vld1q_u8(rows[0].add(off)),
                    vld1q_u8(rows[1].add(off)),
                    vld1q_u8(rows[2].add(off)),
                    vld1q_u8(rows[3].add(off)),
                    vld1q_u8(rows[4].add(off)),
                ];
                let (lo, hi) = (vminq_u8(c[0], c[1]), vmaxq_u8(c[0], c[1]));
                c[0] = lo;
                c[1] = hi;
                let (lo, hi) = (vminq_u8(c[3], c[4]), vmaxq_u8(c[3], c[4]));
                c[3] = lo;
                c[4] = hi;
                let (lo, hi) = (vminq_u8(c[2], c[4]), vmaxq_u8(c[2], c[4]));
                c[2] = lo;
                c[4] = hi;
                let (lo, hi) = (vminq_u8(c[2], c[3]), vmaxq_u8(c[2], c[3]));
                c[2] = lo;
                c[3] = hi;
                let (lo, hi) = (vminq_u8(c[1], c[4]), vmaxq_u8(c[1], c[4]));
                c[1] = lo;
                c[4] = hi;
                let (lo, hi) = (vminq_u8(c[0], c[3]), vmaxq_u8(c[0], c[3]));
                c[0] = lo;
                c[3] = hi;
                let (lo, hi) = (vminq_u8(c[0], c[2]), vmaxq_u8(c[0], c[2]));
                c[0] = lo;
                c[2] = hi;
                let (lo, hi) = (vminq_u8(c[1], c[3]), vmaxq_u8(c[1], c[3]));
                c[1] = lo;
                c[3] = hi;
                let (lo, hi) = (vminq_u8(c[1], c[2]), vmaxq_u8(c[1], c[2]));
                c[1] = lo;
                c[2] = hi;
                c
            };
            let dup_lane0 = |v: &[uint8x16_t; 5]| -> [uint8x16_t; 5] {
                [
                    vdupq_laneq_u8(v[0], 0),
                    vdupq_laneq_u8(v[1], 0),
                    vdupq_laneq_u8(v[2], 0),
                    vdupq_laneq_u8(v[3], 0),
                    vdupq_laneq_u8(v[4], 0),
                ]
            };
            let dup_lane15 = |v: &[uint8x16_t; 5]| -> [uint8x16_t; 5] {
                [
                    vdupq_laneq_u8(v[0], 15),
                    vdupq_laneq_u8(v[1], 15),
                    vdupq_laneq_u8(v[2], 15),
                    vdupq_laneq_u8(v[3], 15),
                    vdupq_laneq_u8(v[4], 15),
                ]
            };
            let run_block = |prev: &[uint8x16_t; 5],
                             cur: &[uint8x16_t; 5],
                             next: &[uint8x16_t; 5]|
             -> uint8x16_t {
                let mut v0 = vextq_u8(prev[0], cur[0], 14);
                let mut v1 = vextq_u8(prev[1], cur[1], 14);
                let mut v2 = vextq_u8(prev[2], cur[2], 14);
                let mut v3 = vextq_u8(prev[3], cur[3], 14);
                let mut v4 = vextq_u8(prev[4], cur[4], 14);
                let mut v5 = vextq_u8(prev[0], cur[0], 15);
                let mut v6 = vextq_u8(prev[1], cur[1], 15);
                let mut v7 = vextq_u8(prev[2], cur[2], 15);
                let mut v8 = vextq_u8(prev[3], cur[3], 15);
                let mut v9 = vextq_u8(prev[4], cur[4], 15);
                let mut v10 = cur[0];
                let mut v11 = cur[1];
                let mut v12 = cur[2];
                let mut v13 = cur[3];
                let mut v14 = cur[4];
                let mut v15 = vextq_u8(cur[0], next[0], 1);
                let mut v16 = vextq_u8(cur[1], next[1], 1);
                let mut v17 = vextq_u8(cur[2], next[2], 1);
                let mut v18 = vextq_u8(cur[3], next[3], 1);
                let mut v19 = vextq_u8(cur[4], next[4], 1);
                let mut v20 = vextq_u8(cur[0], next[0], 2);
                let mut v21 = vextq_u8(cur[1], next[1], 2);
                let mut v22 = vextq_u8(cur[2], next[2], 2);
                let mut v23 = vextq_u8(cur[3], next[3], 2);
                let mut v24 = vextq_u8(cur[4], next[4], 2);
                let (lo, hi) = (vminq_u8(v9, v10), vmaxq_u8(v9, v10));
                v9 = lo;
                v10 = hi;
                let (lo, hi) = (vminq_u8(v8, v9), vmaxq_u8(v8, v9));
                v8 = lo;
                v9 = hi;
                let (lo, hi) = (vminq_u8(v14, v16), vmaxq_u8(v14, v16));
                v14 = lo;
                v16 = hi;
                let (lo, hi) = (vminq_u8(v14, v15), vmaxq_u8(v14, v15));
                v14 = lo;
                v15 = hi;
                let (lo, hi) = (vminq_u8(v2, v5), vmaxq_u8(v2, v5));
                v2 = lo;
                v5 = hi;
                let (lo, hi) = (vminq_u8(v3, v6), vmaxq_u8(v3, v6));
                v3 = lo;
                v6 = hi;
                let (lo, hi) = (vminq_u8(v0, v3), vmaxq_u8(v0, v3));
                v0 = lo;
                v3 = hi;
                let (lo, hi) = (vminq_u8(v4, v7), vmaxq_u8(v4, v7));
                v4 = lo;
                v7 = hi;
                let (lo, hi) = (vminq_u8(v1, v4), vmaxq_u8(v1, v4));
                v1 = lo;
                v4 = hi;
                let (lo, hi) = (vminq_u8(v11, v14), vmaxq_u8(v11, v14));
                v11 = lo;
                v14 = hi;
                let (lo, hi) = (vminq_u8(v8, v11), vmaxq_u8(v8, v11));
                v8 = lo;
                v11 = hi;
                let (lo, hi) = (vminq_u8(v12, v15), vmaxq_u8(v12, v15));
                v12 = lo;
                v15 = hi;
                let (lo, hi) = (vminq_u8(v9, v15), vmaxq_u8(v9, v15));
                v9 = lo;
                v15 = hi;
                let (lo, hi) = (vminq_u8(v9, v12), vmaxq_u8(v9, v12));
                v9 = lo;
                v12 = hi;
                let (lo, hi) = (vminq_u8(v10, v16), vmaxq_u8(v10, v16));
                v10 = lo;
                v16 = hi;
                let (lo, hi) = (vminq_u8(v10, v13), vmaxq_u8(v10, v13));
                v10 = lo;
                v13 = hi;
                let (lo, hi) = (vminq_u8(v17, v23), vmaxq_u8(v17, v23));
                v17 = lo;
                v23 = hi;
                let (lo, hi) = (vminq_u8(v17, v20), vmaxq_u8(v17, v20));
                v17 = lo;
                v20 = hi;
                let (lo, hi) = (vminq_u8(v18, v24), vmaxq_u8(v18, v24));
                v18 = lo;
                v24 = hi;
                let (lo, hi) = (vminq_u8(v18, v21), vmaxq_u8(v18, v21));
                v18 = lo;
                v21 = hi;
                let (lo, hi) = (vminq_u8(v19, v22), vmaxq_u8(v19, v22));
                v19 = lo;
                v22 = hi;
                let (lo, hi) = (vminq_u8(v8, v17), vmaxq_u8(v8, v17));
                v8 = lo;
                v17 = hi;
                let (lo, hi) = (vminq_u8(v9, v18), vmaxq_u8(v9, v18));
                v9 = lo;
                v18 = hi;
                let (lo, hi) = (vminq_u8(v0, v9), vmaxq_u8(v0, v9));
                v0 = lo;
                v9 = hi;
                let (lo, hi) = (vminq_u8(v10, v19), vmaxq_u8(v10, v19));
                v10 = lo;
                v19 = hi;
                let (lo, hi) = (vminq_u8(v1, v19), vmaxq_u8(v1, v19));
                v1 = lo;
                v19 = hi;
                let (lo, hi) = (vminq_u8(v1, v10), vmaxq_u8(v1, v10));
                v1 = lo;
                v10 = hi;
                let (lo, hi) = (vminq_u8(v11, v20), vmaxq_u8(v11, v20));
                v11 = lo;
                v20 = hi;
                let (lo, hi) = (vminq_u8(v2, v20), vmaxq_u8(v2, v20));
                v2 = lo;
                v20 = hi;
                let (lo, hi) = (vminq_u8(v2, v11), vmaxq_u8(v2, v11));
                v2 = lo;
                v11 = hi;
                let (lo, hi) = (vminq_u8(v12, v21), vmaxq_u8(v12, v21));
                v12 = lo;
                v21 = hi;
                let (lo, hi) = (vminq_u8(v3, v21), vmaxq_u8(v3, v21));
                v3 = lo;
                v21 = hi;
                let (lo, hi) = (vminq_u8(v3, v12), vmaxq_u8(v3, v12));
                v3 = lo;
                v12 = hi;
                let (lo, hi) = (vminq_u8(v13, v22), vmaxq_u8(v13, v22));
                v13 = lo;
                v22 = hi;
                let (lo, hi) = (vminq_u8(v4, v22), vmaxq_u8(v4, v22));
                v4 = lo;
                v22 = hi;
                let (lo, hi) = (vminq_u8(v4, v13), vmaxq_u8(v4, v13));
                v4 = lo;
                v13 = hi;
                let (lo, hi) = (vminq_u8(v5, v23), vmaxq_u8(v5, v23));
                v5 = lo;
                v23 = hi;
                let (lo, hi) = (vminq_u8(v5, v14), vmaxq_u8(v5, v14));
                v5 = lo;
                v14 = hi;
                let (lo, hi) = (vminq_u8(v6, v24), vmaxq_u8(v6, v24));
                v6 = lo;
                v24 = hi;
                let (lo, hi) = (vminq_u8(v6, v15), vmaxq_u8(v6, v15));
                v6 = lo;
                v15 = hi;
                let (lo, hi) = (vminq_u8(v7, v16), vmaxq_u8(v7, v16));
                v7 = lo;
                v16 = hi;
                let (lo, hi) = (vminq_u8(v7, v19), vmaxq_u8(v7, v19));
                v7 = lo;
                v19 = hi;
                let (lo, hi) = (vminq_u8(v13, v21), vmaxq_u8(v13, v21));
                v13 = lo;
                v21 = hi;
                let (lo, hi) = (vminq_u8(v15, v23), vmaxq_u8(v15, v23));
                v15 = lo;
                v23 = hi;
                let (lo, hi) = (vminq_u8(v7, v13), vmaxq_u8(v7, v13));
                v7 = lo;
                v13 = hi;
                let (lo, hi) = (vminq_u8(v7, v15), vmaxq_u8(v7, v15));
                v7 = lo;
                v15 = hi;
                let (lo, hi) = (vminq_u8(v1, v9), vmaxq_u8(v1, v9));
                v1 = lo;
                v9 = hi;
                let (lo, hi) = (vminq_u8(v3, v11), vmaxq_u8(v3, v11));
                v3 = lo;
                v11 = hi;
                let (lo, hi) = (vminq_u8(v5, v17), vmaxq_u8(v5, v17));
                v5 = lo;
                v17 = hi;
                let (lo, hi) = (vminq_u8(v11, v17), vmaxq_u8(v11, v17));
                v11 = lo;
                v17 = hi;
                let (lo, hi) = (vminq_u8(v9, v17), vmaxq_u8(v9, v17));
                v9 = lo;
                v17 = hi;
                let (lo, hi) = (vminq_u8(v4, v10), vmaxq_u8(v4, v10));
                v4 = lo;
                v10 = hi;
                let (lo, hi) = (vminq_u8(v6, v12), vmaxq_u8(v6, v12));
                v6 = lo;
                v12 = hi;
                let (lo, hi) = (vminq_u8(v7, v14), vmaxq_u8(v7, v14));
                v7 = lo;
                v14 = hi;
                let (lo, hi) = (vminq_u8(v4, v6), vmaxq_u8(v4, v6));
                v4 = lo;
                v6 = hi;
                let (lo, hi) = (vminq_u8(v4, v7), vmaxq_u8(v4, v7));
                v4 = lo;
                v7 = hi;
                let (lo, hi) = (vminq_u8(v12, v14), vmaxq_u8(v12, v14));
                v12 = lo;
                v14 = hi;
                let (lo, hi) = (vminq_u8(v10, v14), vmaxq_u8(v10, v14));
                v10 = lo;
                v14 = hi;
                let (lo, hi) = (vminq_u8(v6, v7), vmaxq_u8(v6, v7));
                v6 = lo;
                v7 = hi;
                let (lo, hi) = (vminq_u8(v10, v12), vmaxq_u8(v10, v12));
                v10 = lo;
                v12 = hi;
                let (lo, hi) = (vminq_u8(v6, v10), vmaxq_u8(v6, v10));
                v6 = lo;
                v10 = hi;
                let (lo, hi) = (vminq_u8(v6, v17), vmaxq_u8(v6, v17));
                v6 = lo;
                v17 = hi;
                let (lo, hi) = (vminq_u8(v12, v17), vmaxq_u8(v12, v17));
                v12 = lo;
                v17 = hi;
                let (lo, hi) = (vminq_u8(v7, v17), vmaxq_u8(v7, v17));
                v7 = lo;
                v17 = hi;
                let (lo, hi) = (vminq_u8(v7, v10), vmaxq_u8(v7, v10));
                v7 = lo;
                v10 = hi;
                let (lo, hi) = (vminq_u8(v12, v18), vmaxq_u8(v12, v18));
                v12 = lo;
                v18 = hi;
                let (lo, hi) = (vminq_u8(v7, v12), vmaxq_u8(v7, v12));
                v7 = lo;
                v12 = hi;
                let (lo, hi) = (vminq_u8(v10, v18), vmaxq_u8(v10, v18));
                v10 = lo;
                v18 = hi;
                let (lo, hi) = (vminq_u8(v12, v20), vmaxq_u8(v12, v20));
                v12 = lo;
                v20 = hi;
                let (lo, hi) = (vminq_u8(v10, v20), vmaxq_u8(v10, v20));
                v10 = lo;
                v20 = hi;
                let (lo, hi) = (vminq_u8(v10, v12), vmaxq_u8(v10, v12));
                v10 = lo;
                v12 = hi;
                let _ = (
                    v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v13, v14, v15, v16, v17, v18,
                    v19, v20, v21, v22, v23, v24,
                );
                v12
            };

            let mut cur = sort_cols(0);
            let mut prev = dup_lane0(&cur);
            let mut x = 0usize;
            while x + 32 <= w {
                let next = sort_cols(x + 16);
                vst1q_u8(drow.as_mut_ptr().add(x), run_block(&prev, &cur, &next));
                prev = cur;
                cur = next;
                x += 16;
            }
            // Tail: overlapped unaligned block ending at w; columns beyond
            // w-1 replicate the edge.
            let xt = w - 16;
            let c = sort_cols(xt);
            let p = sort_cols(xt - 2);
            // Rotate so lanes 14,15 carry cols xt-2, xt-1 (run_block only
            // consumes prev's top two lanes).
            let p_sh = [
                vextq_u8(p[0], p[0], 2),
                vextq_u8(p[1], p[1], 2),
                vextq_u8(p[2], p[2], 2),
                vextq_u8(p[3], p[3], 2),
                vextq_u8(p[4], p[4], 2),
            ];
            let n = dup_lane15(&c);
            vst1q_u8(drow.as_mut_ptr().add(xt), run_block(&p_sh, &c, &n));
            // Cover any gap between the aligned loop and the tail block.
            let mut xg = x;
            while xg < xt {
                let c = sort_cols(xg);
                let p = sort_cols(xg - 2);
                let n = sort_cols(xg + 2);
                // vext(p, c, 14/15) needs p's lanes 14,15 = cols xg-2, xg-1:
                // p loaded at xg-2 gives lanes 0..15 = cols xg-2..xg+13 —
                // wrong alignment for the shared run_block; recompute via
                // an aligned-style pair: emulate by shifting p so its lane
                // 14 is col xg-2.
                let p_sh = [
                    vextq_u8(p[0], p[0], 2),
                    vextq_u8(p[1], p[1], 2),
                    vextq_u8(p[2], p[2], 2),
                    vextq_u8(p[3], p[3], 2),
                    vextq_u8(p[4], p[4], 2),
                ];
                let n_sh = [
                    vextq_u8(n[0], n[0], 14),
                    vextq_u8(n[1], n[1], 14),
                    vextq_u8(n[2], n[2], 14),
                    vextq_u8(n[3], n[3], 14),
                    vextq_u8(n[4], n[4], 14),
                ];
                vst1q_u8(drow.as_mut_ptr().add(xg), run_block(&p_sh, &c, &n_sh));
                xg += 16;
            }
        }
        true
    }
}

#[cfg(feature = "cuda")]
mod cuda_adapters {
    use super::*;
    use crate::cuda::dispatch::{device_slices, untyped_device_err};
    use crate::cuda::median::launch_median_u8;
    use cudarc::driver::CudaStream;
    use std::sync::Arc;

    fn err(e: impl std::fmt::Display) -> ImageError {
        ImageError::Cuda(e.to_string())
    }

    pub(super) fn median_blur_cuda<const C: usize>(
        src: &Image<u8, C>,
        dst: &mut Image<u8, C>,
        ksize: usize,
        stream: &Arc<CudaStream>,
    ) -> Result<(), ImageError> {
        let ctx = stream.context();
        let (s, d) = device_slices!(src, dst);
        launch_median_u8(ctx, stream, s, d, src.cols(), src.rows(), C, ksize).map_err(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::ImageSize;

    fn sz(w: usize, h: usize) -> ImageSize {
        ImageSize {
            width: w,
            height: h,
        }
    }

    fn median_naive(win: &mut [u8]) -> u8 {
        win.sort_unstable();
        win[win.len() / 2]
    }

    #[test]
    fn networks_match_naive_median() {
        // Exhaustive-ish pseudo-random check of both networks.
        let mut state = 0x12345678u64;
        let mut next = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (state >> 33) as u8
        };
        for _ in 0..2000 {
            let mut w9 = [0u8; 9];
            for v in w9.iter_mut() {
                *v = next();
            }
            let mut sorted = w9.to_vec();
            assert_eq!(median9(&mut w9.clone()), median_naive(&mut sorted));

            let mut w25 = [0u8; 25];
            for v in w25.iter_mut() {
                *v = next();
            }
            let mut sorted = w25.to_vec();
            assert_eq!(median25(&mut w25.clone()), median_naive(&mut sorted));
        }
    }

    /// Pin the 71-exchange sorted-column network: exact median for every
    /// column-sorted 0-1 pattern (zero-one principle => exact for all u8).
    #[test]
    fn reduced_net_exact_on_sorted_columns() {
        // Mirrors the NEON median5_row layout: idx = col*5 + rank.
        const NET71: [(usize, usize); 71] = [
            (9, 10),
            (8, 9),
            (14, 16),
            (14, 15),
            (2, 5),
            (3, 6),
            (0, 3),
            (4, 7),
            (1, 4),
            (11, 14),
            (8, 11),
            (12, 15),
            (9, 15),
            (9, 12),
            (10, 16),
            (10, 13),
            (17, 23),
            (17, 20),
            (18, 24),
            (18, 21),
            (19, 22),
            (8, 17),
            (9, 18),
            (0, 9),
            (10, 19),
            (1, 19),
            (1, 10),
            (11, 20),
            (2, 20),
            (2, 11),
            (12, 21),
            (3, 21),
            (3, 12),
            (13, 22),
            (4, 22),
            (4, 13),
            (5, 23),
            (5, 14),
            (6, 24),
            (6, 15),
            (7, 16),
            (7, 19),
            (13, 21),
            (15, 23),
            (7, 13),
            (7, 15),
            (1, 9),
            (3, 11),
            (5, 17),
            (11, 17),
            (9, 17),
            (4, 10),
            (6, 12),
            (7, 14),
            (4, 6),
            (4, 7),
            (12, 14),
            (10, 14),
            (6, 7),
            (10, 12),
            (6, 10),
            (6, 17),
            (12, 17),
            (7, 17),
            (7, 10),
            (12, 18),
            (7, 12),
            (10, 18),
            (12, 20),
            (10, 20),
            (10, 12),
        ];
        for pattern in 0..6usize.pow(5) {
            let mut v = [0u8; 25];
            let mut ones = 0;
            let mut p = pattern;
            for c in 0..5 {
                let t = p % 6;
                p /= 6;
                for j in 0..t {
                    v[c * 5 + 4 - j] = 1;
                }
                ones += t;
            }
            for &(a, b) in NET71.iter() {
                let (lo, hi) = (v[a].min(v[b]), v[a].max(v[b]));
                v[a] = lo;
                v[b] = hi;
            }
            assert_eq!(v[12], u8::from(ones >= 13), "pattern {pattern}");
        }
    }

    #[test]
    fn rejects_bad_ksize_and_size_mismatch() {
        let src = Image::<u8, 1>::from_size_val(sz(8, 8), 0).unwrap();
        let mut dst = Image::<u8, 1>::from_size_val(sz(8, 8), 0).unwrap();
        assert!(median_blur(&src, &mut dst, 4).is_err());
        assert!(median_blur(&src, &mut dst, 7).is_err());
        let mut small = Image::<u8, 1>::from_size_val(sz(4, 8), 0).unwrap();
        assert!(median_blur(&src, &mut small, 3).is_err());
    }

    #[test]
    fn constant_image_unchanged() {
        let src = Image::<u8, 3>::from_size_val(sz(16, 12), 99).unwrap();
        let mut dst = Image::<u8, 3>::from_size_val(sz(16, 12), 0).unwrap();
        median_blur(&src, &mut dst, 3).unwrap();
        assert!(dst.as_slice().iter().all(|&v| v == 99));
        median_blur(&src, &mut dst, 5).unwrap();
        assert!(dst.as_slice().iter().all(|&v| v == 99));
    }
}

#[cfg(all(test, feature = "cuda"))]
mod cuda_tests {
    use super::*;
    use crate::cuda::color::test_utils::{default_stream, pattern_u8};
    use kornia_image::ImageSize;

    fn sz(w: usize, h: usize) -> ImageSize {
        ImageSize {
            width: w,
            height: h,
        }
    }

    #[test]
    fn median_device_equals_host_byte_exact() {
        let stream = default_stream();
        for (w, h) in [(64usize, 48usize), (67, 43), (5, 4), (1, 1), (2, 9)] {
            for ksize in [3usize, 5] {
                let src = Image::<u8, 1>::new(sz(w, h), pattern_u8(w * h)).unwrap();
                let mut cpu = Image::<u8, 1>::from_size_val(sz(w, h), 0).unwrap();
                median_blur(&src, &mut cpu, ksize).unwrap();

                let d_src = src.to_cuda(&stream).unwrap();
                let mut d_dst = Image::<u8, 1>::zeros_cuda(sz(w, h), &stream).unwrap();
                median_blur(&d_src, &mut d_dst, ksize).unwrap();
                assert_eq!(
                    d_dst.to_host_owned().unwrap().as_slice(),
                    cpu.as_slice(),
                    "{w}x{h} k={ksize}"
                );
            }
        }
        // C3
        let (w, h) = (33usize, 21usize);
        for ksize in [3usize, 5] {
            let src = Image::<u8, 3>::new(sz(w, h), pattern_u8(w * h * 3)).unwrap();
            let mut cpu = Image::<u8, 3>::from_size_val(sz(w, h), 0).unwrap();
            median_blur(&src, &mut cpu, ksize).unwrap();
            let d_src = src.to_cuda(&stream).unwrap();
            let mut d_dst = Image::<u8, 3>::zeros_cuda(sz(w, h), &stream).unwrap();
            median_blur(&d_src, &mut d_dst, ksize).unwrap();
            assert_eq!(
                d_dst.to_host_owned().unwrap().as_slice(),
                cpu.as_slice(),
                "C3 {w}x{h} k={ksize}"
            );
        }
    }
}

#[cfg(test)]
mod probe_tests {
    use super::*;
    use kornia_image::ImageSize;

    #[test]
    #[ignore]
    fn median_bilateral_probe() {
        let (w, h) = (1920usize, 1080usize);
        let data: Vec<u8> = (0..w * h)
            .map(|i| ((i * 2654435761usize) >> 24) as u8)
            .collect();
        let src = Image::<u8, 1>::new(
            ImageSize {
                width: w,
                height: h,
            },
            data,
        )
        .unwrap();
        let mut dst = Image::<u8, 1>::from_size_val(src.size(), 0).unwrap();
        for (name, k) in [("median3", 3usize), ("median5", 5)] {
            let mut best = f64::INFINITY;
            for _ in 0..5 {
                let t = std::time::Instant::now();
                for _ in 0..10 {
                    median_blur(&src, &mut dst, k).unwrap();
                }
                best = best.min(t.elapsed().as_secs_f64() / 10.0);
            }
            println!("{name}: {:.3} ms", best * 1e3);
        }
        let mut best = f64::INFINITY;
        for _ in 0..3 {
            let t = std::time::Instant::now();
            for _ in 0..5 {
                crate::filter::bilateral_filter(&src, &mut dst, 5, 50.0, 50.0).unwrap();
            }
            best = best.min(t.elapsed().as_secs_f64() / 5.0);
        }
        println!("bilateral: {:.3} ms", best * 1e3);
    }
}
