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

    dst.as_slice_mut()
        .par_chunks_mut(w * C)
        .enumerate()
        .for_each(|(y, drow)| {
            // NEON C1 fast path: run the same network on 16 lanes at once
            // (u8 min/max are exact — byte-identical to the scalar walk).
            #[cfg(target_arch = "aarch64")]
            if C == 1 {
                let done = if ksize == 3 {
                    neon::median3_row(s, drow, w, h, y)
                } else {
                    neon::median5_row(s, drow, w, h, y)
                };
                if done {
                    return;
                }
            }
            // Replicate-clamped source row pointers for the window.
            for x in 0..w {
                for c in 0..C {
                    if ksize == 3 {
                        let mut win = [0u8; 9];
                        let mut n = 0;
                        for dy in -1i64..=1 {
                            let sy = (y as i64 + dy).clamp(0, h as i64 - 1) as usize;
                            for dx in -1i64..=1 {
                                let sx = (x as i64 + dx).clamp(0, w as i64 - 1) as usize;
                                win[n] = s[(sy * w + sx) * C + c];
                                n += 1;
                            }
                        }
                        drow[x * C + c] = median9(&mut win);
                    } else {
                        let mut win = [0u8; 25];
                        let mut n = 0;
                        for dy in -r..=r {
                            let sy = (y as i64 + dy).clamp(0, h as i64 - 1) as usize;
                            for dx in -r..=r {
                                let sx = (x as i64 + dx).clamp(0, w as i64 - 1) as usize;
                                win[n] = s[(sy * w + sx) * C + c];
                                n += 1;
                            }
                        }
                        drow[x * C + c] = median25(&mut win);
                    }
                }
            }
        });
    Ok(())
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

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

    /// Scalar fallback for border columns / narrow images — the caller's
    /// generic clamp walk (same exact medians).
    #[allow(clippy::too_many_arguments)]
    fn scalar_px<const TAPS: usize>(
        s: &[u8],
        drow: &mut [u8],
        w: usize,
        h: usize,
        y: usize,
        r: usize,
        net: &[(usize, usize)],
        center: usize,
        x: usize,
    ) {
        let k = 2 * r + 1;
        let mut win = [0u8; TAPS];
        let mut n = 0;
        for dy in 0..k {
            let sy = (y + dy).saturating_sub(r).min(h - 1);
            for dx in 0..k {
                let sx = (x + dx).saturating_sub(r).min(w - 1);
                win[n] = s[sy * w + sx];
                n += 1;
            }
        }
        for &(a, b) in net.iter() {
            let (lo, hi) = (win[a].min(win[b]), win[a].max(win[b]));
            win[a] = lo;
            win[b] = hi;
        }
        drow[x] = win[center];
    }

    /// One output row, C1, 3×3: the classic exact identity
    /// `med9 = med3(max(col_lows), med3(col_mids), min(col_highs))` on 16
    /// lanes (columns sorted lane-wise). Exact median — byte-identical to
    /// the network walk. Returns false for rows too narrow to vectorize.
    pub(super) fn median3_row(s: &[u8], drow: &mut [u8], w: usize, h: usize, y: usize) -> bool {
        if w < 18 {
            return false;
        }
        // SAFETY: loads span x-1..x+16 for x in [1, w-17]; rows clamped.
        unsafe {
            let r0 = s.as_ptr().add(y.saturating_sub(1).min(h - 1) * w);
            let r1 = s.as_ptr().add(y * w);
            let r2 = s.as_ptr().add((y + 1).min(h - 1) * w);
            let mut x = 1usize;
            while x + 16 <= w - 1 {
                // Column sort at offsets -1, 0, +1.
                let sort3 = |p0: *const u8, p1: *const u8, p2: *const u8, off: usize| {
                    let a = vld1q_u8(p0.add(off));
                    let b = vld1q_u8(p1.add(off));
                    let c = vld1q_u8(p2.add(off));
                    let lo = min3(a, b, c);
                    let hi = max3(a, b, c);
                    let mid = med3(a, b, c);
                    (lo, mid, hi)
                };
                let (lo_a, mid_a, hi_a) = sort3(r0, r1, r2, x - 1);
                let (lo_b, mid_b, hi_b) = sort3(r0, r1, r2, x);
                let (lo_c, mid_c, hi_c) = sort3(r0, r1, r2, x + 1);
                let m = med3(
                    max3(lo_a, lo_b, lo_c),
                    med3(mid_a, mid_b, mid_c),
                    min3(hi_a, hi_b, hi_c),
                );
                vst1q_u8(drow.as_mut_ptr().add(x), m);
                x += 16;
            }
            for xx in (0..1).chain(x..w) {
                scalar_px::<9>(s, drow, w, h, y, 1, &super::NET9, 4, xx);
            }
        }
        true
    }

    /// One output row, C1, 5×5: the full Smith network unrolled on NAMED
    /// vectors (no array indexing — keeps everything in registers). Same
    /// exchange list as `NET25`; exact median, byte-identical.
    pub(super) fn median5_row(s: &[u8], drow: &mut [u8], w: usize, h: usize, y: usize) -> bool {
        if w < 20 {
            return false;
        }
        // SAFETY: loads span x-2..x+17 for x in [2, w-18]; rows clamped.
        unsafe {
            let mut rows: [*const u8; 5] = [std::ptr::null(); 5];
            for (dy, row) in rows.iter_mut().enumerate() {
                let sy = (y + dy).saturating_sub(2).min(h - 1);
                *row = s.as_ptr().add(sy * w);
            }
            let mut x = 2usize;
            while x + 16 <= w - 2 {
                let mut v0 = vld1q_u8(rows[0].add(x - 2));
                let mut v1 = vld1q_u8(rows[0].add(x - 1));
                let mut v2 = vld1q_u8(rows[0].add(x));
                let mut v3 = vld1q_u8(rows[0].add(x + 1));
                let mut v4 = vld1q_u8(rows[0].add(x + 2));
                let mut v5 = vld1q_u8(rows[1].add(x - 2));
                let mut v6 = vld1q_u8(rows[1].add(x - 1));
                let mut v7 = vld1q_u8(rows[1].add(x));
                let mut v8 = vld1q_u8(rows[1].add(x + 1));
                let mut v9 = vld1q_u8(rows[1].add(x + 2));
                let mut v10 = vld1q_u8(rows[2].add(x - 2));
                let mut v11 = vld1q_u8(rows[2].add(x - 1));
                let mut v12 = vld1q_u8(rows[2].add(x));
                let mut v13 = vld1q_u8(rows[2].add(x + 1));
                let mut v14 = vld1q_u8(rows[2].add(x + 2));
                let mut v15 = vld1q_u8(rows[3].add(x - 2));
                let mut v16 = vld1q_u8(rows[3].add(x - 1));
                let mut v17 = vld1q_u8(rows[3].add(x));
                let mut v18 = vld1q_u8(rows[3].add(x + 1));
                let mut v19 = vld1q_u8(rows[3].add(x + 2));
                let mut v20 = vld1q_u8(rows[4].add(x - 2));
                let mut v21 = vld1q_u8(rows[4].add(x - 1));
                let mut v22 = vld1q_u8(rows[4].add(x));
                let mut v23 = vld1q_u8(rows[4].add(x + 1));
                let mut v24 = vld1q_u8(rows[4].add(x + 2));
                let (lo, hi) = (vminq_u8(v0, v1), vmaxq_u8(v0, v1));
                v0 = lo;
                v1 = hi;
                let (lo, hi) = (vminq_u8(v3, v4), vmaxq_u8(v3, v4));
                v3 = lo;
                v4 = hi;
                let (lo, hi) = (vminq_u8(v2, v4), vmaxq_u8(v2, v4));
                v2 = lo;
                v4 = hi;
                let (lo, hi) = (vminq_u8(v2, v3), vmaxq_u8(v2, v3));
                v2 = lo;
                v3 = hi;
                let (lo, hi) = (vminq_u8(v6, v7), vmaxq_u8(v6, v7));
                v6 = lo;
                v7 = hi;
                let (lo, hi) = (vminq_u8(v5, v7), vmaxq_u8(v5, v7));
                v5 = lo;
                v7 = hi;
                let (lo, hi) = (vminq_u8(v5, v6), vmaxq_u8(v5, v6));
                v5 = lo;
                v6 = hi;
                let (lo, hi) = (vminq_u8(v9, v10), vmaxq_u8(v9, v10));
                v9 = lo;
                v10 = hi;
                let (lo, hi) = (vminq_u8(v8, v10), vmaxq_u8(v8, v10));
                v8 = lo;
                v10 = hi;
                let (lo, hi) = (vminq_u8(v8, v9), vmaxq_u8(v8, v9));
                v8 = lo;
                v9 = hi;
                let (lo, hi) = (vminq_u8(v12, v13), vmaxq_u8(v12, v13));
                v12 = lo;
                v13 = hi;
                let (lo, hi) = (vminq_u8(v11, v13), vmaxq_u8(v11, v13));
                v11 = lo;
                v13 = hi;
                let (lo, hi) = (vminq_u8(v11, v12), vmaxq_u8(v11, v12));
                v11 = lo;
                v12 = hi;
                let (lo, hi) = (vminq_u8(v15, v16), vmaxq_u8(v15, v16));
                v15 = lo;
                v16 = hi;
                let (lo, hi) = (vminq_u8(v14, v16), vmaxq_u8(v14, v16));
                v14 = lo;
                v16 = hi;
                let (lo, hi) = (vminq_u8(v14, v15), vmaxq_u8(v14, v15));
                v14 = lo;
                v15 = hi;
                let (lo, hi) = (vminq_u8(v18, v19), vmaxq_u8(v18, v19));
                v18 = lo;
                v19 = hi;
                let (lo, hi) = (vminq_u8(v17, v19), vmaxq_u8(v17, v19));
                v17 = lo;
                v19 = hi;
                let (lo, hi) = (vminq_u8(v17, v18), vmaxq_u8(v17, v18));
                v17 = lo;
                v18 = hi;
                let (lo, hi) = (vminq_u8(v21, v22), vmaxq_u8(v21, v22));
                v21 = lo;
                v22 = hi;
                let (lo, hi) = (vminq_u8(v20, v22), vmaxq_u8(v20, v22));
                v20 = lo;
                v22 = hi;
                let (lo, hi) = (vminq_u8(v20, v21), vmaxq_u8(v20, v21));
                v20 = lo;
                v21 = hi;
                let (lo, hi) = (vminq_u8(v23, v24), vmaxq_u8(v23, v24));
                v23 = lo;
                v24 = hi;
                let (lo, hi) = (vminq_u8(v2, v5), vmaxq_u8(v2, v5));
                v2 = lo;
                v5 = hi;
                let (lo, hi) = (vminq_u8(v3, v6), vmaxq_u8(v3, v6));
                v3 = lo;
                v6 = hi;
                let (lo, hi) = (vminq_u8(v0, v6), vmaxq_u8(v0, v6));
                v0 = lo;
                v6 = hi;
                let (lo, hi) = (vminq_u8(v0, v3), vmaxq_u8(v0, v3));
                v0 = lo;
                v3 = hi;
                let (lo, hi) = (vminq_u8(v4, v7), vmaxq_u8(v4, v7));
                v4 = lo;
                v7 = hi;
                let (lo, hi) = (vminq_u8(v1, v7), vmaxq_u8(v1, v7));
                v1 = lo;
                v7 = hi;
                let (lo, hi) = (vminq_u8(v1, v4), vmaxq_u8(v1, v4));
                v1 = lo;
                v4 = hi;
                let (lo, hi) = (vminq_u8(v11, v14), vmaxq_u8(v11, v14));
                v11 = lo;
                v14 = hi;
                let (lo, hi) = (vminq_u8(v8, v14), vmaxq_u8(v8, v14));
                v8 = lo;
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
                let (lo, hi) = (vminq_u8(v13, v16), vmaxq_u8(v13, v16));
                v13 = lo;
                v16 = hi;
                let (lo, hi) = (vminq_u8(v10, v16), vmaxq_u8(v10, v16));
                v10 = lo;
                v16 = hi;
                let (lo, hi) = (vminq_u8(v10, v13), vmaxq_u8(v10, v13));
                v10 = lo;
                v13 = hi;
                let (lo, hi) = (vminq_u8(v20, v23), vmaxq_u8(v20, v23));
                v20 = lo;
                v23 = hi;
                let (lo, hi) = (vminq_u8(v17, v23), vmaxq_u8(v17, v23));
                v17 = lo;
                v23 = hi;
                let (lo, hi) = (vminq_u8(v17, v20), vmaxq_u8(v17, v20));
                v17 = lo;
                v20 = hi;
                let (lo, hi) = (vminq_u8(v21, v24), vmaxq_u8(v21, v24));
                v21 = lo;
                v24 = hi;
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
                let (lo, hi) = (vminq_u8(v0, v18), vmaxq_u8(v0, v18));
                v0 = lo;
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
                let (lo, hi) = (vminq_u8(v14, v23), vmaxq_u8(v14, v23));
                v14 = lo;
                v23 = hi;
                let (lo, hi) = (vminq_u8(v5, v23), vmaxq_u8(v5, v23));
                v5 = lo;
                v23 = hi;
                let (lo, hi) = (vminq_u8(v5, v14), vmaxq_u8(v5, v14));
                v5 = lo;
                v14 = hi;
                let (lo, hi) = (vminq_u8(v15, v24), vmaxq_u8(v15, v24));
                v15 = lo;
                v24 = hi;
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
                vst1q_u8(drow.as_mut_ptr().add(x), v12);
                let _ = (
                    v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v13, v14, v15, v16, v17, v18,
                    v19, v20, v21, v22, v23, v24,
                );
                x += 16;
            }
            for xx in (0..2).chain(x..w) {
                scalar_px::<25>(s, drow, w, h, y, 2, &super::NET25, 12, xx);
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
