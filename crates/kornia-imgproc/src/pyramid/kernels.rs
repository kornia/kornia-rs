//! Separable single-channel pyrDown (OpenCV's scheme).
//!
//! Each source row is filtered horizontally with the 1-4-6-4-1 taps and
//! decimated once into a 5-row ring; each output row then combines five ring
//! rows vertically. That is 5 + 5 taps per output instead of 25, and every
//! source row is read once.

use super::reflect_101;

/// Rows of the 1-4-6-4-1 kernel, and so of the ring buffer.
const TAPS: usize = 5;

/// Narrowest source row the ring kernels take: the interior loop needs
/// reflected indices `1`, `2`, `w - 2` and `w - 3` to be distinct and valid.
pub(super) const MIN_WIDTH: usize = 5;

/// End of the interior output columns: column `x` reads the complete source
/// pairs `x - 1`, `x` and `x + 1`, so `x + 1 < w / 2`.
#[inline]
fn interior_end(w: usize, dw: usize) -> usize {
    (w / 2 - 1).min(dw)
}

/// Five-tap source indices of output column `dx`, reflected (BORDER_REFLECT_101).
#[inline]
fn taps_at(dx: usize, w: usize) -> [usize; TAPS] {
    let c = 2 * dx as i32;
    let w = w as i32;
    [
        reflect_101(c - 2, w) as usize,
        reflect_101(c - 1, w) as usize,
        reflect_101(c, w) as usize,
        reflect_101(c + 1, w) as usize,
        reflect_101(c + 2, w) as usize,
    ]
}

/// Horizontal 1-4-6-4-1 filter and decimation of one u8 row; sums stay below 4081.
#[inline(always)]
fn hrow_u8<const SSE2: bool>(src: &[u8], row: &mut [u16]) {
    let w = src.len();
    let dw = row.len();
    let end = interior_end(w, dw);
    let tap = |s: [u16; TAPS]| s[0] + s[4] + 4 * (s[1] + s[3]) + 6 * s[2];
    for dx in (0..1).chain(end..dw) {
        let t = taps_at(dx, w);
        row[dx] = tap(t.map(|i| src[i] as u16));
    }
    // Pairs (s[2k], s[2k + 1]); output dx uses pairs dx - 1, dx and the even
    // sample of pair dx + 1.
    let start = if SSE2 {
        interior_simd_u8(src, row, end)
    } else {
        interior_avx2_u8(src, row, end)
    };
    let pairs = src[2 * (start - 1)..].chunks_exact(2);
    for (((out, a), b), c) in row[start..end]
        .iter_mut()
        .zip(pairs.clone())
        .zip(pairs.clone().skip(1))
        .zip(pairs.skip(2))
    {
        *out = tap([
            a[0] as u16,
            a[1] as u16,
            b[0] as u16,
            b[1] as u16,
            c[0] as u16,
        ]);
    }
}

/// Horizontal 1-4-6-4-1 filter and decimation of one f32 row, OpenCV's
/// vector expression order: `s2 * 6 + ((s1 + s3) * 4 + (s0 + s4))`.
#[inline(always)]
fn hrow_f32<const SSE2: bool>(src: &[f32], row: &mut [f32]) {
    let w = src.len();
    let dw = row.len();
    let end = interior_end(w, dw);
    let tap = |s: [f32; TAPS]| s[2] * 6.0 + ((s[1] + s[3]) * 4.0 + (s[0] + s[4]));
    for dx in (0..1).chain(end..dw) {
        let t = taps_at(dx, w);
        row[dx] = tap(t.map(|i| src[i]));
    }
    let start = if SSE2 {
        interior_simd_f32(src, row, end)
    } else {
        interior_avx2_f32(src, row, end)
    };
    let pairs = src[2 * (start - 1)..].chunks_exact(2);
    for (((out, a), b), c) in row[start..end]
        .iter_mut()
        .zip(pairs.clone())
        .zip(pairs.clone().skip(1))
        .zip(pairs.skip(2))
    {
        *out = tap([a[0], a[1], b[0], b[1], c[0]]);
    }
}

/// Interior output columns `1..` done with SIMD; returns the first one left.
#[inline]
fn interior_simd_u8(src: &[u8], row: &mut [u16], end: usize) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: SSE2 is part of x86_64; bounds are checked inside.
        return unsafe { hrow_u8_sse2(src, row, end) };
    }
    #[allow(unreachable_code)]
    {
        let _ = (src, row, end);
        1
    }
}

#[inline]
fn interior_simd_f32(src: &[f32], row: &mut [f32], end: usize) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: SSE2 is part of x86_64; bounds are checked inside.
        return unsafe { hrow_f32_sse2(src, row, end) };
    }
    #[allow(unreachable_code)]
    {
        let _ = (src, row, end);
        1
    }
}

/// AVX2 interior, reached only from the `_avx2` entry points (`SSE2 = false`).
#[inline(always)]
fn interior_avx2_u8(src: &[u8], row: &mut [u16], end: usize) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: only `pyrdown_c1_u8_avx2` instantiates the `SSE2 = false`
        // body, after checking AVX2; bounds are checked inside.
        return unsafe { hrow_u8_avx2(src, row, end) };
    }
    #[allow(unreachable_code)]
    {
        let _ = (src, row, end);
        1
    }
}

#[inline(always)]
fn interior_avx2_f32(src: &[f32], row: &mut [f32], end: usize) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: only `pyrdown_c1_f32_avx2` instantiates the `SSE2 = false`
        // body, after checking AVX2; bounds are checked inside.
        return unsafe { hrow_f32_avx2(src, row, end) };
    }
    #[allow(unreachable_code)]
    {
        let _ = (src, row, end);
        1
    }
}

/// [`hrow_u8_sse2`] on 256-bit vectors: 16 outputs per step. The byte-to-u16
/// split works within each 128-bit lane, so no lane crossing is needed.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hrow_u8_avx2(src: &[u8], row: &mut [u16], end: usize) -> usize {
    use std::arch::x86_64::*;
    const STEP: usize = 16;
    let even = _mm256_set1_epi16(0x00ff);
    let mut d = 1;
    // The last load reads bytes `2d + 2 .. 2d + 34`.
    while d + STEP <= end && 2 * d + 34 <= src.len() {
        let p = src.as_ptr().add(2 * d - 2);
        let l = _mm256_loadu_si256(p as *const __m256i);
        let m = _mm256_loadu_si256(p.add(2) as *const __m256i);
        let n = _mm256_loadu_si256(p.add(4) as *const __m256i);
        let (s0, s1) = (_mm256_and_si256(l, even), _mm256_srli_epi16(l, 8));
        let (s2, s3) = (_mm256_and_si256(m, even), _mm256_srli_epi16(m, 8));
        let s4 = _mm256_and_si256(n, even);
        let outer = _mm256_add_epi16(s0, s4);
        let mid = _mm256_slli_epi16(_mm256_add_epi16(s1, s3), 2);
        let center = _mm256_add_epi16(_mm256_slli_epi16(s2, 2), _mm256_slli_epi16(s2, 1));
        let v = _mm256_add_epi16(_mm256_add_epi16(outer, mid), center);
        _mm256_storeu_si256(row.as_mut_ptr().add(d) as *mut __m256i, v);
        d += STEP;
    }
    d
}

/// [`hrow_f32_sse2`] on 256-bit vectors: 8 outputs per step. `shufps` works
/// per 128-bit lane, so every tap comes out in the same lane-swapped order;
/// the taps are combined in that order and one `vpermpd` restores the result
/// (one lane-crossing shuffle per step instead of one per tap).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hrow_f32_avx2(src: &[f32], row: &mut [f32], end: usize) -> usize {
    use std::arch::x86_64::*;
    const STEP: usize = 8;
    const EVEN: i32 = 0b10_00_10_00;
    const ODD: i32 = 0b11_01_11_01;
    // 64-bit chunks (a0 a2)(b0 b2)(a4 a6)(b4 b6) -> (a0 a2)(a4 a6)(b0 b2)(b4 b6).
    const ORDER: i32 = 0b11_01_10_00;
    let (four, six) = (_mm256_set1_ps(4.0), _mm256_set1_ps(6.0));
    let mut d = 1;
    // The last load reads floats `2d + 14 .. 2d + 22`.
    while d + STEP <= end && 2 * d + 22 <= src.len() {
        let p = src.as_ptr().add(2 * d - 2);
        let (a, b) = (_mm256_loadu_ps(p), _mm256_loadu_ps(p.add(8)));
        let (c, e) = (_mm256_loadu_ps(p.add(2)), _mm256_loadu_ps(p.add(10)));
        let (f, g) = (_mm256_loadu_ps(p.add(4)), _mm256_loadu_ps(p.add(12)));
        let s0 = _mm256_shuffle_ps::<EVEN>(a, b);
        let s1 = _mm256_shuffle_ps::<ODD>(a, b);
        let s2 = _mm256_shuffle_ps::<EVEN>(c, e);
        let s3 = _mm256_shuffle_ps::<ODD>(c, e);
        let s4 = _mm256_shuffle_ps::<EVEN>(f, g);
        // Lane-wise, so the order of the elements does not change the bits.
        let v = _mm256_add_ps(
            _mm256_mul_ps(s2, six),
            _mm256_add_ps(
                _mm256_mul_ps(_mm256_add_ps(s1, s3), four),
                _mm256_add_ps(s0, s4),
            ),
        );
        let v = _mm256_permute4x64_pd::<ORDER>(_mm256_castps_pd(v));
        _mm256_storeu_ps(row.as_mut_ptr().add(d), _mm256_castpd_ps(v));
        d += STEP;
    }
    d
}

/// Eight outputs per step: the 16 bytes at `2d - 2`, `2d` and `2d + 2` give,
/// as u16 lanes, the even samples (low bytes) and odd samples (high bytes)
/// of the five taps of outputs `d .. d + 8`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn hrow_u8_sse2(src: &[u8], row: &mut [u16], end: usize) -> usize {
    use std::arch::x86_64::*;
    const STEP: usize = 8;
    let even = _mm_set1_epi16(0x00ff);
    let mut d = 1;
    // The last load reads bytes `2d + 2 .. 2d + 18`.
    while d + STEP <= end && 2 * d + 18 <= src.len() {
        let p = src.as_ptr().add(2 * d - 2);
        let l = _mm_loadu_si128(p as *const __m128i);
        let m = _mm_loadu_si128(p.add(2) as *const __m128i);
        let n = _mm_loadu_si128(p.add(4) as *const __m128i);
        let (s0, s1) = (_mm_and_si128(l, even), _mm_srli_epi16(l, 8));
        let (s2, s3) = (_mm_and_si128(m, even), _mm_srli_epi16(m, 8));
        let s4 = _mm_and_si128(n, even);
        let outer = _mm_add_epi16(s0, s4);
        let mid = _mm_slli_epi16(_mm_add_epi16(s1, s3), 2);
        let center = _mm_add_epi16(_mm_slli_epi16(s2, 2), _mm_slli_epi16(s2, 1));
        let v = _mm_add_epi16(_mm_add_epi16(outer, mid), center);
        _mm_storeu_si128(row.as_mut_ptr().add(d) as *mut __m128i, v);
        d += STEP;
    }
    d
}

/// Four outputs per step: even/odd floats split with `shufps` from the 8
/// floats at `2d - 2` and `2d`, plus the even ones at `2d + 2`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn hrow_f32_sse2(src: &[f32], row: &mut [f32], end: usize) -> usize {
    use std::arch::x86_64::*;
    const STEP: usize = 4;
    const EVEN: i32 = 0b10_00_10_00;
    const ODD: i32 = 0b11_01_11_01;
    let (four, six) = (_mm_set1_ps(4.0), _mm_set1_ps(6.0));
    let mut d = 1;
    // The last load reads floats `2d + 6 .. 2d + 10`.
    while d + STEP <= end && 2 * d + 10 <= src.len() {
        let p = src.as_ptr().add(2 * d - 2);
        let (a, b, c, e) = (
            _mm_loadu_ps(p),
            _mm_loadu_ps(p.add(4)),
            _mm_loadu_ps(p.add(8)),
            _mm_loadu_ps(p.add(2)),
        );
        let f = _mm_loadu_ps(p.add(6));
        let s0 = _mm_shuffle_ps::<EVEN>(a, b);
        let s1 = _mm_shuffle_ps::<ODD>(a, b);
        let s2 = _mm_shuffle_ps::<EVEN>(e, f);
        let s3 = _mm_shuffle_ps::<ODD>(e, f);
        let s4 = _mm_shuffle_ps::<EVEN>(b, c);
        let v = _mm_add_ps(
            _mm_mul_ps(s2, six),
            _mm_add_ps(_mm_mul_ps(_mm_add_ps(s1, s3), four), _mm_add_ps(s0, s4)),
        );
        _mm_storeu_ps(row.as_mut_ptr().add(d), v);
        d += STEP;
    }
    d
}

/// Output widths up to this keep the ring on the stack (20 kB for f32).
const RING_STACK_WIDTH: usize = 1024;

/// Five ring rows of the horizontal pass, keyed by source row. The rows an
/// output needs, after reflection, fall inside five consecutive source rows,
/// so `row % 5` never evicts one still in use.
struct Ring<'a, T> {
    buf: &'a mut [T],
    tags: [usize; TAPS],
    dw: usize,
}

impl<'a, T: Copy + Default> Ring<'a, T> {
    /// `buf` holds `TAPS * dw` elements.
    fn new(buf: &'a mut [T], dw: usize) -> Self {
        Self {
            buf,
            tags: [usize::MAX; TAPS],
            dw,
        }
    }

    /// Fills the slots of source rows `ys` that are not cached yet.
    #[inline(always)]
    fn fill(&mut self, ys: [usize; TAPS], mut hrow: impl FnMut(usize, &mut [T])) {
        for y in ys {
            let slot = y % TAPS;
            if self.tags[slot] != y {
                hrow(y, &mut self.buf[slot * self.dw..(slot + 1) * self.dw]);
                self.tags[slot] = y;
            }
        }
    }

    #[inline(always)]
    fn row(&self, y: usize) -> &[T] {
        let slot = y % TAPS;
        &self.buf[slot * self.dw..(slot + 1) * self.dw]
    }
}

/// Source rows of output row `dy`, reflected (BORDER_REFLECT_101).
#[inline]
fn rows_at(dy: usize, h: usize) -> [usize; TAPS] {
    taps_at(dy, h)
}

/// Single-channel u8 pyrDown on strided rows, `w >= MIN_WIDTH`; output
/// equals the 5x5 integer kernel with `(sum + 128) >> 8` rounding exactly.
/// Row `y` of the source starts at `y * src_stride`, row `y` of the output
/// at `y * dst_stride`.
pub(crate) fn pyrdown_c1_u8(
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    dst: &mut [u8],
    dst_stride: usize,
    dw: usize,
) {
    // The ring is taken here, outside the AVX2 function, so the row loops
    // inline into it without crossing a closure.
    crate::scratch::with_scratch::<u16, { TAPS * RING_STACK_WIDTH }, _>(TAPS * dw, |buf| {
        let ring = &mut Ring::new(buf, dw);
        #[cfg(target_arch = "x86_64")]
        if crate::simd::cpu_features().has_avx2 {
            // SAFETY: AVX2 is present.
            unsafe { pyrdown_c1_u8_avx2(src, src_stride, w, h, dst, dst_stride, ring) };
            return;
        }
        pyrdown_c1_u8_rows::<true>(src, src_stride, w, h, dst, dst_stride, ring);
    });
}

/// The ring kernel compiled for AVX2, with the AVX2 horizontal pass.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn pyrdown_c1_u8_avx2(
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    dst: &mut [u8],
    dst_stride: usize,
    ring: &mut Ring<'_, u16>,
) {
    pyrdown_c1_u8_rows::<false>(src, src_stride, w, h, dst, dst_stride, ring);
}

#[inline(always)]
fn pyrdown_c1_u8_rows<const SSE2: bool>(
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    dst: &mut [u8],
    dst_stride: usize,
    ring: &mut Ring<'_, u16>,
) {
    let dw = ring.dw;
    for dy in 0..h.div_ceil(2) {
        let ys = rows_at(dy, h);
        ring.fill(ys, |y, row| {
            hrow_u8::<SSE2>(&src[y * src_stride..][..w], row)
        });
        let [r0, r1, r2, r3, r4] = ys.map(|y| ring.row(y));
        let out = &mut dst[dy * dst_stride..][..dw];
        let rows = out.iter_mut().zip(r0).zip(r1).zip(r2).zip(r3).zip(r4);
        for (((((o, a), b), c), d), e) in rows {
            // Max 255 * 256 + 128 = 65408 fits u16.
            *o = ((a + e + 4 * (b + d) + 6 * c + 128) >> 8) as u8;
        }
    }
}

/// Single-channel f32 pyrDown, `w >= MIN_WIDTH`, separable.
pub(super) fn pyrdown_c1_f32(src: &[f32], w: usize, h: usize, dst: &mut [f32], dw: usize) {
    crate::scratch::with_scratch::<f32, { TAPS * RING_STACK_WIDTH }, _>(TAPS * dw, |buf| {
        let ring = &mut Ring::new(buf, dw);
        #[cfg(target_arch = "x86_64")]
        if crate::simd::cpu_features().has_avx2 {
            // SAFETY: AVX2 is present.
            unsafe { pyrdown_c1_f32_avx2(src, w, h, dst, ring) };
            return;
        }
        pyrdown_c1_f32_rows::<true>(src, w, h, dst, ring);
    });
}

/// The ring kernel compiled for AVX2, with the AVX2 horizontal pass (no
/// FMA, so the same rounding).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn pyrdown_c1_f32_avx2(
    src: &[f32],
    w: usize,
    h: usize,
    dst: &mut [f32],
    ring: &mut Ring<'_, f32>,
) {
    pyrdown_c1_f32_rows::<false>(src, w, h, dst, ring);
}

#[inline(always)]
fn pyrdown_c1_f32_rows<const SSE2: bool>(
    src: &[f32],
    w: usize,
    h: usize,
    dst: &mut [f32],
    ring: &mut Ring<'_, f32>,
) {
    const SCALE: f32 = 1.0 / 256.0;
    let dw = ring.dw;
    for (dy, out) in dst.chunks_exact_mut(dw).enumerate() {
        let ys = rows_at(dy, h);
        ring.fill(ys, |y, row| hrow_f32::<SSE2>(&src[y * w..(y + 1) * w], row));
        let [r0, r1, r2, r3, r4] = ys.map(|y| ring.row(y));
        let rows = out.iter_mut().zip(r0).zip(r1).zip(r2).zip(r3).zip(r4);
        for (((((o, a), b), c), d), e) in rows {
            *o = (c * 6.0 + ((b + d) * 4.0 + (a + e))) * SCALE;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(n: usize, seed: u64) -> Vec<u8> {
        let mut x = seed;
        (0..n)
            .map(|_| {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (x >> 56) as u8
            })
            .collect()
    }

    /// The SSE2 body (scalar off x86_64) and the dispatched path (AVX2 where
    /// present) give identical outputs, tails included.
    #[test]
    fn isa_paths_agree() {
        for (i, (w, h)) in [
            (5usize, 3usize),
            (17, 9),
            (37, 5),
            (64, 33),
            (127, 64),
            (752, 480),
            // Output wider than the stack ring: heap path.
            (2 * RING_STACK_WIDTH + 9, 7),
        ]
        .into_iter()
        .enumerate()
        {
            let (dw, dh) = (w.div_ceil(2), h.div_ceil(2));
            let src = lcg(w * h, i as u64);
            let (mut a, mut b) = (vec![0u8; dw * dh], vec![0u8; dw * dh]);
            pyrdown_c1_u8(&src, w, w, h, &mut a, dw, dw);
            let mut ring = vec![0u16; TAPS * dw];
            pyrdown_c1_u8_rows::<true>(&src, w, w, h, &mut b, dw, &mut Ring::new(&mut ring, dw));
            assert_eq!(a, b, "u8 {w}x{h}");
            let srcf: Vec<f32> = src.iter().map(|&v| f32::from(v) * 0.37 + 0.11).collect();
            let (mut a, mut b) = (vec![0f32; dw * dh], vec![0f32; dw * dh]);
            pyrdown_c1_f32(&srcf, w, h, &mut a, dw);
            let mut ring = vec![0f32; TAPS * dw];
            pyrdown_c1_f32_rows::<true>(&srcf, w, h, &mut b, &mut Ring::new(&mut ring, dw));
            assert_eq!(a, b, "f32 {w}x{h}");
        }
    }
}
