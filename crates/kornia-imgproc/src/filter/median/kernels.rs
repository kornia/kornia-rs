//! Portable `u16` / `f32` median kernels.
//!
//! Each output row is produced strip by strip ([`STRIP`] output columns at a
//! time) in two passes over a stack scratch:
//!
//! 1. **Vertical**: every source column of the `ksize` window rows is
//!    sorted once ([`sort_col3`] / [`sort_col5`]); the `ksize` horizontally
//!    adjacent windows that share the column reuse it.
//! 2. **Horizontal**: a window is `ksize` adjacent sorted columns. 3×3:
//!    `med3(max(lows), med3(mids), min(highs))`; 5×5: the 71-exchange
//!    sorted-column network ([`super::MED25_SORTED_COLUMNS`]). Both are the
//!    networks of the `u8` NEON kernels.
//!
//! Both passes run on `[K; L]` lane arrays with straight-line
//! compare-exchanges, so the compiler vectorizes them (`L` keys per
//! network wire, one vector register at AVX2 / NEON width). On x86_64 CPUs
//! with AVX2 the row kernel is compiled with AVX2 enabled through a
//! `target_feature` wrapper; there are no intrinsics.
//!
//! Samples are compared through an order-preserving integer key
//! ([`MedianKey`]): the `u16` value itself, and for `f32` the IEEE 754
//! totalOrder key of its bits (the order of [`f32::total_cmp`]), so the
//! median is exact and is always one of the window's samples, bit for bit.

use rayon::prelude::*;

/// Output columns per strip; a multiple of every lane width used here.
const STRIP: usize = 256;

/// Largest window radius (5×5).
const MAX_RADIUS: usize = 2;

/// Sorted columns per strip: the strip plus the window apron on both sides.
/// A horizontal block starting below [`STRIP`] reads at most
/// `STRIP + 2 * MAX_RADIUS` columns.
const STRIP_COLS: usize = STRIP + 2 * MAX_RADIUS;

/// A pixel type with an order-preserving integer sort key.
pub(super) trait MedianKey: Copy + Send + Sync + 'static {
    /// Totally ordered key; `min` / `max` on it vectorize.
    type Key: Copy + Ord + Default + Send + Sync;
    /// Lanes per network wire (one 256-bit vector of keys).
    const LANES: usize;
    /// Order-preserving map to the key.
    fn key(self) -> Self::Key;
    /// Inverse of [`Self::key`].
    fn from_key(k: Self::Key) -> Self;
}

impl MedianKey for u16 {
    type Key = u16;
    const LANES: usize = 16;
    #[inline(always)]
    fn key(self) -> u16 {
        self
    }
    #[inline(always)]
    fn from_key(k: u16) -> u16 {
        k
    }
}

/// IEEE 754 totalOrder key of an `f32` bit pattern: signed-integer order of
/// the result is the order of [`f32::total_cmp`] (`-NaN < -inf < … < -0.0 <
/// +0.0 < … < +inf < +NaN`). The map is its own inverse.
#[inline(always)]
pub(super) fn f32_total_order_key(bits: i32) -> i32 {
    bits ^ (((bits >> 31) as u32) >> 1) as i32
}

impl MedianKey for f32 {
    type Key = i32;
    const LANES: usize = 8;
    #[inline(always)]
    fn key(self) -> i32 {
        f32_total_order_key(self.to_bits() as i32)
    }
    #[inline(always)]
    fn from_key(k: i32) -> f32 {
        f32::from_bits(f32_total_order_key(k) as u32)
    }
}

/// Compare-exchange of wires `a` and `b`: afterwards `v[a] <= v[b]` in every
/// lane.
#[inline(always)]
fn ce<K: Copy + Ord, const L: usize, const N: usize>(v: &mut [[K; L]; N], a: usize, b: usize) {
    let (mut lo, mut hi) = (v[a], v[b]);
    for (lo, hi) in lo.iter_mut().zip(hi.iter_mut()) {
        (*lo, *hi) = ((*lo).min(*hi), (*lo).max(*hi));
    }
    (v[a], v[b]) = (lo, hi);
}

/// Sorts 3-sample columns (3 exchanges).
#[inline(always)]
fn sort_col3<K: Copy + Ord, const L: usize, const N: usize>(v: &mut [[K; L]; N]) {
    ce(v, 0, 1);
    ce(v, 1, 2);
    ce(v, 0, 1);
}

/// Sorts 5-sample columns (optimal, 9 exchanges; Knuth, TAOCP vol. 3,
/// 5.3.4). The `u8` NEON 5×5 kernel's column sort.
#[inline(always)]
fn sort_col5<K: Copy + Ord, const L: usize, const N: usize>(v: &mut [[K; L]; N]) {
    ce(v, 0, 1);
    ce(v, 3, 4);
    ce(v, 2, 4);
    ce(v, 2, 3);
    ce(v, 1, 4);
    ce(v, 0, 3);
    ce(v, 0, 2);
    ce(v, 1, 3);
    ce(v, 1, 2);
}

/// Defines `med25_sorted_columns`, the unrolled
/// [`super::MED25_SORTED_COLUMNS`] network, from the literal pair list.
macro_rules! med25_sorted_columns_fn {
    ([$(($a:literal, $b:literal)),* $(,)?]) => {
        /// Brings the median of a 5×5 window held as 5 sorted columns (wire
        /// `col * 5 + rank`) to wire 12, lane-wise.
        #[inline(always)]
        fn med25_sorted_columns<K: Copy + Ord, const L: usize>(v: &mut [[K; L]; 25]) {
            $(ce(v, $a, $b);)*
        }
    };
}
super::with_med25_sorted_columns!(med25_sorted_columns_fn!());

/// Sorted window-row columns of one strip: rank `r` of strip column `p` is
/// `cols[r][p]`, strip column `p` being image column `xs + p - ksize / 2`
/// (clamped to the image: replicate border).
type Columns<K, const KS: usize> = [[K; STRIP_COLS]; KS];

/// Keys of the `B` samples of channel `c` starting at column `x` of an
/// interleaved row.
// The hot helpers fill arrays with plain loops: `std::array::from_fn` and
// `<[T; N]>::map` are not always inlined, and an out-of-line call is
// compiled without the AVX2 target feature of `median_row_avx2`.
#[inline(always)]
fn load_keys<T: MedianKey, const C: usize, const B: usize>(
    row: &[T],
    x: usize,
    c: usize,
) -> [T::Key; B] {
    let mut k = [T::Key::default(); B];
    if C == 1 {
        let s = &row[x..x + B];
        for l in 0..B {
            k[l] = s[l].key();
        }
    } else {
        let s = &row[x * C..(x + B) * C];
        for l in 0..B {
            k[l] = s[l * C + c].key();
        }
    }
    k
}

/// Sorts the `B` window-row columns starting at image column `x` into strip
/// columns `p..p + B`.
#[inline(always)]
fn sort_block<T: MedianKey, const C: usize, const KS: usize, const B: usize>(
    rows: &[&[T]; KS],
    x: usize,
    c: usize,
    p: usize,
    cols: &mut Columns<T::Key, KS>,
) {
    let mut v = [[T::Key::default(); B]; KS];
    for (wire, row) in v.iter_mut().zip(rows) {
        *wire = load_keys::<T, C, B>(row, x, c);
    }
    if KS == 3 {
        sort_col3(&mut v);
    } else {
        sort_col5(&mut v);
    }
    for (rank, wire) in cols.iter_mut().zip(&v) {
        rank[p..p + B].copy_from_slice(wire);
    }
}

/// Vertical pass for the `n` output columns starting at image column `xs`
/// (strip columns `0..n + ksize - 1`), in contiguous blocks of `B` columns.
// `B` is the lane width: one vector register per wire (wider blocks of
// `i32` keys make LLVM spill and shuffle the 5-sample column sort).
#[inline(always)]
fn sort_strip<T: MedianKey, const C: usize, const KS: usize, const B: usize>(
    rows: &[&[T]; KS],
    w: usize,
    xs: usize,
    n: usize,
    c: usize,
    cols: &mut Columns<T::Key, KS>,
) {
    let r = KS / 2;
    let end = n + 2 * r;
    let mut p = 0;
    while p < end {
        // Image column of strip column `p`, before clamping.
        let x = (xs + p) as isize - r as isize;
        if x >= 0 && x as usize + B <= w && p + B <= end {
            sort_block::<T, C, KS, B>(rows, x as usize, c, p, cols);
            p += B;
        } else {
            let xc = x.clamp(0, w as isize - 1) as usize;
            sort_block::<T, C, KS, 1>(rows, xc, c, p, cols);
            p += 1;
        }
    }
}

/// `[K; L]` of rank `rank`, strip columns `p..p + L`.
#[inline(always)]
fn wire<K: Copy, const KS: usize, const L: usize>(
    cols: &Columns<K, KS>,
    rank: usize,
    p: usize,
) -> [K; L] {
    let mut k = [cols[rank][p]; L];
    k.copy_from_slice(&cols[rank][p..p + L]);
    k
}

/// Medians of the windows centred on strip output columns `x0..x0 + L`.
#[inline(always)]
fn window_medians<K: Copy + Ord, const KS: usize, const L: usize>(
    cols: &Columns<K, KS>,
    x0: usize,
) -> [K; L] {
    if KS == 3 {
        // med9 = med3(max(lows), med3(mids), min(highs)) over the three
        // sorted columns of each window: exact (the u8 3×3 identity).
        let lo: [[K; L]; 3] = [
            wire::<K, KS, L>(cols, 0, x0),
            wire::<K, KS, L>(cols, 0, x0 + 1),
            wire::<K, KS, L>(cols, 0, x0 + 2),
        ];
        let mid: [[K; L]; 3] = [
            wire::<K, KS, L>(cols, 1, x0),
            wire::<K, KS, L>(cols, 1, x0 + 1),
            wire::<K, KS, L>(cols, 1, x0 + 2),
        ];
        let hi: [[K; L]; 3] = [
            wire::<K, KS, L>(cols, 2, x0),
            wire::<K, KS, L>(cols, 2, x0 + 1),
            wire::<K, KS, L>(cols, 2, x0 + 2),
        ];
        let mut m = lo[0];
        for l in 0..L {
            let a = lo[0][l].max(lo[1][l]).max(lo[2][l]);
            let c = hi[0][l].min(hi[1][l]).min(hi[2][l]);
            let b = med3(mid[0][l], mid[1][l], mid[2][l]);
            m[l] = med3(a, b, c);
        }
        m
    } else {
        // Wire `col * 5 + rank`, spelled out so the 25 wires stay in
        // registers (a loop here is not reliably unrolled).
        let w = |i: usize| wire::<K, KS, L>(cols, i % 5, x0 + i / 5);
        let mut v = [
            w(0),
            w(1),
            w(2),
            w(3),
            w(4),
            w(5),
            w(6),
            w(7),
            w(8),
            w(9),
            w(10),
            w(11),
            w(12),
            w(13),
            w(14),
            w(15),
            w(16),
            w(17),
            w(18),
            w(19),
            w(20),
            w(21),
            w(22),
            w(23),
            w(24),
        ];
        med25_sorted_columns(&mut v);
        v[12]
    }
}

/// Exact median of three.
#[inline(always)]
fn med3<K: Copy + Ord>(a: K, b: K, c: K) -> K {
    a.min(b).max(a.max(b).min(c))
}

/// One output row (all channels) of the `KS × KS` median, `L`-lane blocks.
#[inline(always)]
fn median_row<T: MedianKey, const C: usize, const KS: usize, const L: usize>(
    src: &[T],
    drow: &mut [T],
    w: usize,
    h: usize,
    y: usize,
) {
    debug_assert!(STRIP.is_multiple_of(L));
    let rows: [&[T]; KS] = std::array::from_fn(|i| {
        let sy = (y + i).saturating_sub(KS / 2).min(h - 1);
        &src[sy * w * C..(sy + 1) * w * C]
    });
    let mut cols: Columns<T::Key, KS> = [[T::Key::default(); STRIP_COLS]; KS];
    for c in 0..C {
        let mut xs = 0;
        while xs < w {
            let n = STRIP.min(w - xs);
            sort_strip::<T, C, KS, L>(&rows, w, xs, n, c, &mut cols);
            let mut x0 = 0;
            while x0 < n {
                let m = window_medians::<T::Key, KS, L>(&cols, x0);
                let x = xs + x0;
                if C == 1 && x0 + L <= n {
                    for (o, &k) in drow[x..x + L].iter_mut().zip(&m) {
                        *o = T::from_key(k);
                    }
                } else {
                    for (l, &k) in m.iter().enumerate().take(n - x0) {
                        drow[(x + l) * C + c] = T::from_key(k);
                    }
                }
                x0 += L;
            }
            xs += n;
        }
    }
}

/// [`median_row`] compiled with AVX2 enabled.
///
/// # Safety
///
/// The CPU must support AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn median_row_avx2<T: MedianKey, const C: usize, const KS: usize, const L: usize>(
    src: &[T],
    drow: &mut [T],
    w: usize,
    h: usize,
    y: usize,
) {
    median_row::<T, C, KS, L>(src, drow, w, h, y);
}

/// `KS × KS` median over an interleaved `h × w × C` buffer, rows in
/// parallel.
fn median_blur_ks<T: MedianKey, const C: usize, const KS: usize, const L: usize>(
    src: &[T],
    dst: &mut [T],
    w: usize,
    h: usize,
) {
    #[cfg(target_arch = "x86_64")]
    let avx2 = crate::simd::cpu_features().has_avx2;
    let min_rows = h.div_ceil(rayon::current_num_threads() * 4).max(16);
    dst.par_chunks_mut(w * C)
        .enumerate()
        .with_min_len(min_rows)
        .for_each(|(y, drow)| {
            #[cfg(target_arch = "x86_64")]
            if avx2 {
                // SAFETY: AVX2 support was checked at runtime above.
                unsafe { median_row_avx2::<T, C, KS, L>(src, drow, w, h, y) };
                return;
            }
            median_row::<T, C, KS, L>(src, drow, w, h, y);
        });
}

/// Defines the `u16` / `f32` entry points (const lane widths need a
/// concrete type).
macro_rules! median_blur_fn {
    ($name:ident, $t:ty) => {
        /// Exact `ksize × ksize` median, replicate border, over an
        /// interleaved `h × w × C` buffer. `ksize` is 3 or 5 (validated by
        /// the caller); an empty image is a no-op.
        pub(super) fn $name<const C: usize>(
            src: &[$t],
            dst: &mut [$t],
            w: usize,
            h: usize,
            ksize: usize,
        ) {
            if w == 0 || h == 0 {
                return;
            }
            const L: usize = <$t as MedianKey>::LANES;
            if ksize == 3 {
                median_blur_ks::<$t, C, 3, L>(src, dst, w, h);
            } else {
                median_blur_ks::<$t, C, 5, L>(src, dst, w, h);
            }
        }
    };
}

median_blur_fn!(median_blur_u16, u16);
median_blur_fn!(median_blur_f32, f32);

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference: sort the replicate-border window by the key order, take
    /// the middle sample.
    fn naive<T: MedianKey, const C: usize>(src: &[T], w: usize, h: usize, ksize: usize) -> Vec<T> {
        let r = (ksize / 2) as isize;
        let mut out = Vec::with_capacity(src.len());
        for y in 0..h as isize {
            for x in 0..w as isize {
                for c in 0..C {
                    let mut win = Vec::with_capacity(ksize * ksize);
                    for dy in -r..=r {
                        for dx in -r..=r {
                            let sy = (y + dy).clamp(0, h as isize - 1) as usize;
                            let sx = (x + dx).clamp(0, w as isize - 1) as usize;
                            win.push(src[(sy * w + sx) * C + c]);
                        }
                    }
                    win.sort_by_key(|v| v.key());
                    out.push(win[win.len() / 2]);
                }
            }
        }
        out
    }

    fn lcg(state: &mut u64) -> u32 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 33) as u32
    }

    /// Shapes covering one strip, several strips with a partial last one,
    /// widths below a column block and below a lane block, single rows and
    /// columns.
    const SHAPES: [(usize, usize); 9] = [
        (1, 1),
        (2, 3),
        (5, 1),
        (1, 7),
        (7, 6),
        (17, 5),
        (40, 9),
        (256, 4),
        (531, 6),
    ];

    /// Runs `run` on random images from `gen` and compares keys (a
    /// bijection of the bits, so NaNs compare exactly) with [`naive`].
    fn check<T: MedianKey, const C: usize>(
        gen: impl Fn(u32) -> T,
        run: impl Fn(&[T], &mut [T], usize, usize, usize),
    ) where
        T::Key: std::fmt::Debug,
    {
        let keys = |v: &[T]| v.iter().map(|s| s.key()).collect::<Vec<_>>();
        let mut state = 0x5eed_u64;
        for (w, h) in SHAPES {
            let src: Vec<T> = (0..w * h * C).map(|_| gen(lcg(&mut state))).collect();
            for ksize in [3, 5] {
                let mut dst = vec![gen(0); src.len()];
                run(&src, &mut dst, w, h, ksize);
                assert_eq!(
                    keys(&dst),
                    keys(&naive::<T, C>(&src, w, h, ksize)),
                    "{w}x{h} C{C} k{ksize}"
                );
            }
        }
    }

    #[test]
    fn u16_matches_naive_median() {
        // Full range and a narrow range (many ties).
        check::<u16, 1>(|r| r as u16, median_blur_u16::<1>);
        check::<u16, 1>(|r| (r % 4) as u16, median_blur_u16::<1>);
        check::<u16, 3>(|r| r as u16, median_blur_u16::<3>);
    }

    #[test]
    fn f32_matches_naive_median_bit_exact() {
        // Finite values of both signs, both zeros, ±inf and NaNs of both
        // signs: the output is the total-order median, bit for bit.
        const SPECIAL: [f32; 6] = [
            0.0,
            -0.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            -f32::NAN,
        ];
        let gen = |r: u32| {
            if r.is_multiple_of(7) {
                SPECIAL[(r as usize / 7) % SPECIAL.len()]
            } else {
                (r as f32 / u32::MAX as f32 - 0.5) * 100.0
            }
        };
        check::<f32, 1>(gen, median_blur_f32::<1>);
        check::<f32, 3>(gen, median_blur_f32::<3>);
        // Narrow range: many ties, signed zeros among them.
        check::<f32, 1>(
            |r| SPECIAL[r as usize % 2] + (r % 3) as f32,
            median_blur_f32::<1>,
        );
    }

    /// The plain build of the row kernel (the SSE2 / NEON-autovectorized
    /// path), row by row; the entry points above take the AVX2 build on
    /// AVX2 hosts.
    fn plain_rows<T: MedianKey, const C: usize>(
        src: &[T],
        dst: &mut [T],
        w: usize,
        h: usize,
        ksize: usize,
    ) {
        for (y, drow) in dst.chunks_mut(w * C).enumerate() {
            if ksize == 3 {
                median_row::<T, C, 3, 8>(src, drow, w, h, y);
            } else {
                median_row::<T, C, 5, 8>(src, drow, w, h, y);
            }
        }
    }

    #[test]
    fn plain_row_kernel_matches_naive_median() {
        check::<u16, 1>(|r| r as u16, plain_rows::<u16, 1>);
        check::<u16, 3>(|r| r as u16, plain_rows::<u16, 3>);
        check::<f32, 1>(f32::from_bits, plain_rows::<f32, 1>);
        check::<f32, 3>(f32::from_bits, plain_rows::<f32, 3>);
    }

    #[test]
    fn f32_key_is_total_order() {
        let v = [
            -f32::NAN,
            f32::NEG_INFINITY,
            -1.5,
            -f32::MIN_POSITIVE,
            -0.0,
            0.0,
            f32::MIN_POSITIVE,
            2.0,
            f32::INFINITY,
            f32::NAN,
        ];
        for pair in v.windows(2) {
            assert!(pair[0].key() < pair[1].key(), "{pair:?}");
            assert_eq!(pair[0].total_cmp(&pair[1]), std::cmp::Ordering::Less);
        }
        for x in v {
            assert_eq!(f32::from_key(x.key()).to_bits(), x.to_bits());
        }
    }
}
