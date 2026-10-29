//! CPU kernels for the valid-aware depth median.
//!
//! Invalid samples are mapped to a sort key above every valid key (see
//! [`DepthRepr::key`]), so the median of the `n` valid samples of a window
//! is its order statistic of rank `n / 2`.
//!
//! Each output row is produced in two passes over a per-task scratch:
//!
//! 1. **Vertical**: every source column of the `ksize` window rows is
//!    sorted ([`COL3`] / [`COL5`]) and its valid count recorded, once per
//!    column. The `ksize` horizontally adjacent windows that share a column
//!    reuse it.
//! 2. **Horizontal**: a window is `ksize` adjacent sorted columns; a
//!    rank-selection network ([`RANK9`] / [`RANK25`]) brings ranks
//!    `0..=taps / 2` (every rank `n / 2` can take) into place, and rank
//!    `n / 2` is selected lane-wise.
//!
//! In the NEON and CUDA kernels, a block of windows whose samples are all
//! valid (the common case away from holes) takes a plain-median network
//! instead ([`med9`] /
//! [`crate::filter::median::MED25_SORTED_COLUMNS`]): rank
//! `taps / 2` is the same value, for fewer exchanges.
//!
//! On aarch64, single-channel rows of at least one vector run explicit NEON
//! kernels (`u16x8` / `f32x4` per wire, module `neon`). Every other case
//! runs the portable kernels: the same passes on `[K; L]` lane arrays with
//! straight-line compare-exchanges, compiled with AVX2 enabled through a
//! `target_feature` wrapper on x86_64 CPUs that have it. Both are
//! bit-identical (the selections are exact; pinned in the tests). The CUDA
//! kernels in `cuda/depth.rs` are code-generated from the same networks.

use rayon::prelude::*;

/// Widest lane block any kernel uses; the scratch is padded to it.
const MAX_LANES: usize = 16;

/// Width of the vertical-pass blocks.
const COL_BLOCK: usize = 16;

/// Defines a comparator network as a `const` pair list plus an
/// `#[inline(always)]` function that applies it, fully unrolled, to an
/// array of lane vectors. The pair list feeds the tests and the CUDA code
/// generator, the function feeds the CPU kernels, from one literal list.
macro_rules! network {
    ($(#[$doc:meta])* $name:ident, $(#[$fattr:meta])* $apply:ident, [$(($a:literal, $b:literal)),* $(,)?]) => {
        $(#[$doc])*
        // Read by the tests and the CUDA code generator.
        #[cfg_attr(not(any(test, feature = "cuda")), allow(dead_code))]
        pub(crate) const $name: &[(usize, usize)] = &[$(($a, $b)),*];

        network_fn!($(#[$fattr])* $apply, [$(($a, $b)),*]);
    };
}

/// The function half of [`network!`], for a pair list defined elsewhere.
macro_rules! network_fn {
    ($(#[$fattr:meta])* $apply:ident, [$(($a:literal, $b:literal)),* $(,)?]) => {
        $(#[$fattr])*
        #[inline(always)]
        fn $apply<V: Exchange, const N: usize>(v: &mut [V; N]) {
            $(ce(v, $a, $b);)*
        }
    };
}

network!(
    /// Sorts a 3-sample column.
    COL3,
    sort_col3,
    [(0, 1), (1, 2), (0, 1)]
);

network!(
    /// Sorts a 5-sample column (optimal, 9 exchanges; Knuth, TAOCP vol. 3,
    /// 5.3.4).
    COL5,
    sort_col5,
    [(0, 1), (3, 4), (2, 4), (2, 3), (1, 4), (0, 3), (0, 2), (1, 3), (1, 2)]
);

network!(
    /// Rank selection on a 3×3 window held as 3 sorted columns (wire
    /// `col * 3 + rank`): afterwards wires `0..=4` hold ranks `0..=4`.
    /// Derived from Batcher's odd-even merge sort (Knuth, TAOCP vol. 3,
    /// 5.3.4) by greedy comparator deletion, and proven exact on every
    /// column-sorted input by the zero-one principle (see the tests).
    RANK9,
    rank9,
    [
        (2, 3), (0, 2), (4, 6), (5, 7), (1, 2), (5, 6), (0, 4), (1, 5), (2, 6),
        (3, 7), (2, 4), (3, 5), (1, 2), (3, 4), (4, 8), (2, 4), (3, 4),
    ]
);

network!(
    /// Rank selection on a 5×5 window held as 5 sorted columns (wire
    /// `col * 5 + rank`): afterwards wires `0..=12` hold ranks `0..=12`.
    /// Derived like [`RANK9`] and proven the same way.
    RANK25,
    rank25,
    [
        (4, 5), (14, 15), (5, 7), (8, 10), (9, 11), (12, 14), (5, 6), (9, 10),
        (13, 14), (0, 4), (1, 5), (2, 6), (8, 12), (10, 14), (11, 15), (16, 20),
        (17, 21), (18, 22), (19, 23), (2, 4), (3, 5), (10, 12), (11, 13), (18, 20),
        (19, 21), (1, 2), (3, 4), (5, 6), (9, 10), (11, 12), (13, 14), (17, 18),
        (19, 20), (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14),
        (7, 15), (4, 8), (5, 9), (6, 10), (7, 11), (20, 24), (2, 4), (3, 5),
        (6, 8), (7, 9), (10, 12), (11, 13), (22, 24), (1, 2), (3, 4), (5, 6),
        (7, 8), (9, 10), (11, 12), (21, 22), (23, 24), (0, 16), (1, 17), (2, 18),
        (3, 19), (4, 20), (5, 21), (6, 22), (7, 23), (8, 24), (8, 16), (9, 17),
        (10, 18), (11, 19), (12, 20), (13, 21), (4, 8), (5, 9), (6, 10), (7, 11),
        (12, 16), (13, 17), (2, 4), (3, 5), (6, 8), (7, 9), (10, 12), (11, 13),
        (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12),
    ]
);

crate::filter::median::with_med25_sorted_columns!(network_fn!(
    /// Median (wire 12) of a 5×5 window held as 5 sorted columns (wire
    /// `col * 5 + rank`) when all 25 samples are valid:
    /// [`crate::filter::median::MED25_SORTED_COLUMNS`], the sorted-column
    /// network of the `u8` [`crate::filter::median_blur`] 5×5 kernel,
    /// unrolled.
    // Only the NEON kernel takes the all-valid shortcut.
    #[cfg(target_arch = "aarch64")]
    med25,
));

/// Median of a 3×3 window held as 3 sorted columns (wire `col * 3 +
/// rank`) when all 9 samples are valid: the exact identity
/// `med3(max(lows), med3(mids), min(highs))` (the `u8`
/// [`crate::filter::median_blur`] 3×3 kernel uses it too).
#[cfg(any(test, target_arch = "aarch64"))]
#[inline(always)]
fn med9<V: Exchange>(v: &[V; 9]) -> V {
    let max = |a, b| V::exchange(a, b).1;
    let min = |a, b| V::exchange(a, b).0;
    let med3 = |a, b, c| {
        let (lo, hi) = V::exchange(a, b);
        max(lo, min(hi, c))
    };
    med3(
        max(max(v[0], v[3]), v[6]),
        med3(v[1], v[4], v[7]),
        min(min(v[2], v[5]), v[8]),
    )
}

/// A vector of sort keys with a lane-wise compare-exchange.
trait Exchange: Copy {
    /// Lane-wise `(min(a, b), max(a, b))`.
    fn exchange(a: Self, b: Self) -> (Self, Self);
}

/// Compare-exchange of wires `a` and `b`: afterwards `v[a] <= v[b]` in
/// every lane.
#[inline(always)]
fn ce<V: Exchange, const N: usize>(v: &mut [V; N], a: usize, b: usize) {
    let (lo, hi) = V::exchange(v[a], v[b]);
    v[a] = lo;
    v[b] = hi;
}

/// `L` sort keys of dtype `T`, for the portable kernels.
struct Lanes<T: DepthRepr, const L: usize>([T::Key; L]);

impl<T: DepthRepr, const L: usize> Clone for Lanes<T, L> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: DepthRepr, const L: usize> Copy for Lanes<T, L> {}

impl<T: DepthRepr, const L: usize> Exchange for Lanes<T, L> {
    #[inline(always)]
    fn exchange(a: Self, b: Self) -> (Self, Self) {
        (
            Lanes(std::array::from_fn(|l| T::kmin(a.0[l], b.0[l]))),
            Lanes(std::array::from_fn(|l| T::kmax(a.0[l], b.0[l]))),
        )
    }
}

/// Per-dtype depth semantics: which value marks a missing measurement and
/// how samples map to an order-preserving sort key in which invalid samples
/// sort after every valid one.
pub(crate) trait DepthRepr: Copy + Send + Sync + 'static {
    /// Order-preserving sort key.
    type Key: Copy + Send + Sync;
    /// The value that marks a missing measurement.
    const INVALID: Self;
    /// Lanes per block of the 3×3 kernel.
    const LANES3: usize;
    /// Lanes per block of the 5×5 kernel.
    const LANES5: usize;
    /// True when the sample is a measurement: the public rule
    /// ([`super::is_valid_depth_u16`] / [`super::is_valid_depth_f32`]).
    fn is_valid(self) -> bool;
    /// Sort key; invalid samples map to the largest key.
    fn key(self) -> Self::Key;
    /// Inverse of [`Self::key`] for valid samples.
    fn from_key(k: Self::Key) -> Self;
    /// Smaller of two keys.
    fn kmin(a: Self::Key, b: Self::Key) -> Self::Key;
    /// Larger of two keys.
    fn kmax(a: Self::Key, b: Self::Key) -> Self::Key;
    /// Valid-aware median over an interleaved `h × w × C` buffer at the
    /// dtype's lane widths. `ksize` is 3 or 5.
    fn median_blur<const C: usize>(
        src: &[Self],
        dst: &mut [Self],
        w: usize,
        h: usize,
        ksize: usize,
    );
}

impl DepthRepr for u16 {
    type Key = u16;
    const INVALID: u16 = 0;
    // Portable-kernel block widths: 9 or 25 wires of `L` keys stay within
    // the vector register file.
    const LANES3: usize = 16;
    const LANES5: usize = 8;
    #[inline(always)]
    fn is_valid(self) -> bool {
        super::is_valid_depth_u16(self)
    }
    // 0 -> u16::MAX, 1..=65535 -> 0..=65534: order-preserving on valid
    // values, and the invalid marker becomes the unique largest key, whose
    // inverse is the invalid marker again.
    #[inline(always)]
    fn key(self) -> u16 {
        self.wrapping_sub(1)
    }
    #[inline(always)]
    fn from_key(k: u16) -> u16 {
        k.wrapping_add(1)
    }
    #[inline(always)]
    fn kmin(a: u16, b: u16) -> u16 {
        a.min(b)
    }
    #[inline(always)]
    fn kmax(a: u16, b: u16) -> u16 {
        a.max(b)
    }
    fn median_blur<const C: usize>(src: &[u16], dst: &mut [u16], w: usize, h: usize, ksize: usize) {
        if ksize == 3 {
            let simd = simd_median_row::<u16, C, 3, 9>();
            median_blur_ks::<u16, C, 3, 9, { u16::LANES3 }>(src, dst, w, h, simd);
        } else {
            let simd = simd_median_row::<u16, C, 5, 25>();
            median_blur_ks::<u16, C, 5, 25, { u16::LANES5 }>(src, dst, w, h, simd);
        }
    }
}

impl DepthRepr for f32 {
    type Key = f32;
    const INVALID: f32 = f32::NAN;
    const LANES3: usize = 8;
    const LANES5: usize = 4;
    #[inline(always)]
    fn is_valid(self) -> bool {
        super::is_valid_depth_f32(self)
    }
    // Every invalid sample (NaN, ±inf, zero, negative) -> +inf, above every
    // valid (finite, positive) key. Valid keys are the samples themselves:
    // no valid sample is `-0.0`, so equal keys have equal bits and every
    // exact selection (any network, CPU or CUDA) returns the same bits.
    #[inline(always)]
    fn key(self) -> f32 {
        if self.is_valid() {
            self
        } else {
            f32::INFINITY
        }
    }
    #[inline(always)]
    fn from_key(k: f32) -> f32 {
        k
    }
    #[inline(always)]
    fn kmin(a: f32, b: f32) -> f32 {
        a.min(b)
    }
    #[inline(always)]
    fn kmax(a: f32, b: f32) -> f32 {
        a.max(b)
    }
    fn median_blur<const C: usize>(src: &[f32], dst: &mut [f32], w: usize, h: usize, ksize: usize) {
        if ksize == 3 {
            let simd = simd_median_row::<f32, C, 3, 9>();
            median_blur_ks::<f32, C, 3, 9, { f32::LANES3 }>(src, dst, w, h, simd);
        } else {
            let simd = simd_median_row::<f32, C, 5, 25>();
            median_blur_ks::<f32, C, 5, 25, { f32::LANES5 }>(src, dst, w, h, simd);
        }
    }
}

/// Sorted columns of one output row's window rows, for one channel.
///
/// Rank `r` of padded column `p` lives at `rank[r * stride + p]`, its valid
/// count at `count[p]`; padded column `p` is image column `p - ksize / 2`,
/// replicated beyond the image edges.
struct Columns<K> {
    rank: Vec<K>,
    count: Vec<u16>,
    stride: usize,
}

impl<K: Copy> Columns<K> {
    fn new(w: usize, ksize: usize, fill: K) -> Self {
        // Horizontal blocks start below `w` and read `L + ksize - 1`
        // padded columns, `L` dividing `MAX_LANES`.
        let stride = w.div_ceil(MAX_LANES) * MAX_LANES + ksize;
        Self {
            rank: vec![fill; stride * ksize],
            count: vec![0; stride],
            stride,
        }
    }
}

/// `[T; B]` of channel `c` starting at column `x` of an interleaved row.
#[inline(always)]
fn load<T: Copy, const C: usize, const B: usize>(row: &[T], x: usize, c: usize) -> [T; B] {
    if C == 1 {
        let s = &row[x..x + B];
        std::array::from_fn(|l| s[l])
    } else {
        std::array::from_fn(|l| row[(x + l) * C + c])
    }
}

/// Vertical pass for columns `x..x + B`.
#[inline(always)]
fn sort_block<T: DepthRepr, const C: usize, const KS: usize, const B: usize>(
    rows: &[&[T]; KS],
    x: usize,
    c: usize,
    cols: &mut Columns<T::Key>,
) {
    let mut v = [Lanes::<T, B>([T::INVALID.key(); B]); KS];
    let mut n = [0u16; B];
    for (wire, row) in v.iter_mut().zip(rows) {
        let s = load::<T, C, B>(row, x, c);
        for l in 0..B {
            n[l] += u16::from(s[l].is_valid());
            wire.0[l] = s[l].key();
        }
    }
    if KS == 3 {
        sort_col3(&mut v);
    } else {
        sort_col5(&mut v);
    }
    let p = x + KS / 2;
    for (r, wire) in v.iter().enumerate() {
        let o = r * cols.stride + p;
        cols.rank[o..o + B].copy_from_slice(&wire.0);
    }
    cols.count[p..p + B].copy_from_slice(&n);
}

/// Vertical pass for a whole row (channel `c`), plus replicate padding.
#[inline(always)]
fn sort_columns<T: DepthRepr, const C: usize, const KS: usize>(
    src: &[T],
    w: usize,
    sy: &[usize; KS],
    c: usize,
    cols: &mut Columns<T::Key>,
) {
    let rows: [&[T]; KS] = std::array::from_fn(|i| &src[sy[i] * w * C..(sy[i] + 1) * w * C]);
    let mut x = 0;
    while x + COL_BLOCK <= w {
        sort_block::<T, C, KS, COL_BLOCK>(&rows, x, c, cols);
        x += COL_BLOCK;
    }
    while x < w {
        sort_block::<T, C, KS, 1>(&rows, x, c, cols);
        x += 1;
    }
    pad_columns(cols, w, KS / 2);
}

/// Replicate the edge columns into the `r` padding columns on each side.
#[inline(always)]
fn pad_columns<K: Copy>(cols: &mut Columns<K>, w: usize, r: usize) {
    let stride = cols.stride;
    for rank in cols.rank.chunks_exact_mut(stride) {
        let (first, last) = (rank[r], rank[w + r - 1]);
        rank[..r].fill(first);
        rank[w + r..w + 2 * r].fill(last);
    }
    let (first, last) = (cols.count[r], cols.count[w + r - 1]);
    cols.count[..r].fill(first);
    cols.count[w + r..w + 2 * r].fill(last);
}

/// Horizontal pass: rank-`n / 2` key and valid count `n` of the windows
/// centred on columns `x0..x0 + L`.
#[inline(always)]
fn window_medians<T: DepthRepr, const KS: usize, const TAPS: usize, const L: usize>(
    cols: &Columns<T::Key>,
    x0: usize,
) -> ([T::Key; L], [u16; L]) {
    let mut v = [Lanes::<T, L>([T::INVALID.key(); L]); TAPS];
    let mut n = [0u16; L];
    for dc in 0..KS {
        let cs = &cols.count[x0 + dc..x0 + dc + L];
        for l in 0..L {
            n[l] += cs[l];
        }
        for r in 0..KS {
            let o = r * cols.stride + x0 + dc;
            let s = &cols.rank[o..o + L];
            v[dc * KS + r] = Lanes(std::array::from_fn(|l| s[l]));
        }
    }
    if KS == 3 {
        rank9(&mut v);
    } else {
        rank25(&mut v);
    }
    let mut m = v[0].0;
    for (i, wire) in v.iter().enumerate().take(TAPS / 2 + 1).skip(1) {
        for l in 0..L {
            if usize::from(n[l] / 2) == i {
                m[l] = wire.0[l];
            }
        }
    }
    (m, n)
}

/// Source row indices of the window around `y`, replicate border.
#[inline(always)]
fn window_rows<const KS: usize>(y: usize, h: usize) -> [usize; KS] {
    std::array::from_fn(|i| (y + i).saturating_sub(KS / 2).min(h - 1))
}

/// Minimum rows per rayon task.
fn min_rows(h: usize) -> usize {
    h.div_ceil(rayon::current_num_threads() * 4).max(16)
}

/// One output row of the median.
#[inline(always)]
fn median_row<T: DepthRepr, const C: usize, const KS: usize, const TAPS: usize, const L: usize>(
    src: &[T],
    drow: &mut [T],
    w: usize,
    h: usize,
    y: usize,
    cols: &mut Columns<T::Key>,
) {
    let sy = window_rows::<KS>(y, h);
    for c in 0..C {
        sort_columns::<T, C, KS>(src, w, &sy, c, cols);
        for x0 in (0..w).step_by(L) {
            let (m, n) = window_medians::<T, KS, TAPS, L>(cols, x0);
            let out: [T; L] = std::array::from_fn(|l| {
                if n[l] == 0 {
                    T::INVALID
                } else {
                    T::from_key(m[l])
                }
            });
            let lanes = L.min(w - x0);
            if C == 1 && lanes == L {
                drow[x0..x0 + L].copy_from_slice(&out);
            } else {
                for (l, &o) in out.iter().enumerate().take(lanes) {
                    drow[(x0 + l) * C + c] = o;
                }
            }
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
unsafe fn median_row_avx2<
    T: DepthRepr,
    const C: usize,
    const KS: usize,
    const TAPS: usize,
    const L: usize,
>(
    src: &[T],
    drow: &mut [T],
    w: usize,
    h: usize,
    y: usize,
    cols: &mut Columns<T::Key>,
) {
    median_row::<T, C, KS, TAPS, L>(src, drow, w, h, y, cols);
}

/// A SIMD row kernel: computes one output row and returns `true`, or
/// returns `false` (writing nothing) for a shape it does not cover.
type MedianRowFn<T> =
    fn(&[T], &mut [T], usize, usize, usize, &mut Columns<<T as DepthRepr>::Key>) -> bool;

/// The SIMD median row kernel for this dtype / layout, if any.
#[allow(unreachable_code)]
fn simd_median_row<T: SimdDepth, const C: usize, const KS: usize, const TAPS: usize>(
) -> Option<MedianRowFn<T>> {
    #[cfg(target_arch = "aarch64")]
    if C == 1 {
        return Some(neon::median_row::<T, KS, TAPS>);
    }
    None
}

/// Valid-aware median blur, `KS × KS` window, `L`-lane blocks; rows go to
/// `simd` when it covers them, else to the portable [`median_row`].
fn median_blur_ks<
    T: DepthRepr,
    const C: usize,
    const KS: usize,
    const TAPS: usize,
    const L: usize,
>(
    src: &[T],
    dst: &mut [T],
    w: usize,
    h: usize,
    simd: Option<MedianRowFn<T>>,
) {
    dst.par_chunks_mut(w * C)
        .enumerate()
        .with_min_len(min_rows(h))
        .for_each_init(
            || Columns::new(w, KS, T::INVALID.key()),
            |cols, (y, drow)| {
                if simd.is_some_and(|f| f(src, drow, w, h, y, cols)) {
                    return;
                }
                #[cfg(target_arch = "x86_64")]
                if crate::simd::cpu_features().has_avx2 {
                    // SAFETY: AVX2 support was just checked.
                    unsafe { median_row_avx2::<T, C, KS, TAPS, L>(src, drow, w, h, y, cols) };
                    return;
                }
                median_row::<T, C, KS, TAPS, L>(src, drow, w, h, y, cols);
            },
        );
}

/// Valid-aware median blur over an interleaved `h × w × C` buffer. `ksize`
/// is 3 or 5 (validated by the caller). An empty image is a no-op.
pub(crate) fn median_blur_depth<T: DepthRepr, const C: usize>(
    src: &[T],
    dst: &mut [T],
    w: usize,
    h: usize,
    ksize: usize,
) {
    // The row kernels chunk by `w * C` and clamp to row `h - 1`.
    if w == 0 || h == 0 {
        return;
    }
    T::median_blur::<C>(src, dst, w, h, ksize);
}

/// Dtypes with a SIMD row kernel on this architecture.
#[cfg(target_arch = "aarch64")]
use neon::NeonDepth as SimdDepth;

/// Dtypes with a SIMD row kernel on this architecture (none here).
#[cfg(not(target_arch = "aarch64"))]
trait SimdDepth: DepthRepr {}
#[cfg(not(target_arch = "aarch64"))]
impl SimdDepth for u16 {}
#[cfg(not(target_arch = "aarch64"))]
impl SimdDepth for f32 {}

/// NEON row kernels (single channel): the two passes of the portable
/// kernels on one 128-bit vector per wire, `u16x8` or `f32x4`. Same
/// networks, same keys, so bit-identical to [`median_row`] (pinned in the
/// tests).
#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    use super::{
        med25, med9, pad_columns, rank25, rank9, sort_col3, sort_col5, Columns, DepthRepr, Exchange,
    };

    /// `u16x8` sort keys.
    #[derive(Clone, Copy)]
    pub(crate) struct U16x8(uint16x8_t);

    impl Exchange for U16x8 {
        #[inline(always)]
        fn exchange(a: Self, b: Self) -> (Self, Self) {
            // SAFETY: NEON is architectural on aarch64; register-only ops.
            unsafe { (U16x8(vminq_u16(a.0, b.0)), U16x8(vmaxq_u16(a.0, b.0))) }
        }
    }

    /// `f32x4` sort keys (never NaN or negative, see [`DepthRepr::key`]).
    #[derive(Clone, Copy)]
    pub(crate) struct F32x4(float32x4_t);

    impl Exchange for F32x4 {
        #[inline(always)]
        fn exchange(a: Self, b: Self) -> (Self, Self) {
            // SAFETY: NEON is architectural on aarch64; register-only ops.
            // `minnm` / `maxnm` are `f32::min` / `f32::max`.
            unsafe { (F32x4(vminnmq_f32(a.0, b.0)), F32x4(vmaxnmq_f32(a.0, b.0))) }
        }
    }

    /// Per-dtype NEON spelling of [`DepthRepr`]: `W` samples per vector,
    /// valid counts as `u16` lanes (`N`).
    pub(crate) trait NeonDepth: DepthRepr {
        /// Vector of `W` sort keys.
        type V: Exchange;
        /// Vector of `W` `u16` valid counts.
        type N: Copy;
        /// Samples per vector.
        const W: usize;
        /// All-invalid keys.
        fn v_fill() -> Self::V;
        /// Zero counts.
        fn n_zero() -> Self::N;
        /// Lane-wise `a + b`.
        fn n_add(a: Self::N, b: Self::N) -> Self::N;
        /// True when every lane of `n` equals `taps`.
        fn all_eq(n: Self::N, taps: u16) -> bool;
        /// Keys of the `W` samples at `p`, and `n` plus their valid flags.
        ///
        /// # Safety
        ///
        /// `p` is valid for reading `W` samples.
        unsafe fn load_src(p: *const Self, n: Self::N) -> (Self::V, Self::N);
        /// # Safety
        ///
        /// `p` is valid for reading `W` keys.
        unsafe fn ld_key(p: *const Self::Key) -> Self::V;
        /// # Safety
        ///
        /// `p` is valid for writing `W` keys.
        unsafe fn st_key(p: *mut Self::Key, v: Self::V);
        /// # Safety
        ///
        /// `p` is valid for reading `W` counts.
        unsafe fn ld_n(p: *const u16) -> Self::N;
        /// # Safety
        ///
        /// `p` is valid for writing `W` counts.
        unsafe fn st_n(p: *mut u16, n: Self::N);
        /// Lane-wise `v[n / 2]` for `n <= TAPS`.
        fn select<const TAPS: usize>(v: &[Self::V; TAPS], n: Self::N) -> Self::V;
        /// Median output: `from_key(m)`, or invalid where `n == 0`.
        ///
        /// # Safety
        ///
        /// `p` is valid for writing `W` samples.
        unsafe fn st_out(p: *mut Self, m: Self::V, n: Self::N);
    }

    impl NeonDepth for u16 {
        type V = U16x8;
        type N = uint16x8_t;
        const W: usize = 8;
        #[inline(always)]
        fn v_fill() -> U16x8 {
            // SAFETY: NEON is architectural on aarch64; register-only op.
            U16x8(unsafe { vdupq_n_u16(u16::MAX) })
        }
        #[inline(always)]
        fn n_zero() -> uint16x8_t {
            // SAFETY: as above.
            unsafe { vdupq_n_u16(0) }
        }
        #[inline(always)]
        fn n_add(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
            // SAFETY: as above.
            unsafe { vaddq_u16(a, b) }
        }
        #[inline(always)]
        fn all_eq(n: uint16x8_t, taps: u16) -> bool {
            // SAFETY: as above.
            unsafe { vminvq_u16(n) == taps }
        }
        #[inline(always)]
        unsafe fn load_src(p: *const u16, n: uint16x8_t) -> (U16x8, uint16x8_t) {
            let v = vld1q_u16(p);
            // key = v - 1 (wrapping); a non-zero lane counts as valid
            // (vtst gives all-ones, i.e. -1).
            (
                U16x8(vsubq_u16(v, vdupq_n_u16(1))),
                vsubq_u16(n, vtstq_u16(v, v)),
            )
        }
        #[inline(always)]
        unsafe fn ld_key(p: *const u16) -> U16x8 {
            U16x8(vld1q_u16(p))
        }
        #[inline(always)]
        unsafe fn st_key(p: *mut u16, v: U16x8) {
            vst1q_u16(p, v.0)
        }
        #[inline(always)]
        unsafe fn ld_n(p: *const u16) -> uint16x8_t {
            vld1q_u16(p)
        }
        #[inline(always)]
        unsafe fn st_n(p: *mut u16, n: uint16x8_t) {
            vst1q_u16(p, n)
        }
        #[inline(always)]
        fn select<const TAPS: usize>(v: &[U16x8; TAPS], n: uint16x8_t) -> U16x8 {
            // SAFETY: NEON is architectural on aarch64; register-only ops.
            unsafe {
                let k = vshrq_n_u16::<1>(n);
                let mut m = v[0].0;
                for (i, wire) in v.iter().enumerate().take(TAPS / 2 + 1).skip(1) {
                    let hit = vceqq_u16(k, vdupq_n_u16(i as u16));
                    m = vbslq_u16(hit, wire.0, m);
                }
                U16x8(m)
            }
        }
        #[inline(always)]
        unsafe fn st_out(p: *mut u16, m: U16x8, _n: uint16x8_t) {
            // from_key; an all-invalid window selects key u16::MAX, whose
            // inverse is the invalid marker 0.
            vst1q_u16(p, vaddq_u16(m.0, vdupq_n_u16(1)))
        }
    }

    /// Lane-wise [`super::super::is_valid_depth_f32`]: `0 < v < +inf`, all
    /// ones where valid. Both compares are false for NaN.
    #[inline(always)]
    fn valid_f32(v: float32x4_t) -> uint32x4_t {
        // SAFETY: NEON is architectural on aarch64; register-only ops.
        unsafe {
            vandq_u32(
                vcgtq_f32(v, vdupq_n_f32(0.0)),
                vcltq_f32(v, vdupq_n_f32(f32::INFINITY)),
            )
        }
    }

    impl NeonDepth for f32 {
        type V = F32x4;
        type N = uint16x4_t;
        const W: usize = 4;
        #[inline(always)]
        fn v_fill() -> F32x4 {
            // SAFETY: NEON is architectural on aarch64; register-only op.
            F32x4(unsafe { vdupq_n_f32(f32::INFINITY) })
        }
        #[inline(always)]
        fn n_zero() -> uint16x4_t {
            // SAFETY: as above.
            unsafe { vdup_n_u16(0) }
        }
        #[inline(always)]
        fn n_add(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
            // SAFETY: as above.
            unsafe { vadd_u16(a, b) }
        }
        #[inline(always)]
        fn all_eq(n: uint16x4_t, taps: u16) -> bool {
            // SAFETY: as above.
            unsafe { vminv_u16(n) == taps }
        }
        #[inline(always)]
        unsafe fn load_src(p: *const f32, n: uint16x4_t) -> (F32x4, uint16x4_t) {
            let v = vld1q_f32(p);
            let valid = valid_f32(v);
            // Invalid -> +inf, valid samples are their own key, as
            // `DepthRepr::key`.
            let key = vbslq_f32(valid, v, vdupq_n_f32(f32::INFINITY));
            (F32x4(key), vsub_u16(n, vmovn_u32(valid)))
        }
        #[inline(always)]
        unsafe fn ld_key(p: *const f32) -> F32x4 {
            F32x4(vld1q_f32(p))
        }
        #[inline(always)]
        unsafe fn st_key(p: *mut f32, v: F32x4) {
            vst1q_f32(p, v.0)
        }
        #[inline(always)]
        unsafe fn ld_n(p: *const u16) -> uint16x4_t {
            vld1_u16(p)
        }
        #[inline(always)]
        unsafe fn st_n(p: *mut u16, n: uint16x4_t) {
            vst1_u16(p, n)
        }
        #[inline(always)]
        fn select<const TAPS: usize>(v: &[F32x4; TAPS], n: uint16x4_t) -> F32x4 {
            // SAFETY: NEON is architectural on aarch64; register-only ops.
            unsafe {
                let k = vmovl_u16(vshr_n_u16::<1>(n));
                let mut m = v[0].0;
                for (i, wire) in v.iter().enumerate().take(TAPS / 2 + 1).skip(1) {
                    let hit = vceqq_u32(k, vdupq_n_u32(i as u32));
                    m = vbslq_f32(hit, wire.0, m);
                }
                F32x4(m)
            }
        }
        #[inline(always)]
        unsafe fn st_out(p: *mut f32, m: F32x4, n: uint16x4_t) {
            let empty = vceqzq_u32(vmovl_u16(n));
            vst1q_f32(p, vbslq_f32(empty, vdupq_n_f32(f32::NAN), m.0))
        }
    }

    /// Vertical pass for a whole row. Requires `w >= T::W`.
    ///
    /// # Safety
    ///
    /// `src` holds `h * w` samples, `sy` are rows `< h`, `w >= T::W`, and
    /// `cols` was built for width `w` and window `KS`.
    #[inline(always)]
    unsafe fn sort_columns<T: NeonDepth, const KS: usize>(
        src: &[T],
        w: usize,
        sy: &[usize; KS],
        cols: &mut Columns<T::Key>,
    ) {
        let r = KS / 2;
        let stride = cols.stride;
        let rank = cols.rank.as_mut_ptr();
        let count = cols.count.as_mut_ptr();
        let mut x = 0;
        while x < w {
            // The last block overlaps the previous one instead of running
            // past the row end; recomputing columns is idempotent.
            let xs = x.min(w - T::W);
            let mut n = T::n_zero();
            let mut v = [T::v_fill(); KS];
            for (wire, &row) in v.iter_mut().zip(sy) {
                let (k, n2) = T::load_src(src.as_ptr().add(row * w + xs), n);
                *wire = k;
                n = n2;
            }
            if KS == 3 {
                sort_col3(&mut v);
            } else {
                sort_col5(&mut v);
            }
            for (i, &wire) in v.iter().enumerate() {
                T::st_key(rank.add(i * stride + xs + r), wire);
            }
            T::st_n(count.add(xs + r), n);
            x += T::W;
        }
        pad_columns(cols, w, r);
    }

    /// Horizontal pass: rank-`n / 2` keys and valid counts of the windows
    /// centred on columns `x0..x0 + T::W`.
    ///
    /// # Safety
    ///
    /// `x0 < w` for the width `cols` was built for.
    #[inline(always)]
    unsafe fn window<T: NeonDepth, const KS: usize, const TAPS: usize>(
        cols: &Columns<T::Key>,
        x0: usize,
    ) -> (T::V, T::N) {
        let stride = cols.stride;
        let rank = cols.rank.as_ptr();
        let count = cols.count.as_ptr();
        let mut v = [T::v_fill(); TAPS];
        let mut n = T::n_zero();
        for dc in 0..KS {
            n = T::n_add(n, T::ld_n(count.add(x0 + dc)));
            for r in 0..KS {
                v[dc * KS + r] = T::ld_key(rank.add(r * stride + x0 + dc));
            }
        }
        // Fully valid windows (the common case away from holes) only need
        // the plain median, rank `TAPS / 2`; same value, fewer exchanges.
        if T::all_eq(n, TAPS as u16) {
            let m = if KS == 3 {
                let w9: [T::V; 9] = std::array::from_fn(|i| v[i]);
                med9(&w9)
            } else {
                let mut v = v;
                med25(&mut v);
                v[TAPS / 2]
            };
            return (m, n);
        }
        if KS == 3 {
            rank9(&mut v);
        } else {
            rank25(&mut v);
        }
        (T::select(&v, n), n)
    }

    /// One single-channel output row of the median; `false` (nothing
    /// written) when `w < T::W`.
    pub(super) fn median_row<T: NeonDepth, const KS: usize, const TAPS: usize>(
        src: &[T],
        drow: &mut [T],
        w: usize,
        h: usize,
        y: usize,
        cols: &mut Columns<T::Key>,
    ) -> bool {
        if w < T::W {
            return false;
        }
        let sy = super::window_rows::<KS>(y, h);
        // SAFETY: `src` is the `h × w` image, `drow` one `w`-sample row,
        // `sy` rows are `< h` and `w >= T::W`; every block starts at
        // `x0 <= w - T::W`, so its `T::W` outputs lie inside `drow`, and
        // `cols` (built for `w`, `KS`) is padded for the horizontal reads.
        unsafe {
            sort_columns::<T, KS>(src, w, &sy, cols);
            let mut x = 0;
            while x < w {
                let x0 = x.min(w - T::W);
                let (m, n) = window::<T, KS, TAPS>(cols, x0);
                T::st_out(drow.as_mut_ptr().add(x0), m, n);
                x += T::W;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bit-parallel 0/1 simulation: `inputs[j]` is input `j` (bit `i` =
    /// wire `i`). Returns the wires after `net`, bit `j` of wire `i` being
    /// wire `i` of input `j`, for up to 64 inputs at once.
    fn run_zero_one(net: &[(usize, usize)], n: usize, inputs: &[u64]) -> Vec<u64> {
        let mut wires = vec![0u64; n];
        for (j, &input) in inputs.iter().enumerate() {
            for (i, wire) in wires.iter_mut().enumerate() {
                if (input >> i) & 1 == 1 {
                    *wire |= 1 << j;
                }
            }
        }
        for &(a, b) in net {
            let (lo, hi) = (wires[a] & wires[b], wires[a] | wires[b]);
            wires[a] = lo;
            wires[b] = hi;
        }
        wires
    }

    /// Every 0/1 input whose `k` columns of `k` wires (`col * k + rank`)
    /// are each sorted ascending.
    fn column_sorted_inputs(k: usize) -> Vec<u64> {
        let mut out = Vec::new();
        let total = (k + 1).pow(k as u32);
        for code in 0..total {
            let (mut rest, mut bits) = (code, 0u64);
            for col in 0..k {
                let zeros = rest % (k + 1);
                rest /= k + 1;
                for rank in zeros..k {
                    bits |= 1 << (col * k + rank);
                }
            }
            out.push(bits);
        }
        out
    }

    /// Zero-one principle (Knuth 5.3.4), selection form: if, for every 0/1
    /// input of a set closed under monotone maps, wire `i <= max_rank`
    /// holds the rank-`i` order statistic, it does so for every real input
    /// of that set. Column-sorted inputs are such a set.
    fn assert_rank_network(net: &[(usize, usize)], k: usize) {
        let n = k * k;
        assert!(net.iter().all(|&(a, b)| a < b && b < n));
        let max_rank = n / 2;
        for chunk in column_sorted_inputs(k).chunks(64) {
            let wires = run_zero_one(net, n, chunk);
            for (j, &input) in chunk.iter().enumerate() {
                let zeros = n - input.count_ones() as usize;
                for (i, wire) in wires.iter().enumerate().take(max_rank + 1) {
                    // Rank i of a sorted 0/1 sequence is 1 iff zeros <= i.
                    let want = u64::from(zeros <= i);
                    assert_eq!((wire >> j) & 1, want, "k={k} input={input:#x} rank={i}");
                }
            }
        }
    }

    #[test]
    fn rank_networks_select_lower_half_of_sorted_columns() {
        assert_rank_network(RANK9, 3);
        assert_rank_network(RANK25, 5);
    }

    #[test]
    fn median_networks_exact_on_full_sorted_columns() {
        // MED25_SORTED_COLUMNS: wire 12 is the rank-12 order statistic.
        let n = 25;
        for chunk in column_sorted_inputs(5).chunks(64) {
            let wires = run_zero_one(&crate::filter::median::MED25_SORTED_COLUMNS, n, chunk);
            for (j, &input) in chunk.iter().enumerate() {
                let zeros = n - input.count_ones() as usize;
                assert_eq!((wires[12] >> j) & 1, u64::from(zeros <= 12), "{input:#x}");
            }
        }
        // med9 on every column-sorted 0/1 input (min = AND, max = OR).
        #[derive(Clone, Copy)]
        struct Bit(bool);
        impl Exchange for Bit {
            fn exchange(a: Self, b: Self) -> (Self, Self) {
                (Bit(a.0 && b.0), Bit(a.0 || b.0))
            }
        }
        for input in column_sorted_inputs(3) {
            let v: [Bit; 9] = std::array::from_fn(|i| Bit((input >> i) & 1 == 1));
            let zeros = 9 - input.count_ones();
            assert_eq!(med9(&v).0, zeros <= 4, "{input:#x}");
        }
    }

    #[test]
    fn column_networks_sort_all_inputs() {
        for (net, n) in [(COL3, 3usize), (COL5, 5)] {
            let inputs: Vec<u64> = (0..1u64 << n).collect();
            let wires = run_zero_one(net, n, &inputs);
            for i in 1..n {
                // Sorted ascending: a 1 on wire i-1 implies a 1 on wire i.
                assert_eq!(wires[i - 1] & !wires[i], 0, "n={n}");
            }
        }
    }

    /// Deterministic LCG.
    fn lcg(state: &mut u64) -> u32 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 33) as u32
    }

    /// Depth with `holes_pct` % holes (`T::hole`) and duplicates.
    fn random_depth<T: TestDepth>(len: usize, seed: u64, holes_pct: u32) -> Vec<T> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                let r = lcg(&mut state);
                if r % 100 < holes_pct {
                    T::hole(r)
                } else {
                    T::sample(r)
                }
            })
            .collect()
    }

    /// Hole rates of the parity fixtures: mixed windows, mostly-valid
    /// windows, and fully valid images (the NEON all-valid shortcut).
    const HOLE_PCTS: [u32; 3] = [30, 2, 0];

    /// Test-side view of a depth dtype.
    trait TestDepth: SimdDepth + PartialOrd + std::fmt::Debug {
        /// A value no kernel ever writes (outside the sample range and not
        /// the canonical invalid marker), to catch unwritten outputs.
        const POISON: Self;
        /// A random valid sample, with many duplicates.
        fn sample(r: u32) -> Self;
        /// A random invalid sample.
        fn hole(r: u32) -> Self;
        /// Bit pattern, for exact comparison (NaN included).
        fn bits(self) -> u32;
    }

    impl TestDepth for u16 {
        const POISON: u16 = 9999;
        fn sample(r: u32) -> u16 {
            1 + (r >> 20) as u16 % 64
        }
        fn hole(_: u32) -> u16 {
            0
        }
        fn bits(self) -> u32 {
            u32::from(self)
        }
    }

    impl TestDepth for f32 {
        const POISON: f32 = -7.0;
        fn sample(r: u32) -> f32 {
            1.0 + ((r >> 8) % 16) as f32 * 0.25
        }
        /// Every invalid encoding, not only the canonical NaN.
        fn hole(r: u32) -> f32 {
            const HOLES: [f32; 6] = [0.0, -0.0, -1.5, f32::INFINITY, f32::NEG_INFINITY, f32::NAN];
            HOLES[(r >> 8) as usize % HOLES.len()]
        }
        fn bits(self) -> u32 {
            self.to_bits()
        }
    }

    fn bits<T: TestDepth>(v: &[T]) -> Vec<u32> {
        v.iter().map(|&x| x.bits()).collect()
    }

    /// Valid samples (in window order) of the `ksize × ksize` replicate-border
    /// window around `(x, y)`, channel `c`: the scalar reference no network
    /// takes part in.
    fn window_valid<T: TestDepth, const C: usize>(
        src: &[T],
        (w, h): (usize, usize),
        (x, y, c): (usize, usize, usize),
        ksize: usize,
    ) -> Vec<T> {
        let r = (ksize / 2) as isize;
        let mut valid = Vec::new();
        for dy in -r..=r {
            for dx in -r..=r {
                let sy = (y as isize + dy).clamp(0, h as isize - 1) as usize;
                let sx = (x as isize + dx).clamp(0, w as isize - 1) as usize;
                let v = src[(sy * w + sx) * C + c];
                if v.is_valid() {
                    valid.push(v);
                }
            }
        }
        // Valid samples are totally ordered (no NaN).
        valid.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        valid
    }

    /// Scalar reference median: rank `n / 2` of the sorted valid samples,
    /// or the invalid marker.
    fn scalar_median<T: TestDepth, const C: usize>(
        src: &[T],
        w: usize,
        h: usize,
        ksize: usize,
    ) -> Vec<T> {
        let mut out = vec![T::POISON; src.len()];
        for y in 0..h {
            for x in 0..w {
                for c in 0..C {
                    let valid = window_valid::<T, C>(src, (w, h), (x, y, c), ksize);
                    out[(y * w + x) * C + c] =
                        valid.get(valid.len() / 2).copied().unwrap_or(T::INVALID);
                }
            }
        }
        out
    }

    /// Runs one row kernel over the whole image; `None` when it declines a
    /// row (a SIMD kernel below its minimum width).
    fn run_rows<T: DepthRepr, O: Copy>(
        src: &[T],
        (w, h, row_len, ksize): (usize, usize, usize, usize),
        poison: O,
        mut row: impl FnMut(&[T], &mut [O], usize, &mut Columns<T::Key>) -> bool,
    ) -> Option<Vec<O>> {
        let mut out = vec![poison; row_len * h];
        let mut cols = Columns::new(w, ksize, T::INVALID.key());
        for (y, orow) in out.chunks_mut(row_len).enumerate() {
            if !row(src, orow, y, &mut cols) {
                return None;
            }
        }
        Some(out)
    }

    /// Widths 1..=40 hit every tail of the 16-column vertical blocks and
    /// of every lane width (16, 8, 4, NEON 8 / 4); 131 is a long row.
    fn parity_shapes() -> impl Iterator<Item = (usize, usize)> {
        (1..=40).map(|w| (w, 1 + w % 5)).chain([(131, 7), (64, 3)])
    }

    /// Every median row kernel of this build (portable, portable with AVX2,
    /// the SIMD kernel the dispatcher picks, and the dispatched whole-image
    /// entry) against [`scalar_median`]. Returns how many shapes the SIMD
    /// kernel covered.
    fn check_median_paths<
        T: TestDepth,
        const C: usize,
        const KS: usize,
        const TAPS: usize,
        const L: usize,
    >() -> usize {
        let mut simd_covered = 0;
        for (i, (w, h)) in parity_shapes().enumerate() {
            for holes_pct in HOLE_PCTS {
                let seed = 1000 * KS as u64 + 10 * i as u64 + u64::from(holes_pct);
                let src = random_depth::<T>(w * h * C, seed, holes_pct);
                let want = bits(&scalar_median::<T, C>(&src, w, h, KS));
                let ctx = format!(
                    "{} C{C} k={KS} {w}x{h} holes {holes_pct}%",
                    std::any::type_name::<T>()
                );
                let dims = (w, h, w * C, KS);

                let got = run_rows(&src, dims, T::POISON, |s, d, y, cols| {
                    median_row::<T, C, KS, TAPS, L>(s, d, w, h, y, cols);
                    true
                });
                assert_eq!(got.map(|v| bits(&v)), Some(want.clone()), "portable {ctx}");

                #[cfg(target_arch = "x86_64")]
                if crate::simd::cpu_features().has_avx2 {
                    let got = run_rows(&src, dims, T::POISON, |s, d, y, cols| {
                        // SAFETY: AVX2 support was just checked.
                        unsafe { median_row_avx2::<T, C, KS, TAPS, L>(s, d, w, h, y, cols) };
                        true
                    });
                    assert_eq!(got.map(|v| bits(&v)), Some(want.clone()), "avx2 {ctx}");
                }

                if let Some(simd) = simd_median_row::<T, C, KS, TAPS>() {
                    let got = run_rows(&src, dims, T::POISON, |s, d, y, cols| {
                        simd(s, d, w, h, y, cols)
                    });
                    if let Some(got) = got {
                        assert_eq!(bits(&got), want, "simd {ctx}");
                        simd_covered += 1;
                    }
                }

                let mut got = vec![T::POISON; src.len()];
                median_blur_depth::<T, C>(&src, &mut got, w, h, KS);
                assert_eq!(bits(&got), want, "dispatched {ctx}");
            }
        }
        simd_covered
    }

    #[test]
    fn every_median_path_matches_scalar_reference() {
        let simd = [
            check_median_paths::<u16, 1, 3, 9, { u16::LANES3 }>(),
            check_median_paths::<u16, 1, 5, 25, { u16::LANES5 }>(),
            check_median_paths::<f32, 1, 3, 9, { f32::LANES3 }>(),
            check_median_paths::<f32, 1, 5, 25, { f32::LANES5 }>(),
            check_median_paths::<u16, 3, 3, 9, { u16::LANES3 }>(),
            check_median_paths::<u16, 3, 5, 25, { u16::LANES5 }>(),
            check_median_paths::<f32, 3, 3, 9, { f32::LANES3 }>(),
            check_median_paths::<f32, 3, 5, 25, { f32::LANES5 }>(),
        ];
        // The SIMD kernels exist for single-channel rows on aarch64 only;
        // make sure they actually ran there.
        if cfg!(target_arch = "aarch64") {
            assert!(simd[..4].iter().all(|&n| n > 0), "{simd:?}");
        }
        assert!(simd[4..].iter().all(|&n| n == 0), "{simd:?}");
    }

    #[test]
    fn u16_key_is_order_preserving_with_invalid_last() {
        for v in [1u16, 2, 1000, 65534, 65535] {
            assert!(v.key() < 0u16.key());
            assert_eq!(u16::from_key(v.key()), v);
        }
        assert!(1u16.key() < 2u16.key() && 65534u16.key() < 65535u16.key());
    }

    #[test]
    fn f32_key_maps_every_invalid_sample_last() {
        for v in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0, -0.0, -1.5] {
            assert!(!v.is_valid());
            assert_eq!(v.key(), f32::INFINITY, "{v}");
        }
        for v in [f32::MIN_POSITIVE, 1.5, f32::MAX] {
            assert!(v.is_valid());
            assert!(v.key() < f32::INFINITY);
            assert_eq!(f32::from_key(v.key()).to_bits(), v.to_bits());
        }
    }
}
