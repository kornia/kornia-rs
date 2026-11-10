//! CPU kernels for the flying-pixel mask: the median of the valid samples
//! of each 3×3 window, and the deviation test against it.
//!
//! Invalid samples are mapped to a sort key above every valid key (see
//! [`DepthRepr::key`]), so the median of the `n` valid samples of a window
//! is its order statistic of rank `n / 2`.
//!
//! Each output row is produced in two passes over a per-task scratch:
//!
//! 1. **Vertical**: every source column of the 3 window rows is sorted
//!    ([`COL3`]) and its valid count recorded, once per column. The 3
//!    horizontally adjacent windows that share a column reuse it.
//! 2. **Horizontal**: a window is 3 adjacent sorted columns; a
//!    rank-selection network ([`RANK9`]) brings ranks `0..=4` (every rank
//!    `n / 2` can take) into place, and rank `n / 2` is selected lane-wise.
//!
//! In the NEON and CUDA kernels, a block of windows whose samples are all
//! valid (the common case away from holes) takes the plain median of 9
//! instead ([`med9`]): rank 4 is the same value, for fewer exchanges.
//!
//! On aarch64, single-channel rows of at least one vector run explicit NEON
//! kernels (`u16x8` / `f32x4` per wire, module `neon`). Every other case
//! runs the portable kernels: the same passes on `[K; L]` lane arrays with
//! straight-line compare-exchanges, compiled with AVX2 enabled through a
//! `target_feature` wrapper on x86_64 CPUs that have it. Both are
//! bit-identical (the selections are exact; pinned in the tests). The CUDA
//! kernels in `cuda/depth.rs` are code-generated from the same networks.

use rayon::prelude::*;

/// Window side of the flying-pixel test.
pub(crate) const FLYING_KSIZE: usize = 3;

/// Samples per window.
const TAPS: usize = FLYING_KSIZE * FLYING_KSIZE;

/// Window radius: padding columns on each side of a [`Columns`] row.
const R: usize = FLYING_KSIZE / 2;

/// Value written to the mask for a flagged pixel.
pub(crate) const MASK_FLAGGED: u8 = 255;

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
    /// Lanes per block of the portable kernel.
    const LANES: usize;
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
    /// `|a - b|` in `f32`, for two valid samples.
    fn abs_diff_f32(a: Self, b: Self) -> f32;
    /// The sample as `f32`.
    fn as_f32(self) -> f32;
    /// Flying-pixel mask over a single-channel `h × w` buffer.
    fn flying_mask(src: &[Self], mask: &mut [u8], w: usize, h: usize, rule: FlyingRule);
}

/// The caller's flying-pixel thresholds.
#[derive(Clone, Copy)]
pub(crate) struct FlyingRule {
    pub(crate) abs_threshold: f32,
    pub(crate) rel_threshold: f32,
    pub(crate) min_valid: usize,
}

impl DepthRepr for u16 {
    type Key = u16;
    const INVALID: u16 = 0;
    // Portable-kernel block widths: 9 wires of `L` keys stay within the
    // vector register file.
    const LANES: usize = 16;
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
    #[inline(always)]
    fn abs_diff_f32(a: u16, b: u16) -> f32 {
        f32::from(a.abs_diff(b))
    }
    #[inline(always)]
    fn as_f32(self) -> f32 {
        f32::from(self)
    }
    fn flying_mask(src: &[u16], mask: &mut [u8], w: usize, h: usize, rule: FlyingRule) {
        flying_mask_lanes::<u16, { u16::LANES }>(src, mask, w, h, rule, simd_flying_row::<u16>());
    }
}

impl DepthRepr for f32 {
    type Key = f32;
    const INVALID: f32 = f32::NAN;
    const LANES: usize = 8;
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
    #[inline(always)]
    fn abs_diff_f32(a: f32, b: f32) -> f32 {
        (a - b).abs()
    }
    #[inline(always)]
    fn as_f32(self) -> f32 {
        self
    }
    fn flying_mask(src: &[f32], mask: &mut [u8], w: usize, h: usize, rule: FlyingRule) {
        flying_mask_lanes::<f32, { f32::LANES }>(src, mask, w, h, rule, simd_flying_row::<f32>());
    }
}

/// Sorted columns of one output row's window rows.
///
/// Rank `r` of padded column `p` lives at `rank[r * stride + p]`, its valid
/// count at `count[p]`; padded column `p` is image column `p - R`,
/// replicated beyond the image edges.
struct Columns<K> {
    rank: Vec<K>,
    count: Vec<u16>,
    stride: usize,
}

impl<K: Copy> Columns<K> {
    fn new(w: usize, fill: K) -> Self {
        // Horizontal blocks start below `w` and read `L + 2 * R` padded
        // columns, `L` dividing `MAX_LANES`.
        let stride = w.div_ceil(MAX_LANES) * MAX_LANES + 2 * R;
        Self {
            rank: vec![fill; stride * FLYING_KSIZE],
            count: vec![0; stride],
            stride,
        }
    }
}

/// Vertical pass for columns `x..x + B`.
#[inline(always)]
fn sort_block<T: DepthRepr, const B: usize>(
    rows: &[&[T]; FLYING_KSIZE],
    x: usize,
    cols: &mut Columns<T::Key>,
) {
    let mut v = [Lanes::<T, B>([T::INVALID.key(); B]); FLYING_KSIZE];
    let mut n = [0u16; B];
    for (wire, row) in v.iter_mut().zip(rows) {
        let s = &row[x..x + B];
        for l in 0..B {
            n[l] += u16::from(s[l].is_valid());
            wire.0[l] = s[l].key();
        }
    }
    sort_col3(&mut v);
    let p = x + R;
    for (r, wire) in v.iter().enumerate() {
        let o = r * cols.stride + p;
        cols.rank[o..o + B].copy_from_slice(&wire.0);
    }
    cols.count[p..p + B].copy_from_slice(&n);
}

/// Vertical pass for a whole row, plus replicate padding.
#[inline(always)]
fn sort_columns<T: DepthRepr>(
    src: &[T],
    w: usize,
    sy: &[usize; FLYING_KSIZE],
    cols: &mut Columns<T::Key>,
) {
    let rows: [&[T]; FLYING_KSIZE] = std::array::from_fn(|i| &src[sy[i] * w..(sy[i] + 1) * w]);
    let mut x = 0;
    while x + COL_BLOCK <= w {
        sort_block::<T, COL_BLOCK>(&rows, x, cols);
        x += COL_BLOCK;
    }
    while x < w {
        sort_block::<T, 1>(&rows, x, cols);
        x += 1;
    }
    pad_columns(cols, w);
}

/// Replicate the edge columns into the `R` padding columns on each side.
#[inline(always)]
fn pad_columns<K: Copy>(cols: &mut Columns<K>, w: usize) {
    let stride = cols.stride;
    for rank in cols.rank.chunks_exact_mut(stride) {
        let (first, last) = (rank[R], rank[w + R - 1]);
        rank[..R].fill(first);
        rank[w + R..w + 2 * R].fill(last);
    }
    let (first, last) = (cols.count[R], cols.count[w + R - 1]);
    cols.count[..R].fill(first);
    cols.count[w + R..w + 2 * R].fill(last);
}

/// Horizontal pass: rank-`n / 2` key and valid count `n` of the windows
/// centred on columns `x0..x0 + L`.
#[inline(always)]
fn window_medians<T: DepthRepr, const L: usize>(
    cols: &Columns<T::Key>,
    x0: usize,
) -> ([T::Key; L], [u16; L]) {
    let mut v = [Lanes::<T, L>([T::INVALID.key(); L]); TAPS];
    let mut n = [0u16; L];
    for dc in 0..FLYING_KSIZE {
        let cs = &cols.count[x0 + dc..x0 + dc + L];
        for l in 0..L {
            n[l] += cs[l];
        }
        for r in 0..FLYING_KSIZE {
            let o = r * cols.stride + x0 + dc;
            let s = &cols.rank[o..o + L];
            v[dc * FLYING_KSIZE + r] = Lanes(std::array::from_fn(|l| s[l]));
        }
    }
    rank9(&mut v);
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
fn window_rows(y: usize, h: usize) -> [usize; FLYING_KSIZE] {
    std::array::from_fn(|i| (y + i).saturating_sub(R).min(h - 1))
}

/// Minimum rows per rayon task.
fn min_rows(h: usize) -> usize {
    h.div_ceil(rayon::current_num_threads() * 4).max(16)
}

/// A SIMD mask row kernel: computes one mask row and returns `true`, or
/// returns `false` (writing nothing) for a shape it does not cover.
type FlyingRowFn<T> = fn(
    &[T],
    &mut [u8],
    usize,
    usize,
    usize,
    FlyingRule,
    &mut Columns<<T as DepthRepr>::Key>,
) -> bool;

/// The SIMD flying-pixel row kernel for this dtype, if any.
#[allow(unreachable_code)]
fn simd_flying_row<T: SimdDepth>() -> Option<FlyingRowFn<T>> {
    #[cfg(target_arch = "aarch64")]
    return Some(neon::flying_row::<T>);
    None
}

/// One output row of the flying-pixel mask.
#[inline(always)]
fn flying_row<T: DepthRepr, const L: usize>(
    src: &[T],
    mrow: &mut [u8],
    w: usize,
    h: usize,
    y: usize,
    rule: FlyingRule,
    cols: &mut Columns<T::Key>,
) {
    sort_columns(src, w, &window_rows(y, h), cols);
    let srow = &src[y * w..(y + 1) * w];
    for x0 in (0..w).step_by(L) {
        let (m, n) = window_medians::<T, L>(cols, x0);
        let lanes = L.min(w - x0);
        for l in 0..lanes {
            let d = srow[x0 + l];
            mrow[x0 + l] = if is_flying(d, T::from_key(m[l]), n[l], rule) {
                MASK_FLAGGED
            } else {
                0
            };
        }
    }
}

/// [`flying_row`] compiled with AVX2 enabled.
///
/// # Safety
///
/// The CPU must support AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn flying_row_avx2<T: DepthRepr, const L: usize>(
    src: &[T],
    mrow: &mut [u8],
    w: usize,
    h: usize,
    y: usize,
    rule: FlyingRule,
    cols: &mut Columns<T::Key>,
) {
    flying_row::<T, L>(src, mrow, w, h, y, rule, cols);
}

/// Flying-pixel mask, `L`-lane blocks; rows go to `simd` when it covers
/// them, else to the portable [`flying_row`].
fn flying_mask_lanes<T: DepthRepr, const L: usize>(
    src: &[T],
    mask: &mut [u8],
    w: usize,
    h: usize,
    rule: FlyingRule,
    simd: Option<FlyingRowFn<T>>,
) {
    mask.par_chunks_mut(w)
        .enumerate()
        .with_min_len(min_rows(h))
        .for_each_init(
            || Columns::new(w, T::INVALID.key()),
            |cols, (y, mrow)| {
                if simd.is_some_and(|f| f(src, mrow, w, h, y, rule, cols)) {
                    return;
                }
                #[cfg(target_arch = "x86_64")]
                if crate::simd::cpu_features().has_avx2 {
                    // SAFETY: AVX2 support was just checked.
                    unsafe { flying_row_avx2::<T, L>(src, mrow, w, h, y, rule, cols) };
                    return;
                }
                flying_row::<T, L>(src, mrow, w, h, y, rule, cols);
            },
        );
}

/// Flying-pixel mask over a single-channel `h × w` buffer. An empty image
/// is a no-op.
pub(crate) fn flying_pixel_mask<T: DepthRepr>(
    src: &[T],
    mask: &mut [u8],
    w: usize,
    h: usize,
    abs_threshold: f32,
    rel_threshold: f32,
    min_valid: usize,
) {
    if w == 0 || h == 0 {
        return;
    }
    let rule = FlyingRule {
        abs_threshold,
        rel_threshold,
        min_valid,
    };
    T::flying_mask(src, mask, w, h, rule);
}

/// The flying-pixel rule for one pixel: valid and either supported by fewer
/// than `min_valid` valid window samples, or farther than
/// `max(abs_threshold, rel_threshold * d)` from the valid-sample median.
/// Mirrored expression-for-expression by the CUDA kernel.
#[inline(always)]
fn is_flying<T: DepthRepr>(d: T, median: T, count: u16, rule: FlyingRule) -> bool {
    if !d.is_valid() {
        return false;
    }
    if usize::from(count) < rule.min_valid {
        return true;
    }
    let threshold = rule.abs_threshold.max(rule.rel_threshold * d.as_f32());
    T::abs_diff_f32(d, median) > threshold
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
/// networks, same keys, so bit-identical to [`flying_row`] (pinned in the
/// tests).
#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    use super::{
        med9, pad_columns, rank9, sort_col3, window_rows, Columns, DepthRepr, Exchange, FlyingRule,
        FLYING_KSIZE, R, TAPS,
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
        fn select(v: &[Self::V; TAPS], n: Self::N) -> Self::V;
        /// Flying-pixel flags of the `W` samples at `d` against median keys
        /// `m` and valid counts `n`, written as `W` mask bytes.
        ///
        /// # Safety
        ///
        /// `d` is valid for reading `W` samples, `dst` for writing `W` bytes.
        unsafe fn st_flags(dst: *mut u8, d: *const Self, m: Self::V, n: Self::N, rule: FlyingRule);
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
        fn select(v: &[U16x8; TAPS], n: uint16x8_t) -> U16x8 {
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
        unsafe fn st_flags(dst: *mut u8, d: *const u16, m: U16x8, n: uint16x8_t, rule: FlyingRule) {
            let d = vld1q_u16(d);
            let valid = vtstq_u16(d, d);
            let median = vaddq_u16(m.0, vdupq_n_u16(1));
            let diff = vabdq_u16(d, median);
            let abs = vdupq_n_f32(rule.abs_threshold);
            let rel = vdupq_n_f32(rule.rel_threshold);
            let to_f32 = |x: uint16x4_t| vcvtq_f32_u32(vmovl_u16(x));
            let far = |diff: uint16x4_t, d: uint16x4_t| {
                let threshold = vmaxnmq_f32(abs, vmulq_f32(rel, to_f32(d)));
                vmovn_u32(vcgtq_f32(to_f32(diff), threshold))
            };
            let far = vcombine_u16(
                far(vget_low_u16(diff), vget_low_u16(d)),
                far(vget_high_u16(diff), vget_high_u16(d)),
            );
            // Counts are <= TAPS; a larger `min_valid` flags every valid lane
            // either way.
            let min_valid = rule.min_valid.min(usize::from(u16::MAX)) as u16;
            let few = vcltq_u16(n, vdupq_n_u16(min_valid));
            let flag = vandq_u16(valid, vorrq_u16(few, far));
            // All-ones lanes narrow to 0xff = MASK_FLAGGED.
            vst1_u8(dst, vmovn_u16(flag));
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
        fn select(v: &[F32x4; TAPS], n: uint16x4_t) -> F32x4 {
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
        unsafe fn st_flags(dst: *mut u8, d: *const f32, m: F32x4, n: uint16x4_t, rule: FlyingRule) {
            let d = vld1q_f32(d);
            let valid = valid_f32(d);
            // fabd = |d - m| with one rounding, as `(d - m).abs()`.
            let diff = vabdq_f32(d, m.0);
            let threshold = vmaxnmq_f32(
                vdupq_n_f32(rule.abs_threshold),
                vmulq_f32(vdupq_n_f32(rule.rel_threshold), d),
            );
            let far = vcgtq_f32(diff, threshold);
            let min_valid = rule.min_valid.min(u32::MAX as usize) as u32;
            let few = vcltq_u32(vmovl_u16(n), vdupq_n_u32(min_valid));
            let flag = vmovn_u32(vandq_u32(valid, vorrq_u32(few, far)));
            let bytes = vmovn_u16(vcombine_u16(flag, flag));
            let word = vget_lane_u32::<0>(vreinterpret_u32_u8(bytes));
            dst.cast::<u32>().write_unaligned(word);
        }
    }

    /// Vertical pass for a whole row. Requires `w >= T::W`.
    ///
    /// # Safety
    ///
    /// `src` holds `h * w` samples, `sy` are rows `< h`, `w >= T::W`, and
    /// `cols` was built for width `w`.
    #[inline(always)]
    unsafe fn sort_columns<T: NeonDepth>(
        src: &[T],
        w: usize,
        sy: &[usize; FLYING_KSIZE],
        cols: &mut Columns<T::Key>,
    ) {
        let stride = cols.stride;
        let rank = cols.rank.as_mut_ptr();
        let count = cols.count.as_mut_ptr();
        let mut x = 0;
        while x < w {
            // The last block overlaps the previous one instead of running
            // past the row end; recomputing columns is idempotent.
            let xs = x.min(w - T::W);
            let mut n = T::n_zero();
            let mut v = [T::v_fill(); FLYING_KSIZE];
            for (wire, &row) in v.iter_mut().zip(sy) {
                let (k, n2) = T::load_src(src.as_ptr().add(row * w + xs), n);
                *wire = k;
                n = n2;
            }
            sort_col3(&mut v);
            for (i, &wire) in v.iter().enumerate() {
                T::st_key(rank.add(i * stride + xs + R), wire);
            }
            T::st_n(count.add(xs + R), n);
            x += T::W;
        }
        pad_columns(cols, w);
    }

    /// Horizontal pass: rank-`n / 2` keys and valid counts of the windows
    /// centred on columns `x0..x0 + T::W`.
    ///
    /// # Safety
    ///
    /// `x0 < w` for the width `cols` was built for.
    #[inline(always)]
    unsafe fn window<T: NeonDepth>(cols: &Columns<T::Key>, x0: usize) -> (T::V, T::N) {
        let stride = cols.stride;
        let rank = cols.rank.as_ptr();
        let count = cols.count.as_ptr();
        let mut v = [T::v_fill(); TAPS];
        let mut n = T::n_zero();
        for dc in 0..FLYING_KSIZE {
            n = T::n_add(n, T::ld_n(count.add(x0 + dc)));
            for r in 0..FLYING_KSIZE {
                v[dc * FLYING_KSIZE + r] = T::ld_key(rank.add(r * stride + x0 + dc));
            }
        }
        // Fully valid windows (the common case away from holes) only need
        // the plain median, rank `TAPS / 2`; same value, fewer exchanges.
        if T::all_eq(n, TAPS as u16) {
            return (med9(&v), n);
        }
        rank9(&mut v);
        (T::select(&v, n), n)
    }

    /// One output row of the flying-pixel mask; `false` (nothing written)
    /// when `w < T::W`.
    pub(super) fn flying_row<T: NeonDepth>(
        src: &[T],
        mrow: &mut [u8],
        w: usize,
        h: usize,
        y: usize,
        rule: FlyingRule,
        cols: &mut Columns<T::Key>,
    ) -> bool {
        if w < T::W {
            return false;
        }
        let sy = window_rows(y, h);
        // SAFETY: `src` is the `h × w` image, `sy` rows are `< h` and
        // `w >= T::W`; `cols` (built for `w`) is padded for the
        // horizontal reads; the mask row and source row `y` hold `w`
        // entries and every block reads / writes `T::W` of them from
        // `x0 <= w - T::W`.
        unsafe {
            sort_columns(src, w, &sy, cols);
            let srow = src.as_ptr().add(y * w);
            let mut x = 0;
            while x < w {
                let x0 = x.min(w - T::W);
                let (m, n) = window::<T>(cols, x0);
                T::st_flags(mrow.as_mut_ptr().add(x0), srow.add(x0), m, n, rule);
                x += T::W;
            }
        }
        true
    }
}

/// `dst = INVALID` where `mask != 0`, else `src`.
pub(crate) fn remove_flagged<T: DepthRepr>(src: &[T], mask: &[u8], dst: &mut [T]) {
    dst.par_iter_mut()
        .zip(src.par_iter().zip(mask.par_iter()))
        .with_min_len(1 << 14)
        .for_each(|(d, (&s, &m))| *d = if m != 0 { T::INVALID } else { s });
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
    fn rank_network_selects_lower_half_of_sorted_columns() {
        assert_rank_network(RANK9, 3);
    }

    #[test]
    fn med9_exact_on_full_sorted_columns() {
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
        for (net, n) in [(COL3, 3usize)] {
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
        /// A random valid sample, with many duplicates.
        fn sample(r: u32) -> Self;
        /// A random invalid sample.
        fn hole(r: u32) -> Self;
    }

    impl TestDepth for u16 {
        fn sample(r: u32) -> u16 {
            1 + (r >> 20) as u16 % 64
        }
        fn hole(_: u32) -> u16 {
            0
        }
    }

    impl TestDepth for f32 {
        fn sample(r: u32) -> f32 {
            1.0 + ((r >> 8) % 16) as f32 * 0.25
        }
        /// Every invalid encoding, not only the canonical NaN.
        fn hole(r: u32) -> f32 {
            const HOLES: [f32; 6] = [0.0, -0.0, -1.5, f32::INFINITY, f32::NEG_INFINITY, f32::NAN];
            HOLES[(r >> 8) as usize % HOLES.len()]
        }
    }

    /// Valid samples of the 3×3 replicate-border window around `(x, y)`,
    /// sorted: the scalar reference no network takes part in.
    fn window_valid<T: TestDepth>(
        src: &[T],
        (w, h): (usize, usize),
        (x, y): (usize, usize),
    ) -> Vec<T> {
        let r = R as isize;
        let mut valid = Vec::new();
        for dy in -r..=r {
            for dx in -r..=r {
                let sy = (y as isize + dy).clamp(0, h as isize - 1) as usize;
                let sx = (x as isize + dx).clamp(0, w as isize - 1) as usize;
                let v = src[sy * w + sx];
                if v.is_valid() {
                    valid.push(v);
                }
            }
        }
        // Valid samples are totally ordered (no NaN).
        valid.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        valid
    }

    /// Scalar reference flying-pixel mask (the rule of [`is_flying`],
    /// spelled out on the scalar median).
    fn scalar_flying<T: TestDepth>(src: &[T], w: usize, h: usize, rule: FlyingRule) -> Vec<u8> {
        let mut out = vec![0u8; src.len()];
        for y in 0..h {
            for x in 0..w {
                let d = src[y * w + x];
                if !d.is_valid() {
                    continue;
                }
                let valid = window_valid(src, (w, h), (x, y));
                let m = valid[valid.len() / 2];
                let threshold = rule.abs_threshold.max(rule.rel_threshold * d.as_f32());
                if valid.len() < rule.min_valid || T::abs_diff_f32(d, m) > threshold {
                    out[y * w + x] = MASK_FLAGGED;
                }
            }
        }
        out
    }

    /// Runs one row kernel over the whole image; `None` when it declines a
    /// row (a SIMD kernel below its minimum width).
    fn run_rows<T: DepthRepr, O: Copy>(
        src: &[T],
        (w, h): (usize, usize),
        poison: O,
        mut row: impl FnMut(&[T], &mut [O], usize, &mut Columns<T::Key>) -> bool,
    ) -> Option<Vec<O>> {
        let mut out = vec![poison; w * h];
        let mut cols = Columns::new(w, T::INVALID.key());
        for (y, orow) in out.chunks_mut(w).enumerate() {
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

    /// Every flying-pixel row kernel of this build (portable, portable with
    /// AVX2, the SIMD kernel the dispatcher picks, and the dispatched
    /// whole-image entry) against [`scalar_flying`]. Returns how many
    /// shapes the SIMD kernel covered and how many pixels were flagged.
    fn check_flying_paths<T: TestDepth, const L: usize>(scale: f32) -> (usize, usize) {
        let rules = [(1.0, 0.0, 0), (0.5, 0.1, 5), (0.0, 0.0, 10), (4.0, 0.02, 3)];
        let (mut simd_covered, mut flagged) = (0, 0);
        for (i, (w, h)) in parity_shapes().enumerate() {
            for (j, &(abs_threshold, rel_threshold, min_valid)) in rules.iter().enumerate() {
                let rule = FlyingRule {
                    abs_threshold: abs_threshold * scale,
                    rel_threshold,
                    min_valid,
                };
                let holes_pct = HOLE_PCTS[j % HOLE_PCTS.len()];
                let src = random_depth::<T>(w * h, 31 + 10 * i as u64 + j as u64, holes_pct);
                let want = scalar_flying(&src, w, h, rule);
                flagged += want.iter().filter(|&&v| v != 0).count();
                let ctx = format!("{} {w}x{h} rule {j}", std::any::type_name::<T>());
                let dims = (w, h);
                const POISON: u8 = 7;

                let got = run_rows(&src, dims, POISON, |s, m, y, cols| {
                    flying_row::<T, L>(s, m, w, h, y, rule, cols);
                    true
                });
                assert_eq!(got.as_ref(), Some(&want), "portable {ctx}");

                #[cfg(target_arch = "x86_64")]
                if crate::simd::cpu_features().has_avx2 {
                    let got = run_rows(&src, dims, POISON, |s, m, y, cols| {
                        // SAFETY: AVX2 support was just checked.
                        unsafe { flying_row_avx2::<T, L>(s, m, w, h, y, rule, cols) };
                        true
                    });
                    assert_eq!(got.as_ref(), Some(&want), "avx2 {ctx}");
                }

                if let Some(simd) = simd_flying_row::<T>() {
                    let got = run_rows(&src, dims, POISON, |s, m, y, cols| {
                        simd(s, m, w, h, y, rule, cols)
                    });
                    if let Some(got) = got {
                        assert_eq!(got, want, "simd {ctx}");
                        simd_covered += 1;
                    }
                }

                let mut got = vec![POISON; w * h];
                flying_pixel_mask(
                    &src,
                    &mut got,
                    w,
                    h,
                    rule.abs_threshold,
                    rule.rel_threshold,
                    rule.min_valid,
                );
                assert_eq!(got, want, "dispatched {ctx}");
            }
        }
        (simd_covered, flagged)
    }

    #[test]
    fn every_flying_mask_path_matches_scalar_reference() {
        // u16 samples are 1..=64, f32 samples 1.0..=4.75: scale the
        // absolute thresholds so both flag a fair share of pixels.
        let runs = [
            check_flying_paths::<u16, { u16::LANES }>(4.0),
            check_flying_paths::<f32, { f32::LANES }>(0.25),
        ];
        // The fixtures must flag something, or the comparison is vacuous.
        assert!(runs.iter().all(|&(_, flagged)| flagged > 0), "{runs:?}");
        if cfg!(target_arch = "aarch64") {
            assert!(runs.iter().all(|&(simd, _)| simd > 0), "{runs:?}");
        }
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
