//! CPU kernels for label-boundary erosion.
//!
//! The image is processed in horizontal bands of output rows, each on its
//! own rayon task with a band-local distance scratch (L1/L2-resident for
//! typical widths) of `u8` lanes when `t + 1` fits in a byte and `u16`
//! lanes otherwise. A band of rows `[y0, y1)` only depends on the seeds of rows
//! `[y0 - t, y1 + t)` (`t` = distance threshold), so bands are independent
//! and need no full-image intermediate:
//!
//! 1. seed rows: `0` where a pixel has a differently labelled neighbour,
//!    `cap = t + 1` elsewhere (vectorizable compares against the shifted
//!    neighbour rows);
//! 2. vertical two-sweep min-plus pass over the band (`v[y] = min(v[y],
//!    v[y ∓ 1] + 1)`, column lanes, vectorizable);
//! 3. per output row: for Chebyshev the vertical distance is thresholded,
//!    then the horizontal pass and the `fill` / label select. The
//!    horizontal pass is a direct `2t`-shift window minimum while
//!    `t <= DIRECT_MAX_THRESHOLD`, and a two-sweep min-plus scan (NEON
//!    log-step scan on aarch64, scalar elsewhere) above.
//!
//! Distances saturate at `cap`: only the comparison `distance <= t` is ever
//! read, and every value above `t` behaves identically in the min-plus
//! recurrences.

use rayon::prelude::*;

use crate::connected_components::Connectivity;
use crate::parallel::par_row_chunks_mut;

/// Minimum number of output rows per band; a band also re-derives `t` rows
/// above and below itself, so bands are at least `t` rows tall to bound
/// that overhead.
pub(crate) const BAND_MIN_ROWS: usize = 32;

/// Below this pixel count the bands run on the calling thread: the per-band
/// work is too small to amortize rayon task spawning.
pub(crate) const PAR_MIN_PIXELS: usize = 64 * 1024;

/// Output rows scanned together by the horizontal pass. The scan of one
/// row is a serial chain (one carry per vector); interleaving independent
/// rows keeps the SIMD pipelines busy while each chain waits.
pub(crate) const H_ROWS: usize = 4;

/// Largest distance threshold whose horizontal pass is the direct window
/// [`horizontal_window_min_row`] (`2t` shifted minima per pixel) instead of
/// the two-sweep scan, whose cost does not depend on `t`.
pub(crate) const DIRECT_MAX_THRESHOLD: usize = 4;

/// Saturating distance lane of the scratch. Every value `>= cap = t + 1`
/// behaves identically, so a lane type only has to hold `cap`.
pub(crate) trait Dist: Copy + Ord + Send + Sync + 'static {
    /// Distance zero (a seed).
    const ZERO: Self;
    /// `v` in this lane type, saturated to its maximum.
    fn from_u16(v: u16) -> Self;
    /// `self + 1`, saturating.
    fn sat_inc(self) -> Self;
    /// `self + rhs`, saturating.
    fn sat_add(self, rhs: Self) -> Self;
    /// Horizontal two-sweep min-plus pass in place on every `w`-wide row
    /// of `rows` (`w > 0`, `rows.len()` a multiple of `w`).
    fn horizontal_min_plus_rows(rows: &mut [Self], w: usize);
}

impl Dist for u8 {
    const ZERO: Self = 0;
    fn from_u16(v: u16) -> Self {
        u8::try_from(v).unwrap_or(u8::MAX)
    }
    fn sat_inc(self) -> Self {
        self.saturating_add(1)
    }
    fn sat_add(self, rhs: Self) -> Self {
        self.saturating_add(rhs)
    }
    fn horizontal_min_plus_rows(rows: &mut [Self], w: usize) {
        #[cfg(target_arch = "aarch64")]
        {
            // NEON is architectural on aarch64: no runtime check.
            let mut groups = rows.chunks_exact_mut(H_ROWS * w);
            for group in &mut groups {
                horizontal_min_plus_neon_u8::<H_ROWS>(group, w);
            }
            for row in groups.into_remainder().chunks_exact_mut(w) {
                horizontal_min_plus_neon_u8::<1>(row, w);
            }
            return;
        }
        #[allow(unreachable_code)]
        for row in rows.chunks_exact_mut(w) {
            horizontal_min_plus_scalar(row);
        }
    }
}

impl Dist for u16 {
    const ZERO: Self = 0;
    fn from_u16(v: u16) -> Self {
        v
    }
    fn sat_inc(self) -> Self {
        self.saturating_add(1)
    }
    fn sat_add(self, rhs: Self) -> Self {
        self.saturating_add(rhs)
    }
    fn horizontal_min_plus_rows(rows: &mut [Self], w: usize) {
        #[cfg(target_arch = "aarch64")]
        {
            // NEON is architectural on aarch64: no runtime check.
            let mut groups = rows.chunks_exact_mut(H_ROWS * w);
            for group in &mut groups {
                horizontal_min_plus_neon_u16::<H_ROWS>(group, w);
            }
            for row in groups.into_remainder().chunks_exact_mut(w) {
                horizontal_min_plus_neon_u16::<1>(row, w);
            }
            return;
        }
        #[allow(unreachable_code)]
        for row in rows.chunks_exact_mut(w) {
            horizontal_min_plus_scalar(row);
        }
    }
}

/// Erode the label boundaries of a row-major `w × h` label map.
///
/// `threshold` is the distance threshold `radius - 1` (see
/// [`super::distance_threshold`]); `w` and `h` are non-zero and
/// `src.len() == dst.len() == w * h`. Distances use `u8` lanes whenever
/// `cap = threshold + 1` fits in a byte.
pub(crate) fn erode_label_boundaries<T: Copy + PartialEq + Send + Sync>(
    src: &[T],
    dst: &mut [T],
    w: usize,
    h: usize,
    threshold: u16,
    connectivity: Connectivity,
    fill: T,
) {
    if threshold < u16::from(u8::MAX) {
        erode_label_boundaries_with::<T, u8>(src, dst, w, h, threshold, connectivity, fill);
    } else {
        erode_label_boundaries_with::<T, u16>(src, dst, w, h, threshold, connectivity, fill);
    }
}

/// [`erode_label_boundaries`] with an explicit distance lane type `D`;
/// `threshold + 1` must fit in `D`.
pub(crate) fn erode_label_boundaries_with<T: Copy + PartialEq + Send + Sync, D: Dist>(
    src: &[T],
    dst: &mut [T],
    w: usize,
    h: usize,
    threshold: u16,
    connectivity: Connectivity,
    fill: T,
) {
    let band = BAND_MIN_ROWS.max(usize::from(threshold));
    let chunk = band.saturating_mul(w);
    if w * h < PAR_MIN_PIXELS {
        let mut scratch: Vec<D> = Vec::new();
        for (i, out) in dst.chunks_mut(chunk).enumerate() {
            erode_band(
                src,
                out,
                w,
                h,
                i * band,
                threshold,
                connectivity,
                fill,
                &mut scratch,
            );
        }
    } else {
        par_row_chunks_mut(dst, w, band).enumerate().for_each_init(
            Vec::<D>::new,
            |scratch, (i, out)| {
                erode_band(
                    src,
                    out,
                    w,
                    h,
                    i * band,
                    threshold,
                    connectivity,
                    fill,
                    scratch,
                )
            },
        );
    }
}

/// One band: `out` holds the output rows `[y0, y0 + out.len() / w)`.
#[allow(clippy::too_many_arguments)]
fn erode_band<T: Copy + PartialEq, D: Dist>(
    src: &[T],
    out: &mut [T],
    w: usize,
    h: usize,
    y0: usize,
    threshold: u16,
    connectivity: Connectivity,
    fill: T,
    scratch: &mut Vec<D>,
) {
    let t = usize::from(threshold);
    let cap = D::from_u16(threshold + 1);
    let thr = D::from_u16(threshold);
    let y1 = y0 + out.len() / w;
    // Seeds farther than `t` rows away cannot reach the band.
    let e0 = y0.saturating_sub(t);
    let e1 = (y1 + t).min(h);
    let len = (e1 - e0) * w;
    // Grow only: every element of `dist` is written by `seed_row` below,
    // and every element of `row_tmp` by `horizontal_window_min_row`.
    if scratch.len() < len + w {
        scratch.resize(len + w, D::ZERO);
    }
    let (dist, row_tmp) = scratch[..len + w].split_at_mut(len);

    for (i, row) in dist.chunks_exact_mut(w).enumerate() {
        seed_row(src, w, h, e0 + i, connectivity, cap, row);
    }
    // With t == 0 only the seeds themselves are selected: both passes would
    // leave every `0` / `1` value unchanged.
    if t > 0 {
        vertical_min_plus(dist, w);
    }
    let out_rows = out.len() / w;
    if t > 0 && t <= DIRECT_MAX_THRESHOLD {
        // Small threshold: the horizontal pass is a direct window of `2t`
        // shifted minima per row, cheaper than the two-sweep scan.
        for i in 0..out_rows {
            let y = y0 + i;
            let row = &mut dist[(y - e0) * w..(y - e0 + 1) * w];
            if connectivity == Connectivity::Eight {
                threshold_row(row, thr, cap);
            }
            horizontal_window_min_row(row, t, row_tmp);
            let labels = &src[y * w..(y + 1) * w];
            select_row(labels, row_tmp, thr, fill, &mut out[i * w..(i + 1) * w]);
        }
        return;
    }
    let mut i = 0;
    while i < out_rows {
        let n = (out_rows - i).min(H_ROWS);
        let y = y0 + i;
        let rows = &mut dist[(y - e0) * w..(y - e0 + n) * w];
        if t > 0 {
            if connectivity == Connectivity::Eight {
                threshold_row(rows, thr, cap);
            }
            D::horizontal_min_plus_rows(rows, w);
        }
        let labels = src[y * w..(y + n) * w].chunks_exact(w);
        let outs = out[i * w..(i + n) * w].chunks_exact_mut(w);
        for ((l, d), o) in labels.zip(rows.chunks_exact(w)).zip(outs) {
            select_row(l, d, thr, fill, o);
        }
        i += n;
    }
}

/// Seed row `y`: `0` where the pixel differs from one of its neighbours,
/// `cap` elsewhere.
///
/// Out-of-image neighbours are replaced by the clamped (replicated)
/// coordinate. A replicated neighbour is always the pixel itself or another
/// neighbour of the same set (e.g. the up-left neighbour of a top-row pixel
/// becomes its left neighbour), so clamping adds no difference: the image
/// border is not a boundary.
pub(crate) fn seed_row<T: Copy + PartialEq, D: Dist>(
    src: &[T],
    w: usize,
    h: usize,
    y: usize,
    connectivity: Connectivity,
    cap: D,
    out: &mut [D],
) {
    let row = |r: usize| &src[r * w..(r + 1) * w];
    let cur = row(y);
    let up = row(y.saturating_sub(1));
    let dn = row((y + 1).min(h - 1));
    let out = &mut out[..w];

    let code = |differs: bool| if differs { D::ZERO } else { cap };
    out[0] = code(differs_at(up, cur, dn, 0, connectivity));
    if w == 1 {
        return;
    }
    out[w - 1] = code(differs_at(up, cur, dn, w - 1, connectivity));
    if w == 2 {
        return;
    }

    // Interior: every neighbour column exists. Equal-length re-slices let
    // the compiler drop the bounds checks and vectorize the compares.
    let m = w - 2;
    let centre = &cur[1..=m];
    let left = &cur[..m];
    let right = &cur[2..];
    let above = &up[1..=m];
    let below = &dn[1..=m];
    let o = &mut out[1..=m];
    let edges = left.iter().zip(right).zip(above.iter().zip(below));
    match connectivity {
        Connectivity::Four => {
            for ((o, &v), ((&l, &r), (&u, &d))) in o.iter_mut().zip(centre).zip(edges) {
                *o = code((l != v) | (r != v) | (u != v) | (d != v));
            }
        }
        Connectivity::Eight => {
            let corners = up[..m]
                .iter()
                .zip(&up[2..])
                .zip(dn[..m].iter().zip(&dn[2..]));
            for (((o, &v), ((&l, &r), (&u, &d))), ((&ul, &ur), (&dl, &dr))) in
                o.iter_mut().zip(centre).zip(edges).zip(corners)
            {
                let edge = (l != v) | (r != v) | (u != v) | (d != v);
                let corner = (ul != v) | (ur != v) | (dl != v) | (dr != v);
                *o = code(edge | corner);
            }
        }
    }
}

/// Whether pixel `x` of `cur` differs from a neighbour, with clamped
/// coordinates (see [`seed_row`]).
#[inline]
fn differs_at<T: Copy + PartialEq>(
    up: &[T],
    cur: &[T],
    dn: &[T],
    x: usize,
    connectivity: Connectivity,
) -> bool {
    let w = cur.len();
    let xl = x.saturating_sub(1);
    let xr = (x + 1).min(w - 1);
    let v = cur[x];
    let edge = cur[xl] != v || cur[xr] != v || up[x] != v || dn[x] != v;
    match connectivity {
        Connectivity::Four => edge,
        Connectivity::Eight => edge || up[xl] != v || up[xr] != v || dn[xl] != v || dn[xr] != v,
    }
}

/// Vertical two-sweep min-plus pass: every row becomes
/// `min over rows y' of (dist[y'] + |y - y'|)`, lane-wise per column.
pub(crate) fn vertical_min_plus<D: Dist>(dist: &mut [D], w: usize) {
    let rows = dist.len() / w;
    for i in 1..rows {
        let (head, tail) = dist.split_at_mut(i * w);
        let prev = &head[(i - 1) * w..];
        let cur = &mut tail[..w];
        for (c, &p) in cur.iter_mut().zip(prev) {
            *c = (*c).min(p.sat_inc());
        }
    }
    for i in (0..rows.saturating_sub(1)).rev() {
        let (head, tail) = dist.split_at_mut((i + 1) * w);
        let cur = &mut head[i * w..];
        let next = &tail[..w];
        for (c, &n) in cur.iter_mut().zip(next) {
            *c = (*c).min(n.sat_inc());
        }
    }
}

/// Chebyshev step between the two passes: a pixel is within `threshold` of
/// a seed iff some pixel of its row within `threshold` columns has a
/// vertical distance `<= threshold`, so the vertical distance is reduced to
/// `0` (close enough) or `cap`.
fn threshold_row<D: Dist>(row: &mut [D], threshold: D, cap: D) {
    for v in row.iter_mut() {
        *v = if *v <= threshold { D::ZERO } else { cap };
    }
}

/// `out[x] = min over |k| <= t of (row[x + k] + |k|)`, saturating, over
/// in-image `x + k` only: the horizontal min-plus pass restricted to a
/// window of `t` columns, exact for every value compared against `t`.
///
/// Each shift is a pair of equal-length slice zips the compiler vectorizes
/// (NEON / AVX2 / scalar from the same code); `out.len() == row.len()`.
pub(crate) fn horizontal_window_min_row<D: Dist>(row: &[D], t: usize, out: &mut [D]) {
    let w = row.len();
    let out = &mut out[..w];
    out.copy_from_slice(row);
    let mut inc = D::ZERO;
    for k in 1..=t.min(w.saturating_sub(1)) {
        inc = inc.sat_inc();
        // out[x] <- min(out[x], row[x - k] + k), x in k..w
        for (o, &v) in out[k..].iter_mut().zip(&row[..w - k]) {
            *o = (*o).min(v.sat_add(inc));
        }
        // out[x] <- min(out[x], row[x + k] + k), x in 0..w - k
        for (o, &v) in out[..w - k].iter_mut().zip(&row[k..]) {
            *o = (*o).min(v.sat_add(inc));
        }
    }
}

/// `out = dist <= threshold ? fill : label`.
fn select_row<T: Copy, D: Dist>(labels: &[T], dist: &[D], threshold: D, fill: T, out: &mut [T]) {
    for ((o, &l), &d) in out.iter_mut().zip(labels).zip(dist) {
        *o = if d <= threshold { fill } else { l };
    }
}

/// Horizontal two-sweep min-plus pass in place on every `w`-wide row of
/// `rows`: `row[x] = min over x' of (row[x'] + |x - x'|)`, saturating.
#[cfg(test)]
pub(crate) fn horizontal_min_plus_rows<D: Dist>(rows: &mut [D], w: usize) {
    if w == 0 {
        return;
    }
    D::horizontal_min_plus_rows(rows, w);
}

/// Scalar reference for [`horizontal_min_plus_rows`] on one row.
pub(crate) fn horizontal_min_plus_scalar<D: Dist>(row: &mut [D]) {
    let max = D::from_u16(u16::MAX);
    let mut run = max;
    for v in row.iter_mut() {
        run = (*v).min(run.sat_inc());
        *v = run;
    }
    run = max;
    for v in row.iter_mut().rev() {
        run = (*v).min(run.sat_inc());
        *v = run;
    }
}

/// NEON [`horizontal_min_plus_rows`] on `N` rows of width `w` scanned in
/// lock step (`rows.len() == N * w`; otherwise a no-op), for one lane type.
///
/// Per row: a Hillis–Steele log-step scan inside each vector (shift by 1, 2,
/// 4, ... lanes and add the shift as the distance, saturating) plus the
/// carry from the previous vector broadcast and added to a lane ramp. The
/// loop-carried dependency is one broadcast, one add and one min per
/// vector; the `N` rows are independent chains the core overlaps.
/// Bit-exact with [`horizontal_min_plus_scalar`]: both compute the same
/// saturating `min over x' of (row[x'] + |x - x'|)`.
#[cfg(target_arch = "aarch64")]
macro_rules! horizontal_min_plus_neon {
    (
        $name:ident, $t:ty, $lanes:literal, $last:literal,
        dup = $dup:ident, ld = $ld:ident, st = $st:ident, add = $add:ident,
        min = $min:ident, dupl = $dupl:ident, get = $get:ident, ext = $ext:ident,
        fwd = [$(($fs:literal, $fd:literal)),+], bwd = [$(($bs:literal, $bd:literal)),+]
    ) => {
        #[cfg(target_arch = "aarch64")]
        pub(crate) fn $name<const N: usize>(rows: &mut [$t], w: usize) {
            use std::arch::aarch64::*;

            const LANES: usize = $lanes;
            let ramp_up: [$t; LANES] = std::array::from_fn(|i| (i + 1) as $t);
            let ramp_down: [$t; LANES] = std::array::from_fn(|i| (LANES - i) as $t);

            if w == 0 || rows.len() != N * w {
                return;
            }
            let bulk = w - w % LANES;
            let p = rows.as_mut_ptr();
            // SAFETY: NEON is architectural on aarch64. Row `k` is
            // `p + k * w .. p + (k + 1) * w` with `k < N` and
            // `rows.len() == N * w`. Every vector load / store touches
            // `x .. x + LANES` of a row with `x + LANES <= bulk <= w`, and
            // every scalar access an index in `bulk..w`; all inside `rows`,
            // which is exclusively borrowed for the whole call.
            unsafe {
                let maxv = $dup(<$t>::MAX);
                let ramp_up = $ld(ramp_up.as_ptr());
                let ramp_down = $ld(ramp_down.as_ptr());

                // Forward: lane j sees lanes j-1, j-2, j-4, ... (and
                // transitively all lanes <= j), then the previous vector's
                // last value + (j + 1).
                let mut carry = [maxv; N];
                let mut x = 0;
                while x < bulk {
                    for (k, c) in carry.iter_mut().enumerate() {
                        let q = p.add(k * w + x);
                        let mut a = $ld(q);
                        $(a = $min(a, $add($ext::<$fs>(maxv, a), $dup($fd)));)+
                        a = $min(a, $add(*c, ramp_up));
                        $st(q, a);
                        *c = $dupl::<$last>(a);
                    }
                    x += LANES;
                }
                for (k, c) in carry.iter_mut().enumerate() {
                    let r = p.add(k * w);
                    // Forward scalar tail, then the backward scalar tail
                    // (right to left); the backward vector carry starts
                    // from its result.
                    let mut run = $get::<0>(*c);
                    for i in bulk..w {
                        run = (*r.add(i)).min(run.saturating_add(1));
                        *r.add(i) = run;
                    }
                    let mut run = <$t>::MAX;
                    for i in (bulk..w).rev() {
                        run = (*r.add(i)).min(run.saturating_add(1));
                        *r.add(i) = run;
                    }
                    *c = $dup(run);
                }

                // Backward: the vectors from the right, mirrored (shift
                // towards lane 0, descending ramp).
                let mut x = bulk;
                while x > 0 {
                    x -= LANES;
                    for (k, c) in carry.iter_mut().enumerate() {
                        let q = p.add(k * w + x);
                        let mut a = $ld(q);
                        $(a = $min(a, $add($ext::<$bs>(a, maxv), $dup($bd)));)+
                        a = $min(a, $add(*c, ramp_down));
                        $st(q, a);
                        *c = $dupl::<0>(a);
                    }
                }
            }
        }
    };
}

#[cfg(target_arch = "aarch64")]
horizontal_min_plus_neon!(
    horizontal_min_plus_neon_u16,
    u16,
    8,
    7,
    dup = vdupq_n_u16,
    ld = vld1q_u16,
    st = vst1q_u16,
    add = vqaddq_u16,
    min = vminq_u16,
    dupl = vdupq_laneq_u16,
    get = vgetq_lane_u16,
    ext = vextq_u16,
    fwd = [(7, 1), (6, 2), (4, 4)],
    bwd = [(1, 1), (2, 2), (4, 4)]
);

#[cfg(target_arch = "aarch64")]
horizontal_min_plus_neon!(
    horizontal_min_plus_neon_u8,
    u8,
    16,
    15,
    dup = vdupq_n_u8,
    ld = vld1q_u8,
    st = vst1q_u8,
    add = vqaddq_u8,
    min = vminq_u8,
    dupl = vdupq_laneq_u8,
    get = vgetq_lane_u8,
    ext = vextq_u8,
    fwd = [(15, 1), (14, 2), (12, 4), (8, 8)],
    bwd = [(1, 1), (2, 2), (4, 4), (8, 8)]
);
