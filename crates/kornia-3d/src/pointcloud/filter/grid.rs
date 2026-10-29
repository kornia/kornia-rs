//! Exact neighbour queries on a uniform grid of sorted cells (a cell list):
//! the points are sorted by cell, so a query reads its neighbour cells as a
//! few contiguous runs of a coordinate array, and the queries of one cell
//! share the cell lookups. Results equal those of the k-d tree path, which
//! stays as the fallback (keys outside `i32`, and k-NN queries whose
//! neighbourhood leaves the searched cells). Callers pass finite points
//! only; a non-finite point makes the grid constructors return `None`.

use std::ops::Range;

use rayon::prelude::*;

use crate::voxel::{CellRuns, VoxelLayout};

/// `(dx, dy)` columns of the 27-cell block.
const BLOCK_COLUMNS: usize = 9;

/// The block's columns, own column first, then face- then edge-adjacent.
const BLOCK_ORDER: [(i64, i64); BLOCK_COLUMNS] = [
    (0, 0),
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
];

/// Fewest cells handed to one rayon task.
const MIN_CELLS_PER_TASK: usize = 64;

/// The radius grid's cell edge over the radius. Above `1`, a point within
/// the radius of a query is at most one cell away from it even when both
/// keys are rounded. The grid's origin is zero, so a key is
/// `floor(fl(v / s))`: the quotient is off by at most half an ulp, which
/// for `|v / s| < 2³¹` (any `i32` key) is `2⁻²³` cells. Two points within
/// the radius are at most `1 / RADIUS_CELL_SLACK ≈ 1 − 1e-6` cells apart,
/// so their rounded quotients are less than `1 − 1e-6 + 2⁻²²` apart, below
/// one cell since `1e-6 ≈ 2⁻²⁰`: their keys differ by at most one.
const RADIUS_CELL_SLACK: f64 = 1.0 + 1e-6;

/// Relative slack taken off the k-NN termination bound, far above the
/// rounding of the cell bounds and of the keys.
const BOUND_SLACK: f64 = 1e-9;

/// Widest shell (Chebyshev distance in cells) a k-NN query searches before
/// falling back to the k-d tree.
const MAX_SHELL: i64 = 3;

/// Grid sizing rounds of the k-NN grid (see [`CellGrid::for_knn`]).
const KNN_SIZING_ROUNDS: usize = 3;

/// The k-NN grid is resized while a point shares its cell with more than
/// this many times the target load.
const KNN_MAX_LOAD_RATIO: f64 = 4.0;

/// Flat bounding-box axes count as the largest extent over this ratio, so
/// that planar and linear clouds still get several cells.
const FLAT_AXIS_RATIO: f64 = 1000.0;

/// A cloud sorted into the cells of a layout, coordinates in cell order.
/// Rebuilding a long-lived grid reuses its buffers. After a build that
/// returned `false` the contents are unspecified and must not be queried.
#[derive(Debug, Default, Clone)]
pub(super) struct CellGrid {
    /// Cell edge and origin of the layout of the last build.
    cell: f64,
    origin: [f64; 3],
    runs: CellRuns,
    xyz: Vec<[f64; 3]>,
}

impl CellGrid {
    /// Sorts `points` into the cells of `layout`; `false` if a key does not
    /// fit `i32`.
    fn build(&mut self, points: &[[f64; 3]], layout: VoxelLayout) -> bool {
        if self.runs.sort(points, &layout).is_err() {
            return false;
        }
        self.cell = layout.voxel_size();
        self.origin = layout.origin();
        self.runs
            .order
            .par_iter()
            .map(|&i| points[i as usize])
            .collect_into_vec(&mut self.xyz);
        true
    }

    /// Builds the grid of the radius filter: cells slightly wider than
    /// `radius`, so the 27 cells around a query hold all its neighbours.
    pub(super) fn for_radius(&mut self, points: &[[f64; 3]], radius: f64) -> bool {
        match VoxelLayout::new(radius * RADIUS_CELL_SLACK, [0.0; 3]) {
            Ok(layout) => self.build(points, layout),
            Err(_) => false,
        }
    }

    /// Builds the grid of the k-NN search, about `load` points per cell:
    /// sized from the bounding box, then shrunk while the mean number of
    /// points that share a point's cell exceeds `KNN_MAX_LOAD_RATIO · load`
    /// (clustered or surface clouds). `false` if no grid fits.
    pub(super) fn for_knn(&mut self, points: &[[f64; 3]], load: usize) -> bool {
        let n = points.len();
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for p in points {
            for a in 0..3 {
                if !p[a].is_finite() {
                    return false;
                }
                lo[a] = lo[a].min(p[a]);
                hi[a] = hi[a].max(p[a]);
            }
        }
        let ext: [f64; 3] = std::array::from_fn(|a| hi[a] - lo[a]);
        let flat = ext.iter().fold(0.0f64, |m, &e| m.max(e)) / FLAT_AXIS_RATIO;
        let volume: f64 = ext.iter().map(|&e| e.max(flat)).product();
        let load = load.max(1) as f64;
        let mut cell = (volume * load / n as f64).cbrt();
        for round in 0..KNN_SIZING_ROUNDS {
            if round > 0 {
                let shared = self.mean_shared_load();
                if shared <= KNN_MAX_LOAD_RATIO * load {
                    break;
                }
                cell *= (load / shared).cbrt();
            }
            let Ok(layout) = VoxelLayout::new(cell, lo) else {
                return false;
            };
            if !self.build(points, layout) {
                return false;
            }
        }
        true
    }

    /// Mean, over the points, of the number of points in their cell.
    fn mean_shared_load(&self) -> f64 {
        let squares: f64 = self
            .runs
            .starts
            .windows(2)
            .map(|w| {
                let c = f64::from(w[1] - w[0]);
                c * c
            })
            .sum();
        squares / self.xyz.len() as f64
    }

    fn cells(&self) -> usize {
        self.runs.keys.len()
    }

    fn cell_points(&self, c: usize) -> Range<usize> {
        self.runs.starts[c] as usize..self.runs.starts[c + 1] as usize
    }

    /// Appends the sorted-position ranges of the 27 cells around cell `c`
    /// (itself included), nearest first: the cell itself, the rest of its
    /// `z` column, the face-adjacent columns, then the diagonal ones (a k-NN
    /// scan then fills its best list early and rejects most later points).
    /// `hints` keeps one column-search hint per `(dx, dy)` across calls, so
    /// consecutive cells cost amortized O(1) lookups.
    fn block(&self, c: usize, hints: &mut [usize; BLOCK_COLUMNS], out: &mut Vec<Range<usize>>) {
        let k = self.runs.key(c);
        let (x, y, z) = (i64::from(k[0]), i64::from(k[1]), i64::from(k[2]));
        let own = self.cell_points(c);
        let mut push = |r: Range<usize>| {
            if !r.is_empty() {
                out.push(r);
            }
        };
        for (j, &(dx, dy)) in BLOCK_ORDER.iter().enumerate() {
            let r = self
                .runs
                .column(&mut hints[j], x + dx, y + dy, z - 1, z + 1);
            if j == 0 {
                // The own column holds `own`: own cell first, then the rest.
                push(own.clone());
                push(r.start..own.start);
                push(own.end..r.end);
            } else {
                push(r);
            }
        }
    }

    /// Appends the sorted-position ranges of the cells at Chebyshev
    /// distance exactly `l >= 2` from cell `c`. The columns are visited in
    /// ascending key order, so one search hint serves them all.
    fn shell(&self, c: usize, l: i64, out: &mut Vec<Range<usize>>) {
        let k = self.runs.key(c);
        let (x, y, z) = (i64::from(k[0]), i64::from(k[1]), i64::from(k[2]));
        let mut hint = 0;
        let mut push = |r: Range<usize>| {
            if !r.is_empty() {
                out.push(r);
            }
        };
        for dx in -l..=l {
            for dy in -l..=l {
                if dx.abs().max(dy.abs()) == l {
                    push(self.runs.column(&mut hint, x + dx, y + dy, z - l, z + l));
                } else {
                    push(self.runs.column(&mut hint, x + dx, y + dy, z - l, z - l));
                    push(self.runs.column(&mut hint, x + dx, y + dy, z + l, z + l));
                }
            }
        }
    }

    /// Runs `f(cell, positions, slice, scratch)` on every cell, in
    /// parallel, where `slice` is the cell's part of `dst` (one slot per
    /// sorted position). Rayon splits the cell range adaptively, in halves
    /// at cell boundaries and never below [`MIN_CELLS_PER_TASK`] cells, as
    /// `with_min_len` did; no per-cell list of slices is allocated.
    fn for_each_cell<T: Send, S>(
        &self,
        dst: &mut [T],
        init: impl Fn() -> S + Sync + Send,
        f: impl Fn(usize, Range<usize>, &mut [T], &mut S) + Sync + Send,
    ) {
        let starts = &self.runs.starts;
        rayon::iter::split((0..self.cells(), dst), |(cells, dst)| {
            if cells.len() < 2 * MIN_CELLS_PER_TASK {
                return ((cells, dst), None);
            }
            let mid = cells.start + cells.len() / 2;
            let at = (starts[mid] - starts[cells.start]) as usize;
            let (lo, hi) = dst.split_at_mut(at);
            ((cells.start..mid, lo), Some((mid..cells.end, hi)))
        })
        .for_each_init(init, |scratch, (cells, dst)| {
            let mut rest = dst;
            for c in cells {
                let own = self.cell_points(c);
                let (head, tail) = rest.split_at_mut(own.len());
                f(c, own, head, scratch);
                rest = tail;
            }
        });
    }

    /// Scatters per-sorted-position values to input order.
    fn scatter<T: Copy>(&self, sorted: &[T], dst: &mut [T]) {
        for (&i, &v) in self.runs.order.iter().zip(sorted) {
            dst[i as usize] = v;
        }
    }
}

/// Sets `buf` to `n` elements whose values the caller overwrites in full:
/// a buffer that is too short is replaced by a fresh `vec![fill; n]` (zeroed
/// pages for a zero `fill`), a long enough one is only truncated, so reuse
/// writes nothing and a fresh buffer costs what `vec!` does.
pub(super) fn resize_for_overwrite<T: Clone>(buf: &mut Vec<T>, n: usize, fill: T) {
    if buf.len() < n {
        *buf = vec![fill; n];
    } else {
        buf.truncate(n);
    }
}

#[inline]
fn dist2(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
}

/// Per input point, whether at least `cap` points (itself included) lie
/// within `radius` (squared distance `<= radius_sq`, as the k-d tree path).
/// `sorted` is scratch for the per-sorted-position flags.
pub(super) fn radius_dense(
    grid: &CellGrid,
    radius_sq: f64,
    cap: usize,
    sorted: &mut Vec<bool>,
    dense: &mut [bool],
) {
    resize_for_overwrite(sorted, grid.xyz.len(), false);
    grid.for_each_cell(
        sorted,
        || (Vec::new(), [0usize; BLOCK_COLUMNS]),
        |c, own, out, (ranges, hints): &mut (Vec<Range<usize>>, [usize; BLOCK_COLUMNS])| {
            ranges.clear();
            grid.block(c, hints, ranges);
            for (q, slot) in grid.xyz[own].iter().zip(out.iter_mut()) {
                let mut hits = 0usize;
                'search: for r in ranges.iter() {
                    for p in &grid.xyz[r.clone()] {
                        if dist2(q, p) <= radius_sq {
                            hits += 1;
                            if hits >= cap {
                                break 'search;
                            }
                        }
                    }
                }
                *slot = hits >= cap;
            }
        },
    );
    grid.scatter(sorted, dense);
}

/// Per-task scratch of [`knn_mean_distances`].
#[derive(Default)]
struct KnnScratch {
    /// Ranges of the block and shells `2..` searched so far for the
    /// current cell, and where each level starts in `ranges`.
    ranges: Vec<Range<usize>>,
    level_start: Vec<usize>,
    hints: [usize; BLOCK_COLUMNS],
    /// Ascending squared distances of the best candidates.
    best: Vec<f64>,
}

/// Inserts `d` into the ascending `best`, keeping at most `k` entries. The
/// shift is an element loop: the lists are short, and a `memmove` per
/// insertion costs more than the scan.
#[inline]
fn push_best(best: &mut Vec<f64>, k: usize, d: f64) {
    let mut i = best.len();
    if i == k {
        match best.last() {
            Some(&last) if d < last => i -= 1,
            _ => return,
        }
    } else {
        best.push(d);
    }
    while i > 0 && best[i - 1] > d {
        best[i] = best[i - 1];
        i -= 1;
    }
    best[i] = d;
}

/// Per input point, the mean distance to its `hits − 1` nearest other points
/// (`hits` nearest including itself), summed over the ascending squared
/// distances exactly as `mean_of_sorted` does for the k-d tree. Queries
/// whose `hits`-th neighbour may lie beyond [`MAX_SHELL`] cells call
/// `fallback`. `sorted` is scratch for the per-sorted-position means.
pub(super) fn knn_mean_distances(
    grid: &CellGrid,
    hits: usize,
    fallback: &(impl Fn(&[f64; 3]) -> f64 + Sync),
    sorted: &mut Vec<f64>,
    mean: &mut [f64],
) {
    let s = grid.cell;
    let o = grid.origin;
    resize_for_overwrite(sorted, grid.xyz.len(), 0.0);
    grid.for_each_cell(
        sorted,
        KnnScratch::default,
        |c, own, out, sc: &mut KnnScratch| {
            sc.ranges.clear();
            sc.level_start.clear();
            let key = grid.runs.key(c);
            for (q, slot) in grid.xyz[own].iter().zip(out.iter_mut()) {
                sc.best.clear();
                let mut done = false;
                // Level `l`: every cell within Chebyshev distance `l`.
                for l in 1..=MAX_SHELL {
                    let li = (l - 1) as usize;
                    if sc.level_start.len() <= li {
                        sc.level_start.push(sc.ranges.len());
                        if l == 1 {
                            grid.block(c, &mut sc.hints, &mut sc.ranges);
                        } else {
                            grid.shell(c, l, &mut sc.ranges);
                        }
                    }
                    let end = sc
                        .level_start
                        .get(li + 1)
                        .copied()
                        .unwrap_or(sc.ranges.len());
                    for r in &sc.ranges[sc.level_start[li]..end] {
                        for p in &grid.xyz[r.clone()] {
                            push_best(&mut sc.best, hits, dist2(q, p));
                        }
                    }
                    if sc.best.len() == hits {
                        let b = shell_bound(q, &key, l, s, &o);
                        if b > 0.0 && sc.best[hits - 1] <= b * b {
                            done = true;
                            break;
                        }
                    }
                }
                *slot = if done {
                    mean_of_sorted(&sc.best)
                } else {
                    fallback(q)
                };
            }
        },
    );
    grid.scatter(sorted, mean);
}

/// A lower bound on the distance from `q` (in cell `key`) to any point in a
/// cell more than `l` cells away, less a rounding slack.
#[inline]
fn shell_bound(q: &[f64; 3], key: &[i32; 3], l: i64, s: f64, o: &[f64; 3]) -> f64 {
    let mut bound = f64::INFINITY;
    for a in 0..3 {
        let k = i64::from(key[a]);
        let lo = o[a] + (k - l) as f64 * s;
        let hi = o[a] + (k + l + 1) as f64 * s;
        let slack = BOUND_SLACK * (q[a].abs() + o[a].abs() + ((k.abs() + l + 1) as f64) * s);
        bound = bound.min((q[a] - lo).min(hi - q[a]) - slack);
    }
    bound
}

/// Mean of the square roots of ascending squared distances, the first one
/// (the query itself) not counted as a neighbour.
#[inline]
pub(super) fn mean_of_sorted(sorted_sq: &[f64]) -> f64 {
    let neighbours = sorted_sq.len().saturating_sub(1);
    if neighbours == 0 {
        return 0.0;
    }
    let sum: f64 = sorted_sq.iter().map(|d| d.sqrt()).sum();
    sum / neighbours as f64
}
