//! Outlier removal on point clouds, after the radius and statistical outlier
//! filters of PCL (`pcl::RadiusOutlierRemoval`,
//! `pcl::StatisticalOutlierRemoval`) and Open3D
//! (`PointCloud::remove_radius_outlier`,
//! `PointCloud::remove_statistical_outlier`), written from their public
//! documentation.
//!
//! Every filter keeps the surviving points in their input order. The
//! `*_inlier_indices` forms write the kept indices (ascending) into a
//! caller-owned buffer; the `remove_*` forms return the filtered
//! [`PointCloud`] with its colors and normals carried along. Both allocate
//! their search scratch on every call; [`OutlierFilter`] owns it and reuses
//! it across calls, which is the allocation-free path for repeated filtering.
//!
//! Non-finite points (any coordinate NaN or infinite) are dropped, as PCL
//! does: they are never inliers, never count as a neighbour, and are left
//! out of the statistical filter's global mean and deviation. A cloud's
//! result is that of the same cloud with its non-finite points removed,
//! indices mapped back to the input.
//!
//! Neighbour searches run on a uniform grid of cells (a cell list, as in
//! Open3D's fixed-radius search): the points are sorted by cell with the
//! voxel sort of [`crate::voxel`], and each query scans the neighbouring
//! cells as contiguous runs, in parallel over the cells. The radius filter
//! uses cells as wide as the radius; the statistical filter sizes them for
//! about `(k + 1) / 2` points and widens a query's search shell by shell
//! until no unvisited cell can hold a closer point. The search runs on the
//! finite points only, compacted once when the cloud has non-finite ones.
//! The `kiddo` k-d tree (`ImmutableKdTree`) answers the queries the grid
//! cannot (keys outside `i32`, k-NN neighbourhoods beyond three cells).
//! Both give the same result bit for bit.

use std::num::NonZero;

use kiddo::immutable::float::kdtree::ImmutableKdTree;
use kiddo::SquaredEuclidean;
use rayon::prelude::*;

use super::PointCloud;
use grid::CellGrid;

mod grid;

/// Points per k-d tree leaf bucket.
const KD_BUCKET: usize = 32;

/// Smallest number of queries handed to one rayon task.
const MIN_QUERIES_PER_TASK: usize = 256;

/// Smallest cloud the statistical filter searches on a grid; below it the
/// k-d tree alone is cheaper.
const KNN_GRID_MIN_POINTS: usize = 1024;

/// The k-NN grid holds about `hits / KNN_CELL_LOAD_DIVISOR` points per cell
/// (`hits = k + 1`), so the 27 cells around a query usually hold its `k`
/// neighbours: the `k`-NN ball fills under a quarter of a cell of that load.
const KNN_CELL_LOAD_DIVISOR: usize = 2;

type KdTree = ImmutableKdTree<f64, u32, 3, KD_BUCKET>;

/// Errors of the point cloud filters.
#[derive(Debug, thiserror::Error)]
pub enum FilterError {
    /// The search radius is not finite and positive, or its square
    /// overflows `f64`.
    #[error("radius must be finite and positive, got {0}")]
    InvalidRadius(f64),

    /// The neighbour count `k` of the statistical filter is zero.
    #[error("the number of neighbours k must be at least 1")]
    InvalidNeighbourCount,

    /// The standard deviation multiplier is not finite.
    #[error("std_ratio must be finite, got {0}")]
    InvalidStdRatio(f64),

    /// The cloud has more points than `u32` indices can address.
    #[error("{len} points exceed the maximum of {max}")]
    TooManyPoints {
        /// Points in the cloud.
        len: usize,
        /// Largest supported cloud.
        max: usize,
    },

    /// A selected index is not a point of the cloud.
    #[error("index {index} is out of range for {len} points")]
    IndexOutOfRange {
        /// Offending index.
        index: u32,
        /// Points in the cloud.
        len: usize,
    },

    /// The colors or normals of a cloud do not have one entry per point.
    #[error("{attribute} has {len} entries for {expected} points")]
    AttributeLengthMismatch {
        /// `"colors"` or `"normals"`.
        attribute: &'static str,
        /// Entries of the attribute.
        len: usize,
        /// Points in the cloud.
        expected: usize,
    },
}

/// Largest cloud the filters accept: indices are `u32`.
pub const MAX_FILTER_POINTS: usize = u32::MAX as usize;

/// Checks that `cloud`'s colors and normals, when present, have one entry
/// per point.
pub(crate) fn check_attributes(cloud: &PointCloud) -> Result<(), FilterError> {
    let expected = cloud.len();
    if let Some(c) = cloud.colors() {
        if c.len() != expected {
            return Err(FilterError::AttributeLengthMismatch {
                attribute: "colors",
                len: c.len(),
                expected,
            });
        }
    }
    if let Some(n) = cloud.normals() {
        if n.len() != expected {
            return Err(FilterError::AttributeLengthMismatch {
                attribute: "normals",
                len: n.len(),
                expected,
            });
        }
    }
    Ok(())
}

/// The finite points of a cloud that has non-finite ones, compacted in
/// input order, with the input index of each.
#[derive(Debug, Default, Clone)]
struct FiniteSubset {
    points: Vec<[f64; 3]>,
    index: Vec<u32>,
}

impl FiniteSubset {
    /// Compacts the finite points of `points` into `self`. `false`, with
    /// `self` untouched, when every point is finite, so the input is
    /// searched as is.
    fn compact(&mut self, points: &[[f64; 3]]) -> bool {
        let finite = |p: &[f64; 3]| p.iter().all(|v| v.is_finite());
        if points.iter().all(finite) {
            return false;
        }
        self.points.clear();
        self.index.clear();
        for (i, p) in points.iter().enumerate().filter(|(_, p)| finite(p)) {
            self.points.push(*p);
            self.index.push(i as u32);
        }
        true
    }
}

/// Runs `filter` on the finite points of `points` (compacted into `subset`
/// when some are not finite) and writes the kept indices, mapped back to
/// `points`, into `dst` (ascending, as the compaction keeps the input
/// order).
fn on_finite_points(
    subset: &mut FiniteSubset,
    points: &[[f64; 3]],
    dst: &mut Vec<u32>,
    filter: impl FnOnce(&[[f64; 3]], &mut Vec<u32>),
) {
    dst.clear();
    if subset.compact(points) {
        filter(&subset.points, dst);
        for i in dst.iter_mut() {
            *i = subset.index[*i as usize];
        }
    } else {
        filter(points, dst);
    }
}

/// Search buffers of [`OutlierFilter`], apart from the finite subset the
/// search runs on.
#[derive(Debug, Default, Clone)]
struct SearchScratch {
    grid: CellGrid,
    /// Radius rule result per point, in search order and in grid order.
    dense: Vec<bool>,
    sorted_dense: Vec<bool>,
    /// Mean k-NN distance per point, in search order and in grid order.
    mean_dist: Vec<f64>,
    sorted_mean: Vec<f64>,
}

/// Reusable scratch of the radius and statistical outlier filters.
///
/// [`radius_inlier_indices`] and [`statistical_inlier_indices`] allocate
/// their search buffers (the cell grid, its sorted coordinates and the
/// per-point results) on every call. An `OutlierFilter` keeps them and
/// reuses them: after a call on a cloud of `n` points, later calls on clouds
/// of at most `n` points allocate nothing on the grid path. Results are
/// identical to the free functions.
///
/// The k-d tree fallback (keys outside `i32`, k-NN neighbourhoods wider
/// than three cells, statistical clouds under 1024 points) still builds its
/// tree per call, and the parallel grid search allocates a few small
/// per-task lists (the cell ranges and the k-NN candidates).
///
/// # Example
///
/// ```
/// use kornia_3d::pointcloud::filter::OutlierFilter;
///
/// let mut filter = OutlierFilter::new();
/// let mut kept = Vec::new();
/// for points in [
///     vec![[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [5.0, 0.0, 0.0]],
///     vec![[0.0, 0.0, 0.0], [9.0, 0.0, 0.0]],
/// ] {
///     filter.radius_inlier_indices_into(&points, 0.5, 1, &mut kept)?;
/// }
/// assert!(kept.is_empty());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Default, Clone)]
pub struct OutlierFilter {
    finite: FiniteSubset,
    search: SearchScratch,
}

fn check_len(len: usize) -> Result<(), FilterError> {
    if len > MAX_FILTER_POINTS {
        return Err(FilterError::TooManyPoints {
            len,
            max: MAX_FILTER_POINTS,
        });
    }
    Ok(())
}

/// Copies the points of `cloud` at `indices`, with their colors and normals.
///
/// # Arguments
///
/// * `cloud` - Source cloud.
/// * `indices` - Indices of the points to keep, in output order (repeats
///   allowed).
///
/// # Returns
///
/// A cloud with one point per index.
///
/// # Errors
///
/// * [`FilterError::IndexOutOfRange`] if an index is not a point of `cloud`.
/// * [`FilterError::AttributeLengthMismatch`] if `cloud`'s colors or normals
///   do not have one entry per point.
///
/// # Example
///
/// ```
/// use kornia_3d::pointcloud::{filter::select_points, PointCloud};
///
/// let cloud = PointCloud::new(vec![[0.0; 3], [1.0; 3], [2.0; 3]], None, None);
/// let picked = select_points(&cloud, &[2, 0])?;
/// assert_eq!(picked.points(), &vec![[2.0; 3], [0.0; 3]]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn select_points(cloud: &PointCloud, indices: &[u32]) -> Result<PointCloud, FilterError> {
    check_attributes(cloud)?;
    let len = cloud.len();
    if let Some(&index) = indices.iter().find(|&&i| i as usize >= len) {
        return Err(FilterError::IndexOutOfRange { index, len });
    }
    let pick = |i: &u32| *i as usize;
    let points = indices.iter().map(|i| cloud.points()[pick(i)]).collect();
    let colors = cloud
        .colors()
        .map(|c| indices.iter().map(|i| c[pick(i)]).collect());
    let normals = cloud
        .normals()
        .map(|n| indices.iter().map(|i| n[pick(i)]).collect());
    Ok(PointCloud::new(points, colors, normals))
}

/// Indices of the points with at least `min_neighbours` other points within
/// `radius` (radius outlier removal).
///
/// The query point itself is not counted, and a neighbour at exactly
/// `radius` counts. This is PCL's `RadiusOutlierRemoval` with
/// `setMinNeighborsInRadius(min_neighbours)` and Open3D's legacy
/// `PointCloud.remove_radius_outlier(nb_points = min_neighbours, radius)`;
/// Open3D's tensor `remove_radius_outliers` counts the query point itself,
/// so its `nb_points` is `min_neighbours + 1`. Each query stops
/// collecting after `min_neighbours + 1` hits, so the cost does not grow
/// with the density of the inliers.
///
/// A non-finite point (any coordinate NaN or infinite) is never kept and
/// never counts as a neighbour of another point.
///
/// Allocates its search scratch on every call;
/// [`OutlierFilter::radius_inlier_indices_into`] reuses it across calls.
///
/// # Arguments
///
/// * `points` - Input points.
/// * `radius` - Search radius, in the units of `points`.
/// * `min_neighbours` - Neighbours (excluding the point) needed to keep a
///   point; `0` keeps every finite point.
/// * `dst` - Cleared, then filled with the kept indices in ascending order.
///
/// # Returns
///
/// `Ok(())` once `dst` is written.
///
/// # Errors
///
/// * [`FilterError::InvalidRadius`] if `radius` is not finite and positive.
/// * [`FilterError::TooManyPoints`] above [`MAX_FILTER_POINTS`] points.
///
/// # Example
///
/// ```
/// use kornia_3d::pointcloud::filter::radius_inlier_indices;
///
/// let points = [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [5.0, 0.0, 0.0]];
/// let mut kept = Vec::new();
/// radius_inlier_indices(&points, 0.5, 1, &mut kept)?;
/// assert_eq!(kept, vec![0, 1]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn radius_inlier_indices(
    points: &[[f64; 3]],
    radius: f64,
    min_neighbours: usize,
    dst: &mut Vec<u32>,
) -> Result<(), FilterError> {
    OutlierFilter::new().radius_inlier_indices_into(points, radius, min_neighbours, dst)
}

/// [`radius_inlier_indices`] on finite points, into an empty `dst`.
fn radius_inliers_of_finite(
    search: &mut SearchScratch,
    points: &[[f64; 3]],
    radius: f64,
    radius_sq: f64,
    min_neighbours: usize,
    dst: &mut Vec<u32>,
) {
    if min_neighbours == 0 {
        dst.extend(0..points.len() as u32);
        return;
    }
    if points.len() <= min_neighbours {
        return;
    }
    // `min_neighbours + 1` cannot overflow: `min_neighbours < points.len()`.
    let Some(cap) = NonZero::new(min_neighbours + 1) else {
        return;
    };
    let dense = &mut search.dense;
    grid::resize_for_overwrite(dense, points.len(), false);
    if search.grid.for_radius(points, radius) {
        grid::radius_dense(
            &search.grid,
            radius_sq,
            cap.get(),
            &mut search.sorted_dense,
            dense,
        );
    } else {
        radius_dense_kd(points, radius_sq, cap, dense);
    }
    dst.extend(
        dense
            .iter()
            .enumerate()
            .filter(|&(_, &d)| d)
            .map(|(i, _)| i as u32),
    );
}

/// The k-d tree form of [`grid::radius_dense`]: whether the `cap` nearest
/// points within the radius (the point itself included) are all found. The
/// capped search always holds the point itself (distance 0) unless `cap`
/// exact duplicates crowd it out, which also means `cap − 1` neighbours.
fn radius_dense_kd(points: &[[f64; 3]], radius_sq: f64, cap: NonZero<usize>, dense: &mut [bool]) {
    let tree = KdTree::new_from_slice(points);
    dense
        .par_iter_mut()
        .with_min_len(MIN_QUERIES_PER_TASK)
        .zip(points.par_iter())
        .for_each(|(d, p)| {
            *d = tree
                .nearest_n_within::<SquaredEuclidean>(p, radius_sq, cap, false)
                .len()
                >= cap.get();
        });
}

/// Removes the points with fewer than `min_neighbours` other points within
/// `radius`; see [`radius_inlier_indices`] for the exact rule.
///
/// # Arguments
///
/// * `cloud` - Input cloud; colors and normals are carried along.
/// * `radius` - Search radius, in the units of the points.
/// * `min_neighbours` - Neighbours (excluding the point) needed to keep it.
///
/// # Returns
///
/// The kept points, in input order.
///
/// # Errors
///
/// The errors of [`radius_inlier_indices`], and
/// [`FilterError::AttributeLengthMismatch`] for malformed colors or normals.
///
/// # Example
///
/// ```
/// use kornia_3d::pointcloud::{filter::remove_radius_outliers, PointCloud};
///
/// let cloud = PointCloud::new(vec![[0.0; 3], [0.1, 0.0, 0.0], [5.0, 0.0, 0.0]], None, None);
/// assert_eq!(remove_radius_outliers(&cloud, 0.5, 1)?.len(), 2);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn remove_radius_outliers(
    cloud: &PointCloud,
    radius: f64,
    min_neighbours: usize,
) -> Result<PointCloud, FilterError> {
    check_attributes(cloud)?;
    let mut kept = Vec::new();
    radius_inlier_indices(cloud.points(), radius, min_neighbours, &mut kept)?;
    select_points(cloud, &kept)
}

/// Indices of the points whose mean distance to their `k` nearest neighbours
/// is at most `mean + std_ratio · std` of that statistic over the cloud
/// (statistical outlier removal).
///
/// The rule follows PCL's `StatisticalOutlierRemoval` (`setMeanK(k)`,
/// `setStddevMulThresh(std_ratio)`): the query point is not one of its own
/// `k` neighbours, distances are Euclidean, the standard deviation is the
/// sample one (`n − 1` denominator) and the comparison is `<=`. A cloud with
/// at most `k` points uses all other points as neighbours; a single point is
/// kept. The global mean and deviation are summed sequentially in point
/// order, so the result is deterministic.
///
/// A non-finite point (any coordinate NaN or infinite) is never kept, is
/// never a neighbour, and is left out of the global mean and deviation:
/// `n` and the cloud size above count the finite points only.
///
/// A negative `std_ratio` is accepted, as in PCL: the threshold then lies
/// below the mean, so fewer than half of a symmetric distribution is kept.
///
/// Allocates its search scratch on every call;
/// [`OutlierFilter::statistical_inlier_indices_into`] reuses it across
/// calls.
///
/// # Arguments
///
/// * `points` - Input points.
/// * `k` - Neighbours per point, at least `1`.
/// * `std_ratio` - Standard deviation multiplier, any finite value; smaller
///   removes more.
/// * `dst` - Cleared, then filled with the kept indices in ascending order.
///
/// # Returns
///
/// `Ok(())` once `dst` is written.
///
/// # Errors
///
/// * [`FilterError::InvalidNeighbourCount`] if `k == 0`.
/// * [`FilterError::InvalidStdRatio`] if `std_ratio` is not finite.
/// * [`FilterError::TooManyPoints`] above [`MAX_FILTER_POINTS`] points.
///
/// # Example
///
/// ```
/// use kornia_3d::pointcloud::filter::statistical_inlier_indices;
///
/// let mut points: Vec<[f64; 3]> = (0..10).map(|i| [i as f64 * 0.1, 0.0, 0.0]).collect();
/// points.push([50.0, 0.0, 0.0]);
/// let mut kept = Vec::new();
/// statistical_inlier_indices(&points, 2, 1.0, &mut kept)?;
/// assert_eq!(kept, (0..10).collect::<Vec<u32>>());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn statistical_inlier_indices(
    points: &[[f64; 3]],
    k: usize,
    std_ratio: f64,
    dst: &mut Vec<u32>,
) -> Result<(), FilterError> {
    OutlierFilter::new().statistical_inlier_indices_into(points, k, std_ratio, dst)
}

/// [`statistical_inlier_indices`] on finite points, into an empty `dst`.
fn statistical_inliers_of_finite(
    search: &mut SearchScratch,
    points: &[[f64; 3]],
    k: usize,
    std_ratio: f64,
    dst: &mut Vec<u32>,
) {
    let n = points.len();
    if n <= 1 {
        dst.extend(0..n as u32);
        return;
    }
    // The point itself is always among its `k + 1` nearest (distance 0, or a
    // duplicate at distance 0), so the `k + 1` results minus one zero are the
    // `k` neighbours.
    let query = k.min(n - 1) + 1;
    let Some(cap) = NonZero::new(query) else {
        return;
    };
    let mean_dist = &mut search.mean_dist;
    grid::resize_for_overwrite(mean_dist, n, 0.0);
    let on_grid = n >= KNN_GRID_MIN_POINTS
        && search
            .grid
            .for_knn(points, query.div_ceil(KNN_CELL_LOAD_DIVISOR));
    if on_grid {
        let tree = std::sync::OnceLock::new();
        let fallback =
            |p: &[f64; 3]| knn_mean_kd(tree.get_or_init(|| KdTree::new_from_slice(points)), p, cap);
        grid::knn_mean_distances(
            &search.grid,
            query,
            &fallback,
            &mut search.sorted_mean,
            mean_dist,
        );
    } else {
        let tree = KdTree::new_from_slice(points);
        mean_dist
            .par_iter_mut()
            .with_min_len(MIN_QUERIES_PER_TASK)
            .zip(points.par_iter())
            .for_each(|(d, p)| *d = knn_mean_kd(&tree, p, cap));
    }

    let count = n as f64;
    let mean = mean_dist.iter().sum::<f64>() / count;
    let var = mean_dist
        .iter()
        .map(|d| (d - mean) * (d - mean))
        .sum::<f64>()
        / (count - 1.0);
    let threshold = mean + std_ratio * var.sqrt();
    dst.extend(
        mean_dist
            .iter()
            .enumerate()
            .filter(|&(_, &d)| d <= threshold)
            .map(|(i, _)| i as u32),
    );
}

impl OutlierFilter {
    /// Creates an empty scratch.
    ///
    /// # Arguments
    ///
    /// This function takes no arguments.
    ///
    /// # Returns
    ///
    /// A filter with no buffers allocated yet.
    ///
    /// # Errors
    ///
    /// This function is infallible.
    pub fn new() -> Self {
        Self::default()
    }

    /// [`radius_inlier_indices`], reusing this filter's search buffers; see
    /// it for the exact rule. The result is the same.
    ///
    /// # Arguments
    ///
    /// * `points` - Input points.
    /// * `radius` - Search radius, in the units of `points`.
    /// * `min_neighbours` - Neighbours (excluding the point) needed to keep a
    ///   point; `0` keeps every finite point.
    /// * `dst` - Cleared, then filled with the kept indices in ascending order.
    ///
    /// # Returns
    ///
    /// `Ok(())` once `dst` is written.
    ///
    /// # Errors
    ///
    /// * [`FilterError::InvalidRadius`] if `radius` is not finite and positive.
    /// * [`FilterError::TooManyPoints`] above [`MAX_FILTER_POINTS`] points.
    ///
    /// # Example
    ///
    /// ```
    /// use kornia_3d::pointcloud::filter::OutlierFilter;
    ///
    /// let points = [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [5.0, 0.0, 0.0]];
    /// let mut kept = Vec::new();
    /// OutlierFilter::new().radius_inlier_indices_into(&points, 0.5, 1, &mut kept)?;
    /// assert_eq!(kept, vec![0, 1]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn radius_inlier_indices_into(
        &mut self,
        points: &[[f64; 3]],
        radius: f64,
        min_neighbours: usize,
        dst: &mut Vec<u32>,
    ) -> Result<(), FilterError> {
        let radius_sq = radius * radius;
        if !(radius.is_finite() && radius > 0.0 && radius_sq.is_finite()) {
            return Err(FilterError::InvalidRadius(radius));
        }
        check_len(points.len())?;
        let Self { finite, search } = self;
        on_finite_points(finite, points, dst, |finite, dst| {
            radius_inliers_of_finite(search, finite, radius, radius_sq, min_neighbours, dst)
        });
        Ok(())
    }

    /// [`statistical_inlier_indices`], reusing this filter's search buffers;
    /// see it for the exact rule. The result is the same.
    ///
    /// # Arguments
    ///
    /// * `points` - Input points.
    /// * `k` - Neighbours per point, at least `1`.
    /// * `std_ratio` - Standard deviation multiplier, any finite value; a
    ///   negative one puts the threshold below the mean, as in PCL.
    /// * `dst` - Cleared, then filled with the kept indices in ascending order.
    ///
    /// # Returns
    ///
    /// `Ok(())` once `dst` is written.
    ///
    /// # Errors
    ///
    /// * [`FilterError::InvalidNeighbourCount`] if `k == 0`.
    /// * [`FilterError::InvalidStdRatio`] if `std_ratio` is not finite.
    /// * [`FilterError::TooManyPoints`] above [`MAX_FILTER_POINTS`] points.
    ///
    /// # Example
    ///
    /// ```
    /// use kornia_3d::pointcloud::filter::OutlierFilter;
    ///
    /// let mut points: Vec<[f64; 3]> = (0..10).map(|i| [i as f64 * 0.1, 0.0, 0.0]).collect();
    /// points.push([50.0, 0.0, 0.0]);
    /// let mut kept = Vec::new();
    /// OutlierFilter::new().statistical_inlier_indices_into(&points, 2, 1.0, &mut kept)?;
    /// assert_eq!(kept, (0..10).collect::<Vec<u32>>());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn statistical_inlier_indices_into(
        &mut self,
        points: &[[f64; 3]],
        k: usize,
        std_ratio: f64,
        dst: &mut Vec<u32>,
    ) -> Result<(), FilterError> {
        if k == 0 {
            return Err(FilterError::InvalidNeighbourCount);
        }
        if !std_ratio.is_finite() {
            return Err(FilterError::InvalidStdRatio(std_ratio));
        }
        check_len(points.len())?;
        let Self { finite, search } = self;
        on_finite_points(finite, points, dst, |finite, dst| {
            statistical_inliers_of_finite(search, finite, k, std_ratio, dst)
        });
        Ok(())
    }
}

/// Mean distance from `p` to its `cap − 1` nearest other points, from the
/// `cap` nearest (`p` itself first, at distance 0).
fn knn_mean_kd(tree: &KdTree, p: &[f64; 3], cap: NonZero<usize>) -> f64 {
    let hits = tree.nearest_n::<SquaredEuclidean>(p, cap);
    let sorted_sq: Vec<f64> = hits.iter().map(|h| h.distance).collect();
    grid::mean_of_sorted(&sorted_sq)
}

/// Removes the points whose mean distance to their `k` nearest neighbours
/// exceeds `mean + std_ratio · std`; see [`statistical_inlier_indices`] for
/// the exact rule.
///
/// # Arguments
///
/// * `cloud` - Input cloud; colors and normals are carried along.
/// * `k` - Neighbours per point, at least `1`.
/// * `std_ratio` - Standard deviation multiplier, any finite value (negative
///   puts the threshold below the mean, as in PCL).
///
/// # Returns
///
/// The kept points, in input order.
///
/// # Errors
///
/// The errors of [`statistical_inlier_indices`], and
/// [`FilterError::AttributeLengthMismatch`] for malformed colors or normals.
///
/// # Example
///
/// ```
/// use kornia_3d::pointcloud::{filter::remove_statistical_outliers, PointCloud};
///
/// let mut points: Vec<[f64; 3]> = (0..10).map(|i| [i as f64 * 0.1, 0.0, 0.0]).collect();
/// points.push([50.0, 0.0, 0.0]);
/// let cloud = PointCloud::new(points, None, None);
/// assert_eq!(remove_statistical_outliers(&cloud, 2, 1.0)?.len(), 10);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn remove_statistical_outliers(
    cloud: &PointCloud,
    k: usize,
    std_ratio: f64,
) -> Result<PointCloud, FilterError> {
    check_attributes(cloud)?;
    let mut kept = Vec::new();
    statistical_inlier_indices(cloud.points(), k, std_ratio, &mut kept)?;
    select_points(cloud, &kept)
}

#[cfg(test)]
mod tests;
