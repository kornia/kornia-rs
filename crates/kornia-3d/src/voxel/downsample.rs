//! Sort-by-key voxel reductions on the host: every point gets a packed
//! 128-bit sort word (biased voxel key `x, y, z` in the top 96 bits, point
//! index in the low 32), the words are sorted, and each run of equal keys is
//! one voxel, its points in ascending index.

use std::borrow::Cow;

use rayon::prelude::*;

use super::{
    axis_centre, axis_key, VoxelConnectivity, VoxelError, VoxelKey, VoxelLayout, MAX_VOXEL_POINTS,
};
use crate::pointcloud::filter::check_attributes;
use crate::pointcloud::{PointCloud, PointCloudView};

/// Smallest number of points or voxels handed to one rayon task.
const MIN_ITEMS_PER_TASK: usize = 4096;

/// Flips the sign bit so that unsigned order equals signed order.
const KEY_BIAS: u32 = 0x8000_0000;

/// Sort word of a point that has no valid key; sorts after every valid one
/// (a valid word's index is at most [`MAX_VOXEL_POINTS`] − 1 < `u32::MAX`).
const INVALID_WORD: u128 = u128::MAX;

/// Which point represents a voxel in [`voxel_downsample`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DownsampleMode {
    /// The mean of the voxel's points (Open3D `voxel_down_sample`, PCL
    /// `VoxelGrid`). Colors are averaged and rounded to nearest; normals
    /// are averaged and renormalized (a zero mean stays zero).
    Centroid,
    /// The voxel's point with the lowest input index, unchanged.
    First,
    /// The voxel's point nearest to the voxel centre, unchanged; ties go to
    /// the lowest input index.
    NearestToCentre,
}

#[inline]
fn biased(k: i32) -> u32 {
    (k as u32) ^ KEY_BIAS
}

#[inline]
fn unbiased(b: u32) -> i32 {
    (b ^ KEY_BIAS) as i32
}

#[inline]
fn word_from_key(key: &VoxelKey, index: u32) -> u128 {
    (u128::from(biased(key[0])) << 96)
        | (u128::from(biased(key[1])) << 64)
        | (u128::from(biased(key[2])) << 32)
        | u128::from(index)
}

/// The 96-bit key part of a sort word (index dropped).
#[inline]
fn key_bits(word: u128) -> u128 {
    word >> 32
}

#[inline]
fn key_from_word(word: u128) -> VoxelKey {
    [
        unbiased((word >> 96) as u32),
        unbiased((word >> 64) as u32),
        unbiased((word >> 32) as u32),
    ]
}

#[inline]
fn index_from_word(word: u128) -> usize {
    (word as u32) as usize
}

/// Packs biased key components (`key_bits` layout).
#[inline]
fn pack_biased(x: u32, y: u32, z: u32) -> u128 {
    (u128::from(x) << 64) | (u128::from(y) << 32) | u128::from(z)
}

/// Reusable host scratch for the voxel reductions: the sorted words and the
/// run boundaries. Reusing one across calls avoids reallocating them.
#[derive(Debug, Default, Clone)]
pub struct VoxelDownsampler {
    words: Vec<u128>,
    /// Run starts into `words`, plus a final `words.len()`.
    starts: Vec<u32>,
    unique: Vec<u128>,
    voxel_counts: Vec<u8>,
}

impl VoxelDownsampler {
    /// Creates an empty scratch.
    ///
    /// # Arguments
    ///
    /// This function takes no arguments.
    ///
    /// # Returns
    ///
    /// A downsampler with no buffers allocated yet.
    ///
    /// # Errors
    ///
    /// This function is infallible.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sorts the points' words and finds the voxel runs.
    fn sort_and_segment(
        &mut self,
        points: &[[f64; 3]],
        layout: &VoxelLayout,
    ) -> Result<(), VoxelError> {
        let n = points.len();
        if n > MAX_VOXEL_POINTS {
            return Err(VoxelError::TooManyPoints {
                len: n,
                max: MAX_VOXEL_POINTS,
            });
        }
        self.words.clear();
        self.words.resize(n, 0);
        let (s, o, r) = (layout.voxel_size(), layout.origin(), layout.inv_size());
        self.words
            .par_iter_mut()
            .with_min_len(MIN_ITEMS_PER_TASK)
            .zip(points.par_iter())
            .enumerate()
            .for_each(|(i, (w, p))| {
                *w = match (
                    axis_key(p[0], o[0], s, r),
                    axis_key(p[1], o[1], s, r),
                    axis_key(p[2], o[2], s, r),
                ) {
                    (Some(x), Some(y), Some(z)) => word_from_key(&[x, y, z], i as u32),
                    _ => INVALID_WORD,
                };
            });
        if let Some(index) = self
            .words
            .par_iter()
            .with_min_len(MIN_ITEMS_PER_TASK)
            .position_first(|&w| w == INVALID_WORD)
        {
            return Err(VoxelError::PointOutOfRange { index });
        }
        // Words are unique (distinct indices), so the unstable sort is
        // deterministic.
        self.words.par_sort_unstable();

        self.starts.clear();
        let mut prev = None;
        for (i, &w) in self.words.iter().enumerate() {
            let k = key_bits(w);
            if prev != Some(k) {
                self.starts.push(i as u32);
                prev = Some(k);
            }
        }
        self.starts.push(n as u32);
        Ok(())
    }

    /// Voxels found by the last `sort_and_segment`.
    #[inline]
    fn voxel_count(&self) -> usize {
        self.starts.len().saturating_sub(1)
    }

    /// Sort words of voxel `v`, in ascending point index.
    #[inline]
    fn run(&self, v: usize) -> &[u128] {
        voxel_run(&self.words, &self.starts, v)
    }

    /// Groups `cloud` by the voxels of `layout` into this scratch and
    /// returns the grouping as a [`VoxelView`] that borrows it: a reused
    /// scratch does not reallocate its buffers for clouds no larger than an
    /// earlier one. See [`VoxelView`] for the layout of the view.
    ///
    /// # Arguments
    ///
    /// * `cloud` - Input cloud; borrowed by the view, nothing is copied.
    /// * `layout` - Voxel size and origin.
    ///
    /// # Returns
    ///
    /// The cloud grouped by voxel, voxels in ascending key order.
    ///
    /// # Errors
    ///
    /// * [`VoxelError::PointOutOfRange`] for a non-finite point or a key
    ///   outside `i32` (the smallest such index).
    /// * [`VoxelError::TooManyPoints`] above [`MAX_VOXEL_POINTS`] points.
    /// * [`VoxelError::Filter`] if the colors or normals do not have one
    ///   entry per point.
    ///
    /// # Example
    ///
    /// ```
    /// use kornia_3d::pointcloud::PointCloud;
    /// use kornia_3d::voxel::{VoxelDownsampler, VoxelLayout};
    ///
    /// let layout = VoxelLayout::new(1.0, [0.0; 3])?;
    /// let mut ds = VoxelDownsampler::new();
    /// for (points, voxels) in [(vec![[0.5; 3], [3.5; 3], [0.2; 3]], 2), (vec![[0.5; 3]], 1)] {
    ///     let cloud = PointCloud::new(points, None, None);
    ///     let view = ds.view(&cloud, &layout)?;
    ///     assert_eq!(view.len(), voxels);
    /// }
    /// # Ok::<(), kornia_3d::voxel::VoxelError>(())
    /// ```
    pub fn view<'a>(
        &'a mut self,
        cloud: &'a PointCloud,
        layout: &VoxelLayout,
    ) -> Result<VoxelView<'a>, VoxelError> {
        check_attributes(cloud)?;
        self.sort_and_segment(cloud.points(), layout)?;
        Ok(VoxelView {
            cloud,
            layout: *layout,
            words: Cow::Borrowed(&self.words),
            starts: Cow::Borrowed(&self.starts),
        })
    }

    /// The representative point of voxel `v` and the index of the point it
    /// came from (for [`DownsampleMode::Centroid`], the lowest index).
    #[inline]
    fn reduce(
        &self,
        v: usize,
        points: &[[f64; 3]],
        layout: &VoxelLayout,
        mode: DownsampleMode,
    ) -> ([f64; 3], u32) {
        reduce_run(self.run(v), points, layout, mode)
    }

    /// Downsamples `points` to one point per occupied voxel, into
    /// caller-owned buffers. See [`voxel_downsample`] for the rules.
    ///
    /// # Arguments
    ///
    /// * `points` - Input points.
    /// * `layout` - Voxel size and origin.
    /// * `mode` - Which point represents a voxel.
    /// * `dst_points` - Cleared, then one point per voxel in ascending key
    ///   order.
    /// * `dst_index` - Optional; cleared, then per voxel the input index of
    ///   the chosen point (the lowest index of the voxel for
    ///   [`DownsampleMode::Centroid`]).
    ///
    /// # Returns
    ///
    /// `Ok(())` once the outputs are written.
    ///
    /// # Errors
    ///
    /// * [`VoxelError::PointOutOfRange`] for a non-finite point or a key
    ///   outside `i32` (the smallest such index).
    /// * [`VoxelError::TooManyPoints`] above [`MAX_VOXEL_POINTS`] points.
    ///
    /// # Example
    ///
    /// ```
    /// use kornia_3d::voxel::{DownsampleMode, VoxelDownsampler, VoxelLayout};
    ///
    /// let layout = VoxelLayout::new(1.0, [0.0; 3])?;
    /// let points = [[1.5, 0.5, 0.5], [0.25, 0.5, 0.5], [0.75, 0.5, 0.5]];
    /// let mut ds = VoxelDownsampler::new();
    /// let (mut out, mut index) = (Vec::new(), Vec::new());
    /// ds.downsample_into(&points, &layout, DownsampleMode::Centroid, &mut out, Some(&mut index))?;
    /// assert_eq!(out, vec![[0.5, 0.5, 0.5], [1.5, 0.5, 0.5]]);
    /// assert_eq!(index, vec![1, 0]);
    /// # Ok::<(), kornia_3d::voxel::VoxelError>(())
    /// ```
    pub fn downsample_into(
        &mut self,
        points: &[[f64; 3]],
        layout: &VoxelLayout,
        mode: DownsampleMode,
        dst_points: &mut Vec<[f64; 3]>,
        dst_index: Option<&mut Vec<u32>>,
    ) -> Result<(), VoxelError> {
        self.sort_and_segment(points, layout)?;
        let m = self.voxel_count();
        dst_points.clear();
        dst_points.resize(m, [0.0; 3]);
        let this = &*self;
        match dst_index {
            Some(index) => {
                index.clear();
                index.resize(m, 0);
                dst_points
                    .par_iter_mut()
                    .zip(index.par_iter_mut())
                    .with_min_len(MIN_ITEMS_PER_TASK)
                    .enumerate()
                    .for_each(|(v, (p, i))| {
                        let (q, j) = this.reduce(v, points, layout, mode);
                        *p = q;
                        *i = j;
                    });
            }
            None => {
                dst_points
                    .par_iter_mut()
                    .with_min_len(MIN_ITEMS_PER_TASK)
                    .enumerate()
                    .for_each(|(v, p)| *p = this.reduce(v, points, layout, mode).0);
            }
        }
        Ok(())
    }

    /// Per point, the number of occupied voxels among its voxel's
    /// neighbours, into a caller-owned buffer. See
    /// [`voxel_neighbour_counts`].
    ///
    /// # Arguments
    ///
    /// * `points` - Input points.
    /// * `layout` - Voxel size and origin.
    /// * `connectivity` - Which neighbours count.
    /// * `dst` - Cleared, then one count per point, in input order.
    ///
    /// # Returns
    ///
    /// `Ok(())` once `dst` is written.
    ///
    /// # Errors
    ///
    /// As [`downsample_into`](Self::downsample_into).
    pub fn neighbour_counts_into(
        &mut self,
        points: &[[f64; 3]],
        layout: &VoxelLayout,
        connectivity: VoxelConnectivity,
        dst: &mut Vec<u8>,
    ) -> Result<(), VoxelError> {
        self.sort_and_segment(points, layout)?;
        let m = self.voxel_count();
        let Self {
            words,
            starts,
            unique,
            voxel_counts,
        } = self;
        unique.clear();
        unique.extend(starts[..m].iter().map(|&s| key_bits(words[s as usize])));
        voxel_counts.clear();
        voxel_counts.resize(m, 0);
        let max_l1 = connectivity.max_l1();
        let uniq = &*unique;
        voxel_counts
            .par_iter_mut()
            .with_min_len(MIN_ITEMS_PER_TASK)
            .enumerate()
            .for_each_init(
                || [0usize; 9],
                |hints, (v, c)| *c = adjacent_occupied(uniq, uniq[v], max_l1, hints),
            );

        dst.clear();
        dst.resize(points.len(), 0);
        for (v, &c) in voxel_counts.iter().enumerate() {
            for &w in &words[starts[v] as usize..starts[v + 1] as usize] {
                dst[index_from_word(w)] = c;
            }
        }
        Ok(())
    }
}

/// Sort words of voxel `v`: `words[starts[v]..starts[v + 1]]`.
#[inline]
fn voxel_run<'w>(words: &'w [u128], starts: &[u32], v: usize) -> &'w [u128] {
    &words[starts[v] as usize..starts[v + 1] as usize]
}

/// The representative point of the voxel whose sort words are `run`
/// (non-empty, ascending index) and the index of the point it came from
/// (for [`DownsampleMode::Centroid`], the lowest index).
#[inline]
fn reduce_run(
    run: &[u128],
    points: &[[f64; 3]],
    layout: &VoxelLayout,
    mode: DownsampleMode,
) -> ([f64; 3], u32) {
    let first = run[0];
    match mode {
        DownsampleMode::First => (points[index_from_word(first)], first as u32),
        DownsampleMode::Centroid => {
            let mut sum = [0.0f64; 3];
            for &w in run {
                let p = points[index_from_word(w)];
                sum[0] += p[0];
                sum[1] += p[1];
                sum[2] += p[2];
            }
            let c = run.len() as f64;
            ([sum[0] / c, sum[1] / c, sum[2] / c], first as u32)
        }
        DownsampleMode::NearestToCentre => {
            let key = key_from_word(first);
            let (s, o) = (layout.voxel_size(), layout.origin());
            let centre = [
                axis_centre(key[0], o[0], s),
                axis_centre(key[1], o[1], s),
                axis_centre(key[2], o[2], s),
            ];
            let dist = |p: &[f64; 3]| {
                let d = [p[0] - centre[0], p[1] - centre[1], p[2] - centre[2]];
                d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
            };
            let mut best = first;
            let mut best_d = dist(&points[index_from_word(first)]);
            for &w in &run[1..] {
                let d = dist(&points[index_from_word(w)]);
                if d < best_d {
                    best_d = d;
                    best = w;
                }
            }
            (points[index_from_word(best)], best as u32)
        }
    }
}

/// Occupied neighbours of the voxel with packed biased key `key` among the
/// sorted unique keys `unique`, counting offsets with L1 norm `1..=max_l1`.
///
/// For each of the 9 `(dx, dy)` columns, the voxels `z - 1 ..= z + 1` are
/// consecutive in the sorted keys, found by one search that gallops from
/// the column's `hints` entry (the previous voxel's result: voxels visited
/// in ascending order make it amortized O(1)). Columns or `z` values past
/// the `i32` range do not exist.
fn adjacent_occupied(unique: &[u128], key: u128, max_l1: i32, hints: &mut [usize; 9]) -> u8 {
    let x = i64::from((key >> 64) as u32);
    let y = i64::from((key >> 32) as u32);
    let z = i64::from(key as u32);
    let max = i64::from(u32::MAX);
    let mut count = 0u8;
    for dx in -1i64..=1 {
        for dy in -1i64..=1 {
            let lxy = (dx.abs() + dy.abs()) as i32;
            if lxy > max_l1 {
                continue;
            }
            let (tx, ty) = (x + dx, y + dy);
            if !(0..=max).contains(&tx) || !(0..=max).contains(&ty) {
                continue;
            }
            let (tx, ty) = (tx as u32, ty as u32);
            let lo = (z - 1).max(0) as u32;
            let hi = (z + 1).min(max) as u32;
            let first = pack_biased(tx, ty, lo);
            let last = pack_biased(tx, ty, hi);
            let hint = &mut hints[((dx + 1) * 3 + dy + 1) as usize];
            let start = gallop(unique, *hint, first);
            *hint = start;
            for &k in unique[start..].iter().take_while(|&&k| k <= last) {
                let dz = i64::from(k as u32) - z;
                let l1 = lxy + dz.abs() as i32;
                if l1 > 0 && l1 <= max_l1 {
                    count += 1;
                }
            }
        }
    }
    count
}

/// A point cloud grouped by voxel, borrowing the cloud: the voxels in
/// ascending key order (lexicographic `(x, y, z)`), and for each voxel the
/// indices of its points in ascending order. No point data is copied.
///
/// Internally this is the sort of [`VoxelDownsampler`]: one sort word per
/// point (voxel key, then point index), sorted, plus the start of each
/// voxel's run, a compressed list of point indices per voxel.
///
/// Downsampling is a reduction over this grouping:
/// [`first`](Self::first) and [`nearest`](Self::nearest) pick one input
/// point per voxel and return a [`PointCloudView`] of the cloud (zero copy);
/// [`centroids`](Self::centroids) averages each voxel and returns an owned
/// [`PointCloud`], since the averages are new points.
///
/// Built by [`VoxelView::new`] (owns its buffers) or
/// [`VoxelDownsampler::view`] (borrows the scratch, for reuse across
/// calls). [`voxel_downsample`] is the one-call form returning an owned
/// cloud.
///
/// # Example
///
/// ```
/// use kornia_3d::pointcloud::PointCloud;
/// use kornia_3d::voxel::{VoxelLayout, VoxelView};
///
/// let points = vec![[1.5, 0.5, 0.5], [0.25, 0.5, 0.5], [0.75, 0.5, 0.5]];
/// let cloud = PointCloud::new(points, Some(vec![[0; 3], [10; 3], [20; 3]]), None);
/// let view = VoxelView::new(&cloud, &VoxelLayout::new(1.0, [0.0; 3])?)?;
/// assert_eq!(view.keys().collect::<Vec<_>>(), vec![[0, 0, 0], [1, 0, 0]]);
/// assert_eq!(view.voxel(0).map(|v| v.collect::<Vec<_>>()), Some(vec![1, 2]));
///
/// let first = view.first();
/// assert_eq!(first.indices(), &[1, 0]);
/// assert_eq!(first.color(0), Some(&[10; 3]));
///
/// let centroids = view.centroids();
/// assert_eq!(centroids.points(), &vec![[0.5, 0.5, 0.5], [1.5, 0.5, 0.5]]);
/// assert_eq!(centroids.colors(), Some(&vec![[15; 3], [0; 3]]));
/// # Ok::<(), kornia_3d::voxel::VoxelError>(())
/// ```
#[derive(Debug, Clone)]
pub struct VoxelView<'a> {
    cloud: &'a PointCloud,
    layout: VoxelLayout,
    /// Sorted sort words, one per point.
    words: Cow<'a, [u128]>,
    /// Run starts into `words`, plus a final `words.len()`.
    starts: Cow<'a, [u32]>,
}

impl<'a> VoxelView<'a> {
    /// Groups `cloud` by the voxels of `layout`, in buffers owned by the
    /// view. To reuse the buffers across calls, use
    /// [`VoxelDownsampler::view`].
    ///
    /// # Arguments
    ///
    /// * `cloud` - Input cloud; borrowed by the view, nothing is copied.
    /// * `layout` - Voxel size and origin.
    ///
    /// # Returns
    ///
    /// The cloud grouped by voxel, voxels in ascending key order.
    ///
    /// # Errors
    ///
    /// As [`VoxelDownsampler::view`].
    ///
    /// # Example
    ///
    /// ```
    /// use kornia_3d::pointcloud::PointCloud;
    /// use kornia_3d::voxel::{VoxelLayout, VoxelView};
    ///
    /// let cloud = PointCloud::new(vec![[0.1; 3], [-0.1; 3], [0.2; 3]], None, None);
    /// let view = VoxelView::new(&cloud, &VoxelLayout::new(0.5, [0.0; 3])?)?;
    /// assert_eq!(view.len(), 2);
    /// assert_eq!(view.key(0), Some([-1, -1, -1]));
    /// assert_eq!(view.centre(1), Some([0.25; 3]));
    /// # Ok::<(), kornia_3d::voxel::VoxelError>(())
    /// ```
    pub fn new(cloud: &'a PointCloud, layout: &VoxelLayout) -> Result<Self, VoxelError> {
        check_attributes(cloud)?;
        let mut ds = VoxelDownsampler::new();
        ds.sort_and_segment(cloud.points(), layout)?;
        Ok(Self {
            cloud,
            layout: *layout,
            words: Cow::Owned(ds.words),
            starts: Cow::Owned(ds.starts),
        })
    }

    /// Number of occupied voxels.
    ///
    /// # Arguments
    ///
    /// This method takes no arguments beyond `self`.
    ///
    /// # Returns
    ///
    /// The number of voxels holding at least one point.
    ///
    /// # Errors
    ///
    /// This method is infallible.
    #[inline]
    pub fn len(&self) -> usize {
        self.starts.len().saturating_sub(1)
    }

    /// Whether no voxel is occupied (the cloud is empty).
    ///
    /// # Arguments
    ///
    /// This method takes no arguments beyond `self`.
    ///
    /// # Returns
    ///
    /// `true` when [`len`](Self::len) is `0`.
    ///
    /// # Errors
    ///
    /// This method is infallible.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The grouped cloud.
    ///
    /// # Arguments
    ///
    /// This method takes no arguments beyond `self`.
    ///
    /// # Returns
    ///
    /// The borrowed input cloud.
    ///
    /// # Errors
    ///
    /// This method is infallible.
    #[inline]
    pub fn cloud(&self) -> &'a PointCloud {
        self.cloud
    }

    /// The voxel size and origin the cloud was grouped with.
    ///
    /// # Arguments
    ///
    /// This method takes no arguments beyond `self`.
    ///
    /// # Returns
    ///
    /// The layout.
    ///
    /// # Errors
    ///
    /// This method is infallible.
    #[inline]
    pub fn layout(&self) -> &VoxelLayout {
        &self.layout
    }

    /// The key of voxel `v`.
    ///
    /// # Arguments
    ///
    /// * `v` - Voxel position, `0..len()`, in ascending key order.
    ///
    /// # Returns
    ///
    /// The key, or `None` when `v >= len()`.
    ///
    /// # Errors
    ///
    /// This method is infallible.
    #[inline]
    pub fn key(&self, v: usize) -> Option<VoxelKey> {
        if v >= self.len() {
            return None;
        }
        Some(key_from_word(self.words[self.starts[v] as usize]))
    }

    /// The keys of the occupied voxels, ascending (lexicographic `(x, y, z)`).
    ///
    /// # Arguments
    ///
    /// This method takes no arguments beyond `self`.
    ///
    /// # Returns
    ///
    /// An iterator over [`len`](Self::len) keys; it does not allocate.
    ///
    /// # Errors
    ///
    /// This method is infallible.
    pub fn keys(&self) -> impl ExactSizeIterator<Item = VoxelKey> + '_ {
        let words = &*self.words;
        self.starts[..self.len()]
            .iter()
            .map(move |&s| key_from_word(words[s as usize]))
    }

    /// The centre of voxel `v`.
    ///
    /// # Arguments
    ///
    /// * `v` - Voxel position, `0..len()`.
    ///
    /// # Returns
    ///
    /// [`VoxelLayout::centre_from_key`] of the voxel's key, or `None` when
    /// `v >= len()`.
    ///
    /// # Errors
    ///
    /// This method is infallible.
    #[inline]
    pub fn centre(&self, v: usize) -> Option<[f64; 3]> {
        self.key(v).map(|k| self.layout.centre_from_key(&k))
    }

    /// The indices of the points in voxel `v`, ascending.
    ///
    /// # Arguments
    ///
    /// * `v` - Voxel position, `0..len()`.
    ///
    /// # Returns
    ///
    /// An iterator over the voxel's point indices into [`cloud`](Self::cloud)
    /// (it does not allocate), or `None` when `v >= len()`.
    ///
    /// # Errors
    ///
    /// This method is infallible.
    pub fn voxel(
        &self,
        v: usize,
    ) -> Option<impl ExactSizeIterator<Item = u32> + DoubleEndedIterator + '_> {
        if v >= self.len() {
            return None;
        }
        Some(self.run(v).iter().map(|&w| w as u32))
    }

    /// Sort words of voxel `v`, in ascending point index.
    #[inline]
    fn run(&self, v: usize) -> &[u128] {
        voxel_run(&self.words, &self.starts, v)
    }

    /// Per voxel, the input index of the point `mode` picks.
    fn representatives(&self, mode: DownsampleMode) -> PointCloudView<'a> {
        let points = self.cloud.points();
        let indices = (0..self.len())
            .into_par_iter()
            .with_min_len(MIN_ITEMS_PER_TASK)
            .map(|v| reduce_run(self.run(v), points, &self.layout, mode).1)
            .collect();
        // Every index comes from a sort word of this cloud, and the
        // attribute lengths were checked when the view was built.
        PointCloudView::from_checked(self.cloud, Cow::Owned(indices))
    }

    /// Per voxel, the point with the lowest input index
    /// ([`DownsampleMode::First`]), as a view of the cloud: no point is
    /// copied, only one index per voxel is allocated.
    ///
    /// # Arguments
    ///
    /// This method takes no arguments beyond `self`.
    ///
    /// # Returns
    ///
    /// A [`PointCloudView`] with one point per voxel, in ascending key
    /// order; [`PointCloudView::to_point_cloud`] gives the owned cloud of
    /// [`voxel_downsample`] with [`DownsampleMode::First`].
    ///
    /// # Errors
    ///
    /// This method is infallible.
    ///
    /// # Example
    ///
    /// ```
    /// use kornia_3d::pointcloud::PointCloud;
    /// use kornia_3d::voxel::{VoxelLayout, VoxelView};
    ///
    /// let cloud = PointCloud::new(vec![[0.7; 3], [0.2; 3], [1.5; 3]], None, None);
    /// let view = VoxelView::new(&cloud, &VoxelLayout::new(1.0, [0.0; 3])?)?;
    /// assert_eq!(view.first().indices(), &[0, 2]);
    /// # Ok::<(), kornia_3d::voxel::VoxelError>(())
    /// ```
    pub fn first(&self) -> PointCloudView<'a> {
        self.representatives(DownsampleMode::First)
    }

    /// Per voxel, the point nearest to the voxel centre, ties to the lowest
    /// input index ([`DownsampleMode::NearestToCentre`]), as a view of the
    /// cloud: no point is copied, only one index per voxel is allocated.
    ///
    /// # Arguments
    ///
    /// This method takes no arguments beyond `self`.
    ///
    /// # Returns
    ///
    /// A [`PointCloudView`] with one point per voxel, in ascending key
    /// order.
    ///
    /// # Errors
    ///
    /// This method is infallible.
    ///
    /// # Example
    ///
    /// ```
    /// use kornia_3d::pointcloud::PointCloud;
    /// use kornia_3d::voxel::{VoxelLayout, VoxelView};
    ///
    /// let cloud = PointCloud::new(vec![[0.7; 3], [0.2; 3], [1.5; 3]], None, None);
    /// let view = VoxelView::new(&cloud, &VoxelLayout::new(1.0, [0.0; 3])?)?;
    /// assert_eq!(view.nearest().indices(), &[0, 2]);
    /// # Ok::<(), kornia_3d::voxel::VoxelError>(())
    /// ```
    pub fn nearest(&self) -> PointCloudView<'a> {
        self.representatives(DownsampleMode::NearestToCentre)
    }

    /// Per voxel, the mean of its points ([`DownsampleMode::Centroid`]):
    /// colors averaged and rounded half up, normals averaged and
    /// renormalized (a zero mean stays zero). Each voxel is summed in
    /// ascending point index, so the result does not depend on the thread
    /// count.
    ///
    /// # Arguments
    ///
    /// This method takes no arguments beyond `self`.
    ///
    /// # Returns
    ///
    /// An owned cloud with one point per voxel, in ascending key order,
    /// with colors and normals when the input has them.
    ///
    /// # Errors
    ///
    /// This method is infallible.
    ///
    /// # Example
    ///
    /// ```
    /// use kornia_3d::pointcloud::PointCloud;
    /// use kornia_3d::voxel::{VoxelLayout, VoxelView};
    ///
    /// let cloud = PointCloud::new(vec![[0.2; 3], [0.6; 3], [1.5; 3]], None, None);
    /// let view = VoxelView::new(&cloud, &VoxelLayout::new(1.0, [0.0; 3])?)?;
    /// assert_eq!(view.centroids().points(), &vec![[0.4; 3], [1.5; 3]]);
    /// # Ok::<(), kornia_3d::voxel::VoxelError>(())
    /// ```
    pub fn centroids(&self) -> PointCloud {
        let m = self.len();
        let points = self.cloud.points();
        let layout = &self.layout;
        let centroids = (0..m)
            .into_par_iter()
            .with_min_len(MIN_ITEMS_PER_TASK)
            .map(|v| reduce_run(self.run(v), points, layout, DownsampleMode::Centroid).0)
            .collect();
        // Parallel over the voxels, each one reduced sequentially in
        // ascending point index (as `reduce_run`), so the result is the same
        // as a serial pass.
        let colors = self.cloud.colors().map(|c| {
            (0..m)
                .into_par_iter()
                .with_min_len(MIN_ITEMS_PER_TASK)
                .map(|v| mean_color(self.run(v), c))
                .collect()
        });
        let normals = self.cloud.normals().map(|n| {
            (0..m)
                .into_par_iter()
                .with_min_len(MIN_ITEMS_PER_TASK)
                .map(|v| mean_normal(self.run(v), n))
                .collect()
        });
        PointCloud::new(centroids, colors, normals)
    }
}

/// Downsamples a point cloud to one point per occupied voxel (voxel grid
/// filter): [`VoxelView::new`] followed by the reduction `mode` names,
/// copied into an owned cloud.
///
/// The output is in ascending voxel key order (lexicographic `(x, y, z)`),
/// the same for any permutation-free rerun. Open3D's `voxel_down_sample`
/// returns the same centroids in hash-map order; compare with it as sets.
/// For [`DownsampleMode::First`] and [`DownsampleMode::NearestToCentre`]
/// the chosen points are input points: [`VoxelView::first`] and
/// [`VoxelView::nearest`] return them as a view, without the copy.
///
/// # Arguments
///
/// * `cloud` - Input cloud; colors and normals follow `mode` (see
///   [`DownsampleMode`]).
/// * `layout` - Voxel size and origin.
/// * `mode` - Which point represents a voxel.
///
/// # Returns
///
/// One point per occupied voxel.
///
/// # Errors
///
/// As [`VoxelDownsampler::view`].
///
/// # Example
///
/// ```
/// use kornia_3d::pointcloud::PointCloud;
/// use kornia_3d::voxel::{voxel_downsample, DownsampleMode, VoxelLayout};
///
/// let points = vec![[0.1, 0.1, 0.1], [0.3, 0.1, 0.1], [-0.1, 0.1, 0.1]];
/// let cloud = PointCloud::new(points, None, None);
/// let layout = VoxelLayout::new(0.5, [0.0; 3])?;
/// let out = voxel_downsample(&cloud, &layout, DownsampleMode::First)?;
/// assert_eq!(out.points(), &vec![[-0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]);
/// # Ok::<(), kornia_3d::voxel::VoxelError>(())
/// ```
pub fn voxel_downsample(
    cloud: &PointCloud,
    layout: &VoxelLayout,
    mode: DownsampleMode,
) -> Result<PointCloud, VoxelError> {
    let view = VoxelView::new(cloud, layout)?;
    Ok(match mode {
        DownsampleMode::Centroid => view.centroids(),
        DownsampleMode::First => view.first().to_point_cloud(),
        DownsampleMode::NearestToCentre => view.nearest().to_point_cloud(),
    })
}

/// Per-channel mean of a voxel's colors, rounded half up.
fn mean_color(run: &[u128], colors: &[[u8; 3]]) -> [u8; 3] {
    let mut sum = [0u64; 3];
    for &w in run {
        let c = colors[index_from_word(w)];
        for (s, v) in sum.iter_mut().zip(c) {
            *s += u64::from(v);
        }
    }
    let n = run.len() as u64;
    // The mean of u8 values is at most 255.
    sum.map(|s| ((s + n / 2) / n) as u8)
}

/// Mean of a voxel's normals, renormalized; a zero mean stays zero.
fn mean_normal(run: &[u128], normals: &[[f64; 3]]) -> [f64; 3] {
    let mut sum = [0.0f64; 3];
    for &w in run {
        let n = normals[index_from_word(w)];
        sum[0] += n[0];
        sum[1] += n[1];
        sum[2] += n[2];
    }
    let norm = (sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]).sqrt();
    if norm > 0.0 {
        sum.map(|s| s / norm)
    } else {
        sum
    }
}

/// Per point, the number of occupied voxels among the neighbours of the
/// point's voxel (the voxel itself excluded).
///
/// # Arguments
///
/// * `points` - Input points.
/// * `layout` - Voxel size and origin.
/// * `connectivity` - Which neighbours count (at most 6, 18 or 26).
///
/// # Returns
///
/// One count per point, in input order.
///
/// # Errors
///
/// As [`VoxelDownsampler::downsample_into`].
///
/// # Example
///
/// ```
/// use kornia_3d::voxel::{voxel_neighbour_counts, VoxelConnectivity, VoxelLayout};
///
/// let layout = VoxelLayout::new(1.0, [0.0; 3])?;
/// let points = [[0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [1.5, 1.5, 1.5], [9.5, 0.5, 0.5]];
/// let counts = voxel_neighbour_counts(&points, &layout, VoxelConnectivity::TwentySix)?;
/// assert_eq!(counts, vec![2, 2, 2, 0]);
/// let counts = voxel_neighbour_counts(&points, &layout, VoxelConnectivity::Six)?;
/// assert_eq!(counts, vec![1, 1, 0, 0]);
/// # Ok::<(), kornia_3d::voxel::VoxelError>(())
/// ```
pub fn voxel_neighbour_counts(
    points: &[[f64; 3]],
    layout: &VoxelLayout,
    connectivity: VoxelConnectivity,
) -> Result<Vec<u8>, VoxelError> {
    let mut counts = Vec::new();
    VoxelDownsampler::new().neighbour_counts_into(points, layout, connectivity, &mut counts)?;
    Ok(counts)
}

/// Removes the points whose voxel has fewer than `min_neighbours` occupied
/// neighbour voxels (a voxel-grid density filter). Points keep their input
/// order.
///
/// # Arguments
///
/// * `cloud` - Input cloud; colors and normals are carried along.
/// * `layout` - Voxel size and origin.
/// * `connectivity` - Which neighbours count.
/// * `min_neighbours` - Occupied neighbour voxels needed to keep a point;
///   `0` keeps every point.
///
/// # Returns
///
/// The kept points, in input order, as a view of `cloud` (nothing is
/// copied; [`PointCloudView::to_point_cloud`] gives an owned cloud).
///
/// # Errors
///
/// As [`voxel_downsample`].
///
/// # Example
///
/// ```
/// use kornia_3d::pointcloud::PointCloud;
/// use kornia_3d::voxel::{remove_sparse_voxels, VoxelConnectivity, VoxelLayout};
///
/// let cloud = PointCloud::new(vec![[0.5; 3], [1.5, 0.5, 0.5], [9.5, 0.5, 0.5]], None, None);
/// let layout = VoxelLayout::new(1.0, [0.0; 3])?;
/// let out = remove_sparse_voxels(&cloud, &layout, VoxelConnectivity::TwentySix, 1)?;
/// assert_eq!(out.indices(), &[0, 1]);
/// assert_eq!(out.to_point_cloud().points(), &vec![[0.5; 3], [1.5, 0.5, 0.5]]);
/// # Ok::<(), kornia_3d::voxel::VoxelError>(())
/// ```
pub fn remove_sparse_voxels<'a>(
    cloud: &'a PointCloud,
    layout: &VoxelLayout,
    connectivity: VoxelConnectivity,
    min_neighbours: usize,
) -> Result<PointCloudView<'a>, VoxelError> {
    check_attributes(cloud)?;
    let counts = voxel_neighbour_counts(cloud.points(), layout, connectivity)?;
    let kept: Vec<u32> = counts
        .iter()
        .enumerate()
        .filter(|&(_, &c)| usize::from(c) >= min_neighbours)
        .map(|(i, _)| i as u32)
        .collect();
    // Every kept index is a point of `cloud`, whose attributes were checked.
    Ok(PointCloudView::from_checked(cloud, Cow::Owned(kept)))
}

/// The first index of the ascending `keys` whose key is `>= first`,
/// searched forward from `hint` with doubling steps (or by bisection below
/// `hint` when `hint` is past it).
fn gallop(keys: &[u128], hint: usize, first: u128) -> usize {
    let n = keys.len();
    let mut lo = hint.min(n);
    if lo > 0 && keys[lo - 1] >= first {
        return keys[..lo].partition_point(|&k| k < first);
    }
    // Every key below `lo` is `< first`.
    let mut step = 1;
    loop {
        let probe = lo + step - 1;
        if probe >= n {
            return lo + keys[lo..].partition_point(|&k| k < first);
        }
        if keys[probe] >= first {
            return lo + keys[lo..=probe].partition_point(|&k| k < first);
        }
        lo = probe + 1;
        step *= 2;
    }
}

#[cfg(test)]
pub(super) mod test_access {
    //! Exposes the sort words to the sibling tests.
    use super::*;

    /// Sorted words of `points`, for the sort-word ordering test.
    pub(in crate::voxel) fn sorted_words(
        points: &[[f64; 3]],
        layout: &VoxelLayout,
    ) -> Result<Vec<u128>, VoxelError> {
        let mut ds = VoxelDownsampler::new();
        ds.sort_and_segment(points, layout)?;
        Ok(ds.words)
    }
}
