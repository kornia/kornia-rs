//! Sort-by-key voxel reductions on the host: every point gets a packed
//! 128-bit sort word (biased voxel key `x, y, z` in the top 96 bits, point
//! index in the low 32), the words are sorted, and each run of equal keys is
//! one voxel, its points in ascending index. The CUDA path
//! (`super::cuda`) sorts the same words and reduces the runs with the same
//! expressions.

use rayon::prelude::*;

use super::{
    axis_centre, axis_key, VoxelConnectivity, VoxelError, VoxelKey, VoxelLayout, MAX_VOXEL_POINTS,
};
use crate::pointcloud::filter::{check_attributes, select_points};
use crate::pointcloud::PointCloud;

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
    /// # Returns
    ///
    /// A downsampler with no buffers allocated yet.
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
        let (s, o) = (layout.voxel_size(), layout.origin());
        self.words
            .par_iter_mut()
            .with_min_len(MIN_ITEMS_PER_TASK)
            .zip(points.par_iter())
            .enumerate()
            .for_each(|(i, (w, p))| {
                *w = match (
                    axis_key(p[0], o[0], s),
                    axis_key(p[1], o[1], s),
                    axis_key(p[2], o[2], s),
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
        &self.words[self.starts[v] as usize..self.starts[v + 1] as usize]
    }

    /// The representative point of voxel `v` and the index of the point it
    /// came from (for [`DownsampleMode::Centroid`], the lowest index).
    /// Mirrored by `voxel_reduce` in the CUDA source.
    fn reduce(
        &self,
        v: usize,
        points: &[[f64; 3]],
        layout: &VoxelLayout,
        mode: DownsampleMode,
    ) -> ([f64; 3], u32) {
        let run = self.run(v);
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
            .for_each(|(v, c)| *c = adjacent_occupied(uniq, uniq[v], max_l1));

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

/// Occupied neighbours of the voxel with packed biased key `key` among the
/// sorted unique keys `unique`, counting offsets with L1 norm `1..=max_l1`.
///
/// For each of the 9 `(dx, dy)` columns, the voxels `z - 1 ..= z + 1` are
/// consecutive in the sorted keys, found by one binary search. Columns or
/// `z` values past the `i32` range do not exist. Mirrored by
/// `adjacent_occupied` in the CUDA source.
fn adjacent_occupied(unique: &[u128], key: u128, max_l1: i32) -> u8 {
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
            let start = unique.partition_point(|&k| k < first);
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

/// Downsamples a point cloud to one point per occupied voxel (voxel grid
/// filter).
///
/// The output is in ascending voxel key order (lexicographic `(x, y, z)`),
/// the same for any permutation-free rerun and on the device
/// (`CudaVoxelFilter::downsample`). Open3D's `voxel_down_sample` returns
/// the same centroids in hash-map order; compare with it as sets.
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
/// * [`VoxelError::PointOutOfRange`] for a non-finite point or a key
///   outside `i32` (the smallest such index).
/// * [`VoxelError::TooManyPoints`] above [`MAX_VOXEL_POINTS`] points.
/// * [`VoxelError::Filter`] if the colors or normals do not have one entry
///   per point.
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
    check_attributes(cloud)?;
    let mut ds = VoxelDownsampler::new();
    let mut points = Vec::new();
    let mut index = Vec::new();
    ds.downsample_into(cloud.points(), layout, mode, &mut points, Some(&mut index))?;
    let m = points.len();
    let pick = |v: usize| index[v] as usize;
    let (colors, normals) = match mode {
        DownsampleMode::First | DownsampleMode::NearestToCentre => (
            cloud.colors().map(|c| (0..m).map(|v| c[pick(v)]).collect()),
            cloud
                .normals()
                .map(|n| (0..m).map(|v| n[pick(v)]).collect()),
        ),
        DownsampleMode::Centroid => (
            cloud
                .colors()
                .map(|c| (0..m).map(|v| mean_color(ds.run(v), c)).collect()),
            cloud
                .normals()
                .map(|n| (0..m).map(|v| mean_normal(ds.run(v), n)).collect()),
        ),
    };
    Ok(PointCloud::new(points, colors, normals))
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
/// The kept points, in input order.
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
/// assert_eq!(out.points(), &vec![[0.5; 3], [1.5, 0.5, 0.5]]);
/// # Ok::<(), kornia_3d::voxel::VoxelError>(())
/// ```
pub fn remove_sparse_voxels(
    cloud: &PointCloud,
    layout: &VoxelLayout,
    connectivity: VoxelConnectivity,
    min_neighbours: usize,
) -> Result<PointCloud, VoxelError> {
    check_attributes(cloud)?;
    let counts = voxel_neighbour_counts(cloud.points(), layout, connectivity)?;
    let kept: Vec<u32> = counts
        .iter()
        .enumerate()
        .filter(|&(_, &c)| usize::from(c) >= min_neighbours)
        .map(|(i, _)| i as u32)
        .collect();
    select_points(cloud, &kept).map_err(VoxelError::from)
}

#[cfg(test)]
pub(super) mod test_access {
    //! Exposes the sort words to the sibling tests.
    use super::*;

    /// Sorted words of `points`, for the device-sort parity test.
    pub(in crate::voxel) fn sorted_words(
        points: &[[f64; 3]],
        layout: &VoxelLayout,
    ) -> Result<Vec<u128>, VoxelError> {
        let mut ds = VoxelDownsampler::new();
        ds.sort_and_segment(points, layout)?;
        Ok(ds.words)
    }
}
