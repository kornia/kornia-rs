//! Voxel grids over point clouds: a sparse hash grid ([`VoxelGrid`]), a
//! cloud grouped by voxel without copying it ([`VoxelView`]), voxel
//! downsampling as a reduction over that grouping ([`VoxelView::first`],
//! [`VoxelView::nearest`], [`VoxelView::centroids`], or the one-call
//! [`voxel_downsample`]) and a voxel neighbour-count filter
//! ([`remove_sparse_voxels`]), designed after the public APIs of
//! Open3D (`geometry.VoxelGrid`, `PointCloud.voxel_down_sample`) and PCL
//! (`pcl::VoxelGrid`).
//!
//! # Voxel indexing
//!
//! A [`VoxelLayout`] is a voxel edge length `s` and an origin `o`. A point
//! `p` lies in the voxel whose integer key is, per axis,
//! `floor((p - o) / s)` evaluated in `f64` (a division, not a multiply by
//! `1 / s`, so the boundaries are exact multiples of `s`). Keys are `i32`
//! per axis, negative keys included: `-0.5 · s` falls in voxel `-1`. A
//! point whose key is not representable (a non-finite coordinate, or
//! `|p - o| / s` beyond `i32`) is rejected with
//! [`VoxelError::PointOutOfRange`] by the ops that index a whole cloud.
//!
//! # Determinism
//!
//! Every ordered output is sorted, never in hash order: the downsampled
//! cloud is in ascending key order (lexicographic on `(x, y, z)`), and the
//! points of a voxel are reduced in ascending input index. The same input
//! gives the same output bit for bit, on the host and on the device
//! (feature `cuda`, `CudaVoxelFilter`), unlike a hash-map reduction whose
//! output order depends on the hash table.

mod downsample;
mod grid;

#[cfg(feature = "cuda")]
mod cuda;

pub use downsample::{
    remove_sparse_voxels, voxel_downsample, voxel_neighbour_counts, DownsampleMode,
    VoxelDownsampler, VoxelView,
};
pub use grid::VoxelGrid;

#[cfg(feature = "cuda")]
pub use cuda::{CudaNeighbourCounts, CudaVoxelCloud, CudaVoxelFilter, MAX_CUDA_VOXEL_POINTS};

use crate::pointcloud::filter::FilterError;

/// Integer voxel coordinates `(x, y, z)`.
pub type VoxelKey = [i32; 3];

/// Largest cloud the voxel ops accept on the host: point indices are `u32`
/// and `u32::MAX` is reserved.
pub const MAX_VOXEL_POINTS: usize = u32::MAX as usize - 1;

/// Errors of the voxel grid ops.
#[derive(Debug, thiserror::Error)]
pub enum VoxelError {
    /// The voxel size is not finite and positive.
    #[error("voxel size must be finite and positive, got {0}")]
    InvalidVoxelSize(f64),

    /// The grid origin has a non-finite coordinate.
    #[error("voxel grid origin must be finite, got {0:?}")]
    InvalidOrigin([f64; 3]),

    /// A point has a non-finite coordinate or a voxel key outside `i32`.
    /// `index` is the smallest such point index.
    #[error("point {index} is not finite or its voxel key overflows i32")]
    PointOutOfRange {
        /// Index of the first offending point.
        index: usize,
    },

    /// The cloud has more points than the index types can address.
    #[error("{len} points exceed the maximum of {max}")]
    TooManyPoints {
        /// Points in the cloud.
        len: usize,
        /// Largest supported cloud.
        max: usize,
    },

    /// A point cloud operand is malformed (attribute lengths, indices).
    #[error(transparent)]
    Filter(#[from] FilterError),

    /// A device output buffer is smaller than the call needs (feature
    /// `cuda`).
    #[error("output capacity {capacity} < {needed}")]
    CapacityTooSmall {
        /// Capacity of the buffer.
        capacity: usize,
        /// Entries the call needs.
        needed: usize,
    },

    /// A device point buffer holds fewer than `3 · n` coordinates (feature
    /// `cuda`).
    #[error("device point buffer has {len} coordinates, {needed} needed")]
    PointBufferTooSmall {
        /// Coordinates in the buffer.
        len: usize,
        /// `3 · n`.
        needed: usize,
    },

    /// A device buffer belongs to a different stream than the filter using
    /// it (feature `cuda`).
    #[error("device buffer was allocated on a different stream than the filter")]
    StreamMismatch,

    /// A CUDA driver, compile or launch failure (feature `cuda`).
    #[error("CUDA error: {0}")]
    Cuda(String),
}

/// Voxel edge length and grid origin, shared by every voxel op.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VoxelLayout {
    voxel_size: f64,
    origin: [f64; 3],
    /// `1 / voxel_size` when it is a normal `f64`, else `0.0` (every key
    /// then takes the division path of [`axis_key`]).
    inv_size: f64,
}

impl VoxelLayout {
    /// Creates a layout.
    ///
    /// # Arguments
    ///
    /// * `voxel_size` - Voxel edge length, in the units of the points.
    /// * `origin` - Corner of voxel `(0, 0, 0)`. To reproduce Open3D's
    ///   `voxel_down_sample` voxel boundaries pass
    ///   `min_bound - voxel_size / 2`; PCL's `VoxelGrid` uses `[0.0; 3]`.
    ///
    /// # Returns
    ///
    /// The layout.
    ///
    /// # Errors
    ///
    /// [`VoxelError::InvalidVoxelSize`] if `voxel_size` is not finite and
    /// positive; [`VoxelError::InvalidOrigin`] if `origin` is not finite.
    ///
    /// # Example
    ///
    /// ```
    /// use kornia_3d::voxel::VoxelLayout;
    ///
    /// let layout = VoxelLayout::new(0.5, [0.0; 3])?;
    /// assert_eq!(layout.key_from_point(&[-0.25, 0.75, 1.0]), Some([-1, 1, 2]));
    /// assert_eq!(layout.centre_from_key(&[-1, 1, 2]), [-0.25, 0.75, 1.25]);
    /// # Ok::<(), kornia_3d::voxel::VoxelError>(())
    /// ```
    pub fn new(voxel_size: f64, origin: [f64; 3]) -> Result<Self, VoxelError> {
        if !(voxel_size.is_finite() && voxel_size > 0.0) {
            return Err(VoxelError::InvalidVoxelSize(voxel_size));
        }
        if origin.iter().any(|o| !o.is_finite()) {
            return Err(VoxelError::InvalidOrigin(origin));
        }
        let r = 1.0 / voxel_size;
        let inv_size = if r.is_normal() { r } else { 0.0 };
        Ok(Self {
            voxel_size,
            origin,
            inv_size,
        })
    }

    /// Voxel edge length.
    ///
    /// # Arguments
    ///
    /// This method takes no arguments beyond `self`.
    ///
    /// # Returns
    ///
    /// The `voxel_size` given to [`new`](Self::new).
    ///
    /// # Errors
    ///
    /// This method is infallible.
    pub fn voxel_size(&self) -> f64 {
        self.voxel_size
    }

    /// Corner of voxel `(0, 0, 0)`.
    ///
    /// # Arguments
    ///
    /// This method takes no arguments beyond `self`.
    ///
    /// # Returns
    ///
    /// The `origin` given to [`new`](Self::new).
    ///
    /// # Errors
    ///
    /// This method is infallible.
    pub fn origin(&self) -> [f64; 3] {
        self.origin
    }

    /// The key of the voxel containing `point`.
    ///
    /// # Arguments
    ///
    /// * `point` - Point in the units of the layout.
    ///
    /// # Returns
    ///
    /// `floor((point - origin) / voxel_size)` per axis, or `None` if a
    /// coordinate is not finite or the key does not fit `i32`.
    ///
    /// # Errors
    ///
    /// This method is infallible.
    #[inline]
    pub fn key_from_point(&self, point: &[f64; 3]) -> Option<VoxelKey> {
        Some([
            axis_key(point[0], self.origin[0], self.voxel_size, self.inv_size)?,
            axis_key(point[1], self.origin[1], self.voxel_size, self.inv_size)?,
            axis_key(point[2], self.origin[2], self.voxel_size, self.inv_size)?,
        ])
    }

    /// The centre of voxel `key`.
    ///
    /// # Arguments
    ///
    /// * `key` - Voxel key.
    ///
    /// # Returns
    ///
    /// `origin + (key + 0.5) · voxel_size` per axis.
    ///
    /// # Errors
    ///
    /// This method is infallible.
    #[inline]
    pub fn centre_from_key(&self, key: &VoxelKey) -> [f64; 3] {
        [
            axis_centre(key[0], self.origin[0], self.voxel_size),
            axis_centre(key[1], self.origin[1], self.voxel_size),
            axis_centre(key[2], self.origin[2], self.voxel_size),
        ]
    }

    /// `1 / voxel_size`, or `0.0` when that is not a normal `f64`.
    #[inline]
    pub(crate) fn inv_size(&self) -> f64 {
        self.inv_size
    }
}

/// Distance to the nearest integer, in units of `f64::EPSILON ·
/// max(|q|, 1)`, under which [`axis_key`] falls back to the division.
const KEY_SCREEN_EPS: f64 = 4.0;

/// `floor((v - o) / s)` as `i32`, range-checked in `f64` before the cast.
///
/// The quotient is first estimated as `qa = (v - o) · r` with `r = fl(1 / s)`
/// (a multiply, not a division). With `u = 2⁻⁵³`, `qa` is within `(2u + u²)
/// |x|` of `x = (v - o) / s` and the correctly rounded `fl(x)` within `u |x|`,
/// so when `qa` is more than `KEY_SCREEN_EPS · ε · max(|qa|, 1)` (`8u ·
/// max(|qa|, 1)`) away from every integer, `floor(qa) == floor(fl(x))`.
/// Otherwise (near a voxel boundary, `qa` subnormal or zero, `r == 0`, NaN
/// or infinity) the division is evaluated. The key is therefore always
/// exactly `floor(fl((v - o) / s))`. Mirrored by `axis_key` in the CUDA
/// source (which screens with `f32` first); keep the expressions identical.
#[inline]
pub(crate) fn axis_key(v: f64, o: f64, s: f64, r: f64) -> Option<i32> {
    let a = v - o;
    let qa = a * r;
    let fa = qa.floor();
    let tol = KEY_SCREEN_EPS * f64::EPSILON * qa.abs().max(1.0);
    let f = if qa - fa > tol && fa + 1.0 - qa > tol {
        fa
    } else {
        (a / s).floor()
    };
    // NaN is in no range.
    if (i32::MIN as f64..=i32::MAX as f64).contains(&f) {
        Some(f as i32)
    } else {
        None
    }
}

/// `o + (k + 0.5) · s`. Mirrored by `axis_centre` in the CUDA source.
#[inline]
pub(crate) fn axis_centre(k: i32, o: f64, s: f64) -> f64 {
    o + (k as f64 + 0.5) * s
}

/// Neighbourhood of a voxel among the 26 around it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VoxelConnectivity {
    /// The 6 voxels sharing a face.
    Six,
    /// The 18 voxels sharing a face or an edge.
    Eighteen,
    /// The 26 voxels sharing a face, an edge or a corner.
    TwentySix,
}

/// Offsets `(dx, dy, dz)` in `{-1, 0, 1}³ \ {0}` with `|dx| + |dy| + |dz| <=
/// max_l1`, in lexicographic order.
const fn neighbour_offsets<const N: usize>(max_l1: i32) -> [[i32; 3]; N] {
    let mut out = [[0i32; 3]; N];
    let mut n = 0;
    let mut dx: i32 = -1;
    while dx <= 1 {
        let mut dy: i32 = -1;
        while dy <= 1 {
            let mut dz: i32 = -1;
            while dz <= 1 {
                let l1 = dx.abs() + dy.abs() + dz.abs();
                if l1 > 0 && l1 <= max_l1 {
                    out[n] = [dx, dy, dz];
                    n += 1;
                }
                dz += 1;
            }
            dy += 1;
        }
        dx += 1;
    }
    out
}

const OFFSETS_6: [[i32; 3]; 6] = neighbour_offsets::<6>(1);
const OFFSETS_18: [[i32; 3]; 18] = neighbour_offsets::<18>(2);
const OFFSETS_26: [[i32; 3]; 26] = neighbour_offsets::<26>(3);

impl VoxelConnectivity {
    /// The neighbour offsets of this connectivity.
    ///
    /// # Arguments
    ///
    /// This method takes no arguments beyond `self`.
    ///
    /// # Returns
    ///
    /// 6, 18 or 26 offsets `(dx, dy, dz)`, in lexicographic order.
    ///
    /// # Errors
    ///
    /// This method is infallible.
    pub fn offsets(self) -> &'static [[i32; 3]] {
        match self {
            Self::Six => &OFFSETS_6,
            Self::Eighteen => &OFFSETS_18,
            Self::TwentySix => &OFFSETS_26,
        }
    }

    /// Largest `|dx| + |dy| + |dz|` of a neighbour offset.
    pub(crate) fn max_l1(self) -> i32 {
        match self {
            Self::Six => 1,
            Self::Eighteen => 2,
            Self::TwentySix => 3,
        }
    }
}

#[cfg(test)]
mod tests;

#[cfg(all(test, feature = "cuda"))]
mod cuda_tests;
