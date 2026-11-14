//! Truncated signed distance fields on a sparse block grid, with submaps,
//! several sensors per step and several agents per map.
//!
//! [`TsdfVolume`] stores voxels in `B³` blocks (`B` = 8 or 16, a const
//! generic), allocated on demand from a block hash, and fuses pinhole depth
//! cameras ([`PinholeSensor`]) and spherical range sensors
//! ([`SphericalLidar`]) with the projective running mean of KinectFusion
//! (Newcombe et al., ISMAR 2011) and Voxel Hashing (Nießner et al., SIGGRAPH
//! Asia 2013), the same update as Open3D's `t.geometry.VoxelBlockGrid` and
//! nvblox. [`SubmapTsdf`] keeps one active submap per agent; a loop closure
//! moves submaps rigidly ([`SubmapTsdf::set_submap_poses`]) without
//! re-integrating.
//!
//! # Conventions
//!
//! * Poses are named `a_from_b` and map `p_a = R * p_b + t`
//!   ([`Pose3d::transform_point`]); `x.compose(&y)` applies `y` first.
//! * Pinhole frame: x right, y down, z forward; the centre of pixel `(x, y)`
//!   is at `(x, y)`. LiDAR frame: x forward, y left, z up, azimuth
//!   counter-clockwise from +x, elevation positive up.
//! * Distances are metres. The stored `tsdf` is the signed distance divided
//!   by the truncation, in `[-1, 1]`, positive in free space; `weight` is
//!   the observation count, capped at [`TsdfParams::max_weight`].
//!
//! # Determinism
//!
//! Every backend evaluates the same `f32` op sequence (shared with the CUDA
//! kernels through `kornia_oxide_kernels::tsdf_core`): each `+ - * /` is one
//! IEEE op, fused multiply-adds are explicit and correctly rounded, no
//! transcendental runs per voxel (the LiDAR azimuth is [`atan2_det`]). Block
//! pool indices are assigned in ascending key order, every ordered output
//! (blocks, meshes, submaps) is in ascending key order, and a multi-agent
//! batch is applied in `(stamp, agent, sensor)` order, so the stored bits
//! do not depend on thread count, arrival order or hash layout.
//!
//! Departures from VoxelBlockGrid: the weight is an integer count (`u16`)
//! and colour is stored as `u8` with its own capped weight; a voxel is
//! updated only by observations that touched its block in the same step;
//! the pixel rule is `floor(coord + 0.5)`.
//!
//! [`TsdfVolume`]: crate::tsdf::TsdfVolume
//! [`PinholeSensor`]: crate::tsdf::PinholeSensor
//! [`SphericalLidar`]: crate::tsdf::SphericalLidar
//! [`SubmapTsdf`]: crate::tsdf::SubmapTsdf
//! [`SubmapTsdf::set_submap_poses`]: crate::tsdf::SubmapTsdf::set_submap_poses
//! [`Pose3d::transform_point`]: crate::pose::Pose3d::transform_point
//! [`TsdfParams::max_weight`]: crate::tsdf::TsdfParams::max_weight
//! [`atan2_det`]: crate::tsdf::atan2_det

mod alloc;
mod integrate;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod integrate_neon;
mod mesh;
mod raycast;
mod sensor;
mod submap;
mod volume;
mod workspace;

#[cfg(test)]
mod tests;

pub use mesh::{BlockMeshCache, MeshParams, MeshUpdateStats};
pub use raycast::{RaycastMaps, RaycastParams};
pub use sensor::{
    atan2_det, AzimuthGrid, DepthRef, LidarTables, PinholeSensor, RangeSensor, SensorObservation,
    SphericalLidar,
};
pub use submap::{
    AgentStep, BatchReport, MemoryReport, RigSensor, SensorFrame, SensorRig, SpawnPolicy,
    StepReport, SubmapFilter, SubmapFrame, SubmapMeshCache, SubmapParams, SubmapPose, SubmapTsdf,
};
pub use volume::{BlockView, TsdfVolume, Voxel};
pub use workspace::{TsdfWorkspace, WorkspaceHint};

use kornia_image::{ImageError, ImageSize};
use kornia_oxide_kernels::tsdf_core as tc;

use crate::depth::DepthError;
use crate::pose::Pose3d;

#[cfg(all(target_arch = "x86", not(target_feature = "sse2")))]
compile_error!("the TSDF needs SSE2 (x87 extended precision breaks its f32 contract)");

/// Errors of the TSDF module.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum TsdfError {
    /// A depth image is invalid.
    #[error(transparent)]
    Depth(#[from] DepthError),

    /// An image is invalid (size, residency).
    #[error(transparent)]
    Image(#[from] ImageError),

    /// A parameter is out of its documented range.
    #[error("invalid {field}: {value}")]
    InvalidParameter {
        /// Parameter name.
        field: &'static str,
        /// Offending value.
        value: f64,
    },

    /// Pinhole focal lengths are not finite and positive, the principal
    /// point is not finite, or the `f32` constants would not be finite.
    #[error("invalid pinhole intrinsics fx={fx} fy={fy} cx={cx} cy={cy}")]
    InvalidIntrinsics {
        /// Focal length x.
        fx: f64,
        /// Focal length y.
        fy: f64,
        /// Principal point x.
        cx: f64,
        /// Principal point y.
        cy: f64,
    },

    /// The camera has non-zero distortion; undistort the depth first.
    #[error("camera has non-zero distortion; undistort first")]
    DistortedCamera,

    /// A pose is not finite, or its rotation is not orthonormal with
    /// determinant +1 (tolerance `1e-6`).
    #[error("transform is not finite or its rotation is not orthonormal with det +1")]
    InvalidTransform,

    /// The LiDAR beam table is invalid.
    #[error("invalid LiDAR beam table: {0}")]
    InvalidBeamTable(&'static str),

    /// The LiDAR azimuth grid is invalid.
    #[error("invalid azimuth grid start={start} step={step} columns={columns}")]
    InvalidAzimuthGrid {
        /// Azimuth of column 0 (radians).
        start: f64,
        /// Column spacing (radians).
        step: f64,
        /// Number of columns.
        columns: usize,
    },

    /// An image does not have the size of its sensor.
    #[error("image is {got:?}, expected {expected:?}")]
    SizeMismatch {
        /// Size required by the sensor.
        expected: ImageSize,
        /// Size of the given image.
        got: ImageSize,
    },

    /// A colour image was given for a non-pinhole sensor.
    #[error("colour given for a non-pinhole sensor")]
    ColorOnRangeSensor,

    /// Block coordinates outside `[-2^20, 2^20)`.
    #[error("block ({x}, {y}, {z}) is outside the key range")]
    KeyOutOfRange {
        /// Block x.
        x: i32,
        /// Block y.
        y: i32,
        /// Block z.
        z: i32,
    },

    /// A frame names a sensor that is not in its agent's rig.
    #[error("sensor {0:?} is not in the rig")]
    UnknownSensor(SensorId),

    /// A sensor appears twice (in a rig or in one step).
    #[error("sensor {0:?} given twice")]
    DuplicateSensor(SensorId),

    /// The agent was never added.
    #[error("agent {0:?} is not registered")]
    UnknownAgent(AgentId),

    /// The agent was added twice.
    #[error("agent {0:?} registered twice")]
    DuplicateAgent(AgentId),

    /// No submap has this id.
    #[error("submap {0:?} not found")]
    UnknownSubmap(SubmapId),

    /// Two steps of a batch share `(agent, stamp)`.
    #[error("step ({agent:?}, {stamp}) given twice")]
    DuplicateStep {
        /// Agent of the step.
        agent: AgentId,
        /// Stamp of the step.
        stamp: u64,
    },

    /// A step's stamp does not exceed the agent's last applied stamp.
    #[error("agent {agent:?}: stamp {got} <= last applied {last}")]
    StaleStamp {
        /// Agent of the step.
        agent: AgentId,
        /// Last applied stamp.
        last: u64,
        /// Stamp of the rejected step.
        got: u64,
    },

    /// The block pool is full under [`PoolGrowth::Fixed`] (or would exceed
    /// `u32` indices).
    #[error("block pool full: capacity {capacity}, needed {needed}")]
    PoolFull {
        /// Pool capacity in blocks.
        capacity: usize,
        /// Blocks the step needs.
        needed: usize,
    },

    /// Every submap slot is in use under [`PoolGrowth::Fixed`].
    #[error("all {max} submap slots are in use")]
    TooManySubmaps {
        /// Submap slots.
        max: usize,
    },

    /// The mesh has more vertices than `u32` indices address.
    #[error("mesh has {vertices} vertices, more than u32 indices address")]
    MeshTooLarge {
        /// Vertex count.
        vertices: usize,
    },

    /// The calling thread does not round to nearest, or flushes subnormals.
    #[error("floating-point environment is not round-to-nearest without flush-to-zero")]
    FpEnvironment,

    /// A step of [`SubmapTsdf::integrate_batch`] failed.
    #[error("batch step {index} failed: {source}")]
    Batch {
        /// Index of the failing step in the input slice.
        index: usize,
        /// The step's error.
        #[source]
        source: Box<TsdfError>,
    },
}

/// Opaque agent (robot, rig) id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AgentId(pub u32);

/// Opaque sensor id, unique within a rig.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SensorId(pub u32);

/// Submap id: its agent and its spawn sequence number; ordered by agent,
/// then sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SubmapId {
    /// Agent that owns (and alone writes) the submap.
    pub agent: AgentId,
    /// Spawn order within the agent, from 0.
    pub seq: u32,
}

/// Block edge marker; only `BlockDim<8>` and `BlockDim<16>` are supported.
pub struct BlockDim<const B: usize>;

mod sealed {
    pub trait Sealed {}
    impl Sealed for super::BlockDim<8> {}
    impl Sealed for super::BlockDim<16> {}
}

/// Supported block edges (sealed): 8 (nvblox) and 16 (Open3D).
pub trait SupportedBlockDim: sealed::Sealed {
    /// `log2(B)`.
    const LOG2: u32;
    /// `B³`.
    const VOXELS: usize;
}

impl SupportedBlockDim for BlockDim<8> {
    const LOG2: u32 = 3;
    const VOXELS: usize = 512;
}

impl SupportedBlockDim for BlockDim<16> {
    const LOG2: u32 = 4;
    const VOXELS: usize = 4096;
}

/// Packed block coordinates; ascending order is ascending `(x, y, z)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BlockKey(u64);

/// Block coordinate range of a key: `[-2^20, 2^20)`.
const KEY_RANGE: i32 = 1 << 20;

impl BlockKey {
    /// Packs block coordinates.
    ///
    /// # Arguments
    ///
    /// * `x`, `y`, `z` - Block coordinates (voxel coordinate `>> log2(B)`).
    ///
    /// # Returns
    ///
    /// The key.
    ///
    /// # Errors
    ///
    /// [`TsdfError::KeyOutOfRange`] outside `[-2^20, 2^20)`.
    pub fn new(x: i32, y: i32, z: i32) -> Result<Self, TsdfError> {
        if [x, y, z]
            .iter()
            .any(|v| !(-KEY_RANGE..KEY_RANGE).contains(v))
        {
            return Err(TsdfError::KeyOutOfRange { x, y, z });
        }
        Ok(Self(tc::pack_block_key([x, y, z])))
    }

    /// Block coordinates `[x, y, z]`.
    pub fn coords(self) -> [i32; 3] {
        tc::unpack_block_key(self.0)
    }

    /// The packed `u64`.
    pub fn bits(self) -> u64 {
        self.0
    }

    pub(crate) fn from_bits(bits: u64) -> Self {
        Self(bits)
    }
}

/// Grid geometry: voxel size and truncation, with the reciprocals the
/// kernels use (derived once, in `f32`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TsdfLayout {
    voxel_size: f32,
    half_voxel: f32,
    inv_voxel: f32,
    trunc_voxels: f32,
    trunc: f32,
    inv_trunc: f32,
}

const MIN_VOXEL: f32 = 1e-4;
const MAX_VOXEL: f32 = 10.0;
const MAX_TRUNC_VOXELS: f32 = 64.0;

impl TsdfLayout {
    /// Validates and derives the grid constants.
    ///
    /// # Arguments
    ///
    /// * `voxel_size` - Voxel edge in metres, finite, in `[1e-4, 10]`.
    /// * `trunc_voxels` - Truncation in voxels, finite, in `[1, 64]`
    ///   (VoxelBlockGrid's `trunc_voxel_multiplier`).
    ///
    /// # Returns
    ///
    /// The layout; `truncation() = trunc_voxels * voxel_size`.
    ///
    /// # Errors
    ///
    /// [`TsdfError::InvalidParameter`] for a value out of range.
    ///
    /// # Example
    ///
    /// ```
    /// use kornia_3d::tsdf::TsdfLayout;
    /// let layout = TsdfLayout::new(0.02, 4.0)?;
    /// assert_eq!(layout.truncation(), 0.08);
    /// # Ok::<(), kornia_3d::tsdf::TsdfError>(())
    /// ```
    pub fn new(voxel_size: f32, trunc_voxels: f32) -> Result<Self, TsdfError> {
        let inv_voxel = 1.0f32 / voxel_size;
        if !((MIN_VOXEL..=MAX_VOXEL).contains(&voxel_size) && inv_voxel.is_normal()) {
            return Err(TsdfError::InvalidParameter {
                field: "voxel_size",
                value: voxel_size as f64,
            });
        }
        if !(1.0..=MAX_TRUNC_VOXELS).contains(&trunc_voxels) {
            return Err(TsdfError::InvalidParameter {
                field: "trunc_voxels",
                value: trunc_voxels as f64,
            });
        }
        let trunc = trunc_voxels * voxel_size;
        Ok(Self {
            voxel_size,
            half_voxel: voxel_size * 0.5,
            inv_voxel,
            trunc_voxels,
            trunc,
            inv_trunc: 1.0f32 / trunc,
        })
    }

    /// Voxel edge in metres.
    pub fn voxel_size(&self) -> f32 {
        self.voxel_size
    }

    /// Truncation in voxels.
    pub fn trunc_voxels(&self) -> f32 {
        self.trunc_voxels
    }

    /// Truncation distance in metres.
    pub fn truncation(&self) -> f32 {
        self.trunc
    }

    /// Centre of a voxel, `(g + 0.5) * voxel_size` per axis.
    ///
    /// # Arguments
    ///
    /// * `voxel` - Voxel coordinates, `|g| < 2^22`.
    ///
    /// # Returns
    ///
    /// The centre in metres.
    pub fn centre_from_voxel(&self, voxel: [i32; 3]) -> [f32; 3] {
        tc::voxel_centre::<StdOps>(voxel, self.voxel_size, self.half_voxel)
    }

    /// Voxel containing a point (floor rule).
    ///
    /// # Arguments
    ///
    /// * `point` - Point in metres.
    ///
    /// # Returns
    ///
    /// The voxel, `None` outside `|p / voxel_size| < 2^22` or for NaN.
    pub fn voxel_from_point(&self, point: [f32; 3]) -> Option<[i32; 3]> {
        tc::voxel_from_point(point, self.inv_voxel)
    }

    pub(crate) fn grid(&self, log2: u32) -> tc::Grid {
        tc::Grid {
            voxel_size: self.voxel_size,
            half_voxel: self.half_voxel,
            inv_voxel: self.inv_voxel,
            trunc: self.trunc,
            inv_trunc: self.inv_trunc,
            log2,
        }
    }
}

/// Depth (or range) interval `[min, max)` in metres.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DepthRange {
    /// Smallest kept value, `> 0`.
    pub min: f32,
    /// Exclusive upper bound, `> min`, finite.
    pub max: f32,
}

impl DepthRange {
    /// Validates a range.
    ///
    /// # Arguments
    ///
    /// * `min` - Smallest kept value (metres), finite, `> 0`.
    /// * `max` - Exclusive upper bound (metres), finite, `> min`.
    ///
    /// # Returns
    ///
    /// The range.
    ///
    /// # Errors
    ///
    /// [`TsdfError::InvalidParameter`] otherwise.
    pub fn new(min: f32, max: f32) -> Result<Self, TsdfError> {
        let r = Self { min, max };
        r.validate()?;
        Ok(r)
    }

    pub(crate) fn validate(&self) -> Result<(), TsdfError> {
        if !(self.min > 0.0 && self.min < f32::INFINITY) {
            return Err(TsdfError::InvalidParameter {
                field: "range.min",
                value: self.min as f64,
            });
        }
        if !(self.max > self.min && self.max < f32::INFINITY) {
            return Err(TsdfError::InvalidParameter {
                field: "range.max",
                value: self.max as f64,
            });
        }
        Ok(())
    }
}

/// What happens when a step needs more blocks than the pool holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolGrowth {
    /// Fail with [`TsdfError::PoolFull`] (or [`TsdfError::TooManySubmaps`]).
    Fixed,
    /// Double the capacity (an allocating, cold frame).
    Geometric,
}

/// CPU kernel choice; every choice gives identical bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum TsdfKernel {
    /// Portable scalar reference, single-threaded.
    Scalar,
    /// The voxel update of each observation runs one block per rayon task
    /// (current pool), with NEON on aarch64 and the scalar code elsewhere;
    /// the same bits as `Scalar`. Each observation pass enters the pool
    /// once; from outside a pool, rayon's job injector allocates one small
    /// block every 63 entries. Call from inside a rayon pool
    /// (`ThreadPool::install`) for zero allocations.
    #[default]
    Simd,
}

/// Volume parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct TsdfParams {
    /// Weight cap, `>= 1`; `65535` is effectively uncapped.
    pub max_weight: u16,
    /// Colour weight cap, `>= 1` (`u8` colour freezes once a step moves a
    /// channel by less than half a unit, hence a low default).
    pub max_color_weight: u8,
    /// Store colour (`rgbw`, 4 B/voxel).
    pub color: bool,
    /// Block capacity reserved at construction (the hard cap under
    /// [`PoolGrowth::Fixed`]); `block_capacity * B³ <= u32::MAX`.
    pub block_capacity: usize,
    /// Growth policy.
    pub growth: PoolGrowth,
}

impl Default for TsdfParams {
    fn default() -> Self {
        Self {
            max_weight: u16::MAX,
            max_color_weight: 16,
            color: true,
            block_capacity: 1 << 16,
            growth: PoolGrowth::Geometric,
        }
    }
}

impl TsdfParams {
    pub(crate) fn validate(&self, voxels: usize) -> Result<(), TsdfError> {
        if self.max_weight == 0 {
            return Err(TsdfError::InvalidParameter {
                field: "max_weight",
                value: 0.0,
            });
        }
        if self.max_color_weight == 0 {
            return Err(TsdfError::InvalidParameter {
                field: "max_color_weight",
                value: 0.0,
            });
        }
        if self.block_capacity == 0 || self.block_capacity > max_blocks(voxels) {
            return Err(TsdfError::InvalidParameter {
                field: "block_capacity",
                value: self.block_capacity as f64,
            });
        }
        Ok(())
    }
}

/// Largest pool (in blocks) whose voxel indices fit `u32`.
pub(crate) fn max_blocks(voxels: usize) -> usize {
    u32::MAX as usize / voxels
}

/// Counters of one integration step (or the sum over a batch).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct IntegrateStats {
    /// Pixels with a kept depth sample, over all observations.
    pub valid_pixels: u64,
    /// Distinct blocks the step touched.
    pub touched_blocks: u32,
    /// Blocks the step allocated.
    pub new_blocks: u32,
    /// Voxel updates, over all observations.
    pub updated_voxels: u64,
    /// Valid pixels whose truncation band leaves the voxel domain.
    pub out_of_range: u64,
}

impl IntegrateStats {
    pub(crate) fn accumulate(&mut self, o: &IntegrateStats) {
        self.valid_pixels = self.valid_pixels.saturating_add(o.valid_pixels);
        self.touched_blocks = self.touched_blocks.saturating_add(o.touched_blocks);
        self.new_blocks = self.new_blocks.saturating_add(o.new_blocks);
        self.updated_voxels = self.updated_voxels.saturating_add(o.updated_voxels);
        self.out_of_range = self.out_of_range.saturating_add(o.out_of_range);
    }
}

/// Host impl of the shared core's ops: correctly rounded `fma` and `sqrt`
/// on every supported target (`fmadd`/`fsqrt` on aarch64, `vfmadd` or the C
/// library `fmaf` on x86_64).
pub(crate) struct StdOps;

impl tc::F32Ops for StdOps {
    #[inline(always)]
    fn fma(a: f32, b: f32, c: f32) -> f32 {
        a.mul_add(b, c)
    }

    #[inline(always)]
    fn sqrt(x: f32) -> f32 {
        x.sqrt()
    }
}

/// `Err(FpEnvironment)` unless the calling thread rounds to nearest-even
/// without flush-to-zero or denormals-are-zero (read-only check).
pub(crate) fn check_fp_env() -> Result<(), TsdfError> {
    if fp_env_is_default() {
        Ok(())
    } else {
        Err(TsdfError::FpEnvironment)
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
pub(crate) fn fp_env_is_default() -> bool {
    // FZ (24), RMode (22-23), AH (1), FIZ (0)
    const MASK: u64 = (1 << 24) | (0b11 << 22) | (1 << 1) | 1;
    let fpcr: u64;
    // SAFETY: `mrs` only reads the FP control register into a GPR.
    unsafe {
        core::arch::asm!("mrs {}, fpcr", out(reg) fpcr, options(nomem, nostack, preserves_flags));
    }
    fpcr & MASK == 0
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
pub(crate) fn fp_env_is_default() -> bool {
    // FTZ (15), RC (13-14), DAZ (6)
    const MASK: u32 = (1 << 15) | (0b11 << 13) | (1 << 6);
    let mut csr = 0u32;
    let ptr: *mut u32 = &mut csr;
    // SAFETY: `stmxcsr` stores MXCSR into the 4-byte local `ptr` points to.
    unsafe {
        core::arch::asm!("stmxcsr dword ptr [{}]", in(reg) ptr, options(nostack, preserves_flags));
    }
    csr & MASK == 0
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
pub(crate) fn fp_env_is_default() -> bool {
    true
}

/// Tolerance of the orthonormality check of a pose rotation.
const ROTATION_TOLERANCE: f64 = 1e-6;

/// `Ok` for a finite pose with an orthonormal, `det = +1` rotation.
pub(crate) fn check_pose(p: &Pose3d) -> Result<(), TsdfError> {
    let m = p.rotation.to_cols_array();
    let t = p.translation;
    if !(m.iter().all(|v| v.is_finite()) && t.x.is_finite() && t.y.is_finite() && t.z.is_finite()) {
        return Err(TsdfError::InvalidTransform);
    }
    let rtr = p.rotation.transpose() * p.rotation;
    let e = rtr.to_cols_array();
    let id = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let ortho = e
        .iter()
        .zip(id.iter())
        .all(|(a, b)| (a - b).abs() <= ROTATION_TOLERANCE);
    if !(ortho && p.rotation.determinant() > 0.0) {
        return Err(TsdfError::InvalidTransform);
    }
    Ok(())
}

/// The `f32` rows of a validated pose (each entry rounded once).
pub(crate) fn rigid_from_pose(p: &Pose3d) -> Result<tc::Rigid, TsdfError> {
    let m = p.rotation.to_cols_array();
    let mut r = [0.0f32; 9];
    for i in 0..3 {
        for j in 0..3 {
            r[3 * i + j] = m[3 * j + i] as f32;
        }
    }
    let t = [
        p.translation.x as f32,
        p.translation.y as f32,
        p.translation.z as f32,
    ];
    if !r.iter().chain(t.iter()).all(|v| v.is_finite()) {
        return Err(TsdfError::InvalidTransform);
    }
    Ok(tc::Rigid { r, t })
}
