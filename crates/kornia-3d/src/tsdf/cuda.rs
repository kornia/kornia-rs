//! CUDA backend: [`CudaTsdfVolume`] and [`CudaSubmapTsdf`] on the
//! cuda-oxide kernels of `kornia_oxide_kernels` (PTX built ahead of time,
//! `sm_75` and newer; no NVRTC twin).
//!
//! The device state equals the CPU's bit for bit (TSDF contract, section
//! 14): every voxel and pixel formula is the shared core with `PtxOps`, the
//! step's missing blocks are ranked on the device so a new block's pool
//! index is `len + rank` (ascending key, as on the CPU), and every ordered
//! output is in ascending key order. Only hash slots differ.
//!
//! Host side of a warmed step: no heap allocation, launches through
//! `cudarc::driver::result::launch_kernel` with stack parameter arrays,
//! counts kept on the device (grid-stride kernels), one stream sync (the
//! counter download). Device buffers grow geometrically on a cold step;
//! the logical capacities (and [`TsdfError::PoolFull`]) follow the CPU,
//! except that `needed` is a lower bound once a step's misses overflow a
//! miss set of `2^16` slots that the pool cannot hold.

use std::ffi::c_void;
use std::marker::PhantomData;
use std::mem::size_of;
use std::sync::Arc;

use cudarc::driver::{
    result, sys, CudaSlice, CudaStream, DevicePtr, DeviceRepr, DriverError, ValidAsZeroBits,
};
use kornia_oxide_kernels::{self as oxide, tsdf_core as tc, OxideError};

use super::alloc::{prepare, Obs, ObsKind};
use super::mesh::{BlockMeshCache, MeshParams, MeshUpdateStats};
use super::raycast::{check_raycast, core_params, level_out, maps_out, RayOut};
use super::sensor::{DepthRef, SensorObservation};
use super::submap::{
    observation, AgentStep, BatchReport, Book, MemoryReport, SensorRig, SlotIndex, SpawnPolicy,
    StepReport, SubmapFilter, SubmapMeshCache, SubmapParams, SubmapPose,
};
use super::volume::{content_hash, BlockIndex, Pools};
use super::workspace::TsdfWorkspace;
use super::{
    check_fp_env, max_blocks, rigid_from_pose, AgentId, BlockDim, BlockKey, IntegrateStats,
    PoolGrowth, RaycastMaps, RaycastParams, SubmapId, SupportedBlockDim, TsdfError, TsdfLayout,
    TsdfParams, TsdfVolume,
};
use crate::camera::PinholeCamera;
use crate::mesh::TriangleMesh;
use crate::pose::Pose3d;
use crate::registration::{RgbdLevel, RgbdPyramid};

/// Oldest compute capability the PTX runs on.
const MIN_SM: (i32, i32) = (7, 5);
/// Resident 256-thread blocks per SM of the grid-stride kernels.
const BLOCKS_PER_SM: u32 = 8;
/// Byte stride of the per-observation parameter blocks (and the size of
/// the step and mesh parameter buffers).
const PARAM_STRIDE: usize = 256;
/// Alignment of staged images inside their buffers.
const STAGE_ALIGN: usize = 256;
/// Smallest physical block capacity (grown on demand up to the logical one).
const MIN_BLOCKS: usize = 1024;
/// Smallest miss-set size.
const MIN_MISS_SLOTS: usize = 1 << 12;
/// Miss-set size from which a step that cannot fit the pool fails at once
/// instead of growing the set to count every miss (`PoolFull::needed` is
/// exact below it, a lower bound above).
pub(crate) const MAX_EXACT_MISS_SLOTS: usize = 1 << 16;
/// Test-only fault injection: fails one upcoming call of an op on this
/// thread, as a device error would.
#[cfg(test)]
pub(crate) mod fault {
    use std::cell::Cell;

    use super::TsdfError;

    #[derive(Clone, Copy, PartialEq)]
    pub(crate) enum Op {
        Alloc,
        Dtoh,
    }

    thread_local! {
        static ARMED: Cell<Option<(Op, usize)>> = const { Cell::new(None) };
    }

    /// Fails the `skip`-th next call of `op` (0 = the next).
    pub(crate) fn arm(op: Op, skip: usize) {
        ARMED.set(Some((op, skip)));
    }

    /// Whether the armed fault is still pending (it did not fire).
    pub(crate) fn disarm() -> bool {
        ARMED.take().is_some()
    }

    pub(crate) fn hit(op: Op) -> Result<(), TsdfError> {
        match ARMED.get() {
            Some((o, 0)) if o == op => {
                ARMED.set(None);
                Err(TsdfError::Cuda("injected fault".into()))
            }
            Some((o, n)) if o == op => {
                ARMED.set(Some((o, n - 1)));
                Ok(())
            }
            _ => Ok(()),
        }
    }
}

/// `depth_kind` of a `u16` / `f32` depth image (`tsdf.rs`).
const DEPTH_U16: u32 = 0;
const DEPTH_F32: u32 = 1;

const _: () = assert!(size_of::<tc::PinholeParams>() <= PARAM_STRIDE);
const _: () = assert!(size_of::<tc::SphericalParams>() <= PARAM_STRIDE);
const _: () = assert!(size_of::<tc::StepParams>() <= PARAM_STRIDE);
const _: () = assert!(size_of::<tc::MeshParamsDev>() <= PARAM_STRIDE);

fn cuda(e: DriverError) -> TsdfError {
    TsdfError::Cuda(e.to_string())
}

/// Kernel parameter array: the address of each argument (`cuLaunchKernel`
/// copies the values during the call).
macro_rules! args {
    ($($a:expr),* $(,)?) => {
        [$((&$a) as *const _ as *mut c_void),*]
    };
}

// --------------------------------------------------------------- device ----

/// A device buffer and its raw address (launches take addresses).
struct DevBuf<T> {
    _buf: CudaSlice<T>,
    ptr: u64,
    len: usize,
}

impl<T> DevBuf<T> {
    /// Address of element `i`.
    fn at(&self, i: usize) -> u64 {
        self.ptr + (i * size_of::<T>()) as u64
    }
}

/// A loaded kernel.
#[derive(Clone, Copy)]
struct Func(sys::CUfunction);

// SAFETY: a `CUfunction` is a driver handle, usable from any thread of its
// context; the module that owns it is loaded for the life of the process.
unsafe impl Send for Func {}
// SAFETY: as above; launching does not mutate the handle.
unsafe impl Sync for Func {}

struct Kernels {
    touch_pinhole: Func,
    touch_spherical: Func,
    rank: Func,
    assign: Func,
    merge: Func,
    rehash: Func,
    integrate_pinhole: Func,
    integrate_spherical: Func,
    raycast: Func,
    mc_count: Func,
    mc_emit: Func,
    mc_dirty: Func,
}

/// Stream, kernels and the persistent grid size.
struct Dev {
    stream: Arc<CudaStream>,
    k: Kernels,
    grid: u32,
}

impl Dev {
    fn new(stream: Arc<CudaStream>) -> Result<Self, TsdfError> {
        let ctx = stream.context();
        let (major, minor) = ctx.compute_capability().map_err(cuda)?;
        if (major, minor) < MIN_SM {
            return Err(TsdfError::UnsupportedComputeCapability { major, minor });
        }
        if oxide::IS_STUB {
            return Err(TsdfError::KernelsUnavailable);
        }
        let f = |name: &str| -> Result<Func, TsdfError> {
            match oxide::__private::kernel(&stream, name) {
                Ok(k) => Ok(Func(k.cu_function())),
                Err(OxideError::Stub) => Err(TsdfError::KernelsUnavailable),
                Err(e) => Err(TsdfError::Cuda(e.to_string())),
            }
        };
        use oxide::entry as e;
        let k = Kernels {
            touch_pinhole: f(e::TSDF_TOUCH_PINHOLE)?,
            touch_spherical: f(e::TSDF_TOUCH_SPHERICAL)?,
            rank: f(e::TSDF_RANK_MISSES)?,
            assign: f(e::TSDF_ASSIGN_MISSES)?,
            merge: f(e::TSDF_MERGE_KEYS)?,
            rehash: f(e::TSDF_REHASH)?,
            integrate_pinhole: f(e::TSDF_INTEGRATE_PINHOLE)?,
            integrate_spherical: f(e::TSDF_INTEGRATE_SPHERICAL)?,
            raycast: f(e::TSDF_RAYCAST)?,
            mc_count: f(e::TSDF_MC_COUNT)?,
            mc_emit: f(e::TSDF_MC_EMIT)?,
            mc_dirty: f(e::TSDF_MC_DIRTY)?,
        };
        let sms = ctx
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .map_err(cuda)?;
        Ok(Self {
            k,
            grid: (sms.max(1) as u32) * BLOCKS_PER_SM,
            stream,
        })
    }

    fn bind(&self) -> Result<(), TsdfError> {
        self.stream.context().bind_to_thread().map_err(cuda)
    }

    /// Grid of a grid-stride kernel over `n` items, one per thread.
    fn grid_for(&self, n: usize) -> u32 {
        let blocks = n.div_ceil(tc::THREADS as usize).max(1);
        (blocks.min(self.grid as usize)) as u32
    }

    /// # Safety
    ///
    /// `args` match `f`'s parameters in count, order and type, and every
    /// address in them is valid for the kernel's accesses.
    unsafe fn launch(&self, f: Func, grid: u32, args: &mut [*mut c_void]) -> Result<(), TsdfError> {
        // SAFETY: the caller upholds the parameter contract; the stream is
        // live.
        unsafe {
            result::launch_kernel(
                f.0,
                (grid.max(1), 1, 1),
                (tc::THREADS, 1, 1),
                0,
                self.stream.cu_stream(),
                args,
            )
        }
        .map_err(cuda)
    }

    fn alloc<T: DeviceRepr + ValidAsZeroBits>(&self, len: usize) -> Result<DevBuf<T>, TsdfError> {
        #[cfg(test)]
        fault::hit(fault::Op::Alloc)?;
        let buf = self.stream.alloc_zeros::<T>(len.max(1)).map_err(cuda)?;
        let ptr = {
            let (p, _record) = buf.device_ptr(&self.stream);
            p
        };
        Ok(DevBuf {
            _buf: buf,
            ptr,
            len,
        })
    }

    fn htod<T>(&self, dst: u64, src: &[T]) -> Result<(), TsdfError> {
        if src.is_empty() {
            return Ok(());
        }
        // SAFETY: `dst` addresses at least `size_of_val(src)` device bytes
        // (callers size their buffers first); the copy is stream ordered.
        unsafe { result::memcpy_htod_async(dst, src, self.stream.cu_stream()) }.map_err(cuda)
    }

    fn dtoh<T>(&self, dst: &mut [T], src: u64) -> Result<(), TsdfError> {
        #[cfg(test)]
        fault::hit(fault::Op::Dtoh)?;
        if dst.is_empty() {
            return Ok(());
        }
        // SAFETY: `src` addresses at least `size_of_val(dst)` device bytes
        // of `T` values; a pageable destination makes the copy blocking.
        unsafe { result::memcpy_dtoh_async(dst, src, self.stream.cu_stream()) }.map_err(cuda)
    }

    fn dtod(&self, dst: u64, src: u64, bytes: usize) -> Result<(), TsdfError> {
        if bytes == 0 {
            return Ok(());
        }
        // SAFETY: both ranges are live device memory of `bytes` bytes.
        unsafe { result::memcpy_dtod_async(dst, src, bytes, self.stream.cu_stream()) }.map_err(cuda)
    }

    fn fill(&self, dst: u64, byte: u8, bytes: usize) -> Result<(), TsdfError> {
        if bytes == 0 {
            return Ok(());
        }
        // SAFETY: `dst` addresses `bytes` live device bytes.
        unsafe { result::memset_d8_async(dst, byte, bytes, self.stream.cu_stream()) }.map_err(cuda)
    }

    fn sync(&self) -> Result<(), TsdfError> {
        self.stream.synchronize().map_err(cuda)
    }
}

/// `buf` with at least `len` elements: kept, or replaced by a zeroed
/// buffer of `max(len, 2 * old)` (old content dropped).
fn ensure<T: DeviceRepr + ValidAsZeroBits>(
    dev: &Dev,
    buf: &mut DevBuf<T>,
    len: usize,
) -> Result<(), TsdfError> {
    if buf.len < len {
        *buf = dev.alloc(len.max(2 * buf.len))?;
    }
    Ok(())
}

/// `v` with at least `len` elements (geometric growth, cold).
fn ensure_host<T: Clone + Default>(v: &mut Vec<T>, len: usize) {
    if v.len() < len {
        v.resize(len.max(2 * v.len()), T::default());
    }
}

/// Hash slots for `capacity` blocks: a power of two `>= 2 * capacity`.
fn hash_slots(capacity: usize) -> usize {
    (2 * capacity).next_power_of_two().max(2)
}

fn round_up(x: usize, a: usize) -> usize {
    x.div_ceil(a) * a
}

// ---------------------------------------------------------------- index ----

/// The block indices of `headers` volumes side by side: header `h`'s hash
/// at slot `h * slots`, its per-pool arrays at `h * stride`.
struct DevIndex {
    headers: usize,
    stride: usize,
    slots: usize,
    log2_cap: u32,
    keys: DevBuf<u64>,
    vals: DevBuf<u32>,
    sorted_keys: [DevBuf<u64>; 2],
    sorted_pools: [DevBuf<u32>; 2],
    pool_keys: DevBuf<u64>,
    pages: DevBuf<u32>,
    modified: DevBuf<u64>,
    mark: DevBuf<u32>,
    dirty_mark: DevBuf<u32>,
}

impl DevIndex {
    fn new(dev: &Dev, headers: usize, stride: usize) -> Result<Self, TsdfError> {
        let slots = hash_slots(stride);
        let n = headers * stride;
        let keys = dev.alloc::<u64>(headers * slots)?;
        dev.fill(keys.ptr, 0xff, headers * slots * 8)?;
        Ok(Self {
            headers,
            stride,
            slots,
            log2_cap: slots.trailing_zeros(),
            keys,
            vals: dev.alloc(headers * slots)?,
            sorted_keys: [dev.alloc(n)?, dev.alloc(n)?],
            sorted_pools: [dev.alloc(n)?, dev.alloc(n)?],
            pool_keys: dev.alloc(n)?,
            pages: dev.alloc(n)?,
            modified: dev.alloc(n)?,
            mark: dev.alloc(n)?,
            dirty_mark: dev.alloc(n)?,
        })
    }

    fn keys(&self, h: usize) -> u64 {
        self.keys.at(h * self.slots)
    }

    fn vals(&self, h: usize) -> u64 {
        self.vals.at(h * self.slots)
    }

    fn pool<T>(&self, b: &DevBuf<T>, h: usize) -> u64 {
        b.at(h * self.stride)
    }

    /// Re-lays the indices out with `headers` headers of `stride` pools
    /// (both at least the current ones), copying every header's arrays
    /// and rebuilding its hash (cold).
    fn relayout(
        &mut self,
        dev: &Dev,
        heads: &[Head],
        headers: usize,
        stride: usize,
    ) -> Result<(), TsdfError> {
        let next = Self::new(dev, headers, stride)?;
        for (h, hd) in heads.iter().enumerate().take(self.headers) {
            let n = hd.len;
            let copy = |dst: u64, src: u64, size: usize| dev.dtod(dst, src, n * size);
            for half in 0..2 {
                copy(
                    next.pool(&next.sorted_keys[half], h),
                    self.pool(&self.sorted_keys[half], h),
                    8,
                )?;
                copy(
                    next.pool(&next.sorted_pools[half], h),
                    self.pool(&self.sorted_pools[half], h),
                    4,
                )?;
            }
            copy(
                next.pool(&next.pool_keys, h),
                self.pool(&self.pool_keys, h),
                8,
            )?;
            copy(next.pool(&next.pages, h), self.pool(&self.pages, h), 4)?;
            copy(
                next.pool(&next.modified, h),
                self.pool(&self.modified, h),
                8,
            )?;
            if n > 0 {
                let (keys, vals, log2, pk, len) = (
                    next.keys(h),
                    next.vals(h),
                    next.log2_cap,
                    next.pool(&next.pool_keys, h),
                    n as u32,
                );
                // SAFETY: tsdf_rehash(keys, vals, log2_cap, pool_keys, len)
                // over header h's fresh table and its `n` pool keys.
                unsafe {
                    dev.launch(
                        dev.k.rehash,
                        dev.grid_for(n),
                        &mut args![keys, vals, log2, pk, len],
                    )?;
                }
            }
        }
        *self = next;
        Ok(())
    }
}

/// Host mirror of one header's index.
#[derive(Debug, Clone, Copy, Default)]
struct Head {
    /// Allocated blocks.
    len: usize,
    /// Logical capacity (the CPU index's).
    cap: usize,
    /// Current half of the sorted lists.
    half: usize,
    /// Steps applied.
    epoch: u64,
}

/// Voxel pools on the device.
struct DevPools {
    /// Logical page capacity (the CPU pools').
    cap: usize,
    /// Pages in use.
    pages: usize,
    /// Allocated pages.
    phys: usize,
    voxels: usize,
    color: bool,
    tsdf: DevBuf<f32>,
    weight: DevBuf<u16>,
    rgbw: DevBuf<u32>,
}

impl DevPools {
    fn new(
        dev: &Dev,
        cap: usize,
        phys: usize,
        voxels: usize,
        color: bool,
    ) -> Result<Self, TsdfError> {
        Ok(Self {
            cap,
            pages: 0,
            phys,
            voxels,
            color,
            tsdf: dev.alloc(phys * voxels)?,
            weight: dev.alloc(phys * voxels)?,
            rgbw: dev.alloc(if color { phys * voxels } else { 0 })?,
        })
    }

    /// Reallocates `phys` zeroed pages, keeping the used ones (cold).
    fn grow(&mut self, dev: &Dev, phys: usize) -> Result<(), TsdfError> {
        let mut next = Self::new(dev, self.cap, phys, self.voxels, self.color)?;
        let n = self.pages * self.voxels;
        dev.dtod(next.tsdf.ptr, self.tsdf.ptr, n * 4)?;
        dev.dtod(next.weight.ptr, self.weight.ptr, n * 2)?;
        if self.color {
            dev.dtod(next.rgbw.ptr, self.rgbw.ptr, n * 4)?;
        }
        next.pages = self.pages;
        *self = next;
        Ok(())
    }

    fn bytes(&self) -> usize {
        self.phys * self.voxels * if self.color { 10 } else { 6 }
    }
}

// -------------------------------------------------------------- scratch ----

/// One staged observation of a step.
#[derive(Clone, Copy, Default)]
struct Staged {
    spherical: bool,
    pixels: usize,
    params: u64,
    depth: u64,
    depth_kind: u32,
    color: u64,
    tables: u64,
}

/// Reused device scratch and host staging.
struct Scratch {
    step: DevBuf<u8>,
    counters: DevBuf<u64>,
    host_counters: [u64; tc::SC_LEN],
    obs_params: DevBuf<u8>,
    list_counts: DevBuf<u32>,
    hits: DevBuf<u32>,
    news: DevBuf<u32>,
    miss_slots: usize,
    miss_keys: DevBuf<u64>,
    miss_mark: DevBuf<u32>,
    miss_list: DevBuf<u32>,
    miss_rank: DevBuf<u32>,
    new_keys: DevBuf<u64>,
    depth: DevBuf<u8>,
    color: DevBuf<u8>,
    tables: DevBuf<f32>,
    staged: Vec<Staged>,
    /// Last observation mark (pool and miss marks); 0 = never.
    seq: u32,
    // raycast
    views: DevBuf<u8>,
    view_headers: DevBuf<u32>,
    ray_depth: DevBuf<f32>,
    ray_vertices: DevBuf<f32>,
    ray_normals: DevBuf<f32>,
    ray_colors: DevBuf<u8>,
    // meshing
    mesh_params: DevBuf<u8>,
    nv: DevBuf<u32>,
    nt: DevBuf<u32>,
    v_off: DevBuf<u64>,
    t_off: DevBuf<u64>,
    out_vertices: DevBuf<f32>,
    out_normals: DevBuf<f32>,
    out_colors: DevBuf<u8>,
    out_triangles: DevBuf<u32>,
    dirty_list: DevBuf<u32>,
    dirty_count: DevBuf<u32>,
    dirty_seq: u32,
    host_nv: Vec<u32>,
    host_nt: Vec<u32>,
    host_v_off: Vec<u64>,
    host_t_off: Vec<u64>,
    host_dirty: Vec<u32>,
    host_sorted_keys: Vec<u64>,
    host_sorted_pools: Vec<u32>,
    host_order: Vec<(u64, u32)>,
    host_vertices: Vec<[f32; 3]>,
    host_normals: Vec<[f32; 3]>,
    host_colors: Vec<[u8; 3]>,
    host_triangles: Vec<[u32; 3]>,
}

impl Scratch {
    fn new(dev: &Dev, stride: usize) -> Result<Self, TsdfError> {
        let miss_slots = hash_slots(stride).max(MIN_MISS_SLOTS);
        Ok(Self {
            step: dev.alloc(PARAM_STRIDE)?,
            counters: dev.alloc(tc::SC_LEN)?,
            host_counters: [0; tc::SC_LEN],
            obs_params: dev.alloc(PARAM_STRIDE)?,
            list_counts: dev.alloc(2)?,
            hits: dev.alloc(stride)?,
            news: dev.alloc(miss_slots)?,
            miss_slots,
            miss_keys: dev.alloc(miss_slots)?,
            miss_mark: dev.alloc(miss_slots)?,
            miss_list: dev.alloc(miss_slots)?,
            miss_rank: dev.alloc(miss_slots)?,
            new_keys: dev.alloc(miss_slots)?,
            depth: dev.alloc(0)?,
            color: dev.alloc(0)?,
            tables: dev.alloc(0)?,
            staged: Vec::new(),
            seq: 0,
            views: dev.alloc(PARAM_STRIDE)?,
            view_headers: dev.alloc(1)?,
            ray_depth: dev.alloc(0)?,
            ray_vertices: dev.alloc(0)?,
            ray_normals: dev.alloc(0)?,
            ray_colors: dev.alloc(0)?,
            mesh_params: dev.alloc(PARAM_STRIDE)?,
            nv: dev.alloc(0)?,
            nt: dev.alloc(0)?,
            v_off: dev.alloc(0)?,
            t_off: dev.alloc(0)?,
            out_vertices: dev.alloc(0)?,
            out_normals: dev.alloc(0)?,
            out_colors: dev.alloc(0)?,
            out_triangles: dev.alloc(0)?,
            dirty_list: dev.alloc(stride)?,
            dirty_count: dev.alloc(1)?,
            dirty_seq: 0,
            host_nv: Vec::new(),
            host_nt: Vec::new(),
            host_v_off: Vec::new(),
            host_t_off: Vec::new(),
            host_dirty: Vec::new(),
            host_sorted_keys: Vec::new(),
            host_sorted_pools: Vec::new(),
            host_order: Vec::new(),
            host_vertices: Vec::new(),
            host_normals: Vec::new(),
            host_colors: Vec::new(),
            host_triangles: Vec::new(),
        })
    }

    fn bytes(&self) -> usize {
        let b = |len: usize, size: usize| len * size;
        b(self.hits.len, 4)
            + b(self.news.len, 4)
            + b(self.miss_slots, 8 + 4 + 4 + 4 + 8)
            + self.depth.len
            + self.color.len
            + b(self.tables.len, 4)
            + b(self.ray_depth.len, 4)
            + b(self.ray_vertices.len + self.ray_normals.len, 4)
            + self.ray_colors.len
            + b(self.nv.len + self.nt.len, 4)
            + b(self.v_off.len + self.t_off.len, 8)
            + b(
                self.out_vertices.len + self.out_normals.len + self.out_triangles.len,
                4,
            )
            + self.out_colors.len
            + b(self.dirty_list.len, 4)
    }
}

// --------------------------------------------------------------- engine ----

/// Device storage and kernels shared by both CUDA types: an index of
/// `heads.len()` headers over one page arena.
struct Engine {
    dev: Dev,
    log2: u32,
    layout: TsdfLayout,
    params: TsdfParams,
    index: DevIndex,
    heads: Vec<Head>,
    pools: DevPools,
    s: Scratch,
}

/// Where one step's observations come from: `n` of them, `obs(s)` in fold
/// order, already validated.
struct StepInput<'a, 'o> {
    n: usize,
    obs: &'a dyn Fn(usize) -> Result<Obs<'o>, TsdfError>,
}

impl Engine {
    #[allow(clippy::too_many_arguments)]
    fn new(
        stream: Arc<CudaStream>,
        log2: u32,
        layout: TsdfLayout,
        params: TsdfParams,
        headers: usize,
        header_cap: usize,
        arena_cap: usize,
    ) -> Result<Self, TsdfError> {
        let dev = Dev::new(stream)?;
        dev.bind()?;
        let voxels = 1usize << (3 * log2);
        // a fixed pool is allocated whole; a growing one starts small and
        // grows on demand (cold steps)
        let (stride, phys) = match params.growth {
            PoolGrowth::Fixed => (header_cap, arena_cap),
            PoolGrowth::Geometric => (
                header_cap.clamp(1, MIN_BLOCKS),
                arena_cap.clamp(1, MIN_BLOCKS),
            ),
        };
        let index = DevIndex::new(&dev, headers, stride)?;
        let pools = DevPools::new(&dev, arena_cap, phys, voxels, params.color)?;
        let s = Scratch::new(&dev, stride)?;
        let heads = vec![
            Head {
                cap: header_cap,
                ..Head::default()
            };
            headers
        ];
        dev.sync()?;
        Ok(Self {
            dev,
            log2,
            layout,
            params,
            index,
            heads,
            pools,
            s,
        })
    }

    fn voxels(&self) -> usize {
        1usize << (3 * self.log2)
    }

    /// Test hook: `(heads, index headers, index stride, dirty list, hits)`.
    #[cfg(test)]
    fn dims(&self) -> (usize, usize, usize, usize, usize) {
        let ix = &self.index;
        (
            self.heads.len(),
            ix.headers,
            ix.stride,
            self.s.dirty_list.len,
            self.s.hits.len,
        )
    }

    fn device_bytes(&self) -> usize {
        let ix = &self.index;
        ix.headers * (ix.slots * 12 + ix.stride * (2 * 12 + 8 + 4 + 8 + 4 + 4))
            + self.pools.bytes()
            + self.s.bytes()
    }

    /// The next observation mark; on wrap all marks are cleared.
    fn next_seq(&mut self) -> Result<u32, TsdfError> {
        if self.s.seq == u32::MAX {
            let ix = &self.index;
            self.dev.fill(ix.mark.ptr, 0, ix.mark.len * 4)?;
            self.dev
                .fill(self.s.miss_mark.ptr, 0, self.s.miss_mark.len * 4)?;
            self.s.seq = 0;
        }
        self.s.seq += 1;
        Ok(self.s.seq)
    }

    /// Uploads the step's depth, colour, LiDAR tables and parameters.
    fn stage(&mut self, input: &StepInput<'_, '_>) -> Result<(), TsdfError> {
        let (mut depth, mut color, mut tables) = (0usize, 0usize, 0usize);
        for s in 0..input.n {
            let o = (input.obs)(s)?;
            let (bytes, pixels) = match o.depth {
                DepthRef::U16(d) => (d.image().as_slice().len() * 2, d.image().as_slice().len()),
                DepthRef::F32(d) => (d.image().as_slice().len() * 4, d.image().as_slice().len()),
            };
            depth += round_up(bytes, STAGE_ALIGN);
            match o.kind {
                ObsKind::Pinhole(_) => {
                    if o.color.is_some() && self.pools.color {
                        color += round_up(3 * pixels, STAGE_ALIGN);
                    }
                }
                ObsKind::Spherical(p, _) => {
                    let n = (p.height - 1 + 2 * p.height + 2 * p.width) as usize;
                    tables += round_up(n, STAGE_ALIGN / 4);
                }
            }
        }
        let dev = &self.dev;
        ensure(dev, &mut self.s.depth, depth)?;
        ensure(dev, &mut self.s.color, color)?;
        ensure(dev, &mut self.s.tables, tables)?;
        ensure(dev, &mut self.s.obs_params, input.n * PARAM_STRIDE)?;
        ensure(dev, &mut self.s.list_counts, 2 * input.n)?;
        ensure(dev, &mut self.s.hits, input.n * self.index.stride)?;
        ensure(dev, &mut self.s.news, input.n * self.s.miss_slots)?;
        self.s.staged.clear();
        let (mut depth, mut color, mut tables) = (0usize, 0usize, 0usize);
        for s in 0..input.n {
            let o = (input.obs)(s)?;
            let mut st = Staged {
                params: self.s.obs_params.at(s * PARAM_STRIDE),
                depth: self.s.depth.at(depth),
                ..Staged::default()
            };
            match o.depth {
                DepthRef::U16(d) => {
                    let px = d.image().as_slice();
                    dev.htod(st.depth, px)?;
                    st.depth_kind = DEPTH_U16;
                    st.pixels = px.len();
                    depth += round_up(px.len() * 2, STAGE_ALIGN);
                }
                DepthRef::F32(d) => {
                    let px = d.image().as_slice();
                    dev.htod(st.depth, px)?;
                    st.depth_kind = DEPTH_F32;
                    st.pixels = px.len();
                    depth += round_up(px.len() * 4, STAGE_ALIGN);
                }
            }
            match o.kind {
                ObsKind::Pinhole(p) => {
                    dev.htod(st.params, std::slice::from_ref(&p))?;
                    if let (Some(c), true) = (o.color, self.pools.color) {
                        st.color = self.s.color.at(color);
                        dev.htod(st.color, c)?;
                        color += round_up(c.len(), STAGE_ALIGN);
                    }
                }
                ObsKind::Spherical(p, lidar) => {
                    st.spherical = true;
                    dev.htod(st.params, std::slice::from_ref(&p))?;
                    let t = lidar.tables();
                    st.tables = self.s.tables.at(tables);
                    let mut at = st.tables;
                    for part in [t.row_boundaries, t.cos_el, t.sin_el, t.cos_az, t.sin_az] {
                        dev.htod(at, part)?;
                        at += (part.len() * 4) as u64;
                    }
                    tables += round_up(
                        t.row_boundaries.len()
                            + t.cos_el.len()
                            + t.sin_el.len()
                            + t.cos_az.len()
                            + t.sin_az.len(),
                        STAGE_ALIGN / 4,
                    );
                }
            }
            self.s.staged.push(st);
        }
        Ok(())
    }

    /// One integration step into header `h` (contract sections 9-10).
    fn step(
        &mut self,
        h: usize,
        input: &StepInput<'_, '_>,
        growth: PoolGrowth,
    ) -> Result<IntegrateStats, TsdfError> {
        self.dev.bind()?;
        self.stage(input)?;
        let epoch = self.heads[h].epoch + 1;
        loop {
            self.run_step(h, input.n, epoch)?;
            let c = self.s.host_counters;
            let abort = c[tc::SC_ABORT];
            if abort & tc::ABORT_MISS_SET != 0 {
                // A full set holds more misses than the pool can ever
                // take: past `MAX_EXACT_MISS_SLOTS` fail now, not after
                // doubling (and re-running every touch) to count them all.
                let slots = self.s.miss_slots;
                if slots >= MAX_EXACT_MISS_SLOTS && slots > self.pool_room(h, growth) {
                    self.make_room(h, slots + 1, growth)?;
                }
                self.grow_miss_set(input.n)?;
                continue;
            }
            if abort & tc::ABORT_POOL_FULL != 0 {
                self.make_room(h, c[tc::SC_MISSES] as usize, growth)?;
                continue;
            }
            let m = c[tc::SC_MISSES] as usize;
            let hd = &mut self.heads[h];
            hd.len += m;
            if m > 0 {
                hd.half ^= 1;
            }
            hd.epoch = epoch;
            self.pools.pages += m;
            return Ok(IntegrateStats {
                valid_pixels: c[tc::SC_VALID],
                touched_blocks: c[tc::SC_TOUCHED] as u32,
                new_blocks: m as u32,
                updated_voxels: c[tc::SC_UPDATED],
                out_of_range: c[tc::SC_OUT_OF_RANGE],
            });
        }
    }

    /// Launches one attempt of a step and downloads its counters (the
    /// step's one sync). Nothing observable changes on an abort.
    fn run_step(&mut self, h: usize, n: usize, epoch: u64) -> Result<(), TsdfError> {
        let hd = self.heads[h];
        let ix = &self.index;
        let st = tc::StepParams {
            grid: self.layout.grid(self.log2),
            max_weight: self.params.max_weight as u32,
            max_color_weight: self.params.max_color_weight as u32,
            log2_cap: ix.log2_cap,
            log2_miss: self.s.miss_slots.trailing_zeros(),
            len: hd.len as u32,
            pages: self.pools.pages as u32,
            index_cap: hd.cap.min(ix.stride) as u32,
            arena_cap: self.pools.cap.min(self.pools.phys) as u32,
            epoch,
        };
        let dev = &self.dev;
        dev.htod(self.s.step.ptr, std::slice::from_ref(&st))?;
        dev.fill(self.s.counters.ptr, 0, tc::SC_LEN * 8)?;
        dev.fill(self.s.list_counts.ptr, 0, 2 * n * 4)?;
        dev.fill(self.s.miss_keys.ptr, 0xff, self.s.miss_slots * 8)?;
        let (keys, vals) = (ix.keys(h), ix.vals(h));
        let (pool_keys, pages, modified, mark) = (
            ix.pool(&ix.pool_keys, h),
            ix.pool(&ix.pages, h),
            ix.pool(&ix.modified, h),
            ix.pool(&ix.mark, h),
        );
        let s = &self.s;
        let (step, counters) = (s.step.ptr, s.counters.ptr);
        let (miss_keys, miss_mark, miss_list, miss_rank, new_keys) = (
            s.miss_keys.ptr,
            s.miss_mark.ptr,
            s.miss_list.ptr,
            s.miss_rank.ptr,
            s.new_keys.ptr,
        );
        for i in 0..n {
            let o = self.s.staged[i];
            let obs_mark = self.next_seq()?;
            let s = &self.s;
            let dev = &self.dev;
            let (hits, news, lc) = (
                s.hits.at(i * self.index.stride),
                s.news.at(i * s.miss_slots),
                s.list_counts.at(2 * i),
            );
            let grid = dev.grid_for(o.pixels);
            // SAFETY: argument lists mirror the kernels' signatures in
            // tsdf.rs; every buffer is sized for this step by `stage` and
            // the index layout.
            unsafe {
                if o.spherical {
                    dev.launch(
                        dev.k.touch_spherical,
                        grid,
                        &mut args![
                            step,
                            o.params,
                            o.tables,
                            o.depth,
                            o.depth_kind,
                            keys,
                            vals,
                            mark,
                            obs_mark,
                            miss_keys,
                            miss_mark,
                            miss_list,
                            hits,
                            news,
                            lc,
                            counters
                        ],
                    )?;
                } else {
                    dev.launch(
                        dev.k.touch_pinhole,
                        grid,
                        &mut args![
                            step,
                            o.params,
                            o.depth,
                            o.depth_kind,
                            keys,
                            vals,
                            mark,
                            obs_mark,
                            miss_keys,
                            miss_mark,
                            miss_list,
                            hits,
                            news,
                            lc,
                            counters
                        ],
                    )?;
                }
            }
        }
        let ix = &self.index;
        let dev = &self.dev;
        let half = hd.half;
        let (old_keys, old_pools, out_keys, out_pools) = (
            ix.pool(&ix.sorted_keys[half], h),
            ix.pool(&ix.sorted_pools[half], h),
            ix.pool(&ix.sorted_keys[1 - half], h),
            ix.pool(&ix.sorted_pools[1 - half], h),
        );
        let g = dev.grid;
        // SAFETY: as above.
        unsafe {
            dev.launch(
                dev.k.rank,
                g,
                &mut args![step, miss_keys, miss_list, miss_rank, new_keys, counters],
            )?;
            dev.launch(
                dev.k.assign,
                g,
                &mut args![step, new_keys, keys, vals, pool_keys, pages, modified, mark, counters],
            )?;
            dev.launch(
                dev.k.merge,
                g,
                &mut args![step, new_keys, old_keys, old_pools, out_keys, out_pools, counters],
            )?;
        }
        let s = &self.s;
        let (tsdf, weight, rgbw) = (
            self.pools.tsdf.ptr,
            self.pools.weight.ptr,
            self.pools.rgbw.ptr,
        );
        for i in 0..n {
            let o = s.staged[i];
            let (hits, news, lc) = (
                s.hits.at(i * ix.stride),
                s.news.at(i * s.miss_slots),
                s.list_counts.at(2 * i),
            );
            // SAFETY: as above.
            unsafe {
                if o.spherical {
                    dev.launch(
                        dev.k.integrate_spherical,
                        g,
                        &mut args![
                            step,
                            o.params,
                            o.tables,
                            o.depth,
                            o.depth_kind,
                            tsdf,
                            weight,
                            pool_keys,
                            pages,
                            hits,
                            news,
                            lc,
                            miss_rank,
                            modified,
                            counters
                        ],
                    )?;
                } else {
                    dev.launch(
                        dev.k.integrate_pinhole,
                        g,
                        &mut args![
                            step,
                            o.params,
                            o.depth,
                            o.depth_kind,
                            o.color,
                            tsdf,
                            weight,
                            rgbw,
                            pool_keys,
                            pages,
                            hits,
                            news,
                            lc,
                            miss_rank,
                            modified,
                            counters
                        ],
                    )?;
                }
            }
        }
        let mut c = [0u64; tc::SC_LEN];
        dev.dtoh(&mut c, counters)?;
        dev.sync()?;
        self.s.host_counters = c;
        Ok(())
    }

    /// Doubles the miss set (cold) after it overflowed.
    fn grow_miss_set(&mut self, n: usize) -> Result<(), TsdfError> {
        self.resize_miss_set(n, 2 * self.s.miss_slots)
    }

    /// A miss set of `slots` (a power of two) for `n` observations; all
    /// or nothing (every buffer is allocated before any is replaced).
    fn resize_miss_set(&mut self, n: usize, slots: usize) -> Result<(), TsdfError> {
        let dev = &self.dev;
        let miss_keys = dev.alloc(slots)?;
        let miss_mark = dev.alloc(slots)?;
        let miss_list = dev.alloc(slots)?;
        let miss_rank = dev.alloc(slots)?;
        let new_keys = dev.alloc(slots)?;
        let news = dev.alloc(n.max(1) * slots)?;
        let s = &mut self.s;
        (s.miss_keys, s.miss_mark, s.miss_list) = (miss_keys, miss_mark, miss_list);
        (s.miss_rank, s.new_keys, s.news) = (miss_rank, new_keys, news);
        s.miss_slots = slots;
        Ok(())
    }

    /// Most blocks header `h` can still take: its logical capacity under
    /// `Fixed`, the `u32` voxel-index limit otherwise.
    fn pool_room(&self, h: usize, growth: PoolGrowth) -> usize {
        let pages = self.pools.pages;
        match growth {
            PoolGrowth::Fixed => (self.heads[h].cap.saturating_sub(self.heads[h].len))
                .min(self.pools.cap.saturating_sub(pages)),
            PoolGrowth::Geometric => max_blocks(self.voxels()).saturating_sub(pages),
        }
    }

    /// Room for `m` more blocks in header `h`: the CPU's capacity rule
    /// (`ensure_capacity`) on the logical capacities, then the physical
    /// buffers (cold).
    fn make_room(&mut self, h: usize, m: usize, growth: PoolGrowth) -> Result<(), TsdfError> {
        let limit = max_blocks(self.voxels());
        let hd = self.heads[h];
        let (need_index, need_pages) = (hd.len + m, self.pools.pages + m);
        if need_index > hd.cap || need_pages > self.pools.cap {
            if growth == PoolGrowth::Fixed || need_pages > limit {
                let (capacity, needed) = if need_index > hd.cap {
                    (hd.cap, need_index)
                } else {
                    (self.pools.cap, need_pages)
                };
                return Err(TsdfError::PoolFull { capacity, needed });
            }
            if need_index > hd.cap {
                self.heads[h].cap = (2 * hd.cap).max(need_index).min(limit);
            }
            if need_pages > self.pools.cap {
                self.pools.cap = (2 * self.pools.cap).max(need_pages).min(limit);
            }
        }
        self.reserve_phys(need_index, need_pages)
    }

    /// Physical room for `blocks` blocks per header and `pages` pages.
    fn reserve_phys(&mut self, blocks: usize, pages: usize) -> Result<(), TsdfError> {
        if blocks > self.index.stride {
            let cap = self.heads.iter().map(|h| h.cap).max().unwrap_or(blocks);
            let stride = (2 * self.index.stride).max(blocks).min(cap.max(blocks));
            let dev = &self.dev;
            let n = self.s.staged.len().max(1);
            // The scratch first, the index last: a failure leaves the old
            // stride, so a retry regrows whatever is missing.
            ensure(dev, &mut self.s.hits, n * stride)?;
            ensure(dev, &mut self.s.dirty_list, stride)?;
            let slots = hash_slots(stride);
            if slots > self.s.miss_slots {
                self.resize_miss_set(n, slots)?;
            }
            let headers = self.index.headers;
            self.index
                .relayout(&self.dev, &self.heads, headers, stride)?;
        }
        if pages > self.pools.phys {
            let phys = (2 * self.pools.phys)
                .max(pages)
                .min(self.pools.cap.max(pages));
            self.pools.grow(&self.dev, phys)?;
        }
        self.dev.sync()
    }

    /// Raycasts `n` staged views (parameters and header per view already
    /// uploaded) into `out` (contract section 13).
    fn raycast(
        &mut self,
        nviews: usize,
        width: usize,
        height: usize,
        out: RayOut<'_>,
    ) -> Result<(), TsdfError> {
        let px = width * height;
        let dev = &self.dev;
        let want_depth = out.depth.is_some();
        let want_colors = out.colors.is_some();
        ensure(dev, &mut self.s.ray_vertices, 3 * px)?;
        ensure(dev, &mut self.s.ray_normals, 3 * px)?;
        if want_depth {
            ensure(dev, &mut self.s.ray_depth, px)?;
        }
        if want_colors {
            ensure(dev, &mut self.s.ray_colors, 3 * px)?;
        }
        let s = &self.s;
        let ix = &self.index;
        let (views, view_headers, nviews) = (s.views.ptr, s.view_headers.ptr, nviews as u32);
        let (keys, vals, hash_stride, log2_cap) =
            (ix.keys.ptr, ix.vals.ptr, ix.slots as u32, ix.log2_cap);
        let (pages, pool_stride) = (ix.pages.ptr, ix.stride as u32);
        let (tsdf, weight, rgbw, log2) = (
            self.pools.tsdf.ptr,
            self.pools.weight.ptr,
            self.pools.rgbw.ptr,
            self.log2,
        );
        let (w, hgt) = (width as u32, height as u32);
        let out_depth = if want_depth { s.ray_depth.ptr } else { 0 };
        let out_colors = if want_colors { s.ray_colors.ptr } else { 0 };
        let (out_v, out_n) = (s.ray_vertices.ptr, s.ray_normals.ptr);
        // SAFETY: tsdf_raycast's signature; views/headers hold `nviews`
        // entries, outputs `px` pixels.
        unsafe {
            dev.launch(
                dev.k.raycast,
                dev.grid_for(px),
                &mut args![
                    views,
                    view_headers,
                    nviews,
                    keys,
                    vals,
                    hash_stride,
                    log2_cap,
                    pages,
                    pool_stride,
                    tsdf,
                    weight,
                    rgbw,
                    log2,
                    w,
                    hgt,
                    out_depth,
                    out_v,
                    out_n,
                    out_colors
                ],
            )?;
        }
        let RayOut {
            depth,
            vertices,
            normals,
            colors,
        } = out;
        if let Some(d) = depth {
            dev.dtoh(&mut d[..px], out_depth)?;
        }
        dev.dtoh(&mut vertices[..3 * px], out_v)?;
        dev.dtoh(&mut normals[..3 * px], out_n)?;
        if let Some(c) = colors {
            dev.dtoh(&mut c[..3 * px], out_colors)?;
        }
        dev.sync()
    }

    /// Uploads the raycast parameters of views `(params, header)`: a
    /// contiguous array, as the kernel indexes it.
    fn stage_views(&mut self, views: &[(tc::RaycastParams, u32)]) -> Result<(), TsdfError> {
        let dev = &self.dev;
        let size = size_of::<tc::RaycastParams>();
        ensure(dev, &mut self.s.views, views.len() * size)?;
        ensure(dev, &mut self.s.view_headers, views.len())?;
        for (v, (rp, h)) in views.iter().enumerate() {
            dev.htod(self.s.views.at(v * size), std::slice::from_ref(rp))?;
            dev.htod(self.s.view_headers.at(v), std::slice::from_ref(h))?;
        }
        Ok(())
    }

    fn mesh_params(
        &self,
        mp: &MeshParams,
        xf: Option<&tc::Rigid>,
        global: bool,
        base: u32,
    ) -> tc::MeshParamsDev {
        let g = self.layout.grid(self.log2);
        tc::MeshParamsDev {
            world_from_volume: xf.copied().unwrap_or(tc::Rigid {
                r: [0.0; 9],
                t: [0.0; 3],
            }),
            voxel_size: g.voxel_size,
            half_voxel: g.half_voxel,
            log2: self.log2,
            log2_cap: self.index.log2_cap,
            weight_threshold: mp.weight_threshold as u32,
            normals: mp.normals as u32,
            colors: mp.colors as u32,
            transform: xf.is_some() as u32,
            global_index: global as u32,
            index_base: base,
        }
    }

    /// Counts (`tsdf_mc_count`) the meshes of the `n` blocks listed at
    /// `list` in header `h` into `host_nv`/`host_nt`.
    fn mesh_count(&mut self, h: usize, list: u64, n: usize) -> Result<(), TsdfError> {
        let dev = &self.dev;
        // sized by the capacity, so a growing volume does not regrow them
        let cap = n.max(self.index.stride);
        ensure(dev, &mut self.s.nv, cap)?;
        ensure(dev, &mut self.s.nt, cap)?;
        ensure_host(&mut self.s.host_nv, cap);
        ensure_host(&mut self.s.host_nt, cap);
        let ix = &self.index;
        let (mp, keys, vals, pages, pool_keys) = (
            self.s.mesh_params.ptr,
            ix.keys(h),
            ix.vals(h),
            ix.pool(&ix.pages, h),
            ix.pool(&ix.pool_keys, h),
        );
        let (tsdf, weight, nn, nv, nt) = (
            self.pools.tsdf.ptr,
            self.pools.weight.ptr,
            n as u32,
            self.s.nv.ptr,
            self.s.nt.ptr,
        );
        // SAFETY: tsdf_mc_count's signature; `list` holds `n` pools of
        // header h, nv/nt `n` entries.
        unsafe {
            dev.launch(
                dev.k.mc_count,
                (n.min(dev.grid as usize)) as u32,
                &mut args![mp, keys, vals, pages, pool_keys, tsdf, weight, list, nn, nv, nt],
            )?;
        }
        dev.dtoh(&mut self.s.host_nv[..n], nv)?;
        dev.dtoh(&mut self.s.host_nt[..n], nt)?;
        dev.sync()
    }

    /// Exclusive prefix sums of the counted blocks (`host_v_off`,
    /// `host_t_off`); returns the totals. `base` is the vertex count of the
    /// mesh before this pass (for the `u32` index check, as the CPU's
    /// per-block check).
    fn mesh_offsets(&mut self, n: usize, base: usize) -> Result<(usize, usize), TsdfError> {
        let cap = n.max(self.index.stride);
        ensure_host(&mut self.s.host_v_off, cap);
        ensure_host(&mut self.s.host_t_off, cap);
        let (mut v, mut t) = (0usize, 0usize);
        for i in 0..n {
            self.s.host_v_off[i] = v as u64;
            self.s.host_t_off[i] = t as u64;
            v += self.s.host_nv[i] as usize;
            t += self.s.host_nt[i] as usize;
            if base + v > u32::MAX as usize {
                return Err(TsdfError::MeshTooLarge { vertices: base + v });
            }
        }
        Ok((v, t))
    }

    /// Emits (`tsdf_mc_emit`) the counted blocks into the output buffers.
    fn mesh_emit(
        &mut self,
        h: usize,
        list: u64,
        n: usize,
        totals: (usize, usize),
        mp: &MeshParams,
    ) -> Result<(), TsdfError> {
        let dev = &self.dev;
        let cap = n.max(self.index.stride);
        ensure(dev, &mut self.s.v_off, cap)?;
        ensure(dev, &mut self.s.t_off, cap)?;
        ensure(dev, &mut self.s.out_vertices, 3 * totals.0)?;
        ensure(dev, &mut self.s.out_triangles, 3 * totals.1)?;
        if mp.normals {
            ensure(dev, &mut self.s.out_normals, 3 * totals.0)?;
        }
        if mp.colors {
            ensure(dev, &mut self.s.out_colors, 3 * totals.0)?;
        }
        dev.htod(self.s.v_off.ptr, &self.s.host_v_off[..n])?;
        dev.htod(self.s.t_off.ptr, &self.s.host_t_off[..n])?;
        let ix = &self.index;
        let s = &self.s;
        let (mpp, keys, vals, pages, pool_keys) = (
            s.mesh_params.ptr,
            ix.keys(h),
            ix.vals(h),
            ix.pool(&ix.pages, h),
            ix.pool(&ix.pool_keys, h),
        );
        let (tsdf, weight) = (self.pools.tsdf.ptr, self.pools.weight.ptr);
        let rgbw = if mp.colors { self.pools.rgbw.ptr } else { 0 };
        let (nn, v_off, t_off) = (n as u32, s.v_off.ptr, s.t_off.ptr);
        let (ov, ot) = (s.out_vertices.ptr, s.out_triangles.ptr);
        let on = if mp.normals { s.out_normals.ptr } else { 0 };
        let oc = if mp.colors { s.out_colors.ptr } else { 0 };
        // SAFETY: tsdf_mc_emit's signature; outputs hold the counted
        // totals at the uploaded offsets.
        unsafe {
            dev.launch(
                dev.k.mc_emit,
                (n.min(dev.grid as usize)) as u32,
                &mut args![
                    mpp, keys, vals, pages, pool_keys, tsdf, weight, rgbw, list, nn, v_off, t_off,
                    ov, on, oc, ot
                ],
            )?;
        }
        Ok(())
    }

    /// Full extraction of header `h` appended to `mesh` (global indices),
    /// mapped by `xf` when given. On error `mesh` is as it was.
    fn extract(
        &mut self,
        h: usize,
        mp: &MeshParams,
        xf: Option<&tc::Rigid>,
        mesh: &mut TriangleMesh,
    ) -> Result<(), TsdfError> {
        let (v, t) = (mesh.vertices.len(), mesh.triangles.len());
        let (n, c) = (mesh.normals.len(), mesh.colors.len());
        let res = self.extract_append(h, mp, xf, mesh);
        if res.is_err() {
            mesh.vertices.truncate(v);
            mesh.triangles.truncate(t);
            mesh.normals.truncate(n);
            mesh.colors.truncate(c);
        }
        res
    }

    fn extract_append(
        &mut self,
        h: usize,
        mp: &MeshParams,
        xf: Option<&tc::Rigid>,
        mesh: &mut TriangleMesh,
    ) -> Result<(), TsdfError> {
        let hd = self.heads[h];
        let n = hd.len;
        if n == 0 {
            return Ok(());
        }
        let base = mesh.vertices.len();
        let params = self.mesh_params(mp, xf, true, base as u32);
        self.dev
            .htod(self.s.mesh_params.ptr, std::slice::from_ref(&params))?;
        let list = self.index.pool(&self.index.sorted_pools[hd.half], h);
        self.mesh_count(h, list, n)?;
        let totals = self.mesh_offsets(n, base)?;
        self.mesh_emit(h, list, n, totals, mp)?;
        let (nv, nt) = totals;
        let tb = mesh.triangles.len();
        mesh.vertices.resize(base + nv, [0.0; 3]);
        mesh.triangles.resize(tb + nt, [0; 3]);
        let dev = &self.dev;
        dev.dtoh(
            mesh.vertices[base..].as_flattened_mut(),
            self.s.out_vertices.ptr,
        )?;
        dev.dtoh(
            mesh.triangles[tb..].as_flattened_mut(),
            self.s.out_triangles.ptr,
        )?;
        if mp.normals {
            let b = mesh.normals.len();
            mesh.normals.resize(b + nv, [0.0; 3]);
            dev.dtoh(mesh.normals[b..].as_flattened_mut(), self.s.out_normals.ptr)?;
        }
        if mp.colors {
            let b = mesh.colors.len();
            mesh.colors.resize(b + nv, [0; 3]);
            dev.dtoh(mesh.colors[b..].as_flattened_mut(), self.s.out_colors.ptr)?;
        }
        dev.sync()
    }

    /// Incremental meshing of header `h` into `cache` (contract 12.6):
    /// dirty blocks found and meshed on the device, stored in the host
    /// cache slab.
    fn update_cache(
        &mut self,
        h: usize,
        mp: &MeshParams,
        cache: &mut BlockMeshCache,
    ) -> Result<MeshUpdateStats, TsdfError> {
        let hd = self.heads[h];
        let (cache_epoch, cache_len) = cache.begin_update(mp, hd.epoch, hd.len);
        let n = hd.len;
        let dev = &self.dev;
        if self.s.dirty_seq == u32::MAX {
            dev.fill(self.index.dirty_mark.ptr, 0, self.index.dirty_mark.len * 4)?;
            self.s.dirty_seq = 0;
        }
        self.s.dirty_seq += 1;
        dev.fill(self.s.dirty_count.ptr, 0, 4)?;
        let ix = &self.index;
        let s = &self.s;
        let (keys, vals, log2_cap) = (ix.keys(h), ix.vals(h), ix.log2_cap);
        let (pool_keys, modified, len) = (
            ix.pool(&ix.pool_keys, h),
            ix.pool(&ix.modified, h),
            n as u32,
        );
        let (ce, cl, normals) = (cache_epoch, cache_len as u32, mp.normals as u32);
        let (dmark, mark, dlist, dcount) = (
            ix.pool(&ix.dirty_mark, h),
            s.dirty_seq,
            s.dirty_list.ptr,
            s.dirty_count.ptr,
        );
        // SAFETY: tsdf_mc_dirty's signature; per-pool arrays of header h
        // and the dirty list hold `len` (<= stride) entries.
        unsafe {
            dev.launch(
                dev.k.mc_dirty,
                dev.grid_for(n),
                &mut args![
                    keys, vals, log2_cap, pool_keys, modified, len, ce, cl, normals, dmark, mark,
                    dlist, dcount
                ],
            )?;
        }
        let cap = n.max(ix.stride);
        ensure_host(&mut self.s.host_sorted_keys, cap);
        ensure_host(&mut self.s.host_sorted_pools, cap);
        if self.s.host_order.capacity() < cap {
            self.s
                .host_order
                .reserve_exact(cap - self.s.host_order.len());
        }
        let mut count = [0u32; 1];
        dev.dtoh(&mut count, dcount)?;
        dev.dtoh(
            &mut self.s.host_sorted_keys[..n],
            ix.pool(&ix.sorted_keys[hd.half], h),
        )?;
        dev.dtoh(
            &mut self.s.host_sorted_pools[..n],
            ix.pool(&ix.sorted_pools[hd.half], h),
        )?;
        dev.sync()?;
        let nd = count[0] as usize;
        if nd > 0 {
            let params = self.mesh_params(mp, None, false, 0);
            self.dev
                .htod(self.s.mesh_params.ptr, std::slice::from_ref(&params))?;
            let list = self.s.dirty_list.ptr;
            self.mesh_count(h, list, nd)?;
            let totals = self.mesh_offsets(nd, 0)?;
            self.mesh_emit(h, list, nd, totals, mp)?;
            let s = &mut self.s;
            ensure_host(&mut s.host_dirty, nd.max(self.index.stride));
            ensure_host(&mut s.host_vertices, totals.0);
            ensure_host(&mut s.host_triangles, totals.1);
            let dev = &self.dev;
            dev.dtoh(&mut s.host_dirty[..nd], list)?;
            dev.dtoh(
                s.host_vertices[..totals.0].as_flattened_mut(),
                s.out_vertices.ptr,
            )?;
            dev.dtoh(
                s.host_triangles[..totals.1].as_flattened_mut(),
                s.out_triangles.ptr,
            )?;
            if mp.normals {
                ensure_host(&mut s.host_normals, totals.0);
                dev.dtoh(
                    s.host_normals[..totals.0].as_flattened_mut(),
                    s.out_normals.ptr,
                )?;
            }
            if mp.colors {
                ensure_host(&mut s.host_colors, totals.0);
                dev.dtoh(
                    s.host_colors[..totals.0].as_flattened_mut(),
                    s.out_colors.ptr,
                )?;
            }
            dev.sync()?;
            for j in 0..nd {
                let (v0, t0) = (s.host_v_off[j] as usize, s.host_t_off[j] as usize);
                let (v1, t1) = (v0 + s.host_nv[j] as usize, t0 + s.host_nt[j] as usize);
                let part = |on: bool, len: usize| if on { v0..v1 } else { len..len };
                cache.store_parts(
                    s.host_dirty[j] as usize,
                    &s.host_vertices[v0..v1],
                    &s.host_normals[part(mp.normals, 0)],
                    &s.host_colors[part(mp.colors, 0)],
                    &s.host_triangles[t0..t1],
                    mp,
                );
            }
        }
        let s = &mut self.s;
        s.host_order.clear();
        s.host_order.extend(
            s.host_sorted_keys[..n]
                .iter()
                .copied()
                .zip(s.host_sorted_pools[..n].iter().copied()),
        );
        Ok(cache.finish_update(&s.host_order, hd.epoch, nd as u32))
    }

    /// Downloads header `h` into a host index and pools (identity pages).
    fn download(
        &self,
        h: usize,
        index: &mut BlockIndex,
        pools: &mut Pools,
    ) -> Result<(), TsdfError> {
        self.dev.bind()?;
        let hd = self.heads[h];
        let n = hd.len;
        let v = self.voxels();
        let ix = &self.index;
        let dev = &self.dev;
        let mut pool_keys = vec![0u64; n];
        let mut pages = vec![0u32; n];
        let mut modified = vec![0u64; n];
        let mut skeys = vec![0u64; n];
        let mut spools = vec![0u32; n];
        dev.dtoh(&mut pool_keys, ix.pool(&ix.pool_keys, h))?;
        dev.dtoh(&mut pages, ix.pool(&ix.pages, h))?;
        dev.dtoh(&mut modified, ix.pool(&ix.modified, h))?;
        dev.dtoh(&mut skeys, ix.pool(&ix.sorted_keys[hd.half], h))?;
        dev.dtoh(&mut spools, ix.pool(&ix.sorted_pools[hd.half], h))?;
        // the whole arena in three copies, then this header's pages gathered
        let used = self.pools.pages * v;
        let mut tsdf = vec![0f32; used];
        let mut weight = vec![0u16; used];
        let mut rgbw = vec![0u32; if self.pools.color { used } else { 0 }];
        dev.dtoh(&mut tsdf, self.pools.tsdf.ptr)?;
        dev.dtoh(&mut weight, self.pools.weight.ptr)?;
        dev.dtoh(&mut rgbw, self.pools.rgbw.ptr)?;
        dev.sync()?;
        index.clear();
        index.grow(hd.cap);
        pools.grow(hd.cap.max(n));
        pools.tsdf.clear();
        pools.weight.clear();
        pools.rgbw.clear();
        pools.alloc(n);
        for (p, &page) in pages.iter().enumerate() {
            let (dst, src) = (
                p * v..(p + 1) * v,
                page as usize * v..(page as usize + 1) * v,
            );
            pools.tsdf[dst.clone()].copy_from_slice(&tsdf[src.clone()]);
            pools.weight[dst.clone()].copy_from_slice(&weight[src.clone()]);
            if pools.color {
                pools.rgbw[dst].copy_from_slice(&rgbw[src]);
            }
        }
        for (p, &key) in pool_keys.iter().enumerate() {
            index.pool_keys.push(key);
            index.pages.push(p as u32);
            index.modified.push(modified[p]);
            index.mark.push(0);
            tc::hash_insert(
                &mut index.keys,
                &mut index.vals,
                index.log2_cap,
                key,
                p as u32,
            );
        }
        index.sorted.extend(skeys.into_iter().zip(spools));
        index.epoch = hd.epoch;
        Ok(())
    }
}

impl Engine {
    /// At least `n` headers (a submap slot added beyond `max_submaps`; cold).
    fn ensure_headers(&mut self, n: usize) -> Result<(), TsdfError> {
        if n <= self.heads.len() {
            return Ok(());
        }
        // The index first: `heads.len() <= index.headers` must survive a
        // failed relayout (the old index is kept), or later steps index
        // past its buffers.
        if n > self.index.headers {
            let headers = n.max(2 * self.index.headers);
            let stride = self.index.stride;
            self.index
                .relayout(&self.dev, &self.heads, headers, stride)?;
        }
        let cap = self.params.block_capacity;
        self.heads.resize(
            n,
            Head {
                cap,
                ..Head::default()
            },
        );
        Ok(())
    }

    /// Empties header `h` for a new submap (as `BlockIndex::clear`: the
    /// capacity stays).
    fn reset_head(&mut self, h: usize) -> Result<(), TsdfError> {
        let hd = &mut self.heads[h];
        hd.len = 0;
        hd.half = 0;
        hd.epoch = 0;
        self.dev
            .fill(self.index.keys(h), 0xff, self.index.slots * 8)
    }
}

/// Validates a raycast's parameters and pose (as the host types).
fn view_params(
    layout: &TsdfLayout,
    log2: u32,
    color: bool,
    camera: &PinholeCamera,
    size: kornia_image::ImageSize,
    volume_from_camera: &Pose3d,
    params: &RaycastParams,
) -> Result<tc::RaycastParams, TsdfError> {
    let k = check_raycast(camera, size, params, color)?;
    core_params(layout, log2, &k, volume_from_camera, params)
}

// --------------------------------------------------------------- volume ----

/// [`TsdfVolume`] on the GPU: the same layout, parameters, integration,
/// raycast and meshing, with bit-identical results (blocks, voxels, pool
/// order, epochs, counters, meshes, raycast maps). Inputs are host images
/// (uploaded each step); the volume lives in device memory on `stream`.
///
/// Under [`PoolGrowth::Fixed`] the device pools and index are allocated
/// whole at construction; under [`PoolGrowth::Geometric`] they start small
/// and grow on a cold step ([`CudaTsdfVolume::reserve`] pre-sizes them). A
/// warmed integration step allocates nothing on the host and synchronises
/// the stream once (a full mesh twice, an incremental update three times).
pub struct CudaTsdfVolume<const B: usize = 8>
where
    BlockDim<B>: SupportedBlockDim,
{
    e: Engine,
    _block: PhantomData<BlockDim<B>>,
}

impl<const B: usize> CudaTsdfVolume<B>
where
    BlockDim<B>: SupportedBlockDim,
{
    const LOG2: u32 = <BlockDim<B> as SupportedBlockDim>::LOG2;
    const VOXELS: usize = <BlockDim<B> as SupportedBlockDim>::VOXELS;

    /// Creates an empty device volume.
    ///
    /// # Arguments
    ///
    /// * `layout` - Voxel size and truncation.
    /// * `params` - Weight caps, colour, capacity and growth (as
    ///   [`TsdfVolume::new`]).
    /// * `stream` - Stream all work of this volume runs on.
    ///
    /// # Returns
    ///
    /// The volume.
    ///
    /// # Errors
    ///
    /// [`TsdfError::InvalidParameter`] (as [`TsdfVolume::new`]),
    /// [`TsdfError::UnsupportedComputeCapability`] below sm_75,
    /// [`TsdfError::KernelsUnavailable`] with the stub PTX,
    /// [`TsdfError::Cuda`] on a driver error.
    pub fn new(
        layout: TsdfLayout,
        params: TsdfParams,
        stream: Arc<CudaStream>,
    ) -> Result<Self, TsdfError> {
        params.validate(Self::VOXELS)?;
        let cap = params.block_capacity;
        Ok(Self {
            e: Engine::new(stream, Self::LOG2, layout, params, 1, cap, cap)?,
            _block: PhantomData,
        })
    }

    /// Grid geometry.
    pub fn layout(&self) -> &TsdfLayout {
        &self.e.layout
    }

    /// Volume parameters.
    pub fn params(&self) -> &TsdfParams {
        &self.e.params
    }

    /// Allocated blocks.
    pub fn num_blocks(&self) -> usize {
        self.e.heads[0].len
    }

    /// Device bytes held (buffers allocated, not just live blocks).
    pub fn memory_bytes(&self) -> usize {
        self.e.device_bytes()
    }

    /// Reserves `additional_blocks` more blocks (logical and device
    /// capacity), as [`TsdfVolume::reserve`].
    ///
    /// # Arguments
    ///
    /// * `additional_blocks` - Blocks to add to the current count.
    ///
    /// # Returns
    ///
    /// `Ok(())` once the capacity holds them.
    ///
    /// # Errors
    ///
    /// [`TsdfError::PoolFull`] beyond `u32` voxel indices,
    /// [`TsdfError::Cuda`].
    pub fn reserve(&mut self, additional_blocks: usize) -> Result<(), TsdfError> {
        let e = &mut self.e;
        let needed = e.heads[0].len.saturating_add(additional_blocks);
        let limit = max_blocks(Self::VOXELS);
        if needed > limit {
            return Err(TsdfError::PoolFull {
                capacity: limit,
                needed,
            });
        }
        e.dev.bind()?;
        e.heads[0].cap = e.heads[0].cap.max(needed);
        e.pools.cap = e.pools.cap.max(needed);
        e.reserve_phys(needed, needed)
    }

    /// Integrates one step of observations, as [`TsdfVolume::integrate`].
    ///
    /// # Arguments
    ///
    /// * `observations` - The step's observations (host images).
    ///
    /// # Returns
    ///
    /// The step's counters (equal to the CPU's).
    ///
    /// # Errors
    ///
    /// As [`TsdfVolume::integrate`], plus [`TsdfError::Cuda`]; on error the
    /// volume is unchanged (a driver error leaves it unspecified).
    pub fn integrate(
        &mut self,
        observations: &[SensorObservation<'_>],
    ) -> Result<IntegrateStats, TsdfError> {
        check_fp_env()?;
        for o in observations {
            prepare(o.sensor, &o.volume_from_sensor, o.depth, o.color)?;
        }
        let obs = |s: usize| {
            let o = &observations[s];
            prepare(o.sensor, &o.volume_from_sensor, o.depth, o.color)
        };
        let growth = self.e.params.growth;
        self.e.step(
            0,
            &StepInput {
                n: observations.len(),
                obs: &obs,
            },
            growth,
        )
    }

    fn view(
        &mut self,
        camera: &PinholeCamera,
        size: kornia_image::ImageSize,
        volume_from_camera: &Pose3d,
        params: &RaycastParams,
    ) -> Result<(), TsdfError> {
        let rp = view_params(
            &self.e.layout,
            Self::LOG2,
            self.e.params.color,
            camera,
            size,
            volume_from_camera,
            params,
        )?;
        self.e.dev.bind()?;
        self.e.stage_views(&[(rp, 0)])
    }

    /// Raycasts a pinhole view, as [`TsdfVolume::raycast_into`].
    ///
    /// # Arguments
    ///
    /// * `camera` - Intrinsics of the view, without distortion.
    /// * `volume_from_camera` - Camera pose in the volume frame.
    /// * `params` - March range, weight threshold, step cap, colour.
    /// * `maps` - Host output maps (their size is the view's).
    ///
    /// # Returns
    ///
    /// `Ok(())` once the maps hold the view.
    ///
    /// # Errors
    ///
    /// As [`TsdfVolume::raycast_into`], plus [`TsdfError::Cuda`].
    pub fn raycast_into(
        &mut self,
        camera: &PinholeCamera,
        volume_from_camera: &Pose3d,
        params: &RaycastParams,
        maps: &mut RaycastMaps,
    ) -> Result<(), TsdfError> {
        let size = maps.size();
        self.view(camera, size, volume_from_camera, params)?;
        let out = maps_out(size, params, maps)?;
        self.e.raycast(1, size.width, size.height, out)
    }

    /// Raycasts into an RGB-D ICP level, as
    /// [`TsdfVolume::raycast_level_into`].
    ///
    /// # Arguments
    ///
    /// * `volume_from_camera` - Camera pose in the volume frame.
    /// * `params` - March parameters (`colors` is ignored).
    /// * `level` - Level to refill.
    ///
    /// # Returns
    ///
    /// `Ok(())` once the level holds the view.
    ///
    /// # Errors
    ///
    /// As [`CudaTsdfVolume::raycast_into`].
    pub fn raycast_level_into(
        &mut self,
        volume_from_camera: &Pose3d,
        params: &RaycastParams,
        level: &mut RgbdLevel,
    ) -> Result<(), TsdfError> {
        let params = RaycastParams {
            colors: false,
            ..*params
        };
        let size = level.size();
        let camera = level.camera().clone();
        self.view(&camera, size, volume_from_camera, &params)?;
        self.e
            .raycast(1, size.width, size.height, level_out(level)?)
    }

    /// Raycasts every level of an RGB-D ICP pyramid, as
    /// [`TsdfVolume::raycast_pyramid_into`].
    ///
    /// # Arguments
    ///
    /// * `volume_from_camera` - Camera pose in the volume frame.
    /// * `params` - March parameters (`colors` is ignored).
    /// * `pyramid` - Pyramid to refill.
    ///
    /// # Returns
    ///
    /// `Ok(())` once every level holds the view.
    ///
    /// # Errors
    ///
    /// As [`CudaTsdfVolume::raycast_into`].
    pub fn raycast_pyramid_into(
        &mut self,
        volume_from_camera: &Pose3d,
        params: &RaycastParams,
        pyramid: &mut RgbdPyramid,
    ) -> Result<(), TsdfError> {
        for level in pyramid.levels.iter_mut() {
            self.raycast_level_into(volume_from_camera, params, level)?;
        }
        Ok(())
    }

    /// Extracts the full mesh, as [`TsdfVolume::extract_mesh_into`].
    ///
    /// # Arguments
    ///
    /// * `params` - Weight threshold, normals, colours.
    /// * `mesh` - Output mesh (cleared first; its capacity is reused).
    ///
    /// # Returns
    ///
    /// `Ok(())` once `mesh` holds the surface.
    ///
    /// # Errors
    ///
    /// As [`TsdfVolume::extract_mesh_into`], plus [`TsdfError::Cuda`]
    /// (`mesh` is then empty).
    pub fn extract_mesh_into(
        &mut self,
        params: &MeshParams,
        mesh: &mut TriangleMesh,
    ) -> Result<(), TsdfError> {
        check_fp_env()?;
        params.validate(self.e.params.color)?;
        self.e.dev.bind()?;
        mesh.clear();
        self.e.extract(0, params, None, mesh)
    }

    /// Re-meshes the blocks changed since the cache's last update, as
    /// [`TsdfVolume::update_mesh_cache`] (the dirty set and the meshes are
    /// computed on the device; the cache stays on the host).
    ///
    /// # Arguments
    ///
    /// * `params` - Weight threshold, normals, colours.
    /// * `cache` - This volume's cache.
    ///
    /// # Returns
    ///
    /// The update's counters.
    ///
    /// # Errors
    ///
    /// As [`CudaTsdfVolume::extract_mesh_into`].
    pub fn update_mesh_cache(
        &mut self,
        params: &MeshParams,
        cache: &mut BlockMeshCache,
    ) -> Result<MeshUpdateStats, TsdfError> {
        check_fp_env()?;
        params.validate(self.e.params.color)?;
        self.e.dev.bind()?;
        self.e.update_cache(0, params, cache)
    }

    /// Copies the volume into a host volume of the same layout and
    /// parameters (cold: allocates).
    ///
    /// # Arguments
    ///
    /// * `dst` - Host volume; its content is replaced.
    ///
    /// # Returns
    ///
    /// `Ok(())` once `dst` holds the blocks, voxels, pool order and epochs.
    ///
    /// # Errors
    ///
    /// [`TsdfError::InvalidParameter`] for another layout or parameters,
    /// [`TsdfError::Cuda`].
    pub fn download_into(&self, dst: &mut TsdfVolume<B>) -> Result<(), TsdfError> {
        if dst.layout != self.e.layout || dst.params.color != self.e.params.color {
            return Err(TsdfError::InvalidParameter {
                field: "dst",
                value: 0.0,
            });
        }
        self.e.download(0, &mut dst.index, &mut dst.pools)
    }

    /// FNV-1a 64 of the blocks, as [`TsdfVolume::content_hash`]
    /// (downloads; cold).
    ///
    /// # Returns
    ///
    /// The hash.
    ///
    /// # Errors
    ///
    /// [`TsdfError::Cuda`].
    pub fn content_hash(&self) -> Result<u64, TsdfError> {
        let mut index = BlockIndex::with_capacity(1);
        let mut pools = Pools::with_capacity(1, Self::VOXELS, self.e.params.color);
        self.e.download(0, &mut index, &mut pools)?;
        Ok(content_hash(&index, &pools))
    }

    /// Test hook: shrinks the miss set to `slots` (a power of two) so a
    /// step overflows it.
    #[cfg(test)]
    pub(crate) fn shrink_miss_set(&mut self, slots: usize) -> Result<(), TsdfError> {
        self.e.s.miss_slots = slots / 2;
        self.e.grow_miss_set(1)
    }

    /// Test hook: the miss-set size.
    #[cfg(test)]
    pub(crate) fn miss_slots(&self) -> usize {
        self.e.s.miss_slots
    }

    /// Test hook: the engine's `(heads, index headers, stride, dirty list,
    /// hits)` sizes.
    #[cfg(test)]
    pub(crate) fn dims(&self) -> (usize, usize, usize, usize, usize) {
        self.e.dims()
    }

    /// Allocated block keys, ascending (downloads; cold).
    ///
    /// # Arguments
    ///
    /// * `out` - Output keys (cleared first).
    ///
    /// # Errors
    ///
    /// [`TsdfError::Cuda`].
    pub fn block_keys_into(&self, out: &mut Vec<BlockKey>) -> Result<(), TsdfError> {
        let hd = self.e.heads[0];
        let mut keys = vec![0u64; hd.len];
        self.e.dev.bind()?;
        let ix = &self.e.index;
        self.e
            .dev
            .dtoh(&mut keys, ix.pool(&ix.sorted_keys[hd.half], 0))?;
        self.e.dev.sync()?;
        out.clear();
        out.extend(keys.into_iter().map(BlockKey::from_bits));
        Ok(())
    }
}

// --------------------------------------------------------------- submaps ----

/// Host mirror of a device submap slot (its block count; the index itself
/// lives on the device).
#[derive(Debug, Clone, Default)]
pub(crate) struct DevSlot {
    len: usize,
}

impl SlotIndex for DevSlot {
    fn with_capacity(_capacity: usize) -> Self {
        Self::default()
    }

    fn len(&self) -> usize {
        self.len
    }

    fn clear(&mut self) {
        self.len = 0;
    }
}

/// [`super::SubmapTsdf`] on the GPU: the same agents, spawn policy, step
/// order and queries, with bit-identical submaps (blocks, voxels, pool
/// order, epochs, poses), counters, meshes and raycasts. Every submap's
/// index and the voxel arena live on the device; inputs are host images.
///
/// Every submap stays resident: a residency budget
/// ([`SubmapParams::max_resident_bytes`]) is rejected, parking submaps to
/// the host is not implemented.
pub struct CudaSubmapTsdf<const B: usize = 8>
where
    BlockDim<B>: SupportedBlockDim,
{
    book: Book<DevSlot>,
    e: Engine,
    ws: TsdfWorkspace,
    views: Vec<(tc::RaycastParams, u32)>,
    _block: PhantomData<BlockDim<B>>,
}

impl<const B: usize> CudaSubmapTsdf<B>
where
    BlockDim<B>: SupportedBlockDim,
{
    const LOG2: u32 = <BlockDim<B> as SupportedBlockDim>::LOG2;
    const VOXELS: usize = <BlockDim<B> as SupportedBlockDim>::VOXELS;

    /// Creates an empty device map with `max_submaps` slots.
    ///
    /// # Arguments
    ///
    /// * `params` - Layout, voxel parameters and capacities (as
    ///   [`super::SubmapTsdf::new`]); `max_resident_bytes` must be `None`.
    /// * `stream` - Stream all work of this map runs on.
    ///
    /// # Returns
    ///
    /// The map.
    ///
    /// # Errors
    ///
    /// [`TsdfError::InvalidParameter`] (as the host type, or a residency
    /// budget), [`TsdfError::UnsupportedComputeCapability`],
    /// [`TsdfError::KernelsUnavailable`], [`TsdfError::Cuda`].
    pub fn new(params: SubmapParams, stream: Arc<CudaStream>) -> Result<Self, TsdfError> {
        if let Some(b) = params.max_resident_bytes {
            return Err(TsdfError::InvalidParameter {
                field: "max_resident_bytes",
                value: b as f64,
            });
        }
        let book = Book::new(params, Self::VOXELS)?;
        let p = &book.params;
        let e = Engine::new(
            stream,
            Self::LOG2,
            p.layout,
            p.volume.clone(),
            p.max_submaps,
            p.volume.block_capacity,
            p.arena_blocks,
        )?;
        Ok(Self {
            views: Vec::with_capacity(p.max_submaps),
            book,
            e,
            ws: TsdfWorkspace::default(),
            _block: PhantomData,
        })
    }

    /// Map parameters.
    pub fn params(&self) -> &SubmapParams {
        &self.book.params
    }

    /// Registers an agent, as [`super::SubmapTsdf::add_agent`].
    ///
    /// # Arguments
    ///
    /// * `agent` - New agent id.
    /// * `rig` - Its sensors.
    /// * `spawn` - Its spawn policy.
    ///
    /// # Returns
    ///
    /// `Ok(())` once registered.
    ///
    /// # Errors
    ///
    /// [`TsdfError::DuplicateAgent`], [`TsdfError::InvalidParameter`].
    pub fn add_agent(
        &mut self,
        agent: AgentId,
        rig: SensorRig,
        spawn: SpawnPolicy,
    ) -> Result<(), TsdfError> {
        self.book.add_agent(agent, rig, spawn)
    }

    /// Applies one validated step.
    fn apply(&mut self, step: &AgentStep<'_>) -> Result<StepReport, TsdfError> {
        let plan = self.book.plan(step)?;
        self.e.dev.bind()?;
        self.e.ensure_headers(self.book.headers.len())?;
        if plan.spawn {
            self.e.reset_head(plan.slot)?;
        }
        let mut order = std::mem::take(&mut self.ws.frame_order);
        Book::<DevSlot>::order_frames(step, &mut order);
        let growth = self.book.params.volume.growth;
        let rig = &self.book.agents[plan.agent].rig;
        let obs = |s: usize| observation(rig, &plan, step, &order, s);
        let res = self.e.step(
            plan.slot,
            &StepInput {
                n: step.frames.len(),
                obs: &obs,
            },
            growth,
        );
        self.ws.frame_order = order;
        let stats = res?;
        self.book.headers[plan.slot].index.len = self.e.heads[plan.slot].len;
        Ok(self.book.commit(&plan, step, stats))
    }

    /// Integrates one agent step, as [`super::SubmapTsdf::integrate`].
    ///
    /// # Arguments
    ///
    /// * `step` - The step (host images).
    ///
    /// # Returns
    ///
    /// The submap used, whether it was spawned, and the counters.
    ///
    /// # Errors
    ///
    /// As [`super::SubmapTsdf::integrate`], plus [`TsdfError::Cuda`].
    pub fn integrate(&mut self, step: &AgentStep<'_>) -> Result<StepReport, TsdfError> {
        check_fp_env()?;
        self.book.validate_step(step)?;
        self.apply(step)
    }

    /// Integrates several agents' steps in `(stamp, agent)` order, as
    /// [`super::SubmapTsdf::integrate_batch`].
    ///
    /// # Arguments
    ///
    /// * `steps` - Steps of any agents, any order.
    ///
    /// # Returns
    ///
    /// The number of steps and spawns, and the summed counters.
    ///
    /// # Errors
    ///
    /// As [`super::SubmapTsdf::integrate_batch`].
    pub fn integrate_batch(&mut self, steps: &[AgentStep<'_>]) -> Result<BatchReport, TsdfError> {
        check_fp_env()?;
        let mut ws = std::mem::take(&mut self.ws);
        let res = Book::batch(
            self,
            |m| &mut m.book,
            &mut ws,
            steps,
            |m, ws, s| {
                std::mem::swap(&mut m.ws, ws);
                let r = m.apply(s);
                std::mem::swap(&mut m.ws, ws);
                r
            },
        );
        self.ws = ws;
        res
    }

    /// Moves submaps, as [`super::SubmapTsdf::set_submap_poses`].
    ///
    /// # Arguments
    ///
    /// * `poses` - New world poses of some submaps.
    ///
    /// # Returns
    ///
    /// `Ok(())` once applied.
    ///
    /// # Errors
    ///
    /// [`TsdfError::UnknownSubmap`] or [`TsdfError::InvalidTransform`].
    pub fn set_submap_poses(&mut self, poses: &[SubmapPose]) -> Result<(), TsdfError> {
        self.book.set_submap_poses(poses)
    }

    /// Submap ids, ascending.
    pub fn submap_ids(&self) -> impl Iterator<Item = SubmapId> + '_ {
        self.book.submap_ids()
    }

    /// The agent's active submap.
    ///
    /// # Arguments
    ///
    /// * `agent` - Agent id.
    ///
    /// # Returns
    ///
    /// Its active submap, `None` before its first step or if unknown.
    pub fn active_submap(&self, agent: AgentId) -> Option<SubmapId> {
        self.book.active_submap(agent)
    }

    /// A submap's world pose.
    ///
    /// # Arguments
    ///
    /// * `id` - Submap id.
    ///
    /// # Returns
    ///
    /// `world_from_submap`, `None` for an unknown id.
    pub fn submap_pose(&self, id: SubmapId) -> Option<Pose3d> {
        self.book
            .slot_of(id)
            .map(|s| self.book.headers[s].world_from_submap)
    }

    /// Blocks of a submap.
    ///
    /// # Arguments
    ///
    /// * `id` - Submap id.
    ///
    /// # Returns
    ///
    /// Its allocated block count, `None` for an unknown id.
    pub fn submap_blocks(&self, id: SubmapId) -> Option<usize> {
        self.book
            .slot_of(id)
            .map(|s| self.book.headers[s].index.len)
    }

    /// Extracts the selected submaps' meshes in the world frame, as
    /// [`super::SubmapTsdf::extract_mesh_into`].
    ///
    /// # Arguments
    ///
    /// * `filter` - Submaps to mesh.
    /// * `params` - Weight threshold, normals, colours.
    /// * `mesh` - Output mesh (cleared first).
    ///
    /// # Returns
    ///
    /// `Ok(())` once `mesh` holds the surface.
    ///
    /// # Errors
    ///
    /// As [`super::SubmapTsdf::extract_mesh_into`], plus
    /// [`TsdfError::Cuda`] (`mesh` is then empty).
    pub fn extract_mesh_into(
        &mut self,
        filter: SubmapFilter<'_>,
        params: &MeshParams,
        mesh: &mut TriangleMesh,
    ) -> Result<(), TsdfError> {
        check_fp_env()?;
        params.validate(self.e.params.color)?;
        self.book.check_filter(&filter)?;
        self.e.dev.bind()?;
        mesh.clear();
        for i in 0..self.book.live.len() {
            let slot = self.book.live[i] as usize;
            let h = &self.book.headers[slot];
            if filter.matches(h.id) {
                let xf = rigid_from_pose(&h.world_from_submap)?;
                if let Err(e) = self.e.extract(slot, params, Some(&xf), mesh) {
                    mesh.clear();
                    return Err(e);
                }
            }
        }
        Ok(())
    }

    /// Re-meshes the changed blocks of every submap into `cache`, as
    /// [`super::SubmapTsdf::update_mesh_cache`].
    ///
    /// # Arguments
    ///
    /// * `params` - Weight threshold, normals, colours.
    /// * `cache` - This map's cache.
    ///
    /// # Returns
    ///
    /// Summed counters over the submaps.
    ///
    /// # Errors
    ///
    /// [`TsdfError::InvalidParameter`], [`TsdfError::FpEnvironment`],
    /// [`TsdfError::Cuda`].
    pub fn update_mesh_cache(
        &mut self,
        params: &MeshParams,
        cache: &mut SubmapMeshCache,
    ) -> Result<MeshUpdateStats, TsdfError> {
        check_fp_env()?;
        params.validate(self.e.params.color)?;
        self.e.dev.bind()?;
        let mut stats = MeshUpdateStats::default();
        for i in 0..self.book.live.len() {
            let s = self.book.live[i] as usize;
            let c = cache.slot_cache(s, self.book.headers[s].id);
            let st = self.e.update_cache(s, params, c)?;
            stats.accumulate(&st);
        }
        Ok(stats)
    }

    /// Writes `cache`'s meshes of the selected submaps in the world frame,
    /// as [`SubmapMeshCache::write_into`] for a host map.
    ///
    /// # Arguments
    ///
    /// * `cache` - This map's cache.
    /// * `filter` - Submaps to write.
    /// * `mesh` - Output mesh (cleared first).
    ///
    /// # Returns
    ///
    /// `Ok(())` once `mesh` holds the meshes.
    ///
    /// # Errors
    ///
    /// As [`SubmapMeshCache::write_into`].
    pub fn write_mesh_cache_into(
        &self,
        cache: &SubmapMeshCache,
        filter: SubmapFilter<'_>,
        mesh: &mut TriangleMesh,
    ) -> Result<(), TsdfError> {
        cache.write_book(&self.book, filter, mesh)
    }

    /// Stages the views of the selected submaps (ascending id).
    fn stage_rays(
        &mut self,
        camera: &PinholeCamera,
        size: kornia_image::ImageSize,
        world_from_camera: &Pose3d,
        filter: &SubmapFilter<'_>,
        params: &RaycastParams,
    ) -> Result<usize, TsdfError> {
        let k = check_raycast(camera, size, params, self.e.params.color)?;
        self.book.check_view(world_from_camera, filter)?;
        self.views.clear();
        for &slot in &self.book.live {
            let h = &self.book.headers[slot as usize];
            if !filter.matches(h.id) {
                continue;
            }
            let submap_from_camera = h.world_from_submap.inverse().compose(world_from_camera);
            let rp = core_params(
                &self.book.params.layout,
                Self::LOG2,
                &k,
                &submap_from_camera,
                params,
            )?;
            self.views.push((rp, slot));
        }
        self.e.dev.bind()?;
        self.e.stage_views(&self.views)?;
        Ok(self.views.len())
    }

    /// Raycasts the selected submaps (nearest hit wins), as
    /// [`super::SubmapTsdf::raycast_into`].
    ///
    /// # Arguments
    ///
    /// * `camera` - Intrinsics of the view, without distortion.
    /// * `world_from_camera` - Camera pose in the world frame.
    /// * `filter` - Submaps to raycast.
    /// * `params` - March parameters.
    /// * `maps` - Host output maps.
    ///
    /// # Returns
    ///
    /// `Ok(())` once the maps hold the view.
    ///
    /// # Errors
    ///
    /// As [`super::SubmapTsdf::raycast_into`], plus [`TsdfError::Cuda`].
    pub fn raycast_into(
        &mut self,
        camera: &PinholeCamera,
        world_from_camera: &Pose3d,
        filter: SubmapFilter<'_>,
        params: &RaycastParams,
        maps: &mut RaycastMaps,
    ) -> Result<(), TsdfError> {
        let size = maps.size();
        let n = self.stage_rays(camera, size, world_from_camera, &filter, params)?;
        let out = maps_out(size, params, maps)?;
        self.e.raycast(n, size.width, size.height, out)
    }

    /// Raycasts every level of an RGB-D ICP pyramid, as
    /// [`super::SubmapTsdf::raycast_pyramid_into`].
    ///
    /// # Arguments
    ///
    /// * `world_from_camera` - Camera pose in the world frame.
    /// * `filter` - Submaps to raycast.
    /// * `params` - March parameters (`colors` is ignored).
    /// * `pyramid` - Pyramid to refill.
    ///
    /// # Returns
    ///
    /// `Ok(())` once every level holds the view.
    ///
    /// # Errors
    ///
    /// As [`CudaSubmapTsdf::raycast_into`].
    pub fn raycast_pyramid_into(
        &mut self,
        world_from_camera: &Pose3d,
        filter: SubmapFilter<'_>,
        params: &RaycastParams,
        pyramid: &mut RgbdPyramid,
    ) -> Result<(), TsdfError> {
        let params = RaycastParams {
            colors: false,
            ..*params
        };
        for level in pyramid.levels.iter_mut() {
            let size = level.size();
            let camera = level.camera().clone();
            let n = self.stage_rays(&camera, size, world_from_camera, &filter, &params)?;
            self.e
                .raycast(n, size.width, size.height, level_out(level)?)?;
        }
        Ok(())
    }

    /// Memory held: device bytes of everything (or one agent's voxel pages)
    /// and allocated blocks.
    ///
    /// # Arguments
    ///
    /// * `agent` - One agent's submaps, or `None` for the whole map.
    ///
    /// # Returns
    ///
    /// Host bytes (0), device bytes and blocks.
    pub fn memory_bytes(&self, agent: Option<AgentId>) -> MemoryReport {
        let page = self.e.voxels() * if self.e.params.color { 10 } else { 6 };
        let mut blocks = 0;
        for &slot in &self.book.live {
            let h = &self.book.headers[slot as usize];
            if agent.is_none_or(|a| a == h.id.agent) {
                blocks += h.index.len;
            }
        }
        MemoryReport {
            host_bytes: 0,
            device_bytes: if agent.is_some() {
                blocks * page
            } else {
                self.e.device_bytes()
            },
            blocks,
        }
    }

    /// FNV-1a 64 of one submap's blocks, as
    /// [`super::SubmapTsdf::content_hash`] (downloads; cold).
    ///
    /// # Arguments
    ///
    /// * `id` - Submap id.
    ///
    /// # Returns
    ///
    /// The hash, `None` for an unknown id.
    ///
    /// # Errors
    ///
    /// [`TsdfError::Cuda`].
    pub fn content_hash(&self, id: SubmapId) -> Result<Option<u64>, TsdfError> {
        let Some(s) = self.book.slot_of(id) else {
            return Ok(None);
        };
        let mut index = BlockIndex::with_capacity(1);
        let mut pools = Pools::with_capacity(1, Self::VOXELS, self.e.params.color);
        self.e.download(s, &mut index, &mut pools)?;
        Ok(Some(content_hash(&index, &pools)))
    }

    /// Test hook: grows the header table to `n` headers.
    #[cfg(test)]
    pub(crate) fn ensure_headers(&mut self, n: usize) -> Result<(), TsdfError> {
        self.e.dev.bind()?;
        self.e.ensure_headers(n)
    }

    /// Test hook: the engine's `(heads, index headers, stride, dirty list,
    /// hits)` sizes.
    #[cfg(test)]
    pub(crate) fn dims(&self) -> (usize, usize, usize, usize, usize) {
        self.e.dims()
    }

    /// Test hook: a submap's index and voxels on the host.
    #[cfg(test)]
    pub(crate) fn download_submap(&self, id: SubmapId) -> Option<(BlockIndex, Pools)> {
        let s = self.book.slot_of(id)?;
        let mut index = BlockIndex::with_capacity(1);
        let mut pools = Pools::with_capacity(1, Self::VOXELS, self.e.params.color);
        self.e.download(s, &mut index, &mut pools).ok()?;
        Some((index, pools))
    }
}
