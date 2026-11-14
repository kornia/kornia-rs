//! One integration step: validation, allocation over the union of the
//! observations, then one update pass per observation in fold order. The
//! `Scalar` kernel walks an observation's blocks serially; `Simd` runs them
//! as parallel tasks (rayon) inside a pool and serially outside one, NEON
//! per block on aarch64 and scalar elsewhere. A block is written by one task only, so every kernel stores
//! the same bits.

use std::marker::PhantomData;

use kornia_oxide_kernels::tsdf_core::{self as tc, RawDepth};
use rayon::prelude::*;

use super::alloc::{allocate, prepare, touch, touch_par, Obs, ObsKind};
use super::sensor::{DepthRef, SensorObservation};
use super::volume::{BlockIndex, Pools};
use super::workspace::TsdfWorkspace;
use super::{
    check_fp_env, BlockDim, IntegrateStats, StdOps, SupportedBlockDim, TsdfError, TsdfKernel,
    TsdfLayout, TsdfParams, TsdfVolume,
};

/// The storage one step writes.
pub(crate) struct StepTarget<'a> {
    pub(crate) index: &'a mut BlockIndex,
    pub(crate) pools: &'a mut Pools,
    pub(crate) layout: &'a TsdfLayout,
    pub(crate) params: &'a TsdfParams,
    pub(crate) log2: u32,
}

/// Runs one step over `n` observations; `obs(s)` yields observation `s`
/// in fold order (already validated by the caller). All-or-nothing: the
/// only failure (`PoolFull`) happens before any observable mutation.
pub(crate) fn integrate_step<'o>(
    t: StepTarget<'_>,
    ws: &mut TsdfWorkspace,
    n: usize,
    obs: &dyn Fn(usize) -> Result<Obs<'o>, TsdfError>,
) -> Result<IntegrateStats, TsdfError> {
    let grid = t.layout.grid(t.log2);
    let mut stats = IntegrateStats::default();
    ws.begin_step();
    // parallel only inside a pool of >1 thread (same output either way)
    let par = ws.kernel() == TsdfKernel::Simd
        && rayon::current_thread_index().is_some()
        && rayon::current_num_threads() > 1;
    for s in 0..n {
        let o = obs(s)?;
        let r = if par {
            touch_par(&o, &grid, t.index, ws, &mut stats)
        } else {
            touch(&o, &grid, t.index, ws, &mut stats)
        };
        ws.ranges.push(r);
    }
    stats.new_blocks = allocate(t.index, t.pools, t.params.growth, ws)?;
    t.index.epoch += 1;
    let epoch = t.index.epoch;
    for &p in ws.visible.iter().chain(ws.visible_new.iter()) {
        let m = &mut t.index.modified[p as usize];
        if *m != epoch {
            *m = epoch;
            stats.touched_blocks += 1;
        }
    }
    for s in 0..n {
        let o = obs(s)?;
        let r = ws.ranges[s];
        let pass = Pass {
            obs: &o,
            grid: &grid,
            params: t.params,
            index: t.index,
            hits: &ws.visible[r.visible.0..r.visible.1],
            new: &ws.visible_new[r.new.0..r.new.1],
        };
        stats.updated_voxels += match ws.kernel() {
            TsdfKernel::Scalar => pass.serial(t.pools, false),
            // outside a pool rayon's injector would allocate: run serially
            TsdfKernel::Simd if rayon::current_thread_index().is_none() => {
                pass.serial(t.pools, true)
            }
            TsdfKernel::Simd => pass.parallel(t.pools),
        };
    }
    Ok(stats)
}

/// The `B³` voxels of one block; `rgbw` iff the pools store colour.
pub(crate) struct PageMut<'a> {
    pub(crate) tsdf: &'a mut [f32],
    pub(crate) weight: &'a mut [u16],
    pub(crate) rgbw: Option<&'a mut [u32]>,
}

/// The pools as raw pages, so that parallel tasks each borrow one page.
struct PagesMut<'a> {
    tsdf: *mut f32,
    weight: *mut u16,
    rgbw: Option<*mut u32>,
    voxels: usize,
    pages: usize,
    _pools: PhantomData<&'a mut Pools>,
}

// SAFETY: `PagesMut` is `&mut Pools` split by page: it hands out a page only
// to a caller that holds no other borrow of it (`page`), and the pools are
// plain numbers.
#[allow(unsafe_code)]
unsafe impl Send for PagesMut<'_> {}
// SAFETY: as above; shared use only reaches disjoint pages.
#[allow(unsafe_code)]
unsafe impl Sync for PagesMut<'_> {}

impl<'a> PagesMut<'a> {
    fn new(pools: &'a mut Pools) -> Self {
        Self {
            tsdf: pools.tsdf.as_mut_ptr(),
            weight: pools.weight.as_mut_ptr(),
            rgbw: pools.color.then_some(pools.rgbw.as_mut_ptr()),
            voxels: pools.voxels,
            pages: pools.pages(),
            _pools: PhantomData,
        }
    }

    /// Page `page`, `None` past the allocated pages.
    ///
    /// # Safety
    ///
    /// No other borrow of page `page` may be alive while the result is.
    #[allow(unsafe_code)]
    unsafe fn page(&self, page: u32) -> Option<PageMut<'a>> {
        let page = page as usize;
        if page >= self.pages {
            return None;
        }
        let (base, v) = (page * self.voxels, self.voxels);
        // SAFETY: every pool holds `pages * voxels` entries, so
        // `base..base + v` is in bounds; the caller guarantees it is not
        // borrowed elsewhere.
        unsafe {
            Some(PageMut {
                tsdf: std::slice::from_raw_parts_mut(self.tsdf.add(base), v),
                weight: std::slice::from_raw_parts_mut(self.weight.add(base), v),
                rgbw: self
                    .rgbw
                    .map(|r| std::slice::from_raw_parts_mut(r.add(base), v)),
            })
        }
    }
}

/// One observation's update pass over the blocks it touched (hit pools,
/// then new pools, each pool once).
struct Pass<'a, 'o> {
    obs: &'a Obs<'o>,
    grid: &'a tc::Grid,
    params: &'a TsdfParams,
    index: &'a BlockIndex,
    hits: &'a [u32],
    new: &'a [u32],
}

impl Pass<'_, '_> {
    fn serial(&self, pools: &mut Pools, simd: bool) -> u64 {
        let mut updated = 0;
        for &p in self.hits.iter().chain(self.new) {
            let page = pools.page_mut(self.index.pages[p as usize]);
            updated += self.block(p, page, simd);
        }
        updated
    }

    /// One task per block; the counts are integers, so their sum is exact.
    #[allow(unsafe_code)]
    fn parallel(&self, pools: &mut Pools) -> u64 {
        let pages = PagesMut::new(pools);
        let task = |&p: &u32| {
            // SAFETY: `touch` and `allocate` list each pool of an
            // observation once (observation mark), hits and new pools are
            // disjoint, and the page table is injective (each new pool gets
            // a freshly appended page), so no two tasks share a page.
            let page = unsafe { pages.page(self.index.pages[p as usize]) };
            // a page past the table is corruption: panic like `serial`
            self.block(
                p,
                page.expect("page index within the allocated pages"),
                true,
            )
        };
        self.hits
            .par_iter()
            .chain(self.new.par_iter())
            .map(task)
            .sum()
    }

    /// Updates the voxels of local pool `p`; returns how many it updated.
    #[inline(always)]
    fn block(&self, p: u32, page: PageMut<'_>, simd: bool) -> u64 {
        let b = Block {
            kind: &self.obs.kind,
            color: self.obs.color.filter(|_| page.rgbw.is_some()),
            grid: self.grid,
            params: self.params,
            key: self.index.pool_keys[p as usize],
        };
        match self.obs.depth {
            DepthRef::U16(d) => block_pass(&b, d.image().as_slice(), page, simd),
            DepthRef::F32(d) => block_pass(&b, d.image().as_slice(), page, simd),
        }
    }
}

/// One block of one observation, without its samples and storage.
pub(crate) struct Block<'a, 'o> {
    pub(crate) kind: &'a ObsKind<'o>,
    /// Interleaved RGB to fold in (pinhole with colour stored).
    pub(crate) color: Option<&'a [u8]>,
    pub(crate) grid: &'a tc::Grid,
    pub(crate) params: &'a TsdfParams,
    pub(crate) key: u64,
}

#[inline(always)]
fn block_pass<D: RawDepth>(b: &Block<'_, '_>, depth: &[D], page: PageMut<'_>, simd: bool) -> u64 {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    if simd {
        return super::integrate_neon::block(b, depth, page);
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    let _ = simd;
    match b.kind {
        ObsKind::Pinhole(p) => voxel_loop(b, page, b.color, |c| {
            tc::pinhole_sdf::<StdOps, D>(p, depth, c)
        }),
        ObsKind::Spherical(p, lidar) => {
            let bnd = lidar.boundaries();
            voxel_loop(b, page, None, |c| {
                tc::spherical_sdf::<StdOps, D>(p, bnd, depth, c)
            })
        }
    }
}

/// The scalar per-voxel update over one block, `x` fastest.
#[inline(always)]
fn voxel_loop(
    b: &Block<'_, '_>,
    page: PageMut<'_>,
    color: Option<&[u8]>,
    sdf: impl Fn([f32; 3]) -> Option<(f32, u32)>,
) -> u64 {
    let (grid, params) = (b.grid, b.params);
    let e = 1i32 << grid.log2;
    let bc = tc::unpack_block_key(b.key);
    let g0 = [bc[0] * e, bc[1] * e, bc[2] * e];
    let PageMut {
        tsdf,
        weight,
        mut rgbw,
    } = page;
    let mut updated = 0u64;
    let mut i = 0;
    for lz in 0..e {
        for ly in 0..e {
            for lx in 0..e {
                let g = [g0[0] + lx, g0[1] + ly, g0[2] + lz];
                let c = tc::voxel_centre::<StdOps>(g, grid.voxel_size, grid.half_voxel);
                if let Some((d, idx)) = sdf(c) {
                    if let Some(tsd) = tc::truncate(d, grid.trunc, grid.inv_trunc) {
                        let (t, w) =
                            tc::update_tsdf::<StdOps>(tsdf[i], weight[i], tsd, params.max_weight);
                        tsdf[i] = t;
                        weight[i] = w;
                        if let (Some(rgb), Some(rgbw)) = (color, rgbw.as_deref_mut()) {
                            let k = 3 * idx as usize;
                            rgbw[i] = tc::update_color::<StdOps>(
                                rgbw[i],
                                [rgb[k], rgb[k + 1], rgb[k + 2]],
                                params.max_color_weight,
                            );
                        }
                        updated += 1;
                    }
                }
                i += 1;
            }
        }
    }
    updated
}

impl<const B: usize> TsdfVolume<B>
where
    BlockDim<B>: SupportedBlockDim,
{
    /// Integrates one step of observations.
    ///
    /// Allocation covers the union of the observations: every block within
    /// `± truncation` of a measured point is allocated, new blocks in
    /// ascending key order. Then each observation, in slice order, updates
    /// the voxels of the blocks it touched: `sdf = d - z` (pinhole) or
    /// `rho - |p|` (spherical), skipped below `-truncation`, normalised and
    /// clamped to `[-1, 1]`, folded in with the running mean
    /// `t += (tsd - t) / (w + 1)`, `w = min(w + 1, max_weight)`. Colour (when
    /// enabled and given) is averaged the same way with its own cap.
    ///
    /// # Arguments
    ///
    /// * `ws` - Reused scratch.
    /// * `observations` - The step's observations (pose in this volume's
    ///   frame).
    ///
    /// # Returns
    ///
    /// The step's counters.
    ///
    /// # Errors
    ///
    /// Validation errors ([`TsdfError::InvalidTransform`],
    /// [`TsdfError::SizeMismatch`], [`TsdfError::ColorOnRangeSensor`],
    /// [`TsdfError::FpEnvironment`], [`TsdfError::Image`]) and
    /// [`TsdfError::PoolFull`]; on error the volume is unchanged.
    ///
    /// # Example
    ///
    /// ```
    /// use kornia_3d::camera::PinholeCamera;
    /// use kornia_3d::depth::DepthImage;
    /// use kornia_3d::pose::Pose3d;
    /// use kornia_3d::tsdf::*;
    /// use kornia_image::ImageSize;
    ///
    /// let size = ImageSize { width: 32, height: 24 };
    /// let camera = PinholeCamera { fx: 30.0, fy: 30.0, cx: 15.5, cy: 11.5, ..PinholeCamera::IDENTITY };
    /// let sensor = RangeSensor::from(PinholeSensor::new(&camera, size, DepthRange::new(0.1, 5.0)?)?);
    /// let depth = DepthImage::from_size_val(size, 1000u16, 1000.0)?;
    /// let mut volume = TsdfVolume::<8>::new(TsdfLayout::new(0.05, 4.0)?, TsdfParams { block_capacity: 256, ..Default::default() })?;
    /// let mut ws = TsdfWorkspace::new(TsdfKernel::Scalar);
    /// let obs = SensorObservation { sensor: &sensor, volume_from_sensor: Pose3d::IDENTITY, depth: (&depth).into(), color: None };
    /// let stats = volume.integrate(&mut ws, &[obs])?;
    /// assert!(stats.updated_voxels > 0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn integrate(
        &mut self,
        ws: &mut TsdfWorkspace,
        observations: &[SensorObservation<'_>],
    ) -> Result<IntegrateStats, TsdfError> {
        check_fp_env()?;
        ws.check_workers_fp_env()?;
        for o in observations {
            prepare(o.sensor, &o.volume_from_sensor, o.depth, o.color)?;
        }
        let obs = |s: usize| {
            let o = &observations[s];
            prepare(o.sensor, &o.volume_from_sensor, o.depth, o.color)
        };
        let target = StepTarget {
            index: &mut self.index,
            pools: &mut self.pools,
            layout: &self.layout,
            params: &self.params,
            log2: Self::LOG2,
        };
        integrate_step(target, ws, observations.len(), &obs)
    }
}
