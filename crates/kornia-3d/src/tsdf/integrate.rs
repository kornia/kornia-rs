//! One integration step: validation, allocation over the union of the
//! observations, then one update pass per observation in fold order.

use kornia_oxide_kernels::tsdf_core::{self as tc, RawDepth};

use super::alloc::{allocate, prepare, touch, Obs, ObsKind};
use super::sensor::{DepthRef, SensorObservation};
use super::volume::{BlockIndex, Pools};
use super::workspace::TsdfWorkspace;
use super::{
    check_fp_env, BlockDim, IntegrateStats, StdOps, SupportedBlockDim, TsdfError, TsdfLayout,
    TsdfParams, TsdfVolume,
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
    for s in 0..n {
        let o = obs(s)?;
        let r = touch(&o, &grid, t.index, ws, &mut stats);
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
        let pools_v = ws.visible[r.visible.0..r.visible.1].iter();
        let pools_n = ws.visible_new[r.new.0..r.new.1].iter();
        for &p in pools_v.chain(pools_n) {
            stats.updated_voxels += integrate_block(&o, &grid, t.params, t.index, t.pools, p);
        }
    }
    Ok(stats)
}

/// Updates the voxels of local pool `p` from one observation; returns the
/// number of voxels updated.
fn integrate_block(
    o: &Obs<'_>,
    grid: &tc::Grid,
    params: &TsdfParams,
    index: &BlockIndex,
    pools: &mut Pools,
    p: u32,
) -> u64 {
    let key = index.pool_keys[p as usize];
    let base = index.pages[p as usize] as usize * pools.voxels;
    let color = if pools.color { o.color } else { None };
    match o.depth {
        DepthRef::U16(d) => block_pass(
            o,
            d.image().as_slice(),
            color,
            grid,
            params,
            pools,
            key,
            base,
        ),
        DepthRef::F32(d) => block_pass(
            o,
            d.image().as_slice(),
            color,
            grid,
            params,
            pools,
            key,
            base,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn block_pass<D: RawDepth>(
    o: &Obs<'_>,
    depth: &[D],
    color: Option<&[u8]>,
    grid: &tc::Grid,
    params: &TsdfParams,
    pools: &mut Pools,
    key: u64,
    base: usize,
) -> u64 {
    match &o.kind {
        ObsKind::Pinhole(p) => voxel_loop(grid, params, pools, key, base, color, |c| {
            tc::pinhole_sdf::<StdOps, D>(p, depth, c)
        }),
        ObsKind::Spherical(p, lidar) => {
            let bnd = lidar.boundaries();
            voxel_loop(grid, params, pools, key, base, None, |c| {
                tc::spherical_sdf::<StdOps, D>(p, bnd, depth, c)
            })
        }
    }
}

/// The per-voxel update over one block, `x` fastest.
#[inline(always)]
fn voxel_loop(
    grid: &tc::Grid,
    params: &TsdfParams,
    pools: &mut Pools,
    key: u64,
    base: usize,
    color: Option<&[u8]>,
    sdf: impl Fn([f32; 3]) -> Option<(f32, u32)>,
) -> u64 {
    let b = 1i32 << grid.log2;
    let bc = tc::unpack_block_key(key);
    let g0 = [bc[0] * b, bc[1] * b, bc[2] * b];
    let mut updated = 0u64;
    let mut i = base;
    for lz in 0..b {
        for ly in 0..b {
            for lx in 0..b {
                let g = [g0[0] + lx, g0[1] + ly, g0[2] + lz];
                let c = tc::voxel_centre::<StdOps>(g, grid.voxel_size, grid.half_voxel);
                if let Some((d, idx)) = sdf(c) {
                    if let Some(tsd) = tc::truncate(d, grid.trunc, grid.inv_trunc) {
                        let (t, w) = tc::update_tsdf::<StdOps>(
                            pools.tsdf[i],
                            pools.weight[i],
                            tsd,
                            params.max_weight,
                        );
                        pools.tsdf[i] = t;
                        pools.weight[i] = w;
                        if let Some(rgb) = color {
                            let k = 3 * idx as usize;
                            pools.rgbw[i] = tc::update_color::<StdOps>(
                                pools.rgbw[i],
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
