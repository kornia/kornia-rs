//! Step allocation: touch every block within the truncation band of each
//! observation's measurements, look each up first, and sort only the
//! misses; new blocks get pool indices in ascending key order.

use kornia_image::Image;
use kornia_oxide_kernels::tsdf_core::{self as tc, RawDepth};

use super::sensor::{check_observation, DepthRef, RangeSensor, SphericalLidar};
use super::volume::{ensure_capacity, insert_blocks, BlockIndex, Pools};
use super::workspace::{ObsRange, TsdfWorkspace, MISS_FILTER};
use super::{check_pose, rigid_from_pose, IntegrateStats, PoolGrowth, StdOps, TsdfError};
use crate::pose::Pose3d;

/// Projection constants of one observation.
#[derive(Clone, Copy)]
pub(crate) enum ObsKind<'a> {
    Pinhole(tc::PinholeParams),
    Spherical(tc::SphericalParams, &'a SphericalLidar),
}

/// One validated observation of a step.
#[derive(Clone, Copy)]
pub(crate) struct Obs<'a> {
    pub(crate) kind: ObsKind<'a>,
    pub(crate) depth: DepthRef<'a>,
    /// Interleaved RGB of the depth's size (pinhole only).
    pub(crate) color: Option<&'a [u8]>,
}

/// Validates an observation and derives its constants: the pose in `f64`,
/// each `f32` entry rounded once (`sensor_from_volume` is the `f64`
/// inverse of `volume_from_sensor`).
pub(crate) fn prepare<'a>(
    sensor: &'a RangeSensor,
    volume_from_sensor: &Pose3d,
    depth: DepthRef<'a>,
    color: Option<&'a Image<u8, 3>>,
) -> Result<Obs<'a>, TsdfError> {
    check_pose(volume_from_sensor)?;
    check_observation(sensor, &depth, color)?;
    let vfs = rigid_from_pose(volume_from_sensor)?;
    let sfv = rigid_from_pose(&volume_from_sensor.inverse())?;
    let mpu = depth.metres_per_unit();
    let kind = match sensor {
        RangeSensor::Pinhole(s) => ObsKind::Pinhole(s.params(sfv, vfs, mpu)),
        RangeSensor::Spherical(s) => ObsKind::Spherical(s.params(sfv, vfs, mpu), s),
    };
    Ok(Obs {
        kind,
        depth,
        color: color.map(|c| c.as_slice()),
    })
}

/// Touch pass of one observation: hit pools go to `ws.visible` (once each,
/// via `mark`), misses to `ws.obs_misses`.
pub(crate) fn touch(
    obs: &Obs<'_>,
    grid: &tc::Grid,
    index: &mut BlockIndex,
    ws: &mut TsdfWorkspace,
    stats: &mut IntegrateStats,
) -> ObsRange {
    let mark = index.next_mark();
    let v0 = ws.visible.len();
    let m0 = ws.obs_misses.len();
    ws.miss_filter.fill(tc::EMPTY_KEY);
    let mut t = Toucher {
        grid,
        index,
        ws,
        mark,
        prev: None,
        stats,
    };
    match obs.depth {
        DepthRef::U16(d) => t.pass(&obs.kind, d.image().as_slice()),
        DepthRef::F32(d) => t.pass(&obs.kind, d.image().as_slice()),
    }
    ObsRange {
        visible: (v0, ws.visible.len()),
        misses: (m0, ws.obs_misses.len()),
        new: (0, 0),
        mark,
    }
}

struct Toucher<'s> {
    grid: &'s tc::Grid,
    index: &'s mut BlockIndex,
    ws: &'s mut TsdfWorkspace,
    mark: u32,
    prev: Option<([i32; 3], [i32; 3])>,
    stats: &'s mut IntegrateStats,
}

impl Toucher<'_> {
    fn pass<D: RawDepth>(&mut self, kind: &ObsKind<'_>, depth: &[D]) {
        match kind {
            ObsKind::Pinhole(p) => {
                let w = p.width as usize;
                for (i, &raw) in depth.iter().enumerate() {
                    let Some(d) = raw.metres(p.mpu, p.range_min, p.range_max) else {
                        continue;
                    };
                    let ps = tc::pinhole_point(p, (i % w) as u32, (i / w) as u32, d);
                    let vfs = &p.volume_from_sensor;
                    self.point(tc::xform::<StdOps>(&vfs.r, &vfs.t, ps));
                }
            }
            ObsKind::Spherical(p, lidar) => {
                let w = p.width as usize;
                for (i, &raw) in depth.iter().enumerate() {
                    let Some(d) = raw.metres(p.mpu, p.range_min, p.range_max) else {
                        continue;
                    };
                    let (ce, se, ca, sa) = lidar.beam(i / w, i % w);
                    let ps = tc::spherical_point(d, ce, se, ca, sa);
                    let vfs = &p.volume_from_sensor;
                    self.point(tc::xform::<StdOps>(&vfs.r, &vfs.t, ps));
                }
            }
        }
    }

    /// Touches the blocks of one measured volume-frame point.
    #[inline(always)]
    fn point(&mut self, pv: [f32; 3]) {
        self.stats.valid_pixels += 1;
        let Some(bx) = tc::touch_box(pv, self.grid) else {
            self.stats.out_of_range += 1;
            return;
        };
        // a box equal to the previous pixel's adds nothing to the set
        if self.prev == Some(bx) {
            return;
        }
        self.prev = Some(bx);
        let (lo, hi) = bx;
        for x in lo[0]..=hi[0] {
            for y in lo[1]..=hi[1] {
                for z in lo[2]..=hi[2] {
                    self.key(tc::pack_block_key([x, y, z]));
                }
            }
        }
    }

    #[inline(always)]
    fn key(&mut self, key: u64) {
        match self.index.lookup(key) {
            Some(p) => {
                let m = &mut self.index.mark[p as usize];
                if *m != self.mark {
                    *m = self.mark;
                    self.ws.visible.push(p);
                }
            }
            None => {
                let slot = tc::hash_slot(key, MISS_FILTER.trailing_zeros());
                if self.ws.miss_filter[slot] != key {
                    self.ws.miss_filter[slot] = key;
                    self.ws.obs_misses.push(key);
                }
            }
        }
    }
}

/// Sorts and de-duplicates the step's misses, allocates them (pool index =
/// `num_blocks + rank`), then resolves each observation's new pools into
/// `ws.visible_new`. Fails before any observable mutation.
pub(crate) fn allocate(
    index: &mut BlockIndex,
    pools: &mut Pools,
    growth: PoolGrowth,
    ws: &mut TsdfWorkspace,
) -> Result<u32, TsdfError> {
    ws.misses.clear();
    ws.misses.extend_from_slice(&ws.obs_misses);
    ws.misses.sort_unstable();
    ws.misses.dedup();
    let m = ws.misses.len();
    ensure_capacity(index, pools, growth, m)?;
    insert_blocks(index, pools, &ws.misses);
    for r in ws.ranges.iter_mut() {
        let n0 = ws.visible_new.len();
        for &key in &ws.obs_misses[r.misses.0..r.misses.1] {
            let Some(p) = index.lookup(key) else {
                continue;
            };
            let mk = &mut index.mark[p as usize];
            if *mk != r.mark {
                *mk = r.mark;
                ws.visible_new.push(p);
            }
        }
        r.new = (n0, ws.visible_new.len());
    }
    Ok(m as u32)
}
