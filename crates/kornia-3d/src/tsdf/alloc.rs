//! Step allocation: touch every block within the truncation band of each
//! observation's measurements, look each up first, and sort only the
//! misses; new blocks get pool indices in ascending key order.

use kornia_image::Image;
use kornia_oxide_kernels::tsdf_core::{self as tc, RawDepth};
use rayon::prelude::*;

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

/// Samples per chunk of the parallel touch pass. Fixed, so the merge order
/// does not depend on the pool size.
const TOUCH_CHUNK: usize = 1 << 14;
/// Entries of a chunk's direct-mapped key filter (a power of two).
const CHUNK_FILTER: usize = 256;

/// Scratch of one chunk of the parallel touch pass.
#[derive(Debug, Default)]
pub(crate) struct TouchChunk {
    hits: Vec<u32>,
    misses: Vec<u64>,
    filter: Vec<u64>,
    valid: u64,
    out_of_range: u64,
}

/// Sizes the chunks for observations of up to `pixels` samples: like
/// `obs_misses`, a cold chunk may list up to one key per sample.
pub(crate) fn reserve_chunks(chunks: &mut Vec<TouchChunk>, pixels: usize) {
    let n = pixels.div_ceil(TOUCH_CHUNK);
    if chunks.len() < n {
        chunks.resize_with(n, TouchChunk::default);
    }
    for c in chunks.iter_mut() {
        c.hits.reserve(TOUCH_CHUNK);
        c.misses.reserve(TOUCH_CHUNK);
        c.filter.reserve(CHUNK_FILTER);
    }
}

impl TouchChunk {
    fn run<D: RawDepth>(
        &mut self,
        kind: &ObsKind<'_>,
        depth: &[D],
        i0: usize,
        i1: usize,
        grid: &tc::Grid,
        index: &BlockIndex,
    ) {
        self.hits.clear();
        self.misses.clear();
        if self.filter.len() != CHUNK_FILTER {
            self.filter.clear();
            self.filter.resize(CHUNK_FILTER, tc::EMPTY_KEY);
        } else {
            self.filter.fill(tc::EMPTY_KEY);
        }
        self.valid = 0;
        self.out_of_range = 0;
        let mut prev = None;
        match kind {
            ObsKind::Pinhole(p) => {
                let w = p.width as usize;
                for (i, &raw) in depth[i0..i1].iter().enumerate() {
                    let Some(d) = raw.metres(p.mpu, p.range_min, p.range_max) else {
                        continue;
                    };
                    let i = i0 + i;
                    let ps = tc::pinhole_point(p, (i % w) as u32, (i / w) as u32, d);
                    let vfs = &p.volume_from_sensor;
                    self.point(
                        tc::xform::<StdOps>(&vfs.r, &vfs.t, ps),
                        grid,
                        index,
                        &mut prev,
                    );
                }
            }
            ObsKind::Spherical(p, lidar) => {
                let w = p.width as usize;
                for (i, &raw) in depth[i0..i1].iter().enumerate() {
                    let Some(d) = raw.metres(p.mpu, p.range_min, p.range_max) else {
                        continue;
                    };
                    let i = i0 + i;
                    let (ce, se, ca, sa) = lidar.beam(i / w, i % w);
                    let ps = tc::spherical_point(d, ce, se, ca, sa);
                    let vfs = &p.volume_from_sensor;
                    self.point(
                        tc::xform::<StdOps>(&vfs.r, &vfs.t, ps),
                        grid,
                        index,
                        &mut prev,
                    );
                }
            }
        }
    }

    #[inline(always)]
    fn point(
        &mut self,
        pv: [f32; 3],
        grid: &tc::Grid,
        index: &BlockIndex,
        prev: &mut Option<([i32; 3], [i32; 3])>,
    ) {
        self.valid += 1;
        let Some(bx) = tc::touch_box(pv, grid) else {
            self.out_of_range += 1;
            return;
        };
        if *prev == Some(bx) {
            return;
        }
        *prev = Some(bx);
        let (lo, hi) = bx;
        for x in lo[0]..=hi[0] {
            for y in lo[1]..=hi[1] {
                for z in lo[2]..=hi[2] {
                    let key = tc::pack_block_key([x, y, z]);
                    // drops only later repeats: first-occurrence order is kept
                    let slot = tc::hash_slot(key, CHUNK_FILTER.trailing_zeros());
                    if self.filter[slot] == key {
                        continue;
                    }
                    self.filter[slot] = key;
                    match index.lookup(key) {
                        Some(p) => self.hits.push(p),
                        None => self.misses.push(key),
                    }
                }
            }
        }
    }
}

/// [`touch`] with the samples split into fixed chunks run as rayon tasks
/// (read-only on the index), merged serially in chunk order. `visible`
/// keeps the serial first-occurrence order and the misses the same set, so
/// every later result is identical to [`touch`].
pub(crate) fn touch_par(
    obs: &Obs<'_>,
    grid: &tc::Grid,
    index: &mut BlockIndex,
    ws: &mut TsdfWorkspace,
    stats: &mut IntegrateStats,
) -> ObsRange {
    let len = match obs.depth {
        DepthRef::U16(d) => d.image().as_slice().len(),
        DepthRef::F32(d) => d.image().as_slice().len(),
    };
    let n = len.div_ceil(TOUCH_CHUNK);
    if ws.touch_chunks.len() < n {
        ws.touch_chunks.resize_with(n, TouchChunk::default);
    }
    {
        let ix: &BlockIndex = index;
        let chunks = ws.touch_chunks[..n].par_iter_mut().enumerate();
        let run = |(c, ch): (usize, &mut TouchChunk)| {
            let (i0, i1) = (c * TOUCH_CHUNK, ((c + 1) * TOUCH_CHUNK).min(len));
            match obs.depth {
                DepthRef::U16(d) => ch.run(&obs.kind, d.image().as_slice(), i0, i1, grid, ix),
                DepthRef::F32(d) => ch.run(&obs.kind, d.image().as_slice(), i0, i1, grid, ix),
            }
        };
        chunks.for_each(run);
    }
    let mark = index.next_mark();
    let v0 = ws.visible.len();
    let m0 = ws.obs_misses.len();
    ws.miss_filter.fill(tc::EMPTY_KEY);
    let mf = MISS_FILTER.trailing_zeros();
    for ch in &ws.touch_chunks[..n] {
        stats.valid_pixels += ch.valid;
        stats.out_of_range += ch.out_of_range;
        for &p in &ch.hits {
            let m = &mut index.mark[p as usize];
            if *m != mark {
                *m = mark;
                ws.visible.push(p);
            }
        }
        for &key in &ch.misses {
            let slot = tc::hash_slot(key, mf);
            if ws.miss_filter[slot] != key {
                ws.miss_filter[slot] = key;
                ws.obs_misses.push(key);
            }
        }
    }
    ObsRange {
        visible: (v0, ws.visible.len()),
        misses: (m0, ws.obs_misses.len()),
        new: (0, 0),
        mark,
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
