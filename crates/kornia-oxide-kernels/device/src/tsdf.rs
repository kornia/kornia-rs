//! TSDF kernels (kornia-3d `tsdf::cuda`): allocation, integration, raycast
//! and marching cubes. Every per-voxel and per-pixel formula is the shared
//! `tsdf_core` with [`PtxOps`], so each kernel stores the CPU path's bits
//! (kornia-3d TSDF contract, sections 9-14).
//!
//! - Every pointer is a kernel parameter (a pointer loaded from memory
//!   loses its address space); per element, all loads before the stores.
//! - Atomics only as sets, counts and list appends (contract P6): the miss
//!   set is built with CAS, its keys are then ranked, and a new block's
//!   pool index is `len + rank` (ascending key), never an atomic counter.
//! - Device-resident counts drive grid-stride loops, so no launch waits on
//!   the host.

use core::slice;

use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32, DeviceAtomicU64};
use cuda_device::{SharedArray, kernel, thread, warp};

use crate::tsdf_core::{self as tc, MeshVoxels, RawDepth, VoxelSampler};
use crate::tsdf_ops::PtxOps;

type O = PtxOps;

const T: u32 = tc::THREADS;
const R: AtomicOrdering = AtomicOrdering::Relaxed;
const NONE: u32 = u32::MAX;
const FULL: u32 = 0xffff_ffff;
/// `depth_kind` of a `u16` depth image (else `f32`).
const DEPTH_U16: u32 = 0;
/// Largest `B³` (B = 16) and block edge count `3 (B + 1)³`.
const MAX_CUBES: usize = 4096;
const MAX_EDGES: usize = 14739;
/// Unused edge in the vertex-id table.
const NO_VERTEX: u16 = u16::MAX;

#[inline(always)]
fn gtid() -> u32 {
    thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()
}

#[inline(always)]
fn gstride() -> u32 {
    thread::gridDim_x() * thread::blockDim_x()
}

/// Warp sum of `v` added to `*counter` by lane 0; every lane must call.
#[inline(always)]
unsafe fn add_warp_sum(counter: *mut u64, v: u32) {
    let mut s = v;
    s += warp::shuffle_down_sync(FULL, s, 16);
    s += warp::shuffle_down_sync(FULL, s, 8);
    s += warp::shuffle_down_sync(FULL, s, 4);
    s += warp::shuffle_down_sync(FULL, s, 2);
    s += warp::shuffle_down_sync(FULL, s, 1);
    if warp::lane_id() == 0 && s != 0 {
        // SAFETY: `counter` is a live, 8-byte aligned device counter.
        unsafe { DeviceAtomicU64::from_ptr(counter).fetch_add(s as u64, R) };
    }
}

/// Warp sum of `v`, valid in lane 0.
#[inline(always)]
fn warp_sum(v: u32) -> u32 {
    let mut s = v;
    s += warp::shuffle_down_sync(FULL, s, 16);
    s += warp::shuffle_down_sync(FULL, s, 8);
    s += warp::shuffle_down_sync(FULL, s, 4);
    s += warp::shuffle_down_sync(FULL, s, 2);
    s += warp::shuffle_down_sync(FULL, s, 1);
    s
}

/// `#{ i < n : a[i] < k }` of ascending `a`.
#[inline(always)]
unsafe fn lower_bound(a: *const u64, n: u32, k: u64) -> u32 {
    let (mut lo, mut hi) = (0u32, n);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        // SAFETY: mid < n, within `a`.
        if unsafe { *a.add(mid as usize) } < k {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

// --------------------------------------------------------------- touch ----

/// Pointers and constants of one observation's touch pass.
#[derive(Clone, Copy)]
struct Touch {
    keys: *const u64,
    vals: *const u32,
    log2_cap: u32,
    mark: *mut u32,
    obs_mark: u32,
    miss_keys: *mut u64,
    log2_miss: u32,
    miss_mark: *mut u32,
    miss_list: *mut u32,
    hits: *mut u32,
    news: *mut u32,
    list_counts: *mut u32,
    counters: *mut u64,
}

impl Touch {
    /// Adds `key` to the observation's touched set: a hit appends its pool
    /// to `hits` once, a miss joins the step's miss set (CAS) and appends
    /// its slot to `news` once.
    #[inline(always)]
    unsafe fn key(&self, key: u64) {
        // SAFETY (whole fn): every index is a slot of its table or a pool
        // of the index (hash values), and the lists hold one entry per pool
        // or miss slot (marks make each append unique).
        unsafe {
            let keys = self.keys;
            if let Some(s) = tc::hash_find(self.log2_cap, key, |i| *keys.add(i)) {
                let p = *self.vals.add(s);
                if DeviceAtomicU32::from_ptr(self.mark.add(p as usize)).swap(self.obs_mark, R)
                    != self.obs_mark
                {
                    let pos = DeviceAtomicU32::from_ptr(self.list_counts).fetch_add(1, R);
                    *self.hits.add(pos as usize) = p;
                }
                return;
            }
            let mask = (1usize << self.log2_miss) - 1;
            let mut i = tc::hash_slot(key, self.log2_miss);
            let mut slot = NONE;
            let mut probes = 0usize;
            while probes <= mask {
                let cell = DeviceAtomicU64::from_ptr(self.miss_keys.add(i));
                match cell.compare_exchange(tc::EMPTY_KEY, key, R, R) {
                    Ok(_) => {
                        let at = DeviceAtomicU64::from_ptr(self.counters.add(tc::SC_MISSES))
                            .fetch_add(1, R);
                        *self.miss_list.add(at as usize) = i as u32;
                        slot = i as u32;
                        break;
                    }
                    Err(k) if k == key => {
                        slot = i as u32;
                        break;
                    }
                    Err(_) => i = (i + 1) & mask,
                }
                probes += 1;
            }
            if slot == NONE {
                DeviceAtomicU64::from_ptr(self.counters.add(tc::SC_ABORT))
                    .fetch_or(tc::ABORT_MISS_SET, R);
                return;
            }
            if DeviceAtomicU32::from_ptr(self.miss_mark.add(slot as usize)).swap(self.obs_mark, R)
                != self.obs_mark
            {
                let pos = DeviceAtomicU32::from_ptr(self.list_counts.add(1)).fetch_add(1, R);
                *self.news.add(pos as usize) = slot;
            }
        }
    }

    /// Touches the blocks of one measured volume-frame point; `false` when
    /// its band leaves the voxel domain.
    #[inline(always)]
    unsafe fn point(&self, grid: &tc::Grid, pv: [f32; 3]) -> bool {
        let Some((lo, hi)) = tc::touch_box(pv, grid) else {
            return false;
        };
        let mut x = lo[0];
        while x <= hi[0] {
            let mut y = lo[1];
            while y <= hi[1] {
                let mut z = lo[2];
                while z <= hi[2] {
                    // SAFETY: see `key`.
                    unsafe { self.key(tc::pack_block_key([x, y, z])) };
                    z += 1;
                }
                y += 1;
            }
            x += 1;
        }
        true
    }
}

/// Depth sample `i` in metres (`depth_kind` picks `u16` or `f32`).
#[inline(always)]
unsafe fn depth_metres(
    depth: *const u8,
    kind: u32,
    i: u32,
    mpu: f32,
    lo: f32,
    hi: f32,
) -> Option<f32> {
    // SAFETY: `i` indexes the depth image the launcher sized.
    unsafe {
        if kind == DEPTH_U16 {
            (*(depth as *const u16).add(i as usize)).metres(mpu, lo, hi)
        } else {
            (*(depth as *const f32).add(i as usize)).metres(mpu, lo, hi)
        }
    }
}

/// Touch pass of one pinhole observation (contract section 9): every
/// block within `± trunc` of a kept depth sample joins the observation's
/// visible list (hits) or the step's miss set.
///
/// # Safety
///
/// `step`, `obs` hold one parameter block each; `depth` holds `width *
/// height` samples of `depth_kind`; `keys`/`vals` are the block hash of
/// `2^log2_cap` slots, `mark` one entry per pool; `miss_keys`, `miss_mark`,
/// `miss_list` hold `2^log2_miss` entries; `hits` the index capacity,
/// `news` `2^log2_miss` entries; `list_counts` 2, `counters` `SC_LEN`.
#[kernel]
pub unsafe fn tsdf_touch_pinhole(
    step: *const tc::StepParams,
    obs: *const tc::PinholeParams,
    depth: *const u8,
    depth_kind: u32,
    keys: *const u64,
    vals: *const u32,
    mark: *mut u32,
    obs_mark: u32,
    miss_keys: *mut u64,
    miss_mark: *mut u32,
    miss_list: *mut u32,
    hits: *mut u32,
    news: *mut u32,
    list_counts: *mut u32,
    counters: *mut u64,
) {
    // SAFETY: the launcher upholds the layout above.
    unsafe {
        let st = *step;
        let p = *obs;
        let t = Touch {
            keys,
            vals,
            log2_cap: st.log2_cap,
            mark,
            obs_mark,
            miss_keys,
            log2_miss: st.log2_miss,
            miss_mark,
            miss_list,
            hits,
            news,
            list_counts,
            counters,
        };
        let n = p.width * p.height;
        let (mut valid, mut oor) = (0u32, 0u32);
        let mut i = gtid();
        while i < n {
            if let Some(d) = depth_metres(depth, depth_kind, i, p.mpu, p.range_min, p.range_max) {
                valid += 1;
                let ps = tc::pinhole_point(&p, i % p.width, i / p.width, d);
                let v = &p.volume_from_sensor;
                if !t.point(&st.grid, tc::xform::<O>(&v.r, &v.t, ps)) {
                    oor += 1;
                }
            }
            i += gstride();
        }
        add_warp_sum(counters.add(tc::SC_VALID), valid);
        add_warp_sum(counters.add(tc::SC_OUT_OF_RANGE), oor);
    }
}

/// Spherical-LiDAR twin of [`tsdf_touch_pinhole`]; `tables` holds the
/// sensor's `row_boundaries (H - 1) | cos_el (H) | sin_el (H) | cos_az (W)
/// | sin_az (W)`.
///
/// # Safety
///
/// As [`tsdf_touch_pinhole`], plus `tables` as above.
#[kernel]
pub unsafe fn tsdf_touch_spherical(
    step: *const tc::StepParams,
    obs: *const tc::SphericalParams,
    tables: *const f32,
    depth: *const u8,
    depth_kind: u32,
    keys: *const u64,
    vals: *const u32,
    mark: *mut u32,
    obs_mark: u32,
    miss_keys: *mut u64,
    miss_mark: *mut u32,
    miss_list: *mut u32,
    hits: *mut u32,
    news: *mut u32,
    list_counts: *mut u32,
    counters: *mut u64,
) {
    // SAFETY: the launcher upholds the layout above.
    unsafe {
        let st = *step;
        let p = *obs;
        let t = Touch {
            keys,
            vals,
            log2_cap: st.log2_cap,
            mark,
            obs_mark,
            miss_keys,
            log2_miss: st.log2_miss,
            miss_mark,
            miss_list,
            hits,
            news,
            list_counts,
            counters,
        };
        let (w, h) = (p.width as usize, p.height as usize);
        let cos_el = tables.add(h - 1);
        let sin_el = cos_el.add(h);
        let cos_az = sin_el.add(h);
        let sin_az = cos_az.add(w);
        let n = p.width * p.height;
        let (mut valid, mut oor) = (0u32, 0u32);
        let mut i = gtid();
        while i < n {
            if let Some(d) = depth_metres(depth, depth_kind, i, p.mpu, p.range_min, p.range_max) {
                valid += 1;
                let (row, col) = ((i / p.width) as usize, (i % p.width) as usize);
                let ps = tc::spherical_point(
                    d,
                    *cos_el.add(row),
                    *sin_el.add(row),
                    *cos_az.add(col),
                    *sin_az.add(col),
                );
                let v = &p.volume_from_sensor;
                if !t.point(&st.grid, tc::xform::<O>(&v.r, &v.t, ps)) {
                    oor += 1;
                }
            }
            i += gstride();
        }
        add_warp_sum(counters.add(tc::SC_VALID), valid);
        add_warp_sum(counters.add(tc::SC_OUT_OF_RANGE), oor);
    }
}

// ---------------------------------------------------------- allocation ----

/// Ranks the step's misses: `new_keys[rank] = key` and `miss_rank[slot] =
/// rank`, `rank = #{ misses < key }` (keys are distinct, so ranks are a
/// permutation in ascending key order). Sets `ABORT_POOL_FULL` instead when
/// the misses do not fit the pool.
///
/// # Safety
///
/// `miss_keys`/`miss_rank` hold `2^log2_miss` entries, `miss_list` and
/// `new_keys` as many; `counters` `SC_LEN`.
#[kernel]
pub unsafe fn tsdf_rank_misses(
    step: *const tc::StepParams,
    miss_keys: *const u64,
    miss_list: *const u32,
    miss_rank: *mut u32,
    new_keys: *mut u64,
    counters: *mut u64,
) {
    static mut TILE: SharedArray<u64, 256> = SharedArray::UNINIT;
    // SAFETY: the launcher upholds the layout above; TILE indices < T.
    unsafe {
        let st = *step;
        if *counters.add(tc::SC_ABORT) != 0 {
            return;
        }
        let m = *counters.add(tc::SC_MISSES);
        if st.len as u64 + m > st.index_cap as u64 || st.pages as u64 + m > st.arena_cap as u64 {
            if gtid() == 0 {
                DeviceAtomicU64::from_ptr(counters.add(tc::SC_ABORT))
                    .fetch_or(tc::ABORT_POOL_FULL, R);
            }
            return;
        }
        let m = m as u32;
        let tid = thread::threadIdx_x();
        let mut base = thread::blockIdx_x() * T;
        while base < m {
            let i = base + tid;
            let (key, slot) = if i < m {
                let s = *miss_list.add(i as usize);
                (*miss_keys.add(s as usize), s)
            } else {
                (tc::EMPTY_KEY, 0)
            };
            let mut rank = 0u32;
            let mut t0 = 0u32;
            while t0 < m {
                let j = t0 + tid;
                TILE[tid as usize] = if j < m {
                    *miss_keys.add(*miss_list.add(j as usize) as usize)
                } else {
                    tc::EMPTY_KEY
                };
                thread::sync_threads();
                let cnt = if m - t0 < T { m - t0 } else { T };
                let mut k = 0u32;
                while k < cnt {
                    rank += (TILE[k as usize] < key) as u32;
                    k += 1;
                }
                thread::sync_threads();
                t0 += T;
            }
            if i < m {
                *new_keys.add(rank as usize) = key;
                *miss_rank.add(slot as usize) = rank;
            }
            base += thread::gridDim_x() * T;
        }
    }
}

/// Allocates the ranked misses: pool `len + r` gets key `new_keys[r]`,
/// page `pages + r`, cleared epochs, and a hash entry (CAS insert; slots
/// may differ from the CPU's, lookups do not).
///
/// # Safety
///
/// `keys`/`vals` hold `2^log2_cap` slots, the pool arrays `index_cap`
/// entries; `counters` `SC_LEN`.
#[kernel]
pub unsafe fn tsdf_assign_misses(
    step: *const tc::StepParams,
    new_keys: *const u64,
    keys: *mut u64,
    vals: *mut u32,
    pool_keys: *mut u64,
    pages: *mut u32,
    modified: *mut u64,
    mark: *mut u32,
    counters: *const u64,
) {
    // SAFETY: the launcher upholds the layout above; the rank kernel
    // checked `len + m <= index_cap` and `pages + m <= arena_cap`.
    unsafe {
        let st = *step;
        if *counters.add(tc::SC_ABORT) != 0 {
            return;
        }
        let m = *counters.add(tc::SC_MISSES) as u32;
        let mask = (1usize << st.log2_cap) - 1;
        let mut r = gtid();
        while r < m {
            let key = *new_keys.add(r as usize);
            let pool = (st.len + r) as usize;
            *pool_keys.add(pool) = key;
            *pages.add(pool) = st.pages + r;
            *modified.add(pool) = 0;
            *mark.add(pool) = 0;
            let mut i = tc::hash_slot(key, st.log2_cap);
            let mut probes = 0usize;
            while probes <= mask {
                let cell = DeviceAtomicU64::from_ptr(keys.add(i));
                if cell.compare_exchange(tc::EMPTY_KEY, key, R, R).is_ok() {
                    *vals.add(i) = pool as u32;
                    break;
                }
                i = (i + 1) & mask;
                probes += 1;
            }
            r += gstride();
        }
    }
}

/// Merges the ranked misses into the ascending `(key, pool)` list:
/// `old` (`len` entries) and `new_keys` (pools `len + r`) into `out`.
/// A no-op without misses (the host then keeps the old list).
///
/// # Safety
///
/// `old_*` hold `len`, `new_keys` the misses, `out_*` `len + misses`
/// entries; `counters` `SC_LEN`.
#[kernel]
pub unsafe fn tsdf_merge_keys(
    step: *const tc::StepParams,
    new_keys: *const u64,
    old_keys: *const u64,
    old_pools: *const u32,
    out_keys: *mut u64,
    out_pools: *mut u32,
    counters: *const u64,
) {
    // SAFETY: the launcher upholds the layout above.
    unsafe {
        let st = *step;
        if *counters.add(tc::SC_ABORT) != 0 {
            return;
        }
        let m = *counters.add(tc::SC_MISSES) as u32;
        if m == 0 {
            return;
        }
        let n = st.len;
        let mut i = gtid();
        while i < n + m {
            if i < n {
                let k = *old_keys.add(i as usize);
                let at = (i + lower_bound(new_keys, m, k)) as usize;
                *out_keys.add(at) = k;
                *out_pools.add(at) = *old_pools.add(i as usize);
            } else {
                let r = i - n;
                let k = *new_keys.add(r as usize);
                let at = (r + lower_bound(old_keys, n, k)) as usize;
                *out_keys.add(at) = k;
                *out_pools.add(at) = n + r;
            }
            i += gstride();
        }
    }
}

/// Rebuilds a block hash from `pool_keys` (`len` pools) into cleared
/// `keys`/`vals` (a grown table).
///
/// # Safety
///
/// `keys`/`vals` hold `2^log2_cap` slots, all `EMPTY_KEY`; `pool_keys`
/// `len` entries.
#[kernel]
pub unsafe fn tsdf_rehash(
    keys: *mut u64,
    vals: *mut u32,
    log2_cap: u32,
    pool_keys: *const u64,
    len: u32,
) {
    // SAFETY: the launcher upholds the layout above.
    unsafe {
        let mask = (1usize << log2_cap) - 1;
        let mut p = gtid();
        while p < len {
            let key = *pool_keys.add(p as usize);
            let mut i = tc::hash_slot(key, log2_cap);
            let mut probes = 0usize;
            while probes <= mask {
                if DeviceAtomicU64::from_ptr(keys.add(i))
                    .compare_exchange(tc::EMPTY_KEY, key, R, R)
                    .is_ok()
                {
                    *vals.add(i) = p;
                    break;
                }
                i = (i + 1) & mask;
                probes += 1;
            }
            p += gstride();
        }
    }
}

// ----------------------------------------------------------- integrate ----

/// Pointers of one observation's update pass.
#[derive(Clone, Copy)]
struct Pass {
    tsdf: *mut f32,
    weight: *mut u16,
    rgbw: *mut u32,
    color: *const u8,
    pool_keys: *const u64,
    pages: *const u32,
    hits: *const u32,
    news: *const u32,
    list_counts: *const u32,
    miss_rank: *const u32,
    modified: *mut u64,
    counters: *mut u64,
}

impl Pass {
    /// Updates every voxel of the observation's visible blocks (one block
    /// per thread block, grid-stride), `sdf(centre)` giving the signed
    /// distance and pixel of a voxel centre.
    #[inline(always)]
    unsafe fn run(&self, st: &tc::StepParams, sdf: impl Fn([f32; 3]) -> Option<(f32, u32)>) {
        // SAFETY (whole fn): list entries are pools of the index (`len +
        // rank` for new ones), pages are within the pools, pixel indices
        // come from the gated projection, so every access is in bounds.
        unsafe {
            if *self.counters.add(tc::SC_ABORT) != 0 {
                return;
            }
            let g = &st.grid;
            let voxels = 1u32 << (3 * g.log2);
            let nh = *self.list_counts;
            let total = nh + *self.list_counts.add(1);
            let tid = thread::threadIdx_x();
            let (mut updated, mut touched) = (0u32, 0u32);
            let mut e = thread::blockIdx_x();
            while e < total {
                let pool = if e < nh {
                    *self.hits.add(e as usize)
                } else {
                    st.len
                        + *self
                            .miss_rank
                            .add(*self.news.add((e - nh) as usize) as usize)
                } as usize;
                if tid == 0
                    && DeviceAtomicU64::from_ptr(self.modified.add(pool)).swap(st.epoch, R)
                        != st.epoch
                {
                    touched += 1;
                }
                let origin = tc::block_origin(*self.pool_keys.add(pool), g.log2);
                let base = (*self.pages.add(pool) * voxels) as usize;
                let mut v = tid;
                while v < voxels {
                    let l = tc::local_coords(v, g.log2);
                    let gv = [origin[0] + l[0], origin[1] + l[1], origin[2] + l[2]];
                    let c = tc::voxel_centre::<O>(gv, g.voxel_size, g.half_voxel);
                    if let Some((d, idx)) = sdf(c) {
                        if let Some(tsd) = tc::truncate(d, g.trunc, g.inv_trunc) {
                            let i = base + v as usize;
                            let old = *self.tsdf.add(i);
                            let w0 = *self.weight.add(i);
                            let colour = !self.color.is_null();
                            let (s0, rgb) = if colour {
                                let k = 3 * idx as usize;
                                (
                                    *self.rgbw.add(i),
                                    [
                                        *self.color.add(k),
                                        *self.color.add(k + 1),
                                        *self.color.add(k + 2),
                                    ],
                                )
                            } else {
                                (0, [0; 3])
                            };
                            let (t, w) = tc::update_tsdf::<O>(old, w0, tsd, st.max_weight as u16);
                            *self.tsdf.add(i) = t;
                            *self.weight.add(i) = w;
                            if colour {
                                *self.rgbw.add(i) =
                                    tc::update_color::<O>(s0, rgb, st.max_color_weight as u8);
                            }
                            updated += 1;
                        }
                    }
                    v += T;
                }
                e += thread::gridDim_x();
            }
            add_warp_sum(self.counters.add(tc::SC_UPDATED), updated);
            add_warp_sum(self.counters.add(tc::SC_TOUCHED), touched);
        }
    }
}

/// Update pass of one pinhole observation (contract section 8) over its
/// visible blocks: `hits` then `news` (miss slots, pool `len + rank`).
/// `color` is null without colour.
///
/// # Safety
///
/// As the touch pass; pools hold `arena_cap * B³` voxels (`rgbw` too when
/// `color` is not null), `color` `3 * width * height` bytes.
#[kernel]
pub unsafe fn tsdf_integrate_pinhole(
    step: *const tc::StepParams,
    obs: *const tc::PinholeParams,
    depth: *const u8,
    depth_kind: u32,
    color: *const u8,
    tsdf: *mut f32,
    weight: *mut u16,
    rgbw: *mut u32,
    pool_keys: *const u64,
    pages: *const u32,
    hits: *const u32,
    news: *const u32,
    list_counts: *const u32,
    miss_rank: *const u32,
    modified: *mut u64,
    counters: *mut u64,
) {
    // SAFETY: the launcher upholds the layout above.
    unsafe {
        let st = *step;
        let p = *obs;
        let pass = Pass {
            tsdf,
            weight,
            rgbw,
            color,
            pool_keys,
            pages,
            hits,
            news,
            list_counts,
            miss_rank,
            modified,
            counters,
        };
        let n = (p.width * p.height) as usize;
        if depth_kind == DEPTH_U16 {
            let d = slice::from_raw_parts(depth as *const u16, n);
            pass.run(&st, |c| tc::pinhole_sdf::<O, u16>(&p, d, c));
        } else {
            let d = slice::from_raw_parts(depth as *const f32, n);
            pass.run(&st, |c| tc::pinhole_sdf::<O, f32>(&p, d, c));
        }
    }
}

/// Spherical-LiDAR twin of [`tsdf_integrate_pinhole`] (no colour);
/// `tables` as in [`tsdf_touch_spherical`].
///
/// # Safety
///
/// As [`tsdf_integrate_pinhole`].
#[kernel]
pub unsafe fn tsdf_integrate_spherical(
    step: *const tc::StepParams,
    obs: *const tc::SphericalParams,
    tables: *const f32,
    depth: *const u8,
    depth_kind: u32,
    tsdf: *mut f32,
    weight: *mut u16,
    pool_keys: *const u64,
    pages: *const u32,
    hits: *const u32,
    news: *const u32,
    list_counts: *const u32,
    miss_rank: *const u32,
    modified: *mut u64,
    counters: *mut u64,
) {
    // SAFETY: the launcher upholds the layout above.
    unsafe {
        let st = *step;
        let p = *obs;
        let pass = Pass {
            tsdf,
            weight,
            rgbw: core::ptr::null_mut(),
            color: core::ptr::null(),
            pool_keys,
            pages,
            hits,
            news,
            list_counts,
            miss_rank,
            modified,
            counters,
        };
        let bnd = slice::from_raw_parts(tables, (p.height - 1) as usize);
        let n = (p.width * p.height) as usize;
        if depth_kind == DEPTH_U16 {
            let d = slice::from_raw_parts(depth as *const u16, n);
            pass.run(&st, |c| tc::spherical_sdf::<O, u16>(&p, bnd, d, c));
        } else {
            let d = slice::from_raw_parts(depth as *const f32, n);
            pass.run(&st, |c| tc::spherical_sdf::<O, f32>(&p, bnd, d, c));
        }
    }
}

// ------------------------------------------------------------- raycast ----

/// One volume's voxels for the raycast: its hash, page table and the pools.
#[derive(Clone, Copy)]
struct View {
    keys: *const u64,
    vals: *const u32,
    log2_cap: u32,
    pages: *const u32,
    tsdf: *const f32,
    weight: *const u16,
    log2: u32,
}

impl View {
    /// Pool offset of voxel `g`, `None` when its block is unallocated.
    #[inline(always)]
    unsafe fn slot(&self, g: [i32; 3]) -> Option<usize> {
        let l = self.log2;
        let key = tc::pack_block_key([g[0] >> l, g[1] >> l, g[2] >> l]);
        let keys = self.keys;
        // SAFETY: slots and pools come from the hash, pages from the table.
        unsafe {
            let s = tc::hash_find(self.log2_cap, key, |i| *keys.add(i))?;
            let p = *self.vals.add(s);
            let m = (1 << l) - 1;
            let base = (*self.pages.add(p as usize) as usize) << (3 * l);
            Some(base + tc::local_index([g[0] & m, g[1] & m, g[2] & m], l))
        }
    }
}

impl VoxelSampler for View {
    #[inline(always)]
    fn sample(&self, g: [i32; 3]) -> Option<(f32, u16)> {
        // SAFETY: `slot` is within the pools.
        unsafe {
            let i = self.slot(g)?;
            Some((*self.tsdf.add(i), *self.weight.add(i)))
        }
    }
}

/// Raycasts `nviews` volumes (submaps) into one pinhole view (contract
/// section 13): per pixel each view marches with its own parameters, the
/// smallest hit `z` wins (the lower view on a tie). A miss writes zeros.
/// `view_headers[v]` is the header (hash and page-table block) of view
/// `v`; `out_depth`/`out_colors` may be null.
///
/// # Safety
///
/// `views`/`view_headers` hold `nviews` entries; header `h`'s hash is at
/// `keys/vals + h * hash_stride` (`2^log2_cap` slots), its page table at
/// `pages + h * pool_stride`; outputs hold `width * height` pixels (3
/// values each for vertices, normals and colours).
#[kernel]
pub unsafe fn tsdf_raycast(
    views: *const tc::RaycastParams,
    view_headers: *const u32,
    nviews: u32,
    keys: *const u64,
    vals: *const u32,
    hash_stride: u32,
    log2_cap: u32,
    pages: *const u32,
    pool_stride: u32,
    tsdf: *const f32,
    weight: *const u16,
    rgbw: *const u32,
    log2: u32,
    width: u32,
    height: u32,
    out_depth: *mut f32,
    out_vertices: *mut f32,
    out_normals: *mut f32,
    out_colors: *mut u8,
) {
    // SAFETY: the launcher upholds the layout above.
    unsafe {
        let view = |v: u32| {
            let h = *view_headers.add(v as usize) as usize;
            View {
                keys: keys.add(h * hash_stride as usize),
                vals: vals.add(h * hash_stride as usize),
                log2_cap,
                pages: pages.add(h * pool_stride as usize),
                tsdf,
                weight,
                log2,
            }
        };
        let n = width * height;
        let mut i = gtid();
        while i < n {
            let (px, py) = (i % width, i / width);
            let mut best: Option<(tc::RayHit, u32)> = None;
            let mut v = 0u32;
            while v < nviews {
                let rp = *views.add(v as usize);
                if let Some(h) = tc::raycast_pixel::<O, View>(&rp, &view(v), px, py) {
                    let better = match &best {
                        Some((b, _)) => h.z < b.z,
                        None => true,
                    };
                    if better {
                        best = Some((h, v));
                    }
                }
                v += 1;
            }
            let (z, vert, norm) = match &best {
                Some((h, _)) => (h.z, h.vertex, h.normal),
                None => (0.0, [0.0; 3], [0.0; 3]),
            };
            let mut rgb = [0u8; 3];
            if !out_colors.is_null() {
                if let Some((h, v)) = &best {
                    let rp = *views.add(*v as usize);
                    if let Some(g) = tc::voxel_from_point(h.point, rp.inv_voxel) {
                        let vw = view(*v);
                        if let Some(s) = vw.slot(g) {
                            if *weight.add(s) as u32 >= rp.weight_threshold {
                                rgb = tc::unpack_rgb(*rgbw.add(s));
                            }
                        }
                    }
                }
            }
            let k = 3 * i as usize;
            if !out_depth.is_null() {
                *out_depth.add(i as usize) = z;
            }
            *out_vertices.add(k) = vert[0];
            *out_vertices.add(k + 1) = vert[1];
            *out_vertices.add(k + 2) = vert[2];
            *out_normals.add(k) = norm[0];
            *out_normals.add(k + 1) = norm[1];
            *out_normals.add(k + 2) = norm[2];
            if !out_colors.is_null() {
                *out_colors.add(k) = rgb[0];
                *out_colors.add(k + 1) = rgb[1];
                *out_colors.add(k + 2) = rgb[2];
            }
            i += gstride();
        }
    }
}

// ------------------------------------------------------ marching cubes ----

/// The voxels `[-1, B + 1]^3` around one block: its 27 neighbour blocks'
/// pool offsets (`NONE` = unallocated) and the pools.
#[derive(Clone, Copy)]
struct Hood {
    bases: [u32; 27],
    tsdf: *const f32,
    weight: *const u16,
    rgbw: *const u32,
    threshold: u32,
    log2: u32,
}

impl Hood {
    #[inline(always)]
    fn slot(&self, v: [i32; 3]) -> Option<usize> {
        let l = self.log2;
        let ni = ((v[0] >> l) + 1) + 3 * (((v[1] >> l) + 1) + 3 * ((v[2] >> l) + 1));
        let b = self.bases[ni as usize];
        if b == NONE {
            return None;
        }
        let m = (1 << l) - 1;
        Some(b as usize + tc::local_index([v[0] & m, v[1] & m, v[2] & m], l))
    }

    /// Configuration of the cube with min corner `l0` (0 unless all 8
    /// corners are valid).
    #[inline(always)]
    fn conf(&self, l0: [i32; 3]) -> usize {
        let mut f = [0f32; 8];
        for c in 0..8 {
            let o = tc::CORNER[c];
            match self.value([l0[0] + o[0], l0[1] + o[1], l0[2] + o[2]]) {
                Some(v) => f[c] = v,
                None => return 0,
            }
        }
        tc::cube_index(&f)
    }
}

impl MeshVoxels for Hood {
    #[inline(always)]
    fn value(&self, v: [i32; 3]) -> Option<f32> {
        let i = self.slot(v)?;
        // SAFETY: `slot` is within the pools.
        unsafe { (*self.weight.add(i) as u32 >= self.threshold).then(|| *self.tsdf.add(i)) }
    }

    #[inline(always)]
    fn rgb(&self, v: [i32; 3]) -> [u8; 3] {
        match self.slot(v) {
            // SAFETY: `slot` is within the pools; `rgbw` is set with colours.
            Some(i) if !self.rgbw.is_null() => tc::unpack_rgb(unsafe { *self.rgbw.add(i) }),
            _ => [0; 3],
        }
    }
}

/// `(owner, axis)` of block edge `e` (inverse of `edge_key`).
#[inline(always)]
fn edge_of(e: u32, b: u32) -> ([i32; 3], u32) {
    let q = e / 3;
    let s = b + 1;
    (
        [(q % s) as i32, ((q / s) % s) as i32, (q / (s * s)) as i32],
        e % 3,
    )
}

/// Pool offset of the block `key + d` for `d = neighbour i` of `{-1, 0, 1}^3`
/// (x fastest), `NONE` when unallocated.
#[inline(always)]
unsafe fn neighbour_base(
    keys: *const u64,
    vals: *const u32,
    log2_cap: u32,
    pages: *const u32,
    key: u64,
    i: u32,
    log2: u32,
) -> u32 {
    let d = [
        (i % 3) as i32 - 1,
        ((i / 3) % 3) as i32 - 1,
        (i / 9) as i32 - 1,
    ];
    let Some(nk) = tc::neighbour_key(tc::unpack_block_key(key), d) else {
        return NONE;
    };
    // SAFETY: slots and pools come from the hash, pages from the table.
    unsafe {
        match tc::hash_find(log2_cap, nk, |s| *keys.add(s)) {
            Some(s) => *pages.add(*vals.add(s) as usize) << (3 * log2),
            None => NONE,
        }
    }
}

/// Per listed block: vertex and triangle counts of its mesh (contract
/// section 12.4), for the host's prefix sum.
///
/// # Safety
///
/// `list` holds `n` pools of the index; `out_nv`/`out_nt` `n` entries;
/// hash, page table and pools as in the raycast (one header).
#[kernel]
pub unsafe fn tsdf_mc_count(
    mp: *const tc::MeshParamsDev,
    keys: *const u64,
    vals: *const u32,
    pages: *const u32,
    pool_keys: *const u64,
    tsdf: *const f32,
    weight: *const u16,
    list: *const u32,
    n: u32,
    out_nv: *mut u32,
    out_nt: *mut u32,
) {
    static mut BASES: SharedArray<u32, 32> = SharedArray::UNINIT;
    static mut CONF: SharedArray<u8, MAX_CUBES> = SharedArray::UNINIT;
    static mut WSUM: SharedArray<u32, 16> = SharedArray::UNINIT;
    // SAFETY: the launcher upholds the layout above; shared indices are
    // below their sizes (B <= 16, 8 warps).
    unsafe {
        let p = *mp;
        let b = 1u32 << p.log2;
        let cubes = b * b * b;
        let edges = 3 * (b + 1) * (b + 1) * (b + 1);
        let tid = thread::threadIdx_x();
        let mut e = thread::blockIdx_x();
        while e < n {
            let pool = *list.add(e as usize) as usize;
            let key = *pool_keys.add(pool);
            if tid < 27 {
                BASES[tid as usize] =
                    neighbour_base(keys, vals, p.log2_cap, pages, key, tid, p.log2);
            }
            thread::sync_threads();
            let mut hood = Hood {
                bases: [NONE; 27],
                tsdf,
                weight,
                rgbw: core::ptr::null(),
                threshold: p.weight_threshold,
                log2: p.log2,
            };
            for k in 0..27 {
                hood.bases[k] = BASES[k];
            }
            let mut nt = 0u32;
            let mut c = tid;
            while c < cubes {
                let conf = hood.conf(tc::local_coords(c, p.log2));
                CONF[c as usize] = conf as u8;
                nt += tc::tri_count(conf);
                c += T;
            }
            thread::sync_threads();
            let conf_at = |l: [i32; 3]| CONF[tc::local_index(l, p.log2)] as usize;
            let mut nv = 0u32;
            let mut ei = tid;
            while ei < edges {
                let (o, axis) = edge_of(ei, b);
                nv += tc::edge_used(o, axis, b as i32, conf_at) as u32;
                ei += T;
            }
            let (snv, snt) = (warp_sum(nv), warp_sum(nt));
            let w = (tid / 32) as usize;
            if warp::lane_id() == 0 {
                WSUM[w] = snv;
                WSUM[8 + w] = snt;
            }
            thread::sync_threads();
            if tid == 0 {
                let (mut tv, mut tt) = (0u32, 0u32);
                for k in 0..8 {
                    tv += WSUM[k];
                    tt += WSUM[8 + k];
                }
                *out_nv.add(e as usize) = tv;
                *out_nt.add(e as usize) = tt;
            }
            thread::sync_threads();
            e += thread::gridDim_x();
        }
    }
}

/// Per listed block: writes its mesh at `v_off[e]`/`t_off[e]` (vertices
/// in ascending edge index, triangles in cube then table order, wound
/// `(i0, i2, i1)`), indices global (`index_base + v_off + local`) or
/// block-local.
/// Normals and colours are written when `mp` asks for them.
///
/// # Safety
///
/// As [`tsdf_mc_count`]; outputs hold the counted vertices (3 values each)
/// and triangles (3 indices) at the given offsets; `rgbw` is null unless
/// colours are requested.
#[kernel]
pub unsafe fn tsdf_mc_emit(
    mp: *const tc::MeshParamsDev,
    keys: *const u64,
    vals: *const u32,
    pages: *const u32,
    pool_keys: *const u64,
    tsdf: *const f32,
    weight: *const u16,
    rgbw: *const u32,
    list: *const u32,
    n: u32,
    v_off: *const u64,
    t_off: *const u64,
    out_vertices: *mut f32,
    out_normals: *mut f32,
    out_colors: *mut u8,
    out_triangles: *mut u32,
) {
    static mut BASES: SharedArray<u32, 32> = SharedArray::UNINIT;
    static mut CONF: SharedArray<u8, MAX_CUBES> = SharedArray::UNINIT;
    static mut IDS: SharedArray<u16, MAX_EDGES> = SharedArray::UNINIT;
    static mut CHUNK: SharedArray<u32, 256> = SharedArray::UNINIT;
    // SAFETY: the launcher upholds the layout above; shared indices are
    // below their sizes (B <= 16, 256 threads).
    unsafe {
        let p = *mp;
        let b = 1u32 << p.log2;
        let cubes = b * b * b;
        let edges = 3 * (b + 1) * (b + 1) * (b + 1);
        let tid = thread::threadIdx_x();
        let (normals, colors) = (p.normals != 0, p.colors != 0);
        let xf = p.world_from_volume;
        let mut e = thread::blockIdx_x();
        while e < n {
            let pool = *list.add(e as usize) as usize;
            let key = *pool_keys.add(pool);
            let origin = tc::block_origin(key, p.log2);
            let vb = *v_off.add(e as usize);
            let tb = *t_off.add(e as usize);
            if tid < 27 {
                BASES[tid as usize] =
                    neighbour_base(keys, vals, p.log2_cap, pages, key, tid, p.log2);
            }
            thread::sync_threads();
            let mut hood = Hood {
                bases: [NONE; 27],
                tsdf,
                weight,
                rgbw,
                threshold: p.weight_threshold,
                log2: p.log2,
            };
            for k in 0..27 {
                hood.bases[k] = BASES[k];
            }
            let mut c = tid;
            while c < cubes {
                CONF[c as usize] = hood.conf(tc::local_coords(c, p.log2)) as u8;
                c += T;
            }
            thread::sync_threads();

            // vertices: one per used edge, ids in ascending edge index
            let conf_at = |l: [i32; 3]| CONF[tc::local_index(l, p.log2)] as usize;
            let ce = edges.div_ceil(T);
            let (e0, e1) = (
                tid * ce,
                if (tid + 1) * ce < edges {
                    (tid + 1) * ce
                } else {
                    edges
                },
            );
            let mut cnt = 0u32;
            let mut ei = e0;
            while ei < e1 {
                let (o, axis) = edge_of(ei, b);
                let used = tc::edge_used(o, axis, b as i32, conf_at);
                IDS[ei as usize] = if used { 0 } else { NO_VERTEX };
                cnt += used as u32;
                ei += 1;
            }
            CHUNK[tid as usize] = cnt;
            thread::sync_threads();
            if tid == 0 {
                let mut acc = 0u32;
                for k in 0..T as usize {
                    let x = CHUNK[k];
                    CHUNK[k] = acc;
                    acc += x;
                }
            }
            thread::sync_threads();
            let mut id = CHUNK[tid as usize];
            let mut ei = e0;
            while ei < e1 {
                if IDS[ei as usize] != NO_VERTEX {
                    IDS[ei as usize] = id as u16;
                    let (o, axis) = edge_of(ei, b);
                    let (pos, rgb, nrm) = tc::mesh_edge_point::<O, Hood>(
                        &hood,
                        origin,
                        o,
                        axis as usize,
                        p.voxel_size,
                        p.half_voxel,
                        normals,
                        colors,
                    );
                    let (pos, nrm) = if p.transform != 0 {
                        (tc::xform::<O>(&xf.r, &xf.t, pos), tc::rot::<O>(&xf.r, nrm))
                    } else {
                        (pos, nrm)
                    };
                    let k = 3 * (vb + id as u64) as usize;
                    *out_vertices.add(k) = pos[0];
                    *out_vertices.add(k + 1) = pos[1];
                    *out_vertices.add(k + 2) = pos[2];
                    if normals {
                        *out_normals.add(k) = nrm[0];
                        *out_normals.add(k + 1) = nrm[1];
                        *out_normals.add(k + 2) = nrm[2];
                    }
                    if colors {
                        *out_colors.add(k) = rgb[0];
                        *out_colors.add(k + 1) = rgb[1];
                        *out_colors.add(k + 2) = rgb[2];
                    }
                    id += 1;
                }
                ei += 1;
            }
            thread::sync_threads();

            // triangles in cube order, then table order
            let cc = cubes.div_ceil(T);
            let (c0, c1) = (
                tid * cc,
                if (tid + 1) * cc < cubes {
                    (tid + 1) * cc
                } else {
                    cubes
                },
            );
            let mut cnt = 0u32;
            let mut c = c0;
            while c < c1 {
                cnt += tc::tri_count(CONF[c as usize] as usize);
                c += 1;
            }
            CHUNK[tid as usize] = cnt;
            thread::sync_threads();
            if tid == 0 {
                let mut acc = 0u32;
                for k in 0..T as usize {
                    let x = CHUNK[k];
                    CHUNK[k] = acc;
                    acc += x;
                }
            }
            thread::sync_threads();
            let ib = if p.global_index != 0 {
                p.index_base + vb as u32
            } else {
                0
            };
            let mut ti = tb + CHUNK[tid as usize] as u64;
            let mut c = c0;
            while c < c1 {
                let conf = CONF[c as usize] as usize;
                let l0 = tc::local_coords(c, p.log2);
                let row = &tc::TRI_TABLE[conf];
                let vid = |ek: i8| {
                    let ek = ek as usize;
                    let ow = tc::EDGE_OWNER[ek];
                    let o = [l0[0] + ow[0], l0[1] + ow[1], l0[2] + ow[2]];
                    ib + IDS[tc::edge_key(o, tc::EDGE_AXIS[ek], b as i32)] as u32
                };
                let mut j = 0;
                while j + 2 < 16 && row[j] >= 0 {
                    let k = 3 * ti as usize;
                    *out_triangles.add(k) = vid(row[j]);
                    *out_triangles.add(k + 1) = vid(row[j + 2]);
                    *out_triangles.add(k + 2) = vid(row[j + 1]);
                    ti += 1;
                    j += 3;
                }
                c += 1;
            }
            thread::sync_threads();
            e += thread::gridDim_x();
        }
    }
}

/// Incremental meshing: lists every allocated block whose mesh may have
/// changed since the cache epoch, i.e. the neighbours (`{-1, 0, 1}^3` with
/// normals, `{-1, 0}^3` without) of blocks modified after `cache_epoch`
/// or allocated at or after pool `cache_len` (contract section 12.6).
/// `dirty_mark` makes each block appear once; order is free.
///
/// # Safety
///
/// Hash as in the raycast (one header); `pool_keys`/`modified`/
/// `dirty_mark`/`dirty_list` hold `len` entries; `dirty_count` one.
#[kernel]
pub unsafe fn tsdf_mc_dirty(
    keys: *const u64,
    vals: *const u32,
    log2_cap: u32,
    pool_keys: *const u64,
    modified: *const u64,
    len: u32,
    cache_epoch: u64,
    cache_len: u32,
    normals: u32,
    dirty_mark: *mut u32,
    mark: u32,
    dirty_list: *mut u32,
    dirty_count: *mut u32,
) {
    // SAFETY: the launcher upholds the layout above.
    unsafe {
        let mut p = gtid();
        while p < len {
            if *modified.add(p as usize) > cache_epoch || p >= cache_len {
                let bc = tc::unpack_block_key(*pool_keys.add(p as usize));
                let n = if normals != 0 { 27 } else { 8 };
                for i in 0..n {
                    let d = if normals != 0 {
                        [
                            (i % 3) as i32 - 1,
                            ((i / 3) % 3) as i32 - 1,
                            (i / 9) as i32 - 1,
                        ]
                    } else {
                        [
                            -((i & 1) as i32),
                            -(((i >> 1) & 1) as i32),
                            -((i >> 2) as i32),
                        ]
                    };
                    let Some(nk) = tc::neighbour_key(bc, d) else {
                        continue;
                    };
                    if let Some(s) = tc::hash_find(log2_cap, nk, |s| *keys.add(s)) {
                        let q = *vals.add(s);
                        if DeviceAtomicU32::from_ptr(dirty_mark.add(q as usize)).swap(mark, R)
                            != mark
                        {
                            let at = DeviceAtomicU32::from_ptr(dirty_count).fetch_add(1, R);
                            *dirty_list.add(at as usize) = q;
                        }
                    }
                }
            }
            p += gstride();
        }
    }
}
