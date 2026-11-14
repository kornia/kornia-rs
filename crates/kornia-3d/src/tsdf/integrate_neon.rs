//! NEON voxel pass: four voxels of a block x-row per register through the
//! shared core's lane math ([`Lanes`], [`LaneOps`] with `vfmaq_f32` and
//! `vsqrtq_f32`), so every lane holds the scalar pass's bits. Samples are
//! read per active lane by the core's scalar tail; a gated-out lane never
//! indexes an image.

// Every NEON intrinsic is an `unsafe fn`; this module is compiled only where
// NEON is enabled at build time.
#![allow(unsafe_code)]

use core::arch::aarch64::*;
use core::ops::{Add, BitAnd, Div, Mul, Sub};

use kornia_oxide_kernels::tsdf_core::{self as tc, LaneOps, Lanes, RawDepth};

use super::alloc::ObsKind;
use super::integrate::{Block, PageMut};
use super::{StdOps, TsdfParams};

/// Voxels per register; `B` (8 or 16) is a multiple, so rows have no tail.
const LANES: usize = 4;
const _: () = assert!(8 % LANES == 0 && 16 % LANES == 0);

/// Calls a register-only NEON intrinsic.
macro_rules! neon {
    ($e:expr) => {
        // SAFETY: the module is built only with the `neon` target feature,
        // so the intrinsic exists; it reads and writes registers only.
        unsafe { $e }
    };
}

/// Four `f32` lanes.
#[derive(Clone, Copy)]
pub(crate) struct Vf(float32x4_t);

/// Four `u32` lanes.
#[derive(Clone, Copy)]
pub(crate) struct Vu(uint32x4_t);

/// Four `i32` lanes.
#[derive(Clone, Copy)]
pub(crate) struct Vi(int32x4_t);

/// Four lane masks (all ones = set).
#[derive(Clone, Copy)]
pub(crate) struct Vm(uint32x4_t);

impl Add for Vf {
    type Output = Self;
    #[inline(always)]
    fn add(self, o: Self) -> Self {
        Self(neon!(vaddq_f32(self.0, o.0)))
    }
}

impl Sub for Vf {
    type Output = Self;
    #[inline(always)]
    fn sub(self, o: Self) -> Self {
        Self(neon!(vsubq_f32(self.0, o.0)))
    }
}

impl Mul for Vf {
    type Output = Self;
    #[inline(always)]
    fn mul(self, o: Self) -> Self {
        Self(neon!(vmulq_f32(self.0, o.0)))
    }
}

impl Div for Vf {
    type Output = Self;
    #[inline(always)]
    fn div(self, o: Self) -> Self {
        Self(neon!(vdivq_f32(self.0, o.0)))
    }
}

impl Add for Vu {
    type Output = Self;
    #[inline(always)]
    fn add(self, o: Self) -> Self {
        Self(neon!(vaddq_u32(self.0, o.0)))
    }
}

impl BitAnd for Vm {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, o: Self) -> Self {
        Self(neon!(vandq_u32(self.0, o.0)))
    }
}

impl Lanes for Vf {
    type M = Vm;
    type U = Vu;
    type I = Vi;

    #[inline(always)]
    fn splat(x: f32) -> Self {
        Self(neon!(vdupq_n_f32(x)))
    }

    #[inline(always)]
    fn splat_u(x: u32) -> Vu {
        Vu(neon!(vdupq_n_u32(x)))
    }

    #[inline(always)]
    fn lt(a: Self, b: Self) -> Vm {
        Vm(neon!(vcltq_f32(a.0, b.0)))
    }

    #[inline(always)]
    fn le(a: Self, b: Self) -> Vm {
        Vm(neon!(vcleq_f32(a.0, b.0)))
    }

    #[inline(always)]
    fn gt(a: Self, b: Self) -> Vm {
        Vm(neon!(vcgtq_f32(a.0, b.0)))
    }

    #[inline(always)]
    fn ge(a: Self, b: Self) -> Vm {
        Vm(neon!(vcgeq_f32(a.0, b.0)))
    }

    #[inline(always)]
    fn gt_u(a: Vu, b: Vu) -> Vm {
        Vm(neon!(vcgtq_u32(a.0, b.0)))
    }

    #[inline(always)]
    fn select(m: Vm, a: Self, b: Self) -> Self {
        Self(neon!(vbslq_f32(m.0, a.0, b.0)))
    }

    #[inline(always)]
    fn select_u(m: Vm, a: Vu, b: Vu) -> Vu {
        Vu(neon!(vbslq_u32(m.0, a.0, b.0)))
    }

    #[inline(always)]
    fn abs(x: Self) -> Self {
        Self(neon!(vabsq_f32(x.0)))
    }

    #[inline(always)]
    fn neg(x: Self) -> Self {
        Self(neon!(vnegq_f32(x.0)))
    }

    #[inline(always)]
    fn from_u32(x: Vu) -> Self {
        Self(neon!(vcvtq_f32_u32(x.0)))
    }

    #[inline(always)]
    fn from_i32(x: Vi) -> Self {
        Self(neon!(vcvtq_f32_s32(x.0)))
    }
}

/// The NEON fma and sqrt: `vfmaq_f32` (addend first) and `vsqrtq_f32`,
/// both rounded once.
pub(crate) struct NeonOps;

impl LaneOps for NeonOps {
    type V = Vf;

    #[inline(always)]
    fn fma(a: Vf, b: Vf, c: Vf) -> Vf {
        Vf(neon!(vfmaq_f32(c.0, a.0, b.0)))
    }

    #[inline(always)]
    fn sqrt(x: Vf) -> Vf {
        Vf(neon!(vsqrtq_f32(x.0)))
    }
}

impl Vf {
    #[inline(always)]
    pub(crate) fn to_array(self) -> [f32; LANES] {
        let mut a = [0f32; LANES];
        // SAFETY: `a` holds `LANES` writable `f32`.
        unsafe { vst1q_f32(a.as_mut_ptr(), self.0) };
        a
    }

    #[inline(always)]
    pub(crate) fn from_array(a: [f32; LANES]) -> Self {
        // SAFETY: `a` holds `LANES` readable `f32`.
        Self(unsafe { vld1q_f32(a.as_ptr()) })
    }
}

impl Vm {
    #[inline(always)]
    fn any(self) -> bool {
        neon!(vmaxvq_u32(self.0)) != 0
    }

    #[inline(always)]
    pub(crate) fn to_array(self) -> [u32; LANES] {
        u32_lanes(self.0)
    }

    #[inline(always)]
    fn from_array(a: [u32; LANES]) -> Self {
        Self(u32_from_lanes(a))
    }
}

#[cfg(test)]
impl Vu {
    #[inline(always)]
    pub(crate) fn to_array(self) -> [u32; LANES] {
        u32_lanes(self.0)
    }

    #[inline(always)]
    pub(crate) fn from_array(a: [u32; LANES]) -> Self {
        Self(u32_from_lanes(a))
    }
}

#[inline(always)]
fn u32_lanes(v: uint32x4_t) -> [u32; LANES] {
    let mut a = [0u32; LANES];
    // SAFETY: `a` holds `LANES` writable `u32`.
    unsafe { vst1q_u32(a.as_mut_ptr(), v) };
    a
}

#[inline(always)]
fn u32_from_lanes(a: [u32; LANES]) -> uint32x4_t {
    // SAFETY: `a` holds `LANES` readable `u32`.
    unsafe { vld1q_u32(a.as_ptr()) }
}

/// `LANES` consecutive voxels of a page.
struct Chunk<'a> {
    tsdf: &'a mut [f32; LANES],
    weight: &'a mut [u16; LANES],
    rgbw: Option<&'a mut [u32; LANES]>,
}

/// Walks block `key` in pool order (`x` fastest), `LANES` voxels at a time,
/// and calls `chunk` with their sensor-frame centres. The y and z terms of
/// the transform are hoisted per row (same bits: `xform = xform_x ∘
/// xform_yz`). Returns the sum of `chunk`'s counts.
#[inline(always)]
fn block_rows(
    grid: &tc::Grid,
    key: u64,
    sensor_from_volume: &tc::Rigid,
    page: PageMut<'_>,
    mut chunk: impl FnMut([Vf; 3], Chunk<'_>) -> u64,
) -> u64 {
    let (r, t) = (&sensor_from_volume.r, &sensor_from_volume.t);
    let b = 1i32 << grid.log2;
    let bc = tc::unpack_block_key(key);
    let g0 = [bc[0] * b, bc[1] * b, bc[2] * b];
    let (vs, hv) = (grid.voxel_size, grid.half_voxel);
    const IOTA: [i32; LANES] = [0, 1, 2, 3];
    // SAFETY: `IOTA` holds `LANES` readable `i32`.
    let iota = unsafe { vld1q_s32(IOTA.as_ptr()) };
    // a short page would silently skip voxels; `B³` is a multiple of LANES
    let voxels = (b as usize).pow(3);
    assert!(
        page.tsdf.len() == voxels
            && page.weight.len() == voxels
            && page.rgbw.as_ref().is_none_or(|c| c.len() == voxels),
        "page of {voxels} voxels expected"
    );
    let mut tsdf = page.tsdf.as_chunks_mut::<LANES>().0.iter_mut();
    let mut weight = page.weight.as_chunks_mut::<LANES>().0.iter_mut();
    let mut rgbw = page.rgbw.map(|c| c.as_chunks_mut::<LANES>().0.iter_mut());
    let mut updated = 0u64;
    for lz in 0..b {
        let cz = tc::voxel_axis::<StdOps>(g0[2] + lz, vs, hv);
        for ly in 0..b {
            let cy = tc::voxel_axis::<StdOps>(g0[1] + ly, vs, hv);
            let a = tc::xform_yz::<StdOps>(r, t, cy, cz).map(Vf::splat);
            for x0 in (0..b).step_by(LANES) {
                let (Some(tc4), Some(wc4)) = (tsdf.next(), weight.next()) else {
                    unreachable!("page lengths checked above");
                };
                let rc4 = rgbw
                    .as_mut()
                    .map(|it| it.next().expect("page lengths checked above"));
                let gx = Vi(neon!(vaddq_s32(vdupq_n_s32(g0[0] + x0), iota)));
                let cx = tc::voxel_axis::<NeonOps>(gx, vs, hv);
                let s = tc::xform_x::<NeonOps>(r, a, cx);
                let c = Chunk {
                    tsdf: tc4,
                    weight: wc4,
                    rgbw: rc4,
                };
                updated += chunk(s, c);
            }
        }
    }
    updated
}

/// Folds the sampled distances `sdf` of the lanes in `hit` into `c`:
/// truncation, running mean and weight cap in lanes; colour per lane.
/// Lanes outside `hit` or behind the band keep their bits.
#[inline(always)]
fn update(
    c: Chunk<'_>,
    hit: Vm,
    sdf: Vf,
    grid: &tc::Grid,
    params: &TsdfParams,
    color: Option<(&[u8], [u32; LANES])>,
) -> u64 {
    let (keep, tsd) = tc::truncate_lanes::<Vf>(sdf, grid.trunc, grid.inv_trunc);
    let ok = hit & keep;
    if !ok.any() {
        return 0;
    }
    // SAFETY: `c.tsdf` and `c.weight` hold `LANES` entries each.
    let (old, w0) = unsafe {
        (
            Vf(vld1q_f32(c.tsdf.as_ptr())),
            Vu(vmovl_u16(vld1_u16(c.weight.as_ptr()))),
        )
    };
    let (t, w) = tc::update_tsdf_lanes::<NeonOps>(old, w0, tsd, params.max_weight);
    let t = Vf::select(ok, t, old);
    // `w <= max_weight`, so the narrowing is exact
    let w = neon!(vmovn_u32(Vf::select_u(ok, w, w0).0));
    // SAFETY: as above, now written.
    unsafe {
        vst1q_f32(c.tsdf.as_mut_ptr(), t.0);
        vst1_u16(c.weight.as_mut_ptr(), w);
    }
    let okl = ok.to_array();
    if let (Some(rgbw), Some((rgb, idx))) = (c.rgbw, color) {
        for l in 0..LANES {
            if okl[l] != 0 {
                let k = 3 * idx[l] as usize;
                rgbw[l] = tc::update_color::<StdOps>(
                    rgbw[l],
                    [rgb[k], rgb[k + 1], rgb[k + 2]],
                    params.max_color_weight,
                );
            }
        }
    }
    okl.iter().filter(|&&m| m != 0).count() as u64
}

/// Updates the voxels of one block; returns how many it updated
/// (bit-identical to the scalar block pass).
pub(crate) fn block<D: RawDepth>(b: &Block<'_, '_>, depth: &[D], page: PageMut<'_>) -> u64 {
    match b.kind {
        ObsKind::Pinhole(p) => pinhole_block(b, p, depth, page),
        ObsKind::Spherical(p, lidar) => spherical_block(b, p, lidar.boundaries(), depth, page),
    }
}

#[inline(always)]
fn pinhole_block<D: RawDepth>(
    b: &Block<'_, '_>,
    p: &tc::PinholeParams,
    depth: &[D],
    page: PageMut<'_>,
) -> u64 {
    block_rows(b.grid, b.key, &p.sensor_from_volume, page, |s, c| {
        let (ok, uh, vh) = tc::pinhole_coords::<NeonOps>(p, s);
        if !ok.any() {
            return 0;
        }
        let (okl, uh, vh, z) = (ok.to_array(), uh.to_array(), vh.to_array(), s[2].to_array());
        let mut sdf = [0f32; LANES];
        let mut idx = [0u32; LANES];
        let mut hit = [0u32; LANES];
        for l in 0..LANES {
            if okl[l] == 0 {
                continue;
            }
            if let Some((d, k)) = tc::pinhole_sample(p, depth, uh[l], vh[l], z[l]) {
                (sdf[l], idx[l], hit[l]) = (d, k, u32::MAX);
            }
        }
        let color = b.color.map(|rgb| (rgb, idx));
        update(
            c,
            Vm::from_array(hit),
            Vf::from_array(sdf),
            b.grid,
            b.params,
            color,
        )
    })
}

#[inline(always)]
fn spherical_block<D: RawDepth>(
    b: &Block<'_, '_>,
    p: &tc::SphericalParams,
    bnd: &[f32],
    range: &[D],
    page: PageMut<'_>,
) -> u64 {
    block_rows(b.grid, b.key, &p.sensor_from_volume, page, |s, c| {
        let sc = tc::spherical_coords::<NeonOps>(p, s);
        if !sc.ok.any() {
            return 0;
        }
        let (okl, r, sz, uh) = (
            sc.ok.to_array(),
            sc.r.to_array(),
            sc.sz.to_array(),
            sc.uh.to_array(),
        );
        let mut sdf = [0f32; LANES];
        let mut hit = [0u32; LANES];
        for l in 0..LANES {
            if okl[l] == 0 {
                continue;
            }
            if let Some((d, _)) = tc::spherical_sample(p, bnd, range, r[l], sz[l], uh[l]) {
                (sdf[l], hit[l]) = (d, u32::MAX);
            }
        }
        update(
            c,
            Vm::from_array(hit),
            Vf::from_array(sdf),
            b.grid,
            b.params,
            None,
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const GRID: tc::Grid = tc::Grid {
        voxel_size: 0.1,
        half_voxel: 0.05,
        inv_voxel: 10.0,
        trunc: 0.4,
        inv_trunc: 2.5,
        log2: 3,
    };

    /// Runs `block_rows` over a page with the given lengths; returns the
    /// number of chunks visited.
    fn walk(tsdf: usize, weight: usize, rgbw: Option<usize>) -> usize {
        let (mut t, mut w) = (vec![0f32; tsdf], vec![0u16; weight]);
        let mut c = rgbw.map(|n| vec![0u32; n]);
        let page = PageMut {
            tsdf: &mut t,
            weight: &mut w,
            rgbw: c.as_deref_mut(),
        };
        let id = tc::Rigid {
            r: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            t: [0.0; 3],
        };
        let mut chunks = 0;
        block_rows(&GRID, tc::pack_block_key([0, 0, 0]), &id, page, |_, _| {
            chunks += 1;
            0
        });
        chunks
    }

    #[test]
    fn block_rows_visits_every_chunk_of_a_full_page() {
        assert_eq!(walk(512, 512, Some(512)), 512 / LANES);
    }

    #[test]
    #[should_panic(expected = "page of 512 voxels expected")]
    fn block_rows_rejects_a_short_tsdf_page() {
        walk(5, 512, None);
    }

    #[test]
    #[should_panic(expected = "page of 512 voxels expected")]
    fn block_rows_rejects_a_short_weight_page() {
        walk(512, 508, None);
    }

    #[test]
    #[should_panic(expected = "page of 512 voxels expected")]
    fn block_rows_rejects_a_short_colour_page() {
        walk(512, 512, Some(256));
    }
}
