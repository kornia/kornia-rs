//! Shared per-voxel and per-pixel TSDF math: one source for the CPU and the
//! GPU. The host lib loads this file through `#[path]` (edition 2021, stable,
//! `no_std`); the device crate compiles it as `mod tsdf_core` (edition 2024,
//! cuda-oxide). The op order is normative (kornia-3d TSDF contract): each
//! `+ - * /` is one IEEE binary32 op, fma and sqrt come only from `F32Ops`.
//!
//! Rules for this file:
//! - no `std`, no `alloc`, no `unsafe`, no nested file modules (the two
//!   loaders resolve them differently), edition-neutral code;
//! - no inherent `f32::{mul_add, sqrt, floor, min, max, clamp}`;
//! - every fn `#[inline(always)]`: the PTX gate rejects fma in a `.func`.

// Every gate is an ordered compare written so that NaN fails it; rewriting
// them as `contains`/`clamp`/iterators would change the contract's op order.
#![allow(
    clippy::neg_cmp_op_on_partial_ord,
    clippy::manual_range_contains,
    clippy::manual_clamp,
    clippy::needless_range_loop
)]

/// Float ops the shared core cannot spell itself (stable `no_std` has no
/// `f32::mul_add`/`sqrt`). Every impl must be correctly rounded.
pub trait F32Ops {
    /// `RN(a * b + c)`, one rounding.
    fn fma(a: f32, b: f32, c: f32) -> f32;
    /// Correctly rounded square root.
    fn sqrt(x: f32) -> f32;
}

/// Exact floor for finite `|x| < 2^31`; callers range-check first.
#[inline(always)]
pub fn floor_i32(x: f32) -> i32 {
    let t = x as i32;
    t - (((t as f32) > x) as i32)
}

// ---------------------------------------------------------------- grid ----

/// Voxel coordinates satisfy `|g| < VOXEL_LIMIT` on every axis.
pub const VOXEL_LIMIT: i32 = 1 << 22;
/// `VOXEL_LIMIT` as `f32` (exact).
pub const G_F: f32 = 4_194_304.0;
/// Bits per packed block coordinate.
pub const KEY_BITS: u32 = 21;
/// Bias that maps a block coordinate in `[-2^20, 2^20)` to `[0, 2^21)`.
pub const KEY_BIAS: i32 = 1 << 20;
/// Mask of one packed coordinate.
pub const KEY_MASK: u64 = (1 << KEY_BITS) - 1;
/// Empty hash slot; bit 63 is never set in a valid key.
pub const EMPTY_KEY: u64 = u64::MAX;
/// Fibonacci hashing multiplier of the block hash.
pub const HASH_MUL: u64 = 0x9E37_79B9_7F4A_7C15;

/// Grid constants, derived once on the host (`inv_*` in `f32`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Grid {
    /// Voxel edge in metres.
    pub voxel_size: f32,
    /// `voxel_size * 0.5` (exact).
    pub half_voxel: f32,
    /// `1.0 / voxel_size`.
    pub inv_voxel: f32,
    /// Truncation distance in metres.
    pub trunc: f32,
    /// `1.0 / trunc`.
    pub inv_trunc: f32,
    /// `log2` of the block edge in voxels (3 or 4).
    pub log2: u32,
}

/// Packs block coordinates in `[-2^20, 2^20)`; ascending keys are ascending
/// lexicographic `(x, y, z)`.
#[inline(always)]
pub fn pack_block_key(b: [i32; 3]) -> u64 {
    let f = |v: i32| ((v + KEY_BIAS) as u32 as u64) & KEY_MASK;
    (f(b[0]) << (2 * KEY_BITS)) | (f(b[1]) << KEY_BITS) | f(b[2])
}

/// Inverse of [`pack_block_key`].
#[inline(always)]
pub fn unpack_block_key(k: u64) -> [i32; 3] {
    let g = |s: u32| (((k >> s) & KEY_MASK) as i32) - KEY_BIAS;
    [g(2 * KEY_BITS), g(KEY_BITS), g(0)]
}

/// Voxel coordinate of `x` metres, `None` outside `|q| < 2^22` (or NaN).
#[inline(always)]
pub fn grid_coord(x: f32, inv_voxel: f32) -> Option<i32> {
    let q = x * inv_voxel;
    if !(q >= -G_F && q < G_F) {
        return None;
    }
    Some(floor_i32(q))
}

/// Voxel containing `p` (floor rule), `None` outside the domain.
#[inline(always)]
pub fn voxel_from_point(p: [f32; 3], inv_voxel: f32) -> Option<[i32; 3]> {
    Some([
        grid_coord(p[0], inv_voxel)?,
        grid_coord(p[1], inv_voxel)?,
        grid_coord(p[2], inv_voxel)?,
    ])
}

/// Centre of voxel `g`: `fma(g, voxel_size, half_voxel)` per axis, equal to
/// `(g + 0.5) * voxel_size` bit for bit under `|g| < 2^22`.
#[inline(always)]
pub fn voxel_centre<O: F32Ops>(g: [i32; 3], voxel_size: f32, half_voxel: f32) -> [f32; 3] {
    [
        O::fma(g[0] as f32, voxel_size, half_voxel),
        O::fma(g[1] as f32, voxel_size, half_voxel),
        O::fma(g[2] as f32, voxel_size, half_voxel),
    ]
}

/// In-block index `lx + B * (ly + B * lz)` of local coordinates.
#[inline(always)]
pub fn local_index(l: [i32; 3], log2: u32) -> usize {
    (l[0] | ((l[1] | (l[2] << log2)) << log2)) as usize
}

// ---------------------------------------------------------- transforms ----

/// Rigid transform in `f32`: row-major rotation and translation.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rigid {
    /// Row-major rotation, `r[3 * i + j] = R[i][j]`.
    pub r: [f32; 9],
    /// Translation.
    pub t: [f32; 3],
}

/// `R * p + t`; x (the fastest voxel index) is the outermost fma, so a
/// hoisted per-row partial gives the same bits.
#[inline(always)]
pub fn xform<O: F32Ops>(r: &[f32; 9], t: &[f32; 3], p: [f32; 3]) -> [f32; 3] {
    [
        O::fma(r[0], p[0], O::fma(r[1], p[1], O::fma(r[2], p[2], t[0]))),
        O::fma(r[3], p[0], O::fma(r[4], p[1], O::fma(r[5], p[2], t[1]))),
        O::fma(r[6], p[0], O::fma(r[7], p[1], O::fma(r[8], p[2], t[2]))),
    ]
}

/// `R * v`.
#[inline(always)]
pub fn rot<O: F32Ops>(r: &[f32; 9], v: [f32; 3]) -> [f32; 3] {
    [
        O::fma(r[0], v[0], O::fma(r[1], v[1], r[2] * v[2])),
        O::fma(r[3], v[0], O::fma(r[4], v[1], r[5] * v[2])),
        O::fma(r[6], v[0], O::fma(r[7], v[1], r[8] * v[2])),
    ]
}

/// Squared length, `fma(x, x, fma(y, y, z * z))`.
#[inline(always)]
pub fn norm2<O: F32Ops>(v: [f32; 3]) -> f32 {
    O::fma(v[0], v[0], O::fma(v[1], v[1], v[2] * v[2]))
}

/// `v / |v|` (`1 / sqrt` then three multiplies), zero when `|v|^2 <= 0`.
#[inline(always)]
pub fn normalize<O: F32Ops>(v: [f32; 3]) -> [f32; 3] {
    let l2 = norm2::<O>(v);
    if l2 > 0.0 {
        let inv = 1.0 / O::sqrt(l2);
        [v[0] * inv, v[1] * inv, v[2] * inv]
    } else {
        [0.0; 3]
    }
}

// --------------------------------------------------------------- depth ----

/// A raw depth sample (`u16` units or `f32`).
pub trait RawDepth: Copy {
    /// The sample in metres if it is a measurement inside
    /// `[range_min, range_max)`.
    fn metres(self, mpu: f32, range_min: f32, range_max: f32) -> Option<f32>;
}

impl RawDepth for u16 {
    #[inline(always)]
    fn metres(self, mpu: f32, range_min: f32, range_max: f32) -> Option<f32> {
        if self == 0 {
            return None;
        }
        let d = (self as f32) * mpu;
        if d >= range_min && d < range_max {
            Some(d)
        } else {
            None
        }
    }
}

impl RawDepth for f32 {
    #[inline(always)]
    fn metres(self, mpu: f32, range_min: f32, range_max: f32) -> Option<f32> {
        let d = self * mpu;
        // finite and positive, NaN-false
        if !(d > 0.0 && d < f32::INFINITY) {
            return None;
        }
        if d >= range_min && d < range_max {
            Some(d)
        } else {
            None
        }
    }
}

// ------------------------------------------------------------- pinhole ----

/// Per-observation pinhole constants (host-derived, passed verbatim).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PinholeParams {
    /// Volume to sensor.
    pub sensor_from_volume: Rigid,
    /// Sensor to volume.
    pub volume_from_sensor: Rigid,
    /// Focal length x (pixels).
    pub fx: f32,
    /// Focal length y (pixels).
    pub fy: f32,
    /// Principal point x (pixels).
    pub cx: f32,
    /// Principal point y (pixels).
    pub cy: f32,
    /// `(1 / fx) as f32`, from `f64`.
    pub inv_fx: f32,
    /// `(1 / fy) as f32`, from `f64`.
    pub inv_fy: f32,
    /// Image width.
    pub width: u32,
    /// Image height.
    pub height: u32,
    /// `width as f32` (exact).
    pub width_f: f32,
    /// `height as f32` (exact).
    pub height_f: f32,
    /// Metres per depth unit.
    pub mpu: f32,
    /// Smallest kept depth (metres, `> 0`).
    pub range_min: f32,
    /// Depth upper bound (metres, exclusive).
    pub range_max: f32,
}

/// Pixel index of sensor-frame point `s`: cheirality `z >= range_min`, then
/// the pixel rule `floor(coord + 0.5)` inside the image.
#[inline(always)]
pub fn pinhole_pixel<O: F32Ops>(p: &PinholeParams, s: [f32; 3]) -> Option<u32> {
    if !(s[2] >= p.range_min) {
        return None;
    }
    let iz = 1.0 / s[2];
    let u = O::fma(p.fx, s[0] * iz, p.cx);
    let v = O::fma(p.fy, s[1] * iz, p.cy);
    let uh = u + 0.5;
    let vh = v + 0.5;
    if !(uh >= 0.0 && uh < p.width_f && vh >= 0.0 && vh < p.height_f) {
        return None;
    }
    // trunc == floor for non-negative, range-checked values
    let (px, py) = (uh as i32, vh as i32);
    Some((py as u32) * p.width + (px as u32))
}

/// Projective signed distance `d - z` of the voxel centre `c` (volume frame)
/// and its depth-pixel index.
#[inline(always)]
pub fn pinhole_sdf<O: F32Ops, D: RawDepth>(
    p: &PinholeParams,
    depth: &[D],
    c: [f32; 3],
) -> Option<(f32, u32)> {
    let s = xform::<O>(&p.sensor_from_volume.r, &p.sensor_from_volume.t, c);
    let idx = pinhole_pixel::<O>(p, s)?;
    let d = depth[idx as usize].metres(p.mpu, p.range_min, p.range_max)?;
    Some((d - s[2], idx))
}

/// Sensor-frame point of pixel `(px, py)` at depth `d`.
#[inline(always)]
pub fn pinhole_point(p: &PinholeParams, px: u32, py: u32, d: f32) -> [f32; 3] {
    let xn = (px as f32 - p.cx) * p.inv_fx;
    let yn = (py as f32 - p.cy) * p.inv_fy;
    [xn * d, yn * d, d]
}

// ----------------------------------------------------------- spherical ----

/// Per-observation spherical LiDAR constants; the tables are passed as
/// slices next to it.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SphericalParams {
    /// Volume to sensor.
    pub sensor_from_volume: Rigid,
    /// Sensor to volume.
    pub volume_from_sensor: Rigid,
    /// `sin` of the lower field-of-view limit.
    pub fov_low: f32,
    /// `sin` of the upper field-of-view limit.
    pub fov_high: f32,
    /// `(1 / step) as f32`, from `f64`.
    pub inv_az_step: f32,
    /// `(-start / step) as f32`, from `f64`.
    pub az_offset: f32,
    /// Columns.
    pub width: u32,
    /// Rows (beams).
    pub height: u32,
    /// Non-zero: columns wrap around (full turn).
    pub wrap: u32,
    /// Metres per range unit.
    pub mpu: f32,
    /// Smallest kept range (metres, `> 0`).
    pub range_min: f32,
    /// Range upper bound (metres, exclusive).
    pub range_max: f32,
}

/// `#{ j : bnd[j] <= sz }` for strictly ascending `bnd` (binary search).
#[inline(always)]
pub fn count_le(bnd: &[f32], sz: f32) -> u32 {
    let mut lo = 0usize;
    let mut n = bnd.len();
    while n > 0 {
        let half = n / 2;
        if bnd[lo + half] <= sz {
            lo += half + 1;
            n -= half + 1;
        } else {
            n = half;
        }
    }
    lo as u32
}

const ATAN_C1: f32 = f32::from_bits(0xbeaa_a879); // -0.3333165943622589
const ATAN_C2: f32 = f32::from_bits(0x3e4c_6b08); //  0.19962704181671143
const ATAN_C3: f32 = f32::from_bits(0xbe0f_1ec6); // -0.13976582884788513
const ATAN_C4: f32 = f32::from_bits(0x3dc8_9602); //  0.09794236719608307
const ATAN_C5: f32 = f32::from_bits(0xbd6c_a408); // -0.05777361989021301
const ATAN_C6: f32 = f32::from_bits(0x3cbc_beb6); //  0.023040156811475754
const ATAN_C7: f32 = f32::from_bits(0xbb8e_b7d8); // -0.0043554119765758514
/// `pi / 2` rounded to `f32`.
pub const HALF_PI: f32 = f32::from_bits(0x3fc9_0fdb);
/// `pi` rounded to `f32`.
pub const PI: f32 = f32::from_bits(0x4049_0fdb);

/// Deterministic `atan2(y, x)` for finite inputs: octant reduction, then
/// `t + t^3 Q(t^2)` (minimax, degree 15). `(±0, x < 0)` gives `+pi`,
/// `(0, 0)` gives `0`.
#[inline(always)]
pub fn atan2_det<O: F32Ops>(y: f32, x: f32) -> f32 {
    let ax = f32::from_bits(x.to_bits() & 0x7fff_ffff);
    let ay = f32::from_bits(y.to_bits() & 0x7fff_ffff);
    let swap = ay > ax;
    let mx = if swap { ay } else { ax };
    let mn = if swap { ax } else { ay };
    let den = if mx > 0.0 { mx } else { 1.0 };
    let t = mn / den;
    let s = t * t;
    let mut q = ATAN_C7;
    q = O::fma(q, s, ATAN_C6);
    q = O::fma(q, s, ATAN_C5);
    q = O::fma(q, s, ATAN_C4);
    q = O::fma(q, s, ATAN_C3);
    q = O::fma(q, s, ATAN_C2);
    q = O::fma(q, s, ATAN_C1);
    let mut r = O::fma(t * s, q, t);
    if swap {
        r = HALF_PI - r;
    }
    if x < 0.0 {
        r = PI - r;
    }
    if y < 0.0 {
        r = -r;
    }
    r
}

/// Range and range-image index of sensor-frame point `s`: range gate, then
/// the vertical field of view, the beam row (`count_le` over the row
/// boundaries `bnd`) and the column rule.
#[inline(always)]
pub fn spherical_pixel<O: F32Ops>(
    p: &SphericalParams,
    bnd: &[f32],
    s: [f32; 3],
) -> Option<(f32, u32)> {
    let r = O::sqrt(norm2::<O>(s));
    // range gate before any division or atan2_det
    if !(r >= p.range_min) {
        return None;
    }
    let sz = s[2] / r;
    if !(p.fov_low <= sz && sz < p.fov_high) {
        return None;
    }
    let k = count_le(bnd, sz);
    let row = (p.height - 1) - k;
    let phi = atan2_det::<O>(s[1], s[0]);
    let u = O::fma(phi, p.inv_az_step, p.az_offset);
    let uh = u + 0.5;
    if !(uh >= -16_777_216.0 && uh < 16_777_216.0) {
        return None;
    }
    let ci = floor_i32(uh);
    let w = p.width as i32;
    let col = if p.wrap != 0 {
        ci.rem_euclid(w)
    } else if ci >= 0 && ci < w {
        ci
    } else {
        return None;
    };
    Some((r, row * p.width + col as u32))
}

/// Range-space signed distance `rho - |s|` of voxel centre `c` and its
/// range-image index.
#[inline(always)]
pub fn spherical_sdf<O: F32Ops, D: RawDepth>(
    p: &SphericalParams,
    bnd: &[f32],
    range: &[D],
    c: [f32; 3],
) -> Option<(f32, u32)> {
    let s = xform::<O>(&p.sensor_from_volume.r, &p.sensor_from_volume.t, c);
    let (r, idx) = spherical_pixel::<O>(p, bnd, s)?;
    let rho = range[idx as usize].metres(p.mpu, p.range_min, p.range_max)?;
    Some((rho - r, idx))
}

/// Sensor-frame point of a beam at range `d` from its direction tables.
#[inline(always)]
pub fn spherical_point(d: f32, cos_el: f32, sin_el: f32, cos_az: f32, sin_az: f32) -> [f32; 3] {
    [d * (cos_el * cos_az), d * (cos_el * sin_az), d * sin_el]
}

// --------------------------------------------------------------- touch ----

/// Block-coordinate box `[lo, hi]` touched by the band `pv ± trunc` of a
/// measured volume-frame point, `None` when it leaves the voxel domain.
#[inline(always)]
pub fn touch_box(pv: [f32; 3], g: &Grid) -> Option<([i32; 3], [i32; 3])> {
    let mut blo = [0i32; 3];
    let mut bhi = [0i32; 3];
    for a in 0..3 {
        let lo = (pv[a] - g.trunc) * g.inv_voxel;
        let hi = (pv[a] + g.trunc) * g.inv_voxel;
        if !(lo >= -G_F && hi < G_F) {
            return None;
        }
        blo[a] = floor_i32(lo) >> g.log2;
        bhi[a] = floor_i32(hi) >> g.log2;
    }
    Some((blo, bhi))
}

// -------------------------------------------------------------- update ----

/// Truncated, normalised distance in `[-1, 1]`; `None` behind the band
/// (`sdf < -trunc`) or NaN. Free space (`sdf >= trunc`) is exactly `1.0`.
#[inline(always)]
pub fn truncate(sdf: f32, trunc: f32, inv_trunc: f32) -> Option<f32> {
    if !(sdf >= -trunc) {
        return None;
    }
    let x = if sdf < trunc { sdf * inv_trunc } else { 1.0 };
    Some(if x > 1.0 {
        1.0
    } else if x < -1.0 {
        -1.0
    } else {
        x
    })
}

/// Running mean with the uncapped count `n = w0 + 1`:
/// `fma(tsd - old, 1 / n, old)`, weight capped at `max_weight`.
#[inline(always)]
pub fn update_tsdf<O: F32Ops>(old: f32, w0: u16, tsd: f32, max_weight: u16) -> (f32, u16) {
    let n = w0 as u32 + 1;
    let rc = 1.0 / (n as f32);
    let t = O::fma(tsd - old, rc, old);
    let w = if n > max_weight as u32 {
        max_weight
    } else {
        n as u16
    };
    (t, w)
}

/// `c + 0.5` rounded down, clamped to 255 explicitly (`c` in `[0, 255]`).
#[inline(always)]
pub fn round_u8(c: f32) -> u8 {
    let hc = c + 0.5;
    if hc >= 255.0 {
        255
    } else {
        (hc as u32) as u8
    }
}

/// Colour running mean on packed `r | g << 8 | b << 16 | weight << 24`.
#[inline(always)]
pub fn update_color<O: F32Ops>(s: u32, rgb: [u8; 3], max_color_weight: u8) -> u32 {
    let cw0 = s >> 24;
    let rc = 1.0 / ((cw0 + 1) as f32);
    let mut out = 0u32;
    for ch in 0..3 {
        let c_old = ((s >> (8 * ch)) & 0xff) as f32;
        let c_obs = rgb[ch] as f32;
        let c = O::fma(c_obs - c_old, rc, c_old);
        out |= (round_u8(c) as u32) << (8 * ch);
    }
    let cw = if cw0 + 1 > max_color_weight as u32 {
        max_color_weight as u32
    } else {
        cw0 + 1
    };
    out | (cw << 24)
}

/// RGB bytes of a packed colour.
#[inline(always)]
pub fn unpack_rgb(s: u32) -> [u8; 3] {
    [
        (s & 0xff) as u8,
        ((s >> 8) & 0xff) as u8,
        ((s >> 16) & 0xff) as u8,
    ]
}

// ---------------------------------------------------------------- hash ----

/// First probe slot of `key` in a table of `2^log2_cap` slots
/// (`1 <= log2_cap <= 63`).
#[inline(always)]
pub fn hash_slot(key: u64, log2_cap: u32) -> usize {
    (key.wrapping_mul(HASH_MUL) >> (64 - log2_cap)) as usize
}

/// Linear-probe lookup; `keys.len() == vals.len() == 2^log2_cap`.
#[inline(always)]
pub fn hash_lookup(keys: &[u64], vals: &[u32], log2_cap: u32, key: u64) -> Option<u32> {
    let mask = keys.len() - 1;
    let mut i = hash_slot(key, log2_cap);
    for _ in 0..keys.len() {
        let k = keys[i];
        if k == key {
            return Some(vals[i]);
        }
        if k == EMPTY_KEY {
            return None;
        }
        i = (i + 1) & mask;
    }
    None
}

/// Linear-probe insert of a key not in the table; `false` if it is full.
#[inline(always)]
pub fn hash_insert(keys: &mut [u64], vals: &mut [u32], log2_cap: u32, key: u64, val: u32) -> bool {
    let mask = keys.len() - 1;
    let mut i = hash_slot(key, log2_cap);
    for _ in 0..keys.len() {
        if keys[i] == EMPTY_KEY {
            keys[i] = key;
            vals[i] = val;
            return true;
        }
        i = (i + 1) & mask;
    }
    false
}

// ------------------------------------------------------ marching cubes ----

/// Cube corner offsets: c0 (0,0,0), c1 (1,0,0), c2 (1,1,0), c3 (0,1,0),
/// c4 (0,0,1), c5 (1,0,1), c6 (1,1,1), c7 (0,1,1).
pub const CORNER: [[i32; 3]; 8] = [
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
];

/// Owner (lower end) offset of each cube edge.
pub const EDGE_OWNER: [[i32; 3]; 12] = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 0],
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [0, 0, 1],
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
];

/// Axis (0 x, 1 y, 2 z) of each cube edge.
pub const EDGE_AXIS: [u32; 12] = [0, 1, 0, 1, 0, 1, 0, 1, 2, 2, 2, 2];

/// Cube configuration: bit `c` set when corner `c` is inside (`f < 0`).
#[inline(always)]
pub fn cube_index(f: &[f32; 8]) -> usize {
    let mut ci = 0usize;
    for c in 0..8 {
        if f[c] < 0.0 {
            ci |= 1 << c;
        }
    }
    ci
}

/// Index `3 * (o_x + (B + 1) * (o_y + (B + 1) * o_z)) + axis` of the block
/// edge owned by local voxel `o` in `[0, B]^3`.
#[inline(always)]
pub fn edge_key(o: [i32; 3], axis: u32, b: i32) -> usize {
    (3 * (o[0] + (b + 1) * (o[1] + (b + 1) * o[2])) + axis as i32) as usize
}

/// Crossing parameter from the owner `a` to `b`: `fa / (fa - fb)` (signs
/// differ, so the denominator is non-zero).
#[inline(always)]
pub fn edge_t(fa: f32, fb: f32) -> f32 {
    fa / (fa - fb)
}

/// `fma(t, b - a, a)`.
#[inline(always)]
pub fn lerp<O: F32Ops>(a: f32, b: f32, t: f32) -> f32 {
    O::fma(t, b - a, a)
}

/// Edge vertex: the owner centre `ca` moved `t * voxel_size` along `axis`.
#[inline(always)]
pub fn edge_vertex<O: F32Ops>(ca: [f32; 3], axis: usize, voxel_size: f32, t: f32) -> [f32; 3] {
    let mut p = ca;
    p[axis] = O::fma(t, voxel_size, ca[axis]);
    p
}

/// Per-channel colour interpolation, rounded half up.
#[inline(always)]
pub fn lerp_color<O: F32Ops>(t: f32, ca: [u8; 3], cb: [u8; 3]) -> [u8; 3] {
    let mut out = [0u8; 3];
    for ch in 0..3 {
        let a = ca[ch] as f32;
        out[ch] = round_u8(O::fma(t, cb[ch] as f32 - a, a));
    }
    out
}

/// Interpolated, renormalised vertex normal from the owner and far-end
/// gradients (toward `+tsdf`).
#[inline(always)]
pub fn edge_normal<O: F32Ops>(t: f32, ga: [f32; 3], gb: [f32; 3]) -> [f32; 3] {
    normalize::<O>([
        O::fma(t, gb[0] - ga[0], ga[0]),
        O::fma(t, gb[1] - ga[1], ga[1]),
        O::fma(t, gb[2] - ga[2], ga[2]),
    ])
}

/// One gradient component at a valid voxel: central difference, else the
/// one-sided difference toward the valid neighbour, else `0`.
#[inline(always)]
pub fn grad_axis(plus: Option<f32>, centre: f32, minus: Option<f32>) -> f32 {
    match (plus, minus) {
        (Some(p), Some(m)) => p - m,
        (Some(p), None) => p - centre,
        (None, Some(m)) => centre - m,
        (None, None) => 0.0,
    }
}

// ------------------------------------------------------------- raycast ----

/// Per-call raycast constants (host-derived).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RaycastParams {
    /// Camera to volume.
    pub volume_from_camera: Rigid,
    /// Rotation of volume to camera.
    pub camera_from_volume_r: [f32; 9],
    /// Principal point x (pixels).
    pub cx: f32,
    /// Principal point y (pixels).
    pub cy: f32,
    /// `(1 / fx) as f32`.
    pub inv_fx: f32,
    /// `(1 / fy) as f32`.
    pub inv_fy: f32,
    /// March start (camera z, metres).
    pub depth_min: f32,
    /// March end (camera z, metres, exclusive).
    pub depth_max: f32,
    /// Voxel edge (metres).
    pub voxel_size: f32,
    /// `1.0 / voxel_size`.
    pub inv_voxel: f32,
    /// Truncation distance (metres).
    pub trunc: f32,
    /// `B as f32 * voxel_size`: the step over an unallocated block.
    pub block_step: f32,
    /// Smallest weight of a valid voxel (`>= 1`).
    pub weight_threshold: u32,
    /// Hard cap on march steps.
    pub max_steps: u32,
}

/// Read access to stored voxels for the raycast.
pub trait VoxelSampler {
    /// `(tsdf, weight)` of voxel `g`, `None` when its block is unallocated.
    fn sample(&self, g: [i32; 3]) -> Option<(f32, u16)>;
}

/// A raycast hit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayHit {
    /// Camera z of the hit (metres).
    pub z: f32,
    /// Camera-frame vertex `[xn * z, yn * z, z]`.
    pub vertex: [f32; 3],
    /// Camera-frame unit normal facing the camera, zero when invalid.
    pub normal: [f32; 3],
    /// Volume-frame hit point `ray(z)`.
    pub point: [f32; 3],
}

/// Trilinear gradient of the 8 corner values (`f[x + 2y + 4z]`) at the
/// fractional offset `al`.
#[inline(always)]
pub fn trilinear_gradient<O: F32Ops>(f: &[f32; 8], al: [f32; 3]) -> [f32; 3] {
    let (f000, f100, f010, f110) = (f[0], f[1], f[2], f[3]);
    let (f001, f101, f011, f111) = (f[4], f[5], f[6], f[7]);
    let gx = lerp::<O>(
        lerp::<O>(f100 - f000, f110 - f010, al[1]),
        lerp::<O>(f101 - f001, f111 - f011, al[1]),
        al[2],
    );
    let gy = lerp::<O>(
        lerp::<O>(f010 - f000, f110 - f100, al[0]),
        lerp::<O>(f011 - f001, f111 - f101, al[0]),
        al[2],
    );
    let gz = lerp::<O>(
        lerp::<O>(f001 - f000, f101 - f100, al[0]),
        lerp::<O>(f011 - f010, f111 - f110, al[0]),
        al[1],
    );
    [gx, gy, gz]
}

#[inline(always)]
fn valid_sample<S: VoxelSampler>(s: &S, g: [i32; 3], threshold: u32) -> Option<f32> {
    match s.sample(g) {
        Some((f, w)) if w as u32 >= threshold => Some(f),
        _ => None,
    }
}

/// Camera-frame normal at volume point `p` from the trilinear gradient of
/// the 8 surrounding voxel centres, facing the camera; zero unless all 8
/// are valid.
#[inline(always)]
pub fn raycast_normal<O: F32Ops, S: VoxelSampler>(
    rp: &RaycastParams,
    s: &S,
    p: [f32; 3],
    vertex: [f32; 3],
) -> [f32; 3] {
    let mut g0 = [0i32; 3];
    let mut al = [0f32; 3];
    for a in 0..3 {
        let q = O::fma(p[a], rp.inv_voxel, -0.5);
        if !(q >= -G_F && q < G_F - 1.0) {
            return [0.0; 3];
        }
        g0[a] = floor_i32(q);
        al[a] = q - g0[a] as f32;
    }
    let mut f = [0f32; 8];
    for i in 0..8 {
        let g = [
            g0[0] + (i & 1) as i32,
            g0[1] + ((i >> 1) & 1) as i32,
            g0[2] + ((i >> 2) & 1) as i32,
        ];
        match valid_sample(s, g, rp.weight_threshold) {
            Some(v) => f[i] = v,
            None => return [0.0; 3],
        }
    }
    let gc = rot::<O>(&rp.camera_from_volume_r, trilinear_gradient::<O>(&f, al));
    let mut n = normalize::<O>(gc);
    if O::fma(n[0], vertex[0], O::fma(n[1], vertex[1], n[2] * vertex[2])) > 0.0 {
        n = [-n[0], -n[1], -n[2]];
    }
    n
}

/// Marches the ray of pixel `(px, py)` from `depth_min` to the first
/// `+ -> -` zero crossing of valid voxels; `None` on a miss.
#[inline(always)]
pub fn raycast_pixel<O: F32Ops, S: VoxelSampler>(
    rp: &RaycastParams,
    s: &S,
    px: u32,
    py: u32,
) -> Option<RayHit> {
    let xn = (px as f32 - rp.cx) * rp.inv_fx;
    let yn = (py as f32 - rp.cy) * rp.inv_fy;
    let dir = rot::<O>(&rp.volume_from_camera.r, [xn, yn, 1.0]);
    let t = rp.volume_from_camera.t;
    let ray = |z: f32| {
        [
            O::fma(dir[0], z, t[0]),
            O::fma(dir[1], z, t[1]),
            O::fma(dir[2], z, t[2]),
        ]
    };
    let mut z = rp.depth_min;
    // (f, z) of the previous valid sample
    let mut prev: Option<(f32, f32)> = None;
    let mut hit: Option<(f32, f32, f32, f32)> = None;
    // Unallocated samples below `fine_until` step one voxel, not one block.
    let mut fine_until = rp.depth_min;
    // The last step was a block skip.
    let mut skipped = false;
    for _ in 0..rp.max_steps {
        if !(z < rp.depth_max) {
            break;
        }
        let sample = match voxel_from_point(ray(z), rp.inv_voxel) {
            Some(g) => s.sample(g),
            None => None,
        };
        match sample {
            None => {
                if z < fine_until {
                    z += rp.voxel_size;
                    skipped = false;
                } else {
                    z += rp.block_step;
                    skipped = true;
                }
                prev = None;
            }
            Some((_, w)) if (w as u32) < rp.weight_threshold => {
                z += rp.voxel_size;
                prev = None;
                skipped = false;
            }
            // A block skip landed behind the surface: re-march the skipped
            // block one voxel at a time, else the crossing is never seen.
            Some((f, _)) if skipped && f < 0.0 => {
                fine_until = z;
                z = (z - rp.block_step) + rp.voxel_size;
                skipped = false;
            }
            Some((f, _)) => {
                skipped = false;
                if let Some((fp, zp)) = prev {
                    if fp >= 0.0 && f < 0.0 {
                        hit = Some((fp, zp, f, z));
                        break;
                    }
                }
                prev = Some((f, z));
                let st = f * rp.trunc;
                z += if st > rp.voxel_size {
                    st
                } else {
                    rp.voxel_size
                };
            }
        }
    }
    let (fp, zp, f, zc) = hit?;
    let a = fp / (fp - f);
    let z_hit = O::fma(a, zc - zp, zp);
    let vertex = [xn * z_hit, yn * z_hit, z_hit];
    let point = ray(z_hit);
    let normal = raycast_normal::<O, S>(rp, s, point, vertex);
    Some(RayHit {
        z: z_hit,
        vertex,
        normal,
        point,
    })
}

// ------------------------------------------------------------- tables ----

// Lorensen-Cline marching-cubes tables (P. Bourke's public-domain listing, the
// corner and edge numbering above; as in Open3D's MarchingCubesConst.h).

/// Edges crossed by each cube configuration (bit k = edge k).
pub const EDGE_TABLE: [u16; 256] = [
    0x000, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c, 0x80c, 0x905, 0xa0f, 0xb06, 0xc0a,
    0xd03, 0xe09, 0xf00, 0x190, 0x099, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c, 0x99c, 0x895,
    0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90, 0x230, 0x339, 0x033, 0x13a, 0x636, 0x73f, 0x435,
    0x53c, 0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30, 0x3a0, 0x2a9, 0x1a3, 0x0aa,
    0x7a6, 0x6af, 0x5a5, 0x4ac, 0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0, 0x460,
    0x569, 0x663, 0x76a, 0x066, 0x16f, 0x265, 0x36c, 0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963,
    0xa69, 0xb60, 0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0x0ff, 0x3f5, 0x2fc, 0xdfc, 0xcf5, 0xfff,
    0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0, 0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x055, 0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950, 0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6,
    0x2cf, 0x1c5, 0x0cc, 0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0, 0x8c0, 0x9c9,
    0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc, 0x0cc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9,
    0x7c0, 0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c, 0x15c, 0x055, 0x35f, 0x256,
    0x55a, 0x453, 0x759, 0x650, 0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc, 0x2fc,
    0x3f5, 0x0ff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0, 0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f,
    0xd65, 0xc6c, 0x36c, 0x265, 0x16f, 0x066, 0x76a, 0x663, 0x569, 0x460, 0xca0, 0xda9, 0xea3,
    0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac, 0x4ac, 0x5a5, 0x6af, 0x7a6, 0x0aa, 0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c, 0x53c, 0x435, 0x73f, 0x636, 0x13a,
    0x033, 0x339, 0x230, 0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c, 0x69c, 0x795,
    0x49f, 0x596, 0x29a, 0x393, 0x099, 0x190, 0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905,
    0x80c, 0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x000,
];

/// Triangles of each cube configuration as edge triples, `-1` terminated.
#[rustfmt::skip]
pub const TRI_TABLE: [[i8; 16]; 256] = [
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1],
    [3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1],
    [3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1],
    [3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1],
    [9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1],
    [9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1],
    [2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1],
    [8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1],
    [9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1],
    [4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1],
    [3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1],
    [1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1],
    [4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1],
    [4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1],
    [9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1],
    [5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1],
    [2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1],
    [9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1],
    [0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1],
    [2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1],
    [10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1],
    [4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1],
    [5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1],
    [5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1],
    [9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1],
    [0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1],
    [1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1],
    [10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1],
    [8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1],
    [2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1],
    [7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1],
    [9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1],
    [2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1],
    [11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1],
    [9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1],
    [5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1],
    [11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1],
    [11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1],
    [1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1],
    [9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1],
    [5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1],
    [2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1],
    [0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1],
    [5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1],
    [6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1],
    [3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1],
    [6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1],
    [5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1],
    [1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1],
    [10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1],
    [6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1],
    [8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1],
    [7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1],
    [3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1],
    [5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1],
    [0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1],
    [9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1],
    [8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1],
    [5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1],
    [0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1],
    [6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1],
    [10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1],
    [10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1],
    [8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1],
    [1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1],
    [3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1],
    [0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1],
    [10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1],
    [3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1],
    [6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1],
    [9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1],
    [8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1],
    [3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1],
    [6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1],
    [0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1],
    [10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1],
    [10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1],
    [2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1],
    [7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1],
    [7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1],
    [2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1],
    [1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1],
    [11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1],
    [8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1],
    [0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1],
    [7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1],
    [10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1],
    [2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1],
    [6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1],
    [7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1],
    [2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1],
    [1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1],
    [10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1],
    [10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1],
    [0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1],
    [7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1],
    [6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1],
    [8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1],
    [9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1],
    [6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1],
    [4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1],
    [10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1],
    [8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1],
    [0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1],
    [1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1],
    [8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1],
    [10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1],
    [4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1],
    [10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1],
    [5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1],
    [11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1],
    [9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1],
    [6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1],
    [7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1],
    [3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1],
    [7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1],
    [9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1],
    [3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1],
    [6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1],
    [9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1],
    [1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1],
    [4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1],
    [7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1],
    [6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1],
    [3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1],
    [0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1],
    [6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1],
    [0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1],
    [11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1],
    [6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1],
    [5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1],
    [9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1],
    [1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1],
    [1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1],
    [10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1],
    [0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1],
    [5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1],
    [10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1],
    [11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1],
    [9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1],
    [7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1],
    [2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1],
    [8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1],
    [9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1],
    [9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1],
    [1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1],
    [9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1],
    [9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1],
    [5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1],
    [0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1],
    [10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1],
    [2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1],
    [0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1],
    [0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1],
    [9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1],
    [5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1],
    [3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1],
    [5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1],
    [8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1],
    [0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1],
    [9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1],
    [1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1],
    [3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1],
    [4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1],
    [9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1],
    [11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1],
    [11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1],
    [2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1],
    [9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1],
    [3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1],
    [1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1],
    [4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1],
    [4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1],
    [0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1],
    [3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1],
    [3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1],
    [0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1],
    [9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1],
    [1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
];
