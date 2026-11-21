//! `F32x4` and `F32x8` against `wide::f32x8` on a point-to-plane ICP association pass.
//!
//! One generic kernel, monomorphised per lane type, so all three run the same math: transform by
//! `R, t`, pinhole projection, image-bounds, distance and normal-angle gates as masks, the
//! point-to-plane residual, a ray-model sigma, the Huber weight, and the 21 upper-triangle
//! `J^T W J` and 6 `J^T W r` sums. Lanes sum in `f32` over a block of [`BLOCK_POINTS`] points,
//! then the block goes into `f64` totals. The target lookup is not benched: each source point
//! comes with its already-gathered target vertex, normal and variance.

use criterion::{criterion_group, criterion_main, Criterion};
use kornia_algebra::simd::{F32x4, F32x8, Mask32x4, Mask32x8};
use std::hint::black_box;
use std::ops::{Add, AddAssign, BitAnd, Div, Mul, Sub};
use wide::f32x8;

/// The 160x120 frame.
const WIDTH: f32 = 160.0;
const HEIGHT: f32 = 120.0;
const NUM_POINTS: usize = 19200;
/// Points summed in `f32` lanes before the block is added into `f64`.
const BLOCK_POINTS: usize = 128;
/// 21 of `J^T W J`, 6 of `J^T W r`.
const NUM_SUMS: usize = 27;
const MIN_RAY_INCIDENCE: f32 = 0.1;
const SIGMA_FLOOR: f32 = 1e-15;

trait Lanes:
    Copy + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> + AddAssign
{
    type Mask: Copy + BitAnd<Output = Self::Mask>;
    const N: usize;
    fn splat(v: f32) -> Self;
    fn load(src: &[f32]) -> Self;
    fn sqrt(self) -> Self;
    fn abs(self) -> Self;
    fn max(self, rhs: Self) -> Self;
    fn lt(self, rhs: Self) -> Self::Mask;
    fn le(self, rhs: Self) -> Self::Mask;
    fn gt(self, rhs: Self) -> Self::Mask;
    fn ge(self, rhs: Self) -> Self::Mask;
    fn select(mask: Self::Mask, if_true: Self, if_false: Self) -> Self;
    fn bitmask(mask: Self::Mask) -> u32;
    fn sum_f64(self) -> f64;
}

impl Lanes for F32x4 {
    type Mask = Mask32x4;
    const N: usize = 4;
    #[inline(always)]
    fn splat(v: f32) -> Self {
        F32x4::splat(v)
    }
    #[inline(always)]
    fn load(src: &[f32]) -> Self {
        F32x4::from_slice(src).unwrap_or_default()
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        F32x4::sqrt(self)
    }
    #[inline(always)]
    fn abs(self) -> Self {
        F32x4::abs(self)
    }
    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        F32x4::max(self, rhs)
    }
    #[inline(always)]
    fn lt(self, rhs: Self) -> Mask32x4 {
        self.cmplt(rhs)
    }
    #[inline(always)]
    fn le(self, rhs: Self) -> Mask32x4 {
        self.cmple(rhs)
    }
    #[inline(always)]
    fn gt(self, rhs: Self) -> Mask32x4 {
        self.cmpgt(rhs)
    }
    #[inline(always)]
    fn ge(self, rhs: Self) -> Mask32x4 {
        self.cmpge(rhs)
    }
    #[inline(always)]
    fn select(mask: Mask32x4, if_true: Self, if_false: Self) -> Self {
        F32x4::select(mask, if_true, if_false)
    }
    #[inline(always)]
    fn bitmask(mask: Mask32x4) -> u32 {
        mask.bitmask()
    }
    #[inline(always)]
    fn sum_f64(self) -> f64 {
        self.to_array().iter().map(|&x| x as f64).sum()
    }
}

impl Lanes for F32x8 {
    type Mask = Mask32x8;
    const N: usize = 8;
    #[inline(always)]
    fn splat(v: f32) -> Self {
        F32x8::splat(v)
    }
    #[inline(always)]
    fn load(src: &[f32]) -> Self {
        F32x8::from_slice(src).unwrap_or_default()
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        F32x8::sqrt(self)
    }
    #[inline(always)]
    fn abs(self) -> Self {
        F32x8::abs(self)
    }
    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        F32x8::max(self, rhs)
    }
    #[inline(always)]
    fn lt(self, rhs: Self) -> Mask32x8 {
        self.cmplt(rhs)
    }
    #[inline(always)]
    fn le(self, rhs: Self) -> Mask32x8 {
        self.cmple(rhs)
    }
    #[inline(always)]
    fn gt(self, rhs: Self) -> Mask32x8 {
        self.cmpgt(rhs)
    }
    #[inline(always)]
    fn ge(self, rhs: Self) -> Mask32x8 {
        self.cmpge(rhs)
    }
    #[inline(always)]
    fn select(mask: Mask32x8, if_true: Self, if_false: Self) -> Self {
        F32x8::select(mask, if_true, if_false)
    }
    #[inline(always)]
    fn bitmask(mask: Mask32x8) -> u32 {
        mask.bitmask()
    }
    #[inline(always)]
    fn sum_f64(self) -> f64 {
        self.to_array().iter().map(|&x| x as f64).sum()
    }
}

impl Lanes for f32x8 {
    type Mask = f32x8;
    const N: usize = 8;
    #[inline(always)]
    fn splat(v: f32) -> Self {
        f32x8::splat(v)
    }
    #[inline(always)]
    fn load(src: &[f32]) -> Self {
        f32x8::new(src.first_chunk::<8>().copied().unwrap_or_default())
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        f32x8::sqrt(self)
    }
    #[inline(always)]
    fn abs(self) -> Self {
        f32x8::abs(self)
    }
    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        f32x8::max(self, rhs)
    }
    #[inline(always)]
    fn lt(self, rhs: Self) -> f32x8 {
        self.simd_lt(rhs)
    }
    #[inline(always)]
    fn le(self, rhs: Self) -> f32x8 {
        self.simd_le(rhs)
    }
    #[inline(always)]
    fn gt(self, rhs: Self) -> f32x8 {
        self.simd_gt(rhs)
    }
    #[inline(always)]
    fn ge(self, rhs: Self) -> f32x8 {
        self.simd_ge(rhs)
    }
    #[inline(always)]
    fn select(mask: f32x8, if_true: Self, if_false: Self) -> Self {
        mask.bitselect(if_true, if_false)
    }
    #[inline(always)]
    fn bitmask(mask: f32x8) -> u32 {
        mask.to_bitmask()
    }
    #[inline(always)]
    fn sum_f64(self) -> f64 {
        self.to_array().iter().map(|&x| x as f64).sum()
    }
}

/// Source and gathered target points, structure-of-arrays.
struct Soa {
    src: [Vec<f32>; 7],
    tgt: [Vec<f32>; 7],
}

/// Deterministic synthetic frame: a slanted plane seen from 1-3 m, targets offset by a few cm,
/// with some outliers, some lanes behind the camera and some NaN target vertices.
fn make_soa() -> Soa {
    let mut state = 0x2545_f491u32;
    let mut rnd = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state as f32 / u32::MAX as f32
    };
    let mut src: [Vec<f32>; 7] = Default::default();
    let mut tgt: [Vec<f32>; 7] = Default::default();
    for i in 0..NUM_POINTS {
        let (u, v) = ((i % 160) as f32, (i / 160) as f32);
        let z = 1.0 + 2.0 * rnd();
        let p = [(u - 80.0) / 100.0 * z, (v - 60.0) / 100.0 * z, z];
        let n = [0.2f32, -0.1, -0.97];
        let off = if i % 17 == 0 { 0.5 } else { 0.03 };
        let mut q = [p[0] + off * (rnd() - 0.5), p[1], p[2] + off * (rnd() - 0.5)];
        if i % 101 == 0 {
            q[2] = f32::NAN;
        }
        let pz = if i % 53 == 0 { -p[2] } else { p[2] };
        for (c, val) in [p[0], p[1], pz, n[0], n[1], n[2], 0.01 * z]
            .into_iter()
            .enumerate()
        {
            src[c].push(val);
        }
        let nt = [n[0] + 0.05 * rnd(), n[1], n[2]];
        for (c, val) in [q[0], q[1], q[2], nt[0], nt[1], nt[2], 1e-4 * z * z]
            .into_iter()
            .enumerate()
        {
            tgt[c].push(val);
        }
    }
    Soa { src, tgt }
}

#[inline(always)]
fn dot<L: Lanes>(a: [L; 3], b: [L; 3]) -> L {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// The 27 normal-equation sums and the inlier count of one pass.
// index loops: the 21 upper-triangle accumulators are addressed by (row, col)
#[allow(clippy::needless_range_loop)]
#[inline(always)]
fn accumulate<L: Lanes>(soa: &Soa, r: [[f32; 3]; 3], t: [f32; 3]) -> ([f64; NUM_SUMS], u32) {
    let sp = L::splat;
    let r = r.map(|row| row.map(sp));
    let t = t.map(sp);
    let (fx, fy, cx, cy) = (sp(100.0), sp(100.0), sp(80.0), sp(60.0));
    let (zero, one, half) = (sp(0.0), sp(1.0), sp(-0.5));
    let (w_max, h_max) = (sp(WIDTH - 0.5), sp(HEIGHT - 0.5));
    let cos_max = sp(0.9);
    let max_dist_sq = sp(0.1 * 0.1);
    let k = sp(0.01);
    let incidence_sq = sp(MIN_RAY_INCIDENCE * MIN_RAY_INCIDENCE);
    let floor = sp(SIGMA_FLOOR);
    let chunks_per_block = BLOCK_POINTS / L::N;

    let mut block = [zero; NUM_SUMS];
    let mut total = [0.0f64; NUM_SUMS];
    let mut inliers = 0u32;
    let flush = |block: &mut [L; NUM_SUMS], total: &mut [f64; NUM_SUMS]| {
        for (b, t) in block.iter_mut().zip(total.iter_mut()) {
            *t += b.sum_f64();
            *b = zero;
        }
    };
    // fixed-length views, so the per-chunk loads carry no bounds check
    fn fixed(v: &[Vec<f32>; 7]) -> [&[f32; NUM_POINTS]; 7] {
        v.each_ref()
            .map(|x| x.first_chunk::<NUM_POINTS>().expect("NUM_POINTS samples"))
    }
    let (src, tgt) = (fixed(&soa.src), fixed(&soa.tgt));
    for ci in 0..NUM_POINTS / L::N {
        let i = ci * L::N;
        let s = |c: usize| L::load(&src[c][i..i + L::N]);
        let g = |c: usize| L::load(&tgt[c][i..i + L::N]);
        let (ps, ns) = ([s(0), s(1), s(2)], [s(3), s(4), s(5)]);
        let rp = [dot(r[0], ps), dot(r[1], ps), dot(r[2], ps)];
        let p = [rp[0] + t[0], rp[1] + t[1], rp[2] + t[2]];
        let iz = one / p[2];
        let u = fx * p[0] * iz + cx;
        let v = fy * p[1] * iz + cy;
        let in_img = p[2].gt(zero) & u.ge(half) & u.lt(w_max) & v.ge(half) & v.lt(h_max);

        let (vt, nt) = ([g(0), g(1), g(2)], [g(3), g(4), g(5)]);
        let diff = [p[0] - vt[0], p[1] - vt[1], p[2] - vt[2]];
        let nsr = [dot(r[0], ns), dot(r[1], ns), dot(r[2], ns)];
        let mask = in_img & dot(diff, diff).le(max_dist_sq) & dot(nsr, nt).ge(cos_max);
        let bits = L::bitmask(mask);
        if bits != 0 {
            let res = dot(nt, diff);
            let j = [
                p[1] * nt[2] - p[2] * nt[1],
                p[2] * nt[0] - p[0] * nt[2],
                p[0] * nt[1] - p[1] * nt[0],
                nt[0],
                nt[1],
                nt[2],
            ];
            let ray_s = [rp[0] / ps[2], rp[1] / ps[2], rp[2] / ps[2]];
            let c = dot(nt, ray_s);
            let ray_sq = (c * c).max(incidence_sq * dot(ray_s, ray_s));
            let ss = s(6);
            let sigma = (ray_sq * ss * ss + g(6)).sqrt().max(floor);
            let inv_sigma = one / sigma;
            let rn = (res * inv_sigma).abs();
            let huber = L::select(rn.le(k), one, k / rn);
            let w = L::select(mask, huber * inv_sigma * inv_sigma, zero);
            let res = L::select(mask, res, zero);
            let wj = j.map(|ji| w * ji);
            let mut n = 0;
            for a in 0..6 {
                block[21 + a] += wj[a] * res;
                for b in a..6 {
                    block[n] += wj[a] * j[b];
                    n += 1;
                }
            }
            inliers += bits.count_ones();
        }
        if (ci + 1) % chunks_per_block == 0 {
            flush(&mut block, &mut total);
        }
    }
    flush(&mut block, &mut total);
    (total, inliers)
}

fn bench_simd_lanes(c: &mut Criterion) {
    let soa = make_soa();
    let (cz, sz) = (0.02f32.cos(), 0.02f32.sin());
    let r = [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]];
    let t = [0.01, -0.005, 0.02];

    let (s8, n8) = accumulate::<f32x8>(&soa, r, t);
    let (s4, n4) = accumulate::<F32x4>(&soa, r, t);
    let (k8, m8) = accumulate::<F32x8>(&soa, r, t);
    assert_eq!(n8, n4, "inlier counts differ");
    assert_eq!(n8, m8, "inlier counts differ");
    assert!(n4 > 0, "no inliers");
    let max_rel = [s4, k8]
        .iter()
        .flat_map(|s| s8.iter().zip(s))
        .map(|(a, b)| (a - b).abs() / a.abs().max(1e-12))
        .fold(0.0f64, f64::max);
    assert!(
        max_rel < 1e-5,
        "sums differ, max relative error {max_rel:e}"
    );
    println!("inliers {n4} of {NUM_POINTS}, max relative sum difference {max_rel:e}");

    let mut group = c.benchmark_group("icp_association_19200");
    group.bench_function("wide_f32x8", |b| {
        b.iter(|| accumulate::<f32x8>(black_box(&soa), black_box(r), black_box(t)))
    });
    group.bench_function("kornia_f32x4", |b| {
        b.iter(|| accumulate::<F32x4>(black_box(&soa), black_box(r), black_box(t)))
    });
    group.bench_function("kornia_f32x8", |b| {
        b.iter(|| accumulate::<F32x8>(black_box(&soa), black_box(r), black_box(t)))
    });
    group.finish();
}

criterion_group!(benches, bench_simd_lanes);
criterion_main!(benches);
