//! `wide` (portable SIMD: SSE2/AVX on x86, NEON on aarch64) kernel for one association pass.
//!
//! Same arithmetic as `for_each_correspondence` + `accumulate_level`, eight source points per
//! `f32x8`. The target lookup stays scalar per lane — `wide` has no gather, and the lane calls
//! `sample_target` the scalar path uses for bilinear, and an inlined copy of `sample_nearest`
//! for nearest. Everything after it (gates, residual, Jacobian row, sigma, Huber, the 21 + 6
//! sums) is vectorised.
//!
//! Lanes are `f32`, twice the width of `f64x4` and the maps' own precision. The sums are not:
//! each lane sums in `f32` over a block of [`BLOCK`] chunks only, then the block is added into
//! `f64` accumulators, so the rounding of a sum grows with the block, not with the image.
//! Results agree with the `f64` scalar kernel to `f32` rounding (see `simd_kernel_matches_scalar`),
//! and a projection that lands within rounding of a pixel boundary can round to the other pixel,
//! so correspondence counts can differ by a handful.

use super::{
    sample_target, DepthNoise, IcpPlaneCriteria, NormalEquations, RgbdLevel, TargetPx,
    MIN_RAY_INCIDENCE,
};
use crate::registration::rgbd::{is_valid_normal, is_valid_vertex};
use wide::f32x8;

/// Lanes per chunk.
const LANES: usize = 8;

/// Chunks summed in `f32` before the block is added into the `f64` sums: 128 points, so a lane
/// sum carries at most ~16 roundings of `f32` (~1e-6 relative), and the flush (29 x 8 adds in
/// `f64`) is a few percent of the block's work.
const BLOCK: usize = 16;

/// Floor on a row's sigma in metres. Keeps `1 / sigma^2` and its products with the Jacobian
/// (|J| up to ~10 m) and their block sums inside `f32` range (1e30 * 1e2 * 128 < 3.4e38); the
/// `f64` scalar kernel's floor, `sqrt(f64::MIN_POSITIVE)`, would overflow. Only a zero sigma
/// reaches it.
const SIGMA_FLOOR: f32 = 1e-15;

/// Eight valid source points, lane-major: one aligned load per field.
#[derive(Debug, Default, Clone, Copy)]
struct Chunk {
    x: [f32; LANES],
    y: [f32; LANES],
    z: [f32; LANES],
    nx: [f32; LANES],
    ny: [f32; LANES],
    nz: [f32; LANES],
    /// Depth sigma of the point: its map entry, else the noise model at its depth.
    sig: [f32; LANES],
}

/// Valid source points of one level, structure-of-arrays in chunks of eight. Refilled per call
/// into a reused buffer.
#[derive(Debug, Default, Clone)]
pub(crate) struct SourceSoa {
    chunks: Vec<Chunk>,
    /// Real point count; the last chunk is padded past it with inert lanes.
    len: usize,
}

impl SourceSoa {
    /// Refill from a level.
    pub(crate) fn fill(&mut self, src: &RgbdLevel, noise: &DepthNoise) {
        let pad = Chunk {
            z: [1.0; LANES],
            nz: [-1.0; LANES],
            sig: [1.0; LANES],
            ..Default::default()
        };
        // built on the stack and pushed whole: a resize-then-write doubled the memory traffic
        self.chunks.clear();
        let (mut c, mut n) = (pad, 0usize);
        for (i, (v, nn)) in src.vertices.iter().zip(src.normals.iter()).enumerate() {
            if !is_valid_vertex(v) || !is_valid_normal(nn) {
                continue;
            }
            let l = n % LANES;
            c.x[l] = v[0];
            c.y[l] = v[1];
            c.z[l] = v[2];
            c.nx[l] = nn[0];
            c.ny[l] = nn[1];
            c.nz[l] = nn[2];
            c.sig[l] = super::point_depth_sigma(src, i, v[2] as f64, noise) as f32;
            n += 1;
            if l == LANES - 1 {
                self.chunks.push(c);
                c = pad;
            }
        }
        if n % LANES != 0 {
            self.chunks.push(c);
        }
        self.len = n;
    }
}

#[inline]
fn mask_from(b: [bool; LANES]) -> f32x8 {
    f32x8::new(b.map(|x| if x { f32::from_bits(u32::MAX) } else { 0.0 }))
}

/// `max((n . ray)^2, MIN_RAY_INCIDENCE^2 |ray|^2)`, lane-wise.
#[inline]
fn projected_ray_sq(n: [f32x8; 3], ray: [f32x8; 3]) -> f32x8 {
    let c = n[0] * ray[0] + n[1] * ray[1] + n[2] * ray[2];
    let len_sq = ray[0] * ray[0] + ray[1] * ray[1] + ray[2] * ray[2];
    (c * c).max(f32x8::splat((MIN_RAY_INCIDENCE * MIN_RAY_INCIDENCE) as f32) * len_sq)
}

/// Number of block accumulators: 21 of `J^T W J`, 6 of `J^T W r`, the squared residual, `z`.
const NUM_SUMS: usize = 29;

/// Adds every lane of the block sums into the `f64` totals and clears the block.
#[inline]
fn flush(block: &mut [f32x8; NUM_SUMS], total: &mut [f64; NUM_SUMS]) {
    for (b, t) in block.iter_mut().zip(total.iter_mut()) {
        *t += b.to_array().iter().map(|&x| x as f64).sum::<f64>();
        *b = f32x8::splat(0.0);
    }
}

/// One association pass over `len` SoA points; the SIMD twin of `accumulate_level`.
///
/// `packed`, when given, is `tgt`'s [`super::PreparedTarget`] cache: nearest lookups read it,
/// bilinear ones still read `tgt`. `depths`, when given, receives the inlier depths (for the
/// exact median of the final pass).
// index loops: the 21 upper-triangle accumulators are addressed by (row, col)
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub(crate) fn accumulate(
    soa: &SourceSoa,
    src_has_map: bool,
    tgt: &RgbdLevel,
    packed: Option<&[TargetPx]>,
    rotation: &[[f64; 3]; 3],
    translation: &[f64; 3],
    criteria: &IcpPlaneCriteria,
    bilinear: bool,
    mut depths: Option<&mut Vec<f32>>,
) -> NormalEquations {
    let len = soa.len;
    let r = rotation.map(|row| row.map(|v| f32x8::splat(v as f32)));
    let t = translation.map(|v| f32x8::splat(v as f32));
    let intr = &tgt.intrinsics;
    let (tw, th) = (intr.width, intr.height);
    let sp = |v: f64| f32x8::splat(v as f32);
    let (fx, fy, cx, cy) = (sp(intr.fx), sp(intr.fy), sp(intr.cx), sp(intr.cy));
    let cos_max = sp(criteria.max_normal_angle_rad.cos());
    let max_dist_sq = sp(criteria.max_dist_m * criteria.max_dist_m);
    let k = sp(criteria.huber_delta);
    let ray_model = src_has_map
        || !tgt.depth_sigmas.is_empty()
        || !matches!(criteria.depth_noise, DepthNoise::Constant { .. });
    let constant_sigma = sp(match criteria.depth_noise {
        DepthNoise::Constant { sigma_m } => sigma_m,
        DepthNoise::Stereo { .. } => 0.0,
    });
    let zero = f32x8::splat(0.0);
    let one = f32x8::splat(1.0);
    let floor = f32x8::splat(SIGMA_FLOOR);

    let mut block = [zero; NUM_SUMS];
    let mut total = [0.0f64; NUM_SUMS];
    let mut in_block = 0;
    let (mut num_inliers, mut num_associated) = (0usize, 0usize);
    if let Some(d) = depths.as_deref_mut() {
        d.clear();
    }

    for (ci, ch) in soa.chunks.iter().enumerate() {
        let i = ci * LANES;
        let ps = [f32x8::new(ch.x), f32x8::new(ch.y), f32x8::new(ch.z)];
        let ns = [f32x8::new(ch.nx), f32x8::new(ch.ny), f32x8::new(ch.nz)];
        let rp = [
            r[0][0] * ps[0] + r[0][1] * ps[1] + r[0][2] * ps[2],
            r[1][0] * ps[0] + r[1][1] * ps[1] + r[1][2] * ps[2],
            r[2][0] * ps[0] + r[2][1] * ps[1] + r[2][2] * ps[2],
        ];
        let p = [rp[0] + t[0], rp[1] + t[1], rp[2] + t[2]];
        let iz = one / p[2];
        let u = (fx * p[0] * iz + cx).to_array();
        let v = (fy * p[1] * iz + cy).to_array();
        let pz = p[2].to_array();

        // scalar gather, the same samplers as the scalar kernel
        let mut ok = [false; LANES];
        let (mut vt, mut nt) = ([[0.0, 0.0, 1.0]; LANES], [[0.0, 0.0, -1.0]; LANES]);
        let mut sig_t = [1.0f32; LANES];
        let mut var_t = [1.0f32; LANES];
        for l in 0..LANES {
            if i + l >= len || pz[l] <= 0.0 {
                continue;
            }
            if !bilinear {
                // `sample_nearest`, inlined: the call and its `Option` were most of the pass
                if !(u[l] >= -0.5 && v[l] >= -0.5) {
                    continue;
                }
                let (ui, vi) = ((u[l] + 0.5) as usize, (v[l] + 0.5) as usize);
                if ui > tw - 1 || vi > th - 1 {
                    continue;
                }
                let idx = vi * tw + ui;
                if let Some(px) = packed {
                    let q = &px[idx];
                    // packed z is a valid depth or exactly 0, never NaN
                    if q.v[2] <= 0.0 {
                        continue;
                    }
                    ok[l] = true;
                    vt[l] = q.v;
                    nt[l] = q.n;
                    var_t[l] = q.var;
                    continue;
                }
                let (a, b) = (&tgt.vertices[idx], &tgt.normals[idx]);
                if !is_valid_vertex(a) || !is_valid_normal(b) {
                    continue;
                }
                ok[l] = true;
                vt[l] = *a;
                nt[l] = *b;
                if ray_model {
                    sig_t[l] =
                        super::point_depth_sigma(tgt, idx, a[2] as f64, &criteria.depth_noise)
                            as f32;
                }
                continue;
            }
            if let Some(s) = sample_target(tgt, u[l] as f64, v[l] as f64) {
                ok[l] = true;
                vt[l] = s.vertex.map(|c| c as f32);
                nt[l] = s.normal.map(|c| c as f32);
                sig_t[l] = s
                    .depth_sigma
                    .unwrap_or_else(|| criteria.depth_noise.depth_sigma(s.vertex[2]))
                    as f32;
            }
        }
        num_associated += ok.iter().filter(|&&b| b).count();
        if !ok.iter().any(|&b| b) {
            continue;
        }
        let lane = |a: &[[f32; 3]; LANES], c: usize| f32x8::new(std::array::from_fn(|l| a[l][c]));
        let vt = [lane(&vt, 0), lane(&vt, 1), lane(&vt, 2)];
        let nt = [lane(&nt, 0), lane(&nt, 1), lane(&nt, 2)];

        let diff = [p[0] - vt[0], p[1] - vt[1], p[2] - vt[2]];
        let d2 = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2];
        let nsr = [
            r[0][0] * ns[0] + r[0][1] * ns[1] + r[0][2] * ns[2],
            r[1][0] * ns[0] + r[1][1] * ns[1] + r[1][2] * ns[2],
            r[2][0] * ns[0] + r[2][1] * ns[1] + r[2][2] * ns[2],
        ];
        let cos_n = nsr[0] * nt[0] + nsr[1] * nt[1] + nsr[2] * nt[2];
        let mask = mask_from(ok) & d2.simd_le(max_dist_sq) & cos_n.simd_ge(cos_max);
        let bits = mask.to_bitmask();
        if bits == 0 {
            continue;
        }

        let res = nt[0] * diff[0] + nt[1] * diff[1] + nt[2] * diff[2];
        let j = [
            p[1] * nt[2] - p[2] * nt[1],
            p[2] * nt[0] - p[0] * nt[2],
            p[0] * nt[1] - p[1] * nt[0],
            nt[0],
            nt[1],
            nt[2],
        ];
        let sigma = if ray_model {
            let zs = ps[2];
            let ray_s = [rp[0] / zs, rp[1] / zs, rp[2] / zs];
            let ss = f32x8::new(ch.sig);
            let target_var = if packed.is_some() && !bilinear {
                f32x8::new(var_t)
            } else {
                let ray_t = [vt[0] / vt[2], vt[1] / vt[2], one];
                let st = f32x8::new(sig_t);
                projected_ray_sq(nt, ray_t) * st * st
            };
            (projected_ray_sq(nt, ray_s) * ss * ss + target_var).sqrt()
        } else {
            constant_sigma
        }
        .max(floor);
        let inv_sigma = one / sigma;
        let rn = (res * inv_sigma).abs();
        let huber = rn.simd_le(k).bitselect(one, k / rn);
        let w = mask.bitselect(huber * inv_sigma * inv_sigma, zero);
        let res = mask.bitselect(res, zero);

        let wj = j.map(|ji| w * ji);
        let mut n = 0;
        for a in 0..6 {
            block[21 + a] += wj[a] * res;
            for b in a..6 {
                block[n] += wj[a] * j[b];
                n += 1;
            }
        }
        block[27] += res * res;
        block[28] += mask.bitselect(p[2], zero);
        num_inliers += bits.count_ones() as usize;
        if let Some(d) = depths.as_deref_mut() {
            for (l, &zl) in p[2].to_array().iter().enumerate() {
                if bits & (1 << l) != 0 {
                    d.push(zl);
                }
            }
        }
        in_block += 1;
        if in_block == BLOCK {
            flush(&mut block, &mut total);
            in_block = 0;
        }
    }
    flush(&mut block, &mut total);

    let mut a = [[0.0; 6]; 6];
    let mut n = 0;
    for r in 0..6 {
        for c in r..6 {
            a[r][c] = total[n];
            a[c][r] = a[r][c];
            n += 1;
        }
    }
    NormalEquations {
        a,
        b: std::array::from_fn(|i| total[21 + i]),
        num_inliers,
        num_associated,
        num_valid: len,
        sum_sq_residual: total[27],
        depth_scale: super::depth_scale(total[28], num_inliers, depths),
    }
}
