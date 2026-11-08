//! `wide` (portable SIMD: SSE2/AVX on x86, NEON on aarch64) kernel for one association pass.
//!
//! Same arithmetic as `for_each_correspondence` + `accumulate_level`, four source points per
//! `f64x4`. The target lookup stays scalar per lane — `wide` has no gather, and the lane calls
//! `sample_target` the scalar path uses for bilinear, and an inlined copy of `sample_nearest`
//! for nearest, so both kernels pick the same correspondences. Everything after it (gates, residual, Jacobian row, sigma, Huber, the
//! 21 + 6 sums) is vectorised. Sums run in a different order than the scalar loop, so results
//! agree to rounding, not bit for bit.

use super::{
    sample_target, DepthNoise, IcpPlaneCriteria, NormalEquations, RgbdLevel, MIN_RAY_INCIDENCE,
};
use crate::registration::rgbd::{is_valid_normal, is_valid_vertex};
use wide::f64x4;

/// Four valid source points, lane-major: one aligned load per field.
#[derive(Debug, Default, Clone, Copy)]
struct Chunk {
    x: [f64; 4],
    y: [f64; 4],
    z: [f64; 4],
    nx: [f64; 4],
    ny: [f64; 4],
    nz: [f64; 4],
    /// Depth sigma of the point: its map entry, else the noise model at its depth.
    sig: [f64; 4],
}

/// Valid source points of one level, structure-of-arrays in chunks of four, `f64` so the hot
/// loop loads lanes directly. Refilled per call into a reused buffer.
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
            z: [1.0; 4],
            nz: [-1.0; 4],
            sig: [1.0; 4],
            ..Default::default()
        };
        // built on the stack and pushed whole: a resize-then-write doubled the memory traffic
        self.chunks.clear();
        let (mut c, mut n) = (pad, 0usize);
        for (i, (v, nn)) in src.vertices.iter().zip(src.normals.iter()).enumerate() {
            if !is_valid_vertex(v) || !is_valid_normal(nn) {
                continue;
            }
            let l = n % 4;
            c.x[l] = v[0] as f64;
            c.y[l] = v[1] as f64;
            c.z[l] = v[2] as f64;
            c.nx[l] = nn[0] as f64;
            c.ny[l] = nn[1] as f64;
            c.nz[l] = nn[2] as f64;
            c.sig[l] = super::point_depth_sigma(src, i, v[2] as f64, noise);
            n += 1;
            if l == 3 {
                self.chunks.push(c);
                c = pad;
            }
        }
        if n % 4 != 0 {
            self.chunks.push(c);
        }
        self.len = n;
    }
}

#[inline]
fn mask_from(b: [bool; 4]) -> f64x4 {
    let m = |x: bool| if x { f64::from_bits(u64::MAX) } else { 0.0 };
    f64x4::new([m(b[0]), m(b[1]), m(b[2]), m(b[3])])
}

/// `max((n . ray)^2, MIN_RAY_INCIDENCE^2 |ray|^2)`, lane-wise.
#[inline]
fn projected_ray_sq(n: [f64x4; 3], ray: [f64x4; 3]) -> f64x4 {
    let c = n[0] * ray[0] + n[1] * ray[1] + n[2] * ray[2];
    let len_sq = ray[0] * ray[0] + ray[1] * ray[1] + ray[2] * ray[2];
    (c * c).max(f64x4::splat(MIN_RAY_INCIDENCE * MIN_RAY_INCIDENCE) * len_sq)
}

/// One association pass over `len` SoA points; the SIMD twin of `accumulate_level`.
///
/// `depths`, when given, receives the inlier depths (for the exact median of the final pass).
// index loops: the 21 upper-triangle accumulators are addressed by (row, col)
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub(crate) fn accumulate(
    soa: &SourceSoa,
    src_has_map: bool,
    tgt: &RgbdLevel,
    rotation: &[[f64; 3]; 3],
    translation: &[f64; 3],
    criteria: &IcpPlaneCriteria,
    bilinear: bool,
    mut depths: Option<&mut Vec<f32>>,
) -> NormalEquations {
    let len = soa.len;
    let r = rotation.map(|row| row.map(f64x4::splat));
    let t = translation.map(f64x4::splat);
    let intr = &tgt.intrinsics;
    let (tw, th) = (intr.width, intr.height);
    let (fx, fy, cx, cy) = (
        f64x4::splat(intr.fx),
        f64x4::splat(intr.fy),
        f64x4::splat(intr.cx),
        f64x4::splat(intr.cy),
    );
    let cos_max = f64x4::splat(criteria.max_normal_angle_rad.cos());
    let max_dist_sq = f64x4::splat(criteria.max_dist_m * criteria.max_dist_m);
    let k = f64x4::splat(criteria.huber_delta);
    let ray_model = src_has_map
        || !tgt.depth_sigmas.is_empty()
        || !matches!(criteria.depth_noise, DepthNoise::Constant { .. });
    let constant_sigma = f64x4::splat(match criteria.depth_noise {
        DepthNoise::Constant { sigma_m } => sigma_m,
        DepthNoise::Stereo { .. } => 0.0,
    });
    let zero = f64x4::splat(0.0);
    let one = f64x4::splat(1.0);
    let tiny = f64x4::splat(f64::MIN_POSITIVE.sqrt());

    let mut acc = [zero; 21];
    let mut accb = [zero; 6];
    let (mut sum_sq, mut sum_z) = (zero, zero);
    let (mut num_inliers, mut num_associated) = (0usize, 0usize);
    if let Some(d) = depths.as_deref_mut() {
        d.clear();
    }

    for (ci, ch) in soa.chunks.iter().enumerate() {
        let i = ci * 4;
        let ps = [f64x4::new(ch.x), f64x4::new(ch.y), f64x4::new(ch.z)];
        let ns = [f64x4::new(ch.nx), f64x4::new(ch.ny), f64x4::new(ch.nz)];
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
        let mut ok = [false; 4];
        let (mut vt, mut nt) = ([[0.0, 0.0, 1.0]; 4], [[0.0, 0.0, -1.0]; 4]);
        let mut sig_t = [1.0f64; 4];
        for l in 0..4 {
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
                let (a, b) = (&tgt.vertices[idx], &tgt.normals[idx]);
                if !is_valid_vertex(a) || !is_valid_normal(b) {
                    continue;
                }
                ok[l] = true;
                vt[l] = [a[0] as f64, a[1] as f64, a[2] as f64];
                nt[l] = [b[0] as f64, b[1] as f64, b[2] as f64];
                if ray_model {
                    sig_t[l] = super::point_depth_sigma(tgt, idx, vt[l][2], &criteria.depth_noise);
                }
                continue;
            }
            if let Some(s) = sample_target(tgt, u[l], v[l]) {
                ok[l] = true;
                vt[l] = s.vertex;
                nt[l] = s.normal;
                sig_t[l] = s
                    .depth_sigma
                    .unwrap_or_else(|| criteria.depth_noise.depth_sigma(s.vertex[2]));
            }
        }
        num_associated += ok.iter().filter(|&&b| b).count();
        if !ok.iter().any(|&b| b) {
            continue;
        }
        let lane = |a: &[[f64; 3]; 4], c: usize| f64x4::new([a[0][c], a[1][c], a[2][c], a[3][c]]);
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
            let ray_t = [vt[0] / vt[2], vt[1] / vt[2], one];
            let ss = f64x4::new(ch.sig);
            let st = f64x4::new(sig_t);
            (projected_ray_sq(nt, ray_s) * ss * ss + projected_ray_sq(nt, ray_t) * st * st).sqrt()
        } else {
            constant_sigma
        }
        .max(tiny);
        let inv_sigma = one / sigma;
        let rn = (res * inv_sigma).abs();
        let huber = rn.simd_le(k).blend(one, k / rn);
        let w = mask.blend(huber * inv_sigma * inv_sigma, zero);
        let res = mask.blend(res, zero);

        let wj = j.map(|ji| w * ji);
        let mut n = 0;
        for a in 0..6 {
            accb[a] += wj[a] * res;
            for b in a..6 {
                acc[n] += wj[a] * j[b];
                n += 1;
            }
        }
        sum_sq += res * res;
        sum_z += mask.blend(p[2], zero);
        num_inliers += bits.count_ones() as usize;
        if let Some(d) = depths.as_deref_mut() {
            let z = p[2].to_array();
            for (l, &zl) in z.iter().enumerate() {
                if bits & (1 << l) != 0 {
                    d.push(zl as f32);
                }
            }
        }
    }

    let mut a = [[0.0; 6]; 6];
    let mut n = 0;
    for r in 0..6 {
        for c in r..6 {
            a[r][c] = acc[n].reduce_add();
            a[c][r] = a[r][c];
            n += 1;
        }
    }
    NormalEquations {
        a,
        b: accb.map(|x| x.reduce_add()),
        num_inliers,
        num_associated,
        num_valid: len,
        sum_sq_residual: sum_sq.reduce_add(),
        depth_scale: super::depth_scale(sum_z.reduce_add(), num_inliers, depths),
    }
}
