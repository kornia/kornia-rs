//! Post-processing: NMS, top-K, descriptor sampling at sparse keypoint
//! coordinates. Mirrors upstream `Xfeat.detectAndCompute`.

use crate::ops;
use rayon::prelude::*;

/// One detected keypoint with score, in **original image** coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KeyPoint {
    /// Sub-pixel x in original image coords (after rescaling by `rw`).
    pub x: f32,
    /// Sub-pixel y in original image coords.
    pub y: f32,
    /// Final detection score = `K1h(x, y) * H1(x, y)` (heatmap × reliability).
    pub score: f32,
}

/// Run NMS on a `(H, W)` keypoint heatmap, sample reliability at each surviving
/// peak, sort by combined score, take top-K.
#[allow(clippy::too_many_arguments)]
pub fn nms_topk(
    heatmap: &[f32],
    reliability: &[f32],
    h: usize,
    w: usize,
    h_rel: usize,
    w_rel: usize,
    score_threshold: f32,
    top_k: usize,
    rw: f32,
    rh: f32,
) -> Vec<KeyPoint> {
    debug_assert_eq!(heatmap.len(), h * w);
    debug_assert_eq!(reliability.len(), h_rel * w_rel);

    let raw = ops::nms_maxpool_5x5_equality(heatmap, h, w, score_threshold);

    // Match PyTorch's InterpolateSparse2d(bilinear) coordinate convention:
    //   normgrid: g = 2*x/(W-1) - 1
    //   grid_sample(align_corners=False): pos = (g+1)/2 * W_rel - 0.5
    //   → pos = x * W_rel / (W-1) - 0.5
    let x_scale = (w_rel as f32) / ((w as f32) - 1.0);
    let y_scale = (h_rel as f32) / ((h as f32) - 1.0);

    // Sampling reliability at each NMS peak is independent per candidate.
    let mut kps: Vec<KeyPoint> = raw
        .into_par_iter()
        .map(|(kp_score, idx)| {
            let y = (idx / w) as f32;
            let x = (idx % w) as f32;
            let rel = bilinear_sample_single(
                reliability,
                w_rel,
                h_rel,
                x * x_scale - 0.5,
                y * y_scale - 0.5,
            );
            KeyPoint {
                x: x * rw,
                y: y * rh,
                score: kp_score * rel,
            }
        })
        .collect();

    // Sort descending, then truncate.
    kps.sort_unstable_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if kps.len() > top_k {
        kps.truncate(top_k);
    }
    kps
}

fn bilinear_sample_single(buf: &[f32], w: usize, h: usize, x: f32, y: f32) -> f32 {
    let x0 = x.floor();
    let y0 = y.floor();
    let wx = x - x0;
    let wy = y - y0;
    let x0i = (x0 as isize).clamp(0, w as isize - 1) as usize;
    let y0i = (y0 as isize).clamp(0, h as isize - 1) as usize;
    let x1i = (x0i + 1).min(w - 1);
    let y1i = (y0i + 1).min(h - 1);
    let v00 = buf[y0i * w + x0i];
    let v01 = buf[y0i * w + x1i];
    let v10 = buf[y1i * w + x0i];
    let v11 = buf[y1i * w + x1i];
    let v0 = v00 * (1.0 - wx) + v01 * wx;
    let v1 = v10 * (1.0 - wx) + v11 * wx;
    v0 * (1.0 - wy) + v1 * wy
}

/// Bicubic sample of a `(H, W, C)` NHWC descriptor map at sparse (x, y) pixel
/// coordinates **in the descriptor map's own resolution** (i.e. already
/// divided by 8 from the input scale). Output is `kp.len() * c` f32.
///
/// Each descriptor is L2-normalised in-place after sampling, avoiding a second
/// pass over the output buffer.
pub fn bicubic_sample_descriptors(
    desc: &[f32],
    h_d: usize,
    w_d: usize,
    c: usize,
    kps_xy_in_desc_space: &[(f32, f32)],
    out: &mut [f32],
) {
    debug_assert_eq!(desc.len(), h_d * w_d * c);
    debug_assert_eq!(out.len(), kps_xy_in_desc_space.len() * c);

    // Batch 64 keypoints per Rayon task (64 tasks for top_k=4096) to amortize
    // scheduler dispatch overhead across meaningful compute.
    out.par_chunks_mut(c * 64)
        .zip(kps_xy_in_desc_space.par_chunks(64))
        .for_each(|(out_block, kps_block)| {
            for (chunk, &(x, y)) in out_block.chunks_mut(c).zip(kps_block.iter()) {
                bicubic_sample_one(desc, h_d, w_d, c, x, y, chunk);
                let norm = chunk.iter().map(|&v| v * v).sum::<f32>().sqrt();
                let inv = 1.0 / (norm + 1e-12);
                for v in chunk.iter_mut() {
                    *v *= inv;
                }
            }
        });
}

/// Same as `bicubic_sample_descriptors` but `desc` is in f16 (u16) storage.
///
/// Each f16 descriptor value is widened to f32 during the bicubic accumulation.
/// Output is f32 and L2-normalised in-place, identical to the f32 variant.
pub fn bicubic_sample_descriptors_f16(
    desc: &[u16],
    h_d: usize,
    w_d: usize,
    c: usize,
    kps_xy_in_desc_space: &[(f32, f32)],
    out: &mut [f32],
) {
    debug_assert_eq!(desc.len(), h_d * w_d * c);
    debug_assert_eq!(out.len(), kps_xy_in_desc_space.len() * c);

    out.par_chunks_mut(c * 64)
        .zip(kps_xy_in_desc_space.par_chunks(64))
        .for_each(|(out_block, kps_block)| {
            for (chunk, &(x, y)) in out_block.chunks_mut(c).zip(kps_block.iter()) {
                bicubic_sample_one_f16(desc, h_d, w_d, c, x, y, chunk);
                let norm = chunk.iter().map(|&v| v * v).sum::<f32>().sqrt();
                let inv = 1.0 / (norm + 1e-12);
                for v in chunk.iter_mut() {
                    *v *= inv;
                }
            }
        });
}

/// Cubic Hermite coefficients (PyTorch `grid_sample` bicubic, a = -0.75).
#[inline(always)]
fn cubic_weights(t: f32) -> [f32; 4] {
    let a = -0.75f32;
    let t2 = t * t;
    let t3 = t2 * t;
    let s = 1.0 - t;
    let s2 = s * s;
    let s3 = s2 * s;
    let u = 2.0 - t;
    let u2 = u * u;
    let u3 = u2 * u;
    let v = 1.0 + t;
    let v2 = v * v;
    let v3 = v2 * v;
    [
        a * v3 - 5.0 * a * v2 + 8.0 * a * v - 4.0 * a,
        (a + 2.0) * t3 - (a + 3.0) * t2 + 1.0,
        (a + 2.0) * s3 - (a + 3.0) * s2 + 1.0,
        a * u3 - 5.0 * a * u2 + 8.0 * a * u - 4.0 * a,
    ]
}

/// Sample descriptor at a single sub-pixel coordinate.
///
/// Loop order: spatial (j, i) outer, channel inner. Each inner pass reads
/// `c` contiguous f32 values from the NHWC tensor — a single cache-line-aligned
/// load stream per spatial position — vs the previous channel-outer order which
/// scattered 16 stride-`c` loads per channel.
///
/// The c=64 path uses NEON FMLA.4S inline (no helper call, no function boundary).
/// On AArch64, NEON is mandatory ISA, so intrinsics are valid in any unsafe block.
fn bicubic_sample_one(desc: &[f32], h: usize, w: usize, c: usize, x: f32, y: f32, out: &mut [f32]) {
    let ix = x.floor() as isize;
    let iy = y.floor() as isize;

    let kx = cubic_weights(x - ix as f32);
    let ky = cubic_weights(y - iy as f32);

    // Pre-compute all 16 weight products upfront.
    let mut w16 = [0.0f32; 16];
    for j in 0..4usize {
        for i in 0..4usize {
            w16[j * 4 + i] = ky[j] * kx[i];
        }
    }

    // Spatial-outer: each inner pass reads c contiguous f32s (NHWC pixel).
    #[cfg(target_arch = "aarch64")]
    if c == 64 {
        // SAFETY: AArch64 NEON is mandatory. Bounds: xx < w, yy < h, c == 64 →
        // base + 64 ≤ desc.len() and out.len() ≥ 64.
        unsafe {
            use std::arch::aarch64::*;
            // 16 zero-initialised accumulators — one per 4-channel block.
            let op = out.as_mut_ptr();
            let z = vdupq_n_f32(0.0);
            let mut a0 = z;
            let mut a1 = z;
            let mut a2 = z;
            let mut a3 = z;
            let mut a4 = z;
            let mut a5 = z;
            let mut a6 = z;
            let mut a7 = z;
            let mut a8 = z;
            let mut a9 = z;
            let mut a10 = z;
            let mut a11 = z;
            let mut a12 = z;
            let mut a13 = z;
            let mut a14 = z;
            let mut a15 = z;

            for j in 0..4usize {
                let yy = (iy - 1 + j as isize).clamp(0, h as isize - 1) as usize;
                for i in 0..4usize {
                    let xx = (ix - 1 + i as isize).clamp(0, w as isize - 1) as usize;
                    let wt = vdupq_n_f32(w16[j * 4 + i]);
                    let sp = desc.as_ptr().add((yy * w + xx) * 64);
                    a0 = vfmaq_f32(a0, wt, vld1q_f32(sp));
                    a1 = vfmaq_f32(a1, wt, vld1q_f32(sp.add(4)));
                    a2 = vfmaq_f32(a2, wt, vld1q_f32(sp.add(8)));
                    a3 = vfmaq_f32(a3, wt, vld1q_f32(sp.add(12)));
                    a4 = vfmaq_f32(a4, wt, vld1q_f32(sp.add(16)));
                    a5 = vfmaq_f32(a5, wt, vld1q_f32(sp.add(20)));
                    a6 = vfmaq_f32(a6, wt, vld1q_f32(sp.add(24)));
                    a7 = vfmaq_f32(a7, wt, vld1q_f32(sp.add(28)));
                    a8 = vfmaq_f32(a8, wt, vld1q_f32(sp.add(32)));
                    a9 = vfmaq_f32(a9, wt, vld1q_f32(sp.add(36)));
                    a10 = vfmaq_f32(a10, wt, vld1q_f32(sp.add(40)));
                    a11 = vfmaq_f32(a11, wt, vld1q_f32(sp.add(44)));
                    a12 = vfmaq_f32(a12, wt, vld1q_f32(sp.add(48)));
                    a13 = vfmaq_f32(a13, wt, vld1q_f32(sp.add(52)));
                    a14 = vfmaq_f32(a14, wt, vld1q_f32(sp.add(56)));
                    a15 = vfmaq_f32(a15, wt, vld1q_f32(sp.add(60)));
                }
            }
            vst1q_f32(op, a0);
            vst1q_f32(op.add(4), a1);
            vst1q_f32(op.add(8), a2);
            vst1q_f32(op.add(12), a3);
            vst1q_f32(op.add(16), a4);
            vst1q_f32(op.add(20), a5);
            vst1q_f32(op.add(24), a6);
            vst1q_f32(op.add(28), a7);
            vst1q_f32(op.add(32), a8);
            vst1q_f32(op.add(36), a9);
            vst1q_f32(op.add(40), a10);
            vst1q_f32(op.add(44), a11);
            vst1q_f32(op.add(48), a12);
            vst1q_f32(op.add(52), a13);
            vst1q_f32(op.add(56), a14);
            vst1q_f32(op.add(60), a15);
        }
        return;
    }

    out[..c].fill(0.0);
    for j in 0..4usize {
        let yy = (iy - 1 + j as isize).clamp(0, h as isize - 1) as usize;
        for i in 0..4usize {
            let xx = (ix - 1 + i as isize).clamp(0, w as isize - 1) as usize;
            let wt = w16[j * 4 + i];
            let base = (yy * w + xx) * c;
            for ci in 0..c {
                out[ci] += desc[base + ci] * wt;
            }
        }
    }
}

/// f16-storage variant of `bicubic_sample_one`.
///
/// Loads u16 (f16 bits) from `desc` and widens to f32 for accumulation.
/// On aarch64 the c=64 hot path uses FCVTL (NEON half→single) + FMLA.4S.
fn bicubic_sample_one_f16(
    desc: &[u16],
    h: usize,
    w: usize,
    c: usize,
    x: f32,
    y: f32,
    out: &mut [f32],
) {
    let ix = x.floor() as isize;
    let iy = y.floor() as isize;
    let kx = cubic_weights(x - ix as f32);
    let ky = cubic_weights(y - iy as f32);
    let mut w16 = [0.0f32; 16];
    for j in 0..4usize {
        for i in 0..4usize {
            w16[j * 4 + i] = ky[j] * kx[i];
        }
    }

    #[cfg(target_arch = "aarch64")]
    if c == 64 {
        // SAFETY: AArch64 NEON is mandatory. Bounds: xx<w, yy<h, c==64.
        // Uses fcvtl_lo/fcvtl_hi (stable inline-asm wrappers) instead of the
        // nightly vcvt_f32_f16 / vreinterpret_f16_u16 intrinsics.
        unsafe {
            use crate::ops::neon_asm_f16::{fcvtl_hi, fcvtl_lo};
            use std::arch::aarch64::*;
            let op = out.as_mut_ptr();
            let z = vdupq_n_f32(0.0);
            let mut a0 = z;
            let mut a1 = z;
            let mut a2 = z;
            let mut a3 = z;
            let mut a4 = z;
            let mut a5 = z;
            let mut a6 = z;
            let mut a7 = z;
            let mut a8 = z;
            let mut a9 = z;
            let mut a10 = z;
            let mut a11 = z;
            let mut a12 = z;
            let mut a13 = z;
            let mut a14 = z;
            let mut a15 = z;

            for j in 0..4usize {
                let yy = (iy - 1 + j as isize).clamp(0, h as isize - 1) as usize;
                for i in 0..4usize {
                    let xx = (ix - 1 + i as isize).clamp(0, w as isize - 1) as usize;
                    let wt = vdupq_n_f32(w16[j * 4 + i]);
                    let sp = desc.as_ptr().add((yy * w + xx) * 64);
                    // 8 groups of 8 f16 lanes = 64 channels.  Each group → 2 × f32x4.
                    let h0 = vld1q_u16(sp);
                    let h1 = vld1q_u16(sp.add(8));
                    let h2 = vld1q_u16(sp.add(16));
                    let h3 = vld1q_u16(sp.add(24));
                    let h4 = vld1q_u16(sp.add(32));
                    let h5 = vld1q_u16(sp.add(40));
                    let h6 = vld1q_u16(sp.add(48));
                    let h7 = vld1q_u16(sp.add(56));
                    // FCVTL/FCVTL2 via stable inline-asm wrappers (no nightly needed).
                    a0 = vfmaq_f32(a0, wt, fcvtl_lo(h0));
                    a1 = vfmaq_f32(a1, wt, fcvtl_hi(h0));
                    a2 = vfmaq_f32(a2, wt, fcvtl_lo(h1));
                    a3 = vfmaq_f32(a3, wt, fcvtl_hi(h1));
                    a4 = vfmaq_f32(a4, wt, fcvtl_lo(h2));
                    a5 = vfmaq_f32(a5, wt, fcvtl_hi(h2));
                    a6 = vfmaq_f32(a6, wt, fcvtl_lo(h3));
                    a7 = vfmaq_f32(a7, wt, fcvtl_hi(h3));
                    a8 = vfmaq_f32(a8, wt, fcvtl_lo(h4));
                    a9 = vfmaq_f32(a9, wt, fcvtl_hi(h4));
                    a10 = vfmaq_f32(a10, wt, fcvtl_lo(h5));
                    a11 = vfmaq_f32(a11, wt, fcvtl_hi(h5));
                    a12 = vfmaq_f32(a12, wt, fcvtl_lo(h6));
                    a13 = vfmaq_f32(a13, wt, fcvtl_hi(h6));
                    a14 = vfmaq_f32(a14, wt, fcvtl_lo(h7));
                    a15 = vfmaq_f32(a15, wt, fcvtl_hi(h7));
                }
            }
            vst1q_f32(op, a0);
            vst1q_f32(op.add(4), a1);
            vst1q_f32(op.add(8), a2);
            vst1q_f32(op.add(12), a3);
            vst1q_f32(op.add(16), a4);
            vst1q_f32(op.add(20), a5);
            vst1q_f32(op.add(24), a6);
            vst1q_f32(op.add(28), a7);
            vst1q_f32(op.add(32), a8);
            vst1q_f32(op.add(36), a9);
            vst1q_f32(op.add(40), a10);
            vst1q_f32(op.add(44), a11);
            vst1q_f32(op.add(48), a12);
            vst1q_f32(op.add(52), a13);
            vst1q_f32(op.add(56), a14);
            vst1q_f32(op.add(60), a15);
        }
        return;
    }

    // Scalar fallback: convert f16 bits to f32 on load.
    out[..c].fill(0.0);
    for j in 0..4usize {
        let yy = (iy - 1 + j as isize).clamp(0, h as isize - 1) as usize;
        for i in 0..4usize {
            let xx = (ix - 1 + i as isize).clamp(0, w as isize - 1) as usize;
            let wt = w16[j * 4 + i];
            let base = (yy * w + xx) * c;
            for ci in 0..c {
                out[ci] += half::f16::from_bits(desc[base + ci]).to_f32() * wt;
            }
        }
    }
}
