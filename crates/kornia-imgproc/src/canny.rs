//! Canny edge detection — byte-for-byte with `cv2.Canny` (u8 C1,
//! aperture 3).
//!
//! The whole pipeline is integer, so parity is exact by transcription:
//!
//! * Sobel 3×3 (`CV_16S`, `BORDER_REPLICATE`, scale 1) gradients;
//! * magnitude `|dx| + |dy|` (L1, default) or `dx² + dy²` (L2 —
//!   thresholds are clamped to 32767 and squared, mirroring cv2);
//! * thresholds `low = floor(low_thresh)`, `high = floor(high_thresh)`,
//!   swapped if reversed;
//! * non-maximum suppression with OpenCV's fixed-point sector test
//!   (`TG22 = 13573 = tan(22.5°)·2¹⁵`), including its exact tie-break
//!   asymmetries: horizontal `> left, >= right`; vertical `> up, >= down`;
//!   diagonal strictly `>` both, with `s = sign(dx ^ dy)`;
//! * hysteresis as a stack flood from strong pixels over weak candidates —
//!   pure reachability, so the result is traversal-order independent;
//! * output 255 for edges, 0 otherwise.
//!
//! The magnitude ring is padded with zero rows/columns exactly like cv2's,
//! so image-border pixels compete against zero magnitudes, and the map ring
//! is pre-marked non-edge (candidates never touch the border ring).

use kornia_image::{Image, ImageError};
use rayon::prelude::*;

/// `tan(22.5°) · 2^15`, cv2's sector constant.
const TG22: i32 = 13573;

/// Sobel 3×3 `CV_16S` with replicate borders: `dx = [1,2,1]ᵀ ⊗ [-1,0,1]`,
/// `dy = [-1,0,1]ᵀ ⊗ [1,2,1]`. Shared by the CPU path and (in PR 11) the
/// CUDA twin.
pub(crate) fn sobel3_i16(src: &Image<u8, 1>, dx: &mut [i16], dy: &mut [i16]) {
    let (w, h) = (src.cols(), src.rows());
    let s = src.as_slice();
    dx.par_chunks_mut(w)
        .zip(dy.par_chunks_mut(w))
        .enumerate()
        .with_min_len(32)
        .for_each_init(
            || (vec![0i16; w], vec![0i16; w]),
            |(sv, dv), (y, (dxr, dyr))| {
                let ym = y.saturating_sub(1).min(h - 1);
                let yp = (y + 1).min(h - 1);
                let (r0, r1, r2) = (
                    &s[ym * w..ym * w + w],
                    &s[y * w..y * w + w],
                    &s[yp * w..yp * w + w],
                );
                // Vertical pass (separable, exact): sv = r0 + 2·r1 + r2
                // (≤ 1020, fits i16), dv = r2 − r0.
                #[cfg(target_arch = "aarch64")]
                unsafe {
                    use std::arch::aarch64::*;
                    let mut x = 0usize;
                    while x + 8 <= w {
                        let a = vmovl_u8(vld1_u8(r0.as_ptr().add(x)));
                        let b = vmovl_u8(vld1_u8(r1.as_ptr().add(x)));
                        let c = vmovl_u8(vld1_u8(r2.as_ptr().add(x)));
                        let svv = vaddq_u16(vaddq_u16(a, c), vshlq_n_u16(b, 1));
                        let dvv = vsubq_s16(vreinterpretq_s16_u16(c), vreinterpretq_s16_u16(a));
                        vst1q_s16(sv.as_mut_ptr().add(x), vreinterpretq_s16_u16(svv));
                        vst1q_s16(dv.as_mut_ptr().add(x), dvv);
                        x += 8;
                    }
                    for xx in x..w {
                        sv[xx] = r0[xx] as i16 + 2 * r1[xx] as i16 + r2[xx] as i16;
                        dv[xx] = r2[xx] as i16 - r0[xx] as i16;
                    }
                }
                #[cfg(not(target_arch = "aarch64"))]
                for xx in 0..w {
                    sv[xx] = r0[xx] as i16 + 2 * r1[xx] as i16 + r2[xx] as i16;
                    dv[xx] = r2[xx] as i16 - r0[xx] as i16;
                }
                // Horizontal pass: dx = sv[x+1] − sv[x−1] (replicate),
                // dy = dv[x−1] + 2·dv[x] + dv[x+1].
                if w == 1 {
                    dxr[0] = 0;
                    dyr[0] = 4 * dv[0];
                    return;
                }
                dxr[0] = sv[1] - sv[0];
                dyr[0] = dv[0] + 2 * dv[0] + dv[1];
                #[cfg(target_arch = "aarch64")]
                unsafe {
                    use std::arch::aarch64::*;
                    let mut x = 1usize;
                    while x + 8 <= w - 1 {
                        let sm = vld1q_s16(sv.as_ptr().add(x - 1));
                        let sp = vld1q_s16(sv.as_ptr().add(x + 1));
                        vst1q_s16(dxr.as_mut_ptr().add(x), vsubq_s16(sp, sm));
                        let dm = vld1q_s16(dv.as_ptr().add(x - 1));
                        let dc = vld1q_s16(dv.as_ptr().add(x));
                        let dp = vld1q_s16(dv.as_ptr().add(x + 1));
                        vst1q_s16(
                            dyr.as_mut_ptr().add(x),
                            vaddq_s16(vaddq_s16(dm, dp), vshlq_n_s16(dc, 1)),
                        );
                        x += 8;
                    }
                    for xx in x..w - 1 {
                        dxr[xx] = sv[xx + 1] - sv[xx - 1];
                        dyr[xx] = dv[xx - 1] + 2 * dv[xx] + dv[xx + 1];
                    }
                }
                #[cfg(not(target_arch = "aarch64"))]
                for xx in 1..w - 1 {
                    dxr[xx] = sv[xx + 1] - sv[xx - 1];
                    dyr[xx] = dv[xx - 1] + 2 * dv[xx] + dv[xx + 1];
                }
                dxr[w - 1] = sv[w - 1] - sv[w - 2];
                dyr[w - 1] = dv[w - 2] + 2 * dv[w - 1] + dv[w - 1];
            },
        );
}

/// Map codes, mirroring cv2's: 0 = weak candidate, 1 = non-edge, 2 = edge.
const CANDIDATE: u8 = 0;
const NON_EDGE: u8 = 1;
const EDGE: u8 = 2;

/// Canny edge detector for 8-bit single-channel images — byte-for-byte
/// with `cv2.Canny(src, low_thresh, high_thresh, L2gradient=l2_gradient)`
/// (aperture size 3). Output pixels are 255 on edges, 0 elsewhere.
pub fn canny(
    src: &Image<u8, 1>,
    dst: &mut Image<u8, 1>,
    low_thresh: f64,
    high_thresh: f64,
    l2_gradient: bool,
) -> Result<(), ImageError> {
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }
    let (w, h) = (src.cols(), src.rows());

    // cv2's threshold preparation: swap if reversed; L2 clamps to 32767 and
    // squares; floor to int.
    let (mut lo, mut hi) = (low_thresh, high_thresh);
    if lo > hi {
        std::mem::swap(&mut lo, &mut hi);
    }
    if l2_gradient {
        lo = lo.min(32767.0);
        hi = hi.min(32767.0);
        if lo > 0.0 {
            lo *= lo;
        }
        if hi > 0.0 {
            hi *= hi;
        }
    }
    let low = lo.floor() as i64 as i32;
    let high = hi.floor() as i64 as i32;

    #[cfg(feature = "cuda")]
    {
        use crate::try_device;
        try_device!(src, dst, |stream| cuda_adapters::canny_cuda(
            src,
            dst,
            low,
            high,
            l2_gradient,
            stream
        ));
    }

    // Gradients.
    let mut dx = vec![0i16; w * h];
    let mut dy = vec![0i16; w * h];
    sobel3_i16(src, &mut dx, &mut dy);

    // Magnitude with a one-pixel zero ring (cv2 zeroes the halo, so border
    // pixels compete against 0).
    let mstep = w + 2;
    let mut mag = vec![0i32; mstep * (h + 2)];
    mag.par_chunks_mut(mstep)
        .skip(1)
        .take(h)
        .enumerate()
        .for_each(|(y, mrow)| {
            let (dxr, dyr) = (&dx[y * w..y * w + w], &dy[y * w..y * w + w]);
            if l2_gradient {
                for x in 0..w {
                    let (gx, gy) = (dxr[x] as i32, dyr[x] as i32);
                    mrow[x + 1] = gx * gx + gy * gy;
                }
            } else {
                #[cfg(target_arch = "aarch64")]
                unsafe {
                    use std::arch::aarch64::*;
                    let mut x = 0usize;
                    while x + 8 <= w {
                        let gx = vabsq_s16(vld1q_s16(dxr.as_ptr().add(x)));
                        let gy = vabsq_s16(vld1q_s16(dyr.as_ptr().add(x)));
                        // |dx| ≤ 1020, |dy| ≤ 1020: the sum fits u16/i32.
                        let m = vaddq_u16(vreinterpretq_u16_s16(gx), vreinterpretq_u16_s16(gy));
                        let lo = vmovl_u16(vget_low_u16(m));
                        let hi = vmovl_u16(vget_high_u16(m));
                        vst1q_s32(mrow.as_mut_ptr().add(x + 1), vreinterpretq_s32_u32(lo));
                        vst1q_s32(mrow.as_mut_ptr().add(x + 5), vreinterpretq_s32_u32(hi));
                        x += 8;
                    }
                    for xx in x..w {
                        mrow[xx + 1] = (dxr[xx] as i32).abs() + (dyr[xx] as i32).abs();
                    }
                }
                #[cfg(not(target_arch = "aarch64"))]
                for x in 0..w {
                    mrow[x + 1] = (dxr[x] as i32).abs() + (dyr[x] as i32).abs();
                }
            }
        });

    // Non-maximum suppression → map with a NON_EDGE ring. Rows are
    // independent (pure function of three magnitude rows), so this
    // parallelizes; strong seeds are collected per row.
    let mut map = vec![NON_EDGE; mstep * (h + 2)];
    let seeds: Vec<Vec<usize>> = map[mstep..mstep * (h + 1)]
        .par_chunks_mut(mstep)
        .enumerate()
        .map(|(y, mprow)| {
            let mut row_seeds = Vec::new();
            let mag_p = &mag[y * mstep..(y + 1) * mstep];
            let mag_a = &mag[(y + 1) * mstep..(y + 2) * mstep];
            let mag_n = &mag[(y + 2) * mstep..(y + 3) * mstep];
            let (dxr, dyr) = (&dx[y * w..y * w + w], &dy[y * w..y * w + w]);
            #[cfg(not(target_arch = "aarch64"))]
            let j = 0usize;
            #[cfg(target_arch = "aarch64")]
            let j = {
                neon::nms_row(
                    mag_p,
                    mag_a,
                    mag_n,
                    dxr,
                    dyr,
                    mprow,
                    w,
                    low,
                    high,
                    &mut row_seeds,
                )
            };
            for j in j..w {
                let k = j + 1;
                if nms_pixel(mag_p, mag_a, mag_n, dxr[j], dyr[j], k, low, high, mprow) {
                    row_seeds.push(k);
                }
            }
            row_seeds.into_iter().map(|k| (y + 1) * mstep + k).collect()
        })
        .collect();

    // Hysteresis: parallel tile-worklist flood (the CPU mirror of the GPU
    // kernel). Each active tile floods locally; a changed tile wakes its 8
    // neighbors for the next round. Edge marking is monotonic (0 → 2 only)
    // and the fixpoint is pure reachability, so racy halo reads at tile
    // borders can only delay propagation by a round, never change the
    // final set — the result is byte-identical to a serial stack flood.
    let seeds: Vec<usize> = seeds.into_iter().flatten().collect();
    if !seeds.is_empty() {
        hysteresis_parallel(&mut map, w, h, mstep, &seeds);
    }

    // Final pass: 255 where map == EDGE.
    dst.as_slice_mut()
        .par_chunks_mut(w)
        .enumerate()
        .for_each(|(y, drow)| {
            let mrow = &map[(y + 1) * mstep + 1..];
            #[cfg(target_arch = "aarch64")]
            unsafe {
                use std::arch::aarch64::*;
                let two = vdupq_n_u8(EDGE);
                let mut x = 0usize;
                while x + 16 <= w {
                    let m = vld1q_u8(mrow.as_ptr().add(x));
                    vst1q_u8(drow.as_mut_ptr().add(x), vceqq_u8(m, two));
                    x += 16;
                }
                for xx in x..w {
                    drow[xx] = if mrow[xx] == EDGE { 255 } else { 0 };
                }
            }
            #[cfg(not(target_arch = "aarch64"))]
            for (x, d) in drow.iter_mut().enumerate() {
                *d = if mrow[x] == EDGE { 255 } else { 0 };
            }
        });
    Ok(())
}

const TILE_W: usize = 256;
const TILE_H: usize = 64;

/// Shared-map handle for the tile flood: tiles write only their own
/// cells; halo reads may race with a neighbor's monotonic 0→2 writes,
/// which is benign (see call site).
struct MapPtr(*mut u8);
unsafe impl Send for MapPtr {}
unsafe impl Sync for MapPtr {}
impl MapPtr {
    /// Accessor (rather than direct field use) so closures capture the
    /// Sync wrapper, not the raw pointer (edition-2021 disjoint capture).
    fn get(&self) -> *mut u8 {
        self.0
    }
}

fn hysteresis_parallel(map: &mut [u8], w: usize, h: usize, mstep: usize, seeds: &[usize]) {
    let tiles_x = w.div_ceil(TILE_W);
    let tiles_y = h.div_ceil(TILE_H);
    let ntiles = tiles_x * tiles_y;
    let ptr = MapPtr(map.as_mut_ptr());

    // Round 1 floods each tile from its own strong seeds; later rounds
    // only need the tile's 1-pixel boundary ring (the interior is already
    // a local fixpoint — new conversions can only enter from the halo).
    let mut tile_seeds: Vec<Vec<usize>> = vec![Vec::new(); ntiles];
    for &p in seeds {
        let (x, y) = (p % mstep - 1, p / mstep - 1);
        tile_seeds[(y / TILE_H) * tiles_x + x / TILE_W].push(p);
    }

    let neighbors = |p: usize| {
        [
            p - mstep - 1,
            p - mstep,
            p - mstep + 1,
            p - 1,
            p + 1,
            p + mstep - 1,
            p + mstep,
            p + mstep + 1,
        ]
    };

    let flood_tile = |t: usize, init: &[usize], boundary_scan: bool| -> bool {
        let (ty, tx) = (t / tiles_x, t % tiles_x);
        let x0 = tx * TILE_W + 1; // map coords (ring offset)
        let y0 = ty * TILE_H + 1;
        let x1 = (x0 + TILE_W).min(w + 1);
        let y1 = (y0 + TILE_H).min(h + 1);
        // SAFETY: this tile writes only cells in [x0,x1)×[y0,y1); halo
        // reads race only with monotonic single-byte 0→2 writes.
        let m = ptr.get();
        let mut stack: Vec<usize> = init.to_vec();
        let mut changed = false;
        unsafe {
            if boundary_scan {
                // Candidates on the boundary ring adjacent to an edge
                // (halo or own).
                let mut check = |p: usize| {
                    if *m.add(p) == CANDIDATE && neighbors(p).iter().any(|&q| *m.add(q) == EDGE) {
                        *m.add(p) = EDGE;
                        stack.push(p);
                        changed = true;
                    }
                };
                for x in x0..x1 {
                    check(y0 * mstep + x);
                    check((y1 - 1) * mstep + x);
                }
                for y in y0..y1 {
                    check(y * mstep + x0);
                    check(y * mstep + (x1 - 1));
                }
            }
            while let Some(p) = stack.pop() {
                for q in neighbors(p) {
                    if *m.add(q) == CANDIDATE {
                        let qx = q % mstep;
                        let qy = q / mstep;
                        if qx >= x0 && qx < x1 && qy >= y0 && qy < y1 {
                            *m.add(q) = EDGE;
                            stack.push(q);
                            changed = true;
                        }
                    }
                }
            }
        }
        changed
    };

    let wake_neighbors = |t: usize, active: &mut [bool]| {
        let (ty, tx) = (t / tiles_x, t % tiles_x);
        for dy in -1i64..=1 {
            for dx in -1i64..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let (nx, ny) = (tx as i64 + dx, ty as i64 + dy);
                if nx >= 0 && nx < tiles_x as i64 && ny >= 0 && ny < tiles_y as i64 {
                    active[ny as usize * tiles_x + nx as usize] = true;
                }
            }
        }
    };

    // Round 1: seed-driven.
    let changed_tiles: Vec<usize> = (0..ntiles)
        .into_par_iter()
        .filter(|&t| !tile_seeds[t].is_empty())
        .filter(|&t| flood_tile(t, &tile_seeds[t], false))
        .collect();
    let mut active = vec![false; ntiles];
    for t in changed_tiles {
        wake_neighbors(t, &mut active);
    }

    // Later rounds: boundary-ring driven, worklist ping-pong.
    let mut round = 0usize;
    while active.iter().any(|&a| a) && round <= w * h {
        round += 1;
        let changed_tiles: Vec<usize> = (0..ntiles)
            .into_par_iter()
            .filter(|&t| active[t])
            .filter(|&t| flood_tile(t, &[], true))
            .collect();
        active.iter_mut().for_each(|a| *a = false);
        for t in changed_tiles {
            wake_neighbors(t, &mut active);
        }
    }
}

/// Scalar NMS for one pixel — cv2's exact sector conditions and
/// tie-breaks. Writes the map label; returns true for a strong seed. The
/// NEON row kernel and the CUDA kernel evaluate the same integer
/// expressions.
#[allow(clippy::too_many_arguments)]
#[inline]
fn nms_pixel(
    mag_p: &[i32],
    mag_a: &[i32],
    mag_n: &[i32],
    xs: i16,
    ys: i16,
    k: usize,
    low: i32,
    high: i32,
    mprow: &mut [u8],
) -> bool {
    let m = mag_a[k];
    let mut label = NON_EDGE;
    let mut strong = false;
    if m > low {
        let xs = xs as i32;
        let ys = ys as i32;
        let x = xs.abs();
        let y15 = ys.abs() << 15;
        let tg22x = x * TG22;
        let is_max = if y15 < tg22x {
            m > mag_a[k - 1] && m >= mag_a[k + 1]
        } else {
            let tg67x = tg22x + (x << 16);
            if y15 > tg67x {
                m > mag_p[k] && m >= mag_n[k]
            } else {
                let s = if (xs ^ ys) < 0 { -1i64 } else { 1 };
                m > mag_p[(k as i64 - s) as usize] && m > mag_n[(k as i64 + s) as usize]
            }
        };
        if is_max {
            if m > high {
                label = EDGE;
                strong = true;
            } else {
                label = CANDIDATE;
            }
        }
    }
    mprow[k] = label;
    strong
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::{CANDIDATE, EDGE, NON_EDGE, TG22};
    use std::arch::aarch64::*;

    /// Branchless 4-lane NMS over a row: evaluates all three sector tests
    /// and selects by sector mask — every compare is the same integer
    /// compare the scalar path does, so labels are byte-identical. Blocks
    /// whose 4 magnitudes are all <= low are skipped outright (the map
    /// row is pre-filled NON_EDGE). Returns the first unprocessed column.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn nms_row(
        mag_p: &[i32],
        mag_a: &[i32],
        mag_n: &[i32],
        dxr: &[i16],
        dyr: &[i16],
        mprow: &mut [u8],
        w: usize,
        low: i32,
        high: i32,
        seeds: &mut Vec<usize>,
    ) -> usize {
        let mut j = 0usize;
        if w < 4 {
            return 0;
        }
        // SAFETY: for j+4 <= w, k = j+1 reads mag rows at k-1..k+4 within
        // the (w+2)-wide padded rows; dx/dy reads j..j+4 <= w; map writes
        // k..k+4 <= w+1.
        unsafe {
            let lowv = vdupq_n_s32(low);
            let highv = vdupq_n_s32(high);
            let tg22v = vdupq_n_s32(TG22);
            let one = vdupq_n_s32(1);
            let two = vdupq_n_s32(2);
            let zero = vdupq_n_s32(0);
            while j + 4 <= w {
                let k = j + 1;
                let m = vld1q_s32(mag_a.as_ptr().add(k));
                // Fast skip: all four below low → labels stay NON_EDGE.
                if vmaxvq_s32(m) <= low {
                    j += 4;
                    continue;
                }
                let dx4 = vmovl_s16(vld1_s16(dxr.as_ptr().add(j)));
                let dy4 = vmovl_s16(vld1_s16(dyr.as_ptr().add(j)));
                let x = vabsq_s32(dx4);
                let y15 = vshlq_n_s32(vabsq_s32(dy4), 15);
                let tg22x = vmulq_s32(x, tg22v);
                let tg67x = vaddq_s32(tg22x, vshlq_n_s32(x, 16));

                let ma_m1 = vld1q_s32(mag_a.as_ptr().add(k - 1));
                let ma_p1 = vld1q_s32(mag_a.as_ptr().add(k + 1));
                let mp_m1 = vld1q_s32(mag_p.as_ptr().add(k - 1));
                let mp = vld1q_s32(mag_p.as_ptr().add(k));
                let mp_p1 = vld1q_s32(mag_p.as_ptr().add(k + 1));
                let mn_m1 = vld1q_s32(mag_n.as_ptr().add(k - 1));
                let mn = vld1q_s32(mag_n.as_ptr().add(k));
                let mn_p1 = vld1q_s32(mag_n.as_ptr().add(k + 1));

                // Sector masks (mutually exclusive, mirroring the scalar
                // if/else chain).
                let mh = vcltq_s32(y15, tg22x);
                let mv = vcgtq_s32(y15, tg67x);
                let md = vbicq_u32(vbicq_u32(vdupq_n_u32(!0), mh), mv);

                // s = sign(dx ^ dy): lane-select the diagonal neighbors.
                let sneg = vcltq_s32(veorq_s32(dx4, dy4), zero);
                let dp = vbslq_s32(sneg, mp_p1, mp_m1);
                let dn = vbslq_s32(sneg, mn_m1, mn_p1);

                let h_max = vandq_u32(vcgtq_s32(m, ma_m1), vcgeq_s32(m, ma_p1));
                let v_max = vandq_u32(vcgtq_s32(m, mp), vcgeq_s32(m, mn));
                let d_max = vandq_u32(vcgtq_s32(m, dp), vcgtq_s32(m, dn));

                let is_max = vorrq_u32(
                    vorrq_u32(vandq_u32(mh, h_max), vandq_u32(mv, v_max)),
                    vandq_u32(md, d_max),
                );
                let cand = vandq_u32(vcgtq_s32(m, lowv), is_max);
                let strong = vandq_u32(vcgtq_s32(m, highv), cand);

                // label = cand ? (strong ? 2 : 0) : 1
                let label = vbslq_s32(cand, vbslq_s32(strong, two, zero), one);
                let l16 = vmovn_s32(label);
                let l8 = vmovn_s16(vcombine_s16(l16, l16));
                let bytes: [u8; 8] = std::mem::transmute(l8);
                mprow[k..k + 4].copy_from_slice(&bytes[..4]);

                if vmaxvq_u32(strong) != 0 {
                    let sm: [u32; 4] = std::mem::transmute(strong);
                    for (l, &sv) in sm.iter().enumerate() {
                        if sv != 0 {
                            seeds.push(k + l);
                        }
                    }
                }
                j += 4;
            }
        }
        let _ = (CANDIDATE, EDGE, NON_EDGE);
        j
    }
}

#[cfg(feature = "cuda")]
mod cuda_adapters {
    use super::*;
    use crate::cuda::canny::launch_canny_u8;
    use crate::cuda::dispatch::{device_slices, untyped_device_err};
    use cudarc::driver::CudaStream;
    use std::sync::Arc;

    fn err(e: impl std::fmt::Display) -> ImageError {
        ImageError::Cuda(e.to_string())
    }

    pub(super) fn canny_cuda(
        src: &Image<u8, 1>,
        dst: &mut Image<u8, 1>,
        low: i32,
        high: i32,
        l2_gradient: bool,
        stream: &Arc<CudaStream>,
    ) -> Result<(), ImageError> {
        let ctx = stream.context();
        let (s, d) = device_slices!(src, dst);
        launch_canny_u8(
            ctx,
            stream,
            s,
            d,
            src.cols(),
            src.rows(),
            low,
            high,
            l2_gradient,
        )
        .map_err(err)
    }
}

#[cfg(all(test, feature = "cuda"))]
mod cuda_tests {
    use super::*;
    use crate::cuda::color::test_utils::{default_stream, pattern_u8};
    use kornia_image::ImageSize;

    fn sz(w: usize, h: usize) -> ImageSize {
        ImageSize {
            width: w,
            height: h,
        }
    }

    /// Device Canny must be byte-exact vs the CPU path: random content
    /// (dense candidates), structured steps, several thresholds, L1 + L2,
    /// odd sizes.
    #[test]
    fn canny_device_equals_host_byte_exact() {
        let stream = default_stream();
        for (w, h) in [(64usize, 48usize), (67, 43), (128, 128), (17, 9)] {
            for (lo, hi) in [(50.0f64, 150.0f64), (20.0, 60.0), (0.0, 0.0)] {
                for l2 in [false, true] {
                    let src = Image::<u8, 1>::new(sz(w, h), pattern_u8(w * h)).unwrap();
                    let mut cpu = Image::<u8, 1>::from_size_val(sz(w, h), 0).unwrap();
                    canny(&src, &mut cpu, lo, hi, l2).unwrap();

                    let d_src = src.to_cuda(&stream).unwrap();
                    let mut d_dst = Image::<u8, 1>::zeros_cuda(sz(w, h), &stream).unwrap();
                    canny(&d_src, &mut d_dst, lo, hi, l2).unwrap();
                    assert_eq!(
                        d_dst.to_host_owned().unwrap().as_slice(),
                        cpu.as_slice(),
                        "{w}x{h} lo={lo} hi={hi} l2={l2}"
                    );
                }
            }
        }
        // Long weak chain seeded by one strong pixel: exercises multi-sweep
        // hysteresis propagation across many tiles.
        let (w, h) = (512usize, 64usize);
        let mut data = vec![0u8; w * h];
        for x in 0..w {
            for y in 0..h {
                data[y * w + x] = if y == 32 { 180 } else { 0 };
            }
        }
        data[32 * w + 5] = 255;
        let src = Image::<u8, 1>::new(sz(w, h), data).unwrap();
        let mut cpu = Image::<u8, 1>::from_size_val(sz(w, h), 0).unwrap();
        canny(&src, &mut cpu, 100.0, 600.0, false).unwrap();
        let d_src = src.to_cuda(&stream).unwrap();
        let mut d_dst = Image::<u8, 1>::zeros_cuda(sz(w, h), &stream).unwrap();
        canny(&d_src, &mut d_dst, 100.0, 600.0, false).unwrap();
        assert_eq!(d_dst.to_host_owned().unwrap().as_slice(), cpu.as_slice());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::ImageSize;

    fn sz(w: usize, h: usize) -> ImageSize {
        ImageSize {
            width: w,
            height: h,
        }
    }

    #[test]
    fn constant_image_no_edges() {
        let src = Image::<u8, 1>::from_size_val(sz(32, 24), 128).unwrap();
        let mut dst = Image::<u8, 1>::from_size_val(sz(32, 24), 7).unwrap();
        canny(&src, &mut dst, 50.0, 150.0, false).unwrap();
        assert!(dst.as_slice().iter().all(|&v| v == 0));
    }

    #[test]
    fn step_edge_detected_and_binary() {
        let mut data = vec![0u8; 64 * 32];
        for y in 0..32 {
            for x in 32..64 {
                data[y * 64 + x] = 200;
            }
        }
        let src = Image::<u8, 1>::new(sz(64, 32), data).unwrap();
        let mut dst = Image::<u8, 1>::from_size_val(sz(64, 32), 0).unwrap();
        canny(&src, &mut dst, 50.0, 150.0, false).unwrap();
        assert!(dst.as_slice().contains(&255));
        assert!(dst.as_slice().iter().all(|&v| v == 0 || v == 255));
        // edge localized near the step column
        for y in 2..30 {
            assert_eq!(dst.as_slice()[y * 64 + 31], 255, "row {y}");
        }
    }

    #[test]
    fn reversed_thresholds_swapped() {
        let mut data = vec![0u8; 64 * 32];
        for y in 0..32 {
            for x in 32..64 {
                data[y * 64 + x] = 200;
            }
        }
        let src = Image::<u8, 1>::new(sz(64, 32), data).unwrap();
        let mut a = Image::<u8, 1>::from_size_val(sz(64, 32), 0).unwrap();
        let mut b = Image::<u8, 1>::from_size_val(sz(64, 32), 0).unwrap();
        canny(&src, &mut a, 50.0, 150.0, false).unwrap();
        canny(&src, &mut b, 150.0, 50.0, false).unwrap();
        assert_eq!(a.as_slice(), b.as_slice());
    }

    #[test]
    fn size_mismatch_rejected() {
        let src = Image::<u8, 1>::from_size_val(sz(8, 8), 0).unwrap();
        let mut dst = Image::<u8, 1>::from_size_val(sz(4, 8), 0).unwrap();
        assert!(canny(&src, &mut dst, 50.0, 150.0, false).is_err());
    }
}

#[cfg(test)]
mod probe_tests {
    use super::*;
    use kornia_image::ImageSize;

    #[test]
    #[ignore]
    fn canny_stage_probe() {
        let (w, h) = (1920usize, 1080usize);
        // dense content: smoothed pseudo-noise (worst-case NMS/hysteresis)
        let mut state = 0x9e3779b9u64;
        let noise: Vec<u8> = (0..w * h)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                (state >> 33) as u8
            })
            .collect();
        // cheap 3x3 box smooth to mimic GaussianBlur'd noise
        let data: Vec<u8> = (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                let mut acc = 0u32;
                for dy in 0..3i64 {
                    let sy = (y as i64 + dy - 1).clamp(0, h as i64 - 1) as usize;
                    for dx in 0..3i64 {
                        let sx = (x as i64 + dx - 1).clamp(0, w as i64 - 1) as usize;
                        acc += noise[sy * w + sx] as u32;
                    }
                }
                (acc / 9) as u8
            })
            .collect();
        let src = Image::<u8, 1>::new(
            ImageSize {
                width: w,
                height: h,
            },
            data,
        )
        .unwrap();
        let mut dst = Image::<u8, 1>::from_size_val(src.size(), 0).unwrap();

        let mut dx = vec![0i16; w * h];
        let mut dy = vec![0i16; w * h];
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let t = std::time::Instant::now();
            for _ in 0..10 {
                sobel3_i16(&src, &mut dx, &mut dy);
            }
            best = best.min(t.elapsed().as_secs_f64() / 10.0);
        }
        println!("sobel: {:.3} ms", best * 1e3);

        for (name, lo, hi) in [
            ("full", 50.0, 150.0),
            ("no-strong", 50.0, 1e5),
            ("no-cand", 1e5, 1e5 + 1.0),
        ] {
            let mut best = f64::INFINITY;
            for _ in 0..5 {
                let t = std::time::Instant::now();
                for _ in 0..10 {
                    canny(&src, &mut dst, lo, hi, false).unwrap();
                }
                best = best.min(t.elapsed().as_secs_f64() / 10.0);
            }
            println!("{name}: {:.3} ms", best * 1e3);
        }
    }
}
