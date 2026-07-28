//! 128-dimensional descriptor: a 4x4 grid of 8-bin gradient histograms,
//! accumulated with trilinear interpolation.
//!
//! # Numerics
//!
//! Two details carry the bit-exactness. The scatter is sequential in the
//! reference's sample order, because float addition is not associative. And the
//! normalisation tail computes `nrm2` with a **four-lane FMA accumulation
//! followed by a pairwise reduction**, not a scalar sum — see [`nrm2`].

use super::hal::{exp_batch, grow_to, mag_ang_batch};

use super::orient::OrientedKeypoint;

/// Grid width (`SIFT_DESCR_WIDTH`).
pub const DESCR_WIDTH: usize = 4;
/// Orientation bins per cell (`SIFT_DESCR_HIST_BINS`).
pub const DESCR_HIST_BINS: usize = 8;
/// Descriptor length.
pub const DESCR_LEN: usize = DESCR_WIDTH * DESCR_WIDTH * DESCR_HIST_BINS;
/// Patch scale factor (`SIFT_DESCR_SCL_FCTR`).
pub const DESCR_SCL_FCTR: f32 = 3.0;
/// Post-normalisation clamp (`SIFT_DESCR_MAG_THR`).
pub const DESCR_MAG_THR: f32 = 0.2;
/// Quantisation factor (`SIFT_INT_DESCR_FCTR`).
pub const INT_DESCR_FCTR: f32 = 512.0;

#[inline]
fn floor_i(v: f32) -> i32 {
    let i = v as i32;
    i - i32::from(v < i as f32)
}

/// Narrow `[lo, hi]` to the `j` satisfying `-1 < j * coef + off < limit`.
///
/// Both descriptor bin coordinates are affine in `j` (`r_rot = j * sin_t +
/// i * cos_t`, and likewise for `c_rot`), so each bound is a half-plane and the
/// accepted `j` of a row form one contiguous run. `coef == 0` makes the
/// constraint independent of `j`: it either holds for the whole row or kills it.
#[inline]
fn clip_j(coef: f32, off: f32, limit: f32, lo: &mut f32, hi: &mut f32) {
    if coef > 0.0 {
        *lo = lo.max((-1.0 - off) / coef);
        *hi = hi.min((limit - off) / coef);
    } else if coef < 0.0 {
        *lo = lo.max((limit - off) / coef);
        *hi = hi.min((-1.0 - off) / coef);
    } else if !(off > -1.0 && off < limit) {
        // Empty row.
        *lo = 1.0;
        *hi = 0.0;
    }
}

/// The reference accumulates over four SIMD lanes with FMA, then reduces
/// pairwise: `((l0+l1) + (l2+l3))`. A plain scalar sum gives a different value.
fn nrm2(v: &[f32]) -> f32 {
    let (mut a0, mut a1, mut a2, mut a3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let mut k = 0usize;
    while k + 4 <= v.len() {
        a0 = v[k].mul_add(v[k], a0);
        a1 = v[k + 1].mul_add(v[k + 1], a1);
        a2 = v[k + 2].mul_add(v[k + 2], a2);
        a3 = v[k + 3].mul_add(v[k + 3], a3);
        k += 4;
    }
    let mut s = (a0 + a1) + (a2 + a3);
    while k < v.len() {
        s += v[k] * v[k];
        k += 1;
    }
    s
}

/// Per-thread scratch for [`compute_descriptor`].
///
/// The reference batches each HAL primitive over the **whole patch** —
/// `fastAtan2`, `magnitude32f` and `exp32f` each run as one long loop — instead
/// of evaluating four samples at a time between scatters. That turns out to
/// matter a lot: four-at-a-time measured 39 ns per accepted sample against the
/// reference's 22 ns, because the three transcendental emulations and the
/// histogram scatter all compete for registers inside one loop body. These
/// buffers make the same split possible here. One per rayon worker, reused
/// across keypoints, since a patch is a few thousand floats and allocating per
/// keypoint would dominate.
#[derive(Default)]
pub struct DescriptorScratch {
    rbin: Vec<f32>,
    cbin: Vec<f32>,
    dx: Vec<f32>,
    dy: Vec<f32>,
    wt: Vec<f32>,
    mag: Vec<f32>,
    ang: Vec<f32>,
}

impl DescriptorScratch {
    /// Empty scratch; sized on first use.
    ///
    /// The buffers are grown monotonically and written by index with a sample
    /// counter, never cleared: every element read is written first, so a reset
    /// would only cost a pass that changes nothing.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Trilinear scatter of every resolved sample into `hist`.
///
/// Everything up to the eight accumulations is elementwise, so four lanes are
/// bit-identical to the scalar tail; only the accumulation is order-sensitive
/// (float addition does not associate) and it stays scalar, in the reference's
/// sample order. This is the split `calcSIFTDescriptor` itself uses.
fn scatter(sc: &DescriptorScratch, len: usize, ori: f32, bins_per_rad: f32, hist: &mut [f32]) {
    const D: usize = DESCR_WIDTH;
    const N: usize = DESCR_HIST_BINS;
    let mut k = 0usize;

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        // SAFETY: every buffer is `len` long, `k + 4 <= len`, and the scatter
        // indices are bounded by the `rbin`/`cbin` range test applied during
        // collection.
        unsafe {
            let vori = vdupq_n_f32(ori);
            let vn = vdupq_n_s32(N as i32);
            let vone = vdupq_n_s32(1);
            while k + 4 <= len {
                let mut rb = vld1q_f32(sc.rbin.as_ptr().add(k));
                let mut cb = vld1q_f32(sc.cbin.as_ptr().add(k));
                let mut ob = vmulq_n_f32(
                    vsubq_f32(vld1q_f32(sc.ang.as_ptr().add(k)), vori),
                    bins_per_rad,
                );
                // Round toward minus infinity: `floor_i`'s twin.
                let r0 = vcvtmq_s32_f32(rb);
                let c0 = vcvtmq_s32_f32(cb);
                let mut o0 = vcvtmq_s32_f32(ob);
                rb = vsubq_f32(rb, vcvtq_f32_s32(r0));
                cb = vsubq_f32(cb, vcvtq_f32_s32(c0));
                ob = vsubq_f32(ob, vcvtq_f32_s32(o0));

                o0 = vbslq_s32(vcltq_s32(o0, vdupq_n_s32(0)), vaddq_s32(o0, vn), o0);
                o0 = vbslq_s32(vcgeq_s32(o0, vn), vsubq_s32(o0, vn), o0);

                let m = vmulq_f32(
                    vld1q_f32(sc.mag.as_ptr().add(k)),
                    vld1q_f32(sc.wt.as_ptr().add(k)),
                );
                let v_r1 = vmulq_f32(m, rb);
                let v_r0 = vsubq_f32(m, v_r1);
                let v_rc11 = vmulq_f32(v_r1, cb);
                let v_rc10 = vsubq_f32(v_r1, v_rc11);
                let v_rc01 = vmulq_f32(v_r0, cb);
                let v_rc00 = vsubq_f32(v_r0, v_rc01);
                let v111 = vmulq_f32(v_rc11, ob);
                let v110 = vsubq_f32(v_rc11, v111);
                let v101 = vmulq_f32(v_rc10, ob);
                let v100 = vsubq_f32(v_rc10, v101);
                let v011 = vmulq_f32(v_rc01, ob);
                let v010 = vsubq_f32(v_rc01, v011);
                let v001 = vmulq_f32(v_rc00, ob);
                let v000 = vsubq_f32(v_rc00, v001);

                let idx = vaddq_s32(
                    vmulq_n_s32(
                        vaddq_s32(
                            vmulq_n_s32(vaddq_s32(r0, vone), (D + 2) as i32),
                            vaddq_s32(c0, vone),
                        ),
                        (N + 2) as i32,
                    ),
                    o0,
                );

                let mut ib = [0i32; 4];
                vst1q_s32(ib.as_mut_ptr(), idx);
                // The bin coordinates come out too, to gate the dead pairs:
                // the fold only reads cell rows/cols 1..D, so a pair whose
                // row is 0/D+1 or whose column is 0/D+1 writes memory nothing
                // ever reads. That is 36% of the pairs on average.
                let mut rb4 = [0i32; 4];
                vst1q_s32(rb4.as_mut_ptr(), r0);
                let mut cb4 = [0i32; 4];
                vst1q_s32(cb4.as_mut_ptr(), c0);

                // The eight destinations are four *adjacent* pairs — offsets
                // 0,1 / 10,11 / 60,61 / 70,71 — so each pair can accumulate as
                // one 2-lane read-modify-write instead of two scalar ones.
                // Interleaving each pair's two vectors puts lane `t`'s values
                // next to each other: lane t's (a_t, b_t) is exactly one half
                // of vzip1q/vzip2q, handed over REGISTER-TO-REGISTER. The
                // previous form staged the zips through a 32-float stack
                // buffer — 8 stores per 4 samples on the single store-data
                // port, plus a load-behind-store round trip per pair — which
                // the audit measured as a third of the scatter's stores.
                // Elementwise vector add over distinct addresses is
                // bit-identical to the scalar pair, and the t-outer order
                // below is the sample order the reference fixes.
                let z0 = (vzip1q_f32(v000, v001), vzip2q_f32(v000, v001));
                let z1 = (vzip1q_f32(v010, v011), vzip2q_f32(v010, v011));
                let z2 = (vzip1q_f32(v100, v101), vzip2q_f32(v100, v101));
                let z3 = (vzip1q_f32(v110, v111), vzip2q_f32(v110, v111));
                const OFF: [usize; 4] = [0, N + 2, (D + 2) * (N + 2), (D + 3) * (N + 2)];
                let hp = hist.as_mut_ptr();
                // One block per lane `t`, fully unrolled: vget_low/vget_high
                // need a compile-time half. Slot liveness: 0 = (r+1,c+1),
                // 1 = (r+1,c+2), 2 = (r+2,c+1), 3 = (r+2,c+2); a dead slot
                // writes only never-read border cells (see the scalar tail),
                // and skipping it keeps the live accumulation order exact.
                macro_rules! lane_scatter {
                    ($t:expr, $half:ident, $zi:tt) => {{
                        // SAFETY: `rbin`/`cbin` were range-checked during
                        // collection, so `b <= 287` and the widest touched
                        // index is `b + 71 < HISTLEN`.
                        let b = ib[$t] as usize;
                        let (rlo, rhi) = (rb4[$t] >= 0, rb4[$t] <= D as i32 - 2);
                        let (clo, chi) = (cb4[$t] >= 0, cb4[$t] <= D as i32 - 2);
                        if rlo && clo {
                            let h = hp.add(b + OFF[0]);
                            vst1_f32(h, vadd_f32(vld1_f32(h), $half(z0.$zi)));
                        }
                        if rlo && chi {
                            let h = hp.add(b + OFF[1]);
                            vst1_f32(h, vadd_f32(vld1_f32(h), $half(z1.$zi)));
                        }
                        if rhi && clo {
                            let h = hp.add(b + OFF[2]);
                            vst1_f32(h, vadd_f32(vld1_f32(h), $half(z2.$zi)));
                        }
                        if rhi && chi {
                            let h = hp.add(b + OFF[3]);
                            vst1_f32(h, vadd_f32(vld1_f32(h), $half(z3.$zi)));
                        }
                    }};
                }
                // Lane t's pair: t 0/1 = low/high half of zip1, t 2/3 = low/
                // high half of zip2 — [a0,b0,a1,b1] and [a2,b2,a3,b3].
                lane_scatter!(0, vget_low_f32, 0);
                lane_scatter!(1, vget_high_f32, 0);
                lane_scatter!(2, vget_low_f32, 1);
                lane_scatter!(3, vget_high_f32, 1);
                k += 4;
            }
        }
    }

    while k < len {
        let mut obin = (sc.ang[k] - ori) * bins_per_rad;
        let (mut rb, mut cb) = (sc.rbin[k], sc.cbin[k]);
        let (r0, c0) = (floor_i(rb), floor_i(cb));
        let mut o0 = floor_i(obin);
        rb -= r0 as f32;
        cb -= c0 as f32;
        obin -= o0 as f32;
        if o0 < 0 {
            o0 += N as i32;
        }
        if o0 >= N as i32 {
            o0 -= N as i32;
        }

        let m = sc.mag[k] * sc.wt[k];
        let v_r1 = m * rb;
        let v_r0 = m - v_r1;
        let v_rc11 = v_r1 * cb;
        let v_rc10 = v_r1 - v_rc11;
        let v_rc01 = v_r0 * cb;
        let v_rc00 = v_r0 - v_rc01;
        let v_rco111 = v_rc11 * obin;
        let v_rco110 = v_rc11 - v_rco111;
        let v_rco101 = v_rc10 * obin;
        let v_rco100 = v_rc10 - v_rco101;
        let v_rco011 = v_rc01 * obin;
        let v_rco010 = v_rc01 - v_rco011;
        let v_rco001 = v_rc00 * obin;
        let v_rco000 = v_rc00 - v_rco001;

        // Same border-ring elision as the vector block above.
        let (rlo, rhi) = (r0 >= 0, r0 <= D as i32 - 2);
        let (clo, chi) = (c0 >= 0, c0 <= D as i32 - 2);
        let idx = (((r0 + 1) as usize * (D + 2) + (c0 + 1) as usize) * (N + 2)) + o0 as usize;
        if rlo && clo {
            hist[idx] += v_rco000;
            hist[idx + 1] += v_rco001;
        }
        if rlo && chi {
            hist[idx + (N + 2)] += v_rco010;
            hist[idx + (N + 3)] += v_rco011;
        }
        if rhi && clo {
            hist[idx + (D + 2) * (N + 2)] += v_rco100;
            hist[idx + (D + 2) * (N + 2) + 1] += v_rco101;
        }
        if rhi && chi {
            hist[idx + (D + 3) * (N + 2)] += v_rco110;
            hist[idx + (D + 3) * (N + 2) + 1] += v_rco111;
        }
        k += 1;
    }
}

/// Compute one 128-D descriptor.
///
/// `ptx`, `pty` and `scl` are in the **octave's** coordinate frame, and `ori` is
/// the reference's `360 - angle` with a near-360 result collapsed to zero — the
/// caller applies both conversions, as `calcDescriptors` does.
///
/// `sc` is caller-owned scratch; see [`DescriptorScratch`] for why the patch is
/// buffered whole rather than four samples at a time.
#[allow(clippy::too_many_arguments)]
pub fn compute_descriptor(
    img: &[f32],
    w: usize,
    h: usize,
    ptx: f32,
    pty: f32,
    scl: f32,
    ori: f32,
    out: &mut [f32],
    sc: &mut DescriptorScratch,
) {
    const D: usize = DESCR_WIDTH;
    const N: usize = DESCR_HIST_BINS;
    const HISTLEN: usize = (D + 2) * (D + 2) * (N + 2);
    debug_assert_eq!(out.len(), DESCR_LEN);

    let px = ptx.round_ties_even() as i32;
    let py = pty.round_ties_even() as i32;
    let mut cos_t = (ori * (std::f32::consts::PI / 180.0)).cos();
    let mut sin_t = (ori * (std::f32::consts::PI / 180.0)).sin();
    let bins_per_rad = N as f32 / 360.0;
    let exp_scale = -1.0 / (D as f32 * D as f32 * 0.5);
    let hist_width = DESCR_SCL_FCTR * scl;
    let mut radius =
        (hist_width * std::f32::consts::SQRT_2 * (D as f32 + 1.0) * 0.5).round_ties_even() as i32;
    let diag = ((w as f32) * (w as f32) + (h as f32) * (h as f32)).sqrt() as i32;
    if radius > diag {
        radius = diag;
    }
    cos_t /= hist_width;
    sin_t /= hist_width;

    let mut hist = [0.0f32; HISTLEN];

    // Size the buffers to the row span once, then write by index. The
    // reference fills preallocated arrays with a running counter; `Vec::push`
    // costs a capacity check per element, and there are five buffers times
    // ~1400 samples per keypoint.
    let i_lo = (-radius).max(1 - py);
    let i_hi = radius.min(h as i32 - 2 - py);
    let cap = (i_hi - i_lo + 1).max(0) as usize * (2 * radius + 1) as usize;
    grow_to(&mut sc.dx, cap);
    grow_to(&mut sc.dy, cap);
    grow_to(&mut sc.rbin, cap);
    grow_to(&mut sc.cbin, cap);
    grow_to(&mut sc.wt, cap);
    let mut n = 0usize;

    // The row bound is exactly the reference's `r > 0 && r < rows - 1`, hoisted
    // out of the inner test: it depends only on `i`.
    for i in i_lo..=i_hi {
        // Accepted `j` form one contiguous run per row (see `clip_j`). Deriving
        // it skips the ~52% of the square the predicate rejects — measured on
        // apriltags: 5.50M iterations for 2.63M accepted samples.
        //
        // The derived bounds are float arithmetic and the original predicate is
        // NOT float-associative-safe to replace, so the run is widened by one on
        // each side and every surviving `j` still faces the identical test
        // below. Skipping therefore removes provably-rejected iterations only,
        // and cannot move a sample into or out of the histogram.
        let (mut jf_lo, mut jf_hi) = (-(radius as f32), radius as f32);
        let half = D as f32 / 2.0 - 0.5;
        clip_j(
            sin_t,
            i as f32 * cos_t + half,
            D as f32,
            &mut jf_lo,
            &mut jf_hi,
        );
        clip_j(
            cos_t,
            -(i as f32) * sin_t + half,
            D as f32,
            &mut jf_lo,
            &mut jf_hi,
        );
        if jf_lo > jf_hi {
            continue;
        }
        let j_lo = (jf_lo.floor() as i32 - 1).max(-radius).max(1 - px);
        let j_hi = (jf_hi.ceil() as i32 + 1).min(radius).min(w as i32 - 2 - px);

        let r = (py + i) as usize;
        let (row, up, dn) = (r * w, (r - 1) * w, (r + 1) * w);

        // Scalar body, shared by the 4-wide fallback and the tail.
        //
        // SAFETY (both paths): `n` counts accepted samples, and the loops run
        // at most `(i_hi - i_lo + 1) * (2 * radius + 1) == cap` times between
        // them, so `n < cap` and every buffer is `cap` long. The row and
        // column bounds already guarantee all four stencil neighbours exist.
        macro_rules! push_sample {
            ($j:expr) => {{
                let j = $j;
                let c_rot = j as f32 * cos_t - i as f32 * sin_t;
                let r_rot = j as f32 * sin_t + i as f32 * cos_t;
                let rbin = r_rot + D as f32 / 2.0 - 0.5;
                let cbin = c_rot + D as f32 / 2.0 - 0.5;
                if rbin > -1.0 && rbin < D as f32 && cbin > -1.0 && cbin < D as f32 {
                    let c = (px + j) as usize;
                    unsafe {
                        *sc.dx.get_unchecked_mut(n) = img[row + c + 1] - img[row + c - 1];
                        *sc.dy.get_unchecked_mut(n) = img[up + c] - img[dn + c];
                        *sc.rbin.get_unchecked_mut(n) = rbin;
                        *sc.cbin.get_unchecked_mut(n) = cbin;
                        *sc.wt.get_unchecked_mut(n) = (c_rot * c_rot + r_rot * r_rot) * exp_scale;
                    }
                    n += 1;
                }
            }};
        }

        let mut j = j_lo;

        // 4-wide body. `clip_j` widens the accepted run by at most one on each
        // side, so rejects sit only at the run's two ends and almost every
        // block passes whole; a partial block falls back to the scalar body so
        // packing order is preserved. Expression shapes are the scalar's
        // exactly — mul/mul/sub for c_rot, mul/mul/add for r_rot, and
        // mul,mul,add,mul for the weight; **no FMA anywhere** (Rust never
        // contracts, so introducing one would change the descriptor). The
        // hoisted i*sin_t / i*cos_t are the same product the scalar computed
        // per iteration, rounded once — identical value.
        #[cfg(target_arch = "aarch64")]
        unsafe {
            use std::arch::aarch64::*;
            let isin = i as f32 * sin_t;
            let icos = i as f32 * cos_t;
            let half = D as f32 / 2.0 - 0.5;
            let dsize = D as f32;
            let base = vld1q_f32([0.0f32, 1.0, 2.0, 3.0].as_ptr());
            while j + 3 <= j_hi {
                let jv = vaddq_f32(vdupq_n_f32(j as f32), base);
                let c_rot = vsubq_f32(vmulq_n_f32(jv, cos_t), vdupq_n_f32(isin));
                let r_rot = vaddq_f32(vmulq_n_f32(jv, sin_t), vdupq_n_f32(icos));
                let rbin = vaddq_f32(r_rot, vdupq_n_f32(half));
                let cbin = vaddq_f32(c_rot, vdupq_n_f32(half));

                let ok = vandq_u32(
                    vandq_u32(
                        vcgtq_f32(rbin, vdupq_n_f32(-1.0)),
                        vcltq_f32(rbin, vdupq_n_f32(dsize)),
                    ),
                    vandq_u32(
                        vcgtq_f32(cbin, vdupq_n_f32(-1.0)),
                        vcltq_f32(cbin, vdupq_n_f32(dsize)),
                    ),
                );
                if vminvq_u32(ok) != u32::MAX {
                    // Partial block: rejects only at run ends. Scalar keeps
                    // the packed order exact.
                    for jj in j..j + 4 {
                        push_sample!(jj);
                    }
                } else {
                    let c0 = (px + j) as usize;
                    let wt = vmulq_n_f32(
                        vaddq_f32(vmulq_f32(c_rot, c_rot), vmulq_f32(r_rot, r_rot)),
                        exp_scale,
                    );
                    let dxv = vsubq_f32(
                        vld1q_f32(img.as_ptr().add(row + c0 + 1)),
                        vld1q_f32(img.as_ptr().add(row + c0 - 1)),
                    );
                    let dyv = vsubq_f32(
                        vld1q_f32(img.as_ptr().add(up + c0)),
                        vld1q_f32(img.as_ptr().add(dn + c0)),
                    );
                    vst1q_f32(sc.dx.as_mut_ptr().add(n), dxv);
                    vst1q_f32(sc.dy.as_mut_ptr().add(n), dyv);
                    vst1q_f32(sc.rbin.as_mut_ptr().add(n), rbin);
                    vst1q_f32(sc.cbin.as_mut_ptr().add(n), cbin);
                    vst1q_f32(sc.wt.as_mut_ptr().add(n), wt);
                    n += 4;
                }
                j += 4;
            }
        }

        while j <= j_hi {
            push_sample!(j);
            j += 1;
        }
    }

    // One long pass per primitive, as the reference does.
    let len = n;
    grow_to(&mut sc.mag, len);
    grow_to(&mut sc.ang, len);
    exp_batch(&mut sc.wt[..len]);
    mag_ang_batch(
        &sc.dx[..len],
        &sc.dy[..len],
        &mut sc.mag[..len],
        &mut sc.ang[..len],
    );
    scatter(sc, len, ori, bins_per_rad, &mut hist);

    // Fold the circular orientation bins back into the d*d*n array.
    let mut raw = [0.0f32; DESCR_LEN];
    for i in 0..D {
        for j in 0..D {
            let idx = ((i + 1) * (D + 2) + (j + 1)) * (N + 2);
            hist[idx] += hist[idx + N];
            // The reference also folds o-bin N+1 into bin 1, but no scatter can
            // write it: o0 wraps into 0..N-1, so writes reach offset N at most.
            // Folding that guaranteed +0.0 is a no-op (all accumulands are
            // non-negative, so there is no -0.0 for it to normalise).
            for k in 0..N {
                raw[(i * D + j) * N + k] = hist[idx + k];
            }
        }
    }

    // Normalise, clamp at MAG_THR, renormalise, scale and saturate to uchar.
    let thr = nrm2(&raw).sqrt() * DESCR_MAG_THR;
    let mut n2 = 0.0f32;
    for v in raw.iter_mut() {
        let val = v.min(thr);
        *v = val;
        n2 += val * val;
    }
    let scale = INT_DESCR_FCTR / n2.sqrt().max(f32::EPSILON);
    for (o, v) in out.iter_mut().zip(raw.iter()) {
        *o = ((v * scale).round_ties_even()).clamp(0.0, 255.0);
    }
}

/// Convert an oriented keypoint into the descriptor's inputs for its octave.
///
/// The reference rescales by `1 / (1 << octave)` and flips the angle: the stored
/// orientation is counter-clockwise, the descriptor's rotation is clockwise.
pub fn descriptor_inputs(kp: &OrientedKeypoint, octv: i32) -> (f32, f32, f32, f32) {
    let scale = 1.0 / (1 << octv) as f32;
    let mut angle = 360.0 - kp.angle;
    if (angle - 360.0).abs() < f32::EPSILON {
        angle = 0.0;
    }
    (kp.x * scale, kp.y * scale, (kp.size * scale) * 0.5, angle)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load_dump(path: &str) -> Option<(usize, usize, Vec<f32>)> {
        let b = std::fs::read(path).ok()?;
        let rows = i32::from_le_bytes(b[0..4].try_into().unwrap()) as usize;
        let cols = i32::from_le_bytes(b[4..8].try_into().unwrap()) as usize;
        let data: Vec<f32> = b[8..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .take(rows * cols)
            .collect();
        Some((rows, cols, data))
    }

    /// Driven from the REFERENCE keypoints so this isolates the descriptor from
    /// any upstream detector or orientation residual.
    #[test]
    fn descriptor_matches_reference_bitwise() {
        let Some(dir) = std::env::var("KORNIA_SIFT_ORACLE")
            .ok()
            .and_then(|v| v.split(':').next().map(String::from))
        else {
            eprintln!("KORNIA_SIFT_ORACLE unset; skipping");
            return;
        };
        let b = std::fs::read(format!("{dir}/keypoints.bin")).expect("keypoints");
        let n = i32::from_le_bytes(b[0..4].try_into().unwrap()) as usize;
        let Some((_, dcols, want)) = load_dump(&format!("{dir}/descriptors.f32")) else {
            eprintln!("no descriptor dump; skipping");
            return;
        };
        assert_eq!(dcols, DESCR_LEN);

        let mut sc = DescriptorScratch::new();
        let (mut total, mut bad) = (0usize, 0usize);
        for layer in 1..=3usize {
            let Some((h, w, img)) = load_dump(&format!("{dir}/gauss_o0_l{layer}.f32")) else {
                return;
            };

            for i in 0..n {
                let o = 4 + i * 24;
                let packed = i32::from_le_bytes(b[o + 20..o + 24].try_into().unwrap());
                if (packed & 255) != 255 || ((packed >> 8) & 255) != layer as i32 {
                    continue;
                }
                let f =
                    |k: usize| f32::from_le_bytes(b[o + k * 4..o + k * 4 + 4].try_into().unwrap());
                // Octave -1: undo the stored halving, then flip the angle.
                let mut angle = 360.0f32 - f(3);
                if (angle - 360.0).abs() < f32::EPSILON {
                    angle = 0.0;
                }
                let mut got = [0.0f32; DESCR_LEN];
                compute_descriptor(
                    &img,
                    w,
                    h,
                    f(0) * 2.0,
                    f(1) * 2.0,
                    f(2) * 2.0 * 0.5,
                    angle,
                    &mut got,
                    &mut sc,
                );
                let e = &want[i * DESCR_LEN..(i + 1) * DESCR_LEN];
                total += 1;
                if got.iter().zip(e).any(|(a, b)| a.to_bits() != b.to_bits()) {
                    bad += 1;
                }
            }
        }
        eprintln!(
            "  cpu descriptor: {}/{} exact (octave -1)",
            total - bad,
            total
        );
        assert!(total > 0, "no octave -1 keypoints");
        assert_eq!(bad, 0, "{bad} of {total} descriptors differ");
    }
}
