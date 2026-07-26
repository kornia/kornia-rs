//! 128-dimensional descriptor: a 4x4 grid of 8-bin gradient histograms,
//! accumulated with trilinear interpolation.
//!
//! # Numerics
//!
//! Two details carry the bit-exactness. The scatter is sequential in the
//! reference's sample order, because float addition is not associative. And the
//! normalisation tail computes `nrm2` with a **four-lane FMA accumulation
//! followed by a pairwise reduction**, not a scalar sum — see [`nrm2`].

use super::hal::{atan2_deg, exp, magnitude};

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

/// Evaluate up to four buffered samples and scatter them in order.
///
/// `n < 4` falls back to the scalar primitives, which are bit-identical to the
/// lane-wise ones, so the tail needs no special casing beyond the width.
#[allow(clippy::too_many_arguments)]
#[inline]
fn flush(
    dx: &[f32; 4],
    dy: &[f32; 4],
    c_rot: &[f32; 4],
    r_rot: &[f32; 4],
    rbin: &[f32; 4],
    cbin: &[f32; 4],
    n: usize,
    ori: f32,
    bins_per_rad: f32,
    exp_scale: f32,
    hist: &mut [f32],
) {
    const D: usize = DESCR_WIDTH;
    const N: usize = DESCR_HIST_BINS;
    // `dx`/`dy` carry the precomputed magnitude and angle; only the weight is
    // still evaluated, and it vectorises over the four buffered samples.
    let (mag, ang) = (dx, dy);
    let mut wgt = [0.0f32; 4];

    #[cfg(target_arch = "aarch64")]
    if n == 4 {
        use super::hal::x4;
        use std::arch::aarch64::*;
        // SAFETY: NEON is baseline on aarch64; all buffers are 4 wide.
        unsafe {
            let vcr = vld1q_f32(c_rot.as_ptr());
            let vrr = vld1q_f32(r_rot.as_ptr());
            let q = vfmaq_f32(vmulq_f32(vrr, vrr), vcr, vcr);
            vst1q_f32(wgt.as_mut_ptr(), x4::exp(vmulq_n_f32(q, exp_scale)));
        }
    }
    if !(cfg!(target_arch = "aarch64") && n == 4) {
        for k in 0..n {
            wgt[k] = exp((c_rot[k] * c_rot[k] + r_rot[k] * r_rot[k]) * exp_scale);
        }
    }

    for k in 0..n {
        let mut obin = (ang[k] - ori) * bins_per_rad;
        let (mut rb, mut cb) = (rbin[k], cbin[k]);
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

        let m = mag[k] * wgt[k];
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

        let idx = (((r0 + 1) as usize * (D + 2) + (c0 + 1) as usize) * (N + 2)) + o0 as usize;
        hist[idx] += v_rco000;
        hist[idx + 1] += v_rco001;
        hist[idx + (N + 2)] += v_rco010;
        hist[idx + (N + 3)] += v_rco011;
        hist[idx + (D + 2) * (N + 2)] += v_rco100;
        hist[idx + (D + 2) * (N + 2) + 1] += v_rco101;
        hist[idx + (D + 3) * (N + 2)] += v_rco110;
        hist[idx + (D + 3) * (N + 2) + 1] += v_rco111;
    }
}

/// Compute one 128-D descriptor.
///
/// `ptx`, `pty` and `scl` are in the **octave's** coordinate frame, and `ori` is
/// the reference's `360 - angle` with a near-360 result collapsed to zero — the
/// caller applies both conversions, as `calcDescriptors` does.
#[allow(clippy::too_many_arguments)]
pub fn compute_descriptor(
    grad_mag: &[f32],
    grad_ang: &[f32],
    w: usize,
    h: usize,
    ptx: f32,
    pty: f32,
    scl: f32,
    ori: f32,
    out: &mut [f32],
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

    // Evaluate four samples at a time, scatter them one at a time.
    //
    // The scatter order is fixed by the reference and cannot move — float
    // addition is not associative. The *evaluation* feeding it can: `exp`,
    // `atan2` and `magnitude` are pure functions of the sample, and the 4-lane
    // forms are lane-wise identical to the scalar ones. So the batch below is
    // bit-exact by construction, and it is where the time goes: this stage is
    // ~47% of the pipeline and every sample costs three primitive evaluations.
    let mut b_dx = [0.0f32; 4];
    let mut b_dy = [0.0f32; 4];
    let mut b_cr = [0.0f32; 4];
    let mut b_rr = [0.0f32; 4];
    let mut b_rbin = [0.0f32; 4];
    let mut b_cbin = [0.0f32; 4];
    let mut nb = 0usize;

    for i in -radius..=radius {
        for j in -radius..=radius {
            let c_rot = j as f32 * cos_t - i as f32 * sin_t;
            let r_rot = j as f32 * sin_t + i as f32 * cos_t;
            let rbin = r_rot + D as f32 / 2.0 - 0.5;
            let cbin = c_rot + D as f32 / 2.0 - 0.5;
            let (r, c) = (py + i, px + j);

            if !(rbin > -1.0
                && rbin < D as f32
                && cbin > -1.0
                && cbin < D as f32
                && r > 0
                && r < h as i32 - 1
                && c > 0
                && c < w as i32 - 1)
            {
                continue;
            }
            let (r, c) = (r as usize, c as usize);
            // Magnitude and angle come from the layer's precomputed planes; only
            // the Gaussian weight is keypoint-dependent and still evaluated here.
            b_dx[nb] = grad_mag[r * w + c];
            b_dy[nb] = grad_ang[r * w + c];
            b_cr[nb] = c_rot;
            b_rr[nb] = r_rot;
            b_rbin[nb] = rbin;
            b_cbin[nb] = cbin;
            nb += 1;
            if nb == 4 {
                flush(
                    &b_dx,
                    &b_dy,
                    &b_cr,
                    &b_rr,
                    &b_rbin,
                    &b_cbin,
                    4,
                    ori,
                    bins_per_rad,
                    exp_scale,
                    &mut hist,
                );
                nb = 0;
            }
        }
    }
    if nb > 0 {
        flush(
            &b_dx,
            &b_dy,
            &b_cr,
            &b_rr,
            &b_rbin,
            &b_cbin,
            nb,
            ori,
            bins_per_rad,
            exp_scale,
            &mut hist,
        );
    }

    // Fold the circular orientation bins back into the d*d*n array.
    let mut raw = [0.0f32; DESCR_LEN];
    for i in 0..D {
        for j in 0..D {
            let idx = ((i + 1) * (D + 2) + (j + 1)) * (N + 2);
            hist[idx] += hist[idx + N];
            hist[idx + 1] += hist[idx + N + 1];
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

/// Samples per descriptor bin per axis in the rotated-frame kernel.
pub const FAST_SAMP: usize = 4;

/// Bilinear image sample, clamped at the border.
#[inline]
fn tex(img: &[f32], w: usize, h: usize, x: f32, y: f32) -> f32 {
    let x = x.clamp(0.0, (w - 1) as f32);
    let y = y.clamp(0.0, (h - 1) as f32);
    let (x0, y0) = (x as usize, y as usize);
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
    let top = (img[y0 * w + x1] - img[y0 * w + x0]).mul_add(fx, img[y0 * w + x0]);
    let bot = (img[y1 * w + x1] - img[y1 * w + x0]).mul_add(fx, img[y1 * w + x0]);
    (bot - top).mul_add(fy, top)
}

/// Rotated-frame descriptor: **not** bit-exact, and much cheaper at large scales.
///
/// The reference walks every pixel of an axis-aligned bounding square and
/// rotates each one into descriptor space. That square's side grows with
/// `scl`, so the sample count grows as `scl^2` — about 1850 at a typical scale
/// and far more at coarse octaves — and roughly 28% of those samples are then
/// discarded for landing outside the rotated patch.
///
/// This walks a fixed grid *in* the rotated frame and samples the image
/// bilinearly, the way CudaSift and VLFeat do. The sample count becomes constant
/// at `(6 * FAST_SAMP)^2 = 576` regardless of scale, and none is wasted because
/// the grid is the patch. Trilinear interpolation, Gaussian weighting and the
/// normalisation tail are unchanged, so this is a sampling approximation rather
/// than a different descriptor.
///
/// # It does not pay off on CPU
///
/// On GPU the same rewrite is worth 2.9x, because there the stage is bound on
/// shared-memory atomics and the sample count drives them. On CPU the stage is
/// bound on per-sample arithmetic instead, and this trades *fewer* samples for
/// *more work each*: every sample needs four bilinear `tex` fetches (16 loads
/// plus interpolation) where the reference's grid walk reads four neighbours
/// directly. Measured end to end on mh01, `fo=-1`: 103.2 -> 116.9 ms. The
/// pipeline therefore does not select it, and it is kept only for callers
/// operating at scales large enough for the `scl^2` growth to dominate.
#[allow(clippy::too_many_arguments)]
pub fn compute_descriptor_fast(
    img: &[f32],
    w: usize,
    h: usize,
    ptx: f32,
    pty: f32,
    scl: f32,
    ori: f32,
    out: &mut [f32],
) {
    const D: usize = DESCR_WIDTH;
    const N: usize = DESCR_HIST_BINS;
    const HISTLEN: usize = (D + 2) * (D + 2) * (N + 2);
    let nsamp = (D + 2) * FAST_SAMP;

    let cos_o = (ori * (std::f32::consts::PI / 180.0)).cos();
    let sin_o = (ori * (std::f32::consts::PI / 180.0)).sin();
    let bins_per_rad = N as f32 / 360.0;
    let exp_scale = -1.0 / (D as f32 * D as f32 * 0.5);
    let hist_width = DESCR_SCL_FCTR * scl;
    let step = 1.0 / FAST_SAMP as f32;

    let mut hist = [0.0f32; HISTLEN];
    for s in 0..nsamp * nsamp {
        // The grid walks descriptor space directly, so the cell a sample lands
        // in is decided by the loop rather than by the data.
        let cb = -1.0 + ((s % nsamp) as f32 + 0.5) * step;
        let rb = -1.0 + ((s / nsamp) as f32 + 0.5) * step;
        let c_rot = cb - (D as f32 / 2.0 - 0.5);
        let r_rot = rb - (D as f32 / 2.0 - 0.5);
        let jx = hist_width * (c_rot * cos_o + r_rot * sin_o);
        let iy = hist_width * (r_rot * cos_o - c_rot * sin_o);
        let (x, y) = (ptx + jx, pty + iy);
        if x < 1.0 || x > (w - 2) as f32 || y < 1.0 || y > (h - 2) as f32 {
            continue;
        }

        let dx = tex(img, w, h, x + 1.0, y) - tex(img, w, h, x - 1.0, y);
        let dy = tex(img, w, h, x, y - 1.0) - tex(img, w, h, x, y + 1.0);
        let wgt = exp((c_rot * c_rot + r_rot * r_rot) * exp_scale);
        let ang = atan2_deg(dy, dx);
        let mag = magnitude(dx, dy);

        let mut obin = (ang - ori) * bins_per_rad;
        let (mut rf, mut cf) = (rb, cb);
        let (r0, c0) = (floor_i(rf), floor_i(cf));
        let mut o0 = floor_i(obin);
        rf -= r0 as f32;
        cf -= c0 as f32;
        obin -= o0 as f32;
        if o0 < 0 {
            o0 += N as i32;
        }
        if o0 >= N as i32 {
            o0 -= N as i32;
        }
        if r0 < -1 || r0 >= D as i32 || c0 < -1 || c0 >= D as i32 {
            continue;
        }

        // The grid is denser than the reference's pixel walk at small scales and
        // sparser at large ones; rescale so the total weight is comparable.
        let m = mag * wgt * (step * step);
        let v_r1 = m * rf;
        let v_r0 = m - v_r1;
        let v_rc11 = v_r1 * cf;
        let v_rc10 = v_r1 - v_rc11;
        let v_rc01 = v_r0 * cf;
        let v_rc00 = v_r0 - v_rc01;
        let v_rco111 = v_rc11 * obin;
        let v_rco110 = v_rc11 - v_rco111;
        let v_rco101 = v_rc10 * obin;
        let v_rco100 = v_rc10 - v_rco101;
        let v_rco011 = v_rc01 * obin;
        let v_rco010 = v_rc01 - v_rco011;
        let v_rco001 = v_rc00 * obin;
        let v_rco000 = v_rc00 - v_rco001;

        let idx = (((r0 + 1) as usize * (D + 2) + (c0 + 1) as usize) * (N + 2)) + o0 as usize;
        hist[idx] += v_rco000;
        hist[idx + 1] += v_rco001;
        hist[idx + (N + 2)] += v_rco010;
        hist[idx + (N + 3)] += v_rco011;
        hist[idx + (D + 2) * (N + 2)] += v_rco100;
        hist[idx + (D + 2) * (N + 2) + 1] += v_rco101;
        hist[idx + (D + 3) * (N + 2)] += v_rco110;
        hist[idx + (D + 3) * (N + 2) + 1] += v_rco111;
    }

    let mut raw = [0.0f32; DESCR_LEN];
    for i in 0..D {
        for j in 0..D {
            let idx = ((i + 1) * (D + 2) + (j + 1)) * (N + 2);
            hist[idx] += hist[idx + N];
            hist[idx + 1] += hist[idx + N + 1];
            for k in 0..N {
                raw[(i * D + j) * N + k] = hist[idx + k];
            }
        }
    }
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

        let (mut total, mut bad) = (0usize, 0usize);
        for layer in 1..=3usize {
            let Some((h, w, img)) = load_dump(&format!("{dir}/gauss_o0_l{layer}.f32")) else {
                return;
            };
            let mut gm = vec![0.0f32; w * h];
            let mut ga = vec![0.0f32; w * h];
            super::super::scalespace::gradients(&img, w, h, &mut gm, &mut ga);
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
                    &gm,
                    &ga,
                    w,
                    h,
                    f(0) * 2.0,
                    f(1) * 2.0,
                    f(2) * 2.0 * 0.5,
                    angle,
                    &mut got,
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
