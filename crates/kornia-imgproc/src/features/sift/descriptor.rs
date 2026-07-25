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

/// Compute one 128-D descriptor.
///
/// `ptx`, `pty` and `scl` are in the **octave's** coordinate frame, and `ori` is
/// the reference's `360 - angle` with a near-360 result collapsed to zero — the
/// caller applies both conversions, as `calcDescriptors` does.
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
            let dx = img[r * w + c + 1] - img[r * w + c - 1];
            let dy = img[(r - 1) * w + c] - img[(r + 1) * w + c];
            let wgt = exp((c_rot * c_rot + r_rot * r_rot) * exp_scale);
            let ang = atan2_deg(dy, dx);
            let mag = magnitude(dx, dy);

            let mut obin = (ang - ori) * bins_per_rad;
            let (mut rb, mut cb) = (rbin, cbin);
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

            let m = mag * wgt;
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
