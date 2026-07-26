//! Orientation assignment: a 36-bin gradient histogram per keypoint.
//!
//! The accumulation order is fixed by the reference and cannot be reordered:
//! float addition is not associative, and this histogram is summed sequentially
//! in a known sample order. The loop below walks rows-outer, columns-inner for
//! that reason.

use super::detect::RawKeypoint;
use super::hal::{atan2_deg, exp, magnitude};
use super::params::SiftConfig;

/// Histogram bins (`SIFT_ORI_HIST_BINS`).
pub const ORI_HIST_BINS: usize = 36;
/// Patch radius factor (`SIFT_ORI_RADIUS`).
pub const ORI_RADIUS: f32 = 4.5;
/// Gaussian weight sigma factor (`SIFT_ORI_SIG_FCTR`).
pub const ORI_SIG_FCTR: f32 = 1.5;
/// Secondary-peak acceptance ratio (`SIFT_ORI_PEAK_RATIO`).
pub const ORI_PEAK_RATIO: f32 = 0.8;

/// A keypoint with an assigned orientation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrientedKeypoint {
    /// Column in base-image pixels.
    pub x: f32,
    /// Row in base-image pixels.
    pub y: f32,
    /// Diameter of the meaningful neighbourhood.
    pub size: f32,
    /// Contrast at the refined extremum.
    pub response: f32,
    /// Packed octave/layer/sub-layer field.
    pub octave: i32,
    /// Orientation in degrees.
    pub angle: f32,
}

/// Append one oriented keypoint per dominant gradient direction.
///
/// A keypoint with several peaks within `ORI_PEAK_RATIO` of the maximum yields
/// several entries, as the reference does.
pub fn assign_orientations(
    img: &[f32],
    w: usize,
    h: usize,
    kp: &RawKeypoint,
    cfg: &SiftConfig,
    out: &mut Vec<OrientedKeypoint>,
) {
    let octv = kp.octave & 255;
    // `size` is still in the octave's own scale here; the first-octave halving
    // happens later.
    let scl_octv = kp.size * 0.5 / (1 << octv) as f32;
    let radius = (ORI_RADIUS * scl_octv).round_ties_even() as i32;
    let sigma = ORI_SIG_FCTR * scl_octv;
    let expf_scale = -1.0 / (2.0 * sigma * sigma);

    let mut temphist = [0.0f32; ORI_HIST_BINS];
    let (cc, rr) = (kp.cc, kp.rr);

    // The reference's binning loop has a SIMD block and a scalar tail, and they
    // round DIFFERENTLY:
    //
    //   SIMD:   w = w * mag;  then  temphist[bin] += w;   // two roundings
    //   tail:   temphist[bin] += W[k] * Mag[k];           // contracts to one fma
    //
    // So which samples get which depends only on the total count, and the count
    // is `nx * ny` in closed form because the loop is a rectangle clipped to the
    // image interior and the column bound does not depend on the row.
    let y0 = (rr - radius).max(1);
    let y1 = (rr + radius).min(h as i32 - 2);
    let x0 = (cc - radius).max(1);
    let x1 = (cc + radius).min(w as i32 - 2);
    let nx = (x1 - x0 + 1).max(0);
    let ny = (y1 - y0 + 1).max(0);
    let vec_end = (nx * ny) & !3;
    let mut k = 0i32;

    // Samples are buffered four at a time so `magnitude`, `atan2` and the
    // Gaussian weight evaluate lane-wise instead of one sample at a time. The
    // primitives are bit-identical across the two widths, and the binning below
    // still runs in strict sample order with its own `k`, so neither the values
    // nor the two-roundings split depend on where a batch boundary falls.
    let mut b_dx = [0.0f32; 4];
    let mut b_dy = [0.0f32; 4];
    let mut b_q = [0.0f32; 4];
    let mut nb = 0usize;

    let bin_batch = |dx: &[f32; 4],
                     dy: &[f32; 4],
                     q: &[f32; 4],
                     n: usize,
                     k: &mut i32,
                     temphist: &mut [f32; ORI_HIST_BINS]| {
        let mut mag = [0.0f32; 4];
        let mut ang = [0.0f32; 4];
        let mut wgt = [0.0f32; 4];

        #[cfg(target_arch = "aarch64")]
        let vectorised = n == 4;
        #[cfg(not(target_arch = "aarch64"))]
        let vectorised = false;

        #[cfg(target_arch = "aarch64")]
        if n == 4 {
            use super::hal::x4;
            use std::arch::aarch64::*;
            // SAFETY: NEON is baseline on aarch64; every buffer is 4 wide.
            unsafe {
                let vdx = vld1q_f32(dx.as_ptr());
                let vdy = vld1q_f32(dy.as_ptr());
                vst1q_f32(mag.as_mut_ptr(), x4::magnitude(vdx, vdy));
                vst1q_f32(ang.as_mut_ptr(), x4::atan2_deg(vdy, vdx));
                vst1q_f32(
                    wgt.as_mut_ptr(),
                    x4::exp(vmulq_n_f32(vld1q_f32(q.as_ptr()), expf_scale)),
                );
            }
        }
        if !vectorised {
            for t in 0..n {
                mag[t] = magnitude(dx[t], dy[t]);
                ang[t] = atan2_deg(dy[t], dx[t]);
                wgt[t] = exp(q[t] * expf_scale);
            }
        }

        for t in 0..n {
            let mut bin = ((ORI_HIST_BINS as f32 / 360.0) * ang[t]).round_ties_even() as i32;
            if bin >= ORI_HIST_BINS as i32 {
                bin -= ORI_HIST_BINS as i32;
            }
            if bin < 0 {
                bin += ORI_HIST_BINS as i32;
            }
            // NOTE: `magnitude32f`'s vector body composes `recip(rsqrt(s))`
            // from ARM's estimate instructions while its scalar tail issues a
            // real square root. Reproducing that split for the last `len % 4`
            // samples was implemented and measured here and on the GPU: it
            // changes no angle either time. Do not re-try.
            if *k < vec_end {
                temphist[bin as usize] += wgt[t] * mag[t];
            } else {
                temphist[bin as usize] = wgt[t].mul_add(mag[t], temphist[bin as usize]);
            }
            *k += 1;
        }
    };

    for i in -radius..=radius {
        let y = rr + i;
        if y <= 0 || y >= h as i32 - 1 {
            continue;
        }
        for j in -radius..=radius {
            let x = cc + j;
            if x <= 0 || x >= w as i32 - 1 {
                continue;
            }
            let (y, x) = (y as usize, x as usize);
            // Raw differences, the reference's stencil; magnitude and angle are
            // derived per sample rather than read from a precomputed layer.
            b_dx[nb] = img[y * w + x + 1] - img[y * w + x - 1];
            b_dy[nb] = img[(y - 1) * w + x] - img[(y + 1) * w + x];
            b_q[nb] = (i * i + j * j) as f32;
            nb += 1;
            if nb == 4 {
                bin_batch(&b_dx, &b_dy, &b_q, 4, &mut k, &mut temphist);
                nb = 0;
            }
        }
    }
    if nb > 0 {
        bin_batch(&b_dx, &b_dy, &b_q, nb, &mut k, &mut temphist);
    }

    // Smooth with [1,4,6,4,1]/16, wrapping. The reference's scalar form is
    // `(tn2+t2)*1/16 + (tn1+t1)*4/16 + t0*6/16` evaluated left to right, which
    // its build contracts as below — NOT the vector branch's shape.
    let n = ORI_HIST_BINS;
    let mut hist = [0.0f32; ORI_HIST_BINS];
    for i in 0..n {
        let tn2 = temphist[(i + n - 2) % n];
        let tn1 = temphist[(i + n - 1) % n];
        let t0 = temphist[i];
        let t1 = temphist[(i + 1) % n];
        let t2 = temphist[(i + 2) % n];
        // Vector branch's association, matching the binning loop above: the
        // whole translation unit takes one dispatch or the other, so if the
        // accumulation is the SIMD form the smoothing is too.
        hist[i] = (tn2 + t2).mul_add(
            1.0 / 16.0,
            (tn1 + t1).mul_add(4.0 / 16.0, t0 * (6.0 / 16.0)),
        );
    }

    let mut omax = hist[0];
    for &v in hist.iter().skip(1) {
        if v > omax {
            omax = v;
        }
    }
    let mag_thr = omax * ORI_PEAK_RATIO;

    let scale = 1.0f32; // caller applies the first-octave rescale
    for j in 0..n {
        let l = if j > 0 { j - 1 } else { n - 1 };
        let r2 = if j < n - 1 { j + 1 } else { 0 };
        if hist[j] > hist[l] && hist[j] > hist[r2] && hist[j] >= mag_thr {
            let den = hist[l] - 2.0 * hist[j] + hist[r2];
            let mut bin = j as f32 + 0.5 * (hist[l] - hist[r2]) / den;
            bin = if bin < 0.0 {
                n as f32 + bin
            } else if bin >= n as f32 {
                bin - n as f32
            } else {
                bin
            };
            // `360 - (360/n)*bin` is a mul-then-subtract, which the reference's
            // build contracts. Rounding the product separately shifts the angle
            // by an ULP.
            let mut angle = (-(360.0 / n as f32)).mul_add(bin, 360.0);
            if (angle - 360.0).abs() < f32::EPSILON {
                angle = 0.0;
            }
            out.push(OrientedKeypoint {
                x: kp.x * scale,
                y: kp.y * scale,
                size: kp.size * scale,
                response: kp.response,
                octave: kp.octave,
                angle,
            });
        }
    }
    let _ = cfg;
}

#[cfg(test)]
mod tests {
    use super::super::detect::find_extrema;
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

    /// Angles compared as a multiset per position: a keypoint with several
    /// dominant peaks appears as several reference entries at the same point, so
    /// matching positionally and taking the first hit compares unrelated peaks.
    #[test]
    fn orientation_matches_reference() {
        let Some(dir) = std::env::var("KORNIA_SIFT_ORACLE")
            .ok()
            .and_then(|v| v.split(':').next().map(String::from))
        else {
            eprintln!("KORNIA_SIFT_ORACLE unset; skipping");
            return;
        };
        let cfg = SiftConfig::default();
        let n_dog = cfg.n_octave_layers + 2;
        let mut stack: Vec<f32> = Vec::new();
        let (mut hh, mut ww) = (0usize, 0usize);
        for i in 0..n_dog {
            let Some((h, w, p)) = load_dump(&format!("{dir}/dog_o0_l{i}.f32")) else {
                eprintln!("no dog dumps; skipping");
                return;
            };
            hh = h;
            ww = w;
            stack.extend_from_slice(&p);
        }
        let mut kps = Vec::new();
        for layer in 1..=cfg.n_octave_layers {
            find_extrema(&stack, ww, hh, n_dog, layer, 0, &cfg, &mut kps);
        }

        let mut got: Vec<(u32, u32, u32)> = Vec::new();
        for layer in 1..=cfg.n_octave_layers {
            let Some((h2, w2, img)) = load_dump(&format!("{dir}/gauss_o0_l{layer}.f32")) else {
                return;
            };
            let mut o = Vec::new();

            for kp in kps.iter().filter(|k| k.layer == layer as i32) {
                assign_orientations(&img, w2, h2, kp, &cfg, &mut o);
            }
            for k in o {
                // Reference keypoints are stored after the first-octave halving.
                got.push((
                    (k.x * 0.5).to_bits(),
                    (k.y * 0.5).to_bits(),
                    k.angle.to_bits(),
                ));
            }
        }

        let b = std::fs::read(format!("{dir}/keypoints.bin")).expect("keypoints");
        let n = i32::from_le_bytes(b[0..4].try_into().unwrap()) as usize;
        use std::collections::HashMap;
        let mut want: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
        for i in 0..n {
            let o = 4 + i * 24;
            let packed = i32::from_le_bytes(b[o + 20..o + 24].try_into().unwrap());
            if (packed & 255) != 255 {
                continue;
            }
            let f = |k: usize| f32::from_le_bytes(b[o + k * 4..o + k * 4 + 4].try_into().unwrap());
            want.entry((f(0).to_bits(), f(1).to_bits()))
                .or_default()
                .push(f(3).to_bits());
        }
        let mut have: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
        for (x, y, a) in got {
            have.entry((x, y)).or_default().push(a);
        }

        let (mut matched, mut bad) = (0usize, 0usize);
        for (pos, wa) in &want {
            if let Some(ga) = have.get(pos) {
                matched += 1;
                let (mut a, mut c) = (wa.clone(), ga.clone());
                a.sort_unstable();
                c.sort_unstable();
                // One extremum can be reached from neighbouring start pixels and
                // refine onto the same point, so this stage emits it more than
                // once; the reference's `removeDuplicatedSorted` runs AFTER
                // orientation and collapses them. Compare like for like — the
                // assembled pipeline dedups, and `end_to_end` covers that.
                a.dedup();
                c.dedup();
                if a != c {
                    bad += 1;
                    if std::env::var("KORNIA_SIFT_ORIDBG").is_ok() {
                        let f =
                            |v: &Vec<u32>| v.iter().map(|x| f32::from_bits(*x)).collect::<Vec<_>>();
                        eprintln!("    want={:?} got={:?}", f(&a), f(&c));
                    }
                }
            }
        }
        eprintln!(
            "  cpu orientation: positions matched={}/{} angle_set_mismatch={}",
            matched,
            want.len(),
            bad
        );
        assert_eq!(matched, want.len(), "positions dropped by orientation");
        assert_eq!(bad, 0, "{bad} positions have a different angle set");
    }
}
