//! Extrema detection and sub-pixel refinement, bit-exact against the reference.
//!
//! # Why the expression shapes are spelled out
//!
//! The Hessian solved here is near-singular — entries around 2.5e-3 with a
//! determinant around 1e-9, so the cofactor products cancel to about six decimal
//! digits. A single-ULP difference anywhere upstream is amplified roughly 1e5
//! into the solved offset. That makes every contraction load-bearing, and it is
//! why these helpers exist rather than the obvious arithmetic.
//!
//! `f32::mul_add` is the exact twin of the fused ops the reference's build emits.

use rayon::prelude::*;

use super::params::SiftConfig;

/// Border in pixels excluded from the extrema search (`SIFT_IMG_BORDER`).
pub const IMG_BORDER: usize = 5;
/// Maximum refinement iterations (`SIFT_MAX_INTERP_STEPS`).
pub const MAX_INTERP_STEPS: usize = 5;

/// `a*b - c*d` with the reference build's contraction: `c*d` rounds, then fuses.
#[inline(always)]
fn mul_sub(a: f32, b: f32, c: f32, d: f32) -> f32 {
    a.mul_add(b, -(c * d))
}

/// `p0*p1 - q0*q1 + r0*r1`, chained left to right as two fused ops.
#[inline(always)]
fn outer3(p0: f32, p1: f32, q0: f32, q1: f32, r0: f32, r1: f32) -> f32 {
    r0.mul_add(r1, p0.mul_add(p1, -(q0 * q1)))
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn det3(
    a00: f32,
    a01: f32,
    a02: f32,
    a10: f32,
    a11: f32,
    a12: f32,
    a20: f32,
    a21: f32,
    a22: f32,
) -> f32 {
    let m0 = mul_sub(a11, a22, a21, a12);
    let m1 = mul_sub(a10, a22, a20, a12);
    let m2 = mul_sub(a10, a21, a20, a11);
    outer3(a00, m0, a01, m1, a02, m2)
}

/// Round half away from zero, matching `cvRound`'s behaviour on this path.
#[inline(always)]
fn cv_round(v: f32) -> i32 {
    // Ties-to-even, which is what `cvRound` compiles to here.
    let r = v.round_ties_even();
    r as i32
}

/// Integer contrast threshold: `cvFloor(0.5 * contrast / n_layers * 255)`.
///
/// Images are f32 in 0..255, not 0..1, and the reference floors this to an
/// integer before comparing — a detail that changes which extrema survive.
pub fn extrema_threshold(cfg: &SiftConfig) -> f32 {
    (0.5 * cfg.contrast_threshold / cfg.n_octave_layers as f64 * 255.0).floor() as f32
}

/// A refined keypoint, in the coordinates of the pyramid's base.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RawKeypoint {
    /// Column in base-image pixels.
    pub x: f32,
    /// Row in base-image pixels.
    pub y: f32,
    /// Diameter of the meaningful neighbourhood.
    pub size: f32,
    /// Contrast at the refined extremum.
    pub response: f32,
    /// Packed `octave | (layer << 8) | (round((xi + 0.5) * 255) << 16)`.
    pub octave: i32,
    /// Layer within the octave.
    pub layer: i32,
    /// Sub-layer offset from refinement.
    pub xi: f32,
    /// Integer column the refinement converged to.
    pub cc: i32,
    /// Integer row the refinement converged to.
    pub rr: i32,
}

/// Search one DoG layer for extrema and refine the survivors.
///
/// `dog` holds `n_dog` contiguous planes of `w * h`. `layer` must be in
/// `1..n_dog-1` so the neighbours above and below exist.
#[allow(clippy::too_many_arguments)]
pub fn find_extrema(
    dog: &[f32],
    w: usize,
    h: usize,
    n_dog: usize,
    layer: usize,
    octv: usize,
    cfg: &SiftConfig,
    out: &mut Vec<RawKeypoint>,
) {
    debug_assert!(layer >= 1 && layer + 1 < n_dog);
    if w <= 2 * IMG_BORDER || h <= 2 * IMG_BORDER {
        return;
    }
    let plane = w * h;
    let thr = extrema_threshold(cfg);
    let cur = &dog[layer * plane..(layer + 1) * plane];
    let prv = &dog[(layer - 1) * plane..layer * plane];
    let nxt = &dog[(layer + 1) * plane..(layer + 2) * plane];

    // Rows in parallel. The scan is read-only over the DoG and each row owns its
    // output, so the only thing this loses is the append order — which the
    // reference's final sort normalises anyway.
    let mut bands: Vec<(usize, Vec<RawKeypoint>)> = (IMG_BORDER..h - IMG_BORDER)
        .into_par_iter()
        .map(|r| {
            let mut acc = Vec::new();
            // The exact test, unchanged. Everything below only decides which
            // columns reach it.
            let check = |c: usize, acc: &mut Vec<RawKeypoint>| {
                let val = cur[r * w + c];
                if val.abs() <= thr {
                    return;
                }
                // 26-neighbour test: strictly greater than, or strictly less
                // than, every neighbour in the 3x3x3 cube.
                let mut is_max = true;
                let mut is_min = true;
                'nb: for (pi, pl) in [prv, cur, nxt].into_iter().enumerate() {
                    for dr in -1i32..=1 {
                        let rr = (r as i32 + dr) as usize * w;
                        for dc in -1i32..=1 {
                            // The centre is not its own neighbour: comparing it
                            // against itself fails both tests and rejects every
                            // candidate.
                            if pi == 1 && dr == 0 && dc == 0 {
                                continue;
                            }
                            let n = pl[rr + (c as i32 + dc) as usize];
                            if n >= val {
                                is_max = false;
                            }
                            if n <= val {
                                is_min = false;
                            }
                            if !is_max && !is_min {
                                break 'nb;
                            }
                        }
                    }
                }
                if !(is_max || is_min) {
                    return;
                }
                if let Some(kp) = refine(dog, w, h, n_dog, layer, r, c, octv, cfg) {
                    acc.push(kp);
                }
            };

            let (c_lo, c_hi) = (IMG_BORDER, w - IMG_BORDER);
            let mut c = c_lo;
            // Vector prefilter. Measured on apriltags: 34.1% of pixels clear the
            // contrast threshold and enter the 26-neighbour test, but only 0.09%
            // are extrema — so the scalar path spends its time proving negatives.
            //
            // `val >= max(neighbours)` is implied by the strict `val > every
            // neighbour`, so a lane the exact test would accept can never be
            // dropped here; `check` still decides. This is the same conservative
            // prefilter `findScaleSpaceExtrema` uses, including its behaviour on
            // a non-finite DoG, where `FMAX` drops NaN.
            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::*;
                // SAFETY: `r` is in `IMG_BORDER..h - IMG_BORDER` so `r - 1` and
                // `r + 1` are in range, and the loop guard keeps `c + 4` (the
                // widest column touched, from the `c + 1` shift of a 4-wide
                // block) below `w`.
                unsafe {
                    let vthr = vdupq_n_f32(thr);
                    while c + 4 <= c_hi {
                        let val = vld1q_f32(cur.as_ptr().add(r * w + c));
                        // |val| > thr, the same rejection as the scalar head.
                        let mut cond = vcgtq_f32(vabsq_f32(val), vthr);
                        if vmaxvq_u32(cond) == 0 {
                            c += 4;
                            continue;
                        }
                        let (mut vmax, mut vmin) =
                            (vdupq_n_f32(f32::NEG_INFINITY), vdupq_n_f32(f32::INFINITY));
                        for pl in [prv, cur, nxt] {
                            for dr in 0..3usize {
                                let base = pl.as_ptr().add((r + dr - 1) * w + c);
                                for dc in 0..3usize {
                                    // The centre is not its own neighbour.
                                    if std::ptr::eq(pl.as_ptr(), cur.as_ptr()) && dr == 1 && dc == 1
                                    {
                                        continue;
                                    }
                                    let n = vld1q_f32(base.add(dc).sub(1));
                                    vmax = vmaxq_f32(vmax, n);
                                    vmin = vminq_f32(vmin, n);
                                }
                            }
                        }
                        cond =
                            vandq_u32(cond, vorrq_u32(vcgeq_f32(val, vmax), vcleq_f32(val, vmin)));
                        if vmaxvq_u32(cond) != 0 {
                            let mut lanes = [0u32; 4];
                            vst1q_u32(lanes.as_mut_ptr(), cond);
                            for (k, &l) in lanes.iter().enumerate() {
                                if l != 0 {
                                    check(c + k, &mut acc);
                                }
                            }
                        }
                        c += 4;
                    }
                }
            }
            while c < c_hi {
                check(c, &mut acc);
                c += 1;
            }
            (r, acc)
        })
        .filter(|(_, v)| !v.is_empty())
        .collect();
    // Restore row order so a run is reproducible before the final sort.
    bands.sort_by_key(|(r, _)| *r);
    for (_, b) in bands {
        out.extend_from_slice(&b);
    }
}

/// `adjustLocalExtrema`: iterate the 3-D quadratic fit, then reject on contrast
/// and principal-curvature ratio.
#[allow(clippy::too_many_arguments)]
fn refine(
    dog: &[f32],
    w: usize,
    h: usize,
    n_dog: usize,
    layer0: usize,
    r0: usize,
    c0: usize,
    octv: usize,
    cfg: &SiftConfig,
) -> Option<RawKeypoint> {
    const IMG_SCALE: f32 = 1.0 / 255.0;
    const DERIV: f32 = IMG_SCALE * 0.5;
    const SECOND: f32 = IMG_SCALE;
    const CROSS: f32 = IMG_SCALE * 0.25;
    let plane = w * h;

    let (mut rr, mut cc, mut layer) = (r0 as i32, c0 as i32, layer0 as i32);
    let (mut xi, mut xr, mut xc) = (0.0f32, 0.0f32, 0.0f32);
    let mut step = 0usize;

    let at = |pl: &[f32], r: i32, c: i32| pl[r as usize * w + c as usize];

    while step < MAX_INTERP_STEPS {
        let im = &dog[layer as usize * plane..];
        let pv = &dog[(layer - 1) as usize * plane..];
        let nx = &dog[(layer + 1) as usize * plane..];

        let d_dx = (at(im, rr, cc + 1) - at(im, rr, cc - 1)) * DERIV;
        let d_dy = (at(im, rr + 1, cc) - at(im, rr - 1, cc)) * DERIV;
        let d_ds = (at(nx, rr, cc) - at(pv, rr, cc)) * DERIV;

        let v2 = at(im, rr, cc) * 2.0;
        let dxx = (at(im, rr, cc + 1) + at(im, rr, cc - 1) - v2) * SECOND;
        let dyy = (at(im, rr + 1, cc) + at(im, rr - 1, cc) - v2) * SECOND;
        let dss = (at(nx, rr, cc) + at(pv, rr, cc) - v2) * SECOND;
        let dxy = (at(im, rr + 1, cc + 1) - at(im, rr + 1, cc - 1) - at(im, rr - 1, cc + 1)
            + at(im, rr - 1, cc - 1))
            * CROSS;
        let dxs = (at(nx, rr, cc + 1) - at(nx, rr, cc - 1) - at(pv, rr, cc + 1)
            + at(pv, rr, cc - 1))
            * CROSS;
        let dys = (at(nx, rr + 1, cc) - at(nx, rr - 1, cc) - at(pv, rr + 1, cc)
            + at(pv, rr - 1, cc))
            * CROSS;

        let det = det3(dxx, dxy, dxs, dxy, dyy, dys, dxs, dys, dss);
        if det == 0.0 {
            return None;
        }
        let d = 1.0f32 / det;

        let x0 = d * outer3(
            d_dx,
            mul_sub(dyy, dss, dys, dys),
            dxy,
            mul_sub(d_dy, dss, dys, d_ds),
            dxs,
            mul_sub(d_dy, dys, dyy, d_ds),
        );
        let x1 = d * outer3(
            dxx,
            mul_sub(d_dy, dss, dys, d_ds),
            d_dx,
            mul_sub(dxy, dss, dys, dxs),
            dxs,
            mul_sub(dxy, d_ds, d_dy, dxs),
        );
        // x2's first cofactor is written `a11*b2 - b1*a21` in the reference, but
        // x0's third term is its exact negation. Common subexpression
        // elimination evaluates the pair ONCE and negates, and under the
        // reference's contraction the two spellings are not bitwise negations:
        // one rounds `a11*b2` and fuses `b1*a21`, the other does the reverse.
        // Mirror x0's spelling and negate, or x2 lands 1-3 ULP out on ~30% of
        // keypoints — which `size` then exposes.
        let x2 = d * outer3(
            dxx,
            -mul_sub(d_dy, dys, dyy, d_ds),
            dxy,
            mul_sub(dxy, d_ds, d_dy, dxs),
            d_dx,
            mul_sub(dxy, dys, dyy, dxs),
        );

        xi = -x2;
        xr = -x1;
        xc = -x0;

        if xi.abs() < 0.5 && xr.abs() < 0.5 && xc.abs() < 0.5 {
            break;
        }
        const LIM: f32 = 715_827_882.0; // INT_MAX / 3
        if xi.abs() > LIM || xr.abs() > LIM || xc.abs() > LIM {
            return None;
        }
        cc += cv_round(xc);
        rr += cv_round(xr);
        layer += cv_round(xi);

        if layer < 1
            || layer > cfg.n_octave_layers as i32
            || cc < IMG_BORDER as i32
            || cc >= (w - IMG_BORDER) as i32
            || rr < IMG_BORDER as i32
            || rr >= (h - IMG_BORDER) as i32
        {
            return None;
        }
        step += 1;
    }
    if step >= MAX_INTERP_STEPS {
        return None;
    }
    let _ = n_dog;

    let im = &dog[layer as usize * plane..];
    let pv = &dog[(layer - 1) as usize * plane..];
    let nx = &dog[(layer + 1) as usize * plane..];

    let d_dx = (at(im, rr, cc + 1) - at(im, rr, cc - 1)) * DERIV;
    let d_dy = (at(im, rr + 1, cc) - at(im, rr - 1, cc)) * DERIV;
    let d_ds = (at(nx, rr, cc) - at(pv, rr, cc)) * DERIV;
    // `Matx::dot` accumulates `s += a[i]*b[i]`, which contracts to an FMA chain
    // seeded by a plain multiply.
    let mut t = d_dx * xc;
    t = d_dy.mul_add(xr, t);
    t = d_ds.mul_add(xi, t);

    let contr = at(im, rr, cc).mul_add(IMG_SCALE, t * 0.5);
    let contr_scaled = contr.abs() * cfg.n_octave_layers as f32;
    if contr_scaled < cfg.contrast_threshold as f32 {
        return None;
    }

    let v2 = at(im, rr, cc) * 2.0;
    let dxx = (at(im, rr, cc + 1) + at(im, rr, cc - 1) - v2) * SECOND;
    let dyy = (at(im, rr + 1, cc) + at(im, rr - 1, cc) - v2) * SECOND;
    let dxy = (at(im, rr + 1, cc + 1) - at(im, rr + 1, cc - 1) - at(im, rr - 1, cc + 1)
        + at(im, rr - 1, cc - 1))
        * CROSS;
    let tr = dxx + dyy;
    let det = mul_sub(dxx, dyy, dxy, dxy);
    let et = cfg.edge_threshold as f32;
    if det <= 0.0 || tr * tr * et >= (et + 1.0) * (et + 1.0) * det {
        return None;
    }

    let scale = (1 << octv) as f32;
    // The reference calls host `powf(2.f, y)`, which glibc evaluates as exactly
    // its own `exp2f` — and Rust's `f32::exp2` lowers to that same libm symbol,
    // so on this target the CPU path gets it for free where CUDA needed a port.
    let pw = ((layer as f32 + xi) / cfg.n_octave_layers as f32).exp2();
    Some(RawKeypoint {
        x: (cc as f32 + xc) * scale,
        y: (rr as f32 + xr) * scale,
        size: cfg.sigma as f32 * pw * scale * 2.0,
        response: contr.abs(),
        octave: octv as i32 + (layer << 8) + (cv_round((xi + 0.5) * 255.0) << 16),
        layer,
        xi,
        cc,
        rr,
    })
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

    /// Detection on the reference's own dumped DoG planes, compared against its
    /// keypoints bit for bit — the same oracle and the same standard the CUDA
    /// detector is held to.
    #[test]
    fn detect_matches_reference_bitwise() {
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
            let Some((h, w, plane)) = load_dump(&format!("{dir}/dog_o0_l{i}.f32")) else {
                eprintln!("no dog dumps; skipping");
                return;
            };
            hh = h;
            ww = w;
            stack.extend_from_slice(&plane);
        }
        let mut got = Vec::new();
        for layer in 1..=cfg.n_octave_layers {
            find_extrema(&stack, ww, hh, n_dog, layer, 0, &cfg, &mut got);
        }

        // Reference keypoints for octave -1 (the doubled base), pre-rescale.
        let b = std::fs::read(format!("{dir}/keypoints.bin")).expect("keypoints");
        let n = i32::from_le_bytes(b[0..4].try_into().unwrap()) as usize;
        let mut want: Vec<(u32, u32)> = Vec::new();
        for i in 0..n {
            let o = 4 + i * 24;
            let packed = i32::from_le_bytes(b[o + 20..o + 24].try_into().unwrap());
            if (packed & 255) != 255 {
                continue;
            }
            let f = |k: usize| f32::from_le_bytes(b[o + k * 4..o + k * 4 + 4].try_into().unwrap());
            // Undo the first-octave halving to get the octave's own frame.
            want.push(((f(0) * 2.0).to_bits(), (f(1) * 2.0).to_bits()));
        }
        want.sort_unstable();
        want.dedup();

        use std::collections::HashSet;
        let have: HashSet<(u32, u32)> =
            got.iter().map(|k| (k.x.to_bits(), k.y.to_bits())).collect();
        let missing = want.iter().filter(|p| !have.contains(p)).count();
        eprintln!(
            "  cpu detect: produced={} reference={} exact_pos_match={}",
            got.len(),
            want.len(),
            want.len() - missing
        );
        assert_eq!(missing, 0, "{missing} reference positions not reproduced");
    }
}
