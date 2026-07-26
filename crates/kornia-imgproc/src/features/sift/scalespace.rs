//! NEON separable Gaussian blur and scale-space construction for CPU SIFT.
//!
//! # Numerics
//!
//! Bit-exact against the reference, which means reproducing two *different*
//! accumulation shapes — this is not a symmetric filter applied twice:
//!
//! ```text
//! row:    acc = s[0] * k[0];              acc = fma(s[j],          k[j], acc)
//! column: acc = fma(s[0], k[0], 0);       acc = fma(s[j] + s[-j],  k[j], acc)
//! ```
//!
//! The column pass sums the symmetric pair *before* multiplying, the row pass
//! does not. Getting that backwards changes the last bit of most outputs.
//!
//! `vfmaq_f32` is the exact twin of `f32::mul_add`, and `vfmaq_n_f32` broadcasts
//! a scalar coefficient, so the vector and scalar paths agree by construction
//! rather than by tolerance — the border columns fall out of the vector loop and
//! must produce identical bits to the interior.

use rayon::prelude::*;

use super::refl101;

/// Rows per rayon task. Matches the crate's existing sharding granularity: at
/// one row per task the ~2-5 us spawn overhead rivals the per-row work, and the
/// pyramid's upper octaves have few rows to begin with.
const ROWS_PER_TASK: usize = 16;

// Falsified 2026-07-26: tiling the vertical pass into column strips so its
// `ksize`-row window fits L1 (162 KB at the base octave, so it does not) made
// the blur *slower* — 74.4 -> 79.0 ms at a 128-column strip, even with the
// reflected row bases hoisted out of the strip loop. The full-width walk feeds
// the hardware prefetcher long sequential streams, and losing that costs more
// than L1 residency returns. Do not retry without addressing the prefetcher.

/// Horizontal (row) pass.
///
/// The interior is vectorised four columns at a time with unaligned loads: the
/// taps are consecutive, so tap `j` for lanes `x..x+3` is just the source vector
/// at `x - n2 + j`. Only the `n2` columns at each edge need reflection.
pub fn blur_h_f32(src: &[f32], dst: &mut [f32], w: usize, h: usize, kernel: &[f32]) {
    blur_h_f32_mode(src, dst, w, h, kernel, false)
}

/// Horizontal pass, optionally exploiting the kernel's symmetry.
///
/// `symmetric` halves the multiply-adds by folding tap pairs before the
/// multiply — `(s[n2-j] + s[n2+j]) * k[n2+j]` — which is what a normal
/// implementation does and what the *column* pass already does.
///
/// It is not the default because the reference's row filter does **not** pair:
/// it accumulates all `ksize` taps individually, and reproducing that rounding
/// is what makes this bit-exact. So bit-exactness costs roughly twice the
/// multiply-adds on this pass, and this flag is where that cost is given back.
pub fn blur_h_f32_mode(
    src: &[f32],
    dst: &mut [f32],
    w: usize,
    h: usize,
    kernel: &[f32],
    symmetric: bool,
) {
    debug_assert_eq!(src.len(), w * h);
    debug_assert_eq!(dst.len(), w * h);
    let n = kernel.len();
    let n2 = n / 2;

    // Rows are independent in both passes -- this is exactly the parallelism the
    // reference gives up, because its pyramid is built layer by layer on one
    // thread (measured: cv2 SIFT scales 1.13x across six cores).
    dst.par_chunks_mut(w * ROWS_PER_TASK)
        .enumerate()
        .for_each(|(chunk, dchunk)| {
            let y0 = chunk * ROWS_PER_TASK;
            for (yy, d) in dchunk.chunks_mut(w).enumerate() {
                let row = (y0 + yy) * w;
                let s = &src[row..row + w];

                // Left border.
                let lo = n2.min(w);
                for (x, out) in d.iter_mut().enumerate().take(lo) {
                    *out = row_border(s, w, x, n2, kernel);
                }
                // Right border.
                let hi = w.saturating_sub(n2).max(lo);
                for (x, out) in d.iter_mut().enumerate().skip(hi) {
                    *out = row_border(s, w, x, n2, kernel);
                }
                if hi > lo {
                    if symmetric {
                        row_interior_sym(s, &mut d[lo..hi], lo, n2, kernel);
                    } else {
                        row_interior(s, &mut d[lo..hi], lo, n2, kernel);
                    }
                }
            }
        });
}

#[inline]
fn row_border(s: &[f32], w: usize, x: usize, n2: usize, kernel: &[f32]) -> f32 {
    let n = w as i64;
    let mut acc = s[refl101(x as i64 - n2 as i64, n)] * kernel[0];
    for (j, &c) in kernel.iter().enumerate().skip(1) {
        acc = s[refl101(x as i64 - n2 as i64 + j as i64, n)].mul_add(c, acc);
    }
    acc
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn row_interior(s: &[f32], d: &mut [f32], x0: usize, n2: usize, kernel: &[f32]) {
    use std::arch::aarch64::*;
    let len = d.len();
    let mut i = 0usize;
    // Four output vectors per iteration, each with its OWN accumulator.
    //
    // The tap order within any one output is untouched, so this stays bit-exact;
    // what it buys is four independent dependency chains. With a single
    // accumulator the loop is `acc = fma(acc, ...)` repeated `ksize` times, and
    // at ~4-cycle FMA latency that is latency-bound, not throughput-bound:
    // measured 0.98 vector ops/cycle against the ~2 this core can sustain.
    // SAFETY: the caller's interior bounds keep every load inside the row.
    unsafe {
        while i + 16 <= len {
            let b = s.as_ptr().add(x0 + i - n2);
            let k0 = kernel[0];
            let mut a0 = vmulq_n_f32(vld1q_f32(b), k0);
            let mut a1 = vmulq_n_f32(vld1q_f32(b.add(4)), k0);
            let mut a2 = vmulq_n_f32(vld1q_f32(b.add(8)), k0);
            let mut a3 = vmulq_n_f32(vld1q_f32(b.add(12)), k0);
            for (j, &c) in kernel.iter().enumerate().skip(1) {
                let p = b.add(j);
                a0 = vfmaq_n_f32(a0, vld1q_f32(p), c);
                a1 = vfmaq_n_f32(a1, vld1q_f32(p.add(4)), c);
                a2 = vfmaq_n_f32(a2, vld1q_f32(p.add(8)), c);
                a3 = vfmaq_n_f32(a3, vld1q_f32(p.add(12)), c);
            }
            let o = d.as_mut_ptr().add(i);
            vst1q_f32(o, a0);
            vst1q_f32(o.add(4), a1);
            vst1q_f32(o.add(8), a2);
            vst1q_f32(o.add(12), a3);
            i += 16;
        }
        while i + 4 <= len {
            let b = s.as_ptr().add(x0 + i - n2);
            let mut acc = vmulq_n_f32(vld1q_f32(b), kernel[0]);
            for (j, &c) in kernel.iter().enumerate().skip(1) {
                acc = vfmaq_n_f32(acc, vld1q_f32(b.add(j)), c);
            }
            vst1q_f32(d.as_mut_ptr().add(i), acc);
            i += 4;
        }
    }
    // Scalar tail, identical arithmetic.
    while i < len {
        let base = x0 + i - n2;
        let mut acc = s[base] * kernel[0];
        for (j, &c) in kernel.iter().enumerate().skip(1) {
            acc = s[base + j].mul_add(c, acc);
        }
        d[i] = acc;
        i += 1;
    }
}

/// Symmetric row interior: half the taps, four independent accumulators.
#[cfg(target_arch = "aarch64")]
#[inline]
fn row_interior_sym(s: &[f32], d: &mut [f32], x0: usize, n2: usize, kernel: &[f32]) {
    use std::arch::aarch64::*;
    let len = d.len();
    let mut i = 0usize;
    // SAFETY: the caller's interior bounds keep every load inside the row.
    unsafe {
        while i + 16 <= len {
            let ctr = s.as_ptr().add(x0 + i);
            let k0 = kernel[n2];
            let mut a0 = vmulq_n_f32(vld1q_f32(ctr), k0);
            let mut a1 = vmulq_n_f32(vld1q_f32(ctr.add(4)), k0);
            let mut a2 = vmulq_n_f32(vld1q_f32(ctr.add(8)), k0);
            let mut a3 = vmulq_n_f32(vld1q_f32(ctr.add(12)), k0);
            for j in 1..=n2 {
                let c = kernel[n2 + j];
                let p = ctr.add(j);
                let m = ctr.sub(j);
                a0 = vfmaq_n_f32(a0, vaddq_f32(vld1q_f32(p), vld1q_f32(m)), c);
                a1 = vfmaq_n_f32(a1, vaddq_f32(vld1q_f32(p.add(4)), vld1q_f32(m.add(4))), c);
                a2 = vfmaq_n_f32(a2, vaddq_f32(vld1q_f32(p.add(8)), vld1q_f32(m.add(8))), c);
                a3 = vfmaq_n_f32(a3, vaddq_f32(vld1q_f32(p.add(12)), vld1q_f32(m.add(12))), c);
            }
            let o = d.as_mut_ptr().add(i);
            vst1q_f32(o, a0);
            vst1q_f32(o.add(4), a1);
            vst1q_f32(o.add(8), a2);
            vst1q_f32(o.add(12), a3);
            i += 16;
        }
    }
    while i < len {
        let ctr = x0 + i;
        let mut acc = s[ctr] * kernel[n2];
        for j in 1..=n2 {
            acc = (s[ctr + j] + s[ctr - j]).mul_add(kernel[n2 + j], acc);
        }
        d[i] = acc;
        i += 1;
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn row_interior_sym(s: &[f32], d: &mut [f32], x0: usize, n2: usize, kernel: &[f32]) {
    for (i, out) in d.iter_mut().enumerate() {
        let ctr = x0 + i;
        let mut acc = s[ctr] * kernel[n2];
        for j in 1..=n2 {
            acc = (s[ctr + j] + s[ctr - j]).mul_add(kernel[n2 + j], acc);
        }
        *out = acc;
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn row_interior(s: &[f32], d: &mut [f32], x0: usize, n2: usize, kernel: &[f32]) {
    for (i, out) in d.iter_mut().enumerate() {
        let base = x0 + i - n2;
        let mut acc = s[base] * kernel[0];
        for (j, &c) in kernel.iter().enumerate().skip(1) {
            acc = s[base + j].mul_add(c, acc);
        }
        *out = acc;
    }
}

/// Vertical (column) pass, optionally fused with the DoG subtract.
///
/// Whole rows are contiguous, so this vectorises over `x` with no unaligned
/// gathers at all — the natural shape for NEON, and why the column pass is the
/// cheaper of the two despite the symmetric pairing.
///
/// When `lower` is given, `dog` receives `blurred - lower` from the value
/// already in registers, which is the same saving the CUDA path takes.
pub fn blur_v_f32(
    src: &[f32],
    dst: &mut [f32],
    w: usize,
    h: usize,
    kernel: &[f32],
    lower: Option<&[f32]>,
    dog: Option<&mut [f32]>,
) {
    let n2 = kernel.len() / 2;
    let hh = h as i64;

    match dog {
        Some(dg) => dst
            .par_chunks_mut(w * ROWS_PER_TASK)
            .zip(dg.par_chunks_mut(w * ROWS_PER_TASK))
            .enumerate()
            .for_each(|(chunk, (dchunk, gchunk))| {
                let y0 = chunk * ROWS_PER_TASK;
                // One scratch buffer per task, not per row: `column_row` used to
                // allocate this every row, i.e. once per output row per blur.
                let mut pairs = Vec::with_capacity(n2);
                for (yy, (out, g)) in dchunk.chunks_mut(w).zip(gchunk.chunks_mut(w)).enumerate() {
                    let y = y0 + yy;
                    let c0 = refl101(y as i64, hh) * w;
                    column_row(src, out, w, c0, y, n2, hh, kernel, &mut pairs);
                    if let Some(lo) = lower {
                        let lo = &lo[y * w..y * w + w];
                        for ((gv, o), l) in g.iter_mut().zip(out.iter()).zip(lo.iter()) {
                            *gv = *o - *l;
                        }
                    }
                }
            }),
        None => dst
            .par_chunks_mut(w * ROWS_PER_TASK)
            .enumerate()
            .for_each(|(chunk, dchunk)| {
                let y0 = chunk * ROWS_PER_TASK;
                let mut pairs = Vec::with_capacity(n2);
                for (yy, out) in dchunk.chunks_mut(w).enumerate() {
                    let y = y0 + yy;
                    let c0 = refl101(y as i64, hh) * w;
                    column_row(src, out, w, c0, y, n2, hh, kernel, &mut pairs);
                }
            }),
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn column_row(
    src: &[f32],
    out: &mut [f32],
    w: usize,
    c0: usize,
    y: usize,
    n2: usize,
    hh: i64,
    kernel: &[f32],
    pairs: &mut Vec<(usize, usize, f32)>,
) {
    // Precompute the reflected row bases once for the whole row. `pairs` is the
    // caller's per-task scratch, reused across rows.
    pairs.clear();
    for j in 1..=n2 {
        let a = refl101(y as i64 + j as i64, hh) * w;
        let b = refl101(y as i64 - j as i64, hh) * w;
        pairs.push((a, b, kernel[n2 + j]));
    }
    let pairs: &[(usize, usize, f32)] = pairs;
    let k0 = kernel[n2];

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        let mut x = 0usize;
        // Four independent chains, as in the row pass.
        // SAFETY: all offsets are reflected row bases plus `x < w`.
        unsafe {
            while x + 16 <= w {
                let c = src.as_ptr().add(c0 + x);
                let mut a0 = vmulq_n_f32(vld1q_f32(c), k0);
                let mut a1 = vmulq_n_f32(vld1q_f32(c.add(4)), k0);
                let mut a2 = vmulq_n_f32(vld1q_f32(c.add(8)), k0);
                let mut a3 = vmulq_n_f32(vld1q_f32(c.add(12)), k0);
                for &(pa, pb, kc) in pairs {
                    let pa = src.as_ptr().add(pa + x);
                    let pb = src.as_ptr().add(pb + x);
                    a0 = vfmaq_n_f32(a0, vaddq_f32(vld1q_f32(pa), vld1q_f32(pb)), kc);
                    a1 = vfmaq_n_f32(
                        a1,
                        vaddq_f32(vld1q_f32(pa.add(4)), vld1q_f32(pb.add(4))),
                        kc,
                    );
                    a2 = vfmaq_n_f32(
                        a2,
                        vaddq_f32(vld1q_f32(pa.add(8)), vld1q_f32(pb.add(8))),
                        kc,
                    );
                    a3 = vfmaq_n_f32(
                        a3,
                        vaddq_f32(vld1q_f32(pa.add(12)), vld1q_f32(pb.add(12))),
                        kc,
                    );
                }
                let o = out.as_mut_ptr().add(x);
                vst1q_f32(o, a0);
                vst1q_f32(o.add(4), a1);
                vst1q_f32(o.add(8), a2);
                vst1q_f32(o.add(12), a3);
                x += 16;
            }
            while x + 4 <= w {
                let mut acc = vmulq_n_f32(vld1q_f32(src.as_ptr().add(c0 + x)), k0);
                for &(a, b, c) in pairs {
                    let sum = vaddq_f32(
                        vld1q_f32(src.as_ptr().add(a + x)),
                        vld1q_f32(src.as_ptr().add(b + x)),
                    );
                    acc = vfmaq_n_f32(acc, sum, c);
                }
                vst1q_f32(out.as_mut_ptr().add(x), acc);
                x += 4;
            }
        }
        while x < w {
            let mut acc = src[c0 + x] * k0;
            for &(a, b, c) in pairs {
                acc = (src[a + x] + src[b + x]).mul_add(c, acc);
            }
            out[x] = acc;
            x += 1;
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    for x in 0..w {
        let mut acc = src[c0 + x] * k0;
        for &(a, b, c) in pairs {
            acc = (src[a + x] + src[b + x]).mul_add(c, acc);
        }
        out[x] = acc;
    }
}

/// Per-pixel gradient magnitude and angle for one Gaussian layer.
///
/// # Why this is a fusion and not extra work
///
/// Orientation and the descriptor both sample gradients from the same layer, and
/// both recompute `atan2` and `magnitude` per sample — scattered, one sample at
/// a time, because the patch walk is keypoint-relative. Those two values depend
/// only on the pixel, so computing them once per layer converts that scattered
/// scalar work into a contiguous 4-wide sweep, and leaves the samplers with two
/// loads.
///
/// The Gaussian weight is *not* precomputed: its scale factor depends on the
/// keypoint, so it stays where it is.
///
/// Values are identical to the per-sample computation — same inputs, same
/// functions, evaluated earlier — so bit-exactness is preserved.
pub fn gradients(img: &[f32], w: usize, h: usize, mag: &mut [f32], ang: &mut [f32]) {
    debug_assert_eq!(img.len(), w * h);
    if w < 3 || h < 3 {
        mag[..w * h].fill(0.0);
        ang[..w * h].fill(0.0);
        return;
    }
    // Row 0 and row h-1 are never sampled (the callers require `0 < y < h-1`),
    // and neither are columns 0 and w-1.
    mag[..w].fill(0.0);
    ang[..w].fill(0.0);
    mag[(h - 1) * w..h * w].fill(0.0);
    ang[(h - 1) * w..h * w].fill(0.0);

    let rows = 1..h - 1;
    mag[w..(h - 1) * w]
        .par_chunks_mut(w * ROWS_PER_TASK)
        .zip(ang[w..(h - 1) * w].par_chunks_mut(w * ROWS_PER_TASK))
        .enumerate()
        .for_each(|(chunk, (mc, ac))| {
            let y0 = rows.start + chunk * ROWS_PER_TASK;
            for (yy, (mrow, arow)) in mc.chunks_mut(w).zip(ac.chunks_mut(w)).enumerate() {
                let y = y0 + yy;
                if y >= h - 1 {
                    break;
                }
                grad_row(img, w, y, mrow, arow);
            }
        });
}

/// One row of magnitudes and angles.
///
/// Two separate planes, not one interleaved `(mag, ang)` plane. Interleaving was
/// measured on 2026-07-26 and **regressed** — descriptor 98.0 -> 101.6 ms,
/// gradients 30.9 -> 32.2, total +5%. The pair is still two load instructions
/// either way (adjacent addresses do not merge), so there is no gather to save,
/// while `vst2q_f32` adds an interleaving shuffle the two plain stores avoid.
/// Do not retry without a mechanism that addresses both.
#[inline]
fn grad_row(img: &[f32], w: usize, y: usize, mrow: &mut [f32], arow: &mut [f32]) {
    use super::hal::{atan2_deg, magnitude};
    mrow[0] = 0.0;
    arow[0] = 0.0;
    mrow[w - 1] = 0.0;
    arow[w - 1] = 0.0;

    let mut x = 1usize;
    #[cfg(target_arch = "aarch64")]
    {
        use super::hal::x4;
        use std::arch::aarch64::*;
        // SAFETY: `x` stays in `1..w-1`, so every load is inside the row and its
        // vertical neighbours exist for `0 < y < h-1`.
        unsafe {
            while x + 4 < w - 1 {
                let c = img.as_ptr().add(y * w + x);
                let dx = vsubq_f32(vld1q_f32(c.add(1)), vld1q_f32(c.sub(1)));
                let up = img.as_ptr().add((y - 1) * w + x);
                let dn = img.as_ptr().add((y + 1) * w + x);
                let dy = vsubq_f32(vld1q_f32(up), vld1q_f32(dn));
                vst1q_f32(mrow.as_mut_ptr().add(x), x4::magnitude(dx, dy));
                vst1q_f32(arow.as_mut_ptr().add(x), x4::atan2_deg(dy, dx));
                x += 4;
            }
        }
    }
    while x < w - 1 {
        let dx = img[y * w + x + 1] - img[y * w + x - 1];
        let dy = img[(y - 1) * w + x] - img[(y + 1) * w + x];
        mrow[x] = magnitude(dx, dy);
        arow[x] = atan2_deg(dy, dx);
        x += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::super::params::{gaussian_kernel_f32, gaussian_ksize};
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

    /// Bit-exact against the reference's own dumped layers, using the same
    /// oracle the CUDA path is held to.
    #[test]
    fn neon_blur_matches_reference_bitwise() {
        let Some(dir) = std::env::var("KORNIA_SIFT_ORACLE")
            .ok()
            .and_then(|v| v.split(':').next().map(String::from))
        else {
            eprintln!("KORNIA_SIFT_ORACLE unset; skipping");
            return;
        };
        let Some((h, w, base)) = load_dump(&format!("{dir}/gauss_o0_l0.f32")) else {
            eprintln!("no gauss_o0_l0 dump; skipping");
            return;
        };
        let cfg = super::super::params::SiftConfig::default();
        let sigmas = cfg.layer_sigmas();

        let mut prev = base;
        let mut checked = 0usize;
        for (layer, &sigma) in sigmas
            .iter()
            .enumerate()
            .take(cfg.n_octave_layers + 3)
            .skip(1)
        {
            let Some((_, _, want)) = load_dump(&format!("{dir}/gauss_o0_l{layer}.f32")) else {
                break;
            };
            let k = gaussian_kernel_f32(gaussian_ksize(sigma), sigma);
            let mut tmp = vec![0.0f32; w * h];
            let mut got = vec![0.0f32; w * h];
            blur_h_f32(&prev, &mut tmp, w, h, &k);
            blur_v_f32(&tmp, &mut got, w, h, &k, None, None);

            let bad = got
                .iter()
                .zip(&want)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(
                bad,
                0,
                "layer {layer}: {bad} of {} pixels differ",
                got.len()
            );
            checked += 1;
            prev = want;
        }
        assert!(checked > 0, "no layers checked");
        eprintln!("  neon blur: {checked} layers bit-exact");
    }
}
