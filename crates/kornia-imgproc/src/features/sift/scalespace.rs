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

/// Horizontal (row) pass, optionally exploiting the kernel's symmetry.
///
/// The interior is vectorised four columns at a time with unaligned loads: the
/// taps are consecutive, so tap `j` for lanes `x..x+3` is just the source vector
/// at `x - n2 + j`. Only the `n2` columns at each edge need reflection.
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

    // Rows are independent in both passes. The reference parallelises its
    // extrema search and descriptors but builds the pyramid serially, which is
    // most of why it scales 2.15x across six cores here where this does 3.2x.
    //
    // Falsified 2026-07-27: making the split adaptive so small octaves get one
    // task per worker instead of a fixed 16 rows (a 60-row plane otherwise
    // yields 4 tasks for 6 threads) measured no change, p = 0.15. Rayon's
    // work-stealing already absorbs it; task granularity is not what limits the
    // blur's 2.65x scaling.
    dst.par_chunks_mut(w * ROWS_PER_TASK)
        .enumerate()
        .for_each_init(Vec::new, |ext: &mut Vec<f32>, (chunk, dchunk)| {
            let y0 = chunk * ROWS_PER_TASK;
            for (yy, d) in dchunk.chunks_mut(w).enumerate() {
                let row = (y0 + yy) * w;
                let s = &src[row..row + w];

                if !symmetric && w > 2 * n2 {
                    // Materialise the reflected edges once and run the vector
                    // interior over the whole row (FilterEngine's shape). The
                    // old border loops paid `ksize` scalar serial-latency FMAs
                    // and a `refl101` per TAP per border pixel — ~872 scalar
                    // taps/row at the default config, a third of the whole
                    // vector interior's cost, and the ratio doubles each
                    // octave. `row_interior` over the extended row computes
                    // the identical seed-then-ascending-FMA expression on the
                    // identical tap values, so this is bit-exact.
                    ext.resize(w + 2 * n2, 0.0);
                    ext[n2..n2 + w].copy_from_slice(s);
                    let wi = w as i64;
                    for t in 0..n2 {
                        ext[t] = s[refl101(t as i64 - n2 as i64, wi)];
                        ext[n2 + w + t] = s[refl101((w + t) as i64, wi)];
                    }
                    row_interior(ext, d, n2, n2, kernel);
                    continue;
                }

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
    // One argument, not two Options: the DoG difference needs *both* the lower
    // layer and somewhere to put the result, and as separate Options
    // `dog: Some, lower: None` type-checks and silently leaves the caller's
    // buffer untouched. The CUDA side avoids the same trap by exposing two
    // launchers rather than one with optional halves.
    dog: Option<(&[f32], &mut [f32])>,
) {
    let n2 = kernel.len() / 2;
    let hh = h as i64;

    match dog {
        Some((lower, dg)) => dst
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
                    column_row::<true>(
                        src,
                        out,
                        &lower[y * w..y * w + w],
                        g,
                        w,
                        c0,
                        y,
                        n2,
                        hh,
                        kernel,
                        &mut pairs,
                    );
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
                    column_row::<false>(
                        src,
                        out,
                        &[],
                        &mut [],
                        w,
                        c0,
                        y,
                        n2,
                        hh,
                        kernel,
                        &mut pairs,
                    );
                }
            }),
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn column_row<const DOG: bool>(
    src: &[f32],
    out: &mut [f32],
    lo: &[f32],
    dg: &mut [f32],
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
                // The DoG difference is taken from the accumulator that is
                // still in a register, not by reading `out` back: the layer and
                // its difference are produced by the same store pass.
                if DOG {
                    let l = lo.as_ptr().add(x);
                    let d = dg.as_mut_ptr().add(x);
                    vst1q_f32(d, vsubq_f32(a0, vld1q_f32(l)));
                    vst1q_f32(d.add(4), vsubq_f32(a1, vld1q_f32(l.add(4))));
                    vst1q_f32(d.add(8), vsubq_f32(a2, vld1q_f32(l.add(8))));
                    vst1q_f32(d.add(12), vsubq_f32(a3, vld1q_f32(l.add(12))));
                }
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
                if DOG {
                    vst1q_f32(
                        dg.as_mut_ptr().add(x),
                        vsubq_f32(acc, vld1q_f32(lo.as_ptr().add(x))),
                    );
                }
                x += 4;
            }
        }
        while x < w {
            let mut acc = src[c0 + x] * k0;
            for &(a, b, c) in pairs {
                acc = (src[a + x] + src[b + x]).mul_add(c, acc);
            }
            out[x] = acc;
            if DOG {
                dg[x] = acc - lo[x];
            }
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
        if DOG {
            dg[x] = acc - lo[x];
        }
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

    /// The fused DoG store must equal the difference the caller would have
    /// computed from the written layer.
    ///
    /// Runs unconditionally and on every architecture, unlike the oracle tests.
    /// The vertical pass writes the layer and its difference from one
    /// accumulator, and the scalar fallback for non-NEON targets is a separate
    /// piece of code that this host never executes — it shipped without the
    /// difference store once, and only a portable test catches that.
    #[test]
    fn fused_dog_equals_the_explicit_difference() {
        let (w, h) = (37usize, 23usize);
        let mut seed = 0x9E3779B9u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((seed >> 9) % 1000) as f32 * 0.25
        };
        let src: Vec<f32> = (0..w * h).map(|_| rnd()).collect();
        let lower: Vec<f32> = (0..w * h).map(|_| rnd()).collect();
        let kernel = gaussian_kernel_f32(gaussian_ksize(1.6), 1.6);

        let mut layer = vec![0.0f32; w * h];
        let mut dog = vec![0.0f32; w * h];
        blur_v_f32(&src, &mut layer, w, h, &kernel, Some((&lower, &mut dog)));

        let mut plain = vec![0.0f32; w * h];
        blur_v_f32(&src, &mut plain, w, h, &kernel, None);

        assert_eq!(layer, plain, "fusing the DoG changed the layer");
        for i in 0..w * h {
            assert_eq!(
                dog[i].to_bits(),
                (layer[i] - lower[i]).to_bits(),
                "dog[{i}] disagrees with layer - lower"
            );
        }
        assert!(dog.iter().any(|v| *v != 0.0), "dog was never written");
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
            // `symmetric = false`: this compares against the reference dump, and
            // the reference's row filter does not pair its taps.
            blur_h_f32_mode(&prev, &mut tmp, w, h, &k, false);
            blur_v_f32(&tmp, &mut got, w, h, &k, None);

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
