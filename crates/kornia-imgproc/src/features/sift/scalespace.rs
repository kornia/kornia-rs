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

use super::refl101;

/// Horizontal (row) pass.
///
/// The interior is vectorised four columns at a time with unaligned loads: the
/// taps are consecutive, so tap `j` for lanes `x..x+3` is just the source vector
/// at `x - n2 + j`. Only the `n2` columns at each edge need reflection.
pub fn blur_h_f32(src: &[f32], dst: &mut [f32], w: usize, h: usize, kernel: &[f32]) {
    debug_assert_eq!(src.len(), w * h);
    debug_assert_eq!(dst.len(), w * h);
    let n = kernel.len();
    let n2 = n / 2;

    for y in 0..h {
        let row = y * w;
        let s = &src[row..row + w];
        let d = &mut dst[row..row + w];

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
        if hi <= lo {
            continue;
        }
        row_interior(s, &mut d[lo..hi], lo, n2, kernel);
    }
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
    let mut dog = dog;

    for y in 0..h {
        let row = y * w;
        // Row offsets for the centre tap and each symmetric pair.
        let c0 = refl101(y as i64, hh) * w;
        let out = &mut dst[row..row + w];
        column_row(src, out, w, c0, y, n2, hh, kernel);

        if let (Some(lo), Some(dg)) = (lower, dog.as_deref_mut()) {
            let lo = &lo[row..row + w];
            let dg = &mut dg[row..row + w];
            for ((g, o), l) in dg.iter_mut().zip(out.iter()).zip(lo.iter()) {
                *g = *o - *l;
            }
        }
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
) {
    // Precompute the reflected row bases once for the whole row.
    let mut pairs: Vec<(usize, usize, f32)> = Vec::with_capacity(n2);
    for j in 1..=n2 {
        let a = refl101(y as i64 + j as i64, hh) * w;
        let b = refl101(y as i64 - j as i64, hh) * w;
        pairs.push((a, b, kernel[n2 + j]));
    }
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
                for &(pa, pb, kc) in &pairs {
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
                for &(a, b, c) in &pairs {
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
            for &(a, b, c) in &pairs {
                acc = (src[a + x] + src[b + x]).mul_add(c, acc);
            }
            out[x] = acc;
            x += 1;
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    for x in 0..w {
        let mut acc = src[c0 + x] * k0;
        for &(a, b, c) in &pairs {
            acc = (src[a + x] + src[b + x]).mul_add(c, acc);
        }
        out[x] = acc;
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

    /// Scale-space throughput, to size the CPU budget against cv2's 226 ms.
    /// `KORNIA_SIFT_BENCH=1`; skipped otherwise.
    #[test]
    fn bench_neon_scale_space() {
        if std::env::var("KORNIA_SIFT_BENCH").is_err() {
            eprintln!("KORNIA_SIFT_BENCH unset; skipping");
            return;
        }
        let cfg = super::super::params::SiftConfig::default();
        let sigmas = cfg.layer_sigmas();
        // Octave 0 of a doubled 752x480 input, the audit's configuration.
        let (w, h) = (1504usize, 960usize);
        let kernels: Vec<Vec<f32>> = (1..cfg.n_octave_layers + 3)
            .map(|i| gaussian_kernel_f32(gaussian_ksize(sigmas[i]), sigmas[i]))
            .collect();

        let mut a = vec![0.5f32; w * h];
        for (i, v) in a.iter_mut().enumerate() {
            *v = ((i * 37) % 251) as f32;
        }
        let mut tmp = vec![0.0f32; w * h];
        let mut b = vec![0.0f32; w * h];

        let mut ts = Vec::new();
        for rep in 0..6 {
            let t = std::time::Instant::now();
            for k in &kernels {
                blur_h_f32(&a, &mut tmp, w, h, k);
                blur_v_f32(&tmp, &mut b, w, h, k, None, None);
            }
            if rep > 0 {
                ts.push(t.elapsed().as_secs_f64() * 1e3);
            }
        }
        ts.sort_by(f64::total_cmp);
        let oct0 = ts[2];
        // Later octaves are a geometric 1/4 series -> ~4/3 of octave 0.
        eprintln!(
            "  NEON scale-space octave0 {oct0:.1} ms  =>  all octaves ~{:.1} ms (1 thread)",
            oct0 * 4.0 / 3.0
        );
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
