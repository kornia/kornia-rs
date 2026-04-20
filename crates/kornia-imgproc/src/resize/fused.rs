//! Fused preprocessing kernels: `resize + normalize + layout-convert` in one pass.
//!
//! Standard ML preprocessing materializes 3 intermediate buffers between
//! `u8 [H×W×C]` input and `f32 [C×H×W]` tensor output. Each op reads+writes
//! full-frame DRAM. These fused kernels collapse the chain to a single pass:
//! every byte of input is touched once, every byte of output is written once,
//! nothing in between lives outside L1.
//!
//! # First brick
//!
//! [`resize_normalize_to_tensor_u8_to_f32`] — 2× exact downscale (bilinear) +
//! per-channel `(x - mean) / std` + NCHW layout, in one NEON pass. This is
//! the cleanest and hottest fused case: exact-2× doesn't need a coefficient
//! LUT, and NCHW f32 is what ~90% of PyTorch consumers expect.
//!
//! # Numerical note
//!
//! Unlike the separate `resize → normalize` pipeline, which quantizes the
//! 2×2 average to u8 before normalizing, this kernel keeps the 10-bit sum
//! in f32 and folds the `/4` into `scale`. The output is therefore slightly
//! *more* accurate than the separate pipeline, not less.
//!
//! # Planned extensions (not yet implemented)
//!
//! - Arbitrary-ratio fused bilinear (the real hot case for ImageNet/CLIP
//!   pipelines, which resize from ~1-3MP JPEG-decoded input to 224/256).
//! - `f16` / `bf16` output variants for GPU training stacks. bf16 is nearly
//!   free (top-half extract of f32); f16 needs ARMv8.2 `vcvt_f16_f32`.

use rayon::prelude::*;

/// Per-channel ImageNet-style normalization parameters.
///
/// `scale[c] = 1.0 / (std[c] * 255.0)` and `bias[c] = -mean[c] / std[c]` so the
/// fused kernel can compute `out = src_u8 * scale + bias` with a single FMA
/// per lane instead of `(src/255 - mean) / std`.
#[derive(Copy, Clone, Debug)]
pub struct NormalizeParams<const C: usize> {
    /// Per-channel multiplicative factor applied to the raw u8 value.
    pub scale: [f32; C],
    /// Per-channel additive bias applied after `scale`.
    pub bias: [f32; C],
}

impl<const C: usize> NormalizeParams<C> {
    /// Build from per-channel mean+std in [0, 1] range (the PyTorch convention).
    /// `u8 x` → `(x/255 - mean) / std` → `x * (1/(std*255)) + (-mean/std)`.
    pub fn from_mean_std(mean: [f32; C], std: [f32; C]) -> Self {
        let mut scale = [0f32; C];
        let mut bias = [0f32; C];
        for c in 0..C {
            scale[c] = 1.0 / (std[c] * 255.0);
            bias[c] = -mean[c] / std[c];
        }
        Self { scale, bias }
    }
}

/// Fused 2× exact-downscale bilinear + per-channel normalize + `HWC→CHW` layout
/// conversion for RGB u8 → f32.
///
/// # Arguments
///
/// * `src`    — `[src_h × src_w × 3]` row-major u8 input
/// * `src_w`  — input width in pixels (must equal `2 * dst_w`)
/// * `src_h`  — input height in pixels (must equal `2 * dst_h`)
/// * `dst`    — `[3 × dst_h × dst_w]` f32 output (NCHW, channels-first)
/// * `dst_w`  — output width
/// * `dst_h`  — output height
/// * `params` — per-channel pre-combined scale/bias from [`NormalizeParams`]
pub fn resize_normalize_to_tensor_u8_to_f32(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    dst: &mut [f32],
    dst_w: usize,
    dst_h: usize,
    params: &NormalizeParams<3>,
) {
    debug_assert_eq!(src_w, 2 * dst_w);
    debug_assert_eq!(src_h, 2 * dst_h);
    debug_assert_eq!(src.len(), src_h * src_w * 3);
    debug_assert_eq!(dst.len(), 3 * dst_h * dst_w);

    let src_stride = src_w * 3;
    let plane_size = dst_h * dst_w;

    // NCHW output: three contiguous planes. Split the mutable borrow once so
    // each task writes to all 3 planes independently.
    let (r_plane, rest) = dst.split_at_mut(plane_size);
    let (g_plane, b_plane) = rest.split_at_mut(plane_size);

    // Group 8 dst rows per rayon task. Matches `pyrdown_2x_rgb_u8`'s shape —
    // one-row-per-task costs ~5 μs of spawn overhead × 540 rows = ~2.7 ms of
    // unnecessary dispatch at 1080p→540p, which dominates the kernel itself.
    const ROWS_PER_TASK: usize = 8;
    let chunk = dst_w * ROWS_PER_TASK;

    r_plane
        .par_chunks_mut(chunk)
        .zip(g_plane.par_chunks_mut(chunk))
        .zip(b_plane.par_chunks_mut(chunk))
        .enumerate()
        .for_each(|(ti, ((r_chunk, g_chunk), b_chunk))| {
            let y_start = ti * ROWS_PER_TASK;
            let nrows = r_chunk.len() / dst_w;
            for dy in 0..nrows {
                let y = y_start + dy;
                let r0 = &src[(2 * y) * src_stride..(2 * y + 1) * src_stride];
                let r1 = &src[(2 * y + 1) * src_stride..(2 * y + 2) * src_stride];
                let r_row = &mut r_chunk[dy * dst_w..(dy + 1) * dst_w];
                let g_row = &mut g_chunk[dy * dst_w..(dy + 1) * dst_w];
                let b_row = &mut b_chunk[dy * dst_w..(dy + 1) * dst_w];
                fused_row_rgb_u8_to_nchw_f32(r0, r1, r_row, g_row, b_row, dst_w, params);
            }
        });
}

/// Row-level fused kernel: one 2× downscale + normalize + HWC→CHW split.
#[inline(always)]
fn fused_row_rgb_u8_to_nchw_f32(
    r0: &[u8],
    r1: &[u8],
    r_out: &mut [f32],
    g_out: &mut [f32],
    b_out: &mut [f32],
    dst_w: usize,
    params: &NormalizeParams<3>,
) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        fused_row_neon(r0, r1, r_out, g_out, b_out, dst_w, params);
    }
    #[cfg(not(target_arch = "aarch64"))]
    fused_row_scalar(r0, r1, r_out, g_out, b_out, dst_w, params);
}

/// Scalar reference: f32-precise (no u8 quantization between avg and normalize).
#[inline]
#[allow(dead_code)]
fn fused_row_scalar(
    r0: &[u8],
    r1: &[u8],
    r_out: &mut [f32],
    g_out: &mut [f32],
    b_out: &mut [f32],
    dst_w: usize,
    params: &NormalizeParams<3>,
) {
    let sr = params.scale[0] * 0.25;
    let sg = params.scale[1] * 0.25;
    let sb = params.scale[2] * 0.25;
    let br = params.bias[0];
    let bg = params.bias[1];
    let bb = params.bias[2];
    for x in 0..dst_w {
        let base0 = 2 * x * 3;
        let base1 = base0 + 3;
        let sum_r =
            r0[base0] as u32 + r0[base1] as u32 + r1[base0] as u32 + r1[base1] as u32;
        let sum_g = r0[base0 + 1] as u32
            + r0[base1 + 1] as u32
            + r1[base0 + 1] as u32
            + r1[base1 + 1] as u32;
        let sum_b = r0[base0 + 2] as u32
            + r0[base1 + 2] as u32
            + r1[base0 + 2] as u32
            + r1[base1 + 2] as u32;
        r_out[x] = (sum_r as f32) * sr + br;
        g_out[x] = (sum_g as f32) * sg + bg;
        b_out[x] = (sum_b as f32) * sb + bb;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn fused_row_neon(
    r0: &[u8],
    r1: &[u8],
    r_out: &mut [f32],
    g_out: &mut [f32],
    b_out: &mut [f32],
    dst_w: usize,
    params: &NormalizeParams<3>,
) {
    use std::arch::aarch64::*;

    // Fold the 2×2 average's /4 into scale so the hot loop is pure FMA —
    // no integer rounding, no u8 requantization.
    let sr = vdupq_n_f32(params.scale[0] * 0.25);
    let sg = vdupq_n_f32(params.scale[1] * 0.25);
    let sb = vdupq_n_f32(params.scale[2] * 0.25);
    let br = vdupq_n_f32(params.bias[0]);
    let bg = vdupq_n_f32(params.bias[1]);
    let bb = vdupq_n_f32(params.bias[2]);

    // Per-channel 16-output-pixel emitter. Takes the 4 deinterleaved u8x16
    // channel lanes (two adjacent src chunks per row, two rows), computes
    // the 2×2 sum in u16, then widens→cvts→FMAs into 4 f32x4 stores.
    //
    // Written as a macro (not a closure) because unsafe + captured `_v`
    // vectors inside `#[target_feature]` contexts are easier to reason about
    // when fully inlined, and Rust closures don't inherit target_feature.
    macro_rules! emit_channel {
        ($a:expr, $b:expr, $c:expr, $d:expr, $scale:expr, $bias:expr, $out:expr) => {{
            let lo = vaddq_u16(vpaddlq_u8($a), vpaddlq_u8($c)); // out px 0..7
            let hi = vaddq_u16(vpaddlq_u8($b), vpaddlq_u8($d)); // out px 8..15
            let f0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo)));
            let f1 = vcvtq_f32_u32(vmovl_high_u16(lo));
            let f2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi)));
            let f3 = vcvtq_f32_u32(vmovl_high_u16(hi));
            vst1q_f32($out, vfmaq_f32($bias, f0, $scale));
            vst1q_f32($out.add(4), vfmaq_f32($bias, f1, $scale));
            vst1q_f32($out.add(8), vfmaq_f32($bias, f2, $scale));
            vst1q_f32($out.add(12), vfmaq_f32($bias, f3, $scale));
        }};
    }

    let bulk = dst_w & !15;
    let mut x = 0usize;
    while x < bulk {
        // 16 dst px → 32 src px per row → 2 × vld3q_u8 per row.
        let src0 = r0.as_ptr().add(x * 2 * 3);
        let src1 = r1.as_ptr().add(x * 2 * 3);
        let s0 = vld3q_u8(src0);
        let s1 = vld3q_u8(src0.add(48));
        let s2 = vld3q_u8(src1);
        let s3 = vld3q_u8(src1.add(48));

        emit_channel!(s0.0, s1.0, s2.0, s3.0, sr, br, r_out.as_mut_ptr().add(x));
        emit_channel!(s0.1, s1.1, s2.1, s3.1, sg, bg, g_out.as_mut_ptr().add(x));
        emit_channel!(s0.2, s1.2, s2.2, s3.2, sb, bb, b_out.as_mut_ptr().add(x));

        x += 16;
    }

    // Scalar tail for the last (dst_w % 16) pixels.
    let sr_s = params.scale[0] * 0.25;
    let sg_s = params.scale[1] * 0.25;
    let sb_s = params.scale[2] * 0.25;
    let br_s = params.bias[0];
    let bg_s = params.bias[1];
    let bb_s = params.bias[2];
    while x < dst_w {
        let base0 = 2 * x * 3;
        let base1 = base0 + 3;
        let sum_r = *r0.get_unchecked(base0) as u32
            + *r0.get_unchecked(base1) as u32
            + *r1.get_unchecked(base0) as u32
            + *r1.get_unchecked(base1) as u32;
        let sum_g = *r0.get_unchecked(base0 + 1) as u32
            + *r0.get_unchecked(base1 + 1) as u32
            + *r1.get_unchecked(base0 + 1) as u32
            + *r1.get_unchecked(base1 + 1) as u32;
        let sum_b = *r0.get_unchecked(base0 + 2) as u32
            + *r0.get_unchecked(base1 + 2) as u32
            + *r1.get_unchecked(base0 + 2) as u32
            + *r1.get_unchecked(base1 + 2) as u32;
        *r_out.get_unchecked_mut(x) = (sum_r as f32) * sr_s + br_s;
        *g_out.get_unchecked_mut(x) = (sum_g as f32) * sg_s + bg_s;
        *b_out.get_unchecked_mut(x) = (sum_b as f32) * sb_s + bb_s;
        x += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compare the fused kernel's output against an f64 reference on a
    /// synthetic image with a non-trivial remainder (dst_w not divisible
    /// by 16, so both the NEON bulk and the scalar tail are exercised).
    #[test]
    fn fused_2x_normalize_matches_f64_reference() {
        let dst_w = 37;
        let dst_h = 5;
        let src_w = 2 * dst_w;
        let src_h = 2 * dst_h;
        let src: Vec<u8> = (0..src_h * src_w * 3)
            .map(|i| ((i * 7 + 3) % 256) as u8)
            .collect();
        let mut dst = vec![0f32; 3 * dst_h * dst_w];

        let mean = [0.485, 0.456, 0.406];
        let std = [0.229, 0.224, 0.225];
        let params = NormalizeParams::<3>::from_mean_std(mean, std);

        resize_normalize_to_tensor_u8_to_f32(
            &src, src_w, src_h, &mut dst, dst_w, dst_h, &params,
        );

        // f64 reference: (avg(2x2)/255 - mean) / std, CHW layout.
        let plane = dst_h * dst_w;
        for y in 0..dst_h {
            for x in 0..dst_w {
                let b0 = (2 * y) * src_w * 3 + 2 * x * 3;
                let b1 = b0 + 3;
                let b2 = (2 * y + 1) * src_w * 3 + 2 * x * 3;
                let b3 = b2 + 3;
                for ch in 0..3 {
                    let sum = src[b0 + ch] as f64
                        + src[b1 + ch] as f64
                        + src[b2 + ch] as f64
                        + src[b3 + ch] as f64;
                    let expect = ((sum / 4.0) / 255.0 - mean[ch] as f64) / std[ch] as f64;
                    let got = dst[ch * plane + y * dst_w + x] as f64;
                    assert!(
                        (got - expect).abs() < 1e-4,
                        "ch={ch} y={y} x={x} got={got} expect={expect}"
                    );
                }
            }
        }
    }

    /// Corner test: a completely zero src image produces `-mean/std` on
    /// every output pixel (bias only).
    #[test]
    fn fused_2x_normalize_zero_input() {
        let dst_w = 16;
        let dst_h = 2;
        let src = vec![0u8; 2 * dst_w * 2 * dst_h * 3];
        let mut dst = vec![0f32; 3 * dst_h * dst_w];
        let mean = [0.5, 0.25, 0.75];
        let std = [0.5, 0.25, 0.75];
        let params = NormalizeParams::<3>::from_mean_std(mean, std);
        resize_normalize_to_tensor_u8_to_f32(
            &src,
            2 * dst_w,
            2 * dst_h,
            &mut dst,
            dst_w,
            dst_h,
            &params,
        );
        let plane = dst_h * dst_w;
        for ch in 0..3 {
            let expect = -mean[ch] / std[ch];
            for i in 0..plane {
                let got = dst[ch * plane + i];
                assert!(
                    (got - expect).abs() < 1e-6,
                    "ch={ch} i={i} got={got} expect={expect}"
                );
            }
        }
    }
}
