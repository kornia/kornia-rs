//! NEON (aarch64) kernels for kornia-xfeat.
//!
//! Strategy: vectorize the inner channel reduction with 4-way parallel f32x4
//! accumulators (saturating both A78AE FMA pipes), then horizontal-sum at the
//! end of each (output-pixel, output-channel) work item. Bias, optional
//! residual, and the ReLU/Sigmoid/Identity activation fuse into the epilogue.
//!
//! This is the v1 NEON layer — correct, parity-tested, and several × the
//! scalar baseline. A more aggressive v2 (block over c_out tiles, re-pack
//! weights into a SIMD-friendly layout, broadcast input + vector weights to
//! avoid the horizontal reduction) is a follow-up tuned against measured
//! profiler traces — see the project's NEON skill for the rules.

#![allow(unused_imports)]

use std::arch::aarch64::*;

use super::{scalar, Activation, Conv1x1Args, Conv3x3Args};

#[inline]
fn apply_act(x: f32, act: Activation) -> f32 {
    match act {
        Activation::Relu => x.max(0.0),
        Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        Activation::Identity => x,
    }
}

/// NEON 1×1 fused conv with parallel f32x4 accumulators.
///
/// Falls back to scalar if `c_in` is not a multiple of 16 (covers the
/// awkward c_in=1 first-layer-of-skip1 case; everything else in the XFeat
/// graph has c_in ∈ {4, 8, 24, 64, 128} — all multiples of 4, and ≥16 for the
/// hot layers).
pub fn conv1x1_nhwc(args: &Conv1x1Args<'_>, output: &mut [f32]) {
    let &Conv1x1Args {
        input,
        weights,
        bias,
        h,
        w,
        c_in,
        c_out,
        activation,
    } = args;

    if c_in % 16 != 0 {
        scalar::conv1x1_nhwc(args, output);
        return;
    }

    debug_assert_eq!(input.len(), h * w * c_in);
    debug_assert_eq!(output.len(), h * w * c_out);
    debug_assert_eq!(weights.len(), c_out * c_in);
    debug_assert_eq!(bias.len(), c_out);

    // SAFETY: aarch64 always has NEON; pointers + lengths are checked by the
    // debug_assert_eq calls above. No aliasing: input/weights/bias are read-only
    // borrows; output is a unique mutable borrow.
    unsafe {
        let n_pixels = h * w;
        for px in 0..n_pixels {
            let in_ptr = input.as_ptr().add(px * c_in);
            let out_ptr = output.as_mut_ptr().add(px * c_out);
            for (co, &b) in bias.iter().enumerate().take(c_out) {
                let w_ptr = weights.as_ptr().add(co * c_in);
                let acc = dot_f32_16ply(in_ptr, w_ptr, c_in);
                let v = acc + b;
                *out_ptr.add(co) = apply_act(v, activation);
            }
        }
    }
}

/// Dot product of two contiguous f32 vectors of length `n` (must be a multiple
/// of 16), 4-way parallel accumulators saturating the two NEON FMA pipes.
#[inline]
unsafe fn dot_f32_16ply(a: *const f32, b: *const f32, n: usize) -> f32 {
    debug_assert!(n % 16 == 0);
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);
    let mut i = 0;
    while i < n {
        let av0 = vld1q_f32(a.add(i));
        let av1 = vld1q_f32(a.add(i + 4));
        let av2 = vld1q_f32(a.add(i + 8));
        let av3 = vld1q_f32(a.add(i + 12));
        let bv0 = vld1q_f32(b.add(i));
        let bv1 = vld1q_f32(b.add(i + 4));
        let bv2 = vld1q_f32(b.add(i + 8));
        let bv3 = vld1q_f32(b.add(i + 12));
        acc0 = vfmaq_f32(acc0, av0, bv0);
        acc1 = vfmaq_f32(acc1, av1, bv1);
        acc2 = vfmaq_f32(acc2, av2, bv2);
        acc3 = vfmaq_f32(acc3, av3, bv3);
        i += 16;
    }
    let s = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
    vaddvq_f32(s)
}

/// NEON 3×3 stride-1 NHWC fused conv. Same per-output FMA chain structure as
/// 1×1, but with the spatial-window walk on the outside; falls back to
/// scalar if `c_in % 16 != 0` (only hits on the rare layers where it would
/// not pay anyway).
pub fn conv3x3_relu_nhwc(args: &Conv3x3Args<'_>, output: &mut [f32]) {
    conv3x3_generic(args, output, 1);
}

/// NEON 3×3 stride-2 conv.
pub fn conv3x3_s2_relu_nhwc(args: &Conv3x3Args<'_>, output: &mut [f32]) {
    conv3x3_generic(args, output, 2);
}

fn conv3x3_generic(args: &Conv3x3Args<'_>, output: &mut [f32], stride: usize) {
    let &Conv3x3Args {
        input,
        residual,
        weights,
        bias,
        h_in,
        w_in,
        c_in,
        c_out,
        activation,
    } = args;

    if c_in % 16 != 0 {
        // Mirror the scalar dispatcher's `s1/s2` split.
        if stride == 1 {
            scalar::conv3x3_relu_nhwc(args, output);
        } else {
            scalar::conv3x3_s2_relu_nhwc(args, output);
        }
        return;
    }

    let h_out = h_in / stride;
    let w_out = w_in / stride;

    debug_assert_eq!(input.len(), h_in * w_in * c_in);
    debug_assert_eq!(output.len(), h_out * w_out * c_out);
    debug_assert_eq!(weights.len(), c_out * 9 * c_in);
    debug_assert_eq!(bias.len(), c_out);
    if let Some(r) = residual {
        debug_assert_eq!(r.len(), output.len());
    }

    // SAFETY: aarch64 always has NEON; all lengths are checked.
    unsafe {
        let in_ptr = input.as_ptr();
        let out_ptr = output.as_mut_ptr();
        let w_ptr = weights.as_ptr();
        let res_ptr = residual.map(|r| r.as_ptr());

        for oh in 0..h_out {
            for ow in 0..w_out {
                let ih_base = (oh * stride) as isize - 1;
                let iw_base = (ow * stride) as isize - 1;

                for (co, &b) in bias.iter().enumerate().take(c_out) {
                    let mut acc0 = vdupq_n_f32(0.0);
                    let mut acc1 = vdupq_n_f32(0.0);
                    let mut acc2 = vdupq_n_f32(0.0);
                    let mut acc3 = vdupq_n_f32(0.0);

                    for kh in 0..3usize {
                        let ih = ih_base + kh as isize;
                        if ih < 0 || ih >= h_in as isize {
                            continue;
                        }
                        let ih = ih as usize;
                        for kw in 0..3usize {
                            let iw = iw_base + kw as isize;
                            if iw < 0 || iw >= w_in as isize {
                                continue;
                            }
                            let iw = iw as usize;
                            let in_row_ptr = in_ptr.add((ih * w_in + iw) * c_in);
                            let w_row_ptr = w_ptr.add(((co * 3 + kh) * 3 + kw) * c_in);

                            let mut i = 0;
                            while i < c_in {
                                let av0 = vld1q_f32(in_row_ptr.add(i));
                                let av1 = vld1q_f32(in_row_ptr.add(i + 4));
                                let av2 = vld1q_f32(in_row_ptr.add(i + 8));
                                let av3 = vld1q_f32(in_row_ptr.add(i + 12));
                                let bv0 = vld1q_f32(w_row_ptr.add(i));
                                let bv1 = vld1q_f32(w_row_ptr.add(i + 4));
                                let bv2 = vld1q_f32(w_row_ptr.add(i + 8));
                                let bv3 = vld1q_f32(w_row_ptr.add(i + 12));
                                acc0 = vfmaq_f32(acc0, av0, bv0);
                                acc1 = vfmaq_f32(acc1, av1, bv1);
                                acc2 = vfmaq_f32(acc2, av2, bv2);
                                acc3 = vfmaq_f32(acc3, av3, bv3);
                                i += 16;
                            }
                        }
                    }

                    let s = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
                    let mut v = vaddvq_f32(s) + b;
                    let out_off = (oh * w_out + ow) * c_out + co;
                    if let Some(rp) = res_ptr {
                        v += *rp.add(out_off);
                    }
                    *out_ptr.add(out_off) = apply_act(v, activation);
                }
            }
        }
    }
}
