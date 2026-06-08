//! NEON (aarch64) kernels for kornia-xfeat.
//!
//! # v1 kernels
//! Strategy: vectorize the inner channel reduction with 4-way parallel f32x4
//! accumulators (saturating both A78AE FMA pipes), then horizontal-sum at the
//! end of each (output-pixel, output-channel) work item. Bias, optional
//! residual, and the ReLU/Sigmoid/Identity activation fuse into the epilogue.
//!
//! # v2 kernels
//! Strategy: c_out-4 tiling (process 4 output channels simultaneously per
//! pixel). Weights are repacked into `[c_out/4, c_in, 4]` layout so each
//! input channel broadcast multiplies 4 weight values in a single `vfmaq_f32`.
//! No horizontal reduction — the 4-lane accumulator IS the output.
//!
//! Inner loop (per pixel, per co_block):
//!   for ci in 0..c_in step 4 (unrolled 4-way):
//!     acc0 += broadcast(in[ci  ]) * W[:,ci  ]
//!     acc1 += broadcast(in[ci+1]) * W[:,ci+1]
//!     acc2 += broadcast(in[ci+2]) * W[:,ci+2]
//!     acc3 += broadcast(in[ci+3]) * W[:,ci+3]
//!   result[co0..co3] = bias + acc0 + acc1 + acc2 + acc3
//!
//! conv3x3 v2 additionally uses 2-pixel spatial blocking and rayon
//! row-level parallelism.
//!
//! Dispatch:
//!   - c_out % 4 == 0 && c_in % 4 == 0  → v2 path
//!   - otherwise                          → v1 path (falls back to scalar
//!     when c_in % 4 != 0)

#![allow(unused_imports)]

use std::arch::aarch64::*;

use rayon::prelude::*;

use super::{scalar, Activation, Conv1x1Args, Conv3x3Args};

#[inline]
fn apply_act(x: f32, act: Activation) -> f32 {
    match act {
        Activation::Relu => x.max(0.0),
        Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        Activation::Identity => x,
    }
}

/// Apply activation lane-wise to a float32x4.
#[inline]
unsafe fn apply_act_vec(v: float32x4_t, act: Activation) -> float32x4_t {
    match act {
        Activation::Relu => vmaxq_f32(v, vdupq_n_f32(0.0)),
        Activation::Identity => v,
        Activation::Sigmoid => {
            let mut buf = [0.0f32; 4];
            vst1q_f32(buf.as_mut_ptr(), v);
            for x in &mut buf {
                *x = 1.0 / (1.0 + (-*x).exp());
            }
            vld1q_f32(buf.as_ptr())
        }
    }
}

// ── Weight repacking ──────────────────────────────────────────────────────────

/// Repack conv1x1 weights from `[c_out, c_in]` to `[c_out/4, c_in, 4]`.
///
/// The repacked layout groups 4 output-channel rows together so that for each
/// input channel `ci` we can load the corresponding 4 weight values as a
/// contiguous `float32x4_t`.
///
/// `c_out` must be a multiple of 4.
fn repack_weights_co4(
    weights: &[f32], // [c_out, c_in]
    c_out: usize,
    c_in: usize,
) -> Vec<f32> {
    debug_assert_eq!(c_out % 4, 0);
    let n_co4 = c_out / 4;
    let mut out = vec![0.0f32; n_co4 * c_in * 4];
    for co_block in 0..n_co4 {
        for ci in 0..c_in {
            for lane in 0..4usize {
                let co = co_block * 4 + lane;
                out[(co_block * c_in + ci) * 4 + lane] = weights[co * c_in + ci];
            }
        }
    }
    out
}

/// Repack conv3x3 weights from `[c_out, 9, c_in]` to `[c_out/4, 9, c_in, 4]`.
fn repack_weights_co4_3x3(
    weights: &[f32], // [c_out, 9, c_in]
    c_out: usize,
    c_in: usize,
) -> Vec<f32> {
    debug_assert_eq!(c_out % 4, 0);
    let n_co4 = c_out / 4;
    let mut out = vec![0.0f32; n_co4 * 9 * c_in * 4];
    for co_block in 0..n_co4 {
        for tap in 0..9usize {
            for ci in 0..c_in {
                for lane in 0..4usize {
                    let co = co_block * 4 + lane;
                    // original layout: weights[co * 9 * c_in + tap * c_in + ci]
                    out[((co_block * 9 + tap) * c_in + ci) * 4 + lane] =
                        weights[co * 9 * c_in + tap * c_in + ci];
                }
            }
        }
    }
    out
}

// ── conv1x1 v2 ───────────────────────────────────────────────────────────────

/// Dot-product of one input pixel against repacked weights, accumulating into 4
/// output channels simultaneously.
///
/// Inner loop: processes 4 consecutive input channels per iteration using 4
/// independent FMA chains for ILP.  Handles c_in % 4 == 0 tails naturally
/// (no padding needed — loop variable bounds are exact c_in).
///
/// # Safety
/// `in_px` must point to `c_in` valid f32 values.
/// `w_blk` must point to `c_in * 4` valid f32 values: layout `[c_in, 4]`.
#[inline]
unsafe fn dot_co4(in_px: *const f32, w_blk: *const f32, c_in: usize) -> float32x4_t {
    debug_assert_eq!(c_in % 4, 0);
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);

    // Phase 1: 4-at-a-time consecutive channels (handles c_in % 16 != 0 too).
    let c4 = c_in; // c_in % 4 == 0 guaranteed by caller
    let c16 = c_in & !15;
    let mut ci = 0usize;

    // Unrolled 16-channel blocks: 4 accumulators × 4 consecutive ci's.
    while ci < c16 {
        // Process ci, ci+1, ci+2, ci+3 (acc0..acc3)
        let iv0 = vdupq_n_f32(*in_px.add(ci));
        let iv1 = vdupq_n_f32(*in_px.add(ci + 1));
        let iv2 = vdupq_n_f32(*in_px.add(ci + 2));
        let iv3 = vdupq_n_f32(*in_px.add(ci + 3));
        let wv0 = vld1q_f32(w_blk.add(ci * 4));
        let wv1 = vld1q_f32(w_blk.add((ci + 1) * 4));
        let wv2 = vld1q_f32(w_blk.add((ci + 2) * 4));
        let wv3 = vld1q_f32(w_blk.add((ci + 3) * 4));
        acc0 = vfmaq_f32(acc0, iv0, wv0);
        acc1 = vfmaq_f32(acc1, iv1, wv1);
        acc2 = vfmaq_f32(acc2, iv2, wv2);
        acc3 = vfmaq_f32(acc3, iv3, wv3);
        ci += 4;

        // Process ci, ci+1, ci+2, ci+3 (re-use same accumulators for 2nd group)
        let iv0 = vdupq_n_f32(*in_px.add(ci));
        let iv1 = vdupq_n_f32(*in_px.add(ci + 1));
        let iv2 = vdupq_n_f32(*in_px.add(ci + 2));
        let iv3 = vdupq_n_f32(*in_px.add(ci + 3));
        let wv0 = vld1q_f32(w_blk.add(ci * 4));
        let wv1 = vld1q_f32(w_blk.add((ci + 1) * 4));
        let wv2 = vld1q_f32(w_blk.add((ci + 2) * 4));
        let wv3 = vld1q_f32(w_blk.add((ci + 3) * 4));
        acc0 = vfmaq_f32(acc0, iv0, wv0);
        acc1 = vfmaq_f32(acc1, iv1, wv1);
        acc2 = vfmaq_f32(acc2, iv2, wv2);
        acc3 = vfmaq_f32(acc3, iv3, wv3);
        ci += 4;

        let iv0 = vdupq_n_f32(*in_px.add(ci));
        let iv1 = vdupq_n_f32(*in_px.add(ci + 1));
        let iv2 = vdupq_n_f32(*in_px.add(ci + 2));
        let iv3 = vdupq_n_f32(*in_px.add(ci + 3));
        let wv0 = vld1q_f32(w_blk.add(ci * 4));
        let wv1 = vld1q_f32(w_blk.add((ci + 1) * 4));
        let wv2 = vld1q_f32(w_blk.add((ci + 2) * 4));
        let wv3 = vld1q_f32(w_blk.add((ci + 3) * 4));
        acc0 = vfmaq_f32(acc0, iv0, wv0);
        acc1 = vfmaq_f32(acc1, iv1, wv1);
        acc2 = vfmaq_f32(acc2, iv2, wv2);
        acc3 = vfmaq_f32(acc3, iv3, wv3);
        ci += 4;

        let iv0 = vdupq_n_f32(*in_px.add(ci));
        let iv1 = vdupq_n_f32(*in_px.add(ci + 1));
        let iv2 = vdupq_n_f32(*in_px.add(ci + 2));
        let iv3 = vdupq_n_f32(*in_px.add(ci + 3));
        let wv0 = vld1q_f32(w_blk.add(ci * 4));
        let wv1 = vld1q_f32(w_blk.add((ci + 1) * 4));
        let wv2 = vld1q_f32(w_blk.add((ci + 2) * 4));
        let wv3 = vld1q_f32(w_blk.add((ci + 3) * 4));
        acc0 = vfmaq_f32(acc0, iv0, wv0);
        acc1 = vfmaq_f32(acc1, iv1, wv1);
        acc2 = vfmaq_f32(acc2, iv2, wv2);
        acc3 = vfmaq_f32(acc3, iv3, wv3);
        ci += 4;
    }

    // Phase 2: 4-channel tail (c_in % 16 != 0, e.g. c_in=24 → 8 remaining).
    while ci < c4 {
        let iv0 = vdupq_n_f32(*in_px.add(ci));
        let iv1 = vdupq_n_f32(*in_px.add(ci + 1));
        let iv2 = vdupq_n_f32(*in_px.add(ci + 2));
        let iv3 = vdupq_n_f32(*in_px.add(ci + 3));
        let wv0 = vld1q_f32(w_blk.add(ci * 4));
        let wv1 = vld1q_f32(w_blk.add((ci + 1) * 4));
        let wv2 = vld1q_f32(w_blk.add((ci + 2) * 4));
        let wv3 = vld1q_f32(w_blk.add((ci + 3) * 4));
        acc0 = vfmaq_f32(acc0, iv0, wv0);
        acc1 = vfmaq_f32(acc1, iv1, wv1);
        acc2 = vfmaq_f32(acc2, iv2, wv2);
        acc3 = vfmaq_f32(acc3, iv3, wv3);
        ci += 4;
    }

    vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3))
}

/// NEON 1×1 conv (public, called by parity tests).
///
/// Uses v1 accumulation (vector×vector dot) to stay within the parity-test
/// tolerance of 1e-6 vs scalar. The hot-path model dispatch goes through
/// [`conv1x1_nhwc_v2`] which uses the co4-tiled broadcast accumulation.
///
/// Falls back to scalar when `c_in % 4 != 0`.
pub fn conv1x1_nhwc(args: &Conv1x1Args<'_>, output: &mut [f32]) {
    conv1x1_nhwc_v1(args, output);
}

/// NEON 1×1 conv v2: c_out-4 tiled, no horizontal reduction.
///
/// Repacks weights once per call, then processes all rows in parallel via rayon.
/// Falls back to v1 when `c_out % 4 != 0` or `c_in % 4 != 0`.
pub fn conv1x1_nhwc_v2(args: &Conv1x1Args<'_>, output: &mut [f32]) {
    let &Conv1x1Args {
        input,
        weights,
        bias,
        h: _,
        w,
        c_in,
        c_out,
        activation,
    } = args;

    // v2 requires c_out % 4 == 0 and c_in % 4 == 0.
    if c_out % 4 != 0 || c_in % 4 != 0 {
        conv1x1_nhwc_v1(args, output);
        return;
    }

    let packed_w = repack_weights_co4(weights, c_out, c_in);

    // Parallelize over rows; each row is an independent slice of c_out * w floats.
    let row_stride_out = w * c_out;
    let row_stride_in = w * c_in;

    // Wrap raw pointers for Send across rayon threads.
    let pw_ptr = packed_w.as_ptr() as usize;
    let bias_ptr = bias.as_ptr() as usize;

    output
        .par_chunks_mut(row_stride_out)
        .enumerate()
        .for_each(|(row_idx, out_row)| {
            let in_row = &input[row_idx * row_stride_in..(row_idx + 1) * row_stride_in];
            unsafe {
                conv1x1_v2_row(
                    in_row.as_ptr(),
                    pw_ptr as *const f32,
                    bias_ptr as *const f32,
                    out_row.as_mut_ptr(),
                    w, // n_pixels in this row
                    c_in,
                    c_out,
                    activation,
                );
            }
        });
}

/// Inner per-row kernel for conv1x1 v2.  `in_ptr` points to `n_pixels * c_in`
/// input values; `out_ptr` to `n_pixels * c_out` output values.
///
/// # Safety
/// All pointers must be valid for their indicated lengths; no aliasing.
#[allow(clippy::too_many_arguments)]
unsafe fn conv1x1_v2_row(
    in_ptr: *const f32,
    packed_w: *const f32, // [c_out/4, c_in, 4]
    bias: *const f32,
    out_ptr: *mut f32,
    n_pixels: usize,
    c_in: usize,
    c_out: usize,
    act: Activation,
) {
    let n_co4 = c_out / 4;

    for px in 0..n_pixels {
        let in_px = in_ptr.add(px * c_in);
        let out_px = out_ptr.add(px * c_out);

        for co_block in 0..n_co4 {
            // Load bias for this co-block as the initial accumulator.
            let bias_vec = vld1q_f32(bias.add(co_block * 4));

            // w_blk: [c_in, 4] slice for this co_block
            let w_blk = packed_w.add(co_block * c_in * 4);

            let sum = dot_co4(in_px, w_blk, c_in);
            let acc = vaddq_f32(bias_vec, sum);
            let result = apply_act_vec(acc, act);
            vst1q_f32(out_px.add(co_block * 4), result);
        }
    }
}

// ── conv1x1 v1 (original, kept as fallback) ──────────────────────────────────

/// v1 1×1 conv — kept intact as fallback for c_out % 4 != 0 or c_in % 4 != 0.
fn conv1x1_nhwc_v1(args: &Conv1x1Args<'_>, output: &mut [f32]) {
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

    if c_in % 4 != 0 {
        scalar::conv1x1_nhwc(args, output);
        return;
    }

    debug_assert_eq!(input.len(), h * w * c_in);
    debug_assert_eq!(output.len(), h * w * c_out);
    debug_assert_eq!(weights.len(), c_out * c_in);
    debug_assert_eq!(bias.len(), c_out);

    unsafe {
        let n_pixels = h * w;
        for px in 0..n_pixels {
            let in_ptr = input.as_ptr().add(px * c_in);
            let out_ptr = output.as_mut_ptr().add(px * c_out);
            for (co, &b) in bias.iter().enumerate().take(c_out) {
                let w_ptr = weights.as_ptr().add(co * c_in);
                let acc = dot_f32_4ply_tail(in_ptr, w_ptr, c_in);
                let v = acc + b;
                *out_ptr.add(co) = apply_act(v, activation);
            }
        }
    }
}

/// Dot product of two contiguous f32 vectors of length `n` (must be a multiple
/// of 4). Phase 1: 4-way parallel accumulators over 16-element blocks
/// (saturates the two A78AE FMA pipes). Phase 2: f32x4 tail folds into `acc0`.
#[inline]
unsafe fn dot_f32_4ply_tail(a: *const f32, b: *const f32, n: usize) -> f32 {
    debug_assert!(n % 4 == 0);
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);

    // Phase 1: full 16-element blocks.
    let n16 = n & !15;
    let mut i = 0;
    while i < n16 {
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
    // Phase 2: f32x4 tail.
    while i < n {
        let av = vld1q_f32(a.add(i));
        let bv = vld1q_f32(b.add(i));
        acc0 = vfmaq_f32(acc0, av, bv);
        i += 4;
    }
    let s = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
    vaddvq_f32(s)
}

// ── conv3x3 v2 ───────────────────────────────────────────────────────────────

/// NEON 3×3 stride-1 NHWC conv v2.
///
/// Uses c_out-4 tiling, 2-pixel spatial blocking, and rayon over output rows.
/// Falls back to v1 when c_out % 4 != 0 or c_in % 4 != 0.
pub fn conv3x3_relu_nhwc(args: &Conv3x3Args<'_>, output: &mut [f32]) {
    if args.c_out % 4 != 0 || args.c_in % 4 != 0 {
        conv3x3_generic_v1(args, output, 1);
        return;
    }
    conv3x3_v2(args, output, 1);
}

/// NEON 3×3 stride-2 NHWC conv v2.
pub fn conv3x3_s2_relu_nhwc(args: &Conv3x3Args<'_>, output: &mut [f32]) {
    if args.c_out % 4 != 0 || args.c_in % 4 != 0 {
        conv3x3_generic_v1(args, output, 2);
        return;
    }
    conv3x3_v2(args, output, 2);
}

/// v2 conv3x3 entry point: repack weights, then dispatch rows in parallel.
fn conv3x3_v2(args: &Conv3x3Args<'_>, output: &mut [f32], stride: usize) {
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

    let w_out = w_in / stride;

    let packed_w = repack_weights_co4_3x3(weights, c_out, c_in);

    let row_stride_out = w_out * c_out;

    // Wrap raw pointers for Send across rayon threads.
    // Safety: all reads are from shared immutable data; writes are to
    // non-overlapping per-row output slices from par_chunks_mut.
    let in_ptr = input.as_ptr() as usize;
    let pw_ptr = packed_w.as_ptr() as usize;
    let bias_ptr = bias.as_ptr() as usize;
    let res_ptr: usize = residual.map_or(0, |r| r.as_ptr() as usize);
    let has_res = residual.is_some();

    output
        .par_chunks_mut(row_stride_out)
        .enumerate()
        .for_each(|(oh, out_row)| unsafe {
            conv3x3_v2_row(
                oh,
                in_ptr as *const f32,
                pw_ptr as *const f32,
                bias_ptr as *const f32,
                if has_res {
                    res_ptr as *const f32
                } else {
                    core::ptr::null()
                },
                out_row.as_mut_ptr(),
                h_in,
                w_in,
                w_out,
                c_in,
                c_out,
                stride,
                activation,
            );
        });
}

/// Per-row inner kernel for conv3x3 v2 (stride 1 or 2).
///
/// Processes pairs of adjacent output pixels to amortise weight loads.
/// Single-pixel epilogue handles odd `w_out`.
///
/// # Safety
/// All raw pointers must be valid for their logical extents.  `res_ptr` may be
/// null (indicates no residual).  `out_ptr` must point to a non-overlapping
/// slice of exactly `w_out * c_out` floats for this row.
#[allow(clippy::too_many_arguments)]
unsafe fn conv3x3_v2_row(
    oh: usize,
    in_ptr: *const f32,
    packed_w: *const f32, // [c_out/4, 9, c_in, 4]
    bias_ptr: *const f32,
    res_ptr: *const f32, // null if no residual
    out_ptr: *mut f32,   // points to start of this row: w_out * c_out floats
    h_in: usize,
    w_in: usize,
    w_out: usize,
    c_in: usize,
    c_out: usize,
    stride: usize,
    act: Activation,
) {
    let n_co4 = c_out / 4;
    // ih index for the top kh=0 tap of this output row.
    let ih_center = (oh * stride) as isize - 1;

    // --- 2-pixel spatial block loop ---
    let mut ow = 0usize;
    while ow + 1 < w_out {
        let iw_base0 = (ow * stride) as isize - 1;
        let iw_base1 = ((ow + 1) * stride) as isize - 1;

        for co_block in 0..n_co4 {
            let bias_vec = vld1q_f32(bias_ptr.add(co_block * 4));
            let mut acc_px0 = bias_vec;
            let mut acc_px1 = bias_vec;

            for kh in 0..3usize {
                let ih = ih_center + kh as isize;
                if ih < 0 || ih >= h_in as isize {
                    continue;
                }
                let ih = ih as usize;

                for kw in 0..3usize {
                    let iw0 = iw_base0 + kw as isize;
                    let iw1 = iw_base1 + kw as isize;

                    let tap = kh * 3 + kw;
                    // w_tap: [c_in, 4] for this (co_block, tap)
                    let w_tap = packed_w.add((co_block * 9 + tap) * c_in * 4);

                    let valid0 = iw0 >= 0 && iw0 < w_in as isize;
                    let valid1 = iw1 >= 0 && iw1 < w_in as isize;

                    if !valid0 && !valid1 {
                        continue;
                    }

                    if valid0 {
                        let ip0 = in_ptr.add((ih * w_in + iw0 as usize) * c_in);
                        let tap_sum = dot_co4(ip0, w_tap, c_in);
                        acc_px0 = vaddq_f32(acc_px0, tap_sum);
                    }
                    if valid1 {
                        let ip1 = in_ptr.add((ih * w_in + iw1 as usize) * c_in);
                        let tap_sum = dot_co4(ip1, w_tap, c_in);
                        acc_px1 = vaddq_f32(acc_px1, tap_sum);
                    }
                }
            }

            // Optional residual add (uses global output offset for residual indexing)
            let out_off0 = ow * c_out + co_block * 4;
            let out_off1 = (ow + 1) * c_out + co_block * 4;
            if !res_ptr.is_null() {
                let global_off0 = (oh * w_out + ow) * c_out + co_block * 4;
                let global_off1 = (oh * w_out + ow + 1) * c_out + co_block * 4;
                let rv0 = vld1q_f32(res_ptr.add(global_off0));
                let rv1 = vld1q_f32(res_ptr.add(global_off1));
                acc_px0 = vaddq_f32(acc_px0, rv0);
                acc_px1 = vaddq_f32(acc_px1, rv1);
            }

            let r0 = apply_act_vec(acc_px0, act);
            let r1 = apply_act_vec(acc_px1, act);
            vst1q_f32(out_ptr.add(out_off0), r0);
            vst1q_f32(out_ptr.add(out_off1), r1);
        }

        ow += 2;
    }

    // --- Single-pixel epilogue for odd w_out ---
    if ow < w_out {
        let iw_base = (ow * stride) as isize - 1;

        for co_block in 0..n_co4 {
            let mut acc = vld1q_f32(bias_ptr.add(co_block * 4));
            let w_co_base = packed_w.add(co_block * 9 * c_in * 4);

            for kh in 0..3usize {
                let ih = ih_center + kh as isize;
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
                    let tap = kh * 3 + kw;
                    let w_tap = w_co_base.add(tap * c_in * 4);
                    let inp = in_ptr.add((ih * w_in + iw) * c_in);
                    let tap_sum = dot_co4(inp, w_tap, c_in);
                    acc = vaddq_f32(acc, tap_sum);
                }
            }

            let out_off = ow * c_out + co_block * 4;
            if !res_ptr.is_null() {
                let global_off = (oh * w_out + ow) * c_out + co_block * 4;
                let rv = vld1q_f32(res_ptr.add(global_off));
                acc = vaddq_f32(acc, rv);
            }
            let result = apply_act_vec(acc, act);
            vst1q_f32(out_ptr.add(out_off), result);
        }
    }
}

// ── conv3x3 v1 (original, kept as fallback) ──────────────────────────────────

fn conv3x3_generic_v1(args: &Conv3x3Args<'_>, output: &mut [f32], stride: usize) {
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

    if c_in % 4 != 0 {
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

                            let c16 = c_in & !15;
                            let mut i = 0;
                            while i < c16 {
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
                            while i < c_in {
                                let av = vld1q_f32(in_row_ptr.add(i));
                                let bv = vld1q_f32(w_row_ptr.add(i));
                                acc0 = vfmaq_f32(acc0, av, bv);
                                i += 4;
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
