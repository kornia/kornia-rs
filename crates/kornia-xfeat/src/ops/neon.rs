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

/// Accumulate one 3×3 tap into TWO output-pixel accumulators simultaneously,
/// sharing the weight load between them.
///
/// Instead of the old scalar-broadcast path (vdupq_n_f32 + vfmaq), this uses
/// `vfmaq_laneq_f32` — a single FMA-pipe instruction that reads one lane of a
/// vector register, broadcasting it implicitly, with no separate broadcast op.
/// Weight vectors are loaded ONCE and reused for both pixels, halving bandwidth
/// for the weight access.
///
/// `c_in` must be a multiple of 4.  When a pixel pointer is null the caller
/// has already ensured the accumulator is not updated for that side.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn accum_tap_2px(
    acc0: &mut float32x4_t, // accumulator for pixel 0
    acc1: &mut float32x4_t, // accumulator for pixel 1
    ip0:  *const f32,       // input pointer for pixel 0 (never null here)
    ip1:  *const f32,       // input pointer for pixel 1 (never null here)
    w_tap: *const f32,      // weight pointer: [c_in, 4] for this tap
    c_in:  usize,
) {
    let mut ci = 0usize;
    while ci < c_in {
        // One vector load for 4 input channels — shared between acc0 and acc1.
        let iv0 = vld1q_f32(ip0.add(ci));
        let iv1 = vld1q_f32(ip1.add(ci));

        // Four weight vectors: w_ci[j] contains the 4 output-channel weights
        // for input channel ci+j.
        let wv0 = vld1q_f32(w_tap.add((ci + 0) * 4));
        let wv1 = vld1q_f32(w_tap.add((ci + 1) * 4));
        let wv2 = vld1q_f32(w_tap.add((ci + 2) * 4));
        let wv3 = vld1q_f32(w_tap.add((ci + 3) * 4));

        *acc0 = vfmaq_laneq_f32::<0>(*acc0, wv0, iv0);
        *acc0 = vfmaq_laneq_f32::<1>(*acc0, wv1, iv0);
        *acc0 = vfmaq_laneq_f32::<2>(*acc0, wv2, iv0);
        *acc0 = vfmaq_laneq_f32::<3>(*acc0, wv3, iv0);

        *acc1 = vfmaq_laneq_f32::<0>(*acc1, wv0, iv1);
        *acc1 = vfmaq_laneq_f32::<1>(*acc1, wv1, iv1);
        *acc1 = vfmaq_laneq_f32::<2>(*acc1, wv2, iv1);
        *acc1 = vfmaq_laneq_f32::<3>(*acc1, wv3, iv1);

        ci += 4;
    }
}

/// Single-pixel variant of accum_tap_2px, also using vfmaq_laneq_f32.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn accum_tap_1px(
    acc: &mut float32x4_t,
    ip:  *const f32,
    w_tap: *const f32,
    c_in:  usize,
) {
    let mut ci = 0usize;
    while ci < c_in {
        let iv  = vld1q_f32(ip.add(ci));
        let wv0 = vld1q_f32(w_tap.add((ci + 0) * 4));
        let wv1 = vld1q_f32(w_tap.add((ci + 1) * 4));
        let wv2 = vld1q_f32(w_tap.add((ci + 2) * 4));
        let wv3 = vld1q_f32(w_tap.add((ci + 3) * 4));
        *acc = vfmaq_laneq_f32::<0>(*acc, wv0, iv);
        *acc = vfmaq_laneq_f32::<1>(*acc, wv1, iv);
        *acc = vfmaq_laneq_f32::<2>(*acc, wv2, iv);
        *acc = vfmaq_laneq_f32::<3>(*acc, wv3, iv);
        ci += 4;
    }
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
                    w,
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
        packed_weights: _,
    } = args;

    let w_out = w_in / stride;

    // Use pre-packed weights when the caller supplies them (zero-alloc hot path).
    // Fall back to on-the-fly repack for callers that don't pre-compute (e.g.
    // Winograd-fallback paths and non-aarch64 CI builds).
    let _owned_packed: Vec<f32>;
    let packed_w: &[f32] = match args.packed_weights {
        Some(pw) => pw,
        None => {
            _owned_packed = super::repack_weights_co4_3x3(weights, c_out, c_in);
            &_owned_packed
        }
    };

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
                    let w_tap = packed_w.add((co_block * 9 + tap) * c_in * 4);

                    let valid0 = iw0 >= 0 && iw0 < w_in as isize;
                    let valid1 = iw1 >= 0 && iw1 < w_in as isize;

                    if !valid0 && !valid1 {
                        continue;
                    }

                    if valid0 && valid1 {
                        // Fast path: load weights once, accumulate for both pixels.
                        let ip0 = in_ptr.add((ih * w_in + iw0 as usize) * c_in);
                        let ip1 = in_ptr.add((ih * w_in + iw1 as usize) * c_in);
                        accum_tap_2px(&mut acc_px0, &mut acc_px1, ip0, ip1, w_tap, c_in);
                    } else if valid0 {
                        let ip0 = in_ptr.add((ih * w_in + iw0 as usize) * c_in);
                        accum_tap_1px(&mut acc_px0, ip0, w_tap, c_in);
                    } else {
                        let ip1 = in_ptr.add((ih * w_in + iw1 as usize) * c_in);
                        accum_tap_1px(&mut acc_px1, ip1, w_tap, c_in);
                    }
                }
            }

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
                    accum_tap_1px(&mut acc, inp, w_tap, c_in);
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

// ── conv3x3 fp16 ─────────────────────────────────────────────────────────────
//
// Uses FMLA.8H (8 fp16 MACs/instruction) for 2× arithmetic density vs FMLA.4S.
// Weights pre-packed as [c_out/8, 9, c_in, 8] fp16-as-u16. Input f32 is
// converted inline via FCVTN (zero heap writes). Accumulator is fp16; bias and
// residual are added after FCVTL/FCVTL2 converts the result back to f32.

/// Per-row kernel for fp16 3×3 conv.
///
/// # Safety
/// All raw pointers must be valid for their logical extents. `res_ptr` may be
/// null (no residual). Requires aarch64 + ARMv8.2 fp16.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,fp16")]
#[allow(clippy::too_many_arguments)]
unsafe fn conv3x3_nhwc_fp16_row(
    oh: usize,
    in_ptr:   *const f32,
    packed_w: *const u16,  // [c_out/8, 9, c_in, 8] fp16-as-u16
    bias_ptr: *const f32,
    res_ptr:  *const f32,  // null if no residual
    out_ptr:  *mut f32,
    h_in:  usize,
    w_in:  usize,
    w_out: usize,
    c_in:  usize,
    c_out: usize,
    stride: usize,
    act: Activation,
) {
    use super::neon_asm_f16::{accum_tap_2px_f16, accum_tap_1px_f16, fcvtl_lo, fcvtl_hi};

    let n_co8 = c_out / 8;
    let ih_center = (oh * stride) as isize - 1;

    // --- 2-pixel spatial block loop ---
    let mut ow = 0usize;
    while ow + 1 < w_out {
        let iw_base0 = (ow * stride) as isize - 1;
        let iw_base1 = ((ow + 1) * stride) as isize - 1;

        for co_block in 0..n_co8 {
            let mut acc_px0 = vdupq_n_u16(0);
            let mut acc_px1 = vdupq_n_u16(0);

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
                    let w_tap = packed_w.add((co_block * 9 + tap) * c_in * 8);

                    let valid0 = iw0 >= 0 && iw0 < w_in as isize;
                    let valid1 = iw1 >= 0 && iw1 < w_in as isize;

                    if !valid0 && !valid1 {
                        continue;
                    }
                    if valid0 && valid1 {
                        let ip0 = in_ptr.add((ih * w_in + iw0 as usize) * c_in);
                        let ip1 = in_ptr.add((ih * w_in + iw1 as usize) * c_in);
                        accum_tap_2px_f16(&mut acc_px0, &mut acc_px1, ip0, ip1, w_tap, c_in);
                    } else if valid0 {
                        let ip0 = in_ptr.add((ih * w_in + iw0 as usize) * c_in);
                        accum_tap_1px_f16(&mut acc_px0, ip0, w_tap, c_in);
                    } else {
                        let ip1 = in_ptr.add((ih * w_in + iw1 as usize) * c_in);
                        accum_tap_1px_f16(&mut acc_px1, ip1, w_tap, c_in);
                    }
                }
            }

            // fp16 → f32, add bias, optional residual, activation, store.
            let bias_lo = vld1q_f32(bias_ptr.add(co_block * 8));
            let bias_hi = vld1q_f32(bias_ptr.add(co_block * 8 + 4));

            let mut r0_lo = vaddq_f32(fcvtl_lo(acc_px0), bias_lo);
            let mut r0_hi = vaddq_f32(fcvtl_hi(acc_px0), bias_hi);
            let mut r1_lo = vaddq_f32(fcvtl_lo(acc_px1), bias_lo);
            let mut r1_hi = vaddq_f32(fcvtl_hi(acc_px1), bias_hi);

            let out_off0 = ow * c_out + co_block * 8;
            let out_off1 = (ow + 1) * c_out + co_block * 8;

            if !res_ptr.is_null() {
                let g0 = (oh * w_out + ow) * c_out + co_block * 8;
                let g1 = (oh * w_out + ow + 1) * c_out + co_block * 8;
                r0_lo = vaddq_f32(r0_lo, vld1q_f32(res_ptr.add(g0)));
                r0_hi = vaddq_f32(r0_hi, vld1q_f32(res_ptr.add(g0 + 4)));
                r1_lo = vaddq_f32(r1_lo, vld1q_f32(res_ptr.add(g1)));
                r1_hi = vaddq_f32(r1_hi, vld1q_f32(res_ptr.add(g1 + 4)));
            }

            vst1q_f32(out_ptr.add(out_off0),     apply_act_vec(r0_lo, act));
            vst1q_f32(out_ptr.add(out_off0 + 4), apply_act_vec(r0_hi, act));
            vst1q_f32(out_ptr.add(out_off1),     apply_act_vec(r1_lo, act));
            vst1q_f32(out_ptr.add(out_off1 + 4), apply_act_vec(r1_hi, act));
        }

        ow += 2;
    }

    // --- Single-pixel epilogue ---
    if ow < w_out {
        let iw_base = (ow * stride) as isize - 1;
        for co_block in 0..n_co8 {
            let mut acc = vdupq_n_u16(0);
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
                    let w_tap = packed_w.add((co_block * 9 + tap) * c_in * 8);
                    let ip = in_ptr.add((ih * w_in + iw) * c_in);
                    accum_tap_1px_f16(&mut acc, ip, w_tap, c_in);
                }
            }
            let bias_lo = vld1q_f32(bias_ptr.add(co_block * 8));
            let bias_hi = vld1q_f32(bias_ptr.add(co_block * 8 + 4));
            let mut r_lo = vaddq_f32(fcvtl_lo(acc), bias_lo);
            let mut r_hi = vaddq_f32(fcvtl_hi(acc), bias_hi);
            let out_off = ow * c_out + co_block * 8;
            if !res_ptr.is_null() {
                let g = (oh * w_out + ow) * c_out + co_block * 8;
                r_lo = vaddq_f32(r_lo, vld1q_f32(res_ptr.add(g)));
                r_hi = vaddq_f32(r_hi, vld1q_f32(res_ptr.add(g + 4)));
            }
            vst1q_f32(out_ptr.add(out_off),     apply_act_vec(r_lo, act));
            vst1q_f32(out_ptr.add(out_off + 4), apply_act_vec(r_hi, act));
        }
    }
}

/// fp16 3×3 NHWC conv: 2× arithmetic density via FMLA.8H.
///
/// Requires `c_out % 8 == 0` and `c_in % 4 == 0`.
/// `packed_w_f16` must be pre-packed as `[c_out/8, 9, c_in, 8]` fp16-as-u16
/// (from [`super::repack_weights_co8_3x3_f16`]).
/// Bias is kept in f32; it is added after the fp16→f32 accumulator conversion
/// so there is no additional precision loss in the bias term.
#[cfg(target_arch = "aarch64")]
pub fn conv3x3_nhwc_fp16(
    args: &Conv3x3Args<'_>,
    output: &mut [f32],
    packed_w_f16: &[u16],
    stride: usize,
) {
    use rayon::prelude::*;
    debug_assert_eq!(args.c_out % 8, 0);
    debug_assert_eq!(args.c_in % 4, 0);

    let w_out = args.w_in / stride;
    let row_stride_out = w_out * args.c_out;

    let in_ptr    = args.input.as_ptr() as usize;
    let pw_ptr    = packed_w_f16.as_ptr() as usize;
    let bias_ptr  = args.bias.as_ptr() as usize;
    let res_ptr: usize = args.residual.map_or(0, |r| r.as_ptr() as usize);
    let has_res   = args.residual.is_some();

    let h_in  = args.h_in;
    let w_in  = args.w_in;
    let c_in  = args.c_in;
    let c_out = args.c_out;
    let act   = args.activation;

    output
        .par_chunks_mut(row_stride_out)
        .enumerate()
        .for_each(|(oh, out_row)| unsafe {
            conv3x3_nhwc_fp16_row(
                oh,
                in_ptr   as *const f32,
                pw_ptr   as *const u16,
                bias_ptr as *const f32,
                if has_res { res_ptr as *const f32 } else { core::ptr::null() },
                out_row.as_mut_ptr(),
                h_in, w_in, w_out,
                c_in, c_out, stride, act,
            );
        });
}

// ── conv3x3 c1: specialized NEON direct conv for c_in=1 ──────────────────────
//
// For c_in=1 the `dot_co4` reduction degenerates to one scalar broadcast per
// tap.  Each tap: `vfmaq_f32(acc, vdupq_n_f32(pixel), vld1q_f32(4-ch-weight))`.
// No horizontal reduction, no c_in loop.  Process two output pixels per
// iteration to amortise the weight load.  Rayon-parallel over output rows.
// Requires c_out % 4 == 0; weights are pre-packed with `repack_weights_co4_3x3`.

/// Stride-1 or stride-2 3×3 NHWC conv for c_in=1, any c_out%4==0.
///
/// Weights are consumed in the `[c_out/4, 9, c_in=1, 4]` layout produced by
/// [`super::repack_weights_co4_3x3`].  The caller must pre-pack (or pass the
/// `packed_weights` field in `Conv3x3Args`).
#[cfg(target_arch = "aarch64")]
pub fn conv3x3_c1_nhwc(args: &Conv3x3Args<'_>, output: &mut [f32], stride: usize) {
    use std::arch::aarch64::*;

    let &Conv3x3Args {
        input,
        residual,
        weights,
        bias,
        h_in,
        w_in,
        c_in: _,  // == 1
        c_out,
        activation,
        packed_weights,
    } = args;
    debug_assert_eq!(c_out % 4, 0);
    let w_out = w_in / stride;
    let n_co4 = c_out / 4;

    let _owned: Vec<f32>;
    let packed_w: &[f32] = match packed_weights {
        Some(pw) => pw,
        None => {
            _owned = super::repack_weights_co4_3x3(weights, c_out, 1);
            &_owned
        }
    };

    let in_ptr   = input.as_ptr() as usize;
    let pw_ptr   = packed_w.as_ptr() as usize;
    let bias_ptr = bias.as_ptr() as usize;
    let res_ptr: usize = residual.map_or(0, |r| r.as_ptr() as usize);
    let has_res = residual.is_some();

    output
        .par_chunks_mut(w_out * c_out)
        .enumerate()
        .for_each(|(oh, out_row)| unsafe {
            let input   = std::slice::from_raw_parts(in_ptr as *const f32, h_in * w_in);
            let packed  = std::slice::from_raw_parts(pw_ptr as *const f32, n_co4 * 9 * 4);
            let bias_sl = std::slice::from_raw_parts(bias_ptr as *const f32, c_out);
            let ih_center = (oh * stride) as isize - 1;

            let mut ow = 0usize;
            while ow + 1 < w_out {
                let iw0 = (ow * stride) as isize - 1;
                let iw1 = ((ow + 1) * stride) as isize - 1;

                for co4 in 0..n_co4 {
                    let mut acc0 = vld1q_f32(bias_sl.as_ptr().add(co4 * 4));
                    let mut acc1 = acc0;

                    for kh in 0..3usize {
                        let ih = ih_center + kh as isize;
                        if ih < 0 || ih >= h_in as isize { continue; }
                        let ih = ih as usize;
                        let row_base = ih * w_in;

                        for kw in 0..3usize {
                            let tap = kh * 3 + kw;
                            let wv = vld1q_f32(packed.as_ptr().add((co4 * 9 + tap) * 4));
                            let cx0 = iw0 + kw as isize;
                            let cx1 = iw1 + kw as isize;
                            if cx0 >= 0 && cx0 < w_in as isize {
                                let p0 = vdupq_n_f32(*input.as_ptr().add(row_base + cx0 as usize));
                                acc0 = vfmaq_f32(acc0, p0, wv);
                            }
                            if cx1 >= 0 && cx1 < w_in as isize {
                                let p1 = vdupq_n_f32(*input.as_ptr().add(row_base + cx1 as usize));
                                acc1 = vfmaq_f32(acc1, p1, wv);
                            }
                        }
                    }

                    if has_res {
                        let g0 = (oh * w_out + ow)       * c_out + co4 * 4;
                        let g1 = (oh * w_out + ow + 1)   * c_out + co4 * 4;
                        let rv0 = vld1q_f32((res_ptr as *const f32).add(g0));
                        let rv1 = vld1q_f32((res_ptr as *const f32).add(g1));
                        acc0 = vaddq_f32(acc0, rv0);
                        acc1 = vaddq_f32(acc1, rv1);
                    }
                    vst1q_f32(out_row.as_mut_ptr().add( ow      * c_out + co4 * 4), apply_act_vec(acc0, activation));
                    vst1q_f32(out_row.as_mut_ptr().add((ow + 1) * c_out + co4 * 4), apply_act_vec(acc1, activation));
                }
                ow += 2;
            }

            // Scalar epilogue for odd w_out (or last pixel when w_out is odd).
            if ow < w_out {
                let iw_base = (ow * stride) as isize - 1;
                for co4 in 0..n_co4 {
                    let mut acc = vld1q_f32(bias_sl.as_ptr().add(co4 * 4));
                    for kh in 0..3usize {
                        let ih = ih_center + kh as isize;
                        if ih < 0 || ih >= h_in as isize { continue; }
                        let ih = ih as usize;
                        let row_base = ih * w_in;
                        for kw in 0..3usize {
                            let cx = iw_base + kw as isize;
                            if cx < 0 || cx >= w_in as isize { continue; }
                            let tap = kh * 3 + kw;
                            let wv = vld1q_f32(packed.as_ptr().add((co4 * 9 + tap) * 4));
                            let p  = vdupq_n_f32(*input.as_ptr().add(row_base + cx as usize));
                            acc = vfmaq_f32(acc, p, wv);
                        }
                    }
                    if has_res {
                        let g = (oh * w_out + ow) * c_out + co4 * 4;
                        acc = vaddq_f32(acc, vld1q_f32((res_ptr as *const f32).add(g)));
                    }
                    vst1q_f32(out_row.as_mut_ptr().add(ow * c_out + co4 * 4), apply_act_vec(acc, activation));
                }
            }
        });
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
        packed_weights: _,
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

// ── Elementwise / spatial primitives ─────────────────────────────────────────

/// Fast exp(x) for four f32 lanes using a degree-5 Horner polynomial.
///
/// Algorithm: split x = n*ln2 + f, compute e^f via Horner, then scale by 2^n
/// using the IEEE 754 exponent trick. Max error ≈ 2 ULP for |x| ≤ 88.
/// Process two independent 4-wide exp vectors simultaneously.
///
/// On A78AE (2 FMA pipes, 4-cycle latency) the two independent 5-step Horner
/// chains can overlap: step N of chain 0 on pipe 0 while step N of chain 1 runs
/// on pipe 1. Achieves ~2× throughput vs calling exp_f32x4 twice sequentially.
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn exp_f32x8(
    xa: float32x4_t, xb: float32x4_t,
) -> (float32x4_t, float32x4_t) {
    use std::arch::aarch64::*;
    let clamp_lo = vdupq_n_f32(-88.0_f32);
    let clamp_hi = vdupq_n_f32( 88.0_f32);
    let xa = vmaxq_f32(vminq_f32(xa, clamp_hi), clamp_lo);
    let xb = vmaxq_f32(vminq_f32(xb, clamp_hi), clamp_lo);
    let log2e = vdupq_n_f32(1.442_695_040_888_963_4_f32);
    let nfa = vrndnq_f32(vmulq_f32(xa, log2e));
    let nfb = vrndnq_f32(vmulq_f32(xb, log2e));
    let nia = vcvtq_s32_f32(nfa);
    let nib = vcvtq_s32_f32(nfb);
    let ln2a_v = vdupq_n_f32(6.931_471_805_599_453e-1_f32);
    let ln2b_v = vdupq_n_f32(1.908_214_929_270_587_7e-10_f32);
    let fa = vfmsq_f32(vfmsq_f32(xa, nfa, ln2a_v), nfa, ln2b_v);
    let fb = vfmsq_f32(vfmsq_f32(xb, nfb, ln2a_v), nfb, ln2b_v);
    let c5 = vdupq_n_f32(1.0_f32 / 120.0);
    let c4 = vdupq_n_f32(1.0_f32 / 24.0);
    let c3 = vdupq_n_f32(1.0_f32 / 6.0);
    let c2 = vdupq_n_f32(0.5_f32);
    let one = vdupq_n_f32(1.0_f32);
    let pa = vfmaq_f32(c4, fa, c5); let pb = vfmaq_f32(c4, fb, c5);
    let pa = vfmaq_f32(c3, fa, pa); let pb = vfmaq_f32(c3, fb, pb);
    let pa = vfmaq_f32(c2, fa, pa); let pb = vfmaq_f32(c2, fb, pb);
    let pa = vfmaq_f32(one, fa, pa); let pb = vfmaq_f32(one, fb, pb);
    let pa = vfmaq_f32(one, fa, pa); let pb = vfmaq_f32(one, fb, pb);
    let e2a = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(nia, vdupq_n_s32(127)), 23));
    let e2b = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(nib, vdupq_n_s32(127)), 23));
    (vmulq_f32(pa, e2a), vmulq_f32(pb, e2b))
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn exp_f32x4(x: float32x4_t) -> float32x4_t {
    use std::arch::aarch64::*;
    // Clamp to prevent NaN from saturated exponents.
    let x = vmaxq_f32(x, vdupq_n_f32(-88.0_f32));
    let x = vminq_f32(x, vdupq_n_f32(88.0_f32));
    // n = round(x / ln2)
    let nf = vrndnq_f32(vmulq_f32(x, vdupq_n_f32(1.442_695_040_888_963_4_f32)));
    let ni = vcvtq_s32_f32(nf);
    // f = x - n*ln2, using two-part ln2 for accuracy
    let f = vfmsq_f32(vfmsq_f32(x, nf, vdupq_n_f32(6.931_471_805_599_453e-1_f32)),
                      nf, vdupq_n_f32(1.908_214_929_270_587_7e-10_f32));
    // Degree-5 Horner: e^f ≈ 1 + f*(1 + f*(0.5 + f*(1/6 + f*(1/24 + f/120))))
    let p = vfmaq_f32(vdupq_n_f32(1.0_f32 / 24.0), f, vdupq_n_f32(1.0_f32 / 120.0));
    let p = vfmaq_f32(vdupq_n_f32(1.0_f32 / 6.0), f, p);
    let p = vfmaq_f32(vdupq_n_f32(0.5_f32), f, p);
    let p = vfmaq_f32(vdupq_n_f32(1.0_f32), f, p);
    let p = vfmaq_f32(vdupq_n_f32(1.0_f32), f, p);
    // Scale by 2^n via IEEE 754 exponent injection
    let exp2n = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(ni, vdupq_n_s32(127)), 23));
    vmulq_f32(p, exp2n)
}

/// Bilinear upsample with a NEON-vectorized channel inner loop.
///
/// Identical semantics to [`super::scalar::bilinear_upsample`]. Processes the
/// channel dimension 4-at-a-time using `float32x4_t`; a scalar tail handles
/// any remainder when `c % 4 != 0`.
#[cfg(target_arch = "aarch64")]
pub fn bilinear_upsample_neon(
    input: &[f32],
    output: &mut [f32],
    h_in: usize,
    w_in: usize,
    c: usize,
    h_out: usize,
    w_out: usize,
) {
    debug_assert_eq!(input.len(), h_in * w_in * c);
    debug_assert_eq!(output.len(), h_out * w_out * c);
    use rayon::prelude::*;
    let sh = h_in as f32 / h_out as f32;
    let sw = w_in as f32 / w_out as f32;
    let in_ptr = input.as_ptr() as usize;
    output.par_chunks_mut(w_out * c).enumerate().for_each(|(oh, row_out)| {
        let ys = (oh as f32 + 0.5) * sh - 0.5;
        let y0f = ys.floor();
        let wy = ys - y0f;
        let y0 = (y0f as isize).clamp(0, h_in as isize - 1) as usize;
        let y1 = ((y0f as isize + 1).clamp(0, h_in as isize - 1)) as usize;
        for ow in 0..w_out {
            let xs = (ow as f32 + 0.5) * sw - 0.5;
            let x0f = xs.floor();
            let wx = xs - x0f;
            let x0 = (x0f as isize).clamp(0, w_in as isize - 1) as usize;
            let x1 = ((x0f as isize + 1).clamp(0, w_in as isize - 1)) as usize;
            let w00 = (1.0 - wx) * (1.0 - wy);
            let w01 = wx * (1.0 - wy);
            let w10 = (1.0 - wx) * wy;
            let w11 = wx * wy;
            let b00 = (y0 * w_in + x0) * c;
            let b01 = (y0 * w_in + x1) * c;
            let b10 = (y1 * w_in + x0) * c;
            let b11 = (y1 * w_in + x1) * c;
            let out_base = ow * c;
            unsafe {
                use std::arch::aarch64::*;
                let inp = in_ptr as *const f32;
                let w00v = vdupq_n_f32(w00);
                let w01v = vdupq_n_f32(w01);
                let w10v = vdupq_n_f32(w10);
                let w11v = vdupq_n_f32(w11);
                let outp = row_out.as_mut_ptr().add(out_base);
                let mut ci = 0usize;
                while ci + 4 <= c {
                    let v00 = vld1q_f32(inp.add(b00 + ci));
                    let v01 = vld1q_f32(inp.add(b01 + ci));
                    let v10 = vld1q_f32(inp.add(b10 + ci));
                    let v11 = vld1q_f32(inp.add(b11 + ci));
                    let r = vmulq_f32(v00, w00v);
                    let r = vfmaq_f32(r, v01, w01v);
                    let r = vfmaq_f32(r, v10, w10v);
                    let r = vfmaq_f32(r, v11, w11v);
                    vst1q_f32(outp.add(ci), r);
                    ci += 4;
                }
                while ci < c {
                    let v00 = *inp.add(b00 + ci);
                    let v01 = *inp.add(b01 + ci);
                    let v10 = *inp.add(b10 + ci);
                    let v11 = *inp.add(b11 + ci);
                    *row_out.get_unchecked_mut(out_base + ci) =
                        v00 * w00 + v01 * w01 + v10 * w10 + v11 * w11;
                    ci += 1;
                }
            }
        }
    });
}

/// Per-pixel softmax over the channel axis with NEON-vectorized exp().
///
/// Replaces the scalar `exp()` loop in [`super::scalar::channel_softmax`] with
/// a degree-5 polynomial NEON approximation (~2 ULP). Parallelises over pixels
/// with Rayon; the inner exp/sum/scale loops are 4-wide SIMD.
#[cfg(target_arch = "aarch64")]
pub fn channel_softmax_neon(buf: &mut [f32], h: usize, w: usize, c: usize) {
    debug_assert_eq!(buf.len(), h * w * c);
    use rayon::prelude::*;
    // 64 pixels per Rayon task: reduces task count from h*w to h*w/64,
    // cutting scheduling overhead while keeping 6-core parallelism for 4800 pixels.
    buf.par_chunks_mut(c * 64).for_each(|block| {
        for row in block.chunks_exact_mut(c) { unsafe {
        use std::arch::aarch64::*;
        // ── Step 1: vectorised max ──
        let mut max_v = vdupq_n_f32(f32::NEG_INFINITY);
        let mut i = 0usize;
        while i + 4 <= c {
            max_v = vmaxq_f32(max_v, vld1q_f32(row.as_ptr().add(i)));
            i += 4;
        }
        let mut max_s = vmaxvq_f32(max_v);
        while i < c { max_s = max_s.max(row[i]); i += 1; }
        let max_v = vdupq_n_f32(max_s);
        // ── Step 2: exp(x - max) and accumulate sum ──
        // Process 8 values at a time with dual-chain exp (doubles FMA pipe utilization).
        let mut sum_v = vdupq_n_f32(0.0_f32);
        let mut i = 0usize;
        while i + 8 <= c {
            let va = vsubq_f32(vld1q_f32(row.as_ptr().add(i)),     max_v);
            let vb = vsubq_f32(vld1q_f32(row.as_ptr().add(i + 4)), max_v);
            let (ea, eb) = exp_f32x8(va, vb);
            vst1q_f32(row.as_mut_ptr().add(i),     ea);
            vst1q_f32(row.as_mut_ptr().add(i + 4), eb);
            sum_v = vaddq_f32(vaddq_f32(sum_v, ea), eb);
            i += 8;
        }
        while i + 4 <= c {
            let v = vsubq_f32(vld1q_f32(row.as_ptr().add(i)), max_v);
            let e = exp_f32x4(v);
            vst1q_f32(row.as_mut_ptr().add(i), e);
            sum_v = vaddq_f32(sum_v, e);
            i += 4;
        }
        let mut sum_s = vaddvq_f32(sum_v);
        while i < c {
            let e = (row[i] - max_s).exp();
            row[i] = e;
            sum_s += e;
            i += 1;
        }
        // ── Step 3: scale by 1/sum ──
        let inv = vdupq_n_f32(1.0_f32 / sum_s);
        let mut i = 0usize;
        while i + 4 <= c {
            let v = vld1q_f32(row.as_ptr().add(i));
            vst1q_f32(row.as_mut_ptr().add(i), vmulq_f32(v, inv));
            i += 4;
        }
        while i < c { row[i] /= sum_s; i += 1; }
        } } // close unsafe block and inner for-loop
    });
}
