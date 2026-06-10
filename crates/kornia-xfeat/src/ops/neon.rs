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
    ip0: *const f32,        // input pointer for pixel 0 (never null here)
    ip1: *const f32,        // input pointer for pixel 1 (never null here)
    w_tap: *const f32,      // weight pointer: [c_in, 4] for this tap
    c_in: usize,
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
unsafe fn accum_tap_1px(acc: &mut float32x4_t, ip: *const f32, w_tap: *const f32, c_in: usize) {
    let mut ci = 0usize;
    while ci < c_in {
        let iv = vld1q_f32(ip.add(ci));
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
    in_ptr: *const f32,
    packed_w: *const u16, // [c_out/8, 9, c_in, 8] fp16-as-u16
    bias_ptr: *const f32,
    res_ptr: *const f32, // null if no residual
    out_ptr: *mut f32,
    h_in: usize,
    w_in: usize,
    w_out: usize,
    c_in: usize,
    c_out: usize,
    stride: usize,
    act: Activation,
) {
    use super::neon_asm_f16::{accum_tap_1px_f16, accum_tap_2px_f16, fcvtl_hi, fcvtl_lo};

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

            vst1q_f32(out_ptr.add(out_off0), apply_act_vec(r0_lo, act));
            vst1q_f32(out_ptr.add(out_off0 + 4), apply_act_vec(r0_hi, act));
            vst1q_f32(out_ptr.add(out_off1), apply_act_vec(r1_lo, act));
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
            vst1q_f32(out_ptr.add(out_off), apply_act_vec(r_lo, act));
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

    let in_ptr = args.input.as_ptr() as usize;
    let pw_ptr = packed_w_f16.as_ptr() as usize;
    let bias_ptr = args.bias.as_ptr() as usize;
    let res_ptr: usize = args.residual.map_or(0, |r| r.as_ptr() as usize);
    let has_res = args.residual.is_some();

    let h_in = args.h_in;
    let w_in = args.w_in;
    let c_in = args.c_in;
    let c_out = args.c_out;
    let act = args.activation;

    output
        .par_chunks_mut(row_stride_out)
        .enumerate()
        .for_each(|(oh, out_row)| unsafe {
            conv3x3_nhwc_fp16_row(
                oh,
                in_ptr as *const f32,
                pw_ptr as *const u16,
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
                act,
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
        c_in: _, // == 1
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

    let in_ptr = input.as_ptr() as usize;
    let pw_ptr = packed_w.as_ptr() as usize;
    let bias_ptr = bias.as_ptr() as usize;
    let res_ptr: usize = residual.map_or(0, |r| r.as_ptr() as usize);
    let has_res = residual.is_some();

    output
        .par_chunks_mut(w_out * c_out)
        .enumerate()
        .for_each(|(oh, out_row)| unsafe {
            let input = std::slice::from_raw_parts(in_ptr as *const f32, h_in * w_in);
            let packed = std::slice::from_raw_parts(pw_ptr as *const f32, n_co4 * 9 * 4);
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
                        if ih < 0 || ih >= h_in as isize {
                            continue;
                        }
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
                        let g0 = (oh * w_out + ow) * c_out + co4 * 4;
                        let g1 = (oh * w_out + ow + 1) * c_out + co4 * 4;
                        let rv0 = vld1q_f32((res_ptr as *const f32).add(g0));
                        let rv1 = vld1q_f32((res_ptr as *const f32).add(g1));
                        acc0 = vaddq_f32(acc0, rv0);
                        acc1 = vaddq_f32(acc1, rv1);
                    }
                    vst1q_f32(
                        out_row.as_mut_ptr().add(ow * c_out + co4 * 4),
                        apply_act_vec(acc0, activation),
                    );
                    vst1q_f32(
                        out_row.as_mut_ptr().add((ow + 1) * c_out + co4 * 4),
                        apply_act_vec(acc1, activation),
                    );
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
                        if ih < 0 || ih >= h_in as isize {
                            continue;
                        }
                        let ih = ih as usize;
                        let row_base = ih * w_in;
                        for kw in 0..3usize {
                            let cx = iw_base + kw as isize;
                            if cx < 0 || cx >= w_in as isize {
                                continue;
                            }
                            let tap = kh * 3 + kw;
                            let wv = vld1q_f32(packed.as_ptr().add((co4 * 9 + tap) * 4));
                            let p = vdupq_n_f32(*input.as_ptr().add(row_base + cx as usize));
                            acc = vfmaq_f32(acc, p, wv);
                        }
                    }
                    if has_res {
                        let g = (oh * w_out + ow) * c_out + co4 * 4;
                        acc = vaddq_f32(acc, vld1q_f32((res_ptr as *const f32).add(g)));
                    }
                    vst1q_f32(
                        out_row.as_mut_ptr().add(ow * c_out + co4 * 4),
                        apply_act_vec(acc, activation),
                    );
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
unsafe fn exp_f32x8(xa: float32x4_t, xb: float32x4_t) -> (float32x4_t, float32x4_t) {
    use std::arch::aarch64::*;
    let clamp_lo = vdupq_n_f32(-88.0_f32);
    let clamp_hi = vdupq_n_f32(88.0_f32);
    let xa = vmaxq_f32(vminq_f32(xa, clamp_hi), clamp_lo);
    let xb = vmaxq_f32(vminq_f32(xb, clamp_hi), clamp_lo);
    let log2e = vdupq_n_f32(std::f32::consts::LOG2_E);
    let nfa = vrndnq_f32(vmulq_f32(xa, log2e));
    let nfb = vrndnq_f32(vmulq_f32(xb, log2e));
    let nia = vcvtq_s32_f32(nfa);
    let nib = vcvtq_s32_f32(nfb);
    // Two-part ln2: high part is the named constant, low part is the residual
    // correction (LN_2 - (f32)LN_2) for extended-precision range reduction.
    let ln2a_v = vdupq_n_f32(std::f32::consts::LN_2);
    let ln2b_v = vdupq_n_f32(1.908_214_929_270_587_7e-10_f32);
    let fa = vfmsq_f32(vfmsq_f32(xa, nfa, ln2a_v), nfa, ln2b_v);
    let fb = vfmsq_f32(vfmsq_f32(xb, nfb, ln2a_v), nfb, ln2b_v);
    let c5 = vdupq_n_f32(1.0_f32 / 120.0);
    let c4 = vdupq_n_f32(1.0_f32 / 24.0);
    let c3 = vdupq_n_f32(1.0_f32 / 6.0);
    let c2 = vdupq_n_f32(0.5_f32);
    let one = vdupq_n_f32(1.0_f32);
    let pa = vfmaq_f32(c4, fa, c5);
    let pb = vfmaq_f32(c4, fb, c5);
    let pa = vfmaq_f32(c3, fa, pa);
    let pb = vfmaq_f32(c3, fb, pb);
    let pa = vfmaq_f32(c2, fa, pa);
    let pb = vfmaq_f32(c2, fb, pb);
    let pa = vfmaq_f32(one, fa, pa);
    let pb = vfmaq_f32(one, fb, pb);
    let pa = vfmaq_f32(one, fa, pa);
    let pb = vfmaq_f32(one, fb, pb);
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
    let nf = vrndnq_f32(vmulq_f32(x, vdupq_n_f32(std::f32::consts::LOG2_E)));
    let ni = vcvtq_s32_f32(nf);
    // f = x - n*ln2, using two-part ln2 for accuracy
    let f = vfmsq_f32(
        vfmsq_f32(x, nf, vdupq_n_f32(std::f32::consts::LN_2)),
        nf,
        vdupq_n_f32(1.908_214_929_270_587_7e-10_f32),
    );
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
/// 2-pass NEON instance-normalization for single-channel 2-D activation maps.
///
/// **Pass 1** (one Rayon `fold`+`reduce`): accumulates `sum` and `sum_sq`
/// simultaneously using `vaddq_f32` + `vfmaq_f32`, avoiding a separate variance
/// pass and saving one Rayon barrier. **Pass 2**: parallel normalize
/// `(x − mean) × inv_std` using `vsubq_f32` + `vmulq_f32`.
///
/// ~3–4× faster than the scalar 3-pass path on A78AE (4-wide SIMD, 1 fewer barrier).
#[cfg(target_arch = "aarch64")]
pub fn instance_norm_2d_singlech_neon(input: &[f32], output: &mut [f32]) {
    use rayon::prelude::*;
    use std::arch::aarch64::*;

    let n = input.len();
    assert_eq!(n, output.len());
    let n_f = n as f32;

    // ── Pass 1: parallel (sum, sum_sq) via fold+reduce ───────────────────
    const FOLD_CHUNK: usize = 16384;
    let (sum, sum_sq) = input
        .par_chunks(FOLD_CHUNK)
        .fold(
            || (0.0f32, 0.0f32),
            |acc, chunk| {
                let (mut s, mut ss) = acc;
                // NEON 4-wide accumulation.
                let mut vs = unsafe { vdupq_n_f32(0.0) };
                let mut vss = unsafe { vdupq_n_f32(0.0) };
                let ptr = chunk.as_ptr();
                let n4 = chunk.len() / 4;
                for i in 0..n4 {
                    let v = unsafe { vld1q_f32(ptr.add(i * 4)) };
                    vs = unsafe { vaddq_f32(vs, v) };
                    vss = unsafe { vfmaq_f32(vss, v, v) };
                }
                s += unsafe { vaddvq_f32(vs) };
                ss += unsafe { vaddvq_f32(vss) };
                for &x in &chunk[n4 * 4..] {
                    s += x;
                    ss += x * x;
                }
                (s, ss)
            },
        )
        .reduce(
            || (0.0f32, 0.0f32),
            |(s1, ss1), (s2, ss2)| (s1 + s2, ss1 + ss2),
        );

    let mean = sum / n_f;
    let var = (sum_sq / n_f) - mean * mean;
    let inv_std = (var + 1e-5_f32).sqrt().recip();

    // ── Pass 2: parallel normalize ────────────────────────────────────────
    let vmean = unsafe { vdupq_n_f32(mean) };
    let vinv_std = unsafe { vdupq_n_f32(inv_std) };
    let vmean_addr = &vmean as *const float32x4_t as usize;
    let vinv_std_addr = &vinv_std as *const float32x4_t as usize;

    input
        .par_chunks(FOLD_CHUNK)
        .zip(output.par_chunks_mut(FOLD_CHUNK))
        .for_each(|(src, dst)| {
            let vm = unsafe { *(vmean_addr as *const float32x4_t) };
            let vis = unsafe { *(vinv_std_addr as *const float32x4_t) };
            let sp = src.as_ptr();
            let dp = dst.as_mut_ptr();
            let n4 = src.len() / 4;
            for i in 0..n4 {
                let v = unsafe { vld1q_f32(sp.add(i * 4)) };
                let r = unsafe { vmulq_f32(vsubq_f32(v, vm), vis) };
                unsafe { vst1q_f32(dp.add(i * 4), r) };
            }
            for i in n4 * 4..src.len() {
                unsafe { *dp.add(i) = (*sp.add(i) - mean) * inv_std };
            }
        });
}

/// Unfold 8×8 patches + f32→f16 narrowing, NEON-vectorized.
///
/// Identical semantics to [`super::scalar::unfold_8x8_to_f16`] but replaces
/// 8 scalar `f16::from_f32().to_bits()` calls per kh-step with
/// `vld1q_f32` × 2 + `FCVTN` × 2 + `vcombine_u16` + `vst1q_u16`.
/// ~4× fewer cycles per 8 elements on A78AE.
#[cfg(target_arch = "aarch64")]
pub fn unfold_8x8_to_f16_neon(input: &[f32], output: &mut [u16], h_in: usize, w_in: usize) {
    use crate::ops::neon_asm_f16::fcvtn_f32x4_to_f16x4;
    use rayon::prelude::*;
    use std::arch::aarch64::*;

    let h_out = h_in / 8;
    let w_out = w_in / 8;
    debug_assert_eq!(input.len(), h_in * w_in);
    debug_assert_eq!(output.len(), h_out * w_out * 64);

    output
        .par_chunks_mut(w_out * 64)
        .enumerate()
        .for_each(|(oh, row_out)| {
            for ow in 0..w_out {
                unsafe {
                    let in_base = input.as_ptr().add((oh * 8) * w_in + ow * 8);
                    let out_base = row_out.as_mut_ptr().add(ow * 64);
                    for kh in 0..8usize {
                        let in_ptr = in_base.add(kh * w_in);
                        let v_lo = vld1q_f32(in_ptr);
                        let v_hi = vld1q_f32(in_ptr.add(4));
                        let f16_lo = fcvtn_f32x4_to_f16x4(v_lo);
                        let f16_hi = fcvtn_f32x4_to_f16x4(v_hi);
                        let f16_8 = vcombine_u16(f16_lo, f16_hi);
                        vst1q_u16(out_base.add(kh * 8), f16_8);
                    }
                }
            }
        });
}

/// Pixel-shuffle (factor 8) + f16→f32 widening, NEON-vectorized.
///
/// Identical semantics to [`super::scalar::pixel_shuffle_8_f16`] but replaces
/// 8 scalar `f16::to_f32()` calls per output row with one `vld1q_u16` +
/// `FCVTL`/`FCVTL2` + two `vst1q_f32` — 1 cycle per 8 elements vs ~5 scalar.
#[cfg(target_arch = "aarch64")]
pub fn pixel_shuffle_8_f16_neon(input: &[u16], output: &mut [f32], h_in: usize, w_in: usize) {
    use crate::ops::neon_asm_f16::{fcvtl_hi, fcvtl_lo};
    use rayon::prelude::*;
    use std::arch::aarch64::*;

    debug_assert_eq!(input.len(), h_in * w_in * 64);
    debug_assert_eq!(output.len(), h_in * 8 * w_in * 8);

    let w_out = w_in * 8;
    output
        .par_chunks_mut(8 * w_out)
        .zip(input.par_chunks(w_in * 64))
        .for_each(|(super_row, in_row)| {
            for w in 0..w_in {
                unsafe {
                    let in_ptr = in_row.as_ptr().add(w * 64);
                    let out_base = super_row.as_mut_ptr().add(w * 8);
                    for kh in 0..8usize {
                        let h8 = vld1q_u16(in_ptr.add(kh * 8));
                        let lo = fcvtl_lo(h8);
                        let hi = fcvtl_hi(h8);
                        let out_row = out_base.add(kh * w_out);
                        vst1q_f32(out_row, lo);
                        vst1q_f32(out_row.add(4), hi);
                    }
                }
            }
        });
}

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
    output
        .par_chunks_mut(w_out * c)
        .enumerate()
        .for_each(|(oh, row_out)| {
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

/// Fused FPN merge: `x3 += upsample(x4) + upsample(x5)` in ONE Rayon pass.
///
/// Replaces the `bilinear_upsample(x4)` → `bilinear_upsample(x5)` →
/// `add3_inplace` three-dispatch sequence with a single row-parallel pass,
/// eliminating two Rayon barriers and the `x4_up`/`x5_up` intermediate
/// buffers (~2.4 MB of write+read traffic at 480×640).
///
/// Bit-exactness: each upsample tap uses the identical
/// `vmulq`/`vfmaq` chain as [`bilinear_upsample_neon`], and the final sum is
/// grouped `x3 + (up4 + up5)` — the same grouping as
/// `add3_inplace`'s `*x += y + z`. Output is therefore bit-identical to the
/// unfused pipeline.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
pub fn fpn_upsample2_add3_neon(
    x3: &mut [f32],
    x4: &[f32],
    h4: usize,
    w4: usize,
    x5: &[f32],
    h5: usize,
    w5: usize,
    c: usize,
    h_out: usize,
    w_out: usize,
) {
    debug_assert_eq!(x3.len(), h_out * w_out * c);
    debug_assert_eq!(x4.len(), h4 * w4 * c);
    debug_assert_eq!(x5.len(), h5 * w5 * c);
    use rayon::prelude::*;
    let sh4 = h4 as f32 / h_out as f32;
    let sw4 = w4 as f32 / w_out as f32;
    let sh5 = h5 as f32 / h_out as f32;
    let sw5 = w5 as f32 / w_out as f32;
    let p4 = x4.as_ptr() as usize;
    let p5 = x5.as_ptr() as usize;
    x3.par_chunks_mut(w_out * c)
        .enumerate()
        .for_each(|(oh, row_out)| {
            // y-coords/weights for each source (same formulas as bilinear_upsample_neon).
            let ys4 = (oh as f32 + 0.5) * sh4 - 0.5;
            let y0f4 = ys4.floor();
            let wy4 = ys4 - y0f4;
            let y0_4 = (y0f4 as isize).clamp(0, h4 as isize - 1) as usize;
            let y1_4 = ((y0f4 as isize + 1).clamp(0, h4 as isize - 1)) as usize;
            let ys5 = (oh as f32 + 0.5) * sh5 - 0.5;
            let y0f5 = ys5.floor();
            let wy5 = ys5 - y0f5;
            let y0_5 = (y0f5 as isize).clamp(0, h5 as isize - 1) as usize;
            let y1_5 = ((y0f5 as isize + 1).clamp(0, h5 as isize - 1)) as usize;
            for ow in 0..w_out {
                let xs4 = (ow as f32 + 0.5) * sw4 - 0.5;
                let x0f4 = xs4.floor();
                let wx4 = xs4 - x0f4;
                let x0_4 = (x0f4 as isize).clamp(0, w4 as isize - 1) as usize;
                let x1_4 = ((x0f4 as isize + 1).clamp(0, w4 as isize - 1)) as usize;
                let w00_4 = (1.0 - wx4) * (1.0 - wy4);
                let w01_4 = wx4 * (1.0 - wy4);
                let w10_4 = (1.0 - wx4) * wy4;
                let w11_4 = wx4 * wy4;
                let b00_4 = (y0_4 * w4 + x0_4) * c;
                let b01_4 = (y0_4 * w4 + x1_4) * c;
                let b10_4 = (y1_4 * w4 + x0_4) * c;
                let b11_4 = (y1_4 * w4 + x1_4) * c;

                let xs5 = (ow as f32 + 0.5) * sw5 - 0.5;
                let x0f5 = xs5.floor();
                let wx5 = xs5 - x0f5;
                let x0_5 = (x0f5 as isize).clamp(0, w5 as isize - 1) as usize;
                let x1_5 = ((x0f5 as isize + 1).clamp(0, w5 as isize - 1)) as usize;
                let w00_5 = (1.0 - wx5) * (1.0 - wy5);
                let w01_5 = wx5 * (1.0 - wy5);
                let w10_5 = (1.0 - wx5) * wy5;
                let w11_5 = wx5 * wy5;
                let b00_5 = (y0_5 * w5 + x0_5) * c;
                let b01_5 = (y0_5 * w5 + x1_5) * c;
                let b10_5 = (y1_5 * w5 + x0_5) * c;
                let b11_5 = (y1_5 * w5 + x1_5) * c;

                let out_base = ow * c;
                unsafe {
                    use std::arch::aarch64::*;
                    let i4 = p4 as *const f32;
                    let i5 = p5 as *const f32;
                    let w00v4 = vdupq_n_f32(w00_4);
                    let w01v4 = vdupq_n_f32(w01_4);
                    let w10v4 = vdupq_n_f32(w10_4);
                    let w11v4 = vdupq_n_f32(w11_4);
                    let w00v5 = vdupq_n_f32(w00_5);
                    let w01v5 = vdupq_n_f32(w01_5);
                    let w10v5 = vdupq_n_f32(w10_5);
                    let w11v5 = vdupq_n_f32(w11_5);
                    let outp = row_out.as_mut_ptr().add(out_base);
                    let mut ci = 0usize;
                    while ci + 4 <= c {
                        // upsample(x4) tap — identical chain to bilinear_upsample_neon
                        let v00 = vld1q_f32(i4.add(b00_4 + ci));
                        let v01 = vld1q_f32(i4.add(b01_4 + ci));
                        let v10 = vld1q_f32(i4.add(b10_4 + ci));
                        let v11 = vld1q_f32(i4.add(b11_4 + ci));
                        let r4 = vmulq_f32(v00, w00v4);
                        let r4 = vfmaq_f32(r4, v01, w01v4);
                        let r4 = vfmaq_f32(r4, v10, w10v4);
                        let r4 = vfmaq_f32(r4, v11, w11v4);
                        // upsample(x5) tap
                        let v00 = vld1q_f32(i5.add(b00_5 + ci));
                        let v01 = vld1q_f32(i5.add(b01_5 + ci));
                        let v10 = vld1q_f32(i5.add(b10_5 + ci));
                        let v11 = vld1q_f32(i5.add(b11_5 + ci));
                        let r5 = vmulq_f32(v00, w00v5);
                        let r5 = vfmaq_f32(r5, v01, w01v5);
                        let r5 = vfmaq_f32(r5, v10, w10v5);
                        let r5 = vfmaq_f32(r5, v11, w11v5);
                        // x3 + (up4 + up5) — same grouping as add3_inplace
                        let x3v = vld1q_f32(outp.add(ci));
                        vst1q_f32(outp.add(ci), vaddq_f32(x3v, vaddq_f32(r4, r5)));
                        ci += 4;
                    }
                    while ci < c {
                        let r4 = *i4.add(b00_4 + ci) * w00_4
                            + *i4.add(b01_4 + ci) * w01_4
                            + *i4.add(b10_4 + ci) * w10_4
                            + *i4.add(b11_4 + ci) * w11_4;
                        let r5 = *i5.add(b00_5 + ci) * w00_5
                            + *i5.add(b01_5 + ci) * w01_5
                            + *i5.add(b10_5 + ci) * w10_5
                            + *i5.add(b11_5 + ci) * w11_5;
                        let x = row_out.get_unchecked_mut(out_base + ci);
                        *x += r4 + r5;
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
        for row in block.chunks_exact_mut(c) {
            unsafe {
                use std::arch::aarch64::*;
                // ── Step 1: vectorised max ──
                let mut max_v = vdupq_n_f32(f32::NEG_INFINITY);
                let mut i = 0usize;
                while i + 4 <= c {
                    max_v = vmaxq_f32(max_v, vld1q_f32(row.as_ptr().add(i)));
                    i += 4;
                }
                let mut max_s = vmaxvq_f32(max_v);
                while i < c {
                    max_s = max_s.max(row[i]);
                    i += 1;
                }
                let max_v = vdupq_n_f32(max_s);
                // ── Step 2: exp(x - max) and accumulate sum ──
                // Process 8 values at a time with dual-chain exp (doubles FMA pipe utilization).
                let mut sum_v = vdupq_n_f32(0.0_f32);
                let mut i = 0usize;
                while i + 8 <= c {
                    let va = vsubq_f32(vld1q_f32(row.as_ptr().add(i)), max_v);
                    let vb = vsubq_f32(vld1q_f32(row.as_ptr().add(i + 4)), max_v);
                    let (ea, eb) = exp_f32x8(va, vb);
                    vst1q_f32(row.as_mut_ptr().add(i), ea);
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
                while i < c {
                    row[i] /= sum_s;
                    i += 1;
                }
            }
        } // close unsafe block and inner for-loop
    });
}

/// Per-pixel channel softmax on f16 storage (u16 bit-pattern) with NEON-vectorized exp().
///
/// Identical semantics to [`channel_softmax_neon`] but operates on `u16`-encoded f16 values.
/// Each group of 8 u16 is widened to two `float32x4_t` via FCVTL/FCVTL2, exp computed in
/// f32 for numerical stability (same degree-5 polynomial), then narrowed back to f16 via FCVTN.
/// Requires ARMv8.2 fp16 (FCVTL/FCVTL2/FCVTN instructions); caller must check `has_fp16`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,fp16")]
pub unsafe fn channel_softmax_neon_f16_kernel(buf: &mut [u16], h: usize, w: usize, c: usize) {
    debug_assert_eq!(buf.len(), h * w * c);
    use crate::ops::neon_asm_f16::{fcvtl_hi, fcvtl_lo, fcvtn_f32x4_to_f16x4};
    use std::arch::aarch64::*;

    for row in buf.chunks_exact_mut(c) {
        unsafe {
            // ── Step 1: find max across all channels (in f32) ──
            let mut max_v = vdupq_n_f32(f32::NEG_INFINITY);
            let mut i = 0usize;
            while i + 8 <= c {
                let u16x8 = vld1q_u16(row.as_ptr().add(i));
                let fa = fcvtl_lo(u16x8);
                let fb = fcvtl_hi(u16x8);
                max_v = vmaxq_f32(max_v, vmaxq_f32(fa, fb));
                i += 8;
            }
            while i + 4 <= c {
                // Load 4 u16, widen to f32x4. Pad with zeros to form a full uint16x8_t.
                let lo = vld1_u16(row.as_ptr().add(i));
                let u16x8 = vcombine_u16(lo, vdup_n_u16(0));
                let fa = fcvtl_lo(u16x8);
                max_v = vmaxq_f32(max_v, fa);
                i += 4;
            }
            let mut max_s = vmaxvq_f32(max_v);
            while i < c {
                let f = half::f16::from_bits(row[i]).to_f32();
                if f > max_s {
                    max_s = f;
                }
                i += 1;
            }
            let max_v = vdupq_n_f32(max_s);

            // ── Step 2: exp(x - max) in f32, store result back as f16, accumulate sum ──
            let mut sum_v = vdupq_n_f32(0.0_f32);
            let mut i = 0usize;
            // Process 8 u16 at a time: widen to 2×f32x4, dual-chain exp, narrow back.
            while i + 8 <= c {
                let u16x8 = vld1q_u16(row.as_ptr().add(i));
                let fa = fcvtl_lo(u16x8);
                let fb = fcvtl_hi(u16x8);
                let va = vsubq_f32(fa, max_v);
                let vb = vsubq_f32(fb, max_v);
                let (ea, eb) = exp_f32x8(va, vb);
                // Narrow back to f16 and store.
                let lo = fcvtn_f32x4_to_f16x4(ea);
                let hi = fcvtn_f32x4_to_f16x4(eb);
                vst1q_u16(row.as_mut_ptr().add(i), vcombine_u16(lo, hi));
                sum_v = vaddq_f32(vaddq_f32(sum_v, ea), eb);
                i += 8;
            }
            // Process 4 u16 at a time.
            while i + 4 <= c {
                let lo = vld1_u16(row.as_ptr().add(i));
                let u16x8 = vcombine_u16(lo, vdup_n_u16(0));
                let fa = fcvtl_lo(u16x8);
                let va = vsubq_f32(fa, max_v);
                let ea = exp_f32x4(va);
                let lo_out = fcvtn_f32x4_to_f16x4(ea);
                vst1_u16(row.as_mut_ptr().add(i), lo_out);
                sum_v = vaddq_f32(sum_v, ea);
                i += 4;
            }
            let mut sum_s = vaddvq_f32(sum_v);
            // Scalar tail.
            while i < c {
                let f = half::f16::from_bits(row[i]).to_f32();
                let e = (f - max_s).exp();
                row[i] = half::f16::from_f32(e).to_bits();
                sum_s += e;
                i += 1;
            }

            // ── Step 3: scale all exp values by 1/sum ──
            let inv_v = vdupq_n_f32(1.0_f32 / sum_s);
            let mut i = 0usize;
            while i + 8 <= c {
                let u16x8 = vld1q_u16(row.as_ptr().add(i));
                let fa = fcvtl_lo(u16x8);
                let fb = fcvtl_hi(u16x8);
                let ra = vmulq_f32(fa, inv_v);
                let rb = vmulq_f32(fb, inv_v);
                let lo = fcvtn_f32x4_to_f16x4(ra);
                let hi = fcvtn_f32x4_to_f16x4(rb);
                vst1q_u16(row.as_mut_ptr().add(i), vcombine_u16(lo, hi));
                i += 8;
            }
            while i + 4 <= c {
                let lo = vld1_u16(row.as_ptr().add(i));
                let u16x8 = vcombine_u16(lo, vdup_n_u16(0));
                let fa = fcvtl_lo(u16x8);
                let ra = vmulq_f32(fa, inv_v);
                let lo_out = fcvtn_f32x4_to_f16x4(ra);
                vst1_u16(row.as_mut_ptr().add(i), lo_out);
                i += 4;
            }
            while i < c {
                let f = half::f16::from_bits(row[i]).to_f32();
                row[i] = half::f16::from_f32(f / sum_s).to_bits();
                i += 1;
            }
        } // close unsafe block
    }
}

/// Safe wrapper — dispatches to the fp16 kernel above. Must only be called
/// when `cpu_features().has_fp16` is true (asserted in debug builds).
#[cfg(target_arch = "aarch64")]
pub fn channel_softmax_neon_f16(buf: &mut [u16], h: usize, w: usize, c: usize) {
    debug_assert!(
        crate::cpu_features::cpu_features().has_fp16,
        "channel_softmax_neon_f16 called on a CPU without fp16 support"
    );
    // SAFETY: we verified has_fp16 above; the kernel only uses neon+fp16 ops.
    unsafe { channel_softmax_neon_f16_kernel(buf, h, w, c) }
}

/// Rayon-parallel variant: uses 64-pixel chunks (same as scalar) so each chunk
/// fits in L1 (64px × 65ch × 2B = 8.3KB) and Rayon gets 75 work units for good
/// load balancing across 6 cores. The polynomial exp kernel provides ~3–5× speedup
/// over scalar per chunk.
#[cfg(target_arch = "aarch64")]
pub fn channel_softmax_neon_f16_par(buf: &mut [u16], h: usize, w: usize, c: usize) {
    use rayon::prelude::*;
    debug_assert_eq!(buf.len(), h * w * c);
    debug_assert!(
        crate::cpu_features::cpu_features().has_fp16,
        "channel_softmax_neon_f16_par called on a CPU without fp16 support"
    );
    buf.par_chunks_mut(c * 64).for_each(|block| {
        let px = block.len() / c;
        // SAFETY: fp16 checked above; no cross-chunk state.
        unsafe { channel_softmax_neon_f16_kernel(block, 1, px, c) };
    });
}

/// Sidecar-aware per-pixel channel softmax on f16 storage (NEON-vectorized exp).
///
/// Computes softmax over `c_main + 1` logits per pixel: the first `c_main` live in
/// `main` (row stride `c_main`), the last "dustbin" logit lives in `dustbin` (one
/// value per pixel). Mirrors [`channel_softmax_f16_sidecar`]'s reduction order —
/// max/sum over the `c_main` main values first (vectorized), dustbin folded in last
/// (scalar) — so it matches the interleaved scalar reference within the f16-scale
/// softmax tolerance.
///
/// Requires ARMv8.2 fp16; caller must check `has_fp16`.
///
/// # Safety
/// `main.len() >= rows * c_main`, `dustbin.len() >= rows`, fp16 supported.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,fp16")]
unsafe fn channel_softmax_neon_f16_sidecar_kernel(
    main: &mut [u16],
    dustbin: &mut [u16],
    c_main: usize,
) {
    use crate::ops::neon_asm_f16::{fcvtl_hi, fcvtl_lo, fcvtn_f32x4_to_f16x4};
    use std::arch::aarch64::*;

    for (row, d) in main.chunks_exact_mut(c_main).zip(dustbin.iter_mut()) {
        unsafe {
            // ── Step 1: max across the c_main main channels, then fold dustbin ──
            let mut max_v = vdupq_n_f32(f32::NEG_INFINITY);
            let mut i = 0usize;
            while i + 8 <= c_main {
                let u16x8 = vld1q_u16(row.as_ptr().add(i));
                let fa = fcvtl_lo(u16x8);
                let fb = fcvtl_hi(u16x8);
                max_v = vmaxq_f32(max_v, vmaxq_f32(fa, fb));
                i += 8;
            }
            while i + 4 <= c_main {
                let lo = vld1_u16(row.as_ptr().add(i));
                let u16x8 = vcombine_u16(lo, vdup_n_u16(0));
                let fa = fcvtl_lo(u16x8);
                max_v = vmaxq_f32(max_v, fa);
                i += 4;
            }
            let mut max_s = vmaxvq_f32(max_v);
            while i < c_main {
                let f = half::f16::from_bits(row[i]).to_f32();
                if f > max_s {
                    max_s = f;
                }
                i += 1;
            }
            // Dustbin folded in last (matches scalar reference order).
            let dustbin_f = half::f16::from_bits(*d).to_f32();
            if dustbin_f > max_s {
                max_s = dustbin_f;
            }
            let max_v = vdupq_n_f32(max_s);

            // ── Step 2: exp(x - max), store back, accumulate sum (main then dustbin) ──
            let mut sum_v = vdupq_n_f32(0.0_f32);
            let mut i = 0usize;
            while i + 8 <= c_main {
                let u16x8 = vld1q_u16(row.as_ptr().add(i));
                let fa = fcvtl_lo(u16x8);
                let fb = fcvtl_hi(u16x8);
                let va = vsubq_f32(fa, max_v);
                let vb = vsubq_f32(fb, max_v);
                let (ea, eb) = exp_f32x8(va, vb);
                let lo = fcvtn_f32x4_to_f16x4(ea);
                let hi = fcvtn_f32x4_to_f16x4(eb);
                vst1q_u16(row.as_mut_ptr().add(i), vcombine_u16(lo, hi));
                sum_v = vaddq_f32(vaddq_f32(sum_v, ea), eb);
                i += 8;
            }
            while i + 4 <= c_main {
                let lo = vld1_u16(row.as_ptr().add(i));
                let u16x8 = vcombine_u16(lo, vdup_n_u16(0));
                let fa = fcvtl_lo(u16x8);
                let va = vsubq_f32(fa, max_v);
                let ea = exp_f32x4(va);
                let lo_out = fcvtn_f32x4_to_f16x4(ea);
                vst1_u16(row.as_mut_ptr().add(i), lo_out);
                sum_v = vaddq_f32(sum_v, ea);
                i += 4;
            }
            let mut sum_s = vaddvq_f32(sum_v);
            while i < c_main {
                let f = half::f16::from_bits(row[i]).to_f32();
                let e = (f - max_s).exp();
                row[i] = half::f16::from_f32(e).to_bits();
                sum_s += e;
                i += 1;
            }
            // Dustbin exp folded in last.
            let dustbin_e = (dustbin_f - max_s).exp();
            *d = half::f16::from_f32(dustbin_e).to_bits();
            sum_s += dustbin_e;

            // ── Step 3: scale all exp values by 1/sum (main then dustbin) ──
            let inv_v = vdupq_n_f32(1.0_f32 / sum_s);
            let mut i = 0usize;
            while i + 8 <= c_main {
                let u16x8 = vld1q_u16(row.as_ptr().add(i));
                let fa = fcvtl_lo(u16x8);
                let fb = fcvtl_hi(u16x8);
                let ra = vmulq_f32(fa, inv_v);
                let rb = vmulq_f32(fb, inv_v);
                let lo = fcvtn_f32x4_to_f16x4(ra);
                let hi = fcvtn_f32x4_to_f16x4(rb);
                vst1q_u16(row.as_mut_ptr().add(i), vcombine_u16(lo, hi));
                i += 8;
            }
            while i + 4 <= c_main {
                let lo = vld1_u16(row.as_ptr().add(i));
                let u16x8 = vcombine_u16(lo, vdup_n_u16(0));
                let fa = fcvtl_lo(u16x8);
                let ra = vmulq_f32(fa, inv_v);
                let lo_out = fcvtn_f32x4_to_f16x4(ra);
                vst1_u16(row.as_mut_ptr().add(i), lo_out);
                i += 4;
            }
            while i < c_main {
                let f = half::f16::from_bits(row[i]).to_f32();
                row[i] = half::f16::from_f32(f / sum_s).to_bits();
                i += 1;
            }
            let df = half::f16::from_bits(*d).to_f32();
            *d = half::f16::from_f32(df / sum_s).to_bits();
        }
    }
}

/// Rayon-parallel sidecar softmax: 64-pixel chunks of `main` paired with the
/// matching 64 dustbin values. See [`channel_softmax_neon_f16_sidecar_kernel`].
#[cfg(target_arch = "aarch64")]
pub fn channel_softmax_neon_f16_sidecar_par(
    main: &mut [u16],
    dustbin: &mut [u16],
    h: usize,
    w: usize,
    c_main: usize,
) {
    use rayon::prelude::*;
    let m = h * w;
    debug_assert_eq!(main.len(), m * c_main);
    debug_assert_eq!(dustbin.len(), m);
    debug_assert!(
        crate::cpu_features::cpu_features().has_fp16,
        "channel_softmax_neon_f16_sidecar_par called on a CPU without fp16 support"
    );
    main.par_chunks_mut(c_main * 64)
        .zip(dustbin.par_chunks_mut(64))
        .for_each(|(block, dblock)| {
            // SAFETY: fp16 checked above; each (block, dblock) pair is independent.
            unsafe { channel_softmax_neon_f16_sidecar_kernel(block, dblock, c_main) };
        });
}

/// Per-pixel fused sidecar-softmax + pixel-shuffle (c_main = 64), NEON.
///
/// Runs the same sidecar softmax math as [`channel_softmax_neon_f16_sidecar_kernel`]
/// (max over the 64 main channels then dustbin; exp-sum in the same order), writes
/// the normalized dustbin back, and scatters the 64 normalized values — through the
/// identical `FCVTN(x*inv)` f16 round-trip the standalone softmax applies, then
/// `FCVTL`-widened — into the f32 output block. `main_row` is left holding the
/// post-exp (un-normalized) f16 values.
///
/// `out_base` points at the top-left of this pixel's 8×8 output block; rows are
/// `w_out` apart.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,fp16")]
unsafe fn softmax_shuffle_px_f16_neon(
    row: &mut [u16], // 64 main channels for this pixel
    d: &mut u16,     // dustbin for this pixel
    out_base: *mut f32,
    w_out: usize,
) {
    use crate::ops::neon_asm_f16::{fcvtl_hi, fcvtl_lo, fcvtn_f32x4_to_f16x4};
    use std::arch::aarch64::*;
    const C_MAIN: usize = 64;

    unsafe {
        // ── Step 1: max across the 64 main channels, then fold dustbin last. ──
        let mut max_v = vdupq_n_f32(f32::NEG_INFINITY);
        let mut i = 0usize;
        while i + 8 <= C_MAIN {
            let u16x8 = vld1q_u16(row.as_ptr().add(i));
            let fa = fcvtl_lo(u16x8);
            let fb = fcvtl_hi(u16x8);
            max_v = vmaxq_f32(max_v, vmaxq_f32(fa, fb));
            i += 8;
        }
        let mut max_s = vmaxvq_f32(max_v);
        let dustbin_f = half::f16::from_bits(*d).to_f32();
        if dustbin_f > max_s {
            max_s = dustbin_f;
        }
        let max_v = vdupq_n_f32(max_s);

        // ── Step 2: exp(x - max), store back to row, accumulate sum (main then
        //    dustbin), matching the standalone sidecar order exactly. ──
        let mut sum_v = vdupq_n_f32(0.0_f32);
        let mut i = 0usize;
        while i + 8 <= C_MAIN {
            let u16x8 = vld1q_u16(row.as_ptr().add(i));
            let fa = fcvtl_lo(u16x8);
            let fb = fcvtl_hi(u16x8);
            let va = vsubq_f32(fa, max_v);
            let vb = vsubq_f32(fb, max_v);
            let (ea, eb) = exp_f32x8(va, vb);
            let lo = fcvtn_f32x4_to_f16x4(ea);
            let hi = fcvtn_f32x4_to_f16x4(eb);
            vst1q_u16(row.as_mut_ptr().add(i), vcombine_u16(lo, hi));
            sum_v = vaddq_f32(vaddq_f32(sum_v, ea), eb);
            i += 8;
        }
        let mut sum_s = vaddvq_f32(sum_v);
        let dustbin_e = (dustbin_f - max_s).exp();
        *d = half::f16::from_f32(dustbin_e).to_bits();
        sum_s += dustbin_e;

        // ── Step 3 (fused with shuffle): scale each main value by 1/sum through
        //    the f16 round-trip, then FCVTL-widen straight into the f32 output. ──
        let inv_v = vdupq_n_f32(1.0_f32 / sum_s);
        let mut i = 0usize;
        while i + 8 <= C_MAIN {
            let u16x8 = vld1q_u16(row.as_ptr().add(i));
            let fa = fcvtl_lo(u16x8);
            let fb = fcvtl_hi(u16x8);
            let ra = vmulq_f32(fa, inv_v);
            let rb = vmulq_f32(fb, inv_v);
            // f16 round-trip (FCVTN) to match the standalone softmax output bits,
            // then widen (FCVTL) for the f32 heatmap.
            let lo = fcvtn_f32x4_to_f16x4(ra);
            let hi = fcvtn_f32x4_to_f16x4(rb);
            let h8x = vcombine_u16(lo, hi);
            let out_lo = fcvtl_lo(h8x);
            let out_hi = fcvtl_hi(h8x);
            // i == kh*8 → output row kh of this pixel's 8×8 block.
            let out_row = out_base.add((i / 8) * w_out);
            vst1q_f32(out_row, out_lo);
            vst1q_f32(out_row.add(4), out_hi);
            i += 8;
        }

        // Dustbin normalized + written back (parity with sidecar softmax).
        let df = half::f16::from_bits(*d).to_f32();
        *d = half::f16::from_f32(df / sum_s).to_bits();
    }
}

/// Fused sidecar-softmax + pixel-shuffle (c_main = 64), NEON + Rayon.
///
/// Replaces the `channel_softmax_neon_f16_sidecar_par` + `pixel_shuffle_8_f16_neon`
/// sequence with one dispatch. Bit-identical to running those two ops in sequence
/// (same softmax order, same `FCVTN(x*inv)` round-trip before widening).
#[cfg(target_arch = "aarch64")]
pub fn channel_softmax_pixel_shuffle_f16_sidecar_neon(
    main: &mut [u16],
    dustbin: &mut [u16],
    k1h_out: &mut [f32],
    h8: usize,
    w8: usize,
) {
    use rayon::prelude::*;
    const C_MAIN: usize = 64;
    let w_out = w8 * 8;
    debug_assert_eq!(main.len(), h8 * w8 * C_MAIN);
    debug_assert_eq!(dustbin.len(), h8 * w8);
    debug_assert_eq!(k1h_out.len(), h8 * 8 * w_out);
    debug_assert!(
        crate::cpu_features::cpu_features().has_fp16,
        "channel_softmax_pixel_shuffle_f16_sidecar_neon called on a CPU without fp16 support"
    );

    k1h_out
        .par_chunks_mut(8 * w_out)
        .zip(main.par_chunks_mut(w8 * C_MAIN))
        .zip(dustbin.par_chunks_mut(w8))
        .for_each(|((super_row, main_row), dust_row)| {
            for w in 0..w8 {
                let row = &mut main_row[w * C_MAIN..w * C_MAIN + C_MAIN];
                let d = &mut dust_row[w];
                // out_base = top-left of this pixel's 8×8 block.
                let out_base = unsafe { super_row.as_mut_ptr().add(w * 8) };
                // SAFETY: fp16 checked above; each (row, d, block) is disjoint.
                unsafe { softmax_shuffle_px_f16_neon(row, d, out_base, w_out) };
            }
        });
}

/// Channel-wise L2 normalization for f16 storage — NEON + Rayon.
///
/// For each pixel's `c`-channel vector: compute sum-of-squares using NEON
/// FCVTL widening + VFMA, then scale by 1/sqrt(sum + ε) using vectorized MUL.
/// Dispatched by `ops::l2_normalize_channel_f16` on aarch64 when fp16 is present.
#[cfg(target_arch = "aarch64")]
pub fn l2_normalize_channel_f16_neon(buf: &mut [u16], h: usize, w: usize, c: usize) {
    use rayon::prelude::*;
    debug_assert_eq!(buf.len(), h * w * c);
    debug_assert!(
        crate::cpu_features::cpu_features().has_fp16,
        "l2_normalize_channel_f16_neon called on a CPU without fp16 support"
    );
    buf.par_chunks_mut(c).with_min_len(256).for_each(|chunk| {
        // SAFETY: fp16 checked above; each chunk is independent.
        unsafe { l2_norm_row_f16_neon(chunk) };
    });
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,fp16")]
unsafe fn l2_norm_row_f16_neon(chunk: &mut [u16]) {
    use crate::ops::neon_asm_f16::{fcvtl_hi, fcvtl_lo, fcvtn_f32x4_to_f16x4};
    use std::arch::aarch64::*;
    let c = chunk.len();
    let ptr = chunk.as_mut_ptr();

    // Pass 1: NEON sum of squares
    let mut vsum = vdupq_n_f32(0.0_f32);
    let mut i = 0;
    while i + 8 <= c {
        let vi = vld1q_u16(ptr.add(i));
        let fa = fcvtl_lo(vi);
        let fb = fcvtl_hi(vi);
        vsum = vfmaq_f32(vsum, fa, fa);
        vsum = vfmaq_f32(vsum, fb, fb);
        i += 8;
    }
    while i + 4 <= c {
        let lo = vld1_u16(ptr.add(i));
        let vi = vcombine_u16(lo, vdup_n_u16(0));
        let fa = fcvtl_lo(vi);
        vsum = vfmaq_f32(vsum, fa, fa);
        i += 4;
    }
    let mut sum_sq = vaddvq_f32(vsum);
    while i < c {
        let f = half::f16::from_bits(*ptr.add(i)).to_f32();
        sum_sq += f * f;
        i += 1;
    }

    let inv_norm = (sum_sq + 1e-12_f32).sqrt().recip();
    let vinv = vdupq_n_f32(inv_norm);

    // Pass 2: NEON normalize
    i = 0;
    while i + 8 <= c {
        let vi = vld1q_u16(ptr.add(i));
        let fa = fcvtl_lo(vi);
        let fb = fcvtl_hi(vi);
        let ra = vmulq_f32(fa, vinv);
        let rb = vmulq_f32(fb, vinv);
        let lo = fcvtn_f32x4_to_f16x4(ra);
        let hi = fcvtn_f32x4_to_f16x4(rb);
        vst1q_u16(ptr.add(i), vcombine_u16(lo, hi));
        i += 8;
    }
    while i + 4 <= c {
        let lo_in = vld1_u16(ptr.add(i));
        let vi = vcombine_u16(lo_in, vdup_n_u16(0));
        let fa = fcvtl_lo(vi);
        let ra = vmulq_f32(fa, vinv);
        let lo = fcvtn_f32x4_to_f16x4(ra);
        vst1_u16(ptr.add(i), lo);
        i += 4;
    }
    while i < c {
        let f = half::f16::from_bits(*ptr.add(i)).to_f32() * inv_norm;
        *ptr.add(i) = half::f16::from_f32(f).to_bits();
        i += 1;
    }
}

// ── NMS via MaxPool 5×5 equality (NEON) ──────────────────────────────────────

/// Compute the 5×5 sliding-window maximum (stride 1, padding 2 / clamp-border)
/// for a single-channel `(h, w)` image. Writes the result into `out`.
///
/// Uses a separable two-pass approach:
///   1. Horizontal 1×5 max → `hmax` scratch  (NEON vmaxq_f32, 4 px/cycle)
///   2. Vertical   5×1 max → `out`           (sequential scalar)
///
/// Both passes are sequential — at 60×80 (1/8 of 480×640) the compute is
/// ~0.02ms and Rayon dispatch overhead would dominate.
pub(crate) fn maxpool_5x5_neon(input: &[f32], out: &mut [f32], h: usize, w: usize) {
    debug_assert_eq!(input.len(), h * w);
    debug_assert_eq!(out.len(), h * w);

    // ── Pass 1: horizontal max into hmax ─────────────────────────────────────
    // hmax[y * w + x] = max(input[y][x-2 .. x+2])  (boundary → clamp).
    let mut hmax = vec![0.0f32; h * w];
    // Pre-allocate padded row once; reused for every row.
    let mut padded = vec![f32::NEG_INFINITY; w + 4];
    unsafe {
        use std::arch::aarch64::*;
        for y in 0..h {
            let row = &input[y * w..(y + 1) * w];
            let hrow = &mut hmax[y * w..(y + 1) * w];

            padded[2..w + 2].copy_from_slice(row);
            // Clamp-border: replicate edge values.
            padded[0] = row[0];
            padded[1] = row[0];
            padded[w + 2] = row[w - 1];
            padded[w + 3] = row[w - 1];

            let p = padded.as_ptr();
            let o = hrow.as_mut_ptr();

            let mut x = 0usize;
            // Process 4 output pixels at a time.
            while x + 4 <= w {
                // For output pixel x, 5 taps are padded[x], padded[x+1], ..., padded[x+4].
                // For output pixel x+1, taps are padded[x+1] .. padded[x+5], etc.
                // Load 5 float32x4_t vectors at offsets x, x+1, x+2, x+3, x+4.
                let v0 = vld1q_f32(p.add(x));
                let v1 = vld1q_f32(p.add(x + 1));
                let v2 = vld1q_f32(p.add(x + 2));
                let v3 = vld1q_f32(p.add(x + 3));
                let v4 = vld1q_f32(p.add(x + 4));
                let m01 = vmaxq_f32(v0, v1);
                let m23 = vmaxq_f32(v2, v3);
                let m = vmaxq_f32(vmaxq_f32(m01, m23), v4);
                vst1q_f32(o.add(x), m);
                x += 4;
            }
            // Scalar tail for remaining pixels.
            while x < w {
                let mut m = f32::NEG_INFINITY;
                for k in 0..5usize {
                    let v = padded[x + k];
                    if v > m {
                        m = v;
                    }
                }
                hrow[x] = m;
                x += 1;
            }
        }
    }

    // ── Pass 2: vertical max from hmax into out (sequential scalar) ──────────
    // out[y * w + x] = max(hmax[(y-2..y+2)][x])
    for y in 0..h {
        let y0 = if y >= 2 { y - 2 } else { 0 };
        let y1 = (y + 3).min(h); // exclusive, covers y-2..y+2 inclusive
        for x in 0..w {
            let mut m = f32::NEG_INFINITY;
            for vy in y0..y1 {
                let v = hmax[vy * w + x];
                if v > m {
                    m = v;
                }
            }
            out[y * w + x] = m;
        }
    }
}

/// NMS via MaxPool 5×5 equality check — NEON-accelerated.
///
/// For each pixel: keep if `value > threshold` AND `value == max over 5×5 window`.
/// Uses [`maxpool_5x5_neon`] for the window max, then a NEON equality pass.
/// Returns `(value, flat_index)` for each surviving local maximum, in an
/// arbitrary order (same as the scalar variant).
pub fn nms_maxpool_5x5_equality_neon(
    input: &[f32],
    h: usize,
    w: usize,
    threshold: f32,
) -> Vec<(f32, usize)> {
    debug_assert_eq!(input.len(), h * w);

    // Step 1: compute 5×5 sliding max.
    let mut pool = vec![0.0f32; h * w];
    maxpool_5x5_neon(input, &mut pool, h, w);

    // Step 2: collect pixels where input[i] > threshold AND input[i] == pool[i].
    // Sequential scan — 60×80 = 4800 pixels, sequential is ~0.01ms.
    let mut result = Vec::new();
    for (i, (&v, &p)) in input.iter().zip(pool.iter()).enumerate() {
        if v > threshold && v == p {
            result.push((v, i));
        }
    }
    result
}

/// Specialized 3×3 conv for `c_in = 1, c_out = 4` (block1.0, full resolution).
///
/// The generic NEON conv vectorizes the channel reduction, so `c_in = 1`
/// falls back to scalar — at 480×640 that single layer costs ~0.65 ms.
/// Here we exploit `c_out = 4` + NHWC instead: each output pixel is exactly
/// one `float32x4` (its 4 channels), so a pixel is 9 `vld1q_dup_f32` +
/// `vfmaq_f32` against pre-gathered per-tap weight vectors.
///
/// Bit-exactness vs [`super::scalar::conv3x3_relu_nhwc`] at `c_in = 1`:
/// the scalar fallback accumulates per channel with sequential
/// `f32::mul_add` over taps in `kh`-major order, skipping out-of-range
/// taps; each NEON lane here performs the identical FMA sequence in the
/// identical order, then `+ bias` and the activation — so every lane is
/// IEEE-identical to the scalar result (asserted exactly in parity tests).
#[cfg(target_arch = "aarch64")]
pub fn conv3x3_c1_co4_neon(
    input: &[f32],
    weights: &[f32], // [c_out=4, 3, 3, c_in=1] row-major = [co*9 + kh*3 + kw]
    bias: &[f32],
    h: usize,
    w: usize,
    activation: Activation,
    output: &mut [f32],
) {
    debug_assert_eq!(input.len(), h * w);
    debug_assert_eq!(weights.len(), 4 * 9);
    debug_assert_eq!(bias.len(), 4);
    debug_assert_eq!(output.len(), h * w * 4);
    debug_assert!(
        matches!(activation, Activation::Relu | Activation::Identity),
        "conv3x3_c1_co4_neon: unsupported activation"
    );

    // Gather weights into 9 per-tap vectors: wv[tap][lane co] = weights[co*9 + tap].
    let mut wv = [[0.0f32; 4]; 9];
    for (tap, wtap) in wv.iter_mut().enumerate() {
        for (co, lane) in wtap.iter_mut().enumerate() {
            *lane = weights[co * 9 + tap];
        }
    }
    let relu = matches!(activation, Activation::Relu);
    let in_addr = input.as_ptr() as usize;
    debug_assert!(w >= 2 && h >= 1);

    /// One pixel, runtime kw bounds (row edges only). Tap order: kh-major,
    /// kw-minor, borders skipped — identical to the scalar fallback.
    #[inline(always)]
    unsafe fn px1<const KH_LO: usize, const KH_HI: usize>(
        inp: *const f32,
        oh: usize,
        w: usize,
        ow: usize,
        kw_lo: usize,
        kw_hi: usize,
        wvv: &[float32x4_t; 9],
    ) -> float32x4_t {
        let mut acc = vdupq_n_f32(0.0);
        for kh in KH_LO..KH_HI {
            let base = inp.add((oh + kh - 1) * w + ow);
            for kw in kw_lo..kw_hi {
                acc = vfmaq_f32(acc, vld1q_dup_f32(base.add(kw).sub(1)), wvv[kh * 3 + kw]);
            }
        }
        acc
    }

    /// Four interior pixels: fully const-unrolled taps (weight vectors stay in
    /// registers) and 4 independent accumulator chains to hide the 4-cycle FMA
    /// latency. Each pixel's own chain keeps the exact kh-major tap order, so
    /// every lane remains bit-identical to the scalar fallback.
    #[inline(always)]
    unsafe fn px4<const KH_LO: usize, const KH_HI: usize>(
        inp: *const f32,
        oh: usize,
        w: usize,
        ow: usize,
        wvv: &[float32x4_t; 9],
    ) -> [float32x4_t; 4] {
        let mut a = [vdupq_n_f32(0.0); 4];
        for kh in KH_LO..KH_HI {
            let row = inp.add((oh + kh - 1) * w + ow - 1);
            for kw in 0..3usize {
                let wt = wvv[kh * 3 + kw];
                for (p, acc) in a.iter_mut().enumerate() {
                    *acc = vfmaq_f32(*acc, vld1q_dup_f32(row.add(kw + p)), wt);
                }
            }
        }
        a
    }

    #[inline(always)]
    unsafe fn store_px(out: *mut f32, acc: float32x4_t, bv: float32x4_t, relu: bool) {
        let mut r = vaddq_f32(acc, bv);
        if relu {
            r = vmaxq_f32(r, vdupq_n_f32(0.0));
        }
        vst1q_f32(out, r);
    }

    /// Whole row with compile-time kh bounds (so `wvv[kh*3+kw]` const-folds).
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn row_body<const KH_LO: usize, const KH_HI: usize>(
        inp: *const f32,
        oh: usize,
        w: usize,
        wvv: &[float32x4_t; 9],
        bv: float32x4_t,
        relu: bool,
        out: *mut f32,
    ) {
        // Left edge (ow = 0): skip kw = 0.
        store_px(out, px1::<KH_LO, KH_HI>(inp, oh, w, 0, 1, 3, wvv), bv, relu);
        // Interior 1..w-1: 4-pixel blocks, then a 1-pixel tail.
        let mut ow = 1usize;
        while ow + 4 <= w - 1 {
            let a = px4::<KH_LO, KH_HI>(inp, oh, w, ow, wvv);
            for (p, &acc) in a.iter().enumerate() {
                store_px(out.add((ow + p) * 4), acc, bv, relu);
            }
            ow += 4;
        }
        while ow < w - 1 {
            store_px(
                out.add(ow * 4),
                px1::<KH_LO, KH_HI>(inp, oh, w, ow, 0, 3, wvv),
                bv,
                relu,
            );
            ow += 1;
        }
        // Right edge (ow = w-1): skip kw = 2.
        store_px(
            out.add((w - 1) * 4),
            px1::<KH_LO, KH_HI>(inp, oh, w, w - 1, 0, 2, wvv),
            bv,
            relu,
        );
    }

    output
        .par_chunks_mut(w * 4)
        .enumerate()
        .for_each(|(oh, row_out)| unsafe {
            let inp = in_addr as *const f32;
            let wvv: [float32x4_t; 9] = [
                vld1q_f32(wv[0].as_ptr()),
                vld1q_f32(wv[1].as_ptr()),
                vld1q_f32(wv[2].as_ptr()),
                vld1q_f32(wv[3].as_ptr()),
                vld1q_f32(wv[4].as_ptr()),
                vld1q_f32(wv[5].as_ptr()),
                vld1q_f32(wv[6].as_ptr()),
                vld1q_f32(wv[7].as_ptr()),
                vld1q_f32(wv[8].as_ptr()),
            ];
            let bv = vld1q_f32(bias.as_ptr());
            let out = row_out.as_mut_ptr();
            match (oh == 0, oh + 1 == h) {
                (false, false) => row_body::<0, 3>(inp, oh, w, &wvv, bv, relu, out),
                (true, false) => row_body::<1, 3>(inp, oh, w, &wvv, bv, relu, out),
                (false, true) => row_body::<0, 2>(inp, oh, w, &wvv, bv, relu, out),
                (true, true) => row_body::<1, 2>(inp, oh, w, &wvv, bv, relu, out),
            }
        });
}

/// Pack block1.1 weights `[c_out=8, 3, 3, c_in=4]` (row-major
/// `weights[((co*3+kh)*3+kw)*4 + ci]`) into the `vfmaq_laneq` layout
/// `pk[((tap*4 + ci)*2 + cob)*4 + co_lane]`, i.e. for each (tap, ci) two
/// `float32x4` vectors holding the 8 output-channel weights split as
/// `co0..3` (cob=0) and `co4..7` (cob=1). 9·4·2·4 = 288 f32.
#[cfg(target_arch = "aarch64")]
pub fn pack_b1_1_laneq(weights: &[f32]) -> Vec<f32> {
    debug_assert_eq!(weights.len(), 8 * 9 * 4);
    let mut pk = vec![0.0f32; 9 * 4 * 2 * 4];
    for tap in 0..9 {
        for ci in 0..4 {
            for cob in 0..2 {
                for lane in 0..4 {
                    let co = cob * 4 + lane;
                    pk[((tap * 4 + ci) * 2 + cob) * 4 + lane] = weights[(co * 9 + tap) * 4 + ci];
                }
            }
        }
    }
    pk
}

/// Specialized 3×3 **stride-2** conv for `c_in = 4, c_out = 8` (block1.1).
///
/// Each input pixel is one `float32x4` (its 4 channels); each output pixel is
/// two `float32x4` (its 8 channels, split `co0..3` / `co4..7`). The inner
/// product over the 4 input channels uses `vfmaq_laneq_f32`: one input
/// `vld1q` broadcast-per-lane against the pre-packed per-(tap,ci) weight
/// vectors (see [`pack_b1_1_laneq`]). Four output pixels are processed per
/// interior iteration (8 independent accumulator chains) to hide FMA latency.
///
/// Reduction-order note: unlike the scalar fallback (which sums the 4 input
/// channels by horizontally adding a 4-lane accumulator, lanes = ci), this
/// kernel accumulates the 4 ci terms sequentially into the same lane
/// (lanes = co). The two orders differ only by f32 rounding, bounded well
/// inside `TOL_CONV3X3_PENDING = 5e-5` (asserted in the parity test).
///
/// Requires even `h_in`/`w_in` (true for XFeat: 480×640) so the only
/// out-of-range taps are the low borders (`kh=0` at `oh=0`, `kw=0` at `ow=0`).
#[cfg(target_arch = "aarch64")]
pub fn conv3x3_s2_c4_co8_neon(
    input: &[f32],
    packed: &[f32], // pack_b1_1_laneq layout, 288 f32
    bias: &[f32],
    h_in: usize,
    w_in: usize,
    activation: Activation,
    output: &mut [f32],
) {
    let h_out = h_in / 2;
    let w_out = w_in / 2;
    debug_assert_eq!(input.len(), h_in * w_in * 4);
    debug_assert_eq!(packed.len(), 9 * 4 * 2 * 4);
    debug_assert_eq!(bias.len(), 8);
    debug_assert_eq!(output.len(), h_out * w_out * 8);
    debug_assert!(
        matches!(activation, Activation::Relu | Activation::Identity),
        "conv3x3_s2_c4_co8_neon: unsupported activation"
    );
    debug_assert!(h_in % 2 == 0 && w_in % 2 == 0 && w_out >= 2);

    let relu = matches!(activation, Activation::Relu);
    let in_addr = input.as_ptr() as usize;
    let pk_addr = packed.as_ptr() as usize;

    /// Accumulate one (tap, ci) contribution into (acc_lo, acc_hi) for one px.
    #[inline(always)]
    unsafe fn fma_tap(
        acc: &mut (float32x4_t, float32x4_t),
        pk: *const f32,
        tap: usize,
        in_vec: float32x4_t,
    ) {
        // ci=0..3 sequentially into the same accumulator lane.
        let b = pk.add(tap * 4 * 2 * 4);
        acc.0 = vfmaq_laneq_f32::<0>(acc.0, vld1q_f32(b), in_vec);
        acc.1 = vfmaq_laneq_f32::<0>(acc.1, vld1q_f32(b.add(4)), in_vec);
        acc.0 = vfmaq_laneq_f32::<1>(acc.0, vld1q_f32(b.add(8)), in_vec);
        acc.1 = vfmaq_laneq_f32::<1>(acc.1, vld1q_f32(b.add(12)), in_vec);
        acc.0 = vfmaq_laneq_f32::<2>(acc.0, vld1q_f32(b.add(16)), in_vec);
        acc.1 = vfmaq_laneq_f32::<2>(acc.1, vld1q_f32(b.add(20)), in_vec);
        acc.0 = vfmaq_laneq_f32::<3>(acc.0, vld1q_f32(b.add(24)), in_vec);
        acc.1 = vfmaq_laneq_f32::<3>(acc.1, vld1q_f32(b.add(28)), in_vec);
    }

    /// One output pixel with runtime kw bounds (column edges only).
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn px1<const KH_LO: usize, const KH_HI: usize>(
        inp: *const f32,
        pk: *const f32,
        oh: usize,
        ow: usize,
        w_in: usize,
        kw_lo: usize,
        kw_hi: usize,
    ) -> (float32x4_t, float32x4_t) {
        let mut acc = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
        let ih_base = oh * 2; // ih = ih_base + kh - 1
        let iw_base = ow * 2; // iw = iw_base + kw - 1
        for kh in KH_LO..KH_HI {
            let row = inp.add(((ih_base + kh - 1) * w_in + iw_base) * 4);
            for kw in kw_lo..kw_hi {
                let in_vec = vld1q_f32(row.add((kw - 1) * 4));
                fma_tap(&mut acc, pk, kh * 3 + kw, in_vec);
            }
        }
        acc
    }

    /// Four interior output pixels (ow..ow+3), fully unrolled taps.
    #[inline(always)]
    unsafe fn px4<const KH_LO: usize, const KH_HI: usize>(
        inp: *const f32,
        pk: *const f32,
        oh: usize,
        ow: usize,
        w_in: usize,
    ) -> [(float32x4_t, float32x4_t); 4] {
        let z = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
        let mut a = [z, z, z, z];
        let ih_base = oh * 2;
        for kh in KH_LO..KH_HI {
            // For pixel p, iw_base = (ow+p)*2; tap kw uses iw = iw_base+kw-1.
            // Row pointer at the first pixel's leftmost tap column (kw=0).
            let row = inp.add(((ih_base + kh - 1) * w_in + ow * 2 - 1) * 4);
            for kw in 0..3usize {
                let tap = kh * 3 + kw;
                for (p, acc) in a.iter_mut().enumerate() {
                    // pixel p, tap kw → input column offset = p*2 + kw.
                    let in_vec = vld1q_f32(row.add((p * 2 + kw) * 4));
                    fma_tap(acc, pk, tap, in_vec);
                }
            }
        }
        a
    }

    #[inline(always)]
    unsafe fn store_px(
        out: *mut f32,
        acc: (float32x4_t, float32x4_t),
        blo: float32x4_t,
        bhi: float32x4_t,
        relu: bool,
    ) {
        let mut lo = vaddq_f32(acc.0, blo);
        let mut hi = vaddq_f32(acc.1, bhi);
        if relu {
            let z = vdupq_n_f32(0.0);
            lo = vmaxq_f32(lo, z);
            hi = vmaxq_f32(hi, z);
        }
        vst1q_f32(out, lo);
        vst1q_f32(out.add(4), hi);
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn row_body<const KH_LO: usize, const KH_HI: usize>(
        inp: *const f32,
        pk: *const f32,
        oh: usize,
        w_in: usize,
        w_out: usize,
        blo: float32x4_t,
        bhi: float32x4_t,
        relu: bool,
        out: *mut f32,
    ) {
        // Left edge (ow = 0): kw = 0 reads iw = -1 → skip.
        store_px(
            out,
            px1::<KH_LO, KH_HI>(inp, pk, oh, 0, w_in, 1, 3),
            blo,
            bhi,
            relu,
        );
        // Interior 1..w_out-1: 4-pixel blocks then 1-pixel tail.
        let mut ow = 1usize;
        while ow + 4 <= w_out - 1 {
            let a = px4::<KH_LO, KH_HI>(inp, pk, oh, ow, w_in);
            for (p, &acc) in a.iter().enumerate() {
                store_px(out.add((ow + p) * 8), acc, blo, bhi, relu);
            }
            ow += 4;
        }
        while ow < w_out - 1 {
            store_px(
                out.add(ow * 8),
                px1::<KH_LO, KH_HI>(inp, pk, oh, ow, w_in, 0, 3),
                blo,
                bhi,
                relu,
            );
            ow += 1;
        }
        // Right edge (ow = w_out-1): iw_max = 2*(w_out-1)+1 = w_in-1 (even w_in)
        // → kw = 2 in range; full 0..3.
        store_px(
            out.add((w_out - 1) * 8),
            px1::<KH_LO, KH_HI>(inp, pk, oh, w_out - 1, w_in, 0, 3),
            blo,
            bhi,
            relu,
        );
    }

    output
        .par_chunks_mut(w_out * 8)
        .enumerate()
        .for_each(|(oh, row_out)| unsafe {
            let inp = in_addr as *const f32;
            let pk = pk_addr as *const f32;
            let blo = vld1q_f32(bias.as_ptr());
            let bhi = vld1q_f32(bias.as_ptr().add(4));
            let out = row_out.as_mut_ptr();
            // oh=0 → kh=0 reads ih=-1, skip. High border never overflows for
            // even h_in (ih_max = 2*(h_out-1)+1 = h_in-1).
            if oh == 0 {
                row_body::<1, 3>(inp, pk, oh, w_in, w_out, blo, bhi, relu, out);
            } else {
                row_body::<0, 3>(inp, pk, oh, w_in, w_out, blo, bhi, relu, out);
            }
        });
}

/// Pack block1.3 weights `[c_out=24, 3, 3, c_in=8]` (row-major
/// `weights[((co*3+kh)*3+kw)*8 + ci]`) into the `vfmaq_laneq` layout
/// `pk[((tap*8 + ci)*6 + cq)*4 + co_lane]`: for each (tap, ci) six
/// `float32x4` vectors holding the 24 output-channel weights as quads
/// `cq*4 .. cq*4+3`. 9·8·6·4 = 1728 f32.
#[cfg(target_arch = "aarch64")]
pub fn pack_b1_3_laneq(weights: &[f32]) -> Vec<f32> {
    debug_assert_eq!(weights.len(), 24 * 9 * 8);
    let mut pk = vec![0.0f32; 9 * 8 * 6 * 4];
    for tap in 0..9 {
        for ci in 0..8 {
            for cq in 0..6 {
                for lane in 0..4 {
                    let co = cq * 4 + lane;
                    pk[((tap * 8 + ci) * 6 + cq) * 4 + lane] = weights[(co * 9 + tap) * 8 + ci];
                }
            }
        }
    }
    pk
}

/// Specialized 3×3 **stride-2** conv for `c_in = 8, c_out = 24` (block1.3).
///
/// Each input pixel is two `float32x4` (8 channels); each output pixel is six
/// `float32x4` (24 channels). The 8-channel reduction uses `vfmaq_laneq_f32`
/// (one input `vld1q` broadcast-per-lane against the pre-packed per-(tap,ci)
/// weight quads — see [`pack_b1_3_laneq`]). The six output quads form six
/// independent accumulator chains within a pixel, hiding the 4-cycle FMA
/// latency without extra pixel blocking.
///
/// Reduction-order note: the 8 ci terms accumulate sequentially per lane
/// (lanes = co) instead of the scalar fallback's horizontal 4-lane add
/// (lanes = ci), so parity is tolerance- (≤ 5e-5), not bit-, exact.
///
/// Requires even `h_in`/`w_in` (true for XFeat: 240×320) so only the low
/// borders (`kh=0` at `oh=0`, `kw=0` at `ow=0`) are ever out of range.
#[cfg(target_arch = "aarch64")]
pub fn conv3x3_s2_c8_co24_neon(
    input: &[f32],
    packed: &[f32], // pack_b1_3_laneq layout, 1728 f32
    bias: &[f32],
    h_in: usize,
    w_in: usize,
    activation: Activation,
    output: &mut [f32],
) {
    let h_out = h_in / 2;
    let w_out = w_in / 2;
    debug_assert_eq!(input.len(), h_in * w_in * 8);
    debug_assert_eq!(packed.len(), 9 * 8 * 6 * 4);
    debug_assert_eq!(bias.len(), 24);
    debug_assert_eq!(output.len(), h_out * w_out * 24);
    debug_assert!(
        matches!(activation, Activation::Relu | Activation::Identity),
        "conv3x3_s2_c8_co24_neon: unsupported activation"
    );
    debug_assert!(h_in % 2 == 0 && w_in % 2 == 0 && w_out >= 2);

    let relu = matches!(activation, Activation::Relu);
    let in_addr = input.as_ptr() as usize;
    let pk_addr = packed.as_ptr() as usize;

    /// Accumulate one (tap) contribution over all 8 ci into the 6 quads.
    #[inline(always)]
    unsafe fn fma_tap(
        acc: &mut [float32x4_t; 6],
        pk: *const f32,
        tap: usize,
        in_lo: float32x4_t,
        in_hi: float32x4_t,
    ) {
        // Per (tap, ci): 6 weight quads at pk + ((tap*8+ci)*6)*4.
        macro_rules! ci_block {
            ($ci:expr, $inv:expr, $lane:expr) => {{
                let b = pk.add(((tap * 8 + $ci) * 6) * 4);
                acc[0] = vfmaq_laneq_f32::<$lane>(acc[0], vld1q_f32(b), $inv);
                acc[1] = vfmaq_laneq_f32::<$lane>(acc[1], vld1q_f32(b.add(4)), $inv);
                acc[2] = vfmaq_laneq_f32::<$lane>(acc[2], vld1q_f32(b.add(8)), $inv);
                acc[3] = vfmaq_laneq_f32::<$lane>(acc[3], vld1q_f32(b.add(12)), $inv);
                acc[4] = vfmaq_laneq_f32::<$lane>(acc[4], vld1q_f32(b.add(16)), $inv);
                acc[5] = vfmaq_laneq_f32::<$lane>(acc[5], vld1q_f32(b.add(20)), $inv);
            }};
        }
        ci_block!(0, in_lo, 0);
        ci_block!(1, in_lo, 1);
        ci_block!(2, in_lo, 2);
        ci_block!(3, in_lo, 3);
        ci_block!(4, in_hi, 0);
        ci_block!(5, in_hi, 1);
        ci_block!(6, in_hi, 2);
        ci_block!(7, in_hi, 3);
    }

    /// Two output pixels for one (tap): weight quads are loaded once and reused
    /// across both pixels (halves packed-weight read bandwidth) while the 12
    /// accumulators give 12 independent FMA chains.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn fma_tap2(
        a0: &mut [float32x4_t; 6],
        a1: &mut [float32x4_t; 6],
        pk: *const f32,
        tap: usize,
        in0_lo: float32x4_t,
        in0_hi: float32x4_t,
        in1_lo: float32x4_t,
        in1_hi: float32x4_t,
    ) {
        macro_rules! ci_block {
            ($ci:expr, $v0:expr, $v1:expr, $lane:expr) => {{
                let b = pk.add(((tap * 8 + $ci) * 6) * 4);
                let w0 = vld1q_f32(b);
                let w1 = vld1q_f32(b.add(4));
                let w2 = vld1q_f32(b.add(8));
                let w3 = vld1q_f32(b.add(12));
                let w4 = vld1q_f32(b.add(16));
                let w5 = vld1q_f32(b.add(20));
                a0[0] = vfmaq_laneq_f32::<$lane>(a0[0], w0, $v0);
                a1[0] = vfmaq_laneq_f32::<$lane>(a1[0], w0, $v1);
                a0[1] = vfmaq_laneq_f32::<$lane>(a0[1], w1, $v0);
                a1[1] = vfmaq_laneq_f32::<$lane>(a1[1], w1, $v1);
                a0[2] = vfmaq_laneq_f32::<$lane>(a0[2], w2, $v0);
                a1[2] = vfmaq_laneq_f32::<$lane>(a1[2], w2, $v1);
                a0[3] = vfmaq_laneq_f32::<$lane>(a0[3], w3, $v0);
                a1[3] = vfmaq_laneq_f32::<$lane>(a1[3], w3, $v1);
                a0[4] = vfmaq_laneq_f32::<$lane>(a0[4], w4, $v0);
                a1[4] = vfmaq_laneq_f32::<$lane>(a1[4], w4, $v1);
                a0[5] = vfmaq_laneq_f32::<$lane>(a0[5], w5, $v0);
                a1[5] = vfmaq_laneq_f32::<$lane>(a1[5], w5, $v1);
            }};
        }
        ci_block!(0, in0_lo, in1_lo, 0);
        ci_block!(1, in0_lo, in1_lo, 1);
        ci_block!(2, in0_lo, in1_lo, 2);
        ci_block!(3, in0_lo, in1_lo, 3);
        ci_block!(4, in0_hi, in1_hi, 0);
        ci_block!(5, in0_hi, in1_hi, 1);
        ci_block!(6, in0_hi, in1_hi, 2);
        ci_block!(7, in0_hi, in1_hi, 3);
    }

    /// One output pixel with runtime kw bounds (column edges only).
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn px1<const KH_LO: usize, const KH_HI: usize>(
        inp: *const f32,
        pk: *const f32,
        oh: usize,
        ow: usize,
        w_in: usize,
        kw_lo: usize,
        kw_hi: usize,
    ) -> [float32x4_t; 6] {
        let z = vdupq_n_f32(0.0);
        let mut acc = [z; 6];
        let ih_base = oh * 2;
        let iw_base = ow * 2;
        for kh in KH_LO..KH_HI {
            let row = inp.add(((ih_base + kh - 1) * w_in + iw_base) * 8);
            for kw in kw_lo..kw_hi {
                let p = row.add((kw - 1) * 8);
                let in_lo = vld1q_f32(p);
                let in_hi = vld1q_f32(p.add(4));
                fma_tap(&mut acc, pk, kh * 3 + kw, in_lo, in_hi);
            }
        }
        acc
    }

    /// Two interior output pixels (ow, ow+1), full kw range, shared weights.
    #[inline(always)]
    unsafe fn px2<const KH_LO: usize, const KH_HI: usize>(
        inp: *const f32,
        pk: *const f32,
        oh: usize,
        ow: usize,
        w_in: usize,
    ) -> [[float32x4_t; 6]; 2] {
        let z = vdupq_n_f32(0.0);
        let mut a0 = [z; 6];
        let mut a1 = [z; 6];
        let ih_base = oh * 2;
        for kh in KH_LO..KH_HI {
            // Leftmost tap column for pixel ow is iw = ow*2-1.
            let row = inp.add(((ih_base + kh - 1) * w_in + ow * 2 - 1) * 8);
            for kw in 0..3usize {
                let tap = kh * 3 + kw;
                // pixel0 tap kw → col kw; pixel1 tap kw → col kw+2.
                let p0 = row.add(kw * 8);
                let p1 = row.add((kw + 2) * 8);
                fma_tap2(
                    &mut a0,
                    &mut a1,
                    pk,
                    tap,
                    vld1q_f32(p0),
                    vld1q_f32(p0.add(4)),
                    vld1q_f32(p1),
                    vld1q_f32(p1.add(4)),
                );
            }
        }
        [a0, a1]
    }

    #[inline(always)]
    unsafe fn store_px(out: *mut f32, acc: &[float32x4_t; 6], bv: &[float32x4_t; 6], relu: bool) {
        let z = vdupq_n_f32(0.0);
        for q in 0..6 {
            let mut r = vaddq_f32(acc[q], bv[q]);
            if relu {
                r = vmaxq_f32(r, z);
            }
            vst1q_f32(out.add(q * 4), r);
        }
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn row_body<const KH_LO: usize, const KH_HI: usize>(
        inp: *const f32,
        pk: *const f32,
        oh: usize,
        w_in: usize,
        w_out: usize,
        bv: &[float32x4_t; 6],
        relu: bool,
        out: *mut f32,
    ) {
        // Left edge (ow = 0): kw = 0 reads iw = -1 → skip.
        store_px(
            out,
            &px1::<KH_LO, KH_HI>(inp, pk, oh, 0, w_in, 1, 3),
            bv,
            relu,
        );
        // Interior 1..w_out-1 (high border in range for even w_in): 2-pixel
        // blocks (shared weight loads) then a 1-pixel tail.
        let mut ow = 1usize;
        while ow + 2 <= w_out - 1 {
            let a = px2::<KH_LO, KH_HI>(inp, pk, oh, ow, w_in);
            store_px(out.add(ow * 24), &a[0], bv, relu);
            store_px(out.add((ow + 1) * 24), &a[1], bv, relu);
            ow += 2;
        }
        while ow < w_out - 1 {
            store_px(
                out.add(ow * 24),
                &px1::<KH_LO, KH_HI>(inp, pk, oh, ow, w_in, 0, 3),
                bv,
                relu,
            );
            ow += 1;
        }
        // Right edge (ow = w_out-1): iw_max = w_in-1 → full kw 0..3.
        store_px(
            out.add((w_out - 1) * 24),
            &px1::<KH_LO, KH_HI>(inp, pk, oh, w_out - 1, w_in, 0, 3),
            bv,
            relu,
        );
    }

    output
        .par_chunks_mut(w_out * 24)
        .enumerate()
        .for_each(|(oh, row_out)| unsafe {
            let inp = in_addr as *const f32;
            let pk = pk_addr as *const f32;
            let bv: [float32x4_t; 6] = [
                vld1q_f32(bias.as_ptr()),
                vld1q_f32(bias.as_ptr().add(4)),
                vld1q_f32(bias.as_ptr().add(8)),
                vld1q_f32(bias.as_ptr().add(12)),
                vld1q_f32(bias.as_ptr().add(16)),
                vld1q_f32(bias.as_ptr().add(20)),
            ];
            let out = row_out.as_mut_ptr();
            if oh == 0 {
                row_body::<1, 3>(inp, pk, oh, w_in, w_out, &bv, relu, out);
            } else {
                row_body::<0, 3>(inp, pk, oh, w_in, w_out, &bv, relu, out);
            }
        });
}

/// Pack block1.2 weights `[c_out=8, 3, 3, c_in=8]` (row-major
/// `weights[((co*3+kh)*3+kw)*8 + ci]`) into the `vfmaq_laneq` layout
/// `pk[((tap*8 + ci)*2 + cob)*4 + co_lane]`: for each (tap, ci) two
/// `float32x4` vectors holding the 8 output-channel weights split as
/// `co0..3` (cob=0) and `co4..7` (cob=1). 9·8·2·4 = 576 f32.
#[cfg(target_arch = "aarch64")]
pub fn pack_b1_2_laneq(weights: &[f32]) -> Vec<f32> {
    debug_assert_eq!(weights.len(), 8 * 9 * 8);
    let mut pk = vec![0.0f32; 9 * 8 * 2 * 4];
    for tap in 0..9 {
        for ci in 0..8 {
            for cob in 0..2 {
                for lane in 0..4 {
                    let co = cob * 4 + lane;
                    pk[((tap * 8 + ci) * 2 + cob) * 4 + lane] = weights[(co * 9 + tap) * 8 + ci];
                }
            }
        }
    }
    pk
}

/// Specialized 3×3 **stride-1** conv for `c_in = 8, c_out = 8` (block1.2).
///
/// Each input pixel is two `float32x4` (8 channels); each output pixel is two
/// `float32x4` (8 channels, split `co0..3` / `co4..7`). The 8-channel
/// reduction uses `vfmaq_laneq_f32`: the input pixel's two channel quads are
/// broadcast-per-lane against the pre-packed per-(tap,ci) weight quads (see
/// [`pack_b1_2_laneq`]).
///
/// Stride-1 means adjacent output pixels share input columns. The interior is
/// processed in 4-pixel blocks: the 6 input columns spanned by a block
/// (`ow-1 .. ow+4`) are each loaded **once** per `kh` row (two quads apiece)
/// and reused across the three `kw` taps. Eight accumulators (4 px × 2 quads)
/// give 8 independent FMA chains to hide the 4-cycle FMA latency; weight quads
/// are reloaded per (tap, ci) — they stay L1-hot and the const indices keep
/// them off the stack (the spill lesson from 175967c).
///
/// Reduction-order note: the 8 ci terms accumulate sequentially per lane
/// (lanes = co) instead of the scalar fallback's horizontal 4-lane add
/// (lanes = ci), so parity is tolerance- (≤ 5e-5), not bit-, exact.
#[cfg(target_arch = "aarch64")]
pub fn conv3x3_s1_c8_co8_neon(
    input: &[f32],
    packed: &[f32], // pack_b1_2_laneq layout, 576 f32
    bias: &[f32],
    h_in: usize,
    w_in: usize,
    activation: Activation,
    output: &mut [f32],
) {
    debug_assert_eq!(input.len(), h_in * w_in * 8);
    debug_assert_eq!(packed.len(), 9 * 8 * 2 * 4);
    debug_assert_eq!(bias.len(), 8);
    debug_assert_eq!(output.len(), h_in * w_in * 8);
    debug_assert!(
        matches!(activation, Activation::Relu | Activation::Identity),
        "conv3x3_s1_c8_co8_neon: unsupported activation"
    );
    debug_assert!(w_in >= 6);

    let relu = matches!(activation, Activation::Relu);
    let in_addr = input.as_ptr() as usize;
    let pk_addr = packed.as_ptr() as usize;

    /// Accumulate one (tap) contribution over all 8 ci into (acc_lo, acc_hi)
    /// for a single input pixel held as (in_lo, in_hi).
    #[inline(always)]
    unsafe fn fma_tap(
        acc: &mut (float32x4_t, float32x4_t),
        pk: *const f32,
        tap: usize,
        in_lo: float32x4_t,
        in_hi: float32x4_t,
    ) {
        macro_rules! ci_block {
            ($ci:expr, $inv:expr, $lane:expr) => {{
                let b = pk.add(((tap * 8 + $ci) * 2) * 4);
                acc.0 = vfmaq_laneq_f32::<$lane>(acc.0, vld1q_f32(b), $inv);
                acc.1 = vfmaq_laneq_f32::<$lane>(acc.1, vld1q_f32(b.add(4)), $inv);
            }};
        }
        ci_block!(0, in_lo, 0);
        ci_block!(1, in_lo, 1);
        ci_block!(2, in_lo, 2);
        ci_block!(3, in_lo, 3);
        ci_block!(4, in_hi, 0);
        ci_block!(5, in_hi, 1);
        ci_block!(6, in_hi, 2);
        ci_block!(7, in_hi, 3);
    }

    /// One (tap) contribution for four pixels sharing the same weight quads:
    /// the 2 weight quads per ci are loaded once and FMA'd into all four
    /// pixels' accumulators (4 input pixels supplied as 8 channel quads).
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn fma_tap4(
        a0: &mut (float32x4_t, float32x4_t),
        a1: &mut (float32x4_t, float32x4_t),
        a2: &mut (float32x4_t, float32x4_t),
        a3: &mut (float32x4_t, float32x4_t),
        pk: *const f32,
        tap: usize,
        in0: (float32x4_t, float32x4_t),
        in1: (float32x4_t, float32x4_t),
        in2: (float32x4_t, float32x4_t),
        in3: (float32x4_t, float32x4_t),
    ) {
        macro_rules! ci_block {
            ($ci:expr, $v0:expr, $v1:expr, $v2:expr, $v3:expr, $lane:expr) => {{
                let b = pk.add(((tap * 8 + $ci) * 2) * 4);
                let wlo = vld1q_f32(b);
                let whi = vld1q_f32(b.add(4));
                a0.0 = vfmaq_laneq_f32::<$lane>(a0.0, wlo, $v0);
                a0.1 = vfmaq_laneq_f32::<$lane>(a0.1, whi, $v0);
                a1.0 = vfmaq_laneq_f32::<$lane>(a1.0, wlo, $v1);
                a1.1 = vfmaq_laneq_f32::<$lane>(a1.1, whi, $v1);
                a2.0 = vfmaq_laneq_f32::<$lane>(a2.0, wlo, $v2);
                a2.1 = vfmaq_laneq_f32::<$lane>(a2.1, whi, $v2);
                a3.0 = vfmaq_laneq_f32::<$lane>(a3.0, wlo, $v3);
                a3.1 = vfmaq_laneq_f32::<$lane>(a3.1, whi, $v3);
            }};
        }
        // Lanes 0..3 of in*.0 are ci 0..3; lanes 0..3 of in*.1 are ci 4..7.
        ci_block!(0, in0.0, in1.0, in2.0, in3.0, 0);
        ci_block!(1, in0.0, in1.0, in2.0, in3.0, 1);
        ci_block!(2, in0.0, in1.0, in2.0, in3.0, 2);
        ci_block!(3, in0.0, in1.0, in2.0, in3.0, 3);
        ci_block!(4, in0.1, in1.1, in2.1, in3.1, 0);
        ci_block!(5, in0.1, in1.1, in2.1, in3.1, 1);
        ci_block!(6, in0.1, in1.1, in2.1, in3.1, 2);
        ci_block!(7, in0.1, in1.1, in2.1, in3.1, 3);
    }

    /// One output pixel with runtime kw bounds (column edges only).
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn px1<const KH_LO: usize, const KH_HI: usize>(
        inp: *const f32,
        pk: *const f32,
        oh: usize,
        ow: usize,
        w_in: usize,
        kw_lo: usize,
        kw_hi: usize,
    ) -> (float32x4_t, float32x4_t) {
        let mut acc = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
        for kh in KH_LO..KH_HI {
            // ih = oh + kh - 1, iw = ow + kw - 1.
            let row = inp.add(((oh + kh - 1) * w_in + ow) * 8);
            for kw in kw_lo..kw_hi {
                let p = row.add((kw - 1) * 8);
                let in_lo = vld1q_f32(p);
                let in_hi = vld1q_f32(p.add(4));
                fma_tap(&mut acc, pk, kh * 3 + kw, in_lo, in_hi);
            }
        }
        acc
    }

    /// Four interior output pixels (ow..ow+3), full kw range. The 6 shared
    /// input columns per `kh` row are loaded once and reused across taps.
    #[inline(always)]
    unsafe fn px4<const KH_LO: usize, const KH_HI: usize>(
        inp: *const f32,
        pk: *const f32,
        oh: usize,
        ow: usize,
        w_in: usize,
    ) -> [(float32x4_t, float32x4_t); 4] {
        let z = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
        let mut a0 = z;
        let mut a1 = z;
        let mut a2 = z;
        let mut a3 = z;
        for kh in KH_LO..KH_HI {
            // Leftmost shared column is iw = ow - 1 (tap kw=0 of pixel ow).
            let row = inp.add(((oh + kh - 1) * w_in + ow - 1) * 8);
            // Load the 6 spanned input columns once (each two quads).
            let c0 = (vld1q_f32(row), vld1q_f32(row.add(4)));
            let c1 = (vld1q_f32(row.add(8)), vld1q_f32(row.add(12)));
            let c2 = (vld1q_f32(row.add(16)), vld1q_f32(row.add(20)));
            let c3 = (vld1q_f32(row.add(24)), vld1q_f32(row.add(28)));
            let c4 = (vld1q_f32(row.add(32)), vld1q_f32(row.add(36)));
            let c5 = (vld1q_f32(row.add(40)), vld1q_f32(row.add(44)));
            // Pixel p, tap kw → shared column slot (p + kw).
            // kw=0: pixels use cols (0,1,2,3); kw=1: (1,2,3,4); kw=2: (2,3,4,5).
            fma_tap4(
                &mut a0,
                &mut a1,
                &mut a2,
                &mut a3,
                pk,
                kh * 3,
                c0,
                c1,
                c2,
                c3,
            );
            fma_tap4(
                &mut a0,
                &mut a1,
                &mut a2,
                &mut a3,
                pk,
                kh * 3 + 1,
                c1,
                c2,
                c3,
                c4,
            );
            fma_tap4(
                &mut a0,
                &mut a1,
                &mut a2,
                &mut a3,
                pk,
                kh * 3 + 2,
                c2,
                c3,
                c4,
                c5,
            );
        }
        [a0, a1, a2, a3]
    }

    #[inline(always)]
    unsafe fn store_px(
        out: *mut f32,
        acc: (float32x4_t, float32x4_t),
        blo: float32x4_t,
        bhi: float32x4_t,
        relu: bool,
    ) {
        let mut lo = vaddq_f32(acc.0, blo);
        let mut hi = vaddq_f32(acc.1, bhi);
        if relu {
            let z = vdupq_n_f32(0.0);
            lo = vmaxq_f32(lo, z);
            hi = vmaxq_f32(hi, z);
        }
        vst1q_f32(out, lo);
        vst1q_f32(out.add(4), hi);
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn row_body<const KH_LO: usize, const KH_HI: usize>(
        inp: *const f32,
        pk: *const f32,
        oh: usize,
        w_in: usize,
        blo: float32x4_t,
        bhi: float32x4_t,
        relu: bool,
        out: *mut f32,
    ) {
        // Left edge (ow = 0): kw = 0 reads iw = -1 → skip.
        store_px(
            out,
            px1::<KH_LO, KH_HI>(inp, pk, oh, 0, w_in, 1, 3),
            blo,
            bhi,
            relu,
        );
        // Interior 1..w_in-1: 4-pixel blocks then a 1-pixel tail.
        let mut ow = 1usize;
        while ow + 4 <= w_in - 1 {
            let a = px4::<KH_LO, KH_HI>(inp, pk, oh, ow, w_in);
            for (p, &acc) in a.iter().enumerate() {
                store_px(out.add((ow + p) * 8), acc, blo, bhi, relu);
            }
            ow += 4;
        }
        while ow < w_in - 1 {
            store_px(
                out.add(ow * 8),
                px1::<KH_LO, KH_HI>(inp, pk, oh, ow, w_in, 0, 3),
                blo,
                bhi,
                relu,
            );
            ow += 1;
        }
        // Right edge (ow = w_in-1): kw = 2 reads iw = w_in → skip.
        store_px(
            out.add((w_in - 1) * 8),
            px1::<KH_LO, KH_HI>(inp, pk, oh, w_in - 1, w_in, 0, 2),
            blo,
            bhi,
            relu,
        );
    }

    output
        .par_chunks_mut(w_in * 8)
        .enumerate()
        .for_each(|(oh, row_out)| unsafe {
            let inp = in_addr as *const f32;
            let pk = pk_addr as *const f32;
            let blo = vld1q_f32(bias.as_ptr());
            let bhi = vld1q_f32(bias.as_ptr().add(4));
            let out = row_out.as_mut_ptr();
            // oh=0 → kh=0 reads ih=-1, skip. oh=h_in-1 → kh=2 reads ih=h_in,
            // skip. Interior rows use the full 0..3 kh range.
            if oh == 0 {
                row_body::<1, 3>(inp, pk, oh, w_in, blo, bhi, relu, out);
            } else if oh == h_in - 1 {
                row_body::<0, 2>(inp, pk, oh, w_in, blo, bhi, relu, out);
            } else {
                row_body::<0, 3>(inp, pk, oh, w_in, blo, bhi, relu, out);
            }
        });
}
