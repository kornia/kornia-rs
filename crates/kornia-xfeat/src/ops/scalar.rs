//! Scalar reference implementations of every fused / primitive op the model
//! uses. These are the parity oracle the SIMD backends are tested against —
//! correctness-first, no manual unrolling, no tuning.

use super::{Activation, Conv1x1Args, Conv3x3Args};
use rayon::prelude::*;

#[inline]
fn apply_act(x: f32, act: Activation) -> f32 {
    match act {
        Activation::Relu => x.max(0.0),
        Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        Activation::Identity => x,
    }
}

/// 3×3 stride-1 NHWC conv with zero-padding, optional residual, fused activation.
///
/// Weights layout: `[c_out, k_h=3, k_w=3, c_in]` (row-major); index expression
/// `weights[((co * 3 + kh) * 3 + kw) * c_in + ci]`.
///
/// Input layout: `[h, w, c_in]` (NHWC). Output: `[h, w, c_out]`.
pub fn conv3x3_relu_nhwc(args: &Conv3x3Args<'_>, output: &mut [f32]) {
    conv3x3_generic(args, output, 1);
}

/// 3×3 stride-2 NHWC conv. Output spatial extent is `(h_in / 2, w_in / 2)`.
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
        packed_weights: _,
    } = args;
    let h_out = h_in / stride;
    let w_out = w_in / stride;

    debug_assert_eq!(input.len(), h_in * w_in * c_in);
    debug_assert_eq!(output.len(), h_out * w_out * c_out);
    debug_assert_eq!(weights.len(), c_out * 9 * c_in);
    debug_assert_eq!(bias.len(), c_out);
    if let Some(r) = residual {
        debug_assert_eq!(r.len(), output.len());
    }

    // Parallel over output rows — each row is independent (read-only input/weights/bias).
    output
        .par_chunks_mut(w_out * c_out)
        .enumerate()
        .for_each(|(oh, row_out)| {
            let ih_base = (oh * stride) as isize - 1;
            for ow in 0..w_out {
                let iw_base = (ow * stride) as isize - 1;

                for (co, &b) in bias.iter().enumerate().take(c_out) {
                    // 4-way FMA accumulators declared once per (oh, ow, co) work-item,
                    // accumulated across ALL valid (kh, kw) taps — identical structure
                    // to NEON conv3x3_generic which initialises acc0..acc3 to zero once
                    // per (co) and folds only after all taps.
                    let dot = if c_in % 4 == 0 {
                        let mut acc0 = [0.0f32; 4];
                        let mut acc1 = [0.0f32; 4];
                        let mut acc2 = [0.0f32; 4];
                        let mut acc3 = [0.0f32; 4];

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
                                let in_off = (ih * w_in + iw) * c_in;
                                let w_off = ((co * 3 + kh) * 3 + kw) * c_in;
                                // Phase 1: full 16-element blocks, 4 parallel accumulators.
                                let n16 = c_in & !15;
                                let mut i = 0usize;
                                while i < n16 {
                                    for j in 0..4 {
                                        acc0[j] = f32::mul_add(
                                            input[in_off + i + j],
                                            weights[w_off + i + j],
                                            acc0[j],
                                        );
                                        acc1[j] = f32::mul_add(
                                            input[in_off + i + 4 + j],
                                            weights[w_off + i + 4 + j],
                                            acc1[j],
                                        );
                                        acc2[j] = f32::mul_add(
                                            input[in_off + i + 8 + j],
                                            weights[w_off + i + 8 + j],
                                            acc2[j],
                                        );
                                        acc3[j] = f32::mul_add(
                                            input[in_off + i + 12 + j],
                                            weights[w_off + i + 12 + j],
                                            acc3[j],
                                        );
                                    }
                                    i += 16;
                                }
                                // Phase 2: f32x4 tail into acc0 only.
                                while i < c_in {
                                    for j in 0..4 {
                                        acc0[j] = f32::mul_add(
                                            input[in_off + i + j],
                                            weights[w_off + i + j],
                                            acc0[j],
                                        );
                                    }
                                    i += 4;
                                }
                            }
                        }

                        // Horizontal fold: (acc0+acc1)+(acc2+acc3), then sum 4 lanes.
                        // Mirrors NEON: vaddq_f32(vaddq_f32(acc0,acc1), vaddq_f32(acc2,acc3)) + vaddvq_f32.
                        let mut s = [0.0f32; 4];
                        for j in 0..4 {
                            s[j] = (acc0[j] + acc1[j]) + (acc2[j] + acc3[j]);
                        }
                        s[0] + s[1] + s[2] + s[3]
                    } else {
                        // Scalar fallback for c_in ∈ {1, 2, 3} (not hot paths in XFeat).
                        let mut acc_scalar = 0.0f32;
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
                                let in_off = (ih * w_in + iw) * c_in;
                                let w_off = ((co * 3 + kh) * 3 + kw) * c_in;
                                for ci in 0..c_in {
                                    acc_scalar = f32::mul_add(
                                        input[in_off + ci],
                                        weights[w_off + ci],
                                        acc_scalar,
                                    );
                                }
                            }
                        }
                        acc_scalar
                    };

                    let out_off_row = ow * c_out + co;
                    let out_off_abs = (oh * w_out + ow) * c_out + co;
                    let mut v = dot + b;
                    if let Some(r) = residual {
                        v += r[out_off_abs];
                    }
                    row_out[out_off_row] = apply_act(v, activation);
                }
            }
        });
}

/// 1×1 pointwise conv (per-pixel GEMM along channels). Weights `[c_out, c_in]`.
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
    debug_assert_eq!(input.len(), h * w * c_in);
    debug_assert_eq!(output.len(), h * w * c_out);
    debug_assert_eq!(weights.len(), c_out * c_in);
    debug_assert_eq!(bias.len(), c_out);

    for px in 0..(h * w) {
        let in_off = px * c_in;
        let out_off = px * c_out;
        for co in 0..c_out {
            let w_off = co * c_in;
            // 4-way FMA reduction matching NEON dot_f32_4ply_tail structure.
            // Falls back to simple loop when c_in % 4 != 0 (c_in ∈ {1,2,3} only).
            let dot = if c_in % 4 == 0 {
                let mut acc0 = [0.0f32; 4];
                let mut acc1 = [0.0f32; 4];
                let mut acc2 = [0.0f32; 4];
                let mut acc3 = [0.0f32; 4];
                // Phase 1: full 16-element blocks.
                let n16 = c_in & !15;
                let mut i = 0usize;
                while i < n16 {
                    for j in 0..4 {
                        acc0[j] =
                            f32::mul_add(input[in_off + i + j], weights[w_off + i + j], acc0[j]);
                        acc1[j] = f32::mul_add(
                            input[in_off + i + 4 + j],
                            weights[w_off + i + 4 + j],
                            acc1[j],
                        );
                        acc2[j] = f32::mul_add(
                            input[in_off + i + 8 + j],
                            weights[w_off + i + 8 + j],
                            acc2[j],
                        );
                        acc3[j] = f32::mul_add(
                            input[in_off + i + 12 + j],
                            weights[w_off + i + 12 + j],
                            acc3[j],
                        );
                    }
                    i += 16;
                }
                // Phase 2: f32x4 tail, acc0 only.
                while i < c_in {
                    for j in 0..4 {
                        acc0[j] =
                            f32::mul_add(input[in_off + i + j], weights[w_off + i + j], acc0[j]);
                    }
                    i += 4;
                }
                // Horizontal fold: (acc0+acc1)+(acc2+acc3), then sum 4 lanes.
                let mut s = [0.0f32; 4];
                for j in 0..4 {
                    s[j] = (acc0[j] + acc1[j]) + (acc2[j] + acc3[j]);
                }
                s[0] + s[1] + s[2] + s[3]
            } else {
                let mut d = 0.0f32;
                for ci in 0..c_in {
                    d = f32::mul_add(input[in_off + ci], weights[w_off + ci], d);
                }
                d
            };
            output[out_off + co] = apply_act(dot + bias[co], activation);
        }
    }
}

/// `InstanceNorm2d(1, affine=False)` — per-image mean/variance over the single
/// channel, then `(x - mean) / sqrt(var + eps)`. Input is `(H, W, 1)`.
/// Output is the same shape; eps matches PyTorch's default `1e-5`.
pub fn instance_norm_2d_singlech(input: &[f32], output: &mut [f32]) {
    let n = input.len() as f32;
    // Parallel reductions for mean and variance, then parallel normalisation.
    // Use a minimum chunk size so thread-spawn overhead is amortised.
    const MIN_CHUNK: usize = 16384;
    let mean: f32 = input
        .par_iter()
        .with_min_len(MIN_CHUNK)
        .map(|&x| x)
        .sum::<f32>()
        / n;
    let var: f32 = input
        .par_iter()
        .with_min_len(MIN_CHUNK)
        .map(|&x| (x - mean) * (x - mean))
        .sum::<f32>()
        / n;
    let inv_std = 1.0 / (var + 1e-5).sqrt();
    output
        .par_iter_mut()
        .with_min_len(MIN_CHUNK)
        .zip(input.par_iter().with_min_len(MIN_CHUNK))
        .for_each(|(o, &x)| {
            *o = (x - mean) * inv_std;
        });
}

/// `AvgPool2d(kernel=4, stride=4)` on NHWC. Output is `(H/4, W/4, C)`.
pub fn avgpool_4x4_s4(input: &[f32], output: &mut [f32], h_in: usize, w_in: usize, c: usize) {
    let h_out = h_in / 4;
    let w_out = w_in / 4;
    debug_assert_eq!(output.len(), h_out * w_out * c);
    output
        .par_chunks_mut(w_out * c)
        .enumerate()
        .for_each(|(oh, row_out)| {
            for ow in 0..w_out {
                for ci in 0..c {
                    let mut acc = 0.0f32;
                    for kh in 0..4 {
                        for kw in 0..4 {
                            let ih = oh * 4 + kh;
                            let iw = ow * 4 + kw;
                            acc += input[(ih * w_in + iw) * c + ci];
                        }
                    }
                    row_out[ow * c + ci] = acc / 16.0;
                }
            }
        });
}

/// `F.interpolate(..., mode='bilinear', align_corners=False)` from
/// `(H_in, W_in)` to `(H_out, W_out)`, NHWC. Used by the FPN to upsample
/// `x4` and `x5` to the descriptor scale.
pub fn bilinear_upsample(
    input: &[f32],
    output: &mut [f32],
    h_in: usize,
    w_in: usize,
    c: usize,
    h_out: usize,
    w_out: usize,
) {
    let sh = h_in as f32 / h_out as f32;
    let sw = w_in as f32 / w_out as f32;
    output
        .par_chunks_mut(w_out * c)
        .enumerate()
        .for_each(|(oh, row_out)| {
            let ys = (oh as f32 + 0.5) * sh - 0.5;
            let y0 = ys.floor();
            let y1 = y0 + 1.0;
            let wy = ys - y0;
            let y0 = (y0 as isize).clamp(0, h_in as isize - 1) as usize;
            let y1 = (y1 as isize).clamp(0, h_in as isize - 1) as usize;
            for ow in 0..w_out {
                let xs = (ow as f32 + 0.5) * sw - 0.5;
                let x0 = xs.floor();
                let x1 = x0 + 1.0;
                let wx = xs - x0;
                let x0 = (x0 as isize).clamp(0, w_in as isize - 1) as usize;
                let x1 = (x1 as isize).clamp(0, w_in as isize - 1) as usize;
                for ci in 0..c {
                    let v00 = input[(y0 * w_in + x0) * c + ci];
                    let v01 = input[(y0 * w_in + x1) * c + ci];
                    let v10 = input[(y1 * w_in + x0) * c + ci];
                    let v11 = input[(y1 * w_in + x1) * c + ci];
                    let v0 = v00 * (1.0 - wx) + v01 * wx;
                    let v1 = v10 * (1.0 - wx) + v11 * wx;
                    row_out[ow * c + ci] = v0 * (1.0 - wy) + v1 * wy;
                }
            }
        });
}

/// Fused FPN merge: `x3 += upsample(x4) + upsample(x5)` in one parallel pass.
///
/// Scalar fallback for `neon::fpn_upsample2_add3_neon`. Mirrors
/// [`bilinear_upsample`]'s separable-lerp form for each source and
/// [`add3_inplace`]'s `*x += y + z` grouping for the sum, so it is the
/// composition of the unfused scalar pipeline, computed without the
/// intermediate buffers.
#[allow(clippy::too_many_arguments)]
pub fn fpn_upsample2_add3(
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
    let sh4 = h4 as f32 / h_out as f32;
    let sw4 = w4 as f32 / w_out as f32;
    let sh5 = h5 as f32 / h_out as f32;
    let sw5 = w5 as f32 / w_out as f32;
    x3.par_chunks_mut(w_out * c)
        .enumerate()
        .for_each(|(oh, row_out)| {
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
                let xs5 = (ow as f32 + 0.5) * sw5 - 0.5;
                let x0f5 = xs5.floor();
                let wx5 = xs5 - x0f5;
                let x0_5 = (x0f5 as isize).clamp(0, w5 as isize - 1) as usize;
                let x1_5 = ((x0f5 as isize + 1).clamp(0, w5 as isize - 1)) as usize;
                for ci in 0..c {
                    // upsample(x4) — identical lerp form to bilinear_upsample
                    let v00 = x4[(y0_4 * w4 + x0_4) * c + ci];
                    let v01 = x4[(y0_4 * w4 + x1_4) * c + ci];
                    let v10 = x4[(y1_4 * w4 + x0_4) * c + ci];
                    let v11 = x4[(y1_4 * w4 + x1_4) * c + ci];
                    let v0 = v00 * (1.0 - wx4) + v01 * wx4;
                    let v1 = v10 * (1.0 - wx4) + v11 * wx4;
                    let up4 = v0 * (1.0 - wy4) + v1 * wy4;
                    // upsample(x5)
                    let v00 = x5[(y0_5 * w5 + x0_5) * c + ci];
                    let v01 = x5[(y0_5 * w5 + x1_5) * c + ci];
                    let v10 = x5[(y1_5 * w5 + x0_5) * c + ci];
                    let v11 = x5[(y1_5 * w5 + x1_5) * c + ci];
                    let v0 = v00 * (1.0 - wx5) + v01 * wx5;
                    let v1 = v10 * (1.0 - wx5) + v11 * wx5;
                    let up5 = v0 * (1.0 - wy5) + v1 * wy5;
                    // x3 + (up4 + up5) — same grouping as add3_inplace
                    row_out[ow * c + ci] += up4 + up5;
                }
            }
        });
}

/// `a += b + c` over equal-length slices. Used by the FPN fusion sum
/// `x3 + x4_up + x5_up`. The first operand is the accumulator.
pub fn add3_inplace(a: &mut [f32], b: &[f32], c: &[f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), c.len());
    const CHUNK: usize = 4096;
    a.par_chunks_mut(CHUNK)
        .zip(b.par_chunks(CHUNK))
        .zip(c.par_chunks(CHUNK))
        .for_each(|((ai, bi), ci)| {
            for ((x, &y), &z) in ai.iter_mut().zip(bi.iter()).zip(ci.iter()) {
                *x += y + z;
            }
        });
}

/// Write `dst[i] = a[i] + b[i] + c[i]` — single-pass parallel 3-way add into dst.
pub fn add3_from(dst: &mut [f32], a: &[f32], b: &[f32], c: &[f32]) {
    debug_assert_eq!(dst.len(), a.len());
    debug_assert_eq!(dst.len(), b.len());
    debug_assert_eq!(dst.len(), c.len());
    dst.par_iter_mut()
        .zip(a.par_iter().zip(b.par_iter()).zip(c.par_iter()))
        .for_each(|(d, ((&ai, &bi), &ci))| *d = ai + bi + ci);
}

/// L2-normalize each pixel's channel vector. NHWC.
pub fn l2_normalize_channel(buf: &mut [f32], h: usize, w: usize, c: usize) {
    debug_assert_eq!(buf.len(), h * w * c);
    // Each pixel's channel slice is independent — parallel over pixels.
    // Use with_min_len so we get at least 256 pixels per rayon chunk,
    // amortising thread-dispatch overhead for small (H/8×W/8) maps.
    buf.par_chunks_mut(c).with_min_len(256).for_each(|chunk| {
        let sum_sq: f32 = chunk.iter().map(|&v| v * v).sum();
        let inv = 1.0 / (sum_sq + 1e-12).sqrt();
        for v in chunk {
            *v *= inv;
        }
    });
}

/// Apply sigmoid in-place.
pub fn sigmoid_inplace(buf: &mut [f32]) {
    for x in buf {
        *x = 1.0 / (1.0 + (-*x).exp());
    }
}

/// Per-pixel softmax across the channel axis (NHWC). Mirrors
/// `F.softmax(K1, dim=1)` after the layout transpose.
pub fn channel_softmax(buf: &mut [f32], h: usize, w: usize, c: usize) {
    debug_assert_eq!(buf.len(), h * w * c);
    // 64 pixels per Rayon task keeps ~75 tasks for 4800-pixel inputs — enough
    // load-balance without drowning in per-task dispatch overhead.
    buf.par_chunks_mut(c * 64).for_each(|block| {
        for row in block.chunks_mut(c) {
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for x in row.iter_mut() {
                *x = (*x - max).exp();
                sum += *x;
            }
            let inv = 1.0 / sum;
            for x in row.iter_mut() {
                *x *= inv;
            }
        }
    });
}

/// PyTorch's `_unfold2d(x, ws=8)` for a single-channel input.
///
/// Takes an `(H, W, 1)` tensor and produces `(H/8, W/8, 64)` where each output
/// pixel is the 8×8 patch of the input at that 8-cell, flattened in row-major
/// order. This is how the keypoint head sees the InstanceNorm'd grayscale.
pub fn unfold_8x8(input: &[f32], output: &mut [f32], h_in: usize, w_in: usize) {
    debug_assert_eq!(input.len(), h_in * w_in);
    let h_out = h_in / 8;
    let w_out = w_in / 8;
    debug_assert_eq!(output.len(), h_out * w_out * 64);
    output
        .par_chunks_mut(w_out * 64)
        .enumerate()
        .for_each(|(oh, row_out)| {
            for ow in 0..w_out {
                let out_off = ow * 64;
                for kh in 0..8 {
                    for kw in 0..8 {
                        let ih = oh * 8 + kh;
                        let iw = ow * 8 + kw;
                        row_out[out_off + kh * 8 + kw] = input[ih * w_in + iw];
                    }
                }
            }
        });
}

/// Pixel-shuffle (depth-to-space) with factor 8 on NHWC.
///
/// Input `(H, W, 64)` → output `(H*8, W*8, 1)`. Mirrors XFeat's reshape from
/// the 64 keypoint channels at /8 scale back to the original resolution.
pub fn pixel_shuffle_8(input: &[f32], output: &mut [f32], h_in: usize, w_in: usize) {
    debug_assert_eq!(input.len(), h_in * w_in * 64);
    let w_out = w_in * 8;
    debug_assert_eq!(output.len(), h_in * 8 * w_out);
    // Each output super-row (8 rows of w_out pixels) corresponds to one input row.
    output
        .par_chunks_mut(8 * w_out)
        .zip(input.par_chunks(w_in * 64))
        .for_each(|(super_row_out, in_row)| {
            for w in 0..w_in {
                let in_off = w * 64;
                for kh in 0..8 {
                    for kw in 0..8 {
                        super_row_out[kh * w_out + w * 8 + kw] = in_row[in_off + kh * 8 + kw];
                    }
                }
            }
        });
}

/// In-place element-wise add: `a[i] += b[i]`.
pub fn add_inplace(a: &mut [f32], b: &[f32]) {
    debug_assert_eq!(a.len(), b.len());
    // Chunk-based split: LLVM auto-vectorizes the inner loop (vs per-element par_iter).
    const CHUNK: usize = 4096;
    a.par_chunks_mut(CHUNK)
        .zip(b.par_chunks(CHUNK))
        .for_each(|(ai, bi)| {
            for (x, &y) in ai.iter_mut().zip(bi.iter()) {
                *x += y;
            }
        });
}

/// Copy the first `c_out` channels from each pixel of an NHWC tensor, discarding
/// the remainder. Used to strip the softmax dustbin channel (65→64).
pub fn drop_last_channel_nhwc(
    input: &[f32],
    output: &mut [f32],
    h: usize,
    w: usize,
    c_in: usize,
    c_out: usize,
) {
    debug_assert!(c_out < c_in);
    debug_assert_eq!(input.len(), h * w * c_in);
    debug_assert_eq!(output.len(), h * w * c_out);
    output
        .par_chunks_mut(c_out)
        .zip(input.par_chunks(c_in))
        .for_each(|(o, i)| {
            o.copy_from_slice(&i[..c_out]);
        });
}

// ── f16-storage variants of common ops ──────────────────────────────────────

/// Unfold 8×8 patches from a single-channel f32 input into f16 (u16) storage.
pub fn unfold_8x8_to_f16(input: &[f32], output: &mut [u16], h_in: usize, w_in: usize) {
    debug_assert_eq!(input.len(), h_in * w_in);
    let h_out = h_in / 8;
    let w_out = w_in / 8;
    debug_assert_eq!(output.len(), h_out * w_out * 64);
    output
        .par_chunks_mut(w_out * 64)
        .enumerate()
        .for_each(|(oh, row_out)| {
            for ow in 0..w_out {
                let out_off = ow * 64;
                for kh in 0..8 {
                    for kw in 0..8 {
                        let ih = oh * 8 + kh;
                        let iw = ow * 8 + kw;
                        row_out[out_off + kh * 8 + kw] =
                            half::f16::from_f32(input[ih * w_in + iw]).to_bits();
                    }
                }
            }
        });
}

/// Per-pixel channel softmax on f16 storage. exp() computed in f32 for stability.
pub fn channel_softmax_f16(buf: &mut [u16], h: usize, w: usize, c: usize) {
    debug_assert_eq!(buf.len(), h * w * c);
    buf.par_chunks_mut(c * 64).for_each(|block| {
        for row in block.chunks_mut(c) {
            let mut max = f32::NEG_INFINITY;
            for &v in row.iter() {
                let f = half::f16::from_bits(v).to_f32();
                if f > max {
                    max = f;
                }
            }
            let mut sum = 0.0f32;
            for v in row.iter_mut() {
                let f = (half::f16::from_bits(*v).to_f32() - max).exp();
                *v = half::f16::from_f32(f).to_bits();
                sum += f;
            }
            let inv = 1.0 / sum;
            for v in row.iter_mut() {
                let f = half::f16::from_bits(*v).to_f32() * inv;
                *v = half::f16::from_f32(f).to_bits();
            }
        }
    });
}

/// Sidecar-aware per-pixel channel softmax on f16 storage (scalar reference).
///
/// Computes softmax over `c_main + 1` logits per pixel, where the first `c_main`
/// logits live in `main` (row stride `c_main`) and the last logit (the "dustbin")
/// lives in `dustbin` (one value per pixel). The `c_main` normalized outputs are
/// written back to `main` in place; the dustbin's normalized output is written
/// back to `dustbin`.
///
/// To stay numerically equivalent to running `channel_softmax_f16` over a
/// stride-(c_main+1) interleaved buffer, the reduction order matches exactly:
/// the max is taken over the `c_main` main values first then the dustbin last,
/// and the exp-sum sums the main values in the same order before adding the
/// dustbin's exp last.
pub fn channel_softmax_f16_sidecar(
    main: &mut [u16],
    dustbin: &mut [u16],
    h: usize,
    w: usize,
    c_main: usize,
) {
    let m = h * w;
    debug_assert_eq!(main.len(), m * c_main);
    debug_assert_eq!(dustbin.len(), m);
    main.par_chunks_mut(c_main * 64)
        .zip(dustbin.par_chunks_mut(64))
        .for_each(|(block, dblock)| {
            for (row, d) in block.chunks_mut(c_main).zip(dblock.iter_mut()) {
                // max over main values first, dustbin last.
                let mut max = f32::NEG_INFINITY;
                for &v in row.iter() {
                    let f = half::f16::from_bits(v).to_f32();
                    if f > max {
                        max = f;
                    }
                }
                let df = half::f16::from_bits(*d).to_f32();
                if df > max {
                    max = df;
                }
                // exp(x - max): main in order, dustbin last; sum accumulated identically.
                let mut sum = 0.0f32;
                for v in row.iter_mut() {
                    let f = (half::f16::from_bits(*v).to_f32() - max).exp();
                    *v = half::f16::from_f32(f).to_bits();
                    sum += f;
                }
                let de = (df - max).exp();
                *d = half::f16::from_f32(de).to_bits();
                sum += de;
                let inv = 1.0 / sum;
                for v in row.iter_mut() {
                    let f = half::f16::from_bits(*v).to_f32() * inv;
                    *v = half::f16::from_f32(f).to_bits();
                }
                let f = half::f16::from_bits(*d).to_f32() * inv;
                *d = half::f16::from_f32(f).to_bits();
            }
        });
}

/// Pixel-shuffle (depth-to-space) factor 8. f16 input → f32 output.
/// Input `(H, W, 64)` → output `(H*8, W*8)`.
pub fn pixel_shuffle_8_f16(input: &[u16], output: &mut [f32], h_in: usize, w_in: usize) {
    debug_assert_eq!(input.len(), h_in * w_in * 64);
    let w_out = w_in * 8;
    debug_assert_eq!(output.len(), h_in * 8 * w_out);
    output
        .par_chunks_mut(8 * w_out)
        .zip(input.par_chunks(w_in * 64))
        .for_each(|(super_row, in_row)| {
            for w in 0..w_in {
                let in_off = w * 64;
                for kh in 0..8 {
                    for kw in 0..8 {
                        super_row[kh * w_out + w * 8 + kw] =
                            half::f16::from_bits(in_row[in_off + kh * 8 + kw]).to_f32();
                    }
                }
            }
        });
}

/// Fused sidecar-softmax + pixel-shuffle (factor 8). f16 in → f32 out.
///
/// For each pixel: run the sidecar softmax over `64` main channels + the
/// dustbin (identical numerics to [`channel_softmax_f16_sidecar`] — max over the
/// 64 main first then dustbin, exp-sum in the same order), write the normalized
/// dustbin back to the sidecar, then pixel-shuffle that pixel's 64 normalized
/// values into the f32 output (same indexing as [`pixel_shuffle_8_f16`]).
///
/// `main` is left holding the post-exp (un-normalized) f16 values; nothing
/// downstream reads it after this op.
pub fn channel_softmax_pixel_shuffle_f16_sidecar(
    main: &mut [u16],
    dustbin: &mut [u16],
    k1h_out: &mut [f32],
    h8: usize,
    w8: usize,
) {
    const C_MAIN: usize = 64;
    let w_out = w8 * 8;
    debug_assert_eq!(main.len(), h8 * w8 * C_MAIN);
    debug_assert_eq!(dustbin.len(), h8 * w8);
    debug_assert_eq!(k1h_out.len(), h8 * 8 * w_out);

    k1h_out
        .par_chunks_mut(8 * w_out)
        .zip(main.par_chunks_mut(w8 * C_MAIN))
        .zip(dustbin.par_chunks_mut(w8))
        .for_each(|((super_row, main_row), dust_row)| {
            for w in 0..w8 {
                let row = &mut main_row[w * C_MAIN..w * C_MAIN + C_MAIN];
                let d = &mut dust_row[w];

                // Softmax (identical order to channel_softmax_f16_sidecar).
                let mut max = f32::NEG_INFINITY;
                for &v in row.iter() {
                    let f = half::f16::from_bits(v).to_f32();
                    if f > max {
                        max = f;
                    }
                }
                let df = half::f16::from_bits(*d).to_f32();
                if df > max {
                    max = df;
                }
                let mut sum = 0.0f32;
                for v in row.iter_mut() {
                    let f = (half::f16::from_bits(*v).to_f32() - max).exp();
                    *v = half::f16::from_f32(f).to_bits();
                    sum += f;
                }
                let de = (df - max).exp();
                *d = half::f16::from_f32(de).to_bits();
                sum += de;
                let inv = 1.0 / sum;
                // Dustbin normalized + written back (parity with sidecar softmax).
                let dn = half::f16::from_bits(*d).to_f32() * inv;
                *d = half::f16::from_f32(dn).to_bits();

                // Shuffle the 64 normalized main values into the f32 output block.
                // Normalize each value (×inv) through the same f16 round-trip the
                // standalone softmax applies, then widen to f32.
                for kh in 0..8 {
                    for kw in 0..8 {
                        let f = half::f16::from_bits(row[kh * 8 + kw]).to_f32() * inv;
                        let fn16 = half::f16::from_f32(f).to_f32();
                        super_row[kh * w_out + w * 8 + kw] = fn16;
                    }
                }
            }
        });
}

/// L2-normalize each pixel's channel vector in-place. f16 storage.
pub fn l2_normalize_channel_f16(buf: &mut [u16], h: usize, w: usize, c: usize) {
    debug_assert_eq!(buf.len(), h * w * c);
    buf.par_chunks_mut(c).with_min_len(256).for_each(|chunk| {
        let sum_sq: f32 = chunk
            .iter()
            .map(|&v| {
                let f = half::f16::from_bits(v).to_f32();
                f * f
            })
            .sum();
        let inv = 1.0 / (sum_sq + 1e-12).sqrt();
        for v in chunk {
            let f = half::f16::from_bits(*v).to_f32() * inv;
            *v = half::f16::from_f32(f).to_bits();
        }
    });
}

/// NMS via MaxPool 5×5 stride 1 padding 2 equality check.
///
/// Returns a `(value, index)` per local max in raster order; `index` is the
/// flat row-major index into the input. Input must be `(H, W)`.
pub fn nms_maxpool_5x5_equality(
    input: &[f32],
    h: usize,
    w: usize,
    threshold: f32,
) -> Vec<(f32, usize)> {
    debug_assert_eq!(input.len(), h * w);

    // Process rows in 32-row chunks to reduce Vec allocations and rayon
    // task overhead (480 per-row tasks → 15 chunk tasks).
    const CHUNK_H: usize = 32;
    let h_i = h as i32;
    let w_i = w as i32;

    (0..h)
        .step_by(CHUNK_H)
        .collect::<Vec<_>>()
        .into_par_iter()
        .flat_map_iter(|oy_base| {
            let oy_end = (oy_base + CHUNK_H).min(h);
            let mut chunk_out = Vec::new();
            for oy in oy_base..oy_end {
                let oy_i = oy as i32;
                for ox in 0..w {
                    let v = input[oy * w + ox];
                    if v <= threshold {
                        continue;
                    }
                    let ox_i = ox as i32;
                    let mut is_max = true;
                    'outer: for dy in -2i32..=2 {
                        let ny = oy_i + dy;
                        if ny < 0 || ny >= h_i {
                            continue;
                        }
                        let row = (ny as usize) * w;
                        for dx in -2i32..=2 {
                            let nx = ox_i + dx;
                            if nx >= 0 && nx < w_i && input[row + nx as usize] > v {
                                is_max = false;
                                break 'outer;
                            }
                        }
                    }
                    if is_max {
                        chunk_out.push((v, oy * w + ox));
                    }
                }
            }
            chunk_out
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn conv3x3_identity_no_op() {
        // 3-channel input, 3-channel output, identity weights on the center tap.
        let (h, w, c) = (4, 4, 3);
        let mut input = vec![0.0f32; h * w * c];
        for (i, v) in input.iter_mut().enumerate() {
            *v = i as f32;
        }
        let mut weights = vec![0.0f32; c * 9 * c];
        // weights[(co * 9 + 4) * c + co] = 1.0 picks center tap, same channel.
        (0..c).for_each(|co| {
            weights[(co * 9 + 4) * c + co] = 1.0;
        });
        let bias = vec![0.0; c];
        let mut output = vec![0.0; h * w * c];
        conv3x3_relu_nhwc(
            &Conv3x3Args {
                input: &input,
                residual: None,
                weights: &weights,
                bias: &bias,
                h_in: h,
                w_in: w,
                c_in: c,
                c_out: c,
                activation: Activation::Identity,
                packed_weights: None,
            },
            &mut output,
        );
        // Border pixels lose tap weight from padding (zero), but the centers do not.
        // Pixel (2, 2) center tap reads input(2,2): all input values are >= 0, so
        // identity activation returns them unchanged.
        for ci in 0..c {
            assert_abs_diff_eq!(
                output[((2 * w) + 2) * c + ci],
                input[((2 * w) + 2) * c + ci]
            );
        }
    }

    #[test]
    fn conv1x1_per_pixel_dot() {
        // 1x1 conv: each output pixel is bias + input · weights[co].
        let (h, w, c_in, c_out) = (2, 3, 4, 2);
        let mut input = vec![0.0f32; h * w * c_in];
        for (i, v) in input.iter_mut().enumerate() {
            *v = (i as f32) * 0.1;
        }
        let weights = vec![1.0; c_out * c_in];
        let bias = vec![0.5; c_out];
        let mut output = vec![0.0; h * w * c_out];
        conv1x1_nhwc(
            &Conv1x1Args {
                input: &input,
                weights: &weights,
                bias: &bias,
                h,
                w,
                c_in,
                c_out,
                activation: Activation::Identity,
            },
            &mut output,
        );
        // For each pixel, output = bias + sum(input).
        for px in 0..(h * w) {
            let sum: f32 = input[px * c_in..(px + 1) * c_in].iter().sum();
            for co in 0..c_out {
                assert_abs_diff_eq!(output[px * c_out + co], 0.5 + sum, epsilon = 1e-5);
            }
        }
    }

    #[test]
    fn instance_norm_zero_mean_unit_var() {
        let input = (0..64).map(|i| i as f32).collect::<Vec<_>>();
        let mut output = vec![0.0; input.len()];
        instance_norm_2d_singlech(&input, &mut output);
        let mean = output.iter().sum::<f32>() / output.len() as f32;
        let var =
            output.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / output.len() as f32;
        assert_abs_diff_eq!(mean, 0.0, epsilon = 1e-5);
        assert_abs_diff_eq!(var, 1.0, epsilon = 1e-3);
    }

    #[test]
    fn l2_normalize_unit_norm() {
        let (h, w, c) = (2, 3, 4);
        let mut buf = vec![0.0f32; h * w * c];
        for (i, v) in buf.iter_mut().enumerate() {
            *v = (i as f32) + 1.0;
        }
        l2_normalize_channel(&mut buf, h, w, c);
        for px in 0..(h * w) {
            let off = px * c;
            let norm = buf[off..off + c].iter().map(|x| x * x).sum::<f32>().sqrt();
            assert_abs_diff_eq!(norm, 1.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn channel_softmax_sums_to_one() {
        let (h, w, c) = (2, 2, 5);
        let mut buf = (0..h * w * c).map(|i| i as f32 * 0.1).collect::<Vec<_>>();
        channel_softmax(&mut buf, h, w, c);
        for px in 0..(h * w) {
            let sum: f32 = buf[px * c..(px + 1) * c].iter().sum();
            assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn nms_picks_single_max() {
        let h = 7;
        let w = 7;
        let mut img = vec![0.0f32; h * w];
        img[3 * w + 3] = 1.0;
        img[3 * w + 4] = 0.5; // suppressed by 5x5 NMS around (3,3)
        let kp = nms_maxpool_5x5_equality(&img, h, w, 0.1);
        assert_eq!(kp.len(), 1);
        assert_eq!(kp[0].1, 3 * w + 3);
    }

    #[test]
    fn pixel_shuffle_8_inverse_of_unfold() {
        let h_in = 16;
        let w_in = 16;
        let input: Vec<f32> = (0..h_in * w_in).map(|i| i as f32).collect();
        let mut unfolded = vec![0.0; (h_in / 8) * (w_in / 8) * 64];
        unfold_8x8(&input, &mut unfolded, h_in, w_in);
        let mut roundtrip = vec![0.0; h_in * w_in];
        pixel_shuffle_8(&unfolded, &mut roundtrip, h_in / 8, w_in / 8);
        for (a, b) in input.iter().zip(roundtrip.iter()) {
            assert_abs_diff_eq!(*a, *b);
        }
    }
}
