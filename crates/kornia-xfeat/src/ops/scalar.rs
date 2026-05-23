//! Scalar reference implementations of every fused / primitive op the model
//! uses. These are the parity oracle the SIMD backends are tested against —
//! correctness-first, no manual unrolling, no tuning.

use super::{Activation, Conv1x1Args, Conv3x3Args};

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

    for oh in 0..h_out {
        for ow in 0..w_out {
            let ih_base = (oh * stride) as isize - 1;
            let iw_base = (ow * stride) as isize - 1;

            for (co, &b) in bias.iter().enumerate().take(c_out) {
                let mut acc = b;
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
                            acc += input[in_off + ci] * weights[w_off + ci];
                        }
                    }
                }
                let out_off = (oh * w_out + ow) * c_out + co;
                if let Some(r) = residual {
                    acc += r[out_off];
                }
                output[out_off] = apply_act(acc, activation);
            }
        }
    }
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
            let mut acc = bias[co];
            let w_off = co * c_in;
            for ci in 0..c_in {
                acc += input[in_off + ci] * weights[w_off + ci];
            }
            output[out_off + co] = apply_act(acc, activation);
        }
    }
}

/// `InstanceNorm2d(1, affine=False)` — per-image mean/variance over the single
/// channel, then `(x - mean) / sqrt(var + eps)`. Input is `(H, W, 1)`.
/// Output is the same shape; eps matches PyTorch's default `1e-5`.
pub fn instance_norm_2d_singlech(input: &[f32], output: &mut [f32]) {
    let n = input.len() as f32;
    let mean = input.iter().sum::<f32>() / n;
    let var = input.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n;
    let inv_std = 1.0 / (var + 1e-5).sqrt();
    for (o, &x) in output.iter_mut().zip(input.iter()) {
        *o = (x - mean) * inv_std;
    }
}

/// `AvgPool2d(kernel=4, stride=4)` on NHWC. Output is `(H/4, W/4, C)`.
pub fn avgpool_4x4_s4(input: &[f32], output: &mut [f32], h_in: usize, w_in: usize, c: usize) {
    let h_out = h_in / 4;
    let w_out = w_in / 4;
    debug_assert_eq!(output.len(), h_out * w_out * c);
    for oh in 0..h_out {
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
                output[(oh * w_out + ow) * c + ci] = acc / 16.0;
            }
        }
    }
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
    for oh in 0..h_out {
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
                output[(oh * w_out + ow) * c + ci] = v0 * (1.0 - wy) + v1 * wy;
            }
        }
    }
}

/// `a += b + c` over equal-length slices. Used by the FPN fusion sum
/// `x3 + x4_up + x5_up`. The first operand is the accumulator.
pub fn add3_inplace(a: &mut [f32], b: &[f32], c: &[f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), c.len());
    for ((ai, &bi), &ci) in a.iter_mut().zip(b).zip(c) {
        *ai += bi + ci;
    }
}

/// L2-normalize each pixel's channel vector. NHWC.
pub fn l2_normalize_channel(buf: &mut [f32], h: usize, w: usize, c: usize) {
    debug_assert_eq!(buf.len(), h * w * c);
    for px in 0..(h * w) {
        let off = px * c;
        let mut sum_sq = 0.0f32;
        for ci in 0..c {
            sum_sq += buf[off + ci] * buf[off + ci];
        }
        let inv = 1.0 / (sum_sq + 1e-12).sqrt();
        for ci in 0..c {
            buf[off + ci] *= inv;
        }
    }
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
    for px in 0..(h * w) {
        let off = px * c;
        let row = &mut buf[off..off + c];
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
    for oh in 0..h_out {
        for ow in 0..w_out {
            let out_off = (oh * w_out + ow) * 64;
            for kh in 0..8 {
                for kw in 0..8 {
                    let ih = oh * 8 + kh;
                    let iw = ow * 8 + kw;
                    output[out_off + kh * 8 + kw] = input[ih * w_in + iw];
                }
            }
        }
    }
}

/// Pixel-shuffle (depth-to-space) with factor 8 on NHWC.
///
/// Input `(H, W, 64)` → output `(H*8, W*8, 1)`. Mirrors XFeat's reshape from
/// the 64 keypoint channels at /8 scale back to the original resolution.
pub fn pixel_shuffle_8(input: &[f32], output: &mut [f32], h_in: usize, w_in: usize) {
    debug_assert_eq!(input.len(), h_in * w_in * 64);
    let h_out = h_in * 8;
    let w_out = w_in * 8;
    debug_assert_eq!(output.len(), h_out * w_out);
    for h in 0..h_in {
        for w in 0..w_in {
            let in_off = (h * w_in + w) * 64;
            for kh in 0..8 {
                for kw in 0..8 {
                    let oh = h * 8 + kh;
                    let ow = w * 8 + kw;
                    output[oh * w_out + ow] = input[in_off + kh * 8 + kw];
                }
            }
        }
    }
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
    let mut out = Vec::new();
    for oy in 0..h {
        for ox in 0..w {
            let v = input[oy * w + ox];
            if v <= threshold {
                continue;
            }
            let mut is_max = true;
            'outer: for dy in -2i32..=2 {
                for dx in -2i32..=2 {
                    let ny = oy as i32 + dy;
                    let nx = ox as i32 + dx;
                    if ny < 0 || ny >= h as i32 || nx < 0 || nx >= w as i32 {
                        continue;
                    }
                    if input[(ny as usize) * w + nx as usize] > v {
                        is_max = false;
                        break 'outer;
                    }
                }
            }
            if is_max {
                out.push((v, oy * w + ox));
            }
        }
    }
    out
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
