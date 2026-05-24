//! Per-kernel microbenches at every shape XFeat actually uses.
//!
//! Runs both scalar and (on aarch64) NEON for each shape so the
//! speedup table can be read directly from the criterion output.
//! AVX2 micro-benches will be added once the AVX2 kernel is real.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use kornia_xfeat::ops::{scalar, Activation, Conv1x1Args, Conv3x3Args};

#[cfg(target_arch = "aarch64")]
use kornia_xfeat::ops::neon;

/// Representative XFeat shapes (h, w, c_in, c_out). Block transitions are
/// expressed at the *input* spatial extent; stride-2 layers halve it.
const CONV3X3_SHAPES: &[(usize, usize, usize, usize, usize)] = &[
    // (h, w, c_in, c_out, stride). Block 3 mid and block_fusion finale share
    // (60, 80, 64, 64, 1) — listed once.
    (120, 160, 24, 24, 1), // Block 2 mid (residual block)
    (60, 80, 64, 64, 1),   // Block 3 mid / post-FPN
    (60, 80, 24, 64, 2),   // Block 3 entry (stride-2 transition)
    (30, 40, 64, 64, 1),   // Block 4 mid
    (15, 20, 128, 128, 1), // Block 5 mid
];

const CONV1X1_SHAPES: &[(usize, usize, usize, usize)] = &[
    // (h, w, c_in, c_out)
    (60, 80, 64, 64),  // descriptor head / block-fusion finale
    (60, 80, 64, 65),  // keypoint head final (Conv 64->65)
    (60, 80, 128, 64), // Block 5 finale
    (60, 80, 64, 1),   // reliability head finale
];

fn alloc_random(n: usize, seed: u32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(2654435761).wrapping_add(1);
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state as f32) / (u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

fn bench_conv3x3(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv3x3");
    for &(h, w, c_in, c_out, stride) in CONV3X3_SHAPES {
        let h_out = h / stride;
        let w_out = w / stride;
        let input = alloc_random(h * w * c_in, 11);
        let weights = alloc_random(c_out * 9 * c_in, 13);
        let bias = alloc_random(c_out, 17);
        let mut output = vec![0.0f32; h_out * w_out * c_out];

        let ops = h_out * w_out * c_out * 9 * c_in * 2; // 1 mul + 1 add per MAC
        group.throughput(Throughput::Elements(ops as u64));

        let id = format!("{h}x{w}_c{c_in}x{c_out}_s{stride}");

        group.bench_with_input(BenchmarkId::new("scalar", &id), &id, |b, _| {
            b.iter(|| {
                let args = Conv3x3Args {
                    input: &input,
                    residual: None,
                    weights: &weights,
                    bias: &bias,
                    h_in: h,
                    w_in: w,
                    c_in,
                    c_out,
                    activation: Activation::Relu,
                };
                if stride == 1 {
                    scalar::conv3x3_relu_nhwc(&args, black_box(&mut output));
                } else {
                    scalar::conv3x3_s2_relu_nhwc(&args, black_box(&mut output));
                }
            })
        });

        #[cfg(target_arch = "aarch64")]
        group.bench_with_input(BenchmarkId::new("neon", &id), &id, |b, _| {
            b.iter(|| {
                let args = Conv3x3Args {
                    input: &input,
                    residual: None,
                    weights: &weights,
                    bias: &bias,
                    h_in: h,
                    w_in: w,
                    c_in,
                    c_out,
                    activation: Activation::Relu,
                };
                if stride == 1 {
                    neon::conv3x3_relu_nhwc(&args, black_box(&mut output));
                } else {
                    neon::conv3x3_s2_relu_nhwc(&args, black_box(&mut output));
                }
            })
        });
    }
    group.finish();
}

fn bench_conv1x1(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv1x1");
    for &(h, w, c_in, c_out) in CONV1X1_SHAPES {
        let input = alloc_random(h * w * c_in, 21);
        let weights = alloc_random(c_out * c_in, 23);
        let bias = alloc_random(c_out, 27);
        let mut output = vec![0.0f32; h * w * c_out];

        let ops = h * w * c_out * c_in * 2;
        group.throughput(Throughput::Elements(ops as u64));

        let id = format!("{h}x{w}_c{c_in}x{c_out}");

        group.bench_with_input(BenchmarkId::new("scalar", &id), &id, |b, _| {
            b.iter(|| {
                let args = Conv1x1Args {
                    input: &input,
                    weights: &weights,
                    bias: &bias,
                    h,
                    w,
                    c_in,
                    c_out,
                    activation: Activation::Relu,
                };
                scalar::conv1x1_nhwc(&args, black_box(&mut output));
            })
        });

        #[cfg(target_arch = "aarch64")]
        group.bench_with_input(BenchmarkId::new("neon", &id), &id, |b, _| {
            b.iter(|| {
                let args = Conv1x1Args {
                    input: &input,
                    weights: &weights,
                    bias: &bias,
                    h,
                    w,
                    c_in,
                    c_out,
                    activation: Activation::Relu,
                };
                neon::conv1x1_nhwc(&args, black_box(&mut output));
            })
        });
    }
    group.finish();
}

/// Simulated end-to-end XFeat forward pass at 480×640 — every layer in the
/// upstream `modules/model.py` graph called at its real shape, in order,
/// with random weights. Output is meaningless (random weights), timing is real.
///
/// Not a substitute for `XFeat::extract` once the graph is wired — this is the
/// "what would the model cost if we just called the kernels?" number, which is
/// the floor on what a wired-up `extract()` can hit.
fn bench_e2e_480x640(c: &mut Criterion) {
    let mut group = c.benchmark_group("xfeat_e2e_480x640");
    group.sample_size(20); // each iter does ~1-2k FMA ops; 20 samples is plenty

    // Pre-allocate every buffer the forward pass touches. Sized to the largest
    // intermediate per ping-pong slot — the model never needs both buffers at
    // peak simultaneously.
    let alloc = |n: usize, seed: u32| alloc_random(n, seed);
    let alloc_bias = |c_out: usize, seed: u32| alloc_random(c_out, seed);

    // ── Block 1: input is the InstanceNorm'd grayscale at 480×640×1.
    let gray = alloc(480 * 640, 1);
    let mut b1_l1 = vec![0.0f32; 480 * 640 * 4];
    let mut b1_l2 = vec![0.0f32; 240 * 320 * 8];
    let mut b1_l3 = vec![0.0f32; 240 * 320 * 8];
    let mut b1_l4 = vec![0.0f32; 120 * 160 * 24];
    let w_b1_l1 = alloc(4 * 9, 100);
    let b_b1_l1 = alloc_bias(4, 101);
    let w_b1_l2 = alloc(8 * 9 * 4, 102);
    let b_b1_l2 = alloc_bias(8, 103);
    let w_b1_l3 = alloc(8 * 9 * 8, 104);
    let b_b1_l3 = alloc_bias(8, 105);
    let w_b1_l4 = alloc(24 * 9 * 8, 106);
    let b_b1_l4 = alloc_bias(24, 107);

    // Skip1: AvgPool4 + Conv1x1(1→24)
    let mut skip_pooled = vec![0.0f32; 120 * 160];
    let mut skip_out = vec![0.0f32; 120 * 160 * 24];
    let w_skip = alloc(24, 200);
    let b_skip = alloc_bias(24, 201);

    // Block 2: 2× conv3x3 c24→c24 with residual fusion before block.
    let mut b2_l1 = vec![0.0f32; 120 * 160 * 24];
    let mut b2_l2 = vec![0.0f32; 120 * 160 * 24];
    let w_b2_l1 = alloc(24 * 9 * 24, 300);
    let b_b2_l1 = alloc_bias(24, 301);
    let w_b2_l2 = alloc(24 * 9 * 24, 302);
    let b_b2_l2 = alloc_bias(24, 303);

    // Block 3.
    let mut b3_l1 = vec![0.0f32; 60 * 80 * 64];
    let mut b3_l2 = vec![0.0f32; 60 * 80 * 64];
    let mut b3_l3 = vec![0.0f32; 60 * 80 * 64];
    let w_b3_l1 = alloc(64 * 9 * 24, 400);
    let b_b3_l1 = alloc_bias(64, 401);
    let w_b3_l2 = alloc(64 * 9 * 64, 402);
    let b_b3_l2 = alloc_bias(64, 403);
    let w_b3_l3 = alloc(64 * 64, 404);
    let b_b3_l3 = alloc_bias(64, 405);

    // Block 4.
    let mut b4_l1 = vec![0.0f32; 30 * 40 * 64];
    let mut b4_l2 = vec![0.0f32; 30 * 40 * 64];
    let mut b4_l3 = vec![0.0f32; 30 * 40 * 64];
    let w_b4_l1 = alloc(64 * 9 * 64, 500);
    let b_b4_l1 = alloc_bias(64, 501);
    let w_b4_l2 = alloc(64 * 9 * 64, 502);
    let b_b4_l2 = alloc_bias(64, 503);
    let w_b4_l3 = alloc(64 * 9 * 64, 504);
    let b_b4_l3 = alloc_bias(64, 505);

    // Block 5.
    let mut b5_l1 = vec![0.0f32; 15 * 20 * 128];
    let mut b5_l2 = vec![0.0f32; 15 * 20 * 128];
    let mut b5_l3 = vec![0.0f32; 15 * 20 * 128];
    let mut b5_l4 = vec![0.0f32; 15 * 20 * 64];
    let w_b5_l1 = alloc(128 * 9 * 64, 600);
    let b_b5_l1 = alloc_bias(128, 601);
    let w_b5_l2 = alloc(128 * 9 * 128, 602);
    let b_b5_l2 = alloc_bias(128, 603);
    let w_b5_l3 = alloc(128 * 9 * 128, 604);
    let b_b5_l3 = alloc_bias(128, 605);
    let w_b5_l4 = alloc(64 * 128, 606);
    let b_b5_l4 = alloc_bias(64, 607);

    // FPN: upsample x4 and x5 to x3's resolution then add.
    let mut x4_up = vec![0.0f32; 60 * 80 * 64];
    let mut x5_up = vec![0.0f32; 60 * 80 * 64];

    // block_fusion: 2× conv3x3 c64→c64 + Conv2d(64→64, k=1).
    let mut bf_l1 = vec![0.0f32; 60 * 80 * 64];
    let mut bf_l2 = vec![0.0f32; 60 * 80 * 64];
    let mut feats = vec![0.0f32; 60 * 80 * 64];
    let w_bf_l1 = alloc(64 * 9 * 64, 700);
    let b_bf_l1 = alloc_bias(64, 701);
    let w_bf_l2 = alloc(64 * 9 * 64, 702);
    let b_bf_l2 = alloc_bias(64, 703);
    let w_bf_l3 = alloc(64 * 64, 704);
    let b_bf_l3 = alloc_bias(64, 705);

    // Keypoint head: unfold to (60, 80, 64), 3× conv1x1 c64→c64, conv1x1 64→65.
    let mut kp_unfold = vec![0.0f32; 60 * 80 * 64];
    let mut kp_l1 = vec![0.0f32; 60 * 80 * 64];
    let mut kp_l2 = vec![0.0f32; 60 * 80 * 64];
    let mut kp_l3 = vec![0.0f32; 60 * 80 * 64];
    let mut kp_logits = vec![0.0f32; 60 * 80 * 65];
    let w_kp_l1 = alloc(64 * 64, 800);
    let b_kp_l1 = alloc_bias(64, 801);
    let w_kp_l2 = alloc(64 * 64, 802);
    let b_kp_l2 = alloc_bias(64, 803);
    let w_kp_l3 = alloc(64 * 64, 804);
    let b_kp_l3 = alloc_bias(64, 805);
    let w_kp_out = alloc(65 * 64, 806);
    let b_kp_out = alloc_bias(65, 807);

    // Reliability head: 2× conv1x1 c64→c64 + conv1x1 c64→c1 + sigmoid.
    let mut rel_l1 = vec![0.0f32; 60 * 80 * 64];
    let mut rel_l2 = vec![0.0f32; 60 * 80 * 64];
    let mut rel_out = vec![0.0f32; 60 * 80];
    let w_rel_l1 = alloc(64 * 64, 900);
    let b_rel_l1 = alloc_bias(64, 901);
    let w_rel_l2 = alloc(64 * 64, 902);
    let b_rel_l2 = alloc_bias(64, 903);
    let w_rel_out = alloc(64, 904);
    let b_rel_out = alloc_bias(1, 905);

    group.bench_function("forward", |b| {
        b.iter(|| {
            // ── Block 1
            scalar::conv3x3_relu_nhwc(
                &Conv3x3Args {
                    input: &gray,
                    residual: None,
                    weights: &w_b1_l1,
                    bias: &b_b1_l1,
                    h_in: 480,
                    w_in: 640,
                    c_in: 1,
                    c_out: 4,
                    activation: Activation::Relu,
                },
                &mut b1_l1,
            );

            run_conv3x3(&b1_l1, &mut b1_l2, &w_b1_l2, &b_b1_l2, 480, 640, 4, 8, 2);
            run_conv3x3(&b1_l2, &mut b1_l3, &w_b1_l3, &b_b1_l3, 240, 320, 8, 8, 1);
            run_conv3x3(&b1_l3, &mut b1_l4, &w_b1_l4, &b_b1_l4, 240, 320, 8, 24, 2);

            // ── Skip1
            scalar::avgpool_4x4_s4(&gray, &mut skip_pooled, 480, 640, 1);
            scalar::conv1x1_nhwc(
                &Conv1x1Args {
                    input: &skip_pooled,
                    weights: &w_skip,
                    bias: &b_skip,
                    h: 120,
                    w: 160,
                    c_in: 1,
                    c_out: 24,
                    activation: Activation::Identity,
                },
                &mut skip_out,
            );

            // ── Block 2 (residual = b1_l4 + skip_out)
            run_conv3x3_residual(
                &b1_l4,
                &mut b2_l1,
                Some(&skip_out),
                &w_b2_l1,
                &b_b2_l1,
                120,
                160,
                24,
                24,
            );
            run_conv3x3(&b2_l1, &mut b2_l2, &w_b2_l2, &b_b2_l2, 120, 160, 24, 24, 1);

            // ── Block 3
            run_conv3x3(&b2_l2, &mut b3_l1, &w_b3_l1, &b_b3_l1, 120, 160, 24, 64, 2);
            run_conv3x3(&b3_l1, &mut b3_l2, &w_b3_l2, &b_b3_l2, 60, 80, 64, 64, 1);
            run_conv1x1(&b3_l2, &mut b3_l3, &w_b3_l3, &b_b3_l3, 60, 80, 64, 64);

            // ── Block 4
            run_conv3x3(&b3_l3, &mut b4_l1, &w_b4_l1, &b_b4_l1, 60, 80, 64, 64, 2);
            run_conv3x3(&b4_l1, &mut b4_l2, &w_b4_l2, &b_b4_l2, 30, 40, 64, 64, 1);
            run_conv3x3(&b4_l2, &mut b4_l3, &w_b4_l3, &b_b4_l3, 30, 40, 64, 64, 1);

            // ── Block 5
            run_conv3x3(&b4_l3, &mut b5_l1, &w_b5_l1, &b_b5_l1, 30, 40, 64, 128, 2);
            run_conv3x3(&b5_l1, &mut b5_l2, &w_b5_l2, &b_b5_l2, 15, 20, 128, 128, 1);
            run_conv3x3(&b5_l2, &mut b5_l3, &w_b5_l3, &b_b5_l3, 15, 20, 128, 128, 1);
            run_conv1x1(&b5_l3, &mut b5_l4, &w_b5_l4, &b_b5_l4, 15, 20, 128, 64);

            // ── FPN: bilinear upsample x4 (30×40) and x5 (15×20) to x3 (60×80), then add.
            scalar::bilinear_upsample(&b4_l3, &mut x4_up, 30, 40, 64, 60, 80);
            scalar::bilinear_upsample(&b5_l4, &mut x5_up, 15, 20, 64, 60, 80);
            // The add-3 happens in-place on x3 (b3_l3). Take its content as the start.
            let mut fpn_sum = b3_l3.clone();
            scalar::add3_inplace(&mut fpn_sum, &x4_up, &x5_up);

            // ── block_fusion: 2× conv3x3 + conv1x1 final (no activation on last).
            run_conv3x3(&fpn_sum, &mut bf_l1, &w_bf_l1, &b_bf_l1, 60, 80, 64, 64, 1);
            run_conv3x3(&bf_l1, &mut bf_l2, &w_bf_l2, &b_bf_l2, 60, 80, 64, 64, 1);
            run_conv1x1_id(&bf_l2, &mut feats, &w_bf_l3, &b_bf_l3, 60, 80, 64, 64);

            // ── Keypoint head
            scalar::unfold_8x8(&gray, &mut kp_unfold, 480, 640);
            run_conv1x1(&kp_unfold, &mut kp_l1, &w_kp_l1, &b_kp_l1, 60, 80, 64, 64);
            run_conv1x1(&kp_l1, &mut kp_l2, &w_kp_l2, &b_kp_l2, 60, 80, 64, 64);
            run_conv1x1(&kp_l2, &mut kp_l3, &w_kp_l3, &b_kp_l3, 60, 80, 64, 64);
            run_conv1x1_id(&kp_l3, &mut kp_logits, &w_kp_out, &b_kp_out, 60, 80, 64, 65);
            scalar::channel_softmax(&mut kp_logits, 60, 80, 65);

            // ── Reliability head
            run_conv1x1(&feats, &mut rel_l1, &w_rel_l1, &b_rel_l1, 60, 80, 64, 64);
            run_conv1x1(&rel_l1, &mut rel_l2, &w_rel_l2, &b_rel_l2, 60, 80, 64, 64);
            run_conv1x1_id(&rel_l2, &mut rel_out, &w_rel_out, &b_rel_out, 60, 80, 64, 1);
            scalar::sigmoid_inplace(&mut rel_out);

            // ── L2-norm descriptors
            let mut desc = feats.clone();
            scalar::l2_normalize_channel(&mut desc, 60, 80, 64);

            std::hint::black_box(&kp_logits);
            std::hint::black_box(&desc);
            std::hint::black_box(&rel_out);
        });
    });

    group.finish();
}

/// Helper: dispatch through the same vtable as the model would.
#[allow(clippy::too_many_arguments)]
fn run_conv3x3(
    input: &[f32],
    output: &mut [f32],
    weights: &[f32],
    bias: &[f32],
    h: usize,
    w: usize,
    c_in: usize,
    c_out: usize,
    stride: usize,
) {
    let args = Conv3x3Args {
        input,
        residual: None,
        weights,
        bias,
        h_in: h,
        w_in: w,
        c_in,
        c_out,
        activation: Activation::Relu,
    };
    #[cfg(target_arch = "aarch64")]
    {
        if stride == 1 {
            neon::conv3x3_relu_nhwc(&args, output);
        } else {
            neon::conv3x3_s2_relu_nhwc(&args, output);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        if stride == 1 {
            scalar::conv3x3_relu_nhwc(&args, output);
        } else {
            scalar::conv3x3_s2_relu_nhwc(&args, output);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_conv3x3_residual(
    input: &[f32],
    output: &mut [f32],
    residual: Option<&[f32]>,
    weights: &[f32],
    bias: &[f32],
    h: usize,
    w: usize,
    c_in: usize,
    c_out: usize,
) {
    let args = Conv3x3Args {
        input,
        residual,
        weights,
        bias,
        h_in: h,
        w_in: w,
        c_in,
        c_out,
        activation: Activation::Relu,
    };
    #[cfg(target_arch = "aarch64")]
    neon::conv3x3_relu_nhwc(&args, output);
    #[cfg(not(target_arch = "aarch64"))]
    scalar::conv3x3_relu_nhwc(&args, output);
}

#[allow(clippy::too_many_arguments)]
fn run_conv1x1(
    input: &[f32],
    output: &mut [f32],
    weights: &[f32],
    bias: &[f32],
    h: usize,
    w: usize,
    c_in: usize,
    c_out: usize,
) {
    let args = Conv1x1Args {
        input,
        weights,
        bias,
        h,
        w,
        c_in,
        c_out,
        activation: Activation::Relu,
    };
    #[cfg(target_arch = "aarch64")]
    neon::conv1x1_nhwc(&args, output);
    #[cfg(not(target_arch = "aarch64"))]
    scalar::conv1x1_nhwc(&args, output);
}

#[allow(clippy::too_many_arguments)]
fn run_conv1x1_id(
    input: &[f32],
    output: &mut [f32],
    weights: &[f32],
    bias: &[f32],
    h: usize,
    w: usize,
    c_in: usize,
    c_out: usize,
) {
    let args = Conv1x1Args {
        input,
        weights,
        bias,
        h,
        w,
        c_in,
        c_out,
        activation: Activation::Identity,
    };
    #[cfg(target_arch = "aarch64")]
    neon::conv1x1_nhwc(&args, output);
    #[cfg(not(target_arch = "aarch64"))]
    scalar::conv1x1_nhwc(&args, output);
}

criterion_group!(benches, bench_conv3x3, bench_conv1x1, bench_e2e_480x640);
criterion_main!(benches);
