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

criterion_group!(benches, bench_conv3x3, bench_conv1x1);
criterion_main!(benches);
