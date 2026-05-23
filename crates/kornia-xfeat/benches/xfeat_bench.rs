//! Criterion benches for kornia-xfeat.
//!
//! Currently a single placeholder that times the scalar reference conv on a
//! small synthetic input. Real per-kernel + end-to-end benches land
//! alongside the SIMD kernels (since "did we get faster" only makes sense
//! once there's a real kernel to be faster than).

use criterion::{criterion_group, criterion_main, Criterion};
use kornia_xfeat::ops::{Activation, Conv1x1Args};

fn bench_conv1x1_64x64(c: &mut Criterion) {
    let h = 60;
    let w = 80;
    let c_in = 64;
    let c_out = 64;
    let input: Vec<f32> = (0..h * w * c_in).map(|i| (i as f32) * 1e-3).collect();
    let weights: Vec<f32> = (0..c_out * c_in).map(|i| (i as f32) * 1e-4).collect();
    let bias = vec![0.1f32; c_out];
    let mut output = vec![0.0f32; h * w * c_out];

    c.bench_function("conv1x1_60x80_64x64_scalar", |b| {
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
            kornia_xfeat::ops::scalar::conv1x1_nhwc(&args, &mut output);
        })
    });
}

criterion_group!(benches, bench_conv1x1_64x64);
criterion_main!(benches);
