//! Cross-backend parity: every SIMD kernel must produce the same output as
//! the scalar reference within a tight tolerance, on every shape the model
//! actually uses.
//!
//! Pseudo-random inputs are seeded so failures are deterministic.

use kornia_xfeat::ops::{scalar, Activation, Conv1x1Args, Conv3x3Args};

#[cfg(target_arch = "aarch64")]
use kornia_xfeat::ops::neon;

const TOL_ABS: f32 = 1e-4;
const TOL_REL: f32 = 1e-4;

fn lcg_seed(seed: u32) -> impl FnMut() -> f32 {
    let mut state = seed.wrapping_mul(2654435761).wrapping_add(1);
    move || {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        // Map to roughly [-1, 1].
        (state as f32) / (u32::MAX as f32) * 2.0 - 1.0
    }
}

fn approx_eq(a: f32, b: f32) -> bool {
    (a - b).abs() <= TOL_ABS + TOL_REL * a.abs().max(b.abs())
}

fn assert_buffers_match(scalar_out: &[f32], simd_out: &[f32], label: &str) {
    assert_eq!(
        scalar_out.len(),
        simd_out.len(),
        "[{label}] length mismatch"
    );
    let mut worst_idx = 0usize;
    let mut worst_diff = 0.0f32;
    for (i, (&a, &b)) in scalar_out.iter().zip(simd_out.iter()).enumerate() {
        let d = (a - b).abs();
        if d > worst_diff {
            worst_diff = d;
            worst_idx = i;
        }
        if !approx_eq(a, b) {
            panic!("[{label}] mismatch at idx {i}: scalar={a}, simd={b}, diff={d}");
        }
    }
    eprintln!("[{label}] OK — worst diff {worst_diff:.3e} at idx {worst_idx}");
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_conv1x1_parity_60x80x64x64() {
    let (h, w, c_in, c_out) = (60, 80, 64, 64);
    let mut r = lcg_seed(1);
    let input: Vec<f32> = (0..h * w * c_in).map(|_| r()).collect();
    let weights: Vec<f32> = (0..c_out * c_in).map(|_| r()).collect();
    let bias: Vec<f32> = (0..c_out).map(|_| r()).collect();

    for act in [Activation::Relu, Activation::Sigmoid, Activation::Identity] {
        let args = Conv1x1Args {
            input: &input,
            weights: &weights,
            bias: &bias,
            h,
            w,
            c_in,
            c_out,
            activation: act,
        };
        let mut s_out = vec![0.0f32; h * w * c_out];
        let mut n_out = vec![0.0f32; h * w * c_out];
        scalar::conv1x1_nhwc(&args, &mut s_out);
        neon::conv1x1_nhwc(&args, &mut n_out);
        assert_buffers_match(&s_out, &n_out, &format!("conv1x1 act={:?}", act));
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_conv1x1_parity_cin4_now_takes_neon_path() {
    // c_in=4 is a multiple of 4 → goes through the Phase-2-only path
    // (no Phase 1 because c16 = 0). Verifies the tail loop on its own.
    parity_conv1x1(8, 8, 4, 8, Activation::Relu);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_conv1x1_parity_cin24_relu() {
    // c_in=24 is the XFeat Block-2-mid c_in. Exercises Phase 1 (one 16-block)
    // + Phase 2 (two f32x4 chunks). This is the layer the fallback fix targets.
    parity_conv1x1(120, 160, 24, 24, Activation::Relu);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_conv1x1_parity_falls_back_when_cin_lt_4() {
    let (h, w, c_in, c_out) = (8, 8, 3, 8);
    let mut r = lcg_seed(7);
    let input: Vec<f32> = (0..h * w * c_in).map(|_| r()).collect();
    let weights: Vec<f32> = (0..c_out * c_in).map(|_| r()).collect();
    let bias: Vec<f32> = (0..c_out).map(|_| r()).collect();
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
    let mut s_out = vec![0.0f32; h * w * c_out];
    let mut n_out = vec![0.0f32; h * w * c_out];
    scalar::conv1x1_nhwc(&args, &mut s_out);
    neon::conv1x1_nhwc(&args, &mut n_out);
    assert_buffers_match(&s_out, &n_out, "conv1x1 c_in=3 fallback");
}

#[cfg(target_arch = "aarch64")]
fn parity_conv1x1(h: usize, w: usize, c_in: usize, c_out: usize, act: Activation) {
    let mut r = lcg_seed(5);
    let input: Vec<f32> = (0..h * w * c_in).map(|_| r()).collect();
    let weights: Vec<f32> = (0..c_out * c_in).map(|_| r()).collect();
    let bias: Vec<f32> = (0..c_out).map(|_| r()).collect();
    let args = Conv1x1Args {
        input: &input,
        weights: &weights,
        bias: &bias,
        h,
        w,
        c_in,
        c_out,
        activation: act,
    };
    let mut s_out = vec![0.0f32; h * w * c_out];
    let mut n_out = vec![0.0f32; h * w * c_out];
    scalar::conv1x1_nhwc(&args, &mut s_out);
    neon::conv1x1_nhwc(&args, &mut n_out);
    assert_buffers_match(
        &s_out,
        &n_out,
        &format!("conv1x1 {h}x{w} c_in={c_in} c_out={c_out}"),
    );
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_conv3x3_parity_60x80x64x64_relu() {
    parity_conv3x3(60, 80, 64, 64, 1, Activation::Relu, None);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_conv3x3_parity_120x160_c24x24_block2_mid() {
    // Block 2 mid — the c_in=24 case the Phase-2 tail loop was added for.
    parity_conv3x3(120, 160, 24, 24, 1, Activation::Relu, None);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_conv3x3_parity_60x80_c24x64_s2_block3_entry() {
    // Block 3 entry — c_in=24 stride-2 transition.
    parity_conv3x3(60, 80, 24, 64, 2, Activation::Relu, None);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_conv3x3_parity_30x40x64x64_stride1() {
    parity_conv3x3(30, 40, 64, 64, 1, Activation::Relu, None);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_conv3x3_parity_60x80x64x64_stride2() {
    parity_conv3x3(60, 80, 64, 64, 2, Activation::Relu, None);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_conv3x3_parity_with_residual() {
    let (h, w, c_in, c_out, stride) = (60, 80, 64, 64, 1);
    let mut r = lcg_seed(3);
    let input: Vec<f32> = (0..h * w * c_in).map(|_| r()).collect();
    let weights: Vec<f32> = (0..c_out * 9 * c_in).map(|_| r()).collect();
    let bias: Vec<f32> = (0..c_out).map(|_| r()).collect();
    let residual: Vec<f32> = (0..(h / stride) * (w / stride) * c_out)
        .map(|_| r())
        .collect();
    let args = Conv3x3Args {
        input: &input,
        residual: Some(&residual),
        weights: &weights,
        bias: &bias,
        h_in: h,
        w_in: w,
        c_in,
        c_out,
        activation: Activation::Relu,
    };
    let mut s_out = vec![0.0f32; (h / stride) * (w / stride) * c_out];
    let mut n_out = vec![0.0f32; (h / stride) * (w / stride) * c_out];
    scalar::conv3x3_relu_nhwc(&args, &mut s_out);
    neon::conv3x3_relu_nhwc(&args, &mut n_out);
    assert_buffers_match(&s_out, &n_out, "conv3x3 residual");
}

#[cfg(target_arch = "aarch64")]
fn parity_conv3x3(
    h: usize,
    w: usize,
    c_in: usize,
    c_out: usize,
    stride: usize,
    act: Activation,
    _label: Option<&str>,
) {
    let mut r = lcg_seed(2);
    let input: Vec<f32> = (0..h * w * c_in).map(|_| r()).collect();
    let weights: Vec<f32> = (0..c_out * 9 * c_in).map(|_| r()).collect();
    let bias: Vec<f32> = (0..c_out).map(|_| r()).collect();
    let args = Conv3x3Args {
        input: &input,
        residual: None,
        weights: &weights,
        bias: &bias,
        h_in: h,
        w_in: w,
        c_in,
        c_out,
        activation: act,
    };
    let mut s_out = vec![0.0f32; (h / stride) * (w / stride) * c_out];
    let mut n_out = vec![0.0f32; (h / stride) * (w / stride) * c_out];
    if stride == 1 {
        scalar::conv3x3_relu_nhwc(&args, &mut s_out);
        neon::conv3x3_relu_nhwc(&args, &mut n_out);
    } else {
        scalar::conv3x3_s2_relu_nhwc(&args, &mut s_out);
        neon::conv3x3_s2_relu_nhwc(&args, &mut n_out);
    }
    assert_buffers_match(
        &s_out,
        &n_out,
        &format!("conv3x3 {h}x{w} c_in={c_in} c_out={c_out} stride={stride}"),
    );
}

#[cfg(not(target_arch = "aarch64"))]
#[test]
fn neon_skipped_on_non_aarch64() {
    eprintln!("[skip] NEON parity tests only run on aarch64");
}
