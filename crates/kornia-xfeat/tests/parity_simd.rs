//! Cross-backend parity: every SIMD kernel must produce the same output as
//! the scalar reference within a tight tolerance, on every shape the model
//! actually uses.
//!
//! Pseudo-random inputs are seeded so failures are deterministic.
//!
//! # Exactness invariant
//!
//! **Goal:** scalar and NEON produce bit-identical results for every path
//! where `c_in % 4 == 0`.
//!
//! **Condition for bit-exactness:** scalar.rs must use the same 4-way grouped
//! `f32::mul_add` reduction that NEON's `vfmaq_f32` / `vaddvq_f32` implements.
//! When both backends execute the same FMA-based reduction tree in the same
//! order, IEEE 754 round-to-nearest guarantees identical f32 bit patterns.
//!
//! **Current state:** scalar.rs still uses a simple sequential `acc += a * b`
//! loop (no FMA fusion, different accumulation order). Until that rewrite lands
//! the `c_in % 4 == 0` paths tolerate up to 1e-6. Once the canonical scalar
//! reduction is merged the TODO comments below should be replaced with
//! `assert_eq!` calls.
//!
//! **Already bit-exact:** the `c_in % 4 != 0` fallback (c_in ∈ {1, 2, 3})
//! routes NEON directly to the scalar implementation, so the outputs are
//! trivially identical; those tests use `assert_eq!`.

use kornia_xfeat::ops::{scalar, Activation, Conv1x1Args, Conv3x3Args};

#[cfg(target_arch = "aarch64")]
use kornia_xfeat::ops::neon;

/// Tolerance for conv1x1 paths where scalar hasn't yet adopted the canonical
/// 4-way mul_add reduction. Worst observed diff for conv1x1 is ~9.5e-7; this
/// 1e-6 ceiling is the target we maintain until bit-exact equality is demanded.
///
/// TODO(scalar-canonical-reduction): tighten to `assert_eq!` once scalar.rs
/// uses the same FMA-grouped accumulation as NEON.
#[cfg(target_arch = "aarch64")]
const TOL_CONV1X1_PENDING: f32 = 1e-6;

/// Tolerance for conv3x3 paths. The 3×3 spatial loop accumulates 9 tile FMA
/// reductions into the same four vector accumulators; the longer reduction
/// chain widens the scalar/NEON numerical gap compared to conv1x1. Worst
/// observed diff is ~3.6e-5 (c_in=64, full-resolution). Still 3× tighter
/// than the previous 1e-4 budget.
///
/// Tests that still needed relaxing beyond 1e-6 (because the scalar sequential
/// accumulation diverges further with the 9-tap spatial loop):
///   - neon_conv3x3_parity_60x80x64x64_relu         (worst ~3.6e-5)
///   - neon_conv3x3_parity_60x80x64x64_stride2       (worst ~3.1e-5)
///   - neon_conv3x3_parity_30x40x64x64_stride1       (worst ~3.1e-5)
///   - neon_conv3x3_parity_120x160_c24x24_block2_mid (worst ~1.1e-5)
///   - neon_conv3x3_parity_60x80_c24x64_s2           (worst ~9.5e-6)
///   - neon_conv3x3_parity_with_residual              (worst ~4.0e-5)
///
/// TODO(scalar-canonical-reduction): tighten to `assert_eq!` once scalar.rs
/// uses the same FMA-grouped accumulation as NEON.
#[cfg(target_arch = "aarch64")]
const TOL_CONV3X3_PENDING: f32 = 5e-5;

fn lcg_seed(seed: u32) -> impl FnMut() -> f32 {
    let mut state = seed.wrapping_mul(2654435761).wrapping_add(1);
    move || {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        // Map to roughly [-1, 1].
        (state as f32) / (u32::MAX as f32) * 2.0 - 1.0
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Assert that two f32 slices are bit-identical.
/// Used for paths that are always scalar on both sides (NEON falls back to
/// scalar::conv1x1_nhwc / conv3x3_relu_nhwc when c_in % 4 != 0).
#[cfg(target_arch = "aarch64")]
fn assert_buffers_exact(scalar_out: &[f32], simd_out: &[f32], label: &str) {
    assert_eq!(
        scalar_out.len(),
        simd_out.len(),
        "[{label}] length mismatch"
    );
    for (i, (&a, &b)) in scalar_out.iter().zip(simd_out.iter()).enumerate() {
        assert_eq!(
            a,
            b,
            "[{label}] bit mismatch at idx {i}: scalar={a:?} ({:#010x}), simd={b:?} ({:#010x})",
            a.to_bits(),
            b.to_bits(),
        );
    }
    eprintln!("[{label}] OK — bit-exact ({} elements)", scalar_out.len());
}

/// Assert that two f32 slices agree within `tol` (absolute).
/// Used while scalar.rs still uses the sequential-accumulation reduction.
/// TODO(scalar-canonical-reduction): replace callers with assert_buffers_exact.
#[cfg(target_arch = "aarch64")]
fn assert_buffers_within_tol(scalar_out: &[f32], simd_out: &[f32], label: &str, tol: f32) {
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
        if d > tol {
            panic!(
                "[{label}] diff {d:.3e} > tol {tol:.3e} at idx {i}: \
                 scalar={a} ({:#010x}), simd={b} ({:#010x})",
                a.to_bits(),
                b.to_bits(),
            );
        }
    }
    eprintln!("[{label}] OK — worst diff {worst_diff:.3e} at idx {worst_idx}");
}

// ---------------------------------------------------------------------------
// conv1x1 tests
// ---------------------------------------------------------------------------

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
        // TODO(scalar-canonical-reduction): replace with assert_buffers_exact once
        // scalar uses the 4-way mul_add reduction. Observed worst diff: ~9.5e-7.
        assert_buffers_within_tol(
            &s_out,
            &n_out,
            &format!("conv1x1 act={:?}", act),
            TOL_CONV1X1_PENDING,
        );
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_conv1x1_parity_cin4_now_takes_neon_path() {
    // c_in=4 is a multiple of 4 → goes through the Phase-2-only path
    // (no Phase 1 because c16 = 0). Verifies the tail loop on its own.
    // TODO(scalar-canonical-reduction): use assert_buffers_exact. Worst diff: ~2.4e-7.
    parity_conv1x1(8, 8, 4, 8, Activation::Relu);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_conv1x1_parity_cin24_relu() {
    // c_in=24 is the XFeat Block-2-mid c_in. Exercises Phase 1 (one 16-block)
    // + Phase 2 (two f32x4 chunks). This is the layer the fallback fix targets.
    // TODO(scalar-canonical-reduction): use assert_buffers_exact. Worst diff: ~9.5e-7.
    parity_conv1x1(120, 160, 24, 24, Activation::Relu);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_conv1x1_parity_falls_back_when_cin_lt_4() {
    // c_in=3 → NEON delegates to scalar::conv1x1_nhwc. Both sides execute
    // identical scalar code, so the output must be bit-exact.
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
    // NEON calls scalar directly for c_in % 4 != 0 → must be bit-identical.
    assert_buffers_exact(&s_out, &n_out, "conv1x1 c_in=3 fallback");
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
    // TODO(scalar-canonical-reduction): replace with assert_buffers_exact.
    assert_buffers_within_tol(
        &s_out,
        &n_out,
        &format!("conv1x1 {h}x{w} c_in={c_in} c_out={c_out}"),
        TOL_CONV1X1_PENDING,
    );
}

// ---------------------------------------------------------------------------
// conv3x3 tests
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_conv3x3_parity_60x80x64x64_relu() {
    // TODO(scalar-canonical-reduction): use assert_buffers_exact. Worst diff: ~3.6e-5.
    parity_conv3x3(60, 80, 64, 64, 1, Activation::Relu, None);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_conv3x3_parity_120x160_c24x24_block2_mid() {
    // Block 2 mid — the c_in=24 case the Phase-2 tail loop was added for.
    // TODO(scalar-canonical-reduction): use assert_buffers_exact. Worst diff: ~1.1e-5.
    parity_conv3x3(120, 160, 24, 24, 1, Activation::Relu, None);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_conv3x3_parity_60x80_c24x64_s2_block3_entry() {
    // Block 3 entry — c_in=24 stride-2 transition.
    // TODO(scalar-canonical-reduction): use assert_buffers_exact. Worst diff: ~9.5e-6.
    parity_conv3x3(60, 80, 24, 64, 2, Activation::Relu, None);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_conv3x3_parity_30x40x64x64_stride1() {
    // TODO(scalar-canonical-reduction): use assert_buffers_exact. Worst diff: ~3.1e-5.
    parity_conv3x3(30, 40, 64, 64, 1, Activation::Relu, None);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_conv3x3_parity_60x80x64x64_stride2() {
    // TODO(scalar-canonical-reduction): use assert_buffers_exact. Worst diff: ~3.1e-5.
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
    // TODO(scalar-canonical-reduction): replace with assert_buffers_exact.
    // Worst observed diff: ~4.0e-5. Needed TOL_CONV3X3_PENDING (not 1e-6).
    assert_buffers_within_tol(&s_out, &n_out, "conv3x3 residual", TOL_CONV3X3_PENDING);
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
    // TODO(scalar-canonical-reduction): replace with assert_buffers_exact.
    // conv3x3 has a longer reduction chain (9 taps) → wider scalar/NEON gap.
    assert_buffers_within_tol(
        &s_out,
        &n_out,
        &format!("conv3x3 {h}x{w} c_in={c_in} c_out={c_out} stride={stride}"),
        TOL_CONV3X3_PENDING,
    );
}

#[cfg(not(target_arch = "aarch64"))]
#[test]
fn neon_skipped_on_non_aarch64() {
    eprintln!("[skip] NEON parity tests only run on aarch64");
}
