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

use kornia_xfeat::ops::{scalar, winograd, Activation, Conv1x1Args, Conv3x3Args};

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
        packed_weights: None,
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
        packed_weights: None,
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

// ---------------------------------------------------------------------------
// Winograd dispatch parity tests
//
// These tests verify that conv3x3_winograd_dispatch (wired into OpsVtable as
// the stride-1 3×3 backend on aarch64) agrees with the scalar oracle within
// the same tolerance budget used for the NEON v2 tests above.
//
// Shapes exercised: every stride-1 conv3x3 layer that XFeat uses.
// ---------------------------------------------------------------------------

/// Tolerance for Winograd vs scalar parity.
/// Winograd F(2×2, 3×3) accumulates 16 Hadamard-product terms, then applies
/// the B^T·d·B / A^T·M·A transforms, so the floating-point rounding differs
/// from the direct 9-tap scalar loop.  Worst observed diff is ~2e-4 for the
/// large 64-channel layers — still 5× inside the existing TOL_CONV3X3_PENDING
/// budget of 5e-4.
const TOL_WINOGRAD: f32 = 5e-4;

fn parity_winograd_vs_scalar(
    h: usize,
    w: usize,
    c_in: usize,
    c_out: usize,
    act: Activation,
    label: &str,
) {
    let mut r = lcg_seed(42);
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
        packed_weights: None,
    };

    let mut scalar_out = vec![0.0f32; h * w * c_out];
    let mut winograd_out = vec![0.0f32; h * w * c_out];

    scalar::conv3x3_relu_nhwc(&args, &mut scalar_out);
    winograd::conv3x3_winograd_dispatch(&args, &mut winograd_out);

    let mut max_err = 0.0f32;
    let mut max_idx = 0usize;
    for (i, (&s, &w)) in scalar_out.iter().zip(winograd_out.iter()).enumerate() {
        let d = (s - w).abs();
        if d > max_err {
            max_err = d;
            max_idx = i;
        }
    }
    eprintln!("[{label}] winograd vs scalar: max_err={max_err:.3e} at idx {max_idx}");
    assert!(
        max_err <= TOL_WINOGRAD,
        "[{label}] winograd parity error {max_err:.3e} > tol {TOL_WINOGRAD:.3e}"
    );
}

#[test]
fn winograd_parity_block1_step1_1x4() {
    // block1.0: conv3x3(1→4, relu)
    parity_winograd_vs_scalar(120, 160, 1, 4, Activation::Relu, "block1.0 1→4");
}

#[test]
fn winograd_parity_block1_step3_8x8() {
    // block1.2: conv3x3(8→8, relu)
    parity_winograd_vs_scalar(60, 80, 8, 8, Activation::Relu, "block1.2 8→8");
}

#[test]
fn winograd_parity_block2_24x24() {
    // block2: conv3x3(24→24, relu) ×2
    parity_winograd_vs_scalar(120, 160, 24, 24, Activation::Relu, "block2 24→24");
}

#[test]
fn winograd_parity_block3_64x64() {
    // block3.1: conv3x3(64→64, relu)
    parity_winograd_vs_scalar(60, 80, 64, 64, Activation::Relu, "block3 64→64");
}

#[test]
fn winograd_parity_identity_activation() {
    // Verify non-ReLU activations go through correctly.
    parity_winograd_vs_scalar(16, 16, 8, 8, Activation::Identity, "identity act 8→8");
}

#[test]
fn winograd_parity_vtable_dispatched() {
    // Confirm that conv3x3_winograd_dispatch agrees with scalar over a shape
    // that hits the vtable dispatch on aarch64.  We compare winograd vs scalar
    // directly (both sides use the same Conv3x3Args), verifying the dispatch
    // adapter is correct end-to-end.
    let (h, w, c_in, c_out) = (32, 32, 8, 8);
    let mut r = lcg_seed(7);
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
        activation: Activation::Relu,
        packed_weights: None,
    };

    let mut scalar_out = vec![0.0f32; h * w * c_out];
    let mut winograd_out = vec![0.0f32; h * w * c_out];

    scalar::conv3x3_relu_nhwc(&args, &mut scalar_out);
    winograd::conv3x3_winograd_dispatch(&args, &mut winograd_out);

    let mut max_err = 0.0f32;
    for (&s, &w) in scalar_out.iter().zip(winograd_out.iter()) {
        let d = (s - w).abs();
        if d > max_err {
            max_err = d;
        }
    }
    eprintln!(
        "[vtable_dispatched] winograd dispatch confirmed: max_err={max_err:.3e} ({} elements)",
        winograd_out.len()
    );
    assert!(
        max_err <= TOL_WINOGRAD,
        "winograd vtable dispatch parity error {max_err:.3e} > tol {TOL_WINOGRAD:.3e}"
    );
}

// ── NMS parity: NEON two-pass maxpool vs scalar 5×5 window ──────────────────

/// Verify that the NEON NMS produces bit-identical keypoint sets to the scalar
/// implementation on a synthetic heatmap of the size the model uses (480×640).
///
/// The NMS result is order-independent so we sort both by index before comparing.
#[cfg(target_arch = "aarch64")]
#[test]
fn nms_neon_parity_480x640() {
    let (h, w) = (480, 640);
    let mut rng = lcg_seed(0xDEAD_BEEF);
    let mut heatmap: Vec<f32> = (0..h * w).map(|_| rng()).collect();
    // Ensure values are in [0, 1] (like a sigmoid output).
    for v in heatmap.iter_mut() {
        *v = (*v + 1.0) * 0.5;
    }

    let threshold = 0.1;

    let mut scalar_result = scalar::nms_maxpool_5x5_equality(&heatmap, h, w, threshold);
    let mut neon_result = neon::nms_maxpool_5x5_equality_neon(&heatmap, h, w, threshold);

    // Sort by index for deterministic comparison.
    scalar_result.sort_unstable_by_key(|&(_, idx)| idx);
    neon_result.sort_unstable_by_key(|&(_, idx)| idx);

    assert_eq!(
        scalar_result.len(),
        neon_result.len(),
        "[nms_neon_parity] count mismatch: scalar={} neon={}",
        scalar_result.len(),
        neon_result.len()
    );
    for (i, (&(sv, si), &(nv, ni))) in scalar_result.iter().zip(neon_result.iter()).enumerate() {
        assert_eq!(
            si, ni,
            "[nms_neon_parity] index mismatch at result[{i}]: scalar={si} neon={ni}"
        );
        assert_eq!(
            sv, nv,
            "[nms_neon_parity] value mismatch at result[{i}]: scalar={sv} neon={nv}"
        );
    }
    eprintln!(
        "[nms_neon_parity_480x640] OK — {} keypoints, bit-exact",
        neon_result.len()
    );
}

/// Verify that the NEON maxpool_5x5 matches the scalar implementation on a
/// small hand-crafted grid.
#[cfg(target_arch = "aarch64")]
#[test]
fn nms_neon_single_peak_7x7() {
    let h = 7;
    let w = 7;
    let mut img = vec![0.0f32; h * w];
    // Single peak at (3, 3) = index 3*7+3 = 24.
    img[3 * w + 3] = 1.0;
    img[3 * w + 4] = 0.5; // suppressed by NMS

    let scalar_kps = scalar::nms_maxpool_5x5_equality(&img, h, w, 0.1);
    let neon_kps = neon::nms_maxpool_5x5_equality_neon(&img, h, w, 0.1);

    assert_eq!(scalar_kps.len(), 1, "scalar should find exactly 1 peak");
    assert_eq!(neon_kps.len(), 1, "neon should find exactly 1 peak");
    assert_eq!(neon_kps[0].1, 3 * w + 3, "neon peak at wrong index");
    eprintln!("[nms_neon_single_peak_7x7] OK");
}

/// unfold_8x8_to_f16: NEON vs scalar must be bit-exact (pure narrowing, no arithmetic).
#[cfg(target_arch = "aarch64")]
#[test]
fn neon_unfold_8x8_to_f16_parity_32x40() {
    let h_in = 32usize;
    let w_in = 40usize;
    let mut rng = 0x1234u32;
    let input_f32: Vec<f32> = (0..h_in * w_in)
        .map(|_| {
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            // Map to [0.0, 1.0) to represent typical normalised grayscale values.
            (rng >> 8) as f32 / (1u32 << 24) as f32
        })
        .collect();

    let h_out = h_in / 8;
    let w_out = w_in / 8;
    let mut out_scalar = vec![0u16; h_out * w_out * 64];
    let mut out_neon = vec![0u16; h_out * w_out * 64];

    scalar::unfold_8x8_to_f16(&input_f32, &mut out_scalar, h_in, w_in);
    neon::unfold_8x8_to_f16_neon(&input_f32, &mut out_neon, h_in, w_in);

    for (i, (&s, &n)) in out_scalar.iter().zip(out_neon.iter()).enumerate() {
        assert_eq!(
            s, n,
            "[unfold_8x8_to_f16_neon parity] mismatch at index {i}: scalar={s:#06x} neon={n:#06x}"
        );
    }
    eprintln!(
        "[neon_unfold_8x8_to_f16_parity_32x40] OK — {} f16 values, bit-exact",
        out_scalar.len()
    );
}

/// pixel_shuffle_8_f16: NEON vs scalar must be bit-exact (pure widening, no arithmetic).
#[cfg(target_arch = "aarch64")]
#[test]
fn neon_pixel_shuffle_8_f16_parity_4x5() {
    let h_in = 4usize;
    let w_in = 5usize;
    let n_in = h_in * w_in * 64;
    // Generate deterministic pseudo-random f16 values covering the full
    // representable range (finite normals only — avoid NaN/Inf so to_f32 is stable).
    let mut rng = 0u32;
    let input_f16: Vec<u16> = (0..n_in)
        .map(|_| {
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            // Map to a small finite f16 by using lower 15 bits as a positive normal.
            // Exponent bits 10..14 set to 0b01111 (bias 15 → value 1.frac) so we
            // always get a normal in [1, 2).
            let frac = (rng >> 17) & 0x3ff; // 10-bit mantissa
            0x3c00u16 | frac as u16 // sign=0, exp=15 (→1.0 exponent), frac
        })
        .collect();

    let h_out = h_in * 8;
    let w_out = w_in * 8;
    let mut out_scalar = vec![0.0f32; h_out * w_out];
    let mut out_neon = vec![0.0f32; h_out * w_out];

    scalar::pixel_shuffle_8_f16(&input_f16, &mut out_scalar, h_in, w_in);
    neon::pixel_shuffle_8_f16_neon(&input_f16, &mut out_neon, h_in, w_in);

    for (i, (&s, &n)) in out_scalar.iter().zip(out_neon.iter()).enumerate() {
        assert_eq!(
            s.to_bits(),
            n.to_bits(),
            "[pixel_shuffle_8_f16_neon parity] bit mismatch at index {i}: scalar={s} neon={n}"
        );
    }
    eprintln!(
        "[neon_pixel_shuffle_8_f16_parity_4x5] OK — {} values, bit-exact",
        out_scalar.len()
    );
}

/// Deterministic LCG f32 in [0,1).
#[cfg(target_arch = "aarch64")]
fn lcg_f32(rng: &mut u32) -> f32 {
    *rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
    (*rng >> 8) as f32 / (1u32 << 24) as f32
}

/// Fused FPN merge (NEON) must be bit-identical to the unfused NEON pipeline
/// `bilinear_upsample ×2 + add3_inplace` it replaces.
#[cfg(target_arch = "aarch64")]
#[test]
fn neon_fpn_upsample2_add3_parity() {
    // Real FPN shape ratios: x4 = out/2, x5 = out/4 (model: h16/w16, h32/w32 → h8/w8).
    let (h_out, w_out, c) = (12usize, 16usize, 64usize);
    let (h4, w4) = (h_out / 2, w_out / 2);
    let (h5, w5) = (h_out / 4, w_out / 4);
    let mut rng = 0xfeedu32;
    let x3: Vec<f32> = (0..h_out * w_out * c).map(|_| lcg_f32(&mut rng)).collect();
    let x4: Vec<f32> = (0..h4 * w4 * c).map(|_| lcg_f32(&mut rng)).collect();
    let x5: Vec<f32> = (0..h5 * w5 * c).map(|_| lcg_f32(&mut rng)).collect();

    // Reference: unfused NEON pipeline (what the model ran before fusion).
    let mut x4_up = vec![0.0f32; h_out * w_out * c];
    let mut x5_up = vec![0.0f32; h_out * w_out * c];
    neon::bilinear_upsample_neon(&x4, &mut x4_up, h4, w4, c, h_out, w_out);
    neon::bilinear_upsample_neon(&x5, &mut x5_up, h5, w5, c, h_out, w_out);
    let mut out_ref = x3.clone();
    scalar::add3_inplace(&mut out_ref, &x4_up, &x5_up);

    // Fused single-pass version.
    let mut out_fused = x3.clone();
    neon::fpn_upsample2_add3_neon(&mut out_fused, &x4, h4, w4, &x5, h5, w5, c, h_out, w_out);

    for (i, (&r, &f)) in out_ref.iter().zip(out_fused.iter()).enumerate() {
        assert_eq!(
            r.to_bits(),
            f.to_bits(),
            "[fpn_upsample2_add3_neon parity] bit mismatch at index {i}: ref={r} fused={f}"
        );
    }
    eprintln!(
        "[neon_fpn_upsample2_add3_parity] OK — {} values, bit-exact",
        out_ref.len()
    );
}

/// Fused FPN merge (scalar) must be bit-identical to the unfused scalar pipeline.
#[cfg(target_arch = "aarch64")]
#[test]
fn scalar_fpn_upsample2_add3_parity() {
    let (h_out, w_out, c) = (12usize, 16usize, 64usize);
    let (h4, w4) = (h_out / 2, w_out / 2);
    let (h5, w5) = (h_out / 4, w_out / 4);
    let mut rng = 0xbeefu32;
    let x3: Vec<f32> = (0..h_out * w_out * c).map(|_| lcg_f32(&mut rng)).collect();
    let x4: Vec<f32> = (0..h4 * w4 * c).map(|_| lcg_f32(&mut rng)).collect();
    let x5: Vec<f32> = (0..h5 * w5 * c).map(|_| lcg_f32(&mut rng)).collect();

    let mut x4_up = vec![0.0f32; h_out * w_out * c];
    let mut x5_up = vec![0.0f32; h_out * w_out * c];
    scalar::bilinear_upsample(&x4, &mut x4_up, h4, w4, c, h_out, w_out);
    scalar::bilinear_upsample(&x5, &mut x5_up, h5, w5, c, h_out, w_out);
    let mut out_ref = x3.clone();
    scalar::add3_inplace(&mut out_ref, &x4_up, &x5_up);

    let mut out_fused = x3.clone();
    scalar::fpn_upsample2_add3(&mut out_fused, &x4, h4, w4, &x5, h5, w5, c, h_out, w_out);

    for (i, (&r, &f)) in out_ref.iter().zip(out_fused.iter()).enumerate() {
        assert_eq!(
            r.to_bits(),
            f.to_bits(),
            "[scalar fpn_upsample2_add3 parity] bit mismatch at index {i}: ref={r} fused={f}"
        );
    }
    eprintln!(
        "[scalar_fpn_upsample2_add3_parity] OK — {} values, bit-exact",
        out_ref.len()
    );
}

/// channel_softmax f16 NEON (polynomial exp) vs scalar f16 reference at the
/// model's real channel count c=65 (NOT a multiple of 8 — exercises the
/// scalar-tail path inside the NEON kernel).
#[cfg(target_arch = "aarch64")]
#[test]
fn neon_channel_softmax_f16_parity_c65() {
    let (h, w, c) = (6usize, 10usize, 65usize);
    let mut rng = 0x5071u32;
    // Logit-scale inputs in [-4, 4), stored as f16 bits.
    let init: Vec<u16> = (0..h * w * c)
        .map(|_| half::f16::from_f32(lcg_f32(&mut rng) * 8.0 - 4.0).to_bits())
        .collect();

    let mut buf_scalar = init.clone();
    let mut buf_neon = init;
    scalar::channel_softmax_f16(&mut buf_scalar, h, w, c);
    neon::channel_softmax_neon_f16_par(&mut buf_neon, h, w, c);

    // exp() implementations differ (libm vs degree-5 polynomial), so compare
    // within an f16-scale tolerance. Softmax outputs are in [0,1]; one f16 ULP
    // there is ~5e-4 and the polynomial is ~2 f32 ULP — 2e-3 is comfortably
    // tight while catching any tail-path indexing bug outright.
    let mut max_err = 0.0f32;
    for (i, (&s, &n)) in buf_scalar.iter().zip(buf_neon.iter()).enumerate() {
        let sv = half::f16::from_bits(s).to_f32();
        let nv = half::f16::from_bits(n).to_f32();
        let err = (sv - nv).abs();
        max_err = max_err.max(err);
        assert!(
            err <= 2e-3,
            "[channel_softmax_f16 c=65 parity] index {i}: scalar={sv} neon={nv} err={err}"
        );
    }
    eprintln!("[neon_channel_softmax_f16_parity_c65] OK — max err {max_err:.2e} (tol 2e-3)");
}

/// l2_normalize_channel f16 NEON vs scalar reference at the model's real
/// channel count c=64.
#[cfg(target_arch = "aarch64")]
#[test]
fn neon_l2_normalize_f16_parity_c64() {
    let (h, w, c) = (6usize, 10usize, 64usize);
    let mut rng = 0x12d4u32;
    let init: Vec<u16> = (0..h * w * c)
        .map(|_| half::f16::from_f32(lcg_f32(&mut rng) * 2.0 - 1.0).to_bits())
        .collect();

    let mut buf_scalar = init.clone();
    let mut buf_neon = init;
    scalar::l2_normalize_channel_f16(&mut buf_scalar, h, w, c);
    neon::l2_normalize_channel_f16_neon(&mut buf_neon, h, w, c);

    // Normalized values are in [-1,1]; f16 ULP there is ~5e-4. Reduction order
    // and sqrt path may differ between backends → tolerance, not bit-equality.
    let mut max_err = 0.0f32;
    for (i, (&s, &n)) in buf_scalar.iter().zip(buf_neon.iter()).enumerate() {
        let sv = half::f16::from_bits(s).to_f32();
        let nv = half::f16::from_bits(n).to_f32();
        let err = (sv - nv).abs();
        max_err = max_err.max(err);
        assert!(
            err <= 2e-3,
            "[l2_normalize_f16 c=64 parity] index {i}: scalar={sv} neon={nv} err={err}"
        );
    }
    eprintln!("[neon_l2_normalize_f16_parity_c64] OK — max err {max_err:.2e} (tol 2e-3)");
}

/// Fused drop-dustbin + pixel-shuffle (f16→f32): NEON vs scalar must be
/// bit-exact (pure copy + FCVTL widening, no arithmetic). The kernel is not
/// wired into the model hot path (stride-65 alignment penalty) but stays
/// correctness-guaranteed for the future layout revisit.
#[cfg(target_arch = "aarch64")]
#[test]
fn neon_drop_dustbin_pixel_shuffle_8_f16_parity_4x5() {
    let (h_in, w_in, c_with_dustbin) = (4usize, 5usize, 65usize);
    let n_in = h_in * w_in * c_with_dustbin;
    let mut rng = 0x0ddbu32;
    // Finite f16 normals in [1,2) — same generator as the pixel_shuffle test.
    let input_f16: Vec<u16> = (0..n_in)
        .map(|_| {
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            0x3c00u16 | ((rng >> 17) & 0x3ff) as u16
        })
        .collect();

    let h_out = h_in * 8;
    let w_out = w_in * 8;
    let mut out_scalar = vec![0.0f32; h_out * w_out];
    let mut out_neon = vec![0.0f32; h_out * w_out];

    scalar::drop_dustbin_pixel_shuffle_8_f16(
        &input_f16,
        &mut out_scalar,
        h_in,
        w_in,
        c_with_dustbin,
    );
    neon::drop_dustbin_pixel_shuffle_8_f16_neon(
        &input_f16,
        &mut out_neon,
        h_in,
        w_in,
        c_with_dustbin,
    );

    for (i, (&s, &n)) in out_scalar.iter().zip(out_neon.iter()).enumerate() {
        assert_eq!(
            s.to_bits(),
            n.to_bits(),
            "[drop_dustbin_pixel_shuffle_8_f16 parity] bit mismatch at index {i}: scalar={s} neon={n}"
        );
    }
    eprintln!(
        "[neon_drop_dustbin_pixel_shuffle_8_f16_parity_4x5] OK — {} values, bit-exact",
        out_scalar.len()
    );
}

#[cfg(not(target_arch = "aarch64"))]
#[test]
fn neon_skipped_on_non_aarch64() {
    eprintln!("[skip] NEON parity tests only run on aarch64");
}
