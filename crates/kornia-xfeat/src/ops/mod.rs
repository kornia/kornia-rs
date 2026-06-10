//! Fused per-layer kernels (scalar + SIMD).
//!
//! Each kernel takes NHWC `&[f32]` inputs and writes an `&mut [f32]` output.
//! The dispatcher in [`OpsVtable::select`] picks the best available
//! implementation at construction time and stores function pointers; the
//! per-frame hot path has no `cfg`/`if` to traverse.
//!
//! The scalar implementations in [`scalar`] are the parity oracle for every
//! SIMD path. They are correctness-first and intentionally un-tuned.

pub mod gemm_fp16;
pub mod scalar;
pub mod winograd;

#[cfg(target_arch = "aarch64")]
pub mod neon;

#[cfg(target_arch = "aarch64")]
pub mod neon_asm_f16;

#[cfg(target_arch = "x86_64")]
pub mod avx2;

use crate::cpu_features::cpu_features;

/// Activation applied to the conv epilogue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// `max(x, 0)`.
    Relu,
    /// `1 / (1 + exp(-x))`.
    Sigmoid,
    /// Identity — used by some final pointwise layers.
    Identity,
}

/// Static inputs / parameters to a 3×3 NHWC fused conv.
///
/// Weights are packed in `[c_out, k_h=3, k_w=3, c_in]` row-major order. BN
/// folding (if any) is done offline so the kernel sees only conv + bias.
pub struct Conv3x3Args<'a> {
    /// NHWC input tensor, length `h_in * w_in * c_in`.
    pub input: &'a [f32],
    /// Optional residual tensor added in the accumulator before activation.
    /// Same logical shape as the output. `None` for non-residual layers.
    pub residual: Option<&'a [f32]>,
    /// Packed weights `[c_out, k_h=3, k_w=3, c_in]`, length `c_out * 9 * c_in`.
    pub weights: &'a [f32],
    /// Per-output-channel bias (length `c_out`).
    pub bias: &'a [f32],
    /// Input height.
    pub h_in: usize,
    /// Input width.
    pub w_in: usize,
    /// Input channels.
    pub c_in: usize,
    /// Output channels.
    pub c_out: usize,
    /// Activation applied to the conv epilogue.
    pub activation: Activation,
    /// Optional pre-packed weights in `[c_out/4, 9, c_in, 4]` layout.
    /// When `Some`, the NEON v2 backend uses this directly and skips the
    /// per-frame repack. `None` triggers the fallback repack (needed for
    /// backends that don't use this layout, e.g. scalar or Winograd).
    pub packed_weights: Option<&'a [f32]>,
}

/// Static inputs / parameters to a 1×1 fused conv (per-pixel GEMM).
pub struct Conv1x1Args<'a> {
    /// NHWC input tensor, length `h * w * c_in`.
    pub input: &'a [f32],
    /// Weights packed `[c_out, c_in]` row-major.
    pub weights: &'a [f32],
    /// Per-output-channel bias (length `c_out`).
    pub bias: &'a [f32],
    /// Spatial height.
    pub h: usize,
    /// Spatial width.
    pub w: usize,
    /// Input channels.
    pub c_in: usize,
    /// Output channels.
    pub c_out: usize,
    /// Activation applied to the epilogue.
    pub activation: Activation,
}

/// Backend function-pointer table. Picked once at construction; the per-frame
/// hot path never touches `cfg` or feature detection.
#[derive(Debug, Clone, Copy)]
pub(crate) struct OpsVtable {
    pub conv3x3: fn(&Conv3x3Args<'_>, &mut [f32]),
    pub conv3x3_s2: fn(&Conv3x3Args<'_>, &mut [f32]),
    /// f32 1×1 conv. Currently unread: `model::extract` always routes 1×1 convs
    /// through `conv1x1_f16` (which falls back to this same NEON/scalar fn when
    /// fp16 is unavailable). Kept for symmetry with `conv1x1_f16` and as the
    /// f32 entry point for any future non-fp16 1×1 call site.
    #[allow(dead_code)]
    pub conv1x1: fn(&Conv1x1Args<'_>, &mut [f32]),
    /// fp16 1×1 conv — same signature as conv1x1 but uses ARMv8.2 FMLA.8H.
    /// Points to conv1x1 when fp16 is unavailable (same pointer, no branch needed).
    pub conv1x1_f16: fn(&Conv1x1Args<'_>, &mut [f32]),
}

impl OpsVtable {
    /// Pick the best available backend for this CPU.
    pub fn select() -> Self {
        let cpu = cpu_features();
        let _ = cpu;

        #[cfg(target_arch = "aarch64")]
        {
            // Use fp16 conv1x1 when the CPU supports ARMv8.2 half-precision.
            let conv1x1_f16_fn: fn(&Conv1x1Args<'_>, &mut [f32]) = if cpu.has_fp16 {
                neon_asm_f16::conv1x1_nhwc_f16
            } else {
                neon::conv1x1_nhwc_v2
            };
            return Self {
                // Stride-1 3×3 conv: Winograd F(2×2,3×3) — 16 multiplications
                // per 2×2 output tile vs 36 for direct conv, ~2× fewer FLOPs.
                conv3x3: winograd::conv3x3_winograd_dispatch,
                // Stride-2 conv: Winograd only supports stride-1; keep NEON v2.
                conv3x3_s2: neon::conv3x3_s2_relu_nhwc,
                // 1×1 conv: Winograd doesn't apply; use NEON v2 (c_out-4 tiled).
                conv1x1: neon::conv1x1_nhwc_v2,
                // fp16 1×1 conv: use ARMv8.2 FMLA.8H when available.
                conv1x1_f16: conv1x1_f16_fn,
            };
        }

        #[cfg(target_arch = "x86_64")]
        {
            if cpu.has_avx2 && cpu.has_fma {
                return Self {
                    conv3x3: avx2::conv3x3_relu_nhwc,
                    conv3x3_s2: avx2::conv3x3_s2_relu_nhwc,
                    conv1x1: avx2::conv1x1_nhwc,
                    conv1x1_f16: avx2::conv1x1_nhwc,
                };
            }
        }

        #[allow(unreachable_code)]
        Self::scalar()
    }

    /// Force the scalar backend regardless of CPU support. Used by parity tests.
    pub fn scalar() -> Self {
        Self {
            conv3x3: scalar::conv3x3_relu_nhwc,
            conv3x3_s2: scalar::conv3x3_s2_relu_nhwc,
            conv1x1: scalar::conv1x1_nhwc,
            conv1x1_f16: scalar::conv1x1_nhwc,
        }
    }
}

/// Repack weights from `[c_out, 9, c_in]` into `[c_out/4, 9, c_in, 4]` layout.
///
/// The NEON v2 kernel reads weights with the innermost 4-lane dimension
/// contiguous so each `vld1q_f32` loads one co_block-lane per tap.
/// Pre-computing this at model construction eliminates the per-frame alloc.
pub(crate) fn repack_weights_co4_3x3(weights: &[f32], c_out: usize, c_in: usize) -> Vec<f32> {
    debug_assert_eq!(c_out % 4, 0);
    let n_co4 = c_out / 4;
    let mut out = vec![0.0f32; n_co4 * 9 * c_in * 4];
    for co_block in 0..n_co4 {
        for tap in 0..9usize {
            for ci in 0..c_in {
                for lane in 0..4usize {
                    let co = co_block * 4 + lane;
                    out[((co_block * 9 + tap) * c_in + ci) * 4 + lane] =
                        weights[co * 9 * c_in + tap * c_in + ci];
                }
            }
        }
    }
    out
}

/// Repack 3×3 conv weights from `[c_out, 9, c_in]` to `[c_out/8, 9, c_in, 8]` fp16.
///
/// Used by the fp16 direct-conv3x3 kernel. Co-packs 8 output channels into each
/// `uint16x8_t` weight vector so a single FMLA.8H instruction covers all 8 at once.
/// `c_out` must be a multiple of 8.
pub fn repack_weights_co8_3x3_f16(weights: &[f32], c_out: usize, c_in: usize) -> Vec<u16> {
    debug_assert_eq!(c_out % 8, 0);
    let n_co8 = c_out / 8;
    let mut out = vec![0u16; n_co8 * 9 * c_in * 8];
    for co_block in 0..n_co8 {
        for tap in 0..9usize {
            for ci in 0..c_in {
                for lane in 0..8usize {
                    let co = co_block * 8 + lane;
                    out[((co_block * 9 + tap) * c_in + ci) * 8 + lane] =
                        half::f16::from_f32(weights[co * 9 * c_in + tap * c_in + ci]).to_bits();
                }
            }
        }
    }
    out
}

// ── Non-fused primitives used by the model graph ────────────────────────────
pub use scalar::{
    add3_from, add3_inplace, add_inplace, avgpool_4x4_s4, drop_last_channel_nhwc,
    l2_normalize_channel, pixel_shuffle_8, sigmoid_inplace, unfold_8x8,
};

/// Instance-normalize a single-channel 2-D activation map.
///
/// Dispatches to `neon::instance_norm_2d_singlech_neon` on aarch64 (2-pass NEON:
/// fused mean+variance in one Rayon fold, one fewer barrier than scalar 3-pass).
/// Falls back to scalar on other platforms.
pub fn instance_norm_2d_singlech(input: &[f32], output: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    if cpu_features().has_neon {
        return neon::instance_norm_2d_singlech_neon(input, output);
    }
    scalar::instance_norm_2d_singlech(input, output);
}

/// Channel-wise L2 normalization on f16 storage.
///
/// Dispatches to `neon::l2_normalize_channel_f16_neon` on aarch64 with fp16,
/// otherwise falls back to the scalar Rayon path.
pub fn l2_normalize_channel_f16(buf: &mut [u16], h: usize, w: usize, c: usize) {
    #[cfg(target_arch = "aarch64")]
    if cpu_features().has_fp16 {
        return neon::l2_normalize_channel_f16_neon(buf, h, w, c);
    }
    scalar::l2_normalize_channel_f16(buf, h, w, c);
}

/// NMS via MaxPool 5×5 equality check.
///
/// Uses the scalar Rayon path: direct 5×5 neighborhood check with threshold
/// early-exit. More efficient than the NEON dense-maxpool path on the small
/// 1/8-resolution heatmap (60×80) where the alloc + full pass outweighs
/// the SIMD throughput gain.
pub fn nms_maxpool_5x5_equality(
    heatmap: &[f32],
    h: usize,
    w: usize,
    threshold: f32,
) -> Vec<(f32, usize)> {
    scalar::nms_maxpool_5x5_equality(heatmap, h, w, threshold)
}

/// Channel-wise softmax on f16 storage.
///
/// Dispatches to `neon::channel_softmax_neon_f16_par` on aarch64 with fp16
/// (Rayon 800-pixel strips, degree-5 polynomial exp). Falls back to scalar.
pub fn channel_softmax_f16(buf: &mut [u16], h: usize, w: usize, c: usize) {
    #[cfg(target_arch = "aarch64")]
    if cpu_features().has_fp16 {
        return neon::channel_softmax_neon_f16_par(buf, h, w, c);
    }
    scalar::channel_softmax_f16(buf, h, w, c);
}

/// Sidecar-aware channel-wise softmax on f16 storage.
///
/// Softmax over `c_main + 1` logits per pixel: `c_main` in `main` (stride
/// `c_main`) plus one "dustbin" value per pixel in `dustbin`. Normalizes both in
/// place. Numerically equivalent to running [`channel_softmax_f16`] over a
/// stride-(c_main+1) interleaved buffer.
///
/// Dispatches to `neon::channel_softmax_neon_f16_sidecar_par` on aarch64 with
/// fp16; falls back to the scalar reference.
pub fn channel_softmax_f16_sidecar(
    main: &mut [u16],
    dustbin: &mut [u16],
    h: usize,
    w: usize,
    c_main: usize,
) {
    #[cfg(target_arch = "aarch64")]
    if cpu_features().has_fp16 {
        return neon::channel_softmax_neon_f16_sidecar_par(main, dustbin, h, w, c_main);
    }
    scalar::channel_softmax_f16_sidecar(main, dustbin, h, w, c_main);
}

/// Channel-wise softmax dispatcher: NEON on aarch64, scalar elsewhere.
pub fn channel_softmax(buf: &mut [f32], h: usize, w: usize, c: usize) {
    #[cfg(target_arch = "aarch64")]
    if cpu_features().has_neon {
        return neon::channel_softmax_neon(buf, h, w, c);
    }
    scalar::channel_softmax(buf, h, w, c);
}

/// Unfold 8×8 patches + f32→f16 narrowing.
///
/// Dispatches to `neon::unfold_8x8_to_f16_neon` on aarch64 with fp16 (FCVTN vectorized).
/// Falls back to scalar on other platforms.
pub fn unfold_8x8_to_f16(input: &[f32], output: &mut [u16], h_in: usize, w_in: usize) {
    #[cfg(target_arch = "aarch64")]
    if cpu_features().has_fp16 {
        return neon::unfold_8x8_to_f16_neon(input, output, h_in, w_in);
    }
    scalar::unfold_8x8_to_f16(input, output, h_in, w_in);
}

/// Pixel-shuffle (factor 8) + f16→f32 widening.
///
/// Dispatches to `neon::pixel_shuffle_8_f16_neon` on aarch64 (FCVTL vectorized).
/// Falls back to scalar on other platforms.
pub fn pixel_shuffle_8_f16(input: &[u16], output: &mut [f32], h_in: usize, w_in: usize) {
    #[cfg(target_arch = "aarch64")]
    if cpu_features().has_fp16 {
        return neon::pixel_shuffle_8_f16_neon(input, output, h_in, w_in);
    }
    scalar::pixel_shuffle_8_f16(input, output, h_in, w_in);
}

/// Fused sidecar-softmax + pixel-shuffle (c_main = 64) dispatcher.
///
/// Replaces the `channel_softmax_f16_sidecar` + `pixel_shuffle_8_f16` sequence
/// with a single dispatch. Per pixel: softmax over the 64 main channels + the
/// dustbin (dustbin normalized + written back), then the 64 normalized values
/// pixel-shuffled into the f32 `k1h` output. Numerically identical to running
/// the two ops in sequence. `main` is left un-normalized afterward (nothing
/// downstream reads it).
///
/// Dispatches to NEON on aarch64 with fp16; falls back to the scalar reference.
pub fn channel_softmax_pixel_shuffle_f16_sidecar(
    main: &mut [u16],
    dustbin: &mut [u16],
    k1h_out: &mut [f32],
    h8: usize,
    w8: usize,
) {
    #[cfg(target_arch = "aarch64")]
    if cpu_features().has_fp16 {
        return neon::channel_softmax_pixel_shuffle_f16_sidecar_neon(
            main, dustbin, k1h_out, h8, w8,
        );
    }
    scalar::channel_softmax_pixel_shuffle_f16_sidecar(main, dustbin, k1h_out, h8, w8);
}

/// Fused FPN merge dispatcher: `x3 += upsample(x4) + upsample(x5)` in one pass.
///
/// Replaces the 3-dispatch `bilinear_upsample` ×2 + `add3_inplace` sequence —
/// saves two Rayon barriers and the intermediate upsample buffers. Output is
/// bit-identical to the unfused sequence on each backend.
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
    #[cfg(target_arch = "aarch64")]
    if cpu_features().has_neon {
        return neon::fpn_upsample2_add3_neon(x3, x4, h4, w4, x5, h5, w5, c, h_out, w_out);
    }
    scalar::fpn_upsample2_add3(x3, x4, h4, w4, x5, h5, w5, c, h_out, w_out);
}

/// Bilinear upsample dispatcher: NEON on aarch64, scalar elsewhere.
pub fn bilinear_upsample(
    input: &[f32],
    output: &mut [f32],
    h_in: usize,
    w_in: usize,
    c: usize,
    h_out: usize,
    w_out: usize,
) {
    #[cfg(target_arch = "aarch64")]
    if cpu_features().has_neon {
        return neon::bilinear_upsample_neon(input, output, h_in, w_in, c, h_out, w_out);
    }
    scalar::bilinear_upsample(input, output, h_in, w_in, c, h_out, w_out);
}
