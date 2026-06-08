//! Fused per-layer kernels (scalar + SIMD).
//!
//! Each kernel takes NHWC `&[f32]` inputs and writes an `&mut [f32]` output.
//! The dispatcher in [`OpsVtable::select`] picks the best available
//! implementation at construction time and stores function pointers; the
//! per-frame hot path has no `cfg`/`if` to traverse.
//!
//! The scalar implementations in [`scalar`] are the parity oracle for every
//! SIMD path. They are correctness-first and intentionally un-tuned.

pub mod scalar;

#[cfg(target_arch = "aarch64")]
pub mod neon;

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
#[allow(dead_code)] // populated by `select()`, consumed by `model::extract` once layer wiring lands
pub(crate) struct OpsVtable {
    pub conv3x3: fn(&Conv3x3Args<'_>, &mut [f32]),
    pub conv3x3_s2: fn(&Conv3x3Args<'_>, &mut [f32]),
    pub conv1x1: fn(&Conv1x1Args<'_>, &mut [f32]),
}

impl OpsVtable {
    /// Pick the best available backend for this CPU.
    pub fn select() -> Self {
        let cpu = cpu_features();
        let _ = cpu;

        #[cfg(target_arch = "aarch64")]
        {
            return Self {
                conv3x3: neon::conv3x3_relu_nhwc,
                conv3x3_s2: neon::conv3x3_s2_relu_nhwc,
                // v2: c_out-4 tiled co-product with rayon row parallelism;
                // falls back to v1 internally when c_out % 4 != 0.
                conv1x1: neon::conv1x1_nhwc_v2,
            };
        }

        #[cfg(target_arch = "x86_64")]
        {
            if cpu.has_avx2 && cpu.has_fma {
                return Self {
                    conv3x3: avx2::conv3x3_relu_nhwc,
                    conv3x3_s2: avx2::conv3x3_s2_relu_nhwc,
                    conv1x1: avx2::conv1x1_nhwc,
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
        }
    }
}

// ── Non-fused primitives used by the model graph ────────────────────────────
pub use scalar::{
    add3_inplace, add_inplace, avgpool_4x4_s4, bilinear_upsample, channel_softmax,
    drop_last_channel_nhwc, instance_norm_2d_singlech, l2_normalize_channel,
    nms_maxpool_5x5_equality, pixel_shuffle_8, sigmoid_inplace, unfold_8x8,
};
