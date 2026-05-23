//! NEON (aarch64) kernels for kornia-xfeat.
//!
//! Status: stubs that delegate to the scalar reference. The first real NEON
//! kernel (3×3 NHWC fused conv + bias + ReLU) is the next thing to land.
//! Each new kernel must pass parity against [`super::scalar`] before being
//! wired into [`super::OpsVtable::select`].

#![allow(unused_imports)]

use super::{scalar, Conv1x1Args, Conv3x3Args};

/// NEON 3×3 NHWC fused conv + bias + (optional residual) + activation.
///
/// Currently delegates to scalar. The next milestone replaces this with a
/// hand-written `vfmaq_f32` loop over `f32x4` chunks of `c_out`, with
/// parallel accumulators per the NEON skill's chain-length rule.
pub fn conv3x3_relu_nhwc(args: &Conv3x3Args<'_>, output: &mut [f32]) {
    scalar::conv3x3_relu_nhwc(args, output);
}

/// NEON 3×3 stride-2 conv.
pub fn conv3x3_s2_relu_nhwc(args: &Conv3x3Args<'_>, output: &mut [f32]) {
    scalar::conv3x3_s2_relu_nhwc(args, output);
}

/// NEON 1×1 fused conv.
pub fn conv1x1_nhwc(args: &Conv1x1Args<'_>, output: &mut [f32]) {
    scalar::conv1x1_nhwc(args, output);
}
