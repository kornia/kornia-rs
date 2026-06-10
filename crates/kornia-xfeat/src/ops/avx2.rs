//! AVX2 (x86_64) kernels for kornia-xfeat.
//!
//! Status: stubs that delegate to the scalar reference. Real AVX2 kernels
//! land after NEON does — same parity-test bar.

#![allow(unused_imports)]

use super::{scalar, Conv1x1Args, Conv3x3Args};

/// AVX2 3×3 NHWC fused conv. Currently scalar; real impl uses
/// `_mm256_fmadd_ps` over `f32x8` chunks of `c_out`.
pub fn conv3x3_relu_nhwc(args: &Conv3x3Args<'_>, output: &mut [f32]) {
    scalar::conv3x3_relu_nhwc(args, output);
}

/// AVX2 3×3 stride-2 conv.
pub fn conv3x3_s2_relu_nhwc(args: &Conv3x3Args<'_>, output: &mut [f32]) {
    scalar::conv3x3_s2_relu_nhwc(args, output);
}

/// AVX2 1×1 fused conv.
pub fn conv1x1_nhwc(args: &Conv1x1Args<'_>, output: &mut [f32]) {
    scalar::conv1x1_nhwc(args, output);
}
