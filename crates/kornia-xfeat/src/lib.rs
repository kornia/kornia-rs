#![deny(missing_docs)]
#![doc = env!("CARGO_PKG_DESCRIPTION")]
//!
//! # Overview
//!
//! `kornia-xfeat` is a from-scratch CPU implementation of XFeat (Verlab, 2024),
//! a lightweight learned local feature extractor. Unlike the upstream PyTorch
//! reference, this crate has no Python / candle / ONNX runtime in the loop — the
//! forward pass is written directly in Rust, with hand-tuned NEON (aarch64) and
//! AVX2/AVX-512 (x86_64) kernels for the dominant ops (3×3 conv, 1×1 GEMM).
//!
//! ## Design pillars
//!
//! - **Non-batched.** Tensors are `(H, W, C)`; no batch dim. One image per call.
//! - **NHWC end-to-end.** Channel is the contiguous axis — SIMD friendly.
//! - **Fused epilogues.** BN-fold is done offline; conv + bias + ReLU run in one
//!   pass over input/output memory.
//! - **Zero-alloc inference.** A ping-pong arena, sized once at model
//!   construction from a known input resolution, owns every intermediate.
//! - **Parity-tested against the upstream PyTorch model** via committed fixtures.
//!
//! ## Public surface
//!
//! - [`XFeat`] — the model. Construct from packed weights, call [`XFeat::extract`].
//! - [`XFeatOutput`] — borrowed view into the model's keypoint/descriptor/reliability buffers.
//! - [`KeyPoint`] — one detection: `(x, y, score)` + index into the descriptor table.
//! - [`PinholeCamera`]-style preprocessing helpers live under [`preproc`].
//!
//! ## Status
//!
//! Early. The scalar reference is the parity oracle and is correct; NEON / AVX
//! kernels and the offline weight-conversion tool are in progress.

pub mod affinity;
pub mod model;
pub mod ops;
pub mod postproc;
pub mod preproc;
pub mod tensor;
pub mod weights;

mod cpu_features;

pub use model::{XFeat, XFeatConfig};
pub use postproc::KeyPoint;
pub use tensor::NhwcTensor;
pub use weights::{PackedWeights, WeightsError};

/// Borrowed view into a single call's outputs. Lifetime is the same `&mut`
/// borrow on [`XFeat`] used by [`XFeat::extract`]; the next call invalidates it.
#[derive(Debug)]
pub struct XFeatOutput<'a> {
    /// Detected keypoints, post-NMS, sorted by descending score.
    pub keypoints: &'a [KeyPoint],
    /// L2-normalized descriptors, parallel to `keypoints`. Row stride = 64.
    pub descriptors: &'a [f32],
    /// Per-keypoint reliability, parallel to `keypoints`.
    pub reliability: &'a [f32],
}

/// Errors that can come out of [`XFeat::extract`] or model construction.
#[derive(Debug, thiserror::Error)]
pub enum XFeatError {
    /// The input image dimensions don't match what the model was configured for.
    #[error("input shape mismatch: expected {expected:?}, got {got:?}")]
    InputShapeMismatch {
        /// `(height, width)` the model was configured for.
        expected: (usize, usize),
        /// `(height, width)` the caller provided.
        got: (usize, usize),
    },
    /// Image dimensions are not a multiple of 32 (XFeat requires this).
    #[error("input dimensions must be a multiple of 32: got {0}×{1}")]
    InputNotAlignedTo32(usize, usize),
    /// Failure loading or parsing packed weights.
    #[error(transparent)]
    Weights(#[from] WeightsError),
}
