//! XFeat packed weights — loading from safetensors, BN-fold-aware accessors.
//!
//! The on-disk artifact is **not** the upstream PyTorch state dict. It's a
//! kornia-xfeat-specific safetensors file produced by
//! `tools/xfeat-convert`. Two invariants the converter enforces:
//!
//! 1. **BN folded into the preceding conv.** Every BasicLayer in the upstream
//!    model is `Conv2d(bias=False) → BatchNorm2d(affine=False) → ReLU`. The
//!    fold collapses this into a single `Conv2d(bias=True)` with effective
//!    weights `W * γ / sqrt(var + eps)` and bias `β - γ * mean / sqrt(var + eps)`.
//!    Because the upstream model uses `affine=False`, `γ = 1` and `β = 0`,
//!    so the formula simplifies to:
//!    `W_eff = W / sqrt(var + eps)`, `b_eff = -mean / sqrt(var + eps)`.
//! 2. **Weights pre-packed for our NHWC conv layout** `[c_out, k_h, k_w, c_in]`.
//!    PyTorch ships `[c_out, c_in, k_h, k_w]`; we transpose at conversion time
//!    so the runtime kernel reads contiguous bytes.
//!
//! Skip-1's Conv2d, block_fusion's final Conv2d, and the head finals all have
//! native PyTorch biases (no BN), which we pass through.

use std::collections::HashMap;

use safetensors::SafeTensors;

/// Errors when loading or parsing packed weights.
#[derive(Debug, thiserror::Error)]
pub enum WeightsError {
    /// Underlying safetensors parse failure.
    #[error("safetensors parse error: {0}")]
    Parse(String),
    /// Missing expected tensor.
    #[error("missing tensor `{0}` in packed weights")]
    Missing(String),
    /// Tensor present but with the wrong shape.
    #[error("tensor `{name}` has shape {got:?}, expected {expected:?}")]
    BadShape {
        /// Tensor name that mismatched.
        name: String,
        /// Shape as found in the artifact.
        got: Vec<usize>,
        /// Shape the model expected.
        expected: Vec<usize>,
    },
    /// Tensor has an unexpected dtype.
    #[error("tensor `{name}` has dtype {got:?}, expected f32")]
    BadDtype {
        /// Tensor name that mismatched.
        name: String,
        /// Dtype string (e.g. `"F16"`) as found.
        got: String,
    },
    /// SHA-256 of the loaded artifact doesn't match the expected pin.
    #[error("packed weights SHA mismatch: got {got}, expected {expected}")]
    ShaMismatch {
        /// SHA computed from the loaded bytes.
        got: String,
        /// SHA pinned at build time.
        expected: String,
    },
    /// I/O failure reading the weights file.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

/// Expected SHA-256 of the published packed-weights artifact. Verified on every
/// load path (embed / hub / explicit) so we can't silently regress to a wrong
/// checkpoint.
pub const PACKED_WEIGHTS_SHA256: &str =
    // Placeholder — replaced when the converter produces the canonical artifact.
    "0000000000000000000000000000000000000000000000000000000000000000";

/// Packed, BN-folded, NHWC-laid-out weights for the entire XFeat model.
///
/// All tensors are `f32`. Memory is owned by this struct; the model borrows
/// slices into it for the lifetime of the model.
#[derive(Debug)]
pub struct PackedWeights {
    tensors: HashMap<String, Vec<f32>>,
    shapes: HashMap<String, Vec<usize>>,
}

impl PackedWeights {
    /// Load from a safetensors byte blob (e.g. `include_bytes!(...)`).
    pub fn from_safetensors_bytes(bytes: &[u8]) -> Result<Self, WeightsError> {
        let st = SafeTensors::deserialize(bytes).map_err(|e| WeightsError::Parse(e.to_string()))?;
        let mut tensors = HashMap::new();
        let mut shapes = HashMap::new();
        for (name, view) in st.tensors() {
            if view.dtype() != safetensors::Dtype::F32 {
                return Err(WeightsError::BadDtype {
                    name,
                    got: format!("{:?}", view.dtype()),
                });
            }
            let raw = view.data();
            let mut floats = vec![0.0f32; raw.len() / 4];
            for (i, chunk) in raw.chunks_exact(4).enumerate() {
                floats[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
            shapes.insert(name.clone(), view.shape().to_vec());
            tensors.insert(name, floats);
        }
        Ok(Self { tensors, shapes })
    }

    /// Look up a named tensor; error with a clear name if missing.
    pub fn get(&self, name: &str) -> Result<&[f32], WeightsError> {
        self.tensors
            .get(name)
            .map(|v| v.as_slice())
            .ok_or_else(|| WeightsError::Missing(name.into()))
    }

    /// Shape of a named tensor, in the on-disk layout.
    pub fn shape(&self, name: &str) -> Result<&[usize], WeightsError> {
        self.shapes
            .get(name)
            .map(|v| v.as_slice())
            .ok_or_else(|| WeightsError::Missing(name.into()))
    }

    /// Assert shape matches the expectation; clearer error than the
    /// kernel's `debug_assert`.
    pub fn check_shape(&self, name: &str, expected: &[usize]) -> Result<(), WeightsError> {
        let got = self.shape(name)?;
        if got != expected {
            return Err(WeightsError::BadShape {
                name: name.into(),
                got: got.to_vec(),
                expected: expected.to_vec(),
            });
        }
        Ok(())
    }
}

/// Embedded weights byte slice. `None` if the `xfeat-embed` feature isn't on.
///
/// The actual `include_bytes!` is conditionally compiled so that builds
/// without `xfeat-embed` don't carry the 6 MB blob in the rlib.
#[cfg(feature = "xfeat-embed")]
pub fn embedded_bytes() -> &'static [u8] {
    // Placeholder weight file. The real artifact is produced by
    // `tools/xfeat-convert` from the upstream `xfeat.pt` checkpoint.
    include_bytes!("../assets/xfeat_packed_placeholder.safetensors")
}

#[cfg(not(feature = "xfeat-embed"))]
pub fn embedded_bytes() -> &'static [u8] {
    &[]
}
