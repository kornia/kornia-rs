//! Backend abstraction for kornia-vlm inference.
//!
//! Introduces a [`VlmBackend`] enum so callers can select between
//! Candle (default), ONNX Runtime CPU/CUDA, and TensorRT — without
//! changing any model or preprocessing code.
//!
//! # Example
//!
//! ```rust
//! use kornia_vlm::backends::VlmBackend;
//! let b = VlmBackend::default();
//! assert_eq!(b.name(), "candle");
//! ```

#[cfg(feature = "onnx")]
pub mod onnx;

/// Runtime backend for VLM inference.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum VlmBackend {
    /// Pure-Rust Candle backend (default). No extra system dependencies.
    #[default]
    Candle,

    /// ONNX Runtime on CPU. Requires the `onnx` feature flag.
    #[cfg(feature = "onnx")]
    OnnxCpu,

    /// ONNX Runtime with CUDA execution provider.
    /// Requires the `onnx` feature flag and a CUDA-capable GPU.
    #[cfg(feature = "onnx")]
    OnnxCuda {
        /// CUDA device index (0 = first GPU).
        device_id: i32,
    },

    /// ONNX Runtime with TensorRT execution provider.
    /// Full wiring tracked in <https://github.com/kornia/kornia-rs/issues/634>.
    #[cfg(feature = "onnx")]
    TensorRt {
        /// CUDA device index (0 = first GPU).
        device_id: i32,
    },
}

impl VlmBackend {
    /// Returns `true` if this backend uses ONNX Runtime.
    pub fn is_onnx(&self) -> bool {
        #[cfg(feature = "onnx")]
        match self {
            Self::OnnxCpu | Self::OnnxCuda { .. } | Self::TensorRt { .. } => return true,
            _ => {}
        }
        false
    }

    /// Human-readable name for logging and benchmarks.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Candle => "candle",
            #[cfg(feature = "onnx")]
            Self::OnnxCpu => "onnxruntime-cpu",
            #[cfg(feature = "onnx")]
            Self::OnnxCuda { .. } => "onnxruntime-cuda",
            #[cfg(feature = "onnx")]
            Self::TensorRt { .. } => "onnxruntime-tensorrt",
        }
    }
}
