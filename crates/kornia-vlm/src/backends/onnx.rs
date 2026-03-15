//! ONNX Runtime session wrapper for kornia-vlm.
//!
//! Wraps [`ort`] to load `.onnx` models exported from kornia (PyTorch)
//! and run inference within the kornia-vlm pipeline.

use ort::{execution_providers::CUDAExecutionProvider, GraphOptimizationLevel, Session};

use crate::backends::VlmBackend;

/// Errors from building or running an ONNX session.
#[derive(Debug, thiserror::Error)]
pub enum OnnxError {
    /// The backend variant is not an ONNX backend.
    #[error("Backend '{0}' is not an ONNX backend")]
    NotOnnxBackend(&'static str),

    /// ONNX Runtime reported an error.
    #[error("ONNX Runtime error: {0}")]
    OrtError(#[from] ort::Error),
}

/// A loaded ONNX Runtime inference session.
pub struct OnnxSession {
    /// Underlying ORT session.
    pub session: Session,
    /// Backend used to create this session.
    pub backend: VlmBackend,
}

impl OnnxSession {
    /// Load an ONNX model from `path` using the execution provider
    /// matching `backend`.
    pub fn from_file(path: &str, backend: &VlmBackend) -> Result<Self, OnnxError> {
        if !backend.is_onnx() {
            return Err(OnnxError::NotOnnxBackend(backend.name()));
        }

        let mut builder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?;

        match backend {
            VlmBackend::OnnxCpu => {
                log::info!("[kornia-vlm] ONNX Runtime: CPU execution provider");
            }
            VlmBackend::OnnxCuda { device_id } => {
                log::info!("[kornia-vlm] ONNX Runtime: CUDA EP (device {})", device_id);
                builder = builder.with_execution_providers([
                    CUDAExecutionProvider::default()
                        .with_device_id(*device_id)
                        .build(),
                ])?;
            }
            VlmBackend::TensorRt { device_id } => {
                // Full TensorRT EP tracked in https://github.com/kornia/kornia-rs/issues/634
                log::warn!(
                    "[kornia-vlm] TensorRT EP (device {}) not yet wired — using CUDA EP. See #634",
                    device_id
                );
                builder = builder.with_execution_providers([
                    CUDAExecutionProvider::default()
                        .with_device_id(*device_id)
                        .build(),
                ])?;
            }
            VlmBackend::Candle => unreachable!("filtered above"),
        }

        let session = builder.commit_from_file(path)?;
        log::debug!(
            "[kornia-vlm] Loaded '{}': {} inputs, {} outputs",
            path,
            session.inputs.len(),
            session.outputs.len(),
        );

        Ok(Self { session, backend: backend.clone() })
    }

    /// Names of all model inputs.
    pub fn input_names(&self) -> Vec<&str> {
        self.session.inputs.iter().map(|i| i.name.as_str()).collect()
    }

    /// Names of all model outputs.
    pub fn output_names(&self) -> Vec<&str> {
        self.session.outputs.iter().map(|o| o.name.as_str()).collect()
    }
}
