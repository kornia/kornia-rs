# Backend Architecture for kornia-vlm

**Status:** Proposal  
**Author:** [Your Name] (@your-github-username)  
**Date:** February 12, 2026  
**Related:** GSoC 2026 - ONNX and TensorRT Integration Project

## Overview

This document proposes an architecture for integrating ONNX Runtime and TensorRT backends into `kornia-vlm`, enabling multi-backend inference for Vision Language Models while maintaining backward compatibility with the existing Candle implementation.

## Current State

### Existing Implementation

kornia-vlm currently supports three VLM models:
- **PaliGemma** (`src/paligemma/`)
- **SmolVLM** (`src/smolvlm/`)
- **SmolVLM2** (`src/smolvlm2/`)

All models use the **Candle** ML framework exclusively:
```rust
// Current dependency in Cargo.toml
[dependencies]
candle-core = { workspace = true }
candle-nn = { workspace = true }
candle-transformers = { workspace = true }
```

### Existing ONNX Runtime Usage

The repository already has ONNX Runtime integration in `examples/onnx/`, which demonstrates:
- Using `ort = "2.0.0-rc.10"` crate
- Converting kornia tensors to ORT tensors
- Running inference with `Session::builder()`
- Extracting results back to kornia format

**Key learnings from existing ONNX example:**
```rust
// Pattern from examples/onnx/src/main.rs
let model = Session::builder()?
    .with_optimization_level(GraphOptimizationLevel::Level3)?
    .commit_from_file(&model_path)?;

let ort_tensor = ort::value::Tensor::from_array((shape, data))?;
let outputs = model.run(ort::inputs!["input" => ort_tensor])?;
```

## Proposed Architecture

### 1. Backend Abstraction Layer

Introduce a trait-based abstraction for inference backends:
```rust
// crates/kornia-vlm/src/backend/mod.rs

pub trait InferenceBackend: Send + Sync {
    /// Perform inference on an image with a text prompt
    fn infer(
        &mut self,
        image: &Image<u8, 3>,
        prompt: &str,
        max_length: usize,
    ) -> Result<String, BackendError>;
    
    /// Get the name of this backend
    fn backend_name(&self) -> &'static str;
    
    /// Check if this backend supports batching
    fn supports_batch(&self) -> bool {
        false
    }
}

#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    #[error("Preprocessing failed: {0}")]
    PreprocessingError(String),
    
    #[error("Inference failed: {0}")]
    InferenceError(String),
    
    #[error("Postprocessing failed: {0}")]
    PostprocessingError(String),
}
```

### 2. Backend Implementations

#### Candle Backend (Existing)

Wrap existing Candle implementation:
```rust
// crates/kornia-vlm/src/backend/candle.rs

pub struct CandleBackend {
    model: Model,
    tokenizer: Tokenizer,
    device: Device,
}

impl InferenceBackend for CandleBackend {
    fn infer(&mut self, image: &Image<u8, 3>, prompt: &str, max_length: usize) 
        -> Result<String, BackendError> {
        // Existing Candle inference logic
    }
    
    fn backend_name(&self) -> &'static str {
        "Candle"
    }
}
```

#### ONNX Runtime Backend (New)
```rust
// crates/kornia-vlm/src/backend/onnx.rs

use ort::{Session, ExecutionProvider};

pub struct OnnxBackend {
    session: Session,
    tokenizer: Tokenizer,
    provider: ExecutionProvider,
}

impl OnnxBackend {
    pub fn new(
        model_path: impl AsRef<Path>,
        provider: ExecutionProvider,
    ) -> Result<Self, BackendError> {
        let session = Session::builder()?
            .with_execution_providers(&[provider])?
            .commit_from_file(model_path)?;
        
        // Load tokenizer from same directory
        let tokenizer = Tokenizer::from_file(/* ... */)?;
        
        Ok(Self { session, tokenizer, provider })
    }
}

impl InferenceBackend for OnnxBackend {
    fn infer(&mut self, image: &Image<u8, 3>, prompt: &str, max_length: usize)
        -> Result<String, BackendError> {
        // 1. Preprocess image and text
        let image_tensor = self.preprocess_image(image)?;
        let input_ids = self.tokenizer.encode(prompt, false)?;
        
        // 2. Convert to ORT tensors
        let ort_image = ort::value::Tensor::from_array(image_tensor)?;
        let ort_ids = ort::value::Tensor::from_array(input_ids)?;
        
        // 3. Run inference
        let outputs = self.session.run(ort::inputs![
            "image" => ort_image,
            "input_ids" => ort_ids,
        ])?;
        
        // 4. Decode output
        let output_ids = outputs["output"].extract_tensor()?;
        let text = self.tokenizer.decode(output_ids)?;
        
        Ok(text)
    }
    
    fn backend_name(&self) -> &'static str {
        match self.provider {
            ExecutionProvider::CPU(_) => "ONNX-CPU",
            ExecutionProvider::CUDA(_) => "ONNX-CUDA",
            ExecutionProvider::TensorRT(_) => "ONNX-TensorRT",
            _ => "ONNX-Unknown",
        }
    }
}
```

### 3. Execution Provider Configuration
```rust
// crates/kornia-vlm/src/backend/config.rs

#[derive(Debug, Clone)]
pub enum BackendType {
    Candle { device: Device },
    OnnxCpu,
    OnnxCuda { device_id: i32 },
    OnnxTensorRT(TensorRTConfig),
}

#[derive(Debug, Clone)]
pub struct TensorRTConfig {
    pub device_id: i32,
    pub fp16_mode: bool,
    pub int8_mode: bool,
    pub max_workspace_size: usize,
    pub engine_cache_path: Option<PathBuf>,
}

impl Default for TensorRTConfig {
    fn default() -> Self {
        Self {
            device_id: 0,
            fp16_mode: true,
            int8_mode: false,
            max_workspace_size: 1 << 30, // 1GB
            engine_cache_path: Some(PathBuf::from("./trt_cache")),
        }
    }
}

impl BackendType {
    pub fn to_execution_provider(&self) -> ort::ExecutionProvider {
        match self {
            BackendType::OnnxCpu => ExecutionProvider::CPU(Default::default()),
            BackendType::OnnxCuda { device_id } => {
                ExecutionProvider::CUDA(
                    ort::CUDAExecutionProviderOptions::default()
                        .with_device_id(*device_id)
                )
            }
            BackendType::OnnxTensorRT(config) => {
                ExecutionProvider::TensorRT(
                    ort::TensorRTExecutionProviderOptions::default()
                        .with_device_id(config.device_id)
                        .with_fp16(config.fp16_mode)
                        .with_int8(config.int8_mode)
                )
            }
            _ => unreachable!(),
        }
    }
}
```

### 4. Unified VLM Model API
```rust
// crates/kornia-vlm/src/model.rs

pub struct VlmModel {
    backend: Box<dyn InferenceBackend>,
}

impl VlmModel {
    /// Create a new VLM model with the specified backend
    pub fn new(model_path: impl AsRef<Path>, backend_type: BackendType) 
        -> Result<Self, VlmError> {
        let backend: Box<dyn InferenceBackend> = match backend_type {
            BackendType::Candle { device } => {
                Box::new(CandleBackend::new(model_path, device)?)
            }
            BackendType::OnnxCpu | BackendType::OnnxCuda { .. } | BackendType::OnnxTensorRT(_) => {
                let provider = backend_type.to_execution_provider();
                Box::new(OnnxBackend::new(model_path, provider)?)
            }
        };
        
        Ok(Self { backend })
    }
    
    /// Run inference on an image
    pub fn infer(&mut self, image: &Image<u8, 3>, prompt: &str, max_length: usize) 
        -> Result<String, VlmError> {
        self.backend.infer(image, prompt, max_length)
            .map_err(VlmError::from)
    }
    
    /// Get the active backend name
    pub fn backend_name(&self) -> &'static str {
        self.backend.backend_name()
    }
}
```

## Directory Structure
```
crates/kornia-vlm/
├── src/
│   ├── backend/
│   │   ├── mod.rs          # Trait definition + BackendError
│   │   ├── candle.rs       # Candle implementation
│   │   ├── onnx.rs         # ONNX Runtime implementation
│   │   └── config.rs       # BackendType, TensorRTConfig
│   ├── models/
│   │   ├── paligemma/      # Existing (refactor to use backend trait)
│   │   ├── smolvlm/        # Existing (refactor to use backend trait)
│   │   └── smolvlm2/       # Existing (refactor to use backend trait)
│   ├── model.rs            # Unified VlmModel API
│   ├── lib.rs              # Re-exports
│   └── ...
├── benches/
│   └── backend_comparison.rs  # Benchmark all backends
├── examples/
│   ├── paligemma_candle.rs    # Existing
│   ├── paligemma_onnx.rs      # NEW
│   └── backend_selection.rs   # NEW
└── docs/
    └── BACKEND_ARCHITECTURE.md # This document
```

## Dependency Changes
```toml
# crates/kornia-vlm/Cargo.toml

[dependencies]
# Existing
candle-core = { workspace = true, optional = true }
candle-nn = { workspace = true, optional = true }
candle-transformers = { workspace = true, optional = true }

# NEW - ONNX Runtime
ort = { version = "2.0.0-rc.10", optional = true, default-features = false, features = ["half", "load-dynamic"] }

# ... other existing dependencies

[features]
default = ["candle"]
candle = ["candle-core", "candle-nn", "candle-transformers"]
onnx = ["ort"]
onnx-cuda = ["onnx", "ort/cuda"]
onnx-tensorrt = ["onnx", "ort/tensorrt"]
all-backends = ["candle", "onnx-cuda", "onnx-tensorrt"]
```

## Migration Path

### Phase 1: Foundation (Weeks 1-2)
- Create backend trait and module structure
- Wrap existing Candle code in CandleBackend
- Add ONNX dependency
- Ensure existing examples still work

### Phase 2: ONNX Integration (Weeks 3-4)
- Implement OnnxBackend for CPU
- Create ONNX export scripts (Python)
- Validate numerical accuracy
- Add ONNX example

### Phase 3: GPU Acceleration (Weeks 5-6)
- Add CUDA execution provider support
- Benchmark CUDA vs CPU
- Optimize memory transfers

### Phase 4: TensorRT (Weeks 7-8)
- Integrate TensorRT execution provider
- FP16 optimization
- Engine caching
- Jetson testing

### Phase 5: Testing & Documentation (Weeks 9-10)
- Comprehensive test suite
- Performance benchmarks
- User documentation
- Migration guide

## ONNX Model Export

Models need to be exported from PyTorch to ONNX format:
```python
# scripts/export_paligemma.py

import torch
from transformers import PaliGemmaForConditionalGeneration

model = PaliGemmaForConditionalGeneration.from_pretrained("google/paligemma-3b-pt-224")

dummy_image = torch.randn(1, 3, 224, 224)
dummy_text = torch.randint(0, 1000, (1, 10))

torch.onnx.export(
    model,
    (dummy_image, dummy_text),
    "paligemma.onnx",
    input_names=["image", "input_ids"],
    output_names=["output"],
    dynamic_axes={
        "image": {0: "batch_size"},
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "output": {0: "batch_size", 1: "sequence_length"},
    },
    opset_version=17,
)
```

## Performance Expectations

Based on similar implementations:

| Backend | Latency (relative) | Throughput (relative) | Memory |
|---------|-------------------|----------------------|--------|
| Candle CPU | 1.0x (baseline) | 1.0x | Baseline |
| ONNX CPU | 0.9-1.1x | 0.9-1.1x | ~1.2x |
| ONNX CUDA | 0.3-0.5x | 2-3x | ~1.5x |
| TensorRT FP16 | 0.2-0.3x | 3-5x | ~1.3x |
| TensorRT INT8 | 0.1-0.2x | 5-10x | ~1.0x |

*Note: Actual performance depends on model architecture and hardware.*

## Backward Compatibility

Existing code continues to work without changes:
```rust
// This still works exactly as before
let mut model = Paligemma::new(PaligemmaConfig::default())?;
let caption = model.inference(&image, "Describe", 100, true)?;
```

New backend selection API:
```rust
// New: explicit backend selection
let model = VlmModel::new("paligemma.onnx", BackendType::OnnxTensorRT(Default::default()))?;
let caption = model.infer(&image, "Describe", 100)?;
```

## Testing Strategy

### Unit Tests
- Backend trait implementations
- Tensor conversions
- Error handling

### Integration Tests
- Model loading for each backend
- Inference accuracy comparison
- Memory leak detection

### Benchmarks
- Latency measurements
- Throughput tests
- Memory profiling
- Cross-backend comparison

## Open Questions

1. **Model Format:** Should we distribute models in both Candle and ONNX formats, or convert on-demand?
2. **Feature Flags:** Should backends be mutually exclusive or can multiple be enabled?
3. **Tokenizer:** Should tokenizer be backend-specific or shared?
4. **API Design:** Keep model-specific APIs (Paligemma::new) or move to unified VlmModel::new?

## References

- [ONNX Runtime Documentation](https://onnxruntime.ai/)
- [TensorRT Execution Provider](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html)
- [Existing ONNX Example](../../examples/onnx/)
- [ort crate](https://docs.rs/ort/)

## Feedback Welcome

This is a proposal document. Feedback, suggestions, and concerns are welcome:
- GitHub: Open issue or comment on PR
- Discord: #kornia-rs channel

---

**Next Steps:** Once this architecture is reviewed and approved, implementation can begin with Phase 1 foundation work.
