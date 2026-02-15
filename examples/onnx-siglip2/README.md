# onnx-siglip2

A minimal Rust example for running **SigLIP-2 vision encoder inference**
using **ONNX Runtime** on macOS.

This crate demonstrates:

- Loading a SigLIP-2 vision ONNX model
- Dynamically loading `libonnxruntime.dylib`
- Running single-image inference
- Producing a fixed-size embedding vector

The goal is to keep the example **simple, reproducible, and portable**.

---

## Requirements

- Rust (stable)
- macOS (Apple Silicon or Intel)
- ONNX Runtime installed via Homebrew

## Install ONNX Runtime:

```bash
brew install onnxruntime
```

Verify that the dylib exists:

```bash
ls /opt/homebrew/lib/libonnxruntime.dylib
```

## Running the example

```bash
cargo run -- \
  --image-path path/to/image.jpg \
  --onnx-model-path path/to/siglip2_vision.onnx \
  --ort-dylib-path /opt/homebrew/lib/libonnxruntime.dylib
```

Example output:

```bash
inference ms: 58.60
embedding shape: [1, 768]
```

## Notes

- The SigLIP-2 ONNX model is not included
- This example focuses on inference correctness, not training
- Dependencies are intentionally kept minimal
- The ONNX Runtime library is loaded explicitly at runtime for portability

## Tests

A minimal integration test is provided to validate the presence of
ONNX Runtime on the system.

Run tests with :

```bash
cargo test
```

## Motivation

This example exists to:

- Validate ONNX Runtime integration in Rust
- Provide a reference for vision model inference workflows
- Enable future work around benchmarking and edge deployment
