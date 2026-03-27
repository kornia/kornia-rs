# onnx-siglip2

A minimal Rust example for running **SigLIP2 vision encoder inference** using
**ONNX Runtime** via the `ort` crate.

This example demonstrates:

- Resizing and preprocessing an input image using Kornia primitives
- Running a SigLIP2 ONNX vision encoder via ONNX Runtime
- Producing a fixed-size image embedding vector

---

## Getting the Model

The SigLIP2 ONNX model is hosted on the Kornia HuggingFace organisation:

```
https://huggingface.co/kornia
```

Download the exported ONNX vision encoder from there before running.

---

## Requirements

- Rust (stable, ≥ 1.82)
- ONNX Runtime shared library (`libonnxruntime.so` / `onnxruntime.dll` / `libonnxruntime.dylib`)

### Install ONNX Runtime

**Linux (Ubuntu):**
```bash
sudo apt install libonnxruntime-dev
```

**macOS (Homebrew):**
```bash
brew install onnxruntime
```

**Windows:** Download from the [ONNX Runtime releases page](https://github.com/microsoft/onnxruntime/releases) and note the path to `onnxruntime.dll`.

---

## Running the Example

```bash
cargo run -p onnx-siglip2 -- \
  --image-path path/to/image.jpg \
  --onnx-model-path path/to/siglip2_vision.onnx \
  --ort-dylib-path /path/to/libonnxruntime.so
```

Set `RUST_LOG=info` to see timing and embedding shape output:

```bash
RUST_LOG=info cargo run -p onnx-siglip2 -- \
  --image-path path/to/image.jpg \
  --onnx-model-path path/to/siglip2_vision.onnx \
  --ort-dylib-path /path/to/libonnxruntime.so
```

Example output:

```
[INFO  onnx_siglip2] inference time: 58.60 ms
[INFO  onnx_siglip2] embedding shape: [1, 768]
```

---

## Notes

- This is a **reference example** for inference only — no training or model conversion logic
- Image preprocessing (rescale → bicubic resize → normalize) uses existing Kornia ops
- Dependencies are intentionally kept minimal
