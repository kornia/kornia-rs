# Kornia: kornia-tensor-ops

[![Crates.io](https://img.shields.io/crates/v/kornia-tensor-ops.svg)](https://crates.io/crates/kornia-tensor-ops)
[![Documentation](https://docs.rs/kornia-tensor-ops/badge.svg)](https://docs.rs/kornia-tensor-ops)
[![License](https://img.shields.io/crates/l/kornia-tensor-ops.svg)](https://github.com/kornia/kornia/blob/main/LICENSE)

> **Tensor operation utilities for computer vision in Rust.**

## 🚀 Overview
`kornia-tensor-ops` provides higher‑level tensor operations (convolutions, reductions, broadcasting utilities) built on top of `kornia-tensor`.

## 🔑 Key Features
- Convolution and pooling layers
- Reduction ops (sum, mean, max, min) with axis support
- Broadcasting and reshaping helpers
- GPU‑accelerated kernels via optional features

## 📦 Installation
```toml
[dependencies]
kornia-tensor-ops = "0.1.11"
```

## 🛠️ Usage
```rust
use kornia_tensor_ops::conv2d;
// Example: apply a 2D convolution to a tensor
```

## 🤝 Contributing
Contributions are welcome! See our [Contributing Guidelines](CONTRIBUTING.md).

## 📄 License
Apache-2.0
