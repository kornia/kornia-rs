# Kornia: kornia-tensor

[![Crates.io](https://img.shields.io/crates/v/kornia-tensor.svg)](https://crates.io/crates/kornia-tensor)
[![Documentation](https://docs.rs/kornia-tensor/badge.svg)](https://docs.rs/kornia-tensor)
[![License](https://img.shields.io/crates/l/kornia-tensor.svg)](https://github.com/kornia/kornia/blob/main/LICENSE)

> **Tensor operations and core data structures for computer vision in Rust.**

## 🚀 Overview
`kornia-tensor` provides a high‑performance tensor library with support for multi‑dimensional arrays, arithmetic operations, and integration with GPU backends. It forms the foundation for all other Kornia crates.

## 🔑 Key Features
- N‑dimensional tensor type with shape inference
- SIMD‑accelerated arithmetic and linear algebra
- Interoperability with `ndarray` and `tch`
- Optional GPU support via `wgpu` or `cuda` features

## 📦 Installation
```toml
[dependencies]
kornia-tensor = "0.1.11"
```

## 🛠️ Usage
```rust
use kornia_tensor::Tensor;
let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], [2, 2]);
let b = a * 2.0;
```

## 🤝 Contributing
Contributions are welcome! See our [Contributing Guidelines](CONTRIBUTING.md).

## 📄 License
Apache-2.0
