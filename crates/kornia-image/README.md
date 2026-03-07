# Kornia: kornia-image

[![Crates.io](https://img.shields.io/crates/v/kornia-image.svg)](https://crates.io/crates/kornia-image)
[![Documentation](https://docs.rs/kornia-image/badge.svg)](https://docs.rs/kornia-image)
[![License](https://img.shields.io/crates/l/kornia-image.svg)](https://github.com/kornia/kornia/blob/main/LICENSE)

> **Image handling and processing utilities for Rust.**

## 🚀 Overview
`kornia-image` provides image representations, pixel access, format conversion, and basic image processing primitives built on top of `kornia-tensor`.

## 🔑 Key Features
- Typed image structs (`Image<T, C>`)
- Efficient memory layout and zero‑copy conversions
- Common operations: resize, crop, color conversion
- Integration with `kornia-io` for loading/saving

## 📦 Installation
```toml
[dependencies]
kornia-image = "0.1.11"
```

## 🛠️ Usage
```rust
use kornia_image::Image;
// Example: load an image, convert to grayscale, and resize
```

## 🤝 Contributing
Contributions are welcome! See our [Contributing Guidelines](CONTRIBUTING.md).

## 📄 License
Apache-2.0
