# Kornia: kornia-imgproc

[![Crates.io](https://img.shields.io/crates/v/kornia-imgproc.svg)](https://crates.io/crates/kornia-imgproc)
[![Documentation](https://docs.rs/kornia-imgproc/badge.svg)](https://docs.rs/kornia-imgproc)
[![License](https://img.shields.io/crates/l/kornia-imgproc.svg)](https://github.com/kornia/kornia/blob/main/LICENSE)

> **Image processing primitives for Rust.**

## 🚀 Overview
`kornia-imgproc` offers a collection of classic image processing algorithms (filters, edge detection, morphological ops) built on top of `kornia-tensor` and `kornia-image`.

## 🔑 Key Features
- Convolution and filtering utilities
- Edge detection (Sobel, Canny)
- Morphological operations (dilate, erode, opening, closing)
- Color space conversions and histogram utilities

## 📦 Installation
```toml
[dependencies]
kornia-imgproc = "0.1.11"
```

## 🛠️ Usage
```rust
use kornia_imgproc::filters::gaussian_blur;
// Example: apply Gaussian blur to an image
```

## 🤝 Contributing
Contributions are welcome! See our [Contributing Guidelines](CONTRIBUTING.md).

## 📄 License
Apache-2.0
