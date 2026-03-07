# Kornia: kornia-io

[![Crates.io](https://img.shields.io/crates/v/kornia-io.svg)](https://crates.io/crates/kornia-io)
[![Documentation](https://docs.rs/kornia-io/badge.svg)](https://docs.rs/kornia-io)
[![License](https://img.shields.io/crates/l/kornia-io.svg)](https://github.com/kornia/kornia/blob/main/LICENSE)

> **IO utilities for loading and saving images, tensors, and point clouds in Rust.**

## 🚀 Overview
`kornia-io` provides convenient functions to read/write common image formats, video streams, and binary tensor data, bridging the gap between disk storage and the core Kornia data structures.

## 🔑 Key Features
- Image decoding/encoding (PNG, JPEG, TIFF, etc.)
- Video frame extraction via ffmpeg bindings
- Tensor serialization (npy, npz)
- Point‑cloud import/export (PLY, OBJ)
- Seamless integration with `kornia-image` and `kornia-tensor`

## 📦 Installation
```toml
[dependencies]
kornia-io = "0.1.11"
```

## 🛠️ Usage
```rust
use kornia_io::image::load_image;
let img = load_image("photo.png")?;
```

## 🤝 Contributing
Contributions are welcome! See our [Contributing Guidelines](CONTRIBUTING.md).

## 📄 License
Apache-2.0
