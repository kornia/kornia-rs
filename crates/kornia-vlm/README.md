# Kornia: kornia-vlm

[![Crates.io](https://img.shields.io/crates/v/kornia-vlm.svg)](https://crates.io/crates/kornia-vlm)
[![Documentation](https://docs.rs/kornia-vlm/badge.svg)](https://docs.rs/kornia-vlm)
[![License](https://img.shields.io/crates/l/kornia-vlm.svg)](https://github.com/kornia/kornia/blob/main/LICENSE)

> **Vision Language Model (VLM) utilities for Rust.**

## 🚀 Overview
`kornia-vlm` provides tools to integrate vision-language models, enabling image captioning, visual question answering, and multimodal embeddings within the Kornia ecosystem.

## 🔑 Key Features
- Pre‑trained VLM inference wrappers (e.g., CLIP, BLIP)
- Tokenization and text encoding utilities
- Image‑text similarity scoring
- Easy integration with `kornia-tensor` for tensor‑based processing

## 📦 Installation
```toml
[dependencies]
kornia-vlm = "0.1.11"
```

## 🛠️ Usage
```rust
use kornia_vlm::model::ClipModel;
let model = ClipModel::new()?;
let embedding = model.encode_image(&image)?;
```

## 🤝 Contributing
Contributions are welcome! See our [Contributing Guidelines](CONTRIBUTING.md).

## 📄 License
Apache-2.0
