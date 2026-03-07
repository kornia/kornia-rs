# Kornia: kornia-bow

[![Crates.io](https://img.shields.io/crates/v/kornia-bow.svg)](https://crates.io/crates/kornia-bow)
[![Documentation](https://docs.rs/kornia-bow/badge.svg)](https://docs.rs/kornia-bow)
[![License](https://img.shields.io/crates/l/kornia-bow.svg)](https://github.com/kornia/kornia/blob/main/LICENSE)

> **Bag‑of‑words (BoW) utilities for visual place recognition in Rust.**

## 🚀 Overview
`kornia-bow` implements bag‑of‑words pipelines, feature extraction, vocabulary building, and image retrieval, leveraging the core Kornia tensor and image infrastructure.

## 🔑 Key Features
- Feature descriptor extraction (SIFT, ORB, etc.)
- Vocabulary construction and quantization
- Efficient inverted index for image retrieval
- Integration with `kornia-image` and `kornia-tensor`

## 📦 Installation
```toml
[dependencies]
kornia-bow = "0.1.11"
```

## 🛠️ Usage
```rust
use kornia_bow::vocabulary::Vocabulary;
// Example: build a BoW model and query images
```

## 🤝 Contributing
Contributions are welcome! See our [Contributing Guidelines](CONTRIBUTING.md).

## 📄 License
Apache-2.0
