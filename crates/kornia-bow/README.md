# Kornia: kornia-bow

[![Crates.io](https://img.shields.io/crates/v/kornia-bow.svg)](https://crates.io/crates/kornia-bow)
[![Documentation](https://docs.rs/kornia-bow/badge.svg)](https://docs.rs/kornia-bow)
[![License](https://img.shields.io/crates/l/kornia-bow.svg)](https://github.com/kornia/kornia/blob/main/LICENSE)

> **High-Performance Hierarchical Bag of Words (BoW) for Visual Place Recognition.**

## 🚀 Overview

`kornia-bow` implements a hierarchical Bag of Words vocabulary tree, optimized for fast visual place recognition and image retrieval. It allows you to train a vocabulary from a set of descriptors (e.g., ORB, SIFT) and then transform new images into sparse BoW vectors for efficient matching.

## 🔑 Key Features

*   **Hierarchical Vocabulary:** Efficient tree structure (K-means based) for fast quantization of descriptors.
*   **Fast Lookups:** Optimized traversal for transforming features into BoW vectors.
*   **Direct Indexing:** Supports direct index generation for geometric verification.
*   **Metric Agnostic:** Generic implementation supporting different distance metrics (e.g., Hamming distance for binary descriptors).
*   **Serialization:** Built-in `serde` and `bincode` support for saving and loading vocabularies.

## 📦 Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
kornia-bow = "0.1.0"
```

## 🛠️ Usage

### Training and Using a Vocabulary

```rust
use kornia_bow::{Vocabulary, BoW, metric::{Hamming, Feature}};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Train a vocabulary (offline step)
    // Create dummy descriptors for demonstration
    let descriptors = vec![Feature([0u64; 4]); 50];
    let vocab = Vocabulary::<10, Hamming<4>>::train(&descriptors, 2)?;

    // 2. Save vocabulary
    let bytes = bincode::encode_to_vec(&vocab, bincode::config::standard())?;
    // std::fs::write("vocab.bin", bytes)?;

    // 3. Load vocabulary
    // let (loaded_vocab, _): (Vocabulary<10, Hamming<4>>, usize) =
    //     bincode::decode_from_slice(&bytes, bincode::config::standard())?;

    // 4. Transform new image features to BoW vector
    let query_descriptors = vec![Feature([0u64; 4]); 10];
    let bow_vector = vocab.transform(&query_descriptors)?;

    println!("BoW Vector size: {}", bow_vector.0.len());

    Ok(())
}
```

## 🧩 Modules

*   **`bow`**: Core `BoW` and `DirectIndex` types.
*   **`constructor`**: Algorithms for training the vocabulary tree.
*   **`io`**: Input/Output utilities.
*   **`metric`**: Distance metrics (e.g., L2, Hamming).

## 🤝 Contributing

Contributions are welcome! This crate is part of the Kornia workspace. Please refer to the main repository for contribution guidelines.

## 📄 License

This crate is licensed under the Apache-2.0 License.
