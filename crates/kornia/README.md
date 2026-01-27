# Kornia

[![Crates.io](https://img.shields.io/crates/v/kornia.svg)](https://crates.io/crates/kornia)
[![Documentation](https://docs.rs/kornia/badge.svg)](https://docs.rs/kornia)
[![License](https://img.shields.io/crates/l/kornia.svg)](https://github.com/kornia/kornia/blob/main/LICENSE)

> **Low-level Computer Vision Library in Rust.**

## 🚀 Overview

`kornia` is the main entry point for the Kornia computer vision ecosystem in Rust. It aggregates specialized crates into a single, cohesive library, providing a comprehensive suite of tools for image processing, geometric transformations, IO, and AI integration. It is designed to be **fast**, **type-safe**, and **easy to use**.

## 🔑 Ecosystem

This crate re-exports functionality from the following specialized crates:

*   **`kornia-image`**: Strongly-typed image container and memory management.
*   **`kornia-tensor`**: N-dimensional tensor library for low-level data manipulation.
*   **`kornia-io`**: Image and video reading/writing (JPEG, PNG, GStreamer, V4L2).
*   **`kornia-imgproc`**: Image processing algorithms (resize, color, filter, warp).
*   **`kornia-3d`**: 3D vision, point clouds, and geometry.
*   **`kornia-algebra`**: Linear algebra and Lie theory (vectors, matrices, SO3, SE3).
*   **`kornia-vlm`**: Vision Language Models (PaliGemma, SmolVLM).

## 📦 Installation

Add `kornia` to your `Cargo.toml`. You can enable specific features to reduce build times and dependencies:

```toml
[dependencies]
kornia = { version = "0.1.0", features = ["image", "io", "imgproc"] }
```

## 🛠️ Usage

The `prelude` module provides easy access to common types and traits.

```rust
use kornia::image::{Image, ImageSize, allocator::CpuAllocator};
use kornia::io::functional as F;
use kornia::imgproc::resize::{resize_fast_rgb, InterpolationMode};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Read an image
    let image = F::read_image_any_rgb8("tests/data/dog.jpeg")?;
    println!("Loaded image: {:?}", image.size());

    // 2. Resize it
    let new_size = ImageSize { width: 128, height: 128 };
    let mut resized = Image::from_size_val(new_size, 0, CpuAllocator)?;

    resize_fast_rgb(&image, &mut resized, InterpolationMode::Bilinear)?;

    println!("Resized image: {:?}", resized.size());

    Ok(())
}
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) and join the discussion on GitHub.

## 📄 License

This project is licensed under the Apache-2.0 License.