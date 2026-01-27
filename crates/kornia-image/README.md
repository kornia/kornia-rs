# Kornia: kornia-image

[![Crates.io](https://img.shields.io/crates/v/kornia-image.svg)](https://crates.io/crates/kornia-image)
[![Documentation](https://docs.rs/kornia-image/badge.svg)](https://docs.rs/kornia-image)
[![License](https://img.shields.io/crates/l/kornia-image.svg)](https://github.com/kornia/kornia/blob/main/LICENSE)

> **Image types and traits for computer vision in Rust.**

## 🚀 Overview

`kornia-image` provides a strongly-typed image representation for computer vision applications. It is built on top of `kornia-tensor` and offers a flexible memory layout that supports various pixel formats and data types. The library is designed to be zero-copy where possible and allows for easy integration with other libraries in the ecosystem.

## 🔑 Key Features

* **Strongly-typed Image Struct:** The `Image<T, C, A>` struct ensures compile-time safety for pixel types (`T`) and channel counts (`C`).
* **Flexible Memory Management:** Uses the `ImageAllocator` trait to support different memory backends (e.g., CPU, potentially GPU).
* **Rich Operations:** built-in support for casting, scaling, channel splitting/merging, and pixel access.
* **Arrow Integration:** Optional support for converting images to Arrow format for data processing pipelines.
* **Color Space Safety:** Includes typed wrappers for color spaces (e.g., `Rgb8`, `Gray8`) to prevent mixing up image formats.

## 📦 Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
kornia-image = "0.1.0"
```

## 🛠️ Usage

Here is a simple example showing how to create and manipulate an image:

```rust
use kornia_image::{Image, ImageSize, allocator::CpuAllocator};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Create a dummy RGB image (3 channels) with u8 data
    let image_size = ImageSize { width: 10, height: 20 };
    let data = vec![0u8; 10 * 20 * 3];
    let image = Image::<u8, 3, _>::new(image_size, data, CpuAllocator)?;

    println!("Image size: {:?}", image.size());
    println!("Channels: {}", image.num_channels());

    // 2. Cast to f32 and scale values to [0, 1]
    let image_f32 = image.cast_and_scale::<f32>(1.0 / 255.0)?;

    // 3. Access specific pixel (slow, for convenience)
    let pixel_val = image_f32.get_pixel(5, 5, 0)?;
    println!("Pixel at (5,5) ch 0: {}", pixel_val);

    // 4. Split into individual channels
    let channels = image_f32.split_channels()?;
    println!("Split into {} single-channel images", channels.len());

    Ok(())
}
```

## 🧩 Modules

*   **`image`**: Core `Image` struct, `ImageSize`, `ImageLayout`, and `PixelFormat`.
*   **`allocator`**: Memory management utilities and the `ImageAllocator` trait.
*   **`ops`**: Basic image operations.
*   **`color_spaces`**: Typed wrappers for common color spaces.
*   **`arrow`**: (Optional) Utilities for Apache Arrow integration.

## 🤝 Contributing

Contributions are welcome! This crate is part of the Kornia workspace. Please refer to the main repository for contribution guidelines.

## 📄 License

This crate is licensed under the Apache-2.0 License.