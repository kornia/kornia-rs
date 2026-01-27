# Kornia: kornia-tensor

[![Crates.io](https://img.shields.io/crates/v/kornia-tensor.svg)](https://crates.io/crates/kornia-tensor)
[![Documentation](https://docs.rs/kornia-tensor/badge.svg)](https://docs.rs/kornia-tensor)
[![License](https://img.shields.io/crates/l/kornia-tensor.svg)](https://github.com/kornia/kornia/blob/main/LICENSE)

> **Lightweight, high-performance tensor library for computer vision.**

## 🚀 Overview

`kornia-tensor` is a lightweight multi-dimensional array library designed specifically for computer vision applications. It serves as the foundational data structure for the Kornia ecosystem, providing efficient memory management, custom allocators, and safe, zero-copy operations.

## 🔑 Key Features

*   **Multi-Dimensional Arrays:** efficient handling of N-dimensional data with arbitrary shapes and strides.
*   **Zero-Copy Operations:** Supports views and reshaping without copying underlying data.
*   **Custom Allocators:** Trait-based memory management allowing for different backends (CPU, generic storage).
*   **Type Safety:** Uses const generics for compile-time dimension checking (e.g., `Tensor<f32, 2>`).
*   **Serialization:** Optional support for `serde` and `bincode` for easy data persistence.

## 📦 Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
kornia-tensor = "0.1.0"
```

## 🛠️ Usage

Here is a simple example of creating and using a tensor:

```rust
use kornia_tensor::{Tensor, CpuAllocator};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Create a 2x3 tensor from a vector
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor = Tensor::<f32, 2, _>::from_shape_vec([2, 3], data, CpuAllocator)?;

    // 2. Access elements
    if let Some(val) = tensor.get([1, 2]) {
        println!("Value at [1, 2]: {}", val); // Prints 6.0
    }

    // 3. Reshape the tensor (zero-copy if contiguous)
    let reshaped = tensor.reshape([3, 2])?;
    println!("Reshaped dimensions: {:?}", reshaped.shape);

    // 4. Create a tensor of zeros
    let zeros = Tensor::<f32, 3, _>::zeros([2, 2, 3], CpuAllocator);
    println!("Zeros shape: {:?}", zeros.shape);

    Ok(())
}
```

## 🧩 Modules

*   **`tensor`**: The main `Tensor` struct and core implementation.
*   **`view`**: `TensorView` for non-owning access to tensor data.
*   **`storage`**: Low-level buffer management.
*   **`allocator`**: Traits and implementations for memory allocation.

## 🤝 Contributing

Contributions are welcome! This crate is part of the Kornia workspace. Please refer to the main repository for contribution guidelines.

## 📄 License

This crate is licensed under the Apache-2.0 License.
