# Kornia: kornia-tensor-ops

[![Crates.io](https://img.shields.io/crates/v/kornia-tensor-ops.svg)](https://crates.io/crates/kornia-tensor-ops)
[![Documentation](https://docs.rs/kornia-tensor-ops/badge.svg)](https://docs.rs/kornia-tensor-ops)
[![License](https://img.shields.io/crates/l/kornia-tensor-ops.svg)](https://github.com/kornia/kornia/blob/main/LICENSE)

> **Extension crate for higher-level tensor operations in kornia-rs.**

## 🚀 Overview

`kornia-tensor-ops` is an extension layer on top of [`kornia-tensor`](https://crates.io/crates/kornia-tensor).
While `kornia-tensor` provides the minimal, core tensor abstractions (storage, views, allocators, and memory layout), this crate houses higher-level and composite operations that build on those primitives.

This separation keeps the core tensor crate lean and composable, while providing a clear home for operations that involve more complex logic, additional trait bounds, or domain-specific computation.

## 🔑 Key Features

*   **Element-wise Arithmetic:** Add, subtract, multiply, divide tensors.
*   **Scalar Operations:** Multiply by scalar, raise to power (float and integer).
*   **Reductions:** Sum along a dimension, compute mean, find minimum.
*   **Similarity Metrics:** Dot product, cosine similarity, cosine distance.
*   **Low-level Kernels:** Standalone functions operating directly on slices for use outside the `Tensor` abstraction.
*   **Safe API:** Trait-based `TensorOps` interface for intuitive usage with `kornia-tensor`.

## 📦 Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
kornia-tensor-ops = "0.1.0"
```

## 🛠️ Usage

```rust
use kornia_tensor::{Tensor, CpuAllocator};
use kornia_tensor_ops::TensorOps;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let t1 = Tensor::<f32, 2, _>::from_shape_vec([2, 2], vec![1.0, 2.0, 3.0, 4.0], CpuAllocator)?;
    let t2 = Tensor::<f32, 2, _>::from_shape_vec([2, 2], vec![10.0, 20.0, 30.0, 40.0], CpuAllocator)?;

    // Element-wise addition
    let sum = t1.add(&t2)?;

    assert_eq!(sum.get([0, 0]), Some(&11.0));
    println!("Sum: {:?}", sum.as_slice());

    Ok(())
}
```

## 🧩 Modules

*   **`ops`**: High-level trait implementations for tensor operations (`TensorOps`).
*   **`kernels`**: Low-level kernel functions (e.g. dot product, cosine similarity) operating on slices.
*   **`error`**: Error types for tensor operation failures.

## 🏗️ Architecture Note

New tensor operations that go beyond basic storage or view manipulation should be added to this crate rather than to `kornia-tensor`. If an operation requires additional trait bounds (e.g. `num_traits::Float`) or combines multiple lower-level steps, it belongs here.

## 💡 Related Examples

You can find comprehensive examples in the `examples` folder of the repository:

*   [`metrics`](../../examples/metrics): Calculating image metrics using tensor operations.

## 🤝 Contributing

Contributions are welcome! This crate is part of the Kornia workspace. Please refer to the main repository for contribution guidelines.

## 📄 License

This crate is licensed under the Apache-2.0 License.
