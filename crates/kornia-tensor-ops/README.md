# Kornia: kornia-tensor-ops

[![Crates.io](https://img.shields.io/crates/v/kornia-tensor-ops.svg)](https://crates.io/crates/kornia-tensor-ops)
[![Documentation](https://docs.rs/kornia-tensor-ops/badge.svg)](https://docs.rs/kornia-tensor-ops)
[![License](https://img.shields.io/crates/l/kornia-tensor-ops.svg)](https://github.com/kornia/kornia/blob/main/LICENSE)

> **Core tensor operations and kernels for kornia-rs.**

## 🚀 Overview

`kornia-tensor-ops` provides essential mathematical operations for `kornia-tensor`. It implements element-wise arithmetic, reductions, and other low-level kernels required for building complex computer vision algorithms.

## 🔑 Key Features

*   **Arithmetic Operations:** Add, subtract, multiply, divide tensors efficiently.
*   **Broadcasting:** (Planned/Partial) Support for broadcasting operations across dimensions.
*   **Kernels:** Optimized low-level implementations of common mathematical functions.
*   **Safe API:** Trait-based operator overloading for intuitive usage with `kornia-tensor`.

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

*   **`ops`**: High-level trait implementations for tensor operations.
*   **`kernels`**: Low-level kernel implementations.

## 🤝 Contributing

Contributions are welcome! This crate is part of the Kornia workspace. Please refer to the main repository for contribution guidelines.

## 📄 License

This crate is licensed under the Apache-2.0 License.