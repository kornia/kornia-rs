# Kornia: kornia-tensor

[![Crates.io](https://img.shields.io/crates/v/kornia-tensor.svg)](https://crates.io/crates/kornia-tensor)
[![Documentation](https://docs.rs/kornia-tensor/badge.svg)](https://docs.rs/kornia-tensor)
[![License](https://img.shields.io/crates/l/kornia-tensor.svg)](https://github.com/kornia/kornia/blob/main/LICENSE)

> **Lightweight, high-performance tensor library for computer vision.**

## 🚀 Overview

`kornia-tensor` is a lightweight multi-dimensional array library designed specifically for computer vision applications. It serves as the foundational data structure for the Kornia ecosystem, providing a runtime memory model (CPU / CUDA / unified), zero-copy interop, and safe operations.

## 🔑 Key Features

*   **Multi-Dimensional Arrays:** efficient handling of N-dimensional data with arbitrary shapes and strides.
*   **Zero-Copy Operations:** Supports views and reshaping without copying underlying data.
*   **Runtime memory model:** one concrete `Tensor<T, N>` spans Host / Device / Unified memory; the allocator is a runtime handle, not a type parameter.
*   **Ergonomic constructors:** host is the default (no allocator argument); `_in` variants accept a custom allocator handle.
*   **Type Safety:** Uses const generics for compile-time dimension checking (e.g., `Tensor<f32, 2>`).
*   **Serialization:** Optional support for `serde` and `bincode` for easy data persistence.

## 🧠 Memory Model

A tensor's location is described at **runtime**, not in its type. `Tensor<T, N>` carries no
allocator/device type parameter. Its `TensorStorage` holds:

* an `owner: Box<dyn MemoryResource>` — frees the buffer on drop and reports the `MemoryDomain`
  (`Host`, `Device { id }`, or `Unified { id }`);
* an `AllocHandle` (`Arc<dyn TensorAllocator>`) — a cheap runtime handle used only when an op
  allocates (e.g. `zeros`, `cast`); never touched on the element-access hot path;
* a cached pointer read directly by indexing / `as_slice`.

`as_slice`/`as_mut_slice` panic on a non-host (device) tensor — move data with `to_host` / `to_cuda`
(feature `cudarc`) first. Because location is runtime, a single `Vec<Tensor<T, N>>` can hold tensors
from different domains.

## 📦 Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
kornia-tensor = "0.1.0"
```

## 🛠️ Usage

Here is a simple example of creating and using a tensor:

```rust
use kornia_tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Create a 2x3 tensor from a vector (host is the default — no allocator argument)
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor = Tensor::<f32, 2>::from_shape_vec([2, 3], data)?;

    // 2. Access elements
    if let Some(val) = tensor.get([1, 2]) {
        println!("Value at [1, 2]: {}", val); // Prints 6.0
    }

    // 3. Reshape the tensor (zero-copy if contiguous)
    let reshaped = tensor.reshape([3, 2])?;
    println!("Reshaped dimensions: {:?}", reshaped.shape);

    // 4. Create a tensor of zeros
    let zeros = Tensor::<f32, 3>::zeros([2, 2, 3]);
    println!("Zeros shape: {:?}", zeros.shape);

    // 5. Custom allocator? use the `_in` variant:
    //    let t = Tensor::<f32, 2>::from_shape_vec_in([2, 3], data, my_alloc)?;

    Ok(())
}
```

## 🧩 Modules

*   **`tensor`**: The main `Tensor` struct and core implementation.
*   **`view`**: `TensorView` for non-owning access to tensor data.
*   **`storage`**: Low-level buffer management.
*   **`allocator`**: Traits and implementations for memory allocation.

## 💡 Related Examples

You can find comprehensive examples in the `examples` folder of the repository:

*   [`onnx`](../../examples/onnx): Using tensors for ONNX model inference.
*   [`smol_vlm`](../../examples/smol_vlm): Advanced usage of tensors in VLM inference.

## 🤝 Contributing

Contributions are welcome! This crate is part of the Kornia workspace. Please refer to the main repository for contribution guidelines.

## 📄 License

This crate is licensed under the Apache-2.0 License.
