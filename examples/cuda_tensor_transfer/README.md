# CUDA Tensor Transfer Example

This example demonstrates the multi-backend tensor allocator API with CUDA support.

## Features

- CPU ↔ CUDA memory transfers
- Zero-copy operations where possible
- Explicit transfer API with `to_device()` and `to_cpu()` methods
- Support for both Tensors and Images
- Device-aware accessors

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- `cust` crate dependencies

## Usage

```bash
# Run the example
cargo run --example cuda_tensor_transfer --features cuda

# Or from the examples directory
cd examples/cuda_tensor_transfer
cargo run --release
```

## Examples Demonstrated

### 1. Basic Tensor Transfer
Creates a 2D tensor on CPU, transfers it to CUDA, and transfers it back, verifying data integrity.

### 2. Image Transfer
Demonstrates transferring an RGB image between CPU and CUDA memory.

### 3. Multiple Round-trips
Performs multiple CPU ↔ CUDA transfers to verify robustness and data consistency.

## Key Concepts

### Device Abstraction
```rust
let device = tensor.device();
println!("Device: {}", device); // "cpu" or "cuda:0"
```

### Explicit Transfers
```rust
// Transfer to CUDA
let cuda_alloc = CudaAllocator::new(0)?;
let cuda_tensor = cpu_tensor.to_device(cuda_alloc)?;

// Transfer to CPU
let cpu_tensor = cuda_tensor.to_cpu()?;
```

### Device-Aware Operations
```rust
// These operations require CPU tensors
let slice = tensor.as_slice(); // Panics if on GPU
let casted = tensor.cast::<f64>(); // CPU-only for now
```

## Architecture

The allocator API provides:
- `TensorAllocator` trait with device-specific implementations
- `Device` enum for device identification
- Explicit memory transfer methods
- Safe accessors that prevent invalid GPU memory access

## Future Work

- GPU kernel operations (element-wise ops, reductions)
- Multi-GPU support
- Metal and Vulkan backends
- Unified memory support

