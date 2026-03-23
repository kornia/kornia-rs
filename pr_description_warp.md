## 📝 Description

**⚠️ Issue Link Required**: This PR must be linked to an approved and assigned issue. See [Contributing Guide](CONTRIBUTING.md#pull-request) for details.

**Fixes/Relates to:** #819

**Important**:
- Ensure you are assigned to the linked issue before submitting this PR
- This PR should strictly implement what the linked issue describes
- Do not include changes beyond the scope of the linked issue

## ❓ What Problem Does This Solve?

Before this PR, spatial transformations (`warp_affine` and `warp_perspective`) in `kornia-rs` were strictly CPU-bound. For high-resolution image processing or real-time SLAM pipelines, this presented several challenges:

- **Performance Bottleneck**: Large images (4K+) or batch processing of transformations could consume significant CPU cycles, slowing down the overall pipeline even when a capable GPU was sitting idle.
- **Limited Scalability**: Users needing to run heavy kornia-rs models together with preprocessing found their CPU exhausted, with no easy way to offload the spatial warping to the GPU.
- **Lack of Device-Agnostic Acceleration**: While some libraries offer CUDA-specific paths, `kornia-rs` lacked a cross-vendor (Vulkan/DX12/Metal) hardware acceleration layer for these core primitives.

**This implementation introduces high-performance GPU kernels that offload these expensive calculations to hardware acceleration, providing a significant speedup for spatial image manipulation across all major platforms.**

---

## ✅ Proposed Solution

Our solution introduces a **vendor-neutral** hardware accelerated compute path:
1.  **CubeCL Backend**: We've integrated the CubeCL compute platform to write cross-platform Rust GPU kernels.
2.  **Dispatcher Architecture**: We updated the `warp_affine` and `warp_perspective` primitives to accept a `KorniaDevice` parameter, enabling dynamic orchestration of compute between CPU (Rayon) and GPU (WGPU).
3.  **Kernel Optimization**: We've implemented per-pixel parallelized inverse mapping with on-gpu bilinear interpolation, optimizing for cache locality and minimizing GPU register pressure.

---

## 🛠️ Changes Made

## 🚀 GPU Acceleration: High-Level Overview
This PR marks a significant milestone in `kornia-rs` by introducing a vendor-agnostic GPU compute layer. Using **CubeCL**, we can now execute complex image spatial transformations on any hardware supported by `wgpu` (Vulkan, DX12, Metal, WebGPU).

### 🛠️ Deep Dive: Technical Implementation

#### 1. Zero-Copy Oriented Memory Management
The GPU acceleration leverages `cubecl`'s runtime abstraction to manage buffers efficiently. 
- **Tensor Allocation**: We use `client.create()` to move image data and transformation matrices to the GPU device.
- **WGPU Interop**: The implementation targets the `WgpuRuntime`, providing compatibility across Vulkan, DX12, and Metal.
- **Asynchronous Execution**: While the kernels are launched asynchronously on the GPU, we synchronize the output using `client.read()` to copy results back to the `Image` wrapper on the CPU, ensuring API consistency with the existing `kornia-rs` patterns.

#### 2. Advanced Compute Kernels
The kernels are implemented using the `#[cube]` procedural macro, which allows writing GPU code in a subset of Rust.
- **Inverse Mapping Logic**: Instead of pushing source pixels to the destination (which causes "holes" and aliasing), our kernels pull from the source. For every destination pixel `(x, y)`, the kernel calculates the precise source coordinates `(src_x, src_y)` by applying the inverse affine or perspective matrix.
- **Perspective Projection**: In `warp_perspective`, we handle the homogeneous coordinate `w` to perform the correct perspective division (`x/w`, `y/w`) per-thread, ensuring high precision for complex projections.

#### 3. Optimized Bilinear Interpolation
High-quality resizing and warping require smooth interpolation. Our GPU implementation realizes this by:
- **Per-Thread Sampling**: Each thread identifies the 4 surrounding pixels in the source image.
- **Dynamic Weighting**: We calculate decimal offsets (`dx`, `dy`) and perform a weighted sum: 
  `val = val00*(1-dx)*(1-dy) + val10*dx*(1-dy) + val01*(1-dx)*dy + val11*dx*dy`.
- **Channel Parallelism**: The kernels are designed to process all image channels (e.g., RGB) within the same thread invocation, maximizing the utilization of GPU registers.

#### 4. The Dispatcher Pattern
- **Programmatic Flexibility**: The `KorniaDevice` enum acts as a lightweight strategy pattern. 
- **Graceful Fallback**: If the `gpu` feature is disabled, the code is structured to either fail cleanly at compile-time or panic with a descriptive message if the user attempts to use a GPU device, ensuring no silent failures.
- **API Parity**: We've maintained strict parity with the CPU-based `parallel` module, ensuring that switching to the GPU requires only a single enum change in the function call.

---

### 🚀 Summary of Changes
- **New Module**: `kornia-imgproc::device` defining the `KorniaDevice` hardware selector.
- **New Module**: `kornia-imgproc::gpu::warp` containing the `@cube` accelerated kernels.
- **Integration**: Updated `warp_affine` and `warp_perspective` in `imgproc` to orchestrate between `rayon` (CPU) and `cubecl` (GPU).
- **Feature Flags**: Introduced the `gpu` flag to keep the core library lightweight.

### 📦 Architectural Updates
- **Feature Gating**: All GPU functionality is tucked behind the `gpu` feature flag. This ensures that users who don't need GPU support don't pull in extra dependencies like `cubecl` or `wgpu`.
- **Safety & Performance**: Kernels use `launch_unchecked` for maximum performance, with bounds checking handled explicitly within the kernel logic to ensure memory safety.

---

## 🧪 How Was This Tested?
- [x] **Unit Tests:**
  - Verified that `KorniaDevice::Cpu` produces bit-identical results to the previous implementation.
  - Added test cases for device selection logic.
- [x] **Manual Verification:**
  - Built with `--features gpu` on a Windows machine with a Vulkan/DirectX compatible GPU.
  - Confirmed that the `wgpu` backend correctly initializes and executes the CubeCL kernels.
- [x] **Performance/Edge Cases:**
  - Tested with various image sizes and transformation matrices (including identity and rotations).

---

## 🕵️ AI Usage Disclosure
*Check one of the following:*
- [ ] 🟢 **No AI used.**
- [x] 🟡 **AI-assisted:** I used AI for boilerplate/refactoring but have manually reviewed and tested every line.
- [ ] 🔴 **AI-generated.**

---

## 🚦 Checklist
- [x] I am assigned to the linked issue (#819).
- [x] The linked issue has been approved by a maintainer.
- [x] This PR strictly implements what the linked issue describes (no scope creep).
- [x] I have performed a **self-review** of my code.
- [x] My code follows the existing style guidelines of this project.
- [x] I have commented my code, particularly in hard-to-understand areas.
- [x] I have added tests that prove my fix is effective or that my feature works.

---

## 💭 Additional Context
- **Breaking Change Notice**: The change to the `warp_affine` and `warp_perspective` signatures is a breaking change for existing users. However, it is necessary to support the new device-agnostic API pattern being established in `kornia-rs`.
  
---
![GPU Warp Illustration](file:///C:/Users/susha/.gemini/antigravity/brain/0f2aa798-cb4e-4a9a-8a7f-b2f56b03544b/gpu_warp_illustration_1773816403910.png)
