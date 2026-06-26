//! Experimental CubeCL-backed image processing kernels (feature `gpu-cubecl`).
//!
//! This module is the first home for GPU-accelerated operations in `kornia-imgproc`.
//! The current scope is device-to-device kernels only; host-device transfer APIs
//! (`to_device` / `to_host`) will be added in a later PR.
//!
//! # Usage
//!
//! ```no_run
//! use cubecl::Runtime;
//! use cubecl_cuda::CudaRuntime;
//! use kornia_imgproc::gpu::color::launch_gray_from_rgb_f32;
//!
//! let device = <CudaRuntime as Runtime>::Device::default();
//! let client = CudaRuntime::client(&device);
//! // allocate device buffers for a 1920×1080 image …
//! ```

/// GPU-accelerated color conversion kernels.
pub mod color;

/// GPU-accelerated resize kernels (nearest-neighbor and bilinear, f32).
pub mod resize;
