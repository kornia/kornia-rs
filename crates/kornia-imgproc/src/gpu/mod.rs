//! Experimental GPU-accelerated image processing kernels.
//!
//! Two backends are supported, each behind its own feature flag:
//!
//! | Feature      | Backend        | Use for                              |
//! |--------------|----------------|--------------------------------------|
//! | `gpu-cubecl` | CubeCL/CUDA    | General kernels; best for upscale    |
//! | `gpu-cuda`   | native NVRTC   | Downscale; uses `__ldg` (read-only cache) |
//!
//! # Usage (CubeCL)
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

/// GPU-accelerated color conversion kernels (CubeCL).
#[cfg(feature = "gpu-cubecl")]
pub mod color;

/// GPU-accelerated resize kernels via CubeCL (nearest-neighbor and bilinear, f32).
#[cfg(feature = "gpu-cubecl")]
pub mod resize;

/// Native CUDA downscale kernels using `__ldg` read-only cache (feature `gpu-cuda`).
#[cfg(feature = "gpu-cuda")]
pub mod resize_cuda;

/// Native CUDA warp-affine kernels (bilinear and nearest-neighbor, feature `gpu-cuda`).
#[cfg(feature = "gpu-cuda")]
pub mod warp_affine_cuda;

/// Native CUDA color-space conversion kernels (feature `gpu-cuda`).
#[cfg(feature = "gpu-cuda")]
pub mod color_cuda;

/// CUDA texture object RAII wrapper (used internally by warp-affine kernels).
#[cfg(feature = "gpu-cuda")]
mod texture;
