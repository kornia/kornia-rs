//! Experimental GPU-accelerated image processing kernels.
//!
//! Backed by native CUDA kernels compiled at runtime via NVRTC (feature
//! `cuda`), using `cudarc` for device memory and launches.

/// Native CUDA downscale kernels using `__ldg` read-only cache (feature `cuda`).
#[cfg(feature = "cuda")]
pub mod resize_cuda;

/// Native CUDA warp-affine kernels (bilinear and nearest-neighbor, feature `cuda`).
#[cfg(feature = "cuda")]
pub mod warp_affine_cuda;

/// Native CUDA color-space conversion kernels (feature `cuda`).
#[cfg(feature = "cuda")]
pub mod color_cuda;

/// CUDA texture object RAII wrapper (used internally by warp-affine kernels).
#[cfg(feature = "cuda")]
mod texture;
