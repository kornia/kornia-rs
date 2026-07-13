//! Experimental GPU-accelerated image processing kernels.
//!
//! Backed by native CUDA kernels compiled at runtime via NVRTC, using `cudarc`
//! for device memory and launches. The whole module is gated on the `cuda`
//! feature at its declaration in `lib.rs`.

/// Native CUDA downscale kernels using `__ldg` read-only cache.
pub mod resize;

/// Native CUDA warp-affine kernels (bilinear and nearest-neighbor).
pub mod warp_affine;

/// Native CUDA warp-perspective kernels (homography, bilinear / nearest / bicubic / Lanczos-3).
pub mod warp_perspective;

/// Native CUDA color-space conversion kernels.
pub mod color;

/// CUDA texture object RAII wrapper (used internally by warp-affine kernels).
mod texture;
