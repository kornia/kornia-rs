//! cubecl-based GPU/CPU compute kernels for kornia-rs.
//!
//! Prototype crate. Currently provides bilinear u8 RGB resize across two
//! cubecl runtimes (`cubecl-cuda`, `cubecl-cpu`) for cross-backend benchmarking
//! against the production NEON path in `kornia-imgproc`.

pub mod error;
pub mod resize;
pub mod runtime;

pub use error::ResizeError;
