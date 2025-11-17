//! Matrix types module.
//!
//! This module provides matrix types for kornia-rs:
//! - Mat2: 2x2 matrix
//! - Mat3: 3x3 matrix
//! - Mat3A: 3x3 aligned matrix (SIMD-optimized)
//! - Mat4: 4x4 matrix

#[macro_use]
mod mat;

mod mat2;
mod mat3;
mod mat3a;
mod mat4;

pub use mat3a::Mat3AF32;
pub use {mat2::Mat2F32, mat2::Mat2F64};
pub use {mat3::Mat3F32, mat3::Mat3F64};
pub use {mat4::Mat4F32, mat4::Mat4F64};
