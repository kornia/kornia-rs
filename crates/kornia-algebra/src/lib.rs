//! Lie groups and algebras for kornia-rs.
//!
//! This crate provides:
//! - Unified algebraic types (`types` module) using newtype pattern over glam backend
//! - Lie group implementations (SO2, SO3, SE2, SE3)

// Algebraic types and Lie groups
mod lie;
mod matrix;
mod quat;
mod vector;

// Re-export types at crate root for convenience
pub use lie::{SE2, SE3, SO2, SO3};
pub use matrix::{Mat2F32, Mat2F64, Mat3AF32, Mat3F32, Mat3F64, Mat4F32, Mat4F64};
pub use quat::{QuatF32, QuatF64};
pub use vector::{Vec2F32, Vec2F64, Vec3AF32, Vec3F32, Vec3F64, Vec4F32, Vec4F64};

// Backwards-compatible type aliases for default single-precision types.
pub type Mat2 = Mat2F32;
pub type Mat3 = Mat3F32;
pub type Mat3A = Mat3AF32;
pub type Mat4 = Mat4F32;
pub type Quat = QuatF32;

// Type aliases for explicit precision (single precision / f32)
pub type Vec4 = Vec4F32;

// Isometry type aliases
pub type Isometry2F32 = SE2;
pub type Isometry3F32 = SE3;

// Re-export glam types that are used directly
pub use glam::Affine3A;
