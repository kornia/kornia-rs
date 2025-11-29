//! Lie groups and algebras for kornia-rs.
//!
//! This crate provides:
//! - Unified algebraic types (`types` module) using newtype pattern over glam backend
//! - Lie group implementations (SO2F32, SO3F32, SE2F32, SE3F32)

// Algebraic types and Lie groups
mod lie;
mod mat;
mod quat;
mod vec;

// Re-export types at crate root for convenience
pub use lie::{SE2F32, SE3F32, SO2F32, SO3F32};
pub use mat::{Mat2F32, Mat2F64, Mat3AF32, Mat3F32, Mat3F64, Mat4F32, Mat4F64};
pub use quat::{QuatF32, QuatF64};
pub use vec::{Vec2F32, Vec2F64, Vec3AF32, Vec3F32, Vec3F64, Vec4F32, Vec4F64};

// Isometry type aliases (explicit precision)
pub type Isometry2F32 = SE2F32;
pub type Isometry3F32 = SE3F32;
