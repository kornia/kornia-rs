//! Lie groups for kornia-rs.
//!
//! This module provides implementations of Lie groups:
//! - SO2F32: Special Orthogonal group in 2D (rotations)
//! - SO3F32: Special Orthogonal group in 3D (rotations)
//! - SE2F32: Special Euclidean group in 2D (rotations + translations)
//! - SE3F32: Special Euclidean group in 3D (rotations + translations)
//! - RxSO3F32: Direct product of positive scalars and rotations in 3D
//! - Sim3F32: Similarity group in 3D (scaling, rotation and translation)

pub mod rxso3;
pub mod se2;
pub mod se3;
pub mod sim3;
pub mod so2;
pub mod so3;

// Re-export types at module root for convenience
pub use rxso3::RxSO3F32;
pub use se2::SE2F32;
pub use se3::SE3F32;
pub use sim3::Sim3F32;
pub use so2::SO2F32;
pub use so3::SO3F32;
