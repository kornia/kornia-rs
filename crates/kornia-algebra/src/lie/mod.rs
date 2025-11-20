//! Lie groups for kornia-rs.
//!
//! This module provides implementations of Lie groups:
//! - SO2: Special Orthogonal group in 2D (rotations)
//! - SO3: Special Orthogonal group in 3D (rotations)
//! - SE2: Special Euclidean group in 2D (rotations + translations)
//! - SE3: Special Euclidean group in 3D (rotations + translations)

pub mod se2;
pub mod se3;
pub mod so2;
pub mod so3;

// Re-export types at module root for convenience
pub use se2::SE2;
pub use se3::SE3;
pub use so2::SO2;
pub use so3::SO3;
