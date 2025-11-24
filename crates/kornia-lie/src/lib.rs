#![deny(missing_docs)]
#![doc = env!("CARGO_PKG_DESCRIPTION")]
//!
//! # Kornia Lie Groups
//!
//! This crate provides implementations of Lie groups and Lie algebras commonly used in robotics
//! and computer vision for representing rotations and rigid body transformations.
//!
//! ## Supported Groups
//!
//! - **SO(2)**: 2D rotation group
//! - **SE(2)**: 2D rigid body transformations (rotation + translation)
//! - **SO(3)**: 3D rotation group
//! - **SE(3)**: 3D rigid body transformations (rotation + translation)
//!
//! ## Example
//!
//! ```rust
//! use kornia_lie::so3::SO3;
//!
//! // Create a rotation from axis-angle representation
//! let axis_angle = [0.0, 0.0, std::f32::consts::PI / 2.0];
//! let rotation = SO3::from_vec(axis_angle);
//!
//! // Apply the rotation to a point
//! let point = [1.0, 0.0, 0.0];
//! let rotated = rotation.transform(&point);
//! ```

/// Special Euclidean group SE(2) for 2D rigid transformations.
pub mod se2;

/// Special Euclidean group SE(3) for 3D rigid transformations.
pub mod se3;

/// Special Orthogonal group SO(2) for 2D rotations.
pub mod so2;

/// Special Orthogonal group SO(3) for 3D rotations.
pub mod so3;
