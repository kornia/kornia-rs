//! # Kornia Lie Groups
//!
//! Lie group and Lie algebra representations for robotics and 3D geometry.
//!
//! This crate provides efficient implementations of common Lie groups used in
//! robotics, computer vision, and SLAM:
//!
//! - **SO(2)**: 2D rotations
//! - **SE(2)**: 2D rigid transformations (rotation + translation)
//! - **SO(3)**: 3D rotations  
//! - **SE(3)**: 3D rigid transformations (rotation + translation)
//!
//! # Mathematical Background
//!
//! Lie groups provide a principled way to represent and compose rotations and
//! rigid transformations. Each group has an associated Lie algebra that represents
//! tangent space elements and enables optimization on manifolds.
//!
//! # Examples
//!
//! ```no_run
//! // Working with 3D rotations
//! // use kornia_lie::so3::SO3;
//! ```

/// Special Orthogonal group SO(2) for 2D rotations.
///
/// Represents rotations in 2D space as unit complex numbers or 2×2 rotation matrices.
pub mod se2;

/// Special Euclidean group SE(2) for 2D rigid transformations.
///
/// Represents combined rotation and translation in 2D space.
pub mod se3;

/// Special Orthogonal group SO(2) for 2D rotations.
///
/// Represents rotations in 2D as angles or unit complex numbers.
pub mod so2;

/// Special Orthogonal group SO(3) for 3D rotations.
///
/// Represents rotations in 3D space using quaternions, rotation matrices,
/// or axis-angle representations.
pub mod so3;
