#![deny(missing_docs)]
#![doc = env!("CARGO_PKG_DESCRIPTION")]
//!
//! # Kornia Linear Algebra
//!
//! Specialized linear algebra operations for computer vision and robotics applications.
//!
//! ## Key Features
//!
//! - **Rigid Body Transforms**: Compute optimal transformations between point sets
//! - **SVD Decomposition**: Singular Value Decomposition for 3x3 matrices
//! - **Optimized for CV**: Tailored for common computer vision use cases
//!
//! ## Example: Computing Rigid Transformation
//!
//! ```rust
//! use kornia_linalg::rigid::compute_rigid_transform;
//!
//! // Source and target point sets (corresponding points)
//! let src = vec![
//!     [0.0, 0.0, 0.0],
//!     [1.0, 0.0, 0.0],
//!     [0.0, 1.0, 0.0],
//! ];
//!
//! let dst = vec![
//!     [1.0, 1.0, 0.0],
//!     [2.0, 1.0, 0.0],
//!     [1.0, 2.0, 0.0],
//! ];
//!
//! // Compute the rigid transformation (R, t) that maps src to dst
//! let transform = compute_rigid_transform(&src, &dst)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Example: SVD Decomposition
//!
//! ```rust
//! use kornia_linalg::svd::svd_3x3;
//!
//! // Define a 3x3 matrix
//! let matrix = [
//!     [1.0, 2.0, 3.0],
//!     [4.0, 5.0, 6.0],
//!     [7.0, 8.0, 9.0],
//! ];
//!
//! // Compute SVD: A = U * Σ * V^T
//! let (u, s, vt) = svd_3x3(&matrix)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

/// Rigid body transformation computation.
///
/// Algorithms for computing optimal rigid transformations between point sets
/// using SVD-based methods (Kabsch algorithm).
pub mod rigid;

/// Singular Value Decomposition for 3x3 matrices.
///
/// Efficient SVD implementation optimized for small matrices commonly
/// used in computer vision applications.
pub mod svd;
