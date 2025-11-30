#![deny(missing_docs)]
#![doc = env!("CARGO_PKG_DESCRIPTION")]
//!
//! # Kornia ICP (Iterative Closest Point)
//!
//! Efficient implementations of Iterative Closest Point algorithms for point cloud registration
//! and alignment. ICP is a fundamental algorithm in robotics and computer vision for aligning
//! 3D point clouds by iteratively minimizing the distance between corresponding points.
//!
//! ## Features
//!
//! - **Vanilla ICP**: Classic point-to-point ICP algorithm
//! - **Fast Convergence**: Optimized nearest neighbor search and transformation estimation
//! - **Configurable Parameters**: Control convergence criteria and maximum iterations
//!
//! ## Example
//!
//! ```rust
//! use kornia_icp::icp_vanilla;
//!
//! // Define source and target point clouds
//! let source = vec![
//!     [1.0, 0.0, 0.0],
//!     [0.0, 1.0, 0.0],
//!     [0.0, 0.0, 1.0],
//! ];
//!
//! let target = vec![
//!     [1.1, 0.1, 0.0],
//!     [0.1, 1.1, 0.0],
//!     [0.0, 0.1, 1.1],
//! ];
//!
//! // Run ICP to find the transformation
//! let result = icp_vanilla(&source, &target, 50, 1e-6)?;
//! println!("Transformation matrix: {:?}", result.transform);
//! println!("Final error: {}", result.error);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Algorithm Overview
//!
//! ICP iteratively performs these steps:
//! 1. Find nearest neighbors between source and target points
//! 2. Estimate optimal rigid transformation
//! 3. Apply transformation to source points
//! 4. Repeat until convergence or max iterations reached

mod icp_vanilla;
pub use icp_vanilla::*;

/// Internal operations for ICP algorithms.
mod ops;
