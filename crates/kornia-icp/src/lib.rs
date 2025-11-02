#![deny(missing_docs)]
#![doc = env!("CARGO_PKG_DESCRIPTION")]
//!
//! # Iterative Closest Point (ICP) Algorithm
//!
//! Point cloud registration using the Iterative Closest Point algorithm.
//!
//! ICP is a widely-used algorithm for aligning two point clouds by iteratively
//! finding correspondences and estimating the optimal rigid transformation.
//!
//! # Algorithm
//!
//! 1. Find nearest neighbors between source and target point clouds
//! 2. Estimate rigid transformation (rotation + translation)
//! 3. Apply transformation to source points
//! 4. Repeat until convergence
//!
//! # Examples
//!
//! ```no_run
//! // Align two point clouds
//! // let transform = icp_align(&source_cloud, &target_cloud, max_iterations);
//! ```

mod icp_vanilla;
pub use icp_vanilla::*;

/// Internal operations for ICP computation.
mod ops;
