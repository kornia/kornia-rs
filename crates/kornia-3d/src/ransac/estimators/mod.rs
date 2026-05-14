//! Concrete [`super::Estimator`] implementations for kornia-3d's geometric
//! solvers.
//!
//! Each estimator wraps an existing pure-math entry point (`fundamental_8point`,
//! `essential_5pt`, `homography_4pt2d`, `solve_epnp`) so the porting layer
//! adds no new numerical surface — only the trait plumbing that lets the
//! generic RANSAC driver reuse them.

mod epnp;
mod essential;
mod fundamental;
mod homography;

pub use epnp::EPnPEstimator;
pub use essential::EssentialEstimator;
pub use fundamental::FundamentalEstimator;
pub use homography::HomographyEstimator;
