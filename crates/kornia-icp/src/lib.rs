#![deny(missing_docs)]
#![doc = env!("CARGO_PKG_DESCRIPTION")]

mod icp_vanilla;
pub use icp_vanilla::*;

mod ops;

// Re-export the point_cloud_transformation module
mod point_cloud_transformation;
pub use point_cloud_transformation::{compute_centroids, fit_transformation};
