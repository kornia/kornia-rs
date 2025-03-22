#![deny(missing_docs)]
#![doc = env!("CARGO_PKG_DESCRIPTION")]

/// Re-export of glam::DMat3 for use throughout Kornia without direct glam dependency
pub use glam::DMat3;
/// Re-export of glam::DVec3 for use throughout Kornia without direct glam dependency
pub use glam::DVec3;
/// Re-export of glam::Mat3 for use throughout Kornia without direct glam dependency
pub use glam::Mat3;

/// Module to calculate SVD of a 3x3 matrix
pub mod linalg;
