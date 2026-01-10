#![deny(missing_docs)]
#![doc = env!("CARGO_PKG_DESCRIPTION")]

/// I/O utilities for reading and writing 3D data.
pub mod io;

/// Linear algebra utilities.
pub mod linalg;

/// Operations on 3D data processing.
pub mod ops;

/// Point cloud traits.
pub mod pointcloud;

/// Pose estimation algorithms.
pub mod pose;

/// Perspective-n-Point (PnP) solvers.
pub mod pnp;

/// Registration algorithms.
pub mod registration;

/// 3D transforms algorithms.
pub mod transforms;
