#![deny(missing_docs)]
#![doc = env!("CARGO_PKG_DESCRIPTION")]

/// Bundle adjustment solver.
pub mod ba;

/// Pinhole camera model with Brown-Conrady distortion and Kannala-Brandt fisheye model.
pub mod camera;

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

/// Generic RANSAC traits and config shared by all robust estimators.
pub mod ransac;

/// Registration algorithms.
pub mod registration;

/// 3D transforms algorithms.
pub mod transforms;
