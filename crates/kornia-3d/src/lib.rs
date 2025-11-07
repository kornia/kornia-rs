#![deny(missing_docs)]
#![doc = env!("CARGO_PKG_DESCRIPTION")]

/// I/O utilities for reading and writing 3D data.
///
/// Read and write point clouds, meshes, and other 3D data structures in
/// various formats (PLY, OBJ, etc.).
pub mod io;

/// Linear algebra utilities for 3D geometry.
///
/// Matrix operations, decompositions, and numerical algorithms specialized
/// for 3D computer vision tasks.
pub mod linalg;

/// Operations on 3D data processing.
///
/// Common 3D operations including transformations, projections, and
/// geometric computations on point clouds and meshes.
pub mod ops;

/// Point cloud data structures and traits.
///
/// Defines traits and implementations for representing and manipulating
/// 3D point cloud data with efficient storage and access patterns.
pub mod pointcloud;

/// Pose estimation algorithms.
///
/// Algorithms for estimating 3D camera pose and object pose from 2D-3D
/// correspondences or other geometric constraints.
pub mod pose;

/// 3D geometric transformation algorithms.
///
/// Rigid transformations, similarity transforms, and other geometric
/// operations for transforming 3D data including rotation matrices,
/// quaternions, and homogeneous transforms.
pub mod transforms;

/// 3D vector operations and traits.
///
/// Vector arithmetic, normalization, cross products, and other fundamental
/// operations on 3D vectors.
pub mod vector;
