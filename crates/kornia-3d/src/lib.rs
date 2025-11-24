#![deny(missing_docs)]
#![doc = env!("CARGO_PKG_DESCRIPTION")]
//!
//! # Kornia 3D
//!
//! A comprehensive library for 3D computer vision tasks including point cloud processing,
//! pose estimation, and geometric transformations.
//!
//! ## Key Features
//!
//! - **Point Cloud I/O**: Read/write PCD, PLY, and COLMAP formats
//! - **Geometric Transformations**: Affine, homography, and rigid body transforms
//! - **Pose Estimation**: Camera pose and motion estimation algorithms
//! - **Linear Algebra**: Specialized operations for 3D geometry
//!
//! ## Example: Loading a Point Cloud
//!
//! ```rust,no_run
//! use kornia_3d::io::pcd::read_pcd;
//!
//! // Load a point cloud from a PCD file
//! let point_cloud = read_pcd("path/to/cloud.pcd")?;
//! println!("Loaded {} points", point_cloud.len());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Example: Applying Transformations
//!
//! ```rust
//! use kornia_3d::transforms::transform_points;
//!
//! // Define a transformation matrix (4x4 homogeneous)
//! let transform = [
//!     [1.0, 0.0, 0.0, 1.0],
//!     [0.0, 1.0, 0.0, 2.0],
//!     [0.0, 0.0, 1.0, 3.0],
//!     [0.0, 0.0, 0.0, 1.0],
//! ];
//!
//! // Transform a point
//! let point = [1.0, 2.0, 3.0];
//! let transformed = transform_points(&[point], &transform);
//! ```

/// I/O utilities for reading and writing 3D data formats.
///
/// Supports PCD (Point Cloud Data), PLY (Polygon File Format), and COLMAP camera models.
pub mod io;

/// Linear algebra utilities specialized for 3D geometry.
///
/// Provides matrix operations, decompositions, and geometric calculations.
pub mod linalg;

/// Operations on 3D data structures.
///
/// Includes nearest neighbor search, filtering, and point cloud manipulation.
pub mod ops;

/// Point cloud traits and abstractions.
///
/// Defines common interfaces for working with point cloud data structures.
pub mod pointcloud;

/// Pose estimation algorithms for camera and object localization.
///
/// Includes methods for computing affine transformations and homographies.
pub mod pose;

/// 3D geometric transformation algorithms.
///
/// Provides functions for applying rotations, translations, and projections.
pub mod transforms;

/// 3D vector traits and operations.
///
/// Defines vector arithmetic and common vector operations in 3D space.
pub mod vector;
