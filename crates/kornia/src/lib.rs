#![doc = include_str!(concat!("../", env!("CARGO_PKG_README")))]
//!
//! ## Quick Start
//!
//! ```rust
//! use kornia::image::Image;
//! use kornia::imgproc::resize::resize_fast;
//! use kornia::io::functional as imageio;
//!
//! // Read an image
//! let img = imageio::read_image_any("path/to/image.jpg")?;
//!
//! // Resize it
//! let resized = resize_fast(&img, [224, 224])?;
//!
//! // Write it back
//! imageio::write_image_jpeg("output.jpg", &resized)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Core Modules
//!
//! - [`k3d`] - 3D computer vision primitives and point cloud operations
//! - [`icp`] - Iterative Closest Point algorithms for point cloud registration
//! - [`image`] - Image data structures with custom memory allocators
//! - [`imgproc`] - Image processing operations (filters, transforms, etc.)
//! - [`io`] - Image I/O and video capture utilities
//! - [`linalg`] - Linear algebra operations for computer vision
//! - [`tensor`] - Multi-dimensional tensor data structures
//! - [`tensor_ops`] - Operations on tensors

#[doc(inline)]
pub use kornia_3d as k3d;

#[doc(inline)]
pub use kornia_icp as icp;

#[doc(inline)]
pub use kornia_image as image;

#[doc(inline)]
pub use kornia_imgproc as imgproc;

#[doc(inline)]
pub use kornia_io as io;

#[doc(inline)]
pub use kornia_linalg as linalg;

#[doc(inline)]
pub use kornia_tensor as tensor;

#[doc(inline)]
pub use kornia_tensor_ops as tensor_ops;
