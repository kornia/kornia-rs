//! Feature detection and description utilities for computer vision.
//!
//! This module provides implementations of various feature detection algorithms
//! commonly used in computer vision tasks such as image matching, tracking,
//! and structure-from-motion.
//!
//! # Available Algorithms
//!
//! * **FAST Corner Detection** - High-speed corner detector using the Bresenham circle test
//! * **Hessian Response** - Blob detection based on the determinant of the Hessian matrix
//!
//! # Example
//!
//! ```
//! use kornia_image::{Image, ImageSize};
//! use kornia_imgproc::features::fast_feature_detector;
//!
//! let image = Image::<u8, 1>::from_size_val(
//!     ImageSize { width: 100, height: 100 },
//!     0,
//! ).unwrap();
//!
//! let corners = fast_feature_detector(&image, 20, 9).unwrap();
//! ```

mod responses;
pub use responses::*;

mod fast;
pub use fast::*;
