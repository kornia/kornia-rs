//! Feature detection and keypoint extraction.
//!
//! This module provides algorithms for detecting distinctive points (keypoints)
//! in images, which are useful for:
//!
//! - Image matching and registration
//! - Object tracking
//! - 3D reconstruction
//! - Visual SLAM
//!
//! # Available Detectors
//!
//! - **FAST**: Features from Accelerated Segment Test - fast corner detection
//!
//! # Examples
//!
//! Detecting FAST corners in an image for feature matching.

mod responses;
pub use responses::*;

mod fast;
pub use fast::*;
