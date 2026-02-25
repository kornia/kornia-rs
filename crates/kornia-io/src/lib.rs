#![deny(missing_docs)]
#![doc = env!("CARGO_PKG_DESCRIPTION")]

/// Module to handle the error types for the io module.
pub mod error;

/// Module to handle the camera frame rate.
pub mod fps_counter;

/// High-level read and write functions for images.
pub mod functional;

/// TurboJPEG image encoding and decoding.
#[cfg(feature = "turbojpeg")]
pub mod jpegturbo;

/// PNG image encoding and decoding.
pub mod png;

/// JPEG image encoding and decoding.
pub mod jpeg;

/// GStreamer video module for real-time video processing.
#[cfg(feature = "gstreamer")]
pub mod gstreamer;

// NOTE: remove in future release
#[deprecated(since = "0.1.10", note = "Use the gstreamer module instead")]
#[cfg(feature = "gstreamer")]
pub use gstreamer as stream;

/// TIFF image encoding and decoding.
pub mod tiff;

/// Image metadata helpers (EXIF parsing)
pub mod metadata;

/// V4L2 video module for real-time video processing.
#[cfg(all(feature = "v4l", target_os = "linux"))]
pub mod v4l;

/// Internal utility functions for image bit depth conversion.
mod conv_utils;

pub use crate::metadata::{apply_exif_orientation, read_image_jpeg_auto_orient};
