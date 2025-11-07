#![deny(missing_docs)]
#![doc = env!("CARGO_PKG_DESCRIPTION")]

/// Error types for I/O operations.
///
/// Defines [`IoError`] variants for file access, encoding/decoding failures,
/// and format-specific errors.
pub mod error;

/// Frame rate counter for video processing.
///
/// Track and measure frames per second (FPS) during video capture and processing.
pub mod fps_counter;

/// High-level image reading and writing functions.
///
/// Provides convenient functions for reading and writing images in various formats.
/// See [`functional::read_image_any_rgb8`] for automatic format detection.
pub mod functional;

/// TurboJPEG image encoding and decoding (feature-gated).
///
/// Hardware-accelerated JPEG codec using libjpeg-turbo for maximum performance.
/// Requires the `turbojpeg` feature flag.
#[cfg(feature = "turbojpeg")]
pub mod jpegturbo;

/// PNG image encoding and decoding.
///
/// Read and write PNG images with support for various bit depths and color types.
pub mod png;

/// JPEG image encoding and decoding.
///
/// Pure Rust JPEG codec for reading and writing JPEG images.
pub mod jpeg;

/// GStreamer video I/O for real-time video processing (feature-gated).
///
/// Camera capture, RTSP streaming, and video file processing using GStreamer.
/// Requires the `gstreamer` feature flag and system GStreamer libraries.
/// See [`gstreamer::CameraCapture`] for camera access.
#[cfg(feature = "gstreamer")]
pub mod gstreamer;

// NOTE: remove in future release
#[deprecated(since = "0.1.10", note = "Use the gstreamer module instead")]
#[cfg(feature = "gstreamer")]
pub use gstreamer as stream;

/// TIFF image encoding and decoding.
///
/// Read and write TIFF images with support for various compression schemes.
pub mod tiff;

/// Video4Linux2 (V4L2) camera capture (feature-gated, Linux only).
///
/// Direct access to Linux camera devices via the V4L2 API.
/// Requires the `v4l` feature flag and Linux operating system.
#[cfg(all(feature = "v4l", target_os = "linux"))]
pub mod v4l;

/// Internal utility functions for image bit depth conversion.
mod conv_utils;
