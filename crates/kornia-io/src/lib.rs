#![deny(missing_docs)]
#![doc = env!("CARGO_PKG_DESCRIPTION")]
//!
//! # Kornia I/O
//!
//! Image and video I/O utilities for computer vision applications.
//!
//! ## Key Features
//!
//! - **Image Formats**: JPEG, PNG, TIFF encoding/decoding
//! - **Video Capture**: GStreamer, V4L2 for real-time video streaming
//! - **High-Performance**: TurboJPEG support for fast JPEG operations
//! - **Cross-Platform**: Works on Linux, macOS, and Windows
//!
//! ## Example: Reading and Writing Images
//!
//! ```rust,no_run
//! use kornia_io::functional::{read_image_any, write_image_jpeg};
//!
//! // Read any supported image format
//! let img = read_image_any("input.jpg")?;
//!
//! // Write as JPEG with quality setting
//! write_image_jpeg("output.jpg", &img)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Example: Video Capture with GStreamer
//!
//! ```rust,no_run
//! # #[cfg(feature = "gstreamer")]
//! # {
//! use kornia_io::gstreamer::VideoCapture;
//!
//! // Open a video capture device
//! let mut cap = VideoCapture::new("/dev/video0", 640, 480, 30)?;
//!
//! // Read frames
//! while let Some(frame) = cap.read()? {
//!     // Process frame
//!     println!("Got frame: {}x{}", frame.width(), frame.height());
//! }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! # }
//! ```
//!
//! ## Example: V4L2 Camera Control (Linux)
//!
//! ```rust,no_run
//! # #[cfg(all(feature = "v4l", target_os = "linux"))]
//! # {
//! use kornia_io::v4l::VideoStream;
//!
//! // Open a V4L2 device with control over camera settings
//! let mut stream = VideoStream::new("/dev/video0")?;
//! stream.set_resolution(1920, 1080)?;
//! stream.start()?;
//!
//! // Capture frames
//! for _ in 0..100 {
//!     let frame = stream.read()?;
//!     // Process frame
//! }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! # }
//! ```

/// Error types for I/O operations.
///
/// Defines error handling for image and video I/O failures.
pub mod error;

/// Frame rate counter for video processing.
///
/// Utilities to measure and monitor video capture performance.
pub mod fps_counter;

/// High-level image reading and writing functions.
///
/// Provides convenience functions that automatically detect image formats.
pub mod functional;

/// TurboJPEG image encoding and decoding.
///
/// High-performance JPEG operations using libjpeg-turbo.
#[cfg(feature = "turbojpeg")]
pub mod jpegturbo;

/// PNG image encoding and decoding.
///
/// Read and write PNG images with support for various bit depths.
pub mod png;

/// JPEG image encoding and decoding.
///
/// Standard JPEG operations for reading and writing compressed images.
pub mod jpeg;

/// GStreamer video module for real-time video processing.
///
/// Capture video from cameras, files, and network streams using GStreamer.
#[cfg(feature = "gstreamer")]
pub mod gstreamer;

// NOTE: remove in future release
#[deprecated(since = "0.1.10", note = "Use the gstreamer module instead")]
#[cfg(feature = "gstreamer")]
pub use gstreamer as stream;

/// TIFF image encoding and decoding.
///
/// Read and write TIFF images with support for multi-page files.
pub mod tiff;

/// V4L2 video module for real-time video processing on Linux.
///
/// Direct access to Video4Linux2 devices with fine-grained control.
#[cfg(all(feature = "v4l", target_os = "linux"))]
pub mod v4l;

/// Internal utility functions for image bit depth conversion.
mod conv_utils;
