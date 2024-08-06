//! I/O utilities for real-time image and video processing.
#![deny(missing_docs)]

/// Module to handle the error types for the io module.
pub mod error;

/// Module to handle the camera frame rate.
pub mod fps_counter;

/// High-level read and write functions for images.
pub mod functional;

/// TurboJPEG image encoding and decoding.
#[cfg(feature = "jpegturbo")]
pub mod jpeg;

/// GStreamer video module for real-time video processing.
#[cfg(feature = "gstreamer")]
pub mod stream;

pub use crate::error::IoError;
