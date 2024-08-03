/// A module for capturing video streams from different sources.
pub mod camera;

/// Error types for the stream module.
pub mod error;

pub use crate::stream::camera::{CameraCapture, CameraCaptureBuilder};
pub use crate::stream::error::StreamCaptureError;
