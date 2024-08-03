/// A module for capturing video streams from different sources.
pub mod camera;

/// Error types for the stream module.
pub mod error;

/// A module for capturing video streams from different sources.
pub mod capture;

pub use crate::stream::camera::{CameraCapture, CameraCaptureBuilder};
pub use crate::stream::capture::StreamCapture;
pub use crate::stream::error::StreamCaptureError;
