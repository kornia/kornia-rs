use super::stream::{CameraCapture, CameraCaptureBuilder};

#[deprecated(
    since = "0.1.5",
    note = "This module is deprecated and will be removed in the next release. \
    Please use the `StreamCaptureError` type from the `stream` module"
)]
pub type StreamCaptureError = super::stream::StreamCaptureError;

#[deprecated(
    since = "0.1.5",
    note = "This module is deprecated and will be removed in the next release. \
    Please use the `CameraCaptureBuilder` struct from the `stream` module"
)]
pub type WebcamCaptureBuilder = CameraCaptureBuilder;

#[deprecated(
    since = "0.1.5",
    note = "This module is deprecated and will be removed in the next release. \
    Please use the `CameraCapture` struct from the `stream` module"
)]
pub type WebcamCapture = CameraCapture;
