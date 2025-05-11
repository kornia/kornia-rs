use std::any::Any;

use crate::stream::{
    error::StreamCaptureError,
    rtsp::{rtsp_camera_pipeline_description, RTSPCameraConfig},
    v4l2::{v4l2_camera_pipeline_description, V4L2CameraConfig},
    StreamCapture,
};

/// A trait for camera capture configuration.
///
/// This trait allows for different types of camera configurations to be used
/// with the `CameraCapture` struct.
pub trait CameraCaptureConfig: Any {
    /// Returns the configuration as a trait object.
    ///
    /// This method is used for runtime type checking and downcasting.
    fn as_any(&self) -> &dyn Any;
}

/// A camera capture object that grabs frames from a camera.
///
/// This struct wraps a `StreamCapture` and provides a convenient interface
/// for capturing frames from various types of cameras.
pub struct CameraCapture(pub StreamCapture);

impl CameraCapture {
    /// Creates a new CameraCapture object.
    ///
    /// This method constructs a new `CameraCapture` based on the provided configuration.
    /// It supports different types of camera configurations, such as V4L2 and RTSP.
    ///
    /// # Arguments
    ///
    /// * `config` - A trait object implementing `CameraCaptureConfig` that specifies the camera configuration.
    ///
    /// # Returns
    ///
    /// A `Result` containing either a new `CameraCapture` instance or a `StreamCaptureError`.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The configuration is invalid (e.g., empty device or URL)
    /// - The configuration type is unknown
    /// - The `StreamCapture` creation fails
    pub fn new(config: &dyn CameraCaptureConfig) -> Result<Self, StreamCaptureError> {
        let pipeline = if let Some(config) = config.as_any().downcast_ref::<V4L2CameraConfig>() {
            // check that the device is not empty
            if config.device.is_empty() {
                return Err(StreamCaptureError::InvalidConfig(
                    "device is empty".to_string(),
                ));
            }
            v4l2_camera_pipeline_description(&config.device, config.size, config.fps)
        } else if let Some(config) = config.as_any().downcast_ref::<RTSPCameraConfig>() {
            // check that the url is not empty
            if config.url.is_empty() {
                return Err(StreamCaptureError::InvalidConfig(
                    "url is empty".to_string(),
                ));
            }
            rtsp_camera_pipeline_description(&config.url, config.latency)
        } else {
            return Err(StreamCaptureError::InvalidConfig(
                "unknown config type".to_string(),
            ));
        };

        Ok(Self(StreamCapture::new(&pipeline)?))
    }
}

/// Allows `CameraCapture` to be dereferenced to `StreamCapture`.
///
/// This implementation enables direct access to `StreamCapture` methods
/// on a `CameraCapture` instance.
impl std::ops::Deref for CameraCapture {
    type Target = StreamCapture;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Allows `CameraCapture` to be dereferenced to `StreamCapture`.
///
/// This implementation enables direct access to `StreamCapture` methods
/// on a `CameraCapture` instance.
impl std::ops::DerefMut for CameraCapture {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
