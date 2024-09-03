use std::any::Any;

use crate::stream::{
    error::StreamCaptureError,
    rtsp::{rtsp_camera_pipeline_description, RTSPCameraConfig},
    v4l2::{v4l2_camera_pipeline_description, V4L2CameraConfig},
    StreamCapture,
};

/// A trait for camera capture configuration.
pub trait CameraCaptureConfig: Any {
    /// Returns the configuration as a trait object.
    fn as_any(&self) -> &dyn Any;
}

/// A camera capture object that grabs frames from a camera.
//pub struct CameraCapture {
//    stream: StreamCapture,
//}
pub struct CameraCapture(pub StreamCapture);

/// A builder for creating a CameraCapture object
impl CameraCapture {
    /// Creates a new CameraCapture object
    ///
    /// # Arguments
    ///
    /// * `config` - The camera capture configuration
    ///
    /// # Returns
    ///
    /// A CameraCapture object
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

impl std::ops::Deref for CameraCapture {
    type Target = StreamCapture;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
