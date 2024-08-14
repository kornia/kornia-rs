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
pub struct CameraCapture {
    stream: StreamCapture,
}

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
                return Err(StreamCaptureError::InvalidConfig);
            }
            v4l2_camera_pipeline_description(&config.device, config.size, config.fps)
        } else if let Some(config) = config.as_any().downcast_ref::<RTSPCameraConfig>() {
            // check that the url is not empty
            if config.url.is_empty() {
                return Err(StreamCaptureError::InvalidConfig);
            }
            rtsp_camera_pipeline_description(&config.url, config.latency)
        } else {
            return Err(StreamCaptureError::InvalidConfig);
        };

        Ok(Self {
            stream: StreamCapture::new(&pipeline)?,
        })
    }

    /// Starts grabbing frames from the camera.
    ///
    /// # Arguments
    ///
    /// * `f` - A function that processes the image frames
    ///
    /// # Errors
    ///
    /// Returns an error if the image processing function fails.
    pub async fn run<F>(&mut self, f: F) -> Result<(), StreamCaptureError>
    where
        F: Fn(kornia_image::Image<u8, 3>) -> Result<(), Box<dyn std::error::Error>>,
    {
        self.stream
            .run(|img| {
                f(img)?;
                Ok(())
            })
            .await
    }

    /// Stops the camera capture object.
    ///
    /// This function should be called when the camera capture object is no longer needed.
    ///
    /// # Returns
    ///
    /// A Result object with a success message if the camera capture object is stopped successfully.
    pub fn close(&mut self) -> Result<(), StreamCaptureError> {
        self.stream.close()
    }
}
