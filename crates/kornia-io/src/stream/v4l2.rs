use std::any::Any;

use crate::stream::{
    camera::{CameraCapture, CameraCaptureConfig},
    error::StreamCaptureError,
};

use kornia_image::ImageSize;

/// A configuration object for capturing frames from a V4L2 camera.
pub struct V4L2CameraConfig {
    /// The camera device path
    pub device: String,
    /// The desired image size
    pub size: Option<ImageSize>,
    /// The desired frames per second
    pub fps: u32,
}

impl CameraCaptureConfig for V4L2CameraConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl V4L2CameraConfig {
    /// Creates a new V4L2CameraConfig object with default values.
    ///
    /// Note: The default device is "/dev/video0", the default image size is None, and the default fps is 30.
    ///
    /// # Returns
    ///
    /// A V4L2CameraConfig object
    pub fn new() -> Self {
        Self {
            device: "/dev/video0".to_string(),
            size: None,
            fps: 30,
        }
    }

    /// Sets the camera device path for the V4L2CameraConfig.
    ///
    /// # Arguments
    ///
    /// * `device` - The camera device path
    pub fn with_device(mut self, device: &str) -> Self {
        self.device = device.to_string();
        self
    }

    /// Sets the camera device path based on the camera id.
    ///
    /// # Arguments
    ///
    /// * `camera_id` - The desired camera id
    pub fn with_camera_id(mut self, camera_id: u32) -> Self {
        self.device = format!("/dev/video{}", camera_id);
        self
    }

    /// Sets the image size for the V4L2CameraConfig.
    ///
    /// # Arguments
    ///
    /// * `size` - The desired image size
    pub fn with_size(mut self, size: ImageSize) -> Self {
        self.size = Some(size);
        self
    }

    /// Sets the frames per second for the V4L2CameraConfig.
    ///
    /// # Arguments
    ///
    /// * `fps` - The desired frames per second
    pub fn with_fps(mut self, fps: u32) -> Self {
        self.fps = fps;
        self
    }

    /// Create a new [`CameraCapture`] object.
    pub fn build(self) -> Result<CameraCapture, StreamCaptureError> {
        CameraCapture::new(&self)
    }
}

impl Default for V4L2CameraConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Returns a GStreamer pipeline string for capturing frames from a V4L2 camera.
///
/// # Arguments
///
/// * `device` - The camera device path
/// * `size` - The image size to capture
/// * `fps` - The desired frames per second
///
/// # Returns
///
/// A GStreamer pipeline string
pub fn v4l2_camera_pipeline_description(device: &str, size: Option<ImageSize>, fps: u32) -> String {
    let video_resize = if let Some(size) = size {
        format!("! video/x-raw,width={},height={} ", size.width, size.height)
    } else {
        "".to_string()
    };

    format!(
            "v4l2src device={} {}! videorate ! video/x-raw,framerate={}/1 ! videoconvert ! video/x-raw,format=RGB ! appsink name=sink",
            device, video_resize, fps
        )
}
