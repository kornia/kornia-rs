use crate::stream::{error::StreamCaptureError, StreamCapture};
use kornia_image::ImageSize;

/// Returns a GStreamer pipeline string for capturing frames from a V4L2 camera.
///
/// # Arguments
///
/// * `camera_id` - The camera id
/// * `size` - The image size to capture
/// * `fps` - The desired frames per second
///
/// # Returns
///
/// A GStreamer pipeline string
fn v4l2_camera_pipeline_description(camera_id: usize, size: Option<ImageSize>, fps: u32) -> String {
    let video_resize = if let Some(size) = size {
        format!("! video/x-raw,width={},height={} ", size.width, size.height)
    } else {
        "".to_string()
    };

    format!(
            "v4l2src device=/dev/video{} {}! videorate ! video/x-raw,framerate={}/1 ! videoconvert ! video/x-raw,format=RGB ! appsink name=sink",
            camera_id, video_resize, fps
        )
}

/// A builder for creating a CameraCapture object
pub struct CameraCaptureBuilder {
    camera_id: usize,
    size: Option<ImageSize>,
    fps: u32,
}

impl CameraCaptureBuilder {
    /// Creates a new CameraCaptureBuilder object with default values.
    ///
    /// Note: The default camera id is 0 and the default image size is None
    ///
    /// # Returns
    ///
    /// A CameraCaptureBuilder object
    pub fn new() -> Self {
        Self {
            camera_id: 0,
            size: None,
            fps: 30,
        }
    }

    /// Sets the camera id for the CameraCaptureBuilder.
    ///
    /// # Arguments
    ///
    /// * `camera_id` - The desired camera id
    pub fn camera_id(mut self, camera_id: usize) -> Self {
        self.camera_id = camera_id;
        self
    }

    /// Sets the image size for the CameraCaptureBuilder.
    ///
    /// # Arguments
    ///
    /// * `size` - The desired image size
    pub fn with_size(mut self, size: ImageSize) -> Self {
        self.size = Some(size);
        self
    }

    /// Sets the frames per second for the CameraCaptureBuilder.
    ///
    /// # Arguments
    ///
    /// * `fps` - The desired frames per second
    pub fn with_fps(mut self, fps: u32) -> Self {
        self.fps = fps;
        self
    }

    /// Create a new [`CameraCapture`] object.
    pub fn build(self) -> Result<StreamCapture, StreamCaptureError> {
        // create a pipeline specified by the camera id and size
        StreamCapture::new(&v4l2_camera_pipeline_description(
            self.camera_id,
            self.size,
            self.fps,
        ))
    }
}

impl Default for CameraCaptureBuilder {
    fn default() -> Self {
        Self::new()
    }
}
