
use std::any::Any;
use crate::stream::{
    camera::{CameraCapture, CameraCaptureConfig},
    error::StreamCaptureError,
};
use kornia_image::ImageSize;

/// A configuration object for capturing frames using NVIDIA hardware acceleration.
pub struct NVCameraConfig {
    /// The camera device path
    pub device: String,
    /// The desired image size
    pub size: Option<ImageSize>,
    /// The desired frames per second
    pub fps: u32,
    /// Whether to use NV12 format (more lightweight) instead of RGB
    pub use_nv12: bool,
}

impl CameraCaptureConfig for NVCameraConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl NVCameraConfig {
    /// Creates a new NVCameraConfig object with default values.
    pub fn new() -> Self {
        Self {
            device: "/dev/video0".to_string(),
            size: None,
            fps: 30,
            use_nv12: false,
        }
    }

    /// Sets the camera device path for the NVCameraConfig.
    pub fn with_device(mut self, device: &str) -> Self {
        self.device = device.to_string();
        self
    }

    /// Sets the camera device path based on the camera id.
    pub fn with_camera_id(mut self, camera_id: u32) -> Self {
        self.device = format!("/dev/video{}", camera_id);
        self
    }

    /// Sets the image size for the NVCameraConfig.
    pub fn with_size(mut self, size: ImageSize) -> Self {
        self.size = Some(size);
        self
    }

    /// Sets the frames per second for the NVCameraConfig.
    pub fn with_fps(mut self, fps: u32) -> Self {
        self.fps = fps;
        self
    }

    /// Sets whether to use NV12 format.
    pub fn with_nv12(mut self, use_nv12: bool) -> Self {
        self.use_nv12 = use_nv12;
        self
    }

    /// Create a new [`CameraCapture`] object.
    pub fn build(self) -> Result<CameraCapture, StreamCaptureError> {
        CameraCapture::new(&self)
    }
}

impl Default for NVCameraConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Returns a GStreamer pipeline string for capturing frames using NVIDIA hardware acceleration.
pub fn nv_camera_pipeline_description(
    device: &str, 
    size: Option<ImageSize>, 
    fps: u32,
    use_nv12: bool
) -> String {
    let video_resize = if let Some(size) = size {
        format!("! video/x-raw,width={},height={} ", size.width, size.height)
    } else {
        "".to_string()
    };

    // If NV12 format is requested, we'll use a different output format
    let format_conversion = if use_nv12 {
        // NV12 format - more lightweight for real-time applications
        "! nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! appsink name=sink"
    } else {
        // RGB format - convert to standard RGB format
        "! nvvidconv ! video/x-raw,format=RGB ! appsink name=sink"
    };

    format!(
        "v4l2src device={} {}! videorate ! video/x-raw,framerate={}/1 ! nvvideoconvert ! nvv4l2decoder enable-max-performance=true ! nvvideoconvert {}",
        device, video_resize, fps, format_conversion
    )
}