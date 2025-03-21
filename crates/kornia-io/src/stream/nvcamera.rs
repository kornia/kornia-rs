
use std::any::Any;
use crate::stream::{
    camera::{CameraCapture, CameraCaptureConfig},
    error::StreamCaptureError,
};
use crate::stream::v4l2::{V4L2CameraConfig,v4l2_camera_pipeline_base_description};
use kornia_image::ImageSize;

/// Defines the output format for NV camera capture.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NVCameraFormat {
    /// RGB format (default)
    RGB,
    /// NV12 format (more lightweight for real-time applications)
    NV12,
}

impl Default for NVCameraFormat {
    fn default() -> Self {
        Self::RGB
    }
}

/// A configuration object for capturing frames using NVIDIA hardware acceleration.
pub struct NVCameraConfig {
    /// The base V4L2 camera configuration
    pub base_config: V4L2CameraConfig,
    /// The output format to use
    pub format: NVCameraFormat,
    /// Whether to enable maximum performance mode for the decoder
    pub enable_max_performance: bool,
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
            base_config: V4L2CameraConfig::new(),
            format: NVCameraFormat::default(),
            enable_max_performance: true,
        }
    }

    /// Sets the camera device path for the NVCameraConfig.
    pub fn with_device(mut self, device: &str) -> Self {
        self.base_config = self.base_config.with_device(device);
        self
    }

    /// Sets the camera device path based on the camera id.
    pub fn with_camera_id(mut self, camera_id: u32) -> Self {
        self.base_config = self.base_config.with_camera_id(camera_id);
        self
    }

    /// Sets the image size for the NVCameraConfig.
    pub fn with_size(mut self, size: ImageSize) -> Self {
        self.base_config = self.base_config.with_size(size);
        self
    }

    /// Sets the frames per second for the NVCameraConfig.
    pub fn with_fps(mut self, fps: u32) -> Self {
        self.base_config = self.base_config.with_fps(fps);
        self
    }

    /// Sets the output format.
    pub fn with_format(mut self, format: NVCameraFormat) -> Self {
        self.format = format;
        self
    }

    /// Sets the maximum performance mode for the decoder.
    pub fn with_max_performance(mut self, enable: bool) -> Self {
        self.enable_max_performance = enable;
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
    format: NVCameraFormat,
    enable_max_performance: bool
) -> String {
    // Create a basic pipeline structure using the V4L2 base
    let base_pipeline = v4l2_camera_pipeline_base_description(device, size, fps);
    
    // Add NVIDIA-specific processing
    let enable_max_performance_str = if enable_max_performance { "true" } else { "false" };
    
    // Format-specific output configuration
    let format_conversion = match format {
        NVCameraFormat::NV12 => {
            // NV12 format - more lightweight for real-time applications
            "nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! appsink name=sink"
        },
        NVCameraFormat::RGB => {
            // RGB format - convert to standard RGB format 
            "nvvidconv ! video/x-raw,format=RGB ! appsink name=sink"
        }
    };

    format!(
        "{} ! nvvideoconvert ! nvv4l2decoder enable-max-performance={} ! nvvideoconvert ! {}",
        base_pipeline, 
        enable_max_performance_str, 
        format_conversion
    )
}