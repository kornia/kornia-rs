use std::any::Any;

use crate::stream::{
    camera::{CameraCapture, CameraCaptureConfig},
    error::StreamCaptureError,
};

/// A configuration object for capturing frames from a Rtsp camera.
pub struct RTSPCameraConfig {
    /// The url for the Rtsp stream
    pub url: String,
    /// The latency for the Rtsp stream
    pub latency: u32,
}

impl CameraCaptureConfig for RTSPCameraConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl RTSPCameraConfig {
    /// Creates a new RTSPCameraConfig object with default values.
    ///
    /// # Returns
    ///
    /// A RTSPCameraConfig object
    pub fn new() -> Self {
        Self {
            url: String::new(),
            latency: 0,
        }
    }

    /// Sets the url for the RTSPCameraConfig.
    ///
    /// NOTE: usually the url is in the format of `rtsp://username:password@ip:port/stream`
    ///
    /// # Arguments
    ///
    /// * `url` - The url for the Rtsp stream
    pub fn with_url(mut self, url: &str) -> Self {
        self.url = url.to_string();
        self
    }

    /// Sets the latency for the RTSPCameraConfig.
    ///
    /// # Arguments
    ///
    /// * `latency` - The latency for the Rtsp stream
    pub fn with_latency(mut self, latency: u32) -> Self {
        self.latency = latency;
        self
    }

    /// Sets the settings for the RTSPCameraConfig.
    ///
    /// # Arguments
    ///
    /// * `username` - The username for the Rtsp stream
    /// * `password` - The password for the Rtsp stream
    /// * `ip` - The ip address for the Rtsp stream
    /// * `port` - The port for the Rtsp stream
    /// * `stream` - The name of stream
    pub fn with_settings(
        mut self,
        username: &str,
        password: &str,
        ip: &str,
        port: &u16,
        stream: &str,
    ) -> Self {
        self.url = format!(
            "rtsp://{}:{}@{}:{}/{}",
            username, password, ip, port, stream
        );
        self
    }

    /// Create a new [`CameraCapture`] object.
    pub fn build(self) -> Result<CameraCapture, StreamCaptureError> {
        CameraCapture::new(&self)
    }
}

impl Default for RTSPCameraConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Returns a GStreamer pipeline description for capturing frames from a Rtsp camera.
///
/// # Arguments
///
/// * `url` - The url for the Rtsp stream
/// * `latency` - The latency for the Rtsp stream
///
/// # Returns
///
/// A GStreamer pipeline description
pub fn rtsp_camera_pipeline_description(url: &str, latency: u32) -> String {
    format!(
        "rtspsrc location={} latency={} ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! video/x-raw,format=RGB ! appsink name=sink",
        url, latency,
    )
}
