use crate::stream::{error::StreamCaptureError, StreamCapture};

/// Returns a GStreamer pipeline description for capturing frames from a Rtsp camera.
///
/// # Arguments
///
/// * `username` - The username for the Rtsp stream
/// * `password` - The password for the Rtsp stream
/// * `ip` - The ip address for the Rtsp stream
/// * `port` - The port for the Rtsp stream
/// * `stream` - The name of stream
///
/// # Returns
///
/// A GStreamer pipeline description
fn rtsp_camera_pipeline_description(
    username: &str,
    password: &str,
    ip: &str,
    port: &u32,
    stream: &str,
) -> String {
    format!(
        "rtspsrc location=rtsp://{}:{}@{}:{}/{} ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! video/x-raw,format=RGB ! appsink name=sink",
        username, password, ip, port, stream
    )
}

/// A builder for creating a RtspCameraCapture object
pub struct RtspCameraCaptureBuilder {
    username: String,
    password: String,
    ip: String,
    port: u32,
    stream: String,
}

impl RtspCameraCaptureBuilder {
    /// Creates a new CameraCaptureBuilder object with default values.
    pub fn new() -> Self {
        Self {
            username: "".to_string(),
            password: "".to_string(),
            ip: "".to_string(),
            port: 0,
            stream: "".to_string(),
        }
    }

    /// Sets the username for the Rtsp stream.
    ///
    /// # Arguments
    ///
    /// * `username` - The username for the Rtsp stream
    pub fn with_username(mut self, username: &str) -> Self {
        self.username = username.to_string();
        self
    }

    /// Sets the password for the Rtsp stream.
    ///
    /// # Arguments
    ///
    /// * `password` - The password for the Rtsp stream
    pub fn with_password(mut self, password: &str) -> Self {
        self.password = password.to_string();
        self
    }

    /// Sets the ip address for the Rtsp stream.
    ///
    /// # Arguments
    ///
    /// * `ip` - The ip address for the Rtsp stream
    pub fn with_ip(mut self, ip: &str) -> Self {
        self.ip = ip.to_string();
        self
    }

    /// Sets the port for the Rtsp stream.
    ///
    /// # Arguments
    ///
    /// * `port` - The port for the Rtsp stream
    pub fn with_port(mut self, port: u32) -> Self {
        self.port = port;
        self
    }

    /// Sets the stream for the Rtsp stream.
    ///
    /// # Arguments
    ///
    /// * `stream` - The name of stream
    pub fn with_stream(mut self, stream: &str) -> Self {
        self.stream = stream.to_string();
        self
    }

    /// Create a new [`StreamCapture`] object.
    pub fn build(self) -> Result<StreamCapture, StreamCaptureError> {
        // create a pipeline specified by the camera id and size
        StreamCapture::new(&rtsp_camera_pipeline_description(
            &self.username,
            &self.password,
            &self.ip,
            &self.port,
            &self.stream,
        ))
    }
}

impl Default for RtspCameraCaptureBuilder {
    fn default() -> Self {
        Self::new()
    }
}
