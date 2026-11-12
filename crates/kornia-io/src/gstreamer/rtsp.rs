use std::any::Any;

use crate::stream::{
    camera::{CameraCapture, CameraCaptureConfig},
    error::StreamCaptureError,
    quote_pipeline_value,
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
        let username = percent_encode_userinfo(username);
        let password = percent_encode_userinfo(password);
        self.url = format!("rtsp://{username}:{password}@{ip}:{port}/{stream}");
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
///
/// # Errors
///
/// Returns [`StreamCaptureError::InvalidConfig`] if `url` contains characters that could escape
/// the quoted `location` property (`"`, `\` or control characters).
pub fn rtsp_camera_pipeline_description(
    url: &str,
    latency: u32,
) -> Result<String, StreamCaptureError> {
    let url = quote_pipeline_value(url)?;
    Ok(format!(
        "rtspsrc location={url} latency={latency} ! rtph264depay ! avdec_h264 ! videoconvert ! video/x-raw,format=RGB ! appsink name=sink"
    ))
}

// Percent-encodes the userinfo part of a URL so credentials containing `@`, `:`, `/` or spaces
// cannot change how the URL is parsed.
fn percent_encode_userinfo(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for b in value.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn credentials_are_percent_encoded() {
        let cfg = RTSPCameraConfig::new().with_settings("us er", "p@ss:!/", "10.0.0.1", &554, "s");
        assert_eq!(cfg.url, "rtsp://us%20er:p%40ss%3A%21%2F@10.0.0.1:554/s");
    }

    #[test]
    fn pipeline_url_is_quoted_and_validated() -> Result<(), StreamCaptureError> {
        let desc = rtsp_camera_pipeline_description("rtsp://x ! filesink location=/tmp/x", 0)?;
        assert!(desc.starts_with("rtspsrc location=\"rtsp://x ! filesink location=/tmp/x\" "));
        assert!(rtsp_camera_pipeline_description("rtsp://x\" ! fakesink", 0).is_err());
        Ok(())
    }
}
