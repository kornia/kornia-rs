/// Auto exposure modes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutoExposureMode {
    /// Manual exposure - user controls exposure time
    Manual = 1,
    /// Aperture priority - camera automatically adjusts exposure
    Auto = 3,
}

/// Basic camera controls
#[derive(Debug, Clone, PartialEq)]
pub enum CameraControl {
    // Basic image controls
    /// Image brightness (0-255)
    Brightness(u8),
    /// Image contrast (0-100)
    Contrast(u8),

    // Exposure controls
    /// Auto exposure mode
    AutoExposure(AutoExposureMode),
    /// Exposure time in 100µs units (only when AutoExposure is Manual)
    ExposureTime(u16),
    /// Enable/disable dynamic framerate adjustment (the key fix for 30fps!)
    DynamicFramerate(bool),

    // White balance controls
    /// Auto white balance on/off
    AutoWhiteBalance(bool),
    /// White balance temperature in Kelvin (2800-6500, only when AutoWhiteBalance is false)
    WhiteBalanceTemperature(u16),
}

impl CameraControl {
    /// Get the V4L2 control ID for this control type
    fn control_id(&self) -> u32 {
        match self {
            Self::Brightness(_) => 0x00980900,       // V4L2_CID_BRIGHTNESS
            Self::Contrast(_) => 0x00980901,         // V4L2_CID_CONTRAST
            Self::AutoExposure(_) => 0x009a0901,     // V4L2_CID_EXPOSURE_AUTO
            Self::ExposureTime(_) => 0x009a0902,     // V4L2_CID_EXPOSURE_ABSOLUTE
            Self::DynamicFramerate(_) => 0x009a0903, // V4L2_CID_EXPOSURE_AUTO_PRIORITY
            Self::AutoWhiteBalance(_) => 0x0098090c, // V4L2_CID_AUTO_WHITE_BALANCE
            Self::WhiteBalanceTemperature(_) => 0x0098091a, // V4L2_CID_WHITE_BALANCE_TEMPERATURE
        }
    }

    /// Convert to V4L2 control value
    pub fn to_v4l_control(&self) -> v4l::control::Control {
        use v4l::control::{Control, Value};

        let id = self.control_id();
        let value = match self {
            Self::Brightness(v) => Value::Integer(*v as i64),
            Self::Contrast(v) => Value::Integer(*v as i64),
            Self::AutoExposure(mode) => Value::Integer(*mode as i64),
            Self::ExposureTime(v) => Value::Integer(*v as i64),
            Self::DynamicFramerate(v) => Value::Boolean(*v),
            Self::AutoWhiteBalance(v) => Value::Boolean(*v),
            Self::WhiteBalanceTemperature(v) => Value::Integer(*v as i64),
        };

        Control { id, value }
    }
}
