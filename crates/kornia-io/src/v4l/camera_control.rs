/// Auto exposure modes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutoExposureMode {
    /// Auto exposure mode
    Auto = 0,
    /// Manual exposure - user controls exposure time
    Manual = 1,
}

/// Basic camera controls
#[derive(Debug, Clone, PartialEq)]
pub enum CameraControl {
    // User Controls
    /// Image brightness (-1024 to 1023)
    /// BREAKING CHANGE: The type of `Brightness` has been changed from `u8` (0-255) to `i32` (-1024 to 1023).
    /// See the changelog for more details.
    Brightness(i32),
    /// Image contrast (0-255)
    Contrast(u8),
    /// Color saturation (0-255)
    Saturation(u8),
    /// Hue adjustment (-90 to 90)
    Hue(i32),
    /// Image sharpness (0-255)
    Sharpness(u8),

    // Camera Controls
    /// Auto exposure mode
    AutoExposure(AutoExposureMode),
    /// Exposure time in microseconds (500-133000, only when AutoExposure is Manual).
    /// Replaces the legacy `ExposureTime` control, which used a `u16` type.
    ExposureTimeAbsolute(u32),
    /// Frame synchronization
    FrameSync(bool),
    /// PWM configuration (0=30Hz, 1=60Hz)
    PwmConfig(u8),
    /// Bypass mode
    BypassMode(u8),
    /// Override enable
    OverrideEnable(u8),
    /// Height alignment (1-16)
    HeightAlign(u8),
    /// Size alignment (0-2)
    SizeAlign(u8),
    /// Write ISP format (fixed at 1)
    WriteIspFormat(u8),
    /// Low latency mode
    LowLatencyMode(bool),
    /// Preferred stride (0-65535)
    PreferredStride(u16),
    /// Override capture timeout in milliseconds (-1 to 2147483647)
    OverrideCaptureTimeoutMs(i32),

    // Legacy controls (kept for backwards compatibility)
    /// Enable/disable dynamic framerate adjustment
    DynamicFramerate(bool),
    /// Auto white balance on/off
    AutoWhiteBalance(bool),
    /// White balance temperature in Kelvin (2800-6500)
    WhiteBalanceTemperature(u16),
}

impl CameraControl {
    /// Get the V4L2 control ID for this control type
    fn control_id(&self) -> u32 {
        match self {
            // User Controls
            Self::Brightness(_) => 0x00980900,
            Self::Contrast(_) => 0x00980901,
            Self::Saturation(_) => 0x00980902,
            Self::Hue(_) => 0x00980903,
            Self::Sharpness(_) => 0x0098091b,

            // Camera Controls
            Self::AutoExposure(_) => 0x009a0901,
            Self::ExposureTimeAbsolute(_) => 0x009a0902,
            Self::FrameSync(_) => 0x009a092b,
            Self::PwmConfig(_) => 0x009a0935,
            Self::BypassMode(_) => 0x009a2064,
            Self::OverrideEnable(_) => 0x009a2065,
            Self::HeightAlign(_) => 0x009a2066,
            Self::SizeAlign(_) => 0x009a2067,
            Self::WriteIspFormat(_) => 0x009a2068,
            Self::LowLatencyMode(_) => 0x009a206d,
            Self::PreferredStride(_) => 0x009a206e,
            Self::OverrideCaptureTimeoutMs(_) => 0x009a206f,

            // Legacy controls
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
            // User Controls
            Self::Brightness(v) => Value::Integer(*v as i64),
            Self::Contrast(v) => Value::Integer(*v as i64),
            Self::Saturation(v) => Value::Integer(*v as i64),
            Self::Hue(v) => Value::Integer(*v as i64),
            Self::Sharpness(v) => Value::Integer(*v as i64),

            // Camera Controls
            Self::AutoExposure(mode) => Value::Integer(*mode as i64),
            Self::ExposureTimeAbsolute(v) => Value::Integer(*v as i64),
            Self::FrameSync(v) => Value::Boolean(*v),
            Self::PwmConfig(v) => Value::Integer(*v as i64),
            Self::BypassMode(v) => Value::Integer(*v as i64),
            Self::OverrideEnable(v) => Value::Integer(*v as i64),
            Self::HeightAlign(v) => Value::Integer(*v as i64),
            Self::SizeAlign(v) => Value::Integer(*v as i64),
            Self::WriteIspFormat(v) => Value::Integer(*v as i64),
            Self::LowLatencyMode(v) => Value::Boolean(*v),
            Self::PreferredStride(v) => Value::Integer(*v as i64),
            Self::OverrideCaptureTimeoutMs(v) => Value::Integer(*v as i64),

            // Legacy controls
            Self::DynamicFramerate(v) => Value::Boolean(*v),
            Self::AutoWhiteBalance(v) => Value::Boolean(*v),
            Self::WhiteBalanceTemperature(v) => Value::Integer(*v as i64),
        };

        Control { id, value }
    }

    /// Validate control value is within acceptable range
    pub fn validate(&self) -> Result<(), String> {
        match self {
            Self::Brightness(v) => {
                if *v < -1024 || *v > 1023 {
                    return Err(format!("Brightness {v} out of range (-1024 to 1023)"));
                }
            }
            Self::Hue(v) => {
                if *v < -90 || *v > 90 {
                    return Err(format!("Hue {v} out of range (-90 to 90)"));
                }
            }
            Self::ExposureTimeAbsolute(v) => {
                if *v < 500 || *v > 133000 {
                    return Err(format!("Exposure time {v} out of range (500 to 133000)"));
                }
            }
            Self::HeightAlign(v) => {
                if *v < 1 || *v > 16 {
                    return Err(format!("Height align {v} out of range (1 to 16)"));
                }
            }
            Self::SizeAlign(v) => {
                if *v > 2 {
                    return Err(format!("Size align {v} out of range (0 to 2)"));
                }
            }
            Self::PwmConfig(v) => {
                if *v > 1 {
                    return Err(format!("PWM config {v} out of range (0 to 1)"));
                }
            }
            Self::OverrideCaptureTimeoutMs(v) => {
                if *v < -1 {
                    return Err(format!(
                        "Override capture timeout {v} out of range (-1 to 2147483647)"
                    ));
                }
            }
            Self::WhiteBalanceTemperature(v) => {
                if *v < 2800 || *v > 6500 {
                    return Err(format!(
                        "White balance temperature {v} out of range (2800 to 6500)"
                    ));
                }
            }
            // Boolean controls are always valid
            _ => {}
        }
        Ok(())
    }
}
