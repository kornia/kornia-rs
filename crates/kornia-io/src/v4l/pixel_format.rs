//! Pixel format definitions for V4L2 cameras
use std::str::FromStr;
use v4l::FourCC;

/// Supported camera pixel formats
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    /// YUYV 4:2:2 format (uncompressed, good quality, high bandwidth)
    YUYV,
    /// UYVY 4:2:2 format (uncompressed, good quality, high bandwidth)
    UYVY,
    /// Motion JPEG (compressed, excellent quality, lower bandwidth)
    MJPG,
    /// Custom format specified by FourCC bytes
    Custom([u8; 4]),
}

impl PixelFormat {
    /// Convert to V4L2 FourCC
    pub fn to_fourcc(&self) -> FourCC {
        match self {
            Self::YUYV => FourCC::new(b"YUYV"),
            Self::UYVY => FourCC::new(b"UYVY"),
            Self::MJPG => FourCC::new(b"MJPG"),
            Self::Custom(bytes) => FourCC::new(bytes),
        }
    }

    /// Create PixelFormat from V4L2 FourCC
    pub fn from_fourcc(fourcc: FourCC) -> Self {
        match fourcc.str() {
            Ok("YUYV") => Self::YUYV,
            Ok("UYVY") => Self::UYVY,
            Ok("MJPG") => Self::MJPG,
            _ => {
                let bytes = [
                    fourcc.repr[0],
                    fourcc.repr[1],
                    fourcc.repr[2],
                    fourcc.repr[3],
                ];
                Self::Custom(bytes)
            }
        }
    }

    /// Get the bytes per pixel for uncompressed formats
    pub fn bytes_per_pixel(&self) -> Option<usize> {
        match self {
            Self::YUYV => Some(2),   // 4:2:2 format, 2 bytes per pixel average
            Self::UYVY => Some(2),   // 4:2:2 format, 2 bytes per pixel average
            Self::MJPG => None,      // Variable compression
            Self::Custom(_) => None, // Unknown
        }
    }
}

impl Default for PixelFormat {
    fn default() -> Self {
        Self::YUYV // Most compatible format
    }
}

impl std::fmt::Display for PixelFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::YUYV => write!(f, "YUYV"),
            Self::UYVY => write!(f, "UYVY"),
            Self::MJPG => write!(f, "MJPG"),
            Self::Custom(bytes) => {
                let fourcc_str = std::str::from_utf8(bytes).unwrap_or("????");
                write!(f, "{fourcc_str}")
            }
        }
    }
}

impl FromStr for PixelFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "YUYV" => Ok(Self::YUYV),
            "UYVY" => Ok(Self::UYVY),
            "MJPG" => Ok(Self::MJPG),
            _ => Err(format!("Invalid pixel format: {s}")),
        }
    }
}
