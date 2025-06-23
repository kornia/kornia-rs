mod arena;
mod camera_control;
mod pixel_format;
mod stream;

// re-export the camera control and pixel format types
pub use camera_control::{AutoExposureMode, CameraControl};
pub use pixel_format::PixelFormat;

use kornia_image::ImageSize;
use v4l::buffer::Type;
use v4l::io::traits::CaptureStream;
use v4l::video::capture::Parameters;
use v4l::video::Capture;
use v4l::Device;
use v4l::Timestamp;

/// Error types for the v4l2 module.
#[derive(Debug, thiserror::Error)]
pub enum V4L2Error {
    /// Failed to create image
    #[error(transparent)]
    ImageError(#[from] kornia_image::ImageError),

    /// Failed to set parameters
    #[error(transparent)]
    IoError(#[from] std::io::Error),
}

/// Configuration for V4L video capture.
pub struct V4LCameraConfig {
    /// The camera device path
    pub device_path: String,
    /// The desired image size
    pub size: ImageSize,
    /// The desired frames per second
    pub fps: u32,
    /// The desired pixel format
    pub format: PixelFormat,
}

impl Default for V4LCameraConfig {
    fn default() -> Self {
        Self {
            device_path: "/dev/video0".to_string(),
            size: ImageSize {
                width: 640,
                height: 480,
            },
            fps: 30,
            format: PixelFormat::default(),
        }
    }
}

/// V4L video capture.
pub struct V4LVideoCapture {
    stream: stream::Stream,
    pixel_format: PixelFormat,
    device: Device,
}

/// Represents a captured frame
pub struct EncodedFrame {
    /// The buffer of the frame
    pub buffer: arena::V4lBuffer,
    /// The fourcc of the frame
    pub pixel_format: PixelFormat,
    /// The timestamp of the frame
    pub timestamp: Timestamp,
    /// The sequence number of the frame
    pub sequence: u32,
}

impl V4LVideoCapture {
    /// Create a new V4L video capture.
    pub fn new(config: V4LCameraConfig) -> Result<Self, V4L2Error> {
        let device = Device::with_path(&config.device_path)?;

        // Set the format
        let mut format = device.format()?;
        format.width = config.size.width as u32;
        format.height = config.size.height as u32;
        format.fourcc = config.format.to_fourcc();

        device.set_format(&format)?;

        // Verify the format was actually set (camera might not support it)
        let actual_format = device.format()?;
        if actual_format.fourcc != format.fourcc {
            eprintln!(
                "Warning: Requested format {} not supported, using {}",
                config.format,
                PixelFormat::from_fourcc(actual_format.fourcc)
            );
        }

        // Set the frame rate
        let params = Parameters::with_fps(config.fps);
        device.set_params(&params)?;

        // Create the stream
        let stream = stream::Stream::with_buffers(&device, Type::VideoCapture, 4)?;

        Ok(Self {
            stream,
            pixel_format: PixelFormat::from_fourcc(actual_format.fourcc),
            device,
        })
    }

    /// Get the current pixel format
    #[inline]
    pub fn pixel_format(&self) -> PixelFormat {
        self.pixel_format
    }

    /// Set a camera control
    pub fn set_control(&mut self, control: CameraControl) -> Result<(), V4L2Error> {
        self.device
            .set_control(control.to_v4l_control())
            .map_err(V4L2Error::IoError)
    }

    /// Grab a frame from the camera
    pub fn grab(&mut self) -> Result<Option<EncodedFrame>, V4L2Error> {
        let (buffer, metadata) = self.stream.next()?;

        let frame = EncodedFrame {
            buffer: buffer.clone(),
            pixel_format: self.pixel_format,
            timestamp: metadata.timestamp,
            sequence: metadata.sequence,
        };

        Ok(Some(frame))
    }
}
