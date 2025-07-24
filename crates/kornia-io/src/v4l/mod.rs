/// module for camera controls
pub mod camera_control;

mod pixel_format;
mod stream;

// re-export the camera control and pixel format types
pub use pixel_format::PixelFormat;

use crate::v4l::camera_control::{CameraControlTrait, ControlType};
use kornia_image::ImageSize;
use v4l::{
    buffer::Type, control::Value, video::capture::Parameters, video::Capture, Device, Timestamp,
};

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
    /// The number of buffers to use
    pub buffer_size: u32,
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
            buffer_size: 4,
        }
    }
}

/// V4L video capture.
pub struct V4lVideoCapture {
    stream: stream::MmapStream,
    pixel_format: PixelFormat,
    device: Device,
    size: ImageSize,
}

/// Represents a captured frame
pub struct EncodedFrame {
    /// The buffer of the frame
    pub buffer: stream::V4lBuffer,
    /// The image size of the frame
    pub size: ImageSize,
    /// The fourcc of the frame
    pub pixel_format: PixelFormat,
    /// The timestamp of the frame
    pub timestamp: Timestamp,
    /// The sequence number of the frame
    pub sequence: u32,
}

impl V4lVideoCapture {
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
        let mut stream =
            stream::MmapStream::with_buffers(&device, Type::VideoCapture, config.buffer_size)?;
        stream.next_frame()?; // warm up the stream

        Ok(Self {
            stream,
            pixel_format: PixelFormat::from_fourcc(actual_format.fourcc),
            device,
            size: config.size,
        })
    }

    /// Get the current pixel format
    #[inline]
    pub fn pixel_format(&self) -> PixelFormat {
        self.pixel_format
    }

    /// Set a camera control
    pub fn set_control<T: CameraControlTrait>(&mut self, control: T) -> Result<(), V4L2Error> {
        self.device
            .set_control(v4l::Control {
                id: control.control_id(),
                value: match control.value() {
                    ControlType::Integer(value) => Value::Integer(value),
                    ControlType::Boolean(value) => Value::Boolean(value),
                },
            })
            .map_err(V4L2Error::IoError)
    }

    /// Set the timeout for the stream
    pub fn set_timeout(&mut self, timeout: i32) {
        self.stream.set_timeout(Some(timeout));
    }

    /// Grab a frame from the camera
    pub fn grab_frame(&mut self) -> Result<Option<EncodedFrame>, V4L2Error> {
        let Ok((buffer, metadata)) = self.stream.next_frame() else {
            return Ok(None);
        };

        let frame = EncodedFrame {
            buffer,
            size: self.size,
            pixel_format: self.pixel_format,
            timestamp: metadata.timestamp,
            sequence: metadata.sequence,
        };

        Ok(Some(frame))
    }
}
