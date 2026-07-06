/// module for camera controls
pub mod camera_control;

mod pixel_format;
mod stream;

// re-export the camera control and pixel format types
pub use pixel_format::PixelFormat;
pub use stream::MmapBuffer;

use crate::v4l::camera_control::{CameraControlTrait, ControlType};
use kornia_image::ImageSize;
use v4l::{
    buffer::Type, control::Value, video::capture::Parameters, video::Capture, Device, Timestamp,
};

use std::io;

/// Error types for the v4l2 module.
#[derive(Debug, thiserror::Error)]
pub enum V4L2Error {
    /// Failed to create image
    #[error(transparent)]
    ImageError(#[from] kornia_image::ImageError),

    /// Failed to set parameters or read from the device
    #[error(transparent)]
    IoError(#[from] std::io::Error),

    /// Every capture buffer is still held by the caller, so no buffer can be
    /// handed back to the kernel. Drop older frames or increase `buffer_size`.
    #[error("all V4L2 capture buffers are still in use; drop old frames or increase buffer_size")]
    BuffersExhausted,
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

/// Represents a captured frame from a V4L camera.
///
/// The frame can contain either compressed data (e.g., JPEG) or uncompressed data
/// (e.g., YUYV, UYVY). The buffer uses zero-copy semantics via `MmapBuffer`, which
/// keeps the mmap'd memory alive via `Arc<MmapInfo>`.
pub struct EncodedFrame {
    /// The buffer of the frame (zero-copy via mmap)
    pub buffer: MmapBuffer,
    /// The image size of the frame
    pub size: ImageSize,
    /// The pixel format of the frame
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

        // Read back what the driver actually negotiated. Cameras may fall back to
        // a different pixel format and/or clamp the resolution to one they support.
        let actual_format = device.format()?;
        if actual_format.fourcc != format.fourcc {
            log::warn!(
                "requested format {} not supported, using {}",
                config.format,
                PixelFormat::from_fourcc(actual_format.fourcc)
            );
        }
        let actual_size = ImageSize {
            width: actual_format.width as usize,
            height: actual_format.height as usize,
        };
        if actual_size.width != config.size.width || actual_size.height != config.size.height {
            log::warn!(
                "requested size {}x{} not supported, using {}x{}",
                config.size.width,
                config.size.height,
                actual_size.width,
                actual_size.height
            );
        }

        // Set the frame rate
        let params = Parameters::with_fps(config.fps);
        device.set_params(&params)?;

        // Create the stream and prime it (queue buffers + STREAMON) without
        // discarding a frame.
        let mut stream =
            stream::MmapStream::with_buffers(&device, Type::VideoCapture, config.buffer_size)?;
        stream.start_streaming()?;

        Ok(Self {
            stream,
            pixel_format: PixelFormat::from_fourcc(actual_format.fourcc),
            device,
            size: actual_size,
        })
    }

    /// Get the current pixel format
    #[inline]
    pub fn pixel_format(&self) -> PixelFormat {
        self.pixel_format
    }

    /// Get the frame size the driver actually negotiated.
    ///
    /// This may differ from the size requested in [`V4LCameraConfig`] if the
    /// camera clamped the resolution; allocate decode buffers from this value.
    #[inline]
    pub fn size(&self) -> ImageSize {
        self.size
    }

    /// Set the per-frame dequeue timeout in milliseconds.
    ///
    /// `None` (the default) blocks until a frame is ready. `Some(ms)` makes
    /// [`grab_frame`](Self::grab_frame) return `Ok(None)` if no frame arrives
    /// within `ms` milliseconds.
    #[inline]
    pub fn set_timeout(&mut self, timeout_ms: Option<u32>) {
        self.stream.set_timeout(timeout_ms.map(|ms| ms as i32));
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

    /// Grab a raw, zero-copy frame from the camera.
    ///
    /// The returned [`EncodedFrame`] borrows a kernel mmap buffer; it stays valid
    /// (and the buffer is not recycled) until it and all its clones are dropped.
    ///
    /// Returns `Ok(None)` when a configured timeout elapsed with no frame. Errors
    /// are surfaced rather than swallowed: a device error propagates, and
    /// [`V4L2Error::BuffersExhausted`] means every buffer is still held by the
    /// caller (drop older frames or raise `buffer_size`).
    pub fn grab_frame(&mut self) -> Result<Option<EncodedFrame>, V4L2Error> {
        let (buffer, metadata) = match self.stream.next_frame() {
            Ok(frame) => frame,
            Err(e) if e.kind() == io::ErrorKind::TimedOut => return Ok(None),
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                return Err(V4L2Error::BuffersExhausted)
            }
            Err(e) => return Err(V4L2Error::IoError(e)),
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
