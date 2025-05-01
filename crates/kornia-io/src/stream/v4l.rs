use kornia_image::ImageSize;
use v4l::buffer::Type;
use v4l::io::mmap::Stream;
use v4l::io::traits::CaptureStream;
use v4l::video::capture::Parameters;
use v4l::video::Capture;
use v4l::Device;
use v4l::FourCC;
use v4l::Timestamp;

/// Error types for the v4l2 module.
#[derive(Debug, thiserror::Error)]
pub enum V4L2Error {
    /// Failed to open device
    #[error("Failed to open device")]
    OpenDeviceError,
}

/// A configuration object for capturing frames from a V4L2 camera.
pub struct V4LCameraConfig {
    /// The camera device path
    pub device_path: String,
    /// The desired image size
    pub size: ImageSize,
    /// The desired frames per second
    pub fps: u32,
}

/// A video capture object for a V4L2 camera.
pub struct V4LVideoCapture {
    stream: Stream<'static>,
    size: ImageSize,
}

/// A frame of video from a V4L2 camera.
pub struct ImageFrame<'a> {
    /// The buffer of the frame
    pub buffer: &'a [u8],
    /// The size of the frame
    pub size: ImageSize,
    /// The fourcc of the frame
    pub fourcc: FourCC,
    /// The timestamp of the frame
    pub timestamp: Timestamp,
    /// The sequence number of the frame
    pub sequence: u32,
}

impl V4LVideoCapture {
    /// Create a new V4L2 video capture object.
    pub fn new(config: V4LCameraConfig) -> Result<Self, V4L2Error> {
        let mut dev = Device::with_path(config.device_path).unwrap();
        let mut fmt = dev.format().unwrap();
        fmt.width = config.size.width as u32;
        fmt.height = config.size.height as u32;
        fmt.fourcc = FourCC::new(b"MJPG");
        dev.set_format(&fmt).unwrap();
        let params = Parameters::with_fps(config.fps);
        dev.set_params(&params).unwrap();

        let stream = Stream::with_buffers(&mut dev, Type::VideoCapture, 4).unwrap();

        Ok(Self {
            stream,
            size: ImageSize {
                width: config.size.width,
                height: config.size.height,
            },
        })
    }

    /// Grab a frame from the V4L2 camera.
    pub fn grab(&mut self) -> Option<Result<ImageFrame, V4L2Error>> {
        let Ok((buffer, metadata)) = self.stream.next() else {
            return None;
        };
        let frame = ImageFrame {
            buffer,
            size: self.size,
            fourcc: FourCC::new(b"MJPG"),
            timestamp: metadata.timestamp,
            sequence: metadata.sequence, ///////////
        };
        Some(Ok(frame))
    }
}
