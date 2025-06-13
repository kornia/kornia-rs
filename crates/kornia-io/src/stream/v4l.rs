use kornia_image::allocator::ImageAllocator;
use kornia_image::Image;
use kornia_image::ImageSize;
use kornia_tensor::TensorAllocator;
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

    /// Failed to create image
    #[error(transparent)]
    ImageError(#[from] kornia_image::ImageError),
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
pub struct V4LVideoCapture<'a> {
    stream: Stream<'a>,
    size: ImageSize,
}

/// An allocator for a V4L2 camera
#[allow(dead_code)]
#[derive(Clone)]
pub struct V4lAllocator<'a>(&'a [u8]);

impl<'a> ImageAllocator for V4lAllocator<'a> {}

impl<'a> TensorAllocator for V4lAllocator<'a> {
    fn alloc(
        &self,
        layout: std::alloc::Layout,
    ) -> Result<*mut u8, kornia_tensor::allocator::TensorAllocatorError> {
        let ptr = unsafe { std::alloc::alloc(layout) };
        Ok(ptr)
    }

    fn dealloc(&self, _ptr: *mut u8, _layout: std::alloc::Layout) {}
}

/// A frame of video from a V4L2 camera.
pub struct ImageFrame<'a> {
    /// The buffer of the frame
    //pub buffer: &'a [u8],
    pub image: Image<u8, 3, V4lAllocator<'a>>,
    /// The fourcc of the frame
    pub fourcc: FourCC,
    /// The timestamp of the frame
    pub timestamp: Timestamp,
    /// The sequence number of the frame
    pub sequence: u32,
}

impl<'a> V4LVideoCapture<'a> {
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

        let allocator = V4lAllocator(buffer);

        let image = unsafe {
            Image::from_raw_parts(self.size, buffer.as_ptr(), buffer.len(), allocator)
                .map_err(|_| V4L2Error::ImageError)
                .ok()?
        };

        let frame = ImageFrame {
            image,
            fourcc: FourCC::new(b"MJPG"),
            timestamp: metadata.timestamp,
            sequence: metadata.sequence,
        };
        Some(Ok(frame))
    }
}
