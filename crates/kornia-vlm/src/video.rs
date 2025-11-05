use candle_core::DType;
use candle_core::Device;
use candle_core::Shape;
use candle_core::Tensor;
use circular_buffer::CircularBuffer;
use kornia_image::{allocator::ImageAllocator, Image};
use thiserror::Error;

/// Errors that can occur during video processing operations.
#[derive(Debug, Error)]
pub enum VideoError {
    /// Failed to create a video reader instance.
    #[error("Failed to create video reader: {0}")]
    VideoReaderCreation(String),

    /// Failed to start the video reader.
    #[error("Failed to start video reader")]
    VideoReaderStart,

    /// Failed to grab a frame from the video.
    #[error("Failed to grab frame from video")]
    FrameGrabbing,

    /// Failed to close the video reader.
    #[error("Failed to close video reader")]
    VideoReaderClose,

    /// Error from the Candle tensor library.
    #[error("Candle error: {0}")]
    CandleError(#[from] candle_core::Error),

    /// Error from the Kornia image processing library.
    #[error("Kornia image error: {0}")]
    KorniaImage(#[from] kornia_image::ImageError),
}

/// Metadata information for a video.
///
/// Contains timing and structural information about the video,
/// including frame rate, timestamps, and duration.
#[derive(Clone, Debug, Default)]
pub struct VideoMetadata<const N: usize> {
    /// Frames per second of the original video, if available.
    pub fps: Option<u32>,

    /// Timestamps in seconds for each frame in the video.
    pub timestamps: CircularBuffer<N, u32>,

    /// Total duration of the video in seconds, if available.
    pub duration: Option<u32>,
}

/// A video container that holds frames and metadata.
///
/// This struct represents a video as a collection of image frames along with
/// their temporal metadata. It supports various operations like resizing,
/// normalization, and padding for video processing tasks.
///
/// # Generic Parameters
///
/// * `A` - The image allocator type used for frame storage
#[derive(Clone, Default)]
pub struct VideoSample<const N: usize, A: ImageAllocator> {
    /// Circular buffer of image frames that make up the video.
    frames: CircularBuffer<N, Image<u8, 3, A>>,

    /// Metadata containing timing and video information.
    meta: VideoMetadata<N>,

    /// Processing status for each frame in the video.
    ///
    /// Each boolean value indicates whether the corresponding frame has been
    /// processed by operations like `process_frames()`. This helps avoid
    /// redundant processing and tracks which frames have been modified.
    processed: CircularBuffer<N, bool>,
}

impl<const N: usize, A: ImageAllocator + Clone> VideoSample<N, A> {
    /// Create a new Video instance with frames and timestamps.
    ///
    /// # Arguments
    ///
    /// * `frames` - Vector of image frames
    /// * `timestamps` - Vector of timestamps in seconds for each frame
    ///
    /// # Returns
    ///
    /// A new Video instance with the provided frames and metadata
    pub fn new() -> Self {
        Self {
            meta: VideoMetadata {
                fps: None,
                timestamps: CircularBuffer::new(),
                duration: None,
            },
            frames: CircularBuffer::new(),
            processed: CircularBuffer::new(),
        }
    }

    /// Add a new frame to the video with its timestamp.
    ///
    /// # Arguments
    ///
    /// * `frame` - The image frame to add
    /// * `timestamp` - Timestamp of the frame in seconds
    pub fn add_frame(&mut self, frame: Image<u8, 3, A>, timestamp: u32) {
        self.frames.push_back(frame);
        self.processed.push_back(false);
        self.meta.timestamps.push_back(timestamp);
    }

    /// Process all frames using a closure that modifies each frame in-place.
    ///
    /// This method applies the provided closure to each unprocessed frame as a mutable reference,
    /// allowing for in-place modifications of the frame data. Frames that have already been
    /// processed (marked in metadata) are automatically skipped to avoid redundant operations.
    ///
    /// # Arguments
    ///
    /// * `processor` - A closure that takes a mutable reference to an Image frame and returns
    ///   a Result, allowing for in-place modifications and error handling
    ///
    /// # Returns
    ///
    /// A `Result` indicating success or the first error encountered during processing
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use kornia_vlm::video::VideoSample;
    /// use kornia_tensor::CpuAllocator;
    /// use kornia_image::Image;
    /// let mut video = VideoSample::<32, CpuAllocator>::default();
    /// // Apply some processing to each frame
    /// video.process_frames(|frame| {
    ///     // Example: modify frame data (e.g., apply a filter)
    ///     println!("Processing frame with size: {:?}", frame.size());
    ///     // frame modifications would go here
    ///     Ok(())
    /// }).unwrap();
    /// ```
    pub fn process_frames<F>(&mut self, mut processor: F) -> Result<(), VideoError>
    where
        F: FnMut(&mut Image<u8, 3, A>) -> Result<(), VideoError>,
    {
        for (frame, processed) in self.frames.iter_mut().zip(self.processed.iter_mut()) {
            if *processed {
                continue;
            }
            processor(frame)?;
            *processed = true;
        }
        Ok(())
    }

    /// Get a reference to the frames without processing them.
    ///
    /// # Returns
    ///
    /// A reference to the frames vector
    pub fn frames(&self) -> &CircularBuffer<N, Image<u8, 3, A>> {
        &self.frames
    }

    /// Convert the video frames into a tensor representation.
    ///
    /// This method converts all video frames into a single 4D tensor with the format
    /// `N x 3 x H x W` where:
    /// - `N` is the number of frames
    /// - `3` is the number of color channels (RGB)
    /// - `H` is the height of each frame
    /// - `W` is the width of each frame
    ///
    /// The frames are converted to F32 dtype and the color channels are permuted
    /// from HWC (Height-Width-Channel) to CHW (Channel-Height-Width) format,
    /// which is the standard format expected by most neural network models.
    ///
    /// # Arguments
    ///
    /// * `device` - The device (CPU/CUDA) where the tensor should be allocated
    ///
    /// # Returns
    ///
    /// A `Result` containing either:
    /// - `Ok(Tensor)` - A 4D tensor of shape `[N, 3, H, W]` with F32 dtype
    /// - `Err(VideoError)` - If tensor creation or operations fail
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use kornia_vlm::video::VideoSample;
    /// use kornia_tensor::CpuAllocator;
    /// use kornia_image::Image;
    /// use candle_core::Device;
    /// let video = VideoSample::<32, CpuAllocator>::default();
    /// let device = Device::Cpu;
    /// let tensor = video.into_tensor(candle_core::DType::F32, &device).unwrap();
    /// println!("Tensor shape: {:?}", tensor.dims()); // [N, 3, H, W]
    /// ```
    pub fn into_tensor(&self, dtype: DType, device: &Device) -> Result<Tensor, VideoError> {
        let mut tensors = vec![];
        for i in 0..self.frames.len() {
            let tensor = Tensor::from_vec(
                self.frames[i].to_vec(),
                Shape::from_dims(&[self.frames[i].size().height, self.frames[i].size().width, 3]),
                device,
            )?
            .permute(vec![2, 0, 1])?
            .to_dtype(dtype)?;

            tensors.push(tensor);
        }

        Ok(Tensor::stack(&tensors, 0)?)
    }

    /// Get a reference to the video metadata.
    ///
    /// Returns metadata containing timing and structural information about the video,
    /// including frame timestamps, FPS, duration, and processing status for each frame.
    ///
    /// # Returns
    ///
    /// A reference to the `VideoMetadata` containing:
    /// - `fps`: Original video frame rate (if available)
    /// - `timestamps`: Vector of frame timestamps in seconds
    /// - `duration`: Total video duration in seconds (if available)
    /// - `processed`: Vector indicating which frames have been processed
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use kornia_vlm::video::VideoSample;
    /// use kornia_tensor::CpuAllocator;
    /// let video = VideoSample::<32, CpuAllocator>::default();
    /// let metadata = video.metadata();
    /// if let Some(fps) = metadata.fps {
    ///     println!("Video FPS: {}", fps);
    /// }
    /// println!("Number of frames: {}", metadata.timestamps.len());
    /// ```
    pub fn metadata(&self) -> &VideoMetadata<N> {
        &self.meta
    }
}
