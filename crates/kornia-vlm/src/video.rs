//! Video processing module for Vision Language Models.
//!
//! This module provides video loading and processing capabilities with different sampling strategies.
//! The video functionality requires the `gstreamer` feature to be enabled.
//!
//! # Features
//!
//! To use video functionality, enable the `gstreamer` feature:
//!
//! ```toml
//! [dependencies]
//! kornia-vlm = { version = "0.1", features = ["gstreamer"] }
//! ```
//!
//! # Examples
//!
//! ```no_run
//! # #[cfg(feature = "gstreamer")]
//! # {
//! use kornia_vlm::video::{Video, VideoSamplingMethod};
//! use kornia_tensor::CpuAllocator;
//!
//! let video = Video::from_video_path(
//!     "video.mp4",
//!     VideoSamplingMethod::Uniform(30),
//!     60, // max_frames
//!     CpuAllocator,
//! ).unwrap();
//! # }
//! ```

use std::collections::VecDeque;

use candle_core::DType;
use candle_core::Device;
use candle_core::Shape;
use candle_core::Tensor;
use kornia_image::{allocator::ImageAllocator, Image};
#[cfg(feature = "gstreamer")]
use kornia_io::gstreamer::{video::ImageFormat as IoImageFormat, video::VideoReader};
#[cfg(feature = "gstreamer")]
use log::debug;
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

    /// Error from the image processing library.
    #[error("Image error: {0}")]
    Image(#[from] image::ImageError),

    /// Error from the Candle tensor library.
    #[error("Candle error: {0}")]
    CandleError(#[from] candle_core::Error),

    /// Error from the Kornia image processing library.
    #[error("Kornia image error: {0}")]
    KorniaImage(#[from] kornia_image::ImageError),
}

/// Video sampling strategies for extracting frames from a video.
///
/// Different sampling methods provide various ways to select frames from
/// a video sequence for processing or analysis.
#[derive(Debug)]
pub enum VideoSamplingMethod {
    /// Uniformly sample n frames across the entire video duration.
    ///
    /// This method divides the video into equal segments and takes one frame
    /// from each segment, ensuring even temporal distribution.
    Uniform(usize),

    /// Sample frames at a specific rate (frames per second).
    ///
    /// This method attempts to extract the specified number of frames per
    /// second from the video, useful for maintaining temporal consistency.
    Fps(usize),

    /// Take the first n frames from the video.
    ///
    /// This method simply extracts frames sequentially from the beginning
    /// of the video until the specified count is reached.
    FirstN(usize),

    /// Take frames at specific indices.
    ///
    /// This method allows precise control over which frames are extracted
    /// by specifying their exact positions in the video sequence.
    Indices(Vec<usize>),
}

/// Metadata information for a video.
///
/// Contains timing and structural information about the video,
/// including frame rate, timestamps, and duration.
#[derive(Clone, Debug, Default)]
pub struct VideoMetadata {
    /// Frames per second of the original video, if available.
    pub fps: Option<u32>,

    /// Timestamps in seconds for each frame in the video.
    pub timestamps: VecDeque<u32>,

    /// Total duration of the video in seconds, if available.
    pub duration: Option<u32>,

    /// Processing status for each frame in the video.
    ///
    /// Each boolean value indicates whether the corresponding frame has been
    /// processed by operations like `process_frames()`. This helps avoid
    /// redundant processing and tracks which frames have been modified.
    pub processed: VecDeque<bool>,
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
#[derive(Clone)]
pub struct VideoBuffer<A: ImageAllocator> {
    /// Vector of image frames that make up the video.
    frames: VecDeque<Image<u8, 3, A>>, // TODO: with max frame required, this can be a fixed size array via const generics

    /// Metadata containing timing and video information.
    metadata: VideoMetadata,
}

impl<A: ImageAllocator + Clone> VideoBuffer<A> {
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
    pub fn new(frames: Vec<Image<u8, 3, A>>, timestamps: Vec<u32>) -> Self {
        Self {
            metadata: VideoMetadata {
                fps: None,
                timestamps: VecDeque::from(timestamps),
                duration: None,
                processed: VecDeque::from(vec![false; frames.len()]),
            },
            frames: VecDeque::from(frames),
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
        self.metadata.timestamps.push_back(timestamp);
        self.metadata.processed.push_back(false);
    }

    /// Remove old frames to maintain a maximum frame count.
    ///
    /// This method removes frames from the beginning of the video if the total
    /// number of frames exceeds the specified maximum. Both frames and their
    /// corresponding timestamps and processing status are removed to maintain
    /// consistency across all metadata.
    ///
    /// # Arguments
    ///
    /// * `max_frames` - Maximum number of frames to keep
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use kornia_vlm::video::Video;
    /// # use kornia_tensor::CpuAllocator;
    /// # let mut video = Video::<CpuAllocator>::new(vec![], vec![]);
    /// // Keep only the most recent 10 frames
    /// video.remove_old_frames(10);
    /// ```
    pub fn remove_old_frames(&mut self, max_frames: usize) {
        if self.frames.len() > max_frames {
            let excess = self.frames.len() - max_frames;
            self.frames.drain(0..excess);
            self.metadata.timestamps.drain(0..excess);
            self.metadata.processed.drain(0..excess);
        }
    }

    /// Create a Video from a video file path using kornia-io's VideoReader functionality.
    ///
    /// This method uses kornia-io's VideoReader which provides a higher-level interface
    /// for video reading. Note: This method has some limitations compared to the GStreamer-based
    /// implementation:
    /// - Currently only supports RGB8 format
    /// - Uses GstAllocator for frames, requiring conversion to the target allocator
    /// - May have different performance characteristics
    /// - Works differently as it's designed for streaming rather than batch processing
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the video file
    /// * `sampling` - How to sample frames from the video
    /// * `max_frames` - Maximum number of frames to extract
    /// * `allocator` - Target allocator for the final frames
    #[cfg(feature = "gstreamer")]
    pub fn from_video_path<P: AsRef<std::path::Path>>(
        path: P,
        sampling: VideoSamplingMethod,
        max_frames: usize,
        allocator: A,
    ) -> Result<Self, VideoError> {
        let mut video_reader = VideoReader::new(&path, IoImageFormat::Rgb8).map_err(|e| {
            VideoError::VideoReaderCreation(format!("Path: {:?}, Error: {:?}", path.as_ref(), e))
        })?;

        video_reader
            .start()
            .map_err(|_| VideoError::VideoReaderStart)?;

        let fps = video_reader.get_fps().map(|f| f.round() as u32);
        let duration = video_reader.get_duration().map(|d| d.as_secs() as u32);

        let mut all_frames = Vec::new();
        let mut all_pts = Vec::new();
        let mut frames = Vec::new();
        let mut frame_pts = Vec::new();
        let mut frame_idx = 0;
        let mut indices_set = std::collections::HashSet::new();
        let mut fps_next_time = 0.0;
        let mut fps_interval = 0.0;

        // Precompute for Indices
        if let VideoSamplingMethod::Indices(ref idxs) = sampling {
            indices_set = idxs.iter().cloned().collect();
        }

        // For Fps sampling, calculate interval
        if let VideoSamplingMethod::Fps(user_fps) = sampling {
            if user_fps > 0 {
                fps_interval = 1.0 / (user_fps as f64);
            }
        }

        // Always collect all frames for Uniform, otherwise sample as we go
        let collect_all = matches!(sampling, VideoSamplingMethod::Uniform(_));

        // Give the pipeline time to start up and begin buffering frames
        std::thread::sleep(std::time::Duration::from_millis(100));

        // Read frames from the video with a more patient approach
        let mut consecutive_no_frames = 0;
        let max_consecutive_no_frames = 50; // More patience for streaming
        let frame_wait_ms = 33; // ~30fps equivalent wait time

        loop {
            match video_reader.grab_rgb8() {
                Ok(Some(gst_image)) => {
                    consecutive_no_frames = 0; // Reset counter when we get a frame

                    // Convert GstAllocator image to target allocator
                    let size = gst_image.size();
                    let gst_data = gst_image.as_slice();

                    let img = Image::<u8, 3, A>::from_size_slice(size, gst_data, allocator.clone())
                        .map_err(VideoError::KorniaImage)?;

                    // Get current position for timestamp - using frame index as fallback
                    let current_pos = video_reader
                        .get_pos()
                        .map(|d| d.as_nanos() as u64)
                        .unwrap_or_else(|| {
                            // Fallback: estimate based on frame index and fps
                            if let Some(fps_val) = fps {
                                (frame_idx as u64 * 1_000_000_000) / fps_val as u64
                            } else {
                                frame_idx as u64 * 33_333_333 // Assume 30fps
                            }
                        });

                    if collect_all {
                        all_frames.push(img);
                        all_pts.push(current_pos);
                        debug!(
                            "[kornia-io] Collected frame {} for later uniform sampling (pos: {})",
                            frame_idx, current_pos
                        );
                    } else {
                        let mut take = false;
                        match &sampling {
                            VideoSamplingMethod::Fps(_) => {
                                if fps_interval == 0.0 {
                                    take = true;
                                } else {
                                    let current_time = current_pos as f64 / 1_000_000_000.0;
                                    if current_time >= fps_next_time {
                                        take = true;
                                        fps_next_time += fps_interval;
                                    }
                                }
                            }
                            VideoSamplingMethod::FirstN(n) => {
                                if frames.len() < *n && frames.len() < max_frames {
                                    take = true;
                                }
                            }
                            VideoSamplingMethod::Indices(_) => {
                                if indices_set.contains(&frame_idx) {
                                    take = true;
                                }
                            }
                            _ => {}
                        }
                        if take && frames.len() < max_frames {
                            frames.push(img);
                            frame_pts.push(current_pos);
                            debug!("[kornia-io] ✓ SAMPLED frame {} (pos: {}, timestamp: {}s) - Total sampled: {}", 
                                   frame_idx, current_pos, current_pos as f64 / 1_000_000_000.0, frames.len());
                        } else if take {
                            debug!(
                                "[kornia-io] ✗ Skipped frame {} due to max_frames limit",
                                frame_idx
                            );
                        }

                        // Check if we should break early for different sampling methods
                        match &sampling {
                            VideoSamplingMethod::FirstN(n) => {
                                if frames.len() >= *n.min(&max_frames) {
                                    debug!("[kornia-io] Breaking early for FirstN: collected {} frames", frames.len());
                                    break;
                                }
                            }
                            VideoSamplingMethod::Fps(_) => {
                                if frames.len() >= max_frames {
                                    debug!(
                                        "[kornia-io] Breaking early for Fps: collected {} frames",
                                        frames.len()
                                    );
                                    break;
                                }
                            }
                            VideoSamplingMethod::Indices(indices) => {
                                // Check if we've collected all requested indices within max_frames
                                let remaining_indices: Vec<_> = indices
                                    .iter()
                                    .filter(|&&idx| idx >= frame_idx && frames.len() < max_frames)
                                    .collect();
                                if remaining_indices.is_empty() {
                                    debug!("[kornia-io] Breaking early for Indices: collected all requested frames within limits");
                                    break;
                                }
                            }
                            _ => {}
                        }
                        // General max_frames check for all non-uniform methods
                        if frames.len() >= max_frames {
                            break;
                        }
                    }
                    frame_idx += 1;
                }
                Ok(None) => {
                    consecutive_no_frames += 1;

                    // Wait a bit for more frames
                    std::thread::sleep(std::time::Duration::from_millis(frame_wait_ms));

                    // Check if we've reached the end of the video
                    if let Some(duration) = video_reader.get_duration() {
                        if let Some(current_pos) = video_reader.get_pos() {
                            // Add some tolerance for the end detection
                            if current_pos.as_millis() + 100 >= duration.as_millis() {
                                break;
                            }
                        }
                    }

                    // If we haven't gotten frames for a while, break
                    if consecutive_no_frames > max_consecutive_no_frames {
                        break;
                    }
                }
                Err(_) => {
                    return Err(VideoError::FrameGrabbing);
                }
            }

            // Safety break if we've been running too long
            if frame_idx > 10000 {
                // Reasonable upper limit
                break;
            }
        }

        // If Uniform, select N evenly spaced frames from all_frames
        let timestamps: Vec<u32> = if collect_all {
            let total = all_frames.len();
            let mut ts = Vec::new();
            if let VideoSamplingMethod::Uniform(n) = sampling {
                if n > 0 && total > 0 {
                    let num = n.min(max_frames);
                    debug!(
                        "[kornia-io] Uniform sampling: selecting {} frames from {} total frames",
                        num, total
                    );

                    // Pre-calculate all indices we need
                    let mut indices_to_sample = Vec::new();
                    for i in 0..num {
                        let idx = if num == 1 {
                            total / 2 // Take middle frame if only 1 frame requested
                        } else {
                            ((i as f64) * (total as f64 - 1.0) / (num as f64 - 1.0)).round()
                                as usize
                        };
                        indices_to_sample.push(idx);
                    }

                    debug!(
                        "[kornia-io] Pre-calculated indices to sample: {:?}",
                        indices_to_sample
                    );

                    // Now sample only the frames we need
                    for (i, &idx) in indices_to_sample.iter().enumerate() {
                        frames.push(all_frames[idx].clone());
                        ts.push((all_pts[idx] / 1_000_000_000) as u32);
                        debug!("[kornia-io] ✓ SAMPLED frame at index {} (pos: {}, timestamp: {}s) - Uniform selection {}/{}", 
                               idx, all_pts[idx], all_pts[idx] as f64 / 1_000_000_000.0, i + 1, num);
                    }
                }
                // If more than max_frames were pushed (shouldn't happen), truncate
                if frames.len() > max_frames {
                    frames.truncate(max_frames);
                    ts.truncate(max_frames);
                }
            }
            ts
        } else {
            // Other sampling: use frame_pts for timestamps
            frame_pts
                .iter()
                .map(|&p| (p / 1_000_000_000) as u32)
                .collect()
        };

        video_reader
            .close()
            .map_err(|_| VideoError::VideoReaderClose)?;

        debug!(
            "Video loaded with kornia-io: sampled frames = {}, total frames processed = {}, duration = {:?} seconds, fps = {:?}",
            frames.len(),
            if collect_all { all_frames.len() } else { frame_idx },
            duration,
            fps
        );

        // Log summary of all sampled frames
        debug!("[kornia-io] === SAMPLING SUMMARY ===");
        debug!("[kornia-io] Sampling method: {:?}", sampling);
        debug!("[kornia-io] Total frames sampled: {}", frames.len());
        for (i, &timestamp) in timestamps.iter().enumerate() {
            debug!("[kornia-io]   Sample {}: timestamp {}s", i + 1, timestamp);
        }
        debug!("[kornia-io] ============================");

        Ok(Self {
            metadata: VideoMetadata {
                fps,
                timestamps: VecDeque::from(timestamps),
                duration,
                processed: VecDeque::from(vec![false; frames.len()]),
            },
            frames: VecDeque::from(frames),
        })
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
    /// # use kornia_vlm::video::{Video, VideoSamplingMethod};
    /// # use kornia_tensor::CpuAllocator;
    /// # let mut video = Video::<CpuAllocator>::new(vec![], vec![]);
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
        for (frame, processed) in self.frames.iter_mut().zip(&mut self.metadata.processed) {
            if *processed {
                continue; // Skip already processed frames
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
    pub fn frames(&self) -> &VecDeque<Image<u8, 3, A>> {
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
    /// # use kornia_vlm::video::{Video, VideoSamplingMethod};
    /// # use kornia_tensor::CpuAllocator;
    /// # use candle_core::Device;
    /// # let video = Video::<CpuAllocator>::new(vec![], vec![]);
    /// let device = Device::Cpu;
    /// let tensor = video.into_tensor(&device).unwrap();
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
    /// # use kornia_vlm::video::{Video, VideoSamplingMethod};
    /// # use kornia_tensor::CpuAllocator;
    /// # let video = Video::<CpuAllocator>::new(vec![], vec![]);
    /// let metadata = video.metadata();
    /// if let Some(fps) = metadata.fps {
    ///     println!("Video FPS: {}", fps);
    /// }
    /// println!("Number of frames: {}", metadata.timestamps.len());
    /// ```
    pub fn metadata(&self) -> &VideoMetadata {
        &self.metadata
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "gstreamer")]
    use super::*;
    #[cfg(feature = "gstreamer")]
    use kornia_tensor::CpuAllocator;

    // RUST_LOG=debug cargo test -p kornia-vlm test_smolvlm2_reading_video --features gstreamer -- --nocapture --ignored
    #[test]
    #[cfg(feature = "gstreamer")]
    #[ignore = "Requires GStreamer + test files"]
    fn test_smolvlm2_reading_video() {
        let _ = env_logger::builder().is_test(true).try_init();

        let _video = VideoBuffer::<CpuAllocator>::from_video_path(
            "../../example_video.mp4",
            VideoSamplingMethod::Uniform(1),
            60, // max_frames
            CpuAllocator,
        )
        .unwrap();
        println!(
            "Video loaded successfully with {} frames.",
            _video.frames.len()
        );
        println!("First frame shape: {:?}.", _video.frames[0].0.shape);
    }
}
