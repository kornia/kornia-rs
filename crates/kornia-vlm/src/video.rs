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
//! let video = Video::from_video_path_io(
//!     "video.mp4",
//!     VideoSamplingMethod::Uniform(30),
//!     60, // max_frames
//!     CpuAllocator,
//! ).unwrap();
//! # }
//! ```

use candle_core::Device;
use candle_core::Shape;
use candle_core::Tensor;
use kornia_image::{allocator::ImageAllocator, Image, ImageSize};
use kornia_imgproc::{interpolation::InterpolationMode, resize::resize_fast_rgb};
#[cfg(feature = "gstreamer")]
use kornia_io::gstreamer::{video::ImageFormat as IoImageFormat, video::VideoReader};
#[cfg(feature = "gstreamer")]
use log::debug;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum VideoError {
    #[error("Failed to create video reader: {0}")]
    VideoReaderCreation(String),
    #[error("Failed to start video reader")]
    VideoReaderStart,
    #[error("Failed to grab frame from video")]
    FrameGrabbing,
    #[error("Failed to close video reader")]
    VideoReaderClose,
    #[error("Image error: {0}")]
    Image(#[from] image::ImageError),
    #[error("Candle error: {0}")]
    CandleError(#[from] candle_core::Error),
    #[error("Kornia image error: {0}")]
    KorniaImage(#[from] kornia_image::ImageError),
}

#[derive(Debug)]
pub enum VideoSamplingMethod {
    Uniform(usize),      // Uniformly sample n frames
    Fps(usize),          // Number of frames to sample per second
    FirstN(usize),       // Take the first n frames
    Indices(Vec<usize>), // Take frames at specified indices
}

#[derive(Clone, Debug)]
pub struct VideoMetadata {
    pub fps: Option<u32>,
    pub timestamps: Vec<u32>,  // seconds
    pub duration: Option<u32>, // seconds
}

pub struct Video<A: ImageAllocator> {
    pub frames: Vec<Image<u8, 3, A>>,
    pub frames_tensor: Option<Tensor>,
    pub metadata: VideoMetadata,
}

impl<A: ImageAllocator + Clone> Video<A> {
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
                        .map_err(|e| VideoError::KorniaImage(e))?;

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

        println!(
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
            frames,
            frames_tensor: None,
            metadata: VideoMetadata {
                fps,
                timestamps,
                duration,
            },
        })
    }

    pub fn resize(
        &mut self,
        new_size: ImageSize,
        interpolation: InterpolationMode,
        alloc: A,
    ) -> Result<(), VideoError> {
        for i in 0..self.frames.len() {
            let mut buf = Image::<u8, 3, A>::from_size_val(new_size, 0, alloc.clone())?;
            resize_fast_rgb(&self.frames[i], &mut buf, interpolation)?;
            self.frames[i] = buf;
        }
        Ok(())
    }

    pub fn normalize_and_rescale(
        &mut self,
        mean: [f32; 3],
        std: [f32; 3],
        rescale_factor: f32,
        device: &Device,
    ) -> Result<(), VideoError> {
        let mean_tensor = Tensor::from_slice(&mean, &[3, 1, 1], device)?;
        let std_tensor = Tensor::from_slice(&std, &[3, 1, 1], device)?;
        let rescale_tensor = Tensor::from_slice(&[rescale_factor], &[1], device)?;
        let mut tensors = vec![];
        for i in 0..self.frames.len() {
            let mut tensor = Tensor::from_vec(
                self.frames[i].to_vec(),
                Shape::from_dims(&[self.frames[i].size().height, self.frames[i].size().width, 3]),
                device,
            )?
            .permute(vec![2, 0, 1])?
            .to_dtype(candle_core::DType::F32)?;

            // Normalize: scale to [0,1]
            tensor = tensor.broadcast_mul(&rescale_tensor)?;
            tensor = tensor
                .broadcast_sub(&mean_tensor)?
                .broadcast_div(&std_tensor)?;

            tensors.push(tensor);
        }

        self.frames_tensor = Some(Tensor::stack(&tensors, 0)?);

        Ok(())
    }

    /// Pad video frames spatially and temporally.
    ///
    /// - padded_size: target (height, width)
    /// - max_num_frames: target number of frames
    /// - fill: value to use for padding (e.g., 0)
    /// - alloc: allocator for new frames
    pub fn pad(
        &mut self,
        padded_size: ImageSize,
        max_num_frames: usize,
        fill: u8,
        alloc: A,
    ) -> Result<(), VideoError> {
        // Pad each frame spatially if needed
        for i in 0..self.frames.len() {
            let frame = &self.frames[i];
            let size = frame.size();
            if size.width < padded_size.width || size.height < padded_size.height {
                let mut padded =
                    Image::<u8, 3, A>::from_size_val(padded_size, fill, alloc.clone())?;
                let img_slice = frame.as_slice();
                let padded_img_slice = padded.as_slice_mut();
                let width = size.width;
                let height = size.height;
                let new_width = padded_size.width;
                for y in 0..height.min(padded_size.height) {
                    let src_offset = y * width * 3;
                    let dst_offset = y * new_width * 3;
                    let row_bytes = width.min(new_width) * 3;
                    padded_img_slice[dst_offset..dst_offset + row_bytes]
                        .copy_from_slice(&img_slice[src_offset..src_offset + row_bytes]);
                }
                self.frames[i] = padded;
            }
        }

        // Pad temporally (add blank frames if needed)
        let cur_frames = self.frames.len();
        if cur_frames < max_num_frames {
            let blank = Image::<u8, 3, A>::from_size_val(padded_size, fill, alloc.clone())?;
            for _ in 0..(max_num_frames - cur_frames) {
                self.frames.push(blank.clone());
            }
        }

        Ok(())
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

        let _video = Video::<CpuAllocator>::from_video_path(
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
