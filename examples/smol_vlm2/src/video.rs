/// Video processing module for Vision Language Models.
///
/// This module provides video loading and processing capabilities with different sampling strategies.
/// The video functionality requires the `gstreamer` feature to be enabled.
///
/// # Features
///
/// To use video functionality, enable the `gstreamer` feature:
///
/// ```toml
/// [dependencies]
/// kornia-vlm = { version = "0.1", features = ["gstreamer"] }
/// ```
///
/// # Examples
///
/// ```no_run
/// # #[cfg(feature = "gstreamer")]
/// # {
/// use kornia_vlm::video::{Video, VideoSamplingMethod};
/// use kornia_tensor::CpuAllocator;
///
/// let video = Video::from_video_path(
///     "video.mp4",
///     VideoSamplingMethod::Uniform(30),
///     CpuAllocator,
/// ).unwrap();
/// # }
/// ```
#[cfg(feature = "gstreamer")]
use kornia_io::gstreamer::{video::ImageFormat as IoImageFormat, video::VideoReader};
use kornia_vlm::video::VideoError;
use kornia_vlm::video::VideoSample;

use kornia_image::allocator::ImageAllocator;

/// Video sampling strategies for extracting frames from a video.
///
/// Different sampling methods provide various ways to select frames from
/// a video sequence for processing or analysis.
#[derive(Debug)]
#[allow(dead_code)]
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

#[cfg(not(feature = "gstreamer"))]
#[allow(dead_code)]
pub fn from_video_path<P: AsRef<std::path::Path>, A: ImageAllocator>(
    _path: P,
    _sampling: VideoSamplingMethod,
    _allocator: A,
) -> Result<VideoSample<32, A>, VideoError> {
    panic!("This function requires the 'gstreamer' feature to be enabled.");
}

#[cfg(feature = "gstreamer")]
pub fn from_video_path<const N: usize, P: AsRef<std::path::Path>, A: ImageAllocator>(
    path: P,
    sampling: VideoSamplingMethod,
    allocator: A,
) -> Result<VideoSample<N, A>, VideoError> {
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
                use kornia_image::Image;

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
                    log::debug!(
                        "[kornia-io] Collected frame {} for later uniform sampling (pos: {})",
                        frame_idx,
                        current_pos
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
                            if frames.len() < *n && frames.len() < N {
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
                    if take && frames.len() < N {
                        frames.push(img);
                        frame_pts.push(current_pos);
                        log::debug!("[kornia-io] ✓ SAMPLED frame {} (pos: {}, timestamp: {}s) - Total sampled: {}",
                                   frame_idx, current_pos, current_pos as f64 / 1_000_000_000.0, frames.len());
                    } else if take {
                        log::debug!(
                            "[kornia-io] ✗ Skipped frame {} due to N (max frame) limit",
                            frame_idx
                        );
                    }

                    // Check if we should break early for different sampling methods
                    match &sampling {
                        VideoSamplingMethod::FirstN(n) => {
                            if frames.len() >= *n.min(&N) {
                                log::debug!(
                                    "[kornia-io] Breaking early for FirstN: collected {} frames",
                                    frames.len()
                                );
                                break;
                            }
                        }
                        VideoSamplingMethod::Fps(_) => {
                            if frames.len() >= N {
                                log::debug!(
                                    "[kornia-io] Breaking early for Fps: collected {} frames",
                                    frames.len()
                                );
                                break;
                            }
                        }
                        VideoSamplingMethod::Indices(indices) => {
                            // Check if we've collected all requested indices within N
                            let remaining_indices: Vec<_> = indices
                                .iter()
                                .filter(|&&idx| idx >= frame_idx && frames.len() < N)
                                .collect();
                            if remaining_indices.is_empty() {
                                log::debug!("[kornia-io] Breaking early for Indices: collected all requested frames within limits");
                                break;
                            }
                        }
                        _ => {}
                    }
                    // General N max frames check for all non-uniform methods
                    if frames.len() >= N {
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
                let num = n.min(N);
                log::debug!(
                    "[kornia-io] Uniform sampling: selecting {} frames from {} total frames",
                    num,
                    total
                );

                // Pre-calculate all indices we need
                let mut indices_to_sample = Vec::new();
                for i in 0..num {
                    let idx = if num == 1 {
                        total / 2 // Take middle frame if only 1 frame requested
                    } else {
                        ((i as f64) * (total as f64 - 1.0) / (num as f64 - 1.0)).round() as usize
                    };
                    indices_to_sample.push(idx);
                }

                log::debug!(
                    "[kornia-io] Pre-calculated indices to sample: {:?}",
                    indices_to_sample
                );

                // Now sample only the frames we need
                for (i, &idx) in indices_to_sample.iter().enumerate() {
                    frames.push(all_frames[idx].clone());
                    ts.push((all_pts[idx] / 1_000_000_000) as u32);
                    log::debug!("[kornia-io] ✓ SAMPLED frame at index {} (pos: {}, timestamp: {}s) - Uniform selection {}/{}",
                               idx, all_pts[idx], all_pts[idx] as f64 / 1_000_000_000.0, i + 1, num);
                }
            }
            // If more than N were pushed (shouldn't happen), truncate
            if frames.len() > N {
                frames.truncate(N);
                ts.truncate(N);
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

    log::debug!(
            "Video loaded with kornia-io: sampled frames = {}, total frames processed = {}, duration = {:?} seconds, fps = {:?}",
            frames.len(),
            if collect_all { all_frames.len() } else { frame_idx },
            duration,
            fps
        );

    // Log summary of all sampled frames
    log::debug!("[kornia-io] === SAMPLING SUMMARY ===");
    log::debug!("[kornia-io] Sampling method: {:?}", sampling);
    log::debug!("[kornia-io] Total frames sampled: {}", frames.len());
    for (i, &timestamp) in timestamps.iter().enumerate() {
        log::debug!("[kornia-io]   Sample {}: timestamp {}s", i + 1, timestamp);
    }
    log::debug!("[kornia-io] ============================");

    let mut video_sample = VideoSample::new();
    for (frame, timestamp) in frames.into_iter().zip(timestamps.into_iter()) {
        video_sample.add_frame(frame, timestamp);
    }
    Ok(video_sample)
}
