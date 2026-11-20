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
///
/// let video = Video::from_video_path(
///     "video.mp4",
///     VideoSamplingMethod::Uniform(30),
/// ).unwrap();
/// # }
/// ```
#[cfg(feature = "gstreamer")]
use kornia_io::gstreamer::{video::ImageFormat as IoImageFormat, video::VideoReader, Grab};
use kornia_vlm::video::VideoError;
use kornia_vlm::video::VideoSample;

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

/// Estimate missing timestamps relative to the most recent observed PTS.
#[cfg(feature = "gstreamer")]
fn frame_position_ns(
    pts: Option<std::time::Duration>,
    frame_idx: usize,
    last_pts: &mut Option<(usize, u64)>,
    negotiated_fps: impl FnOnce() -> Option<f64>,
) -> u64 {
    if let Some(pts) = pts {
        let position = pts.as_nanos() as u64;
        *last_pts = Some((frame_idx, position));
        return position;
    }

    let (anchor_idx, anchor_position) = last_pts.unwrap_or((0, 0));
    let fps = negotiated_fps().filter(|fps| *fps > 0.0).unwrap_or(30.0);
    let elapsed = ((frame_idx - anchor_idx) as f64 * 1_000_000_000.0 / fps) as u64;
    anchor_position.saturating_add(elapsed)
}

/// Start the schedule at the first frame position, which may be estimated.
/// After a gap, skip scheduled slots that have already elapsed.
#[cfg(feature = "gstreamer")]
fn take_fps_frame(position_ns: u64, next_slot: &mut Option<u128>, fps: usize) -> bool {
    if fps == 0 {
        return true;
    }
    // Scale timestamps so each slot is exactly one billion units, even at fractional periods.
    const SLOT: u128 = 1_000_000_000;
    let scaled_position = u128::from(position_ns) * fps as u128;
    let threshold = next_slot.get_or_insert(scaled_position);
    // Include fractional slots that floor to this nanosecond.
    let position = scaled_position + fps as u128 - 1;
    if position < *threshold {
        return false;
    }
    *threshold += ((position - *threshold) / SLOT + 1) * SLOT;
    true
}

#[cfg(not(feature = "gstreamer"))]
#[allow(dead_code)]
/// Fallback that panics when the `gstreamer` feature is disabled.
pub fn from_video_path<P: AsRef<std::path::Path>>(
    _path: P,
    _sampling: VideoSamplingMethod,
) -> Result<VideoSample<32>, VideoError> {
    panic!("This function requires the 'gstreamer' feature to be enabled.");
}

#[cfg(feature = "gstreamer")]
/// Loads a video file and samples frames according to `sampling`.
///
/// Returned timestamps are whole seconds on the frame PTS timeline, truncated from
/// nanoseconds. Missing PTS values are estimated from the latest observed PTS and
/// frame rate (30 FPS if unknown). Before any PTS is observed, estimates start at zero.
///
/// # Arguments
///
/// * `path` - Path to the input video.
/// * `sampling` - Strategy for choosing frames from the stream.
///
/// # Returns
///
/// A `VideoSample<N>` containing up to `N` selected frames and their timestamps.
///
/// # Errors
///
/// Returns a `VideoError` if creating or starting the reader, grabbing or
/// converting a frame, or closing the reader fails.
///
/// # Example
///
/// ```no_run
/// # use crate::video::{from_video_path, VideoSamplingMethod};
/// let sample = from_video_path::<32, _>("clip.mov", VideoSamplingMethod::Fps(5));
/// # let _ = sample;
/// ```
pub fn from_video_path<const N: usize, P: AsRef<std::path::Path>>(
    path: P,
    sampling: VideoSamplingMethod,
) -> Result<VideoSample<N>, VideoError> {
    let mut video_reader = VideoReader::new(&path, IoImageFormat::Rgb8).map_err(|e| {
        VideoError::VideoReaderCreation(format!("Path: {:?}, Error: {:?}", path.as_ref(), e))
    })?;

    video_reader
        .start()
        .map_err(|_| VideoError::VideoReaderStart)?;

    let fps = video_reader
        .get_fps()
        .map(|f| f.round() as u32)
        .filter(|fps| *fps > 0);
    let duration = video_reader.get_duration().map(|d| d.as_secs() as u32);

    let mut all_frames = Vec::new();
    let mut all_pts = Vec::new();
    let mut frames = Vec::new();
    let mut frame_pts = Vec::new();
    let mut frame_idx = 0;
    let mut last_pts = None;
    let mut indices_set = std::collections::HashSet::new();
    let mut fps_next_time = None;

    // Precompute for Indices
    if let VideoSamplingMethod::Indices(ref idxs) = sampling {
        indices_set = idxs.iter().cloned().collect();
    }

    // Always collect all frames for Uniform, otherwise sample as we go
    let collect_all = matches!(sampling, VideoSamplingMethod::Uniform(_));

    // Read frames until the reader reports end-of-stream or a pipeline error.
    loop {
        match video_reader.try_grab_rgb8() {
            Ok(Grab::Frame {
                image: gst_image,
                pts,
            }) => {
                use kornia_image::Image;

                // Convert kornia_image::allocator::host_alloc() image to target allocator
                let size = gst_image.size();
                let gst_data = gst_image.as_slice();

                let img = Image::<u8, 3>::from_size_slice(size, gst_data)
                    .map_err(VideoError::KorniaImage)?;

                let current_pos =
                    frame_position_ns(pts, frame_idx, &mut last_pts, || video_reader.get_fps());

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
                        VideoSamplingMethod::Fps(fps) => {
                            take = take_fps_frame(current_pos, &mut fps_next_time, *fps);
                        }
                        VideoSamplingMethod::FirstN(n) => {
                            if frames.len() < *n && frames.len() < N {
                                take = true;
                            }
                        }
                        VideoSamplingMethod::Indices(_) => {
                            take = indices_set.contains(&frame_idx);
                        }
                        _ => {}
                    }
                    if take && frames.len() < N {
                        frames.push(img);
                        frame_pts.push(current_pos);
                        log::debug!("[kornia-io] sampled frame {} (pos: {}, timestamp: {}s) - total sampled: {}",
                                   frame_idx, current_pos, current_pos as f64 / 1_000_000_000.0, frames.len());
                    } else if take {
                        log::debug!(
                            "[kornia-io] skipped frame {} due to N (max frame) limit",
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
            // The video ended normally: every decoded frame has been read.
            Ok(Grab::Eos) => break,
            // No frame is buffered yet; the decoder is still working, so back off briefly.
            Ok(Grab::Pending) => std::thread::sleep(std::time::Duration::from_millis(5)),
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
                    log::debug!("[kornia-io] sampled frame at index {} (pos: {}, timestamp: {}s) - uniform selection {}/{}",
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
    for (frame, timestamp) in frames.into_iter().zip(timestamps) {
        video_sample.add_frame(frame, timestamp);
    }
    Ok(video_sample)
}

#[cfg(all(test, feature = "gstreamer"))]
mod tests {
    use super::{frame_position_ns, take_fps_frame};
    use std::time::Duration;

    #[test]
    fn fps_sampling_keeps_missing_pts_on_the_observed_timeline() {
        let mut last_pts = None;
        let mut next_time = None;
        let selected: Vec<_> = (0..=60)
            .filter_map(|index| {
                let pts = match index {
                    0 => Some(Duration::from_secs(10)),
                    30 => Some(Duration::from_millis(11_250)),
                    _ => None,
                };
                let position = frame_position_ns(pts, index, &mut last_pts, || Some(30.0));
                take_fps_frame(position, &mut next_time, 1).then_some((index, position))
            })
            .collect();
        assert_eq!(
            selected,
            [
                (0, 10_000_000_000),
                (30, 11_250_000_000),
                (53, 12_016_666_666)
            ]
        );
    }

    #[test]
    fn fps_sampling_starts_at_first_pts() {
        for origin in [0.0_f64, 10.0] {
            let mut next_time = None;
            let selected: Vec<f64> = [0.0, 0.033, 0.067, 0.1, 1.0, 1.033, 2.0]
                .into_iter()
                .map(|time| time + origin)
                .filter(|time| take_fps_frame((*time * 1e9).round() as u64, &mut next_time, 1))
                .collect();
            assert_eq!(selected, [origin, origin + 1.0, origin + 2.0]);
        }
    }

    #[test]
    fn fps_sampling_skips_missed_slots_after_timestamp_gap() {
        for times in [
            [0.0_f64, 10.4, 10.433, 10.467, 10.999, 11.0, 12.0],
            [10.0, 15.4, 15.433, 15.467, 15.999, 16.0, 17.0],
        ] {
            let mut next_time = None;
            let selected: Vec<_> = times
                .into_iter()
                .filter(|time| take_fps_frame((*time * 1e9).round() as u64, &mut next_time, 1))
                .collect();
            assert_eq!(selected, [times[0], times[1], times[5], times[6]]);
        }
    }

    #[test]
    fn fps_sampling_keeps_floor_quantized_30_fps_frames() {
        let mut next_slot = None;
        for index in 0..90 {
            let pts_ns = 42_000_001 + index * 1_000_000_000 / 30;
            assert!(
                take_fps_frame(pts_ns, &mut next_slot, 30),
                "dropped frame {index} at {pts_ns} ns"
            );
        }
    }

    #[test]
    fn fps_sampling_keeps_exact_slots() {
        let mut next_slot = None;
        let selected: Vec<_> = [
            100_000_000,
            299_999_999,
            300_000_000,
            300_000_001,
            500_000_000,
        ]
        .into_iter()
        .filter(|&ns| take_fps_frame(ns, &mut next_slot, 5))
        .collect();
        assert_eq!(selected, [100_000_000, 300_000_000, 500_000_000]);
    }

    #[test]
    fn zero_fps_interval_keeps_every_frame() {
        let mut next_time = None;
        for time in [10.0, 10.033, 10.067] {
            assert!(take_fps_frame((time * 1e9) as u64, &mut next_time, 0));
        }
    }
}
