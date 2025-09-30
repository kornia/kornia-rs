use candle_core::Device;
use candle_core::Shape;
use candle_core::Tensor;
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use kornia_image::{allocator::ImageAllocator, Image, ImageSize};
use kornia_imgproc::{interpolation::InterpolationMode, resize::resize_fast_rgb};
use log::debug;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum VideoError {
    #[error("GStreamer error: {0}")]
    GStreamer(#[from] gst::glib::Error),
    #[error("GStreamer state change error: {0}")]
    StateChange(#[from] gst::StateChangeError),
    #[error("GStreamer BoolError: {0}")]
    BoolError(#[from] gst::glib::BoolError),
    #[error("GStreamer option error")]
    OptionError,
    #[error("Image error: {0}")]
    Image(#[from] image::ImageError),
    #[error("Candle error: {0}")]
    CandleError(#[from] candle_core::Error),
    #[error("Kornia image error: {0}")]
    KorniaImage(#[from] kornia_image::ImageError),
}

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
    pub fn from_video_path<P: AsRef<std::path::Path>>(
        path: P,
        sampling: VideoSamplingMethod,
        max_frames: usize,
        allocator: A,
    ) -> Result<Self, VideoError> {
        gst::init()?;

        // Build pipeline string with the provided path
        let pipeline_str = format!(
            "filesrc location={} ! decodebin ! videoconvert ! video/x-raw,format=RGB ! appsink name=sink",
            path.as_ref().to_string_lossy()
        );
        let pipeline = gst::parse::launch(&pipeline_str)?;
        let pipeline = pipeline
            .downcast::<gst::Pipeline>()
            .map_err(|_| VideoError::OptionError)
            .unwrap();
        let appsink = pipeline
            .by_name("sink")
            .ok_or(VideoError::OptionError)
            .unwrap()
            .downcast::<gst_app::AppSink>()
            .map_err(|_| VideoError::OptionError)
            .unwrap();
        pipeline.set_state(gst::State::Playing)?;

        let mut all_frames = Vec::new();
        let mut all_pts = Vec::new();
        let mut frames = Vec::new();
        let mut frame_pts = Vec::new();
        let mut frame_idx = 0;
        let mut indices_set = std::collections::HashSet::new();
        let mut fps_next_pts = 0;
        let mut fps_interval = 0;
        let mut first_pts = None;

        // Precompute for Indices
        if let VideoSamplingMethod::Indices(ref idxs) = sampling {
            indices_set = idxs.iter().cloned().collect();
        }

        // For Fps, we need to know framerate. We'll try to get it from caps.
        if let VideoSamplingMethod::Fps(user_fps) = sampling {
            // The interval between sampled frames in nanoseconds
            if user_fps > 0 {
                fps_interval = 1_000_000_000u64 / (user_fps as u64);
            }
        }

        // Always collect all frames for Uniform, otherwise sample as we go
        let collect_all = matches!(sampling, VideoSamplingMethod::Uniform(_));

        while let Ok(sample) = appsink.pull_sample() {
            let buffer = sample.buffer().ok_or(VideoError::OptionError).unwrap();
            let caps = sample.caps().ok_or(VideoError::OptionError).unwrap();
            let s = caps.structure(0).ok_or(VideoError::OptionError).unwrap();
            let width = s
                .get::<i32>("width")
                .map_err(|_| VideoError::OptionError)
                .unwrap() as usize;
            let height = s
                .get::<i32>("height")
                .map_err(|_| VideoError::OptionError)
                .unwrap() as usize;
            let map = buffer.map_readable()?;
            let data = map.as_slice();
            use kornia_image::ImageSize;
            let pts = buffer.pts().map(|t| t.nseconds()).unwrap_or(0);

            debug!(
                "Frame {}: PTS = {}, size = {}x{}, data len = {}",
                frame_idx,
                pts,
                width,
                height,
                data.len()
            );

            // Handle rowstride: copy only width*3 bytes per row
            let rowstride = data.len() / height;
            let expected_row = width * 3;
            if rowstride < expected_row {
                eprintln!(
                    "[kornia-vlm] Rowstride ({}) is less than expected row size ({}). Data/caps may be corrupt. caps={:?}",
                    rowstride, expected_row, s
                );
                return Err(VideoError::OptionError);
            }
            let mut tight_data = Vec::with_capacity(width * height * 3);
            for y in 0..height {
                let start = y * rowstride;
                let end = start + expected_row;
                if end > data.len() {
                    eprintln!(
                        "[kornia-vlm] Row {} out of bounds: start={}, end={}, data.len={}, caps={:?}",
                        y, start, end, data.len(), s
                    );
                    return Err(VideoError::OptionError);
                }
                tight_data.extend_from_slice(&data[start..end]);
            }
            let img = Image::<u8, 3, A>::from_size_slice(
                ImageSize { width, height },
                &tight_data,
                allocator.clone(),
            )
            .map_err(|e| {
                eprintln!("[kornia-vlm] from_size_slice error: {:?}", e);
                VideoError::OptionError
            })?;
            // .unwrap();

            if collect_all {
                all_frames.push(img);
                all_pts.push(pts);
            } else {
                let mut take = false;
                match &sampling {
                    VideoSamplingMethod::Fps(_) => {
                        if fps_interval == 0 {
                            take = true;
                        } else {
                            if first_pts.is_none() {
                                first_pts = Some(pts);
                                fps_next_pts = pts;
                            }
                            if pts >= fps_next_pts {
                                take = true;
                                fps_next_pts += fps_interval;
                            }
                        }
                    }
                    VideoSamplingMethod::FirstN(n) => {
                        if frames.len() < *n && frames.len() < max_frames {
                            take = true;
                        } else {
                            break;
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
                    frame_pts.push(pts);
                }
                // For all methods except Uniform, stop if max_frames reached
                if frames.len() >= max_frames {
                    break;
                }
            }
            frame_idx += 1;
        }

        // If Uniform, select N evenly spaced frames from all_frames
        let timestamps: Vec<u32> = if collect_all {
            // Uniform: select N evenly spaced frames from all_frames and all_pts
            let total = all_frames.len();
            let mut ts = Vec::new();
            if let VideoSamplingMethod::Uniform(n) = sampling {
                if n > 0 && total > 0 {
                    let num = n.min(max_frames);
                    for i in 0..num {
                        let idx =
                            ((i as f64) * (total as f64 - 1.0) / (n as f64 - 1.0)).round() as usize;
                        frames.push(all_frames[idx].clone());
                        ts.push((all_pts[idx] / 1_000_000_000) as u32);
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
        // Query the total duration from the pipeline (in nanoseconds)
        let duration = pipeline
            .query_duration::<gst::ClockTime>()
            .map(|d| (d.nseconds() / 1_000_000_000) as u32);
        pipeline.set_state(gst::State::Null)?;

        // Try to get FPS from caps (if available)
        let fps = if let Some(caps) = appsink.caps() {
            if let Some(s) = caps.structure(0) {
                if let Ok(frac) = s.get::<gst::Fraction>("framerate") {
                    let num = frac.numer();
                    let denom = frac.denom();
                    if denom > 0 {
                        Some((num as f32 / denom as f32).round() as u32)
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        println!(
            "Video loaded: total frames = {}, sampled frames = {}, duration = {:?} seconds, fps = {:?}",
            all_frames.len(),
            frames.len(),
            duration,
            fps
        );

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
            resize_fast_rgb(&mut self.frames[i], &mut buf, interpolation)?;
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
    use super::*;
    use kornia_tensor::CpuAllocator;

    // cargo test -p kornia-vlm test_smolvlm2_video_reading --features cuda -- --nocapture --ignored
    #[test]
    #[ignore = "Requires GStreamer"]
    fn test_smolvlm2_video_reading() {
        let _video = Video::<CpuAllocator>::from_video_path(
            "../../example_video.mp4",
            VideoSamplingMethod::Uniform(30),
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
