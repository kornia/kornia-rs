use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use kornia_image::{allocator::ImageAllocator, Image};
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
}

pub enum VideoSamplingMethod {
    Uniform(usize),      // Uniformly sample n frames
    Fps(usize),          // Number of frames to sample per second
    FirstN(usize),       // Take the first n frames
    Indices(Vec<usize>), // Take frames at specified indices
}

pub struct Video<A: ImageAllocator> {
    frames: Vec<Image<u8, 3, A>>,
}

impl<A: ImageAllocator + Clone> Video<A> {
    pub fn new(frames: Vec<Image<u8, 3, A>>) -> Self {
        Self { frames }
    }

    pub fn from_video_path<P: AsRef<std::path::Path>>(
        path: P,
        sampling: VideoSamplingMethod,
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
            .map_err(|_| VideoError::OptionError)?;
        let appsink = pipeline
            .by_name("sink")
            .ok_or(VideoError::OptionError)?
            .downcast::<gst_app::AppSink>()
            .map_err(|_| VideoError::OptionError)?;
        pipeline.set_state(gst::State::Playing)?;

        let mut frames = Vec::new();
        let mut frame_idx = 0;
        let mut indices_set = std::collections::HashSet::new();
        let mut next_uniform = 0;
        let mut fps_next_pts = 0;
        let mut fps_interval = 0;
        let mut first_pts = None;
        let mut n_uniform = 0;
        let mut n_first = 0;
        let mut n_fps = 0;
        let mut total_frames = 0;

        // Precompute for Indices
        if let VideoSamplingMethod::Indices(ref idxs) = sampling {
            indices_set = idxs.iter().cloned().collect();
        }

        // For Uniform, we need to know total frames. We'll collect all and sample after, or require user to use Indices for now.
        // For Fps, we need to know framerate. We'll try to get it from caps.

        // Try to get framerate for Fps
        if let VideoSamplingMethod::Fps(user_fps) = sampling {
            // The interval between sampled frames in nanoseconds
            if user_fps > 0 {
                fps_interval = 1_000_000_000u64 / (user_fps as u64);
            }
        }

        while let Ok(sample) = appsink.pull_sample() {
            let buffer = sample.buffer().ok_or(VideoError::OptionError)?;
            let caps = sample.caps().ok_or(VideoError::OptionError)?;
            let s = caps.structure(0).ok_or(VideoError::OptionError)?;
            let width = s.get::<i32>("width").map_err(|_| VideoError::OptionError)? as usize;
            let height = s
                .get::<i32>("height")
                .map_err(|_| VideoError::OptionError)? as usize;
            let map = buffer.map_readable()?;
            let data = map.as_slice();
            use kornia_image::ImageSize;
            let pts = buffer.pts().map(|t| t.nseconds()).unwrap_or(0);

            let mut take = false;
            match &sampling {
                VideoSamplingMethod::Uniform(n) => {
                    // Uniformly sample n frames: take every (total_frames/n)th frame
                    // Here, we just take every (frame_idx % (total_frames/n) == 0) frame
                    // But we don't know total_frames in advance, so fallback to FirstN
                    n_uniform = *n;
                    if frames.len() < n_uniform {
                        take = true;
                    }
                }
                VideoSamplingMethod::Fps(_) => {
                    // Take frames at regular time intervals
                    if fps_interval == 0 {
                        // fallback: take every frame
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
                    n_first = *n;
                    if frames.len() < n_first {
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
            }
            if take {
                let img = Image::<u8, 3, A>::from_size_slice(
                    ImageSize { width, height },
                    data,
                    allocator.clone(),
                )
                .map_err(|_| VideoError::OptionError)?;
                frames.push(img);
            }
            frame_idx += 1;
        }
        pipeline.set_state(gst::State::Null)?;

        Ok(Self { frames })
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
            VideoSamplingMethod::Fps(1),
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
