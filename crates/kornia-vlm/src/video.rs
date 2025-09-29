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

pub struct Video<A: ImageAllocator> {
    frames: Vec<Image<u8, 3, A>>,
}

impl<A: ImageAllocator + Clone> Video<A> {
    pub fn new(frames: Vec<Image<u8, 3, A>>) -> Self {
        Self { frames }
    }

    pub fn from_video_path<P: AsRef<std::path::Path>>(
        path: P,
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
            let img = Image::<u8, 3, A>::from_size_slice(
                ImageSize { width, height },
                data,
                allocator.clone(),
            )
            .map_err(|_| VideoError::OptionError)?;
            frames.push(img);
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
        let _video =
            Video::<CpuAllocator>::from_video_path("../../example_video.mp4", CpuAllocator)
                .unwrap();
        println!(
            "Video loaded successfully with {} frames.",
            _video.frames.len()
        );
        println!("First frame shape: {:?}.", _video.frames[0].0.shape);
    }
}
