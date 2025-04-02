use super::StreamCaptureError;
use gstreamer::prelude::*;
use kornia_image::{Image, ImageSize};
use std::path::Path;

/// The codec to use for the video writer.
pub enum VideoCodec {
    /// H.264 codec.
    H264,
}

/// The format of the image to write to the video file.
///
/// Usually will be the combination of the image format and the pixel type.
pub enum ImageFormat {
    /// 8-bit RGB format.
    Rgb8,
    /// 8-bit mono format.
    Mono8,
}

/// A struct for writing video files.
pub struct VideoWriter {
    pipeline: gstreamer::Pipeline,
    appsrc: gstreamer_app::AppSrc,
    fps: i32,
    format: ImageFormat,
    counter: u64,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl VideoWriter {
    /// Create a new VideoWriter.
    ///
    /// # Arguments
    ///
    /// * `path` - The path to save the video file.
    /// * `codec` - The codec to use for the video writer.
    /// * `format` - The expected image format.
    /// * `fps` - The frames per second of the video.
    /// * `size` - The size of the video.
    pub fn new(
        path: impl AsRef<Path>,
        codec: VideoCodec,
        format: ImageFormat,
        fps: i32,
        size: ImageSize,
    ) -> Result<Self, StreamCaptureError> {
        // make sure that we do not initialize gstreamer several times
        if !gstreamer::INITIALIZED.load(std::sync::atomic::Ordering::Relaxed) {
            gstreamer::init()?;
        }

        // TODO: Add support for other codecs
        #[allow(unreachable_patterns)]
        let _codec = match codec {
            VideoCodec::H264 => "x264enc",
            _ => {
                return Err(StreamCaptureError::InvalidConfig(
                    "Unsupported codec".to_string(),
                ))
            }
        };

        // TODO: Add support for other formats
        let format_str = match format {
            ImageFormat::Mono8 => "GRAY8",
            ImageFormat::Rgb8 => "RGB",
        };

        let path = path.as_ref().to_owned();

        let pipeline_str = format!(
            "appsrc name=src ! \
            videoconvert ! video/x-raw,format=I420 ! \
            x264enc ! \
            video/x-h264,profile=main ! \
            h264parse ! \
            mp4mux ! \
            filesink location={}",
            path.to_string_lossy()
        );

        let pipeline = gstreamer::parse::launch(&pipeline_str)?
            .dynamic_cast::<gstreamer::Pipeline>()
            .map_err(StreamCaptureError::DowncastPipelineError)?;

        let appsrc = pipeline
            .by_name("src")
            .ok_or_else(|| StreamCaptureError::GetElementByNameError)?
            .dynamic_cast::<gstreamer_app::AppSrc>()
            .map_err(StreamCaptureError::DowncastPipelineError)?;

        appsrc.set_format(gstreamer::Format::Time);

        let caps = gstreamer::Caps::builder("video/x-raw")
            .field("format", format_str)
            .field("width", size.width as i32)
            .field("height", size.height as i32)
            .field("framerate", gstreamer::Fraction::new(fps, 1))
            .build();

        appsrc.set_caps(Some(&caps));

        appsrc.set_is_live(true);
        appsrc.set_property("block", false);

        Ok(Self {
            pipeline,
            appsrc,
            fps,
            format,
            counter: 0,
            handle: None,
        })
    }

    /// Start the video writer.
    ///
    /// Set the pipeline to playing and launch a task to handle the bus messages.
    pub fn start(&mut self) -> Result<(), StreamCaptureError> {
        // set the pipeline to playing
        self.pipeline.set_state(gstreamer::State::Playing)?;

        let bus = self.pipeline.bus().ok_or(StreamCaptureError::BusError)?;

        // launch a task to handle the bus messages, exit when EOS is received and set the pipeline to null
        let handle = std::thread::spawn(move || {
            for msg in bus.iter_timed(gstreamer::ClockTime::NONE) {
                match msg.view() {
                    gstreamer::MessageView::Eos(..) => {
                        log::debug!("gstreamer received EOS");
                        break;
                    }
                    gstreamer::MessageView::Error(err) => {
                        log::error!(
                            "Error from {:?}: {} ({:?})",
                            msg.src().map(|s| s.path_string()),
                            err.error(),
                            err.debug()
                        );
                        break;
                    }
                    _ => {}
                }
            }
        });

        self.handle = Some(handle);

        Ok(())
    }

    ///  Close the video writer.
    ///
    /// Set the pipeline to null and join the thread.
    ///
    pub fn close(&mut self) -> Result<(), StreamCaptureError> {
        // send end of stream to the appsrc
        self.appsrc.end_of_stream()?;

        if let Some(handle) = self.handle.take() {
            handle.join().expect("Failed to join thread");
        }

        self.pipeline.set_state(gstreamer::State::Null)?;

        Ok(())
    }
    /// Write an image to the video file.
    ///
    /// # Arguments
    ///
    /// * `img` - The image to write to the video file.
    // TODO: explore supporting write_async
    pub fn write<const C: usize>(&mut self, img: &Image<u8, C>) -> Result<(), StreamCaptureError> {
        // check if the image channels are correct
        match self.format {
            ImageFormat::Mono8 => {
                if C != 1 {
                    return Err(StreamCaptureError::InvalidImageFormat(format!(
                        "Invalid number of channels: expected 1, got {}",
                        C
                    )));
                }
            }
            ImageFormat::Rgb8 => {
                if C != 3 {
                    return Err(StreamCaptureError::InvalidImageFormat(format!(
                        "Invalid number of channels: expected 3, got {}",
                        C
                    )));
                }
            }
        }

        // TODO: verify is there is a cheaper way to copy the buffer
        let mut buffer = gstreamer::Buffer::from_mut_slice(img.as_slice().to_vec());

        let pts =
            gstreamer::ClockTime::from_nseconds(self.counter * 1_000_000_000 / self.fps as u64);
        let duration = gstreamer::ClockTime::from_nseconds(1_000_000_000 / self.fps as u64);

        let buffer_ref = buffer.get_mut().expect("Failed to get buffer");
        buffer_ref.set_pts(Some(pts));
        buffer_ref.set_duration(Some(duration));

        self.counter += 1;

        if let Err(err) = self.appsrc.push_buffer(buffer) {
            return Err(StreamCaptureError::InvalidConfig(err.to_string()));
        }

        Ok(())
    }
}

impl Drop for VideoWriter {
    fn drop(&mut self) {
        if self.handle.is_some() {
            self.close().expect("Failed to close video writer");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ImageFormat, VideoCodec, VideoWriter};
    use kornia_image::{Image, ImageSize};

    #[ignore = "need gstreamer in CI"]
    #[test]
    fn video_writer_rgb8u() -> Result<(), Box<dyn std::error::Error>> {
        let tmp_dir = tempfile::tempdir()?;
        std::fs::create_dir_all(tmp_dir.path())?;

        let file_path = tmp_dir.path().join("test.mp4");

        let size = ImageSize {
            width: 6,
            height: 4,
        };

        let mut writer =
            VideoWriter::new(&file_path, VideoCodec::H264, ImageFormat::Rgb8, 30, size)?;
        writer.start()?;

        let img = Image::<u8, 3>::new(size, vec![0; size.width * size.height * 3])?;
        writer.write(&img)?;
        writer.close()?;

        assert!(file_path.exists(), "File does not exist: {:?}", file_path);

        Ok(())
    }

    #[ignore = "need gstreamer in CI"]
    #[test]
    fn video_writer_mono8u() -> Result<(), Box<dyn std::error::Error>> {
        let tmp_dir = tempfile::tempdir()?;
        std::fs::create_dir_all(tmp_dir.path())?;

        let file_path = tmp_dir.path().join("test.mp4");

        let size = ImageSize {
            width: 6,
            height: 4,
        };

        let mut writer =
            VideoWriter::new(&file_path, VideoCodec::H264, ImageFormat::Mono8, 30, size)?;
        writer.start()?;

        let img = Image::<u8, 1>::new(size, vec![0; size.width * size.height])?;
        writer.write(&img)?;
        writer.close()?;

        assert!(file_path.exists(), "File does not exist: {:?}", file_path);

        Ok(())
    }
}
