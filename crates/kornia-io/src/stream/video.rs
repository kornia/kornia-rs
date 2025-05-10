use super::{StreamCapture, StreamCaptureError};
use gstreamer::prelude::*;
use kornia_image::{Image, ImageSize};
use std::{path::Path, time::Duration};

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

/// A enum representing the state of [VideoReader] pipeline.
///
/// For more info, refer to <https://gstreamer.freedesktop.org/documentation/additional/design/states.html?gi-language=c>
pub enum State {
    /// This is the initial state of a pipeline.
    Null,
    /// The element should be prepared to go to [State::Paused]
    Ready,
    /// The video is paused.
    Paused,
    /// The video is playing.
    Playing,
}

impl From<gstreamer::State> for State {
    fn from(value: gstreamer::State) -> Self {
        match value {
            gstreamer::State::VoidPending => State::Null,
            gstreamer::State::Null => State::Null,
            gstreamer::State::Ready => State::Ready,
            gstreamer::State::Paused => State::Paused,
            gstreamer::State::Playing => State::Playing,
        }
    }
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

pub use gstreamer::SeekFlags;

/// A struct for reading video files
pub struct VideoReader(StreamCapture);

impl VideoReader {
    /// Creates a new `VideoReader`
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the video file to be read.
    /// * `format` - The expected image format.
    /// * `fps` - The frames per second of the video.
    /// * `size` - The size of the video.
    pub fn new(path: impl AsRef<Path>, format: ImageFormat) -> Result<Self, StreamCaptureError> {
        // TODO: Support more formats
        let video_format = match format {
            ImageFormat::Rgb8 => "RGB",
            ImageFormat::Mono8 => "GRAY8",
        };

        let pipeline = format!(
            "filesrc location=\"{}\" ! \
            decodebin ! \
            videoconvert ! \
            video/x-raw,format={} ! \
            appsink name=sink sync=true",
            path.as_ref().to_string_lossy(),
            video_format
        );

        let capture = StreamCapture::new(&pipeline)?;

        Ok(Self(capture))
    }

    /// Starts the video reader pipeline
    #[inline]
    pub fn start(&mut self) -> Result<(), StreamCaptureError> {
        self.0.start()
    }

    /// Pauses the video reader pipeline
    #[inline]
    pub fn pause(&mut self) -> Result<(), StreamCaptureError> {
        self.0.get_pipeline().set_state(gstreamer::State::Paused)?;
        Ok(())
    }

    /// Close the video reader pipeline
    #[inline]
    pub fn close(&self) -> Result<(), StreamCaptureError> {
        self.0.close()
    }

    /// Gets the current FPS of the video
    #[inline]
    pub fn get_fps(&self) -> Option<f64> {
        self.0.get_fps()
    }

    /// Grabs the last captured image frame.
    ///
    /// # Returns
    ///
    /// An Option containing the last captured Image or None if no image has been captured yet.
    #[inline]
    pub fn grab(&mut self) -> Result<Option<Image<u8, 3>>, StreamCaptureError> {
        self.0.grab()
    }

    /// Gets the current state of the video pipeline
    #[inline]
    pub fn get_state(&self) -> State {
        self.0.get_pipeline().current_state().into()
    }

    /// Gets the current position in the video.
    ///
    /// # Returns
    ///
    /// * `Some(Duration)` - The current position as a Duration from the start of the video
    /// * `None` - If the position could not be determined
    pub fn get_pos(&self) -> Option<Duration> {
        let clock_time = self
            .0
            .get_pipeline()
            .query_position::<gstreamer::format::ClockTime>()?;

        let duration = Duration::from_nanos(clock_time.nseconds());
        Some(duration)
    }

    /// Gets the total duration of the video.
    ///
    /// # Returns
    ///
    /// * `Some(Duration)` - The total duration of the video
    /// * `None` - If the video duration could not be determined
    pub fn get_duration(&self) -> Option<Duration> {
        let clock_time = self
            .0
            .get_pipeline()
            .query_duration::<gstreamer::format::ClockTime>()?;

        let duration = Duration::from_nanos(clock_time.nseconds());
        Some(duration)
    }

    /// Seeks to a specific position in the video.
    ///
    /// # Arguments
    ///
    /// * `pos` - The position to seek to, as a Duration from the start of the video.
    ///
    /// # Returns
    ///
    /// `true` if the seek operation was successful, `false` otherwise.
    pub fn seek(&self, seek_flags: gstreamer::SeekFlags, pos: Duration) -> bool {
        let pipeline = self.0.get_pipeline();

        // Convert the Duration to ClockTime (nanoseconds)
        let clock_time = gstreamer::ClockTime::from_nseconds(pos.as_nanos() as u64);

        pipeline.seek_simple(seek_flags, clock_time).is_ok()
    }

    /// Sets the playback speed of the video.
    ///
    /// # Arguments
    ///
    /// * `speed` - The playback speed factor. 1.0 is normal speed, 0.5 is half speed, 2.0 is
    ///   double speed, etc.
    ///
    /// # Returns
    ///
    /// `true` if the speed change operation was successful, `false` otherwise.
    pub fn set_playback_speed(&self, speed: f64) -> bool {
        if speed <= 0.0 {
            return false; // Speed must be positive
        }

        let pipeline = self.0.get_pipeline();

        // Get current position to maintain the playback position
        let position = match pipeline.query_position::<gstreamer::format::ClockTime>() {
            Some(pos) => pos,
            None => return false, // Can't determine current position
        };

        // Seek with the new rate
        pipeline
            .seek(
                speed,
                gstreamer::SeekFlags::FLUSH | gstreamer::SeekFlags::ACCURATE,
                gstreamer::SeekType::Set,
                position,
                gstreamer::SeekType::None,
                gstreamer::ClockTime::NONE,
            )
            .is_ok()
    }

    /// Restart the video from the beginning
    pub fn restart(&mut self) -> Result<(), StreamCaptureError> {
        let pipeline = self.0.get_pipeline_mut();
        pipeline.set_state(gstreamer::State::Null)?;
        std::thread::sleep(Duration::from_micros(1));
        self.start()?;
        Ok(())
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
