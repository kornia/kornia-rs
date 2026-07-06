use super::{capture::StreamerState, error::VideoReaderError, StreamCapture, StreamCaptureError};
use gstreamer::prelude::*;
use kornia_image::{Image, ImageSize};
use std::{path::Path, time::Duration};

pub use gstreamer::SeekFlags;

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
        self.pipeline.set_state(gstreamer::State::Null)?;

        if let Some(handle) = self.handle.take() {
            if handle.join().is_err() {
                return Err(StreamCaptureError::JoinThreadError);
            }
        }

        Ok(())
    }
    /// Write an image to the video file.
    ///
    /// # Arguments
    ///
    /// * `img` - The image to write to the video file.
    // TODO: explore supporting write_async
    pub fn write<const C: usize>(&mut self, img: &Image<u8, C>) -> Result<(), StreamCaptureError> {
        self.validate_channels::<C>()?;

        // A single copy into a GStreamer-owned buffer is unavoidable here because
        // `img` is only borrowed. Callers that can yield ownership should use
        // [`write_owned`](Self::write_owned) to avoid the copy.
        let buffer = gstreamer::Buffer::from_mut_slice(img.as_slice().to_vec());
        self.push_frame(buffer)
    }

    /// Write an image to the video file without copying its pixel data.
    ///
    /// Unlike [`write`](Self::write) — which must copy the borrowed image into a
    /// GStreamer-owned buffer — this takes ownership of `img` and hands its backing
    /// memory directly to GStreamer, keeping the image alive for the buffer's
    /// lifetime. Use it on hot paths when the caller can give up the frame.
    pub fn write_owned<const C: usize>(
        &mut self,
        img: Image<u8, C>,
    ) -> Result<(), StreamCaptureError> {
        self.validate_channels::<C>()?;

        // Wrap the owned image so GStreamer can borrow its bytes with no copy; the
        // image is dropped (freeing its memory) when the buffer is released.
        struct ImageBytes<const C: usize>(Image<u8, C>);
        impl<const C: usize> AsRef<[u8]> for ImageBytes<C> {
            fn as_ref(&self) -> &[u8] {
                self.0.as_slice()
            }
        }

        let buffer = gstreamer::Buffer::from_slice(ImageBytes(img));
        self.push_frame(buffer)
    }

    /// Check that the image's channel count matches the writer's configured format.
    fn validate_channels<const C: usize>(&self) -> Result<(), StreamCaptureError> {
        let expected = match self.format {
            ImageFormat::Mono8 => 1,
            ImageFormat::Rgb8 => 3,
        };
        if C != expected {
            return Err(StreamCaptureError::InvalidImageFormat(format!(
                "Invalid number of channels: expected {expected}, got {C}"
            )));
        }
        Ok(())
    }

    /// Stamp `buffer` with the next PTS/duration and push it into the appsrc.
    fn push_frame(&mut self, mut buffer: gstreamer::Buffer) -> Result<(), StreamCaptureError> {
        let pts =
            gstreamer::ClockTime::from_nseconds(self.counter * 1_000_000_000 / self.fps as u64);
        let duration = gstreamer::ClockTime::from_nseconds(1_000_000_000 / self.fps as u64);

        let buffer_ref = buffer.get_mut().ok_or(StreamCaptureError::GetBufferError)?;
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
            if let Err(e) = self.close() {
                log::warn!("Failed to close video writer safely on drop: {:?}", e);
            }
        }
    }
}

/// A struct for reading video files
pub struct VideoReader(StreamCapture, ImageFormat);

impl VideoReader {
    /// Creates a new `VideoReader`
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the video file to be read.
    /// * `format` - The expected image format.
    pub fn new(path: impl AsRef<Path>, format: ImageFormat) -> Result<Self, VideoReaderError> {
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

        Ok(Self(capture, format))
    }

    /// Starts the video reader pipeline
    #[inline]
    pub fn start(&mut self) -> Result<(), VideoReaderError> {
        self.0.start().map_err(VideoReaderError::StreamCaptureError)
    }

    /// Pauses the video reader pipeline
    #[inline]
    pub fn pause(&mut self) -> Result<(), VideoReaderError> {
        self.0
            .pipeline
            .set_state(gstreamer::State::Paused)
            .map_err(StreamCaptureError::from)?;
        Ok(())
    }

    /// Close the video reader pipeline
    #[inline]
    pub fn close(&self) -> Result<(), VideoReaderError> {
        self.0.close()?;
        Ok(())
    }

    /// Gets the current FPS of the video
    #[inline]
    pub fn get_fps(&self) -> Option<f64> {
        self.0.get_fps()
    }

    /// Grabs the last captured frame as an RGB8 image.
    ///
    /// Returns [`VideoReaderError::StreamCaptureError`] with
    /// [`StreamCaptureError::InvalidImageFormat`] if the reader was created with a
    /// non-RGB [`ImageFormat`] (use [`grab_mono8`](Self::grab_mono8) instead).
    #[inline]
    pub fn grab_rgb8(&mut self) -> Result<Option<Image<u8, 3>>, VideoReaderError> {
        if !matches!(self.1, ImageFormat::Rgb8) {
            return Err(StreamCaptureError::InvalidImageFormat(
                "reader was created with a non-RGB format; use grab_mono8".to_string(),
            )
            .into());
        }
        self.0
            .grab_rgb8()
            .map_err(VideoReaderError::StreamCaptureError)
    }

    /// Grabs the last captured frame as a single-channel (mono8) image.
    ///
    /// Returns [`VideoReaderError::StreamCaptureError`] with
    /// [`StreamCaptureError::InvalidImageFormat`] if the reader was created with a
    /// non-mono [`ImageFormat`] (use [`grab_rgb8`](Self::grab_rgb8) instead).
    #[inline]
    pub fn grab_mono8(&mut self) -> Result<Option<Image<u8, 1>>, VideoReaderError> {
        if !matches!(self.1, ImageFormat::Mono8) {
            return Err(StreamCaptureError::InvalidImageFormat(
                "reader was created with a non-mono format; use grab_rgb8".to_string(),
            )
            .into());
        }
        self.0
            .grab_mono8()
            .map_err(VideoReaderError::StreamCaptureError)
    }

    /// Gets the current state of the video pipeline
    #[inline]
    pub fn get_state(&self) -> StreamerState {
        self.0.get_state()
    }

    /// Gets the current position in the video.
    ///
    /// # Returns
    ///
    /// * `Some(Duration)` - The current position as a Duration from the start of the video in nanoseconds
    /// * `None` - If the position could not be determined
    pub fn get_pos(&self) -> Option<Duration> {
        let clock_time = self
            .0
            .pipeline
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
            .pipeline
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
    /// * `Ok(())` - If the seek operation was successful.
    /// * `Err(VideoReaderError)` - If the seek operation failed.
    pub fn seek(
        &self,
        seek_flags: gstreamer::SeekFlags,
        pos: Duration,
    ) -> Result<(), VideoReaderError> {
        let pipeline = &self.0.pipeline;

        // Convert the Duration to ClockTime (nanoseconds)
        let clock_time = gstreamer::ClockTime::from_nseconds(pos.as_nanos() as u64);

        pipeline
            .seek_simple(seek_flags, clock_time)
            .map_err(|_| VideoReaderError::SeekError)
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
    pub fn set_playback_speed(&self, speed: f64) -> Result<(), VideoReaderError> {
        if speed <= 0.0 {
            return Err(VideoReaderError::InvalidPlaybackSpeed); // Speed must be positive
        }

        let pipeline = &self.0.pipeline;

        // Get current position to maintain the playback position
        let position = pipeline
            .query_position::<gstreamer::format::ClockTime>()
            .ok_or(VideoReaderError::CurrentPosError)?;

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
            .map_err(|_| VideoReaderError::SeekError)
    }

    /// Resets the video to the beginning without changing its state.
    ///
    /// This function seeks the video to the origin (start) but does not stop or start the pipeline.
    pub fn reset(&self) -> Result<(), VideoReaderError> {
        let pipeline = &self.0.pipeline;
        pipeline
            .seek_simple(
                gstreamer::SeekFlags::FLUSH | gstreamer::SeekFlags::ACCURATE,
                gstreamer::ClockTime::ZERO,
            )
            .map_err(|_| VideoReaderError::SeekError)?;
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

        assert!(file_path.exists(), "File does not exist: {file_path:?}");

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

        assert!(file_path.exists(), "File does not exist: {file_path:?}");

        Ok(())
    }
}
