use super::{
    capture::StreamerState, error::VideoReaderError, set_location_property, StreamCapture,
    StreamCaptureError,
};
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
    /// Whether `start` has been called since the last `close`.
    started: bool,
    /// The first pipeline error seen by `write`, returned again by `close`.
    error: Option<gstreamer::glib::Error>,
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
    ///
    /// # Returns
    ///
    /// A writer that encodes H.264 into an MP4 file at `path`; call [`Self::start`] before
    /// writing.
    ///
    /// # Errors
    ///
    /// Returns [`StreamCaptureError::InvalidConfig`] if the codec is unsupported or `fps` is not
    /// positive, and [`StreamCaptureError::GStreamerError`] if GStreamer cannot be initialized
    /// or the pipeline cannot be built (e.g. `x264enc` is not installed).
    pub fn new(
        path: impl AsRef<Path>,
        codec: VideoCodec,
        format: ImageFormat,
        fps: i32,
        size: ImageSize,
    ) -> Result<Self, StreamCaptureError> {
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

        // The output path is set as a property after parsing (see `set_location_property`).
        let pipeline_desc = "appsrc name=src ! \
            videoconvert ! video/x-raw,format=I420 ! \
            x264enc ! \
            video/x-h264,profile=main ! \
            h264parse ! \
            mp4mux ! \
            filesink name=filesink";

        Self::from_pipeline_description(pipeline_desc, path.as_ref(), format, fps, size)
    }

    /// Creates a writer from a pipeline description that starts with an `appsrc` named `src`
    /// and ends with a `filesink` named `filesink`, whose `location` is set to `path`.
    ///
    /// # Errors
    ///
    /// Returns [`StreamCaptureError::InvalidConfig`] if `fps` is not positive, and the errors of
    /// parsing the pipeline or finding its `src` and `filesink` elements.
    fn from_pipeline_description(
        pipeline_desc: &str,
        path: &Path,
        format: ImageFormat,
        fps: i32,
        size: ImageSize,
    ) -> Result<Self, StreamCaptureError> {
        // make sure that we do not initialize gstreamer several times
        if !gstreamer::INITIALIZED.load(std::sync::atomic::Ordering::Relaxed) {
            gstreamer::init()?;
        }

        // TODO: Add support for other formats
        let format_str = match format {
            ImageFormat::Mono8 => "GRAY8",
            ImageFormat::Rgb8 => "RGB",
        };

        if fps <= 0 {
            return Err(StreamCaptureError::InvalidConfig(format!(
                "fps must be positive, got {fps}"
            )));
        }

        let pipeline = gstreamer::parse::launch(pipeline_desc)?
            .dynamic_cast::<gstreamer::Pipeline>()
            .map_err(StreamCaptureError::DowncastPipelineError)?;
        set_location_property(&pipeline, "filesink", path)?;

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
            started: false,
            error: None,
        })
    }

    /// Start the video writer.
    ///
    /// Sets the pipeline to playing. Calling it again before [`Self::close`] has no further
    /// effect.
    ///
    /// # Returns
    ///
    /// `Ok(())` once the pipeline has been asked to start playing.
    ///
    /// # Errors
    ///
    /// Returns [`StreamCaptureError::SetPipelineStateError`] if the pipeline cannot start.
    pub fn start(&mut self) -> Result<(), StreamCaptureError> {
        self.pipeline.set_state(gstreamer::State::Playing)?;
        self.started = true;
        Ok(())
    }

    /// Close the video writer.
    ///
    /// Sends end-of-stream and waits until it reaches the file sink, so the muxer can finish the
    /// file, then sets the pipeline to null. Setting the pipeline to null first would tear it
    /// down before end-of-stream arrives, and the wait would never end.
    ///
    /// GStreamer keeps bus messages until they are read, so an error raised while writing (for
    /// example caps the encoder cannot accept) is still found here and returned, instead of
    /// leaving a silently empty file.
    ///
    /// # Returns
    ///
    /// `Ok(())` once end-of-stream has reached the file sink and the pipeline is stopped. If no
    /// frame was written, the file may still not be playable.
    ///
    /// # Errors
    ///
    /// Returns [`StreamCaptureError::GStreamerError`] with the first pipeline error,
    /// [`StreamCaptureError::GstreamerFlowError`] if end-of-stream cannot be sent,
    /// [`StreamCaptureError::BusError`] if the pipeline has no bus, and
    /// [`StreamCaptureError::SetPipelineStateError`] if the pipeline cannot be stopped. The
    /// pipeline is set to null in every case except a missing bus.
    pub fn close(&mut self) -> Result<(), StreamCaptureError> {
        let eos = self.appsrc.end_of_stream();
        let waited = match (std::mem::take(&mut self.started), self.error.take()) {
            // The pipeline already failed, so end-of-stream will not reach the sink.
            (_, Some(err)) => Ok(Some(err)),
            (true, None) => self.wait_for_eos(eos.is_ok()),
            (false, None) => Ok(None),
        };

        // Stop the pipeline even if waiting failed, then report the most useful error.
        let stopped = self.pipeline.set_state(gstreamer::State::Null);
        if let Some(err) = waited? {
            return Err(err.into());
        }
        stopped?;
        eos?;
        Ok(())
    }

    /// Waits for end-of-stream to reach the sink and returns the pipeline error, if any.
    ///
    /// When end-of-stream could not be sent nothing will arrive, so only an error that is
    /// already on the bus is looked for.
    fn wait_for_eos(
        &self,
        eos_sent: bool,
    ) -> Result<Option<gstreamer::glib::Error>, StreamCaptureError> {
        let bus = self.pipeline.bus().ok_or(StreamCaptureError::BusError)?;
        let types = [gstreamer::MessageType::Eos, gstreamer::MessageType::Error];
        let msg = if eos_sent {
            bus.timed_pop_filtered(gstreamer::ClockTime::NONE, &types)
        } else {
            bus.pop_filtered(&types)
        };
        Ok(msg.and_then(|msg| pipeline_error(&msg)))
    }

    /// Write an image to the video file.
    ///
    /// # Arguments
    ///
    /// * `img` - The image to write to the video file.
    ///
    /// # Returns
    ///
    /// `Ok(())` once the frame has been queued for encoding.
    ///
    /// # Errors
    ///
    /// Returns [`StreamCaptureError::InvalidImageFormat`] if the image has the wrong number of
    /// channels, [`StreamCaptureError::GStreamerError`] if the pipeline has failed (this and
    /// every later frame is then rejected, and [`Self::close`] returns the same error),
    /// [`StreamCaptureError::GetBufferError`] if the frame buffer cannot be prepared, and
    /// [`StreamCaptureError::InvalidConfig`] if the frame cannot be pushed into the pipeline.
    // TODO: explore supporting write_async
    pub fn write<const C: usize>(&mut self, img: &Image<u8, C>) -> Result<(), StreamCaptureError> {
        // check if the image channels are correct
        match self.format {
            ImageFormat::Mono8 => {
                if C != 1 {
                    return Err(StreamCaptureError::InvalidImageFormat(format!(
                        "Invalid number of channels: expected 1, got {C}"
                    )));
                }
            }
            ImageFormat::Rgb8 => {
                if C != 3 {
                    return Err(StreamCaptureError::InvalidImageFormat(format!(
                        "Invalid number of channels: expected 3, got {C}"
                    )));
                }
            }
        }

        // TODO: verify is there is a cheaper way to copy the buffer
        let mut buffer = gstreamer::Buffer::from_mut_slice(img.as_slice().to_vec());

        let pts =
            gstreamer::ClockTime::from_nseconds(self.counter * 1_000_000_000 / self.fps as u64);
        let duration = gstreamer::ClockTime::from_nseconds(1_000_000_000 / self.fps as u64);

        let buffer_ref = buffer.get_mut().ok_or(StreamCaptureError::GetBufferError)?;
        buffer_ref.set_pts(Some(pts));
        buffer_ref.set_duration(Some(duration));

        self.counter += 1;

        // A failed pipeline does not make `push_buffer` fail: frames would keep queueing in
        // `appsrc` until `close`. Check the bus so the error is returned at the next frame.
        if self.error.is_none() {
            let types = [gstreamer::MessageType::Error];
            if let Some(msg) = self.pipeline.bus().and_then(|bus| bus.pop_filtered(&types)) {
                self.error = pipeline_error(&msg);
            }
        }
        if let Some(err) = &self.error {
            return Err(err.clone().into());
        }

        if let Err(err) = self.appsrc.push_buffer(buffer) {
            return Err(StreamCaptureError::InvalidConfig(err.to_string()));
        }

        Ok(())
    }
}

/// Logs a bus error message and returns its error; `None` for any other message.
fn pipeline_error(msg: &gstreamer::Message) -> Option<gstreamer::glib::Error> {
    let gstreamer::MessageView::Error(err) = msg.view() else {
        log::debug!("gstreamer received {:?}", msg.type_());
        return None;
    };
    log::error!(
        "Error from {:?}: {} ({:?})",
        msg.src().map(|s| s.path_string()),
        err.error(),
        err.debug()
    );
    Some(err.error())
}

impl Drop for VideoWriter {
    fn drop(&mut self) {
        if self.started {
            if let Err(e) = self.close() {
                log::warn!("Failed to close video writer safely on drop: {:?}", e);
            }
        }
    }
}

/// A struct for reading video files
pub struct VideoReader(StreamCapture);

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

        // The input path is set as a property after parsing (see `set_location_property`);
        // the pipeline only starts reading once `start` sets it to PLAYING.
        let pipeline = format!(
            "filesrc name=filesrc ! \
            decodebin ! \
            videoconvert ! \
            video/x-raw,format={video_format} ! \
            appsink name=sink sync=true"
        );

        let capture = StreamCapture::new(&pipeline)?;
        set_location_property(&capture.pipeline, "filesrc", path.as_ref())?;

        Ok(Self(capture))
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

    /// Grabs the last captured image frame.
    ///
    /// # Returns
    ///
    /// An Option containing the last captured Image or None if no image has been captured yet.
    #[inline]
    pub fn grab_rgb8(&mut self) -> Result<Option<Image<u8, 3>>, VideoReaderError> {
        self.0
            .grab_rgb8()
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
    use super::{ImageFormat, VideoCodec, VideoReader, VideoWriter};
    use kornia_image::{Image, ImageSize};

    /// Regression: paths were quoted into the pipeline string, which rejected `\`
    /// (breaking Windows paths). The path must now reach `filesrc` verbatim.
    #[test]
    fn video_reader_accepts_any_path() -> Result<(), Box<dyn std::error::Error>> {
        use gstreamer::prelude::*;
        let path = std::path::Path::new(r"C:\my videos\clip ! fakesink name=x.mp4");
        let reader = VideoReader::new(path, ImageFormat::Rgb8)?;
        let src = reader
            .0
            .pipeline
            .by_name("filesrc")
            .ok_or("missing filesrc")?;
        assert_eq!(
            src.property::<Option<String>>("location").as_deref(),
            path.to_str()
        );
        Ok(())
    }

    /// Runs `f` on its own thread and returns its result, or `None` if it has not finished
    /// after `timeout` (the thread is then left running).
    fn finishes_within<T: Send + 'static>(
        timeout: std::time::Duration,
        f: impl FnOnce() -> T + Send + 'static,
    ) -> Option<T> {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(f());
        });
        rx.recv_timeout(timeout).ok()
    }

    /// `close` must return after end-of-stream has reached the file sink, so every written
    /// frame ends up in the file. It used to set the pipeline to `Null` first, so the bus thread
    /// it then joined never saw end-of-stream and `close` could hang forever.
    ///
    /// Uses uncompressed video in Matroska (gst-plugins-base and -good only) instead of the
    /// H.264 pipeline of `VideoWriter::new`, so it runs without x264.
    #[test]
    fn video_writer_close_returns_and_keeps_every_frame() -> Result<(), Box<dyn std::error::Error>>
    {
        const FRAMES: usize = 5;
        let size = ImageSize {
            width: 64,
            height: 48,
        };
        let tmp_dir = tempfile::tempdir()?;
        let file_path = tmp_dir.path().join("test.mkv");

        let mut writer = VideoWriter::from_pipeline_description(
            "appsrc name=src ! videoconvert ! video/x-raw,format=I420 ! \
             matroskamux ! filesink name=filesink",
            &file_path,
            ImageFormat::Rgb8,
            30,
            size,
        )?;
        // A second start must not change anything (it used to spawn a second bus thread that
        // could take the end-of-stream message and leave close waiting forever).
        writer.start()?;
        writer.start()?;
        for i in 0..FRAMES {
            let img =
                Image::<u8, 3>::new(size, vec![(i * 40) as u8; size.width * size.height * 3])?;
            writer.write(&img)?;
        }
        let closed = finishes_within(std::time::Duration::from_secs(10), move || {
            writer.close().map_err(|e| e.to_string())
        });
        assert_eq!(closed, Some(Ok(())), "close did not return within 10 s");

        let frames = count_frames(&file_path, FRAMES)?;
        assert_eq!(frames, FRAMES, "frames read back from the file");
        Ok(())
    }

    /// Reads `path` back and returns how many frames arrive, stopping at `expected` or after
    /// about 5 seconds.
    fn count_frames(
        path: &std::path::Path,
        expected: usize,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        let mut reader = VideoReader::new(path, ImageFormat::Rgb8)?;
        reader.start()?;
        let mut frames = 0;
        for _ in 0..500 {
            if reader.grab_rgb8()?.is_some() {
                frames += 1;
                if frames == expected {
                    break;
                }
            } else {
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        }
        reader.close()?;
        Ok(frames)
    }

    /// A pipeline error raised while writing (here, RGB frames forced into GRAY8 caps, so the
    /// caps cannot be negotiated) must be returned by `write` and by `close`, instead of `Ok`
    /// with an empty file.
    #[test]
    fn video_writer_close_returns_the_pipeline_error() -> Result<(), Box<dyn std::error::Error>> {
        let size = ImageSize {
            width: 64,
            height: 48,
        };
        let tmp_dir = tempfile::tempdir()?;
        let mut writer = VideoWriter::from_pipeline_description(
            "appsrc name=src ! video/x-raw,format=GRAY8 ! videoconvert ! \
             matroskamux ! filesink name=filesink",
            &tmp_dir.path().join("test.mkv"),
            ImageFormat::Rgb8,
            30,
            size,
        )?;
        writer.start()?;
        let frame = Image::<u8, 3>::new(size, vec![0; size.width * size.height * 3])?;
        // Frames are accepted until the pipeline has posted its error; from then on `write`
        // reports it instead of queueing frames that will never be encoded.
        let mut write_error = false;
        for _ in 0..200 {
            if matches!(
                writer.write(&frame),
                Err(super::StreamCaptureError::GStreamerError(_))
            ) {
                write_error = true;
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(write_error, "write must report the pipeline error");
        let closed = finishes_within(std::time::Duration::from_secs(10), move || {
            matches!(
                writer.close(),
                Err(super::StreamCaptureError::GStreamerError(_))
            )
        });
        assert_eq!(
            closed,
            Some(true),
            "close must return the pipeline error within 10 s"
        );

        // The pipeline fails only once the first frame flows, after the last `write`, so the
        // error is found by `close` while it waits for end-of-stream.
        let mut writer = VideoWriter::from_pipeline_description(
            "appsrc name=src ! video/x-raw,format=GRAY8 ! videoconvert ! \
             matroskamux ! filesink name=filesink",
            &tmp_dir.path().join("test2.mkv"),
            ImageFormat::Rgb8,
            30,
            size,
        )?;
        writer.start()?;
        writer.write(&frame)?;
        let closed = finishes_within(std::time::Duration::from_secs(10), move || {
            matches!(
                writer.close(),
                Err(super::StreamCaptureError::GStreamerError(_))
            )
        });
        assert_eq!(
            closed,
            Some(true),
            "close must find the pipeline error while waiting"
        );
        Ok(())
    }

    #[ignore = "need gstreamer in CI"]
    #[test]
    fn video_writer_rgb8u() -> Result<(), Box<dyn std::error::Error>> {
        let tmp_dir = tempfile::tempdir()?;
        std::fs::create_dir_all(tmp_dir.path())?;

        let file_path = tmp_dir.path().join("test.mp4");

        // x264 cannot encode tiny frames such as 6x4 (caps not negotiated); 16x16 works.
        let size = ImageSize {
            width: 16,
            height: 16,
        };

        let mut writer =
            VideoWriter::new(&file_path, VideoCodec::H264, ImageFormat::Rgb8, 30, size)?;
        writer.start()?;

        let img = Image::<u8, 3>::new(size, vec![0; size.width * size.height * 3])?;
        writer.write(&img)?;
        writer.close()?;

        assert_eq!(
            count_frames(&file_path, 1)?,
            1,
            "frame read back from {file_path:?}"
        );

        Ok(())
    }

    #[ignore = "need gstreamer in CI"]
    #[test]
    fn video_writer_mono8u() -> Result<(), Box<dyn std::error::Error>> {
        let tmp_dir = tempfile::tempdir()?;
        std::fs::create_dir_all(tmp_dir.path())?;

        let file_path = tmp_dir.path().join("test.mp4");

        // x264 cannot encode tiny frames such as 6x4 (caps not negotiated); 16x16 works.
        let size = ImageSize {
            width: 16,
            height: 16,
        };

        let mut writer =
            VideoWriter::new(&file_path, VideoCodec::H264, ImageFormat::Mono8, 30, size)?;
        writer.start()?;

        let img = Image::<u8, 1>::new(size, vec![0; size.width * size.height])?;
        writer.write(&img)?;
        writer.close()?;

        assert_eq!(
            count_frames(&file_path, 1)?,
            1,
            "frame read back from {file_path:?}"
        );

        Ok(())
    }
}
