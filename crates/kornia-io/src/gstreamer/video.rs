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

/// How long [`VideoWriter::close`] waits for end-of-stream to reach the file sink.
const CLOSE_TIMEOUT: Duration = Duration::from_secs(30);

/// A struct for writing video files.
pub struct VideoWriter {
    pipeline: gstreamer::Pipeline,
    appsrc: gstreamer_app::AppSrc,
    /// The pipeline's bus. Auto-flush is off, so messages posted while the pipeline goes to
    /// null (such as the file sink failing to close the file) can still be read.
    bus: gstreamer::Bus,
    fps: i32,
    format: ImageFormat,
    counter: u64,
    /// Whether `start` has succeeded since the last `close`.
    started: bool,
    /// Whether end-of-stream has already been read from the bus during this run.
    eos_received: bool,
    /// The first pipeline error seen by `write`, returned again by `close`.
    error: Option<gstreamer::glib::Error>,
    /// How long `close` waits for end-of-stream.
    close_timeout: Duration,
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
    /// Returns [`StreamCaptureError::InvalidConfig`] if `fps` is not positive or the size cannot
    /// be encoded (MP4 needs a width and height of at least 16, and H.264 4:2:0 needs them even),
    /// and [`StreamCaptureError::GStreamerError`] if GStreamer cannot be initialized or the
    /// pipeline cannot be built (e.g. `x264enc` is not installed).
    pub fn new(
        path: impl AsRef<Path>,
        codec: VideoCodec,
        format: ImageFormat,
        fps: i32,
        size: ImageSize,
    ) -> Result<Self, StreamCaptureError> {
        if size.width < 16
            || size.height < 16
            || !size.width.is_multiple_of(2)
            || !size.height.is_multiple_of(2)
        {
            return Err(StreamCaptureError::InvalidConfig(format!(
                "H.264 in MP4 needs an even width and height of at least 16, got {}x{}",
                size.width, size.height
            )));
        }

        let encoder = match codec {
            VideoCodec::H264 => "x264enc ! video/x-h264,profile=main ! h264parse",
        };

        // The output path is set as a property after parsing (see `set_location_property`).
        let pipeline_desc = format!(
            "appsrc name=src ! videoconvert ! video/x-raw,format=I420 ! {encoder} ! \
             mp4mux ! filesink name=filesink"
        );

        Self::from_pipeline_description(&pipeline_desc, path.as_ref(), format, fps, size)
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

        let bus = pipeline.bus().ok_or(StreamCaptureError::BusError)?;
        // By default the bus is flushed when the pipeline goes to null, which drops an error
        // the file sink posts in that same transition when it cannot close the file.
        pipeline.set_auto_flush_bus(false);
        // Log errors as soon as they are posted, even if `write` is not called again. `Pass`
        // keeps the message queued for `write` and `close`.
        bus.set_sync_handler(|_, msg| {
            if let gstreamer::MessageView::Error(err) = msg.view() {
                log::error!("{}", detailed_error(msg, err));
            }
            gstreamer::BusSyncReply::Pass
        });

        Ok(Self {
            pipeline,
            appsrc,
            bus,
            fps,
            format,
            counter: 0,
            started: false,
            eos_received: false,
            error: None,
            close_timeout: CLOSE_TIMEOUT,
        })
    }

    /// Start the video writer.
    ///
    /// Sets the pipeline to playing. Calling it again before [`Self::close`] has no effect.
    /// After a failed start the writer can be started again (e.g. once the output directory
    /// exists).
    ///
    /// # Returns
    ///
    /// `Ok(())` once the pipeline has been asked to start playing.
    ///
    /// # Errors
    ///
    /// Returns [`StreamCaptureError::GStreamerError`] with the pipeline's own error if it cannot
    /// start (e.g. the output file cannot be opened), or
    /// [`StreamCaptureError::SetPipelineStateError`] if it fails without posting one.
    pub fn start(&mut self) -> Result<(), StreamCaptureError> {
        if self.started {
            return Ok(());
        }
        self.error = None;
        self.eos_received = false;

        if let Err(state_err) = self.pipeline.set_state(gstreamer::State::Playing) {
            if let Err(e) = self.pipeline.set_state(gstreamer::State::Null) {
                log::warn!("failed to reset the pipeline after a failed start: {e}");
            }
            // Drain the bus, so no error of this attempt is left for the next run, and return
            // the elements' own cause (e.g. "Could not open file") instead of the bare
            // state-change error.
            let cause = self.read_bus();
            return Err(cause.map_or(state_err.into(), Into::into));
        }
        self.started = true;
        Ok(())
    }

    /// Close the video writer.
    ///
    /// Sends end-of-stream and waits until it reaches the file sink, so the muxer can finish the
    /// file, then sets the pipeline to null and checks what was posted while stopping (the file
    /// sink reports a failure to close the file only then).
    ///
    /// GStreamer keeps bus messages until they are read, so an error raised while writing (for
    /// example caps the encoder cannot accept) is still found here and returned, instead of
    /// leaving a silently broken file.
    ///
    /// The wait for end-of-stream is bounded (30 s). Setting the pipeline to null can still block
    /// if the file sink is stuck inside a blocking write (e.g. to a pipe whose reader stopped
    /// reading); GStreamer cannot interrupt that system call, so neither can this method or
    /// `Drop`.
    ///
    /// # Returns
    ///
    /// `Ok(())` once end-of-stream has reached the file sink and the pipeline has stopped
    /// without errors. If no frame was written, the file may still not be playable.
    ///
    /// # Errors
    ///
    /// Returns [`StreamCaptureError::PipelineNotRunning`] if the writer was not started (or is
    /// already closed), [`StreamCaptureError::GStreamerError`] with the first pipeline error or
    /// if end-of-stream did not reach the sink in time,
    /// [`StreamCaptureError::GstreamerFlowError`] if end-of-stream cannot be sent, and
    /// [`StreamCaptureError::SetPipelineStateError`] if the pipeline cannot be stopped. Once
    /// started, the pipeline is set to null in every case.
    pub fn close(&mut self) -> Result<(), StreamCaptureError> {
        if !std::mem::take(&mut self.started) {
            return Err(StreamCaptureError::PipelineNotRunning);
        }
        let finished = match self.error.take().or_else(|| self.read_bus()) {
            // The pipeline already failed, so end-of-stream would not reach the sink.
            Some(err) => Err(err.into()),
            None if self.eos_received => Ok(()),
            None => self.finish_stream(),
        };

        // Stop the pipeline even if finishing failed, then report the most useful error.
        let stopped = self.pipeline.set_state(gstreamer::State::Null);
        let stop_error = self.read_bus();
        finished?;
        if let Some(err) = stop_error {
            return Err(err.into());
        }
        stopped?;
        Ok(())
    }

    /// Sends end-of-stream and waits, for at most `close_timeout`, until it reaches the file sink
    /// or the pipeline posts an error.
    fn finish_stream(&mut self) -> Result<(), StreamCaptureError> {
        self.appsrc.end_of_stream()?;
        // `u64::MAX` nanoseconds is GStreamer's "no time" and `from_nseconds` panics on it, so
        // clamp to the largest valid time first.
        let nanos = u64::try_from(self.close_timeout.as_nanos())
            .unwrap_or(u64::MAX)
            .min(gstreamer::ClockTime::MAX.nseconds());
        let timeout = gstreamer::ClockTime::from_nseconds(nanos);
        let types = [gstreamer::MessageType::Eos, gstreamer::MessageType::Error];
        let Some(msg) = self.bus.timed_pop_filtered(timeout, &types) else {
            return Err(gstreamer::glib::Error::new(
                gstreamer::CoreError::Failed,
                &format!(
                    "end-of-stream did not reach the file sink within {:?}; the file may be \
                     incomplete",
                    self.close_timeout
                ),
            )
            .into());
        };
        match msg.view() {
            gstreamer::MessageView::Error(err) => Err(detailed_error(&msg, err).into()),
            _ => Ok(()),
        }
    }

    /// Reads every message queued on the bus, remembers whether end-of-stream was among them and
    /// returns the first error.
    fn read_bus(&mut self) -> Option<gstreamer::glib::Error> {
        let mut first_error = None;
        while let Some(msg) = self.bus.pop() {
            match msg.view() {
                gstreamer::MessageView::Eos(_) => self.eos_received = true,
                gstreamer::MessageView::Error(err) if first_error.is_none() => {
                    first_error = Some(detailed_error(&msg, err));
                }
                _ => {}
            }
        }
        first_error
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
    /// Returns [`StreamCaptureError::PipelineNotRunning`] if the writer was not started,
    /// [`StreamCaptureError::InvalidImageFormat`] if the image has the wrong number of channels,
    /// [`StreamCaptureError::GStreamerError`] if the pipeline has failed (this and every later
    /// frame is then rejected, and [`Self::close`] returns the same error),
    /// [`StreamCaptureError::GetBufferError`] if the frame buffer cannot be prepared, and
    /// [`StreamCaptureError::GstreamerFlowError`] if the pipeline no longer accepts frames
    /// (it is flushing or has reached end-of-stream).
    // TODO: explore supporting write_async
    pub fn write<const C: usize>(&mut self, img: &Image<u8, C>) -> Result<(), StreamCaptureError> {
        if !self.started {
            return Err(StreamCaptureError::PipelineNotRunning);
        }

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
            self.error = self.read_bus();
        }
        if let Some(err) = &self.error {
            return Err(err.clone().into());
        }

        self.appsrc.push_buffer(buffer)?;
        Ok(())
    }
}

/// The bus error with GStreamer's debug details (which name the failing element) added to the
/// message on one line, e.g. `Internal data stream error. (gstbasesrc.c(3187): gst_base_src_loop
/// (): /GstPipeline:pipeline0/GstAppSrc:src: streaming stopped, reason not-negotiated (-4))`. The
/// GStreamer error domain and code are kept.
fn detailed_error(
    msg: &gstreamer::Message,
    err: &gstreamer::message::Error,
) -> gstreamer::glib::Error {
    let base = err.error();
    let details = match err.debug() {
        Some(debug) => debug.replace('\n', " "),
        None => msg
            .src()
            .map(|s| s.path_string().to_string())
            .unwrap_or_default(),
    };
    let text = format!("{} ({details})", base.message());
    if let Some(code) = base.kind::<gstreamer::CoreError>() {
        gstreamer::glib::Error::new(code, &text)
    } else if let Some(code) = base.kind::<gstreamer::LibraryError>() {
        gstreamer::glib::Error::new(code, &text)
    } else if let Some(code) = base.kind::<gstreamer::ResourceError>() {
        gstreamer::glib::Error::new(code, &text)
    } else if let Some(code) = base.kind::<gstreamer::StreamError>() {
        gstreamer::glib::Error::new(code, &text)
    } else {
        gstreamer::glib::Error::new(gstreamer::CoreError::Failed, &text)
    }
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

    /// Grabs the oldest decoded frame as an RGB image.
    ///
    /// The reader must have been created with [`ImageFormat::Rgb8`]. As with
    /// [`StreamCapture::grab_rgb8`], the image is a read-only view of the GStreamer buffer when
    /// its rows are tightly packed and an owned copy when GStreamer padded them; call `.clone()`
    /// to get a writable image in either case.
    ///
    /// # Returns
    ///
    /// An Option containing the oldest decoded Image, or None if no frame is buffered yet.
    ///
    /// # Errors
    ///
    /// Returns [`VideoReaderError::StreamCaptureError`] wrapping
    /// [`StreamCaptureError::InvalidImageFormat`] if the reader was created with another format,
    /// or any other error of [`StreamCapture::grab_rgb8`].
    #[inline]
    pub fn grab_rgb8(&mut self) -> Result<Option<Image<u8, 3>>, VideoReaderError> {
        self.0
            .grab_rgb8()
            .map_err(VideoReaderError::StreamCaptureError)
    }

    /// Grabs the oldest decoded frame as a grayscale image.
    ///
    /// The reader must have been created with [`ImageFormat::Mono8`]. As with
    /// [`StreamCapture::grab_gray8`], tightly packed rows are borrowed read-only and padded
    /// rows are copied. Call `.clone()` for a writable image.
    ///
    /// # Arguments
    ///
    /// * `self` - The mutable video reader.
    ///
    /// # Returns
    ///
    /// The oldest decoded image, or `None` if no frame is buffered yet.
    ///
    /// # Errors
    ///
    /// Returns [`VideoReaderError::StreamCaptureError`] wrapping
    /// [`StreamCaptureError::InvalidImageFormat`] if the pipeline produces a different format,
    /// or any other error of [`StreamCapture::grab_gray8`].
    ///
    /// # Example
    ///
    /// ```no_run
    /// use kornia_io::gstreamer::video::{ImageFormat, VideoReader};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut reader = VideoReader::new("input.mp4", ImageFormat::Mono8)?;
    /// reader.start()?;
    /// if let Some(frame) = reader.grab_gray8()? {
    ///     println!("{}x{}", frame.width(), frame.height());
    /// }
    /// reader.close()?;
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn grab_gray8(&mut self) -> Result<Option<Image<u8, 1>>, VideoReaderError> {
        self.0
            .grab_gray8()
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
    use super::{ImageFormat, StreamCaptureError, VideoCodec, VideoReader, VideoWriter};
    use gstreamer::prelude::*;
    use kornia_image::{Image, ImageSize};
    use std::time::Duration;

    /// Raw RGB into QuickTime. Unlike Matroska, `qtmux` writes its index only at end-of-stream,
    /// so a file closed before end-of-stream reaches the muxer cannot be read back, and the
    /// frames round-trip without any colour conversion. Needs only gst-plugins-good.
    const RAW_MOV: &str =
        "appsrc name=src ! video/x-raw,format=RGB ! qtmux ! filesink name=filesink";

    const SIZE: ImageSize = ImageSize {
        width: 64,
        height: 48,
    };

    /// A frame whose every byte is `value`.
    fn frame(value: u8) -> Result<Image<u8, 3>, Box<dyn std::error::Error>> {
        Ok(Image::new(SIZE, vec![value; SIZE.width * SIZE.height * 3])?)
    }

    /// Writes one frame per value and closes the writer, failing if `close` does not return
    /// within 10 s.
    fn write_and_close(
        mut writer: VideoWriter,
        values: &[u8],
    ) -> Result<(), Box<dyn std::error::Error>> {
        for &v in values {
            writer.write(&frame(v)?)?;
        }
        let closed = finishes_within(Duration::from_secs(10), move || {
            writer.close().map_err(|e| e.to_string())
        });
        assert_eq!(closed, Some(Ok(())), "close did not return Ok within 10 s");
        Ok(())
    }

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
    /// after `timeout` (the thread is then left running). A panic in `f` is re-raised here, so
    /// it is not mistaken for a hang.
    fn finishes_within<T: Send + 'static>(
        timeout: Duration,
        f: impl FnOnce() -> T + Send + 'static,
    ) -> Option<T> {
        let (tx, rx) = std::sync::mpsc::channel();
        let handle = std::thread::spawn(move || {
            let _ = tx.send(f());
        });
        match rx.recv_timeout(timeout) {
            Ok(value) => Some(value),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => None,
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => match handle.join() {
                Err(panic) => std::panic::resume_unwind(panic),
                Ok(()) => None,
            },
        }
    }

    /// `close` must return after end-of-stream has reached the file sink, so every written
    /// frame ends up in the file. It used to set the pipeline to `Null` first, so the bus thread
    /// it then joined never saw end-of-stream and `close` could hang forever.
    #[test]
    fn video_writer_close_returns_and_keeps_every_frame() -> Result<(), Box<dyn std::error::Error>>
    {
        let tmp_dir = tempfile::tempdir()?;
        let file_path = tmp_dir.path().join("test.mov");
        let mut writer = VideoWriter::from_pipeline_description(
            RAW_MOV,
            &file_path,
            ImageFormat::Rgb8,
            30,
            SIZE,
        )?;
        // Errors posted while the pipeline goes to null (e.g. the file sink failing to close the
        // file) are only readable if the bus is not flushed in that transition.
        assert!(!writer.pipeline.is_auto_flush_bus());
        // `u64::MAX` nanoseconds is GStreamer's "no time" value; it must be clamped, not panic.
        writer.close_timeout = Duration::from_nanos(u64::MAX);
        // A second start must not change anything (it used to spawn a second bus thread that
        // could take the end-of-stream message and leave close waiting forever).
        writer.start()?;
        writer.start()?;
        write_and_close(writer, &[0, 40, 80, 120, 160])?;
        assert_eq!(frame_values(&file_path)?, [0, 40, 80, 120, 160]);
        Ok(())
    }

    #[test]
    fn video_reader_grabs_mono8_frames() -> Result<(), Box<dyn std::error::Error>> {
        let tmp_dir = tempfile::tempdir()?;
        let file_path = tmp_dir.path().join("gray-read.mov");
        let mut writer = VideoWriter::from_pipeline_description(
            RAW_MOV,
            &file_path,
            ImageFormat::Rgb8,
            30,
            SIZE,
        )?;
        writer.start()?;
        write_and_close(writer, &[80])?;

        let mut reader = VideoReader::new(&file_path, ImageFormat::Mono8)?;
        assert!(reader.grab_gray8()?.is_none());
        reader.start()?;
        let mut image = None;
        for _ in 0..200 {
            image = reader.grab_gray8()?;
            if image.is_some() {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        reader.close()?;
        let image = image.ok_or("no grayscale frame received")?;
        assert_eq!(image.size(), SIZE);
        assert_eq!(image.as_slice(), vec![80; SIZE.width * SIZE.height]);
        Ok(())
    }

    /// Reads `path` back to its end and returns the value of the first byte of every frame, so
    /// missing, duplicated or blank frames all show up; a reader error is returned as an error. Stops 1 s after the last frame (the
    /// reader cannot report end-of-stream) or after 10 s.
    fn frame_values(path: &std::path::Path) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let mut reader = VideoReader::new(path, ImageFormat::Rgb8)?;
        let bus = reader
            .0
            .pipeline
            .bus()
            .ok_or("reader pipeline has no bus")?;
        reader.start()?;
        let start = std::time::Instant::now();
        let mut last_frame = start;
        let mut values = Vec::new();
        while last_frame.elapsed() < Duration::from_secs(1)
            && start.elapsed() < Duration::from_secs(10)
        {
            // Report a reader failure (e.g. no H.264 decoder installed) instead of "0 frames".
            if let Some(msg) = bus.pop_filtered(&[gstreamer::MessageType::Error]) {
                if let gstreamer::MessageView::Error(err) = msg.view() {
                    return Err(super::detailed_error(&msg, err).into());
                }
            }
            match reader.grab_rgb8()? {
                Some(img) => {
                    values.push(img.as_slice()[0]);
                    last_frame = std::time::Instant::now();
                }
                None => std::thread::sleep(Duration::from_millis(10)),
            }
        }
        reader.close()?;
        Ok(values)
    }

    /// `write` and `close` before `start` must fail instead of reporting success for a file that
    /// is never written. The early `close` also used to send end-of-stream into the stopped
    /// `appsrc`, which then rejected every frame of the next run with `Pad is EOS`.
    #[test]
    fn video_writer_requires_start_and_survives_an_early_close(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let tmp_dir = tempfile::tempdir()?;
        let file_path = tmp_dir.path().join("test.mov");
        let mut writer = VideoWriter::from_pipeline_description(
            RAW_MOV,
            &file_path,
            ImageFormat::Rgb8,
            30,
            SIZE,
        )?;
        assert!(matches!(
            writer.write(&frame(0)?),
            Err(StreamCaptureError::PipelineNotRunning)
        ));
        assert!(matches!(
            writer.close(),
            Err(StreamCaptureError::PipelineNotRunning)
        ));
        writer.start()?;
        for v in [10, 20, 30] {
            writer.write(&frame(v)?)?;
        }
        writer.close()?;
        // A second close used to send end-of-stream into a stopped `appsrc` again and fail with
        // `Flushing`.
        assert!(matches!(
            writer.close(),
            Err(StreamCaptureError::PipelineNotRunning)
        ));
        assert_eq!(frame_values(&file_path)?, [10, 20, 30]);
        Ok(())
    }

    /// `close` must give up when end-of-stream never reaches the sink (here a `valve` drops it)
    /// instead of waiting forever. A sink blocked inside a `write` system call is a different
    /// case: stopping the pipeline then blocks too, and nothing can interrupt it.
    #[test]
    fn video_writer_close_gives_up_when_end_of_stream_never_arrives(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let tmp_dir = tempfile::tempdir()?;
        let mut writer = VideoWriter::from_pipeline_description(
            "appsrc name=src ! valve drop=true ! filesink name=filesink",
            &tmp_dir.path().join("test.raw"),
            ImageFormat::Rgb8,
            30,
            SIZE,
        )?;
        writer.close_timeout = Duration::from_millis(300);
        writer.start()?;
        writer.write(&frame(0)?)?;
        let closed = finishes_within(Duration::from_secs(10), move || match writer.close() {
            Err(StreamCaptureError::GStreamerError(err)) => err.to_string(),
            other => format!("unexpected: {other:?}"),
        })
        .ok_or("close did not return within 10 s")?;
        assert!(
            closed.contains("end-of-stream did not reach the file sink"),
            "{closed}"
        );
        Ok(())
    }

    /// A test-only pass-through element that posts an error while it goes from READY to NULL,
    /// the moment `filesink` reports that it could not close its file (a full disk or a dropped
    /// network mount, which a test cannot create).
    mod fail_on_stop {
        use gstreamer::glib;
        use gstreamer::prelude::*;
        use gstreamer::subclass::prelude::*;

        mod imp {
            use super::*;

            #[derive(Default)]
            pub struct FailOnStop;

            #[glib::object_subclass]
            impl ObjectSubclass for FailOnStop {
                const NAME: &'static str = "KorniaTestFailOnStop";
                type Type = super::FailOnStop;
                type ParentType = gstreamer::Bin;
            }

            impl ObjectImpl for FailOnStop {
                fn constructed(&self) {
                    self.parent_constructed();
                    let bin = self.obj();
                    let identity = gstreamer::ElementFactory::make("identity")
                        .build()
                        .expect("identity is a core element");
                    bin.add(&identity).expect("add identity");
                    for name in ["sink", "src"] {
                        let target = identity.static_pad(name).expect("identity pad");
                        let ghost = gstreamer::GhostPad::builder_with_target(&target)
                            .expect("ghost pad")
                            .name(name)
                            .build();
                        bin.add_pad(&ghost).expect("add ghost pad");
                    }
                }
            }

            impl GstObjectImpl for FailOnStop {}

            impl ElementImpl for FailOnStop {
                fn metadata() -> Option<&'static gstreamer::subclass::ElementMetadata> {
                    static METADATA: std::sync::OnceLock<gstreamer::subclass::ElementMetadata> =
                        std::sync::OnceLock::new();
                    Some(METADATA.get_or_init(|| {
                        gstreamer::subclass::ElementMetadata::new(
                            "Fail on stop",
                            "Generic",
                            "Posts an error when going from READY to NULL (tests only)",
                            "kornia",
                        )
                    }))
                }

                fn change_state(
                    &self,
                    transition: gstreamer::StateChange,
                ) -> Result<gstreamer::StateChangeSuccess, gstreamer::StateChangeError>
                {
                    if transition == gstreamer::StateChange::ReadyToNull {
                        gstreamer::element_imp_error!(
                            self,
                            gstreamer::ResourceError::Close,
                            ("Error closing file"),
                            ["simulated by a test"]
                        );
                    }
                    self.parent_change_state(transition)
                }
            }

            impl BinImpl for FailOnStop {}
        }

        glib::wrapper! {
            pub struct FailOnStop(ObjectSubclass<imp::FailOnStop>)
                @extends gstreamer::Bin, gstreamer::Element, gstreamer::Object;
        }

        /// Registers the element as `korniatestfailonstop`.
        pub fn register() -> Result<(), Box<dyn std::error::Error>> {
            gstreamer::init()?;
            gstreamer::Element::register(
                None,
                "korniatestfailonstop",
                gstreamer::Rank::NONE,
                FailOnStop::static_type(),
            )?;
            Ok(())
        }
    }

    /// An error posted while the pipeline goes to null (here a test element fails as `filesink`
    /// does when it cannot close the file) must be returned by `close`. The bus used to be
    /// flushed in that transition, so `close` returned `Ok` for an incomplete file.
    #[test]
    fn video_writer_close_returns_an_error_posted_while_stopping(
    ) -> Result<(), Box<dyn std::error::Error>> {
        fail_on_stop::register()?;
        let tmp_dir = tempfile::tempdir()?;
        let mut writer = VideoWriter::from_pipeline_description(
            "appsrc name=src ! video/x-raw,format=RGB ! qtmux ! korniatestfailonstop ! \
             filesink name=filesink",
            &tmp_dir.path().join("test.mov"),
            ImageFormat::Rgb8,
            30,
            SIZE,
        )?;
        writer.start()?;
        writer.write(&frame(0)?)?;
        let closed = finishes_within(Duration::from_secs(10), move || match writer.close() {
            Err(StreamCaptureError::GStreamerError(err)) => Ok(err.to_string()),
            other => Err(format!("{other:?}")),
        });
        match closed {
            Some(Ok(msg)) => assert!(msg.contains("Error closing file"), "{msg}"),
            other => panic!("close must return the error posted while stopping, got {other:?}"),
        }
        Ok(())
    }

    /// `new` must reject sizes the MP4/H.264 pipeline can never encode, instead of failing
    /// later with a generic error and an empty file.
    #[test]
    fn video_writer_new_rejects_sizes_mp4_cannot_encode() -> Result<(), Box<dyn std::error::Error>>
    {
        let tmp_dir = tempfile::tempdir()?;
        let path = tmp_dir.path().join("test.mp4");
        for (width, height) in [(15, 16), (16, 8), (17, 16), (16, 17), (0, 0)] {
            let size = ImageSize { width, height };
            assert!(
                matches!(
                    VideoWriter::new(&path, VideoCodec::H264, ImageFormat::Rgb8, 30, size),
                    Err(StreamCaptureError::InvalidConfig(_))
                ),
                "{width}x{height} must be rejected"
            );
        }
        Ok(())
    }

    /// A failed `start` (here the output directory does not exist yet) must return the real
    /// cause, and must not leave its errors on the bus where they used to fail every frame of the
    /// next, healthy run.
    #[test]
    fn video_writer_failed_start_does_not_poison_the_next_run(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let tmp_dir = tempfile::tempdir()?;
        let dir = tmp_dir.path().join("not-created-yet");
        let file_path = dir.join("test.mov");
        let mut writer = VideoWriter::from_pipeline_description(
            RAW_MOV,
            &file_path,
            ImageFormat::Rgb8,
            30,
            SIZE,
        )?;
        match writer.start() {
            Err(StreamCaptureError::GStreamerError(err)) => assert!(
                err.to_string().contains("Could not open file"),
                "start must return the file sink's error, got: {err}"
            ),
            other => panic!("start must fail with the pipeline error, got {other:?}"),
        }
        // The failed attempt must leave the pipeline stopped, ready to start again.
        assert_eq!(writer.pipeline.current_state(), gstreamer::State::Null);
        std::fs::create_dir(&dir)?;
        writer.start()?;
        write_and_close(writer, &[50, 60])?;
        assert_eq!(frame_values(&file_path)?, [50, 60]);
        Ok(())
    }

    /// An end-of-stream already on the bus when `write` checks it for errors must not be lost.
    /// `write` used to discard every non-error message, and `close` then waited forever for an
    /// end-of-stream that had already arrived.
    #[test]
    fn video_writer_close_returns_after_write_saw_end_of_stream(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let tmp_dir = tempfile::tempdir()?;
        let file_path = tmp_dir.path().join("test.mov");
        let mut writer = VideoWriter::from_pipeline_description(
            RAW_MOV,
            &file_path,
            ImageFormat::Rgb8,
            30,
            SIZE,
        )?;
        writer.start()?;
        writer.write(&frame(70)?)?;
        // As if the stream had ended on its own: end-of-stream reaches the sink and the bus. Wait
        // for that message, then put it back for `write` to find.
        writer.appsrc.end_of_stream()?;
        let eos = writer
            .bus
            .timed_pop_filtered(
                gstreamer::ClockTime::from_seconds(5),
                &[gstreamer::MessageType::Eos],
            )
            .ok_or("end-of-stream did not reach the bus")?;
        writer.bus.post(eos)?;
        // `appsrc` is at end-of-stream, so this frame is rejected; what matters is that `write`
        // reads the bus here and must keep the end-of-stream it finds.
        assert!(matches!(
            writer.write(&frame(80)?),
            Err(StreamCaptureError::GstreamerFlowError(_))
        ));
        let closed = finishes_within(Duration::from_secs(10), move || {
            writer.close().map_err(|e| e.to_string())
        });
        assert_eq!(closed, Some(Ok(())), "close did not return Ok within 10 s");
        assert_eq!(frame_values(&file_path)?, [70]);
        Ok(())
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
                Err(StreamCaptureError::GStreamerError(_))
            ) {
                write_error = true;
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(write_error, "write must report the pipeline error");
        let closed = finishes_within(Duration::from_secs(10), move || match writer.close() {
            Err(StreamCaptureError::GStreamerError(err)) => Ok(err.to_string()),
            other => Err(format!("{other:?}")),
        });
        // The error carries GStreamer's debug details, not just "Internal data stream error."
        match closed {
            Some(Ok(msg)) => assert!(msg.contains("not-negotiated"), "{msg}"),
            other => panic!("close must return the pipeline error within 10 s, got {other:?}"),
        }

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
        let closed = finishes_within(Duration::from_secs(10), move || {
            matches!(writer.close(), Err(StreamCaptureError::GStreamerError(_)))
        });
        assert_eq!(
            closed,
            Some(true),
            "close must find the pipeline error while waiting"
        );
        Ok(())
    }

    #[ignore = "needs x264enc and h264parse (gst-plugins-ugly/-bad), which CI does not install"]
    #[test]
    fn video_writer_rgb8u() -> Result<(), Box<dyn std::error::Error>> {
        let tmp_dir = tempfile::tempdir()?;
        std::fs::create_dir_all(tmp_dir.path())?;

        let file_path = tmp_dir.path().join("test.mp4");

        // The smallest size the pipeline accepts: mp4mux needs at least 16x16.
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
            frame_values(&file_path)?.len(),
            1,
            "frames read back from {file_path:?}"
        );

        Ok(())
    }

    #[ignore = "needs x264enc and h264parse (gst-plugins-ugly/-bad), which CI does not install"]
    #[test]
    fn video_writer_mono8u() -> Result<(), Box<dyn std::error::Error>> {
        let tmp_dir = tempfile::tempdir()?;
        std::fs::create_dir_all(tmp_dir.path())?;

        let file_path = tmp_dir.path().join("test.mp4");

        // The smallest size the pipeline accepts: mp4mux needs at least 16x16.
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
            frame_values(&file_path)?.len(),
            1,
            "frames read back from {file_path:?}"
        );

        Ok(())
    }
}
