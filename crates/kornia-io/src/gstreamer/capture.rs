use super::{image_from_gst_buffer, video::detailed_error};
use crate::stream::error::StreamCaptureError;
use gstreamer::prelude::*;
use kornia_image::Image;
use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

#[derive(Clone, Copy)]
enum GrabFormat {
    Rgb8,
    Gray8,
}

impl From<GrabFormat> for gstreamer_video::VideoFormat {
    fn from(format: GrabFormat) -> Self {
        match format {
            GrabFormat::Rgb8 => Self::Rgb,
            GrabFormat::Gray8 => Self::Gray8,
        }
    }
}

impl std::fmt::Display for GrabFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Rgb8 => "grab_rgb8",
            Self::Gray8 => "grab_gray8",
        })
    }
}

/// The terminal state a capture pipeline reports on its bus.
#[derive(Default)]
struct StreamStatus {
    eos: EosStatus,
    previous_eos: Option<EosStatus>,
    error: Option<gstreamer::glib::Error>,
}

/// EOS belongs to the segment selected by the last seek, not just to the pipeline.
#[derive(Clone, Copy, Default)]
struct EosStatus {
    seqnum: Option<gstreamer::Seqnum>,
    received: bool,
}

impl EosStatus {
    fn record(&mut self, seqnum: gstreamer::Seqnum) {
        if self.seqnum.is_none_or(|expected| expected == seqnum) {
            self.received = true;
        }
    }
}

/// Image and metadata extracted from one sample before terminal-state classification.
struct CapturedFrame<const C: usize> {
    image: Image<u8, C>,
    pts: Option<Duration>,
}

/// The outcome of [`StreamCapture::try_grab_rgb8`] and [`StreamCapture::try_grab_gray8`].
///
/// Unlike [`StreamCapture::grab_rgb8`] and [`StreamCapture::grab_gray8`], which report `None`
/// for a frame that has not arrived yet, the end of the stream and a pipeline error alike, this
/// enum keeps the three cases apart.
pub enum Grab<const C: usize> {
    /// A decoded frame and the timestamp carried by the same GStreamer buffer.
    Frame {
        /// The decoded image.
        image: Image<u8, C>,
        /// Buffer presentation timestamp, or `None` when absent.
        /// It may jump backwards after a seek/reset; it is not a wall-clock timestamp.
        pts: Option<Duration>,
    },
    /// No frame is buffered yet; the stream is still running.
    Pending,
    /// The stream reached end-of-stream and every decoded frame has been grabbed.
    Eos,
}

/// A enum representing the state of [VideoReader] pipeline.
///
/// For more info, refer to <https://gstreamer.freedesktop.org/documentation/additional/design/states.html?gi-language=c>
pub enum StreamerState {
    /// This is the initial state of a pipeline.
    Null,
    /// The element should be prepared to go to [State::Paused]
    Ready,
    /// The video is paused.
    Paused,
    /// The video is playing.
    Playing,
}

impl From<gstreamer::State> for StreamerState {
    fn from(value: gstreamer::State) -> Self {
        match value {
            gstreamer::State::VoidPending => StreamerState::Null,
            gstreamer::State::Null => StreamerState::Null,
            gstreamer::State::Ready => StreamerState::Ready,
            gstreamer::State::Paused => StreamerState::Paused,
            gstreamer::State::Playing => StreamerState::Playing,
        }
    }
}

/// Frames retained when a custom pipeline leaves max-buffers unbounded.
const DEFAULT_MAX_BUFFERS: u32 = 5;
const CLOSE_EOS_TIMEOUT: Duration = Duration::from_secs(3);

/// Represents a stream capture pipeline using GStreamer.
pub struct StreamCapture {
    pub(crate) pipeline: gstreamer::Pipeline,
    /// The pipeline's bus. Auto-flush is off, so messages posted while the pipeline goes to null
    /// can still be read.
    bus: gstreamer::Bus,
    appsink: gstreamer_app::AppSink,
    /// End-of-stream and errors seen on the bus, updated by its sync handler.
    status: Arc<Mutex<StreamStatus>>,
    /// Serialize start, close and seeks without blocking the bus handler on the status lock.
    lifecycle_lock: Mutex<()>,
}

impl StreamCapture {
    /// Creates a new StreamCapture instance with the given pipeline description.
    ///
    /// Samples are pulled from GStreamer's appsink queue. A zero `max-buffers` is replaced
    /// by five; explicit nonzero bounds and the description's drop/sync settings are kept.
    /// By default the queue applies backpressure. Use `max-buffers=1 drop=true` for live
    /// latest-frame delivery. See <https://gstreamer.freedesktop.org/documentation/app/appsink.html>.
    ///
    /// # Arguments
    ///
    /// * `pipeline_desc` - A string describing the GStreamer pipeline.
    ///
    /// # Returns
    ///
    /// A Result containing the StreamCapture instance or a StreamCaptureError.
    ///
    /// # Errors
    ///
    /// Returns an error if GStreamer initialization or pipeline construction fails, the named
    /// appsink or bus is unavailable.
    pub fn new(pipeline_desc: &str) -> Result<Self, StreamCaptureError> {
        if !gstreamer::INITIALIZED.load(std::sync::atomic::Ordering::Relaxed) {
            gstreamer::init()?;
        }

        let pipeline = gstreamer::parse::launch(pipeline_desc)?
            .dynamic_cast::<gstreamer::Pipeline>()
            .map_err(StreamCaptureError::DowncastPipelineError)?;

        let appsink = pipeline
            .by_name("sink")
            .ok_or_else(|| StreamCaptureError::GetElementByNameError)?
            .dynamic_cast::<gstreamer_app::AppSink>()
            .map_err(StreamCaptureError::DowncastPipelineError)?;

        if appsink.max_buffers() == 0 {
            appsink.set_max_buffers(DEFAULT_MAX_BUFFERS);
        }
        // EOS must be observable without a consumer draining the remaining samples.
        appsink.set_wait_on_eos(false);
        let status = Arc::new(Mutex::new(StreamStatus::default()));

        let bus = pipeline.bus().ok_or(StreamCaptureError::BusError)?;
        // By default the bus is flushed when the pipeline goes to null, which drops an error
        // posted in that same transition. `Pass` keeps the message queued for `drain_bus`.
        pipeline.set_auto_flush_bus(false);
        bus.set_sync_handler({
            let status = status.clone();
            move |_bus, msg| {
                if let Ok(mut status) = status.lock() {
                    match msg.view() {
                        gstreamer::MessageView::Eos(_) => {
                            status.eos.record(msg.seqnum());
                            // Preserve old-segment EOS received during a rejected seek.
                            if status.eos.seqnum != Some(msg.seqnum()) {
                                if let Some(previous) = &mut status.previous_eos {
                                    previous.record(msg.seqnum());
                                }
                            }
                        }
                        gstreamer::MessageView::Error(err) => {
                            let error = detailed_error(msg, err);
                            log::error!("{error}");
                            status.error.get_or_insert(error);
                        }
                        _ => {}
                    }
                }
                gstreamer::BusSyncReply::Pass
            }
        });

        Ok(Self {
            pipeline,
            bus,
            appsink,
            status,
            lifecycle_lock: Mutex::new(()),
        })
    }

    /// Gets the negotiated framerate from the appsink sink-pad caps.
    ///
    /// # Arguments
    ///
    /// * `self` - The stream capture.
    ///
    /// # Returns
    ///
    /// `None` before negotiation or after closing, `Some(0.0)` for an unknown/variable rate,
    /// otherwise the current negotiated FPS. Queued frames may predate a caps change.
    ///
    /// # Errors
    ///
    /// This method does not return errors.
    pub fn get_fps(&self) -> Option<f64> {
        let caps = self.appsink.static_pad("sink")?.current_caps()?;
        let fps = caps
            .structure(0)
            .and_then(|s| s.get::<gstreamer::Fraction>("framerate").ok())
            .unwrap_or_else(|| gstreamer::Fraction::new(0, 1));
        Some(fps.numer() as f64 / fps.denom() as f64)
    }

    /// Sets whether the appsink synchronizes sample delivery with the pipeline clock.
    ///
    /// # Arguments
    ///
    /// * `sync` - Whether to synchronize with the pipeline clock.
    ///
    /// # Returns
    ///
    /// No value is returned.
    ///
    /// # Errors
    ///
    /// This method does not return errors.
    pub(crate) fn set_sync(&self, sync: bool) {
        self.appsink.set_property("sync", sync);
    }

    /// Sets the bounded appsink capacity while the pipeline is stopped.
    pub(crate) fn set_max_buffers(&mut self, max_buffers: u32) -> Result<(), StreamCaptureError> {
        if max_buffers == 0 {
            return Err(StreamCaptureError::InvalidConfig(
                "max_buffers must be greater than zero".into(),
            ));
        }
        let (_, current, pending) = self.pipeline.state(gstreamer::ClockTime::ZERO);
        if current != gstreamer::State::Null || pending != gstreamer::State::VoidPending {
            return Err(StreamCaptureError::InvalidConfig(
                "max_buffers must be configured while the pipeline is stopped".into(),
            ));
        }
        self.appsink.set_max_buffers(max_buffers);
        Ok(())
    }

    /// Gets the current state of the stream pipeline
    pub fn get_state(&self) -> StreamerState {
        self.pipeline.current_state().into()
    }

    /// Returns whether the stream has reached end-of-stream.
    ///
    /// End-of-stream is read from the pipeline's bus, so it does not depend on the position
    /// reaching the duration or on a timeout. Frames decoded before it are still returned by
    /// [`Self::try_grab_rgb8`] and [`Self::try_grab_gray8`], so this can be `true` while such
    /// frames remain buffered.
    ///
    /// # Arguments
    ///
    /// * `self` - The stream capture.
    ///
    /// # Returns
    ///
    /// `true` once the pipeline has posted end-of-stream; frames decoded before it may still
    /// remain buffered for the `try_grab_*` methods.
    pub fn is_eos(&self) -> bool {
        self.drain_bus();
        self.status.lock().is_ok_and(|status| status.eos.received)
    }

    /// Returns the first error the pipeline posted on its bus, if any.
    ///
    /// A pipeline that cancels its own stream (an undecodable file, a missing decoder, failed
    /// negotiation) sets this instead of silently delivering no more frames.
    ///
    /// # Arguments
    ///
    /// * `self` - The stream capture.
    ///
    /// # Returns
    ///
    /// The GStreamer error, with the failing element and its debug details, or `None` if the
    /// pipeline has not failed.
    pub fn stream_error(&self) -> Option<gstreamer::glib::Error> {
        self.drain_bus();
        self.status
            .lock()
            .ok()
            .and_then(|status| status.error.clone())
    }

    /// Starts the stream capture pipeline and processes messages on the bus.
    pub fn start(&self) -> Result<(), StreamCaptureError> {
        let _lifecycle = self
            .lifecycle_lock
            .lock()
            .map_err(|_| StreamCaptureError::MutexPoisonError)?;
        self.drain_bus();
        // Starting a stopped pipeline begins a fresh run, so its previous terminal state is
        // discarded. Resuming a PLAYING/PAUSED one keeps it: a pipeline that already reached
        // end-of-stream stays there until it is seeked, and forgetting that would report
        // `Pending` forever instead of `Eos`.
        if matches!(
            self.pipeline.current_state(),
            gstreamer::State::Null | gstreamer::State::Ready
        ) {
            *self
                .status
                .lock()
                .map_err(|_| StreamCaptureError::MutexPoisonError)? = StreamStatus::default();
        }
        self.pipeline.set_state(gstreamer::State::Playing)?;
        Ok(())
    }

    /// Grabs the oldest captured frame as an RGB image.
    ///
    /// NOTE: when GStreamer delivers tightly packed rows (`width * 3` divisible by 4, or a
    /// producer-set layout without padding), the image is a read-only view that borrows the
    /// GStreamer buffer without copying, and writing to it (e.g. `as_slice_mut`) panics. Padded
    /// rows are copied into an owned image. Call `.clone()` to get an owned, writable copy in
    /// either case.
    ///
    /// # Returns
    ///
    /// An Option containing the oldest captured Image, or None if no frame is buffered yet.
    ///
    /// `None` cannot distinguish "no frame yet" from end-of-stream or a pipeline error; use
    /// [`Self::try_grab_rgb8`] or [`Self::is_eos`]/[`Self::stream_error`] to tell them apart.
    ///
    /// # Errors
    ///
    /// Returns [`StreamCaptureError::GetCapsError`] if the sample caps cannot be parsed as video
    /// caps (e.g. audio or arbitrary `application/*` caps),
    /// [`StreamCaptureError::InvalidImageFormat`] if the frames are not `RGB` (e.g. `BGR`,
    /// `RGBx`, or an encoded format such as `image/jpeg`) or their size or layout is invalid,
    /// [`StreamCaptureError::BufferSizeMismatch`] if the buffer is too small for the frame,
    /// [`StreamCaptureError::GetBufferError`] if the sample has no buffer or cannot be mapped.
    pub fn grab_rgb8(&mut self) -> Result<Option<Image<u8, 3>>, StreamCaptureError> {
        Ok(self
            .grab_image::<3>(GrabFormat::Rgb8)?
            .map(|frame| frame.image))
    }

    /// Grabs the oldest captured frame as a grayscale image.
    ///
    /// Tightly packed rows are borrowed read-only from the GStreamer buffer. Padded rows are
    /// copied into an owned image. Call `.clone()` for a writable image in either case.
    ///
    /// # Arguments
    ///
    /// * `self` - The mutable stream capture.
    ///
    /// # Returns
    ///
    /// The oldest captured image, or `None` if no frame is buffered yet.
    ///
    /// `None` cannot distinguish "no frame yet" from end-of-stream or a pipeline error; use
    /// [`Self::try_grab_gray8`] or [`Self::is_eos`]/[`Self::stream_error`] to tell them apart.
    ///
    /// # Errors
    ///
    /// Returns [`StreamCaptureError::InvalidImageFormat`] if the frame is not `GRAY8` or its
    /// layout is invalid. Other caps, mapping, buffer size, and lock errors are returned as by
    /// [`Self::grab_rgb8`].
    ///
    /// # Example
    ///
    /// ```no_run
    /// use kornia_io::gstreamer::StreamCapture;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let pipeline = "videotestsrc ! video/x-raw,format=GRAY8 ! appsink name=sink sync=false";
    /// let mut capture = StreamCapture::new(pipeline)?;
    /// capture.start()?;
    /// if let Some(frame) = capture.grab_gray8()? {
    ///     println!("{}x{}", frame.width(), frame.height());
    /// }
    /// capture.close()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn grab_gray8(&mut self) -> Result<Option<Image<u8, 1>>, StreamCaptureError> {
        Ok(self
            .grab_image::<1>(GrabFormat::Gray8)?
            .map(|frame| frame.image))
    }

    /// Pulls a queued sample without waiting and maps the image and its own PTS.
    fn grab_image<const C: usize>(
        &mut self,
        grab_format: GrabFormat,
    ) -> Result<Option<CapturedFrame<C>>, StreamCaptureError> {
        self.drain_bus();
        let Some(sample) = self.appsink.try_pull_sample(gstreamer::ClockTime::ZERO) else {
            return Ok(None);
        };
        let caps = sample.caps().ok_or_else(|| {
            StreamCaptureError::GetCapsError("Failed to get the caps".to_string())
        })?;
        let info = gstreamer_video::VideoInfo::from_caps(caps)
            .map_err(|e| StreamCaptureError::GetCapsError(e.to_string()))?;
        let expected_format: gstreamer_video::VideoFormat = grab_format.into();
        if info.format() != expected_format {
            return Err(StreamCaptureError::InvalidImageFormat(format!(
                "{grab_format} needs {expected_format} frames, but the pipeline produces {} frames",
                info.format()
            )));
        }

        // Zero-copy when the rows are tightly packed; otherwise the padded rows are copied
        // into a packed image. See `image_from_gst_buffer`.
        let buffer = sample
            .buffer_owned()
            .ok_or(StreamCaptureError::GetBufferError)?;
        let pts = buffer
            .pts()
            .map(|time| Duration::from_nanos(time.nseconds()));
        let image = image_from_gst_buffer(buffer, &info)?;

        Ok(Some(CapturedFrame { image, pts }))
    }

    /// Tries to grab the oldest captured frame as an RGB image, distinguishing a frame from
    /// "no frame yet", end-of-stream and a pipeline error.
    ///
    /// Grabs do not wait. Samples already queued before pausing remain pullable.
    ///
    /// Frames decoded before end-of-stream or before an error are returned first; the terminal
    /// state is reported once the buffer is empty. A caller can therefore loop on this method
    /// without any timeout: `Grab::Frame` continues, `Grab::Eos` stops, `Grab::Pending` backs
    /// off, and a failed pipeline is returned as `Err`.
    ///
    /// # Arguments
    ///
    /// * `self` - The mutable stream capture.
    ///
    /// # Returns
    ///
    /// [`Grab::Frame`] with the oldest buffered frame, [`Grab::Pending`] when none is buffered
    /// yet, or [`Grab::Eos`] once the stream has ended and no frame is left.
    ///
    /// # Errors
    ///
    /// Returns [`StreamCaptureError::StreamError`] if the pipeline posted an error on its bus
    /// (e.g. a file that cannot be decoded), [`StreamCaptureError::MutexPoisonError`] if the
    /// status lock is poisoned, and any error of [`Self::grab_rgb8`] for the frame itself.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use kornia_io::gstreamer::{Grab, StreamCapture};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut capture = StreamCapture::new(
    ///     "videotestsrc num-buffers=2 ! video/x-raw,format=RGB ! appsink name=sink sync=false",
    /// )?;
    /// capture.start()?;
    /// loop {
    ///     match capture.try_grab_rgb8()? {
    ///         Grab::Frame { image: frame, .. } => println!("{}x{}", frame.width(), frame.height()),
    ///         Grab::Eos => break,
    ///         Grab::Pending => std::thread::sleep(std::time::Duration::from_millis(1)),
    ///     }
    /// }
    /// capture.close()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn try_grab_rgb8(&mut self) -> Result<Grab<3>, StreamCaptureError> {
        self.try_grab_image::<3>(GrabFormat::Rgb8)
    }

    /// Tries to grab the oldest captured frame as a grayscale image, distinguishing a frame from
    /// "no frame yet", end-of-stream and a pipeline error.
    ///
    /// See [`Self::try_grab_rgb8`] for the loop contract.
    ///
    /// # Arguments
    ///
    /// * `self` - The mutable stream capture.
    ///
    /// # Returns
    ///
    /// [`Grab::Frame`] with the oldest buffered frame, [`Grab::Pending`] when none is buffered
    /// yet, or [`Grab::Eos`] once the stream has ended and no frame is left.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::try_grab_rgb8`], with [`Self::grab_gray8`] for the frame itself.
    pub fn try_grab_gray8(&mut self) -> Result<Grab<1>, StreamCaptureError> {
        self.try_grab_image::<1>(GrabFormat::Gray8)
    }

    /// Grabs a frame or classifies the stream as pending, ended or failed. When a terminal state
    /// is observed, the queue is re-checked once so a frame that arrived in between is returned.
    fn try_grab_image<const C: usize>(
        &mut self,
        grab_format: GrabFormat,
    ) -> Result<Grab<C>, StreamCaptureError> {
        if let Some(CapturedFrame { image, pts }) = self.grab_image::<C>(grab_format)? {
            return Ok(Grab::Frame { image, pts });
        }

        let (error, eos) = {
            let status = self
                .status
                .lock()
                .map_err(|_| StreamCaptureError::MutexPoisonError)?;
            (status.error.clone(), status.eos.received)
        };
        // The final frame can land between the empty check above and the terminal state below:
        // take one more look before reporting the end.
        if error.is_some() || eos {
            if let Some(CapturedFrame { image, pts }) = self.grab_image::<C>(grab_format)? {
                return Ok(Grab::Frame { image, pts });
            }
        }
        if let Some(error) = error {
            return Err(StreamCaptureError::StreamError(error));
        }
        if eos {
            return Ok(Grab::Eos);
        }
        Ok(Grab::Pending)
    }

    /// Closes capture and lets any recording branches finish writing.
    ///
    /// An active stream is sent EOS with appsink backpressure and clock pacing disabled.
    /// The pipeline is stopped even if it rejects EOS or does not finish within three seconds.
    ///
    /// # Arguments
    ///
    /// * `self` - The stream capture.
    ///
    /// # Returns
    ///
    /// `Ok(())` once the pipeline is stopped.
    ///
    /// # Errors
    ///
    /// Returns an error if EOS is rejected, finalization times out, a state change fails,
    /// or the lifecycle lock is poisoned.
    pub fn close(&self) -> Result<(), StreamCaptureError> {
        let _lifecycle = self
            .lifecycle_lock
            .lock()
            .map_err(|_| StreamCaptureError::MutexPoisonError)?;
        let (_, current, pending) = self.pipeline.state(gstreamer::ClockTime::ZERO);
        if current == gstreamer::State::Null && pending == gstreamer::State::VoidPending {
            return Ok(());
        }

        let drop = self.appsink.is_drop();
        let sync = self.appsink.property::<bool>("sync");
        let finished = if !self.is_eos() && self.stream_error().is_none() {
            self.appsink.set_drop(true);
            self.appsink.set_sync(false);
            // Wake any producer already waiting for room in the queue.
            let _ = self.appsink.try_pull_sample(gstreamer::ClockTime::ZERO);
            self.finish_stream(current, pending)
        } else {
            Ok(())
        };

        let stopped = self.pipeline.set_state(gstreamer::State::Null);
        if stopped.is_ok() {
            self.appsink.set_drop(drop);
            self.appsink.set_sync(sync);
        }
        self.drain_bus();
        finished?;
        stopped?;
        Ok(())
    }

    fn finish_stream(
        &self,
        current: gstreamer::State,
        pending: gstreamer::State,
    ) -> Result<(), StreamCaptureError> {
        // Pipeline EOS is only posted in Playing, including if close follows pause.
        if pending != gstreamer::State::Playing
            && (current != gstreamer::State::Playing || pending == gstreamer::State::Paused)
        {
            self.pipeline.set_state(gstreamer::State::Playing)?;
        }

        let deadline = Instant::now() + CLOSE_EOS_TIMEOUT;
        let (sent, result) = std::sync::mpsc::sync_channel(1);
        // A custom branch can still delay a serialized event. Send it off-thread so
        // close can transition to NULL and release the branch if it does not finish.
        self.pipeline.call_async(move |pipeline| {
            let _ = sent.send(pipeline.send_event(gstreamer::event::Eos::new()));
        });
        match result.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
            Ok(true) => {}
            Ok(false) if self.is_eos() => return Ok(()),
            Ok(false) => return Err(StreamCaptureError::SendEosError),
            Err(_) => return Err(Self::eos_timeout()),
        }

        let remaining = deadline.saturating_duration_since(Instant::now());
        let timeout = gstreamer::ClockTime::from_nseconds(remaining.as_nanos() as u64);
        let types = [gstreamer::MessageType::Eos, gstreamer::MessageType::Error];
        // is_eos drains the bus, so check cached EOS/error only after the wait.
        match self.bus.timed_pop_filtered(timeout, &types) {
            Some(msg) => match msg.view() {
                gstreamer::MessageView::Eos(_) => Ok(()),
                gstreamer::MessageView::Error(err) => Err(detailed_error(&msg, err).into()),
                _ => unreachable!("only EOS and errors were requested"),
            },
            None if self.is_eos() => Ok(()),
            None => Err(self
                .stream_error()
                .map_or_else(Self::eos_timeout, Into::into)),
        }
    }

    fn eos_timeout() -> StreamCaptureError {
        gstreamer::glib::Error::new(
            gstreamer::CoreError::Failed,
            "capture EOS did not reach all sinks within three seconds; recording may be incomplete",
        )
        .into()
    }

    /// Sends a seek while tracking the EOS sequence number of its segment.
    ///
    /// GStreamer propagates the seek seqnum to the segment's EOS message:
    /// <https://gstreamer.freedesktop.org/documentation/additional/design/seqnums.html>.
    /// The status lock must not be held across send_event: its sync bus handler can run there.
    ///
    /// # Arguments
    ///
    /// * `rate` - Playback rate.
    /// * `flags` - Seek flags; only `FLUSH` clears the previous EOS state.
    /// * `position` - Start position in stream time.
    ///
    /// # Returns
    ///
    /// `Ok(())` when the pipeline accepts the seek.
    ///
    /// # Errors
    ///
    /// Returns a seek error if rejected, an invalid-configuration error for a non-flushing
    /// seek on a non-dropping queue, or a capture error if a lock is poisoned.
    pub(crate) fn seek(
        &self,
        rate: f64,
        flags: gstreamer::SeekFlags,
        position: gstreamer::ClockTime,
    ) -> Result<(), super::error::VideoReaderError> {
        // Occupancy can change before send_event; reject without inspecting or draining the queue.
        if !flags.contains(gstreamer::SeekFlags::FLUSH) && !self.appsink.is_drop() {
            return Err(StreamCaptureError::InvalidConfig(
                "non-flushing seeks on a non-dropping queue can block; use SeekFlags::FLUSH".into(),
            )
            .into());
        }
        let _lifecycle = self
            .lifecycle_lock
            .lock()
            .map_err(|_| StreamCaptureError::MutexPoisonError)?;
        let event = gstreamer::event::Seek::new(
            rate,
            flags,
            gstreamer::SeekType::Set,
            position,
            gstreamer::SeekType::None,
            gstreamer::ClockTime::NONE,
        );
        {
            let mut status = self
                .status
                .lock()
                .map_err(|_| StreamCaptureError::MutexPoisonError)?;
            status.previous_eos = Some(status.eos);
            status.eos = EosStatus {
                seqnum: Some(event.seqnum()),
                received: !flags.contains(gstreamer::SeekFlags::FLUSH) && status.eos.received,
            };
        }
        let accepted = self.pipeline.send_event(event);
        let mut status = self
            .status
            .lock()
            .map_err(|_| StreamCaptureError::MutexPoisonError)?;
        if let Some(previous) = status.previous_eos.take() {
            if !accepted {
                status.eos = previous;
            } else if !flags.contains(gstreamer::SeekFlags::FLUSH) {
                status.eos.received |= previous.received;
            }
        }
        if accepted {
            Ok(())
        } else {
            Err(super::error::VideoReaderError::SeekError)
        }
    }

    /// Reads every message queued on the bus, so the queue cannot grow without bound. The sync
    /// handler records end-of-stream and errors as they are posted, so the messages themselves
    /// can be discarded.
    fn drain_bus(&self) {
        while self.bus.pop().is_some() {}
    }
}

impl Drop for StreamCapture {
    /// Ensures that the StreamCapture is properly closed when dropped.
    fn drop(&mut self) {
        if let Err(e) = self.close() {
            log::warn!("Failed to close stream safely on drop: {:?}", e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{sync::mpsc, time::Instant};

    const DEADLINE: Duration = Duration::from_secs(5);
    const CAPTURE_PIPELINE: &str = "appsrc name=src format=time ! video/x-raw,format=RGB,width=4,height=1,framerate=1/1 ! appsink name=sink sync=false async=false";

    /// Notify only after appsink has enqueued each sample; never consume it in the callback.
    fn queue_notifications(capture: &StreamCapture) -> mpsc::Receiver<()> {
        let (tx, rx) = mpsc::channel();
        capture.appsink.set_callbacks(
            gstreamer_app::AppSinkCallbacks::builder()
                .new_sample(move |_| {
                    let _ = tx.send(());
                    Ok(gstreamer::FlowSuccess::Ok)
                })
                .build(),
        );
        rx
    }

    fn source(
        capture: &StreamCapture,
    ) -> Result<gstreamer_app::AppSrc, Box<dyn std::error::Error>> {
        Ok(capture
            .pipeline
            .by_name("src")
            .ok_or("missing source")?
            .downcast::<gstreamer_app::AppSrc>()
            .map_err(|_| "not appsrc")?)
    }

    fn push_frame(source: &gstreamer_app::AppSrc, value: u8) -> Result<(), gstreamer::FlowError> {
        let mut buffer = gstreamer::Buffer::from_mut_slice(vec![value; 12]);
        buffer
            .get_mut()
            .unwrap()
            .set_pts(gstreamer::ClockTime::from_seconds(value as u64));
        source.push_buffer(buffer)?;
        Ok(())
    }

    #[test]
    fn bounded_queue_preserves_slow_consumer_frames_and_pts(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut capture =
            StreamCapture::new(&format!("{CAPTURE_PIPELINE} max-buffers=2 drop=false"))?;
        let queued = queue_notifications(&capture);
        capture.start()?;
        let src = source(&capture)?;
        for value in 0..20 {
            push_frame(&src, value)?;
        }
        src.end_of_stream()?;
        queued.recv_timeout(DEADLINE)?;
        queued.recv_timeout(DEADLINE)?;
        // The consumer starts only after the bounded queue is full.
        let mut values = Vec::new();
        let deadline = Instant::now() + DEADLINE;
        loop {
            match capture.try_grab_rgb8()? {
                Grab::Frame { image, pts } => {
                    let value = image.as_slice()[0];
                    assert_eq!(pts, Some(Duration::from_secs(value as u64)));
                    values.push(value);
                }
                Grab::Eos => break,
                Grab::Pending => std::thread::sleep(Duration::from_millis(1)),
            }
            assert!(Instant::now() < deadline, "queue did not drain to EOS");
        }
        capture.close()?;
        assert_eq!(values, (0..20).collect::<Vec<u8>>());
        Ok(())
    }

    #[test]
    fn dropping_queue_keeps_latest_frame_without_a_consumer(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut capture =
            StreamCapture::new(&format!("{CAPTURE_PIPELINE} max-buffers=1 drop=true"))?;
        let queued = queue_notifications(&capture);
        capture.start()?;
        let src = source(&capture)?;
        for value in 0..20 {
            push_frame(&src, value)?;
        }
        for _ in 0..20 {
            queued.recv_timeout(DEADLINE)?;
        }
        let Grab::Frame { image, pts } = capture.try_grab_rgb8()? else {
            panic!("missing frame")
        };
        assert_eq!(image.as_slice(), &[19; 12]);
        assert_eq!(pts, Some(Duration::from_secs(19)));
        capture.close()?;
        Ok(())
    }

    #[test]
    fn flush_discards_queued_frames() -> Result<(), Box<dyn std::error::Error>> {
        let mut capture = StreamCapture::new(CAPTURE_PIPELINE)?;
        let queued = queue_notifications(&capture);
        capture.start()?;
        push_frame(&source(&capture)?, 100)?;
        queued.recv_timeout(DEADLINE)?;
        let pad = capture
            .appsink
            .static_pad("sink")
            .ok_or("missing sink pad")?;
        assert!(pad.send_event(gstreamer::event::FlushStart::new()));
        assert!(pad.send_event(gstreamer::event::FlushStop::new(true)));
        assert!(matches!(capture.try_grab_rgb8()?, Grab::Pending));
        capture.close()?;
        Ok(())
    }

    #[test]
    fn failed_seek_keeps_queued_frames() -> Result<(), Box<dyn std::error::Error>> {
        let mut capture = StreamCapture::new(CAPTURE_PIPELINE)?;
        let queued = queue_notifications(&capture);
        capture.start()?;
        push_frame(&source(&capture)?, 10)?;
        queued.recv_timeout(DEADLINE)?;
        // Appsrc defaults to a non-seekable stream and rejects this seek without flushing.
        assert!(capture
            .seek(1.0, gstreamer::SeekFlags::FLUSH, gstreamer::ClockTime::ZERO)
            .is_err());
        let Grab::Frame { image, .. } = capture.try_grab_rgb8()? else {
            panic!("seek discarded data")
        };
        assert_eq!(image.as_slice(), &[10; 12]);
        capture.close()?;
        Ok(())
    }

    #[test]
    fn fps_comes_from_negotiated_caps() -> Result<(), Box<dyn std::error::Error>> {
        for (rate, expected) in [("1/1", 1.0), ("0/1", 0.0), ("30000/1001", 30000.0 / 1001.0)] {
            let capture = StreamCapture::new(&CAPTURE_PIPELINE.replace("1/1", rate))?;
            assert_eq!(capture.get_fps(), None);
            let queued = queue_notifications(&capture);
            capture.start()?;
            push_frame(&source(&capture)?, 0)?;
            queued.recv_timeout(DEADLINE)?;
            assert_eq!(capture.get_fps(), Some(expected));
            capture.close()?;
        }
        Ok(())
    }

    #[test]
    fn absent_pts_and_pause_resume_keep_queued_frames() -> Result<(), Box<dyn std::error::Error>> {
        let mut capture = StreamCapture::new(CAPTURE_PIPELINE)?;
        let queued = queue_notifications(&capture);
        capture.start()?;
        let src = source(&capture)?;
        // Appsrc may timestamp the first buffer at zero; a later absent PTS must stay absent.
        push_frame(&src, 0)?;
        queued.recv_timeout(DEADLINE)?;
        assert!(matches!(capture.try_grab_rgb8()?, Grab::Frame { .. }));
        src.push_buffer(gstreamer::Buffer::from_mut_slice(vec![7u8; 12]))?;
        queued.recv_timeout(DEADLINE)?;
        capture.pipeline.set_state(gstreamer::State::Paused)?;
        capture
            .pipeline
            .state(gstreamer::ClockTime::from_seconds(5))
            .0?;
        let Grab::Frame { image, pts } = capture.try_grab_rgb8()? else {
            panic!("pause discarded the sample")
        };
        assert_eq!(image.as_slice(), &[7; 12]);
        assert_eq!(pts, None);
        push_frame(&src, 8)?;
        capture.start()?;
        let deadline = Instant::now() + DEADLINE;
        loop {
            match capture.try_grab_rgb8()? {
                Grab::Frame { image, pts } => {
                    assert_eq!(image.as_slice(), &[8; 12]);
                    assert_eq!(pts, Some(Duration::from_secs(8)));
                    break;
                }
                Grab::Pending => std::thread::sleep(Duration::from_millis(1)),
                Grab::Eos => panic!("pause discarded the sample"),
            }
            assert!(Instant::now() < deadline);
        }
        capture.close()?;
        Ok(())
    }

    #[test]
    fn default_bound_preserves_explicit_queue_settings() -> Result<(), Box<dyn std::error::Error>> {
        let default = StreamCapture::new(CAPTURE_PIPELINE)?;
        assert_eq!(default.appsink.max_buffers(), DEFAULT_MAX_BUFFERS);
        assert!(!default.appsink.is_drop());
        let custom = StreamCapture::new(&format!("{CAPTURE_PIPELINE} max-buffers=2 drop=true"))?;
        assert_eq!(custom.appsink.max_buffers(), 2);
        assert!(custom.appsink.is_drop());
        Ok(())
    }

    #[test]
    fn close_and_drop_finalize_recording_with_full_capture_queue(
    ) -> Result<(), Box<dyn std::error::Error>> {
        use super::super::video::{ImageFormat, VideoReader};

        for drop_capture in [false, true] {
            let tmp = tempfile::tempdir()?;
            let path = tmp.path().join("capture.mov");
            let location =
                super::super::quote_pipeline_value(path.to_str().ok_or("invalid path")?)?;
            // qtmux writes its index at EOS, so successful playback checks actual finalization.
            let capture = StreamCapture::new(&format!(
                "appsrc name=src is-live=true format=time !                  video/x-raw,format=RGB,width=64,height=48,framerate=30/1 ! tee name=t                  t. ! queue ! appsink name=sink sync=false max-buffers=1 drop=false                  t. ! queue ! qtmux ! filesink location={location}"
            ))?;
            let queued = queue_notifications(&capture);
            let (blocked_tx, blocked_rx) = mpsc::channel();
            let seen = std::sync::atomic::AtomicBool::new(false);
            capture
                .appsink
                .static_pad("sink")
                .ok_or("missing sink pad")?
                .add_probe(gstreamer::PadProbeType::BUFFER, move |_, _| {
                    if seen.swap(true, std::sync::atomic::Ordering::Relaxed) {
                        let _ = blocked_tx.send(());
                    }
                    gstreamer::PadProbeReturn::Ok
                });

            capture.start()?;
            let src = source(&capture)?;
            for (index, value) in [20, 40].into_iter().enumerate() {
                let mut buffer = gstreamer::Buffer::from_mut_slice(vec![value; 64 * 48 * 3]);
                let buffer_mut = buffer.get_mut().ok_or("new buffer is shared")?;
                buffer_mut.set_pts(gstreamer::ClockTime::from_nseconds(
                    index as u64 * 33_333_333,
                ));
                buffer_mut.set_duration(gstreamer::ClockTime::from_nseconds(33_333_333));
                src.push_buffer(buffer)?;
            }
            queued.recv_timeout(DEADLINE)?;
            blocked_rx.recv_timeout(DEADLINE)?;
            if drop_capture {
                drop(capture);
            } else {
                capture.close()?;
                assert!(!capture.appsink.is_drop(), "close changed the queue policy");
            }

            let mut reader = VideoReader::new(path, ImageFormat::Rgb8)?;
            reader.start()?;
            let deadline = Instant::now() + DEADLINE;
            let mut recorded = Vec::new();
            loop {
                match reader.try_grab_rgb8()? {
                    Grab::Frame { image, .. } => recorded.push(image.as_slice()[0]),
                    Grab::Eos => break,
                    Grab::Pending => std::thread::sleep(Duration::from_millis(1)),
                }
                assert!(Instant::now() < deadline, "recorded file did not reach EOS");
            }
            assert_eq!(recorded, [20, 40]);
            reader.close()?;
        }
        Ok(())
    }

    #[test]
    fn close_and_drop_release_a_full_queue() -> Result<(), Box<dyn std::error::Error>> {
        for drop_capture in [false, true] {
            let capture =
                StreamCapture::new(&format!("{CAPTURE_PIPELINE} max-buffers=1 drop=false"))?;
            let queued = queue_notifications(&capture);
            let (blocked_tx, blocked_rx) = mpsc::channel();
            let count = std::sync::atomic::AtomicUsize::new(0);
            capture
                .appsink
                .static_pad("sink")
                .ok_or("missing sink pad")?
                .add_probe(gstreamer::PadProbeType::BUFFER, move |_, _| {
                    if count.fetch_add(1, std::sync::atomic::Ordering::Relaxed) == 1 {
                        let _ = blocked_tx.send(());
                    }
                    gstreamer::PadProbeReturn::Ok
                });
            capture.start()?;
            let src = source(&capture)?;
            push_frame(&src, 0)?;
            push_frame(&src, 1)?;
            queued.recv_timeout(DEADLINE)?;
            blocked_rx.recv_timeout(DEADLINE)?;
            let (done_tx, done_rx) = mpsc::channel();
            let worker = std::thread::spawn(move || {
                if drop_capture {
                    drop(capture);
                } else {
                    capture.close().unwrap();
                }
                let _ = done_tx.send(());
            });
            done_rx.recv_timeout(DEADLINE)?;
            worker.join().map_err(|_| "shutdown panicked")?;
        }
        Ok(())
    }
}
