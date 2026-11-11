use super::{image_from_gst_buffer, video::detailed_error};
use crate::stream::error::StreamCaptureError;
use gstreamer::prelude::*;
use kornia_image::Image;
use std::sync::{Arc, Mutex};

/// How many decoded frames the appsink may queue for the next grab. When the queue is full the
/// sink blocks the streaming thread (backpressure), so a slow consumer never loses frames.
const FRAME_BUFFER_CAPACITY: u32 = 16;

/// The terminal state a capture pipeline reports on its bus.
#[derive(Default)]
struct StreamStatus {
    eos: bool,
    error: Option<gstreamer::glib::Error>,
}

/// A decoded frame and the presentation timestamp the pipeline reported for it.
struct GrabbedFrame<const C: usize> {
    image: Image<u8, C>,
    pts: Option<gstreamer::ClockTime>,
}

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

/// The outcome of [`StreamCapture::try_grab_rgb8`] and [`StreamCapture::try_grab_gray8`].
///
/// Unlike [`StreamCapture::grab_rgb8`] and [`StreamCapture::grab_gray8`], which report `None`
/// for a frame that has not arrived yet, the end of the stream and a pipeline error alike, this
/// enum keeps the three cases apart.
pub enum Grab<const C: usize> {
    /// A decoded frame and the timestamp the pipeline reported for it.
    Frame {
        /// The decoded image.
        image: Image<u8, C>,
        /// The frame's presentation timestamp (`buffer.pts()`), when the pipeline sets one.
        pts: Option<gstreamer::ClockTime>,
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

/// Represents a stream capture pipeline using GStreamer.
pub struct StreamCapture {
    pub(crate) pipeline: gstreamer::Pipeline,
    /// The pipeline's bus. Auto-flush is off, so messages posted while the pipeline goes to null
    /// can still be read.
    bus: gstreamer::Bus,
    /// The element that queues decoded frames. Its bounded queue applies backpressure and is
    /// flushed by GStreamer on a flushing seek or a state change to null.
    appsink: gstreamer_app::AppSink,
    fps: Arc<Mutex<gstreamer::Fraction>>,
    /// End-of-stream and errors seen on the bus, updated by its sync handler.
    status: Arc<Mutex<StreamStatus>>,
}

impl StreamCapture {
    /// Creates a new StreamCapture instance with the given pipeline description.
    ///
    /// # Arguments
    ///
    /// * `pipeline_desc` - A string describing the GStreamer pipeline.
    ///
    /// # Returns
    ///
    /// A Result containing the StreamCapture instance or a StreamCaptureError.
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

        // Bounded queue: appsink blocks the streaming thread while the queue is full
        // (backpressure) instead of dropping frames. GStreamer itself wakes it on a flushing
        // seek and flushes the queue on a state change to Null.
        appsink.set_max_buffers(FRAME_BUFFER_CAPACITY);
        appsink.set_drop(false);
        // Post end-of-stream on the bus as soon as the sink sees it, without waiting for the
        // queued frames to be pulled: the terminal state is exposed while frames remain buffered.
        appsink.set_wait_on_eos(false);

        let fps = Arc::new(Mutex::new(gstreamer::Fraction::new(1, 1)));
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
                        gstreamer::MessageView::Eos(_) => status.eos = true,
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
            fps,
            status,
        })
    }

    /// Gets the current fps of the stream, or `0.0` if the caps give no fixed framerate.
    pub fn get_fps(&self) -> Option<f64> {
        self.fps
            .lock()
            .ok()
            .map(|fps| fps.numer() as f64 / fps.denom() as f64)
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
    /// # Returns
    ///
    /// `true` once the pipeline has posted end-of-stream; frames decoded before it may still
    /// remain buffered for the `try_grab_*` methods.
    pub fn is_eos(&self) -> bool {
        self.drain_bus();
        self.status.lock().is_ok_and(|status| status.eos)
    }

    /// Returns the first error the pipeline posted on its bus, if any.
    ///
    /// A pipeline that cancels its own stream (an undecodable file, a missing decoder, failed
    /// negotiation) sets this instead of silently delivering no more frames.
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
        self.drain_bus();
        // Starting a stopped pipeline begins a fresh run, so its previous terminal state is
        // discarded. Resuming a PLAYING/PAUSED one keeps it: a pipeline that already reached
        // end-of-stream stays there until it is seeked, and forgetting that would report
        // `Pending` forever instead of `Eos`.
        //
        // The buffered frames are left alone: a state change to Null flushes them inside
        // GStreamer, while resuming a paused pipeline must keep them.
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
    /// # Errors
    ///
    /// Returns [`StreamCaptureError::GetCapsError`] if the sample caps cannot be parsed as video
    /// caps (e.g. audio or arbitrary `application/*` caps),
    /// [`StreamCaptureError::InvalidImageFormat`] if the frames are not `RGB` (e.g. `BGR`,
    /// `RGBx`, or an encoded format such as `image/jpeg`) or their size or layout is invalid,
    /// [`StreamCaptureError::BufferSizeMismatch`] if the buffer is too small for the frame,
    /// [`StreamCaptureError::GetBufferError`] if it cannot be mapped, and
    /// [`StreamCaptureError::MutexPoisonError`] if the frame queue lock is poisoned.
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

    fn grab_image<const C: usize>(
        &mut self,
        grab_format: GrabFormat,
    ) -> Result<Option<GrabbedFrame<C>>, StreamCaptureError> {
        // Keeps the bus queue bounded for callers that only ever use the grab methods; the sync
        // handler has already recorded EOS and errors.
        self.drain_bus();

        // Pulling takes the oldest frame out of the appsink's bounded queue, which also releases
        // the streaming thread if it is blocked because the queue is full.
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

        // A missing framerate means a variable or unknown rate, which GStreamer writes as 0/1.
        let fps = caps
            .structure(0)
            .and_then(|structure| structure.get::<gstreamer::Fraction>("framerate").ok())
            .unwrap_or_else(|| gstreamer::Fraction::new(0, 1));
        *self
            .fps
            .lock()
            .map_err(|_| StreamCaptureError::MutexPoisonError)? = fps;

        let buffer = sample
            .buffer_owned()
            .ok_or(StreamCaptureError::GetBufferError)?;
        // The timestamp must be read before the buffer is moved into the image.
        let pts = buffer.pts();
        // Zero-copy when the rows are tightly packed; otherwise the padded rows are copied
        // into a packed image. See `image_from_gst_buffer`.
        let image = image_from_gst_buffer(buffer, &info)?;

        Ok(Some(GrabbedFrame { image, pts }))
    }

    /// Tries to grab the oldest captured frame as an RGB image, distinguishing a frame from
    /// "no frame yet", end-of-stream and a pipeline error.
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
    /// [`Grab::Frame`] with the oldest buffered frame and its presentation timestamp,
    /// [`Grab::Pending`] when none is buffered yet, or [`Grab::Eos`] once the stream has ended
    /// and no frame is left.
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
    ///         Grab::Frame { image, pts } => {
    ///             println!("{}x{} at {pts:?}", image.width(), image.height());
    ///         }
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
    /// [`Grab::Frame`] with the oldest buffered frame and its presentation timestamp,
    /// [`Grab::Pending`] when none is buffered yet, or [`Grab::Eos`] once the stream has ended
    /// and no frame is left.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::try_grab_rgb8`], with [`Self::grab_gray8`] for the frame itself.
    pub fn try_grab_gray8(&mut self) -> Result<Grab<1>, StreamCaptureError> {
        self.try_grab_image::<1>(GrabFormat::Gray8)
    }

    fn try_grab_image<const C: usize>(
        &mut self,
        grab_format: GrabFormat,
    ) -> Result<Grab<C>, StreamCaptureError> {
        if let Some(frame) = self.grab_image::<C>(grab_format)? {
            return Ok(Grab::Frame {
                image: frame.image,
                pts: frame.pts,
            });
        }

        let status = self
            .status
            .lock()
            .map_err(|_| StreamCaptureError::MutexPoisonError)?;
        if let Some(error) = &status.error {
            return Err(StreamCaptureError::StreamError(error.clone()));
        }
        if status.eos {
            return Ok(Grab::Eos);
        }
        Ok(Grab::Pending)
    }

    /// Closes the stream capture pipeline.
    ///
    /// Setting the pipeline to `Null` stops and flushes it: the appsink's queue is discarded and
    /// a streaming thread blocked on a full queue is woken, so this cannot deadlock. No
    /// end-of-stream event is sent first: with a filled sink queue it would wait for that same
    /// streaming thread, and a state change to `Null` finalizes downstream sinks anyway.
    pub fn close(&self) -> Result<(), StreamCaptureError> {
        self.pipeline.set_state(gstreamer::State::Null)?;
        // Auto-flush is off, so drop the messages posted while stopping: they retain references
        // to the pipeline elements.
        self.drain_bus();
        Ok(())
    }

    /// Forgets end-of-stream after a flushing seek, which clears EOS in GStreamer and lets the
    /// pipeline deliver frames again. Pipeline errors are kept: they do not recover.
    pub(crate) fn clear_eos(&self) {
        if let Ok(mut status) = self.status.lock() {
            status.eos = false;
        }
    }

    /// Restores end-of-stream after a failed seek, which did not clear it in GStreamer.
    pub(crate) fn mark_eos(&self) {
        if let Ok(mut status) = self.status.lock() {
            status.eos = true;
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
