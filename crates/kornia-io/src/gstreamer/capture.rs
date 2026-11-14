use super::{image_from_gst_buffer, video::detailed_error};
use crate::stream::error::StreamCaptureError;
use circular_buffer::FixedCircularBuffer;
use gstreamer::prelude::*;
use kornia_image::Image;
use std::sync::{Arc, Mutex};

// utility struct to store the frame buffer
struct FrameBuffer {
    buffer: gstreamer::Buffer,
    /// Caps of the sample. The frame geometry, format and layout are read from them when the
    /// frame is grabbed, so problems with them are returned by the grab instead of ending the
    /// stream on the streaming thread.
    caps: gstreamer::Caps,
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

/// The terminal state a capture pipeline reports on its bus.
#[derive(Default)]
struct StreamStatus {
    eos: bool,
    error: Option<gstreamer::glib::Error>,
}

/// The outcome of [`StreamCapture::try_grab_rgb8`] and [`StreamCapture::try_grab_gray8`].
///
/// Unlike [`StreamCapture::grab_rgb8`] and [`StreamCapture::grab_gray8`], which report `None`
/// for a frame that has not arrived yet, the end of the stream and a pipeline error alike, this
/// enum keeps the three cases apart.
pub enum Grab<const C: usize> {
    /// A decoded frame.
    Frame(Image<u8, C>),
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
    circular_buffer: Arc<Mutex<FixedCircularBuffer<FrameBuffer, 5>>>,
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
    ///
    /// # Errors
    ///
    /// Returns an error if GStreamer initialization or pipeline construction fails, the named
    /// appsink or bus is unavailable, or the appsink flush probe cannot be installed.
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

        let circular_buffer = Arc::new(Mutex::new(FixedCircularBuffer::new()));
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

        let sink_pad = appsink
            .static_pad("sink")
            .ok_or_else(|| StreamCaptureError::InvalidConfig("appsink has no sink pad".into()))?;
        sink_pad
            .add_probe(
                gstreamer::PadProbeType::EVENT_DOWNSTREAM | gstreamer::PadProbeType::EVENT_FLUSH,
                {
                    let circular_buffer = circular_buffer.clone();
                    move |_pad, info| {
                        if info
                            .event()
                            .is_some_and(|event| event.type_() == gstreamer::EventType::FlushStop)
                        {
                            // FLUSH_STOP takes the sink pad's stream lock before probes run,
                            // so an in-flight new_sample callback finishes before this clear.
                            // See GStreamer's FLUSH_START/STOP event design:
                            // https://gstreamer.freedesktop.org/documentation/additional/design/events.html
                            let mut ring = match circular_buffer.lock() {
                                Ok(ring) => ring,
                                // Discard invalidated buffers even after a panic. Keep the
                                // poison flag: subsequent grabs still return MutexPoisonError.
                                Err(poisoned) => poisoned.into_inner(),
                            };
                            ring.clear();
                        }
                        gstreamer::PadProbeReturn::Ok
                    }
                },
            )
            .ok_or_else(|| {
                StreamCaptureError::InvalidConfig("could not install appsink flush probe".into())
            })?;

        appsink.set_callbacks(
            gstreamer_app::AppSinkCallbacks::builder()
                .new_sample({
                    let circular_buffer = circular_buffer.clone();
                    let fps = fps.clone();
                    let status = status.clone();

                    move |sink| match Self::extract_frame_buffer(sink) {
                        Ok((frame_buffer, fps_fraction)) => {
                            circular_buffer
                                .lock()
                                .map_err(|_| gstreamer::FlowError::Error)?
                                .push_back(frame_buffer);
                            *fps.lock().map_err(|_| gstreamer::FlowError::Error)? = fps_fraction;
                            Ok(gstreamer::FlowSuccess::Ok)
                        }
                        // Returning `Eos` here ended the stream silently; report the error to
                        // the caller and let GStreamer post it on the bus.
                        Err(err) => {
                            if let Ok(mut status) = status.lock() {
                                status.error.get_or_insert_with(|| {
                                    gstreamer::glib::Error::new(
                                        gstreamer::CoreError::Failed,
                                        &err.to_string(),
                                    )
                                });
                            }
                            Err(gstreamer::FlowError::Error)
                        }
                    }
                })
                .build(),
        );

        Ok(Self {
            pipeline,
            bus,
            circular_buffer,
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
        self.status.lock().is_ok_and(|status| status.eos)
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
        self.circular_buffer
            .lock()
            .map_err(|_| StreamCaptureError::MutexPoisonError)?
            .clear();
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
    /// [`StreamCaptureError::GetBufferError`] if it cannot be mapped, and
    /// [`StreamCaptureError::MutexPoisonError`] if the frame queue lock is poisoned.
    pub fn grab_rgb8(&mut self) -> Result<Option<Image<u8, 3>>, StreamCaptureError> {
        self.grab_image::<3>(GrabFormat::Rgb8)
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
        self.grab_image::<1>(GrabFormat::Gray8)
    }

    /// Pops the oldest buffered frame and converts it to an image with `C` channels.
    fn grab_image<const C: usize>(
        &mut self,
        grab_format: GrabFormat,
    ) -> Result<Option<Image<u8, C>>, StreamCaptureError> {
        // Keeps the bus queue bounded for callers that only ever use the grab methods; the sync
        // handler has already recorded EOS and errors.
        self.drain_bus();
        // Pop in its own statement so the lock is released before the frame is mapped or
        // repacked: the appsink streaming thread needs it to push the next sample.
        let frame_buffer = self
            .circular_buffer
            .lock()
            .map_err(|_| StreamCaptureError::MutexPoisonError)?
            .pop_front();
        let Some(FrameBuffer { buffer, caps }) = frame_buffer else {
            return Ok(None);
        };

        let info = gstreamer_video::VideoInfo::from_caps(&caps)
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
        let image = image_from_gst_buffer(buffer, &info)?;

        Ok(Some(image))
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
    ///         Grab::Frame(frame) => println!("{}x{}", frame.width(), frame.height()),
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
        if let Some(image) = self.grab_image::<C>(grab_format)? {
            return Ok(Grab::Frame(image));
        }

        let (error, eos) = {
            let status = self
                .status
                .lock()
                .map_err(|_| StreamCaptureError::MutexPoisonError)?;
            (status.error.clone(), status.eos)
        };
        // The final frame can land between the empty check above and the terminal state below:
        // take one more look before reporting the end.
        if error.is_some() || eos {
            if let Some(image) = self.grab_image::<C>(grab_format)? {
                return Ok(Grab::Frame(image));
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

    /// Closes the stream capture pipeline.
    pub fn close(&self) -> Result<(), StreamCaptureError> {
        // A pipeline that already ended, or failed, rejects a second end-of-stream event.
        if !self.is_eos() && self.stream_error().is_none() {
            let res = self.pipeline.send_event(gstreamer::event::Eos::new());
            if !res {
                return Err(StreamCaptureError::SendEosError);
            }
        }
        self.pipeline.set_state(gstreamer::State::Null)?;
        // Auto-flush is off, so drop the messages posted while stopping: they retain references
        // to the pipeline elements.
        self.drain_bus();
        self.circular_buffer
            .lock()
            .map_err(|_| StreamCaptureError::MutexPoisonError)?
            .clear();
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

    /// Extracts a frame buffer from the AppSink.
    ///
    /// # Arguments
    ///
    /// * `appsink` - The AppSink to extract the frame buffer from.
    ///
    /// # Returns
    ///
    /// A Result containing the extracted FrameBuffer or a StreamCaptureError.
    fn extract_frame_buffer(
        appsink: &gstreamer_app::AppSink,
    ) -> Result<(FrameBuffer, gstreamer::Fraction), StreamCaptureError> {
        let sample = appsink.pull_sample()?;

        let caps = sample.caps().ok_or_else(|| {
            StreamCaptureError::GetCapsError("Failed to get the caps".to_string())
        })?;

        // A missing framerate means a variable or unknown rate, which GStreamer writes as 0/1.
        // Failing here would silently end the stream: errors in this callback never reach the
        // caller, so every other check on the caps happens in `grab_image`.
        let fps = caps
            .structure(0)
            .and_then(|structure| structure.get::<gstreamer::Fraction>("framerate").ok())
            .unwrap_or_else(|| gstreamer::Fraction::new(0, 1));

        let buffer = sample
            .buffer_owned()
            .ok_or_else(|| StreamCaptureError::GetBufferError)?;

        let frame_buffer = FrameBuffer {
            buffer,
            caps: caps.to_owned(),
        };

        Ok((frame_buffer, fps))
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
    use std::{sync::mpsc, time::Duration};

    const CAPTURE_PIPELINE: &str = "appsrc name=src format=time ! video/x-raw,format=RGB,width=4,height=1,framerate=1/1 ! appsink name=sink sync=false async=false";
    const DEADLINE: Duration = Duration::from_secs(5);

    /// The real constructor's probe must clear a frame queued by a callback already in flight.
    #[test]
    fn flush_stop_discards_inflight_sample() -> Result<(), Box<dyn std::error::Error>> {
        let capture = StreamCapture::new(CAPTURE_PIPELINE)?;
        let sink = capture
            .pipeline
            .by_name("sink")
            .ok_or("missing sink")?
            .downcast::<gstreamer_app::AppSink>()
            .map_err(|_| "not appsink")?;
        let source = capture
            .pipeline
            .by_name("src")
            .ok_or("missing src")?
            .downcast::<gstreamer_app::AppSrc>()
            .map_err(|_| "not appsrc")?;
        let pad = sink.static_pad("sink").ok_or("missing sink pad")?;
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let release_rx = Mutex::new(release_rx);
        sink.set_callbacks(
            gstreamer_app::AppSinkCallbacks::builder()
                .new_sample({
                    let ring = capture.circular_buffer.clone();
                    move |sink| {
                        let (frame, _) = StreamCapture::extract_frame_buffer(sink)
                            .map_err(|_| gstreamer::FlowError::Error)?;
                        entered_tx
                            .send(())
                            .map_err(|_| gstreamer::FlowError::Error)?;
                        release_rx
                            .lock()
                            .map_err(|_| gstreamer::FlowError::Error)?
                            .recv_timeout(DEADLINE)
                            .map_err(|_| gstreamer::FlowError::Error)?;
                        ring.lock()
                            .map_err(|_| gstreamer::FlowError::Error)?
                            .push_back(frame);
                        Ok(gstreamer::FlowSuccess::Ok)
                    }
                })
                .build(),
        );
        let (stopped_tx, stopped_rx) = mpsc::channel();
        pad.add_probe(
            gstreamer::PadProbeType::EVENT_DOWNSTREAM | gstreamer::PadProbeType::EVENT_FLUSH,
            move |_pad, info| {
                if info
                    .event()
                    .is_some_and(|event| event.type_() == gstreamer::EventType::FlushStop)
                {
                    let _ = stopped_tx.send(());
                }
                gstreamer::PadProbeReturn::Ok
            },
        );
        capture.start()?;
        source.push_buffer(gstreamer::Buffer::from_mut_slice(vec![100u8; 12]))?;
        entered_rx.recv_timeout(DEADLINE)?;
        let (attempt_tx, attempt_rx) = mpsc::channel();
        let flush = std::thread::spawn(move || {
            let _ = attempt_tx.send(());
            pad.send_event(gstreamer::event::FlushStart::new())
                && pad.send_event(gstreamer::event::FlushStop::new(true))
        });
        attempt_rx.recv_timeout(DEADLINE)?;
        let early_stop = stopped_rx.recv_timeout(Duration::from_millis(100));
        release_tx.send(())?;
        stopped_rx.recv_timeout(DEADLINE)?;
        let flushed = flush.join().map_err(|_| "flush thread panicked")?;
        let empty = capture
            .circular_buffer
            .lock()
            .map_err(|_| "poisoned queue")?
            .is_empty();
        capture.pipeline.set_state(gstreamer::State::Null)?;
        assert!(
            matches!(early_stop, Err(mpsc::RecvTimeoutError::Timeout)),
            "flush-stop overtook a sample callback"
        );
        assert!(flushed, "flush events were rejected");
        assert!(empty, "a pre-flush callback's frame survived flush-stop");
        Ok(())
    }

    /// Clearing invalidated frames must not hide the queue's existing poison error.
    #[test]
    fn flush_stop_keeps_queue_poison_visible() -> Result<(), Box<dyn std::error::Error>> {
        let mut capture = StreamCapture::new(CAPTURE_PIPELINE)?;
        capture.start()?;
        let ring = capture.circular_buffer.clone();
        assert!(std::thread::spawn(move || {
            let _guard = ring.lock().unwrap();
            panic!("poison the frame queue");
        })
        .join()
        .is_err());
        let sink = capture.pipeline.by_name("sink").ok_or("missing sink")?;
        let pad = sink.static_pad("sink").ok_or("missing sink pad")?;
        assert!(pad.send_event(gstreamer::event::FlushStart::new()));
        assert!(pad.send_event(gstreamer::event::FlushStop::new(true)));
        let result = capture.try_grab_rgb8();
        capture.pipeline.set_state(gstreamer::State::Null)?;
        assert!(matches!(result, Err(StreamCaptureError::MutexPoisonError)));
        Ok(())
    }
}
