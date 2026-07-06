use super::frame::GstFrame;
use crate::stream::error::StreamCaptureError;
use circular_buffer::FixedCircularBuffer;
use gstreamer::prelude::*;
use gstreamer_video::VideoInfo;
use kornia_image::Image;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

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
///
/// Captured frames are held in a fixed-capacity ring of the 5 most recent frames;
/// under back-pressure the oldest frames are dropped so `grab_*` always returns
/// fresh data. Fatal pipeline errors are watched on a background bus thread and
/// surfaced through the `grab_*` methods and [`StreamCapture::last_error`].
pub struct StreamCapture {
    pub(crate) pipeline: gstreamer::Pipeline,
    circular_buffer: Arc<Mutex<FixedCircularBuffer<gstreamer::Sample, 5>>>,
    fps: Arc<Mutex<gstreamer::Fraction>>,
    /// Last fatal error reported on the pipeline bus, if any.
    bus_error: Arc<Mutex<Option<String>>>,
    /// Signals the bus-watch thread to stop.
    bus_stop: Arc<AtomicBool>,
    /// Handle to the background bus-watch thread.
    bus_handle: Arc<Mutex<Option<JoinHandle<()>>>>,
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

        let circular_buffer = Arc::new(Mutex::new(FixedCircularBuffer::new()));
        let fps = Arc::new(Mutex::new(gstreamer::Fraction::new(1, 1)));
        let bus_error = Arc::new(Mutex::new(None));
        let bus_stop = Arc::new(AtomicBool::new(false));

        // Watch the pipeline bus on a background thread so fatal pipeline errors
        // are surfaced to the caller instead of silently stalling the appsink.
        let bus_handle = pipeline.bus().map(|bus| {
            let bus_error = bus_error.clone();
            let bus_stop = bus_stop.clone();
            std::thread::spawn(move || {
                while !bus_stop.load(Ordering::Relaxed) {
                    let Some(msg) = bus.timed_pop(gstreamer::ClockTime::from_mseconds(100)) else {
                        continue;
                    };
                    match msg.view() {
                        gstreamer::MessageView::Error(err) => {
                            let text = format!("{} ({:?})", err.error(), err.debug());
                            log::error!(
                                "gstreamer pipeline error from {:?}: {text}",
                                msg.src().map(|s| s.path_string())
                            );
                            if let Ok(mut slot) = bus_error.lock() {
                                *slot = Some(text);
                            }
                            break;
                        }
                        gstreamer::MessageView::Eos(..) => {
                            log::debug!("gstreamer received EOS");
                            break;
                        }
                        _ => {}
                    }
                }
            })
        });

        appsink.set_callbacks(
            gstreamer_app::AppSinkCallbacks::builder()
                .new_sample({
                    let circular_buffer = circular_buffer.clone();
                    let fps = fps.clone();

                    move |sink| {
                        Self::extract_sample(sink)
                            .map_err(|_| gstreamer::FlowError::Eos)
                            .and_then(|(sample, fps_fraction)| {
                                circular_buffer
                                    .lock()
                                    .map_err(|_| gstreamer::FlowError::Error)?
                                    .push_back(sample);
                                *fps.lock().map_err(|_| gstreamer::FlowError::Error)? =
                                    fps_fraction;
                                Ok(gstreamer::FlowSuccess::Ok)
                            })
                    }
                })
                .build(),
        );

        Ok(Self {
            pipeline,
            circular_buffer,
            fps,
            bus_error,
            bus_stop,
            bus_handle: Arc::new(Mutex::new(bus_handle)),
        })
    }

    /// Returns the last fatal error reported on the pipeline bus, if any.
    ///
    /// Once a pipeline error is recorded here, no further frames will arrive; the
    /// grab methods surface it as [`StreamCaptureError::PipelineError`].
    pub fn last_error(&self) -> Option<String> {
        self.bus_error.lock().ok().and_then(|e| e.clone())
    }

    /// Returns `true` while the pipeline has not reported a fatal bus error.
    pub fn is_running(&self) -> bool {
        self.last_error().is_none()
    }

    /// Gets the current fps of the stream
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

    /// Starts the stream capture pipeline.
    ///
    /// The pipeline bus is watched on a background thread (spawned in
    /// [`new`](Self::new)); fatal errors are recorded and surfaced through the
    /// grab methods and [`last_error`](Self::last_error).
    pub fn start(&self) -> Result<(), StreamCaptureError> {
        self.circular_buffer
            .lock()
            .map_err(|_| StreamCaptureError::MutexPoisonError)?
            .clear();
        self.pipeline.set_state(gstreamer::State::Playing)?;
        Ok(())
    }

    /// Grabs the most recent captured frame as a generic, format-aware [`GstFrame`].
    ///
    /// The frame holds a zero-copy view of the mapped buffer plus the negotiated
    /// video info (format, size, row strides) and timing metadata (PTS/duration).
    /// Convert it to a typed image via [`GstFrame::to_image_u8`] /
    /// [`GstFrame::to_image_u16`], or read the raw bytes via [`GstFrame::as_bytes`].
    ///
    /// Returns `Ok(None)` when no frame is buffered yet, or
    /// [`StreamCaptureError::PipelineError`] if the pipeline reported a fatal error
    /// and the buffer has drained.
    pub fn grab(&mut self) -> Result<Option<GstFrame>, StreamCaptureError> {
        let sample = {
            let mut circular_buffer = self
                .circular_buffer
                .lock()
                .map_err(|_| StreamCaptureError::MutexPoisonError)?;
            circular_buffer.pop_front()
        };

        let Some(sample) = sample else {
            // No frame available — surface a fatal pipeline error if one occurred,
            // rather than returning `None` forever on a dead pipeline.
            if let Some(err) = self.last_error() {
                return Err(StreamCaptureError::PipelineError(err));
            }
            return Ok(None);
        };

        let caps = sample
            .caps()
            .ok_or_else(|| StreamCaptureError::GetCapsError("missing caps".to_string()))?;
        let info = VideoInfo::from_caps(caps).map_err(|e| {
            StreamCaptureError::GetCapsError(format!("not a raw video frame: {e}"))
        })?;

        let buffer = sample
            .buffer_owned()
            .ok_or(StreamCaptureError::GetBufferError)?;
        let pts = buffer.pts();
        let duration = buffer.duration();
        let map = buffer
            .into_mapped_buffer_readable()
            .map_err(|_| StreamCaptureError::GetBufferError)?;

        Ok(Some(GstFrame::new(map, info, pts, duration)))
    }

    /// Grabs the last captured frame as an RGB8 image (pipeline format `RGB`).
    pub fn grab_rgb8(&mut self) -> Result<Option<Image<u8, 3>>, StreamCaptureError> {
        match self.grab()? {
            Some(frame) => Ok(Some(frame.to_image_u8::<3>()?)),
            None => Ok(None),
        }
    }

    /// Grabs the last captured frame as a mono8 image (pipeline format `GRAY8`).
    pub fn grab_mono8(&mut self) -> Result<Option<Image<u8, 1>>, StreamCaptureError> {
        match self.grab()? {
            Some(frame) => Ok(Some(frame.to_image_u8::<1>()?)),
            None => Ok(None),
        }
    }

    /// Grabs the last captured frame as a mono16 image (pipeline format `GRAY16_LE`).
    pub fn grab_mono16(&mut self) -> Result<Option<Image<u16, 1>>, StreamCaptureError> {
        match self.grab()? {
            Some(frame) => Ok(Some(frame.to_image_u16::<1>()?)),
            None => Ok(None),
        }
    }

    /// Closes the stream capture pipeline.
    pub fn close(&self) -> Result<(), StreamCaptureError> {
        // Stop and join the bus-watch thread first so it cannot outlive the pipeline.
        self.bus_stop.store(true, Ordering::Relaxed);
        if let Ok(mut guard) = self.bus_handle.lock() {
            if let Some(handle) = guard.take() {
                let _ = handle.join();
            }
        }

        let res = self.pipeline.send_event(gstreamer::event::Eos::new());
        if !res {
            return Err(StreamCaptureError::SendEosError);
        }
        self.pipeline.set_state(gstreamer::State::Null)?;
        self.circular_buffer
            .lock()
            .map_err(|_| StreamCaptureError::MutexPoisonError)?
            .clear();
        Ok(())
    }

    /// Pulls the next sample from the AppSink and reads its framerate from caps.
    ///
    /// The full [`gstreamer::Sample`] (buffer + caps) is kept so that [`grab`] can
    /// build a format-aware [`GstFrame`] with correct strides and metadata.
    fn extract_sample(
        appsink: &gstreamer_app::AppSink,
    ) -> Result<(gstreamer::Sample, gstreamer::Fraction), StreamCaptureError> {
        let sample = appsink.pull_sample()?;

        let caps = sample.caps().ok_or_else(|| {
            StreamCaptureError::GetCapsError("Failed to get the caps".to_string())
        })?;

        let structure = caps.structure(0).ok_or_else(|| {
            StreamCaptureError::GetCapsError("Failed to get the structure".to_string())
        })?;

        // Framerate is optional (some sources omit it); default to 0/1 if absent.
        let fps = structure
            .get::<gstreamer::Fraction>("framerate")
            .unwrap_or_else(|_| gstreamer::Fraction::new(0, 1));

        Ok((sample, fps))
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
