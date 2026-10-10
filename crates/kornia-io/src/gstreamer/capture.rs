use super::image_from_gst_buffer;
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
    circular_buffer: Arc<Mutex<FixedCircularBuffer<FrameBuffer, 5>>>,
    fps: Arc<Mutex<gstreamer::Fraction>>,
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

        appsink.set_callbacks(
            gstreamer_app::AppSinkCallbacks::builder()
                .new_sample({
                    let circular_buffer = circular_buffer.clone();
                    let fps = fps.clone();

                    move |sink| {
                        Self::extract_frame_buffer(sink)
                            .map_err(|_| gstreamer::FlowError::Eos)
                            .and_then(|(frame_buffer, fps_fraction)| {
                                circular_buffer
                                    .lock()
                                    .map_err(|_| gstreamer::FlowError::Error)?
                                    .push_back(frame_buffer);
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

    /// Starts the stream capture pipeline and processes messages on the bus.
    pub fn start(&self) -> Result<(), StreamCaptureError> {
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
        if info.format() != gstreamer_video::VideoFormat::Rgb {
            return Err(StreamCaptureError::InvalidImageFormat(format!(
                "grab_rgb8 needs RGB frames, but the pipeline produces {} frames",
                info.format()
            )));
        }

        // Zero-copy when the rows are tightly packed; otherwise the padded rows are copied
        // into a packed image. See `image_from_gst_buffer`.
        let image = image_from_gst_buffer(buffer, &info)?;

        Ok(Some(image))
    }

    /// Closes the stream capture pipeline.
    pub fn close(&self) -> Result<(), StreamCaptureError> {
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
        // caller, so every other check on the caps happens in `grab_rgb8`.
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
