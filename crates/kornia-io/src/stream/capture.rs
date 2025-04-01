use crate::stream::error::StreamCaptureError;
use gstreamer::prelude::*;
use kornia_image::Image;
use std::sync::{Arc, Mutex};

// utility struct to store the frame buffer
struct FrameBuffer {
    buffer: gstreamer::MappedBuffer<gstreamer::buffer::Readable>,
    width: usize,
    height: usize,
}

/// Represents a stream capture pipeline using GStreamer.
pub struct StreamCapture {
    pipeline: gstreamer::Pipeline,
    last_frame: Arc<Mutex<Option<FrameBuffer>>>,
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

        let last_frame = Arc::new(Mutex::new(None));

        appsink.set_callbacks(
            gstreamer_app::AppSinkCallbacks::builder()
                .new_sample({
                    let last_frame = last_frame.clone();
                    move |sink| {
                        last_frame
                            .lock()
                            .map_err(|_| gstreamer::FlowError::Error)
                            .and_then(|mut guard| {
                                Self::extract_frame_buffer(sink)
                                    .map(|frame_buffer| {
                                        guard.replace(frame_buffer);
                                        gstreamer::FlowSuccess::Ok
                                    })
                                    .map_err(|_| gstreamer::FlowError::Error)
                            })
                    }
                })
                .build(),
        );

        Ok(Self {
            pipeline,
            last_frame,
        })
    }

    /// Starts the stream capture pipeline and processes messages on the bus.
    pub fn start(&self) -> Result<(), StreamCaptureError> {
        self.pipeline.set_state(gstreamer::State::Playing)?;

        let bus = self
            .pipeline
            .bus()
            .ok_or_else(|| StreamCaptureError::BusError)?;

        // handle bus messages
        bus.set_sync_handler(|_bus, _msg| gstreamer::BusSyncReply::Pass);

        Ok(())
    }

    /// Grabs the last captured image frame.
    ///
    /// # Returns
    ///
    /// An Option containing the last captured Image or None if no image has been captured yet.
    pub fn grab(&mut self) -> Result<Option<Image<u8, 3>>, StreamCaptureError> {
        let mut last_frame = self
            .last_frame
            .lock()
            .map_err(|_| StreamCaptureError::LockError)?;

        last_frame.take().map_or(Ok(None), |frame_buffer| {
            // TODO: solve the zero copy issue
            // https://discourse.gstreamer.org/t/zero-copy-video-frames/3856/2
            let img = Image::<u8, 3>::new(
                [frame_buffer.width, frame_buffer.height].into(),
                frame_buffer.buffer.to_owned(),
            )
            .map_err(|_| StreamCaptureError::CreateImageFrameError)?;

            Ok(Some(img))
        })
    }

    /// Closes the stream capture pipeline.
    pub fn close(&self) -> Result<(), StreamCaptureError> {
        let res = self.pipeline.send_event(gstreamer::event::Eos::new());
        if !res {
            return Err(StreamCaptureError::SendEosError);
        }

        self.pipeline.set_state(gstreamer::State::Null)?;

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
    ) -> Result<FrameBuffer, StreamCaptureError> {
        let sample = appsink.pull_sample()?;

        let caps = sample
            .caps()
            .ok_or_else(|| StreamCaptureError::GetCapsError)?;

        let structure = caps
            .structure(0)
            .ok_or_else(|| StreamCaptureError::GetStructureError)?;

        let height = structure
            .get::<i32>("height")
            .map_err(|_| StreamCaptureError::GetHeightError)? as usize;

        let width = structure
            .get::<i32>("width")
            .map_err(|_| StreamCaptureError::GetWidthError)? as usize;

        let buffer = sample
            .buffer_owned()
            .ok_or_else(|| StreamCaptureError::GetBufferError)?
            .into_mapped_buffer_readable()
            .map_err(|_| StreamCaptureError::GetBufferError)?;

        let frame_buffer = FrameBuffer {
            buffer,
            width,
            height,
        };

        Ok(frame_buffer)
    }
}

impl Drop for StreamCapture {
    /// Ensures that the StreamCapture is properly closed when dropped.
    fn drop(&mut self) {
        self.close().expect("Failed to close StreamCapture");
    }
}
