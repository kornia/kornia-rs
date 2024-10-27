use std::sync::{Arc, Mutex};

use crate::stream::error::StreamCaptureError;
use gst::prelude::*;
use kornia_image::{Image, ImageSize};

/// Represents a stream capture pipeline using GStreamer.
pub struct StreamCapture {
    pipeline: gst::Pipeline,
    last_frame: Arc<Mutex<Option<Image<u8, 3>>>>,
    running: bool,
    handle: Option<std::thread::JoinHandle<()>>,
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
        gst::init()?;

        let pipeline = gst::parse::launch(pipeline_desc)?
            .dynamic_cast::<gst::Pipeline>()
            .map_err(StreamCaptureError::DowncastPipelineError)?;

        let appsink = pipeline
            .by_name("sink")
            .ok_or_else(|| StreamCaptureError::GetElementByNameError)?
            .dynamic_cast::<gst_app::AppSink>()
            .map_err(StreamCaptureError::DowncastPipelineError)?;

        let last_frame = Arc::new(Mutex::new(None));

        appsink.set_callbacks(
            gst_app::AppSinkCallbacks::builder()
                .new_sample({
                    let last_frame = last_frame.clone();
                    move |sink| match Self::extract_image_frame(sink) {
                        Ok(frame) => {
                            // SAFETY: we have a lock on the last_frame
                            *last_frame.lock().unwrap() = Some(frame);
                            Ok(gst::FlowSuccess::Ok)
                        }
                        Err(_) => Err(gst::FlowError::Error),
                    }
                })
                .build(),
        );

        Ok(Self {
            pipeline,
            last_frame,
            running: false,
            handle: None,
        })
    }

    /// Starts the stream capture pipeline and processes messages on the bus.
    pub fn start(&mut self) -> Result<(), StreamCaptureError> {
        self.pipeline.set_state(gst::State::Playing)?;
        self.running = true;

        let bus = self
            .pipeline
            .bus()
            .ok_or_else(|| StreamCaptureError::BusError)?;

        let handle = std::thread::spawn(move || {
            for msg in bus.iter_timed(gst::ClockTime::NONE) {
                use gst::MessageView;
                match msg.view() {
                    MessageView::Eos(..) => {
                        break;
                    }
                    MessageView::Error(err) => {
                        eprintln!(
                            "Error from {:?}: {} ({:?})",
                            msg.src().map(|s| s.path_string()),
                            err.error(),
                            err.debug()
                        );
                        break;
                    }
                    _ => (),
                }
            }
        });

        self.handle = Some(handle);

        Ok(())
    }

    /// Grabs the last captured image frame.
    ///
    /// # Returns
    ///
    /// An Option containing the last captured Image or None if no image has been captured yet.
    pub fn grab(&self) -> Result<Option<Image<u8, 3>>, StreamCaptureError> {
        if !self.running {
            return Err(StreamCaptureError::PipelineNotRunning);
        }

        // SAFETY: we have a lock on the last_frame
        Ok(self.last_frame.lock().unwrap().take())
    }

    /// Closes the stream capture pipeline.
    pub fn close(&mut self) -> Result<(), StreamCaptureError> {
        let res = self.pipeline.send_event(gst::event::Eos::new());
        if !res {
            return Err(StreamCaptureError::SendEosError);
        }

        if let Some(handle) = self.handle.take() {
            handle.join().expect("Failed to join thread");
        }

        self.pipeline.set_state(gst::State::Null)?;
        self.running = false;
        Ok(())
    }

    /// Extracts an image frame from the AppSink.
    ///
    /// # Arguments
    ///
    /// * `appsink` - The AppSink to extract the frame from.
    ///
    /// # Returns
    ///
    /// A Result containing the extracted Image or a StreamCaptureError.
    fn extract_image_frame(appsink: &gst_app::AppSink) -> Result<Image<u8, 3>, StreamCaptureError> {
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
            .buffer()
            .ok_or_else(|| StreamCaptureError::GetBufferError)?
            .map_readable()?;

        Image::<u8, 3>::new(ImageSize { width, height }, buffer.as_slice().to_vec())
            .map_err(|_| StreamCaptureError::CreateImageFrameError)
    }
}

impl Drop for StreamCapture {
    /// Ensures that the StreamCapture is properly closed when dropped.
    fn drop(&mut self) {
        self.close().expect("Failed to close StreamCapture");
    }
}
