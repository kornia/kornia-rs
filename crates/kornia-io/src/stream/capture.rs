use crate::stream::error::StreamCaptureError;
use gst::prelude::*;
use kornia_image::Image;

/// Represents a stream capture pipeline using GStreamer.
pub struct StreamCapture {
    pipeline: gst::Pipeline,
    appsink: gst_app::AppSink,
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

        Ok(Self { pipeline, appsink })
    }

    /// Starts the stream capture pipeline and processes messages on the bus.
    pub fn start(&self) -> Result<(), StreamCaptureError> {
        self.pipeline.set_state(gst::State::Playing)?;

        let bus = self
            .pipeline
            .bus()
            .ok_or_else(|| StreamCaptureError::BusError)?;

        // handle bus messages
        bus.set_sync_handler(|_bus, msg| {
            println!("msg_type: {:?}", msg.view());
            match msg.view() {
                gst::MessageView::Eos(eos) => {
                    eprintln!("eos message: {:?}", eos);
                }
                gst::MessageView::Error(e) => {
                    eprintln!("error message: {:?}", e);
                }
                _ => (),
            }
            gst::BusSyncReply::Pass
        });

        Ok(())
    }

    /// Grabs the last captured image frame.
    ///
    /// # Returns
    ///
    /// An Option containing the last captured Image or None if no image has been captured yet.
    pub fn grab(&self) -> Result<Option<Image<u8, 3>>, StreamCaptureError> {
        self.appsink
            .try_pull_sample(gst::ClockTime::ZERO)
            .map(Self::extract_image_frame)
            .transpose()
    }

    /// Closes the stream capture pipeline.
    pub fn close(&self) -> Result<(), StreamCaptureError> {
        let res = self.pipeline.send_event(gst::event::Eos::new());
        if !res {
            return Err(StreamCaptureError::SendEosError);
        }

        self.pipeline.set_state(gst::State::Null)?;

        Ok(())
    }

    /// Extracts an image frame from the AppSink.
    ///
    /// # Arguments
    ///
    /// * `sample` - The sample to extract the frame from.
    ///
    /// # Returns
    ///
    /// A Result containing the extracted Image or a StreamCaptureError.
    fn extract_image_frame(sample: gst::Sample) -> Result<Image<u8, 3>, StreamCaptureError> {
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

        Image::<u8, 3>::new([width, height].into(), buffer.to_owned())
            .map_err(|_| StreamCaptureError::CreateImageFrameError)
    }
}

impl Drop for StreamCapture {
    /// Ensures that the StreamCapture is properly closed when dropped.
    fn drop(&mut self) {
        self.close().expect("Failed to close StreamCapture");
    }
}
