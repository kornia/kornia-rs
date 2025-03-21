use std::sync::{Arc, Mutex};

use crate::stream::error::StreamCaptureError;
use gst::prelude::*;
use kornia_image::Image;
use kornia_imgproc::color::nv12_to_rgb;

// utility struct to store the frame buffer
struct FrameBuffer {
    buffer: gst::MappedBuffer<gst::buffer::Readable>,
    sample: gst::Sample,
    width: usize,
    height: usize,
}

/// Represents a stream capture pipeline using GStreamer.
pub struct StreamCapture {
    pipeline: gst::Pipeline,
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
                    move |sink| {
                        last_frame
                            .lock()
                            .map_err(|_| gst::FlowError::Error)
                            .and_then(|mut guard| {
                                Self::extract_frame_buffer(sink)
                                    .map(|frame_buffer| {
                                        guard.replace(frame_buffer);
                                        gst::FlowSuccess::Ok
                                    })
                                    .map_err(|_| gst::FlowError::Error)
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
        self.pipeline.set_state(gst::State::Playing)?;

        let bus = self
            .pipeline
            .bus()
            .ok_or_else(|| StreamCaptureError::BusError)?;

        // handle bus messages
        bus.set_sync_handler(|_bus, _msg| gst::BusSyncReply::Pass);

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
            // Get the format from the caps
            let caps = frame_buffer.sample.caps().ok_or_else(|| StreamCaptureError::GetCapsError)?;
            let structure = caps.structure(0).ok_or_else(|| StreamCaptureError::GetStructureError)?;
            
            // Check if the format is NV12 or RGB
            let format = structure.get::<String>("format")
                .map_err(|_| StreamCaptureError::GetFormatError)?;
            
            let img = if format == "NV12" {
                // Convert NV12 to RGB 
                nv12_to_rgb::<StreamCaptureError>(
                    frame_buffer.buffer.as_slice(),
                    frame_buffer.width,
                    frame_buffer.height,
                ).map_err(|_| StreamCaptureError::CreateImageFrameError)?
            } else {
                // Assume RGB format
                Image::<u8, 3>::new(
                    [frame_buffer.width, frame_buffer.height].into(),
                    frame_buffer.buffer.to_owned(),
                ).map_err(|_| StreamCaptureError::CreateImageFrameError)?
            };
    
            Ok(Some(img))
        })
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

    /// Extracts a frame buffer from the AppSink.
    ///
    /// # Arguments
    ///
    /// * `appsink` - The AppSink to extract the frame buffer from.
    ///
    /// # Returns
    ///
    /// A Result containing the extracted FrameBuffer or a StreamCaptureError.
    fn extract_frame_buffer(appsink: &gst_app::AppSink) -> Result<FrameBuffer, StreamCaptureError> {
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
            sample: sample.clone(),
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
