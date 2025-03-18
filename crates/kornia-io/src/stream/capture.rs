use std::sync::{Arc, Mutex};

use crate::stream::error::StreamCaptureError;
use gst::prelude::*;
use kornia_image::Image;

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
                // Convert NV12 to RGB - note the Self:: prefix
                Self::nv12_to_rgb(
                    &frame_buffer.buffer,
                    frame_buffer.width,
                    frame_buffer.height,
                )?
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

    // Helper function to convert NV12 to RGB
    fn nv12_to_rgb(
        buffer: &[u8],
        width: usize,
        height: usize,
    ) -> Result<Image<u8, 3>, StreamCaptureError> {
        // Allocate memory for RGB output
        let mut rgb_data = vec![0u8; width * height * 3];
        
        // Implement NV12 to RGB conversion
        // NV12 format: Y plane followed by interleaved U and V planes
        let y_plane_size = width * height;
        let uv_plane_offset = y_plane_size;
        
        for y in 0..height {
            for x in 0..width {
                let y_index = y * width + x;
                let y_value = buffer[y_index] as f32;
                
                // UV values are subsampled (one U,V pair per 2x2 Y values)
                let uv_index = uv_plane_offset + (y / 2) * width + (x / 2) * 2;
                let u_value = buffer[uv_index] as f32 - 128.0;
                let v_value = buffer[uv_index + 1] as f32 - 128.0;
                
                // Standard YUV to RGB conversion
                let r = y_value + 1.402 * v_value;
                let g = y_value - 0.344136 * u_value - 0.714136 * v_value;
                let b = y_value + 1.772 * u_value;
                
                // Clamp values to 0-255 range
                let r = r.max(0.0).min(255.0) as u8;
                let g = g.max(0.0).min(255.0) as u8;
                let b = b.max(0.0).min(255.0) as u8;
                
                // Write to output buffer
                let rgb_index = y_index * 3;
                rgb_data[rgb_index] = r;
                rgb_data[rgb_index + 1] = g;
                rgb_data[rgb_index + 2] = b;
            }
        }
        
        Image::<u8, 3>::new(
            [width, height].into(),
            rgb_data,
        ).map_err(|_| StreamCaptureError::CreateImageFrameError)
    }
}

impl Drop for StreamCapture {
    /// Ensures that the StreamCapture is properly closed when dropped.
    fn drop(&mut self) {
        self.close().expect("Failed to close StreamCapture");
    }
}
