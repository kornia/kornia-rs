use std::sync::{Arc, Mutex};

use crate::stream::error::StreamCaptureError;
use gst::prelude::*;
use kornia_image::Image;

/// Utility struct to hold the frame buffer data for the last captured frame.
#[derive(Debug, Clone)]
struct FrameBuffer {
    data: *const u8,
    len: usize,
    cols: usize,
    rows: usize,
}

unsafe impl Send for FrameBuffer {}
unsafe impl Sync for FrameBuffer {}

struct BufferPool {
    buffers: Vec<Option<FrameBuffer>>,
    last_active_buffer_index: usize,
}

impl BufferPool {
    const BUFFER_POOL_SIZE: usize = 4;

    pub fn new() -> Self {
        Self {
            buffers: vec![None; Self::BUFFER_POOL_SIZE],
            last_active_buffer_index: 0,
        }
    }

    pub fn enqueue(&mut self, buffer: FrameBuffer) {
        self.buffers[self.last_active_buffer_index] = Some(buffer);
        self.last_active_buffer_index =
            (self.last_active_buffer_index + 1) % Self::BUFFER_POOL_SIZE;
    }

    pub fn dequeue(&mut self) -> Option<&FrameBuffer> {
        let buffer = &self.buffers[self.last_active_buffer_index];
        if buffer.is_none() {
            return None;
        }

        let buffer = buffer.as_ref();
        buffer
    }

    pub fn print_debug(&self) {
        println!(">>> [StreamCapture] active buffers");
        for i in 0..self.buffers.len() {
            if let Some(buffer) = &self.buffers[i] {
                println!("index: {}, {:?}", i, buffer.data);
            }
        }
        println!(">>>");
    }
}

/// Represents a stream capture pipeline using GStreamer.
pub struct StreamCapture {
    pipeline: gst::Pipeline,
    running: bool,
    handle: Option<std::thread::JoinHandle<()>>,
    buffer_pool: Arc<Mutex<BufferPool>>,
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

        // create a buffer pool
        let buffer_pool = Arc::new(Mutex::new(BufferPool::new()));

        appsink.set_callbacks(
            gst_app::AppSinkCallbacks::builder()
                .new_sample({
                    let buffer_pool = buffer_pool.clone();
                    move |sink| match Self::extract_image_frame(sink) {
                        Ok(new_frame) => {
                            // SAFETY: we have a lock on the buffer_pool
                            if let Ok(mut buffer_pool) = buffer_pool.lock() {
                                buffer_pool.enqueue(new_frame);
                            }

                            Ok(gst::FlowSuccess::Ok)
                        }
                        Err(_) => Err(gst::FlowError::Error),
                    }
                })
                .build(),
        );

        Ok(Self {
            pipeline,
            running: false,
            handle: None,
            buffer_pool,
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
    pub fn grab(&mut self) -> Result<Option<Image<u8, 3>>, StreamCaptureError> {
        if !self.running {
            return Err(StreamCaptureError::PipelineNotRunning);
        }

        // SAFETY: we have a lock on the buffer_pool
        let mut buffer_pool = self.buffer_pool.lock().unwrap();
        let Some(frame_buffer) = buffer_pool.dequeue() else {
            return Ok(None);
        };

        let frame_buffer = std::mem::ManuallyDrop::new(frame_buffer);

        // SAFETY: this operation is safe because we know the frame_buffer is valid
        let image = unsafe {
            Image::from_raw_parts(
                [frame_buffer.cols, frame_buffer.rows].into(),
                frame_buffer.data,
                frame_buffer.len,
            )
            .map_err(|_| StreamCaptureError::CreateImageFrameError)?
        };

        //buffer_pool.print_debug();

        Ok(Some(image))
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
    /// A Result containing the extracted FrameBuffer or a StreamCaptureError.
    fn extract_image_frame(appsink: &gst_app::AppSink) -> Result<FrameBuffer, StreamCaptureError> {
        let sample = appsink.pull_sample().map_err(|_| gst::FlowError::Eos)?;

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

        //println!(
        //    "[StreamCapture] {:?} successfully mapped buffer",
        //    buffer.as_ptr()
        //);

        // SAFETY: we need to forget the buffer because we are not going to drop it
        let buffer = std::mem::ManuallyDrop::new(buffer);

        let frame_buffer = FrameBuffer {
            data: buffer.as_ptr(),
            len: buffer.len(),
            cols: width,
            rows: height,
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
