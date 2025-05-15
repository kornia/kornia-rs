use crate::stream::error::StreamCaptureError;
use circular_buffer::CircularBuffer;
use gstreamer::prelude::*;
use kornia_image::Image;
use kornia_tensor::{
    storage::TensorStorage, tensor::get_strides_from_shape, CpuAllocator, ParentDeallocator, Tensor,
};
use std::sync::{Arc, Mutex};

#[allow(dead_code)]
pub(crate) struct GstParentDeallocator(gstreamer::Buffer);

impl ParentDeallocator for GstParentDeallocator {
    fn dealloc(&self) {
        // When gstreamer::Buffer will be dropped, it will automatically
        // reduce the reference count as this memory is managed by gstreamer
    }
}

// utility struct to store the frame buffer
struct FrameBuffer {
    buffer: gstreamer::Buffer,
    width: i32,
    height: i32,
}

/// Represents a stream capture pipeline using GStreamer.
pub struct StreamCapture {
    pipeline: gstreamer::Pipeline,
    circular_buffer: Arc<Mutex<CircularBuffer<5, FrameBuffer>>>,
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

        let circular_buffer = Arc::new(Mutex::new(CircularBuffer::new()));

        appsink.set_callbacks(
            gstreamer_app::AppSinkCallbacks::builder()
                .new_sample({
                    let circular_buffer = circular_buffer.clone();
                    move |sink| {
                        Self::extract_frame_buffer(sink)
                            .map_err(|_| gstreamer::FlowError::Eos)
                            .and_then(|frame_buffer| {
                                let mut guard = circular_buffer
                                    .lock()
                                    .map_err(|_| gstreamer::FlowError::Error)?;
                                guard.push_back(frame_buffer);
                                Ok(gstreamer::FlowSuccess::Ok)
                            })
                    }
                })
                .build(),
        );

        Ok(Self {
            pipeline,
            circular_buffer,
        })
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

    /// Grabs the last captured image frame.
    ///
    /// # Returns
    ///
    /// An Option containing the last captured Image or None if no image has been captured yet.
    pub fn grab(&mut self) -> Result<Option<Image<u8, 3>>, StreamCaptureError> {
        let mut circular_buffer = self
            .circular_buffer
            .lock()
            .map_err(|_| StreamCaptureError::MutexPoisonError)?;
        if let Some(mut frame_buffer) = circular_buffer.pop_front() {
            let width = frame_buffer.width as usize;
            let height = frame_buffer.height as usize;

            // Create a mapping of the buffer without moving it out of frame_buffer
            let mut buffer_map = frame_buffer
                .buffer
                .make_mut()
                .map_writable()
                .map_err(|_| StreamCaptureError::GetBufferError)?;

            let frame_data_slice = buffer_map.as_mut_slice();
            let frame_data_ptr = frame_data_slice.as_mut_ptr();

            let length = frame_data_slice.len();
            let shape = [height, width, 3];
            let strides = get_strides_from_shape(shape);

            // Drop the buffer_map as it is a reference of Buffer
            drop(buffer_map);

            let gst_parent_deallocator = Arc::new(GstParentDeallocator(frame_buffer.buffer));

            let tensor = unsafe {
                Tensor {
                    shape,
                    strides,
                    storage: TensorStorage::from_raw_parts(
                        frame_data_ptr,
                        length,
                        CpuAllocator::with_parent_relation(gst_parent_deallocator),
                    ),
                }
            };

            let image = Image(tensor);
            return Ok(Some(image));
        }
        Ok(None)
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
    ) -> Result<FrameBuffer, StreamCaptureError> {
        let sample = appsink.pull_sample()?;

        let caps = sample.caps().ok_or_else(|| {
            StreamCaptureError::GetCapsError("Failed to get the caps".to_string())
        })?;

        let structure = caps.structure(0).ok_or_else(|| {
            StreamCaptureError::GetCapsError("Failed to get the structure".to_string())
        })?;

        let height = structure
            .get::<i32>("height")
            .map_err(|e| StreamCaptureError::GetCapsError(e.to_string()))?;

        let width = structure
            .get::<i32>("width")
            .map_err(|e| StreamCaptureError::GetCapsError(e.to_string()))?;

        let buffer = sample
            .buffer_owned()
            .ok_or_else(|| StreamCaptureError::GetBufferError)?;

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

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::blocking::get;
    use std::{fs::File, path::PathBuf, time::Duration};
    use tempfile::{tempdir, TempDir};

    const FILE_NAME: &str = "video.mp4";
    const VIDEO_LINK: &str =
        "https://github.com/kornia/tutorials/raw/refs/heads/master/data/sharpening.mp4";

    fn download_video<'a>() -> (PathBuf, TempDir) {
        let response = get(VIDEO_LINK).expect("Failed to download video");
        let temp_dir = tempdir().expect("Failed to create temp directory");
        let temp_file_path = temp_dir.path().join(FILE_NAME);
        let mut temp_file = File::create(&temp_file_path).expect("Failed to create temp file");

        std::io::copy(
            &mut response.bytes().expect("Failed to read response").as_ref(),
            &mut temp_file,
        )
        .expect("Failed to write video to temp file");

        println!("Video downloaded to: {:?}", temp_file_path);
        (temp_file_path, temp_dir)
    }

    #[test]
    fn test_image_mutability() {
        let (video_path, _temp_dir) = download_video();

        // Mock pipeline description for testing
        let pipeline_desc = format!(
            "filesrc location=\"{}\" ! decodebin ! videoconvert ! video/x-raw,format=RGB ! appsink name=sink sync=true",
            video_path.to_str().unwrap()
        );

        // Create a new StreamCapture instance
        let mut stream_capture =
            StreamCapture::new(&pipeline_desc).expect("Failed to create StreamCapture");

        // Start the stream capture
        stream_capture
            .start()
            .expect("Failed to start StreamCapture");

        std::thread::sleep(Duration::from_secs(1));

        // Grab an image frame
        let mut image = stream_capture
            .grab()
            .expect("Failed to grab image")
            .expect("No image captured");

        // Modify the image tensor data
        let tensor = &mut image.0; // Access the tensor inside the Image
        let data = tensor.as_slice_mut();

        // Modify the first pixel's RGB values
        data[0] = 255; // Red
        data[1] = 0; // Green
        data[2] = 0; // Blue

        // Verify the modification
        assert_eq!(data[0], 255);
        assert_eq!(data[1], 0);
        assert_eq!(data[2], 0);

        // Close the stream capture
        stream_capture
            .close()
            .expect("Failed to close StreamCapture");
    }
}
