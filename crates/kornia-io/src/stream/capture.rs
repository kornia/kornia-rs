use super::GstAllocator;
use crate::stream::error::StreamCaptureError;
use circular_buffer::CircularBuffer;
use gstreamer::prelude::*;
use kornia_image::{Image, ImageSize};
use std::sync::{Arc, Mutex};

// utility struct to store the frame buffer
struct FrameBuffer {
    buffer: gstreamer::Buffer,
    width: i32,
    height: i32,
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
    circular_buffer: Arc<Mutex<CircularBuffer<5, FrameBuffer>>>,
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

        let circular_buffer = Arc::new(Mutex::new(CircularBuffer::new()));
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
    /// NOTE: the image is grabbed as readable buffer, so you must be careful when modifying the
    /// image data as would cause undefined behavior.
    ///
    /// # Returns
    ///
    /// An Option containing the last captured Image or None if no image has been captured yet.
    pub fn grab(&mut self) -> Result<Option<Image<u8, 3, GstAllocator>>, StreamCaptureError> {
        let mut circular_buffer = self
            .circular_buffer
            .lock()
            .map_err(|_| StreamCaptureError::MutexPoisonError)?;

        let Some(frame_buffer) = circular_buffer.pop_front() else {
            return Ok(None);
        };

        // unpack the frame buffer
        let width = frame_buffer.width;
        let height = frame_buffer.height;
        let buffer = frame_buffer.buffer;

        let mapped_buffer = buffer
            .into_mapped_buffer_readable()
            .map_err(|_| StreamCaptureError::GetBufferError)?;

        let data_ptr = mapped_buffer.as_ptr();
        let data_len = mapped_buffer.len();

        // We are using custom `GstAllocator` and storing `gstreamer::MappedBuffer`, as the buffer
        // is reference counted storage maintained by gstreamer and when it is dropped the
        // `data_ptr` becomes dangling. To avoid this, we are keeping the `MappedBuffer` with
        // `Image`.
        let alloc = GstAllocator(mapped_buffer.into_buffer());

        let image = unsafe {
            Image::from_raw_parts(
                ImageSize {
                    width: width as usize,
                    height: height as usize,
                },
                data_ptr,
                data_len,
                alloc,
            )
            .map_err(StreamCaptureError::ImageError)
        }?;

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

        let structure = caps.structure(0).ok_or_else(|| {
            StreamCaptureError::GetCapsError("Failed to get the structure".to_string())
        })?;

        let height = structure
            .get::<i32>("height")
            .map_err(|e| StreamCaptureError::GetCapsError(e.to_string()))?;

        let width = structure
            .get::<i32>("width")
            .map_err(|e| StreamCaptureError::GetCapsError(e.to_string()))?;

        let fps = structure
            .get::<gstreamer::Fraction>("framerate")
            .map_err(|e| StreamCaptureError::GetCapsError(e.to_string()))?;

        let buffer = sample
            .buffer_owned()
            .ok_or_else(|| StreamCaptureError::GetBufferError)?;

        let frame_buffer = FrameBuffer {
            buffer,
            width,
            height,
        };

        Ok((frame_buffer, fps))
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

    fn download_video() -> (PathBuf, TempDir) {
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
    #[ignore = "need gstreamer in CI"]
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
