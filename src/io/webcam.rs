use crate::image::{Image, ImageSize};
use anyhow::Result;
use gst::prelude::*;
use thiserror::Error;

/// An error type for WebcamCapture
#[derive(Error, Debug)]
pub enum StreamCaptureError {
    #[error("Failed to initialize GStreamer")]
    GStreamerError(#[from] gst::glib::Error),

    #[error("Failed to downcast pipeline")]
    DowncastPipelineError(gst::Element),

    #[error("Failed to downcast appsink")]
    DowncastAppSinkError,

    #[error("Failed to get the bus")]
    BusError,

    #[error("Failed to set the pipeline state")]
    SetPipelineStateError(#[from] gst::StateChangeError),

    #[error("Failed to pull sample from appsink")]
    PullSampleError(#[from] gst::glib::BoolError),

    #[error("Failed to get the caps from the sample")]
    GetCapsError,

    #[error("Failed to get the structure")]
    GetStructureError,

    // TODO: figure out the #[from] macro for this error
    #[error("Failed to get the height from the structure")]
    GetHeightError,

    // TODO: figure out the #[from] macro for this error
    #[error("Failed to get the width from the structure")]
    GetWidthError,

    #[error("Failed to get the buffer from the sample")]
    GetBufferError,

    #[error("Failed to create an image frame")]
    CreateImageFrameError,

    // TODO: support later on ImageError
    #[error("Failed processing the image frame")]
    ProcessImageFrameError(#[from] Box<dyn std::error::Error>),

    #[error("Failed to send eos event")]
    SendEosError,

    #[error("Pipeline cancelled by the user")]
    PipelineCancelled,
}

/// A builder for creating a WebcamCapture object
pub struct WebcamCaptureBuilder {
    camera_id: usize,
    size: Option<ImageSize>,
    fps: u32,
}

impl WebcamCaptureBuilder {
    /// Creates a new WebcamCaptureBuilder object with default values.
    ///
    /// Note: The default camera id is 0 and the default image size is None
    ///
    /// # Returns
    ///
    /// A WebcamCaptureBuilder object
    pub fn new() -> Self {
        Self {
            camera_id: 0,
            size: None,
            fps: 30,
        }
    }

    /// Sets the camera id for the WebcamCaptureBuilder.
    ///
    /// # Arguments
    ///
    /// * `camera_id` - The desired camera id
    pub fn camera_id(mut self, camera_id: usize) -> Self {
        self.camera_id = camera_id;
        self
    }

    /// Sets the image size for the WebcamCaptureBuilder.
    ///
    /// # Arguments
    ///
    /// * `size` - The desired image size
    pub fn with_size(mut self, size: ImageSize) -> Self {
        self.size = Some(size);
        self
    }

    /// Sets the frames per second for the WebcamCaptureBuilder.
    ///
    /// # Arguments
    ///
    /// * `fps` - The desired frames per second
    pub fn with_fps(mut self, fps: u32) -> Self {
        self.fps = fps;
        self
    }

    /// Create a new [`WebcamCapture`] object.
    pub fn build(self) -> Result<WebcamCapture, StreamCaptureError> {
        WebcamCapture::new(self.camera_id, self.size, self.fps)
    }
}

impl Default for WebcamCaptureBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// A webcam capture object that grabs frames from the camera
/// using GStreamer.
///
/// # Example
///
/// ```no_run
/// use kornia_rs::{image::ImageSize, io::webcam::WebcamCaptureBuilder};
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///   // create a webcam capture object with camera id 0
///   // and force the image size to 640x480
///   let mut webcam = WebcamCaptureBuilder::new()
///     .camera_id(0)
///     .with_fps(30)
///     .with_size(ImageSize {
///       width: 640,
///       height: 480,
///   })
///   .build()?;
///
///   // start grabbing frames from the camera
///   webcam.run(|img| {
///     println!("Image: {:?}", img.size());
///     Ok(())
///   }).await?;
///
///   Ok(())
/// }
/// ```
pub struct WebcamCapture {
    pipeline: gst::Pipeline,
    receiver: tokio::sync::mpsc::Receiver<Image<u8, 3>>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl WebcamCapture {
    /// Creates a new WebcamCapture object.
    ///
    /// # Arguments
    ///
    /// * `camera_id` - The camera id used for capturing images
    /// * `size` - The image size used for resizing directly from the camera
    ///
    /// # Returns
    ///
    /// A WebcamCapture object
    fn new(
        camera_id: usize,
        size: Option<ImageSize>,
        fps: u32,
    ) -> Result<Self, StreamCaptureError> {
        // initialize GStreamer
        gst::init()?;

        // create a pipeline specified by the camera id and size
        let pipeline_str = Self::gst_pipeline_string(camera_id, size, fps);

        let pipeline = gst::parse::launch(&pipeline_str)?
            .dynamic_cast::<gst::Pipeline>()
            .map_err(|e| StreamCaptureError::DowncastPipelineError(e))?;

        let appsink = pipeline
            .by_name("sink")
            .ok_or_else(|| StreamCaptureError::DowncastAppSinkError)?
            .dynamic_cast::<gst_app::AppSink>()
            .map_err(|e| StreamCaptureError::DowncastPipelineError(e))?;

        // the sender and receiver for the image frames
        let (tx, rx) = tokio::sync::mpsc::channel(10);

        appsink.set_callbacks(
            gst_app::AppSinkCallbacks::builder()
                .new_sample(move |sink| match Self::extract_image_frame(sink) {
                    Ok(frame) => {
                        if tx.blocking_send(frame).is_err() {
                            Err(gst::FlowError::Error)
                        } else {
                            Ok(gst::FlowSuccess::Ok)
                        }
                    }
                    Err(_) => Err(gst::FlowError::Error),
                })
                .build(),
        );

        Ok(Self {
            pipeline,
            receiver: rx,
            handle: None,
        })
    }

    /// Runs the webcam capture object and grabs frames from the camera
    ///
    /// # Arguments
    ///
    /// * `f` - A function that takes an image frame
    pub async fn run<F>(&mut self, f: F) -> Result<(), StreamCaptureError>
    where
        F: Fn(Image<u8, 3>) -> Result<(), Box<dyn std::error::Error>>,
    {
        // start the pipeline
        let pipeline = &self.pipeline;
        pipeline.set_state(gst::State::Playing)?;

        let bus = pipeline.bus().ok_or_else(|| StreamCaptureError::BusError)?;

        // start a thread to handle the messages from the bus
        let handle = std::thread::spawn(move || {
            for msg in bus.iter_timed(gst::ClockTime::NONE) {
                use gst::MessageView;
                match msg.view() {
                    MessageView::Eos(..) => break,
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

        // start grabbing frames from the camera
        while let Some(img) = self.receiver.recv().await {
            f(img)?;
        }

        Ok(())
    }

    /// Closes the webcam capture object
    pub fn close(&mut self) -> Result<(), StreamCaptureError> {
        let res = self.pipeline.send_event(gst::event::Eos::new());
        if !res {
            return Err(StreamCaptureError::SendEosError);
        }
        self.handle.take().map(|h| h.join());
        self.pipeline.set_state(gst::State::Null)?;
        Ok(())
    }

    /// Returns a GStreamer pipeline string for the given camera id and size
    ///
    /// # Arguments
    ///
    /// * `camera_id` - The camera id
    /// * `size` - The image size to capture
    /// * `fps` - The desired frames per second
    ///
    /// # Returns
    ///
    /// A GStreamer pipeline string
    fn gst_pipeline_string(camera_id: usize, size: Option<ImageSize>, fps: u32) -> String {
        let video_resize = if let Some(size) = size {
            format!("! video/x-raw,width={},height={} ", size.width, size.height)
        } else {
            "".to_string()
        };

        format!(
            "v4l2src device=/dev/video{} {}! videorate ! video/x-raw,framerate={}/1 ! videoconvert ! video/x-raw,format=RGB ! appsink name=sink",
            camera_id, video_resize, fps
        )
    }

    /// Extracts an image frame from the appsink
    ///
    /// # Arguments
    ///
    /// * `appsink` - The AppSink
    ///
    /// # Returns
    ///
    /// An image frame
    fn extract_image_frame(appsink: &gst_app::AppSink) -> Result<Image<u8, 3>, StreamCaptureError> {
        // pull the sample from the appsink
        let sample = appsink.pull_sample()?;

        let caps = sample
            .caps()
            .ok_or_else(|| StreamCaptureError::GetCapsError)?;

        let structure = caps
            .structure(0)
            .ok_or_else(|| StreamCaptureError::GetStructureError)?;

        // get the image size
        let height = structure
            .get::<i32>("height")
            .map_err(|_| StreamCaptureError::GetHeightError)? as usize;

        let width = structure
            .get::<i32>("width")
            .map_err(|_| StreamCaptureError::GetWidthError)? as usize;

        // get the buffer from the sample
        let buffer = sample
            .buffer()
            .ok_or_else(|| StreamCaptureError::GetBufferError)?
            .map_readable()?;

        // create an image frame
        Image::<u8, 3>::new(ImageSize { width, height }, buffer.as_slice().to_vec())
            .map_err(|_| StreamCaptureError::CreateImageFrameError)
    }
}

impl Drop for WebcamCapture {
    fn drop(&mut self) {
        self.close().expect("Failed to close webcam");
    }
}
