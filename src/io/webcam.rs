use crate::image::{Image, ImageSize};
use anyhow::Result;
use gst::prelude::*;

/// A builder for creating a WebcamCapture object
pub struct WebcamCaptureBuilder {
    camera_id: usize,
    size: Option<ImageSize>,
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

    /// Create a new [`WebcamCapture`] object.
    pub fn build(self) -> Result<WebcamCapture> {
        WebcamCapture::new(self.camera_id, self.size)
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
    handle: Vec<tokio::task::JoinHandle<()>>,
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
    fn new(camera_id: usize, size: Option<ImageSize>) -> Result<Self> {
        gst::init()?;

        // create a pipeline specified by the camera id and size
        let pipeline_str = Self::gst_pipeline_string(camera_id, size);
        let pipeline = gst::parse::launch(&pipeline_str)?
            .downcast::<gst::Pipeline>()
            .map_err(|_| anyhow::anyhow!("Failed to downcast pipeline"))?;

        let appsink = pipeline
            .by_name("sink")
            .ok_or_else(|| anyhow::anyhow!("Failed to get sink"))?
            .dynamic_cast::<gst_app::AppSink>()
            .map_err(|_| anyhow::anyhow!("Failed to cast to AppSink"))?;

        let (tx, rx) = tokio::sync::mpsc::channel(50);

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
            handle: vec![],
        })
    }

    /// Runs the webcam capture object and grabs frames from the camera
    ///
    /// # Arguments
    ///
    /// * `f` - A function that takes an image frame
    pub async fn run<F>(&mut self, f: F) -> Result<()>
    where
        F: Fn(Image<u8, 3>) -> Result<()>,
    {
        // start the pipeline
        let pipeline = &self.pipeline;
        pipeline.set_state(gst::State::Playing)?;

        let bus = pipeline
            .bus()
            .ok_or_else(|| anyhow::anyhow!("Failed to get bus"))?;

        // start a thread to handle the messages from the bus
        //let handle = std::thread::spawn(move || {
        let handle = tokio::task::spawn(async move {
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
        self.handle.push(handle);

        // start grabbing frames from the camera
        while let Some(img) = self.receiver.recv().await {
            f(img)?;
        }

        Ok(())
    }

    pub async fn close(&mut self) -> Result<()> {
        self.pipeline.send_event(gst::event::Eos::new());
        while let Some(h) = self.handle.pop() {
            h.await?;
        }
        self.pipeline.set_state(gst::State::Null)?;
        Ok(())
    }

    /// Returns a GStreamer pipeline string for the given camera id and size
    ///
    /// # Arguments
    ///
    /// * `camera_id` - The camera id
    /// * `size` - The image size
    ///
    /// # Returns
    ///
    /// A GStreamer pipeline string
    fn gst_pipeline_string(camera_id: usize, size: Option<ImageSize>) -> String {
        let video_resize = if let Some(size) = size {
            format!(
                " ! video/x-raw,width={},height={},framerate=30/1",
                size.width, size.height
            )
        } else {
            "".to_string()
        };

        format!(
            "v4l2src device=/dev/video{} {}! videoconvert ! videoscale ! video/x-raw,format=RGB ! appsink name=sink",
            camera_id, video_resize
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
    fn extract_image_frame(appsink: &gst_app::AppSink) -> Result<Image<u8, 3>> {
        let sample = appsink.pull_sample()?;
        let caps = sample
            .caps()
            .ok_or_else(|| anyhow::anyhow!("Failed to get caps from sample"))?;
        let structure = caps
            .structure(0)
            .ok_or_else(|| anyhow::anyhow!("Failed to get structure"))?;
        let height = structure.get::<i32>("height")? as usize;
        let width = structure.get::<i32>("width")? as usize;

        let buffer = sample
            .buffer()
            .ok_or_else(|| anyhow::anyhow!("Failed to get buffer from sample"))?;
        let map = buffer.map_readable()?;
        Image::<u8, 3>::new(ImageSize { width, height }, map.as_slice().to_vec())
    }
}
