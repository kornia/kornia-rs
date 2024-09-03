use crate::stream::error::StreamCaptureError;
use futures::prelude::*;
use gst::prelude::*;
use kornia_image::{Image, ImageSize};

/// A camera capture object that grabs frames from a GStreamer pipeline.
///
/// # Example
///
/// ```no_run
/// use kornia::image::Image;
/// use kornia::io::stream::StreamCapture;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///   // create a capture object to grab frames from a camera
///   let mut capture = StreamCapture::new(
///     "v4l2src device=/dev/video0 ! videoconvert ! video/x-raw,format=RGB ! appsink name=sink")?;
///
///   // start grabbing frames from the camera
///   capture.run(|img: Image<u8, 3>| {
///     println!("Image: {:?}", img.size());
///     Ok(())
///   }).await?;
///
///   Ok(())
/// }
/// ```
pub struct StreamCapture {
    pipeline: gst::Pipeline,
    // TODO: pass Image<u8, 3> as a generic type
    //receiver: tokio::sync::mpsc::Receiver<Image<u8, 3>>,
    handle: Option<tokio::task::JoinHandle<()>>,
}

impl StreamCapture {
    /// Creates a new StreamCapture object
    ///
    /// NOTE: The pipeline description should contain an appsink element with the name "sink".
    ///
    /// # Arguments
    ///
    /// * `pipeline_desc` - The GStreamer pipeline description.
    ///
    /// # Returns
    ///
    /// A StreamCapture object
    pub fn new(pipeline_desc: &str) -> Result<Self, StreamCaptureError> {
        // initialize GStreamer
        gst::init()?;

        let pipeline = gst::parse::launch(pipeline_desc)?
            .dynamic_cast::<gst::Pipeline>()
            .map_err(StreamCaptureError::DowncastPipelineError)?;

        // TODO: this block can be defined in the run method so that we can pass to the sender and receiver
        // a generic type.

        //let appsink = pipeline
        //    .by_name("sink")
        //    .ok_or_else(|| StreamCaptureError::DowncastAppSinkError)?
        //    .dynamic_cast::<gst_app::AppSink>()
        //    .map_err(StreamCaptureError::DowncastPipelineError)?;

        //// the sender and receiver for the image frames
        //let (tx, rx) = tokio::sync::mpsc::channel(10);

        //appsink.set_callbacks(
        //    gst_app::AppSinkCallbacks::builder()
        //        .new_sample(move |sink| match Self::extract_image_frame(sink) {
        //            Ok(frame) => {
        //                if tx.blocking_send(frame).is_err() {
        //                    Err(gst::FlowError::Error)
        //                } else {
        //                    Ok(gst::FlowSuccess::Ok)
        //                }
        //            }
        //            Err(_) => Err(gst::FlowError::Error),
        //        })
        //        .build(),
        //);

        Ok(Self {
            pipeline,
            //receiver: rx,
            handle: None,
        })
    }

    /// Runs the capture object and grabs frames from the source
    ///
    /// # Arguments
    ///
    /// * `f` - A function that takes an image frame
    // TODO: implement run_with_shutdown to pass a shutdown signal to the capture object
    pub async fn run<F>(&mut self, mut f: F) -> Result<(), StreamCaptureError>
    where
        F: FnMut(Image<u8, 3>) -> Result<(), Box<dyn std::error::Error>>,
    {
        // the sender and receiver for the image frames
        let (tx, mut rx) = tokio::sync::mpsc::channel(10);

        let appsink = self
            .pipeline
            .by_name("sink")
            .ok_or_else(|| StreamCaptureError::DowncastAppSinkError)?
            .dynamic_cast::<gst_app::AppSink>()
            .map_err(StreamCaptureError::DowncastPipelineError)?;

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

        // start the pipeline
        self.pipeline.set_state(gst::State::Playing)?;

        let bus = self
            .pipeline
            .bus()
            .ok_or_else(|| StreamCaptureError::BusError)?;

        //// start a thread to handle the messages from the bus
        let mut messages = bus.stream();

        //let (err_tx, mut err_rx) = tokio::sync::mpsc::channel(1);
        let (signal_tx, mut signal_rx) = tokio::sync::watch::channel(());

        let handle = tokio::spawn(async move {
            while let Some(msg) = messages.next().await {
                use gst::MessageView;
                match msg.view() {
                    MessageView::Eos(..) => {
                        println!("EOS");
                        break;
                    }
                    MessageView::Error(err) => {
                        eprintln!(
                            "Error from {:?}: {} ({:?})",
                            msg.src().map(|s| s.path_string()),
                            err.error(),
                            err.debug()
                        );

                        let _ = signal_tx.send(());

                        break;
                    }
                    _ => (),
                }
            }
        });

        // NOTE: no clear that we need to keep the handle to avoid the stream capture to be killed by signal
        self.handle = Some(handle);

        //// start grabbing frames from the source and close the capture object if an error occurs

        loop {
            tokio::select! {
                Some(img) = rx.recv() => {
                    f(img)?;
                }
                _ = signal_rx.changed() => {
                    self.close()?;
                    return Err(StreamCaptureError::PipelineCancelled);
                }
                else => break,
            }
        }

        Ok(())
    }

    /// Closes the capture object
    pub fn close(&mut self) -> Result<(), StreamCaptureError> {
        let res = self.pipeline.send_event(gst::event::Eos::new());
        if !res {
            return Err(StreamCaptureError::SendEosError);
        }
        if let Some(handle) = self.handle.take() {
            handle.abort()
        }
        self.pipeline.set_state(gst::State::Null)?;
        Ok(())
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

impl Drop for StreamCapture {
    fn drop(&mut self) {
        self.close().expect("Failed to close webcam");
    }
}
