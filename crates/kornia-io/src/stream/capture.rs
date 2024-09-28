use std::sync::Arc;

use crate::stream::error::StreamCaptureError;
use futures::prelude::*;
use gst::prelude::*;
use kornia_image::{Image, ImageSize};

/// Represents a stream capture pipeline using GStreamer.
pub struct StreamCapture {
    pipeline: gst::Pipeline,
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

        Ok(Self { pipeline })
    }

    /// Runs the stream capture pipeline and processes each frame with the provided function.
    ///
    /// # Arguments
    ///
    /// * `f` - A function that processes each captured image frame.
    ///
    /// # Returns
    ///
    /// A Result indicating success or a StreamCaptureError.
    pub async fn run<F>(&self, mut f: F) -> Result<(), StreamCaptureError>
    where
        F: FnMut(Image<u8, 3>) -> Result<(), Box<dyn std::error::Error>>,
    {
        self.run_internal(
            |img| futures::future::ready(f(img)),
            None::<futures::future::Ready<()>>,
        )
        .await
    }

    /// Runs the stream capture pipeline with a termination signal and processes each frame.
    ///
    /// # Arguments
    ///
    /// * `f` - A function that processes each captured image frame.
    /// * `signal` - A future that, when resolved, will terminate the capture.
    ///
    /// # Returns
    ///
    /// A Result indicating success or a StreamCaptureError.
    pub async fn run_with_termination<F, Fut, S>(
        &self,
        f: F,
        signal: S,
    ) -> Result<(), StreamCaptureError>
    where
        F: FnMut(Image<u8, 3>) -> Fut,
        Fut: Future<Output = Result<(), Box<dyn std::error::Error>>>,
        S: Future<Output = ()>,
    {
        self.run_internal(f, Some(signal)).await
    }

    /// Internal method to run the stream capture pipeline.
    ///
    /// # Arguments
    ///
    /// * `f` - A function that processes each captured image frame.
    /// * `signal` - An optional future that, when resolved, will terminate the capture.
    ///
    /// # Returns
    ///
    /// A Result indicating success or a StreamCaptureError.
    async fn run_internal<F, Fut, S>(
        &self,
        mut f: F,
        signal: Option<S>,
    ) -> Result<(), StreamCaptureError>
    where
        F: FnMut(Image<u8, 3>) -> Fut,
        Fut: Future<Output = Result<(), Box<dyn std::error::Error>>>,
        S: Future<Output = ()>,
    {
        let (tx, mut rx) = tokio::sync::mpsc::channel(10);
        let (signal_tx, mut signal_rx) = tokio::sync::watch::channel(());
        let signal_tx = Arc::new(signal_tx);

        self.setup_pipeline(&tx)?;

        // Start the pipeline
        self.pipeline.set_state(gst::State::Playing)?;

        // Set up bus message handling
        let bus = self
            .pipeline
            .bus()
            .ok_or_else(|| StreamCaptureError::BusError)?;
        self.spawn_bus_handler(bus, signal_tx.clone());

        let mut sig = signal.map(|s| Box::pin(s.fuse()));

        loop {
            tokio::select! {
                img = rx.recv() => {
                    if let Some(img) = img {
                        f(img).await?;
                    } else {
                        break;
                    }
                }
                _ = signal_rx.changed() => {
                    self.close()?;
                    break;
                }
                _ = async { if let Some(ref mut s) = sig { s.as_mut().await } }, if sig.is_some() => {
                    self.close()?;
                    break;
                }
                else => break,
            }
        }

        Ok(())
    }

    /// Sets up the GStreamer pipeline for capturing.
    ///
    /// # Arguments
    ///
    /// * `tx` - A channel sender for transmitting captured image frames.
    ///
    /// # Returns
    ///
    /// A Result indicating success or a StreamCaptureError.
    fn setup_pipeline(
        &self,
        tx: &tokio::sync::mpsc::Sender<Image<u8, 3>>,
    ) -> Result<(), StreamCaptureError> {
        let appsink = self.get_appsink()?;
        self.set_appsink_callbacks(appsink, tx.clone());
        Ok(())
    }

    /// Retrieves the AppSink element from the pipeline.
    ///
    /// # Returns
    ///
    /// A Result containing the AppSink or a StreamCaptureError.
    fn get_appsink(&self) -> Result<gst_app::AppSink, StreamCaptureError> {
        self.pipeline
            .by_name("sink")
            .ok_or_else(|| StreamCaptureError::GetElementByNameError)?
            .dynamic_cast::<gst_app::AppSink>()
            .map_err(StreamCaptureError::DowncastPipelineError)
    }

    /// Sets up callbacks for the AppSink element.
    ///
    /// # Arguments
    ///
    /// * `appsink` - The AppSink element to set callbacks for.
    /// * `tx` - A channel sender for transmitting captured image frames.
    fn set_appsink_callbacks(
        &self,
        appsink: gst_app::AppSink,
        tx: tokio::sync::mpsc::Sender<Image<u8, 3>>,
    ) {
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
    }

    /// Spawns a task to handle GStreamer bus messages.
    ///
    /// This method creates an asynchronous task that listens for messages on the GStreamer bus.
    /// It handles End of Stream (EOS) and Error messages, and signals any errors to the main loop.
    ///
    /// # Arguments
    ///
    /// * `bus` - The GStreamer bus to listen for messages on.
    /// * `signal_tx` - A shared sender to signal when an error occurs or the stream ends.
    ///
    /// # Notes
    ///
    /// This method spawns a Tokio task that runs until an EOS or Error message is received,
    /// or until the bus is closed.
    fn spawn_bus_handler(&self, bus: gst::Bus, signal_tx: Arc<tokio::sync::watch::Sender<()>>) {
        let mut messages = bus.stream();
        tokio::spawn(async move {
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
    }

    /// Closes the stream capture pipeline.
    ///
    /// # Returns
    ///
    /// A Result indicating success or a StreamCaptureError.
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
