use std::path::Path;
use std::sync::{Arc, Mutex};

use gst::prelude::*;
use gst::{ClockTime, Fraction}; // Import Fraction

use kornia_image::{Image, ImageSize};

use super::StreamCaptureError;

/// The video codec used for encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoCodec {
    /// H.264 (AVC) codec. Widely compatible, good for MP4, MKV.
    H264,
    /// VP9 codec. Good quality, often used with WebM.
    VP9,
    /// AV1 codec. Newer, high efficiency, often used with MKV, WebM.
    AV1,
    // TODO: Add HEVC(H.265), FFV1 (lossless), etc.
}

/// The container format for the output video file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoContainer {
    /// MP4 container (.mp4). Common, good compatibility (especially with H.264).
    MP4,
    /// Matroska container (.mkv). Flexible, supports many codecs (H.264, VP9, AV1).
    MKV,
    /// WebM container (.webm). Primarily for web use, typically with VP9 or AV1.
    WebM,
    // TODO: Add AVI, etc.
}

/// The pixel format of the image data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFormat {
    /// 8-bit RGB format (Red, Green, Blue).
    Rgb8,
    /// 8-bit BGR format (Blue, Green, Red). Common in OpenCV.
    Bgr8,
    /// 8-bit monochrome (grayscale) format.
    Mono8,
    // TODO: Add RGBA, BGRA, YUV formats, float formats
}

impl ImageFormat {
    /// Returns the number of channels for the format.
    fn channels(&self) -> usize {
        match self {
            ImageFormat::Rgb8 => 3,
            ImageFormat::Bgr8 => 3,
            ImageFormat::Mono8 => 1,
        }
    }

    /// Returns the GStreamer format string.
    fn to_gst_format_str(&self) -> &'static str {
        match self {
            ImageFormat::Rgb8 => "RGB",
            ImageFormat::Bgr8 => "BGR",
            ImageFormat::Mono8 => "GRAY8",
        }
    }

    /// Creates an ImageFormat from a GStreamer format string.
    fn from_gst_format_str(format_str: &str) -> Result<Self, StreamCaptureError> {
        match format_str {
            "RGB" => Ok(ImageFormat::Rgb8),
            "BGR" => Ok(ImageFormat::Bgr8),
            "GRAY8" => Ok(ImageFormat::Mono8),
            unsupported => Err(StreamCaptureError::InvalidImageFormat(format!(
                "Unsupported GStreamer format negotiated: {}",
                unsupported
            ))),
        }
    }
}

/// A struct for writing video files using GStreamer.
pub struct VideoWriter {
    pipeline: gst::Pipeline,
    appsrc: gst_app::AppSrc,
    fps: i32,
    format: ImageFormat,
    counter: u64,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl VideoWriter {
    /// Create a new VideoWriter.
    ///
    /// # Arguments
    ///
    /// * `path` - The path to save the video file.
    /// * `codec` - The video codec to use for encoding.
    /// * `container` - The container format for the output file.
    /// * `format` - The expected pixel format of the input images.
    /// * `fps` - The frames per second of the output video.
    /// * `size` - The width and height of the video frames.
    ///
    /// # Errors
    ///
    /// Returns an error if GStreamer initialization fails, the pipeline cannot be
    /// constructed (e.g., missing plugins, invalid codec/container combination),
    /// or elements cannot be found or configured.
    pub fn new(
        path: impl AsRef<Path>,
        codec: VideoCodec,
        container: VideoContainer,
        format: ImageFormat,
        fps: i32,
        size: ImageSize,
    ) -> Result<Self, StreamCaptureError> {
        gst::init()?;

        let path_str = path.as_ref().to_string_lossy();
        let format_str = format.to_gst_format_str();

        // Select GStreamer elements based on codec and container
        let (encoder_str, parser_str, muxer_str) = match (codec, container) {
            (VideoCodec::H264, VideoContainer::MP4) => ("x264enc", "h264parse", "mp4mux"),
            (VideoCodec::H264, VideoContainer::MKV) => ("x264enc", "h264parse", "matroskamux"),
            // VP9 typically uses WebM or MKV
            (VideoCodec::VP9, VideoContainer::WebM) => ("vp9enc", "vp9parse", "webmmux"),
            (VideoCodec::VP9, VideoContainer::MKV) => ("vp9enc", "vp9parse", "matroskamux"),
            // AV1 typically uses MKV or WebM
            (VideoCodec::AV1, VideoContainer::MKV) => ("av1enc", "av1parse", "matroskamux"),
            (VideoCodec::AV1, VideoContainer::WebM) => ("av1enc", "av1parse", "webmmux"),
            // Add more combinations or return error for unsupported ones
            _ => {
                return Err(StreamCaptureError::InvalidConfig(format!(
                    "Unsupported codec ({:?}) and container ({:?}) combination.",
                    codec, container
                )))
            }
        };

        // Construct the pipeline string dynamically
        // We convert to I420 (a common YUV format) before encoding, as many encoders prefer/require it.
        // TODO: Allow specifying encoder parameters (bitrate, quality/CRF)
        let pipeline_str = format!(
            "appsrc name=src ! \
            videoconvert ! video/x-raw,format=I420 ! \
            {encoder_str} ! \
            {parser_str} ! \
            {muxer_str} ! \
            filesink location=\"{path}\"",
            src = "src", // Use named parameters for clarity
            encoder_str = encoder_str,
            parser_str = parser_str,
            muxer_str = muxer_str,
            path = path_str // Ensure path is properly quoted if it contains spaces etc.
        );

        log::debug!("GStreamer writer pipeline: {}", pipeline_str);

        let pipeline = gst::parse::launch(&pipeline_str)?
            .dynamic_cast::<gst::Pipeline>()
            .map_err(|_| StreamCaptureError::DowncastPipelineError("Failed to downcast pipeline".to_string()))?;

        let appsrc = pipeline
            .by_name("src")
            .ok_or_else(|| StreamCaptureError::GetElementByNameError("Failed to get appsrc element".to_string()))?
            .dynamic_cast::<gst_app::AppSrc>()
            .map_err(|_| StreamCaptureError::DowncastPipelineError("Failed to downcast appsrc".to_string()))?;

        // Configure AppSrc
        appsrc.set_format(gst::Format::Time);

        let caps = gst::Caps::builder("video/x-raw")
            .field("format", format_str)
            .field("width", size.width as i32)
            .field("height", size.height as i32)
            .field("framerate", Fraction::new(fps, 1))
            .build();

        appsrc.set_caps(Some(&caps));
        appsrc.set_is_live(true); // Treat as a live source
        appsrc.set_property("block", false); // Non-blocking push-buffer

        Ok(Self {
            pipeline,
            appsrc,
            fps,
            format,
            counter: 0,
            handle: None,
        })
    }

    /// Start the video writer pipeline.
    ///
    /// Sets the pipeline state to Playing and starts a background thread
    /// to monitor the GStreamer bus for messages (EOS, errors).
    ///
    /// # Errors
    ///
    /// Returns an error if the pipeline state cannot be set or the bus cannot be retrieved.
    pub fn start(&mut self) -> Result<(), StreamCaptureError> {
        if self.handle.is_some() {
            log::warn!("VideoWriter already started.");
            return Ok(());
        }

        self.pipeline.set_state(gst::State::Playing)?;
        log::info!("VideoWriter pipeline set to Playing.");

        let pipeline_weak = self.pipeline.downgrade(); // Use weak reference for thread
        let bus = self.pipeline.bus().ok_or(StreamCaptureError::BusError)?;

        let handle = std::thread::spawn(move || {
            for msg in bus.iter_timed(ClockTime::NONE) {
                let _pipeline = match pipeline_weak.upgrade() {
                    Some(p) => p,
                    None => {
                        log::debug!("Pipeline already dropped, exiting bus thread.");
                        break;
                    }
                };
                match msg.view() {
                    gst::MessageView::Eos(..) => {
                        log::debug!("Writer bus received EOS");
                        break; // End of stream received, exit thread
                    }
                    gst::MessageView::Error(err) => {
                        log::error!(
                            "Writer bus error from {:?}: {} ({:?})",
                            msg.src().map(|s| s.path_string()),
                            err.error(),
                            err.debug()
                        );
                        // Optional: Set pipeline to Null on error? Depends on desired behavior.
                        // if let Some(p) = pipeline_weak.upgrade() {
                        //     let _ = p.set_state(gst::State::Null);
                        // }
                        break; // Exit thread on error
                    }
                    gst::MessageView::Warning(warn) => {
                         log::warn!(
                            "Writer bus warning from {:?}: {} ({:?})",
                            msg.src().map(|s| s.path_string()),
                            warn.error(),
                            warn.debug()
                        );
                    }
                    _ => {} // Ignore other messages
                }
            }
            log::debug!("VideoWriter bus monitoring thread finished.");
        });

        self.handle = Some(handle);
        Ok(())
    }

    /// Writes a single image frame to the video file.
    ///
    /// # Arguments
    ///
    /// * `img` - The image to write. Its format (number of channels) must match
    ///           the `ImageFormat` specified during `VideoWriter::new`.
    ///
    /// # Errors
    ///
    /// Returns an error if the image format is incorrect, the buffer cannot be
    /// created, or GStreamer fails to accept the buffer (e.g., pipeline error,
    /// buffer queue full in non-blocking mode).
    pub fn write<const C: usize>(&mut self, img: &Image<u8, C>) -> Result<(), StreamCaptureError> {
        // Check if the image channels match the expected format
        if C != self.format.channels() {
            return Err(StreamCaptureError::InvalidImageFormat(format!(
                "Invalid number of channels: expected {}, got {}. Format is {:?}.",
                self.format.channels(),
                C,
                self.format
            )));
        }

        // Check image dimensions match (optional, but good practice)
        // if img.size() != self.size { ... }

        // TODO: Explore zero-copy options if performance becomes critical.
        // This involves potentially allocating GStreamer buffers and writing
        // directly into their mapped memory, or using custom allocators.
        // For now, copy the data.
        let mut buffer = gst::Buffer::from_mut_slice(img.as_slice().to_vec());

        // Calculate Presentation Timestamp (PTS) and Duration
        // Duration is 1/fps
        let frame_duration_ns = 1_000_000_000 / self.fps as u64;
        let pts = ClockTime::from_nseconds(self.counter * frame_duration_ns);
        let duration = ClockTime::from_nseconds(frame_duration_ns);

        // Set buffer metadata
        let buffer_ref = buffer.get_mut().ok_or_else(|| StreamCaptureError::InvalidConfig("Failed to get mutable buffer reference".to_string()))?;
        buffer_ref.set_pts(Some(pts));
        buffer_ref.set_duration(Some(duration));
        // buffer_ref.set_offset(self.counter); // Optional: Set frame number as offset

        self.counter += 1;

        // Push the buffer into the appsrc
        // This might return an error if the pipeline is not PLAYING,
        // or if the downstream elements cannot keep up (if block=false).
        self.appsrc.push_buffer(buffer).map_err(|flow_ret| {
            log::error!("Failed to push buffer into appsrc: {:?}", flow_ret);
            StreamCaptureError::WriteError(format!("Failed to push buffer: {:?}", flow_ret))
        })?;

        Ok(())
    }

    /// Finalizes the video file.
    ///
    /// Sends an End-Of-Stream (EOS) signal to the pipeline, waits for the
    /// processing to complete (joins the bus monitoring thread), and sets
    /// the pipeline state to Null.
    ///
    /// This should be called explicitly when done writing. It is also called
    /// implicitly when the `VideoWriter` is dropped.
    ///
    /// # Errors
    ///
    /// Returns an error if sending EOS fails, the bus thread fails to join,
    /// or setting the pipeline state to Null fails.
    pub fn close(&mut self) -> Result<(), StreamCaptureError> {
        if self.handle.is_none() {
            // Already closed or never started
            return Ok(());
        }
        log::debug!("Closing VideoWriter...");

        // Send End-Of-Stream signal to appsrc. This signals downstream elements
        // that no more data will be pushed.
        self.appsrc.end_of_stream()?;
        log::debug!("EOS sent to appsrc.");

        // Wait for the bus monitoring thread to finish.
        // This thread should exit when it sees the EOS message propagated through the pipeline.
        if let Some(handle) = self.handle.take() {
            if let Err(e) = handle.join() {
                log::error!("VideoWriter bus thread panicked: {:?}", e);
                // Decide if this should be a hard error or just a warning
                // return Err(StreamCaptureError::InternalError("Bus thread panicked".to_string()));
            } else {
                 log::debug!("VideoWriter bus thread joined successfully.");
            }
        }

        // Set the pipeline state to Null to release resources.
        self.pipeline.set_state(gst::State::Null)?;
        log::info!("VideoWriter pipeline set to Null.");

        Ok(())
    }
}

impl Drop for VideoWriter {
    /// Ensures `close()` is called when the `VideoWriter` goes out of scope.
    fn drop(&mut self) {
        if self.handle.is_some() {
            if let Err(e) = self.close() {
                log::error!("Error closing video writer in drop: {}", e);
                // Avoid panicking in drop, just log the error.
            }
        }
    }
}

// ==========================================================================
// VideoReader
// ==========================================================================

/// A struct for reading video files using GStreamer.
///
/// Decodes video files frame by frame and provides them as `kornia_image::Image`.
pub struct VideoReader {
    pipeline: gst::Pipeline,
    appsink: gst_app::AppSink,
    fps: f64,
    format: ImageFormat,
    size: ImageSize,
    duration: Option<ClockTime>, // Store duration after querying
    is_eos: Arc<Mutex<bool>>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl VideoReader {
    /// Create a new VideoReader for the given path.
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the video file to read.
    ///
    /// # Errors
    ///
    /// Returns an error if GStreamer initialization fails, the pipeline cannot be
    /// constructed (e.g., file not found, invalid format, missing plugins),
    /// or elements cannot be found or configured.
    pub fn new(path: impl AsRef<Path>) -> Result<Self, StreamCaptureError> {
        gst::init()?;

        let path = path.as_ref().to_owned();
        if !path.exists() {
             return Err(StreamCaptureError::FileNotFound(path.to_string_lossy().to_string()));
        }
        let path_str = path.to_string_lossy();


        // Create a pipeline using decodebin for automatic format detection.
        // Request RGB, BGR, or GRAY8 output from videoconvert.
        // Use emit-signals=true on appsink to get notified about new samples.
        // sync=false allows processing frames as fast as possible.
        let pipeline_str = format!(
            "filesrc location=\"{path}\" ! \
            decodebin ! \
            videoconvert ! \
            video/x-raw,format=(string){{RGB,BGR,GRAY8}} ! \
            appsink name=sink emit-signals=true sync=false",
             path = path_str // Ensure path is properly quoted
        );

        log::debug!("GStreamer reader pipeline: {}", pipeline_str);

        let pipeline = gst::parse::launch(&pipeline_str)?
            .dynamic_cast::<gst::Pipeline>()
            .map_err(|_| StreamCaptureError::DowncastPipelineError("Failed to downcast pipeline".to_string()))?;

        let appsink = pipeline
            .by_name("sink")
            .ok_or_else(|| StreamCaptureError::GetElementByNameError("Failed to get appsink element".to_string()))?
            .dynamic_cast::<gst_app::AppSink>()
            .map_err(|_| StreamCaptureError::DowncastPipelineError("Failed to downcast appsink".to_string()))?;

        // Appsink configuration (optional but good practice)
        appsink.set_property("emit-signals", true);
        // appsink.set_property("max-buffers", 10); // Limit buffer queue size
        // appsink.set_property("drop", true); // Drop old buffers if queue is full

        // Initialize with placeholder values; they will be updated in start()
        let reader = Self {
            pipeline,
            appsink,
            fps: 0.0,
            format: ImageFormat::Rgb8, // Default, will be updated
            size: ImageSize { width: 0, height: 0 },
            duration: None,
            is_eos: Arc::new(Mutex::new(false)),
            handle: None,
        };

        Ok(reader)
    }

    /// Start the video reader pipeline and determine video properties.
    ///
    /// Sets the pipeline state to Playing, waits for negotiation to complete,
    /// queries video properties (size, format, fps, duration), and starts a
    /// background thread to monitor the GStreamer bus.
    ///
    /// # Errors
    ///
    /// Returns an error if the pipeline state cannot be set, properties cannot
    /// be queried (e.g., invalid video file, negotiation timeout), or the bus
    /// cannot be retrieved.
    pub fn start(&mut self) -> Result<(), StreamCaptureError> {
         if self.handle.is_some() {
            log::warn!("VideoReader already started.");
            return Ok(());
        }

        // Set pipeline to Playing to start data flow and negotiation
        self.pipeline.set_state(gst::State::Playing)?;
        log::info!("VideoReader pipeline set to Playing.");

        // Wait for the pipeline to reach the Playing state or encounter an error.
        // This ensures negotiation has happened or failed.
        let state_change = self.pipeline.state(ClockTime::from_seconds(10)); // Increased timeout
        match state_change {
             Ok((_state, current, _pending)) if current == gst::State::Playing => {
                 log::debug!("Pipeline reached PLAYING state successfully.");
             }
             Ok((_state, current, _pending)) => {
                 log::error!("Pipeline failed to reach PLAYING state, current state: {:?}", current);
                 // Try to get error from bus
                 if let Some(bus) = self.pipeline.bus() {
                     if let Some(msg) = bus.timed_pop_filtered(ClockTime::from_mseconds(100), gst::MessageType::ERROR) {
                         let err_msg = msg.view().error().map(|e| format!("{} ({:?})", e.error(), e.debug())).unwrap_or_else(|| "Unknown error".to_string());
                         self.pipeline.set_state(gst::State::Null)?; // Cleanup
                         return Err(StreamCaptureError::PipelineError(format!("Failed to start pipeline: {}", err_msg)));
                     }
                 }
                 self.pipeline.set_state(gst::State::Null)?; // Cleanup
                 return Err(StreamCaptureError::PipelineError(format!(
                    "Pipeline failed to reach PLAYING state (current: {:?}) within timeout.", current
                 )));
             }
             Err(err) => {
                 log::error!("Failed to get pipeline state: {:?}", err);
                 self.pipeline.set_state(gst::State::Null)?; // Cleanup
                 return Err(StreamCaptureError::PipelineError(format!(
                    "Failed to get pipeline state: {:?}", err
                 )));
             }
        }


        // Query properties from the negotiated caps on the appsink's sink pad
        let sinkpad = self.appsink.static_pad("sink")
            .ok_or_else(|| StreamCaptureError::InternalError("Failed to get appsink sink pad".to_string()))?;

        // It might take a moment for caps to be available after PLAYING state
        let caps = sinkpad.current_caps().or_else(|| {
            log::warn!("Caps not immediately available, waiting briefly...");
            std::thread::sleep(std::time::Duration::from_millis(100));
            sinkpad.current_caps()
        }).ok_or_else(|| StreamCaptureError::InvalidConfig("Failed to get negotiated caps from appsink pad.".to_string()))?;

        log::debug!("Negotiated caps: {}", caps.to_string());

        let structure = caps.structure(0).ok_or_else(|| {
            StreamCaptureError::InvalidConfig("Failed to get structure from caps".to_string())
        })?;

        // Extract video properties
        let width = structure.get::<i32>("width")?;
        let height = structure.get::<i32>("height")?;
        self.size = ImageSize {
            width: width as usize,
            height: height as usize,
        };

        let format_str = structure.get::<String>("format")?;
        self.format = ImageFormat::from_gst_format_str(&format_str)?;

        if let Ok(fps_frac) = structure.get::<Fraction>("framerate") {
            if fps_frac.denom() != 0 {
                self.fps = fps_frac.numer() as f64 / fps_frac.denom() as f64;
            } else {
                 log::warn!("Invalid framerate fraction (denominator is 0), defaulting to 0.0");
                 self.fps = 0.0; // Or query duration and frame count if available
            }
        } else {
            log::warn!("Could not determine video framerate from caps, defaulting to 0.0");
            self.fps = 0.0; // Consider querying duration/frame count later if needed
        }

        log::info!(
            "Video properties: Size={}x{}, Format={:?}, FPS={:.2}",
            self.size.width, self.size.height, self.format, self.fps
        );

        // Query duration (can sometimes fail, so store Option)
        self.duration = self.pipeline.query_duration::<ClockTime>();
        if let Some(dur) = self.duration {
             log::info!("Video duration: {:.2} seconds", dur.seconds_f64());
        } else {
             log::warn!("Could not determine video duration.");
        }


        // Set up bus message handling in a separate thread
        let bus = self.pipeline.bus().ok_or(StreamCaptureError::BusError)?;
        let is_eos_clone = self.is_eos.clone();
        let pipeline_weak = self.pipeline.downgrade(); // Use weak reference

        let handle = std::thread::spawn(move || {
            for msg in bus.iter_timed(ClockTime::NONE) {
                 let _pipeline = match pipeline_weak.upgrade() {
                    Some(p) => p,
                    None => {
                        log::debug!("Pipeline already dropped, exiting bus thread.");
                        break;
                    }
                };
                match msg.view() {
                    gst::MessageView::Eos(..) => {
                        log::debug!("Reader bus received EOS");
                        let mut is_eos = is_eos_clone.lock().unwrap();
                        *is_eos = true;
                        break; // Exit thread on EOS
                    }
                    gst::MessageView::Error(err) => {
                        log::error!(
                            "Reader bus error from {:?}: {} ({:?})",
                            msg.src().map(|s| s.path_string()),
                            err.error(),
                            err.debug()
                        );
                        // Also set EOS flag on error to stop reading attempts
                        let mut is_eos = is_eos_clone.lock().unwrap();
                        *is_eos = true;
                        break; // Exit thread on error
                    }
                     gst::MessageView::Warning(warn) => {
                         log::warn!(
                            "Reader bus warning from {:?}: {} ({:?})",
                            msg.src().map(|s| s.path_string()),
                            warn.error(),
                            warn.debug()
                        );
                    }
                    _ => {} // Ignore other messages
                }
            }
             log::debug!("VideoReader bus monitoring thread finished.");
        });

        self.handle = Some(handle);

        Ok(())
    }

    /// Read the next frame from the video.
    ///
    /// Pulls a sample from the appsink, extracts the image data and its
    /// presentation timestamp (PTS).
    ///
    /// # Returns
    ///
    /// * `Ok(Some((image, pts)))` - If a frame was successfully read.
    /// * `Ok(None)` - If the end of the stream (EOS) has been reached.
    /// * `Err(StreamCaptureError)` - If an error occurred during reading (e.g.,
    ///   pipeline error, buffer mapping failed, format mismatch, timeout).
    ///
    /// # Type Parameters
    ///
    /// * `C` - The expected number of channels for the output `Image`. This *must*
    ///         match the number of channels of the `ImageFormat` determined during `start()`.
    pub fn read<const C: usize>(&mut self) -> Result<Option<(Image<u8, C>, ClockTime)>, StreamCaptureError> {
        // Check EOS flag first
        if *self.is_eos.lock().unwrap() {
            log::trace!("read() called after EOS detected.");
            return Ok(None);
        }

        // Check if the requested channel count C matches the video's actual format
        if C != self.format.channels() {
            return Err(StreamCaptureError::InvalidImageFormat(format!(
                "Invalid number of channels requested: expected {} (for format {:?}), got {}",
                self.format.channels(),
                self.format,
                C
            )));
        }

        // Pull a sample from appsink. Use try_pull_sample for non-blocking,
        // or pull_sample for blocking behavior. Add a reasonable timeout.
        // let timeout = gst::ClockTime::from_seconds(5); // 5 second timeout
        // match self.appsink.try_pull_sample(timeout) {
        match self.appsink.pull_sample() { // Blocking pull
            Ok(sample) => {
                // Got a sample, process it
                let buffer = sample.buffer().ok_or_else(|| {
                    StreamCaptureError::ReadError("Failed to get buffer from sample".to_string())
                })?;

                // Get Presentation Timestamp (PTS)
                let pts = buffer.pts().unwrap_or(ClockTime::NONE); // Use NONE if PTS is missing

                // Map the buffer to access data
                let map = buffer.map_readable().map_err(|_| {
                    StreamCaptureError::ReadError("Failed to map buffer readable".to_string())
                })?;

                let data = map.as_slice();

                // Verify buffer size (optional but recommended)
                let expected_size = self.size.width * self.size.height * C;
                if data.len() < expected_size {
                     return Err(StreamCaptureError::ReadError(format!(
                        "Buffer size mismatch: expected at least {} bytes, got {} bytes",
                        expected_size,
                        data.len()
                    )));
                } else if data.len() > expected_size {
                    // This can happen due to padding/stride, log a warning if significant
                     log::trace!(
                        "Buffer size larger than expected ({} > {}), using expected size.",
                        data.len(), expected_size
                    );
                }


                // Create kornia_image::Image by copying the data
                // TODO: Explore zero-copy if needed, potentially returning a wrapper
                // around the GstMapInfo instead of a new Image.
                let img = Image::<u8, C>::new(self.size, data[..expected_size].to_vec())?;

                Ok(Some((img, pts)))
            }
            // Ok(None) => { // Only for try_pull_sample with timeout
            //     // Timeout occurred, check if EOS happened concurrently
            //     if *self.is_eos.lock().unwrap() {
            //         Ok(None)
            //     } else {
            //         Err(StreamCaptureError::ReadError(
            //             "Timeout while waiting for frame".to_string(),
            //         ))
            //     }
            // }
            Err(gst::FlowError::Eos) => {
                // End of stream reached
                log::debug!("Appsink pull_sample returned EOS.");
                let mut is_eos = self.is_eos.lock().unwrap();
                *is_eos = true;
                Ok(None)
            }
            Err(err) => {
                // Handle other GStreamer flow errors
                log::error!("Appsink pull_sample error: {:?}", err);
                 // Also set EOS flag on error to stop reading attempts
                let mut is_eos = self.is_eos.lock().unwrap();
                *is_eos = true;
                Err(StreamCaptureError::ReadError(format!(
                    "GStreamer pull_sample error: {:?}",
                    err
                )))
            }
        }
    }

    /// Get the frames per second (FPS) of the video.
    ///
    /// This value is determined after `start()` is called.
    pub fn fps(&self) -> f64 {
        self.fps
    }

    /// Get the size (width, height) of the video frames.
    ///
    /// This value is determined after `start()` is called.
    pub fn size(&self) -> ImageSize {
        self.size
    }

    /// Get the pixel format of the video frames.
    ///
    /// This value is determined after `start()` is called.
    pub fn format(&self) -> ImageFormat {
        self.format // Return by value as it's Copy
    }

    /// Check if the end of the video stream has been reached.
    pub fn is_eos(&self) -> bool {
        *self.is_eos.lock().unwrap()
    }

    /// Close the video reader and release resources.
    ///
    /// Sets the pipeline state to Null and waits for the bus monitoring thread
    /// to complete.
    ///
    /// This should be called explicitly when done reading. It is also called
    /// implicitly when the `VideoReader` is dropped.
    ///
    /// # Errors
    ///
    /// Returns an error if setting the pipeline state fails or the bus thread
    /// fails to join.
    pub fn close(&mut self) -> Result<(), StreamCaptureError> {
        if self.handle.is_none() {
             // Already closed or never started
            return Ok(());
        }
        log::debug!("Closing VideoReader...");

        // Set pipeline to Null first to stop data flow and signal threads
        self.pipeline.set_state(gst::State::Null)?;
        log::info!("VideoReader pipeline set to Null.");

        // Wait for the bus monitoring thread to finish.
        // Setting state to Null should eventually trigger EOS or an error, causing the thread to exit.
        if let Some(handle) = self.handle.take() {
             if let Err(e) = handle.join() {
                log::error!("VideoReader bus thread panicked: {:?}", e);
                // return Err(StreamCaptureError::InternalError("Bus thread panicked".to_string()));
            } else {
                 log::debug!("VideoReader bus thread joined successfully.");
            }
        }

        // Clear the EOS flag in case close is called before EOS was naturally reached
        *self.is_eos.lock().unwrap() = true;

        Ok(())
    }

    /// Seek to the specified position in the video.
    ///
    /// # Arguments
    ///
    /// * `position_secs` - The target position in seconds from the beginning.
    ///
    /// # Errors
    ///
    /// Returns an error if the seek operation fails (e.g., file not seekable,
    /// invalid position).
    ///
    /// **Note:** Seeking might be inaccurate depending on the video encoding (key frames).
    pub fn seek(&mut self, position_secs: f64) -> Result<(), StreamCaptureError> {
        if position_secs < 0.0 {
            return Err(StreamCaptureError::SeekError("Seek position cannot be negative".to_string()));
        }

        // Convert seconds to GStreamer ClockTime (nanoseconds)
        let position_ns = ClockTime::from_seconds_f64(position_secs);

        // Seek flags:
        // - FLUSH: Discard any data currently in the pipeline before seeking. Important for responsiveness.
        // - KEY_UNIT: Seek to the nearest key frame before the target position (more accurate for some formats).
        // - ACCURATE: Try to seek to the exact frame (can be slow, might not be supported by all formats/demuxers).
        // Using KEY_UNIT is often a good balance.
        let seek_flags = gst::SeekFlags::FLUSH | gst::SeekFlags::KEY_UNIT; // Or gst::SeekFlags::ACCURATE

        log::debug!("Seeking to {} ns ({:.2} s)", position_ns.nseconds(), position_secs);

        // Perform the seek operation
        // seek(rate, flags, start_type, start, stop_type, stop)
        if self.pipeline.seek(
            1.0, // Playback rate (1.0 for normal)
            seek_flags,
            gst::SeekType::Set, // Start position is absolute
            position_ns,
            gst::SeekType::None, // No specific stop position
            ClockTime::NONE,
        ) {
            // Seek command sent successfully. Now wait for it to complete.
            // Querying the position or waiting for state change can help confirm.
            // let state_change = self.pipeline.state(ClockTime::from_seconds(2)); // Wait briefly
            // log::debug!("State after seek attempt: {:?}", state_change);

            // Reset the EOS flag after a successful seek command is issued.
            // The pipeline might send a new EOS if the seek position is at/after the end.
            let mut is_eos = self.is_eos.lock().unwrap();
            *is_eos = false;
            log::debug!("EOS flag reset after seek.");

            // It might be necessary to discard the first frame after seeking if it's inaccurate.
            // Consider adding logic for that if needed.

            Ok(())
        } else {
            log::error!("Seek operation failed to be performed.");
            Err(StreamCaptureError::SeekError("Seek operation failed".to_string()))
        }
    }

    /// Get the total duration of the video in seconds.
    ///
    /// Returns `None` if the duration could not be determined (e.g., live stream,
    /// corrupted file, or query failed). The duration is queried during `start()`.
    pub fn duration_secs(&self) -> Option<f64> {
        self.duration.map(|d| d.seconds_f64())
    }

     /// Get the total duration of the video as a GStreamer `ClockTime`.
    ///
    /// Returns `None` if the duration could not be determined.
    pub fn duration(&self) -> Option<ClockTime> {
        self.duration
    }
}

impl Drop for VideoReader {
     /// Ensures `close()` is called when the `VideoReader` goes out of scope.
    fn drop(&mut self) {
        if self.handle.is_some() { // Check if it was started
            if let Err(e) = self.close() {
                log::error!("Error closing video reader in drop: {}", e);
                // Avoid panicking in drop
            }
        } else {
            // If never started, still ensure pipeline is set to Null if it exists
             let _ = self.pipeline.set_state(gst::State::Null);
        }
    }
}


// ==========================================================================
// Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*; // Import items from parent module
    use kornia_image::{Image, ImageSize};
    use std::path::PathBuf; // Use PathBuf for owned paths

    // Helper function to create a simple test image
    fn create_test_image<const C: usize>(size: ImageSize, frame_num: u8) -> Result<Image<u8, C>, kornia_image::ImageError> {
        let mut data = Vec::with_capacity(size.width * size.height * C);
        for y in 0..size.height {
            for x in 0..size.width {
                if C == 1 { // Mono8
                    data.push(((x + y + frame_num as usize) % 255) as u8);
                } else if C == 3 { // RGB or BGR
                    data.push(((x + frame_num as usize) % 255) as u8); // R or B
                    data.push(((y + frame_num as usize) % 255) as u8); // G
                    data.push(((x + y + frame_num as usize) % 255) as u8); // B or R
                } else {
                    panic!("Unsupported channel count in test helper");
                }
            }
        }
        Image::<u8, C>::new(size, data)
    }

    // Helper to get a temporary file path
    fn temp_video_path(dir: &tempfile::TempDir, filename: &str) -> PathBuf {
         dir.path().join(filename)
    }

    // -- VideoWriter Tests --

    #[ignore = "Requires GStreamer plugins (good, base, x264) installed"]
    #[test]
    fn video_writer_h264_mp4_rgb8() -> Result<(), Box<dyn std::error::Error>> {
        let tmp_dir = tempfile::tempdir()?;
        let file_path = temp_video_path(&tmp_dir, "test_h264_rgb.mp4");
        let size = ImageSize { width: 64, height: 48 };
        let fps = 30;
        let num_frames = 5;

        let mut writer = VideoWriter::new(
            &file_path,
            VideoCodec::H264,
            VideoContainer::MP4,
            ImageFormat::Rgb8,
            fps,
            size,
        )?;
        writer.start()?;

        for i in 0..num_frames {
            let img = create_test_image::<3>(size, i as u8)?;
            writer.write(&img)?;
        }
        writer.close()?;

        assert!(file_path.exists(), "File was not created: {:?}", file_path);
        assert!(std::fs::metadata(&file_path)?.len() > 0, "File is empty");
        Ok(())
    }

     #[ignore = "Requires GStreamer plugins (good, base, x264) installed"]
    #[test]
    fn video_writer_h264_mkv_bgr8() -> Result<(), Box<dyn std::error::Error>> {
        let tmp_dir = tempfile::tempdir()?;
        let file_path = temp_video_path(&tmp_dir, "test_h264_bgr.mkv");
        let size = ImageSize { width: 64, height: 48 };
        let fps = 15;
        let num_frames = 3;

        let mut writer = VideoWriter::new(
            &file_path,
            VideoCodec::H264,
            VideoContainer::MKV,
            ImageFormat::Bgr8, // Test BGR
            fps,
            size,
        )?;
        writer.start()?;

        for i in 0..num_frames {
            let img = create_test_image::<3>(size, i as u8)?; // Create 3-channel image
            writer.write(&img)?;
        }
        writer.close()?;

        assert!(file_path.exists(), "File was not created: {:?}", file_path);
        assert!(std::fs::metadata(&file_path)?.len() > 0, "File is empty");
        Ok(())
    }

    #[ignore = "Requires GStreamer plugins (good, base, x264) installed"]
    #[test]
    fn video_writer_h264_mp4_mono8() -> Result<(), Box<dyn std::error::Error>> {
        let tmp_dir = tempfile::tempdir()?;
        let file_path = temp_video_path(&tmp_dir, "test_h264_mono.mp4");
        let size = ImageSize { width: 32, height: 32 };
        let fps = 25;
        let num_frames = 4;

        let mut writer = VideoWriter::new(
            &file_path,
            VideoCodec::H264,
            VideoContainer::MP4,
            ImageFormat::Mono8, // Test Mono8
            fps,
            size,
        )?;
        writer.start()?;

        for i in 0..num_frames {
            let img = create_test_image::<1>(size, i as u8)?; // Create 1-channel image
            writer.write(&img)?;
        }
        writer.close()?;

        assert!(file_path.exists(), "File was not created: {:?}", file_path);
        assert!(std::fs::metadata(&file_path)?.len() > 0, "File is empty");
        Ok(())
    }

     #[ignore = "Requires GStreamer plugins (good, ugly, libav, vp9) installed"]
    #[test]
    fn video_writer_vp9_webm_rgb8() -> Result<(), Box<dyn std::error::Error>> {
        let tmp_dir = tempfile::tempdir()?;
        let file_path = temp_video_path(&tmp_dir, "test_vp9_rgb.webm");
        let size = ImageSize { width: 80, height: 60 };
        let fps = 20;
        let num_frames = 6;

        let mut writer = VideoWriter::new(
            &file_path,
            VideoCodec::VP9, // Test VP9
            VideoContainer::WebM, // Test WebM
            ImageFormat::Rgb8,
            fps,
            size,
        )?;
        writer.start()?;

        for i in 0..num_frames {
            let img = create_test_image::<3>(size, i as u8)?;
            writer.write(&img)?;
        }
        writer.close()?;

        assert!(file_path.exists(), "File was not created: {:?}", file_path);
        assert!(std::fs::metadata(&file_path)?.len() > 0, "File is empty");
        Ok(())
    }

    // -- VideoReader Tests --

    // This test requires a pre-existing video file.
    // Create one using ffmpeg:
    // ffmpeg -f lavfi -i testsrc=duration=5:size=128x72:rate=30 -pix_fmt rgb24 test_reader_rgb.mp4
    // ffmpeg -f lavfi -i testsrc=duration=5:size=128x72:rate=30 -pix_fmt gray8 test_reader_mono.mp4
    // ffmpeg -f lavfi -i testsrc=duration=5:size=128x72:rate=30 -pix_fmt bgr24 test_reader_bgr.mp4
    #[ignore = "Requires GStreamer plugins and a test video file (e.g., test_reader_rgb.mp4)"]
    #[test]
    fn video_reader_basic_rgb() -> Result<(), Box<dyn std::error::Error>> {
        let test_file = "test_reader_rgb.mp4"; // Adjust path if needed
        if !Path::new(test_file).exists() {
             eprintln!("Skipping test: File '{}' not found.", test_file);
             return Ok(()); // Skip test if file doesn't exist
        }

        let mut reader = VideoReader::new(test_file)?;
        reader.start()?; // Start determines properties

        // Check determined properties (adjust expected values based on test_file)
        assert!(reader.fps() > 29.0 && reader.fps() < 31.0, "Unexpected FPS: {}", reader.fps());
        assert_eq!(reader.size(), ImageSize { width: 128, height: 72 }, "Unexpected size");
        assert_eq!(reader.format(), ImageFormat::Rgb8, "Unexpected format"); // Assuming RGB input
        assert!(reader.duration_secs().is_some(), "Duration should be known");
        assert!(reader.duration_secs().unwrap() > 4.9 && reader.duration_secs().unwrap() < 5.1, "Unexpected duration");

        // Read the first frame
        let frame_data = reader.read::<3>()?; // Request 3 channels for RGB
        assert!(frame_data.is_some(), "Failed to read the first frame");
        let (frame, pts) = frame_data.unwrap();
        assert_eq!(frame.size(), reader.size(), "Frame size mismatch");
        assert!(pts.nseconds() < 100_000_000, "First frame PTS seems too high"); // PTS should be near 0

        println!("First frame PTS: {} ns", pts.nseconds());

        // Read a few more frames
        let mut frame_count = 1;
        while let Some((_img, _pts)) = reader.read::<3>()? {
            frame_count += 1;
            if frame_count >= 5 { break; }
        }
        assert!(frame_count >= 5, "Failed to read multiple frames");

        // Test seeking
        let seek_time = 1.5; // Seek to 1.5 seconds
        reader.seek(seek_time)?;
        assert!(!reader.is_eos(), "EOS flag should be reset after seek");

        // Read frame after seek
        let frame_data_after_seek = reader.read::<3>()?;
        assert!(frame_data_after_seek.is_some(), "Failed to read frame after seeking");
        let (_frame_after_seek, pts_after_seek) = frame_data_after_seek.unwrap();
        println!("Frame PTS after seeking to {}s: {} ns", seek_time, pts_after_seek.nseconds());
        // PTS after seek should be >= seek time (allowing for keyframe inaccuracy)
        assert!(pts_after_seek.seconds_f64() >= seek_time - 0.5, "PTS after seek is too early");


        // Read until EOS
        while reader.read::<3>()?.is_some() {
            // Keep reading
        }
        assert!(reader.is_eos(), "Reader should be EOS after reading all frames");

        // Reading again after EOS should return None
        assert!(reader.read::<3>()?.is_none(), "Read after EOS should return None");

        reader.close()?;

        Ok(())
    }


    // -- Roundtrip Tests --

    #[ignore = "Requires GStreamer plugins (good, base, x264) installed"]
    #[test]
    fn video_reader_writer_roundtrip_h264_mp4_rgb() -> Result<(), Box<dyn std::error::Error>> {
        let tmp_dir = tempfile::tempdir()?;
        let file_path = temp_video_path(&tmp_dir, "roundtrip_h264_rgb.mp4");
        let size = ImageSize { width: 128, height: 72 };
        let fps = 24;
        let num_frames = 10;
        let codec = VideoCodec::H264;
        let container = VideoContainer::MP4;
        let format = ImageFormat::Rgb8;

        // --- Write Phase ---
        let mut writer = VideoWriter::new(&file_path, codec, container, format, fps, size)?;
        writer.start()?;
        let mut first_frame_data: Option<Vec<u8>> = None;
        for i in 0..num_frames {
            let img = create_test_image::<3>(size, i as u8)?;
            if i == 0 {
                first_frame_data = Some(img.data.clone()); // Save first frame data for comparison
            }
            writer.write(&img)?;
        }
        writer.close()?;
        assert!(file_path.exists());

        // --- Read Phase ---
        let mut reader = VideoReader::new(&file_path)?;
        reader.start()?;

        assert_eq!(reader.size(), size, "Read size mismatch");
        // Note: FPS might not be perfectly exact after encoding/decoding
        assert!((reader.fps() - fps as f64).abs() < 1.0, "Read FPS mismatch: expected ~{}, got {}", fps, reader.fps());
        // Format might be converted during encoding (e.g., to YUV) and back during decoding.
        // GStreamer's decodebin usually converts back to the requested format (RGB/BGR/GRAY8).
        assert_eq!(reader.format(), format, "Read format mismatch"); // Check if it decodes back to RGB

        let mut read_count = 0;
        let mut first_read_frame_data: Option<Vec<u8>> = None;
        let mut last_pts = ClockTime::from_nseconds(0);

        while let Some((frame, pts)) = reader.read::<3>()? { // Read as 3 channels
             assert_eq!(frame.size(), size, "Read frame size mismatch");
             if read_count == 0 {
                 first_read_frame_data = Some(frame.data.clone());
                 assert!(pts.nseconds() < (1_000_000_000 / fps as u64), "First frame PTS too high");
             }
             assert!(pts >= last_pts, "Timestamps should be monotonic increasing");
             last_pts = pts;
             read_count += 1;
        }

        assert_eq!(read_count, num_frames, "Incorrect number of frames read");
        assert!(reader.is_eos(), "Reader should be EOS after roundtrip");

        // Optional: Compare first frame data (lossy compression means they won't be identical)
        // This requires a more sophisticated comparison (e.g., PSNR) or writing lossless.
        if let (Some(written), Some(read)) = (first_frame_data, first_read_frame_data) {
            assert_eq!(written.len(), read.len(), "Data length mismatch for first frame");
            // Simple check: ensure not *all* pixels are drastically different
            let diff_count = written.iter().zip(read.iter()).filter(|(a, b)| (*a as i16 - *b as i16).abs() > 10).count();
            assert!(diff_count < (written.len() / 2), "More than half the pixels differ significantly in the first frame (expected due to lossy compression, but check threshold)");
        } else {
            panic!("Could not retrieve frame data for comparison");
        }


        reader.close()?;
        Ok(())
    }

     #[ignore = "Requires GStreamer plugins (good, base, x264) installed"]
    #[test]
    fn video_reader_writer_roundtrip_h264_mkv_bgr() -> Result<(), Box<dyn std::error::Error>> {
        let tmp_dir = tempfile::tempdir()?;
        let file_path = temp_video_path(&tmp_dir, "roundtrip_h264_bgr.mkv");
        let size = ImageSize { width: 96, height: 64 };
        let fps = 10;
        let num_frames = 8;
        let codec = VideoCodec::H264;
        let container = VideoContainer::MKV;
        let format = ImageFormat::Bgr8; // Write BGR

        // --- Write Phase ---
        let mut writer = VideoWriter::new(&file_path, codec, container, format, fps, size)?;
        writer.start()?;
        for i in 0..num_frames {
            let img = create_test_image::<3>(size, i as u8)?; // 3 channels
            writer.write(&img)?;
        }
        writer.close()?;
        assert!(file_path.exists());

        // --- Read Phase ---
        let mut reader = VideoReader::new(&file_path)?;
        reader.start()?;

        assert_eq!(reader.size(), size, "Read size mismatch");
        assert!((reader.fps() - fps as f64).abs() < 1.0, "Read FPS mismatch");
        assert_eq!(reader.format(), format, "Read format mismatch"); // Expect BGR back

        let mut read_count = 0;
        while let Some((frame, _pts)) = reader.read::<3>()? { // Read as 3 channels
             assert_eq!(frame.size(), size, "Read frame size mismatch");
             read_count += 1;
        }

        assert_eq!(read_count, num_frames, "Incorrect number of frames read");
        assert!(reader.is_eos(), "Reader should be EOS after roundtrip");
        reader.close()?;
        Ok(())
    }

     #[ignore = "Requires GStreamer plugins (good, base, x264) installed"]
    #[test]
    fn video_reader_writer_roundtrip_h264_mp4_mono() -> Result<(), Box<dyn std::error::Error>> {
        let tmp_dir = tempfile::tempdir()?;
        let file_path = temp_video_path(&tmp_dir, "roundtrip_h264_mono.mp4");
        let size = ImageSize { width: 40, height: 40 };
        let fps = 12;
        let num_frames = 6;
        let codec = VideoCodec::H264;
        let container = VideoContainer::MP4;
        let format = ImageFormat::Mono8; // Write Mono8

        // --- Write Phase ---
        let mut writer = VideoWriter::new(&file_path, codec, container, format, fps, size)?;
        writer.start()?;
        for i in 0..num_frames {
            let img = create_test_image::<1>(size, i as u8)?; // 1 channel
            writer.write(&img)?;
        }
        writer.close()?;
        assert!(file_path.exists());

        // --- Read Phase ---
        let mut reader = VideoReader::new(&file_path)?;
        reader.start()?;

        assert_eq!(reader.size(), size, "Read size mismatch");
        assert!((reader.fps() - fps as f64).abs() < 1.0, "Read FPS mismatch");
        assert_eq!(reader.format(), format, "Read format mismatch"); // Expect Mono8 back

        let mut read_count = 0;
        while let Some((frame, _pts)) = reader.read::<1>()? { // Read as 1 channel
             assert_eq!(frame.size(), size, "Read frame size mismatch");
             read_count += 1;
        }

        assert_eq!(read_count, num_frames, "Incorrect number of frames read");
        assert!(reader.is_eos(), "Reader should be EOS after roundtrip");
        reader.close()?;
        Ok(())
    }
}

