#![allow(clippy::collapsible_match)]
use std::path::Path;
use std::sync::{Arc, Mutex};

use gstreamer::prelude::*;
use gstreamer::{ClockTime, Fraction, MessageType, State, StateChangeSuccess};

use gstreamer_app as gst_app;

use kornia_image::{Image, ImageSize};

use super::StreamCaptureError;

// --- Enums ---

/// The video codec used for encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoCodec {
    /// H.264 (AVC) codec.
    H264,
    /// VP9 codec.
    VP9,
    /// AV1 codec.
    AV1,
}

/// The container format for the output video file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoContainer {
    /// MP4 container (.mp4).
    MP4,
    /// Matroska container (.mkv).
    MKV,
    /// WebM container (.webm).
    WebM,
}

/// The pixel format of the image data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFormat {
    /// 8-bit RGB format (Red, Green, Blue).
    Rgb8,
    /// 8-bit BGR format (Blue, Green, Red).
    Bgr8,
    /// 8-bit monochrome (grayscale) format.
    Mono8,
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

    /// Returns the GStreamer format string for raw video caps.
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

// ==========================================================================
// VideoWriter
// ==========================================================================

/// A struct for writing video files using GStreamer.
pub struct VideoWriter {
    pipeline: gstreamer::Pipeline,
    appsrc: gst_app::AppSrc,
    fps: i32,
    format: ImageFormat,
    size: ImageSize,
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
    /// Returns an error if GStreamer initialization fails, the pipeline cannot be
    /// constructed, or elements cannot be found or configured.
    pub fn new(
        path: impl AsRef<Path>,
        codec: VideoCodec,
        container: VideoContainer,
        format: ImageFormat,
        fps: i32,
        size: ImageSize,
    ) -> Result<Self, StreamCaptureError> {
        gstreamer::init()?;

        let path_str = path.as_ref().to_string_lossy();
        let format_str = format.to_gst_format_str();

        let (encoder_str, parser_str, muxer_str) = Self::select_elements(codec, container)?;

        let pipeline_str = format!(
            "appsrc name=src ! video/x-raw,format={format},width={width},height={height},framerate={fps}/1 ! \
            videoconvert ! video/x-raw,format=I420 ! \
            {encoder_str} ! \
            {parser_str} ! \
            {muxer_str} ! \
            filesink location=\"{path}\"",
            format = format_str,
            width = size.width,
            height = size.height,
            fps = fps,
            encoder_str = encoder_str,
            parser_str = parser_str,
            muxer_str = muxer_str,
            path = path_str
        );

        log::debug!("GStreamer writer pipeline: {}", pipeline_str);

        let pipeline = gstreamer::parse::launch(&pipeline_str)?
            .dynamic_cast::<gstreamer::Pipeline>()
            .map_err(StreamCaptureError::DowncastPipelineError)?;

        let appsrc = pipeline
            .by_name("src")
            .ok_or(StreamCaptureError::GetElementByNameError)?
            .dynamic_cast::<gst_app::AppSrc>()
            .map_err(StreamCaptureError::DowncastPipelineError)?;

        appsrc.set_format(gstreamer::Format::Time);
        let caps = gstreamer::Caps::builder("video/x-raw")
            .field("format", format_str)
            .field("width", size.width as i32)
            .field("height", size.height as i32)
            .field("framerate", Fraction::new(fps, 1))
            .build();
        appsrc.set_caps(Some(&caps));
        appsrc.set_is_live(false);
        appsrc.set_property("block", true);

        Ok(Self {
            pipeline,
            appsrc,
            fps,
            format,
            size,
            counter: 0,
            handle: None,
        })
    }

    /// Selects the appropriate GStreamer encoder, parser, and muxer elements.
    fn select_elements(
        codec: VideoCodec,
        container: VideoContainer,
    ) -> Result<(&'static str, &'static str, &'static str), StreamCaptureError> {
        match (codec, container) {
            (VideoCodec::H264, VideoContainer::MP4) => Ok(("x264enc", "h264parse", "mp4mux")),
            (VideoCodec::H264, VideoContainer::MKV) => Ok(("x264enc", "h264parse", "matroskamux")),
            (VideoCodec::VP9, VideoContainer::WebM) => Ok(("vp9enc", "vp9parse", "webmmux")),
            (VideoCodec::VP9, VideoContainer::MKV) => Ok(("vp9enc", "vp9parse", "matroskamux")),
            (VideoCodec::AV1, VideoContainer::MKV) => Ok(("av1enc", "av1parse", "matroskamux")),
            (VideoCodec::AV1, VideoContainer::WebM) => Ok(("av1enc", "av1parse", "webmmux")),
            _ => Err(StreamCaptureError::InvalidConfig(format!(
                "Unsupported codec ({:?}) and container ({:?}) combination.",
                codec, container
            ))),
        }
    }

    /// Start the video writer pipeline.
    ///
    /// # Errors
    /// Returns an error if the pipeline fails to reach the PLAYING state.
    pub fn start(&mut self) -> Result<(), StreamCaptureError> {
        if self.handle.is_some() {
            log::warn!("VideoWriter already started.");
            return Ok(());
        }

        let state_change = self.pipeline.set_state(gstreamer::State::Playing)?;
        if state_change != StateChangeSuccess::Success {
            log::warn!(
                "Pipeline state change for writer returned: {:?}",
                state_change
            );
            if state_change == StateChangeSuccess::Async {
                let (inner_res, current, _pending) =
                    self.pipeline.state(ClockTime::from_seconds(5));
                match inner_res {
                    Ok(_) if current == State::Playing => {
                        log::debug!("Writer pipeline reached PLAYING after async wait.");
                    }
                    _ => {
                        return Err(StreamCaptureError::InvalidConfig(format!(
                            "Writer pipeline failed to reach PLAYING state after async, current: {:?}, result: {:?}",
                            current, inner_res
                         )));
                    }
                }
            } else if state_change == StateChangeSuccess::NoPreroll {
                log::debug!("Writer pipeline started with NO_PREROLL.");
            } else {
                return Err(StreamCaptureError::InvalidConfig(format!(
                    "Unexpected writer pipeline state change result: {:?}",
                    state_change
                )));
            }
        }

        log::info!("VideoWriter pipeline set to Playing.");

        let pipeline_weak = self.pipeline.downgrade();
        let bus = self.pipeline.bus().ok_or(StreamCaptureError::BusError)?;

        let handle = std::thread::spawn(move || {
            for msg in bus.iter_timed(ClockTime::NONE) {
                let _pipeline = match pipeline_weak.upgrade() {
                    Some(p) => p,
                    None => break,
                };
                match msg.view() {
                    gstreamer::MessageView::Eos(..) => {
                        log::debug!("Writer bus received EOS");
                        break;
                    }
                    gstreamer::MessageView::Error(err) => {
                        log::error!(
                            "Writer bus error from {:?}: {} ({:?})",
                            msg.src().map(|s| s.path_string()),
                            err.error(),
                            err.debug()
                        );
                    }
                    gstreamer::MessageView::Warning(warn) => {
                        log::warn!(
                            "Writer bus warning from {:?}: {} ({:?})",
                            msg.src().map(|s| s.path_string()),
                            warn.error(),
                            warn.debug()
                        );
                    }
                    _ => {}
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
    /// * `img` - The image frame to write.
    ///
    /// # Errors
    /// Returns an error if the image format/size is incorrect or if pushing the buffer fails.
    pub fn write<const C: usize>(&mut self, img: &Image<u8, C>) -> Result<(), StreamCaptureError> {
        if C != self.format.channels() {
            return Err(StreamCaptureError::InvalidImageFormat(format!(
                "Invalid number of channels: expected {}, got {}. Format is {:?}.",
                self.format.channels(),
                C,
                self.format
            )));
        }
        if img.size() != self.size {
            return Err(StreamCaptureError::InvalidImageFormat(format!(
                "Invalid image size: expected {:?}, got {:?}.",
                self.size,
                img.size()
            )));
        }

        let mut buffer = gstreamer::Buffer::from_mut_slice(img.as_slice().to_vec());

        let frame_duration_ns = 1_000_000_000 / self.fps as u64;
        let pts = ClockTime::from_nseconds(self.counter * frame_duration_ns);
        let duration = ClockTime::from_nseconds(frame_duration_ns);

        let buffer_ref = buffer.get_mut().ok_or_else(|| {
            StreamCaptureError::InvalidConfig("Failed to get mutable buffer reference".to_string())
        })?;
        buffer_ref.set_pts(Some(pts));
        buffer_ref.set_duration(Some(duration));

        self.counter += 1;

        let flow_ret = self.appsrc.push_buffer(buffer)?;
        if flow_ret != gstreamer::FlowSuccess::Ok {
            log::error!("Failed to push buffer into appsrc: {:?}", flow_ret);
            return Err(StreamCaptureError::InvalidConfig(format!(
                "Failed to push buffer: {:?}",
                flow_ret
            )));
        }

        Ok(())
    }

    /// Finalizes the video file (sends EOS, waits for completion, sets state to NULL).
    ///
    /// # Errors
    /// Returns an error if sending EOS or changing state fails.
    pub fn close(&mut self) -> Result<(), StreamCaptureError> {
        if self.handle.is_none() {
            return Ok(());
        }
        log::debug!("Closing VideoWriter...");

        self.appsrc.end_of_stream()?;

        if let Some(bus) = self.pipeline.bus() {
            log::debug!("Waiting for EOS message on writer bus...");
            bus.timed_pop_filtered(
                ClockTime::from_seconds(10),
                &[MessageType::Eos, MessageType::Error],
            );
            log::debug!("EOS or Error message received/timeout on writer bus.");
        } else {
            log::warn!("Could not get bus to wait for EOS.");
        }

        if let Some(handle) = self.handle.take() {
            if let Err(e) = handle.join() {
                log::error!("VideoWriter bus thread panicked: {:?}", e);
            } else {
                log::debug!("VideoWriter bus thread joined successfully.");
            }
        }

        self.pipeline.set_state(gstreamer::State::Null)?;

        log::info!("VideoWriter pipeline set to Null.");
        Ok(())
    }
}

impl Drop for VideoWriter {
    fn drop(&mut self) {
        if self.handle.is_some() {
            if let Err(e) = self.close() {
                log::error!("Error closing video writer in drop: {}", e);
            }
        }
    }
}

// ==========================================================================
// VideoReader
// ==========================================================================

/// A struct for reading video files using GStreamer.
pub struct VideoReader {
    pipeline: gstreamer::Pipeline,
    appsink: gst_app::AppSink,
    fps: f64,
    format: ImageFormat,
    size: ImageSize,
    duration: Option<ClockTime>,
    is_eos: Arc<Mutex<bool>>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl VideoReader {
    /// Create a new VideoReader for the given path.
    ///
    /// # Arguments
    /// * `path` - The path to the video file.
    ///
    /// # Errors
    /// Returns an error if GStreamer initialization fails, the file doesn't exist,
    /// or the pipeline cannot be constructed.
    pub fn new(path: impl AsRef<Path>) -> Result<Self, StreamCaptureError> {
        gstreamer::init()?;

        let path = path.as_ref().to_owned();
        if !path.exists() {
            return Err(StreamCaptureError::InvalidConfig(format!(
                "File not found: {}",
                path.display()
            )));
        }
        let path_str = path.to_string_lossy();

        let pipeline_str = format!(
            "filesrc location=\"{path}\" ! \
            decodebin ! \
            videoconvert ! \
            video/x-raw,format={{RGB,BGR,GRAY8}} ! \
            appsink name=sink emit-signals=true sync=false",
            path = path_str
        );

        log::debug!("GStreamer reader pipeline: {}", pipeline_str);

        let pipeline = gstreamer::parse::launch(&pipeline_str)?
            .dynamic_cast::<gstreamer::Pipeline>()
            .map_err(StreamCaptureError::DowncastPipelineError)?;

        let appsink = pipeline
            .by_name("sink")
            .ok_or(StreamCaptureError::GetElementByNameError)?
            .dynamic_cast::<gst_app::AppSink>()
            .map_err(StreamCaptureError::DowncastPipelineError)?;

        appsink.set_property("emit-signals", true);

        let reader = Self {
            pipeline,
            appsink,
            fps: 0.0,
            format: ImageFormat::Rgb8,
            size: ImageSize {
                width: 0,
                height: 0,
            },
            duration: None,
            is_eos: Arc::new(Mutex::new(false)),
            handle: None,
        };

        Ok(reader)
    }

    /// Start the video reader pipeline and determine video properties (size, format, fps, duration).
    ///
    /// # Errors
    /// Returns an error if the pipeline fails to start, fails to negotiate caps,
    /// or cannot determine video properties.
    pub fn start(&mut self) -> Result<(), StreamCaptureError> {
        if self.handle.is_some() {
            log::warn!("VideoReader already started.");
            return Ok(());
        }

        let state_change_result = self.pipeline.set_state(gstreamer::State::Playing);

        match state_change_result {
            Ok(StateChangeSuccess::Success) => {
                log::debug!("Pipeline reached PLAYING state successfully.");
            }
            Ok(StateChangeSuccess::NoPreroll) => {
                log::debug!("Pipeline reached PLAYING state with NO_PREROLL.");
            }
            Ok(StateChangeSuccess::Async) => {
                log::debug!("Pipeline state change is ASYNC, waiting for completion...");
                let state_result_tuple = self.pipeline.state(ClockTime::from_seconds(10));
                let (inner_result, current_state, _) = state_result_tuple;

                match inner_result {
                    Ok(_) if current_state == State::Playing => {
                        log::debug!("Pipeline reached PLAYING state after async.");
                    }
                    Ok(unexpected_success) => {
                        log::warn!("Pipeline reached unexpected success state {:?} after async ({:?}). Assuming PLAYING.", unexpected_success, current_state);
                        if current_state != State::Playing {
                            let _ = self.pipeline.set_state(gstreamer::State::Null);
                            return Err(StreamCaptureError::InvalidConfig(format!(
                                 "Pipeline reached success state {:?} but is not PLAYING (current: {:?}) after async.", unexpected_success, current_state
                              )));
                        }
                    }
                    Err(err) => {
                        log::error!("Pipeline failed to reach PLAYING state after async: {:?}, current state: {:?}", err, current_state);
                        let _ = self.pipeline.set_state(gstreamer::State::Null);
                        return Err(StreamCaptureError::InvalidConfig(format!(
                            "Pipeline failed state change after async: {:?}",
                            err
                        )));
                    }
                }
            }
            Err(err) => {
                log::error!("Pipeline failed to change state to PLAYING: {:?}", err);
                if let Some(bus) = self.pipeline.bus() {
                    if let Some(msg) =
                        bus.timed_pop_filtered(ClockTime::from_mseconds(100), &[MessageType::Error])
                    {
                        if let gstreamer::MessageView::Error(error_msg) = msg.view() {
                            log::error!(
                                "Bus error message: {} ({:?})",
                                error_msg.error(),
                                error_msg.debug()
                            );
                        }
                    }
                }
                let _ = self.pipeline.set_state(gstreamer::State::Null);
                return Err(StreamCaptureError::from(err));
            }
        }

        log::info!("VideoReader pipeline set to Playing.");

        let sinkpad = self.appsink.static_pad("sink").ok_or_else(|| {
            StreamCaptureError::InvalidConfig("Failed to get appsink sink pad".to_string())
        })?;

        let caps = sinkpad
            .current_caps()
            .or_else(|| {
                log::warn!(
                    "Caps not immediately available on sinkpad, trying appsink caps property..."
                );
                self.appsink.caps()
            })
            .or_else(|| {
                log::warn!("Caps still not available, waiting briefly for preroll/data...");
                std::thread::sleep(std::time::Duration::from_millis(200));
                sinkpad.current_caps().or_else(|| self.appsink.caps())
            })
            .ok_or_else(|| {
                StreamCaptureError::InvalidConfig(
                    "Failed to get negotiated caps from appsink after waiting.".to_string(),
                )
            })?;

        log::debug!("Negotiated caps: {}", caps.to_string());

        let structure = caps.structure(0).ok_or_else(|| {
            StreamCaptureError::InvalidConfig("Failed to get structure from caps".to_string())
        })?;

        let width = structure
            .get::<i32>("width")
            .map_err(|e| StreamCaptureError::GetCapsError(format!("width: {:?}", e)))?;
        let height = structure
            .get::<i32>("height")
            .map_err(|e| StreamCaptureError::GetCapsError(format!("height: {:?}", e)))?;
        self.size = ImageSize {
            width: width as usize,
            height: height as usize,
        };

        let format_str = structure
            .get::<String>("format")
            .map_err(|e| StreamCaptureError::GetCapsError(format!("format: {:?}", e)))?;
        self.format = ImageFormat::from_gst_format_str(&format_str)?;

        if let Ok(fps_frac) = structure.get::<Fraction>("framerate") {
            if fps_frac.denom() != 0 {
                self.fps = fps_frac.numer() as f64 / fps_frac.denom() as f64;
            } else {
                log::warn!("Invalid framerate fraction (denominator is 0), defaulting to 0.0");
                self.fps = 0.0;
            }
        } else {
            log::warn!("Could not determine video framerate from caps, defaulting to 0.0");
            self.fps = 0.0;
        }

        log::info!(
            "Video properties: Size={}x{}, Format={:?}, FPS={:.2}",
            self.size.width,
            self.size.height,
            self.format,
            self.fps
        );

        self.duration = self.pipeline.query_duration::<ClockTime>();
        if let Some(dur) = self.duration {
            log::info!("Video duration: {:.2} seconds", dur.seconds_f64());
        } else {
            log::warn!("Could not determine video duration.");
        }

        let bus = self.pipeline.bus().ok_or(StreamCaptureError::BusError)?;
        let is_eos_clone = self.is_eos.clone();
        let pipeline_weak = self.pipeline.downgrade();

        let handle = std::thread::spawn(move || {
            for msg in bus.iter_timed(ClockTime::NONE) {
                let _pipeline = match pipeline_weak.upgrade() {
                    Some(p) => p,
                    None => break,
                };
                match msg.view() {
                    gstreamer::MessageView::Eos(..) => {
                        log::debug!("Reader bus received EOS");
                        let mut is_eos = is_eos_clone.lock().unwrap();
                        *is_eos = true;
                    }
                    gstreamer::MessageView::Error(err) => {
                        log::error!(
                            "Reader bus error from {:?}: {} ({:?})",
                            msg.src().map(|s| s.path_string()),
                            err.error(),
                            err.debug()
                        );
                        let mut is_eos = is_eos_clone.lock().unwrap();
                        *is_eos = true;
                        break;
                    }
                    gstreamer::MessageView::Warning(warn) => {
                        log::warn!(
                            "Reader bus warning from {:?}: {} ({:?})",
                            msg.src().map(|s| s.path_string()),
                            warn.error(),
                            warn.debug()
                        );
                    }
                    _ => {}
                }
            }
            log::debug!("VideoReader bus monitoring thread finished.");
        });

        self.handle = Some(handle);

        Ok(())
    }

    /// Read the next frame from the video.
    ///
    /// Returns `Ok(Some((image, timestamp)))` if a frame is successfully read.
    /// Returns `Ok(None)` if the end of the stream is reached.
    /// Returns `Err` if an error occurs during reading or processing.
    ///
    /// # Type Parameters
    /// * `C` - The expected number of channels in the output image.
    ///
    /// # Errors
    /// Returns an error if the requested channel count `C` doesn't match the video's format,
    /// or if `pull_sample` fails, or if processing the sample fails.
    pub fn read<const C: usize>(
        &mut self,
    ) -> Result<Option<(Image<u8, C>, ClockTime)>, StreamCaptureError> {
        if *self.is_eos.lock().unwrap() {
            log::trace!("EOS flag is set, trying to pull potentially buffered sample.");
            match self.appsink.try_pull_sample(ClockTime::ZERO) {
                Some(sample) => {
                    log::trace!("Processing potentially buffered sample after EOS flag set.");
                    if C != self.format.channels() {
                        return Err(StreamCaptureError::InvalidImageFormat(format!(
                            "Invalid number of channels requested after EOS: expected {} (for format {:?}), got {}",
                            self.format.channels(), self.format, C
                        )));
                    }
                    return self.process_sample::<C>(sample);
                }
                None => {
                    log::trace!("No buffered sample found after EOS flag set.");
                    return Ok(None);
                }
            }
        }

        if C != self.format.channels() {
            return Err(StreamCaptureError::InvalidImageFormat(format!(
                "Invalid number of channels requested: expected {} (for format {:?}), got {}",
                self.format.channels(),
                self.format,
                C
            )));
        }

        match self.appsink.pull_sample() {
            Ok(sample) => self.process_sample::<C>(sample),
            Err(err) => {
                // Check the EOS flag again *after* the pull failed.
                // If the flag is now true, it means EOS likely happened concurrently.
                if *self.is_eos.lock().unwrap() {
                    log::debug!(
                        "pull_sample failed ({:?}), but EOS flag is set. Assuming EOS.",
                        err
                    );
                    Ok(None)
                } else {
                    // If EOS flag is not set, this is an unexpected error.
                    log::error!("Appsink pull_sample failed unexpectedly: {:?}", err);
                    let mut is_eos = self.is_eos.lock().unwrap();
                    *is_eos = true; // Treat any pull error as fatal for reading
                    Err(StreamCaptureError::InvalidConfig(format!(
                        "GStreamer pull_sample failed: {:?}",
                        err
                    )))
                }
            }
        }
    }

    /// Helper function to process a GStreamer sample into an Image.
    fn process_sample<const C: usize>(
        &self,
        sample: gstreamer::Sample,
    ) -> Result<Option<(Image<u8, C>, ClockTime)>, StreamCaptureError> {
        if C != self.format.channels() {
            return Err(StreamCaptureError::InvalidImageFormat(format!(
                "Internal Error: process_sample called with wrong channel count C={}. Expected {}.",
                C,
                self.format.channels()
            )));
        }

        let buffer = sample.buffer().ok_or(StreamCaptureError::GetBufferError)?;
        let pts = buffer.pts().unwrap_or_default();

        let map = buffer
            .map_readable()
            .map_err(|_| StreamCaptureError::CreateImageFrameError)?;

        let data = map.as_slice();
        let expected_size = self.size.width * self.size.height * C;

        if data.len() < expected_size {
            log::warn!(
                "Buffer size smaller than expected: expected {} bytes, got {} bytes. PTS: {:?}. Format: {:?}, Size: {:?}",
                expected_size, data.len(), pts, self.format, self.size
            );
            return Err(StreamCaptureError::InvalidConfig(format!(
                "Buffer size mismatch: expected {} bytes, got {} bytes. PTS: {:?}",
                expected_size,
                data.len(),
                pts
            )));
        } else if data.len() > expected_size {
            log::trace!(
                "Buffer size larger than expected ({} > {}), using expected size. PTS: {:?}",
                data.len(),
                expected_size,
                pts
            );
        }

        let img = Image::<u8, C>::new(self.size, data[..expected_size].to_vec())
            .map_err(|e| StreamCaptureError::ProcessImageFrameError(Box::new(e)))?;

        Ok(Some((img, pts)))
    }

    /// Get the frames per second (FPS) of the video.
    pub fn fps(&self) -> f64 {
        self.fps
    }
    /// Get the size (width, height) of the video frames.
    pub fn size(&self) -> ImageSize {
        self.size
    }
    /// Get the pixel format of the video frames.
    pub fn format(&self) -> ImageFormat {
        self.format
    }
    /// Check if the end of the video stream has been reached or a fatal error occurred.
    pub fn is_eos(&self) -> bool {
        *self.is_eos.lock().unwrap()
    }

    /// Close the video reader and release resources.
    ///
    /// # Errors
    /// Returns an error if setting the pipeline state to NULL fails.
    pub fn close(&mut self) -> Result<(), StreamCaptureError> {
        if self.handle.is_none() {
            return Ok(());
        }
        log::debug!("Closing VideoReader...");

        if let Err(err) = self.pipeline.set_state(gstreamer::State::Null) {
            log::error!("Failed to set reader pipeline state to NULL: {:?}", err);
        } else {
            log::info!("VideoReader pipeline set to Null.");
        }

        if let Some(handle) = self.handle.take() {
            *self.is_eos.lock().unwrap() = true;
            if let Err(e) = handle.join() {
                log::error!("VideoReader bus thread panicked: {:?}", e);
            } else {
                log::debug!("VideoReader bus thread joined successfully.");
            }
        }

        *self.is_eos.lock().unwrap() = true;

        Ok(())
    }

    /// Seek to the specified position in the video.
    ///
    /// # Arguments
    /// * `position_secs` - The target position in seconds from the start of the video.
    ///
    /// # Errors
    /// Returns an error if seeking is attempted before starting, the position is negative,
    /// or the seek operation fails in the pipeline.
    pub fn seek(&mut self, position_secs: f64) -> Result<(), StreamCaptureError> {
        if self.handle.is_none() {
            return Err(StreamCaptureError::InvalidConfig(
                "Cannot seek before starting the reader.".to_string(),
            ));
        }
        if position_secs < 0.0 {
            return Err(StreamCaptureError::InvalidConfig(
                "Seek position cannot be negative".to_string(),
            ));
        }

        let position_ns = ClockTime::from_seconds_f64(position_secs);
        let seek_flags = gstreamer::SeekFlags::FLUSH
            | gstreamer::SeekFlags::ACCURATE
            | gstreamer::SeekFlags::KEY_UNIT;

        log::debug!(
            "Seeking to {} ns ({:.2} s)",
            position_ns.nseconds(),
            position_secs
        );

        *self.is_eos.lock().unwrap() = false;

        let seek_result = self.pipeline.seek(
            1.0,
            seek_flags,
            gstreamer::SeekType::Set,
            position_ns,
            gstreamer::SeekType::Set,
            ClockTime::NONE,
        );

        if let Err(err) = seek_result {
            log::error!("Seek operation failed: {:?}", err);
            *self.is_eos.lock().unwrap() = true;
            return Err(StreamCaptureError::InvalidConfig(format!(
                "Seek failed: {}",
                err
            )));
        }

        log::debug!("Seek event sent successfully. Waiting for pipeline to settle...");

        let state_result_tuple = self.pipeline.state(ClockTime::from_seconds(5));
        let (inner_result, current_state, pending_state) = state_result_tuple;

        match inner_result {
            Ok(_) if current_state == State::Playing => {
                log::debug!("Pipeline settled in PLAYING state after seek.");
            }
            Ok(StateChangeSuccess::NoPreroll) if current_state == State::Playing => {
                log::debug!("Pipeline settled in PLAYING (NoPreroll) state after seek.");
            }
            _ => {
                log::warn!("Pipeline state after seek is {:?} ({:?}, pending {:?}). May affect subsequent reads.", current_state, inner_result, pending_state);
            }
        }

        *self.is_eos.lock().unwrap() = false;
        log::debug!("EOS flag reset after seek.");

        Ok(())
    }

    /// Get the total duration of the video in seconds, if known.
    pub fn duration_secs(&self) -> Option<f64> {
        self.duration.map(|d| d.seconds_f64())
    }
    /// Get the total duration of the video as a GStreamer `ClockTime`, if known.
    pub fn duration(&self) -> Option<ClockTime> {
        self.duration
    }
}

impl Drop for VideoReader {
    fn drop(&mut self) {
        if self.handle.is_some() {
            if let Err(e) = self.close() {
                log::error!("Error closing video reader in drop: {}", e);
            }
        } else {
            let _ = self.pipeline.set_state(gstreamer::State::Null);
        }
    }
}

// --- Tests ---
#[cfg(test)]
mod tests {
    use super::{ImageFormat, VideoCodec, VideoContainer, VideoReader, VideoWriter};
    use gstreamer::ClockTime;
    use kornia_image::{Image, ImageSize};
    use std::path::{Path, PathBuf};

    fn create_test_image<const C: usize>(
        size: ImageSize,
        frame_num: u8,
    ) -> Result<Image<u8, C>, kornia_image::ImageError> {
        let mut data = Vec::with_capacity(size.width * size.height * C);
        for y in 0..size.height {
            for x in 0..size.width {
                if C == 1 {
                    data.push(((x + y + frame_num as usize) % 255) as u8);
                } else if C == 3 {
                    data.push(frame_num);
                    data.push((y % 255) as u8);
                    data.push((x % 255) as u8);
                } else {
                    panic!("Unsupported channel count in test helper");
                }
            }
        }
        Image::<u8, C>::new(size, data)
    }

    fn temp_video_path(dir: &tempfile::TempDir, filename: &str) -> PathBuf {
        dir.path().join(filename)
    }

    fn create_dummy_video(
        path: &Path,
        codec: VideoCodec,
        container: VideoContainer,
        format: ImageFormat,
        size: ImageSize,
        fps: i32,
        num_frames: u8,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut writer = VideoWriter::new(path, codec, container, format, fps, size)?;
        writer.start()?;
        for i in 0..num_frames {
            match format.channels() {
                1 => {
                    let img = create_test_image::<1>(size, i)?;
                    writer.write(&img)?;
                }
                3 => {
                    let img = create_test_image::<3>(size, i)?;
                    writer.write(&img)?;
                }
                _ => panic!("Unsupported channel count in dummy video creation"),
            }
        }
        writer.close()?;
        println!("Finished writing dummy video: {:?}", path);
        Ok(())
    }

    fn setup_test() {
        let _ = gstreamer::init();
    }

    #[ignore = "Requires GStreamer plugins (good, base, libav, x264) installed"]
    #[test]
    fn video_writer_h264_mp4_rgb8() -> Result<(), Box<dyn std::error::Error>> {
        setup_test();
        let tmp_dir = tempfile::tempdir()?;
        let file_path = temp_video_path(&tmp_dir, "test_h264_rgb.mp4");
        let size = ImageSize {
            width: 64,
            height: 48,
        };
        let fps = 30;
        let num_frames = 5;

        create_dummy_video(
            &file_path,
            VideoCodec::H264,
            VideoContainer::MP4,
            ImageFormat::Rgb8,
            size,
            fps,
            num_frames,
        )?;

        assert!(file_path.exists(), "File was not created: {:?}", file_path);
        assert!(
            std::fs::metadata(&file_path)?.len() > 100,
            "File seems too small"
        );
        Ok(())
    }

    #[ignore = "Requires GStreamer plugins (good, base, libav, x264) installed"]
    #[test]
    fn video_writer_h264_mkv_bgr8() -> Result<(), Box<dyn std::error::Error>> {
        setup_test();
        let tmp_dir = tempfile::tempdir()?;
        let file_path = temp_video_path(&tmp_dir, "test_h264_bgr.mkv");
        let size = ImageSize {
            width: 64,
            height: 48,
        };
        let fps = 15;
        let num_frames = 3;

        create_dummy_video(
            &file_path,
            VideoCodec::H264,
            VideoContainer::MKV,
            ImageFormat::Bgr8,
            size,
            fps,
            num_frames,
        )?;

        assert!(file_path.exists(), "File was not created: {:?}", file_path);
        assert!(
            std::fs::metadata(&file_path)?.len() > 100,
            "File seems too small"
        );
        Ok(())
    }

    #[ignore = "Requires GStreamer plugins (good, base, libav, x264) installed"]
    #[test]
    fn video_writer_h264_mp4_mono8() -> Result<(), Box<dyn std::error::Error>> {
        setup_test();
        let tmp_dir = tempfile::tempdir()?;
        let file_path = temp_video_path(&tmp_dir, "test_h264_mono.mp4");
        let size = ImageSize {
            width: 32,
            height: 32,
        };
        let fps = 25;
        let num_frames = 4;

        create_dummy_video(
            &file_path,
            VideoCodec::H264,
            VideoContainer::MP4,
            ImageFormat::Mono8,
            size,
            fps,
            num_frames,
        )?;

        assert!(file_path.exists(), "File was not created: {:?}", file_path);
        assert!(
            std::fs::metadata(&file_path)?.len() > 100,
            "File seems too small"
        );
        Ok(())
    }

    #[ignore = "Requires GStreamer plugins (good, base, libav, vp9) installed"]
    #[test]
    fn video_writer_vp9_webm_rgb8() -> Result<(), Box<dyn std::error::Error>> {
        setup_test();
        let tmp_dir = tempfile::tempdir()?;
        let file_path = temp_video_path(&tmp_dir, "test_vp9_rgb.webm");
        let size = ImageSize {
            width: 80,
            height: 60,
        };
        let fps = 20;
        let num_frames = 6;

        create_dummy_video(
            &file_path,
            VideoCodec::VP9,
            VideoContainer::WebM,
            ImageFormat::Rgb8,
            size,
            fps,
            num_frames,
        )?;

        assert!(file_path.exists(), "File was not created: {:?}", file_path);
        assert!(
            std::fs::metadata(&file_path)?.len() > 100,
            "File seems too small"
        );
        Ok(())
    }

    #[ignore = "Requires GStreamer plugins (good, base, libav, x264) installed"]
    #[test]
    fn video_reader_basic_rgb() -> Result<(), Box<dyn std::error::Error>> {
        setup_test();
        let tmp_dir = tempfile::tempdir()?;
        let file_path = temp_video_path(&tmp_dir, "test_reader_rgb.mp4");
        let size = ImageSize {
            width: 128,
            height: 72,
        };
        let fps = 30;
        let num_frames = 15;

        create_dummy_video(
            &file_path,
            VideoCodec::H264,
            VideoContainer::MP4,
            ImageFormat::Rgb8,
            size,
            fps,
            num_frames,
        )?;

        let mut reader = VideoReader::new(&file_path)?;
        reader.start()?;

        assert!(
            (reader.fps() - fps as f64).abs() < 1.5,
            "Unexpected FPS: expected ~{}, got {}",
            fps,
            reader.fps()
        );
        assert_eq!(reader.size(), size, "Unexpected size");
        assert!(
            [ImageFormat::Rgb8, ImageFormat::Bgr8].contains(&reader.format()),
            "Unexpected format: {:?}",
            reader.format()
        );
        assert!(reader.duration_secs().is_some(), "Duration should be known");
        let expected_duration = num_frames as f64 / fps as f64;
        assert!(
            (reader.duration_secs().unwrap() - expected_duration).abs() < 0.2,
            "Unexpected duration: expected ~{:.2}, got {:.2}",
            expected_duration,
            reader.duration_secs().unwrap_or(-1.0)
        );

        let frame_data = reader.read::<3>()?;
        assert!(frame_data.is_some(), "Failed to read the first frame");
        let (frame, pts) = frame_data.unwrap();
        assert_eq!(frame.size(), reader.size(), "Frame size mismatch");
        assert!(
            pts.nseconds() < (1_000_000_000 / fps as u64),
            "First frame PTS seems too high: {}",
            pts.nseconds()
        );

        println!("First frame PTS: {} ns", pts.nseconds());

        let mut frame_count = 1;
        while let Some((_img, _pts)) = reader.read::<3>()? {
            frame_count += 1;
            if frame_count >= 5 {
                break;
            }
        }
        assert!(
            frame_count >= 5,
            "Failed to read multiple frames (read {})",
            frame_count
        );

        let seek_time = 0.2;
        reader.seek(seek_time)?;
        assert!(!reader.is_eos(), "EOS flag should be reset after seek");

        let frame_data_after_seek = reader.read::<3>()?;
        assert!(
            frame_data_after_seek.is_some(),
            "Failed to read frame after seeking"
        );
        let (_frame_after_seek, pts_after_seek) = frame_data_after_seek.unwrap();
        println!(
            "Frame PTS after seeking to {}s: {} ns ({:.3}s)",
            seek_time,
            pts_after_seek.nseconds(),
            pts_after_seek.seconds_f64()
        );
        let frame_duration_secs = 1.0 / reader.fps();
        assert!(
            (pts_after_seek.seconds_f64() - seek_time).abs() < frame_duration_secs * 1.5,
            "PTS after seek ({:.3}s) is too far from seek target ({:.3}s)",
            pts_after_seek.seconds_f64(),
            seek_time
        );

        let mut post_seek_count = 1;
        while reader.read::<3>()?.is_some() {
            post_seek_count += 1;
        }
        assert!(
            reader.is_eos(),
            "Reader should be EOS after reading all frames post-seek"
        );
        println!("Read {} frames after seeking.", post_seek_count);
        let expected_remaining =
            ((reader.duration_secs().unwrap_or(0.0) - seek_time) * reader.fps()).round() as i32;
        println!("Expected remaining frames: ~{}", expected_remaining);
        assert!(
            (post_seek_count as i32 - expected_remaining).abs() <= 2,
            "Unexpected number of frames after seek"
        );

        assert!(
            reader.read::<3>()?.is_none(),
            "Read after EOS should return None"
        );

        reader.close()?;
        Ok(())
    }

    #[ignore = "Requires GStreamer plugins (good, base, libav, x264) installed"]
    #[test]
    fn video_reader_writer_roundtrip_h264_mp4_rgb() -> Result<(), Box<dyn std::error::Error>> {
        setup_test();
        let tmp_dir = tempfile::tempdir()?;
        let file_path = temp_video_path(&tmp_dir, "roundtrip_h264_rgb.mp4");
        let size = ImageSize {
            width: 128,
            height: 72,
        };
        let fps = 24;
        let num_frames = 10;
        let codec = VideoCodec::H264;
        let container = VideoContainer::MP4;
        let format = ImageFormat::Rgb8;

        let mut first_frame_written_data: Option<Vec<u8>> = None;
        {
            let mut writer = VideoWriter::new(&file_path, codec, container, format, fps, size)?;
            writer.start()?;
            for i in 0..num_frames {
                let img = create_test_image::<3>(size, i as u8)?;
                if i == 0 {
                    first_frame_written_data = Some(img.data.clone());
                }
                writer.write(&img)?;
            }
            writer.close()?;
        }
        assert!(file_path.exists());

        let mut reader = VideoReader::new(&file_path)?;
        reader.start()?;

        assert_eq!(reader.size(), size, "Read size mismatch");
        assert!(
            (reader.fps() - fps as f64).abs() < 1.5,
            "Read FPS mismatch: expected ~{}, got {}",
            fps,
            reader.fps()
        );
        let read_format = reader.format();
        assert!(
            [ImageFormat::Rgb8, ImageFormat::Bgr8].contains(&read_format),
            "Read format mismatch: {:?}",
            read_format
        );

        let mut read_count = 0;
        let mut first_read_frame_data: Option<Vec<u8>> = None;
        let mut last_pts = ClockTime::from_nseconds(0);

        while let Some((frame, pts)) = reader.read::<3>()? {
            assert_eq!(
                frame.size(),
                size,
                "Read frame {} size mismatch",
                read_count
            );
            if read_count == 0 {
                first_read_frame_data = Some(frame.data.clone());
                assert!(
                    pts.nseconds() < (1_000_000_000 / fps as u64),
                    "First frame PTS too high"
                );
            }
            assert!(
                pts >= last_pts,
                "Timestamps should be monotonic increasing (or equal): prev={}, current={}",
                last_pts,
                pts
            );
            last_pts = pts;
            read_count += 1;
        }

        assert_eq!(read_count, num_frames, "Incorrect number of frames read");
        assert!(reader.is_eos(), "Reader should be EOS after roundtrip");

        if let (Some(written_data), Some(read_data)) =
            (first_frame_written_data, first_read_frame_data)
        {
            assert_eq!(
                written_data.len(),
                read_data.len(),
                "Data length mismatch for first frame"
            );

            let mut diff_sum: u64 = 0;
            let mut max_diff: u8 = 0;

            if read_format == format {
                for (w, r) in written_data.iter().zip(read_data.iter()) {
                    let diff = (*w as i16 - *r as i16).abs() as u8;
                    diff_sum += diff as u64;
                    max_diff = max_diff.max(diff);
                }
            } else if read_format == ImageFormat::Bgr8 && format == ImageFormat::Rgb8 {
                for i in 0..(size.width * size.height) {
                    let w_r = written_data[i * 3 + 0];
                    let w_g = written_data[i * 3 + 1];
                    let w_b = written_data[i * 3 + 2];
                    let r_b = read_data[i * 3 + 0];
                    let r_g = read_data[i * 3 + 1];
                    let r_r = read_data[i * 3 + 2];

                    let diff_r = (w_r as i16 - r_r as i16).abs() as u8;
                    let diff_g = (w_g as i16 - r_g as i16).abs() as u8;
                    let diff_b = (w_b as i16 - r_b as i16).abs() as u8;

                    diff_sum += diff_r as u64 + diff_g as u64 + diff_b as u64;
                    max_diff = max_diff.max(diff_r).max(diff_g).max(diff_b);
                }
            } else {
                println!("Skipping pixel comparison due to unexpected format conversion.");
                diff_sum = u64::MAX;
            }

            let avg_diff = diff_sum as f64 / written_data.len() as f64;
            println!(
                "First frame comparison: Avg diff = {:.2}, Max diff = {}",
                avg_diff, max_diff
            );
            assert!(
                avg_diff < 30.0,
                "Average pixel difference too high ({:.2}) in the first frame",
                avg_diff
            );
            assert!(
                max_diff < 200,
                "Max pixel difference too high ({}) in the first frame",
                max_diff
            );
        } else {
            panic!("Could not retrieve frame data for comparison");
        }

        reader.close()?;
        Ok(())
    }

    #[ignore = "Requires GStreamer plugins (good, base, libav, x264) installed"]
    #[test]
    fn video_reader_writer_roundtrip_h264_mkv_bgr() -> Result<(), Box<dyn std::error::Error>> {
        setup_test();
        let tmp_dir = tempfile::tempdir()?;
        let file_path = temp_video_path(&tmp_dir, "roundtrip_h264_bgr.mkv");
        let size = ImageSize {
            width: 96,
            height: 64,
        };
        let fps = 10;
        let num_frames = 8;
        let codec = VideoCodec::H264;
        let container = VideoContainer::MKV;
        let format = ImageFormat::Bgr8;

        {
            let mut writer = VideoWriter::new(&file_path, codec, container, format, fps, size)?;
            writer.start()?;
            for i in 0..num_frames {
                let img = create_test_image::<3>(size, i as u8)?;
                writer.write(&img)?;
            }
            writer.close()?;
        }
        assert!(file_path.exists());

        let mut reader = VideoReader::new(&file_path)?;
        reader.start()?;

        assert_eq!(reader.size(), size, "Read size mismatch");
        assert!((reader.fps() - fps as f64).abs() < 1.5, "Read FPS mismatch");
        assert!(
            [ImageFormat::Rgb8, ImageFormat::Bgr8].contains(&reader.format()),
            "Read format mismatch: {:?}",
            reader.format()
        );

        let mut read_count = 0;
        while let Some((frame, _pts)) = reader.read::<3>()? {
            assert_eq!(
                frame.size(),
                size,
                "Read frame {} size mismatch",
                read_count
            );
            read_count += 1;
        }

        assert_eq!(read_count, num_frames, "Incorrect number of frames read");
        assert!(reader.is_eos(), "Reader should be EOS after roundtrip");
        reader.close()?;
        Ok(())
    }

    #[ignore = "Requires GStreamer plugins (good, base, libav, x264) installed"]
    #[test]
    fn video_reader_writer_roundtrip_h264_mp4_mono() -> Result<(), Box<dyn std::error::Error>> {
        setup_test();
        let tmp_dir = tempfile::tempdir()?;
        let file_path = temp_video_path(&tmp_dir, "roundtrip_h264_mono.mp4");
        let size = ImageSize {
            width: 40,
            height: 40,
        };
        let fps = 12;
        let num_frames = 6;
        let codec = VideoCodec::H264;
        let container = VideoContainer::MP4;
        let format = ImageFormat::Mono8;

        {
            let mut writer = VideoWriter::new(&file_path, codec, container, format, fps, size)?;
            writer.start()?;
            for i in 0..num_frames {
                let img = create_test_image::<1>(size, i as u8)?;
                writer.write(&img)?;
            }
            writer.close()?;
        }
        assert!(file_path.exists());

        let mut reader = VideoReader::new(&file_path)?;
        reader.start()?;

        assert_eq!(reader.size(), size, "Read size mismatch");
        assert!((reader.fps() - fps as f64).abs() < 1.5, "Read FPS mismatch");
        assert_eq!(
            reader.format(),
            format,
            "Read format mismatch: {:?}",
            reader.format()
        );

        let mut read_count = 0;
        while let Some((frame, _pts)) = reader.read::<1>()? {
            assert_eq!(
                frame.size(),
                size,
                "Read frame {} size mismatch",
                read_count
            );
            read_count += 1;
        }

        assert_eq!(read_count, num_frames, "Incorrect number of frames read");
        assert!(reader.is_eos(), "Reader should be EOS after roundtrip");
        reader.close()?;
        Ok(())
    }

    #[ignore = "Requires GStreamer plugins (good, base, libav, vp9) installed"]
    #[test]
    fn video_reader_writer_roundtrip_vp9_webm_rgb() -> Result<(), Box<dyn std::error::Error>> {
        setup_test();
        let tmp_dir = tempfile::tempdir()?;
        let file_path = temp_video_path(&tmp_dir, "roundtrip_vp9_rgb.webm");
        let size = ImageSize {
            width: 70,
            height: 50,
        };
        let fps = 18;
        let num_frames = 7;
        let codec = VideoCodec::VP9;
        let container = VideoContainer::WebM;
        let format = ImageFormat::Rgb8;

        {
            let mut writer = VideoWriter::new(&file_path, codec, container, format, fps, size)?;
            writer.start()?;
            for i in 0..num_frames {
                let img = create_test_image::<3>(size, i as u8)?;
                writer.write(&img)?;
            }
            writer.close()?;
        }
        assert!(file_path.exists());

        let mut reader = VideoReader::new(&file_path)?;
        reader.start()?;

        assert_eq!(reader.size(), size, "Read size mismatch");
        assert!((reader.fps() - fps as f64).abs() < 1.5, "Read FPS mismatch");
        assert!(
            [ImageFormat::Rgb8, ImageFormat::Bgr8].contains(&reader.format()),
            "Read format mismatch: {:?}",
            reader.format()
        );

        let mut read_count = 0;
        while let Some((frame, _pts)) = reader.read::<3>()? {
            assert_eq!(
                frame.size(),
                size,
                "Read frame {} size mismatch",
                read_count
            );
            read_count += 1;
        }

        assert_eq!(read_count, num_frames, "Incorrect number of frames read");
        assert!(reader.is_eos(), "Reader should be EOS after roundtrip");
        reader.close()?;
        Ok(())
    }
}
