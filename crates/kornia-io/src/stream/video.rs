use std::path::Path;
use std::sync::{Arc, Mutex};

use gst::prelude::*;
use gst_app;

use kornia_image::{Image, ImageSize};
use super::StreamCaptureError;

// Keep From impl for ImageError -> InvalidImageFormat
impl From<kornia_image::ImageError> for StreamCaptureError {
    fn from(err: kornia_image::ImageError) -> Self {
        StreamCaptureError::InvalidImageFormat(format!("Failed to create image: {}", err))
    }
}

/// The codec to use for the video writer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoCodec {
    /// H.264 codec.
    H264,
}

/// The format of the image to write to the video file or read from it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFormat {
    /// 8-bit RGB format.
    Rgb8,
    /// 8-bit mono format.
    Mono8,
}

/// A struct for writing video files.
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
    /// * `codec` - The codec to use for the video writer.
    /// * `format` - The expected image format.
    /// * `fps` - The frames per second of the video.
    /// * `size` - The size of the video.
    pub fn new(
        path: impl AsRef<Path>,
        codec: VideoCodec,
        format: ImageFormat,
        fps: i32,
        size: ImageSize,
    ) -> Result<Self, StreamCaptureError> {
        gst::init().map_err(StreamCaptureError::GstInitError)?;

        #[allow(unreachable_patterns)]
        let _codec = match codec {
            VideoCodec::H264 => "x264enc",
            _ => return Err(StreamCaptureError::InvalidConfig("Unsupported codec".to_string())),
        };

        let format_str = match format {
            ImageFormat::Mono8 => "GRAY8",
            ImageFormat::Rgb8 => "RGB",
        };

        let path = path.as_ref().to_owned();
        let location_str = if cfg!(windows) { path.to_string_lossy().replace('\\', "/") } else { path.to_string_lossy().into_owned() };

        let pipeline_str = format!(
            "appsrc name=src ! \
            videoconvert ! video/x-raw,format=I420 ! \
            x264enc ! \
            video/x-h264,profile=main ! \
            h264parse ! \
            mp4mux ! \
            filesink location=\"{}\"",
            location_str
        );
        log::debug!("Writer Pipeline: {}", pipeline_str);

        // Assuming DowncastPipelineError(gst::Element) still exists in error.rs
        let pipeline = gst::parse::launch(&pipeline_str)
            .map_err(|e| StreamCaptureError::InvalidConfig(format!("Failed to parse pipeline: {}", e)))?
            .dynamic_cast::<gst::Pipeline>()
            .map_err(|e| StreamCaptureError::DowncastPipelineError(e.upcast()))?;

        let appsrc_name = "src";
        let appsrc = pipeline
            .by_name(appsrc_name)
            .ok_or_else(|| StreamCaptureError::GetElementByNameError)?
            .dynamic_cast::<gst_app::AppSrc>()
            .map_err(|e| StreamCaptureError::DowncastPipelineError(e.upcast()))?;

        appsrc.set_format(gst::Format::Time);
        let caps = gst::Caps::builder("video/x-raw")
            .field("format", format_str)
            .field("width", size.width as i32)
            .field("height", size.height as i32)
            .field("framerate", gst::Fraction::new(fps, 1))
            .build();
        appsrc.set_caps(Some(&caps));
        appsrc.set_is_live(true);
        appsrc.set_property("block", false);

        Ok(Self { pipeline, appsrc, fps, format, counter: 0, handle: None })
    }

    /// Start the video writer.
    ///
    /// Set the pipeline to playing and launch a task to handle the bus messages.
    pub fn start(&mut self) -> Result<(), StreamCaptureError> {
        self.pipeline.set_state(gst::State::Playing)?;
        let bus = self.pipeline.bus().ok_or(StreamCaptureError::BusError)?;
        let pipeline_weak = self.pipeline.downgrade(); // Use weak ref for thread

        let handle = std::thread::spawn(move || {
            for msg in bus.iter_timed(gst::ClockTime::NONE) {
                if pipeline_weak.upgrade().is_none() {
                    log::warn!("Writer bus thread: Pipeline already dropped, exiting.");
                    break;
                }
                match msg.view() {
                    gst::MessageView::Eos(..) => {
                        log::debug!("Writer bus thread received EOS");
                        break;
                    }
                    gst::MessageView::Error(err) => {
                        log::error!("Writer bus thread error from {:?}: {} ({:?})", msg.src().map(|s| s.path_string()), err.error(), err.debug());
                        break;
                    }
                    _ => {}
                }
            }
            log::debug!("Writer bus thread finished.");
        });
        self.handle = Some(handle);
        Ok(())
    }

    /// Close the video writer.
    /// Sends EOS, joins background bus thread, waits briefly, sets pipeline to NULL.
    pub fn close(&mut self) -> Result<(), StreamCaptureError> {
        log::debug!("VideoWriter::close START (Simplified)");

        log::debug!("Sending EOS to appsrc...");
        match self.appsrc.end_of_stream() {
            Ok(_) => log::debug!("EOS signal sent successfully."),
            Err(gst::FlowError::NotLinked) => log::warn!("Appsrc already EOS or not linked when sending EOS."),
            Err(err) => {
                log::error!("Failed to send EOS to appsrc: {:?}", err);
                return Err(StreamCaptureError::GstreamerFlowError(err));
            }
        }
        log::debug!("Sent EOS signal.");

        // Join the original background bus handler thread.
        if let Some(handle) = self.handle.take() {
            log::debug!("Joining writer bus thread (background)...");
            if let Err(e) = handle.join() {
                log::error!("Failed to join writer bus thread: {:?}", e);
            } else {
                log::debug!("Writer bus thread joined successfully.");
            }
        } else {
            log::warn!("No bus thread handle found to join in close. Was start() called?");
        }
        log::debug!("Finished joining bus thread step.");

        // Wait briefly after joining thread, before setting NULL
        log::debug!("Waiting briefly after bus thread join before setting NULL...");
        std::thread::sleep(std::time::Duration::from_millis(500)); // Keep 0.5s delay

        log::debug!("Setting writer pipeline to NULL.");
        self.pipeline.set_state(gst::State::Null)?;
        log::debug!("Pipeline set to NULL.");

        log::debug!("VideoWriter::close END (Simplified)");
        Ok(())
    }


    /// Write an image to the video file.
    pub fn write<const C: usize>(&mut self, img: &Image<u8, C>) -> Result<(), StreamCaptureError> {
        match self.format {
            ImageFormat::Mono8 => {
                if C != 1 { return Err(StreamCaptureError::InvalidImageFormat(format!("Channels: expected 1, got {}", C))); }
            }
            ImageFormat::Rgb8 => {
                if C != 3 { return Err(StreamCaptureError::InvalidImageFormat(format!("Channels: expected 3, got {}", C))); }
            }
        }
        let mut buffer = gst::Buffer::from_mut_slice(img.as_slice().to_vec());
        let pts = gst::ClockTime::from_nseconds(self.counter * 1_000_000_000 / self.fps as u64);
        let duration = gst::ClockTime::from_nseconds(1_000_000_000 / self.fps as u64);
        let buffer_ref = buffer.get_mut().ok_or_else(|| StreamCaptureError::InvalidConfig("Failed to get mutable buffer reference".to_string()))?;
        buffer_ref.set_pts(Some(pts));
        buffer_ref.set_duration(Some(duration));
        self.counter += 1;
        self.appsrc.push_buffer(buffer).map_err(StreamCaptureError::GstreamerFlowError)?;
        Ok(())
    }
}

impl Drop for VideoWriter {
    fn drop(&mut self) {
        if self.handle.is_some() {
            log::debug!("Closing VideoWriter in Drop.");
            if let Err(e) = self.close() { log::error!("Error closing video writer in drop: {}", e); }
        } else {
            if let Err(e) = self.pipeline.set_state(gst::State::Null) { log::error!("Error setting writer pipeline NULL in drop: {}", e); }
        }
    }
}

///// --- VideoReader ---

/// A struct for reading video files.
pub struct VideoReader {
    pipeline: gst::Pipeline,
    appsink: gst_app::AppSink,
    fps: f64,
    format: ImageFormat,
    size: ImageSize,
    is_eos: Arc<Mutex<bool>>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl VideoReader {
    /// Create a new VideoReader.
    pub fn new(path: impl AsRef<Path>) -> Result<Self, StreamCaptureError> {
        gst::init().map_err(StreamCaptureError::GstInitError)?;
        let path = path.as_ref().to_owned();
        if !path.exists() { return Err(StreamCaptureError::FileNotFound(path.to_string_lossy().to_string())); }

        let path_str = path.to_string_lossy();
        let pipeline_str = format!(
            "filesrc location=\"{}\" ! \
            decodebin ! \
            videoconvert ! \
            video/x-raw,format=(string){{RGB,GRAY8}} ! \
            appsink name=sink emit-signals=true sync=false max-buffers=5 drop=true",
            path_str
        );
        log::debug!("Reader Pipeline: {}", pipeline_str);

        let pipeline = gst::parse::launch(&pipeline_str)
            .map_err(|e| StreamCaptureError::InvalidConfig(format!("Failed to parse pipeline: {}", e)))?
            .dynamic_cast::<gst::Pipeline>()
            .map_err(|e| StreamCaptureError::DowncastPipelineError(e.upcast()))?;

        let appsink_name = "sink";
        let appsink = pipeline
            .by_name(appsink_name)
            .ok_or_else(|| StreamCaptureError::GetElementByNameError)?
            .dynamic_cast::<gst_app::AppSink>()
            .map_err(|e| StreamCaptureError::DowncastPipelineError(e.upcast()))?;

        // Configure appsink
        appsink.set_property("emit-signals", true);
        appsink.set_property("sync", false);
        appsink.set_property("max-buffers", 5u32);
        appsink.set_property("drop", true);

        let reader = Self { pipeline, appsink, fps: 0.0, format: ImageFormat::Rgb8, size: ImageSize {width: 0, height: 0}, is_eos: Arc::new(Mutex::new(false)), handle: None };
        Ok(reader)
    }

    /// Start the video reader.
    pub fn start(&mut self) -> Result<(), StreamCaptureError> {
        if self.handle.is_some() { log::warn!("VideoReader already started."); return Ok(()); }

        log::debug!("Setting reader pipeline to Playing...");
        self.pipeline.set_state(gst::State::Playing)?;

        // Wait for Preroll/First Sample
        log::debug!("Waiting for pipeline to preroll (pulling first preroll sample)...");
        let preroll_timeout = gst::ClockTime::from_seconds(15);

        // try_pull_preroll returns Option<Sample> in your version
        let preroll_sample = match self.appsink.try_pull_preroll(preroll_timeout) {
            Some(sample) => {
                // Successfully prerolled
                log::debug!("Pipeline prerolled successfully (received preroll sample).");
                sample
            }
            None => {
                // Timeout or EOS waiting for preroll sample
                self.pipeline.set_state(gst::State::Null)?;
                log::error!("Timeout or EOS waiting for pipeline preroll sample.");
                return Err(StreamCaptureError::InvalidConfig(
                    "Timeout waiting for pipeline preroll sample".to_string(),
                ));
            }
        };

        // Now get caps from sample.
        log::debug!("Getting caps from preroll sample...");
        if let Some(caps) = preroll_sample.caps() {
            log::debug!("Retrieved caps: {}", caps.to_string());
            if let Some(structure) = caps.structure(0) {
                // Get video dimensions
                let width = structure.get::<i32>("width")
                    .map_err(|_| StreamCaptureError::GetWidthError)?;
                let height = structure.get::<i32>("height")
                    .map_err(|_| StreamCaptureError::GetHeightError)?;
                self.size = ImageSize { width: width as usize, height: height as usize };

                // Get framerate
                if let Ok(fps_frac) = structure.get::<gst::Fraction>("framerate") {
                    if fps_frac.numer() > 0 && fps_frac.denom() > 0 {
                        self.fps = fps_frac.numer() as f64 / fps_frac.denom() as f64;
                    } else {
                        log::warn!("Invalid framerate fraction ({}/{}) in caps, using 0.0",
                        fps_frac.numer(), fps_frac.denom());
                        self.fps = 0.0;
                    }
                } else {
                    log::warn!("Could not determine video framerate from caps, using default of 0.0");
                    self.fps = 0.0;
                }

                // Get format
                if let Ok(format_str) = structure.get::<String>("format") {
                    self.format = match format_str.as_str() {
                        "RGB" => ImageFormat::Rgb8,
                        "GRAY8" => ImageFormat::Mono8,
                        unsupported_format => {
                            self.pipeline.set_state(gst::State::Null)?;
                            return Err(StreamCaptureError::InvalidImageFormat(
                                format!("Unsupported format: {}", unsupported_format)
                            ));
                        }
                    };
                } else {
                    self.pipeline.set_state(gst::State::Null)?;
                    return Err(StreamCaptureError::InvalidConfig(
                        "Failed to get format from caps".to_string()
                    ));
                }
            } else {
                self.pipeline.set_state(gst::State::Null)?;
                return Err(StreamCaptureError::GetStructureError);
            }
        } else {
            self.pipeline.set_state(gst::State::Null)?;
            log::error!("Failed to get caps from sample even after successful preroll!");
            return Err(StreamCaptureError::GetCapsError);
        }

        // Start bus thread
        log::debug!("Starting bus message handler thread...");
        let bus = self.pipeline.bus().ok_or(StreamCaptureError::BusError)?;
        let is_eos_clone = self.is_eos.clone();
        let pipeline_weak = self.pipeline.downgrade();
        let handle = std::thread::spawn(move || {
            for msg in bus.iter_timed(gst::ClockTime::NONE) {
                if pipeline_weak.upgrade().is_none() {
                    log::warn!("Reader bus thread: Pipeline dropped, exiting.");
                    break;
                }
                match msg.view() {
                    gst::MessageView::Eos(..) => {
                        log::debug!("Reader bus thread received EOS");
                        let mut flag = is_eos_clone.lock().unwrap();
                        *flag = true;
                        break;
                    }
                    gst::MessageView::Error(err) => {
                        log::error!(
                        "Reader bus thread error from {:?}: {} ({:?})",
                        msg.src().map(|s| s.path_string()),
                        err.error(),
                        err.debug()
                    );
                        let mut flag = is_eos_clone.lock().unwrap();
                        *flag = true;
                        break;
                    }
                    gst::MessageView::Warning(warn) => {
                        log::warn!(
                        "Reader bus thread warning from {:?}: {} ({:?})",
                        msg.src().map(|s| s.path_string()),
                        warn.error(),
                        warn.debug()
                    );
                    }
                    _ => {}
                }
            }
            log::debug!("Reader bus thread finished.");
        });
        self.handle = Some(handle);

        log::info!("VideoReader started successfully: Size={:?}, FPS={:.2}, Format={:?}",
        self.size, self.fps, self.format);
        Ok(())
    }


    /// Read a frame from the video.
    pub fn read<const C: usize>(&mut self) -> Result<Option<Image<u8, C>>, StreamCaptureError> {
        if *self.is_eos.lock().unwrap() { return Ok(None); }

        let expected_channels = match self.format {
            ImageFormat::Mono8 => 1,
            ImageFormat::Rgb8 => 3
        };

        if C != expected_channels {
            return Err(StreamCaptureError::InvalidImageFormat(
                format!("Channels: expected {}, got {}", expected_channels, C)
            ));
        }

        let timeout = gst::ClockTime::from_seconds(5);

        // try_pull_sample returns Option<Sample>
        match self.appsink.try_pull_sample(timeout) {
            Some(sample) => {
                let buffer = sample.buffer().ok_or_else(|| StreamCaptureError::GetBufferError)?;
                let map = buffer.map_readable().map_err(|_| StreamCaptureError::CreateImageFrameError)?;
                let data = map.as_slice();
                let expected_size = self.size.width * self.size.height * C;

                if data.len() != expected_size {
                    log::warn!(
                    "Buffer size mismatch: expected {} bytes, got {} bytes. Check video format/pipeline.",
                    expected_size,
                    data.len()
                );

                    if data.len() < expected_size {
                        return Err(StreamCaptureError::InvalidConfig(format!(
                            "Buffer too small: expected {} bytes, got {} bytes",
                            expected_size, data.len()
                        )));
                    }
                }

                let img = Image::<u8, C>::new(self.size, data[..expected_size].to_vec())?;
                Ok(Some(img))
            }
            None => {
                // Could be EOS or timeout
                if *self.is_eos.lock().unwrap() {
                    Ok(None)
                } else {
                    log::trace!("try_pull_sample returned None - possible timeout");
                    Ok(None)
                }
            }
        }
    }

    /// Get the frames per second of the video.
    pub fn fps(&self) -> f64 { self.fps }

    /// Get the size of the video frames.
    pub fn size(&self) -> ImageSize { self.size }

    /// Get the format of the video frames.
    pub fn format(&self) -> ImageFormat { self.format }

    /// Check if the video has reached the end (based on EOS flag).
    pub fn is_eos(&self) -> bool { *self.is_eos.lock().unwrap() }

    /// Close the video reader.
    pub fn close(&mut self) -> Result<(), StreamCaptureError> {
        log::debug!("Setting reader pipeline to NULL.");
        self.pipeline.set_state(gst::State::Null)?;
        if let Some(handle) = self.handle.take() {
            log::debug!("Joining reader bus thread...");
            if let Err(e) = handle.join() { log::error!("Failed to join reader bus thread: {:?}", e); }
            else { log::debug!("Reader bus thread joined."); }
        }
        Ok(())
    }

    /// Seek to the specified position in seconds.
    pub fn seek(&mut self, position_secs: f64) -> Result<(), StreamCaptureError> {
        log::debug!("Seeking to {:.3} seconds", position_secs);
        let position_ns = gst::ClockTime::from_seconds_f64(position_secs);
        let seek_flags = gst::SeekFlags::FLUSH | gst::SeekFlags::KEY_UNIT | gst::SeekFlags::ACCURATE;
        self.pipeline.seek(1.0, seek_flags, gst::SeekType::Set, position_ns, gst::SeekType::None, gst::ClockTime::NONE)
            .map_err(StreamCaptureError::PullSampleError)?;
        { let mut is_eos = self.is_eos.lock().unwrap(); *is_eos = false; }
        log::trace!("Waiting briefly after seek trigger...");
        let _ = self.pipeline.state(gst::ClockTime::from_nseconds(100 * 1_000_000));
        log::debug!("Seek triggered and brief wait complete.");
        Ok(())
    }

    /// Get the duration of the video in seconds.
    pub fn duration(&self) -> Option<f64> {
        match self.pipeline.query_duration::<gst::ClockTime>() {
            Some(duration) => Some(duration.seconds_f64()),
            None => { log::warn!("Could not query duration from pipeline."); None }
        }
    }
}

// Drop implementation for VideoReader
impl Drop for VideoReader {
    fn drop(&mut self) {
        log::debug!("Closing VideoReader in Drop implementation.");
        if self.handle.is_some() {
            if let Err(e) = self.close() { log::error!("Error closing video reader in drop: {}", e); }
        } else {
            if let Err(e) = self.pipeline.set_state(gst::State::Null) { log::error!("Error setting reader pipeline NULL in drop: {}", e); }
        }
    }
}


///// --- Tests ---
#[cfg(test)]
mod tests {
    use super::{ImageFormat, VideoCodec, VideoReader, VideoWriter};
    use kornia_image::{Image, ImageSize};

    // Helper to ensure GStreamer logs are captured by tests
    fn setup_test_logging() {
        // Attempt to initialize env_logger. If it fails, logging is already initialized.
        let _ = env_logger::builder().is_test(true).try_init();
    }

    // Helper function to create a dummy video file for testing
    fn create_dummy_video(file_path: &std::path::Path, num_frames: u32) -> Result<(), Box<dyn std::error::Error>> {
        let size = ImageSize { width: 64, height: 48 }; // Small size for tests
        let fps = 10;
        let mut writer = VideoWriter::new(
            file_path,
            VideoCodec::H264,
            ImageFormat::Rgb8,
            fps,
            size,
        )?;
        writer.start()?;

        for i in 0..num_frames {
            let frame_val = (i * 255 / num_frames) as u8;
            let data = vec![frame_val; size.width * size.height * 3]; // Simple solid color frame
            let img = Image::<u8, 3>::new(size, data)?;
            writer.write(&img)?;
        }
        writer.close()?;
        println!("Created dummy video: {:?}", file_path);
        Ok(())
    }


    #[test]
    #[ignore = "needs gstreamer installed and configured"]
    fn video_writer_rgb8u() -> Result<(), Box<dyn std::error::Error>> {
        setup_test_logging();
        let tmp_dir = tempfile::tempdir()?;
        let file_path = tmp_dir.path().join("test_writer_rgb.mp4");
        create_dummy_video(&file_path, 5)?;
        assert!(file_path.exists(), "File does not exist: {:?}", file_path);
        // Add check for file size > 0
        assert!(std::fs::metadata(&file_path)?.len() > 0, "File is empty");
        Ok(())
    }

    #[test]
    #[ignore = "needs gstreamer installed and configured"]
    fn video_writer_mono8u() -> Result<(), Box<dyn std::error::Error>> {
        setup_test_logging();
        let tmp_dir = tempfile::tempdir()?;
        let file_path = tmp_dir.path().join("test_writer_mono.mp4");

        let size = ImageSize { width: 64, height: 48 };
        let fps = 10;
        let mut writer = VideoWriter::new(
            &file_path,
            VideoCodec::H264,
            ImageFormat::Mono8, // Specify Mono8 format
            fps,
            size,
        )?;
        writer.start()?;
        let data = vec![128u8; size.width * size.height]; // Gray frame
        let img = Image::<u8, 1>::new(size, data)?;
        writer.write(&img)?;
        writer.close()?;

        assert!(file_path.exists(), "File does not exist: {:?}", file_path);
        assert!(std::fs::metadata(&file_path)?.len() > 0, "File is empty");
        Ok(())
    }

    #[test]
    #[ignore = "needs gstreamer installed and configured"]
    fn video_reader_basic_metadata() -> Result<(), Box<dyn std::error::Error>> {
        setup_test_logging();
        let tmp_dir = tempfile::tempdir()?;
        let file_path = tmp_dir.path().join("test_reader_basic.mp4");
        let num_frames = 15;
        create_dummy_video(&file_path, num_frames)?;

        let mut reader = VideoReader::new(&file_path)?;
        reader.start()?; // Start should succeed and get metadata

        assert_eq!(reader.size().width, 64);
        assert_eq!(reader.size().height, 48);
        assert!((reader.fps() - 10.0).abs() < 0.1, "FPS mismatch: {}", reader.fps()); // Allow small tolerance for FPS
        assert_eq!(reader.format(), ImageFormat::Rgb8); // create_dummy_video writes RGB
        assert!(reader.duration().is_some());
        // Duration might not be exact due to encoding/muxing
        assert!((reader.duration().unwrap() - (num_frames as f64 / 10.0)).abs() < 0.2, "Duration mismatch: {:?}", reader.duration());

        reader.close()?;
        Ok(())
    }


    #[test]
    #[ignore = "needs gstreamer installed and configured"]
    fn video_reader_read_frames() -> Result<(), Box<dyn std::error::Error>> {
        setup_test_logging();
        let tmp_dir = tempfile::tempdir()?;
        let file_path = tmp_dir.path().join("test_reader_frames.mp4");
        let num_frames = 12;
        create_dummy_video(&file_path, num_frames)?;

        let mut reader = VideoReader::new(&file_path)?;
        reader.start()?;

        let mut frame_count = 0;
        while let Some(frame) = reader.read::<3>()? { // Read RGB
            assert_eq!(frame.size().width, 64);
            assert_eq!(frame.size().height, 48);
            frame_count += 1;
        }

        assert_eq!(frame_count, num_frames as usize, "Incorrect number of frames read");
        assert!(reader.is_eos(), "EOS flag should be set after reading all frames");

        // Try reading again after EOS, should return None
        assert!(reader.read::<3>()?.is_none(), "Reading after EOS should return None");

        reader.close()?;
        Ok(())
    }

    #[test]
    #[ignore = "needs gstreamer installed and configured"]
    fn video_reader_seek() -> Result<(), Box<dyn std::error::Error>> {
        setup_test_logging();
        let tmp_dir = tempfile::tempdir()?;
        let file_path = tmp_dir.path().join("test_reader_seek.mp4");
        let num_frames = 30; // 3 seconds at 10 fps
        create_dummy_video(&file_path, num_frames)?;

        let mut reader = VideoReader::new(&file_path)?;
        reader.start()?;

        // Read a few frames first
        let _frame1 = reader.read::<3>()?;
        let _frame2 = reader.read::<3>()?;
        assert!(_frame1.is_some());
        assert!(_frame2.is_some());

        // Seek to approximately 1.5 seconds (which should be around frame 15)
        reader.seek(1.5)?;
        assert!(!reader.is_eos(), "EOS flag should be reset after seek");

        // Read the frame after seeking
        let seek_frame_opt = reader.read::<3>()?;
        assert!(seek_frame_opt.is_some(), "Failed to read frame after seeking");

        // Seek near the end
        reader.seek(2.8)?;
        let end_frame_opt = reader.read::<3>()?;
        assert!(end_frame_opt.is_some(), "Failed to read frame after seeking near end");

        // Read until EOS
        let mut count_after_seek = 0;
        while reader.read::<3>()?.is_some() {
            count_after_seek += 1;
        }
        // Should only have read one more frame (frame 29) after seeking to 2.8s
        assert!(count_after_seek <= 1, "Read too many frames after seeking near end");
        assert!(reader.is_eos(), "EOS not set after reading post-seek");


        reader.close()?;
        Ok(())
    }


    #[test]
    #[ignore = "needs gstreamer installed and configured"]
    fn video_reader_writer_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
        setup_test_logging();
        let tmp_dir = tempfile::tempdir()?;
        let file_path = tmp_dir.path().join("roundtrip.mp4");
        let size = ImageSize { width: 128, height: 96 };
        let fps = 15;
        let num_frames = 10;

        // Create a test pattern
        let mut writer = VideoWriter::new(&file_path, VideoCodec::H264, ImageFormat::Rgb8, fps, size)?;
        writer.start()?;
        for i in 0..num_frames {
            let mut data = Vec::with_capacity(size.width * size.height * 3);
            for y in 0..size.height {
                for x in 0..size.width {
                    data.push((x % 255) as u8); // R
                    data.push((y % 255) as u8); // G
                    data.push((i * 10) as u8);   // B changes
                }
            }
            let img = Image::<u8, 3>::new(size, data)?;
            writer.write(&img)?;
        }
        writer.close()?;

        // Read back the video
        let mut reader = VideoReader::new(&file_path)?;
        reader.start()?;

        assert_eq!(reader.size(), size);
        assert!((reader.fps() - fps as f64).abs() < 1.0); // FPS can be less precise after encoding
        assert_eq!(reader.format(), ImageFormat::Rgb8);

        let mut read_count = 0;
        while let Some(frame) = reader.read::<3>()? {
            assert_eq!(frame.size(), size);
            read_count += 1;
        }

        assert_eq!(read_count, num_frames as usize, "Roundtrip frame count mismatch");
        assert!(reader.is_eos());

        reader.close()?;
        Ok(())
    }
}