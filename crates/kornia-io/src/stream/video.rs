use std::path::Path;

use gst::prelude::*;

use kornia_image::{Image, ImageSize};

use super::StreamCaptureError;

use std::sync::{Arc, Mutex};

/// The codec to use for the video writer.
pub enum VideoCodec {
    /// H.264 codec.
    H264,
}

/// The format of the image to write to the video file.
///
/// Usually will be the combination of the image format and the pixel type.
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
        gst::init()?;

        // TODO: Add support for other codecs
        #[allow(unreachable_patterns)]
        let _codec = match codec {
            VideoCodec::H264 => "x264enc",
            _ => {
                return Err(StreamCaptureError::InvalidConfig(
                    "Unsupported codec".to_string(),
                ))
            }
        };

        // TODO: Add support for other formats
        let format_str = match format {
            ImageFormat::Mono8 => "GRAY8",
            ImageFormat::Rgb8 => "RGB",
        };

        let path = path.as_ref().to_owned();

        let pipeline_str = format!(
            "appsrc name=src ! \
            videoconvert ! video/x-raw,format=I420 ! \
            x264enc ! \
            video/x-h264,profile=main ! \
            h264parse ! \
            mp4mux ! \
            filesink location={}",
            path.to_string_lossy()
        );

        let pipeline = gst::parse::launch(&pipeline_str)?
            .dynamic_cast::<gst::Pipeline>()
            .map_err(StreamCaptureError::DowncastPipelineError)?;

        let appsrc = pipeline
            .by_name("src")
            .ok_or_else(|| StreamCaptureError::GetElementByNameError)?
            .dynamic_cast::<gst_app::AppSrc>()
            .map_err(StreamCaptureError::DowncastPipelineError)?;

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

        Ok(Self {
            pipeline,
            appsrc,
            fps,
            format,
            counter: 0,
            handle: None,
        })
    }

    /// Start the video writer.
    ///
    /// Set the pipeline to playing and launch a task to handle the bus messages.
    pub fn start(&mut self) -> Result<(), StreamCaptureError> {
        // set the pipeline to playing
        self.pipeline.set_state(gst::State::Playing)?;

        let bus = self.pipeline.bus().ok_or(StreamCaptureError::BusError)?;

        // launch a task to handle the bus messages, exit when EOS is received and set the pipeline to null
        let handle = std::thread::spawn(move || {
            for msg in bus.iter_timed(gst::ClockTime::NONE) {
                match msg.view() {
                    gst::MessageView::Eos(..) => {
                        log::debug!("gstreamer received EOS");
                        break;
                    }
                    gst::MessageView::Error(err) => {
                        log::error!(
                            "Error from {:?}: {} ({:?})",
                            msg.src().map(|s| s.path_string()),
                            err.error(),
                            err.debug()
                        );
                        break;
                    }
                    _ => {}
                }
            }
        });

        self.handle = Some(handle);

        Ok(())
    }

    ///  Close the video writer.
    ///
    /// Set the pipeline to null and join the thread.
    ///
    pub fn close(&mut self) -> Result<(), StreamCaptureError> {
        // send end of stream to the appsrc
        self.appsrc.end_of_stream()?;

        if let Some(handle) = self.handle.take() {
            handle.join().expect("Failed to join thread");
        }

        self.pipeline.set_state(gst::State::Null)?;

        Ok(())
    }
    /// Write an image to the video file.
    ///
    /// # Arguments
    ///
    /// * `img` - The image to write to the video file.
    // TODO: explore supporting write_async
    pub fn write<const C: usize>(&mut self, img: &Image<u8, C>) -> Result<(), StreamCaptureError> {
        // check if the image channels are correct
        match self.format {
            ImageFormat::Mono8 => {
                if C != 1 {
                    return Err(StreamCaptureError::InvalidImageFormat(format!(
                        "Invalid number of channels: expected 1, got {}",
                        C
                    )));
                }
            }
            ImageFormat::Rgb8 => {
                if C != 3 {
                    return Err(StreamCaptureError::InvalidImageFormat(format!(
                        "Invalid number of channels: expected 3, got {}",
                        C
                    )));
                }
            }
        }

        // TODO: verify is there is a cheaper way to copy the buffer
        let mut buffer = gst::Buffer::from_mut_slice(img.as_slice().to_vec());

        let pts = gst::ClockTime::from_nseconds(self.counter * 1_000_000_000 / self.fps as u64);
        let duration = gst::ClockTime::from_nseconds(1_000_000_000 / self.fps as u64);

        let buffer_ref = buffer.get_mut().expect("Failed to get buffer");
        buffer_ref.set_pts(Some(pts));
        buffer_ref.set_duration(Some(duration));

        self.counter += 1;

        if let Err(err) = self.appsrc.push_buffer(buffer) {
            return Err(StreamCaptureError::InvalidConfig(err.to_string()));
        }

        Ok(())
    }
}

impl Drop for VideoWriter {
    fn drop(&mut self) {
        if self.handle.is_some() {
            self.close().expect("Failed to close video writer");
        }
    }
}

/// A struct for reading video files.
///
/// This reader uses GStreamer to decode video files and provides access to
/// the video frames as Images.
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
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the video file to read.
    pub fn new(path: impl AsRef<Path>) -> Result<Self, StreamCaptureError> {
        gst::init()?;

        let path = path.as_ref().to_owned();

        // Create a pipeline that decodes video to raw frames
        // Note: We escape the file path properly to handle paths with special characters
        let path_str = path.to_string_lossy();

        let pipeline_str = format!(
            "filesrc location=\"{}\" ! \
            decodebin ! \
            videoconvert ! \
            video/x-raw,format=(string){{RGB,GRAY8}} ! \
            appsink name=sink emit-signals=true sync=false",
            path_str
        );

        let pipeline = gst::parse::launch(&pipeline_str)?
            .dynamic_cast::<gst::Pipeline>()
            .map_err(StreamCaptureError::DowncastPipelineError)?;

        let appsink = pipeline
            .by_name("sink")
            .ok_or_else(|| StreamCaptureError::GetElementByNameError)?
            .dynamic_cast::<gst_app::AppSink>()
            .map_err(StreamCaptureError::DowncastPipelineError)?;

        // Start with default values that will be updated after pipeline starts
        let reader = Self {
            pipeline,
            appsink,
            fps: 0.0,
            format: ImageFormat::Rgb8, // Default format
            size: ImageSize {
                width: 0,
                height: 0,
            },
            is_eos: Arc::new(Mutex::new(false)),
            handle: None,
        };

        Ok(reader)
    }

    /// Start the video reader.
    ///
    /// This method will start the pipeline and update video properties from the
    /// negotiated caps.
    pub fn start(&mut self) -> Result<(), StreamCaptureError> {
        self.pipeline.set_state(gst::State::Playing)?;

        // Allow pipeline to initialize and verify it's playing
        match self.pipeline.state(gst::ClockTime::from_seconds(5)) {
            Ok(state) if state.1 != gst::State::Playing => {
                return Err(StreamCaptureError::InvalidConfig(
                    "Failed to start pipeline".to_string(),
                ));
            }
            Err(err) => {
                return Err(StreamCaptureError::InvalidConfig(format!(
                    "Failed to get pipeline state: {:?}",
                    err
                )));
            }
            _ => {} // Pipeline is playing successfully
        }

        // Get video information from appsink caps
        if let Some(caps) = self.appsink.caps() {
            if let Some(structure) = caps.structure(0) {
                // Get video dimensions
                if let (Ok(width), Ok(height)) = (
                    structure.get::<i32>("width"),
                    structure.get::<i32>("height"),
                ) {
                    self.size = ImageSize {
                        width: width as usize,
                        height: height as usize,
                    };
                } else {
                    return Err(StreamCaptureError::InvalidConfig(
                        "Failed to get video dimensions from caps".to_string(),
                    ));
                }

                // Get framerate
                if let Ok(fps) = structure.get::<gst::Fraction>("framerate") {
                    self.fps = fps.numer() as f64 / fps.denom() as f64;
                } else {
                    log::warn!("Could not determine video framerate, using default of 0.0");
                }

                // Get format
                if let Ok(format_str) = structure.get::<String>("format") {
                    self.format = match format_str.as_str() {
                        "RGB" => ImageFormat::Rgb8,
                        "GRAY8" => ImageFormat::Mono8,
                        unsupported_format => {
                            return Err(StreamCaptureError::InvalidImageFormat(format!(
                                "Unsupported GStreamer format negotiated: {}",
                                unsupported_format
                            )));
                        }
                    };
                } else {
                    return Err(StreamCaptureError::InvalidConfig(
                        "Failed to get format from caps".to_string(),
                    ));
                }
            } else {
                return Err(StreamCaptureError::InvalidConfig(
                    "Failed to get caps structure from appsink".to_string(),
                ));
            }
        } else {
            return Err(StreamCaptureError::InvalidConfig(
                "Failed to get caps from appsink".to_string(),
            ));
        }

        // Set up bus message handling
        let bus = self.pipeline.bus().ok_or(StreamCaptureError::BusError)?;
        let is_eos_clone = self.is_eos.clone();

        let handle = std::thread::spawn(move || {
            for msg in bus.iter_timed(gst::ClockTime::NONE) {
                match msg.view() {
                    gst::MessageView::Eos(..) => {
                        log::debug!("gstreamer received EOS");
                        let mut is_eos = is_eos_clone.lock().unwrap();
                        *is_eos = true;
                        break;
                    }
                    gst::MessageView::Error(err) => {
                        log::error!(
                            "Error from {:?}: {} ({:?})",
                            msg.src().map(|s| s.path_string()),
                            err.error(),
                            err.debug()
                        );
                        let mut is_eos = is_eos_clone.lock().unwrap();
                        *is_eos = true;
                        break;
                    }
                    _ => {}
                }
            }
        });

        self.handle = Some(handle);

        Ok(())
    }

    /// Read a frame from the video.
    ///
    /// Returns an Image if a frame is available, or None if the end of the video has been reached.
    ///
    /// # Type Parameters
    ///
    /// * `C` - The number of channels in the returned Image. Must match the format of the video.
    pub fn read<const C: usize>(&mut self) -> Result<Option<Image<u8, C>>, StreamCaptureError> {
        // Check if we've already reached EOS
        if *self.is_eos.lock().unwrap() {
            return Ok(None);
        }

        // Check if the image channels match expected format before pulling sample
        let expected_channels = match self.format {
            ImageFormat::Mono8 => 1,
            ImageFormat::Rgb8 => 3,
        };

        if C != expected_channels {
            return Err(StreamCaptureError::InvalidImageFormat(format!(
                "Invalid number of channels: expected {}, got {}",
                expected_channels, C
            )));
        }

        // Get a sample with timeout to prevent indefinite blocking
        let timeout = gst::ClockTime::from_seconds(5);
        match self.appsink.try_pull_sample(timeout) {
            Ok(Some(sample)) => {
                // Get the buffer from the sample
                let buffer = sample.buffer().ok_or_else(|| {
                    StreamCaptureError::InvalidConfig(
                        "Failed to get buffer from sample".to_string(),
                    )
                })?;

                // Map the buffer as read-only
                let map = buffer.map_readable().map_err(|_| {
                    StreamCaptureError::InvalidConfig("Failed to map buffer".to_string())
                })?;

                // Get the data
                let data = map.as_slice();

                // Calculate expected buffer size based on image dimensions and channels
                let expected_size = self.size.width * self.size.height * C;
                if data.len() != expected_size {
                    log::warn!(
                        "Buffer size mismatch: expected {} bytes, got {} bytes",
                        expected_size,
                        data.len()
                    );

                    // If the buffer is too small, return an error
                    if data.len() < expected_size {
                        return Err(StreamCaptureError::InvalidConfig(format!(
                            "Buffer too small: expected {} bytes, got {} bytes",
                            expected_size,
                            data.len()
                        )));
                    }

                    // If the buffer is larger, we'll just use the needed portion
                }

                // Create an image from the data
                let img = Image::<u8, C>::new(self.size, data[..expected_size].to_vec())?;

                Ok(Some(img))
            }
            Ok(None) => {
                // No sample available within timeout, check if EOS
                if *self.is_eos.lock().unwrap() {
                    Ok(None)
                } else {
                    Err(StreamCaptureError::InvalidConfig(
                        "Timeout while waiting for frame".to_string(),
                    ))
                }
            }
            Err(gst::FlowError::Eos) => {
                // Set EOS flag and return None to indicate end of stream
                let mut is_eos = self.is_eos.lock().unwrap();
                *is_eos = true;
                log::debug!("VideoReader reached EOS");
                Ok(None)
            }
            Err(err) => {
                // Handle other GStreamer flow errors
                log::error!("Appsink pull_sample error: {:?}", err);
                Err(StreamCaptureError::InvalidConfig(format!(
                    "GStreamer pull_sample error: {:?}",
                    err
                )))
            }
        }
    }

    /// Get the frames per second of the video.
    pub fn fps(&self) -> f64 {
        self.fps
    }

    /// Get the size of the video frames.
    pub fn size(&self) -> ImageSize {
        self.size
    }

    /// Get the format of the video frames.
    pub fn format(&self) -> &ImageFormat {
        &self.format
    }

    /// Check if the video has reached the end.
    pub fn is_eos(&self) -> bool {
        *self.is_eos.lock().unwrap()
    }

    /// Close the video reader.
    ///
    /// Set the pipeline to null state and join the bus thread.
    pub fn close(&mut self) -> Result<(), StreamCaptureError> {
        self.pipeline.set_state(gst::State::Null)?;

        // Join the bus thread if it exists
        if let Some(handle) = self.handle.take() {
            handle.join().expect("Failed to join thread");
        }

        Ok(())
    }

    /// Seek to the specified position in seconds.
    ///
    /// # Arguments
    ///
    /// * `position_secs` - The position to seek to in seconds.
    pub fn seek(&mut self, position_secs: f64) -> Result<(), StreamCaptureError> {
        // Convert seconds to GStreamer ClockTime (nanoseconds)
        let position_ns = gst::ClockTime::from_nseconds((position_secs * 1_000_000_000.0) as u64);
        let seek_flags = gst::SeekFlags::FLUSH | gst::SeekFlags::KEY_UNIT;

        // Perform seek operation
        if !self.pipeline.seek(
            1.0,
            seek_flags,
            gst::SeekType::Set,
            position_ns,
            gst::SeekType::None,
            gst::ClockTime::NONE,
        ) {
            return Err(StreamCaptureError::InvalidConfig(
                "Seek operation failed".to_string(),
            ));
        }

        // Reset EOS flag after successful seek
        let mut is_eos = self.is_eos.lock().unwrap();
        *is_eos = false;

        // Wait for the seek operation to complete
        let _ = self.pipeline.state(gst::ClockTime::from_seconds(1));

        Ok(())
    }

    /// Get the duration of the video in seconds.
    ///
    /// Returns None if the duration cannot be determined.
    pub fn duration(&self) -> Option<f64> {
        // Query the duration from the pipeline
        self.pipeline
            .query_duration::<gst::ClockTime>()
            .map(|duration| duration.seconds_f64())
    }
}

impl Drop for VideoReader {
    fn drop(&mut self) {
        if let Err(e) = self.close() {
            log::error!("Error closing video reader in drop: {}", e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ImageFormat, VideoCodec, VideoReader, VideoWriter};
    use kornia_image::{Image, ImageSize};

    #[ignore = "need gstreamer in CI"]
    #[test]
    fn video_writer_rgb8u() -> Result<(), Box<dyn std::error::Error>> {
        let tmp_dir = tempfile::tempdir()?;
        std::fs::create_dir_all(tmp_dir.path())?;
        let file_path = tmp_dir.path().join("test.mp4");
        let size = ImageSize {
            width: 6,
            height: 4,
        };
        let mut writer =
            VideoWriter::new(&file_path, VideoCodec::H264, ImageFormat::Rgb8, 30, size)?;
        writer.start()?;
        let img = Image::<u8, 3>::new(size, vec![0; size.width * size.height * 3])?;
        writer.write(&img)?;
        writer.close()?;
        assert!(file_path.exists(), "File does not exist: {:?}", file_path);
        Ok(())
    }

    #[ignore = "need gstreamer in CI"]
    #[test]
    fn video_writer_mono8u() -> Result<(), Box<dyn std::error::Error>> {
        let tmp_dir = tempfile::tempdir()?;
        std::fs::create_dir_all(tmp_dir.path())?;
        let file_path = tmp_dir.path().join("test.mp4");
        let size = ImageSize {
            width: 6,
            height: 4,
        };
        let mut writer =
            VideoWriter::new(&file_path, VideoCodec::H264, ImageFormat::Mono8, 30, size)?;
        writer.start()?;
        let img = Image::<u8, 1>::new(size, vec![0; size.width * size.height])?;
        writer.write(&img)?;
        writer.close()?;
        assert!(file_path.exists(), "File does not exist: {:?}", file_path);
        Ok(())
    }

    #[ignore = "need gstreamer in CI"]
    #[test]
    fn video_reader_basic() -> Result<(), Box<dyn std::error::Error>> {
        let test_file = "test_video.mp4"; // Adjust path as needed for your test
        let mut reader = VideoReader::new(test_file)?;
        reader.start()?;

        // Test video metadata
        println!("Video FPS: {}", reader.fps());
        println!(
            "Video Size: {}x{}",
            reader.size().width,
            reader.size().height
        );
        println!("Video Format: {:?}", reader.format());

        if let Some(duration) = reader.duration() {
            println!("Video Duration: {} seconds", duration);
        } else {
            println!("Video Duration: Unknown");
        }

        // Test reading first frame
        let frame = reader.read::<3>()?;
        assert!(frame.is_some(), "Failed to read first frame");

        // Test seeking
        reader.seek(1.0)?; // Seek to 1 second
        assert!(!reader.is_eos(), "EOS flag should be reset after seek");

        // Test proper closure
        reader.close()?;

        Ok(())
    }

    #[ignore = "need gstreamer in CI"]
    #[test]
    fn video_reader_writer_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
        // Create a simple test video
        let tmp_dir = tempfile::tempdir()?;
        let file_path = tmp_dir.path().join("roundtrip.mp4");
        let size = ImageSize {
            width: 320,
            height: 240,
        };

        // Create a test pattern
        let mut data = Vec::with_capacity(size.width * size.height * 3);
        for y in 0..size.height {
            for x in 0..size.width {
                data.push((x % 255) as u8); // R
                data.push((y % 255) as u8); // G
                data.push(((x + y) % 255) as u8); // B
            }
        }

        let img = Image::<u8, 3>::new(size, data)?;

        // Write test video with 5 frames
        let mut writer =
            VideoWriter::new(&file_path, VideoCodec::H264, ImageFormat::Rgb8, 30, size)?;
        writer.start()?;

        for _ in 0..5 {
            writer.write(&img)?;
        }

        writer.close()?;

        // Read back the video
        let mut reader = VideoReader::new(&file_path)?;
        reader.start()?;

        // The format and size should match our input
        assert_eq!(reader.size().width, size.width);
        assert_eq!(reader.size().height, size.height);

        // We should be able to read at least one frame
        let frame = reader.read::<3>()?;
        assert!(frame.is_some(), "Failed to read frame from written video");

        reader.close()?;

        Ok(())
    }
}
