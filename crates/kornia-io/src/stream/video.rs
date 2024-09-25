use std::path::Path;

use futures::prelude::*;
use gst::prelude::*;

use async_std::task;
use kornia_image::{Image, ImageFormat, ImageSize};

use super::StreamCaptureError;

/// The codec to use for the video writer.
pub enum VideoWriterCodec {
    /// H.264 codec.
    H264,
}

/// A struct for writing video files.
pub struct VideoWriter {
    pipeline: gst::Pipeline,
    appsrc: gst_app::AppSrc,
    fps: i32,
    counter: u64,
}

impl VideoWriter {
    /// Create a new VideoWriter.
    ///
    /// # Arguments
    ///
    /// * `path` - The path to save the video file.
    /// * `codec` - The codec to use for the video writer.
    /// * `fps` - The frames per second of the video.
    /// * `size` - The size of the video.
    pub fn new(
        path: impl AsRef<Path>,
        codec: VideoWriterCodec,
        format: ImageFormat,
        fps: i32,
        size: ImageSize,
    ) -> Result<Self, StreamCaptureError> {
        gst::init()?;

        // TODO: Add support for other codecs
        #[allow(unreachable_patterns)]
        let _codec = match codec {
            VideoWriterCodec::H264 => "x264enc",
            _ => {
                return Err(StreamCaptureError::InvalidConfig(
                    "Unsupported codec".to_string(),
                ))
            }
        };

        // TODO: Add support for other formats
        let format = match format {
            ImageFormat::Mono8U => "GRAY8",
            ImageFormat::Rgb8U => "RGB",
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
            .field("format", format)
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
            counter: 0,
        })
    }

    /// Start the video writer
    pub fn start(&self) -> Result<(), StreamCaptureError> {
        self.pipeline.set_state(gst::State::Playing)?;

        let bus = self.pipeline.bus().ok_or(StreamCaptureError::BusError)?;
        let mut messages = bus.stream();

        // launch a task to handle the bus messages, exit when EOS is received and set the pipeline to null
        task::spawn({
            let pipeline = self.pipeline.clone();
            async move {
                while let Some(msg) = messages.next().await {
                    match msg.view() {
                        gst::MessageView::Eos(..) => {
                            println!("EOS");
                            break;
                        }
                        gst::MessageView::Error(err) => {
                            eprintln!(
                                "Error from {:?}: {} ({:?})",
                                msg.src().map(|s| s.path_string()),
                                err.error(),
                                err.debug()
                            );
                        }
                        _ => {}
                    }
                }
                pipeline.set_state(gst::State::Null).unwrap();
            }
        });

        Ok(())
    }

    /// Stop the video writer
    pub fn stop(&mut self) -> Result<(), StreamCaptureError> {
        // Send end of stream to the appsrc
        self.appsrc
            .end_of_stream()
            .map_err(StreamCaptureError::GstreamerFlowError)?;

        Ok(())
    }

    /// Write an image to the video file.
    ///
    /// # Arguments
    ///
    /// * `img` - The image to write to the video file.
    // TODO: support write_async
    pub fn write<const C: usize>(&mut self, img: &Image<u8, C>) -> Result<(), StreamCaptureError> {
        // check if the image channels are correct
        if C != 1 && C != 3 {
            return Err(StreamCaptureError::InvalidConfig(
                "Invalid number of channels".to_string(),
            ));
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
        self.stop().unwrap_or_else(|e| {
            eprintln!("Error stopping video writer: {:?}", e);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::{ImageFormat, VideoWriter, VideoWriterCodec};
    use kornia_image::{Image, ImageSize};

    #[test]
    #[ignore = "TODO: fix this test as there's a race condition in the gstreamer flow"]
    fn video_writer() -> Result<(), Box<dyn std::error::Error>> {
        let tmp_dir = tempfile::tempdir()?;
        std::fs::create_dir_all(tmp_dir.path())?;

        let file_path = tmp_dir.path().join("test.mp4");

        let size = ImageSize {
            width: 6,
            height: 4,
        };
        let mut writer = VideoWriter::new(
            &file_path,
            VideoWriterCodec::H264,
            ImageFormat::RGB8U,
            30,
            size,
        )?;
        writer.start()?;

        let img = Image::new(size, vec![0; size.width * size.height * 3])?;
        writer.write(&img)?;
        writer.stop()?;

        assert!(file_path.exists(), "File does not exist: {:?}", file_path);

        Ok(())
    }
}
