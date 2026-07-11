use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use gstreamer_video as gst_video;
use kornia::image::{Image, ImageSize};
use kornia_vlm::smolvlm2::{InputMedia, Line, Message, Role, SmolVlm2, SmolVlm2Config};
use kornia_vlm::video::VideoSample;

use crate::Args;
use std::error::Error;

/// Run SmolVLM2 over a video file.
///
/// Reads the file with `--fps` frame-rate control at the GStreamer pipeline
/// (`videorate`), keeps a rolling buffer of the last `max_frames_in_buffer`
/// frames, and passes the buffer to SmolVLM2 as `InputMedia::Video` so the model
/// sees temporal context. Results are logged to Rerun.
///
/// ```bash
/// cargo run --bin smol_vlm2_video --video-file video.mp4 --prompt "What do you see?" --fps 2
/// ```
pub fn video_file_demo(args: &Args) -> Result<(), Box<dyn Error>> {
    // Use ip_address and port from Args
    let rec = rerun::RecordingStreamBuilder::new("SmolVLM2 Example: Video Understanding")
        .connect_grpc_opts(format!(
            "rerun+http://{}:{}/proxy",
            args.ip_address, args.port
        ))?;

    gst::init()?;

    let pipeline_str = format!(
        "filesrc location={} ! decodebin ! videoconvert ! videorate ! video/x-raw,format=RGB,framerate={}/1 ! appsink name=sink",
        args.video_file.as_ref().expect("Expected video files to be present when calling this function."),
        args.fps
    );
    let pipeline = gst::parse::launch(&pipeline_str)?;
    let appsink = pipeline
        .clone()
        .dynamic_cast::<gst::Bin>()
        .map_err(|_| "Failed to cast pipeline to Bin")?
        .by_name("sink")
        .ok_or("Failed to find sink element")?
        .dynamic_cast::<gst_app::AppSink>()
        .map_err(|_| "Failed to cast sink to AppSink")?;

    pipeline.set_state(gst::State::Playing)?;
    let mut smolvlm2 = SmolVlm2::new(SmolVlm2Config::default())?;

    let prompt = &args.prompt as &str;
    let mut frame_idx = 0;

    const MAX_FRAMES_IN_BUFFER: usize = 32;

    // Keeping around 50 frames caused CUDA OOM on a 24GB GPU, so we use 32 as a safety margin.
    let mut video_buffer = VideoSample::<MAX_FRAMES_IN_BUFFER>::new();

    // FPS tracking variables
    let mut last_frame_time = std::time::Instant::now();
    while let Ok(sample) = appsink.pull_sample() {
        let buffer = sample.buffer().ok_or("No buffer in sample")?;
        let map = buffer.map_readable().map_err(|_| "Failed to map buffer")?;
        let caps = sample.caps().ok_or("No caps in sample")?;
        let s = gst_video::VideoInfo::from_caps(caps)?;
        let width = s.width() as usize;
        let height = s.height() as usize;
        let img_size = ImageSize { width, height };
        let rgb_slice = map.as_ref();
        let image = Image::<u8, 3>::new(img_size, rgb_slice.to_vec())?;

        video_buffer.add_frame(image, frame_idx);

        smolvlm2.clear_context()?;

        // Create a message using Line::Video for proper video understanding
        let video_message = Message {
            role: Role::User,
            content: vec![
                Line::Video,
                Line::Text {
                    text: prompt.to_string(),
                },
            ],
        };

        // Use the entire video buffer for video understanding
        let response = smolvlm2.inference(
            vec![video_message],
            Some(InputMedia::Video(vec![&mut video_buffer])),
            20,
        )?;

        // Log image and text to rerun (all using rgb_slice for image)
        rec.log(
            "video_capture",
            &rerun::Image::from_elements(rgb_slice, img_size.into(), rerun::ColorModel::RGB),
        )?;
        rec.log("prompt", &rerun::TextDocument::new(prompt))?;
        rec.log("response", &rerun::TextDocument::new(response.as_str()))?;

        // Log buffer status
        rec.log(
            "buffer_info",
            &rerun::TextDocument::new(format!(
                "Frame {}: Buffer contains {} frames (max: {})",
                frame_idx,
                video_buffer.frames().len(),
                MAX_FRAMES_IN_BUFFER
            )),
        )?;

        if args.debug {
            println!("Frame {frame_idx}: {response}");

            // Calculate and report FPS for the last frame only
            let current_time = std::time::Instant::now();
            let frame_duration = current_time.duration_since(last_frame_time);
            let instantaneous_fps = if frame_duration.as_secs_f64() > 0.0 {
                1.0 / frame_duration.as_secs_f64()
            } else {
                0.0
            };
            println!("Processing FPS (last frame): {:.2}", instantaneous_fps);
            last_frame_time = current_time;
        }
        frame_idx += 1;
    }
    pipeline.set_state(gst::State::Null)?;
    Ok(())
}
