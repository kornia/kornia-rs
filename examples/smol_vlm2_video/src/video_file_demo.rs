use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use gstreamer_video as gst_video;
use kornia::{
    image::{Image, ImageSize},
    tensor::CpuAllocator,
};
use kornia_vlm::smolvlm2::{InputMedia, Line, Message, Role, SmolVlm2, SmolVlm2Config};
use kornia_vlm::video::VideoSample;

use crate::Args;
use std::error::Error;

/// Video file demo with intelligent frame buffering and native video understanding.
///
/// This function demonstrates how to process video files with SmolVLM2 using proper
/// video understanding with `Line::Video` and the `Video` struct, providing comprehensive
/// temporal context analysis with configurable FPS streaming.
///
/// ## Features:
/// - **Native Video Understanding**: Uses `Line::Video` and `InputMedia::Video` for true video comprehension
/// - **FPS Streaming Control**: Processes frames at the specified FPS rate to control processing speed
/// - **Automatic Frame Management**: Maintains a rolling buffer of frames with configurable size
/// - **Memory Optimization**: Automatically removes old frames to prevent memory accumulation
/// - **Temporal Context**: Analyzes entire video sequences for motion and temporal understanding
/// - **Real-time Logging**: Streams results to Rerun for visualization
///
/// ## Video Buffer Management:
/// - Creates a `Video<CpuAllocator>` object to manage frame history
/// - Adds new frames with timestamps for temporal tracking
/// - Automatically removes old frames when buffer exceeds `max_frames_in_buffer`
/// - Passes entire video buffer to SmolVLM2 for holistic video analysis
///
/// ## FPS Control:
/// - Uses the `--fps` argument to control video frame rate at the GStreamer pipeline level
/// - Employs `videorate` element to limit frame rate at the source, not just processing delays
/// - Ensures actual frame rate control rather than post-processing timing adjustments
/// - Provides consistent, hardware-level frame rate limiting
///
/// ## Usage:
/// ```bash
/// cargo run --bin smol_vlm2_video --video-file video.mp4 --prompt "What do you see?" --fps 2
/// ```
pub fn video_file_demo(args: &Args) -> Result<(), Box<dyn Error>> {
    // Use ip_address and port from Args
    let rec = rerun::RecordingStreamBuilder::new("SmolVLM2 Example: Video Understanding")
        .connect_grpc_opts(
            format!("rerun+http://{}:{}/proxy", args.ip_address, args.port),
            rerun::default_flush_timeout(),
        )?;

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
    let mut video_buffer = VideoSample::<MAX_FRAMES_IN_BUFFER, CpuAllocator>::new();

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
        let image = Image::<u8, 3, CpuAllocator>::new(img_size, rgb_slice.to_vec(), CpuAllocator)?;

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
            CpuAllocator,
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
