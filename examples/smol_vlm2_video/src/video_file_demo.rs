use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use gstreamer_video as gst_video;
use kornia::{
    image::{Image, ImageSize},
    tensor::CpuAllocator,
};
use kornia_vlm::smolvlm2::{InputMedia, Line, Message, Role, SmolVlm2, SmolVlm2Config};
use kornia_vlm::video::Video;

use crate::Args;
use std::error::Error;

/// Video file demo with intelligent frame buffering and native video understanding.
///
/// This function demonstrates how to process video files with SmolVLM2 using proper
/// video understanding with `Line::Video` and the `Video` struct, providing comprehensive
/// temporal context analysis.
///
/// ## Features:
/// - **Native Video Understanding**: Uses `Line::Video` and `InputMedia::Video` for true video comprehension
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
/// ## Usage:
/// ```bash
/// cargo run --bin smol_vlm2_video --video-file video.mp4 --prompt "What do you see?"
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
        "filesrc location={} ! decodebin ! videoconvert ! video/x-raw,format=RGB ! appsink name=sink",
        args.video_file.as_ref().expect("Expected video files to be present when calling this function.")
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

    // === Video Understanding Implementation ===
    // This implementation uses Line::Video and InputMedia::Video for proper video understanding:
    //    - Uses Line::Video in message content
    //    - Passes entire Video<CpuAllocator> via InputMedia::Video
    //    - Leverages SmolVLM2's native video processing for temporal analysis
    //    - Provides holistic understanding of motion and temporal relationships
    //
    // The video buffer maintains a rolling window of frames with automatic cleanup
    // to prevent memory accumulation during long video processing.

    // Create a video object to manage frames with a rolling buffer
    let mut video_buffer = Video::<CpuAllocator>::new(vec![], vec![]);
    let max_frames_in_buffer = 16; // After around keeping 50 frames, CUDA OOM for 24gb GPU

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

        video_buffer.add_frame(image.clone(), frame_idx);
        video_buffer.remove_old_frames(max_frames_in_buffer);

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
            InputMedia::Video(vec![&mut video_buffer]),
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
                max_frames_in_buffer
            )),
        )?;

        if args.debug {
            println!("Frame {frame_idx}: {response}");
            println!(
                "Video buffer contains {} frames",
                video_buffer.frames().len()
            );
            println!("Using video understanding with Line::Video");
        }
        frame_idx += 1;
    }
    pipeline.set_state(gst::State::Null)?;
    Ok(())
}
