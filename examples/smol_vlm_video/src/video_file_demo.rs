use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use gstreamer_video as gst_video;
use kornia::{
    image::{Image, ImageSize},
    tensor::CpuAllocator,
};
use kornia_vlm::smolvlm::{utils::SmolVlmConfig, SmolVlm};

use crate::Args;
use std::error::Error;

pub fn video_file_demo(args: &Args) -> Result<(), Box<dyn Error>> {
    // Use ip_address and port from Args
    let rec = rerun::RecordingStreamBuilder::new("SmolVLM Example: Video File Captioning")
        .connect_grpc_opts(format!(
            "rerun+http://{}:{}/proxy",
            args.ip_address, args.port
        ))?;

    gst::init()?;

    let pipeline_str = format!(
        "filesrc location={} ! decodebin ! videoconvert ! video/x-raw,format=RGB ! appsink name=sink",
        args.video_file.as_ref().expect("Expected video files to be present when calling this function.")
    );
    let pipeline = gst::parse_launch(&pipeline_str)?;
    let appsink = pipeline
        .clone()
        .dynamic_cast::<gst::Bin>()
        .map_err(|_| "Failed to cast pipeline to Bin")?
        .by_name("sink")
        .ok_or("Failed to find sink element")?
        .dynamic_cast::<gst_app::AppSink>()
        .map_err(|_| "Failed to cast sink to AppSink")?;

    pipeline.set_state(gst::State::Playing)?;
    let mut smolvlm = SmolVlm::new(SmolVlmConfig::default())?;

    let prompt = &args.prompt as &str;
    let mut frame_idx = 0;

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
        smolvlm.clear_context()?;
        let response = smolvlm.inference(prompt, Some(image.clone()), 20, CpuAllocator)?;

        // Log image and text to rerun (all using rgb_slice for image)
        rec.log(
            "video_capture",
            &rerun::Image::from_elements(rgb_slice, img_size.into(), rerun::ColorModel::RGB),
        )?;
        rec.log("prompt", &rerun::TextDocument::new(prompt))?;
        rec.log("response", &rerun::TextDocument::new(response.as_str()))?;

        if args.debug {
            println!("Frame {frame_idx}: {response}");
        }
        frame_idx += 1;
    }
    pipeline.set_state(gst::State::Null)?;
    Ok(())
}
