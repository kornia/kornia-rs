use crate::Args;
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use gstreamer_video as gst_video;
use kornia::{
    image::{Image, ImageSize},
    tensor::CpuAllocator,
    vlm::smolvlm::{utils::SmolVlmConfig, SmolVlm},
};
use rerun;
use std::error::Error;

pub fn video_file_demo(args: &Args, path: &str) -> Result<(), Box<dyn Error>> {
    // Create rerun recording stream (match video_demo.rs)
    let ip_address = "192.168.1.9";
    let port = 9999;
    let rec = rerun::RecordingStreamBuilder::new("SmolVLM Example: Video File Captioning")
        .connect_grpc_opts(
            format!("rerun+http://{ip_address}:{port}/proxy"),
            rerun::default_flush_timeout(),
        )?;

    gst::init()?;
    println!("🎬 Reading video file: {}", path);
    let pipeline_str = format!("filesrc location={} ! decodebin ! videoconvert ! video/x-raw,format=RGB ! appsink name=sink", path);
    let pipeline = gst::parse_launch(&pipeline_str)?;
    let appsink = pipeline
        .clone()
        .dynamic_cast::<gst::Bin>()
        .unwrap()
        .by_name("sink")
        .unwrap()
        .dynamic_cast::<gst_app::AppSink>()
        .unwrap();

    pipeline.set_state(gst::State::Playing)?;
    let mut smolvlm = SmolVlm::new(SmolVlmConfig::default())?;

    let prompt = "What is the the color of the closest car? (Just color only)";
    let mut frame_idx = 0;

    while let Some(sample) = appsink.pull_sample().ok() {
        let buffer = sample.buffer().ok_or("No buffer in sample")?;
        let map = buffer.map_readable().map_err(|_| "Failed to map buffer")?;
        let caps = sample.caps().ok_or("No caps in sample")?;
        let s = gst_video::VideoInfo::from_caps(&caps)?;
        let width = s.width() as usize;
        let height = s.height() as usize;
        let img_size = ImageSize { width, height };
        let rgb_slice = map.as_ref();
        let image = Image::<u8, 3, CpuAllocator>::new(img_size, rgb_slice.to_vec(), CpuAllocator)?;
        smolvlm.clear_context();
        let response = smolvlm.inference(prompt, Some(image.clone()), 20, CpuAllocator)?;

        // Log image and text to rerun (all using rgb_slice for image)
        rec.log(
            "video_capture",
            &rerun::Image::from_elements(rgb_slice, img_size.into(), rerun::ColorModel::RGB),
        )
        .unwrap();
        rec.log("prompt", &rerun::TextDocument::new(prompt))
            .unwrap();
        rec.log("response", &rerun::TextDocument::new(response.as_str()))
            .unwrap();

        println!("Frame {}: {}", frame_idx, response);
        frame_idx += 1;
    }
    pipeline.set_state(gst::State::Null)?;
    Ok(())
}
