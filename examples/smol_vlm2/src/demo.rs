//! Simple video understanding demo
//!
//! Usage: cargo run --example demo <video_path>
use crate::video::{from_video_path, VideoSamplingMethod};
use kornia_tensor::CpuAllocator; // Only keep if used
use kornia_vlm::smolvlm2::Line;
use kornia_vlm::smolvlm2::{InputMedia, Message, Role, SmolVlm2, SmolVlm2Config};

#[cfg(feature = "gstreamer")]
pub fn run_video_demo(
    video_path: &str,
    sampling: &str,
    sample_frames: usize,
    prompt: &str,
    max_tokens: usize,
) {
    let sampling_method = match sampling.to_lowercase().as_str() {
        "uniform" => VideoSamplingMethod::Uniform(sample_frames),
        "fps" => VideoSamplingMethod::Fps(sample_frames),
        "firstn" => VideoSamplingMethod::FirstN(sample_frames),
        "indices" => VideoSamplingMethod::Indices((0..sample_frames).collect()),
        _ => VideoSamplingMethod::Uniform(sample_frames),
    };

    let video = from_video_path::<32, _, CpuAllocator>(video_path, sampling_method, CpuAllocator);
    let mut video = match video {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Failed to load video: {:?}", e);
            std::process::exit(1);
        }
    };

    println!("Sampled {} frames from video.", video.frames().len());

    // Prepare SmolVLM2 model
    let config = SmolVlm2Config {
        debug: true,
        ..Default::default()
    };
    let mut model =
        SmolVlm2::<32, CpuAllocator>::new(config).expect("Failed to create SmolVLM2 model");

    // Prepare prompt
    let messages = vec![Message {
        role: Role::User,
        content: vec![
            Line::Video,
            Line::Text {
                text: prompt.to_string(),
            },
        ],
    }];

    // Run inference
    let response = model
        .inference(
            messages,
            InputMedia::Video(vec![&mut video]),
            max_tokens,
            CpuAllocator,
        )
        .unwrap_or_else(|e| format!("Model error: {:?}", e));

    println!("Model response: {}", response);
}
