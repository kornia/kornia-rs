//! Simple video understanding demo
//!
//! Usage: cargo run --example demo <video_path>
// use crate::video::{from_video_path, VideoSamplingMethod};
use kornia_tensor::CpuAllocator;
use kornia_vlm::smolvlm2::{InputMedia, Line, Message, Role, SmolVlm2, SmolVlm2Config};

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
            Some(InputMedia::Video(vec![&mut video])),
            max_tokens,
            CpuAllocator,
        )
        .unwrap_or_else(|e| format!("Model error: {:?}", e));

    println!("Model response: {}", response);
}

#[cfg(test)]
mod tests {
    use kornia_io::{jpeg::read_image_jpeg_rgb8, png::read_image_png_rgb8};
    use std::path::Path;

    // RUST_LOG=debug cargo test -p smol_vlm2 --features "gstreamer,cuda,flash-attn" -- --nocapture --ignored test_smolvlm2_image_inference_speed
    // cargo test --release -p smol_vlm2 --features "gstreamer,cuda,flash-attn" -- --nocapture --ignored test_smolvlm2_image_inference_speed
    /// Single image inference speed test
    #[test]
    #[ignore = "Requires CUDA"]
    fn test_smolvlm2_image_inference_speed() {
        env_logger::init();

        log::info!("============================================================");
        log::info!("SMOLVLM2 RUST IMAGE INFERENCE SPEED TEST");
        log::info!("============================================================");

        let path = Path::new("../../100462016.jpeg");
        let image = match path.extension().and_then(|ext| ext.to_str()) {
            Some("jpg") | Some("jpeg") => read_image_jpeg_rgb8(path).ok(),
            Some("png") => read_image_png_rgb8(path).ok(),
            _ => None,
        };
        let image = match image {
            Some(img) => img,
            None => {
                log::info!("Image file not found or failed to load: {:?}", path);
                return;
            }
        };

        let config = SmolVlm2Config {
            seed: 42,
            do_sample: false,
            debug: true,
            ..Default::default()
        };
        let mut model = SmolVlm2::<32, CpuAllocator>::new(config).unwrap();
        let sample_len = 500;

        let mut inference_times = Vec::new();
        let runs = 3;
        let prompts = [
            "Describe the image.",
            "What is the appearance of this image in details?",
            "What do you see?",
        ];
        for prompt in prompts.iter() {
            for run in 1..=runs {
                model.clear_context().unwrap();
                let start_time = std::time::Instant::now();
                let response = model
                    .inference(
                        vec![Message {
                            role: Role::User,
                            content: vec![
                                Line::Image,
                                Line::Text {
                                    text: prompt.to_string(),
                                },
                            ],
                        }],
                        Some(InputMedia::Images(vec![image.clone()])),
                        sample_len,
                        CpuAllocator,
                    )
                    .unwrap_or_else(|e| format!("Inference failed: {:?}", e));
                let duration = start_time.elapsed();
                let time_secs = duration.as_secs_f64();
                inference_times.push(time_secs);
                log::info!(
                    "Prompt: {} | Run {}: inference completed in {:.3}s",
                    prompt,
                    run,
                    time_secs
                );
                log::info!("Model response: {}", response);
            }
        }

        if !inference_times.is_empty() {
            let avg_time = inference_times.iter().sum::<f64>() / inference_times.len() as f64;
            log::info!("Average inference time: {:.3}s", avg_time);
        }
    }
    use super::*;

    // RUST_LOG=debug cargo test -p smol_vlm2 --features "gstreamer,cuda,flash-attn" -- --nocapture --ignored test_smolvlm2_video_inference_speed
    // cargo test --release -p smol_vlm2 --features "gstreamer,cuda,flash-attn" -- --nocapture --ignored test_smolvlm2_video_inference_speed
    /// Video inference speed test (previously speed comparison)
    #[test]
    #[ignore = "Video inference speed test - requires CUDA/gstreamer"]
    fn test_smolvlm2_video_inference_speed() {
        env_logger::init();

        log::info!("============================================================");
        log::info!("SMOLVLM2 RUST SPEED TEST RESULTS");
        log::info!("============================================================");

        // Video test section (measure video loading and model inference speed)
        #[cfg(feature = "gstreamer")]
        {
            log::info!("\n==================== Video 32 frames ====================\n");

            let video_path = Path::new("../../example_video.mp4");
            if video_path.exists() {
                let video_prompts = [
                    "Describe what happens in this video.",
                    "What do you see in this video?",
                    "Summarize the video content.",
                ];

                let mut video_total_time = 0.0;
                let mut inference_total_time = 0.0;
                let mut video_test_count = 0;

                let config = SmolVlm2Config {
                    seed: 42,
                    do_sample: false,
                    debug: true,
                    ..Default::default()
                };
                let mut model = SmolVlm2::<32, CpuAllocator>::new(config)
                    .expect("Failed to create SmolVLM2 model");

                for (i, prompt) in video_prompts.iter().enumerate() {
                    model.clear_context().unwrap();
                    log::info!("Video Test {}: '{}'", i + 1, prompt);

                    let mut load_times = Vec::new();
                    let mut inference_times = Vec::new();
                    for run in 1..=2 {
                        // Measure video loading
                        let start_load = std::time::Instant::now();
                        let video_result = from_video_path::<32, _, CpuAllocator>(
                            video_path,
                            VideoSamplingMethod::FirstN(32),
                            CpuAllocator,
                        );
                        let load_duration = start_load.elapsed();
                        let load_secs = load_duration.as_secs_f64();
                        load_times.push(load_secs);

                        match video_result {
                            Ok(mut video) => {
                                log::info!(
                                    "  Run {}: loaded {} frames in {:.3}s",
                                    run,
                                    video.frames().len(),
                                    load_secs
                                );
                                video_total_time += load_secs;
                                video_test_count += 1;

                                // Measure model inference
                                let start_infer = std::time::Instant::now();
                                let messages = vec![Message {
                                    role: Role::User,
                                    content: vec![
                                        Line::Video,
                                        Line::Text {
                                            text: prompt.to_string(),
                                        },
                                    ],
                                }];
                                let response = model
                                    .inference(
                                        messages,
                                        Some(InputMedia::Video(vec![&mut video])),
                                        500,
                                        CpuAllocator,
                                    )
                                    .unwrap_or_else(|e| format!("Model error: {:?}", e));
                                let infer_duration = start_infer.elapsed();
                                let infer_secs = infer_duration.as_secs_f64();
                                inference_times.push(infer_secs);
                                inference_total_time += infer_secs;
                                log::info!("  Run {}: model inference in {:.3}s", run, infer_secs);
                                log::info!("  Model response: {}", response);
                            }
                            Err(e) => {
                                log::info!("  Failed to load video: {}", e);
                                break;
                            }
                        }
                    }

                    if !load_times.is_empty() {
                        let avg_load = load_times.iter().sum::<f64>() / load_times.len() as f64;
                        log::info!("  Average load time: {:.3}s", avg_load);
                    }
                    if !inference_times.is_empty() {
                        let avg_infer =
                            inference_times.iter().sum::<f64>() / inference_times.len() as f64;
                        log::info!("  Average inference time: {:.3}s", avg_infer);
                    }
                }

                // Overall video performance summary
                if video_test_count > 0 {
                    let video_avg_time = video_total_time / video_test_count as f64;
                    let inference_avg_time = inference_total_time / video_test_count as f64;
                    log::info!("🏁 Video 32 frames Section Performance:");
                    log::info!("   Average Load Time: {:.3}s", video_avg_time);
                    log::info!("   Average Inference Time: {:.3}s", inference_avg_time);
                    log::info!("   Total Tests: {}", video_test_count);
                }
            } else {
                log::info!("  Video file not found: {:?}", video_path);
            }
        }

        #[cfg(not(feature = "gstreamer"))]
        {
            log::info!("\n==================== Video Test ====================\n");
            log::info!("  Video testing requires 'gstreamer' feature to be enabled");
            log::info!("  Run with: cargo test --features \"gstreamer,cuda\"");
        }
    }

    /// Single image inference test (mirrors previous library test)
    #[test]
    #[ignore = "Requires CUDA"]
    fn test_smolvlm2_text_inference() {
        env_logger::init();

        let path = std::path::Path::new("../../100462016.jpeg");

        // Load image (JPEG/PNG)
        let image = match path.extension().and_then(|ext| ext.to_str()) {
            Some("jpg") | Some("jpeg") => read_image_jpeg_rgb8(path).ok(),
            Some("png") => read_image_png_rgb8(path).ok(),
            _ => None,
        };
        let image = match image {
            Some(img) => img,
            None => {
                log::info!("Image file not found or failed to load: {:?}", path);
                return;
            }
        };

        let config = SmolVlm2Config {
            seed: 42,
            do_sample: false,
            debug: true,
            ..Default::default()
        };
        let mut model = SmolVlm2::<32, CpuAllocator>::new(config).unwrap();

        let prompt = "Describe the image.";
        let sample_len = 500;

        let response = model
            .inference(
                vec![Message {
                    role: Role::User,
                    content: vec![
                        Line::Image,
                        Line::Text {
                            text: prompt.to_string(),
                        },
                    ],
                }],
                Some(InputMedia::Images(vec![image.into_inner()])),
                sample_len,
                CpuAllocator,
            )
            .unwrap_or_else(|e| format!("Inference failed: {:?}", e));

        log::info!("Model response: {}", response);
    }
}
