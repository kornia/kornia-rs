use kornia::{
    image::{color_spaces::Rgb8, ImageSize},
    imgproc::{self, color::YuvToRgbMode},
    io::{
        fps_counter::FpsCounter,
        jpeg,
        v4l::{camera_control, PixelFormat, V4LCameraConfig, V4lVideoCapture},
    },
    tensor::CpuAllocator,
};
use kornia_vlm::smolvlm2::{InputMedia, Line, Message, Role, SmolVlm2, SmolVlm2Config};
use kornia_vlm::video::VideoSample;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

pub fn video_demo(args: &crate::Args) -> Result<(), Box<dyn std::error::Error>> {
    // Use ip_address and port from Args
    let rec = rerun::RecordingStreamBuilder::new("SmolVLM2 Example: Video Understanding")
        .connect_grpc_opts(format!(
            "rerun+http://{}:{}/proxy",
            args.ip_address, args.port
        ))?;

    // Create the cancellation token for the video capture
    let cancel_token = Arc::new(AtomicBool::new(false));

    // register a signal handler for graceful shutdown
    ctrlc::set_handler({
        let cancel_token = cancel_token.clone();
        move || {
            cancel_token.store(true, Ordering::SeqCst);
            println!("Sending timer cancel signal.");
        }
    })?;

    // Create the video capture object
    let img_size = ImageSize {
        width: 640,
        height: 480,
    };

    let pixel_format = match args.pixel_format.as_deref().unwrap_or("MJPG") {
        "MJPG" | "mjpg" => PixelFormat::MJPG,
        "YUYV" | "yuyv" => PixelFormat::YUYV,
        // Add more formats as needed
        other => {
            eprintln!("Unknown pixel format: {other}. Defaulting to MJPG.");
            PixelFormat::MJPG
        }
    };
    let mut webcam = V4lVideoCapture::new(V4LCameraConfig {
        device_path: format!("/dev/video{}", args.camera_id),
        size: img_size,
        fps: args.fps,
        format: pixel_format,
        buffer_size: 4,
    })?;

    println!("📹 Starting webcam capture...");
    println!("Requested FPS: {0}", args.fps);
    println!("Image size: {img_size:?}");

    if let Err(e) = webcam.set_control(ExposureDynamicFramerate(false)) {
        println!("⚠️ Could not disable dynamic framerate: {e}");
    }

    // Enable auto exposure and auto white balance for best image quality
    if let Err(e) = webcam.set_control(AutoExposure(AutoExposureMode::Priority)) {
        println!("⚠️ Could not enable aperture priority mode: {e}");
    }

    if let Err(e) = webcam.set_control(WhiteBalanceAutomatic(true)) {
        println!("⚠️ Could not enable white balance automatic: {e}");
    }

    let mut fps_counter = FpsCounter::new();

    // Pre-allocate RGB image buffer outside the loop
    let mut rgb_image = Rgb8::from_size_val(img_size, 0, CpuAllocator)?;

    let prompt = &args.prompt as &str;
    let mut smolvlm2 = SmolVlm2::new(SmolVlm2Config::default())?;

    // === Video Understanding Implementation ===
    // This implementation uses Line::Video and InputMedia::Video for proper video understanding:
    //    - Uses Line::Video in message content
    //    - Passes entire Video<CpuAllocator> via InputMedia::Video
    //    - Leverages SmolVLM2's native video processing for temporal analysis
    //    - Provides holistic understanding of motion and temporal relationships
    //
    // The video buffer maintains a rolling window of frames with automatic cleanup
    // to prevent memory accumulation during long camera streaming.

    // Create a video object to manage frames with a rolling buffer
    const MAX_FRAMES_IN_BUFFER: usize = 32; // After around keeping 50 frames, CUDA OOM for 24gb GPU
    let mut frame_idx = 0;
    let mut video_buffer = VideoSample::<MAX_FRAMES_IN_BUFFER, CpuAllocator>::new();

    while !cancel_token.load(Ordering::SeqCst) {
        let Some(frame) = webcam.grab_frame()? else {
            continue;
        };

        // Convert YUYV to RGB if needed - reuse the pre-allocated buffer
        let buf = frame.buffer.as_slice();
        let decode_result: Result<(), Box<dyn std::error::Error>> = match frame.pixel_format {
            PixelFormat::YUYV => {
                imgproc::color::convert_yuyv_to_rgb_u8(buf, &mut rgb_image, YuvToRgbMode::Bt601Full)
                    .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
            }
            PixelFormat::MJPG => jpeg::decode_image_jpeg_rgb8(buf, &mut rgb_image)
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),
            _ => {
                return Err(format!("Unsupported format: {}", frame.pixel_format).into());
            }
        };
        if let Err(e) = decode_result {
            eprintln!("[WARN] Skipping corrupted frame: {e}");
            continue;
        }

        // Add frame to video buffer and manage buffer size
        video_buffer.add_frame(rgb_image.clone(), frame_idx);

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

        // Log the frame to rerun
        rec.log(
            "video_capture",
            &rerun::Image::from_elements(
                rgb_image.as_slice(),
                img_size.into(),
                rerun::ColorModel::RGB,
            ),
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
            fps_counter.update();
            println!("FPS: {:.2}", fps_counter.fps());
            println!("Frame {frame_idx}: {response}");
            println!(
                "Video buffer contains {} frames",
                video_buffer.frames().len()
            );
            println!("Using video understanding with Line::Video");
        }

        frame_idx += 1;
    }

    Ok(())
}

// some camera controls obtained from: v4l2-ctl --device=/dev/video0 --all

#[derive(Debug)]
struct ExposureDynamicFramerate(pub bool);

#[rustfmt::skip]
impl camera_control::CameraControlTrait for ExposureDynamicFramerate {
    fn name(&self) -> &str { "dynamic_framerate" }
    fn control_id(&self) -> u32 { 0x009a0903 }
    fn value(&self) -> camera_control::ControlType {
        camera_control::ControlType::Boolean(self.0)
    }
    fn description(&self) -> String { "Dynamic framerate control".to_string() }
}

#[derive(Debug, Copy, Clone)]
enum AutoExposureMode {
    #[allow(dead_code)]
    Manual = 1,
    #[allow(dead_code)]
    Priority = 3, // Aperture Priority Mode
}

#[derive(Debug)]
struct AutoExposure(pub AutoExposureMode);

#[rustfmt::skip]
impl camera_control::CameraControlTrait for AutoExposure {
    fn name(&self) -> &str { "auto_exposure" }
    fn control_id(&self) -> u32 { 0x009a0901 }
    fn value(&self) -> camera_control::ControlType {
        camera_control::ControlType::Integer(self.0 as i64)
    }
    fn description(&self) -> String { "Auto exposure control".to_string() }
}

#[derive(Debug)]
struct WhiteBalanceAutomatic(pub bool);

#[rustfmt::skip]
impl camera_control::CameraControlTrait for WhiteBalanceAutomatic {
    fn name(&self) -> &str { "white_balance_automatic" }
    fn control_id(&self) -> u32 { 0x0098090c }
    fn value(&self) -> camera_control::ControlType {
        camera_control::ControlType::Boolean(self.0)
    }
    fn description(&self) -> String { "White balance automatic control".to_string() }
}
