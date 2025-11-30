use argh::FromArgs;
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
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

#[derive(FromArgs)]
/// Capture frames from a webcam and log to Rerun
struct Args {
    /// the camera id to use
    #[argh(option, short = 'c', default = "0")]
    camera_id: u32,

    /// the frames per second to record
    #[argh(option, short = 'f', default = "30")]
    fps: u32,

    /// the pixel format to use
    #[argh(option, short = 'p', default = "PixelFormat::YUYV")]
    pixel_format: PixelFormat,

    /// debug the frame rate
    #[argh(switch)]
    debug: bool,
}

pub fn v4l_demo() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = argh::from_env();

    // Create the recording stream
    let rec = rerun::RecordingStreamBuilder::new("Kornia Webcapture App").spawn()?;

    // Create the cancellation token for the video capture
    let cancel_token = Arc::new(AtomicBool::new(false));
    let cancel_token_clone = Arc::clone(&cancel_token);

    // register a signal handler for graceful shutdown
    ctrlc::set_handler(move || {
        cancel_token_clone.store(true, Ordering::SeqCst);
        println!("Sending timer cancel signal !!");
    })?;

    // Create the video capture object
    let img_size = ImageSize {
        width: 640,
        height: 480,
    };

    let mut webcam = V4lVideoCapture::new(V4LCameraConfig {
        device_path: format!("/dev/video{}", args.camera_id),
        size: img_size,
        fps: args.fps,
        format: args.pixel_format,
        buffer_size: 4,
    })?;

    println!("📹 Starting webcam capture...");
    println!("Requested FPS: {0}", args.fps);
    println!("Image size: {img_size:?}");

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

    while !cancel_token.load(Ordering::SeqCst) {
        let Some(frame) = webcam.grab_frame()? else {
            continue;
        };

        // Convert YUYV to RGB if needed - reuse the pre-allocated buffer
        let buf = frame.buffer.as_slice();
        match frame.pixel_format {
            PixelFormat::YUYV => {
                imgproc::color::convert_yuyv_to_rgb_u8(
                    buf,
                    &mut rgb_image,
                    YuvToRgbMode::Bt601Full,
                )?;
            }
            PixelFormat::MJPG => {
                jpeg::decode_image_jpeg_rgb8(buf, &mut rgb_image)?;
            }
            _ => {
                return Err(format!("Unsupported format: {}", frame.pixel_format).into());
            }
        }

        // Log the frame to rerun
        rec.log(
            "video_capture",
            &rerun::Image::from_elements(
                rgb_image.as_slice(),
                img_size.into(),
                rerun::ColorModel::RGB,
            ),
        )?;

        if args.debug {
            fps_counter.update();
            println!("FPS: {:.2}", fps_counter.fps());
        }
    }

    Ok(())
}

// some camera controls obtained from: v4l2-ctl --device=/dev/video0 --all

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
