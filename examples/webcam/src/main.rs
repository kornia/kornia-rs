use argh::FromArgs;
use kornia::{
    image::{Image, ImageSize},
    imgproc::{self, color::YuvToRgbMode},
    io::{
        fps_counter::FpsCounter,
        jpeg,
        v4l::{AutoExposureMode, CameraControl, PixelFormat, V4LCameraConfig, V4LVideoCapture},
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
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

    let mut webcam = V4LVideoCapture::new(V4LCameraConfig {
        device_path: format!("/dev/video{}", args.camera_id),
        size: img_size,
        fps: args.fps,
        format: args.pixel_format,
    })?;

    println!("📹 Starting webcam capture...");
    println!("Requested FPS: {}", args.fps);
    println!("Image size: {:?}", img_size);

    // Enable auto exposure and auto white balance for best image quality
    if let Err(e) = webcam.set_control(CameraControl::AutoExposure(AutoExposureMode::Auto)) {
        println!("⚠️ Could not enable auto exposure: {}", e);
    }

    if let Err(e) = webcam.set_control(CameraControl::AutoWhiteBalance(true)) {
        println!("⚠️ Could not enable auto white balance: {}", e);
    }

    // For manual control, disable auto and set specific values
    if let Err(e) = webcam.set_control(CameraControl::AutoExposure(AutoExposureMode::Auto)) {
        println!("⚠️ Could not set manual exposure: {}", e);
    }

    if let Err(e) = webcam.set_control(CameraControl::AutoWhiteBalance(false)) {
        println!("⚠️ Could not disable auto white balance: {}", e);
    }

    let mut fps_counter = FpsCounter::new();

    while !cancel_token.load(Ordering::SeqCst) {
        let Ok(Some(frame)) = webcam.grab() else {
            continue;
        };

        // Convert YUYV to RGB if needed
        let mut rgb_image = Image::<u8, 3, CpuAllocator>::from_size_val(img_size, 0, CpuAllocator)?;

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
