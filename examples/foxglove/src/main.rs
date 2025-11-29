use argh::FromArgs;
use foxglove::{
    schemas::{CompressedImage, Timestamp},
    WebSocketServer,
};
use kornia_image::ImageSize;
use kornia_io::v4l::{PixelFormat, V4LCameraConfig, V4lVideoCapture};
use std::{
    sync::atomic::{AtomicBool, Ordering},
    sync::Arc,
};

#[derive(FromArgs)]
/// Foxglove demo application
struct Args {
    /// the port to use
    #[argh(option, short = 'c', default = "0")]
    camera_id: u32,

    /// the frames per second to record
    #[argh(option, short = 'f', default = "30")]
    fps: u32,
}

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let env = env_logger::Env::default().default_filter_or("debug");
    env_logger::init_from_env(env);

    let args: Args = argh::from_env();

    // create the cancellation token
    let cancel_token = Arc::new(AtomicBool::new(false));
    let cancel_token_clone = Arc::clone(&cancel_token);

    // register a signal handler for graceful shutdown
    ctrlc::set_handler(move || {
        cancel_token_clone.store(true, Ordering::SeqCst);
        println!("Sending timer cancel signal !!");
    })?;

    // create the web socket server
    let server = WebSocketServer::new().start_blocking()?;

    // the pixel format to use
    let pixel_format = PixelFormat::MJPG;

    // start the camera capture
    let mut camera = V4lVideoCapture::new(V4LCameraConfig {
        device_path: format!("/dev/video{}", args.camera_id),
        size: ImageSize {
            width: 640,
            height: 480,
        },
        fps: args.fps,
        format: pixel_format,
        buffer_size: 4,
    })?;

    while !cancel_token.load(Ordering::SeqCst) {
        let Some(frame) = camera.grab_frame()? else {
            continue;
        };

        // convert the frame to a compressed image
        let compressed_image = CompressedImage {
            format: pixel_format.as_str().to_lowercase(),
            data: frame.buffer.into_vec().into(),
            timestamp: Some(Timestamp::new(
                frame.timestamp.sec as u32,
                frame.timestamp.usec as u32,
            )),
            frame_id: "camera".to_string(),
        };

        // send the compressed image to the web socket server
        foxglove::log!("/camera/compressed", compressed_image);
    }

    // wait for the server to stop
    server.stop().wait_blocking();

    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn main() {
    panic!("This example is only supported on Linux due to V4L dependency.");
}
