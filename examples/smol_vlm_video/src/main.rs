#[cfg(target_os = "linux")]
mod video_demo;
#[cfg(target_os = "linux")]
mod video_file_demo;

#[cfg(target_os = "linux")]
use argh::FromArgs;

#[cfg(target_os = "linux")]
#[derive(FromArgs)]
/// Capture frames from a webcam or video file and log to Rerun
struct Args {
    /// prompt to use for the model (required)
    #[argh(option)]
    prompt: String,
    /// the camera id to use (ignored if --video-file is set)
    #[argh(option, short = 'c', default = "0")]
    camera_id: u32,

    /// the frames per second to record (ignored if --video-file is set)
    #[argh(option, short = 'f', default = "30")]
    fps: u32,

    /// the pixel format to use (ignored if --video-file is set)
    #[argh(option, short = 'p')]
    pixel_format: Option<String>,

    /// path to a video file to use instead of webcam
    #[argh(option)]
    video_file: Option<String>,

    /// IP address of the rerun viewer device
    #[argh(option, short = 'i')]
    ip_address: String,

    /// port number of the rerun viewer device
    #[argh(option, short = 'P')]
    port: u16,

    /// debug the frame rate
    #[argh(switch)]
    debug: bool,
}

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = argh::from_env();
    let _pixel_format = args.pixel_format.as_deref().unwrap_or("MJPG");
    if args.video_file.is_some() {
        video_file_demo::video_file_demo(&args)
    } else {
        video_demo::video_demo(&args)
    }
}

#[cfg(not(target_os = "linux"))]
fn main() {
    panic!("This example is only supported on Linux due to V4L dep.");
}
