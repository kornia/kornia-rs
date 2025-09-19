#[cfg(target_os = "linux")]
mod video_demo;
#[cfg(target_os = "linux")]
mod video_file_demo;

use argh::FromArgs;

#[derive(FromArgs)]
/// Capture frames from a webcam or video file and log to Rerun
struct Args {
    /// the camera id to use (ignored if --video-file is set)
    #[argh(option, short = 'c', default = "0")]
    camera_id: u32,

    /// the frames per second to record
    #[argh(option, short = 'f', default = "30")]
    fps: u32,

    /// the pixel format to use (ignored if --video-file is set)
    #[argh(option, short = 'p')]
    pixel_format: Option<String>,

    /// debug the frame rate
    #[argh(switch)]
    debug: bool,

    /// path to a video file to use instead of webcam
    #[argh(option)]
    video_file: Option<String>,
}

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = argh::from_env();
    let _pixel_format = args.pixel_format.as_deref().unwrap_or("MJPG");
    if let Some(path) = &args.video_file {
        video_file_demo::video_file_demo(&args, path)
    } else {
        video_demo::video_demo(&args)
    }
}

#[cfg(not(target_os = "linux"))]
fn main() {
    panic!("This example is only supported on Linux due to V4L dep.");
}
