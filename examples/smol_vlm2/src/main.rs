mod demo;
mod video;

use argh::FromArgs;

#[derive(FromArgs)]
/// video understanding demo CLI
struct DemoArgs {
    /// path to video file
    #[argh(option)]
    #[allow(dead_code)]
    video_path: String,

    /// sampling method: uniform, fps, firstn, indices
    #[argh(option)]
    #[allow(dead_code)]
    sampling: String,

    /// number of frames to sample (for uniform, fps, firstn)
    #[argh(option, default = "8")]
    #[allow(dead_code)]
    sample_frames: usize,

    /// maximum number of generated tokens
    #[argh(option, default = "128")]
    #[allow(dead_code)]
    max_tokens: usize,

    /// prompt for the model
    #[argh(option)]
    prompt: String,
}

fn main() {
    let mut args: DemoArgs = argh::from_env();
    if args.prompt == "Describe" {
        args.prompt = "Describe the video.".to_string();
    }

    #[cfg(feature = "gstreamer")]
    {
        demo::run_video_demo(
            &args.video_path,
            &args.sampling,
            args.sample_frames,
            &args.prompt,
            args.max_tokens,
        );
    }
    #[cfg(not(feature = "gstreamer"))]
    {
        eprintln!("This demo requires the 'gstreamer' feature to be enabled.");
        std::process::exit(1);
    }
}
