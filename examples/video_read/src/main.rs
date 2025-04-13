use argh::FromArgs;
use kornia::{
    image::ImageSize,
    io::stream::{
        video::{ImageFormat, VideoCodec, VideoReader},
        V4L2CameraConfig, VideoWriter,
    },
};
use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};

#[derive(FromArgs, Debug)]
/// Read the video from given file
struct Args {
    /// the camera id to use
    #[argh(option, short = 'c', default = "0")]
    camera_id: u32,
    /// duration of webcam stream to record
    #[argh(option, short = 'd', default = "30")]
    duration: u64,
    /// path to the input video file
    #[argh(option, short = 'o')]
    output: PathBuf,
    /// width of the video
    #[argh(option, short = 'w', default = "640")]
    width: usize,
    /// height of the video
    #[argh(option, short = 'h', default = "480")]
    height: usize,
    /// the frames per second of video3
    #[argh(option, short = 'f', default = "30")]
    fps: u32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = argh::from_env();

    // Ensure the input path ends with .mp4
    if args.output.extension().and_then(|ext| ext.to_str()) != Some("mp4") {
        return Err("Input file must have .mp4 extension".into());
    }

    // start the recoding stream
    let rec = Arc::new(rerun::RecordingStreamBuilder::new("Kornia Webcam to Video").spawn()?);

    // allocate the image buffers
    let frame_size = ImageSize {
        width: args.width,
        height: args.height,
    };

    // create a webcam capture object
    let mut webcam = V4L2CameraConfig::new()
        .with_camera_id(args.camera_id)
        .with_fps(args.fps)
        .with_size(frame_size)
        .build()?;

    webcam.start()?;

    // create a video writer object
    let mut video_writer = VideoWriter::new(
        args.output.clone(),
        VideoCodec::H264,
        ImageFormat::Rgb8,
        args.fps as i32,
        frame_size,
    )?;

    video_writer.start()?;

    // create a cancel token to stop the webcam capture
    let cancel_token = Arc::new(AtomicBool::new(false));
    let webcam_cancel_token = Arc::new(AtomicBool::new(false));

    ctrlc::set_handler({
        let cancel_token = cancel_token.clone();
        move || {
            println!("Received Ctrl-C signal. Sending cancel signal !!");
            cancel_token.store(true, Ordering::SeqCst);
        }
    })?;

    // launch timer for webcam
    std::thread::spawn({
        let webcam_cancel_token = webcam_cancel_token.clone();
        move || {
            std::thread::sleep(Duration::from_secs(args.duration));
            println!("Stopping Webcam Recording");
            webcam_cancel_token.store(true, Ordering::SeqCst);
        }
    });

    // Webcam recording loop
    while !cancel_token.load(Ordering::SeqCst) && !webcam_cancel_token.load(Ordering::SeqCst) {
        if let Some(image) = webcam.grab()? {
            video_writer.write(&image)?;

            rec.log_static(
                "webcam",
                &rerun::Image::from_elements(
                    image.as_slice(),
                    image.size().into(),
                    rerun::ColorModel::RGB,
                ),
            )?;
        }
    }

    // Close webcam and video writer after recording is done
    webcam.close()?;
    video_writer.close()?;

    println!("Starting video playback");

    // create a video reader object
    let mut video_reader = VideoReader::new(
        args.output,
        kornia::io::stream::video::ImageFormat::Rgb8,
        args.fps,
        frame_size,
    )?;

    // start the video reader capture
    video_reader.start()?;

    // Video playback loop
    while !cancel_token.load(Ordering::SeqCst) {
        if let Some(image) = video_reader.grab()? {
            rec.log_static(
                "video",
                &rerun::Image::from_elements(
                    image.as_slice(),
                    image.size().into(),
                    rerun::ColorModel::RGB,
                ),
            )?;
        }
    }

    video_reader.close()?;

    Ok(())
}
