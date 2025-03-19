use argh::FromArgs;
use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

use kornia::{
    image::ImageSize,
    io::stream::{
        video::{ImageFormat, VideoCodec},
        V4L2CameraConfig, VideoWriter,
    },
};

#[derive(FromArgs)]
/// Record video from a webcam
struct Args {
    /// path to the output video file
    #[argh(option, short = 'o')]
    output: PathBuf,

    /// the camera id to use
    #[argh(option, short = 'c', default = "0")]
    camera_id: u32,

    /// the frames per second to record
    #[argh(option, short = 'f', default = "30")]
    fps: i32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = argh::from_env();

    // Ensure the output path ends with .mp4
    if args.output.extension().and_then(|ext| ext.to_str()) != Some("mp4") {
        return Err("Output file must have a .mp4 extension".into());
    }

    // start the recording stream
    let rec = rerun::RecordingStreamBuilder::new("Kornia Video Write App").spawn()?;

    // allocate the image buffers
    let frame_size = ImageSize {
        width: 640,
        height: 480,
    };

    // create a webcam capture object with camera id 0
    // and force the image size to 640x480
    let mut webcam = V4L2CameraConfig::new()
        .with_camera_id(args.camera_id)
        .with_fps(args.fps as u32)
        .with_size(frame_size)
        .build()?;

    // start the webcam capture
    webcam.start()?;

    // start the video writer
    let mut video_writer = VideoWriter::new(
        args.output,
        VideoCodec::H264,
        ImageFormat::Rgb8,
        args.fps,
        frame_size,
    )?;

    // open the pipeline to start writing the video
    video_writer.start()?;

    // create a cancel token to stop the webcam capture
    let cancel_token = Arc::new(AtomicBool::new(false));

    ctrlc::set_handler({
        let cancel_token = cancel_token.clone();
        move || {
            println!("Received Ctrl-C signal. Sending cancel signal !!");
            cancel_token.store(true, Ordering::SeqCst);
        }
    })?;

    while !cancel_token.load(Ordering::SeqCst) {
        // start grabbing frames from the webcam
        if let Some(img) = webcam.grab()? {
            // write the image to the video writer
            video_writer.write(&img)?;

            // log the image
            rec.log_static(
                "image",
                &rerun::Image::from_elements(
                    img.as_slice(),
                    img.size().into(),
                    rerun::ColorModel::RGB,
                ),
            )?;
        }
    }

    // stop the video writer
    video_writer.close()?;

    Ok(())
}
