use clap::Parser;
use std::{
    path::Path,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
};
use tokio::signal;

use kornia::{
    image::{ops, Image, ImageSize},
    imgproc,
    io::{
        fps_counter::FpsCounter,
        stream::{StreamCaptureError, V4L2CameraConfig, VideoWriter},
    },
};

#[derive(Parser)]
struct Args {
    #[arg(short, long, default_value = "0")]
    camera_id: u32,

    #[arg(short, long, default_value = "30")]
    fps: u32,

    #[arg(short, long)]
    duration: Option<u64>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // start the recording stream
    let rec = rerun::RecordingStreamBuilder::new("Kornia Webcapture App").spawn()?;

    // allocate the image buffers
    let frame_size = ImageSize {
        width: 640,
        height: 480,
    };

    // create a webcam capture object with camera id 0
    // and force the image size to 640x480
    let webcam = V4L2CameraConfig::new()
        .with_camera_id(args.camera_id)
        .with_fps(args.fps)
        .with_size(frame_size)
        .build()?;

    // start the video writer
    let mut video_writer = VideoWriter::new(Path::new("output.mp4"), args.fps as f32, frame_size)?;
    video_writer.start()?;

    let video_writer = Arc::new(Mutex::new(video_writer));

    // start grabbing frames from the camera
    webcam
        .run_with_termination(
            |img| {
                let rec = rec.clone();
                let video_writer = video_writer.clone();
                async move {
                    // write the image to the video writer
                    video_writer.lock().unwrap().write(img)?;
                    //println!("Wrote frame");

                    // log the image
                    //rec.log_static(
                    //    "image",
                    //    &rerun::Image::from_elements(
                    //        img.as_slice(),
                    //        img.size().into(),
                    //        rerun::ColorModel::RGB,
                    //    ),
                    //)?;

                    Ok(())
                }
            },
            async {
                signal::ctrl_c().await.expect("Failed to listen for Ctrl+C");
                println!("👋 Finished recording. Closing app.");
            },
        )
        .await?;

    // stop the video writer
    video_writer.lock().unwrap().stop()?;

    Ok(())
}
