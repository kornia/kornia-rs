use clap::Parser;
use std::{path::PathBuf, sync::Arc};
use tokio::signal;
use tokio::sync::Mutex;

use kornia::{
    image::ImageSize,
    io::stream::{
        video::{ImageFormat, VideoCodec},
        V4L2CameraConfig, VideoWriter,
    },
};

#[derive(Parser)]
struct Args {
    #[arg(short, long)]
    output: PathBuf,

    #[arg(short, long, default_value = "0")]
    camera_id: u32,

    #[arg(short, long, default_value = "30")]
    fps: i32,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // setup logging
    let _ = env_logger::builder().is_test(true).try_init();

    let args = Args::parse();

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
    let webcam = V4L2CameraConfig::new()
        .with_camera_id(args.camera_id)
        .with_fps(args.fps as u32)
        .with_size(frame_size)
        .build()?;

    // start the video writer
    let video_writer = VideoWriter::new(
        args.output,
        VideoCodec::H264,
        ImageFormat::Rgb8,
        args.fps,
        frame_size,
    )?;
    let video_writer = Arc::new(Mutex::new(video_writer));
    video_writer.lock().await.start()?;

    // start grabbing frames from the camera
    webcam
        .run_with_termination(
            |img| {
                let rec = rec.clone();
                let video_writer = video_writer.clone();
                async move {
                    // write the image to the video writer
                    video_writer.lock().await.write(&img)?;

                    // log the image
                    rec.log_static(
                        "image",
                        &rerun::Image::from_elements(
                            img.as_slice(),
                            img.size().into(),
                            rerun::ColorModel::RGB,
                        ),
                    )?;
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
    video_writer.lock().await.stop()?;

    Ok(())
}
