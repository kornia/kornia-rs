use clap::Parser;
use std::{path::Path, sync::Arc};
use tokio::signal;
use tokio::sync::Mutex;

use kornia::{
    image::{Image, ImageSize},
    io::stream::{video::VideoWriterCodec, V4L2CameraConfig, VideoWriter},
};

#[derive(Parser)]
struct Args {
    #[arg(short, long, default_value = "0")]
    camera_id: u32,

    #[arg(short, long, default_value = "30")]
    fps: i32,

    #[arg(short, long)]
    duration: Option<u64>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

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
        Path::new("output.mp4"),
        VideoWriterCodec::H264,
        args.fps,
        frame_size,
    )?;
    let video_writer = Arc::new(Mutex::new(video_writer));
    video_writer.lock().await.start()?;

    // Create a channel to send frames to the video writer
    let (tx, rx) = tokio::sync::mpsc::channel::<Arc<Mutex<Image<u8, 3>>>>(32);
    let rx = Arc::new(Mutex::new(rx));

    // Spawn a task to read frames from the camera and send them to the video writer
    let video_writer_task = tokio::spawn({
        let rx = rx.clone();
        let video_writer = video_writer.clone();
        async move {
            while let Some(img) = rx.lock().await.recv().await {
                // lock the image and write it to the video writer
                let img = img.lock().await;
                video_writer
                    .lock()
                    .await
                    .write(&img)
                    .expect("Failed to write image to video writer");
            }
            Ok::<_, Box<dyn std::error::Error + Send + Sync>>(())
        }
    });

    // Visualization thread
    let visualization_task = tokio::spawn({
        let rec = rec.clone();
        let rx = rx.clone();
        async move {
            while let Some(img) = rx.lock().await.recv().await {
                // lock the image and log it
                let img = img.lock().await;
                rec.log_static(
                    "image",
                    &rerun::Image::from_elements(
                        img.as_slice(),
                        img.size().into(),
                        rerun::ColorModel::RGB,
                    ),
                )?;
            }
            Ok::<_, Box<dyn std::error::Error + Send + Sync>>(())
        }
    });

    // start grabbing frames from the camera
    let capture = webcam.run_with_termination(
        |img| {
            let tx = tx.clone();
            async move {
                // send the image to the video writer and the visualization
                tx.send(Arc::new(Mutex::new(img))).await?;
                Ok(())
            }
        },
        async {
            signal::ctrl_c().await.expect("Failed to listen for Ctrl+C");
            println!("👋 Finished recording. Closing app.");
        },
    );

    tokio::select! {
        _ = capture => (),
        _ = video_writer_task => (),
        _ = visualization_task => (),
        _ = signal::ctrl_c() => (),
    }

    video_writer
        .lock()
        .await
        .stop()
        .expect("Failed to stop video writer");

    Ok(())
}
