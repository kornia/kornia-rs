use argh::FromArgs;
use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};
use tokio::signal;
use tokio::sync::Mutex;

use kornia::{
    image::ImageSize,
    io::stream::{
        video::{ImageFormat, VideoCodec},
        V4L2CameraConfig, VideoWriter,
    },
};

#[derive(FromArgs)]
/// Record video from a webcam using background tasks
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

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
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
    let webcam = Arc::new(Mutex::new(
        V4L2CameraConfig::new()
            .with_camera_id(args.camera_id)
            .with_fps(args.fps as u32)
            .with_size(frame_size)
            .build()?,
    ));

    // start the webcam capture
    webcam.lock().await.start()?;

    // start the video writer
    let video_writer = Arc::new(Mutex::new(VideoWriter::new(
        args.output,
        VideoCodec::H264,
        ImageFormat::Rgb8,
        args.fps,
        frame_size,
    )?));

    // start the video writer
    video_writer.lock().await.start()?;

    // token to cancel the tasks
    let cancel_token = Arc::new(AtomicBool::new(false));

    ctrlc::set_handler({
        let cancel_token = cancel_token.clone();
        move || {
            println!("Received Ctrl-C signal. Sending cancel signal !!");
            cancel_token.store(true, Ordering::SeqCst);
        }
    })?;

    // Create a channel to send frames to the video writer
    let (tx, rx) = tokio::sync::mpsc::channel(50);
    let rx = Arc::new(Mutex::new(rx));

    // Worker to read frames from the camera and send them to the video writer
    let write_task = tokio::spawn({
        let video_writer = video_writer.clone();
        let rx = rx.clone();
        async move {
            while let Some(img) = rx.lock().await.recv().await {
                // lock the image and write it to the video writer
                video_writer.lock().await.write(&img)?;
            }
            Ok::<_, Box<dyn std::error::Error + Send + Sync>>(())
        }
    });

    // Worker to visualize the frames
    let visualization_task = tokio::spawn({
        let rx = rx.clone();
        async move {
            while let Some(img) = rx.lock().await.recv().await {
                // lock the image and visualize it with rerun
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

    // Worker to grab frames from the camera and send them to the video writer
    let capture_task = tokio::spawn({
        let webcam = webcam.clone();
        async move {
            while !cancel_token.load(Ordering::SeqCst) {
                // read the image from the camera
                if let Some(img) = webcam.lock().await.grab()? {
                    // send the image to broadcast channel
                    if let Err(e) = tx.send(img).await {
                        println!("Error sending image to channel: {:?}", e.to_string());
                    }
                }
            }
            Ok::<_, Box<dyn std::error::Error + Send + Sync>>(())
        }
    });

    tokio::select! {
        _ = write_task => {
            video_writer.lock().await.close()?;
        }
        _ = capture_task => {
            webcam.lock().await.close()?;
        }
        _ = visualization_task => (),
        _ = signal::ctrl_c() => (),
    }

    Ok(())
}
