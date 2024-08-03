use clap::Parser;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    {Arc, Mutex},
};

use kornia::{
    image::ImageSize,
    imgproc,
    io::{
        fps_counter::FpsCounter,
        stream::{StreamCapture, StreamCaptureError},
    },
};

#[derive(Parser)]
struct Args {
    #[arg(short, long, default_value = "0")]
    camera_id: usize,

    #[arg(short, long, default_value = "30")]
    fps: u32,

    #[arg(short, long)]
    duration: Option<u64>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // start the recording stream
    let rec = rerun::RecordingStreamBuilder::new("Kornia Stream Capture App").spawn()?;

    // create a webcam capture object with camera id 0
    // and force the image size to 640x480
    let mut capture = StreamCapture::new(
        //"v4l2src device=/dev/video0 ! videoconvert ! video/x-raw,format=RGB,width=640,height=480 ! appsink name=sink",
        "souphttpsrc location=http://192.168.1.139:81/stream ! jpegparse ! jpegdec ! appsink name=sink",
    )?;

    // create a cancel token to stop the webcam capture
    let cancel_token = Arc::new(AtomicBool::new(false));

    // create a shared fps counter
    let fps_counter = Arc::new(Mutex::new(FpsCounter::new()));

    ctrlc::set_handler({
        let cancel_token = cancel_token.clone();
        move || {
            println!("Received Ctrl-C signal. Sending cancel signal !!");
            cancel_token.store(true, Ordering::SeqCst);
        }
    })?;

    // we launch a timer to cancel the token after a certain duration
    tokio::spawn({
        let cancel_token = cancel_token.clone();
        async move {
            if let Some(duration_secs) = args.duration {
                tokio::time::sleep(tokio::time::Duration::from_secs(duration_secs)).await;
                println!("Sending timer cancel signal !!");
                cancel_token.store(true, Ordering::SeqCst);
            }
        }
    });

    // start grabbing frames from the camera
    capture
        .run(|img| {
            // check if the cancel token is set, if so we return an error to stop the pipeline
            if cancel_token.load(Ordering::SeqCst) {
                return Err(StreamCaptureError::PipelineCancelled.into());
            }

            // update the fps counter
            fps_counter
                .lock()
                .expect("Failed to lock fps counter")
                .new_frame();

            // log the image
            rec.log_static("image", &rerun::Image::try_from(img.data)?)?;

            Ok(())
        })
        .await?;

    // NOTE: this is important to close the webcam properly, otherwise the app will hang
    capture.close()?;

    println!("Finished recording. Closing app.");

    Ok(())
}
