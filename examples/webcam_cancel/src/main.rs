use clap::Parser;
use std::sync::{Arc, Mutex};
use tokio_util::sync::CancellationToken;

use kornia_rs::io::fps_counter::FpsCounter;
use kornia_rs::{image::ImageSize, io::webcam::WebcamCaptureBuilder};

#[derive(Parser)]
struct Args {
    #[arg(short, long, default_value = "0")]
    camera_id: usize,

    #[arg(short, long)]
    duration: Option<u64>,
}

#[derive(thiserror::Error, Debug)]
pub enum CancelledError {
    #[error("Cancelled")]
    Cancelled,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // start the recording stream
    let rec = rerun::RecordingStreamBuilder::new("Kornia Webcapture App").spawn()?;

    // create a webcam capture object with camera id 0
    // and force the image size to 640x480
    let mut webcam = WebcamCaptureBuilder::new()
        .camera_id(args.camera_id)
        .with_size(ImageSize {
            width: 640,
            height: 480,
        })
        .build()?;

    // create a cancel token to stop the webcam capture
    let cancel_token = CancellationToken::new();
    let child_token = cancel_token.child_token();

    let fps_counter = Arc::new(Mutex::new(FpsCounter::new()));

    let join_handle = tokio::spawn(async move {
        tokio::select! {
            _ = webcam.run(|img| {
                    // lets resize the image to 256x256
                    let img = kornia_rs::resize::resize_fast(
                        &img,
                        kornia_rs::image::ImageSize {
                            width: 256,
                            height: 256,
                        },
                        kornia_rs::resize::InterpolationMode::Bilinear,
                    )?;

                    // convert the image to f32 and normalize before processing
                    let img = img.cast_and_scale::<f32>(1. / 255.)?;

                    // convert the image to grayscale and binarize
                    let gray = kornia_rs::color::gray_from_rgb(&img)?;
                    let bin = kornia_rs::threshold::threshold_binary(&gray, 0.35, 0.65)?;

                    // update the fps counter
                    fps_counter
                        .lock()
                        .expect("Failed to lock fps counter")
                        .new_frame();

                    // log the image
                    rec.log("image", &rerun::Image::try_from(img.data)?)?;
                    rec.log("binary", &rerun::Image::try_from(bin.data)?)?;

                    Ok(())
                }) => { Ok(()) }
            _ = child_token.cancelled() => {
                println!("Received cancel signal. Closing webcam.");
                webcam.close().await.expect("Failed to close webcam");
                Err(CancelledError::Cancelled)
            }
        }
    });

    // we launch a timer to cancel the token after a certain duration
    tokio::spawn(async move {
        if let Some(duration_secs) = args.duration {
            tokio::time::sleep(tokio::time::Duration::from_secs(duration_secs)).await;
            println!("Sending cancel signal !!");
            cancel_token.cancel();
        }
    });

    join_handle.await??;

    println!("Finished recording. Closing app.");

    // TODO: close rerun::RecordingStream
    // rec.close()?;

    Ok(())
}
