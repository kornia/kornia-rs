use clap::Parser;
use std::sync::{Arc, Mutex};

use kornia_rs::io::fps_counter::FpsCounter;
use kornia_rs::{image::ImageSize, io::webcam::WebcamCaptureBuilder};

#[derive(Parser)]
struct Args {
    #[arg(short, long, default_value = "0")]
    camera_id: usize,
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

    // start grabbing frames from the camera
    let fps_counter = Arc::new(Mutex::new(FpsCounter::new()));

    webcam
        .run(|img| {
            // lets resize the image to 256x256
            let img = kornia_rs::resize::resize_fast(
                &img,
                kornia_rs::image::ImageSize {
                    width: 256,
                    height: 256,
                },
                kornia_rs::resize::InterpolationMode::Bilinear,
            )?;

            // update the fps counter
            fps_counter
                .lock()
                .expect("Failed to lock fps counter")
                .new_frame();

            // log the image
            rec.log("image", &rerun::Image::try_from(img.data)?)?;

            Ok(())
        })
        .await?;

    // TODO: close rerun::RecordingStream
    // rec.close()?;

    Ok(())
}
