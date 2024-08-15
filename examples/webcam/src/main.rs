use clap::Parser;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    {Arc, Mutex},
};

use kornia::{
    image::{Image, ImageSize},
    imgproc,
    io::{
        fps_counter::FpsCounter,
        stream::{StreamCaptureError, V4L2CameraConfig},
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

    // create a webcam capture object with camera id 0
    // and force the image size to 640x480
    let mut webcam = V4L2CameraConfig::new()
        .with_camera_id(args.camera_id)
        .with_fps(args.fps)
        .with_size(ImageSize {
            width: 640,
            height: 480,
        })
        .build()?;

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

    // allocate the image buffers
    let new_size = ImageSize {
        width: 256,
        height: 256,
    };

    let mut img_resized = Image::from_size_val(new_size, 0u8)?;
    let mut gray = Image::from_size_val(new_size, 0f32)?;
    let mut bin = Image::from_size_val(new_size, 0f32)?;

    // start grabbing frames from the camera
    webcam
        .run(|img| {
            // check if the cancel token is set, if so we return an error to stop the pipeline
            if cancel_token.load(Ordering::SeqCst) {
                return Err(StreamCaptureError::PipelineCancelled.into());
            }

            // lets resize the image to 256x256
            imgproc::resize::resize_fast(
                &img,
                &mut img_resized,
                new_size,
                imgproc::interpolation::InterpolationMode::Bilinear,
            )?;

            // convert the image to f32 and normalize before processing
            let img = img_resized.clone().cast_and_scale::<f32>(1. / 255.)?;

            // convert the image to grayscale and binarize
            imgproc::color::gray_from_rgb(&img, &mut gray)?;
            imgproc::threshold::threshold_binary(&gray, &mut bin, 0.35, 0.65)?;

            // update the fps counter
            fps_counter
                .lock()
                .expect("Failed to lock fps counter")
                .new_frame();

            // log the image
            rec.log_static("image", &rerun::Image::try_from(img.data)?)?;
            rec.log_static("binary", &rerun::Image::try_from(bin.clone().data)?)?;

            Ok(())
        })
        .await?;

    // NOTE: this is important to close the webcam properly, otherwise the app will hang
    webcam.close()?;

    println!("Finished recording. Closing app.");

    Ok(())
}
