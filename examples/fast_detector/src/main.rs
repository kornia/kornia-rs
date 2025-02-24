use clap::Parser;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use kornia::{
    image::{Image, ImageSize},
    imgproc,
    io::{fps_counter::FpsCounter, stream::V4L2CameraConfig},
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
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

    // start the background pipeline
    webcam.start()?;

    // create a cancel token to stop the webcam capture
    let cancel_token = Arc::new(AtomicBool::new(false));

    // create a shared fps counter
    let mut fps_counter = FpsCounter::new();

    ctrlc::set_handler({
        let cancel_token = cancel_token.clone();
        move || {
            println!("Received Ctrl-C signal. Sending cancel signal !!");
            cancel_token.store(true, Ordering::SeqCst);
        }
    })?;

    // we launch a timer to cancel the token after a certain duration
    std::thread::spawn({
        let cancel_token = cancel_token.clone();
        move || {
            if let Some(duration_secs) = args.duration {
                std::thread::sleep(std::time::Duration::from_secs(duration_secs));
                println!("Sending timer cancel signal !!");
                cancel_token.store(true, Ordering::SeqCst);
            }
        }
    });

    // preallocate images
    let mut gray = Image::from_size_val(
        ImageSize {
            width: 640,
            height: 480,
        },
        0u8,
    )?;

    // start grabbing frames from the camera
    while !cancel_token.load(Ordering::SeqCst) {
        let Some(img) = webcam.grab()? else {
            continue;
        };

        // convert the image to grayscale
        imgproc::color::gray_from_rgb_u8(&img, &mut gray)?;

        // detect the fast features
        let keypoints = imgproc::features::fast_feature_detector(&gray, 10)?;

        fps_counter.update();
        println!("FPS: {}", fps_counter.fps());

        // log the image
        rec.log_static(
            "image",
            &rerun::Image::from_elements(img.as_slice(), img.size().into(), rerun::ColorModel::RGB),
        )?;

        // log the grayscale image
        // rec.log_static(
        //     "gray",
        //     &rerun::Image::from_elements(gray.as_slice(), gray.size().into(), rerun::ColorModel::L),
        // )?;

        // log the keypoints
        let points = keypoints
            .iter()
            .map(|k| (k[0] as f32, k[1] as f32))
            .collect::<Vec<_>>();

        rec.log_static(
            "image/keypoints",
            &rerun::Points2D::new(points).with_colors([[0, 0, 255]]),
        )?;
    }

    // NOTE: this is important to close the webcam properly, otherwise the app will hang
    webcam.close()?;

    println!("Finished recording. Closing app.");

    Ok(())
}
