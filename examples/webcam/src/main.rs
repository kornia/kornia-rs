use argh::FromArgs;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use kornia::{
    image::{ops, Image, ImageSize},
    imgproc,
    io::{fps_counter::FpsCounter, stream::V4L2CameraConfig},
};

#[derive(FromArgs)]
/// Capture frames from a webcam and log to Rerun
struct Args {
    /// the camera id to use
    #[argh(option, short = 'c', default = "0")]
    camera_id: u32,

    /// the frames per second to record
    #[argh(option, short = 'f', default = "30")]
    fps: u32,

    /// the duration in seconds to run the app
    #[argh(option, short = 'd')]
    duration: Option<u64>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = argh::from_env();

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

    // allocate the image buffers
    let new_size = ImageSize {
        width: 256,
        height: 256,
    };

    // preallocate images
    let mut img_resized = Image::from_size_val(new_size, 0u8)?;
    let mut img_f32 = Image::from_size_val(new_size, 0f32)?;
    let mut gray = Image::from_size_val(new_size, 0f32)?;
    let mut bin = Image::from_size_val(new_size, 0f32)?;

    // start grabbing frames from the camera
    while !cancel_token.load(Ordering::SeqCst) {
        let Some(img) = webcam.grab()? else {
            continue;
        };

        // lets resize the image to 256x256
        imgproc::resize::resize_fast(
            &img,
            &mut img_resized,
            imgproc::interpolation::InterpolationMode::Bilinear,
        )?;

        // convert the image to f32 and normalize before processing
        ops::cast_and_scale(&img_resized, &mut img_f32, 1. / 255.)?;

        // convert the image to grayscale and binarize
        imgproc::color::gray_from_rgb(&img_f32, &mut gray)?;
        imgproc::threshold::threshold_binary(&gray, &mut bin, 0.35, 0.65)?;

        // update the fps counter
        fps_counter.update();

        // log the image
        rec.log_static(
            "image",
            &rerun::Image::from_elements(img.as_slice(), img.size().into(), rerun::ColorModel::RGB),
        )?;

        // log the binary image
        rec.log_static(
            "binary",
            &rerun::Image::from_elements(bin.as_slice(), bin.size().into(), rerun::ColorModel::L),
        )?;
    }

    // NOTE: this is important to close the webcam properly, otherwise the app will hang
    webcam.close()?;

    println!("Finished recording. Closing app.");

    Ok(())
}
