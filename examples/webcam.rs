use anyhow::Result;
use clap::Parser;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};

use kornia_rs::io::fps_counter::FpsCounter;
use kornia_rs::{image::ImageSize, io::webcam::WebcamCaptureBuilder};

#[derive(Parser)]
struct Args {
    #[arg(short, long, default_value = "0")]
    camera_id: usize,

    #[arg(short, long)]
    sleep_sec: Option<u64>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
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
    let cancel_token = Arc::new(AtomicBool::new(false));

    // create a thread to stop the webcam capture
    let timer_thread = {
        let cancel_token = cancel_token.clone();
        std::thread::spawn(move || {
            if let Some(args) = args.sleep_sec {
                std::thread::sleep(std::time::Duration::from_secs(args));
                cancel_token.store(true, Ordering::SeqCst);
            }
        })
    };

    // start grabbing frames from the camera
    let fps_counter = Arc::new(Mutex::new(FpsCounter::new()));

    webcam.run(cancel_token, |img| {
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
        let bin = kornia_rs::threshold::threshold_binary(&gray, 0.5, 1.0)?;

        // update the fps counter
        fps_counter.lock().unwrap().new_frame();

        // log the image
        rec.log("binary", &rerun::Image::try_from(bin.data)?)?;

        Ok(())
    })?;

    // wait for the threads to finish
    timer_thread.join().expect("timer thread panicked");

    println!("Finished recording");

    // TODO: close rerun::RecordingStream
    // rec.close()?;

    Ok(())
}
