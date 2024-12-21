use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use kornia::io::stream::V4L2CameraConfig;
use kornia_depth::DepthAnything;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    // create a rerun recorder
    let rec = rerun::RecordingStreamBuilder::new("depth_anything").spawn()?;

    // create a depth anything model
    let mut depth_anything = DepthAnything::new(None, None)?;

    // create a webcam capture object
    let mut webcam = V4L2CameraConfig::new()
        .with_size([640, 480].into())
        .build()?;

    // start the background pipeline
    webcam.start()?;

    // create a cancel token to stop the webcam capture
    let cancel_token = Arc::new(AtomicBool::new(false));

    ctrlc::set_handler({
        let cancel_token = cancel_token.clone();
        move || {
            println!("Received Ctrl-C signal. Sending cancel signal !!");
            cancel_token.store(true, Ordering::SeqCst);
        }
    })?;

    let mut img_count = 0;

    // start grabbing frames from the camera
    while !cancel_token.load(Ordering::SeqCst) {
        let Some(img) = webcam.grab()? else {
            continue;
        };

        let now = std::time::Instant::now();

        println!("Processing image {}", img_count);

        let depth = depth_anything.forward(&img)?;

        let elapsed = now.elapsed();
        println!("Elapsed time: {:?}", elapsed);

        // log the image
        rec.log_static(
            "image",
            &rerun::Image::from_elements(img.as_slice(), img.size().into(), rerun::ColorModel::RGB),
        )?;

        let depth_viz = depth.scale_and_cast::<u8>(255.0)?;
        let depth_image = rerun::DepthImage::new(
            depth_viz.as_slice().to_vec(),
            rerun::ImageFormat::depth(depth_viz.size().into(), rerun::ChannelDatatype::U8),
        );

        rec.log(
            "world/camera",
            &rerun::Pinhole::from_focal_length_and_resolution([200.0, 200.0], [640.0, 480.0]),
        )?;

        rec.log_static("world/camera/depth", &depth_image)?;

        img_count += 1;
    }

    // NOTE: this is important to close the webcam properly, otherwise the app will hang
    webcam.close()?;

    println!("Finished recording. Closing app.");

    Ok(())
}
