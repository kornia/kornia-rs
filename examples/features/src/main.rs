use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use kornia::{
    image::{ops, Image},
    imgproc,
    io::stream::V4L2CameraConfig,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // start the recording stream
    let rec = rerun::RecordingStreamBuilder::new("Kornia Webcapture App").spawn()?;

    // image size
    let size = [640, 480].into();

    // create a webcam capture object with camera id 0
    // and force the image size to 640x480
    let mut webcam = V4L2CameraConfig::new().with_size(size).build()?;

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

    // preallocate images
    let mut img_f32 = Image::from_size_val(size, 0f32)?;
    let mut gray = Image::from_size_val(size, 0f32)?;
    let mut hessian = Image::from_size_val(size, 0f32)?;
    let mut corners = Image::from_size_val(size, 0f32)?;

    // start grabbing frames from the camera
    while !cancel_token.load(Ordering::SeqCst) {
        let Some(img) = webcam.grab()? else {
            continue;
        };

        // convert to grayscale
        ops::cast_and_scale(&img, &mut img_f32, 1. / 255.)?;
        imgproc::color::gray_from_rgb(&img_f32, &mut gray)?;

        // compute the hessian response
        imgproc::features::hessian_response(&gray, &mut hessian)?;

        // compute the corners
        imgproc::threshold::threshold_binary(&hessian, &mut corners, 0.01, 1.0)?;

        // log the image
        rec.log_static(
            "image",
            &rerun::Image::from_elements(img.as_slice(), img.size().into(), rerun::ColorModel::RGB),
        )?;

        // log the corners
        rec.log_static(
            "corners",
            &rerun::Image::from_elements(
                corners.as_slice(),
                corners.size().into(),
                rerun::ColorModel::L,
            ),
        )?;
    }

    // NOTE: this is important to close the webcam properly, otherwise the app will hang
    webcam.close()?;

    println!("Finished recording. Closing app.");

    Ok(())
}
