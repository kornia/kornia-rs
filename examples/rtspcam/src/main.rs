use argh::FromArgs;
use kornia::{
    image::{ops, Image},
    imgproc,
    io::{fps_counter::FpsCounter, stream::RTSPCameraConfig},
};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

#[derive(FromArgs)]
/// RTSP Camera Capture and stream to ReRun
struct Args {
    /// the username to access the camera
    #[argh(option, short = 'u')]
    username: String,

    /// the password to access the camera
    #[argh(option, short = 'p')]
    password: String,

    /// the camera ip address
    #[argh(option)]
    camera_ip: String,

    /// the camera port
    #[argh(option)]
    camera_port: u16,

    /// the camera stream
    #[argh(option, short = 's')]
    stream: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = argh::from_env();

    // start the recording stream
    let rec = rerun::RecordingStreamBuilder::new("Kornia Rtsp Stream Capture App").spawn()?;

    //// create a stream capture object
    let mut capture = RTSPCameraConfig::new()
        .with_settings(
            &args.username,
            &args.password,
            &args.camera_ip,
            &args.camera_port,
            &args.stream,
        )
        .build()?;

    // start the stream capture
    capture.start()?;

    // create a cancel token to stop the stream capture
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

    // preallocate images
    let mut img_f32 = Image::<f32, 3>::from_size_val([640, 360].into(), 0.0)?;
    let mut gray = Image::<f32, 1>::from_size_val(img_f32.size(), 0.0)?;

    while !cancel_token.load(Ordering::SeqCst) {
        // start grabbing frames from the camera
        let Some(img) = capture.grab()? else {
            continue;
        };

        // cast the image to floating point and convert to grayscale
        ops::cast_and_scale(&img, &mut img_f32, 1.0 / 255.0)?;
        imgproc::color::gray_from_rgb(&img_f32, &mut gray)?;

        // update the fps counter
        fps_counter.update();

        // log the image
        rec.log_static(
            "image",
            &rerun::Image::from_elements(img.as_slice(), img.size().into(), rerun::ColorModel::RGB),
        )?;

        // log the grayscale image
        rec.log_static(
            "gray",
            &rerun::Image::from_elements(gray.as_slice(), gray.size().into(), rerun::ColorModel::L),
        )?;

        rec.log_static("fps", &rerun::Scalar::new(fps_counter.fps() as f64))?;
    }

    capture.close()?;

    Ok(())
}
