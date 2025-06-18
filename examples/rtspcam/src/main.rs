use argh::FromArgs;
use kornia::{
    image::{ops, Image},
    imgproc,
    io::{fps_counter::FpsCounter, stream::RTSPCameraConfig},
    tensor::CpuAllocator,
};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

#[derive(FromArgs)]
/// RTSP Camera Capture and stream to ReRun
struct Args {
    /// the full RTSP URL (e.g., rtsp://user:pass@ip:port/stream)
    #[argh(option, short = 'r')]
    rtsp_url: Option<String>,

    /// the username to access the camera
    #[argh(option, short = 'u')]
    username: Option<String>,

    /// the password to access the camera
    #[argh(option, short = 'p')]
    password: Option<String>,

    /// the camera ip address
    #[argh(option)]
    camera_ip: Option<String>,

    /// the camera port
    #[argh(option)]
    camera_port: Option<u16>,

    /// the camera stream
    #[argh(option, short = 's')]
    stream: Option<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = argh::from_env();

    // start the recording stream
    let rec = rerun::RecordingStreamBuilder::new("Kornia Rtsp Stream Capture App").spawn()?;

    // create a stream capture object
    let mut capture = if let Some(url) = &args.rtsp_url {
        RTSPCameraConfig::new().with_url(url).build()?
    } else {
        RTSPCameraConfig::new()
            .with_settings(
                args.username.as_ref().ok_or("Username required")?,
                args.password.as_ref().ok_or("Password required")?,
                args.camera_ip.as_ref().ok_or("Camera IP required")?,
                &args.camera_port.ok_or("Camera port required")?,
                args.stream.as_ref().ok_or("Stream name required")?,
            )
            .build()?
    };

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
    let mut img_f32 = None;
    let mut gray = None;

    while !cancel_token.load(Ordering::SeqCst) {
        // start grabbing frames from the camera
        let Some(img) = capture.grab()? else {
            continue;
        };

        // initialize images lazily
        let img_f32_ref = img_f32.get_or_insert_with(|| {
            Image::<f32, 3, _>::from_size_val(img.size(), 0.0, CpuAllocator)
                .expect("Failed to create image")
        });
        let gray_ref = gray.get_or_insert_with(|| {
            Image::<f32, 1, _>::from_size_val(img.size(), 0.0, CpuAllocator)
                .expect("Failed to create image")
        });

        // cast the image to floating point and convert to grayscale
        ops::cast_and_scale(&img, img_f32_ref, 1.0 / 255.0)?;
        imgproc::color::gray_from_rgb(img_f32_ref, gray_ref)?;

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
            &rerun::Image::from_elements(
                gray_ref.as_slice(),
                gray_ref.size().into(),
                rerun::ColorModel::L,
            ),
        )?;

        rec.log_static("fps", &rerun::Scalars::new([fps_counter.fps() as f64]))?;
    }

    capture.close()?;

    Ok(())
}
