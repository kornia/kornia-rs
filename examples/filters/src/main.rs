use argh::FromArgs;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use kornia::{
    image::{ops, Image},
    imgproc,
    io::stream::V4L2CameraConfig,
};

#[derive(FromArgs)]
/// Apply a separable filter to an image
struct Args {
    /// the filter to apply
    #[argh(option)]
    filter: String,

    /// the kernel size for the horizontal filter
    #[argh(option)]
    kx: usize,

    /// the kernel size for the vertical filter
    #[argh(option)]
    ky: usize,

    /// the sigma for the gaussian filter
    #[argh(option)]
    sigma_x: Option<f32>,

    /// the sigma for the gaussian filter
    #[argh(option)]
    sigma_y: Option<f32>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = argh::from_env();

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
    let mut img_f32_filtered = Image::from_size_val(size, 0f32)?;
    // start grabbing frames from the camera
    while !cancel_token.load(Ordering::SeqCst) {
        let Some(img) = webcam.grab()? else {
            continue;
        };

        // convert to grayscale
        ops::cast_and_scale(&img, &mut img_f32, 1. / 255.)?;

        match args.filter.to_lowercase().as_str() {
            "box" => {
                imgproc::filter::box_blur(&img_f32, &mut img_f32_filtered, (args.kx, args.ky))?;
            }
            "gaussian" => {
                let sigma_x = args.sigma_x.unwrap_or(0.5);
                let sigma_y = args.sigma_y.unwrap_or(0.5);
                imgproc::filter::gaussian_blur(
                    &img_f32,
                    &mut img_f32_filtered,
                    (args.kx, args.ky),
                    (sigma_x, sigma_y),
                )?;
            }
            "sobel" => {
                let mut img_f32_filtered_sobel = Image::from_size_val(size, 0f32)?;
                imgproc::filter::sobel(&img_f32, &mut img_f32_filtered_sobel, args.kx)?;

                // we need to normalize the sobel filter to 0-1
                imgproc::normalize::normalize_min_max(
                    &img_f32_filtered_sobel,
                    &mut img_f32_filtered,
                    0.0,
                    1.0,
                )?;
            }
            _ => {
                return Err(format!("Invalid filter: {}", args.filter).into());
            }
        }

        // log the image
        rec.log_static(
            "image",
            &rerun::Image::from_elements(img.as_slice(), img.size().into(), rerun::ColorModel::RGB),
        )?;

        // log the blurred image
        rec.log_static(
            "filtered",
            &rerun::Image::from_elements(
                img_f32_filtered.as_slice(),
                img_f32_filtered.size().into(),
                rerun::ColorModel::RGB,
            ),
        )?;
    }

    // NOTE: this is important to close the webcam properly, otherwise the app will hang
    webcam.close()?;

    println!("Finished recording. Closing app.");

    Ok(())
}
