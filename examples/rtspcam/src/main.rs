use clap::Parser;
use kornia::{
    image::{ops, Image},
    imgproc,
    io::{fps_counter::FpsCounter, stream::RTSPCameraConfig},
};
use std::sync::{Arc, Mutex};
use tokio::signal;

#[derive(Parser)]
struct Args {
    #[arg(short, long)]
    username: String,

    #[arg(short, long)]
    password: String,

    #[arg(long)]
    camera_ip: String,

    #[arg(long)]
    camera_port: u16,

    #[arg(short, long)]
    stream: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // start the recording stream
    let rec = rerun::RecordingStreamBuilder::new("Kornia Rtsp Stream Capture App").spawn()?;

    //// create a stream capture object
    let capture = RTSPCameraConfig::new()
        .with_settings(
            &args.username,
            &args.password,
            &args.camera_ip,
            &args.camera_port,
            &args.stream,
        )
        .build()?;

    // create a shared fps counter
    let fps_counter = Arc::new(Mutex::new(FpsCounter::new()));

    // preallocate images
    let mut img_f32 = Image::<f32, 3>::from_size_val([640, 360].into(), 0.0)?;
    let mut gray = Image::<f32, 1>::from_size_val(img_f32.size(), 0.0)?;

    // start grabbing frames from the camera
    capture
        .run_with_termination(
            |img| {
                // update the fps counter
                fps_counter
                    .lock()
                    .expect("Failed to lock fps counter")
                    .new_frame();

                // cast the image to floating point and convert to grayscale
                ops::cast_and_scale(&img, &mut img_f32, 1.0 / 255.0)?;
                imgproc::color::gray_from_rgb(&img_f32, &mut gray)?;

                // log the image
                rec.log_static(
                    "image",
                    &rerun::Image::from_elements(
                        img.as_slice(),
                        img.size().into(),
                        rerun::ColorModel::RGB,
                    ),
                )?;

                // log the grayscale image
                rec.log_static(
                    "gray",
                    &rerun::Image::from_elements(
                        gray.as_slice(),
                        gray.size().into(),
                        rerun::ColorModel::L,
                    ),
                )?;

                Ok(())
            },
            async {
                signal::ctrl_c().await.expect("Failed to listen for Ctrl+C");
                println!("👋 Finished recording. Closing app.");
            },
        )
        .await?;

    Ok(())
}
