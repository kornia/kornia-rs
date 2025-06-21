use argh::FromArgs;
use kornia::{
    image::{ops, Image, ImageSize},
    imgproc,
    io::{
        fps_counter::FpsCounter,
        stream::V4L2CameraConfig,
        v4l2::{V4LCameraConfig, V4LVideoCapture},
    },
    tensor::CpuAllocator,
};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
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
    //let mut webcam = V4L2CameraConfig::new()
    //    .with_camera_id(args.camera_id)
    //    .with_fps(args.fps)
    //    .with_size(ImageSize {
    //        width: 640,
    //        height: 480,
    //    })
    //    .build()?;

    //// start the background pipeline
    //webcam.start()?;
    let img_size = ImageSize {
        width: 320,
        height: 180,
    };

    let mut webcam = V4LVideoCapture::new(V4LCameraConfig {
        device_path: "/dev/video0".to_string(),
        size: img_size,
        fps: args.fps,
    })?;

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
    //let mut img_resized = Image::from_size_val(new_size, 0u8)?;
    //let mut img_f32 = Image::from_size_val(new_size, 0f32)?;
    //let mut gray = Image::from_size_val(new_size, 0f32)?;
    //let mut bin = Image::from_size_val(new_size, 0f32)?;

    let mut img_rgb8 = Image::from_size_val(img_size, 0u8, CpuAllocator)?;

    // start grabbing frames from the camera
    while !cancel_token.load(Ordering::SeqCst) {
        let Some(Ok(img)) = webcam.grab() else {
            continue;
        };

        println!("img.size: {:?}", img_size);
        println!("img.fourcc: {:?}", img.fourcc);
        println!("img.timestamp: {:?}", img.timestamp);
        println!("img.sequence: {:?}", img.sequence);

        //kornia::io::jpeg::decode_image_jpeg_rgb8(&img.buffer, &mut img_rgb8)?;
        convert_yuyv_to_rgb(&img.buffer, &mut img_rgb8);

        rec.log_static(
            "image",
            &rerun::Image::from_elements(
                img_rgb8.as_slice(),
                img_size.into(),
                rerun::ColorModel::RGB,
            ),
        )?;

        // lets resize the image to 256x256
        //imgproc::resize::resize_fast(
        //    &img,
        //    &mut img_resized,
        //    imgproc::interpolation::InterpolationMode::Bilinear,
        //)?;

        //// convert the image to f32 and normalize before processing
        //ops::cast_and_scale(&img_resized, &mut img_f32, 1. / 255.)?;

        //// convert the image to grayscale and binarize
        //imgproc::color::gray_from_rgb(&img_f32, &mut gray)?;
        //imgproc::threshold::threshold_binary(&gray, &mut bin, 0.35, 0.65)?;

        //// update the fps counter
        //fps_counter.update();

        //// log the image
        //rec.log_static(
        //    "image",
        //    &rerun::Image::from_elements(img.as_slice(), img.size().into(), rerun::ColorModel::RGB),
        //)?;

        //// log the binary image
        //rec.log_static(
        //    "binary",
        //    &rerun::Image::from_elements(bin.as_slice(), bin.size().into(), rerun::ColorModel::L),
        //)?;
    }

    // NOTE: this is important to close the webcam properly, otherwise the app will hang
    //webcam.close()?;

    println!("Finished recording. Closing app.");

    Ok(())
}

/// Convert YUYV to RGB - optimized version
fn convert_yuyv_to_rgb(yuyv_data: &[u8], rgb_image: &mut Image<u8, 3, CpuAllocator>) {
    let rgb_data = rgb_image.as_slice_mut();

    yuyv_data
        .chunks_exact(4)
        .zip(rgb_data.chunks_exact_mut(6)) // 6 bytes = 2 RGB pixels
        .for_each(|(yuyv_chunk, rgb_chunk)| {
            // Extract YUYV components
            let y0 = yuyv_chunk[0] as i32;
            let u = yuyv_chunk[1] as i32 - 128;
            let y1 = yuyv_chunk[2] as i32;
            let v = yuyv_chunk[3] as i32 - 128;

            // Precompute shared chroma components (since U,V are shared)
            let u_r = 0;
            let u_g = -11 * u; // -0.344 * 32 ≈ -11
            let u_b = 57 * u; // 1.772 * 32 ≈ 57
            let v_r = 45 * v; // 1.402 * 32 ≈ 45
            let v_g = -23 * v; // -0.714 * 32 ≈ -23
            let v_b = 0;

            // Convert both pixels using integer arithmetic (scaled by 32)
            let r0 = ((y0 << 5) + u_r + v_r) >> 5;
            let g0 = ((y0 << 5) + u_g + v_g) >> 5;
            let b0 = ((y0 << 5) + u_b + v_b) >> 5;

            let r1 = ((y1 << 5) + u_r + v_r) >> 5;
            let g1 = ((y1 << 5) + u_g + v_g) >> 5;
            let b1 = ((y1 << 5) + u_b + v_b) >> 5;

            // Write both RGB pixels at once
            rgb_chunk[0] = r0.clamp(0, 255) as u8;
            rgb_chunk[1] = g0.clamp(0, 255) as u8;
            rgb_chunk[2] = b0.clamp(0, 255) as u8;
            rgb_chunk[3] = r1.clamp(0, 255) as u8;
            rgb_chunk[4] = g1.clamp(0, 255) as u8;
            rgb_chunk[5] = b1.clamp(0, 255) as u8;
        });
}
