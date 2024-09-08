use clap::Parser;
use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
};

use kornia::{
    dnn::rtdetr::RTDETRDetectorBuilder,
    io::{
        fps_counter::FpsCounter,
        stream::{StreamCaptureError, V4L2CameraConfig},
    },
};

#[derive(Parser)]
struct Args {
    #[arg(short, long, default_value = "0")]
    camera_id: u32,

    #[arg(short, long, default_value = "5")]
    fps: u32,

    #[arg(short, long)]
    model_path: PathBuf,

    #[arg(short, long)]
    ort_dylib_path: PathBuf,

    #[arg(short, long, default_value = "8")]
    num_threads: usize,

    #[arg(short, long, default_value = "0.75")]
    score_threshold: f32,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // start the recording stream
    let rec = rerun::RecordingStreamBuilder::new("Kornia RTDETR App").spawn()?;

    // create a webcam capture object with camera id 0
    // and force the image size to 640x480
    let camera = V4L2CameraConfig::new()
        .with_camera_id(args.camera_id)
        .with_size([640, 480].into())
        .with_fps(args.fps)
        .build()?;

    let detector = RTDETRDetectorBuilder::new(args.model_path, args.ort_dylib_path)?
        .with_num_threads(args.num_threads)
        .build()?;

    // create a cancel token to stop the webcam capture
    let cancel_token = Arc::new(AtomicBool::new(false));

    // create a shared fps counter
    let fps_counter = Arc::new(Mutex::new(FpsCounter::new()));

    ctrlc::set_handler({
        let cancel_token = cancel_token.clone();
        move || {
            println!("Received Ctrl-C signal. Sending cancel signal !!");
            cancel_token.store(true, Ordering::SeqCst);
        }
    })?;

    // start grabbing frames from the camera
    camera
        .run(|img| {
            // check if the cancel token is set, if so we return an error to stop the pipeline
            if cancel_token.load(Ordering::SeqCst) {
                return Err(StreamCaptureError::PipelineCancelled.into());
            }

            // run the detector
            let detections = detector.run(&img)?;

            // filter the detections by score
            let detections = detections
                .into_iter()
                .filter(|d| d.score > args.score_threshold);

            // update the fps counter
            fps_counter
                .lock()
                .expect("Failed to lock fps counter")
                .new_frame();

            // log the detections
            let mut boxes_mins = Vec::new();
            let mut boxes_sizes = Vec::new();
            let mut class_ids = Vec::new();
            for detection in detections {
                boxes_mins.push((detection.x, detection.y));
                boxes_sizes.push((detection.w, detection.h));
                class_ids.push(detection.label as u16);
            }

            // log the image
            rec.log_static(
                "image",
                &rerun::Image::from_elements(
                    img.as_slice(),
                    img.size().into(),
                    rerun::ColorModel::RGB,
                ),
            )?;

            // log the detections
            rec.log_static(
                "detections",
                &rerun::Boxes2D::from_mins_and_sizes(boxes_mins, boxes_sizes)
                    .with_class_ids(class_ids),
            )?;

            Ok(())
        })
        .await?;

    // NOTE: this is important to close the webcam properly, otherwise the app will hang
    camera.close()?;

    println!("Finished recording. Closing app.");

    Ok(())
}
