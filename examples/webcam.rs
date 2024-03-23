use kornia_rs::{image::ImageSize, io::webcam::WebcamCaptureBuilder};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // start the recording stream
    let rec = rerun::RecordingStreamBuilder::new("Kornia App")
        // set the batcher config to flush every 10 bytes to improve performance
        .batcher_config(rerun::log::DataTableBatcherConfig {
            flush_num_bytes: 10,
            ..Default::default()
        })
        .spawn()?;

    // create a webcam capture object with camera id 0
    // and force the image size to 640x480
    let webcam = WebcamCaptureBuilder::new()
        .camera_id(0)
        .with_size(ImageSize {
            width: 640,
            height: 480,
        })
        .build()?;

    webcam.run(|img| {
        // lets resize the image to 256x256
        let img = kornia_rs::resize::resize_fast(
            &img,
            kornia_rs::image::ImageSize {
                width: 256,
                height: 256,
            },
            kornia_rs::resize::InterpolationMode::Bilinear,
        )
        .unwrap();

        // convert the image to f32 and normalize before processing
        let img = img.cast_and_scale::<f32>(1. / 255.).unwrap();

        // convert the image to grayscale and binarize
        let gray = kornia_rs::color::gray_from_rgb(&img).unwrap();
        let bin = kornia_rs::threshold::threshold_binary(&gray, 0.5, 1.0).unwrap();

        // log the image
        let _ = rec.log("binary", &rerun::Image::try_from(bin.data).unwrap());
    })?;

    Ok(())
}
