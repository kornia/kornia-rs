use clap::Parser;
use kornia::imgproc::interpolation::InterpolationMode;
use std::path::PathBuf;

use kornia::image::{image, Image, ImageSize};
use kornia::io::functional as F;

#[derive(Parser)]
struct Args {
    #[arg(short, long)]
    image_path: PathBuf,

    #[arg(short, long)]
    onnx_model_path: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // read the image
    let image: Image<u8, 3> = F::read_image_any(&args.image_path)?;

    // read the onnx model

    use ort::{GraphOptimizationLevel, Session};

    let model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file(&args.onnx_model_path)?;

    let new_size = (320, 320).into();

    let mut image_resized = Image::from_size_val(new_size, 0)?;

    kornia::imgproc::resize::resize_fast(
        &image,
        &mut image_resized,
        new_size,
        InterpolationMode::Bilinear,
    )?;

    let image = image_resized.clone().cast_and_scale::<f32>(1. / 255.)?;

    let image_ncwh = image.to_tensor_nchw();
    let image_ncwh = image_ncwh.as_standard_layout().to_owned();

    // run the model
    let outputs = model.run(ort::inputs![
        "images" => image_ncwh,
    ]?)?;

    // get the output
    let loc = outputs["loc"].try_extract_tensor::<f32>()?;
    let conf = outputs["conf"].try_extract_tensor::<f32>()?;
    let iou = outputs["iou"].try_extract_tensor::<f32>()?;

    println!("loc: {:?}", loc);
    println!("conf: {:?}", conf);
    println!("iou: {:?}", iou);

    Ok(())
}
